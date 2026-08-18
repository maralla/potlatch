//! QA agent: monitors configured branches for new commits, gathers the
//! project's open QA-labeled GitLab issues into a context file, and invokes
//! the model to perform end-user functionality testing against the running
//! system (HTTP APIs, CLI invocations, configuration loading). The model
//! authors and runs its own Python test scripts and maintains its own
//! knowledge files in the agent's session directory; the agent's only file
//! responsibility is the QA-issues context file. Creates GitLab issues for
//! non-trivial findings.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient};
use crate::agents::workspace::{GitLabAgentBootstrap, GitLabAgentRuntime, gitlab_banner};
#[cfg(test)]
use crate::core::agent::StructuredOutput;
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, compat, structured_output};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::cycle::Step;
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;

pub(crate) const NAME: &str = "qa";
const MAX_INSTANCES: usize = 1;

const QA_LABEL: &str = crate::agents::labels::QA;
const DO_NOT_IMPLEMENT_LABEL: &str = crate::agents::labels::DO_NOT_IMPLEMENT;

/// A finding's severity, as the model reports it via the `qa_report` tool.
/// The contract only admits the four names below; a severity the model
/// invents is folded to `Low` by [`QaOutput::normalize`], preserving the
/// long-standing behavior of treating unrecognized severity as non-blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Severity {
    Critical,
    High,
    Medium,
    #[default]
    Low,
}

const SEVERITIES: &[&str] = &["critical", "high", "medium", "low"];

impl Severity {
    fn as_str(&self) -> &'static str {
        match self {
            Severity::Critical => "critical",
            Severity::High => "high",
            Severity::Medium => "medium",
            Severity::Low => "low",
        }
    }

    fn is_non_trivial(&self) -> bool {
        !matches!(self, Severity::Low)
    }

    fn priority(&self) -> u8 {
        match self {
            Severity::Critical => 1,
            Severity::High => 2,
            Severity::Medium | Severity::Low => 3,
        }
    }
}

/// One finding as the model described it via the `qa_report` tool's
/// `findings` array, before the empty-title/description defensive filtering
/// in [`normalize_qa_findings`] is applied.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQaFinding {
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    severity: Severity,
    #[serde(default)]
    file: String,
}

/// One clarification question as the model described it via the
/// `qa_report` tool's `clarifications` array, before the empty-question
/// filtering in [`normalize_clarifications`] is applied.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawClarification {
    #[serde(default)]
    question: String,
    #[serde(default)]
    context: String,
}

/// The QA agent's typed structured-output contract. The model calls the
/// `qa_report` tool with its findings and clarification questions; core
/// deserializes the captured JSON into this type (see
/// [`AgentModel::complete_typed`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct QaOutput {
    #[serde(default)]
    findings: Vec<RawQaFinding>,
    #[serde(default)]
    clarifications: Vec<RawClarification>,
}

structured_output! {
    impl QaOutput {
        tool_name: "qa_report";
        tool_description: "The findings and clarification questions from this QA run.";
        schema: object("Everything this QA run found.", {
            required findings: array(
                "Test findings (bugs). Empty array if no bugs found.",
                object("One bug this run reproduced.", {
                    required title: string("Short actionable title for the finding."),
                    required description: string(
                        "Detailed description with steps to reproduce, expected vs actual behavior, and impact."
                    ),
                    required severity: string_enum(
                        "Severity: \"critical\", \"high\", \"medium\", or \"low\".",
                        SEVERITIES
                    ),
                    optional file: string(
                        "Source file and line number if known (e.g. \"src/path/to/file.rs:123\"). Omit if unknown."
                    ),
                })
            ),
            optional clarifications: array(
                "Clarification questions for humans. Empty array if none.",
                object("One question a human has to answer.", {
                    required question: string("The clarification question."),
                    required context: string(
                        "Context explaining why the question is needed."
                    ),
                })
            ),
        });
    /// Tolerated: a severity the model invented or spelled differently, which
    /// becomes `"low"` so an odd label never blocks a whole QA run.
        normalize(value) {
            compat::each_in_array(value, "findings", |finding| {
                compat::normalize_enum(finding, "severity", SEVERITIES, "low");
            });
        }
    }
}

/// Drop findings with an empty title or description (the model occasionally
/// emits a placeholder entry).
fn normalize_qa_findings(raw: Vec<RawQaFinding>) -> Vec<QaFinding> {
    raw.into_iter()
        .filter_map(|f| {
            let title = f.title.trim().to_string();
            if title.is_empty() {
                return None;
            }
            let description = f.description.trim().to_string();
            if description.is_empty() {
                return None;
            }
            Some(QaFinding {
                title,
                description,
                severity: f.severity,
                file: f.file,
            })
        })
        .collect()
}

/// Drop clarifications with an empty question.
fn normalize_clarifications(raw: Vec<RawClarification>) -> Vec<ClarificationQuestion> {
    raw.into_iter()
        .filter_map(|c| {
            let question = c.question.trim().to_string();
            if question.is_empty() {
                return None;
            }
            Some(ClarificationQuestion {
                question,
                context: c.context,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QaConfig {
    poll_interval_secs: u64,
    branches: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct QaAgentSettings {
    #[serde(default = "default_qa_poll_interval")]
    poll_interval_secs: u64,
    #[serde(default = "default_branches")]
    branches: Vec<String>,
}

fn default_qa_poll_interval() -> u64 {
    300
}

fn default_branches() -> Vec<String> {
    vec!["main".to_string()]
}

impl QaAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        let settings: Self = raw
            .clone()
            .try_into()
            .context("qa agent settings from config")?;
        ensure!(
            !settings.branches.is_empty(),
            "[agent.qa] branches must not be empty"
        );
        Ok(settings)
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// A borrowing view over the [`GitLabAgentRuntime`] fields the QA cycle
/// needs. Built fresh from `&GitLabAgentRuntime` at each use site rather
/// than stored, so QA never owns a second `GitRepo`/`GitLabClient` — and,
/// since it is never stored alongside the runtime it borrows from, it
/// can't become self-referential.
struct AgentState<'a> {
    agent_id: &'a str,
    sessions_dir: &'a str,
    git_repo: &'a GitRepo,
    glab: &'a GitLabClient,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &GitLabAgentRuntime) -> AgentState<'_> {
        AgentState {
            agent_id: &runtime.agent_id,
            sessions_dir: &runtime.sessions_dir,
            git_repo: &runtime.git_repo,
            glab: &runtime.gitlab,
        }
    }

    fn sha_history_path(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_qa_sha_history.json", self.agent_id))
    }

    fn qa_issues_path(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_qa_issues.md", self.agent_id))
    }

    /// Path to a compact listing of *all* open project issues (not just
    /// QA-labeled), refreshed each cycle. The model reads this to check
    /// whether a finding is already tracked before reporting it, so the QA
    /// agent doesn't file duplicates of issues other agents or humans have
    /// already opened.
    fn open_issues_path(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_open_issues.md", self.agent_id))
    }

    fn knowledge_dir(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_qa_knowledge", self.agent_id))
    }

    fn test_scripts_dir(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_test_scripts", self.agent_id))
    }
}

pub(crate) struct QaAgent {
    runtime: GitLabAgentRuntime,
    config: QaConfig,
}

// ---------------------------------------------------------------------------
// Persistence types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
struct ShaHistory(HashMap<String, String>);

// ---------------------------------------------------------------------------
// Parsed output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct QaFinding {
    title: String,
    description: String,
    severity: Severity,
    file: String,
}

#[derive(Debug, Clone)]
struct ClarificationQuestion {
    question: String,
    context: String,
}

// ---------------------------------------------------------------------------
// CoreAgent impl
// ---------------------------------------------------------------------------

impl CoreAgent for QaAgent {
    type Settings = QaAgentSettings;
    const MAX_INSTANCES: Option<usize> = Some(MAX_INSTANCES);

    fn name() -> &'static str {
        NAME
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime.core
    }

    fn banner(config: &Config, banner: &mut Banner) {
        gitlab_banner(config, banner);
    }

    fn parse_settings(
        _config: &Config,
        section: &crate::core::config::AgentSection,
    ) -> Result<Self::Settings> {
        QaAgentSettings::from_raw(&section.raw)
    }

    fn validate_settings(
        config: &Config,
        _section: &crate::core::config::AgentSection,
        _settings: &Self::Settings,
    ) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_gitlab_repo()?;
        Ok(())
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec::polling(
            "qa_poll",
            Duration::from_secs(self.config.poll_interval_secs),
        )]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "qa_poll" => {
                let scope = crate::agents::scope_label_filter(&self.runtime.scope_label);
                let state = AgentState::from_runtime(&self.runtime);
                qa_cycle(&state, &self.config, &self.runtime.model, scope)
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = GitLabAgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        let agent_settings = ctx.settings;
        let config = QaConfig {
            poll_interval_secs: agent_settings.poll_interval_secs,
            branches: agent_settings.branches,
        };
        Ok(Self { runtime, config })
    }

    fn on_shutdown(&mut self) {}
}

// ---------------------------------------------------------------------------
// QA cycle
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct IssueObservation {
    iid: u64,
    title: String,
    description: String,
    labels: Vec<String>,
}

impl From<&gitlab::Issue> for IssueObservation {
    fn from(issue: &gitlab::Issue) -> Self {
        Self {
            iid: issue.iid,
            title: issue.title.clone(),
            description: issue.description.clone(),
            labels: issue.labels.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommentObservation {
    author: String,
    body: String,
}

impl From<&gitlab::Comment> for CommentObservation {
    fn from(comment: &gitlab::Comment) -> Self {
        Self {
            author: comment.author.clone(),
            body: comment.body.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IssueContextObservation {
    issue: IssueObservation,
    comments: Option<Vec<CommentObservation>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QaQuery {
    ShutdownRequested,
    ShaHistory,
    RemoteBranchSha { branch: String },
    OpenIssues,
    IssueComments { issue_iid: u64 },
    CurrentTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QaFact {
    ShutdownRequested(bool),
    ShaHistory(ShaHistory),
    RemoteBranchSha(String),
    OpenIssues(Vec<IssueObservation>),
    IssueComments(Option<Vec<CommentObservation>>),
    CurrentTime(chrono::DateTime<chrono::Utc>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QaAction {
    FetchRepository,
    FetchBranches {
        branches: Vec<String>,
    },
    CheckoutRemoteBranch {
        branch: String,
    },
    WriteQaIssuesContext {
        issues: Vec<IssueContextObservation>,
    },
    WriteOpenIssuesContext {
        issues: Vec<IssueObservation>,
    },
    PrepareChangedFiles {
        base: String,
    },
    InvokeQaModel {
        prompt: String,
        branch: String,
    },
    CloseAnsweredClarification {
        issue_iid: u64,
    },
    CreateClarification {
        title: String,
        description: String,
    },
    AddQaLabel {
        issue_iid: u64,
    },
    AddDoNotImplementLabel {
        issue_iid: u64,
    },
    CreateFinding {
        title: String,
        description: String,
    },
    AddPriorityLabel {
        issue_iid: u64,
        priority: u8,
    },
    AddScopeLabel {
        issue_iid: u64,
    },
    SaveShaHistory(ShaHistory),
}

enum QaOutcome {
    Done,
    Failed(anyhow::Error),
    ChangedFiles(Vec<String>),
    ModelCompleted {
        findings: Vec<RawQaFinding>,
        clarifications: Vec<RawClarification>,
    },
    IssueCreated(u64),
}

trait QaPort {
    fn shutdown_requested(&self) -> bool;
    fn sha_history(&self) -> ShaHistory;
    fn remote_branch_sha(&self, branch: &str) -> Result<String>;
    fn open_issues(&self) -> Result<Vec<IssueObservation>>;
    /// Comments are best effort in both places the legacy cycle read them.
    fn issue_comments(&self, issue_iid: u64) -> Option<Vec<CommentObservation>>;
    fn current_time(&self) -> chrono::DateTime<chrono::Utc>;
    fn execute(&mut self, action: &QaAction) -> QaOutcome;
}

type QaStep = Step<QaQuery, QaAction>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum QaStage {
    ShutdownBeforeCycle,
    FetchRepository,
    FetchBranches,
    ObserveHistory,
    NextBranch,
    ObserveBranchSha,
    CheckoutBranch,
    ShutdownAfterCheckout,
    ObserveIssues,
    NextContextComments,
    ObserveContextComments,
    WriteQaContext,
    WriteOpenContext,
    ShutdownAfterContext,
    PrepareChangedFiles,
    InvokeModel,
    ShutdownAfterModel,
    NextAnsweredClarification,
    ObserveAnswerComments,
    CloseAnsweredClarification,
    NextClarification,
    CreateClarification,
    LabelClarificationQa,
    LabelClarificationDoNotImplement,
    NextFinding,
    ObserveFindingTime,
    CreateFinding,
    LabelFindingPriority,
    LabelFindingQa,
    LabelFindingScope,
    SaveHistory,
    Finish,
}

struct QaMachine<'a> {
    agent_id: &'a str,
    branches: VecDeque<String>,
    all_branches: Vec<String>,
    scope_label: Option<&'a str>,
    qa_issues_path: String,
    open_issues_path: String,
    knowledge_dir: String,
    test_scripts_dir: String,
    stage: QaStage,
    history: ShaHistory,
    selected_branch: Option<(String, String, String)>,
    branch: String,
    prev_sha: String,
    cur_sha: String,
    all_issues: Vec<IssueObservation>,
    qa_issues: Vec<IssueObservation>,
    context_issues: Vec<IssueContextObservation>,
    context_queue: VecDeque<IssueObservation>,
    current_issue: Option<IssueObservation>,
    changed_files: Vec<String>,
    clarifications: VecDeque<ClarificationQuestion>,
    clarification: Option<ClarificationQuestion>,
    findings: VecDeque<QaFinding>,
    finding: Option<QaFinding>,
    created_iid: Option<u64>,
    now: Option<chrono::DateTime<chrono::Utc>>,
}

impl<'a> QaMachine<'a> {
    fn new(
        agent_id: &'a str,
        branches: &[String],
        scope_label: Option<&'a str>,
        input: AnalysisInput<'_>,
    ) -> Self {
        Self {
            agent_id,
            branches: branches.iter().cloned().collect(),
            all_branches: branches.to_vec(),
            scope_label,
            qa_issues_path: input.qa_issues_path.to_string(),
            open_issues_path: input.open_issues_path.to_string(),
            knowledge_dir: input.knowledge_dir.to_string(),
            test_scripts_dir: input.test_scripts_dir.to_string(),
            stage: QaStage::ShutdownBeforeCycle,
            history: ShaHistory::default(),
            selected_branch: None,
            branch: String::new(),
            prev_sha: String::new(),
            cur_sha: String::new(),
            all_issues: Vec::new(),
            qa_issues: Vec::new(),
            context_issues: Vec::new(),
            context_queue: VecDeque::new(),
            current_issue: None,
            changed_files: Vec::new(),
            clarifications: VecDeque::new(),
            clarification: None,
            findings: VecDeque::new(),
            finding: None,
            created_iid: None,
            now: None,
        }
    }

    fn next_step(&mut self) -> QaStep {
        loop {
            match self.stage {
                QaStage::ShutdownBeforeCycle
                | QaStage::ShutdownAfterCheckout
                | QaStage::ShutdownAfterContext
                | QaStage::ShutdownAfterModel => {
                    return QaStep::Observe(QaQuery::ShutdownRequested);
                }
                QaStage::FetchRepository => return QaStep::Act(QaAction::FetchRepository),
                QaStage::FetchBranches => {
                    return QaStep::Act(QaAction::FetchBranches {
                        branches: self.all_branches.clone(),
                    });
                }
                QaStage::ObserveHistory => return QaStep::Observe(QaQuery::ShaHistory),
                QaStage::NextBranch => match self.branches.pop_front() {
                    Some(branch) => {
                        self.branch = branch;
                        self.stage = QaStage::ObserveBranchSha;
                    }
                    None => match self.selected_branch.take() {
                        Some((branch, prev_sha, cur_sha)) => {
                            self.branch = branch;
                            self.prev_sha = prev_sha;
                            self.cur_sha = cur_sha;
                            self.stage = QaStage::CheckoutBranch;
                        }
                        None => self.stage = QaStage::Finish,
                    },
                },
                QaStage::ObserveBranchSha => {
                    return QaStep::Observe(QaQuery::RemoteBranchSha {
                        branch: self.branch.clone(),
                    });
                }
                QaStage::CheckoutBranch => {
                    return QaStep::Act(QaAction::CheckoutRemoteBranch {
                        branch: self.branch.clone(),
                    });
                }
                QaStage::ObserveIssues => return QaStep::Observe(QaQuery::OpenIssues),
                QaStage::NextContextComments => match self.context_queue.pop_front() {
                    Some(issue) => {
                        self.current_issue = Some(issue);
                        self.stage = QaStage::ObserveContextComments;
                    }
                    None => self.stage = QaStage::WriteQaContext,
                },
                QaStage::ObserveContextComments | QaStage::ObserveAnswerComments => {
                    return QaStep::Observe(QaQuery::IssueComments {
                        issue_iid: self.current_issue().iid,
                    });
                }
                QaStage::WriteQaContext => {
                    return QaStep::Act(QaAction::WriteQaIssuesContext {
                        issues: self.context_issues.clone(),
                    });
                }
                QaStage::WriteOpenContext => {
                    return QaStep::Act(QaAction::WriteOpenIssuesContext {
                        issues: self.all_issues.clone(),
                    });
                }
                QaStage::PrepareChangedFiles => {
                    let base = if self.prev_sha.is_empty() {
                        "HEAD~1".to_string()
                    } else {
                        self.prev_sha.clone()
                    };
                    return QaStep::Act(QaAction::PrepareChangedFiles { base });
                }
                QaStage::InvokeModel => {
                    return QaStep::Act(QaAction::InvokeQaModel {
                        prompt: self.prompt(),
                        branch: self.branch.clone(),
                    });
                }
                QaStage::NextAnsweredClarification => match self.context_queue.pop_front() {
                    Some(issue) => {
                        self.current_issue = Some(issue);
                        self.stage = QaStage::ObserveAnswerComments;
                    }
                    None => self.stage = QaStage::NextClarification,
                },
                QaStage::CloseAnsweredClarification => {
                    return QaStep::Act(QaAction::CloseAnsweredClarification {
                        issue_iid: self.current_issue().iid,
                    });
                }
                QaStage::NextClarification => match self.clarifications.pop_front() {
                    Some(clarification) => {
                        self.clarification = Some(clarification);
                        self.stage = QaStage::CreateClarification;
                    }
                    None => self.stage = QaStage::NextFinding,
                },
                QaStage::CreateClarification => {
                    let question = self.clarification();
                    return QaStep::Act(QaAction::CreateClarification {
                        title: question.question.clone(),
                        description: clarification_description(question),
                    });
                }
                QaStage::LabelClarificationQa => {
                    return QaStep::Act(QaAction::AddQaLabel {
                        issue_iid: self.created_iid(),
                    });
                }
                QaStage::LabelClarificationDoNotImplement => {
                    return QaStep::Act(QaAction::AddDoNotImplementLabel {
                        issue_iid: self.created_iid(),
                    });
                }
                QaStage::NextFinding => match self.findings.pop_front() {
                    Some(finding)
                        if !finding.severity.is_non_trivial()
                            || self
                                .qa_issues
                                .iter()
                                .any(|issue| issue.title == finding.title) =>
                    {
                        continue;
                    }
                    Some(finding) => {
                        self.finding = Some(finding);
                        self.stage = QaStage::ObserveFindingTime;
                    }
                    None => self.stage = QaStage::SaveHistory,
                },
                QaStage::ObserveFindingTime => return QaStep::Observe(QaQuery::CurrentTime),
                QaStage::CreateFinding => {
                    let finding = self.finding();
                    return QaStep::Act(QaAction::CreateFinding {
                        title: finding.title.clone(),
                        description: finding_description(
                            finding,
                            &self.branch,
                            &self.cur_sha,
                            self.now.expect("finding time observed before creation"),
                        ),
                    });
                }
                QaStage::LabelFindingPriority => {
                    return QaStep::Act(QaAction::AddPriorityLabel {
                        issue_iid: self.created_iid(),
                        priority: self.finding().severity.priority(),
                    });
                }
                QaStage::LabelFindingQa => {
                    return QaStep::Act(QaAction::AddQaLabel {
                        issue_iid: self.created_iid(),
                    });
                }
                QaStage::LabelFindingScope => {
                    if self.scope_label.is_none() {
                        self.stage = QaStage::NextFinding;
                        continue;
                    }
                    return QaStep::Act(QaAction::AddScopeLabel {
                        issue_iid: self.created_iid(),
                    });
                }
                QaStage::SaveHistory => {
                    self.history
                        .0
                        .insert(self.branch.clone(), self.cur_sha.clone());
                    return QaStep::Act(QaAction::SaveShaHistory(self.history.clone()));
                }
                QaStage::Finish => return QaStep::Finish,
            }
        }
    }

    fn current_issue(&self) -> &IssueObservation {
        self.current_issue
            .as_ref()
            .expect("issue stage has an issue")
    }

    fn clarification(&self) -> &ClarificationQuestion {
        self.clarification
            .as_ref()
            .expect("clarification stage has a question")
    }

    fn finding(&self) -> &QaFinding {
        self.finding.as_ref().expect("finding stage has a finding")
    }

    fn created_iid(&self) -> u64 {
        self.created_iid
            .expect("label stage follows issue creation")
    }

    fn prompt(&self) -> String {
        build_qa_prompt(
            self.agent_id,
            &GitContext {
                branch: &self.branch,
                prev_sha: &self.prev_sha,
                cur_sha: &self.cur_sha,
                changed_files: &self.changed_files,
            },
            &AnalysisInput {
                qa_issues_path: &self.qa_issues_path,
                open_issues_path: &self.open_issues_path,
                knowledge_dir: &self.knowledge_dir,
                test_scripts_dir: &self.test_scripts_dir,
            },
        )
    }

    fn apply_fact(&mut self, fact: Result<QaFact>) -> Result<()> {
        match (&self.stage, fact) {
            (QaStage::ShutdownBeforeCycle, Ok(QaFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    QaStage::Finish
                } else {
                    QaStage::FetchRepository
                };
            }
            (QaStage::ShutdownAfterCheckout, Ok(QaFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    QaStage::Finish
                } else {
                    QaStage::ObserveIssues
                };
            }
            (QaStage::ShutdownAfterContext, Ok(QaFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    QaStage::Finish
                } else {
                    QaStage::PrepareChangedFiles
                };
            }
            (QaStage::ShutdownAfterModel, Ok(QaFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    QaStage::Finish
                } else {
                    self.context_queue = self
                        .qa_issues
                        .iter()
                        .filter(|issue| {
                            issue
                                .labels
                                .iter()
                                .any(|label| label == DO_NOT_IMPLEMENT_LABEL)
                        })
                        .cloned()
                        .collect();
                    QaStage::NextAnsweredClarification
                };
            }
            (QaStage::ObserveHistory, Ok(QaFact::ShaHistory(history))) => {
                self.history = history;
                self.stage = QaStage::NextBranch;
            }
            (QaStage::ObserveBranchSha, Ok(QaFact::RemoteBranchSha(sha))) => {
                let previous = self
                    .history
                    .0
                    .get(&self.branch)
                    .cloned()
                    .unwrap_or_default();
                if previous != sha && self.selected_branch.is_none() {
                    self.selected_branch = Some((self.branch.clone(), previous, sha));
                }
                self.stage = QaStage::NextBranch;
            }
            (QaStage::ObserveIssues, Ok(QaFact::OpenIssues(issues))) => {
                self.all_issues = issues;
                self.qa_issues = self
                    .all_issues
                    .iter()
                    .filter(|issue| issue.labels.iter().any(|label| label == QA_LABEL))
                    .cloned()
                    .collect();
                self.context_queue = self.qa_issues.iter().cloned().collect();
                self.stage = QaStage::NextContextComments;
            }
            (QaStage::ObserveContextComments, Ok(QaFact::IssueComments(comments))) => {
                self.context_issues.push(IssueContextObservation {
                    issue: self.current_issue().clone(),
                    comments,
                });
                self.stage = QaStage::NextContextComments;
            }
            (QaStage::ObserveAnswerComments, Ok(QaFact::IssueComments(comments))) => {
                let answered = comments
                    .unwrap_or_default()
                    .iter()
                    .any(|comment| !is_potlatch_author(&comment.author));
                self.stage = if answered {
                    QaStage::CloseAnsweredClarification
                } else {
                    QaStage::NextAnsweredClarification
                };
            }
            (QaStage::ObserveFindingTime, Ok(QaFact::CurrentTime(now))) => {
                self.now = Some(now);
                self.stage = QaStage::CreateFinding;
            }
            (stage, Ok(fact)) => anyhow::bail!("qa port answered {stage:?} with {fact:?}"),
            (_, Err(error)) => return Err(error),
        }
        Ok(())
    }

    fn apply_outcome(&mut self, outcome: QaOutcome) -> Result<()> {
        match (&self.stage, outcome) {
            (QaStage::FetchRepository, QaOutcome::Done) => self.stage = QaStage::FetchBranches,
            (QaStage::FetchBranches, QaOutcome::Done) => self.stage = QaStage::ObserveHistory,
            (QaStage::CheckoutBranch, QaOutcome::Done) => {
                self.stage = QaStage::ShutdownAfterCheckout
            }
            (QaStage::WriteQaContext, QaOutcome::Done) => self.stage = QaStage::WriteOpenContext,
            (QaStage::WriteOpenContext, QaOutcome::Done) => {
                self.stage = QaStage::ShutdownAfterContext
            }
            (QaStage::PrepareChangedFiles, QaOutcome::ChangedFiles(files)) => {
                self.changed_files = files;
                self.stage = QaStage::InvokeModel;
            }
            (
                QaStage::InvokeModel,
                QaOutcome::ModelCompleted {
                    findings,
                    clarifications,
                },
            ) => {
                self.findings = normalize_qa_findings(findings).into();
                self.clarifications = normalize_clarifications(clarifications).into();
                self.stage = QaStage::ShutdownAfterModel;
            }
            (QaStage::CloseAnsweredClarification, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::NextAnsweredClarification
            }
            (QaStage::CreateClarification, QaOutcome::IssueCreated(iid)) => {
                self.created_iid = Some(iid);
                self.stage = QaStage::LabelClarificationQa;
            }
            (QaStage::CreateClarification, QaOutcome::Failed(_)) => {
                self.stage = QaStage::NextClarification
            }
            (QaStage::LabelClarificationQa, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::LabelClarificationDoNotImplement
            }
            (QaStage::LabelClarificationDoNotImplement, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::NextClarification
            }
            (QaStage::CreateFinding, QaOutcome::IssueCreated(iid)) => {
                self.created_iid = Some(iid);
                self.stage = QaStage::LabelFindingPriority;
            }
            (QaStage::CreateFinding, QaOutcome::Failed(_)) => self.stage = QaStage::NextFinding,
            (QaStage::LabelFindingPriority, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::LabelFindingQa
            }
            (QaStage::LabelFindingQa, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::LabelFindingScope
            }
            (QaStage::LabelFindingScope, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::NextFinding
            }
            (QaStage::SaveHistory, QaOutcome::Done | QaOutcome::Failed(_)) => {
                self.stage = QaStage::Finish
            }
            (_, QaOutcome::Failed(error)) => return Err(error),
            (stage, _) => anyhow::bail!("qa port reported an unexpected outcome for {stage:?}"),
        }
        Ok(())
    }
}

fn observe_qa(port: &dyn QaPort, query: &QaQuery) -> Result<QaFact> {
    Ok(match query {
        QaQuery::ShutdownRequested => QaFact::ShutdownRequested(port.shutdown_requested()),
        QaQuery::ShaHistory => QaFact::ShaHistory(port.sha_history()),
        QaQuery::RemoteBranchSha { branch } => {
            QaFact::RemoteBranchSha(port.remote_branch_sha(branch)?)
        }
        QaQuery::OpenIssues => QaFact::OpenIssues(port.open_issues()?),
        QaQuery::IssueComments { issue_iid } => {
            QaFact::IssueComments(port.issue_comments(*issue_iid))
        }
        QaQuery::CurrentTime => QaFact::CurrentTime(port.current_time()),
    })
}

fn run_qa_cycle(machine: &mut QaMachine, port: &mut dyn QaPort) -> Result<()> {
    loop {
        match machine.next_step() {
            Step::Observe(query) => machine.apply_fact(observe_qa(port, &query))?,
            Step::Act(action) => machine.apply_outcome(port.execute(&action))?,
            Step::Finish => return Ok(()),
        }
    }
}

struct LiveQaPort<'a> {
    state: &'a AgentState<'a>,
    model: &'a AgentModel,
    scope_label: Option<&'a str>,
}

fn qa_required(result: Result<()>) -> QaOutcome {
    match result {
        Ok(()) => QaOutcome::Done,
        Err(error) => QaOutcome::Failed(error),
    }
}

impl QaPort for LiveQaPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.model.shutdown().load(Ordering::SeqCst)
    }

    fn sha_history(&self) -> ShaHistory {
        load_sha_history(self.state)
    }

    fn remote_branch_sha(&self, branch: &str) -> Result<String> {
        self.state.git_repo.remote_short_sha(branch)
    }

    fn open_issues(&self) -> Result<Vec<IssueObservation>> {
        Ok(self
            .state
            .glab
            .list_issues()?
            .iter()
            .map(IssueObservation::from)
            .collect())
    }

    fn issue_comments(&self, issue_iid: u64) -> Option<Vec<CommentObservation>> {
        match self.state.glab.get_issue_comments(issue_iid) {
            Ok(comments) => Some(comments.iter().map(CommentObservation::from).collect()),
            Err(error) => {
                warn!("Failed to fetch comments for issue #{issue_iid}: {error}");
                None
            }
        }
    }

    fn current_time(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now()
    }

    fn execute(&mut self, action: &QaAction) -> QaOutcome {
        match action {
            QaAction::FetchRepository => qa_required(self.state.git_repo.fetch()),
            QaAction::FetchBranches { branches } => {
                let branches: Vec<&str> = branches.iter().map(String::as_str).collect();
                qa_required(self.state.git_repo.fetch_branches(&branches))
            }
            QaAction::CheckoutRemoteBranch { branch } => {
                qa_required(self.state.git_repo.checkout_remote_branch(branch))
            }
            QaAction::WriteQaIssuesContext { issues } => {
                qa_required(write_qa_issues_context(self.state, issues))
            }
            QaAction::WriteOpenIssuesContext { issues } => {
                qa_required(write_open_issues_context(self.state, issues))
            }
            QaAction::PrepareChangedFiles { base } => QaOutcome::ChangedFiles(
                self.state
                    .git_repo
                    .changed_files_since(base)
                    .unwrap_or_default(),
            ),
            QaAction::InvokeQaModel { prompt, branch } => {
                match self.model.complete_typed::<QaOutput>(
                    prompt,
                    &InvokeOptions {
                        activity_label: Some(format!(
                            "{} QA analysis on {}",
                            self.state.agent_id, branch
                        )),
                        ..InvokeOptions::default()
                    },
                ) {
                    Ok(completion) => QaOutcome::ModelCompleted {
                        findings: completion.output.findings,
                        clarifications: completion.output.clarifications,
                    },
                    Err(error) => QaOutcome::Failed(error),
                }
            }
            QaAction::CloseAnsweredClarification { issue_iid } => {
                qa_required(self.state.glab.close_issue(*issue_iid))
            }
            QaAction::CreateClarification { title, description }
            | QaAction::CreateFinding { title, description } => {
                match self.state.glab.create_issue(title, description) {
                    Ok(iid) => QaOutcome::IssueCreated(iid),
                    Err(error) => QaOutcome::Failed(error),
                }
            }
            QaAction::AddQaLabel { issue_iid } => {
                qa_required(self.state.glab.add_issue_label(*issue_iid, QA_LABEL))
            }
            QaAction::AddDoNotImplementLabel { issue_iid } => qa_required(
                self.state
                    .glab
                    .add_issue_label(*issue_iid, DO_NOT_IMPLEMENT_LABEL),
            ),
            QaAction::AddPriorityLabel {
                issue_iid,
                priority,
            } => qa_required(
                self.state
                    .glab
                    .add_issue_label(*issue_iid, &gitlab::priority_label(*priority)),
            ),
            QaAction::AddScopeLabel { issue_iid } => {
                let Some(label) = self.scope_label else {
                    return QaOutcome::Done;
                };
                qa_required(self.state.glab.add_issue_label(*issue_iid, label))
            }
            QaAction::SaveShaHistory(history) => qa_required(save_sha_history(self.state, history)),
        }
    }
}

fn qa_cycle(
    state: &AgentState,
    config: &QaConfig,
    model: &AgentModel,
    scope_label: Option<&str>,
) -> Result<()> {
    let qa_issues_path = state.qa_issues_path();
    let open_issues_path = state.open_issues_path();
    let knowledge_dir = state.knowledge_dir();
    let test_scripts_dir = state.test_scripts_dir();
    let mut machine = QaMachine::new(
        state.agent_id,
        &config.branches,
        scope_label,
        AnalysisInput {
            qa_issues_path: &qa_issues_path.to_string_lossy(),
            open_issues_path: &open_issues_path.to_string_lossy(),
            knowledge_dir: &knowledge_dir.to_string_lossy(),
            test_scripts_dir: &test_scripts_dir.to_string_lossy(),
        },
    );
    let mut port = LiveQaPort {
        state,
        model,
        scope_label,
    };
    run_qa_cycle(&mut machine, &mut port)
}

fn clarification_description(question: &ClarificationQuestion) -> String {
    format!(
        "{}\n\n---\n*This is a QA clarification question. Please answer in a comment. The QA agent will pick up answers automatically.*",
        question.context
    )
}

fn finding_description(
    finding: &QaFinding,
    branch: &str,
    cur_sha: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    format!(
        "**Severity:** {}\n**File:** {}\n**Branch:** {}\n**Commit:** {}\n\n{}\n\n---\n*Found by QA agent on {}*",
        finding.severity.as_str(),
        if finding.file.is_empty() {
            "n/a"
        } else {
            &finding.file
        },
        branch,
        cur_sha,
        finding.description,
        now.format("%Y-%m-%d %H:%M UTC")
    )
}

/// Render the QA-labeled issues (with comments) to the agent's context file.
/// The model reads this file directly; it never calls any tool to fetch GitLab.
fn write_qa_issues_context(state: &AgentState, issues: &[IssueContextObservation]) -> Result<()> {
    let mut out = String::new();
    out.push_str("# QA-labeled open issues\n\n");
    out.push_str(&format!(
        "_Fetched by the QA harness on {}. Read this file to learn what to test and to see answers to your past clarification questions._\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));

    if issues.is_empty() {
        out.push_str(
            "_No open QA-labeled issues. Test the changes in this commit as an end user._\n",
        );
    } else {
        for context in issues {
            let issue = &context.issue;
            out.push_str(&format!("## #{} — {}\n", issue.iid, issue.title));
            let labels = if issue.labels.is_empty() {
                "(none)".to_string()
            } else {
                issue.labels.join(", ")
            };
            out.push_str(&format!("**Labels:** {labels}\n\n"));
            if !issue.description.trim().is_empty() {
                out.push_str(issue.description.trim());
                out.push_str("\n\n");
            }
            match &context.comments {
                Some(comments) if !comments.is_empty() => {
                    out.push_str("### Comments\n\n");
                    for c in comments {
                        let author = if is_potlatch_author(&c.author) {
                            format!("{} (potlatch)", c.author)
                        } else {
                            c.author.clone()
                        };
                        out.push_str(&format!("**{author}:**\n{}\n\n", c.body.trim()));
                    }
                }
                Some(_) => {}
                None => {
                    out.push_str("### Comments\n\n_(failed to load comments)_\n\n");
                }
            }
            out.push_str("---\n\n");
        }
    }

    let path = state.qa_issues_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create sessions dir {}", parent.display()))?;
    }
    fs::write(&path, &out).with_context(|| format!("Failed to write {}", path.display()))?;
    debug!(
        "{}: Wrote QA issues context to {} ({} issue(s))",
        state.agent_id,
        path.display(),
        issues.len()
    );
    Ok(())
}

/// Render a compact, dedup-oriented listing of *all* open project issues (not
/// just QA-labeled ones) to a separate file. The model reads this to check
/// whether a finding duplicates an issue that's already tracked — by the QA
/// agent itself, another agent, or a human — before reporting it.
///
/// Only the title and a one-line description preview are included per issue;
/// acceptance-criteria detail lives in the QA-issues file. Keeping this view
/// compact lets the model scan the full open-issue set cheaply for duplicates
/// without a second copy of every issue body.
fn write_open_issues_context(state: &AgentState, issues: &[IssueObservation]) -> Result<()> {
    let mut out = String::new();
    out.push_str("# All open project issues (for duplicate checking)\n\n");
    out.push_str(&format!(
        "_Fetched by the QA harness on {}. Read this file before reporting a finding to check whether an issue with the same problem is already open. Do not report a finding that duplicates an issue listed here — reference the existing issue instead._\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    ));

    if issues.is_empty() {
        out.push_str("_No open issues. Every finding you report will be a new issue._\n");
    } else {
        out.push_str(&format!("_{} open issue(s) total._\n\n", issues.len()));
        for issue in issues {
            let labels = if issue.labels.is_empty() {
                String::new()
            } else {
                format!(" `[{}]`", issue.labels.join(", "))
            };
            let preview: String = issue
                .description
                .trim()
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(160)
                .collect();
            let preview = if preview.is_empty() {
                String::new()
            } else {
                format!(" — {preview}")
            };
            out.push_str(&format!(
                "- #{iid} — {title}{labels}{preview}\n",
                iid = issue.iid,
                title = issue.title
            ));
        }
    }

    let path = state.open_issues_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create sessions dir {}", parent.display()))?;
    }
    fs::write(&path, &out).with_context(|| format!("Failed to write {}", path.display()))?;
    debug!(
        "{}: Wrote open issues context to {} ({} issue(s))",
        state.agent_id,
        path.display(),
        issues.len()
    );
    Ok(())
}

/// Whether a comment author is a potlatch-owned bot (potlatch-* or qa-* agents).
fn is_potlatch_author(author: &str) -> bool {
    author.starts_with("potlatch") || author.starts_with("qa-")
}

// ---------------------------------------------------------------------------
// SHA history persistence
// ---------------------------------------------------------------------------

fn load_sha_history(state: &AgentState) -> ShaHistory {
    // Missing, unreadable, and corrupt history have historically reset QA
    // history: tolerant, warn and default rather than failing the cycle.
    match crate::agents::state::StateStore::new(state.sha_history_path()).load() {
        Ok(history) => history.unwrap_or_default(),
        Err(error) => {
            warn!("Failed to load SHA history: {:#}", error);
            ShaHistory::default()
        }
    }
}

fn save_sha_history(state: &AgentState, history: &ShaHistory) -> Result<()> {
    let store = crate::agents::state::StateStore::new(state.sha_history_path());
    store.save(history)
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

struct GitContext<'a> {
    branch: &'a str,
    prev_sha: &'a str,
    cur_sha: &'a str,
    changed_files: &'a [String],
}

struct AnalysisInput<'a> {
    qa_issues_path: &'a str,
    open_issues_path: &'a str,
    knowledge_dir: &'a str,
    test_scripts_dir: &'a str,
}

fn build_qa_prompt(agent_id: &str, git: &GitContext, input: &AnalysisInput) -> String {
    let changed_files_str = if git.changed_files.is_empty() {
        "(could not determine)"
    } else {
        &git.changed_files.join(", ")
    };

    let branch = git.branch;
    let prev_sha = git.prev_sha;
    let cur_sha = git.cur_sha;
    let qa_issues_path = input.qa_issues_path;
    let open_issues_path = input.open_issues_path;
    let knowledge_dir = input.knowledge_dir;
    let test_scripts_dir = input.test_scripts_dir;

    format!(
        r##"You are a QA agent ({agent_id}) for this project. A new commit landed on branch {branch}.

## Your Role

You are an **end-user tester**, not a code reviewer. Your job is to verify that the system's features work end-to-end from a real user's perspective. Test the *running system* through its external surfaces:
- HTTP APIs (via `curl` or `python3` + `urllib`/`requests`)
- CLI invocations the way a real user would call them
- Configuration loading and behavior from a user's perspective
- Data flow through the system as observed externally

**Your testing is organized around user-facing functionality, not around issues.** You maintain a map of the system's user-facing functions, and a test plan with cases for each function. GitLab issues are a source of context — they describe known bugs, acceptance criteria, and testing instructions that you fold into your functionality-based test plan — but they do not dictate your test organization. You test the functionality a commit touches, and you regression-test the functions you have tracked across prior cycles.

You may use `read`, `grep`, and `glob` **only** to discover how to exercise the system (which endpoints exist, which CLI flags are available, how to invoke the binary) in service of testing a function. Never assert on internal code paths or read the project's own tests to judge correctness; that is the developer's responsibility, not yours.

## File Locations

The harness has prepared the following absolute paths for you. Use `read` and `write` with `outside_cwd: true` to access them (they live outside the working directory).

- **Read** QA issues (harness-written; do not modify): `{qa_issues_path}`
  This file lists every open QA-labeled issue with its description and comments. Read it to gather acceptance criteria, known bugs, and testing instructions for the functions in scope this cycle. It also contains answers to clarification questions you have asked previously. Issues carrying the `do-not-implement` label are clarification threads (questions for humans, plus their answers) — absorb their answers, do not treat them as features to test. Fold each issue's acceptance criteria into the relevant function's test cases in your test plan; do not create per-issue test scripts.
- **Read** all open issues (harness-written; do not modify): `{open_issues_path}`
  A compact listing of every open project issue — not just QA-labeled ones — with its iid, title, labels, and a one-line description preview. Read it before reporting a finding to check whether an issue is already tracked (by the QA agent, another agent, or a human). Do not report a finding that duplicates an issue listed here. This file is refreshed from GitLab at the start of every QA run, so it reflects the current open-issue set.
- **Read/write** your knowledge (persists across runs): `{knowledge_dir}/`
  - `functionality.md` — **your primary artifact.** The map of every user-facing function the system exposes, grouped by external surface (e.g. HTTP API endpoints, CLI commands, data flows). For each function: its name, how an end user invokes it, what it should do, and the test cases that cover it. Keep this comprehensive and up to date — it is the backbone of your test plan.
  - `test_cases.md` — your test plan, organized **by function** (not by issue). Each function section lists its test cases with steps, expected results, and the script that runs them. When a QA-labeled issue provides acceptance criteria for a function, add them as test cases under that function's section — do not create a separate per-issue section. This keeps your regression suite function-oriented.
  - `qa_context.md` — running notes, conventions, environment quirks
  - `requirements.md` — requirements you have discovered
  Read them at the start of each run to recall what you learned. Update them with `write` (`outside_cwd: true`, full overwrite) or `edit` (`outside_cwd: true`, targeted changes) when you discover something new.
- **Read/write/run** your test scripts: `{test_scripts_dir}/`
  Write Python scripts here via `write` (`outside_cwd: true`), **named by the function they test** (e.g. `{test_scripts_dir}/test_p4_search.py`, `{test_scripts_dir}/test_ukb_document_lifecycle.py`), not by issue number or commit SHA. A single script covers all cases for one function — happy path, edge cases, error handling, and regression scenarios as test functions within it. Update existing scripts with `edit` (`outside_cwd: true`) for targeted changes. Run them yourself via `shell` with `outside_cwd: true`: `python3 {test_scripts_dir}/test_p4_search.py`. The harness does not load, save, or run test scripts — you do.
- **Read** `qa.md` at the repo root (relative path, normal `read`) for project-specific QA instructions, if present.

## Recent Changes

- Branch: {branch}
- Previous SHA: {prev_sha}
- Current SHA: {cur_sha}
- Changed files: {changed_files_str}

Use `git log --oneline -5` and the changed files above to identify which user-facing functions this commit touches, then look up those functions in your `functionality.md` map and in `{qa_issues_path}` for any acceptance criteria or known bugs related to them.

## Hard Rules

1. **Read the QA issues file and your knowledge files first.** Before any other action, read `{qa_issues_path}` with `read` (`outside_cwd: true`) to gather acceptance criteria and testing instructions for the functions in scope. Also read your knowledge files under `{knowledge_dir}/` to recall your functionality map and test plan.
2. **Never fetch GitLab yourself.** Do not call `glab`, `fetch`, or any tool to read issues/MRs/commits from GitLab. The harness has already gathered the open QA-labeled issues into `{qa_issues_path}` and the full open-issue listing into `{open_issues_path}` — read those files.
3. **Never report a duplicate finding.** Before reporting a finding, read `{open_issues_path}` and check whether an open issue already describes the same problem (by the QA agent, another agent, or a human). If it does, do not report that finding — the issue is already tracked. Compare by the underlying problem, not just exact-title match: a finding about "login returns 500 on empty password" duplicates an issue titled "Auth API crashes on malformed input" even though the wording differs. Only report a finding if no open issue covers the same root cause.
4. **Never mutate GitLab.** Do not post comments or create/edit issues via tools. The harness creates GitLab issues from your structured result.
5. **Never modify the working directory.** Do not use `write` or `edit` to create, modify, or delete anything inside the checked-out repo. Do not run `cd`, `git checkout`, `git commit`, or any command that mutates the repo tree.
6. **Never run unit tests, build commands, or liveness/ops endpoints.** Do not run `cargo test`, `go test`, `go build`, `go vet`, `pytest`, `npm test`, or similar — these are the developer's responsibility and redundant for end-user testing; build commands also write artifacts into the repo. Do not test ops/liveness/health endpoints (`/ping`, `/monitor`, `/health`, `/metrics`, `/ready`, etc.) unless a QA-labeled issue explicitly asks you to — they are not functionality and testing them is noise. Allowed commands: `curl`/`python3` against real functionality APIs, invoking an already-built CLI binary the way a user would, and `python3` to run your own test scripts.

## Your Task

1. **Read `{qa_issues_path}`** (`read`, `outside_cwd: true`) to gather acceptance criteria, known bugs, and testing instructions for the functions in scope this cycle. Also read `{open_issues_path}` (`read`, `outside_cwd: true`) to load the full open-issue set you'll dedup against when reporting findings.
2. Read your knowledge files under `{knowledge_dir}/` — especially `functionality.md` and `test_cases.md` — to recall your functionality map and test plan.
3. **Maintain your functionality map.** Using the changed files + `git log`, identify which user-facing functions this commit touches. Ensure each is represented in `functionality.md` with its external surface, how to invoke it, and what it should do. If you discovered a new function, add it. If a function's behavior changed, update its description.
4. **Maintain your test plan.** For each touched function, ensure `test_cases.md` has a section with test cases covering: the happy path, edge cases, error handling, and any acceptance criteria from QA-labeled issues related to that function. Add new cases as needed. Fold issue-specific acceptance criteria into the function's section — do not create per-issue sections or per-issue scripts.
5. **Test the touched functions.** For each touched function, run its test cases end-to-end as an end user against the **real functionality APIs**. Author or update Python scripts under `{test_scripts_dir}/` (named by function, not by issue) and run them via `python3` (`shell` with `outside_cwd: true`). Each test must assert the function's expected behavior as observed externally.
6. **Run regression tests for all tracked functions.** After testing the new commit's functions, re-run your test scripts for all functions in `functionality.md` — not just the ones this commit touched. A commit that delivers one feature can break an unrelated feature; regressions are the whole point of maintaining a function-oriented test suite. If a previously-passing test case now fails, report it as a finding (severity: high or critical). If a function was removed or its test cases are no longer relevant, note that in `test_cases.md` (mark it retired) but do not report it as a finding. If `functionality.md`/`test_cases.md` are empty or do not yet exist (first run), build them now from what you discover, then test.
7. Before reporting any finding, check it against `{open_issues_path}`. Skip a finding if an open issue already covers the same root cause — do not file a duplicate.
8. Update your knowledge files under `{knowledge_dir}/` with anything new you learned, including which functions you tested and their results. When recording results, record what you actually tested (the functionality APIs and the outcome), not a generic "regression PASS" label.
9. If a function's expected behavior is ambiguous and you cannot proceed without guessing, emit a clarification question instead of guessing.

Report genuine bugs, security vulnerabilities, race conditions, correctness issues, and incomplete feature implementations you encounter **while testing as an end user**. Each finding must be actionable: a real problem that could cause incorrect behavior, data loss, a security breach, instability, or a feature that doesn't actually work as intended. Do NOT report stylistic preferences, cosmetic issues, or minor nitpicks. TODO/FIXME comments and `unimplemented!()`/`todo!()` markers are acceptable — do not flag their mere presence; only flag when the surrounding feature is functionally broken as observed from the outside.

Only critical, high, and medium findings will be created as GitLab issues; low-severity findings are logged but not tracked. Findings and clarification questions may both be reported in the same run."##
    )
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use chrono::TimeZone;

    use super::*;
    use crate::core::agent::schema::conformance;

    struct FakeQaPort {
        trace: RefCell<Vec<String>>,
        shutdown: RefCell<VecDeque<bool>>,
        history: ShaHistory,
        shas: HashMap<String, String>,
        issues: Vec<IssueObservation>,
        comments: HashMap<u64, Option<Vec<CommentObservation>>>,
        findings: Vec<RawQaFinding>,
        clarifications: Vec<RawClarification>,
        next_iid: Cell<u64>,
        fail_required: Option<&'static str>,
        fail_best_effort: Vec<&'static str>,
        saved: RefCell<Option<ShaHistory>>,
    }

    impl FakeQaPort {
        fn successful() -> Self {
            Self {
                trace: RefCell::new(Vec::new()),
                shutdown: RefCell::new(VecDeque::new()),
                history: ShaHistory::default(),
                shas: HashMap::from([("main".to_string(), "new123".to_string())]),
                issues: Vec::new(),
                comments: HashMap::new(),
                findings: Vec::new(),
                clarifications: Vec::new(),
                next_iid: Cell::new(100),
                fail_required: None,
                fail_best_effort: Vec::new(),
                saved: RefCell::new(None),
            }
        }

        fn record(&self, event: impl Into<String>) {
            self.trace.borrow_mut().push(event.into());
        }

        fn failed(&self, name: &'static str) -> bool {
            self.fail_required == Some(name) || self.fail_best_effort.contains(&name)
        }
    }

    impl QaPort for FakeQaPort {
        fn shutdown_requested(&self) -> bool {
            self.record("observe:shutdown");
            self.shutdown.borrow_mut().pop_front().unwrap_or(false)
        }

        fn sha_history(&self) -> ShaHistory {
            self.record("observe:history");
            self.history.clone()
        }

        fn remote_branch_sha(&self, branch: &str) -> Result<String> {
            self.record(format!("observe:sha:{branch}"));
            Ok(self.shas.get(branch).cloned().unwrap())
        }

        fn open_issues(&self) -> Result<Vec<IssueObservation>> {
            self.record("observe:issues");
            Ok(self.issues.clone())
        }

        fn issue_comments(&self, issue_iid: u64) -> Option<Vec<CommentObservation>> {
            self.record(format!("observe:comments:{issue_iid}"));
            self.comments.get(&issue_iid).cloned().unwrap_or_default()
        }

        fn current_time(&self) -> chrono::DateTime<chrono::Utc> {
            self.record("observe:time");
            chrono::Utc.with_ymd_and_hms(2026, 8, 17, 12, 0, 0).unwrap()
        }

        fn execute(&mut self, action: &QaAction) -> QaOutcome {
            let (name, event) = match action {
                QaAction::FetchRepository => ("fetch", "act:fetch".to_string()),
                QaAction::FetchBranches { .. } => ("fetch_branches", "act:fetch_branches".into()),
                QaAction::CheckoutRemoteBranch { branch } => {
                    ("checkout", format!("act:checkout:{branch}"))
                }
                QaAction::WriteQaIssuesContext { .. } => ("write_qa", "act:write_qa".into()),
                QaAction::WriteOpenIssuesContext { .. } => ("write_open", "act:write_open".into()),
                QaAction::PrepareChangedFiles { base } => {
                    ("changed_files", format!("act:changed_files:{base}"))
                }
                QaAction::InvokeQaModel { .. } => ("model", "act:model".into()),
                QaAction::CloseAnsweredClarification { issue_iid } => {
                    ("close", format!("act:close:{issue_iid}"))
                }
                QaAction::CreateClarification { title, .. } => (
                    "create_clarification",
                    format!("act:create_clarification:{title}"),
                ),
                QaAction::AddQaLabel { issue_iid } => ("qa_label", format!("act:qa:{issue_iid}")),
                QaAction::AddDoNotImplementLabel { issue_iid } => {
                    ("dni_label", format!("act:dni:{issue_iid}"))
                }
                QaAction::CreateFinding { title, .. } => {
                    ("create_finding", format!("act:create_finding:{title}"))
                }
                QaAction::AddPriorityLabel {
                    issue_iid,
                    priority,
                } => (
                    "priority_label",
                    format!("act:priority:{issue_iid}:{priority}"),
                ),
                QaAction::AddScopeLabel { issue_iid } => {
                    ("scope_label", format!("act:scope:{issue_iid}"))
                }
                QaAction::SaveShaHistory(_) => ("save", "act:save".into()),
            };
            self.record(event);
            if self.failed(name) {
                return QaOutcome::Failed(anyhow::anyhow!("injected {name} failure"));
            }
            match action {
                QaAction::PrepareChangedFiles { .. } => {
                    QaOutcome::ChangedFiles(vec!["src/lib.rs".to_string()])
                }
                QaAction::InvokeQaModel { .. } => QaOutcome::ModelCompleted {
                    findings: self.findings.clone(),
                    clarifications: self.clarifications.clone(),
                },
                QaAction::CreateClarification { .. } | QaAction::CreateFinding { .. } => {
                    let iid = self.next_iid.get();
                    self.next_iid.set(iid + 1);
                    QaOutcome::IssueCreated(iid)
                }
                QaAction::SaveShaHistory(history) => {
                    *self.saved.borrow_mut() = Some(history.clone());
                    QaOutcome::Done
                }
                _ => QaOutcome::Done,
            }
        }
    }

    fn qa_issue(iid: u64, title: &str, clarification: bool) -> IssueObservation {
        let mut labels = vec![QA_LABEL.to_string()];
        if clarification {
            labels.push(DO_NOT_IMPLEMENT_LABEL.to_string());
        }
        IssueObservation {
            iid,
            title: title.to_string(),
            description: "existing issue".to_string(),
            labels,
        }
    }

    fn run_fake(port: &mut FakeQaPort) -> Result<()> {
        let mut machine = QaMachine::new(
            "qa-0",
            &["main".to_string()],
            Some("scope::test"),
            AnalysisInput {
                qa_issues_path: "/sessions/qa.md",
                open_issues_path: "/sessions/open.md",
                knowledge_dir: "/sessions/knowledge",
                test_scripts_dir: "/sessions/scripts",
            },
        );
        run_qa_cycle(&mut machine, port)
    }

    #[test]
    fn qa_machine_records_full_ordered_trace_and_uses_generated_iids() {
        let mut port = FakeQaPort::successful();
        port.issues = vec![qa_issue(7, "Old question", true)];
        port.comments.insert(
            7,
            Some(vec![CommentObservation {
                author: "alice".into(),
                body: "Use staging".into(),
            }]),
        );
        port.clarifications = vec![RawClarification {
            question: "Which tenant?".into(),
            context: "Needed for coverage".into(),
        }];
        port.findings = vec![RawQaFinding {
            title: "Search fails".into(),
            description: "Search returned 500".into(),
            severity: Severity::High,
            file: "src/search.rs:9".into(),
        }];

        run_fake(&mut port).unwrap();

        assert_eq!(
            *port.trace.borrow(),
            vec![
                "observe:shutdown",
                "act:fetch",
                "act:fetch_branches",
                "observe:history",
                "observe:sha:main",
                "act:checkout:main",
                "observe:shutdown",
                "observe:issues",
                "observe:comments:7",
                "act:write_qa",
                "act:write_open",
                "observe:shutdown",
                "act:changed_files:HEAD~1",
                "act:model",
                "observe:shutdown",
                // Answered clarification closure is complete before either
                // kind of issue creation begins.
                "observe:comments:7",
                "act:close:7",
                "act:create_clarification:Which tenant?",
                "act:qa:100",
                "act:dni:100",
                "observe:time",
                "act:create_finding:Search fails",
                "act:priority:101:2",
                "act:qa:101",
                "act:scope:101",
                "act:save",
            ]
        );
        assert_eq!(
            port.saved.borrow().as_ref().unwrap().0.get("main"),
            Some(&"new123".to_string())
        );
    }

    #[test]
    fn qa_machine_aborts_required_failures_and_never_saves_sha() {
        for failure in ["fetch", "write_qa", "write_open", "model"] {
            let mut port = FakeQaPort::successful();
            port.fail_required = Some(failure);
            assert!(run_fake(&mut port).is_err(), "{failure}");
            assert!(port.saved.borrow().is_none(), "{failure}");
        }
    }

    #[test]
    fn qa_machine_continues_after_best_effort_failures_and_saves_sha_last() {
        let mut port = FakeQaPort::successful();
        port.issues = vec![qa_issue(7, "Old question", true)];
        port.comments.insert(
            7,
            Some(vec![CommentObservation {
                author: "alice".into(),
                body: "answered".into(),
            }]),
        );
        port.findings = vec![RawQaFinding {
            title: "Bug".into(),
            description: "broken".into(),
            severity: Severity::Medium,
            file: String::new(),
        }];
        port.fail_best_effort = vec!["close", "priority_label", "qa_label", "scope_label"];

        run_fake(&mut port).unwrap();
        assert!(port.saved.borrow().is_some());
        assert_eq!(
            port.trace.borrow().last().map(String::as_str),
            Some("act:save")
        );
    }

    #[test]
    fn qa_machine_honors_shutdown_at_start_and_after_model_without_saving_sha() {
        let mut stopped = FakeQaPort::successful();
        stopped.shutdown.borrow_mut().push_back(true);
        run_fake(&mut stopped).unwrap();
        assert_eq!(*stopped.trace.borrow(), vec!["observe:shutdown"]);

        let mut cancelled = FakeQaPort::successful();
        cancelled
            .shutdown
            .borrow_mut()
            .extend([false, false, false, true]);
        run_fake(&mut cancelled).unwrap();
        assert!(cancelled.trace.borrow().contains(&"act:model".to_string()));
        assert!(!cancelled.trace.borrow().contains(&"act:save".to_string()));
        assert!(cancelled.saved.borrow().is_none());
    }

    #[test]
    fn qa_machine_honors_shutdown_after_checkout_and_context_preparation() {
        for (answers, last_event) in [
            (vec![false, true], "observe:shutdown"),
            (vec![false, false, true], "observe:shutdown"),
        ] {
            let mut port = FakeQaPort::successful();
            port.shutdown.borrow_mut().extend(answers);
            run_fake(&mut port).unwrap();
            assert_eq!(
                port.trace.borrow().last().map(String::as_str),
                Some(last_event)
            );
            assert!(!port.trace.borrow().contains(&"act:model".to_string()));
            assert!(port.saved.borrow().is_none());
        }
    }

    #[test]
    fn qa_machine_observes_all_branch_shas_but_tests_only_first_changed_branch() {
        let mut port = FakeQaPort::successful();
        port.shas.insert("release".into(), "release-new".into());
        let mut machine = QaMachine::new(
            "qa-0",
            &["main".to_string(), "release".to_string()],
            None,
            AnalysisInput {
                qa_issues_path: "/q",
                open_issues_path: "/o",
                knowledge_dir: "/k",
                test_scripts_dir: "/t",
            },
        );
        run_qa_cycle(&mut machine, &mut port).unwrap();
        let trace = port.trace.borrow();
        assert!(trace.windows(3).any(|steps| steps
            == [
                "observe:sha:main",
                "observe:sha:release",
                "act:checkout:main"
            ]));
    }

    // --- Config parsing ---

    #[test]
    fn qa_settings_from_raw_parses_defaults() {
        let raw: toml::Value = toml::from_str("").unwrap();
        let settings = QaAgentSettings::from_raw(&raw).unwrap();
        assert_eq!(settings.poll_interval_secs, 300);
        assert_eq!(settings.branches, vec!["main".to_string()]);
    }

    #[test]
    fn qa_settings_from_raw_parses_custom_values() {
        let raw: toml::Value = toml::from_str(
            r#"
            poll_interval_secs = 120
            branches = ["main", "develop"]
            "#,
        )
        .unwrap();
        let settings = QaAgentSettings::from_raw(&raw).unwrap();
        assert_eq!(settings.poll_interval_secs, 120);
        assert_eq!(settings.branches, vec!["main", "develop"]);
    }

    #[test]
    fn qa_settings_rejects_empty_branches() {
        let raw: toml::Value = toml::from_str(r#"branches = []"#).unwrap();
        assert!(QaAgentSettings::from_raw(&raw).is_err());
    }

    // --- QaOutput contract ---

    #[test]
    fn qa_contract_passes_the_shared_conformance_suite() {
        conformance::assert_contract::<QaOutput>();
    }

    #[test]
    fn qa_output_deserializes_findings_and_clarifications() {
        let output = conformance::assert_accepts::<QaOutput>(serde_json::json!({
            "findings": [
                {
                    "title": "SQL injection in query.rs",
                    "description": "User input is not sanitized",
                    "severity": "critical",
                    "file": "src/query.rs:42"
                }
            ],
            "clarifications": [
                {"question": "What framework?", "context": "Need to know the test framework"}
            ]
        }));
        let findings = normalize_qa_findings(output.findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].title, "SQL injection in query.rs");
        assert_eq!(findings[0].severity, Severity::Critical);
        assert_eq!(findings[0].file, "src/query.rs:42");

        let clarifications = normalize_clarifications(output.clarifications);
        assert_eq!(clarifications.len(), 1);
        assert_eq!(clarifications[0].question, "What framework?");
        assert!(clarifications[0].context.contains("test framework"));
    }

    #[test]
    fn qa_output_accepts_a_clean_run_with_no_findings() {
        let output = conformance::assert_accepts::<QaOutput>(serde_json::json!({"findings": []}));
        assert!(output.findings.is_empty());
        assert!(output.clarifications.is_empty());
    }

    #[test]
    fn qa_output_requires_the_findings_array_even_when_empty() {
        assert_eq!(
            conformance::assert_rejects::<QaOutput>(serde_json::json!({})),
            "$.findings: required property is missing"
        );
    }

    #[test]
    fn qa_output_normalizes_case_and_defaults_unknown_severity_to_low() {
        let output = conformance::assert_accepts::<QaOutput>(serde_json::json!({
            "findings": [
                {"title": "Bug", "description": "d", "severity": "HIGH"},
                {"title": "Odd", "description": "d", "severity": "apocalyptic"}
            ]
        }));
        assert_eq!(output.findings[0].severity, Severity::High);
        assert_eq!(output.findings[1].severity, Severity::Low);
    }

    #[test]
    fn qa_output_reports_the_offending_finding_by_index() {
        assert_eq!(
            conformance::assert_rejects::<QaOutput>(serde_json::json!({
                "findings": [
                    {"title": "Bug", "description": "d", "severity": "high"},
                    {"title": "Missing description", "severity": "low"}
                ]
            })),
            "$.findings[1].description: required property is missing"
        );
    }

    #[test]
    fn qa_output_rejects_undeclared_finding_properties() {
        let error = conformance::assert_rejects::<QaOutput>(serde_json::json!({
            "findings": [{"title": "Bug", "description": "d", "severity": "high", "owner": "me"}]
        }));
        assert!(
            error.starts_with("$.findings[0].owner: unexpected property"),
            "{error}"
        );
    }

    #[test]
    fn qa_output_requires_context_on_every_clarification() {
        assert_eq!(
            conformance::assert_rejects::<QaOutput>(serde_json::json!({
                "findings": [],
                "clarifications": [{"question": "Which env?"}]
            })),
            "$.clarifications[0].context: required property is missing"
        );
    }

    #[test]
    fn normalize_qa_findings_skips_empty_title_or_description() {
        let raw = vec![
            RawQaFinding {
                title: "".into(),
                description: "no title".into(),
                ..Default::default()
            },
            RawQaFinding {
                title: "No desc".into(),
                description: "".into(),
                ..Default::default()
            },
            RawQaFinding {
                title: "Bug A".into(),
                description: "d1".into(),
                severity: Severity::High,
                file: "a.rs:1".into(),
            },
        ];
        let findings = normalize_qa_findings(raw);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].title, "Bug A");
    }

    #[test]
    fn normalize_qa_findings_defaults_missing_severity_to_low() {
        let raw = vec![RawQaFinding {
            title: "Bug".into(),
            description: "d".into(),
            ..Default::default()
        }];
        let findings = normalize_qa_findings(raw);
        assert_eq!(findings[0].severity, Severity::Low);
        assert_eq!(findings[0].file, "");
    }

    #[test]
    fn normalize_clarifications_skips_empty_question() {
        let raw = vec![
            RawClarification {
                question: "".into(),
                context: "no question".into(),
            },
            RawClarification {
                question: "Which DB?".into(),
                context: "Need to know the database to write migration tests".into(),
            },
        ];
        let questions = normalize_clarifications(raw);
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].question, "Which DB?");
        assert!(questions[0].context.contains("migration tests"));
    }

    // --- Severity helpers ---

    #[test]
    fn severity_is_non_trivial_filters_low() {
        assert!(Severity::Critical.is_non_trivial());
        assert!(Severity::High.is_non_trivial());
        assert!(Severity::Medium.is_non_trivial());
        assert!(!Severity::Low.is_non_trivial());
    }

    #[test]
    fn severity_priority_maps_correctly() {
        assert_eq!(Severity::Critical.priority(), 1);
        assert_eq!(Severity::High.priority(), 2);
        assert_eq!(Severity::Medium.priority(), 3);
        assert_eq!(Severity::Low.priority(), 3);
    }

    // --- SHA history ---

    #[test]
    fn sha_history_load_returns_empty_for_new() {
        let history = ShaHistory::default();
        assert!(history.0.is_empty());
    }

    #[test]
    fn sha_history_serializes_and_deserializes() {
        let mut h = ShaHistory::default();
        h.0.insert("main".to_string(), "abc123".to_string());
        let json = serde_json::to_string(&h).unwrap();
        let back: ShaHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(back.0.get("main").map(String::as_str), Some("abc123"));
    }

    // Real-file round trip through `load_sha_history`/`save_sha_history`,
    // which is the mechanism `qa_cycle` relies on to only advance a branch's
    // recorded SHA once a cycle has fully completed (findings/clarifications
    // processed and `model.complete_typed` returned `Ok`). `GitRepo::new` and
    // `GitLabClient::for_test` are file/network-free constructors, so this
    // exercises the real `AgentState` methods without touching git or GitLab.

    /// Owns the resources a test [`AgentState`] borrows from, standing in
    /// for the [`GitLabAgentRuntime`] fields QA's cycle needs.
    struct TestRuntime {
        agent_id: String,
        sessions_dir: String,
        git_repo: GitRepo,
        glab: GitLabClient,
    }

    fn test_runtime(sessions_dir: &str, agent_id: &str) -> TestRuntime {
        TestRuntime {
            agent_id: agent_id.to_string(),
            sessions_dir: sessions_dir.to_string(),
            git_repo: GitRepo::new(
                std::env::temp_dir().to_string_lossy().into_owned(),
                Arc::new(AtomicBool::new(false)),
            ),
            glab: GitLabClient::for_test("/tmp/unused-repo"),
        }
    }

    fn test_agent_state(rt: &TestRuntime) -> AgentState<'_> {
        AgentState {
            agent_id: &rt.agent_id,
            sessions_dir: &rt.sessions_dir,
            git_repo: &rt.git_repo,
            glab: &rt.glab,
        }
    }

    #[test]
    fn load_sha_history_defaults_to_empty_when_no_file_exists_yet() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-qa-sha-history-missing-{}",
            std::process::id()
        ));
        let rt = test_runtime(&dir.to_string_lossy(), "qa-0");
        let state = test_agent_state(&rt);
        let history = load_sha_history(&state);
        assert!(history.0.is_empty());
    }

    #[test]
    fn save_then_load_sha_history_round_trips_a_branchs_recorded_sha() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-qa-sha-history-roundtrip-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "qa-1");
        let state = test_agent_state(&rt);

        let mut history = load_sha_history(&state);
        assert!(history.0.is_empty());

        // Mirrors the single line at the end of `qa_cycle` that advances the
        // watched branch's SHA — this only runs after every required step
        // upstream (fetch, list_issues, model.complete_typed) succeeded.
        history
            .0
            .insert("main".to_string(), "cur-sha-1".to_string());
        save_sha_history(&state, &history).unwrap();

        let reloaded = load_sha_history(&state);
        assert_eq!(
            reloaded.0.get("main").map(String::as_str),
            Some("cur-sha-1")
        );

        let _ = fs::remove_file(state.sha_history_path());
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn corrupt_sha_history_file_resets_to_empty_instead_of_failing() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-qa-sha-history-corrupt-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "qa-2");
        let state = test_agent_state(&rt);
        fs::write(state.sha_history_path(), b"not valid json").unwrap();

        let history = load_sha_history(&state);
        assert!(history.0.is_empty());

        // Tolerant, but not lossy: the bad file is quarantined beside the
        // original rather than being deleted outright.
        assert!(!state.sha_history_path().exists());
        let quarantined: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("quarantined"))
            .collect();
        assert_eq!(quarantined.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsupported_sha_history_version_resets_to_empty_instead_of_failing() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-qa-sha-history-unsupported-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "qa-3");
        let state = test_agent_state(&rt);
        fs::write(state.sha_history_path(), br#"{"version":5,"state":{}}"#).unwrap();

        let history = load_sha_history(&state);
        assert!(history.0.is_empty());
        assert!(!state.sha_history_path().exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_sha_history_reads_the_legacy_unversioned_format_and_migrates_it() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-qa-sha-history-legacy-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "qa-4");
        let state = test_agent_state(&rt);
        let path = state.sha_history_path();

        // The bare pre-envelope payload written by older builds: a plain
        // JSON object of branch -> SHA (matches `ShaHistory`'s newtype
        // transparent serialization).
        fs::write(&path, br#"{"main":"abc123"}"#).unwrap();

        let history = load_sha_history(&state);
        assert_eq!(history.0.get("main").map(String::as_str), Some("abc123"));

        // Transparently migrated to the v1 envelope on disk.
        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["main"], "abc123");

        let _ = fs::remove_dir_all(&dir);
    }

    // --- is_potlatch_author ---

    #[test]
    fn is_potlatch_author_detects_potlatch_and_qa_bots() {
        assert!(is_potlatch_author("potlatch"));
        assert!(is_potlatch_author("potlatch-worker-0"));
        assert!(is_potlatch_author("qa-0"));
        assert!(!is_potlatch_author("alice"));
        assert!(!is_potlatch_author("bob.miller"));
    }

    // --- Prompt builder ---

    #[test]
    fn build_qa_prompt_includes_file_locations_and_end_user_frame() {
        let prompt = build_qa_prompt(
            "qa-0",
            &GitContext {
                branch: "main",
                prev_sha: "abc123",
                cur_sha: "def456",
                changed_files: &["src/main.rs".to_string()],
            },
            &AnalysisInput {
                qa_issues_path: "/sessions/qa-0_qa_issues.md",
                open_issues_path: "/sessions/qa-0_open_issues.md",
                knowledge_dir: "/sessions/qa-0_qa_knowledge",
                test_scripts_dir: "/sessions/qa-0_test_scripts",
            },
        );
        // End-user tester framing.
        assert!(prompt.contains("end-user tester"));
        assert!(!prompt.contains("Scrutinize the codebase"));
        // Functionality-first organization; issues are context, not the test plan.
        assert!(prompt.contains("organized around user-facing functionality"));
        assert!(prompt.contains("source of context"));
        assert!(prompt.contains("Read the QA issues file and your knowledge files first"));
        assert!(prompt.contains("acceptance criteria"));
        assert!(
            prompt.contains("Fold issue-specific acceptance criteria into the function's section")
        );
        // functionality.md is the primary artifact.
        assert!(prompt.contains("functionality.md"));
        assert!(prompt.contains("primary artifact"));
        assert!(prompt.contains("named by the function they test"));
        // Ops/liveness endpoints banned unless an issue asks.
        assert!(prompt.contains("liveness/ops endpoints"));
        assert!(prompt.contains("/ping"));
        assert!(prompt.contains("/monitor"));
        assert!(prompt.contains("/health"));
        // File locations.
        assert!(prompt.contains("/sessions/qa-0_qa_issues.md"));
        assert!(prompt.contains("/sessions/qa-0_open_issues.md"));
        assert!(prompt.contains("/sessions/qa-0_qa_knowledge"));
        assert!(prompt.contains("/sessions/qa-0_test_scripts"));
        // outside_cwd usage instruction.
        assert!(prompt.contains("outside_cwd: true"));
        // GitLab hard rule.
        assert!(prompt.contains("Never fetch GitLab yourself"));
        assert!(prompt.contains("glab"));
        // Unit-test/build ban.
        assert!(prompt.contains("cargo test"));
        assert!(prompt.contains("go test"));
        assert!(prompt.contains("pytest"));
        // Cwd mutation ban.
        assert!(prompt.contains("Never modify the working directory"));
        // Dedup against the full open-issue listing.
        assert!(prompt.contains("Never report a duplicate finding"));
        assert!(prompt.contains("already tracked"));
        assert!(prompt.contains("root cause"));
        // Regression testing of all tracked functions.
        assert!(prompt.contains("Run regression tests for all tracked functions"));
        assert!(prompt.contains("test_cases.md"));
        assert!(prompt.contains("regressions are the whole point"));
        // Git context.
        assert!(prompt.contains("main"));
        assert!(prompt.contains("abc123"));
        assert!(prompt.contains("def456"));
        assert!(prompt.contains("src/main.rs"));
        // Structured-output presentation belongs to the backend vendor, not
        // the role prompt.
        assert!(!prompt.contains("qa_report"));
        assert!(!prompt.contains("output contract"));
        assert!(prompt.contains("findings"));
        assert!(prompt.contains("clarification questions"));
        assert!(!prompt.contains("tool has two fields"));
        assert!(!prompt.contains("JSON array of objects"));
        // No text-marker fallback or removed blocks.
        assert!(!prompt.contains("TEXT MARKER FALLBACK"));
        assert!(!prompt.contains("QA_FINDINGS_BEGIN"));
        assert!(!prompt.contains("QA_CLARIFICATION_BEGIN"));
        assert!(!prompt.contains("QA_TEST_SCRIPTS_BEGIN"));
        assert!(!prompt.contains("QA_CONTEXT_BEGIN"));
        assert!(!prompt.contains("QA_TEST_CASES_BEGIN"));
        assert!(!prompt.contains("QA_FUNCTIONALITY_BEGIN"));
        assert!(!prompt.contains("QA_REQUIREMENTS_BEGIN"));
    }

    #[test]
    fn build_qa_prompt_instructs_model_to_author_and_run_python_scripts() {
        let prompt = build_qa_prompt(
            "qa-0",
            &GitContext {
                branch: "main",
                prev_sha: "",
                cur_sha: "def",
                changed_files: &[],
            },
            &AnalysisInput {
                qa_issues_path: "/x/issues.md",
                open_issues_path: "/x/open_issues.md",
                knowledge_dir: "/x/knowledge",
                test_scripts_dir: "/x/scripts",
            },
        );
        assert!(prompt.contains("python3"));
        assert!(prompt.contains("test_p4_search.py"));
        assert!(prompt.contains("named by the function they test"));
        assert!(prompt.contains("Write Python scripts"));
    }

    #[test]
    fn qa_output_deserializes_multiple_findings_via_tool_definition_schema() {
        // Exercises the full round trip through `QaOutput::tool_definition()`'s
        // schema shape, not just ad hoc JSON.
        let tool = QaOutput::tool_definition();
        assert_eq!(tool.name, "qa_report");

        let output = conformance::assert_accepts::<QaOutput>(serde_json::json!({
            "findings": [
                {
                    "title": "Search returns 500 on empty query",
                    "description": "GET /search?q= returns 500 instead of 400",
                    "severity": "high",
                    "file": "src/handler.go:42"
                },
                {
                    "title": "Memory leak in cache",
                    "description": "Cache grows unbounded",
                    "severity": "medium"
                }
            ]
        }));
        let findings = normalize_qa_findings(output.findings);
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].title, "Search returns 500 on empty query");
        assert_eq!(findings[0].severity, Severity::High);
        assert_eq!(findings[0].file, "src/handler.go:42");
        assert_eq!(findings[1].title, "Memory leak in cache");
        assert_eq!(findings[1].severity, Severity::Medium);
        assert_eq!(findings[1].file, "");
    }
}
