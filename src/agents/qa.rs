//! QA agent: monitors configured branches for new commits, gathers the
//! project's open QA-labeled GitLab issues into a context file, and invokes
//! the model to perform end-user functionality testing against the running
//! system (HTTP APIs, CLI invocations, configuration loading). The model
//! authors and runs its own Python test scripts and maintains its own
//! knowledge files in the agent's session directory; the agent's only file
//! responsibility is the QA-issues context file. Creates GitLab issues for
//! non-trivial findings.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient};
use crate::agents::workspace::{
    GitLabAgentBootstrap, gitlab_banner, validate_instance_id, validate_max_instances,
};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, ObjectSchema, SchemaField, StructuredOutput};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::PeriodicTaskSpec;

pub(crate) const NAME: &str = "qa";
const MAX_INSTANCES: usize = 1;

const QA_LABEL: &str = crate::agents::labels::QA;
const DO_NOT_IMPLEMENT_LABEL: &str = crate::agents::labels::DO_NOT_IMPLEMENT;

/// A finding's severity, as the model reports it via the `qa_report` tool.
/// Unknown or missing values default to `Low`, preserving the pre-typed
/// behavior of treating unrecognized severity as non-blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Severity {
    Critical,
    High,
    Medium,
    #[default]
    Low,
}

impl<'de> Deserialize<'de> for Severity {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.trim().to_ascii_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            _ => Self::Low,
        })
    }
}

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
struct QaOutput {
    #[serde(default)]
    findings: Vec<RawQaFinding>,
    #[serde(default)]
    clarifications: Vec<RawClarification>,
}

impl StructuredOutput for QaOutput {
    fn tool_name() -> &'static str {
        "qa_report"
    }

    fn tool_description() -> &'static str {
        "Emit your QA test findings and clarification questions as structured JSON. This is the primary output channel — Potlatch reads the tool's JSON, not your streamed text. Call this exactly once with your results."
    }

    fn schema() -> ObjectSchema {
        ObjectSchema::new()
            .property(
                "findings",
                SchemaField::array(
                    "Test findings (bugs). Empty array if no bugs found.",
                    SchemaField::object(
                        ObjectSchema::new()
                            .property(
                                "title",
                                SchemaField::string("Short actionable title for the finding."),
                            )
                            .property(
                                "description",
                                SchemaField::string(
                                    "Detailed description with steps to reproduce, expected vs actual behavior, and impact.",
                                ),
                            )
                            .property(
                                "severity",
                                SchemaField::string_enum(
                                    "Severity: \"critical\", \"high\", \"medium\", or \"low\".",
                                    &["critical", "high", "medium", "low"],
                                ),
                            )
                            .property(
                                "file",
                                SchemaField::string(
                                    "Source file and line number if known (e.g. \"src/path/to/file.rs:123\"). Omit if unknown.",
                                ),
                            )
                            .required("title")
                            .required("description")
                            .required("severity"),
                    ),
                ),
            )
            .property(
                "clarifications",
                SchemaField::array(
                    "Clarification questions for humans. Empty array if none.",
                    SchemaField::object(
                        ObjectSchema::new()
                            .property("question", SchemaField::string("The clarification question."))
                            .property(
                                "context",
                                SchemaField::string("Context explaining why the question is needed."),
                            )
                            .required("question")
                            .required("context"),
                    ),
                ),
            )
            .required("findings")
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
struct QaAgentSettings {
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

struct AgentState {
    agent_id: String,
    sessions_dir: String,
    git_repo: GitRepo,
    glab: GitLabClient,
}

impl AgentState {
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
    state: AgentState,
    model: AgentModel,
    config: QaConfig,
    scope_label: String,
}

// ---------------------------------------------------------------------------
// Persistence types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        NAME
    }

    fn agent_id(&self) -> &str {
        &self.state.agent_id
    }

    fn shutdown(&self) -> &Arc<AtomicBool> {
        self.model.shutdown()
    }

    fn banner(config: &Config, banner: &mut Banner) {
        gitlab_banner(config, banner);
    }

    fn validate_config(config: &Config, section: &crate::core::config::AgentSection) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_gitlab_repo()?;
        validate_max_instances(NAME, section.core.instances, MAX_INSTANCES)?;
        QaAgentSettings::from_raw(&section.raw)?;
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
                let scope = crate::agents::scope_label_filter(&self.scope_label);
                qa_cycle(&self.state, &self.config, &self.model, scope)
            }
            _ => Ok(()),
        }
    }

    fn from_spawn(ctx: crate::core::workflow::AgentSpawnContext) -> Result<Self> {
        let section = ctx
            .workflow
            .config
            .agent(NAME)
            .context("[agent.qa] section required")?;
        validate_max_instances(NAME, section.core.instances, MAX_INSTANCES)?;
        validate_instance_id(NAME, ctx.instance_id, MAX_INSTANCES)?;
        let agent_settings = QaAgentSettings::from_raw(&section.raw)?;
        let runtime = GitLabAgentBootstrap::new(
            &ctx,
            NAME,
            ModelPreferences {
                structured_output_tools: Some(vec![QaOutput::tool_definition()]),
                ..ModelPreferences::default()
            },
        )
        .build()?;
        let state = AgentState {
            agent_id: runtime.agent_id,
            sessions_dir: runtime.sessions_dir,
            git_repo: runtime.git_repo,
            glab: runtime.gitlab,
        };
        let config = QaConfig {
            poll_interval_secs: agent_settings.poll_interval_secs,
            branches: agent_settings.branches,
        };
        Ok(Self {
            state,
            model: runtime.model,
            config,
            scope_label: runtime.scope_label,
        })
    }

    fn on_shutdown(&mut self) {}
}

// ---------------------------------------------------------------------------
// QA cycle
// ---------------------------------------------------------------------------

fn qa_cycle(
    state: &AgentState,
    config: &QaConfig,
    model: &AgentModel,
    scope_label: Option<&str>,
) -> Result<()> {
    let shutdown = model.shutdown();
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // --- Branch commit detection ---

    state.git_repo.fetch()?;
    let branch_strs: Vec<&str> = config.branches.iter().map(|s| s.as_str()).collect();
    state.git_repo.fetch_branches(&branch_strs)?;

    let mut sha_history = load_sha_history(state);
    let mut branches_with_new_commits: Vec<(&str, String, String)> = Vec::new();
    for branch in &config.branches {
        let cur_sha = state.git_repo.remote_short_sha(branch)?;
        let prev_sha = sha_history.0.get(branch).cloned();
        match prev_sha {
            Some(prev) if prev == cur_sha => {
                debug!("Branch {branch} unchanged at {cur_sha}");
            }
            _ => {
                info!(
                    "{}: Branch {} has new commits ({} -> {})",
                    state.agent_id,
                    branch,
                    prev_sha.as_deref().unwrap_or("(none)"),
                    cur_sha
                );
                branches_with_new_commits.push((branch, prev_sha.unwrap_or_default(), cur_sha));
            }
        }
    }

    if branches_with_new_commits.is_empty() {
        debug!(
            "{}: No new commits on any watched branch, skipping",
            state.agent_id
        );
        return Ok(());
    }

    let (branch, prev_sha, cur_sha) = &branches_with_new_commits[0];
    state.git_repo.checkout_remote_branch(branch)?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // --- Gather GitLab context: all open issues + QA-labeled subset ---

    let all_issues = state.glab.list_issues()?;
    let qa_issues: Vec<gitlab::Issue> = all_issues
        .iter()
        .filter(|i| i.labels.iter().any(|l| l == QA_LABEL))
        .cloned()
        .collect();
    let qa_issue_titles: Vec<String> = qa_issues.iter().map(|i| i.title.clone()).collect();
    write_qa_issues_context(state, &qa_issues)?;
    write_open_issues_context(state, &all_issues)?;
    debug!(
        "{}: {} open issue(s) total, {} QA-labeled",
        state.agent_id,
        all_issues.len(),
        qa_issues.len()
    );

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // --- QA analysis ---

    let changed_files = if prev_sha.is_empty() {
        state
            .git_repo
            .changed_files_since("HEAD~1")
            .unwrap_or_default()
    } else {
        state
            .git_repo
            .changed_files_since(prev_sha)
            .unwrap_or_default()
    };

    let prompt = build_qa_prompt(
        &state.agent_id,
        &GitContext {
            branch,
            prev_sha,
            cur_sha,
            changed_files: &changed_files,
        },
        &AnalysisInput {
            qa_issues_path: &state.qa_issues_path().to_string_lossy(),
            open_issues_path: &state.open_issues_path().to_string_lossy(),
            knowledge_dir: &state.knowledge_dir().to_string_lossy(),
            test_scripts_dir: &state.test_scripts_dir().to_string_lossy(),
        },
    );

    let completion = model.complete_typed::<QaOutput>(
        &prompt,
        &InvokeOptions {
            activity_label: Some(format!("{} QA analysis on {}", state.agent_id, branch)),
            ..InvokeOptions::default()
        },
    )?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // --- Parse outputs ---

    let findings = normalize_qa_findings(completion.output.findings);
    let clarification_questions = normalize_clarifications(completion.output.clarifications);

    // --- Close answered clarification issues ---
    // A clarification issue carries QA + DO_NOT_IMPLEMENT; if a non-potlatch user
    // has commented on it, the question is answered and the issue can be closed.

    for issue in &qa_issues {
        if !issue.labels.iter().any(|l| l == DO_NOT_IMPLEMENT_LABEL) {
            continue;
        }
        match state.glab.get_issue_comments(issue.iid) {
            Ok(comments) => {
                let answered = comments.iter().any(|c| !is_potlatch_author(&c.author));
                if answered {
                    if let Err(e) = state.glab.close_issue(issue.iid) {
                        warn!(
                            "{}: Failed to close answered clarification issue #{}: {}",
                            state.agent_id, issue.iid, e
                        );
                    } else {
                        info!(
                            "{}: Closed clarification issue #{} (answered)",
                            state.agent_id, issue.iid
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    "{}: Failed to fetch comments for clarification issue #{}: {}",
                    state.agent_id, issue.iid, e
                );
            }
        }
    }

    // --- Create new clarification issues ---

    for q in &clarification_questions {
        let description = format!(
            "{}\n\n---\n*This is a QA clarification question. Please answer in a comment. The QA agent will pick up answers automatically.*",
            q.context
        );
        match state.glab.create_issue(&q.question, &description) {
            Ok(issue_iid) => {
                let _ = state.glab.add_issue_label(issue_iid, QA_LABEL);
                let _ = state
                    .glab
                    .add_issue_label(issue_iid, DO_NOT_IMPLEMENT_LABEL);
                info!(
                    "{}: Created clarification issue #{}: {}",
                    state.agent_id, issue_iid, q.question
                );
            }
            Err(e) => {
                warn!(
                    "{}: Failed to create clarification issue: {}",
                    state.agent_id, e
                );
            }
        }
    }

    // --- Create GitLab issues for non-trivial findings ---
    // Dedup against currently-open QA-labeled issue titles so we don't re-file
    // an issue that's still open. Closed issues are not in the list, so a
    // regression after a fix can legitimately be re-filed.

    let scope = scope_label;
    let mut created_count = 0;
    for finding in &findings {
        if !finding.severity.is_non_trivial() {
            info!(
                "{}: Skipping low-severity finding: {}",
                state.agent_id, finding.title
            );
            continue;
        }
        if qa_issue_titles.iter().any(|t| t == &finding.title) {
            debug!(
                "{}: Open QA issue with same title exists, skipping: {}",
                state.agent_id, finding.title
            );
            continue;
        }

        let description = format!(
            "**Severity:** {}\n**File:** {}\n**Branch:** {}\n**Commit:** {}\n\n{}\n\n---\n*Found by QA agent on {}*",
            finding.severity.as_str(),
            if finding.file.is_empty() {
                "n/a".to_string()
            } else {
                finding.file.clone()
            },
            branch,
            cur_sha,
            finding.description,
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
        );

        match state.glab.create_issue(&finding.title, &description) {
            Ok(issue_iid) => {
                let priority = finding.severity.priority();
                let _ = state
                    .glab
                    .add_issue_label(issue_iid, &gitlab::priority_label(priority));
                let _ = state.glab.add_issue_label(issue_iid, QA_LABEL);
                if let Some(lbl) = scope {
                    let _ = state.glab.add_issue_label(issue_iid, lbl);
                }
                info!(
                    "{}: Created GitLab issue #{} for finding: {}",
                    state.agent_id, issue_iid, finding.title
                );
                created_count += 1;
            }
            Err(e) => {
                warn!(
                    "{}: Failed to create GitLab issue for finding '{}': {}",
                    state.agent_id, finding.title, e
                );
            }
        }
    }

    if created_count > 0 {
        info!(
            "{}: Created {} new GitLab issue(s) from QA findings",
            state.agent_id, created_count
        );
    }

    // --- Update SHA history ---
    //
    // The `qa_report` tool call is mandatory (enforced by `complete_typed`),
    // so every successful run means the model called it — with an empty
    // `findings` array at minimum — and is therefore considered tested, even
    // if it also asked clarification questions.
    sha_history.0.insert(branch.to_string(), cur_sha.clone());
    save_sha_history(state, &sha_history);

    Ok(())
}

/// Render the QA-labeled issues (with comments) to the agent's context file.
/// The model reads this file directly; it never calls any tool to fetch GitLab.
fn write_qa_issues_context(state: &AgentState, issues: &[gitlab::Issue]) -> Result<()> {
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
        for issue in issues {
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
            match state.glab.get_issue_comments(issue.iid) {
                Ok(comments) if !comments.is_empty() => {
                    out.push_str("### Comments\n\n");
                    for c in &comments {
                        let author = if is_potlatch_author(&c.author) {
                            format!("{} (potlatch)", c.author)
                        } else {
                            c.author.clone()
                        };
                        out.push_str(&format!("**{author}:**\n{}\n\n", c.body.trim()));
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(
                        "{}: Failed to fetch comments for issue #{}: {}",
                        state.agent_id, issue.iid, e
                    );
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
fn write_open_issues_context(state: &AgentState, issues: &[gitlab::Issue]) -> Result<()> {
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
    // Missing, unreadable, and corrupt history have historically reset QA history.
    crate::core::state::StateStore::new(state.sha_history_path())
        .load()
        .ok()
        .flatten()
        .unwrap_or_default()
}

fn save_sha_history(state: &AgentState, history: &ShaHistory) {
    let store = crate::core::state::StateStore::new(state.sha_history_path());
    if let Err(e) = store.save(history) {
        warn!("Failed to save SHA history: {}", e);
    }
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
3. **Never report a duplicate finding.** Before including a finding in the `qa_report` tool call, read `{open_issues_path}` and check whether an open issue already describes the same problem (by the QA agent, another agent, or a human). If it does, do not report that finding — the issue is already tracked. Compare by the underlying problem, not just exact-title match: a finding about "login returns 500 on empty password" duplicates an issue titled "Auth API crashes on malformed input" even though the wording differs. Only report a finding if no open issue covers the same root cause.
4. **Never mutate GitLab.** Do not post comments or create/edit issues via tools. The harness creates GitLab issues from your `qa_report` tool call's `findings` and `clarifications`.
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

## Output Format

Call the `qa_report` tool exactly once with your results — this tool call is the only output channel Potlatch reads; there is no text-based fallback. The tool has two fields:
- `findings` (required): a JSON array of objects with `title`, `description`, `severity` (critical/high/medium/low), and optional `file` (e.g. "src/path/to/file.rs:123"). Empty array if no bugs found.
- `clarifications` (optional): a JSON array of objects with `question` and `context`. Omit or use empty array if no clarification needed.

Only critical, high, and medium findings will be created as GitLab issues; low-severity findings are logged but not tracked. The `file` field is optional — leave it empty when the finding is observed externally and you cannot tie it to a specific source location. Findings and clarification questions may both be emitted in the same run."##
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- QaOutput deserialization ---

    #[test]
    fn qa_output_deserializes_findings_and_clarifications() {
        let output: QaOutput = serde_json::from_value(serde_json::json!({
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
        }))
        .unwrap();
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
    fn qa_output_defaults_to_empty_when_fields_absent() {
        let output: QaOutput = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(output.findings.is_empty());
        assert!(output.clarifications.is_empty());
    }

    #[test]
    fn qa_output_normalizes_case_and_defaults_unknown_severity_to_low() {
        let output = serde_json::from_value::<QaOutput>(serde_json::json!({
            "findings": [
                {"title": "Bug", "description": "d", "severity": "HIGH"},
                {"title": "Odd", "description": "d", "severity": "apocalyptic"}
            ]
        }))
        .unwrap();
        assert_eq!(output.findings[0].severity, Severity::High);
        assert_eq!(output.findings[1].severity, Severity::Low);
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
        // The `qa_report` tool is the only output channel.
        assert!(prompt.contains("Call the `qa_report` tool"));
        assert!(prompt.contains("findings"));
        assert!(prompt.contains("clarifications"));
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

        let output: QaOutput = serde_json::from_value(serde_json::json!({
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
        }))
        .unwrap();
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
