use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::agents::gitlab::{self, GitLabClient};
use crate::agents::ssh_util::{shell_single_quote, validate_remote_path, validate_ssh_identity};
use crate::agents::workspace::{GitLabAgentBootstrap, GitLabAgentRuntime, gitlab_banner};
use crate::agents::write_task_context_file;
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, compat, structured_output};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::cycle::Step;
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;

pub(crate) const NAME: &str = "ops";
const MAX_INSTANCES: usize = 1;

/// One issue proposal as the model described it via the `ops_report` tool's
/// `issues` array, before the empty-field defensive filtering in
/// [`normalize_ops_issues`] is applied.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOpsIssue {
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    priority: Option<u64>,
    #[serde(default)]
    log_line: String,
}

/// The ops agent's typed structured-output contract. The model calls the
/// `ops_report` tool with its log analysis findings; core validates the
/// captured JSON against [`OpsOutput::schema`] and deserializes it (see
/// [`AgentModel::complete_typed`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpsOutput {
    #[serde(default)]
    issues: Vec<RawOpsIssue>,
}

structured_output! {
    impl OpsOutput {
        tool_name: "ops_report";
        tool_description: "The new actionable issues found during this log analysis run.";
        schema: object("Everything this log analysis run found.", {
            required issues: array(
                "New actionable issues found in the logs. Empty array if nothing new.",
                object("One issue to file from the logs.", {
                    required title: string("Short actionable issue title."),
                    required description: string(
                        "Markdown body with log evidence, likely code area, impact, and suggested remediation."
                    ),
                    optional priority: integer_enum(
                        "Priority: 1 (critical/blocking), 2 (high), 3 (normal).",
                        &[1, 2, 3]
                    ),
                    required log_line: string(
                        "Exact representative log line from the session file."
                    ),
                })
            ),
        });
    /// Tolerated: `sample_line` as a legacy name for `log_line`, and a
    /// priority outside 1-3 (dropped, so the issue is filed unprioritized).
        normalize(value) {
            compat::each_in_array(value, "issues", |issue| {
                compat::rename_property(issue, "sample_line", "log_line");
                compat::drop_integer_outside(issue, "priority", &[1, 2, 3]);
            });
        }
    }
}

/// Drop issue proposals with an empty title, description, or log line (the
/// model occasionally emits a placeholder entry), and clamp `priority` to
/// the valid 1..=3 range instead of erroring on an out-of-range value.
fn normalize_ops_issues(raw: Vec<RawOpsIssue>) -> Vec<OpsIssueProposal> {
    raw.into_iter()
        .filter_map(|item| {
            let title = item.title.trim().to_string();
            if title.is_empty() {
                return None;
            }
            let description = item.description.trim().to_string();
            if description.is_empty() {
                return None;
            }
            let log_line = item.log_line.trim().to_string();
            if log_line.is_empty() {
                return None;
            }
            let priority = item
                .priority
                .filter(|priority| (1..=3).contains(priority))
                .map(|priority| priority as u8);
            Some(OpsIssueProposal {
                title,
                description,
                priority,
                log_line,
            })
        })
        .collect()
}

const LOG_WINDOW_HOURS: i64 = 2;
const MAX_TAIL_LINES: u32 = 100_000;
const MAX_SCRAPE_FILES_KEPT: usize = 10;
const MIN_LOG_BYTES_FOR_ANALYSIS: usize = 20;

#[derive(Debug, Clone)]
struct OpsConfig {
    poll_interval_secs: u64,
    log_sources: Vec<OpsLogSource>,
}

#[derive(Debug, Clone)]
struct OpsLogSource {
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OpsAgentSettings {
    #[serde(default = "default_ops_poll_interval")]
    poll_interval_secs: u64,
    ssh_user: Option<String>,
    ssh_host: Option<String>,
    log_path: Option<String>,
    #[serde(default)]
    logs: Vec<OpsLogSourceSettings>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpsLogSourceSettings {
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

fn default_ops_poll_interval() -> u64 {
    600
}

impl OpsAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        let mut settings: Self = raw
            .clone()
            .try_into()
            .context("ops agent settings from config")?;
        let legacy_present = settings.ssh_user.is_some()
            || settings.ssh_host.is_some()
            || settings.log_path.is_some();
        if legacy_present {
            let source = OpsLogSourceSettings {
                ssh_user: settings
                    .ssh_user
                    .take()
                    .context("ssh_user is required for [agent.ops]")?,
                ssh_host: settings
                    .ssh_host
                    .take()
                    .context("ssh_host is required for [agent.ops]")?,
                log_path: settings
                    .log_path
                    .take()
                    .context("log_path is required for [agent.ops]")?,
            };
            settings.logs.push(source);
        }
        ensure!(
            !settings.logs.is_empty(),
            "at least one log source is required for [agent.ops]"
        );
        for (idx, source) in settings.logs.iter().enumerate() {
            source.validate(idx)?;
        }
        Ok(settings)
    }
}

impl OpsLogSourceSettings {
    fn validate(&self, idx: usize) -> Result<()> {
        ensure!(
            !self.ssh_user.trim().is_empty(),
            "ssh_user is required for [agent.ops].logs[{idx}]"
        );
        ensure!(
            !self.ssh_host.trim().is_empty(),
            "ssh_host is required for [agent.ops].logs[{idx}]"
        );
        ensure!(
            !self.log_path.trim().is_empty(),
            "log_path is required for [agent.ops].logs[{idx}]"
        );
        Ok(())
    }

    fn into_source(self) -> OpsLogSource {
        OpsLogSource {
            ssh_user: self.ssh_user.trim().to_string(),
            ssh_host: self.ssh_host.trim().to_string(),
            log_path: self.log_path.trim().to_string(),
        }
    }
}

/// A borrowing view over the [`GitLabAgentRuntime`] fields ops's path
/// helpers need. Built fresh from `&GitLabAgentRuntime` at each use site
/// rather than stored — ops never owns a second copy of `sessions_dir`
/// or `agent_id`, and this is never stored alongside the runtime it
/// borrows from, so it can't become self-referential.
struct AgentState<'a> {
    sessions_dir: &'a str,
    agent_id: &'a str,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &GitLabAgentRuntime) -> AgentState<'_> {
        AgentState {
            sessions_dir: &runtime.sessions_dir,
            agent_id: &runtime.agent_id,
        }
    }

    fn ensure_sessions_dir(&self) -> Result<()> {
        fs::create_dir_all(self.sessions_dir).context("Failed to create sessions directory")?;
        Ok(())
    }

    fn history_path(&self) -> PathBuf {
        Path::new(self.sessions_dir).join(format!("{}_issue_history.json", self.agent_id))
    }

    fn scrape_path(&self, unix_ts: u64) -> PathBuf {
        Path::new(self.sessions_dir).join(format!("{}-scrape-{unix_ts}.log", self.agent_id))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct OpsIssueHistoryEntry {
    gitlab_issue_iid: u64,
    title: String,
    log_line: String,
    created_at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct OpsIssueHistory {
    #[serde(default)]
    entries: Vec<OpsIssueHistoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpsIssueProposal {
    title: String,
    description: String,
    priority: Option<u8>,
    log_line: String,
}

pub(crate) struct OpsAgent {
    runtime: GitLabAgentRuntime,
    config: OpsConfig,
}

impl CoreAgent for OpsAgent {
    type Settings = OpsAgentSettings;
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
        OpsAgentSettings::from_raw(&section.raw)
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
            "log_scrape",
            Duration::from_secs(self.config.poll_interval_secs),
        )]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "log_scrape" => {
                let scope = crate::agents::scope_label_filter(&self.runtime.scope_label);
                let model = &self.runtime.model;
                let shutdown = Arc::clone(model.shutdown());
                let state = AgentState::from_runtime(&self.runtime);
                ops_cycle(
                    &state,
                    &self.config,
                    &self.runtime.gitlab,
                    model,
                    Arc::clone(&shutdown),
                    scope,
                )
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = GitLabAgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        AgentState::from_runtime(&runtime).ensure_sessions_dir()?;
        let agent_settings = ctx.settings;
        let config = OpsConfig {
            poll_interval_secs: agent_settings.poll_interval_secs,
            log_sources: agent_settings
                .logs
                .into_iter()
                .map(OpsLogSourceSettings::into_source)
                .collect(),
        };
        Ok(Self { runtime, config })
    }

    fn on_shutdown(&mut self) {}
}

// ---------------------------------------------------------------------------
// Ops role port
// ---------------------------------------------------------------------------

/// Immutable snapshot of one configured log source, as the ops machine's
/// decisions see it. Role-local on purpose: the machine never holds the
/// live [`OpsConfig`], only the fields one scrape needs, and it can never
/// mutate what it observed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LogSourceObservation {
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

impl LogSourceObservation {
    fn from_source(source: &OpsLogSource) -> Self {
        Self {
            ssh_user: source.ssh_user.clone(),
            ssh_host: source.ssh_host.clone(),
            log_path: source.log_path.clone(),
        }
    }

    fn target(&self) -> String {
        format!("{}@{}:{}", self.ssh_user, self.ssh_host, self.log_path)
    }

    /// The banner that separates this source's lines from the next one's in
    /// the joined analysis input.
    fn section(&self, window_log: &str) -> String {
        format!("===== Log source: {} =====\n{window_log}", self.target())
    }
}

/// The single wall-clock reading one cycle takes: the UNIX timestamp that
/// names this cycle's scrape/analysis files, and the instant its log window
/// is measured back from. Observed once so the file names and the history
/// entries a cycle writes all agree on when the cycle happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CycleClock {
    unix_ts: u64,
    now: DateTime<Utc>,
}

/// One question the ops machine asks before it decides anything. Every read
/// the cycle performs is one of these, so a recorded trace shows the
/// observations in the order the machine needed them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsQuery {
    ShutdownRequested,
    CycleClock,
    RemoteLogTail { source: LogSourceObservation },
    IssueHistory,
    HistoryFilePath,
}

/// The answer to one [`OpsQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsFact {
    ShutdownRequested(bool),
    CycleClock(CycleClock),
    RemoteLogTail(String),
    IssueHistory(OpsIssueHistory),
    HistoryFilePath(String),
}

/// A single side effect (or the single model invocation) the ops machine asks
/// the port to perform. Every variant is one step: the machine never hands
/// over a batch, so the recorded order of these *is* the ops role's mutation
/// order — including the per-proposal create → priority → scope ordering and
/// the one final history save.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsAction {
    /// File preparation: gather the open issues and merge requests the model
    /// dedups against into this cycle's context file.
    WriteGitLabContextFile {
        unix_ts: u64,
    },
    WriteScrapeFile {
        unix_ts: u64,
        window_log: String,
    },
    PruneScrapeFiles,
    /// Create the history file if this is the first cycle that ever ran; a
    /// no-op once it exists, so an existing history is never overwritten.
    EnsureHistoryFile(OpsIssueHistory),
    /// File preparation: the model's primary analysis input.
    WriteAnalysisFile {
        unix_ts: u64,
        window_log: String,
    },
    InvokeAnalysisModel {
        prompt: String,
    },
    CreateIssue {
        title: String,
        description: String,
    },
    /// Best effort, exactly like the `let Err(e) = …` warn it replaces.
    AddPriorityLabel {
        issue_iid: u64,
        priority: u8,
    },
    /// Best effort, same as above.
    AddScopeLabel {
        issue_iid: u64,
    },
    /// The cycle's single history write, after every issue it managed to
    /// create has been appended.
    SaveHistory(OpsIssueHistory),
}

/// What the port reports back after executing one [`OpsAction`].
enum OpsOutcome {
    /// The action completed and has nothing to report.
    Done,
    /// The action failed. Whether that aborts the cycle or is merely logged
    /// is decided by the stage that asked for it, mirroring which call sites
    /// used `?` and which were wrapped in a `match`.
    Failed(anyhow::Error),
    ContextFileWritten(String),
    AnalysisFileWritten(String),
    Analyzed(Vec<RawOpsIssue>),
    IssueCreated(u64),
}

/// The narrow surface the ops cycle needs. Object-safe and role-local: it is
/// the ops role's own observe/execute vocabulary, not a stand-in for the
/// GitLab API, ssh, the filesystem, or the model backend (see
/// [`crate::agents::claim::ClaimPort`] for the same reasoning at claim
/// granularity).
trait OpsPort {
    fn shutdown_requested(&self) -> bool;
    fn cycle_clock(&self) -> CycleClock;
    fn remote_log_tail(&self, source: &LogSourceObservation) -> Result<String>;
    fn issue_history(&self) -> Result<OpsIssueHistory>;
    fn history_file_path(&self) -> Result<String>;
    fn execute(&mut self, action: &OpsAction) -> OpsOutcome;
}

type OpsStep = Step<OpsQuery, OpsAction>;

// ---------------------------------------------------------------------------
// Pure ops decisions
// ---------------------------------------------------------------------------

/// Whether the scraped window carries enough bytes to be worth a model
/// invocation. Pure, so the threshold policy is characterized without ssh.
fn window_is_analyzable(window_log: &str) -> bool {
    window_log.trim().len() >= MIN_LOG_BYTES_FOR_ANALYSIS
}

/// The history entry one created issue contributes. Pure — the entry a cycle
/// appends is a function of the proposal, the assigned IID, and the cycle
/// clock, never of the order the port happened to answer in.
fn history_entry(
    proposal: &OpsIssueProposal,
    issue_iid: u64,
    now_iso: &str,
) -> OpsIssueHistoryEntry {
    OpsIssueHistoryEntry {
        gitlab_issue_iid: issue_iid,
        title: proposal.title.clone(),
        log_line: proposal.log_line.clone(),
        created_at: now_iso.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Ops state machine
// ---------------------------------------------------------------------------

/// Where the ops cycle is. Each variant names the single next observation,
/// action, or pure transition, so [`OpsMachine::next_step`] is a function of
/// this plus what the machine already learned — never of the world.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsStage {
    ShutdownBeforeCycle,
    ObserveClock,
    WriteGitLabContext,
    ShutdownAfterContext,
    NextLogSource,
    ObserveLogTail,
    ShutdownAfterLogTail,
    WriteScrapeFile,
    PruneScrapeFiles,
    ObserveHistory,
    EnsureHistoryFile,
    WriteAnalysisFile,
    ObserveHistoryPath,
    InvokeAnalysisModel,
    NextProposal,
    CreateProposalIssue,
    AddProposalPriorityLabel,
    AddProposalScopeLabel,
    SaveHistory,
    Finish,
}

/// The ops role's orchestration state. Holds only plain data — no GitLab
/// client, no ssh command, no model — so every decision it makes is a pure
/// function of what it has observed so far.
struct OpsMachine<'a> {
    agent_id: &'a str,
    has_scope_label: bool,
    stage: OpsStage,
    sources: std::collections::VecDeque<LogSourceObservation>,
    source: Option<LogSourceObservation>,
    clock: Option<CycleClock>,
    raw_tail: String,
    window_sections: Vec<String>,
    window_log: String,
    gitlab_context_path: String,
    analysis_path: String,
    history_path: String,
    history: OpsIssueHistory,
    proposals: std::collections::VecDeque<OpsIssueProposal>,
    proposal: Option<OpsIssueProposal>,
    created_iid: Option<u64>,
    created: usize,
}

impl<'a> OpsMachine<'a> {
    fn new(agent_id: &'a str, sources: &[OpsLogSource], has_scope_label: bool) -> Self {
        Self {
            agent_id,
            has_scope_label,
            stage: OpsStage::ShutdownBeforeCycle,
            sources: sources
                .iter()
                .map(LogSourceObservation::from_source)
                .collect(),
            source: None,
            clock: None,
            raw_tail: String::new(),
            window_sections: Vec::new(),
            window_log: String::new(),
            gitlab_context_path: String::new(),
            analysis_path: String::new(),
            history_path: String::new(),
            history: OpsIssueHistory::default(),
            proposals: std::collections::VecDeque::new(),
            proposal: None,
            created_iid: None,
            created: 0,
        }
    }

    fn clock(&self) -> CycleClock {
        self.clock
            .expect("cycle stages run only after the clock is observed")
    }

    fn source(&self) -> &LogSourceObservation {
        self.source
            .as_ref()
            .expect("a log source is set before it is scraped")
    }

    fn proposal(&self) -> &OpsIssueProposal {
        self.proposal
            .as_ref()
            .expect("a proposal is set before it is filed")
    }

    fn created_iid(&self) -> u64 {
        self.created_iid
            .expect("labels are only applied after the issue was created")
    }

    /// The single next thing to do. Pure: it only resolves stages that need
    /// no port interaction (the log window join, the analyzability gate, the
    /// per-proposal label decisions) before handing back an observation or an
    /// action.
    fn next_step(&mut self) -> OpsStep {
        loop {
            match &self.stage {
                OpsStage::ShutdownBeforeCycle
                | OpsStage::ShutdownAfterContext
                | OpsStage::ShutdownAfterLogTail => {
                    return OpsStep::Observe(OpsQuery::ShutdownRequested);
                }
                OpsStage::ObserveClock => return OpsStep::Observe(OpsQuery::CycleClock),
                OpsStage::WriteGitLabContext => {
                    info!(
                        "{}: Fetching GitLab issues and merge requests for deduplication context",
                        self.agent_id
                    );
                    return OpsStep::Act(OpsAction::WriteGitLabContextFile {
                        unix_ts: self.clock().unix_ts,
                    });
                }
                OpsStage::NextLogSource => match self.sources.pop_front() {
                    Some(source) => {
                        self.source = Some(source);
                        self.stage = OpsStage::ObserveLogTail;
                    }
                    None => {
                        self.window_log = self.window_sections.join("\n\n");
                        if window_is_analyzable(&self.window_log) {
                            self.stage = OpsStage::WriteScrapeFile;
                        } else {
                            info!(
                                "{}: Log window too small to analyze ({} bytes)",
                                self.agent_id,
                                self.window_log.trim().len()
                            );
                            self.stage = OpsStage::Finish;
                        }
                    }
                },
                OpsStage::ObserveLogTail => {
                    let source = self.source().clone();
                    info!(
                        "{}: Fetching last {}h of logs from {}",
                        self.agent_id,
                        LOG_WINDOW_HOURS,
                        source.target()
                    );
                    return OpsStep::Observe(OpsQuery::RemoteLogTail { source });
                }
                OpsStage::WriteScrapeFile => {
                    return OpsStep::Act(OpsAction::WriteScrapeFile {
                        unix_ts: self.clock().unix_ts,
                        window_log: self.window_log.clone(),
                    });
                }
                OpsStage::PruneScrapeFiles => return OpsStep::Act(OpsAction::PruneScrapeFiles),
                OpsStage::ObserveHistory => return OpsStep::Observe(OpsQuery::IssueHistory),
                OpsStage::EnsureHistoryFile => {
                    return OpsStep::Act(OpsAction::EnsureHistoryFile(self.history.clone()));
                }
                OpsStage::WriteAnalysisFile => {
                    return OpsStep::Act(OpsAction::WriteAnalysisFile {
                        unix_ts: self.clock().unix_ts,
                        window_log: self.window_log.clone(),
                    });
                }
                OpsStage::ObserveHistoryPath => {
                    return OpsStep::Observe(OpsQuery::HistoryFilePath);
                }
                OpsStage::InvokeAnalysisModel => {
                    return OpsStep::Act(OpsAction::InvokeAnalysisModel {
                        prompt: build_analysis_prompt(
                            &self.analysis_path,
                            &self.history_path,
                            &self.gitlab_context_path,
                        ),
                    });
                }
                OpsStage::NextProposal => match self.proposals.pop_front() {
                    Some(proposal) => {
                        self.proposal = Some(proposal);
                        self.created_iid = None;
                        self.stage = OpsStage::CreateProposalIssue;
                    }
                    None => {
                        self.stage = if self.created > 0 {
                            OpsStage::SaveHistory
                        } else {
                            OpsStage::Finish
                        };
                    }
                },
                OpsStage::CreateProposalIssue => {
                    let proposal = self.proposal();
                    return OpsStep::Act(OpsAction::CreateIssue {
                        title: proposal.title.clone(),
                        description: proposal.description.clone(),
                    });
                }
                OpsStage::AddProposalPriorityLabel => match self.proposal().priority {
                    Some(priority) => {
                        return OpsStep::Act(OpsAction::AddPriorityLabel {
                            issue_iid: self.created_iid(),
                            priority,
                        });
                    }
                    None => self.stage = OpsStage::AddProposalScopeLabel,
                },
                OpsStage::AddProposalScopeLabel => {
                    if !self.has_scope_label {
                        self.stage = OpsStage::NextProposal;
                        continue;
                    }
                    return OpsStep::Act(OpsAction::AddScopeLabel {
                        issue_iid: self.created_iid(),
                    });
                }
                OpsStage::SaveHistory => {
                    return OpsStep::Act(OpsAction::SaveHistory(self.history.clone()));
                }
                OpsStage::Finish => return OpsStep::Finish,
            }
        }
    }

    /// Feed back the answer to the observation the machine just asked for.
    /// `Err` here means the cycle itself fails, exactly where the original
    /// code used `?` on a read.
    fn apply_fact(&mut self, fact: Result<OpsFact>) -> Result<()> {
        match (&self.stage, fact) {
            (OpsStage::ShutdownBeforeCycle, Ok(OpsFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    OpsStage::Finish
                } else {
                    OpsStage::ObserveClock
                };
            }
            (OpsStage::ObserveClock, Ok(OpsFact::CycleClock(clock))) => {
                self.clock = Some(clock);
                self.stage = OpsStage::WriteGitLabContext;
            }
            (OpsStage::ShutdownAfterContext, Ok(OpsFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    OpsStage::Finish
                } else {
                    OpsStage::NextLogSource
                };
            }
            (OpsStage::ObserveLogTail, Ok(OpsFact::RemoteLogTail(raw))) => {
                self.raw_tail = raw;
                self.stage = OpsStage::ShutdownAfterLogTail;
            }
            (OpsStage::ShutdownAfterLogTail, Ok(OpsFact::ShutdownRequested(stop))) => {
                if stop {
                    self.stage = OpsStage::Finish;
                } else {
                    let now = self.clock().now;
                    let (window_log, parsed_timestamps) =
                        filter_log_to_time_window(&self.raw_tail, now);
                    if !parsed_timestamps {
                        warn!(
                            "{}: Could not parse timestamps in remote log tail for {}; using full tail for analysis",
                            self.agent_id,
                            self.source().target()
                        );
                    }
                    if !window_log.trim().is_empty() {
                        self.window_sections
                            .push(self.source().section(&window_log));
                    }
                    self.stage = OpsStage::NextLogSource;
                }
            }
            (OpsStage::ObserveHistory, Ok(OpsFact::IssueHistory(history))) => {
                self.history = history;
                self.stage = OpsStage::EnsureHistoryFile;
            }
            (OpsStage::ObserveHistoryPath, Ok(OpsFact::HistoryFilePath(path))) => {
                self.history_path = path;
                self.stage = OpsStage::InvokeAnalysisModel;
            }
            (stage, Ok(fact)) => {
                anyhow::bail!("ops port answered {stage:?} with {fact:?}");
            }
            (_, Err(e)) => return Err(e),
        }
        Ok(())
    }

    /// Feed back the outcome of the action the machine just asked for.
    fn apply_outcome(&mut self, outcome: OpsOutcome) -> Result<()> {
        match (&self.stage, outcome) {
            (OpsStage::WriteGitLabContext, OpsOutcome::ContextFileWritten(path)) => {
                self.gitlab_context_path = path;
                self.stage = OpsStage::ShutdownAfterContext;
            }
            (OpsStage::WriteScrapeFile, OpsOutcome::Done) => {
                self.stage = OpsStage::PruneScrapeFiles;
            }
            (OpsStage::PruneScrapeFiles, OpsOutcome::Done) => {
                self.stage = OpsStage::ObserveHistory;
            }
            (OpsStage::EnsureHistoryFile, OpsOutcome::Done) => {
                self.stage = OpsStage::WriteAnalysisFile;
            }
            (OpsStage::WriteAnalysisFile, OpsOutcome::AnalysisFileWritten(path)) => {
                self.analysis_path = path;
                self.stage = OpsStage::ObserveHistoryPath;
            }
            (OpsStage::InvokeAnalysisModel, OpsOutcome::Analyzed(issues)) => {
                self.proposals = normalize_ops_issues(issues).into();
                if self.proposals.is_empty() {
                    info!(
                        "{}: No new actionable errors found in log window",
                        self.agent_id
                    );
                    self.stage = OpsStage::Finish;
                } else {
                    self.stage = OpsStage::NextProposal;
                }
            }
            (OpsStage::CreateProposalIssue, OpsOutcome::IssueCreated(issue_iid)) => {
                info!(
                    "{}: Created GitLab issue #{}: {}",
                    self.agent_id,
                    issue_iid,
                    self.proposal().title
                );
                self.created_iid = Some(issue_iid);
                let entry =
                    history_entry(self.proposal(), issue_iid, &self.clock().now.to_rfc3339());
                self.history.entries.push(entry);
                self.created += 1;
                self.stage = OpsStage::AddProposalPriorityLabel;
            }
            // A create that fails is warned about and skipped: no history
            // entry, and the cycle moves on to the next proposal.
            (OpsStage::CreateProposalIssue, OpsOutcome::Failed(e)) => {
                warn!(
                    "{}: Failed to create issue for {}: {}",
                    self.agent_id,
                    self.proposal().title,
                    e
                );
                self.stage = OpsStage::NextProposal;
            }
            (OpsStage::AddProposalPriorityLabel, OpsOutcome::Done) => {
                self.stage = OpsStage::AddProposalScopeLabel;
            }
            (OpsStage::AddProposalPriorityLabel, OpsOutcome::Failed(e)) => {
                warn!(
                    "{}: Failed to set priority label on #{}: {}",
                    self.agent_id,
                    self.created_iid(),
                    e
                );
                self.stage = OpsStage::AddProposalScopeLabel;
            }
            (OpsStage::AddProposalScopeLabel, OpsOutcome::Done) => {
                self.stage = OpsStage::NextProposal;
            }
            (OpsStage::AddProposalScopeLabel, OpsOutcome::Failed(e)) => {
                warn!(
                    "{}: Failed to add scope label on #{}: {}",
                    self.agent_id,
                    self.created_iid(),
                    e
                );
                self.stage = OpsStage::NextProposal;
            }
            (OpsStage::SaveHistory, OpsOutcome::Done) => {
                info!(
                    "{}: Created {} new GitLab issue(s) from log analysis",
                    self.agent_id, self.created
                );
                self.stage = OpsStage::Finish;
            }
            // Everything else the cycle performed with `?` aborts it.
            (_, OpsOutcome::Failed(e)) => return Err(e),
            (stage, _) => {
                anyhow::bail!("ops port reported an unexpected outcome for {stage:?}");
            }
        }
        Ok(())
    }
}

/// Ask the port one question.
fn observe_ops(port: &dyn OpsPort, query: &OpsQuery) -> Result<OpsFact> {
    Ok(match query {
        OpsQuery::ShutdownRequested => OpsFact::ShutdownRequested(port.shutdown_requested()),
        OpsQuery::CycleClock => OpsFact::CycleClock(port.cycle_clock()),
        OpsQuery::RemoteLogTail { source } => OpsFact::RemoteLogTail(port.remote_log_tail(source)?),
        OpsQuery::IssueHistory => OpsFact::IssueHistory(port.issue_history()?),
        OpsQuery::HistoryFilePath => OpsFact::HistoryFilePath(port.history_file_path()?),
    })
}

fn run_ops_cycle(machine: &mut OpsMachine, port: &mut dyn OpsPort) -> Result<()> {
    loop {
        match machine.next_step() {
            Step::Observe(query) => machine.apply_fact(observe_ops(port, &query))?,
            Step::Act(action) => machine.apply_outcome(port.execute(&action))?,
            Step::Finish => return Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Live ops port
// ---------------------------------------------------------------------------

/// The ops port backed by the real runtime: this file's only place where an
/// ops decision meets ssh, the filesystem, GitLab, or the model.
struct LiveOpsPort<'a> {
    state: &'a AgentState<'a>,
    gitlab: &'a GitLabClient,
    model: &'a AgentModel,
    shutdown: &'a AtomicBool,
    scope_label: Option<&'a str>,
}

fn required(result: Result<()>) -> OpsOutcome {
    match result {
        Ok(()) => OpsOutcome::Done,
        Err(e) => OpsOutcome::Failed(e),
    }
}

impl OpsPort for LiveOpsPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn cycle_clock(&self) -> CycleClock {
        CycleClock {
            unix_ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            now: Utc::now(),
        }
    }

    fn remote_log_tail(&self, source: &LogSourceObservation) -> Result<String> {
        fetch_remote_log_tail(&source.ssh_user, &source.ssh_host, &source.log_path)
    }

    fn issue_history(&self) -> Result<OpsIssueHistory> {
        load_history(&self.state.history_path())
    }

    fn history_file_path(&self) -> Result<String> {
        absolute_path(&self.state.history_path())
    }

    fn execute(&mut self, action: &OpsAction) -> OpsOutcome {
        match action {
            OpsAction::WriteGitLabContextFile { unix_ts } => {
                match write_gitlab_context_file(self.state, self.gitlab, *unix_ts) {
                    Ok(path) => OpsOutcome::ContextFileWritten(path),
                    Err(e) => OpsOutcome::Failed(e),
                }
            }
            OpsAction::WriteScrapeFile {
                unix_ts,
                window_log,
            } => {
                let scrape_path = self.state.scrape_path(*unix_ts);
                required(fs::write(&scrape_path, window_log).with_context(|| {
                    format!("Failed to write scrape file {}", scrape_path.display())
                }))
            }
            OpsAction::PruneScrapeFiles => {
                required(prune_old_scrape_files(self.state, MAX_SCRAPE_FILES_KEPT))
            }
            OpsAction::EnsureHistoryFile(history) => {
                required(ensure_history_file(self.state, history))
            }
            OpsAction::WriteAnalysisFile {
                unix_ts,
                window_log,
            } => {
                match write_task_context_file(
                    self.state.sessions_dir,
                    &format!("{}-analysis-{unix_ts}.log", self.state.agent_id),
                    window_log,
                ) {
                    Ok(path) => OpsOutcome::AnalysisFileWritten(path),
                    Err(e) => OpsOutcome::Failed(e),
                }
            }
            OpsAction::InvokeAnalysisModel { prompt } => self.invoke_analysis_model(prompt),
            OpsAction::CreateIssue { title, description } => {
                match self.gitlab.create_issue(title, description) {
                    Ok(issue_iid) => OpsOutcome::IssueCreated(issue_iid),
                    Err(e) => OpsOutcome::Failed(e),
                }
            }
            OpsAction::AddPriorityLabel {
                issue_iid,
                priority,
            } => required(
                self.gitlab
                    .add_issue_label(*issue_iid, &gitlab::priority_label(*priority)),
            ),
            OpsAction::AddScopeLabel { issue_iid } => {
                let Some(label) = self.scope_label else {
                    return OpsOutcome::Done;
                };
                required(self.gitlab.add_issue_label(*issue_iid, label))
            }
            OpsAction::SaveHistory(history) => {
                required(save_history(&self.state.history_path(), history))
            }
        }
    }
}

impl LiveOpsPort<'_> {
    fn invoke_analysis_model(&self, prompt: &str) -> OpsOutcome {
        match self.model.complete_typed::<OpsOutput>(
            prompt,
            &InvokeOptions {
                activity_label: Some(format!("{} analyzing logs", self.state.agent_id)),
                ..InvokeOptions::default()
            },
        ) {
            Ok(completion) => OpsOutcome::Analyzed(completion.output.issues),
            Err(e) => OpsOutcome::Failed(e),
        }
    }
}

fn ops_cycle(
    state: &AgentState,
    config: &OpsConfig,
    gitlab: &GitLabClient,
    model: &AgentModel,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut machine = OpsMachine::new(state.agent_id, &config.log_sources, scope_label.is_some());
    let mut port = LiveOpsPort {
        state,
        gitlab,
        model,
        shutdown: shutdown.as_ref(),
        scope_label,
    };
    run_ops_cycle(&mut machine, &mut port)
}

fn build_analysis_prompt(log_path: &str, history_path: &str, gitlab_context_path: &str) -> String {
    format!(
        r#"You are an operations agent triaging log data for errors and exceptions.

Read these context files before proposing any new GitLab issues:

1. Log session file (primary analysis input; may contain multiple source sections):
{log_path}

2. Ops issue history (issues previously created by this agent, with related log lines):
{history_path}

3. Current GitLab context (all open issues and merge requests with descriptions and comments):
{gitlab_context_path}

Analyze ONLY the log session file for new errors, exceptions, panics, fatal failures, or repeated error patterns.

Use the issue history and GitLab context files to avoid creating duplicate issues for problems that are already tracked, discussed, or being addressed.

For each candidate problem, inspect the current project codebase in your workspace before proposing an issue. Use the code to decide whether the logged failure is still actionable.

If the current code appears to already fix or guard against the logged failure, do not propose an issue for it unless the log evidence clearly shows the fixed code path is still failing. Treat those cases as already addressed.

Treat explicit code or doc comments describing a behavior, fallback, limitation, or error-handling path as evidence that the handling is intentional. Do not propose an issue whose requested "fix" would merely reverse or remove that explicitly documented behavior. This applies only to comments that clearly describe the exact path under analysis, not TODOs, guesses, or unrelated commentary.

Use the project codebase to map log errors to likely code paths, root causes, and concrete remediation steps. Do not rely on production host, deployment, or infrastructure details beyond what appears in the log file.

For each NEW distinct problem that is not already covered, propose one GitLab issue.

Rules:
- Report no issues if there are no NEW actionable errors.
- Do not propose issues for errors already represented in the history or GitLab context files.
- Do not propose issues for errors that appear already fixed in the current codebase.
- Do not propose issues to change behavior that a relevant code or doc comment explicitly identifies as intentional.
- Titles must be specific and actionable.
- Descriptions must cite log evidence and relevant code context from the repository.
"#,
        log_path = log_path,
        history_path = history_path,
        gitlab_context_path = gitlab_context_path,
    )
}

fn absolute_path(path: &Path) -> Result<String> {
    Ok(fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned())
}

fn ensure_history_file(state: &AgentState, history: &OpsIssueHistory) -> Result<()> {
    let path = state.history_path();
    if path.exists() {
        return Ok(());
    }
    save_history(&path, history)
}

fn write_gitlab_context_file(
    state: &AgentState,
    gitlab: &GitLabClient,
    unix_ts: u64,
) -> Result<String> {
    let content = build_gitlab_context(gitlab)?;
    write_task_context_file(
        state.sessions_dir,
        &format!("{}-gitlab-context-{unix_ts}.md", state.agent_id),
        &content,
    )
}

fn build_gitlab_context(gitlab: &GitLabClient) -> Result<String> {
    let mut out = String::from("# Current GitLab issues and merge requests\n\n");

    let issues = gitlab.list_issues()?;
    out.push_str("## Open issues\n\n");
    if issues.is_empty() {
        out.push_str("(none)\n\n");
    } else {
        for issue in &issues {
            append_issue_context(&mut out, gitlab, issue)?;
        }
    }

    let mrs = gitlab.list_merge_requests()?;
    out.push_str("## Open merge requests\n\n");
    if mrs.is_empty() {
        out.push_str("(none)\n");
    } else {
        for mr in &mrs {
            append_mr_context(&mut out, gitlab, mr)?;
        }
    }

    Ok(out)
}

fn append_issue_context(
    out: &mut String,
    gitlab: &GitLabClient,
    issue: &gitlab::Issue,
) -> Result<()> {
    let labels = if issue.labels.is_empty() {
        "none".to_string()
    } else {
        issue.labels.join(", ")
    };
    out.push_str(&format!(
        "### Issue #{}: {}\nLabels: {}\n\nDescription:\n{}\n",
        issue.iid, issue.title, labels, issue.description
    ));

    match gitlab.get_issue_comments(issue.iid) {
        Ok(comments) if !comments.is_empty() => {
            out.push_str("\nComments:\n");
            for comment in &comments {
                out.push_str(&format!("- {}: {}\n", comment.author, comment.body));
            }
        }
        Ok(_) => {}
        Err(e) => {
            warn!(
                "Ops: Failed to fetch comments for issue #{}: {}",
                issue.iid, e
            );
        }
    }

    out.push('\n');
    Ok(())
}

fn append_mr_context(
    out: &mut String,
    gitlab: &GitLabClient,
    mr: &gitlab::MergeRequest,
) -> Result<()> {
    let labels = mr
        .labels
        .as_ref()
        .filter(|labels| !labels.is_empty())
        .map(|labels| labels.join(", "))
        .unwrap_or_else(|| "none".to_string());
    out.push_str(&format!(
        "### MR !{}: {}\nLabels: {}\nSource branch: {}\nTarget branch: {}\n\nDescription:\n{}\n",
        mr.iid, mr.title, labels, mr.source_branch, mr.target_branch, mr.description
    ));

    match gitlab.get_mr_comments(mr.iid) {
        Ok(comments) if !comments.is_empty() => {
            out.push_str("\nComments:\n");
            for comment in &comments {
                out.push_str(&comment.format_for_prompt());
                out.push('\n');
            }
        }
        Ok(_) => {}
        Err(e) => {
            warn!("Ops: Failed to fetch comments for MR !{}: {}", mr.iid, e);
        }
    }

    out.push('\n');
    Ok(())
}

fn load_history(path: &Path) -> Result<OpsIssueHistory> {
    // Strict: unlike worker/PMO-claim/QA, a corrupt or unsupported history
    // file is quarantined (never losing bytes) but still surfaced as an
    // error rather than silently reset — the caller decides whether to
    // fail the cycle.
    let store = crate::agents::state::StateStore::new(path);
    if path.exists() {
        let content = fs::read(path)
            .with_context(|| format!("Failed to read issue history at {}", path.display()))?;
        if content.iter().all(u8::is_ascii_whitespace) {
            return Ok(OpsIssueHistory::default());
        }
    }
    Ok(store.load()?.unwrap_or_default())
}

fn save_history(path: &Path, history: &OpsIssueHistory) -> Result<()> {
    crate::agents::state::StateStore::new(path).save(history)
}

fn fetch_remote_log_tail(ssh_user: &str, ssh_host: &str, log_path: &str) -> Result<String> {
    validate_ssh_identity(ssh_user, "ssh_user")?;
    validate_ssh_identity(ssh_host, "ssh_host")?;
    validate_remote_path(log_path)?;

    let target = format!("{ssh_user}@{ssh_host}");
    let quoted_path = shell_single_quote(log_path);
    let remote_cmd = format!("test -r {quoted_path} && tail -n {MAX_TAIL_LINES} {quoted_path}");

    let output = Command::new("ssh")
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            &target,
            &remote_cmd,
        ])
        .output()
        .with_context(|| format!("Failed to execute ssh for {target}"))?;

    if !output.status.success() {
        bail!(
            "ssh log fetch failed for {target}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_line_timestamp(line: &str) -> Option<DateTime<Utc>> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return None;
    }

    let candidates = [
        trimmed.get(0..32).unwrap_or(trimmed),
        trimmed.get(0..26).unwrap_or(trimmed),
        trimmed.get(0..23).unwrap_or(trimmed),
        trimmed.get(0..19).unwrap_or(trimmed),
    ];

    for candidate in candidates {
        if let Ok(dt) = DateTime::parse_from_rfc3339(candidate) {
            return Some(dt.with_timezone(&Utc));
        }
        if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, "%Y-%m-%d %H:%M:%S") {
            return Some(Utc.from_utc_datetime(&dt));
        }
        if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, "%Y/%m/%d %H:%M:%S") {
            return Some(Utc.from_utc_datetime(&dt));
        }
        if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, "%Y-%m-%dT%H:%M:%S") {
            return Some(Utc.from_utc_datetime(&dt));
        }
    }

    None
}

fn filter_log_to_time_window(log: &str, now: DateTime<Utc>) -> (String, bool) {
    let cutoff = now - chrono::Duration::hours(LOG_WINDOW_HOURS);
    let mut saw_timestamp = false;
    let mut include_continuation = false;
    let mut kept = Vec::new();

    for line in log.lines() {
        if let Some(ts) = parse_line_timestamp(line) {
            saw_timestamp = true;
            include_continuation = ts >= cutoff;
            if include_continuation {
                kept.push(line);
            }
        } else if include_continuation {
            kept.push(line);
        }
    }

    if !saw_timestamp {
        return (log.to_string(), false);
    }

    (kept.join("\n"), true)
}

fn prune_old_scrape_files(state: &AgentState, keep: usize) -> Result<()> {
    let prefix = format!("{}-scrape-", state.agent_id);
    let mut scrape_files = Vec::new();
    for entry in fs::read_dir(state.sessions_dir)
        .with_context(|| format!("Failed to read {}", state.sessions_dir))?
    {
        let entry = entry.context("Failed to read sessions directory entry")?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if name.starts_with(&prefix) && name.ends_with(".log") {
            scrape_files.push(entry.path());
        }
    }

    if scrape_files.len() <= keep {
        return Ok(());
    }

    scrape_files.sort_unstable();
    let delete_count = scrape_files.len().saturating_sub(keep);
    for path in scrape_files.into_iter().take(delete_count) {
        if let Err(e) = fs::remove_file(&path) {
            warn!("Failed to remove old scrape file {}: {e}", path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::conformance;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    // -----------------------------------------------------------------
    // Ops state machine driven against a recording fake port. Every
    // observation and mutation the cycle performs lands in one ordered
    // trace, so a full scrape/analyze/file flow can be replayed — and its
    // exact mutation order asserted — without ssh, GitLab, the filesystem,
    // or a model.
    // -----------------------------------------------------------------

    const TEST_AGENT: &str = "ops-0";
    const TEST_UNIX_TS: u64 = 1_781_000_000;
    /// A log line inside the two-hour window measured back from
    /// [`test_clock`], so the real timestamp filter is exercised.
    const RECENT_LOG: &str = "2026-06-15 11:30:00 ERROR connection pool exhausted";
    const CONTEXT_PATH: &str = "/sessions/ops-0-gitlab-context.md";
    const ANALYSIS_PATH: &str = "/sessions/ops-0-analysis.log";
    const HISTORY_PATH: &str = "/sessions/ops-0_issue_history.json";

    fn test_clock() -> CycleClock {
        CycleClock {
            unix_ts: TEST_UNIX_TS,
            now: Utc.with_ymd_and_hms(2026, 6, 15, 12, 0, 0).unwrap(),
        }
    }

    fn log_source(host: &str) -> OpsLogSource {
        OpsLogSource {
            ssh_user: "deploy".to_string(),
            ssh_host: host.to_string(),
            log_path: "/var/log/app/app.log".to_string(),
        }
    }

    fn observation(host: &str) -> LogSourceObservation {
        LogSourceObservation::from_source(&log_source(host))
    }

    /// The joined analysis input the machine assembles for `hosts`, in
    /// configuration order.
    fn expected_window(hosts: &[&str]) -> String {
        hosts
            .iter()
            .map(|host| observation(host).section(RECENT_LOG))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn proposal(title: &str, priority: Option<u64>) -> RawOpsIssue {
        RawOpsIssue {
            title: title.to_string(),
            description: format!("{title} — log evidence and remediation."),
            priority,
            log_line: RECENT_LOG.to_string(),
        }
    }

    /// A recording ops port. Answers are scripted per observation kind, and
    /// failures are injected by naming the step that should fail, so tests
    /// read as "this world, then this trace".
    struct FakeOpsPort {
        trace: RefCell<Vec<OpsStep>>,
        shutdown_answers: RefCell<VecDeque<bool>>,
        clock: CycleClock,
        log_tail: String,
        history: OpsIssueHistory,
        analysis: Vec<RawOpsIssue>,
        next_iid: Cell<u64>,
        saved_history: RefCell<Option<OpsIssueHistory>>,
        ensured_history: RefCell<Option<OpsIssueHistory>>,
        failing_queries: Vec<OpsQuery>,
        failing_actions: Vec<std::mem::Discriminant<OpsAction>>,
        failing_issue_titles: Vec<String>,
    }

    impl FakeOpsPort {
        fn new(analysis: Vec<RawOpsIssue>) -> Self {
            Self {
                trace: RefCell::new(Vec::new()),
                shutdown_answers: RefCell::new(VecDeque::new()),
                clock: test_clock(),
                log_tail: RECENT_LOG.to_string(),
                history: OpsIssueHistory::default(),
                analysis,
                next_iid: Cell::new(100),
                saved_history: RefCell::new(None),
                ensured_history: RefCell::new(None),
                failing_queries: Vec::new(),
                failing_actions: Vec::new(),
                failing_issue_titles: Vec::new(),
            }
        }

        fn with_shutdown_answers(self, answers: &[bool]) -> Self {
            self.shutdown_answers.borrow_mut().extend(answers);
            self
        }

        fn with_log_tail(mut self, tail: &str) -> Self {
            self.log_tail = tail.to_string();
            self
        }

        fn with_history(mut self, history: OpsIssueHistory) -> Self {
            self.history = history;
            self
        }

        fn failing_query(mut self, query: OpsQuery) -> Self {
            self.failing_queries.push(query);
            self
        }

        /// Fail one action variant regardless of its payload. Naming the step
        /// keeps the test readable where the payload is a whole log window or
        /// a rendered prompt.
        fn failing_action(mut self, action: &OpsAction) -> Self {
            self.failing_actions.push(std::mem::discriminant(action));
            self
        }

        fn failing_issue(mut self, title: &str) -> Self {
            self.failing_issue_titles.push(title.to_string());
            self
        }

        fn record(&self, step: OpsStep) {
            self.trace.borrow_mut().push(step);
        }

        fn observe(&self, query: OpsQuery) -> Result<()> {
            self.record(OpsStep::Observe(query.clone()));
            if self.failing_queries.contains(&query) {
                anyhow::bail!("injected failure observing {query:?}");
            }
            Ok(())
        }
    }

    impl OpsPort for FakeOpsPort {
        fn shutdown_requested(&self) -> bool {
            self.record(OpsStep::Observe(OpsQuery::ShutdownRequested));
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }

        fn cycle_clock(&self) -> CycleClock {
            self.record(OpsStep::Observe(OpsQuery::CycleClock));
            self.clock
        }

        fn remote_log_tail(&self, source: &LogSourceObservation) -> Result<String> {
            self.observe(OpsQuery::RemoteLogTail {
                source: source.clone(),
            })?;
            Ok(self.log_tail.clone())
        }

        fn issue_history(&self) -> Result<OpsIssueHistory> {
            self.observe(OpsQuery::IssueHistory)?;
            Ok(self.history.clone())
        }

        fn history_file_path(&self) -> Result<String> {
            self.observe(OpsQuery::HistoryFilePath)?;
            Ok(HISTORY_PATH.to_string())
        }

        fn execute(&mut self, action: &OpsAction) -> OpsOutcome {
            self.record(OpsStep::Act(action.clone()));
            if self
                .failing_actions
                .contains(&std::mem::discriminant(action))
            {
                return OpsOutcome::Failed(anyhow::anyhow!(
                    "injected failure executing {action:?}"
                ));
            }
            match action {
                OpsAction::WriteGitLabContextFile { .. } => {
                    OpsOutcome::ContextFileWritten(CONTEXT_PATH.to_string())
                }
                OpsAction::WriteAnalysisFile { .. } => {
                    OpsOutcome::AnalysisFileWritten(ANALYSIS_PATH.to_string())
                }
                OpsAction::EnsureHistoryFile(history) => {
                    *self.ensured_history.borrow_mut() = Some(history.clone());
                    OpsOutcome::Done
                }
                OpsAction::InvokeAnalysisModel { .. } => {
                    OpsOutcome::Analyzed(self.analysis.clone())
                }
                OpsAction::CreateIssue { title, .. } => {
                    if self.failing_issue_titles.iter().any(|t| t == title) {
                        return OpsOutcome::Failed(anyhow::anyhow!(
                            "injected failure creating issue {title:?}"
                        ));
                    }
                    let iid = self.next_iid.get();
                    self.next_iid.set(iid + 1);
                    OpsOutcome::IssueCreated(iid)
                }
                OpsAction::SaveHistory(history) => {
                    *self.saved_history.borrow_mut() = Some(history.clone());
                    OpsOutcome::Done
                }
                _ => OpsOutcome::Done,
            }
        }
    }

    struct FakeRun {
        result: Result<()>,
        trace: Vec<OpsStep>,
        saved_history: Option<OpsIssueHistory>,
        ensured_history: Option<OpsIssueHistory>,
    }

    fn run_ops(port: &mut FakeOpsPort, hosts: &[&str], has_scope_label: bool) -> FakeRun {
        let sources: Vec<OpsLogSource> = hosts.iter().copied().map(log_source).collect();
        let mut machine = OpsMachine::new(TEST_AGENT, &sources, has_scope_label);
        let result = run_ops_cycle(&mut machine, port);
        FakeRun {
            result,
            trace: port.trace.borrow().clone(),
            saved_history: port.saved_history.borrow().clone(),
            ensured_history: port.ensured_history.borrow().clone(),
        }
    }

    fn observe(query: OpsQuery) -> OpsStep {
        OpsStep::Observe(query)
    }

    fn act(action: OpsAction) -> OpsStep {
        OpsStep::Act(action)
    }

    fn shutdown() -> OpsStep {
        observe(OpsQuery::ShutdownRequested)
    }

    /// The steps every cycle performs from the first shutdown check through
    /// the model invocation, for a single log source whose window is large
    /// enough to analyze.
    fn steps_up_to_analysis(hosts: &[&str]) -> Vec<OpsStep> {
        let window = expected_window(hosts);
        let mut steps = vec![
            shutdown(),
            observe(OpsQuery::CycleClock),
            act(OpsAction::WriteGitLabContextFile {
                unix_ts: TEST_UNIX_TS,
            }),
            shutdown(),
        ];
        for host in hosts {
            steps.push(observe(OpsQuery::RemoteLogTail {
                source: observation(host),
            }));
            steps.push(shutdown());
        }
        steps.extend([
            act(OpsAction::WriteScrapeFile {
                unix_ts: TEST_UNIX_TS,
                window_log: window.clone(),
            }),
            act(OpsAction::PruneScrapeFiles),
            observe(OpsQuery::IssueHistory),
            act(OpsAction::EnsureHistoryFile(OpsIssueHistory::default())),
            act(OpsAction::WriteAnalysisFile {
                unix_ts: TEST_UNIX_TS,
                window_log: window,
            }),
            observe(OpsQuery::HistoryFilePath),
            act(OpsAction::InvokeAnalysisModel {
                prompt: build_analysis_prompt(ANALYSIS_PATH, HISTORY_PATH, CONTEXT_PATH),
            }),
        ]);
        steps
    }

    fn created_entry(iid: u64, title: &str) -> OpsIssueHistoryEntry {
        OpsIssueHistoryEntry {
            gitlab_issue_iid: iid,
            title: title.to_string(),
            log_line: RECENT_LOG.to_string(),
            created_at: test_clock().now.to_rfc3339(),
        }
    }

    #[test]
    fn ops_cycle_files_each_issue_as_create_then_priority_then_scope_and_saves_history_once() {
        let mut port = FakeOpsPort::new(vec![
            proposal("Fix DB timeout", Some(1)),
            proposal("Fix memory leak", None),
        ]);
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        let mut expected = steps_up_to_analysis(&["prod-1.example.com"]);
        expected.extend([
            act(OpsAction::CreateIssue {
                title: "Fix DB timeout".to_string(),
                description: "Fix DB timeout — log evidence and remediation.".to_string(),
            }),
            act(OpsAction::AddPriorityLabel {
                issue_iid: 100,
                priority: 1,
            }),
            act(OpsAction::AddScopeLabel { issue_iid: 100 }),
            act(OpsAction::CreateIssue {
                title: "Fix memory leak".to_string(),
                description: "Fix memory leak — log evidence and remediation.".to_string(),
            }),
            // No priority on this proposal, so the scope label follows the
            // create directly.
            act(OpsAction::AddScopeLabel { issue_iid: 101 }),
            act(OpsAction::SaveHistory(OpsIssueHistory {
                entries: vec![
                    created_entry(100, "Fix DB timeout"),
                    created_entry(101, "Fix memory leak"),
                ],
            })),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn ops_cycle_stops_before_any_side_effect_when_shutdown_is_already_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[true]);
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        assert_eq!(run.trace, vec![shutdown()]);
    }

    #[test]
    fn ops_cycle_stops_after_the_context_file_when_shutdown_is_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[false, true]);
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                shutdown(),
                observe(OpsQuery::CycleClock),
                act(OpsAction::WriteGitLabContextFile {
                    unix_ts: TEST_UNIX_TS
                }),
                shutdown(),
            ]
        );
    }

    #[test]
    fn ops_cycle_stops_after_a_log_tail_without_writing_a_scrape_file_when_shutdown_is_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[false, false, true]);
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                shutdown(),
                observe(OpsQuery::CycleClock),
                act(OpsAction::WriteGitLabContextFile {
                    unix_ts: TEST_UNIX_TS
                }),
                shutdown(),
                observe(OpsQuery::RemoteLogTail {
                    source: observation("prod-1.example.com"),
                }),
                shutdown(),
            ]
        );
    }

    #[test]
    fn ops_cycle_scrapes_every_configured_source_in_configuration_order() {
        let mut port = FakeOpsPort::new(Vec::new());
        let hosts = ["prod-1.example.com", "prod-2.example.com"];
        let run = run_ops(&mut port, &hosts, true);

        assert!(run.result.is_ok());
        // Both sources are scraped before anything is written, and the two
        // windows are joined in configuration order.
        assert_eq!(run.trace, steps_up_to_analysis(&hosts));
        assert!(run.trace.contains(&act(OpsAction::WriteScrapeFile {
            unix_ts: TEST_UNIX_TS,
            window_log: expected_window(&hosts),
        })));
    }

    #[test]
    fn ops_cycle_skips_the_model_entirely_when_every_log_line_predates_the_window() {
        // Filtered out by the two-hour window, so no source contributes a
        // section and the joined window stays empty.
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_log_tail("2026-06-15 09:00:00 ERROR long-settled failure");
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                shutdown(),
                observe(OpsQuery::CycleClock),
                act(OpsAction::WriteGitLabContextFile {
                    unix_ts: TEST_UNIX_TS
                }),
                shutdown(),
                observe(OpsQuery::RemoteLogTail {
                    source: observation("prod-1.example.com"),
                }),
                shutdown(),
            ]
        );
        assert!(run.saved_history.is_none());
    }

    #[test]
    fn ops_cycle_saves_no_history_when_the_model_reports_no_issues() {
        let mut port = FakeOpsPort::new(Vec::new());
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        assert_eq!(run.trace, steps_up_to_analysis(&["prod-1.example.com"]));
        assert!(run.saved_history.is_none());
    }

    #[test]
    fn ops_cycle_appends_to_an_existing_history_and_still_saves_exactly_once() {
        let existing = OpsIssueHistory {
            entries: vec![created_entry(7, "Previously filed")],
        };
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(2))])
            .with_history(existing.clone());
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        // The history the model's context file is guaranteed to exist for is
        // the one loaded from disk, before this cycle appended anything.
        assert_eq!(run.ensured_history, Some(existing));
        assert_eq!(
            run.saved_history,
            Some(OpsIssueHistory {
                entries: vec![
                    created_entry(7, "Previously filed"),
                    created_entry(100, "Fix DB timeout"),
                ],
            })
        );
        assert_eq!(
            run.trace
                .iter()
                .filter(|step| matches!(step, OpsStep::Act(OpsAction::SaveHistory(_))))
                .count(),
            1
        );
    }

    #[test]
    fn ops_cycle_skips_a_proposal_whose_creation_failed_without_labeling_or_recording_it() {
        let mut port = FakeOpsPort::new(vec![
            proposal("Doomed", Some(1)),
            proposal("Fix memory leak", Some(2)),
        ])
        .failing_issue("Doomed");
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        assert!(run.result.is_ok());
        let mut expected = steps_up_to_analysis(&["prod-1.example.com"]);
        expected.extend([
            act(OpsAction::CreateIssue {
                title: "Doomed".to_string(),
                description: "Doomed — log evidence and remediation.".to_string(),
            }),
            // No label steps for the failed create; the cycle moves straight
            // to the next proposal.
            act(OpsAction::CreateIssue {
                title: "Fix memory leak".to_string(),
                description: "Fix memory leak — log evidence and remediation.".to_string(),
            }),
            act(OpsAction::AddPriorityLabel {
                issue_iid: 100,
                priority: 2,
            }),
            act(OpsAction::AddScopeLabel { issue_iid: 100 }),
            act(OpsAction::SaveHistory(OpsIssueHistory {
                entries: vec![created_entry(100, "Fix memory leak")],
            })),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn ops_cycle_records_a_created_issue_even_when_both_label_writes_fail() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .failing_action(&OpsAction::AddPriorityLabel {
                issue_iid: 0,
                priority: 0,
            })
            .failing_action(&OpsAction::AddScopeLabel { issue_iid: 0 });
        let run = run_ops(&mut port, &["prod-1.example.com"], true);

        // Labels are best effort: the issue exists, so it is recorded and the
        // history is still saved.
        assert!(run.result.is_ok());
        assert_eq!(
            run.saved_history,
            Some(OpsIssueHistory {
                entries: vec![created_entry(100, "Fix DB timeout")],
            })
        );
    }

    #[test]
    fn ops_cycle_omits_the_scope_label_when_no_scope_label_is_configured() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))]);
        let run = run_ops(&mut port, &["prod-1.example.com"], false);

        assert!(run.result.is_ok());
        assert!(
            !run.trace
                .iter()
                .any(|step| matches!(step, OpsStep::Act(OpsAction::AddScopeLabel { .. }))),
            "{:?}",
            run.trace
        );
    }

    #[test]
    fn ops_cycle_aborts_when_a_required_write_fails() {
        for failing in [
            OpsAction::WriteGitLabContextFile { unix_ts: 0 },
            OpsAction::WriteScrapeFile {
                unix_ts: 0,
                window_log: String::new(),
            },
            OpsAction::PruneScrapeFiles,
            OpsAction::EnsureHistoryFile(OpsIssueHistory::default()),
            OpsAction::WriteAnalysisFile {
                unix_ts: 0,
                window_log: String::new(),
            },
            OpsAction::InvokeAnalysisModel {
                prompt: String::new(),
            },
            OpsAction::SaveHistory(OpsIssueHistory::default()),
        ] {
            let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
                .failing_action(&failing);
            let run = run_ops(&mut port, &["prod-1.example.com"], true);
            assert!(
                run.result.is_err(),
                "expected {failing:?} to abort the cycle"
            );
        }
    }

    #[test]
    fn ops_cycle_aborts_when_a_required_read_fails() {
        for failing in [
            OpsQuery::RemoteLogTail {
                source: observation("prod-1.example.com"),
            },
            OpsQuery::IssueHistory,
            OpsQuery::HistoryFilePath,
        ] {
            let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
                .failing_query(failing.clone());
            let run = run_ops(&mut port, &["prod-1.example.com"], true);
            assert!(
                run.result.is_err(),
                "expected {failing:?} to abort the cycle"
            );
        }
    }

    #[test]
    fn ops_settings_require_ssh_and_log_path() {
        let err = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                ssh_user = ""
                ssh_host = "prod.example.com"
                log_path = "/var/log/app/app.log"
                "#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("ssh_user"));

        let settings = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                ssh_user = "deploy"
                ssh_host = "prod.example.com"
                log_path = "/var/log/app/app.log"
                "#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(settings.logs.len(), 1);
        assert_eq!(settings.logs[0].ssh_user, "deploy");
        assert_eq!(settings.logs[0].log_path, "/var/log/app/app.log");
    }

    #[test]
    fn ops_settings_accept_multiple_log_sources() {
        let settings = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                logs = [
                    { ssh_user = "deploy", ssh_host = "prod-1.example.com", log_path = "/var/log/app/app.log" },
                    { ssh_user = "deploy", ssh_host = "prod-2.example.com", log_path = "/var/log/app/worker.log" },
                ]
                "#,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(settings.logs.len(), 2);
        assert_eq!(settings.logs[0].ssh_host, "prod-1.example.com");
        assert_eq!(settings.logs[1].log_path, "/var/log/app/worker.log");
    }

    #[test]
    fn filter_log_to_time_window_keeps_recent_lines() {
        let now = Utc.with_ymd_and_hms(2026, 6, 15, 12, 0, 0).unwrap();
        let recent = "2026-06-15 11:30:00 ERROR something broke\nstack line";
        let old = "2026-06-15 09:00:00 ERROR old failure";
        let log = format!("{old}\n{recent}");
        let (filtered, parsed) = filter_log_to_time_window(&log, now);
        assert!(parsed);
        assert!(filtered.contains("something broke"));
        assert!(!filtered.contains("old failure"));
    }

    #[test]
    fn filter_log_to_time_window_parses_slash_date_format() {
        let now = Utc.with_ymd_and_hms(2026, 6, 16, 8, 50, 9).unwrap();
        let recent = "2026/06/16 06:50:09 ERROR something broke\nstack line";
        let old = "2026/06/16 04:50:09 ERROR old failure";
        let log = format!("{old}\n{recent}");
        let (filtered, parsed) = filter_log_to_time_window(&log, now);
        assert!(parsed);
        assert!(filtered.contains("something broke"));
        assert!(!filtered.contains("old failure"));
    }

    #[test]
    fn parse_line_timestamp_reads_slash_date_format() {
        let ts = parse_line_timestamp("2026/06/16 06:50:09 ERROR timeout").unwrap();
        assert_eq!(ts, Utc.with_ymd_and_hms(2026, 6, 16, 6, 50, 9).unwrap());
    }

    #[test]
    fn gitlab_context_markdown_sections_are_structured() {
        use crate::agents::gitlab::{Comment, Issue, MergeRequest};

        let issue = Issue {
            iid: 7,
            title: "Fix timeout".to_string(),
            description: "Handle DB timeouts".to_string(),
            labels: vec!["bug".to_string()],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };
        let issue_section = format!(
            "### Issue #{}: {}\nLabels: {}\n\nDescription:\n{}\n",
            issue.iid,
            issue.title,
            issue.labels.join(", "),
            issue.description
        );
        assert!(issue_section.contains("Issue #7"));
        assert!(issue_section.contains("Handle DB timeouts"));

        let mr = MergeRequest {
            iid: 3,
            title: "Fix timeout MR".to_string(),
            description: "Implementation".to_string(),
            source_branch: "issue-7".to_string(),
            target_branch: "main".to_string(),
            state: "opened".to_string(),
            sha: None,
            labels: Some(vec!["bug".to_string()]),
            has_conflicts: false,
        };
        let mr_section = format!(
            "### MR !{}: {}\nLabels: {}\nSource branch: {}\nTarget branch: {}\n\nDescription:\n{}\n",
            mr.iid,
            mr.title,
            mr.labels.as_ref().unwrap().join(", "),
            mr.source_branch,
            mr.target_branch,
            mr.description
        );
        assert!(mr_section.contains("MR !3"));
        assert!(
            Comment {
                id: 1,
                body: "looks good".to_string(),
                author: "alice".to_string(),
                discussion_id: "d1".to_string(),
                discussion_resolvable: false,
                location: None,
                location_details: None,
            }
            .format_for_prompt()
            .contains("looks good")
        );
    }

    #[test]
    fn ops_contract_passes_the_shared_conformance_suite() {
        conformance::assert_contract::<OpsOutput>();
    }

    #[test]
    fn ops_output_deserializes_issue() {
        let output = conformance::assert_accepts::<OpsOutput>(serde_json::json!({
            "issues": [
                {"title": "Fix DB timeout", "description": "details", "priority": 2, "log_line": "ERROR timeout"}
            ]
        }));
        let issues = normalize_ops_issues(output.issues);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].title, "Fix DB timeout");
        assert_eq!(issues[0].log_line, "ERROR timeout");
        assert_eq!(issues[0].priority, Some(2));
    }

    #[test]
    fn history_roundtrip_json() {
        let dir = std::env::temp_dir().join(format!("potlatch-ops-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");
        let history = OpsIssueHistory {
            entries: vec![OpsIssueHistoryEntry {
                gitlab_issue_iid: 42,
                title: "Issue".to_string(),
                log_line: "ERROR x".to_string(),
                created_at: "t0".to_string(),
            }],
        };
        save_history(&path, &history).unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded, history);

        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["entries"][0]["gitlab_issue_iid"], 42);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_history_reads_the_legacy_unversioned_format_and_migrates_it() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-history-legacy-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");
        let history = OpsIssueHistory {
            entries: vec![OpsIssueHistoryEntry {
                gitlab_issue_iid: 7,
                title: "Legacy issue".to_string(),
                log_line: "ERROR legacy".to_string(),
                created_at: "t0".to_string(),
            }],
        };
        // The bare pre-envelope payload written by older builds.
        fs::write(&path, serde_json::to_vec(&history).unwrap()).unwrap();

        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded, history);

        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_history_treats_a_missing_file_as_empty_default() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-history-missing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("history.json");

        assert_eq!(load_history(&path).unwrap(), OpsIssueHistory::default());
    }

    #[test]
    fn load_history_treats_an_empty_or_whitespace_only_file_as_empty_default() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-history-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        fs::write(&path, b"").unwrap();
        assert_eq!(load_history(&path).unwrap(), OpsIssueHistory::default());

        fs::write(&path, b"  \n\t").unwrap();
        assert_eq!(load_history(&path).unwrap(), OpsIssueHistory::default());
        // Never quarantined — this is expected, valid "no history yet" content.
        assert!(path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_history_is_strict_and_quarantines_malformed_json() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-history-corrupt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");
        fs::write(&path, b"not valid json").unwrap();

        let error = load_history(&path).unwrap_err();
        assert!(error.to_string().contains("Malformed state JSON"));

        // Strict: quarantined (bytes preserved), and the error propagates —
        // unlike worker/PMO-claim/QA, Ops history is not silently reset.
        assert!(!path.exists());
        let quarantined: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("quarantined"))
            .collect();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(fs::read(quarantined[0].path()).unwrap(), b"not valid json");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_history_is_strict_and_quarantines_an_unsupported_version() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-history-unsupported-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");
        fs::write(&path, br#"{"version":9,"state":{"entries":[]}}"#).unwrap();

        let error = load_history(&path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Unsupported state envelope version 9")
        );
        assert!(!path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // History persistence ordering: `ops_cycle` only calls `save_history`
    // after the create-issue loop, and only when at least one issue was
    // actually created (`created > 0`). `ensure_history_file` runs earlier,
    // before analysis, and is a no-op once a history file already exists.
    // The create/label GitLab calls themselves are real network calls
    // (`GitLabClient`) and are not characterized here — see note below.
    // -----------------------------------------------------------------

    #[test]
    fn ensure_history_file_creates_file_only_when_absent() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-ensure-history-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = AgentState {
            sessions_dir: &sessions_dir,
            agent_id: "ops-0",
        };
        let path = state.history_path();
        assert!(!path.exists());

        ensure_history_file(&state, &OpsIssueHistory::default()).unwrap();
        assert!(path.exists());

        // Write a sentinel entry directly, then call ensure again: it must
        // not overwrite an existing file (would silently drop history).
        let sentinel = OpsIssueHistory {
            entries: vec![OpsIssueHistoryEntry {
                gitlab_issue_iid: 99,
                title: "sentinel".to_string(),
                log_line: "x".to_string(),
                created_at: "t0".to_string(),
            }],
        };
        save_history(&path, &sentinel).unwrap();
        ensure_history_file(&state, &OpsIssueHistory::default()).unwrap();
        assert_eq!(load_history(&path).unwrap(), sentinel);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_old_scrape_files_keeps_only_the_newest_n() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-ops-prune-scrape-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = AgentState {
            sessions_dir: &sessions_dir,
            agent_id: "ops-0",
        };
        for ts in [100u64, 200, 300, 400, 500] {
            fs::write(state.scrape_path(ts), "log data").unwrap();
        }

        prune_old_scrape_files(&state, 2).unwrap();

        let mut remaining: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("-scrape-"))
            .collect();
        remaining.sort();
        assert_eq!(
            remaining,
            vec![
                "ops-0-scrape-400.log".to_string(),
                "ops-0-scrape-500.log".to_string()
            ]
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_old_scrape_files_is_a_noop_when_under_the_limit() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-prune-scrape-noop-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = AgentState {
            sessions_dir: &sessions_dir,
            agent_id: "ops-0",
        };
        fs::write(state.scrape_path(100), "log data").unwrap();

        prune_old_scrape_files(&state, 10).unwrap();

        assert!(state.scrape_path(100).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_remote_path_rejects_shell_metacharacters() {
        assert!(validate_remote_path("/var/log/app.log").is_ok());
        assert!(validate_remote_path("/var/log/app;rm -rf /").is_err());
    }

    #[test]
    fn build_analysis_prompt_references_context_files() {
        let prompt = build_analysis_prompt(
            "/tmp/project-sessions/ops-0-analysis-1.log",
            "/tmp/project-sessions/ops-0_issue_history.json",
            "/tmp/project-sessions/ops-0-gitlab-context-1.md",
        );
        assert!(prompt.contains("/tmp/project-sessions/ops-0-analysis-1.log"));
        assert!(prompt.contains("/tmp/project-sessions/ops-0_issue_history.json"));
        assert!(prompt.contains("/tmp/project-sessions/ops-0-gitlab-context-1.md"));
        assert!(!prompt.contains("ssh_host"));
        assert!(!prompt.contains("fingerprint"));
        assert!(prompt.contains("Log session file"));
        assert!(prompt.contains("GitLab context"));
        assert!(prompt.contains("inspect the current project codebase"));
        assert!(prompt.contains("already fixed in the current codebase"));
        assert!(prompt.contains("explicit code or doc comments"));
        assert!(prompt.contains("handling is intentional"));
        assert!(prompt.contains("Descriptions must cite log evidence and relevant code context"));
        assert!(!prompt.contains("ops_report"));
        assert!(!prompt.contains("output contract"));
        assert!(!prompt.contains("tool's `issues` field"));
        assert!(!prompt.contains("JSON array of objects"));
    }

    #[test]
    fn ops_output_deserializes_multiple_issues() {
        let output = conformance::assert_accepts::<OpsOutput>(serde_json::json!({
            "issues": [
                {
                    "title": "Fix DB timeout",
                    "description": "The DB connection pool is exhausted",
                    "priority": 1,
                    "log_line": "ERROR timeout connecting to DB"
                },
                {
                    "title": "Fix memory leak",
                    "description": "Goroutine leak in worker",
                    "priority": 2,
                    "log_line": "panic: goroutine leak detected"
                }
            ]
        }));
        let issues = normalize_ops_issues(output.issues);
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].title, "Fix DB timeout");
        assert_eq!(issues[0].priority, Some(1));
        assert_eq!(issues[1].title, "Fix memory leak");
        assert_eq!(issues[1].priority, Some(2));
    }

    #[test]
    fn ops_output_accepts_a_run_that_found_nothing() {
        let output = conformance::assert_accepts::<OpsOutput>(serde_json::json!({"issues": []}));
        assert!(output.issues.is_empty());
    }

    #[test]
    fn ops_output_requires_the_issues_array_even_when_empty() {
        assert_eq!(
            conformance::assert_rejects::<OpsOutput>(serde_json::json!({})),
            "$.issues: required property is missing"
        );
    }

    #[test]
    fn ops_output_reports_the_offending_issue_by_index() {
        assert_eq!(
            conformance::assert_rejects::<OpsOutput>(serde_json::json!({
                "issues": [
                    {"title": "t", "description": "d", "log_line": "ERR"},
                    {"title": "t", "description": "d"}
                ]
            })),
            "$.issues[1].log_line: required property is missing"
        );
    }

    #[test]
    fn ops_output_accepts_sample_line_as_a_legacy_log_line() {
        let output = conformance::assert_accepts::<OpsOutput>(serde_json::json!({
            "issues": [{"title": "t", "description": "d", "sample_line": "ERROR boom"}]
        }));
        assert_eq!(output.issues[0].log_line, "ERROR boom");
    }

    #[test]
    fn ops_output_drops_a_priority_outside_the_allowed_grades() {
        let output = conformance::assert_accepts::<OpsOutput>(serde_json::json!({
            "issues": [{"title": "t", "description": "d", "log_line": "ERR", "priority": 9}]
        }));
        assert_eq!(output.issues[0].priority, None);
    }

    #[test]
    fn ops_output_rejects_undeclared_issue_properties() {
        let error = conformance::assert_rejects::<OpsOutput>(serde_json::json!({
            "issues": [{"title": "t", "description": "d", "log_line": "ERR", "severity": "high"}]
        }));
        assert!(
            error.starts_with("$.issues[0].severity: unexpected property"),
            "{error}"
        );
    }

    #[test]
    fn normalize_ops_issues_skips_empty_fields() {
        let raw = vec![
            RawOpsIssue {
                title: "".into(),
                description: "d".into(),
                log_line: "ERR".into(),
                ..Default::default()
            },
            RawOpsIssue {
                title: "No log line".into(),
                description: "d".into(),
                log_line: "".into(),
                ..Default::default()
            },
            RawOpsIssue {
                title: "Fix X".into(),
                description: "d".into(),
                log_line: "ERR".into(),
                priority: Some(3),
            },
        ];
        let issues = normalize_ops_issues(raw);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].title, "Fix X");
    }
}
