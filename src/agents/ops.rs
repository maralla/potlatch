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

use super::state::StateStore;
use crate::agents::artifact::write_task_context_file;
use crate::agents::forge::{self, ForgeClient, scope_label_filter};
use crate::agents::ssh_util::{shell_single_quote, validate_remote_path, validate_ssh_identity};
use crate::agents::workspace::{AgentBootstrap, AgentWorkspace, repo_banner};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, compat, structured_output};
use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;

mod grafana;

pub(crate) const NAME: &str = "ops";
const MAX_INSTANCES: usize = 1;
const DEFAULT_LOG_WINDOW_INTERVAL: Duration = Duration::from_secs(2 * 60 * 60);
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(10 * 60);
const MAX_TAIL_LINES: u32 = 100_000;
const MAX_SCRAPE_FILES_KEPT: usize = 10;
const MIN_LOG_BYTES_FOR_ANALYSIS: usize = 20;

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

#[derive(Debug, Clone)]
struct OpsConfig {
    poll_interval: Duration,
    log_window_interval: Duration,
    log_sources: Vec<OpsLogSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OpsLogSource {
    Ssh(OpsSshLogSource),
    GrafanaElasticsearch(Box<grafana::GrafanaLogSource>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpsSshLogSource {
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OpsAgentSettings {
    #[serde(
        default = "default_ops_poll_interval",
        deserialize_with = "crate::core::config::duration::deserialize"
    )]
    poll_interval: Duration,
    poll_interval_secs: Option<u64>,
    #[serde(
        default = "default_log_window_interval",
        deserialize_with = "crate::core::config::duration::deserialize"
    )]
    log_window_interval: Duration,
    ssh_user: Option<String>,
    ssh_host: Option<String>,
    log_path: Option<String>,
    #[serde(default)]
    logs: Vec<OpsLogSourceSettings>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OpsLogSourceSettings {
    Ssh(OpsSshLogSourceSettings),
    Typed(TypedOpsLogSourceSettings),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TypedOpsLogSourceSettings {
    #[serde(rename = "grafana")]
    GrafanaElasticsearch(grafana::GrafanaLogSourceSettings),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpsSshLogSourceSettings {
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

fn default_ops_poll_interval() -> Duration {
    DEFAULT_POLL_INTERVAL
}

fn default_log_window_interval() -> Duration {
    DEFAULT_LOG_WINDOW_INTERVAL
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
            let source = OpsSshLogSourceSettings {
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
            settings.logs.push(OpsLogSourceSettings::Ssh(source));
        }
        ensure!(
            !settings.logs.is_empty(),
            "at least one log source is required for [agent.ops]"
        );
        ensure!(
            settings.poll_interval_secs.is_none(),
            "poll_interval_secs was replaced by poll_interval for [agent.ops]"
        );
        ensure!(
            !settings.poll_interval.is_zero(),
            "poll_interval must be greater than zero for [agent.ops]"
        );
        ensure!(
            !settings.log_window_interval.is_zero(),
            "log_window_interval must be greater than zero for [agent.ops]"
        );
        chrono::Duration::from_std(settings.log_window_interval)
            .context("log_window_interval is too large for [agent.ops]")?;
        for (idx, source) in settings.logs.iter().enumerate() {
            source.validate(idx)?;
        }
        Ok(settings)
    }
}

impl OpsLogSourceSettings {
    fn validate(&self, idx: usize) -> Result<()> {
        match self {
            Self::Ssh(source) => source.validate(idx),
            Self::Typed(TypedOpsLogSourceSettings::GrafanaElasticsearch(source)) => {
                source.validate(idx)
            }
        }
    }

    fn into_source(self) -> Result<OpsLogSource> {
        match self {
            Self::Ssh(source) => Ok(OpsLogSource::Ssh(source.into_source())),
            Self::Typed(TypedOpsLogSourceSettings::GrafanaElasticsearch(source)) => Ok(
                OpsLogSource::GrafanaElasticsearch(Box::new(source.into_source()?)),
            ),
        }
    }
}

impl OpsSshLogSourceSettings {
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

    fn into_source(self) -> OpsSshLogSource {
        OpsSshLogSource {
            ssh_user: self.ssh_user.trim().to_string(),
            ssh_host: self.ssh_host.trim().to_string(),
            log_path: self.log_path.trim().to_string(),
        }
    }
}

/// A borrowing view over the [`AgentWorkspace`] fields ops's path
/// helpers need. Built fresh from `&AgentWorkspace` at each use site
/// rather than stored — ops never owns a second copy of `sessions_dir`
/// or `agent_id`, and this is never stored alongside the runtime it
/// borrows from, so it can't become self-referential.
struct AgentState<'a> {
    sessions_dir: &'a str,
    agent_id: &'a str,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &AgentWorkspace) -> AgentState<'_> {
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
    runtime: AgentWorkspace,
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
        repo_banner(config, banner);
    }

    fn parse_settings(_config: &Config, section: &AgentSection) -> Result<Self::Settings> {
        OpsAgentSettings::from_raw(&section.raw)
    }

    fn validate_settings(
        config: &Config,
        _section: &AgentSection,
        _settings: &Self::Settings,
    ) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_repo_url()?;
        Ok(())
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec::polling(
            "log_scrape",
            self.config.poll_interval,
        )]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "log_scrape" => {
                let scope = scope_label_filter(&self.runtime.scope_label);
                let model = &self.runtime.model;
                let shutdown = Arc::clone(model.shutdown());
                let state = AgentState::from_runtime(&self.runtime);
                ops_cycle(
                    &state,
                    &self.config,
                    self.runtime.forge.as_ref(),
                    model,
                    Arc::clone(&shutdown),
                    scope,
                )
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = AgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        AgentState::from_runtime(&runtime).ensure_sessions_dir()?;
        let agent_settings = ctx.settings;
        let config = OpsConfig {
            poll_interval: agent_settings.poll_interval,
            log_window_interval: agent_settings.log_window_interval,
            log_sources: agent_settings
                .logs
                .into_iter()
                .map(OpsLogSourceSettings::into_source)
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(Self { runtime, config })
    }

    fn on_shutdown(&mut self) {}
}

// ---------------------------------------------------------------------------
// Ops role port
// ---------------------------------------------------------------------------

/// Immutable snapshot of one configured log source as the ops cycle sees it.
/// Role-local on purpose: the cycle receives only the fields one scrape needs
/// and cannot mutate the configured source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LogSourceObservation {
    source: OpsLogSource,
}

impl LogSourceObservation {
    fn from_source(source: &OpsLogSource) -> Self {
        Self {
            source: source.clone(),
        }
    }

    fn target(&self) -> String {
        match &self.source {
            OpsLogSource::Ssh(source) => {
                format!(
                    "{}@{}:{}",
                    source.ssh_user, source.ssh_host, source.log_path
                )
            }
            OpsLogSource::GrafanaElasticsearch(source) => source.target(),
        }
    }

    /// The banner that separates this source's lines from the next one's in
    /// the joined analysis input.
    fn section(&self, window_log: &str) -> String {
        format!("===== Log source: {} =====\n{window_log}", self.target())
    }

    fn prepare_window(
        &self,
        raw_log: &str,
        now: DateTime<Utc>,
        log_window_interval: Duration,
    ) -> (String, bool) {
        match &self.source {
            OpsLogSource::Ssh(_) => filter_log_to_time_window(raw_log, now, log_window_interval),
            OpsLogSource::GrafanaElasticsearch(_) => (raw_log.to_string(), true),
        }
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

/// The narrow typed surface the ops cycle needs. It is role-local rather than
/// a stand-in for the GitLab API, ssh, the filesystem, or the model backend
/// (see [`crate::agents::claim::ClaimPort`] for the same reasoning at claim
/// granularity).
trait OpsPort {
    fn shutdown_requested(&self) -> bool;
    fn cycle_clock(&self) -> CycleClock;
    fn fetch_logs(
        &self,
        source: &LogSourceObservation,
        clock: CycleClock,
        log_window_interval: Duration,
    ) -> Result<String>;
    fn write_gitlab_context_file(&mut self, unix_ts: u64) -> Result<String>;
    fn write_scrape_file(&mut self, unix_ts: u64, window_log: &str) -> Result<()>;
    fn prune_scrape_files(&mut self) -> Result<()>;
    fn issue_history(&self) -> Result<OpsIssueHistory>;
    fn ensure_history_file(&mut self, history: &OpsIssueHistory) -> Result<()>;
    fn write_analysis_file(&mut self, unix_ts: u64, window_log: &str) -> Result<String>;
    fn history_file_path(&self) -> Result<String>;
    fn invoke_analysis_model(&mut self, prompt: &str) -> Result<Vec<RawOpsIssue>>;
    fn create_issue(&mut self, title: &str, description: &str) -> Result<u64>;
    fn add_priority_label(&mut self, issue_iid: u64, priority: u8) -> Result<()>;
    fn add_scope_label(&mut self, issue_iid: u64) -> Result<()>;
    fn save_history(&mut self, history: &OpsIssueHistory) -> Result<()>;
}

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

fn run_ops_cycle(
    agent_id: &str,
    sources: &[OpsLogSource],
    log_window_interval: Duration,
    has_scope_label: bool,
    port: &mut dyn OpsPort,
) -> Result<()> {
    if port.shutdown_requested() {
        return Ok(());
    }
    let clock = port.cycle_clock();

    info!("{agent_id}: Fetching GitLab issues and merge requests for deduplication context");
    let gitlab_context_path = port.write_gitlab_context_file(clock.unix_ts)?;
    if port.shutdown_requested() {
        return Ok(());
    }

    let mut window_sections = Vec::new();
    for configured_source in sources {
        let source = LogSourceObservation::from_source(configured_source);
        info!(
            "{agent_id}: Fetching last {} of logs from {}",
            humantime::format_duration(log_window_interval),
            source.target()
        );
        let raw_tail = port.fetch_logs(&source, clock, log_window_interval)?;
        if port.shutdown_requested() {
            return Ok(());
        }

        let (window_log, parsed_timestamps) =
            source.prepare_window(&raw_tail, clock.now, log_window_interval);
        if !parsed_timestamps {
            warn!(
                "{agent_id}: Could not parse timestamps in SSH log tail for {}; using full tail for analysis",
                source.target()
            );
        }
        if !window_log.trim().is_empty() {
            window_sections.push(source.section(&window_log));
        }
    }

    let window_log = window_sections.join("\n\n");
    if !window_is_analyzable(&window_log) {
        info!(
            "{agent_id}: Log window too small to analyze ({} bytes)",
            window_log.trim().len()
        );
        return Ok(());
    }

    let work = crate::ui::WorkTimer::start();
    port.write_scrape_file(clock.unix_ts, &window_log)?;
    port.prune_scrape_files()?;
    let mut history = port.issue_history()?;
    port.ensure_history_file(&history)?;
    let analysis_path = port.write_analysis_file(clock.unix_ts, &window_log)?;
    let history_path = port.history_file_path()?;
    let prompt = build_analysis_prompt(&analysis_path, &history_path, &gitlab_context_path);
    let proposals = normalize_ops_issues(port.invoke_analysis_model(&prompt)?);
    if proposals.is_empty() {
        info!("{agent_id}: No new actionable errors found in log window");
        return Ok(());
    }

    let mut created = 0;
    for proposal in proposals {
        let issue_iid = match port.create_issue(&proposal.title, &proposal.description) {
            Ok(issue_iid) => issue_iid,
            Err(e) => {
                warn!(
                    "{agent_id}: Failed to create issue for {}: {e}",
                    proposal.title
                );
                continue;
            }
        };

        info!(
            "{agent_id}: Created GitLab issue #{}: {}",
            issue_iid, proposal.title
        );
        history
            .entries
            .push(history_entry(&proposal, issue_iid, &clock.now.to_rfc3339()));
        created += 1;

        if let Some(priority) = proposal.priority
            && let Err(e) = port.add_priority_label(issue_iid, priority)
        {
            warn!("{agent_id}: Failed to set priority label on #{issue_iid}: {e}");
        }
        if has_scope_label && let Err(e) = port.add_scope_label(issue_iid) {
            warn!("{agent_id}: Failed to add scope label on #{issue_iid}: {e}");
        }
    }

    if created > 0 {
        port.save_history(&history)?;
        info!("{agent_id}: Created {created} new GitLab issue(s) from log analysis");
    }
    info!(
        "{agent_id}: Log analysis done (analyzed for {})",
        crate::ui::format_work_duration(work.elapsed_seconds())
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Live ops port
// ---------------------------------------------------------------------------

/// The ops port backed by the real runtime: this file's only place where an
/// ops decision meets ssh, the filesystem, GitLab, or the model.
struct LiveOpsPort<'a> {
    state: &'a AgentState<'a>,
    forge: &'a dyn ForgeClient,
    model: &'a AgentModel,
    shutdown: &'a AtomicBool,
    scope_label: Option<&'a str>,
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

    fn fetch_logs(
        &self,
        source: &LogSourceObservation,
        clock: CycleClock,
        log_window_interval: Duration,
    ) -> Result<String> {
        match &source.source {
            OpsLogSource::Ssh(source) => {
                fetch_remote_log_tail(&source.ssh_user, &source.ssh_host, &source.log_path)
            }
            OpsLogSource::GrafanaElasticsearch(source) => grafana::fetch_logs(
                source,
                clock.now
                    - chrono::Duration::from_std(log_window_interval)
                        .expect("validated OPS log window interval"),
                clock.now,
            ),
        }
    }

    fn write_gitlab_context_file(&mut self, unix_ts: u64) -> Result<String> {
        write_gitlab_context_file(self.state, self.forge, unix_ts)
    }

    fn write_scrape_file(&mut self, unix_ts: u64, window_log: &str) -> Result<()> {
        let scrape_path = self.state.scrape_path(unix_ts);
        fs::write(&scrape_path, window_log)
            .with_context(|| format!("Failed to write scrape file {}", scrape_path.display()))
    }

    fn prune_scrape_files(&mut self) -> Result<()> {
        prune_old_scrape_files(self.state, MAX_SCRAPE_FILES_KEPT)
    }

    fn issue_history(&self) -> Result<OpsIssueHistory> {
        load_history(&self.state.history_path())
    }

    fn ensure_history_file(&mut self, history: &OpsIssueHistory) -> Result<()> {
        ensure_history_file(self.state, history)
    }

    fn write_analysis_file(&mut self, unix_ts: u64, window_log: &str) -> Result<String> {
        write_task_context_file(
            self.state.sessions_dir,
            &format!("{}-analysis-{unix_ts}.log", self.state.agent_id),
            window_log,
        )
    }

    fn history_file_path(&self) -> Result<String> {
        absolute_path(&self.state.history_path())
    }

    fn invoke_analysis_model(&mut self, prompt: &str) -> Result<Vec<RawOpsIssue>> {
        Ok(self
            .model
            .complete_typed::<OpsOutput>(
                prompt,
                &InvokeOptions {
                    activity_label: Some(format!("{} analyzing logs", self.state.agent_id)),
                    ..InvokeOptions::default()
                },
            )?
            .output
            .issues)
    }

    fn create_issue(&mut self, title: &str, description: &str) -> Result<u64> {
        self.forge.create_issue(title, description)
    }

    fn add_priority_label(&mut self, issue_iid: u64, priority: u8) -> Result<()> {
        self.forge
            .add_issue_label(issue_iid, &forge::priority_label(priority))
    }

    fn add_scope_label(&mut self, issue_iid: u64) -> Result<()> {
        match self.scope_label {
            Some(label) => self.forge.add_issue_label(issue_iid, label),
            None => Ok(()),
        }
    }

    fn save_history(&mut self, history: &OpsIssueHistory) -> Result<()> {
        save_history(&self.state.history_path(), history)
    }
}

fn ops_cycle(
    state: &AgentState,
    config: &OpsConfig,
    forge: &dyn ForgeClient,
    model: &AgentModel,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut port = LiveOpsPort {
        state,
        forge,
        model,
        shutdown: shutdown.as_ref(),
        scope_label,
    };
    run_ops_cycle(
        state.agent_id,
        &config.log_sources,
        config.log_window_interval,
        scope_label.is_some(),
        &mut port,
    )
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
    forge: &dyn ForgeClient,
    unix_ts: u64,
) -> Result<String> {
    let content = build_gitlab_context(forge)?;
    write_task_context_file(
        state.sessions_dir,
        &format!("{}-gitlab-context-{unix_ts}.md", state.agent_id),
        &content,
    )
}

fn build_gitlab_context(forge: &dyn ForgeClient) -> Result<String> {
    let mut out = String::from("# Current GitLab issues and merge requests\n\n");

    let issues = forge.list_issues()?;
    out.push_str("## Open issues\n\n");
    if issues.is_empty() {
        out.push_str("(none)\n\n");
    } else {
        for issue in &issues {
            append_issue_context(&mut out, forge, issue)?;
        }
    }

    let mrs = forge.list_merge_requests()?;
    out.push_str("## Open merge requests\n\n");
    if mrs.is_empty() {
        out.push_str("(none)\n");
    } else {
        for mr in &mrs {
            append_mr_context(&mut out, forge, mr)?;
        }
    }

    Ok(out)
}

fn append_issue_context(
    out: &mut String,
    forge: &dyn ForgeClient,
    issue: &forge::Issue,
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

    match forge.get_issue_comments(issue.iid) {
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
    forge: &dyn ForgeClient,
    mr: &forge::MergeRequest,
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

    match forge.get_mr_comments(mr.iid) {
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
    let store = StateStore::new(path);
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
    StateStore::new(path).save(history)
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

fn filter_log_to_time_window(
    log: &str,
    now: DateTime<Utc>,
    log_window_interval: Duration,
) -> (String, bool) {
    let cutoff = now
        - chrono::Duration::from_std(log_window_interval)
            .expect("validated OPS log window interval");
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
    // Direct ops cycle driven against a recording fake port. Every observation
    // and mutation lands in one ordered trace, so a full scrape/analyze/file
    // flow can be replayed — and its exact mutation order asserted — without
    // ssh, GitLab, the filesystem, or a model.
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
        OpsLogSource::Ssh(OpsSshLogSource {
            ssh_user: "deploy".to_string(),
            ssh_host: host.to_string(),
            log_path: "/var/log/app/app.log".to_string(),
        })
    }

    fn grafana_source() -> OpsLogSource {
        OpsLogSource::GrafanaElasticsearch(Box::new(
            grafana::GrafanaLogSourceSettings {
                url: "https://grafana.example.com".to_string(),
                datasource_uid: "elastic-1".to_string(),
                index: "app-logs".to_string(),
                org_id: 1,
                username: "ops".to_string(),
                password: "secret".to_string(),
                filter: "level:ERROR".to_string(),
            }
            .into_source()
            .unwrap(),
        ))
    }

    fn observation(host: &str) -> LogSourceObservation {
        LogSourceObservation::from_source(&log_source(host))
    }

    fn ssh_settings(source: &OpsLogSourceSettings) -> &OpsSshLogSourceSettings {
        match source {
            OpsLogSourceSettings::Ssh(source) => source,
            OpsLogSourceSettings::Typed(_) => panic!("expected SSH settings"),
        }
    }

    /// The joined analysis input the cycle assembles for `hosts`, in
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

    /// A recording ops port. Operation order is kept as plain strings while
    /// payloads are captured separately for focused assertions.
    struct FakeOpsPort {
        trace: RefCell<Vec<String>>,
        shutdown_answers: RefCell<VecDeque<bool>>,
        clock: CycleClock,
        log_tail: String,
        history: OpsIssueHistory,
        analysis: Vec<RawOpsIssue>,
        next_iid: Cell<u64>,
        fetched_log_sources: RefCell<Vec<LogSourceObservation>>,
        context_timestamps: RefCell<Vec<u64>>,
        scrape_files: RefCell<Vec<(u64, String)>>,
        analysis_files: RefCell<Vec<(u64, String)>>,
        analysis_prompts: RefCell<Vec<String>>,
        created_issues: RefCell<Vec<(String, String)>>,
        priority_labels: RefCell<Vec<(u64, u8)>>,
        scope_labels: RefCell<Vec<u64>>,
        saved_histories: RefCell<Vec<OpsIssueHistory>>,
        ensured_histories: RefCell<Vec<OpsIssueHistory>>,
        failing_operations: Vec<String>,
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
                fetched_log_sources: RefCell::new(Vec::new()),
                context_timestamps: RefCell::new(Vec::new()),
                scrape_files: RefCell::new(Vec::new()),
                analysis_files: RefCell::new(Vec::new()),
                analysis_prompts: RefCell::new(Vec::new()),
                created_issues: RefCell::new(Vec::new()),
                priority_labels: RefCell::new(Vec::new()),
                scope_labels: RefCell::new(Vec::new()),
                saved_histories: RefCell::new(Vec::new()),
                ensured_histories: RefCell::new(Vec::new()),
                failing_operations: Vec::new(),
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

        fn failing_operation(mut self, operation: &str) -> Self {
            self.failing_operations.push(operation.to_string());
            self
        }

        fn failing_issue(mut self, title: &str) -> Self {
            self.failing_issue_titles.push(title.to_string());
            self
        }

        fn record(&self, operation: &str) {
            self.trace.borrow_mut().push(operation.to_string());
        }

        fn perform(&self, operation: &str) -> Result<()> {
            self.record(operation);
            if self
                .failing_operations
                .iter()
                .any(|failing| failing == operation)
            {
                anyhow::bail!("injected failure performing {operation}");
            }
            Ok(())
        }
    }

    impl OpsPort for FakeOpsPort {
        fn shutdown_requested(&self) -> bool {
            self.record("shutdown_requested");
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }

        fn cycle_clock(&self) -> CycleClock {
            self.record("cycle_clock");
            self.clock
        }

        fn fetch_logs(
            &self,
            source: &LogSourceObservation,
            _clock: CycleClock,
            _log_window_interval: Duration,
        ) -> Result<String> {
            self.perform("fetch_logs")?;
            self.fetched_log_sources.borrow_mut().push(source.clone());
            Ok(self.log_tail.clone())
        }

        fn write_gitlab_context_file(&mut self, unix_ts: u64) -> Result<String> {
            self.perform("write_gitlab_context_file")?;
            self.context_timestamps.borrow_mut().push(unix_ts);
            Ok(CONTEXT_PATH.to_string())
        }

        fn write_scrape_file(&mut self, unix_ts: u64, window_log: &str) -> Result<()> {
            self.perform("write_scrape_file")?;
            self.scrape_files
                .borrow_mut()
                .push((unix_ts, window_log.to_string()));
            Ok(())
        }

        fn prune_scrape_files(&mut self) -> Result<()> {
            self.perform("prune_scrape_files")
        }

        fn issue_history(&self) -> Result<OpsIssueHistory> {
            self.perform("issue_history")?;
            Ok(self.history.clone())
        }

        fn ensure_history_file(&mut self, history: &OpsIssueHistory) -> Result<()> {
            self.perform("ensure_history_file")?;
            self.ensured_histories.borrow_mut().push(history.clone());
            Ok(())
        }

        fn write_analysis_file(&mut self, unix_ts: u64, window_log: &str) -> Result<String> {
            self.perform("write_analysis_file")?;
            self.analysis_files
                .borrow_mut()
                .push((unix_ts, window_log.to_string()));
            Ok(ANALYSIS_PATH.to_string())
        }

        fn history_file_path(&self) -> Result<String> {
            self.perform("history_file_path")?;
            Ok(HISTORY_PATH.to_string())
        }

        fn invoke_analysis_model(&mut self, prompt: &str) -> Result<Vec<RawOpsIssue>> {
            self.perform("invoke_analysis_model")?;
            self.analysis_prompts.borrow_mut().push(prompt.to_string());
            Ok(self.analysis.clone())
        }

        fn create_issue(&mut self, title: &str, description: &str) -> Result<u64> {
            self.perform("create_issue")?;
            self.created_issues
                .borrow_mut()
                .push((title.to_string(), description.to_string()));
            if self.failing_issue_titles.iter().any(|t| t == title) {
                anyhow::bail!("injected failure creating issue {title:?}");
            }
            let iid = self.next_iid.get();
            self.next_iid.set(iid + 1);
            Ok(iid)
        }

        fn add_priority_label(&mut self, issue_iid: u64, priority: u8) -> Result<()> {
            self.perform("add_priority_label")?;
            self.priority_labels
                .borrow_mut()
                .push((issue_iid, priority));
            Ok(())
        }

        fn add_scope_label(&mut self, issue_iid: u64) -> Result<()> {
            self.perform("add_scope_label")?;
            self.scope_labels.borrow_mut().push(issue_iid);
            Ok(())
        }

        fn save_history(&mut self, history: &OpsIssueHistory) -> Result<()> {
            self.perform("save_history")?;
            self.saved_histories.borrow_mut().push(history.clone());
            Ok(())
        }
    }

    fn run_ops(port: &mut FakeOpsPort, hosts: &[&str], has_scope_label: bool) -> Result<()> {
        let sources: Vec<OpsLogSource> = hosts.iter().copied().map(log_source).collect();
        run_ops_cycle(
            TEST_AGENT,
            &sources,
            DEFAULT_LOG_WINDOW_INTERVAL,
            has_scope_label,
            port,
        )
    }

    /// The operations every cycle performs from the first shutdown check through
    /// the model invocation, for a single log source whose window is large
    /// enough to analyze.
    fn operations_up_to_analysis(source_count: usize) -> Vec<String> {
        let mut operations = vec![
            "shutdown_requested",
            "cycle_clock",
            "write_gitlab_context_file",
            "shutdown_requested",
        ];
        for _ in 0..source_count {
            operations.push("fetch_logs");
            operations.push("shutdown_requested");
        }
        operations.extend([
            "write_scrape_file",
            "prune_scrape_files",
            "issue_history",
            "ensure_history_file",
            "write_analysis_file",
            "history_file_path",
            "invoke_analysis_model",
        ]);
        operations.into_iter().map(str::to_string).collect()
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
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());

        let mut expected = operations_up_to_analysis(1);
        expected.extend(
            [
                "create_issue",
                "add_priority_label",
                "add_scope_label",
                "create_issue",
                "add_scope_label",
                "save_history",
            ]
            .map(str::to_string),
        );
        assert_eq!(*port.trace.borrow(), expected);
        assert_eq!(
            *port.created_issues.borrow(),
            vec![
                (
                    "Fix DB timeout".to_string(),
                    "Fix DB timeout — log evidence and remediation.".to_string(),
                ),
                (
                    "Fix memory leak".to_string(),
                    "Fix memory leak — log evidence and remediation.".to_string(),
                ),
            ]
        );
        assert_eq!(*port.priority_labels.borrow(), vec![(100, 1)]);
        assert_eq!(*port.scope_labels.borrow(), vec![100, 101]);
        assert_eq!(
            *port.saved_histories.borrow(),
            vec![OpsIssueHistory {
                entries: vec![
                    created_entry(100, "Fix DB timeout"),
                    created_entry(101, "Fix memory leak"),
                ],
            }]
        );
    }

    #[test]
    fn ops_cycle_stops_before_any_side_effect_when_shutdown_is_already_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[true]);
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        assert_eq!(*port.trace.borrow(), vec!["shutdown_requested"]);
    }

    #[test]
    fn ops_cycle_stops_after_the_context_file_when_shutdown_is_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[false, true]);
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        assert_eq!(
            *port.trace.borrow(),
            vec![
                "shutdown_requested",
                "cycle_clock",
                "write_gitlab_context_file",
                "shutdown_requested",
            ]
        );
        assert_eq!(*port.context_timestamps.borrow(), vec![TEST_UNIX_TS]);
    }

    #[test]
    fn ops_cycle_stops_after_a_log_tail_without_writing_a_scrape_file_when_shutdown_is_requested() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_shutdown_answers(&[false, false, true]);
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        assert_eq!(
            *port.trace.borrow(),
            vec![
                "shutdown_requested",
                "cycle_clock",
                "write_gitlab_context_file",
                "shutdown_requested",
                "fetch_logs",
                "shutdown_requested",
            ]
        );
        assert_eq!(
            *port.fetched_log_sources.borrow(),
            vec![observation("prod-1.example.com")]
        );
        assert!(port.scrape_files.borrow().is_empty());
    }

    #[test]
    fn ops_cycle_scrapes_every_configured_source_in_configuration_order() {
        let mut port = FakeOpsPort::new(Vec::new());
        let hosts = ["prod-1.example.com", "prod-2.example.com"];
        assert!(run_ops(&mut port, &hosts, true).is_ok());
        // Both sources are scraped before anything is written, and the two
        // windows are joined in configuration order.
        assert_eq!(*port.trace.borrow(), operations_up_to_analysis(hosts.len()));
        assert_eq!(
            *port.fetched_log_sources.borrow(),
            vec![
                observation("prod-1.example.com"),
                observation("prod-2.example.com"),
            ]
        );
        assert_eq!(
            *port.scrape_files.borrow(),
            vec![(TEST_UNIX_TS, expected_window(&hosts))]
        );
        assert_eq!(
            *port.analysis_files.borrow(),
            vec![(TEST_UNIX_TS, expected_window(&hosts))]
        );
        assert_eq!(
            *port.analysis_prompts.borrow(),
            vec![build_analysis_prompt(
                ANALYSIS_PATH,
                HISTORY_PATH,
                CONTEXT_PATH
            )]
        );
    }

    #[test]
    fn ops_cycle_keeps_grafana_results_already_bounded_by_the_server() {
        let old_log = "2026-06-15 09:00:00 ERROR old SSH line";
        let mut port = FakeOpsPort::new(Vec::new()).with_log_tail(old_log);
        let sources = vec![log_source("prod.example.com"), grafana_source()];

        run_ops_cycle(
            TEST_AGENT,
            &sources,
            DEFAULT_LOG_WINDOW_INTERVAL,
            true,
            &mut port,
        )
        .unwrap();

        let observations: Vec<_> = sources
            .iter()
            .map(LogSourceObservation::from_source)
            .collect();
        assert_eq!(*port.fetched_log_sources.borrow(), observations);
        let grafana_window = observations[1].section(old_log);
        assert_eq!(
            *port.scrape_files.borrow(),
            vec![(TEST_UNIX_TS, grafana_window)]
        );
    }

    #[test]
    fn ops_cycle_skips_the_model_entirely_when_every_log_line_predates_the_window() {
        // Filtered out by the two-hour window, so no source contributes a
        // section and the joined window stays empty.
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .with_log_tail("2026-06-15 09:00:00 ERROR long-settled failure");
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        assert_eq!(
            *port.trace.borrow(),
            vec![
                "shutdown_requested",
                "cycle_clock",
                "write_gitlab_context_file",
                "shutdown_requested",
                "fetch_logs",
                "shutdown_requested",
            ]
        );
        assert!(port.saved_histories.borrow().is_empty());
    }

    #[test]
    fn ops_cycle_saves_no_history_when_the_model_reports_no_issues() {
        let mut port = FakeOpsPort::new(Vec::new());
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        assert_eq!(*port.trace.borrow(), operations_up_to_analysis(1));
        assert!(port.saved_histories.borrow().is_empty());
    }

    #[test]
    fn ops_cycle_appends_to_an_existing_history_and_still_saves_exactly_once() {
        let existing = OpsIssueHistory {
            entries: vec![created_entry(7, "Previously filed")],
        };
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(2))])
            .with_history(existing.clone());
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        // The history the model's context file is guaranteed to exist for is
        // the one loaded from disk, before this cycle appended anything.
        assert_eq!(*port.ensured_histories.borrow(), vec![existing]);
        assert_eq!(
            *port.saved_histories.borrow(),
            vec![OpsIssueHistory {
                entries: vec![
                    created_entry(7, "Previously filed"),
                    created_entry(100, "Fix DB timeout"),
                ],
            }]
        );
        assert_eq!(
            port.trace
                .borrow()
                .iter()
                .filter(|operation| operation.as_str() == "save_history")
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
        assert!(run_ops(&mut port, &["prod-1.example.com"], true).is_ok());
        let mut expected = operations_up_to_analysis(1);
        expected.extend(
            [
                "create_issue",
                "create_issue",
                "add_priority_label",
                "add_scope_label",
                "save_history",
            ]
            .map(str::to_string),
        );
        assert_eq!(*port.trace.borrow(), expected);
        assert_eq!(
            *port.created_issues.borrow(),
            vec![
                (
                    "Doomed".to_string(),
                    "Doomed — log evidence and remediation.".to_string(),
                ),
                (
                    "Fix memory leak".to_string(),
                    "Fix memory leak — log evidence and remediation.".to_string(),
                ),
            ]
        );
        assert_eq!(*port.priority_labels.borrow(), vec![(100, 2)]);
        assert_eq!(*port.scope_labels.borrow(), vec![100]);
        assert_eq!(
            *port.saved_histories.borrow(),
            vec![OpsIssueHistory {
                entries: vec![created_entry(100, "Fix memory leak")],
            }]
        );
    }

    #[test]
    fn ops_cycle_records_a_created_issue_even_when_both_label_writes_fail() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
            .failing_operation("add_priority_label")
            .failing_operation("add_scope_label");
        let result = run_ops(&mut port, &["prod-1.example.com"], true);

        // Labels are best effort: the issue exists, so it is recorded and the
        // history is still saved.
        assert!(result.is_ok());
        assert_eq!(
            *port.saved_histories.borrow(),
            vec![OpsIssueHistory {
                entries: vec![created_entry(100, "Fix DB timeout")],
            }]
        );
    }

    #[test]
    fn ops_cycle_omits_the_scope_label_when_no_scope_label_is_configured() {
        let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))]);
        assert!(run_ops(&mut port, &["prod-1.example.com"], false).is_ok());
        assert!(
            !port
                .trace
                .borrow()
                .iter()
                .any(|operation| operation == "add_scope_label"),
            "{:?}",
            port.trace.borrow()
        );
        assert!(port.scope_labels.borrow().is_empty());
    }

    #[test]
    fn ops_cycle_aborts_when_a_required_write_fails() {
        for failing in [
            "write_gitlab_context_file",
            "write_scrape_file",
            "prune_scrape_files",
            "ensure_history_file",
            "write_analysis_file",
            "invoke_analysis_model",
            "save_history",
        ] {
            let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
                .failing_operation(failing);
            let result = run_ops(&mut port, &["prod-1.example.com"], true);
            assert!(result.is_err(), "expected {failing} to abort the cycle");
        }
    }

    #[test]
    fn ops_cycle_aborts_when_a_required_read_fails() {
        for failing in ["fetch_logs", "issue_history", "history_file_path"] {
            let mut port = FakeOpsPort::new(vec![proposal("Fix DB timeout", Some(1))])
                .failing_operation(failing);
            let result = run_ops(&mut port, &["prod-1.example.com"], true);
            assert!(result.is_err(), "expected {failing} to abort the cycle");
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
        assert_eq!(settings.poll_interval, DEFAULT_POLL_INTERVAL);
        assert_eq!(settings.log_window_interval, DEFAULT_LOG_WINDOW_INTERVAL);
        assert_eq!(ssh_settings(&settings.logs[0]).ssh_user, "deploy");
        assert_eq!(
            ssh_settings(&settings.logs[0]).log_path,
            "/var/log/app/app.log"
        );
    }

    #[test]
    fn ops_settings_accept_multiple_log_sources() {
        let settings = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                poll_interval = "45s"
                log_window_interval = "30m"
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
        assert_eq!(settings.poll_interval, Duration::from_secs(45));
        assert_eq!(settings.log_window_interval, Duration::from_secs(30 * 60));
        assert_eq!(
            ssh_settings(&settings.logs[0]).ssh_host,
            "prod-1.example.com"
        );
        assert_eq!(
            ssh_settings(&settings.logs[1]).log_path,
            "/var/log/app/worker.log"
        );
    }

    #[test]
    fn ops_settings_reject_zero_log_window() {
        let error = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                log_window_interval = "0s"
                logs = [
                    { ssh_user = "deploy", ssh_host = "prod.example.com", log_path = "/var/log/app.log" },
                ]
                "#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("duration must be greater than zero"));
    }

    #[test]
    fn ops_settings_reject_zero_or_deprecated_poll_interval() {
        for (field, expected) in [
            (
                "poll_interval = \"0s\"",
                "duration must be greater than zero",
            ),
            ("poll_interval_secs = 60", "poll_interval_secs was replaced"),
        ] {
            let raw = format!(
                r#"
                {field}
                logs = [
                    {{ ssh_user = "deploy", ssh_host = "prod.example.com", log_path = "/var/log/app.log" }},
                ]
                "#
            );
            let error = OpsAgentSettings::from_raw(&toml::from_str(&raw).unwrap()).unwrap_err();
            assert!(format!("{error:#}").contains(expected));
        }
    }

    #[test]
    fn ops_settings_accept_mixed_ssh_and_grafana_sources() {
        let settings = OpsAgentSettings::from_raw(
            &toml::from_str(
                r#"
                logs = [
                    { ssh_user = "deploy", ssh_host = "prod.example.com", log_path = "/var/log/app.log" },
                    { type = "grafana", url = "https://grafana.example.com", datasource_uid = "elastic-1", index = "app-logs", username = "ops", password = "secret", filter = "level:ERROR" },
                ]
                "#,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(settings.logs.len(), 2);
        let OpsLogSourceSettings::Typed(TypedOpsLogSourceSettings::GrafanaElasticsearch(grafana)) =
            &settings.logs[1]
        else {
            panic!("expected Grafana Elasticsearch settings");
        };
        assert_eq!(grafana.org_id, 1);
        assert_eq!(grafana.filter, "level:ERROR");
        let debug = format!("{settings:?}");
        assert!(!debug.contains("\"secret\""));
    }

    #[test]
    fn filter_log_to_time_window_keeps_recent_lines() {
        let now = Utc.with_ymd_and_hms(2026, 6, 15, 12, 0, 0).unwrap();
        let recent = "2026-06-15 11:30:00 ERROR something broke\nstack line";
        let old = "2026-06-15 09:00:00 ERROR old failure";
        let log = format!("{old}\n{recent}");
        let (filtered, parsed) = filter_log_to_time_window(&log, now, DEFAULT_LOG_WINDOW_INTERVAL);
        assert!(parsed);
        assert!(filtered.contains("something broke"));
        assert!(!filtered.contains("old failure"));
    }

    #[test]
    fn filter_log_to_time_window_uses_configured_interval() {
        let now = Utc.with_ymd_and_hms(2026, 6, 15, 12, 0, 0).unwrap();
        let log = "2026-06-15 07:00:00 ERROR five hours old";
        let (short_window, _) =
            filter_log_to_time_window(log, now, Duration::from_secs(2 * 60 * 60));
        let (long_window, _) =
            filter_log_to_time_window(log, now, Duration::from_secs(6 * 60 * 60));
        assert!(short_window.is_empty());
        assert!(long_window.contains("five hours old"));
    }

    #[test]
    fn filter_log_to_time_window_parses_slash_date_format() {
        let now = Utc.with_ymd_and_hms(2026, 6, 16, 8, 50, 9).unwrap();
        let recent = "2026/06/16 06:50:09 ERROR something broke\nstack line";
        let old = "2026/06/16 04:50:09 ERROR old failure";
        let log = format!("{old}\n{recent}");
        let (filtered, parsed) = filter_log_to_time_window(&log, now, DEFAULT_LOG_WINDOW_INTERVAL);
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
        use crate::agents::forge::{Comment, Issue, MergeRequest};

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
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-history-legacy-{}",
            std::process::id()
        ));
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
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-history-missing-{}",
            std::process::id()
        ));
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
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-history-corrupt-{}",
            std::process::id()
        ));
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
    // The create/label forge calls themselves are real network calls
    // and are not characterized here — see note below.
    // -----------------------------------------------------------------

    #[test]
    fn ensure_history_file_creates_file_only_when_absent() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-ops-ensure-history-{}",
            std::process::id()
        ));
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
    fn build_analysis_prompt_references_context_files_without_secrets() {
        let prompt = build_analysis_prompt(
            "/tmp/project-sessions/ops-0-analysis-1.log",
            "/tmp/project-sessions/ops-0_issue_history.json",
            "/tmp/project-sessions/ops-0-gitlab-context-1.md",
        );
        // Every context path the model must act on is embedded...
        assert!(prompt.contains("/tmp/project-sessions/ops-0-analysis-1.log"));
        assert!(prompt.contains("/tmp/project-sessions/ops-0_issue_history.json"));
        assert!(prompt.contains("/tmp/project-sessions/ops-0-gitlab-context-1.md"));
        // ...and no SSH connection details leak into the prompt.
        assert!(!prompt.contains("ssh_host"));
        assert!(!prompt.contains("fingerprint"));
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
