use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

use crate::agents::gitlab::{self, GitLabClient};
use crate::agents::settings;
use crate::agents::workspace::{
    ensure_agent_repo, extract_project_name, require_gitlab_repo, sessions_dir, work_dir,
};
use crate::agents::write_task_context_file;
use crate::core::agent::{AgentHandoff, InvokeOptions};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};

pub(crate) const NAME: &str = "ops";
const MAX_INSTANCES: usize = 1;

const OPS_ISSUES_BEGIN: &str = "OPS_ISSUES_BEGIN";
const OPS_ISSUES_END: &str = "OPS_ISSUES_END";
const LOG_WINDOW_HOURS: i64 = 2;
const MAX_TAIL_LINES: u32 = 100_000;
const MAX_SCRAPE_FILES_KEPT: usize = 10;
const MIN_LOG_BYTES_FOR_ANALYSIS: usize = 20;

#[derive(Debug, Clone)]
struct OpsConfig {
    poll_interval_secs: u64,
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OpsAgentSettings {
    #[serde(default = "default_ops_poll_interval")]
    poll_interval_secs: u64,
    ssh_user: String,
    ssh_host: String,
    log_path: String,
}

fn default_ops_poll_interval() -> u64 {
    600
}

impl OpsAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        let settings: Self = raw
            .clone()
            .try_into()
            .context("ops agent settings from config")?;
        ensure!(
            !settings.ssh_user.trim().is_empty(),
            "ssh_user is required for [agent.ops]"
        );
        ensure!(
            !settings.ssh_host.trim().is_empty(),
            "ssh_host is required for [agent.ops]"
        );
        ensure!(
            !settings.log_path.trim().is_empty(),
            "log_path is required for [agent.ops]"
        );
        Ok(settings)
    }
}

struct AgentState {
    sessions_dir: String,
    agent_id: String,
}

impl AgentState {
    fn ensure_sessions_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.sessions_dir).context("Failed to create sessions directory")?;
        Ok(())
    }

    fn history_path(&self) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_issue_history.json", self.agent_id))
    }

    fn scrape_path(&self, unix_ts: u64) -> PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}-scrape-{unix_ts}.log", self.agent_id))
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
    state: AgentState,
    gitlab: GitLabClient,
    model: AgentModel,
    config: OpsConfig,
    scope_label: String,
}

impl CoreAgent for OpsAgent {
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        NAME
    }

    fn model(&self) -> &AgentModel {
        &self.model
    }

    fn banner(_config: &Config, banner: &mut Banner) {
        if let Some(repo) = settings::settings().gitlab_repo() {
            banner.set_once("repo", repo);
        }
    }

    fn validate_config(section: &crate::core::config::AgentSection) -> Result<()> {
        validate_instance_count(section.core.instances)
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: "log_scrape",
            interval: Duration::from_secs(self.config.poll_interval_secs),
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 5000,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "log_scrape" => {
                let scope = crate::agents::scope_label_filter(&self.scope_label);
                let model = &self.model;
                let shutdown = Arc::clone(model.shutdown());
                if let Err(e) = ops_cycle(
                    &self.state,
                    &self.config,
                    &self.gitlab,
                    model,
                    Arc::clone(&shutdown),
                    scope,
                ) && !shutdown.load(Ordering::SeqCst)
                {
                    error!("{}: Cycle error: {}", self.state.agent_id, e);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn from_spawn(ctx: crate::core::workflow::AgentSpawnContext) -> Result<Self> {
        let gitlab_repo = require_gitlab_repo()?;
        let section = ctx
            .workflow
            .config
            .agent(NAME)
            .context("[agent.ops] section required")?;
        validate_instance_count(section.core.instances)?;
        ensure!(
            ctx.instance_id < MAX_INSTANCES,
            "[agent.ops] invalid instance id {} (only instance 0 is supported)",
            ctx.instance_id
        );
        let agent_settings = OpsAgentSettings::from_raw(&section.raw)?;
        let project_name = extract_project_name(&gitlab_repo)?;
        let agent_id = format!("ops-{}", ctx.instance_id);
        ensure_agent_repo(
            &ctx.workflow.base_dir,
            &gitlab_repo,
            &project_name,
            &agent_id,
        )?;
        let working_dir = work_dir(&ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&ctx.workflow.base_dir, &project_name);
        let state = AgentState {
            sessions_dir,
            agent_id: agent_id.clone(),
        };
        state.ensure_sessions_dir()?;
        let config = OpsConfig {
            poll_interval_secs: agent_settings.poll_interval_secs,
            ssh_user: agent_settings.ssh_user.trim().to_string(),
            ssh_host: agent_settings.ssh_host.trim().to_string(),
            log_path: agent_settings.log_path.trim().to_string(),
        };
        let gitlab = GitLabClient::new(working_dir.clone(), &gitlab_repo)?;
        let model = AgentModel::connect(&ctx, "ops", working_dir, ModelPreferences::default())?;
        let global = settings::settings();
        Ok(Self {
            state,
            gitlab,
            model,
            config,
            scope_label: global.scope_label.clone(),
        })
    }

    fn on_shutdown(&mut self) {}
}

pub(crate) fn validate_instance_count(instances: usize) -> Result<()> {
    if instances > MAX_INSTANCES {
        bail!(
            "[agent.ops] supports at most one instance (instances must be 0 or 1, got {instances})"
        );
    }
    Ok(())
}

fn ops_cycle(
    state: &AgentState,
    config: &OpsConfig,
    gitlab: &GitLabClient,
    model: &AgentModel,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
) -> Result<()> {
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    info!(
        "{}: Fetching GitLab issues and merge requests for deduplication context",
        state.agent_id
    );
    let gitlab_context_path = write_gitlab_context_file(state, gitlab, unix_ts)?;
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    info!(
        "{}: Fetching last {}h of logs from {}@{}:{}",
        state.agent_id, LOG_WINDOW_HOURS, config.ssh_user, config.ssh_host, config.log_path
    );

    let raw_log = fetch_remote_log_tail(&config.ssh_user, &config.ssh_host, &config.log_path)?;
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    let now = Utc::now();
    let (window_log, parsed_timestamps) = filter_log_to_time_window(&raw_log, now);
    if !parsed_timestamps {
        warn!(
            "{}: Could not parse timestamps in remote log tail; using full tail for analysis",
            state.agent_id
        );
    }

    if window_log.trim().len() < MIN_LOG_BYTES_FOR_ANALYSIS {
        info!(
            "{}: Log window too small to analyze ({} bytes)",
            state.agent_id,
            window_log.trim().len()
        );
        return Ok(());
    }

    let scrape_path = state.scrape_path(unix_ts);
    fs::write(&scrape_path, &window_log)
        .with_context(|| format!("Failed to write scrape file {}", scrape_path.display()))?;
    prune_old_scrape_files(state, MAX_SCRAPE_FILES_KEPT)?;

    let history = load_history(&state.history_path())?;
    ensure_history_file(state, &history)?;

    let analysis_path = write_task_context_file(
        &state.sessions_dir,
        &format!("{}-analysis-{unix_ts}.log", state.agent_id),
        &window_log,
    )?;
    let history_path = absolute_path(&state.history_path())?;

    let prompt = build_analysis_prompt(&analysis_path, &history_path, &gitlab_context_path);
    let agent_output = model.complete(
        &prompt,
        &InvokeOptions {
            activity_label: Some(format!("{} analyzing logs", state.agent_id)),
            ..InvokeOptions::default()
        },
    )?;

    if !agent_output.has_final_result_text && agent_output.response.trim().is_empty() {
        warn!(
            "{}: Model returned no usable analysis output",
            state.agent_id
        );
        return Ok(());
    }

    let proposals = extract_ops_issues(&agent_output);
    if proposals.is_empty() {
        info!(
            "{}: No new actionable errors found in log window",
            state.agent_id
        );
        return Ok(());
    }

    let now_iso = now.to_rfc3339();
    let mut history = history;
    let mut created = 0usize;
    for proposal in proposals {
        match gitlab.create_issue(&proposal.title, &proposal.description) {
            Ok(issue_iid) => {
                info!(
                    "{}: Created GitLab issue #{}: {}",
                    state.agent_id, issue_iid, proposal.title
                );
                if let Some(p) = proposal.priority
                    && let Err(e) = gitlab.add_issue_label(issue_iid, &gitlab::priority_label(p))
                {
                    warn!(
                        "{}: Failed to set priority label on #{}: {}",
                        state.agent_id, issue_iid, e
                    );
                }
                if let Some(lbl) = scope_label
                    && let Err(e) = gitlab.add_issue_label(issue_iid, lbl)
                {
                    warn!(
                        "{}: Failed to add scope label {:?} on #{}: {}",
                        state.agent_id, lbl, issue_iid, e
                    );
                }

                history.entries.push(OpsIssueHistoryEntry {
                    gitlab_issue_iid: issue_iid,
                    title: proposal.title.clone(),
                    log_line: proposal.log_line.clone(),
                    created_at: now_iso.clone(),
                });
                created += 1;
            }
            Err(e) => {
                warn!(
                    "{}: Failed to create issue for {}: {}",
                    state.agent_id, proposal.title, e
                );
            }
        }
    }

    if created > 0 {
        save_history(&state.history_path(), &history)?;
        info!(
            "{}: Created {} new GitLab issue(s) from log analysis",
            state.agent_id, created
        );
    }

    Ok(())
}

fn build_analysis_prompt(log_path: &str, history_path: &str, gitlab_context_path: &str) -> String {
    format!(
        r#"You are an operations agent triaging log data for errors and exceptions.

Read these context files before proposing any new GitLab issues:

1. Log session file (primary analysis input):
{log_path}

2. Ops issue history (issues previously created by this agent, with related log lines):
{history_path}

3. Current GitLab context (all open issues and merge requests with descriptions and comments):
{gitlab_context_path}

Analyze ONLY the log session file for new errors, exceptions, panics, fatal failures, or repeated error patterns.

Use the issue history and GitLab context files to avoid creating duplicate issues for problems that are already tracked, discussed, or being addressed.

Use the project codebase in your workspace only as auxiliary context: map log errors to likely code paths, root causes, and concrete remediation steps. Do not rely on production host, deployment, or infrastructure details beyond what appears in the log file.

For each NEW distinct problem that is not already covered, propose one GitLab issue.

Return ONLY a machine-readable block:

{begin}
[
  {{
    "title": "Short actionable issue title",
    "description": "Markdown body with log evidence, likely code area, impact, and suggested remediation",
    "priority": 1,
    "log_line": "exact representative log line from the session file"
  }}
]
{end}

Rules:
- Return an empty JSON array [] if there are no NEW actionable errors.
- Do not propose issues for errors already represented in the history or GitLab context files.
- Titles must be specific and actionable.
- Descriptions must cite log evidence and, when helpful, relevant code context from the repository.
"#,
        log_path = log_path,
        history_path = history_path,
        gitlab_context_path = gitlab_context_path,
        begin = OPS_ISSUES_BEGIN,
        end = OPS_ISSUES_END,
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
        &state.sessions_dir,
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
    if !path.exists() {
        return Ok(OpsIssueHistory::default());
    }
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read issue history at {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(OpsIssueHistory::default());
    }
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse issue history JSON at {}", path.display()))
}

fn save_history(path: &Path, history: &OpsIssueHistory) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create issue history directory")?;
    }
    let json =
        serde_json::to_string_pretty(history).context("Failed to serialize issue history")?;
    fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn validate_remote_path(path: &str) -> Result<()> {
    ensure!(!path.is_empty(), "log_path must not be empty");
    for ch in path.chars() {
        ensure!(
            !matches!(ch, '\0' | '\n' | '\r' | ';' | '|' | '&' | '$' | '`'),
            "log_path contains unsupported character: {ch:?}"
        );
    }
    Ok(())
}

fn validate_ssh_identity(value: &str, field: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{field} must not be empty");
    ensure!(
        !value.chars().any(|c| c.is_whitespace()),
        "{field} must not contain whitespace"
    );
    Ok(())
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

fn extract_ops_issues_block(text: &str) -> Option<String> {
    let start = text.find(OPS_ISSUES_BEGIN)?;
    let body_start = start + OPS_ISSUES_BEGIN.len();
    let rest = &text[body_start..];
    let end_rel = rest.find(OPS_ISSUES_END)?;
    let body = rest[..end_rel].trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct OpsIssueRaw {
    title: String,
    description: String,
    priority: Option<u8>,
    #[serde(default, alias = "sample_line")]
    log_line: String,
}

fn parse_ops_issues_json(json: &str) -> Vec<OpsIssueProposal> {
    let parsed: Vec<OpsIssueRaw> = match serde_json::from_str(json) {
        Ok(items) => items,
        Err(e) => {
            warn!("Failed to parse OPS issues JSON: {e}");
            return Vec::new();
        }
    };

    parsed
        .into_iter()
        .filter_map(|item| {
            let title = item.title.trim();
            if title.is_empty() {
                return None;
            }
            Some(OpsIssueProposal {
                title: title.to_string(),
                description: item.description.trim().to_string(),
                priority: item.priority,
                log_line: item.log_line.trim().to_string(),
            })
        })
        .collect()
}

fn extract_ops_issues(agent_output: &AgentHandoff) -> Vec<OpsIssueProposal> {
    let text = if !agent_output.response.trim().is_empty() {
        agent_output.response.as_str()
    } else {
        agent_output.instructions.as_deref().unwrap_or("")
    };

    if let Some(block) = extract_ops_issues_block(text) {
        return parse_ops_issues_json(&block);
    }

    parse_ops_issues_json(text.trim())
}

fn prune_old_scrape_files(state: &AgentState, keep: usize) -> Result<()> {
    let prefix = format!("{}-scrape-", state.agent_id);
    let mut scrape_files = Vec::new();
    for entry in fs::read_dir(&state.sessions_dir)
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
        assert_eq!(settings.ssh_user, "deploy");
        assert_eq!(settings.log_path, "/var/log/app/app.log");
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
    fn extract_ops_issues_reads_block() {
        let output = AgentHandoff {
            response: format!(
                "analysis\n{OPS_ISSUES_BEGIN}\n[{{\"title\":\"Fix DB timeout\",\"description\":\"details\",\"priority\":2,\"log_line\":\"ERROR timeout\"}}]\n{OPS_ISSUES_END}\n"
            ),
            has_final_result_text: true,
            ..AgentHandoff::default()
        };
        let issues = extract_ops_issues(&output);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].title, "Fix DB timeout");
        assert_eq!(issues[0].log_line, "ERROR timeout");
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
    }

    #[test]
    fn validate_instance_count_allows_zero_or_one() {
        assert!(validate_instance_count(0).is_ok());
        assert!(validate_instance_count(1).is_ok());
        assert!(validate_instance_count(2).is_err());
    }
}
