use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

use super::{claim, issue_in_scope, labels, pmo_cursor_ask, strip_internal_markers};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient, Issue};
use crate::agents::settings;
use crate::agents::workspace::{
    ensure_agent_repo, extract_project_name, require_gitlab_repo, sessions_dir, work_dir,
};
use crate::core::agent::{AgentHandoff, HandoffSubIssue, InvokeOptions};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};

const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";

/// The PMO's structured-output tool definition. Passed to the harness via
/// `session/new` so the harness registers a generic `StructuredOutputTool`
/// named `plan`. The model calls it with its triage decision as structured
/// JSON.
fn plan_tool_definition() -> serde_json::Value {
    serde_json::json!({
        "name": "plan",
        "description": "Emit your triage decision as structured JSON. This is the primary output channel — Potlatch reads the tool's JSON, not your streamed text. Call this exactly once with your decision and the fields relevant to it.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "description": "Your triage decision. Must be exactly one of: \"guide_worker\", \"split\", \"already_done\", \"needs_clarification\", \"wait_for_dependency\".",
                    "type": "string",
                    "enum": ["guide_worker", "split", "already_done", "needs_clarification", "wait_for_dependency"]
                },
                "instructions": {
                    "description": "For guide_worker: 3-5 sentences, one clear action for the worker. Posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable.",
                    "type": "string"
                },
                "sub_issues": {
                    "description": "For split: the sub-issues to create. Each must have a title and description.",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "description": "Concise sub-issue title.",
                                "type": "string"
                            },
                            "description": {
                                "description": "Scope and acceptance criteria. If this sub-issue depends on another, reference it by title.",
                                "type": "string"
                            },
                            "priority": {
                                "description": "Priority: 1 (critical/blocking), 2 (high/depended-on), 3 (normal/independent).",
                                "type": "integer",
                                "enum": [1, 2, 3]
                            },
                            "depends_on": {
                                "description": "1-based index of another sub-issue this one depends on (omit or 0 if none).",
                                "type": "integer"
                            }
                        },
                        "required": ["title", "description"]
                    }
                },
                "reason": {
                    "description": "For already_done: why the codebase already satisfies the issue.",
                    "type": "string"
                },
                "question": {
                    "description": "For needs_clarification: specific questions for a human. Posted as a GitLab comment.",
                    "type": "string"
                },
                "plan_text": {
                    "description": "For needs_clarification: your current best plan for this issue. The system will update the issue description with this text so humans can see and refine your proposed approach. Write a structured plan including scope, proposed approach, and any open questions. Each refinement cycle overwrites the description with an improved version.",
                    "type": "string"
                },
                "dependency_issue_iid": {
                    "description": "For wait_for_dependency: the IID (number) of the existing open issue this issue depends on and must wait for. Must be a positive integer.",
                    "type": "integer"
                }
            },
            "required": ["decision"]
        }
    })
}

#[derive(Debug, Clone)]
struct PmoConfig {
    poll_interval_secs: u64,
    cursor_ask_via_gitlab: bool,
    cursor_ask_gitlab_timeout_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PmoAgentSettings {
    #[serde(default = "default_pmo_poll_interval")]
    poll_interval_secs: u64,
    #[serde(default)]
    cursor_ask_via_gitlab: bool,
    #[serde(default = "default_cursor_ask_gitlab_timeout_secs")]
    cursor_ask_gitlab_timeout_secs: u64,
}

fn default_pmo_poll_interval() -> u64 {
    180
}

fn default_cursor_ask_gitlab_timeout_secs() -> u64 {
    600
}

impl PmoAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        raw.clone()
            .try_into()
            .context("pmo agent settings from config")
    }
}

struct AgentState {
    sessions_dir: String,
    working_dir: String,
    agent_id: String,
    project_name: String,
}

impl AgentState {
    fn ensure_sessions_dir(&self) -> Result<()> {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        fs::create_dir_all(ctx_dir).context("Failed to create .potlatch-context directory")?;

        Ok(())
    }

    fn state_path(&self) -> path::PathBuf {
        path::Path::new(&self.sessions_dir).join(format!("{}_state.json", &self.agent_id))
    }

    fn claim_label(&self) -> String {
        format!("claimed:{}", &self.agent_id)
    }

    fn save_state(&self, issue_iid: u64) {
        let path = self.state_path();
        let json = format!(r#"{{"claimed_issue_iid":{}}}"#, issue_iid);
        if let Err(e) = fs::write(&path, json) {
            warn!("Failed to save PMO state: {}", e);
        }
    }

    fn clear_state(&self) {
        let _ = fs::remove_file(self.state_path());
    }

    fn context_path(&self, id: u64) -> path::PathBuf {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        let file_name = format!("{}-issue-{}.md", &self.agent_id, id);
        ctx_dir.join(&file_name)
    }

    fn task_path(&self) -> String {
        let base = path::Path::new(&self.sessions_dir);
        base.join(format!("{}_pending_split.json", &self.agent_id))
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) struct PmoAgent {
    state: AgentState,
    git_repo: GitRepo,
    gitlab: GitLabClient,
    model: AgentModel,
    config: PmoConfig,
    scope_label: String,
    claimed_issue_iid: Option<u64>,
}

impl CoreAgent for PmoAgent {
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        "pmo"
    }

    fn model(&self) -> &AgentModel {
        &self.model
    }

    fn banner(_config: &Config, banner: &mut Banner) {
        if let Some(repo) = settings::settings().gitlab_repo() {
            banner.set_once("repo", repo);
        }
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: "gitlab_poll",
            interval: Duration::from_secs(self.config.poll_interval_secs),
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 5000,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "gitlab_poll" => {
                let scope = crate::agents::scope_label_filter(&self.scope_label);
                let model = &self.model;
                let shutdown = Arc::clone(model.shutdown());
                if let Err(e) = pmo_cycle(
                    &self.state,
                    &self.git_repo,
                    &self.gitlab,
                    model,
                    &mut self.claimed_issue_iid,
                    Arc::clone(&shutdown),
                    scope,
                    &self.config,
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
            .agent("pmo")
            .context("[agent.pmo] section required")?;
        let settings = PmoAgentSettings::from_raw(&section.raw)?;
        let project_name = extract_project_name(&gitlab_repo)?;
        let agent_id = format!("pmo-{}", ctx.instance_id);
        ensure_agent_repo(
            &ctx.workflow.base_dir,
            &gitlab_repo,
            &project_name,
            &agent_id,
        )?;
        let pmo_dir = work_dir(&ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&ctx.workflow.base_dir, &project_name);
        let state = AgentState {
            sessions_dir,
            working_dir: pmo_dir.clone(),
            agent_id: agent_id.clone(),
            project_name,
        };
        state.ensure_sessions_dir()?;
        let config = PmoConfig {
            poll_interval_secs: settings.poll_interval_secs,
            cursor_ask_via_gitlab: settings.cursor_ask_via_gitlab,
            cursor_ask_gitlab_timeout_secs: settings.cursor_ask_gitlab_timeout_secs,
        };
        let git_repo = GitRepo::new(state.working_dir.clone());
        let gitlab = GitLabClient::new(state.working_dir.clone(), &gitlab_repo)?;
        let model = AgentModel::connect(
            &ctx,
            "pmo",
            state.working_dir.clone(),
            ModelPreferences {
                structured_output_tools: Some(vec![plan_tool_definition()]),
                ..ModelPreferences::default()
            },
        )?;
        let agent_settings = settings::settings();
        let scope = agent_settings.scope_label_filter();
        let claimed_issue_iid = try_resume_pmo_state(&state, &gitlab, scope);
        if let Some(iid) = claimed_issue_iid {
            match (gitlab.get_issue(iid), gitlab.list_issues()) {
                (Ok(issue), Ok(issues)) => {
                    if let Err(e) = refresh_pmo_issue_context_file(&state, &gitlab, &issue, &issues)
                    {
                        warn!(
                            "{}: Could not refresh PMO context file after resuming claim on #{}: {}",
                            &state.agent_id, iid, e
                        );
                    }
                }
                (Err(e), _) => warn!(
                    "{}: Could not fetch issue #{} to refresh context after resume: {}",
                    &state.agent_id, iid, e
                ),
                (_, Err(e)) => warn!(
                    "{}: Could not list issues to refresh context after resume: {}",
                    &state.agent_id, e
                ),
            }
        }
        Ok(Self {
            state,
            git_repo,
            gitlab,
            model,
            config,
            scope_label: agent_settings.scope_label.clone(),
            claimed_issue_iid,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.state.agent_id);
        if let Some(issue_iid) = self.claimed_issue_iid {
            info!(
                "{}: Preserving claim on issue #{} for restart",
                self.state.agent_id, issue_iid
            );
            self.state.save_state(issue_iid);
        }
        info!("{}: Stopped", self.state.agent_id);
    }
}

#[allow(clippy::too_many_arguments)]
fn pmo_cycle(
    state: &AgentState,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    model: &AgentModel,
    claimed_issue_iid: &mut Option<u64>,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
    pmo_config: &PmoConfig,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;

    // Ensure we're on the latest upstream — hard reset if checkout fails
    if let Err(e) = git_repo.checkout_remote_branch(&default_branch) {
        warn!(
            "{}: Failed to checkout {}: {}, forcing reset",
            &state.agent_id, default_branch, e
        );

        let _ = git_repo.reset_hard();
        git_repo.checkout_remote_branch(&default_branch)?;
    }

    // If we still hold a claim from a previous run, decide what to do.
    if let Some(held_iid) = *claimed_issue_iid {
        // Pending splits take priority — handled below
        let pending_file_check = state.task_path();
        let has_pending = load_pending_split(&pending_file_check)?;
        if has_pending.is_some() {
            // Fall through to pending split handling
        } else if let Ok(issue) = gitlab.get_issue(held_iid) {
            if !issue_in_scope(&issue, scope_label) {
                info!(
                    "{}: Held issue #{} left scope label {:?}, releasing",
                    &state.agent_id, held_iid, scope_label
                );

                let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                *claimed_issue_iid = None;

                state.clear_state();
            } else if issue.labels.contains(&labels::PMO_PENDING.to_string()) {
                // PMO is in plan-refinement mode: the issue description holds
                // the PMO's draft plan, and the comment thread has the Q&A.
                // Re-triage only when there are new human comments since the
                // last PMO triage — otherwise just wait.
                let issues = gitlab.list_issues();
                match issues {
                    Ok(issues) => {
                        if let Err(e) =
                            refresh_pmo_issue_context_file(state, gitlab, &issue, &issues)
                        {
                            warn!(
                                "{}: Could not refresh PMO context file while pmo-pending on #{}: {}",
                                &state.agent_id, held_iid, e
                            );
                        }

                        if has_new_comments_since_last_pmo_comment(gitlab, &issue, &state.agent_id)
                        {
                            info!(
                                "{}: Issue #{} has new comments, re-triaging pmo-pending refinement",
                                &state.agent_id, held_iid
                            );
                            // Re-run triage with the refreshed context. The PMO
                            // may refine the plan (needs_clarification again) or
                            // reach a final decision. Either way, pmo-pending
                            // stays — only a human removes it.
                            match process_action_required_issue(
                                state,
                                gitlab,
                                model,
                                &issue,
                                &issues,
                                scope_label,
                                Arc::clone(&shutdown),
                                pmo_config,
                            ) {
                                Ok(keep_claim) => {
                                    if !keep_claim {
                                        // The PMO reached a final decision — but
                                        // pmo-pending is still on the issue (the
                                        // PMO never removes it). The decision
                                        // was already acted on inside
                                        // process_action_required_issue. Release
                                        // the claim so the normal flow continues.
                                        let _ =
                                            claim::release_claim(gitlab, held_iid, &state.agent_id);
                                        *claimed_issue_iid = None;
                                        state.clear_state();
                                    }
                                    // If keep_claim, the PMO refined the plan and
                                    // is still waiting. Keep the claim and wait
                                    // for the next cycle.
                                }
                                Err(e) => {
                                    warn!(
                                        "{}: Failed to re-triage pmo-pending issue #{}: {}",
                                        &state.agent_id, held_iid, e
                                    );
                                }
                            }
                        } else {
                            debug!(
                                "{}: Issue #{} still pmo-pending, no new comments, waiting",
                                &state.agent_id, held_iid
                            );
                        }
                    }
                    Err(e) => warn!(
                        "{}: Could not list issues to refresh context (pmo-pending #{}): {}",
                        &state.agent_id, held_iid, e
                    ),
                }

                return Ok(());
            } else {
                // No longer pending (human removed the label) — release claim so it
                // can be re-processed as a fresh action-required issue.
                info!(
                    "{}: Releasing claim on issue #{} from previous run",
                    &state.agent_id, held_iid
                );

                let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                *claimed_issue_iid = None;

                state.clear_state();
            }
        } else {
            // Can't fetch issue — release to be safe
            let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
            *claimed_issue_iid = None;

            state.clear_state();
        }
    }

    let pending_file = state.task_path();
    if let Some(pending_split) = load_pending_split(&pending_file)? {
        info!(
            "{}: Resuming pending split for issue #{} ({} sub-issues remaining)",
            &state.agent_id,
            pending_split.parent_issue_iid,
            pending_split.sub_issues.len()
        );

        match (
            gitlab.get_issue(pending_split.parent_issue_iid),
            gitlab.list_issues(),
        ) {
            (Ok(parent_issue), Ok(issues)) => {
                if let Err(e) =
                    refresh_pmo_issue_context_file(state, gitlab, &parent_issue, &issues)
                {
                    warn!(
                        "{}: Could not refresh PMO context file before pending split on #{}: {}",
                        &state.agent_id, pending_split.parent_issue_iid, e
                    );
                }
            }
            (Err(e), _) => warn!(
                "{}: Could not fetch parent issue #{} for context refresh: {}",
                &state.agent_id, pending_split.parent_issue_iid, e
            ),
            (_, Err(e)) => warn!(
                "{}: Could not list issues for context refresh (pending split): {}",
                &state.agent_id, e
            ),
        }

        match resume_split(&pending_file, gitlab, &pending_split, scope_label) {
            Ok(_) => {
                info!(
                    "{}: Successfully completed pending split for issue #{}",
                    &state.agent_id, pending_split.parent_issue_iid
                );

                delete_pending_split(&pending_file)?;
                // Release the claim from the split
                if let Some(held_iid) = *claimed_issue_iid {
                    let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                    *claimed_issue_iid = None;

                    state.clear_state();
                }
            }
            Err(e) => {
                error!(
                    "{}: Failed to complete pending split: {}",
                    &state.agent_id, e
                );
            }
        }

        return Ok(());
    }

    let issues = gitlab.list_issues()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    for issue in &issues {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !should_process_issue(issue, scope_label) {
            continue;
        }

        if claim::is_claimed(&issue.labels) {
            debug!(
                "{}: Issue #{} already claimed, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if !claim::try_claim_issue(gitlab, issue.iid, &state.agent_id, shutdown.as_ref())? {
            info!(
                "{}: Failed to claim issue #{}, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_claim(gitlab, issue.iid, &state.agent_id);
            return Ok(());
        }

        *claimed_issue_iid = Some(issue.iid);

        state.save_state(issue.iid);

        info!(
            "{}: Processing issue #{}: {}",
            &state.agent_id, issue.iid, issue.title
        );

        match process_action_required_issue(
            state,
            gitlab,
            model,
            issue,
            &issues,
            scope_label,
            Arc::clone(&shutdown),
            pmo_config,
        ) {
            Ok(keep_claim) => {
                info!(
                    "{}: Successfully processed issue #{}",
                    &state.agent_id, issue.iid
                );
                if keep_claim {
                    info!(
                        "{}: Keeping claim on issue #{} (pmo-pending)",
                        &state.agent_id, issue.iid
                    );
                } else {
                    claim::release_claim(gitlab, issue.iid, &state.agent_id)?;
                    *claimed_issue_iid = None;

                    state.clear_state();
                }
            }
            Err(e) => {
                error!(
                    "{}: Failed to process issue #{}: {}",
                    &state.agent_id, issue.iid, e
                );

                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }

                claim::release_claim(gitlab, issue.iid, &state.agent_id)?;
                *claimed_issue_iid = None;

                state.clear_state();
            }
        }

        break;
    }

    // Assign default priority to open issues that lack a priority label.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    assign_default_priority(&state.agent_id, &issues, gitlab, scope_label);

    // Close stale pmo-processed issues that have not been picked up for over 1 hour.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    close_stale_processed_issues(&state.agent_id, &issues, gitlab, scope_label);

    Ok(())
}

fn assign_default_priority(
    agent_id: &str,
    issues: &[Issue],
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) {
    for issue in issues {
        if issue.state != "opened" {
            continue;
        }
        if !issue_in_scope(issue, scope_label) {
            continue;
        }
        let has_priority = issue
            .labels
            .iter()
            .any(|l| l.starts_with(gitlab::PRIORITY_LABEL_PREFIX));
        if !has_priority {
            let label = gitlab::priority_label(gitlab::DEFAULT_PRIORITY);
            debug!(
                "{}: Assigning default {} to issue #{}",
                agent_id, label, issue.iid
            );
            let _ = gitlab.add_issue_label(issue.iid, &label);
        }
    }
}

const STALE_THRESHOLD_SECS: u64 = 3600; // 1 hour

fn close_stale_processed_issues(
    agent_id: &str,
    issues: &[Issue],
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    for issue in issues {
        if issue.state != "opened" {
            continue;
        }
        if !issue_in_scope(issue, scope_label) {
            continue;
        }
        if !issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
            continue;
        }

        let Some(ref updated_str) = issue.updated_at else {
            continue;
        };

        let Some(updated_epoch) = parse_iso8601_to_epoch(updated_str) else {
            debug!(
                "{}: Could not parse updated_at for issue #{}: {}",
                agent_id, issue.iid, updated_str
            );
            continue;
        };

        if now.saturating_sub(updated_epoch) >= STALE_THRESHOLD_SECS {
            info!(
                "{}: Issue #{} has been pmo-processed for over 1 hour with no activity, closing",
                agent_id, issue.iid
            );
            let _ = gitlab.add_issue_comment(
                issue.iid,
                "Closing this issue — it has been marked as `pmo-processed` for over 1 hour with no further activity.",
            );
            let _ = gitlab.close_issue(issue.iid);
        }
    }
}

fn parse_iso8601_to_epoch(s: &str) -> Option<u64> {
    // GitLab returns timestamps like "2026-03-18T05:30:00.000Z" or "2026-03-18T05:30:00+00:00"
    // Parse manually to avoid pulling in a datetime crate.
    let s = s.trim().trim_end_matches('Z');
    let s = if let Some(pos) = s.rfind('+') {
        // Strip timezone offset like +00:00
        if pos > 10 { &s[..pos] } else { s }
    } else if let Some(pos) = s.rfind('-') {
        // Could be timezone offset like -05:00, but also date separator
        // Only strip if it looks like a timezone (position > 10, i.e. after the date part)
        if pos > 18 { &s[..pos] } else { s }
    } else {
        s
    };
    // Strip fractional seconds
    let s = if let Some(pos) = s.find('.') {
        &s[..pos]
    } else {
        s
    };
    // Now s should be "YYYY-MM-DDTHH:MM:SS"
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 {
        return None;
    }
    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
    let time_parts: Vec<u64> = parts[1].split(':').filter_map(|p| p.parse().ok()).collect();
    if date_parts.len() != 3 || time_parts.len() != 3 {
        return None;
    }
    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, min, sec) = (time_parts[0], time_parts[1], time_parts[2]);

    // Days from year 0 to start of month (non-leap)
    const DAYS_BEFORE_MONTH: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if !(1..=12).contains(&month) {
        return None;
    }
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let leap_extra = if is_leap && month > 2 { 1 } else { 0 };

    let days_since_epoch = (year - 1970) * 365 + (year - 1969) / 4 - (year - 1901) / 100
        + (year - 1601) / 400
        + DAYS_BEFORE_MONTH[(month - 1) as usize]
        + leap_extra
        + day
        - 1;

    Some(days_since_epoch * 86400 + hour * 3600 + min * 60 + sec)
}

fn should_process_issue(issue: &Issue, scope_label: Option<&str>) -> bool {
    if !issue_in_scope(issue, scope_label) {
        return false;
    }

    // Must have action-required label
    if !issue.labels.contains(&ACTION_REQUIRED_LABEL.to_string()) {
        return false;
    }

    // Skip if already processed by PMO
    if issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
        return false;
    }

    // Skip if PMO is waiting for human clarification
    if issue.labels.contains(&labels::PMO_PENDING.to_string()) {
        return false;
    }

    // Skip drafts
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return false;
    }

    true
}

/// Returns `Ok(true)` if the PMO should keep its claim (pmo-pending / needs clarification).
/// Returns `Ok(false)` if the claim can be released.
#[allow(clippy::too_many_arguments)]
fn process_action_required_issue(
    state: &AgentState,
    gitlab: &GitLabClient,
    model: &AgentModel,
    issue: &Issue,
    all_issues: &[Issue],
    scope_label: Option<&str>,
    shutdown: Arc<AtomicBool>,
    pmo_config: &PmoConfig,
) -> Result<bool> {
    let parent_priority = issue.priority();
    let context_path = refresh_pmo_issue_context_file(state, gitlab, issue, all_issues)?;
    let prompt = build_split_prompt(state, issue, &context_path, parent_priority)?;

    let ask_handler: Option<
        std::sync::Arc<dyn crate::core::model::acp::client::CursorAskQuestionHandler>,
    > = if pmo_config.cursor_ask_via_gitlab {
        let timeout = if pmo_config.cursor_ask_gitlab_timeout_secs > 0 {
            Some(std::time::Duration::from_secs(
                pmo_config.cursor_ask_gitlab_timeout_secs,
            ))
        } else {
            None
        };
        Some(std::sync::Arc::new(
            pmo_cursor_ask::GitLabIssueCursorAskHandler::new(
                issue.iid,
                gitlab.clone(),
                Arc::clone(&shutdown),
                timeout,
            ),
        ))
    } else {
        None
    };

    info!(
        "{}: PMO agent triaging issue #{}",
        &state.agent_id, issue.iid
    );
    let mut agent_output = model.complete(
        &prompt,
        &InvokeOptions {
            cursor_ask_question_handler: ask_handler,
            activity_label: Some(format!("{} triaging issue #{}", &state.agent_id, issue.iid)),
            ..InvokeOptions::default()
        },
    )?;
    info!(
        "{}: PMO agent finished triaging issue #{}",
        &state.agent_id, issue.iid
    );
    // Apply the structured JSON the model emitted via the `plan` tool.
    // This populates the handoff's structured fields (decision, sub_issues,
    // instructions, etc.) which the decision functions below read.
    let had_plan_output = handoff_has_plan_output(&agent_output);
    agent_output = apply_pmo_handoff(agent_output);
    // The model must call the `plan` tool. If it didn't, bail with a clear
    // error rather than falling through to the decision predicates (which
    // would all return false and hit the SPLIT path with a misleading
    // "No sub-issues" error).
    if !had_plan_output {
        let preview = pmo_truncate_utf8_by_bytes(agent_output.response.trim(), 800);
        warn!(
            "PMO: Model did not call `plan` tool for issue #{} (response preview, {} bytes): {}",
            issue.iid,
            preview.len(),
            preview
        );
        anyhow::bail!(
            "PMO did not call `plan` tool for issue #{}; retrying later",
            issue.iid
        );
    }
    if agent_output.decision.is_none() && agent_output.sub_issues.is_empty() {
        let preview = pmo_truncate_utf8_by_bytes(agent_output.response.trim(), 800);
        warn!(
            "PMO: Plan tool output had no decision and no sub-issues for issue #{} (response preview, {} bytes): {}",
            issue.iid,
            preview.len(),
            preview
        );
        anyhow::bail!(
            "PMO plan output for issue #{} had no decision; retrying later",
            issue.iid
        );
    }

    // --- NEEDS_CLARIFICATION: PMO needs human input, refine plan ---
    if pmo_needs_clarification(&agent_output) {
        let question = strip_internal_markers(&extract_clarification_question(&agent_output));
        info!(
            "PMO: Issue #{} needs clarification, marking pmo-pending and updating plan",
            issue.iid
        );

        // Update the issue description with the PMO's current plan draft, so
        // humans can see and refine the proposed approach directly in the
        // GitLab issue body.
        if let Some(plan_text) = extract_plan_text(&agent_output) {
            let cleaned = strip_internal_markers(&plan_text);
            if !cleaned.is_empty()
                && let Err(e) = gitlab.update_issue_description(issue.iid, &cleaned)
            {
                warn!(
                    "PMO: Failed to update issue #{} description with plan: {}",
                    issue.iid, e
                );
            }
        }

        gitlab.add_issue_comment(
            issue.iid,
            &format!(
                "**PMO needs clarification before proceeding:**\n\n{}\n\n\
                 Please reply to this comment with the requested information. \
                 The PMO will refine the plan based on your feedback. \
                 Remove the `pmo-pending` label when you are satisfied with the plan to let the PMO proceed.",
                question
            ),
        )?;

        gitlab.add_issue_label(issue.iid, labels::PMO_PENDING)?;
        return Ok(true); // keep claim
    }

    // --- ALREADY_DONE: work is already implemented, close the issue ---
    if is_pmo_already_done_response(&agent_output) {
        let reason = strip_internal_markers(&extract_already_done_reason(&agent_output));
        info!("PMO: Issue #{} is already implemented, closing", issue.iid);
        gitlab.add_issue_comment(
            issue.iid,
            &format!(
                "**PMO: Closing — this work is already implemented.**\n\n{}",
                reason
            ),
        )?;
        let _ = gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL);
        let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
        gitlab.close_issue(issue.iid)?;
        return Ok(false);
    }

    // --- WAIT_FOR_DEPENDENCY: issue depends on another open issue, park it ---
    if pmo_wait_for_dependency(&agent_output) {
        let Some(dep_iid) = agent_output.depends_on_issue else {
            warn!(
                "PMO: wait_for_dependency decision for issue #{} but no dependency_issue_iid provided, releasing claim for retry",
                issue.iid
            );
            anyhow::bail!(
                "PMO wait_for_dependency for issue #{} missing dependency_issue_iid; retrying later",
                issue.iid
            );
        };
        info!(
            "PMO: Issue #{} depends on open issue #{}, parking",
            issue.iid, dep_iid
        );
        let label = format!("waiting-on-issue:#{dep_iid}");
        let _ = gitlab.add_issue_label(issue.iid, &label);
        let _ = gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL);
        let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
        gitlab.add_issue_comment(
            issue.iid,
            &format!(
                "**PMO: Parking — this issue depends on issue #{dep_iid} which is still open.**\n\n\
                 The worker will skip this issue until #{} is closed, then resume automatically.",
                dep_iid
            ),
        )?;
        return Ok(false);
    }

    // --- GUIDE_WORKER: single focused retry instruction ---
    if pmo_guides_worker(&agent_output) {
        let guidance = strip_internal_markers(&extract_guidance(&agent_output));
        if guidance.trim().is_empty() {
            warn!(
                "PMO: GUIDE_WORKER output for issue #{} had no usable guidance, releasing claim for retry",
                issue.iid
            );
            anyhow::bail!(
                "PMO GUIDE_WORKER output for issue #{} had no usable guidance; retrying later",
                issue.iid
            );
        }
        info!("PMO: Issue #{} needs guidance, not splitting", issue.iid);
        gitlab.add_issue_comment(issue.iid, &format_pmo_guidance_comment(&guidance))?;
        gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
        return Ok(false);
    }

    // --- SPLIT: create sub-issues, close the parent as a task container ---
    // Sub-issues come from the structured `plan` tool output (populated by
    // `apply_pmo_handoff` into `agent_output.sub_issues`). No text-marker
    // parsing — the model must call the `plan` tool with a `sub_issues` array.
    let sub_issues = agent_output.sub_issues.clone();

    if sub_issues.is_empty() {
        let preview = pmo_truncate_utf8_by_bytes(agent_output.response.trim(), 800);
        warn!(
            "PMO: No sub-issues from plan tool for issue #{} (response preview, {} bytes): {}",
            issue.iid,
            preview.len(),
            preview
        );

        anyhow::bail!(
            "PMO split output for issue #{} was not machine-readable; retrying later",
            issue.iid
        );
    }

    let pending_file = state.task_path();

    let pending_split = PendingSplit {
        parent_issue_iid: issue.iid,
        parent_issue_title: issue.title.clone(),
        parent_priority: issue.priority(),
        sub_issues: sub_issues.clone(),
        created_issue_ids: Vec::new(),
    };

    save_pending_split(&pending_file, &pending_split)?;

    match resume_split(&pending_file, gitlab, &pending_split, scope_label) {
        Ok(_) => {
            info!(
                "{}: Successfully split issue #{} into sub-issues",
                &state.agent_id, issue.iid
            );
            delete_pending_split(&pending_file)?;
        }
        Err(e) => {
            error!(
                "{}: Failed to create all sub-issues: {}",
                &state.agent_id, e
            );
            return Err(e);
        }
    }

    // Close the parent issue — it served as a task container, the real work
    // is now tracked in the sub-issues.
    info!(
        "{}: Closing parent issue #{} (task container)",
        &state.agent_id, issue.iid
    );

    let _ = gitlab.add_issue_comment(
        issue.iid,
        "Closing this issue — it has been split into sub-issues above. \
         The sub-issues now track the actual work.",
    );

    let _ = gitlab.close_issue(issue.iid);

    Ok(false)
}

fn build_existing_issues_summary(current_iid: u64, all_issues: &[Issue]) -> String {
    let mut lines = Vec::new();
    for issue in all_issues {
        if issue.iid == current_iid || issue.state != "opened" {
            continue;
        }
        let in_progress = issue.labels.contains(&"in-progress".to_string())
            || issue.labels.iter().any(|l| l.starts_with("claimed:"));
        let status = if in_progress { "IN-PROGRESS" } else { "OPEN" };
        let p = issue.priority();
        lines.push(format!(
            "- #{} [{}] (priority {p}): {}",
            issue.iid, status, issue.title
        ));
    }
    if lines.is_empty() {
        "No other open issues.".to_string()
    } else {
        lines.join("\n")
    }
}

/// Markdown for the "Comments" section of `pmo-issue-*.md` via [`GitLabClient::get_issue_comments`].
fn pmo_gitlab_comments_section(gitlab: &GitLabClient, issue_iid: u64) -> (String, usize) {
    match gitlab.get_issue_comments(issue_iid) {
        Ok(comments) => {
            let n = comments.len();
            let text = if comments.is_empty() {
                "No comments yet.".to_string()
            } else {
                comments
                    .iter()
                    .map(|c| c.format_for_prompt())
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            (text, n)
        }
        Err(e) => {
            warn!(
                "PMO: failed to fetch GitLab issue comments for #{}: {}",
                issue_iid, e
            );
            (
                format!(
                    "**ERROR: Potlatch could not load GitLab issue comments.**\n\n\
                     PMO triage may be unreliable until this works (check `glab` auth, network, and that `repo_path` is the git clone root).\n\n\
                     ```text\n{e}\n```"
                ),
                0,
            )
        }
    }
}

/// Rewrites `pmo-issue-<iid>.md` under [`BREEZE_CONTEXT_DIR`]. Call whenever PMO holds or resumes
/// work on an issue (new claim, resumed claim, pmo-pending poll, pending split).
fn refresh_pmo_issue_context_file(
    state: &AgentState,
    gitlab: &GitLabClient,
    issue: &Issue,
    all_issues: &[Issue],
) -> Result<String> {
    let (comments_text, gitlab_note_count) = pmo_gitlab_comments_section(gitlab, issue.iid);
    let existing_issues_text = build_existing_issues_summary(issue.iid, all_issues);
    let closed_mr_text = pmo_closed_mr_section(gitlab, issue.iid);

    let generated_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "# PMO Triage Context\n\nProject: {project_name}\nIssue: #{iid} {title}\n\n## Issue description\n{description}\n\n## Comments and worker feedback\n{comments}\n\n## Closed merge request context\n{closed_mr}\n\n## Existing open issues\n{existing}\n\n---\n_Potlatch: PMO refreshed this file; {gitlab_note_count} GitLab note(s) in the comments section; UNIX ts {generated_ts}._\n",
        project_name = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        description = issue.description,
        comments = comments_text,
        closed_mr = closed_mr_text,
        existing = existing_issues_text,
        gitlab_note_count = gitlab_note_count,
        generated_ts = generated_ts,
    );

    let path = state.context_path(issue.iid);
    fs::write(&path, &body)
        .with_context(|| format!("Failed to write PMO context file {}", &state.agent_id,))?;

    let context_path = fs::canonicalize(&path)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();

    Ok(context_path)
}

/// Build the "Closed merge request context" section for the PMO context file.
///
/// When a worker hands off an issue to PMO after failing to resolve reviewer
/// feedback, it closes the MR and posts a reason as an issue comment. But the
/// MR's own comment threads (reviewer feedback, worker replies) and the diff
/// carry context the issue comment alone doesn't capture. This section fetches
/// the most recent closed MR for the issue (keyed by the `issue-<iid>` branch
/// convention) and includes its comments and a truncated diff.
///
/// Returns "No closed MRs found for this issue." when there are none (the
/// common case for issues that were never assigned to a worker).
fn pmo_closed_mr_section(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let branch_name = format!("issue-{issue_iid}");
    let mr_iids = match gitlab.find_mrs_by_source_branch(&branch_name) {
        Ok(iids) => iids,
        Err(e) => {
            warn!("PMO: failed to search MRs for issue #{issue_iid} branch {branch_name}: {e}");
            return format!(
                "**ERROR: Potlatch could not search merge requests for this issue.**\n\n```\n{e}\n```"
            );
        }
    };

    // Find the most recent non-open MR (closed or merged). Open MRs are
    // excluded — they're handled by the worker/reviewer flow, not PMO.
    let mr_iid = mr_iids
        .iter()
        .find(|&&iid| match gitlab.get_merge_request(iid) {
            Ok(mr) => mr.state != "opened",
            Err(_) => false,
        })
        .copied();

    let Some(mr_iid) = mr_iid else {
        return "No closed MRs found for this issue.".to_string();
    };

    let mr = match gitlab.get_merge_request(mr_iid) {
        Ok(mr) => mr,
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid}: {e}");
            return format!(
                "**ERROR: Potlatch could not load merge request !{mr_iid}.**\n\n```\n{e}\n```"
            );
        }
    };

    let comments_text = match gitlab.get_mr_comments(mr_iid) {
        Ok(comments) if comments.is_empty() => "No MR comments.".to_string(),
        Ok(comments) => comments
            .iter()
            .map(|c| c.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n"),
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid} comments: {e}");
            format!("**ERROR: could not load MR comments: {e}**")
        }
    };

    let diff_text = match gitlab.get_merge_request_changes(mr_iid) {
        Ok(snapshot) if snapshot.patch.is_empty() => "No diff available.".to_string(),
        Ok(snapshot) => {
            let files_list = if snapshot.files.is_empty() {
                String::new()
            } else {
                format!(
                    "\n\nChanged files ({}):\n{}\n",
                    snapshot.files.len(),
                    snapshot
                        .files
                        .iter()
                        .take(50)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            };
            let patch = truncate_diff_for_pmo(&snapshot.patch);
            if snapshot.overflow {
                format!(
                    "Diff (truncated — too large for full inclusion):\n```\n{patch}\n```{files_list}\n\n_Note: the full diff exceeds the context limit. See the MR in GitLab for the complete changes._"
                )
            } else {
                format!("Diff:\n```\n{patch}\n```{files_list}")
            }
        }
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid} diff: {e}");
            format!("**ERROR: could not load MR diff: {e}**")
        }
    };

    format!(
        "MR: !{mr_iid} {mr_title}\nState: {mr_state}\nSource branch: {source_branch} → Target: {target_branch}\n\n### MR comments\n{comments}\n\n### MR diff\n{diff}",
        mr_iid = mr.iid,
        mr_title = mr.title,
        mr_state = mr.state,
        source_branch = mr.source_branch,
        target_branch = mr.target_branch,
        comments = comments_text,
        diff = diff_text,
    )
}

/// Truncate a diff patch for inclusion in the PMO context file. Keeps the
/// first 8000 characters (enough to see the shape of changes without bloating
/// the context) with a marker when truncated.
fn truncate_diff_for_pmo(patch: &str) -> String {
    const MAX_DIFF_CHARS: usize = 8_000;
    if patch.len() <= MAX_DIFF_CHARS {
        return patch.to_string();
    }
    let mut end = MAX_DIFF_CHARS;
    while !patch.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!(
        "{}\n\n[...diff truncated at {} chars, {}/{} bytes shown...]",
        &patch[..end],
        MAX_DIFF_CHARS,
        end,
        patch.len()
    )
}

fn build_split_prompt(
    state: &AgentState,
    issue: &Issue,
    context_path: &str,
    parent_priority: u8,
) -> Result<String> {
    let prompt = format!(
        r#"You are a Project Management Office (PMO) agent responsible for triaging issues that an automated worker agent could not implement.

PROJECT: {project}

ISSUE #{iid}: {title}  _(summary only — not sufficient by itself)_

TASK CONTEXT FILE (you MUST open and read this path on disk — it has the full picture):
{context_path}

CONTEXT:
An automated worker agent attempted to implement this issue but was unable to complete it.
Potlatch wrote the path above as a markdown file: **full issue description**, **every GitLab issue comment** (including worker rejection / PMO notes), **closed merge request context** (MR comments, reviewer feedback, and diff from the worker's closed MR, when one exists), and **the list of other open issues**. That file is the authoritative written context for this triage.
- Use your **file-reading** capability on the absolute path and read it **end-to-end** before you decide the situation is unclear.
- The single line `ISSUE #…: title` in this prompt is **not** a substitute for the file; do not claim "no context" or choose NEEDS_CLARIFICATION only because you did not read the task context file.
- The **Closed merge request context** section is especially important when the worker closed an MR after failing to resolve reviewer feedback — the MR comments and diff show what the reviewer asked for and what the worker tried.
Your job is to analyze the failure reason (from the file + repo when needed) and take the appropriate action.

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You do NOT write new production code — you inspect the existing project state, then write comments and create issue descriptions as needed
- The worker agent has FULL ACCESS to shell commands (rm, mv, git, etc.) and all build/test tools
- If the worker claimed it "cannot run commands" or "cannot delete files", that is WRONG — it CAN. Instruct it clearly.
- Before deciding GUIDE_WORKER or SPLIT, you MUST verify whether the issue is already implemented in the current project state when that is plausible from the issue, comments, or worker output. If the behavior/tests/code already exist, choose ALREADY_DONE so Potlatch will close the issue and add a comment.

## Output — call the `plan` tool with your decision

Call the `plan` tool with your decision as its arguments. Potlatch reads the tool's arguments directly — your streamed text is ignored for decision parsing. This is the ONLY output channel; you MUST call `plan` with your decision. The tool's parameters are:

- `decision` (required): one of `guide_worker`, `split`, `already_done`, `needs_clarification`, `wait_for_dependency`
- `instructions` (for guide_worker): 3-5 sentences, one clear action for the worker — posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable.
- `sub_issues` (for split): array of `{{"title": "...", "description": "...", "priority": 1-3, "depends_on": N}}` where `depends_on` is the 1-based index of another sub-issue this one depends on (omit or 0 if none). Example: if sub-issue 2 builds on sub-issue 1's work, set `"depends_on": 1`. The system automatically labels dependent issues so the worker won't start them until their dependency is closed.
- `reason` (for already_done): why the codebase already satisfies the issue
- `question` (for needs_clarification): specific questions for a human. Posted as a GitLab comment.
- `plan_text` (for needs_clarification): your current best plan for this issue. The system updates the issue description with this text so humans can see and refine your proposed approach. Write a structured plan including scope, proposed approach, and any open questions. Each refinement cycle overwrites the description with an improved version.
- `dependency_issue_iid` (for wait_for_dependency): the IID of the existing open issue this issue depends on and must wait for

Only fill the parameter(s) relevant to your decision. Everything you put in `instructions`, `reason`, or `question` is human-facing and posted to GitLab verbatim — do not include any internal markers, harness instructions, or meta-commentary.

DECISION — choose EXACTLY ONE of the following:

1. GUIDE_WORKER — Use ONLY when ALL of these are true:
   a) The issue describes a SINGLE, focused task (not a list of modules/files/components)
   b) The worker failed due to a specific misunderstanding, wrong command, or simple technical obstacle
   c) The fix is ONE clear action (e.g. "use flag X instead of Y", "the config file is at path Z")
   If your guidance would enumerate 2+ independent modules, files, or components, you MUST choose SPLIT instead.

2. SPLIT — Use when ANY of these are true:
   - The issue is a "task container" describing a broad goal (e.g. "add tests for module X", "refactor all Y", "check full code for Z") — these ALWAYS need splitting into concrete sub-tasks
   - The issue involves work on 2+ independent modules, files, or components
   - The issue is too large (estimated >500 lines of non-test code, or >1500 lines total including tests; auto-generated code does not count)
   - The issue description or worker rejection lists multiple distinct things to do
   - Your guidance would need to enumerate 2+ independent items
   Do NOT force a speculative split. If you cannot clearly define the sub-issues with concrete titles and descriptions, choose NEEDS_CLARIFICATION instead and draft your best understanding of the decomposition in `plan_text`.
   When splitting, the PARENT ISSUE will be CLOSED automatically as a task container. The sub-issues become the real tracked work.
   Each sub-issue should target ~500 lines of non-test code, ~1500 total including tests; auto-generated code does not count.

   PRIORITY LEVELS:
   - 1 = Critical: blocking other work, security fix, core dependency that other sub-issues depend on
   - 2 = High: important feature, depended on by lower-priority sub-issues
   - 3 = Normal: independent work, enhancements, nice-to-haves
   The parent issue has priority {parent_priority}. Sub-issues that are dependencies for others should get higher priority (lower number). Independent leaf tasks can inherit the parent priority or be lower.

   DEPENDENCIES:
   When a sub-issue cannot be started until another sub-issue is done, set `"depends_on"` to the 1-based index of the dependency. Example: if sub-issue 2 builds on the primitives from sub-issue 1, set `"depends_on": 1` on sub-issue 2. The system parks sub-issue 2 (labels it `waiting-on-issue:#N`) so the worker skips it until sub-issue 1 is closed. Use this for real build-order dependencies — do not set `depends_on` for issues that are merely related or could run in parallel.

3. ALREADY_DONE — Use when the work described in the issue is ALREADY fully implemented in the codebase:
   - The worker's output or your analysis shows the feature/tests/code already exists
   - There is nothing left to implement — the issue is simply outdated or redundant
   - Prefer ALREADY_DONE over GUIDE_WORKER or SPLIT when the required behavior is already present in the repository as it exists now

4. NEEDS_CLARIFICATION — Use when you need more information from a human, OR when you are unsure how to decompose the issue:
   - After reading the **task context file** and (if needed) the repo, the issue is still too vague to determine scope or intent
   - The worker's rejection and the issue (as given in that file) still don't give enough to guide or split
   - You need specific information from a human (e.g. which modules to cover, what the acceptance criteria are)
   - You are unsure how to split the issue into well-defined sub-issues
   Do **not** use this option because you skipped reading the task context file.
   When you choose NEEDS_CLARIFICATION, write your current best plan in `plan_text` — the system will update the issue description so the human can see your proposed approach. Post your specific questions in `question` — they'll appear as a comment. You may be re-triaged multiple times as the human replies; each time, refine `plan_text` with your updated understanding. The `pmo-pending` label stays until a human removes it — when you reach a confident decision during refinement, still use `needs_clarification` and describe your recommendation in `question`. The human will remove the label to trigger final processing.

5. WAIT_FOR_DEPENDENCY — Use when this issue cannot be implemented until another EXISTING OPEN issue is closed:
   - The dependency is a real build-order blocker (the code from the other issue is a prerequisite)
   - Set `dependency_issue_iid` to the IID of the blocking issue (it must appear in the EXISTING OPEN ISSUES list)
   - The system parks this issue (labels it `waiting-on-issue:#N`) so the worker skips it until the dependency closes, then resumes automatically
   - Do NOT use this for issues that are merely related or could run in parallel — only for real prerequisites
   - Do NOT use this as a substitute for SPLIT's `depends_on` (that's for sub-issues you're creating now); use WAIT_FOR_DEPENDENCY only when the dependency is an already-existing separate issue

DUPLICATE / OVERLAP RULES (STRICT):
- Review the EXISTING OPEN ISSUES list above before creating any sub-issue.
- Do NOT create a sub-issue that duplicates or substantially overlaps with an existing open issue.
- If an existing OPEN (not IN-PROGRESS) issue covers part of the work, reference it (e.g. "See existing #42") instead of creating a new sub-issue for that part.
- If an existing IN-PROGRESS issue already covers it, simply skip that part entirely — do not create a sub-issue or reference.
- If ALL sub-issues would duplicate existing issues, choose GUIDE_WORKER instead and tell the worker which existing issues already cover the work.

INSTRUCTIONS:
1. Open and read the **entire** TASK CONTEXT FILE at the absolute path above (description, GitLab comments, closed MR context, existing issues). Do this first.
2. From that file, read the issue description and **all** comments — especially the worker's rejection reason. If there is a **Closed merge request context** section, read the MR comments and diff to understand what the reviewer asked for and what the worker tried.
3. Review the EXISTING OPEN ISSUES section in that same file to see what is already tracked.
4. TASK CONTAINER TEST: Does the issue describe a broad goal that involves multiple independent pieces of work (e.g. "add tests for all modules", "refactor X across the codebase", "check code for Y")? If YES → SPLIT. The parent issue is just a container; the real work is in the sub-issues.
5. GUIDANCE TEST: Is there ONE specific thing the worker misunderstood or did wrong? If YES → GUIDE_WORKER.
6. ENUMERATION TEST: If your guidance would list 2+ independent modules, files, or components → SPLIT, not GUIDE_WORKER.
7. CLARITY TEST: Only if the task context file plus (if needed) repo inspection still leaves intent unclear → NEEDS_CLARIFICATION.
8. COMPLETION TEST: Does the worker's output, the comments in the file, or your direct inspection of the current project state indicate the work is already fully implemented in the codebase? If YES → ALREADY_DONE.
9. DEPENDENCY TEST: Does this issue require code from another EXISTING OPEN issue to be implemented first? If YES → WAIT_FOR_DEPENDENCY (set `dependency_issue_iid`).
10. Choose EXACTLY ONE of GUIDE_WORKER, SPLIT, ALREADY_DONE, NEEDS_CLARIFICATION, or WAIT_FOR_DEPENDENCY — never combine them.
11. When in doubt between GUIDE_WORKER and SPLIT, prefer SPLIT — it's better to create focused sub-issues than to give the worker a laundry list.
12. Call the `plan` tool with the JSON shape above. This is the last step — your turn is not complete until you call `plan`.

Proceed with analyzing the issue autonomously.
"#,
        project = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        context_path = context_path,
        parent_priority = parent_priority,
    );

    Ok(prompt)
}

#[derive(Debug, Serialize, Deserialize)]
struct PendingSplit {
    parent_issue_iid: u64,
    parent_issue_title: String,
    #[serde(default)]
    parent_priority: u8,
    sub_issues: Vec<HandoffSubIssue>,
    created_issue_ids: Vec<u64>,
}

// --- Decision predicates (structured-fields only) ---
// The PMO's decision is read from the `plan` tool's JSON via
// `apply_pmo_handoff`, which populates `AgentHandoff.decision` and the
// supporting fields. No text-marker parsing — the harness guarantees the
// structured path.

fn pmo_needs_clarification(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("needs_clarification"))
        || output.needs_clarification.is_some()
        || output.question.is_some()
}

fn pmo_guides_worker(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("guide_worker"))
        || output.instructions.is_some()
}

fn is_pmo_already_done_response(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("already_done"))
}

fn pmo_wait_for_dependency(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("wait_for_dependency"))
        || output.depends_on_issue.is_some()
}

// --- Field extractors (structured-fields only) ---

fn extract_already_done_reason(agent_output: &AgentHandoff) -> String {
    if let Some(reason) = &agent_output.reason {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "The work described in this issue is already fully implemented in the codebase.".to_string()
}

fn extract_clarification_question(agent_output: &AgentHandoff) -> String {
    if let Some(question) = &agent_output.question {
        let trimmed = question.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(clarification) = &agent_output.needs_clarification {
        let trimmed = clarification.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "The PMO agent could not determine how to proceed with this issue. Please provide more details about the expected scope and acceptance criteria.".to_string()
}

/// Check whether there are new human comments on the issue since the last
/// PMO comment. Used by the reconcile phase to decide whether to re-triage
/// a `pmo-pending` issue. Comments are in chronological order from the
/// GitLab discussions API. Returns `true` if any comment after the last
/// PMO-authored comment was written by a different author.
fn has_new_comments_since_last_pmo_comment(
    gitlab: &GitLabClient,
    issue: &Issue,
    pmo_agent_id: &str,
) -> bool {
    let Ok(comments) = gitlab.get_issue_comments(issue.iid) else {
        return false;
    };

    // Find the index of the last PMO-authored comment.
    let last_pmo_idx = comments
        .iter()
        .rposition(|c| c.author == pmo_agent_id || c.body.contains("**PMO needs clarification"));

    match last_pmo_idx {
        Some(idx) => {
            // Any non-system comment after the last PMO comment is a "new" human comment.
            comments[idx + 1..]
                .iter()
                .any(|c| !c.author.is_empty() && c.author != pmo_agent_id)
        }
        None => {
            // No PMO comment found — if there are any human comments, they're all "new".
            comments
                .iter()
                .any(|c| !c.author.is_empty() && c.author != pmo_agent_id)
        }
    }
}

fn extract_guidance(agent_output: &AgentHandoff) -> String {
    if let Some(instructions) = &agent_output.instructions {
        let trimmed = instructions.trim();
        if !trimmed.is_empty() {
            return cap_guidance_length(trimmed);
        }
    }
    String::new()
}

/// Cap guidance length — keep it brief and actionable. Truncates at the last
/// sentence boundary within the limit, falling back to an ellipsis suffix.
fn cap_guidance_length(raw: &str) -> String {
    const MAX: usize = 500;
    if raw.len() <= MAX {
        return raw.to_string();
    }
    let truncated: String = raw.chars().take(MAX).collect();
    if let Some(last_period) = truncated.rfind('.') {
        truncated[..=last_period].to_string()
    } else {
        format!("{}...", truncated)
    }
}

/// Format the worker-facing PMO guidance as a GitLab issue comment. The
/// guidance body is posted as plain text (no markers) under a header that
/// gives reviewers context. Both humans and the worker agent read it from
/// the comment stream.
fn format_pmo_guidance_comment(guidance: &str) -> String {
    let trimmed = guidance.trim();
    format!("**PMO guidance for the worker agent:**\n\n{trimmed}")
}

fn pmo_truncate_utf8_by_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Parse a JSON value as an issue IID, accepting numbers, numeric strings,
/// and strings with a leading `#` (e.g. `727`, `"727"`, `"#727"`). Returns
/// `None` for zero or non-numeric values.
fn parse_iid_value(val: &Value) -> Option<u64> {
    let n = val.as_u64().or_else(|| {
        let s = val.as_str()?;
        let s = s.trim().trim_start_matches('#').trim();
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u64>().ok()
    });
    n.filter(|n| *n > 0)
}

/// Check whether the agent's structured output contains a `plan` entry
/// (i.e. the model called the `plan` structured-output tool).
fn handoff_has_plan_output(handoff: &AgentHandoff) -> bool {
    handoff
        .structured_outputs
        .as_ref()
        .is_some_and(|s| s.get("plan").is_some())
}

/// Get a reference to the `plan` tool's captured JSON from
/// `agent_output.structured_outputs`, or `None` if the tool wasn't called.
fn plan_output(agent_output: &AgentHandoff) -> Option<&serde_json::Value> {
    agent_output
        .structured_outputs
        .as_ref()
        .and_then(|s| s.get("plan"))
}

/// Extract the `plan_text` field from the `plan` tool's structured output.
/// This is the PMO's proposed plan that gets written into the issue description
/// during the needs_clarification refinement loop.
fn extract_plan_text(agent_output: &AgentHandoff) -> Option<String> {
    let plan = plan_output(agent_output)?;
    plan.get("plan_text")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Apply the structured JSON the model emitted via the `plan` structured-output
/// tool. Populates the handoff's structured fields (`decision`, `instructions`,
/// `sub_issues`, `reason`, `question`, `needs_clarification`, `depends_on_issue`)
/// from the JSON. Returns the handoff unchanged when `structured_outputs` is
/// absent or has no `plan` entry.
fn apply_pmo_handoff(mut handoff: AgentHandoff) -> AgentHandoff {
    let Some(outputs) = handoff.structured_outputs.take() else {
        return handoff;
    };
    let Some(plan) = outputs.get("plan").cloned() else {
        // Put it back — not our tool.
        handoff.structured_outputs = Some(outputs);
        return handoff;
    };

    if let Some(d) = plan.get("decision").and_then(Value::as_str) {
        let d = d.trim();
        if !d.is_empty() {
            handoff.decision = Some(d.to_lowercase());
        }
    }

    if let Some(instructions) = plan.get("instructions").and_then(Value::as_str) {
        let t = instructions.trim();
        if !t.is_empty() {
            handoff.instructions = Some(t.to_string());
        }
    }

    if let Some(reason) = plan.get("reason").and_then(Value::as_str) {
        let t = reason.trim();
        if !t.is_empty() {
            handoff.reason = Some(t.to_string());
        }
    }

    if let Some(question) = plan.get("question").and_then(Value::as_str) {
        let t = question.trim();
        if !t.is_empty() {
            handoff.question = Some(t.to_string());
            handoff.needs_clarification = Some(t.to_string());
        }
    }

    // PMO wait_for_dependency: the model may use different field names or
    // provide the IID as a string (e.g. "#727" or "727"). Accept any of a
    // set of plausible keys and parse the leading digits.
    for key in [
        "dependency_issue_iid",
        "dependency_iid",
        "depends_on_issue",
        "blocked_by",
        "dependency",
    ] {
        if let Some(val) = plan.get(key)
            && let Some(n) = parse_iid_value(val)
        {
            handoff.depends_on_issue = Some(n);
            break;
        }
    }

    if let Some(sub_issues) = plan.get("sub_issues").and_then(Value::as_array) {
        let parsed: Vec<HandoffSubIssue> = sub_issues
            .iter()
            .filter_map(|s| {
                let title = s.get("title").and_then(Value::as_str)?.trim().to_string();
                if title.is_empty() {
                    return None;
                }
                let description = s
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if description.is_empty() {
                    return None;
                }
                let priority = s
                    .get("priority")
                    .and_then(Value::as_u64)
                    .filter(|p| (1..=3).contains(p))
                    .map(|p| p as u8);
                // depends_on is a 1-based index into the sub_issues array
                // (matching how the model references "sub-issue 1"). 0/absent
                // means no dependency.
                let depends_on = s
                    .get("depends_on")
                    .and_then(Value::as_u64)
                    .map(|d| d as usize)
                    .unwrap_or(0);
                Some(HandoffSubIssue {
                    title,
                    description,
                    priority,
                    depends_on,
                })
            })
            .collect();
        if !parsed.is_empty() {
            handoff.sub_issues = parsed;
        }
    }

    info!(
        "PMO: applied structured plan output (decision={:?}, sub_issues={})",
        handoff.decision.as_deref().unwrap_or(""),
        handoff.sub_issues.len()
    );

    handoff
}

fn save_pending_split(path: &str, pending: &PendingSplit) -> Result<()> {
    let json = serde_json::to_string_pretty(pending)?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).context("Failed to create .potlatch-context for pending split")?;
    }
    fs::write(path, json).context("Failed to save pending split file")?;
    info!(
        "PMO: Saved pending split for issue #{} with {} sub-issues to {}",
        pending.parent_issue_iid,
        pending.sub_issues.len(),
        path
    );
    Ok(())
}

fn load_pending_split(path: &str) -> Result<Option<PendingSplit>> {
    if !std::path::Path::new(path).exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(path).context("Failed to read pending split file")?;
    let pending: PendingSplit =
        serde_json::from_str(&content).context("Failed to parse pending split file")?;

    Ok(Some(pending))
}

fn delete_pending_split(path: &str) -> Result<()> {
    if std::path::Path::new(path).exists() {
        fs::remove_file(path).context("Failed to delete pending split file")?;
        info!("PMO: Deleted pending split file {}", path);
    }
    Ok(())
}

fn try_resume_pmo_state(
    state: &AgentState,
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) -> Option<u64> {
    let path = state.state_path();
    let content = fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&content).ok()?;
    let issue_iid = v.get("claimed_issue_iid")?.as_u64()?;

    let claim_label = state.claim_label();
    match gitlab.get_issue(issue_iid) {
        Ok(issue) => {
            if issue.state != "opened" {
                info!(
                    "{}: Previously claimed issue #{} is {}, discarding state",
                    &state.agent_id, issue_iid, issue.state
                );

                state.clear_state();

                return None;
            }
            if !issue.labels.contains(&claim_label) {
                info!(
                    "{}: Claim label missing from issue #{}, discarding state",
                    &state.agent_id, issue_iid
                );

                state.clear_state();

                return None;
            }
            if !issue_in_scope(&issue, scope_label) {
                info!(
                    "{}: Resumed issue #{} outside scope label {:?}, discarding state",
                    &state.agent_id, issue_iid, scope_label
                );

                state.clear_state();
                return None;
            }

            info!("{}: Resumed claim on issue #{}", &state.agent_id, issue_iid);

            Some(issue_iid)
        }
        Err(e) => {
            warn!(
                "{}: Failed to verify issue #{}: {}, discarding state",
                &state.agent_id, issue_iid, e
            );

            state.clear_state();

            None
        }
    }
}

fn resume_split(
    pending_file: &str,
    gitlab: &GitLabClient,
    pending: &PendingSplit,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut created_issue_ids = pending.created_issue_ids.clone();
    let total_sub_issues = pending.sub_issues.len();
    let already_created = created_issue_ids.len();

    // Create remaining sub-issues
    for (index, sub_issue) in pending.sub_issues.iter().enumerate() {
        // Skip already created sub-issues
        if index < already_created {
            debug!(
                "PMO: Skipping already created sub-issue {}/{}",
                index + 1,
                total_sub_issues
            );
            continue;
        }

        let sub_issue_title = if sub_issue.title.is_empty() {
            &pending.parent_issue_title
        } else {
            &sub_issue.title
        };

        let resolved_description = sub_issue.description.clone();

        match gitlab.create_issue(sub_issue_title, &resolved_description) {
            Ok(sub_issue_iid) => {
                info!(
                    "PMO: Created sub-issue #{}: {}",
                    sub_issue_iid, sub_issue_title
                );

                let p = sub_issue.priority.unwrap_or(pending.parent_priority);
                if let Err(e) = gitlab.add_issue_label(sub_issue_iid, &gitlab::priority_label(p)) {
                    warn!(
                        "PMO: Failed to set priority label on #{}: {}",
                        sub_issue_iid, e
                    );
                }

                if let Some(lbl) = scope_label
                    && let Err(e) = gitlab.add_issue_label(sub_issue_iid, lbl)
                {
                    warn!(
                        "PMO: Failed to add scope label {:?} on #{}: {}",
                        lbl, sub_issue_iid, e
                    );
                }

                // If this sub-issue declares a dependency, add a temporary
                // `do-not-implement` label so the worker skips it immediately.
                // The label is swapped for the real `waiting-on-issue:#N` label
                // in the post-loop pass once all IIDs are known. This closes
                // the race: the issue is blocked from the moment it's created.
                if sub_issue.depends_on > 0
                    && let Err(e) = gitlab.add_issue_label(sub_issue_iid, labels::DO_NOT_IMPLEMENT)
                {
                    warn!(
                        "PMO: Failed to add temporary do-not-implement label to sub-issue #{sub_issue_iid}: {e}"
                    );
                }

                created_issue_ids.push(sub_issue_iid);

                let updated_pending = PendingSplit {
                    parent_issue_iid: pending.parent_issue_iid,
                    parent_issue_title: pending.parent_issue_title.clone(),
                    parent_priority: pending.parent_priority,
                    sub_issues: pending.sub_issues.clone(),
                    created_issue_ids: created_issue_ids.clone(),
                };
                save_pending_split(pending_file, &updated_pending)?;
            }
            Err(e) => {
                error!(
                    "PMO: Failed to create sub-issue {}/{}: {}",
                    index + 1,
                    total_sub_issues,
                    e
                );
                return Err(e);
            }
        }
    }

    // Post-loop pass: replace the temporary `do-not-implement` label with the
    // real `waiting-on-issue:#N` label now that all sub-issue IIDs are known.
    // Both backward and forward references are handled — the temporary label
    // blocked the worker throughout the creation loop, so there's no race.
    for (index, sub_issue) in pending.sub_issues.iter().enumerate() {
        if sub_issue.depends_on == 0 {
            continue;
        }
        let dep_idx = sub_issue.depends_on - 1;
        let Some(&dep_iid) = created_issue_ids.get(dep_idx) else {
            warn!(
                "PMO: sub-issue {} depends on sub-issue {} but the dependency was not created, leaving do-not-implement label",
                index + 1,
                sub_issue.depends_on
            );
            continue;
        };
        let Some(&dependent_iid) = created_issue_ids.get(index) else {
            continue;
        };
        // Swap: remove the temporary hold, add the real dependency label.
        let _ = gitlab.remove_issue_label(dependent_iid, labels::DO_NOT_IMPLEMENT);
        let label = format!("waiting-on-issue:#{dep_iid}");
        if let Err(e) = gitlab.add_issue_label(dependent_iid, &label) {
            warn!("PMO: Failed to add dependency label {label} to sub-issue #{dependent_iid}: {e}");
        } else {
            info!("PMO: Sub-issue #{dependent_iid} depends on #{dep_iid}, labeled {label}");
        }
    }

    // All sub-issues created successfully, add comment to parent issue
    if !created_issue_ids.is_empty() {
        let sub_issue_links = created_issue_ids
            .iter()
            .map(|iid| format!("- #{}", iid))
            .collect::<Vec<_>>()
            .join("\n");

        gitlab.add_issue_comment(
            pending.parent_issue_iid,
            &format!(
                "This issue has been split into {} smaller sub-issues by the PMO Agent:\n\n{}\n\n\
                 Each sub-issue is designed to stay around ~500 lines of non-test code (~1500 total including tests). Auto-generated code is excluded from these limits.",
                created_issue_ids.len(),
                sub_issue_links
            ),
        )?;

        // Remove action-required and mark as processed
        gitlab.remove_issue_label(pending.parent_issue_iid, ACTION_REQUIRED_LABEL)?;
        gitlab.add_issue_label(pending.parent_issue_iid, PMO_PROCESSED_LABEL)?;

        info!(
            "PMO: Successfully completed split of issue #{} into {} sub-issues",
            pending.parent_issue_iid,
            created_issue_ids.len()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_split_prompt_documents_plan_tool_only() {
        let state = AgentState {
            sessions_dir: "/tmp".into(),
            working_dir: "/tmp".into(),
            agent_id: "pmo-test".into(),
            project_name: "test-proj".into(),
        };
        let issue = Issue {
            iid: 42,
            title: "Worker could not complete".into(),
            description: "Details".into(),
            labels: vec![],
            state: "opened".into(),
            created_at: None,
            updated_at: None,
        };
        let prompt = build_split_prompt(&state, &issue, "/abs/pmo-issue-42.md", 2).unwrap();
        // The plan tool is the ONLY output channel.
        assert!(prompt.contains("call the `plan` tool"));
        assert!(prompt.contains("`decision`"));
        assert!(prompt.contains("guide_worker"));
        assert!(prompt.contains("split"));
        assert!(prompt.contains("already_done"));
        assert!(prompt.contains("needs_clarification"));
        // No text-marker fallback.
        assert!(!prompt.contains("SUB_ISSUE_N:"));
        assert!(!prompt.contains("GUIDE_WORKER\n"));
        assert!(!prompt.contains("ALREADY_DONE\n"));
        assert!(!prompt.contains("NEEDS_CLARIFICATION\n"));
        assert!(!prompt.contains("PUBLIC_COMMENT_BEGIN"));
        assert!(!prompt.contains("text-marker"));
        // Reasoning guidance is intact.
        assert!(prompt.contains("TASK CONTAINER TEST"));
        assert!(prompt.contains("DUPLICATE / OVERLAP RULES"));
    }

    #[test]
    fn apply_pmo_handoff_populates_structured_fields_for_split() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "split",
                "sub_issues": [
                    {"title": "First", "description": "Do the first thing", "priority": 1},
                    {"title": "Second", "description": "Do the second thing", "priority": 2, "depends_on": 1}
                ]
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("split"));
        assert_eq!(applied.sub_issues.len(), 2);
        assert_eq!(applied.sub_issues[0].title, "First");
        assert_eq!(applied.sub_issues[0].priority, Some(1));
        assert_eq!(applied.sub_issues[0].depends_on, 0);
        assert_eq!(applied.sub_issues[1].title, "Second");
        assert_eq!(applied.sub_issues[1].priority, Some(2));
        assert_eq!(applied.sub_issues[1].depends_on, 1);
        assert!(applied.structured_outputs.is_none());
    }

    #[test]
    fn apply_pmo_handoff_populates_instructions_for_guide_worker() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "guide_worker",
                "instructions": "Use flag --foo instead of --bar."
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("guide_worker"));
        assert_eq!(
            applied.instructions.as_deref(),
            Some("Use flag --foo instead of --bar.")
        );
        assert!(pmo_guides_worker(&applied));
    }

    #[test]
    fn apply_pmo_handoff_populates_reason_for_already_done() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "already_done",
                "reason": "The feature exists in src/lib.rs."
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("already_done"));
        assert!(applied.reason.as_deref().unwrap().contains("src/lib.rs"));
        assert!(is_pmo_already_done_response(&applied));
    }

    #[test]
    fn apply_pmo_handoff_populates_question_for_needs_clarification() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "needs_clarification",
                "question": "Which modules should be covered?"
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("needs_clarification"));
        assert!(applied.question.as_deref().unwrap().contains("modules"));
        assert!(pmo_needs_clarification(&applied));
    }

    #[test]
    fn apply_pmo_handoff_populates_depends_on_issue_for_wait_for_dependency() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "wait_for_dependency",
                "dependency_issue_iid": 47
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("wait_for_dependency"));
        assert_eq!(applied.depends_on_issue, Some(47));
        assert!(pmo_wait_for_dependency(&applied));
    }

    #[test]
    fn apply_pmo_handoff_ignores_zero_dependency_issue_iid() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "wait_for_dependency",
                "dependency_issue_iid": 0
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision.as_deref(), Some("wait_for_dependency"));
        assert_eq!(applied.depends_on_issue, None);
    }

    #[test]
    fn apply_pmo_handoff_accepts_string_dependency_issue_iid() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "wait_for_dependency",
                "dependency_issue_iid": "#727"
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.depends_on_issue, Some(727));
    }

    #[test]
    fn apply_pmo_handoff_accepts_alternative_field_names() {
        for key in [
            "dependency_iid",
            "depends_on_issue",
            "blocked_by",
            "dependency",
        ] {
            let handoff = AgentHandoff {
                structured_outputs: Some(serde_json::json!({"plan": {
                    "decision": "wait_for_dependency",
                    key: 727
                }})),
                ..Default::default()
            };
            let applied = apply_pmo_handoff(handoff);
            assert_eq!(
                applied.depends_on_issue,
                Some(727),
                "failed for alternative field name `{key}`"
            );
        }
    }

    #[test]
    fn apply_pmo_handoff_none_leaves_handoff_unchanged() {
        let handoff = AgentHandoff {
            response: "some text".into(),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.decision, None);
        assert_eq!(applied.instructions, None);
        assert_eq!(applied.response, "some text");
    }

    #[test]
    fn apply_pmo_handoff_skips_empty_sub_issue_titles() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "split",
                "sub_issues": [
                    {"title": "", "description": "no title"},
                    {"title": "Valid", "description": "has title"}
                ]
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.sub_issues.len(), 1);
        assert_eq!(applied.sub_issues[0].title, "Valid");
    }

    #[test]
    fn apply_pmo_handoff_skips_empty_sub_issue_descriptions() {
        let handoff = AgentHandoff {
            structured_outputs: Some(serde_json::json!({"plan": {
                "decision": "split",
                "sub_issues": [
                    {"title": "No desc", "description": ""},
                    {"title": "Valid", "description": "has desc"}
                ]
            }})),
            ..Default::default()
        };
        let applied = apply_pmo_handoff(handoff);
        assert_eq!(applied.sub_issues.len(), 1);
        assert_eq!(applied.sub_issues[0].title, "Valid");
    }

    #[test]
    fn test_should_process_issue() {
        let mut issue = Issue {
            iid: 1,
            title: "Test issue".to_string(),
            description: "Test".to_string(),
            labels: vec![ACTION_REQUIRED_LABEL.to_string()],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };
        assert!(should_process_issue(&issue, None));

        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            PMO_PROCESSED_LABEL.to_string(),
        ];
        assert!(!should_process_issue(&issue, None));

        issue.labels = vec![];
        assert!(!should_process_issue(&issue, None));

        issue.title = "[Draft] Test".to_string();
        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(!should_process_issue(&issue, None));

        issue.title = "Normal issue".to_string();
        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            labels::PMO_PENDING.to_string(),
        ];
        assert!(!should_process_issue(&issue, None));

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string(), "potlatch".to_string()];
        assert!(!should_process_issue(&issue, Some("other-scope")));
        assert!(should_process_issue(&issue, Some("potlatch")));
    }

    #[test]
    fn extract_guidance_returns_structured_instructions() {
        let output = AgentHandoff {
            instructions: Some("Use the existing config loader.".into()),
            ..Default::default()
        };
        assert_eq!(extract_guidance(&output), "Use the existing config loader.");
    }

    #[test]
    fn extract_guidance_returns_empty_when_no_instructions() {
        let output = AgentHandoff {
            response: "some response text".into(),
            ..Default::default()
        };
        assert!(extract_guidance(&output).is_empty());
    }

    #[test]
    fn extract_guidance_truncates_long_instructions() {
        let long = "Do this. ".repeat(80);
        let output = AgentHandoff {
            instructions: Some(long),
            ..Default::default()
        };
        let extracted = extract_guidance(&output);
        assert!(extracted.len() <= 502);
        assert!(extracted.starts_with("Do this."));
    }

    #[test]
    fn format_pmo_guidance_comment_includes_header_and_body() {
        let comment = format_pmo_guidance_comment("Use --foo instead of --bar.");
        assert!(comment.contains("**PMO guidance for the worker agent:**"));
        assert!(comment.contains("Use --foo instead of --bar."));
        // No internal markers — the comment is plain human-facing text.
        assert!(!comment.contains("PMO_GUIDANCE_BEGIN"));
        assert!(!comment.contains("PMO_GUIDANCE_END"));
    }

    #[test]
    fn format_pmo_guidance_comment_preserves_multiline_body() {
        let body = "Step one: do X.\nStep two: do Y.";
        let comment = format_pmo_guidance_comment(body);
        assert!(comment.contains(body));
    }

    #[test]
    fn format_pmo_guidance_comment_includes_header_even_when_body_is_whitespace() {
        let comment = format_pmo_guidance_comment("   ");
        assert!(comment.contains("**PMO guidance for the worker agent:**"));
        assert!(!comment.contains("PMO_GUIDANCE_BEGIN"));
    }

    #[test]
    fn decision_predicates_use_structured_fields_only() {
        // needs_clarification
        let nc = AgentHandoff {
            decision: Some("needs_clarification".into()),
            ..Default::default()
        };
        assert!(pmo_needs_clarification(&nc));
        assert!(!pmo_guides_worker(&nc));
        assert!(!is_pmo_already_done_response(&nc));

        // guide_worker
        let gw = AgentHandoff {
            decision: Some("guide_worker".into()),
            ..Default::default()
        };
        assert!(pmo_guides_worker(&gw));
        assert!(!pmo_needs_clarification(&gw));

        // already_done
        let ad = AgentHandoff {
            decision: Some("already_done".into()),
            ..Default::default()
        };
        assert!(is_pmo_already_done_response(&ad));
        assert!(!pmo_guides_worker(&ad));

        // wait_for_dependency
        let wd = AgentHandoff {
            decision: Some("wait_for_dependency".into()),
            depends_on_issue: Some(42),
            ..Default::default()
        };
        assert!(pmo_wait_for_dependency(&wd));
        assert!(!pmo_guides_worker(&wd));
        assert!(!is_pmo_already_done_response(&wd));
        assert!(!pmo_needs_clarification(&wd));

        // depends_on_issue alone also triggers the predicate (structured field)
        let wd_field = AgentHandoff {
            depends_on_issue: Some(7),
            ..Default::default()
        };
        assert!(pmo_wait_for_dependency(&wd_field));

        // No decision
        let none = AgentHandoff::default();
        assert!(!pmo_needs_clarification(&none));
        assert!(!pmo_guides_worker(&none));
        assert!(!is_pmo_already_done_response(&none));
        assert!(!pmo_wait_for_dependency(&none));
    }

    #[test]
    fn decision_predicates_ignore_text_markers() {
        // Text markers in the response must NOT trigger decisions — only
        // structured fields count.
        let output = AgentHandoff {
            response: "GUIDE_WORKER\nINSTRUCTIONS:\ndo something".into(),
            ..Default::default()
        };
        assert!(!pmo_guides_worker(&output));
        assert!(!pmo_needs_clarification(&output));

        let output = AgentHandoff {
            response: "ALREADY_DONE\nREASON: done".into(),
            ..Default::default()
        };
        assert!(!is_pmo_already_done_response(&output));
    }

    #[test]
    fn extract_already_done_reason_uses_structured_field() {
        let output = AgentHandoff {
            reason: Some("Implemented in module X.".into()),
            ..Default::default()
        };
        assert!(extract_already_done_reason(&output).contains("module X"));
    }

    #[test]
    fn extract_already_done_reason_defaults_when_no_field() {
        let output = AgentHandoff::default();
        let reason = extract_already_done_reason(&output);
        assert!(!reason.is_empty());
    }

    #[test]
    fn extract_clarification_question_uses_structured_field() {
        let output = AgentHandoff {
            question: Some("Which modules?".into()),
            ..Default::default()
        };
        assert_eq!(extract_clarification_question(&output), "Which modules?");
    }

    #[test]
    fn extract_clarification_question_defaults_when_no_field() {
        let output = AgentHandoff::default();
        let q = extract_clarification_question(&output);
        assert!(!q.is_empty());
    }

    #[test]
    fn extract_plan_text_reads_from_structured_output() {
        let output = AgentHandoff {
            structured_outputs: Some(serde_json::json!({
                "plan": {
                    "decision": "needs_clarification",
                    "plan_text": "## Plan Draft\n\n1. Implement X\n2. Test X\n\nOpen question: which config?"
                }
            })),
            ..Default::default()
        };
        let plan = extract_plan_text(&output).unwrap();
        assert!(plan.contains("## Plan Draft"));
        assert!(plan.contains("Implement X"));
    }

    #[test]
    fn extract_plan_text_returns_none_when_absent() {
        let output = AgentHandoff::default();
        assert!(extract_plan_text(&output).is_none());
    }

    #[test]
    fn extract_plan_text_returns_none_when_empty() {
        let output = AgentHandoff {
            structured_outputs: Some(serde_json::json!({
                "plan": {
                    "decision": "needs_clarification",
                    "plan_text": "  "
                }
            })),
            ..Default::default()
        };
        assert!(extract_plan_text(&output).is_none());
    }

    #[test]
    fn test_priority_from_labels() {
        assert_eq!(
            gitlab::priority_from_labels(&["priority::1".to_string()]),
            1
        );
        assert_eq!(
            gitlab::priority_from_labels(&["priority::2".to_string(), "in-progress".to_string()]),
            2
        );
        assert_eq!(
            gitlab::priority_from_labels(&["in-progress".to_string()]),
            3
        );
    }

    #[test]
    fn truncate_diff_for_pmo_keeps_short_diffs_unchanged() {
        let diff = "diff --git a/file.go b/file.go\n+hello\n";
        assert_eq!(truncate_diff_for_pmo(diff), diff);
    }

    #[test]
    fn truncate_diff_for_pmo_truncates_long_diffs_with_marker() {
        let diff = "x".repeat(10_000);
        let result = truncate_diff_for_pmo(&diff);
        assert!(result.contains("[...diff truncated at"));
        assert!(result.len() < diff.len());
        // Should still start with the original content.
        assert!(result.starts_with('x'));
    }

    #[test]
    fn truncate_diff_for_pmo_respects_char_boundary() {
        // Multi-byte UTF-8 characters must not be split.
        let diff = format!("{}\n", "α".repeat(4_000)); // each α is 2 bytes
        let result = truncate_diff_for_pmo(&diff);
        // Result is valid UTF-8 (no panic), and either truncated or full.
        assert!(result.contains('α'));
    }
}
