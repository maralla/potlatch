use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{
    claim, extract_public_comment_block, issue_in_scope, strip_public_comment_blocks,
    write_task_context_file,
};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{GitLabClient, Issue};
use crate::agents::settings;
use crate::agents::workspace::{
    ensure_agent_repo, extract_project_name, require_gitlab_repo, sessions_dir, work_dir,
};
use crate::core::agent::{AgentHandoff, InvokeOptions};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};

const WORKING_ON_LABEL: &str = "in-progress";
/// Root-level file updated by Potlatch after each successful worker run (impl or MR feedback).
const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";
const PMO_PENDING_LABEL: &str = "pmo-pending";
const NEED_AI_WORKER_LABEL: &str = "need-ai-worker";
/// Human/workflow pause: worker skips the issue (no close) and releases its hold until removed.
const WORKER_PENDING_LABEL: &str = "pending";
/// Reviewer-only workflow: worker skips and stops tracking while reviewer may still process the MR.
const WORKER_REVIEW_ONLY_LABEL: &str = "review-only";
/// ACP runtime message when `cancel_check` returns true.
const WORKER_AGENT_CANCELLED_MSG: &str = "Agent cancelled by external condition";

/// The worker's structured-output tool definition. Passed to the harness via
/// `session/new` so the harness registers a generic `StructuredOutputTool`
/// named `handoff`. The model calls it with structured JSON instead of
/// emitting text markers.
fn handoff_tool_definition() -> serde_json::Value {
    serde_json::json!({
        "name": "handoff",
        "description": "Emit your implementation output as structured JSON. This is the primary output channel — Potlatch reads the tool's JSON, not text markers in your response. Call this once when you're done (or when you need to signal a dependency/split/clarification). All fields are optional — include only the ones relevant to your outcome.",
        "parameters": {
            "type": "object",
            "properties": {
                "mr_title": {
                    "type": "string",
                    "description": "Short MR title (max 8-10 words). Focus on WHAT, not HOW. No markdown."
                },
                "mr_description": {
                    "type": "string",
                    "description": "Full MR description in markdown with ## Goal, ## Implementation, ## Testing sections."
                },
                "changes_summary": {
                    "type": "string",
                    "description": "A concise sentence summarizing the substance of changes made (for commit messages)."
                },
                "depends_on_issue": {
                    "type": "integer",
                    "description": "IID of a dependency issue that must close before this work can proceed. Set when the issue is hard-blocked on another open issue."
                },
                "needs_split": {
                    "type": "string",
                    "description": "Reason the issue needs splitting into smaller issues."
                },
                "needs_clarification": {
                    "type": "string",
                    "description": "What information is needed from a human to proceed."
                },
                "cannot_implement": {
                    "type": "boolean",
                    "description": "Set to true when the issue cannot be implemented (too broad, unclear, blocked)."
                },
                "cannot_resolve": {
                    "type": "boolean",
                    "description": "Set to true when reviewer feedback cannot be resolved autonomously."
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for cannot_implement, cannot_resolve, or no-code-changes."
                },
                "public_comment": {
                    "type": "string",
                    "description": "Human-facing GitLab comment text (separate from MR description)."
                },
                "mark_discussions_resolved": {
                    "type": "boolean",
                    "description": "Whether to mark open review discussions as resolved after your reply."
                },
                "post_plain_comment": {
                    "type": "boolean",
                    "description": "Whether to post a new plain MR comment (non-resolvable)."
                }
            }
        }
    })
}

#[derive(Debug, Clone)]
struct WorkerConfig {
    poll_interval_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct WorkerAgentSettings {
    #[serde(default = "default_worker_poll_interval")]
    poll_interval_secs: u64,
}

fn default_worker_poll_interval() -> u64 {
    60
}

impl WorkerAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        raw.clone()
            .try_into()
            .context("worker agent settings from config")
    }
}

/// The single issue a worker is pinned to for its full lifecycle.
#[derive(Clone)]
struct ActiveIssue {
    issue_iid: u64,
    mr_iid: Option<u64>,
    branch_name: Option<String>,
    mr_created: bool,
}

struct AgentState {
    project_name: String,
    agent_id: String,
    working_dir: String,
    sessions_dir: String,

    git_repo: GitRepo,
    glab: GitLabClient,
}

impl AgentState {
    fn session_file_path(&self, issue_iid: u64) -> std::path::PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_issue_{}.json", &self.agent_id, issue_iid))
    }

    fn load_session(&self, issue_iid: u64) -> Option<SessionFile> {
        let path = self.session_file_path(issue_iid);
        let content = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn load_implementation_summary(&self, issue_number: u64) -> String {
        if let Some(session) = self.load_session(issue_number)
            && let Some(summary) = session.implementation_summary
        {
            return summary;
        }
        "No previous implementation summary available.".to_string()
    }

    fn cleanup_session(&self, issue_iid: u64) {
        let path = self.session_file_path(issue_iid);
        if fs::remove_file(&path).is_ok() {
            info!(
                "{}: Cleaned up session file for issue #{}",
                &self.agent_id, issue_iid
            );
        }
    }

    fn save_session(&self, issue_iid: u64, mr_iid: u64) -> Result<()> {
        let session = SessionFile {
            issue_iid,
            mr_iid,
            agent_id: Some(self.agent_id.clone()),
            implementation_summary: None,
        };

        let path = self.session_file_path(issue_iid);
        let json = serde_json::to_string_pretty(&session)?;
        fs::write(&path, json).context("Failed to write session file")?;
        Ok(())
    }

    fn save_session_with_summary(&self, issue_iid: u64, mr_iid: u64, summary: &str) -> Result<()> {
        let session = SessionFile {
            issue_iid,
            mr_iid,
            agent_id: Some(self.agent_id.clone()),
            implementation_summary: Some(summary.to_string()),
        };

        let path = self.session_file_path(issue_iid);
        let json = serde_json::to_string_pretty(&session)?;
        fs::write(&path, json).context("Failed to write session file")?;
        Ok(())
    }

    fn release_worker_hold_pending_gitlab_only(&self, issue_iid: u64) {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (no close)",
            &self.agent_id, issue_iid, WORKER_PENDING_LABEL
        );

        let _ = claim::release_claim(&self.glab, issue_iid, &self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
        self.cleanup_session(issue_iid);
    }

    fn release_worker_hold_review_only(&self, issue_iid: u64) {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (review only)",
            &self.agent_id, issue_iid, WORKER_REVIEW_ONLY_LABEL
        );

        self.clear_resumed_issue_state(issue_iid);
    }

    fn clear_resumed_issue_state(&self, issue_iid: u64) {
        let default_branch = self
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());

        let branch = format!("issue-{}", issue_iid);

        let _ = self.git_repo.reset_hard();
        let _ = self.git_repo.checkout_remote_branch(&default_branch);
        let _ = self.git_repo.delete_local_branch(&branch);

        let _ = claim::release_claim(&self.glab, issue_iid, &self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
    }

    fn abandon_closed_issue(&self, issue_iid: u64, mr_iid: Option<u64>) {
        info!(
            "{}: Issue #{} was closed externally, abandoning work",
            &self.agent_id, issue_iid
        );

        if let Some(mr) = mr_iid {
            let _ = self
                .glab
                .add_mr_comment(mr, "Closing this MR — the linked issue has been closed.");
            let _ = self.glab.close_mr(mr);
        }

        let branch = format!("issue-{}", issue_iid);
        let default_branch = self
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());

        let _ = self.git_repo.reset_hard();
        let _ = self.git_repo.checkout_remote_branch(&default_branch);
        let _ = self.git_repo.delete_local_branch(&branch);
        self.git_repo.delete_remote_branch_best_effort(&branch);
        let _ = claim::release_claim(&self.glab, issue_iid, &self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
    }

    fn cleanup_on_shutdown(&self, active: &Option<ActiveIssue>) {
        let Some(a) = active else { return };

        // Always keep the claim and session so this worker resumes on restart.
        // The session file (even with mr_iid=0) tells try_resume_session that
        // this worker owns the issue.
        info!(
            "{}: Preserving claim on issue #{} for restart (MR: {})",
            &self.agent_id,
            a.issue_iid,
            a.mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
        );

        let mr_iid = a.mr_iid.unwrap_or(0);
        let _ = self.save_session(a.issue_iid, mr_iid);
        let _ = self.git_repo.reset_hard();
    }
}

pub(crate) struct WorkerAgent {
    state: AgentState,
    model: AgentModel,
    config: WorkerConfig,
    scope_label: String,
    active: Option<ActiveIssue>,
}

impl CoreAgent for WorkerAgent {
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        "worker"
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
                if let Err(e) = worker_cycle(&self.state, model, &mut self.active, &shutdown, scope)
                    && !shutdown.load(Ordering::SeqCst)
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
            .agent("worker")
            .context("[agent.worker] section required")?;
        let settings = WorkerAgentSettings::from_raw(&section.raw)?;
        let project_name = extract_project_name(&gitlab_repo)?;
        let agent_id = format!("worker-{}", ctx.instance_id);
        ensure_agent_repo(
            &ctx.workflow.base_dir,
            &gitlab_repo,
            &project_name,
            &agent_id,
        )?;
        let working_dir = work_dir(&ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&ctx.workflow.base_dir, &project_name);
        let git_repo = GitRepo::new(working_dir.clone());
        let glab = GitLabClient::new(working_dir.clone(), &gitlab_repo)?;
        let state = AgentState {
            project_name,
            agent_id: agent_id.clone(),
            working_dir: working_dir.clone(),
            sessions_dir,
            git_repo,
            glab,
        };
        let config = WorkerConfig {
            poll_interval_secs: settings.poll_interval_secs,
        };
        let model = AgentModel::connect(
            &ctx,
            "worker",
            state.working_dir.clone(),
            ModelPreferences {
                structured_output_tools: Some(vec![handoff_tool_definition()]),
                ..ModelPreferences::default()
            },
        )?;
        let agent_settings = settings::settings();
        let scope = agent_settings.scope_label_filter();
        let mut active =
            try_resume_session(&state, scope).or_else(|| find_claimed_issue(&state, scope));
        active = clear_resumed_issue_if_ignored(&state, active, scope);
        if let Some(ref a) = active {
            if let Some(mr_iid) = a.mr_iid {
                info!(
                    "{}: Resumed issue #{} with MR !{}",
                    &state.agent_id, a.issue_iid, mr_iid
                );
            } else {
                info!(
                    "{}: Resumed issue #{} (no MR yet)",
                    &state.agent_id, a.issue_iid
                );
            }
        }
        Ok(Self {
            state,
            model,
            config,
            scope_label: agent_settings.scope_label.clone(),
            active,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.state.agent_id);
        self.state.cleanup_on_shutdown(&self.active);
        info!("{}: Stopped", self.state.agent_id);
    }
}

fn clear_resumed_issue_if_ignored(
    state: &AgentState,
    active: Option<ActiveIssue>,
    scope_label: Option<&str>,
) -> Option<ActiveIssue> {
    let active_issue = active?;

    let issue = match state.glab.get_issue(active_issue.issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            warn!(
                "{}: Failed to verify resumed issue #{}: {}, dropping resume state",
                &state.agent_id, active_issue.issue_iid, e
            );

            state.clear_resumed_issue_state(active_issue.issue_iid);

            return None;
        }
    };

    if !issue_in_scope(&issue, scope_label) {
        info!(
            "{}: Resumed issue #{} is outside scope label {:?}, dropping resume state",
            &state.agent_id, issue.iid, scope_label
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    if issue.state != "opened" {
        info!(
            "{}: Resumed issue #{} is {}, dropping resume state",
            &state.agent_id, issue.iid, issue.state
        );
        state.abandon_closed_issue(issue.iid, active_issue.mr_iid);
        return None;
    }

    if issue_has_worker_pending_label(&issue.labels) {
        info!(
            "{}: Resumed issue #{} has `{}` label; releasing worker hold",
            &state.agent_id, issue.iid, WORKER_PENDING_LABEL
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    if issue_has_worker_review_only_label(&issue.labels) {
        state.release_worker_hold_review_only(issue.iid);
        return None;
    }

    if has_worker_resume_abandon_label(&issue.labels) {
        info!(
            "{}: Resumed issue #{} has blocking labels {:?}, abandoning resume state",
            &state.agent_id, issue.iid, issue.labels
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    Some(active_issue)
}

/// Returns `true` when the worker should keep watching the active MR issue.
/// On release, clears `active` and returns `false` so the cycle can poll for new work.
fn retain_active_mr_issue(
    state: &AgentState,
    active_issue: &ActiveIssue,
    active: &mut Option<ActiveIssue>,
    scope_label: Option<&str>,
) -> bool {
    let Some(mr_iid) = active_issue.mr_iid else {
        return true;
    };

    let issue = match state.glab.get_issue(active_issue.issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            warn!(
                "{}: Failed to verify active issue #{}: {}, releasing worker state",
                &state.agent_id, active_issue.issue_iid, e
            );
            state.clear_resumed_issue_state(active_issue.issue_iid);
            *active = None;
            return false;
        }
    };

    if issue.state != "opened" {
        state.abandon_closed_issue(active_issue.issue_iid, Some(mr_iid));
        *active = None;
        return false;
    }

    if !issue_in_scope(&issue, scope_label) {
        info!(
            "{}: Issue #{} left scope label {:?}, releasing worker state",
            &state.agent_id, active_issue.issue_iid, scope_label
        );
        state.clear_resumed_issue_state(active_issue.issue_iid);
        *active = None;
        return false;
    }

    if issue_has_worker_pending_label(&issue.labels) {
        info!(
            "{}: Issue #{} has `{}` — stopping MR watch (issue stays open)",
            &state.agent_id, active_issue.issue_iid, WORKER_PENDING_LABEL
        );
        state.clear_resumed_issue_state(active_issue.issue_iid);
        *active = None;
        return false;
    }

    if issue_has_worker_review_only_label(&issue.labels) {
        state.release_worker_hold_review_only(active_issue.issue_iid);
        *active = None;
        return false;
    }

    true
}

fn worker_cycle(
    state: &AgentState,
    model: &AgentModel,
    active: &mut Option<ActiveIssue>,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<()> {
    // If we have an active issue with an MR, watch the MR unless it should be released.
    if let Some(a) = active.clone()
        && a.mr_iid.is_some()
    {
        let still_tracking = retain_active_mr_issue(state, &a, active, scope_label);
        if still_tracking
            && let Some(a) = &*active
            && let Some(mr_iid) = a.mr_iid
        {
            match state.glab.get_merge_request(mr_iid) {
                Ok(mr) => {
                    if mr.state == "merged" || mr.state == "closed" {
                        info!(
                            "{}: MR !{} is {}, releasing issue #{}",
                            &state.agent_id, mr_iid, mr.state, a.issue_iid
                        );

                        let branch = format!("issue-{}", a.issue_iid);
                        let default_branch = state
                            .git_repo
                            .get_default_branch()
                            .unwrap_or("main".to_string());

                        let _ = state.git_repo.reset_hard();
                        let _ = state.git_repo.checkout_remote_branch(&default_branch);
                        let _ = state.git_repo.delete_local_branch(&branch);

                        if mr.state == "merged" {
                            state.git_repo.delete_remote_branch_best_effort(&branch);
                        }

                        let _ = claim::release_claim(&state.glab, a.issue_iid, &state.agent_id);
                        let _ = state.glab.remove_issue_label(a.issue_iid, WORKING_ON_LABEL);

                        state.cleanup_session(a.issue_iid);

                        if mr.state == "merged" {
                            close_issue_best_effort(&state.glab, a.issue_iid);
                        }

                        *active = None;
                    } else {
                        match handle_mr_comments(state, model, &mr, Some(a.issue_iid), false) {
                            Ok(true) => {
                                info!(
                                    "{}: Issue #{} abandoned, MR !{} closed",
                                    &state.agent_id, a.issue_iid, mr_iid
                                );

                                let _ =
                                    claim::release_claim(&state.glab, a.issue_iid, &state.agent_id);

                                state.cleanup_session(a.issue_iid);
                                *active = None;
                            }
                            Err(e) => {
                                if shutdown.load(Ordering::SeqCst) {
                                    return Ok(());
                                }
                                if handle_worker_issue_processing_cancelled(state, a.issue_iid, &e)
                                {
                                    *active = None;
                                    return Ok(());
                                }

                                error!(
                                    "{}: Failed to handle comments for MR !{}: {}",
                                    &state.agent_id, mr_iid, e
                                );
                            }
                            Ok(false) => {}
                        }

                        if active.is_some() {
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    warn!("{}: Failed to check MR !{}: {}", &state.agent_id, mr_iid, e);
                    if active.is_some() {
                        return Ok(());
                    }
                }
            }
        }
    }

    // If we have an active issue without an MR, we were interrupted before
    // creating the MR. Re-attempt implementation from scratch.
    if let Some(ref a) = *active
        && a.mr_iid.is_none()
        && !a.mr_created
    {
        let issue_iid = a.issue_iid;

        // Check if the issue was closed externally or left the scope label.
        let mut released = false;
        match state.glab.get_issue(issue_iid) {
            Ok(issue) if issue.state != "opened" => {
                state.abandon_closed_issue(issue_iid, None);
                released = true;
            }
            Ok(issue) if !issue_in_scope(&issue, scope_label) => {
                info!(
                    "{}: Active issue #{} left scope label {:?}, releasing",
                    &state.agent_id, issue_iid, scope_label
                );
                state.clear_resumed_issue_state(issue_iid);
                released = true;
            }
            Ok(issue) if issue_has_worker_pending_label(&issue.labels) => {
                info!(
                    "{}: Active issue #{} has `{}` — yielding (issue stays open)",
                    &state.agent_id, issue_iid, WORKER_PENDING_LABEL
                );
                state.clear_resumed_issue_state(issue_iid);
                released = true;
            }
            Ok(issue) if issue_has_worker_review_only_label(&issue.labels) => {
                state.release_worker_hold_review_only(issue_iid);
                released = true;
            }
            Err(e) => {
                warn!(
                    "{}: Failed to verify active issue #{}: {}, releasing",
                    &state.agent_id, issue_iid, e
                );
                state.clear_resumed_issue_state(issue_iid);
                released = true;
            }
            Ok(_) => {}
        }

        if released {
            *active = None;
        } else {
            info!(
                "{}: Active issue #{} has no MR, re-attempting implementation",
                &state.agent_id, issue_iid
            );

            match state.glab.get_issue(issue_iid) {
                Ok(issue) => {
                    let mut current = ActiveIssue {
                        issue_iid,
                        mr_iid: None,
                        branch_name: None,
                        mr_created: false,
                    };

                    match process_issue(state, model, &issue, &mut current, scope_label) {
                        Ok(_) => {
                            if current.mr_created {
                                if should_track_worker_issue(state, issue_iid) {
                                    *active = Some(current);
                                } else {
                                    *active = None;
                                }
                            } else {
                                let _ =
                                    claim::release_claim(&state.glab, issue_iid, &state.agent_id);
                                state.cleanup_session(issue_iid);
                                *active = None;
                            }
                        }
                        Err(e) => {
                            if shutdown.load(Ordering::SeqCst) {
                                *active = Some(current);
                                return Ok(());
                            }
                            if handle_worker_issue_processing_cancelled(state, issue_iid, &e) {
                                *active = None;
                                return Ok(());
                            }
                            error!(
                                "{}: Failed to re-process issue #{}: {}",
                                &state.agent_id, issue_iid, e
                            );

                            let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
                            let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                            state.cleanup_session(issue_iid);
                            *active = None;
                            if let Some(ref branch) = current.branch_name {
                                let default_branch = state
                                    .git_repo
                                    .get_default_branch()
                                    .unwrap_or("main".to_string());
                                let _ = state.git_repo.reset_hard();
                                let _ = state.git_repo.checkout_remote_branch(&default_branch);
                                let _ = state.git_repo.delete_local_branch(branch);
                            }
                        }
                    }

                    return Ok(());
                }
                Err(e) => {
                    warn!(
                        "{}: Failed to fetch issue #{} for re-attempt: {}, releasing",
                        &state.agent_id, issue_iid, e
                    );

                    let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
                    let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                    state.cleanup_session(issue_iid);
                    *active = None;
                }
            }
        }
    }

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // No active issue — try to pick up an orphaned session first
    if active.is_none() {
        *active = try_adopt_orphaned_session(state, shutdown, scope_label);
        if let Some(a) = &*active {
            info!(
                "{}: Adopted orphaned issue #{} with MR !{}",
                &state.agent_id,
                a.issue_iid,
                a.mr_iid.unwrap_or(0)
            );
            return Ok(());
        }
    }

    // No orphaned sessions — poll for new issues
    if try_handle_need_ai_worker_mr(state, model, shutdown, scope_label)? {
        return Ok(());
    }

    // No labeled MR work — poll for new issues
    let issues = state.glab.list_issues()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    for issue in issues {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if should_skip_issue(&issue) {
            continue;
        }

        if !issue_in_scope(&issue, scope_label) {
            continue;
        }

        // Check if this issue is parked waiting on a dependency issue.
        if let Some(dep_issue_iid) = extract_waiting_on_issue_iid(&issue.labels) {
            match state.glab.get_issue(dep_issue_iid) {
                Ok(dep) if dep.state == "closed" => {
                    // Dependency closed — strip the waiting label and proceed to claim.
                    info!(
                        "{}: Issue #{} dependency issue #{} closed, resuming",
                        &state.agent_id, issue.iid, dep_issue_iid
                    );
                    let label = format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{dep_issue_iid}");
                    let _ = state.glab.remove_issue_label(issue.iid, &label);
                }
                Ok(_) => {
                    debug!(
                        "{}: Issue #{} waiting on issue #{} (not yet closed), skipping",
                        &state.agent_id, issue.iid, dep_issue_iid
                    );
                    continue;
                }
                Err(e) => {
                    let err_str = e.to_string();
                    // A 404 means the dependency issue was deleted or never
                    // existed — drop the label and resume.
                    if err_str.contains("404") {
                        info!(
                            "{}: Issue #{} dependency issue #{} not found (deleted or never existed), dropping dependency label and resuming",
                            &state.agent_id, issue.iid, dep_issue_iid
                        );
                        let label = format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{dep_issue_iid}");
                        let _ = state.glab.remove_issue_label(issue.iid, &label);
                    } else {
                        warn!(
                            "{}: Issue #{} waiting on issue #{} — failed to check dependency state: {}, skipping this cycle",
                            &state.agent_id, issue.iid, dep_issue_iid, err_str
                        );
                        continue;
                    }
                }
            }
        }

        if claim::is_claimed(&issue.labels) {
            debug!(
                "{}: Issue #{} already claimed, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if !claim::try_claim_issue(&state.glab, issue.iid, &state.agent_id, shutdown)? {
            info!(
                "{}: Failed to claim issue #{}, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_claim(&state.glab, issue.iid, &state.agent_id);
            return Ok(());
        }

        // Persist a pre-session immediately so that even a SIGKILL leaves
        // a record of which issue this worker owns. mr_iid=0 means no MR yet.
        let _ = state.save_session(issue.iid, 0);

        info!(
            "{}: Implementing issue #{}: {}",
            &state.agent_id, issue.iid, issue.title
        );

        let mut current = ActiveIssue {
            issue_iid: issue.iid,
            mr_iid: None,
            branch_name: None,
            mr_created: false,
        };

        match process_issue(state, model, &issue, &mut current, scope_label) {
            Ok(_) => {
                if current.mr_created {
                    if should_track_worker_issue(state, issue.iid) {
                        *active = Some(current);
                    }
                } else {
                    // Rejected or no MR — release claim
                    let _ = claim::release_claim(&state.glab, issue.iid, &state.agent_id);
                }
            }
            Err(e) => {
                if shutdown.load(Ordering::SeqCst) {
                    // Shutdown during processing — store as active for cleanup
                    *active = Some(current);
                    return Ok(());
                }
                if handle_worker_issue_processing_cancelled(state, issue.iid, &e) {
                    *active = None;
                    break;
                }
                error!(
                    "{}: Failed to process issue #{}: {}",
                    &state.agent_id, issue.iid, e
                );
                let _ = claim::release_claim(&state.glab, issue.iid, &state.agent_id);
                let _ = state.glab.remove_issue_label(issue.iid, WORKING_ON_LABEL);

                if let Some(ref branch) = current.branch_name {
                    let default_branch = state
                        .git_repo
                        .get_default_branch()
                        .unwrap_or("main".to_string());

                    let _ = state.git_repo.reset_hard();
                    let _ = state.git_repo.checkout_remote_branch(&default_branch);
                    let _ = state.git_repo.delete_local_branch(branch);
                }
            }
        }

        break;
    }

    Ok(())
}

fn mr_has_label(mr: &crate::agents::gitlab::MergeRequest, label: &str) -> bool {
    mr.labels
        .as_ref()
        .is_some_and(|ls| ls.iter().any(|l| l.eq_ignore_ascii_case(label)))
}

/// Handle open MRs labeled `need-ai-worker` as worker tasks:
/// claim MR directly, address unresolved discussions, then release claim.
fn try_handle_need_ai_worker_mr(
    state: &AgentState,
    model: &AgentModel,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<bool> {
    let mut mrs = state.glab.list_merge_requests()?;
    mrs.sort_by_key(|mr| mr.iid);
    for mr in mrs {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(false);
        }
        if mr.state != "opened" {
            continue;
        }
        if !mr_has_label(&mr, NEED_AI_WORKER_LABEL) {
            continue;
        }
        if !super::mr_in_scope(&mr, scope_label) {
            continue;
        }
        if claim::is_mr_claimed(&mr.labels) {
            continue;
        }
        let unresolved = state.glab.get_unresolved_discussion_ids(mr.iid)?;
        if unresolved.is_empty() {
            continue;
        }
        if !claim::try_claim_mr(&state.glab, mr.iid, &state.agent_id, shutdown)? {
            continue;
        }
        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_mr_claim(&state.glab, mr.iid, &state.agent_id);
            return Ok(false);
        }
        info!(
            "{}: Handling labeled MR !{} (`{}`) with {} unresolved discussion(s)",
            &state.agent_id,
            mr.iid,
            NEED_AI_WORKER_LABEL,
            unresolved.len()
        );
        let result = handle_mr_comments(state, model, &mr, None, true);
        let _ = claim::release_mr_claim(&state.glab, mr.iid, &state.agent_id);
        result?;
        return Ok(true);
    }
    Ok(false)
}

fn should_skip_issue(issue: &Issue) -> bool {
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return true;
    }

    if issue
        .labels
        .contains(&super::labels::DO_NOT_IMPLEMENT.to_string())
    {
        return true;
    }

    if issue_has_worker_pending_label(&issue.labels) {
        return true;
    }

    if issue_has_worker_review_only_label(&issue.labels) {
        return true;
    }

    if has_worker_skip_label(&issue.labels) {
        return true;
    }

    false
}

fn issue_has_worker_pending_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_PENDING_LABEL.to_string())
}

fn issue_has_worker_review_only_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_REVIEW_ONLY_LABEL.to_string())
}

fn worker_should_cancel_issue_processing(issue: &Issue) -> bool {
    issue.state != "opened"
        || issue_has_worker_review_only_label(&issue.labels)
        || issue_has_worker_pending_label(&issue.labels)
}

fn worker_issue_cancel_check(
    glab: GitLabClient,
    issue_iid: u64,
) -> Arc<dyn Fn() -> bool + Send + Sync> {
    Arc::new(move || {
        glab.get_issue(issue_iid)
            .ok()
            .is_some_and(|issue| worker_should_cancel_issue_processing(&issue))
    })
}

fn is_worker_agent_cancelled(err: &anyhow::Error) -> bool {
    err.to_string().contains(WORKER_AGENT_CANCELLED_MSG)
}

/// Stop in-flight work when the issue was closed, switched to review-only,
/// or marked `pending` by a human. Returns true when the error was handled
/// as an intentional stop.
fn handle_worker_issue_processing_cancelled(
    state: &AgentState,
    issue_iid: u64,
    err: &anyhow::Error,
) -> bool {
    if !is_worker_agent_cancelled(err) {
        return false;
    }

    match state.glab.get_issue(issue_iid) {
        Ok(issue) if issue_has_worker_pending_label(&issue.labels) => {
            info!(
                "{}: Issue #{} marked `{}` mid-run — releasing worker hold (issue stays open)",
                &state.agent_id, issue_iid, WORKER_PENDING_LABEL
            );
            // Leave the issue open and the `pending` label in place; just drop
            // our claim and session so another worker can pick it up once a
            // human removes `pending`.
            let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
            let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
            state.cleanup_session(issue_iid);
            true
        }
        Ok(issue) if issue_has_worker_review_only_label(&issue.labels) => {
            state.release_worker_hold_review_only(issue_iid);
            true
        }
        Ok(issue) if issue.state != "opened" => {
            state.abandon_closed_issue(issue_iid, None);
            true
        }
        Ok(_) => {
            info!(
                "{}: Stopped work on issue #{} after external cancel signal",
                &state.agent_id, issue_iid
            );
            let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
            let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
            state.cleanup_session(issue_iid);
            true
        }
        Err(e) => {
            warn!(
                "{}: Cancelled while working on issue #{} but failed to re-fetch issue: {}",
                &state.agent_id, issue_iid, e
            );
            let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
            state.cleanup_session(issue_iid);
            true
        }
    }
}

fn stop_worker_issue_if_review_only(state: &AgentState, issue_iid: u64) -> bool {
    let Ok(issue) = state.glab.get_issue(issue_iid) else {
        return false;
    };
    if !issue_has_worker_review_only_label(&issue.labels) {
        return false;
    }
    state.release_worker_hold_review_only(issue_iid);
    true
}

fn should_track_worker_issue(state: &AgentState, issue_iid: u64) -> bool {
    if state
        .glab
        .get_issue(issue_iid)
        .is_ok_and(|issue| issue_has_worker_review_only_label(&issue.labels))
    {
        state.release_worker_hold_review_only(issue_iid);
        return false;
    }

    true
}

fn close_issue_best_effort(gitlab: &GitLabClient, issue_iid: u64) {
    if let Err(e) = gitlab.close_issue(issue_iid) {
        debug!(
            "Issue #{}: could not close after merged MR (may already be closed): {}",
            issue_iid, e
        );
    }
}

#[derive(Debug, Clone, Copy)]
enum ClosesLinkedMr {
    Open(u64),
    Merged,
}

/// MRs whose description contains `Closes #issue_iid` (case-insensitive), preferring an open MR.
fn closes_keyword_mr_status(gitlab: &GitLabClient, issue_iid: u64) -> Option<ClosesLinkedMr> {
    let mrs = gitlab.list_merge_requests().ok()?;
    let mut opens: Vec<u64> = Vec::new();
    let mut any_merged = false;
    for mr in mrs {
        if !crate::agents::gitlab::mr_description_closes_issue(&mr.description, issue_iid) {
            continue;
        }
        match mr.state.as_str() {
            "opened" => opens.push(mr.iid),
            "merged" => any_merged = true,
            _ => {}
        }
    }
    if let Some(&best) = opens.iter().min() {
        return Some(ClosesLinkedMr::Open(best));
    }
    if any_merged {
        return Some(ClosesLinkedMr::Merged);
    }
    None
}

#[derive(Debug, Clone, Copy)]
enum ResolvedTrackedMr {
    Track(u64),
    MergedCloseIssue,
    None,
}

/// Prefer an open MR linked via `Closes #issue`, then session MR if still open, then branch `issue-N`.
fn resolve_tracked_mr_for_worker_issue(
    gitlab: &GitLabClient,
    issue_iid: u64,
    session_mr_iid: u64,
) -> ResolvedTrackedMr {
    match closes_keyword_mr_status(gitlab, issue_iid) {
        Some(ClosesLinkedMr::Open(id)) => return ResolvedTrackedMr::Track(id),
        Some(ClosesLinkedMr::Merged) => return ResolvedTrackedMr::MergedCloseIssue,
        None => {}
    }
    if session_mr_iid > 0
        && let Ok(mr) = gitlab.get_merge_request(session_mr_iid)
        && mr.state == "opened"
    {
        return ResolvedTrackedMr::Track(session_mr_iid);
    }
    if let Some(id) = find_open_mr_for_issue(gitlab, issue_iid) {
        return ResolvedTrackedMr::Track(id);
    }
    ResolvedTrackedMr::None
}

fn has_worker_resume_abandon_label(labels: &[String]) -> bool {
    labels.contains(&ACTION_REQUIRED_LABEL.to_string())
        || labels.contains(&PMO_PROCESSED_LABEL.to_string())
        || labels.contains(&PMO_PENDING_LABEL.to_string())
}

fn has_worker_skip_label(labels: &[String]) -> bool {
    labels.contains(&WORKING_ON_LABEL.to_string()) || has_worker_resume_abandon_label(labels)
}

fn process_issue(
    state: &AgentState,
    model: &AgentModel,
    issue: &Issue,
    current: &mut ActiveIssue,
    scope_label: Option<&str>,
) -> Result<Option<u64>> {
    if stop_worker_issue_if_review_only(state, issue.iid) {
        return Ok(None);
    }

    match closes_keyword_mr_status(&state.glab, issue.iid) {
        Some(ClosesLinkedMr::Open(mr_iid)) => {
            info!(
                "Issue #{} has open MR !{} (linked via Closes #{}), tracking it",
                issue.iid, mr_iid, issue.iid
            );

            current.mr_iid = Some(mr_iid);
            current.mr_created = true;

            let _ = state.glab.add_issue_label(issue.iid, WORKING_ON_LABEL);
            let _ = state.save_session(issue.iid, mr_iid);

            return Ok(Some(mr_iid));
        }
        Some(ClosesLinkedMr::Merged) => {
            info!(
                "Issue #{}: merged MR already references it via Closes #; closing issue",
                issue.iid
            );
            close_issue_best_effort(&state.glab, issue.iid);
            state.cleanup_session(issue.iid);
            return Ok(None);
        }
        None => {}
    }

    // Open MR on branch issue-<n> (no Closes # link required)
    if let Some(mr_iid) = find_open_mr_for_issue(&state.glab, issue.iid) {
        info!(
            "Issue #{} already has open MR !{}, tracking it",
            issue.iid, mr_iid
        );

        current.mr_iid = Some(mr_iid);
        current.mr_created = true;
        let _ = state.glab.add_issue_label(issue.iid, WORKING_ON_LABEL);
        state.save_session(issue.iid, mr_iid)?;
        return Ok(Some(mr_iid));
    }

    let default_branch = state.git_repo.get_default_branch()?;
    state.git_repo.fetch()?;
    let _ = state.git_repo.reset_hard();

    let branch_name = format!("issue-{}", issue.iid);

    let branch_existed = if state.git_repo.remote_branch_exists(&branch_name)? {
        info!(
            "Branch {} already exists on remote, checking if it's stale",
            branch_name
        );
        state.git_repo.checkout_remote_branch(&branch_name)?;

        // Check if the branch has any diff against the target — if not, it's
        // stale (content already merged). Delete and start fresh.
        if !state.git_repo.has_diff_against(&default_branch)? {
            warn!(
                "Branch {} has no diff against {}, discarding stale branch",
                branch_name, default_branch
            );

            let _ = state.git_repo.reset_hard();
            state.git_repo.checkout_remote_branch(&default_branch)?;
            let _ = state.git_repo.delete_local_branch(&branch_name);
            let _ = state.git_repo.delete_remote_branch(&branch_name);

            state
                .git_repo
                .create_branch_from(&branch_name, &default_branch)?;

            false
        } else if !state.git_repo.try_merge(&default_branch)? {
            warn!(
                "Branch {} has conflicts with {}, creating fresh branch instead",
                branch_name, default_branch
            );

            let _ = state.git_repo.reset_hard();
            state.git_repo.checkout_remote_branch(&default_branch)?;
            let _ = state.git_repo.delete_local_branch(&branch_name);
            state
                .git_repo
                .create_branch_from(&branch_name, &default_branch)?;

            false
        } else {
            true
        }
    } else {
        state
            .git_repo
            .create_branch_from(&branch_name, &default_branch)?;
        false
    };

    current.branch_name = Some(branch_name.clone());

    if stop_worker_issue_if_review_only(state, issue.iid) {
        return Ok(None);
    }

    state.glab.add_issue_label(issue.iid, WORKING_ON_LABEL)?;

    let gl_comments = format_issue_comments_for_worker_context(&state.glab, issue.iid);

    let prompt = if branch_existed {
        build_continuation_prompt(state, issue, &gl_comments)?
    } else {
        build_implementation_prompt(state, issue, &gl_comments)?
    };

    let agent_output = match model.complete(
        &prompt,
        &InvokeOptions {
            cancel_check: Some(worker_issue_cancel_check(state.glab.clone(), issue.iid)),
            activity_label: Some(format!(
                "{} implementing issue #{}",
                &state.agent_id, issue.iid
            )),
            ..InvokeOptions::default()
        },
    ) {
        Ok(output) => apply_worker_handoff(output),
        Err(e) if handle_worker_issue_processing_cancelled(state, issue.iid, &e) => {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    info!(
        "{}: Worker agent finished issue #{}",
        &state.agent_id, issue.iid
    );

    if output_signals_cannot_implement(&agent_output) {
        let reason = if output_needs_split(&agent_output) {
            warn!("Issue #{} is too broad, needs splitting", issue.iid);
            let split_reason = extract_split_reason(&agent_output);
            format!(
                "This issue needs to be split into smaller, focused issues:\n\n{}",
                split_reason
            )
        } else {
            warn!("Issue #{} needs clarification", issue.iid);
            extract_clarification(&agent_output)
        };
        state.glab.add_issue_comment(issue.iid, &reason)?;

        // If the branch already had an open MR, close it
        if let Some(mr_iid) = find_open_mr_for_issue(&state.glab, issue.iid) {
            state.glab.add_mr_comment(
                mr_iid,
                &format!(
                    "Closing this MR — the issue cannot be implemented:\n\n{}",
                    reason
                ),
            )?;
            let _ = state.glab.close_mr(mr_iid);
        }

        // Reset git to a clean state — keep remote branch for potential retry
        let default_branch = state
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());
        let _ = state.git_repo.reset_hard();
        let _ = state.git_repo.checkout_remote_branch(&default_branch);
        let _ = state.git_repo.delete_local_branch(&branch_name);

        state.glab.remove_issue_label(issue.iid, WORKING_ON_LABEL)?;
        state
            .glab
            .add_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        current.branch_name = None;
        info!(
            "Issue #{} requires user action, labeled with '{}'",
            issue.iid, ACTION_REQUIRED_LABEL
        );
        return Ok(None);
    }

    // If the agent declared a dependency on another issue that is not yet
    // closed, park this issue: label it `waiting-on-issue:#N`, remove
    // `in-progress`, release the claim, and reset git state. No MR is
    // created — the work can't proceed until the dependency closes. When
    // the dependency issue closes, the worker resume-path strips the label
    // and the issue becomes claimable again.
    if let Some(dep_issue_iid) = extract_depends_on_issue(&agent_output) {
        let dep_closed = state
            .glab
            .get_issue(dep_issue_iid)
            .map(|dep| dep.state == "closed")
            .unwrap_or(false);
        if !dep_closed {
            let label = waiting_on_issue_label(dep_issue_iid);
            let _ = state.glab.add_issue_label(issue.iid, &label);
            let _ = state.glab.remove_issue_label(issue.iid, WORKING_ON_LABEL);
            let _ = state.glab.add_issue_comment(
                issue.iid,
                &format!(
                    "Implementation cannot proceed until issue #{} is closed. \
                     Parking this issue until the dependency resolves.",
                    dep_issue_iid
                ),
            );
            let _ = claim::release_claim(&state.glab, issue.iid, &state.agent_id);
            state.cleanup_session(issue.iid);

            // Reset git to a clean state — keep the remote branch for resume.
            let default_branch = state
                .git_repo
                .get_default_branch()
                .unwrap_or("main".to_string());
            let _ = state.git_repo.reset_hard();
            let _ = state.git_repo.checkout_remote_branch(&default_branch);
            let _ = state.git_repo.delete_local_branch(&branch_name);

            info!(
                "{}: Issue #{} parked waiting on issue #{} (dependency open), released claim",
                &state.agent_id, issue.iid, dep_issue_iid
            );
            current.mr_created = false;
            current.branch_name = None;
            return Ok(None);
        }
    }

    // The agent may have already committed changes itself (it has full shell
    // access). Stage+commit any remaining uncommitted work, then check whether
    // the branch diverges from the base at all.
    state.git_repo.add_all()?;

    let mr_title = extract_mr_title(&agent_output, &issue.title);

    if state.git_repo.has_staged_changes()? {
        let commit_message = build_commit_message(&mr_title, issue.iid);
        state.git_repo.commit(&commit_message)?;
    }

    if !state.git_repo.has_diff_against(&default_branch)? {
        warn!(
            "Issue #{}: agent produced no code changes, retrying",
            issue.iid
        );
        anyhow::bail!("{}", no_code_changes_retry_error_message(issue.iid));
    }

    state.git_repo.push(&branch_name)?;
    let mr_description = format!(
        "Closes #{}\n\n{}",
        issue.iid,
        extract_mr_description(&agent_output)
    );

    let mr_iid = state.glab.create_merge_request(
        &branch_name,
        &default_branch,
        &mr_title,
        &mr_description,
    )?;
    current.mr_iid = Some(mr_iid);
    current.mr_created = true;

    info!("Created MR !{} for issue #{}", mr_iid, issue.iid);

    if let Some(lbl) = scope_label
        && let Err(e) = state.glab.add_mr_label_with_transient_retries(mr_iid, lbl)
    {
        warn!(
            "{}: Failed to add scope label {:?} to MR !{} (permanent error): {}",
            &state.agent_id, lbl, mr_iid, e
        );
    }

    let impl_summary = extract_mr_description(&agent_output);
    state.save_session_with_summary(issue.iid, mr_iid, &impl_summary)?;

    Ok(Some(mr_iid))
}

// ---------------------------------------------------------------------------
// MR comment handling
// ---------------------------------------------------------------------------

/// Returns `Ok(true)` if the agent decided the issue cannot be resolved and
/// the MR was closed + issue rejected.
fn handle_mr_comments(
    state: &AgentState,
    model: &AgentModel,
    mr: &crate::agents::gitlab::MergeRequest,
    linked_issue_iid: Option<u64>,
    comments_only_mode: bool,
) -> Result<bool> {
    let latest_mr = state.glab.get_merge_request(mr.iid)?;
    let unresolved_ids = state.glab.get_unresolved_discussion_ids(latest_mr.iid)?;
    let all_comments = state.glab.get_mr_comments(latest_mr.iid)?;
    let plain_comments: Vec<_> = all_comments
        .iter()
        .filter(|c| !c.discussion_resolvable)
        .cloned()
        .collect();

    if unresolved_ids.is_empty()
        && plain_comments.is_empty()
        && (comments_only_mode || !latest_mr.has_conflicts)
    {
        return Ok(false);
    }

    if !unresolved_ids.is_empty() {
        info!(
            "MR !{} has {} unresolved discussion(s) to address",
            latest_mr.iid,
            unresolved_ids.len()
        );
    }

    if latest_mr.has_conflicts {
        info!("MR !{} has merge conflicts to resolve", latest_mr.iid);
    }

    let unresolved_comments = if unresolved_ids.is_empty() {
        Vec::new()
    } else {
        let unresolved_set: HashSet<&str> = unresolved_ids.iter().map(String::as_str).collect();
        all_comments
            .iter()
            .filter(|c| unresolved_set.contains(c.discussion_id.as_str()))
            .cloned()
            .collect()
    };

    state.git_repo.fetch_branches(&[
        latest_mr.target_branch.as_str(),
        latest_mr.source_branch.as_str(),
    ])?;
    let _ = state.git_repo.reset_hard();

    state
        .git_repo
        .checkout_remote_branch(&latest_mr.source_branch)?;

    // Remember the remote HEAD so we can detect changes after the agent runs,
    // even if the agent disobeys and commits/pushes itself.
    let pre_agent_sha = latest_mr
        .sha
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or(
            state
                .git_repo
                .rev_parse(&format!("origin/{}", latest_mr.source_branch))?,
        );

    // Build a diff snapshot from the latest source/target state before we merge
    // target into source locally for conflict handling.
    let diff_context_content = build_mr_diff_context(
        &state.project_name,
        &latest_mr,
        &state.git_repo,
        &state.glab,
    );

    // Merge the latest target branch so the worker has up-to-date upstream code.
    let local_merge_clean = state.git_repo.merge_no_abort(&latest_mr.target_branch)?;
    if !local_merge_clean {
        warn!(
            "MR !{}: source branch has conflicts with {}, worker agent will resolve them",
            latest_mr.iid, latest_mr.target_branch
        );
    }
    let requires_conflict_resolution = latest_mr.has_conflicts || !local_merge_clean;
    let merge_conflict_status = build_merge_conflict_status_section(
        &latest_mr,
        requires_conflict_resolution,
        local_merge_clean,
        &state.git_repo,
    )?;

    let issue_number = linked_issue_iid
        .or_else(|| extract_issue_number_from_branch(&latest_mr.source_branch).ok());
    if let Some(issue_iid) = issue_number
        && stop_worker_issue_if_review_only(state, issue_iid)
    {
        return Ok(false);
    }
    let issue_context = issue_number
        .map(|n| load_issue_context(&state.glab, n))
        .transpose()?
        .unwrap_or_else(|| "No linked issue context available for this MR.".to_string());
    let implementation_summary = issue_number
        .map(|n| state.load_implementation_summary(n))
        .unwrap_or_else(|| "No previous implementation summary available.".to_string());

    let unresolved_comments_text = format_comments_for_prompt(&unresolved_comments);
    let plain_comments_text = format_comments_for_prompt(&plain_comments);
    let all_comments_text = format_comments_for_prompt(&all_comments);

    let combined_context_content =
        build_combined_mr_feedback_context(CombinedMrFeedbackContextInput {
            project_name: &state.project_name,
            mr: &latest_mr,
            issue_context: &issue_context,
            implementation_summary: &implementation_summary,
            merge_conflict_status: &merge_conflict_status,
            unresolved_comments_text: &unresolved_comments_text,
            plain_comments_text: &plain_comments_text,
            all_comments_text: &all_comments_text,
            diff_context: &diff_context_content,
        });
    // Write the context to disk for archival/debugging, but inject the content
    // directly into the prompt so the model doesn't waste turns reading it back.
    let _combined_context_path = write_task_context_file(
        &state.sessions_dir,
        &format!(
            "{}-mr-feedback-and-diff-{}.md",
            &state.agent_id, latest_mr.iid
        ),
        &combined_context_content,
    )?;

    let feedback_scope_rules = get_feedback_scope_rules();
    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are addressing reviewer feedback on a merge request in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You MUST delete, rename, or move files as needed — do not ask permission or suggest it
- Leave staging, committing, pushing, and merge request creation to the system
- The task context above includes all comments and the diff — review it, then act directly
- Address all unresolved thread feedback and actionable plain MR comments autonomously
- Make all necessary code changes to resolve the comments
- Keep the original issue requirements in mind while addressing feedback
- If the workspace has merge conflict markers (<<<<<<< / ======= / >>>>>>>), resolve ALL of them before doing anything else. Edit each conflicted file to keep the correct version.
- The **Merge conflict status** section in the task context above is verified by Potlatch. Do NOT claim conflicts are fixed unless that section would be clean after your edits and you commit/push the resolution.
- Potlatch will refuse to mark review threads resolved while GitLab still reports merge conflicts or conflict markers remain in the branch.
- Potlatch already fetched `origin/{}` and merged it into your workspace when conflicts were reported. Edit the listed conflicted files, remove all conflict markers, and leave committing/pushing to Potlatch. Do not claim the conflict is fixed until the **Merge conflict status** section shows a clean merge with the fetched target tip.

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly.
2. The task context is already included above. Review the "Unresolved MR comments to address", "Plain MR comments to consider", and "Full MR comment history for context" sections. Do NOT use read on the task context — it's already in your prompt.
3. Use inline comment locations (`path:line` or `path:start-end`) from the comments to find the corresponding code and make targeted fixes. Grep for the relevant symbol, read only the surrounding lines, then edit.
4. First, check for merge conflicts using the **Merge conflict status** section and your workspace. If any exist, resolve ALL conflicts in every file, commit the resolution, and verify the target branch merges cleanly before claiming completion.
5. Review the original issue and what was implemented
6. Review ALL comments to understand the full conversation and context, including simple comments that do not require resolution.
7. Identify which feedback items still need action. Treat comments in "Unresolved MR comments to address" as actionable threaded feedback. Also consider comments in "Plain MR comments to consider" actionable when they ask for changes, but remember they are plain MR comments and cannot be marked resolved. Use the full comment history only for context, clarification, and avoiding stale assumptions.
8. Make the necessary code changes to address all unresolved threaded feedback and any actionable plain MR comments
9. If the reviewer asked you to delete, rename, or move files, make those file changes.
10. Ensure changes align with both the original requirements and reviewer feedback
11. If the reviewer says code changes are too large (above ~1500 lines total or ~500 non-test lines), you have TWO options:
   a) Adjust your implementation to reduce changed lines — simplify, remove unnecessary changes, trim scope
   b) If you cannot reasonably reduce the size, respond with CANNOT_RESOLVE so the issue is rejected and the problem is reported back
   Do NOT try to split the issue yourself — that is handled by the PMO agent, not you.
12. If you determine that the feedback cannot be resolved without additional human input (e.g. the requirements are ambiguous, the reviewer is asking for something outside the scope of the issue, or the necessary information is missing), respond with:
   CANNOT_RESOLVE
   REASON: <explain concisely why this cannot be resolved autonomously and what input is needed>
13. If the reviewer asked you to fix the MR title or description, include updated versions in your response:
   MR_TITLE: <SHORT title (max 8-10 words) stating the main feature or fix — no enumeration of details, no markdown. It must describe the overall MR, not just the latest incremental change. Do NOT change the title just because you made another follow-up commit; keep it stable unless the reviewer explicitly asks for a title fix or the current title is clearly wrong for the whole MR.>
   MR_DESCRIPTION:
   <full description with goal, implementation, and testing sections — NEVER include PUBLIC_COMMENT_BEGIN/END here; those markers are only for thread replies below; NEVER paste or quote text from repo-root notes.md here>
14. After addressing feedback, provide a summary:
   CHANGES_SUMMARY: <A concise sentence summarizing the substance of the changes made — this will be used as the git commit message, so it must convey the main idea of what was changed>
   The summary must reflect the actual source/MR metadata changes you made in this run. Do not mention a reviewer concern as fixed unless the final diff or MR metadata actually changed to address it.
15. For any human-facing GitLab comment/reply text, include a stable block:
   PUBLIC_COMMENT_BEGIN
   <only the final comment text to post publicly; no progress updates, no tool/log output>
   PUBLIC_COMMENT_END
   Keep this public reply concise. Do NOT include a `Validation:` section, test/lint command lists, passed/failed command output, or unrelated repository backlog notes.
   The public reply must exactly match the committed changes from this run. Mention only feedback items you actually resolved in code or MR metadata. If you did not change code/metadata for an item, say so with MARK_DISCUSSIONS_RESOLVED: no instead of implying it was fixed.
16. Control whether GitLab should mark open review discussions as resolved after your reply:
   - `MARK_DISCUSSIONS_RESOLVED: yes` — only when you have actually fixed what the reviewer asked for (code and/or MR title/description updates they requested), so the thread can be considered addressed.
   - For merge-conflict feedback: use `yes` only after you committed and pushed a branch that merges cleanly with `origin/{}` with no conflict markers left. If conflicts remain, use `no`.
   - `MARK_DISCUSSIONS_RESOLVED: yes` is also correct when you verified that no code change is needed because the branch already satisfies the reviewer request. In that case, PUBLIC_COMMENT must explain the existing behavior specifically instead of saying only "no changes needed".
   - `MARK_DISCUSSIONS_RESOLVED: no` — when your reply does not resolve the comment (e.g. partial progress, disagreement, or anything that still needs the reviewer). The system will still post your reply on each thread but will **not** mark discussions resolved.
   - If you omit this line: the system assumes `yes` only when it detects branch changes: new commits (including rebases) on the MR branch or the remote branch tip moved. For title/description-only fixes, set `MARK_DISCUSSIONS_RESOLVED: yes` explicitly when the feedback is resolved.
   - Plain MR comments cannot be marked resolved.
17. Control whether GitLab should post a new normal MR comment for plain, non-resolvable MR comments:
   - The system will NOT post normal MR comments for plain MR comments unless you explicitly include `POST_PLAIN_COMMENT: yes`.
   - Use `POST_PLAIN_COMMENT: yes` only when a new public reply is necessary for a plain MR comment. Include PUBLIC_COMMENT with the exact comment body to post.
   - If a plain MR comment needs no public reply, or if your response would only repeat that no further changes were needed, omit `POST_PLAIN_COMMENT` or set `POST_PLAIN_COMMENT: no`.
18. Before you finish, edit repo-root notes.md only if you can add lines that pass the **NOTES.MD** rules in your main worker instructions (same as implementation runs): **no** backticks, **no** file paths, **no** repo-specific symbol names, **no** code tours — and **no** bullets that merely **summarize what you did** this run in "timeless" wording (that still belongs in the MR, not notes). **No** lines about how to write notes or what notes are for. If nothing meets that bar, leave notes.md unchanged. Never copy notes.md into MR_DESCRIPTION, MR_TITLE, PUBLIC_COMMENT, or any GitLab field.

Proceed with addressing the feedback autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        latest_mr.iid,
        latest_mr.title,
        combined_context_content,
        latest_mr.target_branch,
        feedback_scope_rules,
        latest_mr.target_branch
    );

    let agent_output = if let Some(issue_iid) = issue_number {
        info!(
            "{}: Worker agent addressing MR !{} feedback for issue #{}",
            &state.agent_id, latest_mr.iid, issue_iid
        );
        let output = match model.complete(
            &prompt,
            &InvokeOptions {
                cancel_check: Some(worker_issue_cancel_check(state.glab.clone(), issue_iid)),
                activity_label: Some(format!(
                    "{} addressing MR !{} feedback",
                    &state.agent_id, latest_mr.iid
                )),
                ..InvokeOptions::default()
            },
        ) {
            Ok(output) => apply_worker_handoff(output),
            Err(e) if handle_worker_issue_processing_cancelled(state, issue_iid, &e) => {
                return Ok(false);
            }
            Err(e) => return Err(e),
        };
        info!(
            "{}: Worker agent finished MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        output
    } else {
        info!(
            "{}: Worker agent addressing MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        let output = model.complete(
            &prompt,
            &InvokeOptions {
                activity_label: Some(format!(
                    "{} addressing MR !{} feedback",
                    &state.agent_id, latest_mr.iid
                )),
                ..InvokeOptions::default()
            },
        )?;
        let output = apply_worker_handoff(output);
        info!(
            "{}: Worker agent finished MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        output
    };

    if output_signals_cannot_resolve(&agent_output) {
        let reason = extract_cannot_resolve_reason(&agent_output);
        warn!(
            "MR !{} cannot be resolved autonomously: {}",
            latest_mr.iid, reason
        );

        if let Some(issue_number) = issue_number {
            abandon_mr(state, &latest_mr, issue_number, &reason)?;
        } else {
            state.glab.add_mr_comment(
                latest_mr.iid,
                &format!(
                    "Cannot resolve this MR feedback autonomously:\n\n{}",
                    reason
                ),
            )?;
            let _ = state.glab.close_mr(latest_mr.iid);
        }

        return Ok(true);
    }

    // If the agent provided updated MR_TITLE / MR_DESCRIPTION that differ from
    // the current MR metadata (e.g. the reviewer asked for a better title),
    // update the MR. With the `handoff` tool, the model always provides
    // mr_title/mr_description — only write when something actually changed.
    let new_title = extract_explicit_mr_title(&agent_output);
    let new_desc = extract_explicit_mr_description(&agent_output);
    let title_changed = new_title.as_deref().is_some_and(|t| t != latest_mr.title);
    let desc_changed = new_desc != "Implementation completed." && new_desc != latest_mr.description;
    if title_changed || desc_changed {
        let title = if title_changed {
            new_title.as_deref().unwrap_or(&latest_mr.title)
        } else {
            &latest_mr.title
        };

        let desc = if desc_changed {
            &new_desc
        } else {
            &latest_mr.description
        };

        if let Err(e) = state
            .glab
            .update_mr_title_description(latest_mr.iid, title, desc)
        {
            warn!("Failed to update MR !{} metadata: {}", latest_mr.iid, e);
        } else {
            info!(
                "Updated MR !{} title/description from agent feedback",
                latest_mr.iid
            );
        }
    }

    // Detect all changes the agent made: working tree, staged, or committed
    // (even if the agent disobeyed and ran git commit/push itself).
    state.git_repo.fetch_branches(&[
        latest_mr.target_branch.as_str(),
        latest_mr.source_branch.as_str(),
    ])?;
    let mut has_new_changes = state.git_repo.has_changes_since(&pre_agent_sha)?;

    if state.git_repo.is_merge_in_progress()? {
        if state.git_repo.stage_resolved_unmerged_paths()? {
            info!(
                "MR !{}: staged merge-conflict files with no remaining conflict markers",
                latest_mr.iid
            );
        }
        state.git_repo.add_all()?;
        has_new_changes = state.git_repo.has_changes_since(&pre_agent_sha)?;
    } else if has_new_changes {
        state.git_repo.add_all()?;
        let summary_for_commit = extract_changes_summary(&agent_output);
        if state.git_repo.has_staged_changes()? {
            let commit_msg = build_commit_message(&summary_for_commit, issue_number.unwrap_or(0));
            state.git_repo.commit(&commit_msg)?;
        }
    }

    let merge_commit_msg = build_commit_message(
        &format!(
            "Merge origin/{} into {}",
            latest_mr.target_branch, latest_mr.source_branch
        ),
        issue_number.unwrap_or(0),
    );
    if state.git_repo.complete_merge_if_ready(&merge_commit_msg)? {
        has_new_changes = true;
        info!(
            "MR !{}: concluded in-progress merge with origin/{}",
            latest_mr.iid, latest_mr.target_branch
        );
    }

    let mut conflicts_unresolved = state.git_repo.merge_conflicts_present()?;
    if requires_conflict_resolution {
        state.git_repo.fetch_branches(&[
            latest_mr.target_branch.as_str(),
            latest_mr.source_branch.as_str(),
        ])?;
        if !state
            .git_repo
            .verify_up_to_date_with_target(&latest_mr.target_branch)?
        {
            conflicts_unresolved = true;
            warn!(
                "MR !{}: branch still does not merge cleanly with origin/{} (fetched latest target and source)",
                latest_mr.iid, latest_mr.target_branch
            );
        } else if state.git_repo.has_changes_since(&pre_agent_sha)? {
            has_new_changes = true;
            state.git_repo.add_all()?;
            if state.git_repo.has_staged_changes()? {
                state.git_repo.commit(&merge_commit_msg)?;
            }
        }
    }

    if !conflicts_unresolved {
        conflicts_unresolved = state.git_repo.merge_conflicts_present()?;
    }

    let diff_highlights = if has_new_changes {
        build_diff_highlights_since(&state.git_repo, &pre_agent_sha)
    } else {
        None
    };

    if has_new_changes && conflicts_unresolved {
        warn!(
            "MR !{}: not pushing — merge conflicts with origin/{} are still unresolved",
            latest_mr.iid, latest_mr.target_branch
        );
    } else if has_new_changes {
        state.git_repo.push(&latest_mr.source_branch)?;
        info!(
            "Pushed changes addressing feedback for MR !{}",
            latest_mr.iid
        );
    } else {
        info!(
            "Agent processed comments for MR !{} but made no code changes",
            latest_mr.iid
        );
    }

    let mr_now = state.glab.get_merge_request(latest_mr.iid)?;
    if mr_now.has_conflicts {
        conflicts_unresolved = true;
        warn!(
            "MR !{}: GitLab still reports merge conflicts after worker run",
            latest_mr.iid
        );
    }
    let mr_gitlab_surface_changed = merge_request_surface_changed(&latest_mr, &mr_now);

    // Only auto-resolve when the branch tip actually changed. MR metadata-only
    // changes (title/description/labels) can happen without addressing feedback.
    let post_origin_head = state
        .git_repo
        .rev_parse(&format!("origin/{}", latest_mr.source_branch))
        .unwrap_or_else(|_| pre_agent_sha.clone());
    let branch_tip_changed = post_origin_head.trim() != pre_agent_sha.trim();
    let implicit_resolve_discussions = has_new_changes || branch_tip_changed;
    if mr_gitlab_surface_changed && !implicit_resolve_discussions {
        info!(
            "MR !{} metadata changed without branch updates; discussions will remain open unless explicitly requested",
            latest_mr.iid
        );
    }

    // Re-fetch unresolved discussions — the original list may have been empty
    // if we were triggered by has_conflicts alone. After pushing, resolve all
    // remaining open discussions (including conflict comments).
    let ids_to_resolve = if unresolved_ids.is_empty() {
        state
            .glab
            .get_unresolved_discussion_ids(latest_mr.iid)
            .unwrap_or_default()
    } else {
        unresolved_ids
    };

    let should_post_plain_comment = should_post_plain_comment(&agent_output);
    let needs_reply_body =
        !ids_to_resolve.is_empty() || (!plain_comments.is_empty() && should_post_plain_comment);

    if requires_conflict_resolution && conflicts_unresolved {
        info!(
            "MR !{}: merge conflicts with origin/{} remain; skipping GitLab replies until the branch merges cleanly",
            latest_mr.iid, latest_mr.target_branch
        );
        return Ok(false);
    }

    let reply_body = if needs_reply_body {
        let reply_raw = if let Some(block) = extract_worker_public_comment(&agent_output) {
            block
        } else if let Some(reply) = build_feedback_resolution_reply(
            &agent_output,
            has_new_changes,
            diff_highlights.as_deref(),
        ) {
            reply
        } else {
            return Err(anyhow::anyhow!(
                "worker produced no source changes and no feedback reply for MR !{}",
                latest_mr.iid
            ));
        };
        Some(strip_worker_reply_boilerplate(&reply_raw))
    } else {
        None
    };
    let resolve_discussions = feedback_discussions_may_be_resolved(
        &agent_output,
        implicit_resolve_discussions,
        conflicts_unresolved,
    );
    if conflicts_unresolved && parse_mark_discussions_resolved(&agent_output) == Some(true) {
        warn!(
            "MR !{}: ignoring agent request to mark discussions resolved while merge conflicts remain",
            latest_mr.iid
        );
    }
    if !resolve_discussions && !ids_to_resolve.is_empty() {
        info!(
            "MR !{}: posting feedback replies without resolving discussions (MARK_DISCUSSIONS_RESOLVED: no and no implicit resolving actions)",
            latest_mr.iid
        );
    }

    for discussion_id in &ids_to_resolve {
        let Some(reply_body) = reply_body.as_deref() else {
            continue;
        };
        if let Err(e) = state
            .glab
            .reply_to_discussion(latest_mr.iid, discussion_id, reply_body)
        {
            warn!("Failed to reply to discussion {}: {}", discussion_id, e);
        }
        if resolve_discussions
            && let Err(e) = state.glab.resolve_discussion(latest_mr.iid, discussion_id)
        {
            warn!("Failed to resolve discussion {}: {}", discussion_id, e);
        }
    }

    if !plain_comments.is_empty()
        && should_post_plain_comment
        && let Some(reply_body) = reply_body.as_deref()
        && let Err(e) = state.glab.add_mr_comment(latest_mr.iid, reply_body)
    {
        warn!(
            "Failed to post MR !{} reply for plain comments: {}",
            latest_mr.iid, e
        );
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// Session file management (shared directory)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionFile {
    issue_iid: u64,
    /// 0 means no MR created yet (issue claimed but implementation not done).
    mr_iid: u64,
    #[serde(default)]
    agent_id: Option<String>,
    implementation_summary: Option<String>,
}

/// On startup, try to resume a session this worker previously owned.
/// Checks the stored agent_id first, then falls back to checking GitLab labels.
fn try_resume_session(state: &AgentState, scope_label: Option<&str>) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", &state.agent_id);
    let issue_prefix = format!("{}_issue_", &state.agent_id);

    let entries = match fs::read_dir(&state.sessions_dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("Failed to read sessions directory: {}", e);
            return None;
        }
    };

    for entry in entries.flatten() {
        let Ok(file_name) = entry.file_name().into_string() else {
            continue;
        };
        if !file_name.starts_with(&issue_prefix) || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix(&issue_prefix)
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };

        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = state.load_session(issue_iid) else {
            continue;
        };

        // Fast path: session file records which agent owned it
        if let Some(ref stored_id) = session.agent_id {
            if stored_id == &state.agent_id {
                let issue = match state.glab.get_issue(issue_iid) {
                    Ok(i) => i,
                    Err(e) => {
                        warn!(
                            "{}: Failed to verify issue #{} for session resume: {}, skipping",
                            &state.agent_id, issue_iid, e
                        );
                        continue;
                    }
                };

                if issue.state != "opened" {
                    let mr_iid = (session.mr_iid > 0).then_some(session.mr_iid);
                    state.abandon_closed_issue(issue_iid, mr_iid);
                    continue;
                }

                if !issue_in_scope(&issue, scope_label) {
                    continue;
                }

                if issue_has_worker_pending_label(&issue.labels) {
                    state.release_worker_hold_pending_gitlab_only(issue_iid);
                    continue;
                }

                if issue_has_worker_review_only_label(&issue.labels) {
                    state.release_worker_hold_review_only(issue_iid);
                    continue;
                }

                match resolve_tracked_mr_for_worker_issue(&state.glab, issue_iid, session.mr_iid) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(&state.glab, issue_iid);

                        let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
                        let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

                        state.cleanup_session(issue_iid);
                        continue;
                    }
                    ResolvedTrackedMr::Track(mr_iid) => {
                        info!(
                            "{}: Found session file for issue #{} (MR: !{}), resuming",
                            &state.agent_id, issue_iid, mr_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: Some(mr_iid),
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: true,
                        });
                    }
                    ResolvedTrackedMr::None => {
                        info!(
                            "{}: Found session file for issue #{} (no MR yet), resuming",
                            &state.agent_id, issue_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: None,
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: false,
                        });
                    }
                }
            }
            // Session belongs to a different agent — skip
            continue;
        }

        // Fallback for old session files without agent_id: check GitLab labels
        match state.glab.get_issue(issue_iid) {
            Ok(issue) if issue.state != "opened" => {
                let mr_iid = (session.mr_iid > 0).then_some(session.mr_iid);
                state.abandon_closed_issue(issue_iid, mr_iid);
            }
            Ok(issue)
                if issue.labels.contains(&claim_label) && issue_in_scope(&issue, scope_label) =>
            {
                if issue_has_worker_pending_label(&issue.labels) {
                    state.release_worker_hold_pending_gitlab_only(issue_iid);
                    continue;
                }

                if issue_has_worker_review_only_label(&issue.labels) {
                    state.release_worker_hold_review_only(issue_iid);
                    continue;
                }

                match resolve_tracked_mr_for_worker_issue(&state.glab, issue_iid, session.mr_iid) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(&state.glab, issue_iid);

                        let _ = claim::release_claim(&state.glab, issue_iid, &state.agent_id);
                        let _ = &state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

                        state.cleanup_session(issue_iid);
                        continue;
                    }
                    ResolvedTrackedMr::Track(mr_iid) => {
                        info!(
                            "{}: Found unclaimed session for issue #{} with matching label, resuming (MR !{})",
                            &state.agent_id, issue_iid, mr_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: Some(mr_iid),
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: true,
                        });
                    }
                    ResolvedTrackedMr::None => {
                        info!(
                            "{}: Found unclaimed session for issue #{} with matching label, resuming",
                            &state.agent_id, issue_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: None,
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: false,
                        });
                    }
                }
            }

            Ok(_) => {
                debug!(
                    "{}: Session for issue #{} exists but claim label not found",
                    &state.agent_id, issue_iid
                );
            }
            Err(e) => {
                warn!(
                    "{}: Failed to verify issue #{} on GitLab: {}, skipping",
                    &state.agent_id, issue_iid, e
                );
            }
        }
    }

    None
}

/// Scan all open GitLab issues for this worker's claim label.
/// Used as a fallback when the session file is missing (e.g. hard kill / crash).
fn find_claimed_issue(state: &AgentState, scope_label: Option<&str>) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", &state.agent_id);

    let issues = match state.glab.list_issues() {
        Ok(i) => i,
        Err(e) => {
            warn!(
                "{}: Failed to scan issues for orphaned claims: {}",
                &state.agent_id, e
            );

            return None;
        }
    };

    for issue in &issues {
        if issue.state != "opened" {
            continue;
        }

        if !issue.labels.contains(&claim_label) {
            continue;
        }

        if !issue_in_scope(issue, scope_label) {
            continue;
        }

        if issue_has_worker_pending_label(&issue.labels) {
            state.release_worker_hold_pending_gitlab_only(issue.iid);
            continue;
        }

        if issue_has_worker_review_only_label(&issue.labels) {
            state.release_worker_hold_review_only(issue.iid);
            continue;
        }

        match resolve_tracked_mr_for_worker_issue(&state.glab, issue.iid, 0) {
            ResolvedTrackedMr::MergedCloseIssue => {
                close_issue_best_effort(&state.glab, issue.iid);

                let _ = claim::release_claim(&state.glab, issue.iid, &state.agent_id);
                let _ = state.glab.remove_issue_label(issue.iid, WORKING_ON_LABEL);

                state.cleanup_session(issue.iid);
                continue;
            }
            ResolvedTrackedMr::Track(mr_iid) => {
                info!(
                    "{}: Found orphaned claim on issue #{} (MR: !{}), adopting it",
                    &state.agent_id, issue.iid, mr_iid
                );

                return Some(ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: Some(mr_iid),
                    branch_name: Some(format!("issue-{}", issue.iid)),
                    mr_created: true,
                });
            }
            ResolvedTrackedMr::None => {
                info!(
                    "{}: Found orphaned claim on issue #{} (no MR), adopting it",
                    &state.agent_id, issue.iid
                );

                return Some(ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: None,
                    branch_name: Some(format!("issue-{}", issue.iid)),
                    mr_created: false,
                });
            }
        }
    }

    None
}

/// Try to adopt an orphaned session file (issue has no claim label from any worker).
fn try_adopt_orphaned_session(
    state: &AgentState,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Option<ActiveIssue> {
    let entries = fs::read_dir(&state.sessions_dir).ok()?;

    let issue_prefix = format!("{}_issue_", &state.agent_id);

    for entry in entries.flatten() {
        let Ok(file_name) = entry.file_name().into_string() else {
            continue;
        };

        if !file_name.starts_with(&issue_prefix) || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix(&issue_prefix)
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };

        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = state.load_session(issue_iid) else {
            continue;
        };

        // Skip sessions that belong to a specific agent — those should be
        // resumed by their owner via try_resume_session, not adopted.
        if session.agent_id.is_some() {
            continue;
        }

        let Ok(issue) = state.glab.get_issue(issue_iid) else {
            continue;
        };

        if issue.state != "opened" {
            state.cleanup_session(issue_iid);
            continue;
        }

        if !issue_in_scope(&issue, scope_label) {
            continue;
        }

        if issue_has_worker_pending_label(&issue.labels) {
            continue;
        }

        if issue_has_worker_review_only_label(&issue.labels) {
            continue;
        }

        // Skip if already claimed by someone
        if claim::is_claimed(&issue.labels) {
            continue;
        }

        let (mr_iid, mr_created) = if session.mr_iid > 0 {
            // Check if the MR is still open
            if let Ok(mr) = state.glab.get_merge_request(session.mr_iid) {
                if mr.state == "merged" || mr.state == "closed" {
                    info!(
                        "{}: Orphaned session for issue #{} has {} MR !{}, cleaning up",
                        &state.agent_id, issue_iid, mr.state, session.mr_iid
                    );

                    state.cleanup_session(issue_iid);
                    let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                    continue;
                }
            } else {
                continue;
            }
            (Some(session.mr_iid), true)
        } else {
            // No MR yet — check if one was created in the meantime
            match find_open_mr_for_issue(&state.glab, issue_iid) {
                Some(mr) => (Some(mr), true),
                None => (None, false),
            }
        };

        // Try to claim this issue
        match claim::try_claim_issue(&state.glab, issue_iid, &state.agent_id, shutdown) {
            Ok(true) => {
                info!(
                    "{}: Adopted orphaned issue #{} (MR: {})",
                    &state.agent_id,
                    issue_iid,
                    mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
                );

                return Some(ActiveIssue {
                    issue_iid,
                    mr_iid,
                    branch_name: Some(format!("issue-{}", issue_iid)),
                    mr_created,
                });
            }
            _ => continue,
        }
    }

    None
}

// ---------------------------------------------------------------------------
// MR lookup
// ---------------------------------------------------------------------------

/// Find an *open* MR for an issue. Closed/merged MRs are ignored.
fn find_open_mr_for_issue(gitlab: &GitLabClient, issue_iid: u64) -> Option<u64> {
    let branch_name = format!("issue-{}", issue_iid);
    gitlab
        .find_open_mr_by_source_branch(&branch_name)
        .ok()
        .flatten()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_issue_number_from_branch(branch_name: &str) -> Result<u64> {
    if let Some(num_str) = branch_name.strip_prefix("issue-") {
        num_str
            .parse::<u64>()
            .context("Failed to parse issue number from branch name")
    } else {
        anyhow::bail!("Branch name does not match issue-<number> format")
    }
}

fn no_code_changes_retry_error_message(issue_iid: u64) -> String {
    format!(
        "Worker produced no code changes for issue #{}; retrying without marking action-required",
        issue_iid
    )
}

fn load_issue_context(gitlab: &GitLabClient, issue_number: u64) -> Result<String> {
    match gitlab.get_issue(issue_number) {
        Ok(issue) => Ok(format!(
            "Issue #{}: {}\n\nDescription:\n{}\n\nLabels: {}",
            issue.iid,
            issue.title,
            issue.description,
            issue.labels.join(", ")
        )),
        Err(_) => Ok(format!("Issue #{} (details not available)", issue_number)),
    }
}

fn abandon_mr(
    state: &AgentState,
    mr: &crate::agents::gitlab::MergeRequest,
    issue_iid: u64,
    reason: &str,
) -> Result<()> {
    state.glab.add_mr_comment(
        mr.iid,
        &format!(
            "Closing this MR — the issue cannot be resolved autonomously:\n\n{}",
            reason
        ),
    )?;
    let _ = state.glab.close_mr(mr.iid);

    let default_branch = state
        .git_repo
        .get_default_branch()
        .unwrap_or("main".to_string());
    let _ = state.git_repo.reset_hard();
    let _ = state.git_repo.checkout_remote_branch(&default_branch);
    let _ = state.git_repo.delete_local_branch(&mr.source_branch);

    let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
    state
        .glab
        .add_issue_label(issue_iid, ACTION_REQUIRED_LABEL)?;
    state.glab.add_issue_comment(
        issue_iid,
        &format!(
            "This issue requires additional human input before it can be implemented:\n\n{}",
            reason
        ),
    )?;

    state.cleanup_session(issue_iid);

    info!(
        "Abandoned MR !{} and rejected issue #{} with action-required",
        mr.iid, issue_iid
    );
    Ok(())
}

fn output_signals_cannot_implement(agent_output: &AgentHandoff) -> bool {
    agent_output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("cannot_implement"))
        || agent_output.response.contains("CANNOT_IMPLEMENT")
}

fn output_signals_cannot_resolve(agent_output: &AgentHandoff) -> bool {
    agent_output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("cannot_resolve"))
        || agent_output.response.contains("CANNOT_RESOLVE")
}

fn output_needs_split(agent_output: &AgentHandoff) -> bool {
    handoff_output(agent_output)
        .and_then(|ho| ho.get("needs_split").and_then(serde_json::Value::as_str))
        .is_some_and(|s| !s.trim().is_empty())
        || agent_output
            .decision
            .as_deref()
            .is_some_and(|d| d.eq_ignore_ascii_case("needs_split"))
        || agent_output.response.contains("NEEDS_SPLIT")
}

fn extract_cannot_resolve_reason(agent_output: &AgentHandoff) -> String {
    if let Some(block) = extract_worker_public_comment(agent_output) {
        return block;
    }
    if let Some(reason) = &agent_output.reason {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(pos) = agent_output.response.find("REASON:") {
        let reason = &agent_output.response[pos + 7..];
        if let Some(end) = reason.find('\n') {
            return reason[..end].trim().to_string();
        }
        return reason.trim().to_string();
    }
    "The implementation cannot proceed without additional human input.".to_string()
}

fn build_commit_message(title: &str, issue_iid: u64) -> String {
    let first_line = title.lines().next().unwrap_or(title).trim();
    // Truncate to a reasonable commit title length
    let title_truncated = if first_line.len() > 72 {
        format!("{}...", &first_line[..69])
    } else {
        first_line.to_string()
    };
    if issue_iid > 0 {
        format!("{}\n\nRefs #{}", title_truncated, issue_iid)
    } else {
        title_truncated
    }
}

fn extract_changes_summary(agent_output: &AgentHandoff) -> String {
    if let Some(s) = handoff_output(agent_output).and_then(|ho| {
        ho.get("changes_summary")
            .and_then(serde_json::Value::as_str)
    }) {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return strip_markdown_formatting(trimmed);
        }
    }
    if let Some(pos) = agent_output.response.find("CHANGES_SUMMARY:") {
        let raw = &agent_output.response[pos + 16..];
        let line = if let Some(end) = raw.find('\n') {
            raw[..end].trim()
        } else {
            raw.trim()
        };
        return strip_markdown_formatting(line);
    }
    "Changes made to address reviewer feedback.".to_string()
}

/// Strips leading `Resolved without code changes:` / `Addressed feedback:` from the posted reply.
/// If there is no substantive text after that prefix, returns the original string unchanged.
fn strip_worker_reply_boilerplate(text: &str) -> String {
    if let Some(block) = extract_public_comment_block(text) {
        return block;
    }

    fn strip_diff_highlights_block(s: &str) -> String {
        let mut out: Vec<&str> = Vec::new();
        for line in s.lines() {
            if line.trim().eq_ignore_ascii_case("Diff highlights:") {
                break;
            }
            out.push(line);
        }
        out.join("\n").trim().to_string()
    }

    let trimmed = text.trim();
    const PREFIXES: &[&[u8]] = &[b"resolved without code changes:", b"addressed feedback:"];
    for prefix in PREFIXES {
        let b = trimmed.as_bytes();
        if b.len() >= prefix.len() && b[..prefix.len()].eq_ignore_ascii_case(prefix) {
            let suffix = trimmed[prefix.len()..].trim();
            if suffix.is_empty() {
                return trimmed.to_string();
            }
            let cleaned = strip_diff_highlights_block(suffix);
            let base = if cleaned.is_empty() { suffix } else { &cleaned };
            let sanitized = base.trim();
            return if sanitized.is_empty() {
                "Addressed the requested feedback.".to_string()
            } else {
                sanitized.to_string()
            };
        }
    }
    let cleaned = strip_diff_highlights_block(trimmed);
    let base = if cleaned.is_empty() {
        trimmed
    } else {
        &cleaned
    };
    let sanitized = base.trim();
    if sanitized.is_empty() {
        "Addressed the requested feedback.".to_string()
    } else {
        sanitized.to_string()
    }
}

/// Parses `MARK_DISCUSSIONS_RESOLVED: yes|no` from the agent response (case-insensitive value).
fn parse_mark_discussions_resolved(agent_output: &AgentHandoff) -> Option<bool> {
    // Structured field first (from the `handoff` tool).
    if let Some(ho) = handoff_output(agent_output)
        && let Some(v) = ho
            .get("mark_discussions_resolved")
            .and_then(serde_json::Value::as_bool)
    {
        return Some(v);
    }
    // Fallback: text marker `MARK_DISCUSSIONS_RESOLVED: yes/no`.
    const KEY: &str = "mark_discussions_resolved:";
    for line in agent_output.response.lines() {
        let lower = line.to_lowercase();
        if let Some(idx) = lower.find(KEY) {
            let val = line[idx + KEY.len()..].trim();
            let v = val.to_lowercase();
            if matches!(v.as_str(), "yes" | "true" | "1") {
                return Some(true);
            }
            if matches!(v.as_str(), "no" | "false" | "0") {
                return Some(false);
            }
        }
    }
    None
}

fn should_post_plain_comment(agent_output: &AgentHandoff) -> bool {
    // Structured field first (from the `handoff` tool).
    if let Some(ho) = handoff_output(agent_output)
        && let Some(v) = ho
            .get("post_plain_comment")
            .and_then(serde_json::Value::as_bool)
    {
        return v;
    }
    // Fallback: text marker `POST_PLAIN_COMMENT: yes/no`.
    const KEY: &str = "post_plain_comment:";
    for line in agent_output.response.lines() {
        let lower = line.to_lowercase();
        if let Some(idx) = lower.find(KEY) {
            let val = line[idx + KEY.len()..].trim();
            let v = val.to_lowercase();
            return matches!(v.as_str(), "yes" | "true" | "1");
        }
    }
    false
}

/// Whether to call GitLab `resolve` on discussions after posting the worker reply.
fn should_resolve_mr_feedback_discussions(
    agent_output: &AgentHandoff,
    implicit_from_actions: bool,
) -> bool {
    parse_mark_discussions_resolved(agent_output).unwrap_or(implicit_from_actions)
}

/// True when MR title, description, or labels differ between two GitLab snapshots (e.g. metadata
/// edit, label added/removed).
fn merge_request_surface_changed(
    before: &crate::agents::gitlab::MergeRequest,
    after: &crate::agents::gitlab::MergeRequest,
) -> bool {
    before.title.trim() != after.title.trim()
        || before.description.trim() != after.description.trim()
        || before.labels != after.labels
}

fn build_feedback_resolution_reply(
    agent_output: &AgentHandoff,
    has_new_changes: bool,
    diff_highlights: Option<&str>,
) -> Option<String> {
    if has_new_changes {
        let summary = extract_changes_summary(agent_output);
        if let Some(diff) = diff_highlights
            && !diff.trim().is_empty()
        {
            return Some(format!(
                "Addressed feedback:\n\n{}\n\nDiff highlights:\n{}",
                summary, diff
            ));
        }
        return Some(format!("Addressed feedback:\n\n{}", summary));
    }
    extract_no_change_resolution_reason(agent_output)
        .map(|reason| format!("Resolved without code changes:\n\n{}", reason))
}

fn build_diff_highlights_since(git_repo: &GitRepo, base_ref: &str) -> Option<String> {
    let files = git_repo.changed_files_since(base_ref).ok()?;
    if files.is_empty() {
        return None;
    }
    let shortstat = git_repo.diff_shortstat_since(base_ref).unwrap_or_default();

    const MAX_FILES: usize = 8;
    let mut lines: Vec<String> = files
        .iter()
        .take(MAX_FILES)
        .map(|f| format!("- {}", f))
        .collect();
    if files.len() > MAX_FILES {
        lines.push(format!("- ... and {} more files", files.len() - MAX_FILES));
    }
    if !shortstat.trim().is_empty() {
        lines.push(format!("- {}", shortstat.trim()));
    }
    Some(lines.join("\n"))
}

fn build_mr_diff_context(
    project_name: &str,
    mr: &crate::agents::gitlab::MergeRequest,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
) -> String {
    const MAX_DIFF_CHARS: usize = 120_000;
    const MAX_FILES: usize = 200;

    let mut source_note = "Source: GitLab MR changes API (matches MR diff view).".to_string();
    let (diff_stat, changed_files, mut diff_patch, overflow_note, diff_refs_line) =
        match gitlab.get_merge_request_changes(mr.iid) {
            Ok(snapshot) => {
                let mut files = snapshot.files;
                if files.len() > MAX_FILES {
                    files.truncate(MAX_FILES);
                }
                let stat = if files.is_empty() {
                    "(no changed files in MR changes payload)".to_string()
                } else {
                    format!("{} files changed", files.len())
                };
                let overflow_note = if snapshot.overflow {
                    Some("GitLab reported diff overflow; parts of the MR diff may be omitted.")
                } else {
                    None
                };
                let diff_refs = format!(
                    "Base commit: {}\nStart commit: {}\nHead commit: {}",
                    snapshot.base_sha.as_deref().unwrap_or("(unknown)"),
                    snapshot.start_sha.as_deref().unwrap_or("(unknown)"),
                    snapshot.head_sha.as_deref().unwrap_or("(unknown)")
                );
                (stat, files, snapshot.patch, overflow_note, Some(diff_refs))
            }
            Err(e) => {
                source_note = format!(
                    "Source: local git fallback (failed to read GitLab MR changes API: {}).",
                    e
                );
                (
                    git_repo
                        .diff_stat_against(&mr.target_branch)
                        .unwrap_or_else(|err| format!("(failed to compute diff stat: {})", err)),
                    git_repo
                        .changed_files_against(&mr.target_branch)
                        .unwrap_or_default(),
                    git_repo
                        .diff_patch_against(&mr.target_branch)
                        .unwrap_or_else(|err| format!("(failed to compute diff patch: {})", err)),
                    None,
                    None,
                )
            }
        };

    if diff_patch.len() > MAX_DIFF_CHARS {
        truncate_utf8_string_in_place(&mut diff_patch, MAX_DIFF_CHARS);
        diff_patch.push_str(
            "\n\n[diff truncated by potlatch: patch exceeded size limit; full diff may contain additional context]\n",
        );
    }
    if diff_patch.trim().is_empty() {
        diff_patch = "(No patch diff available.)".to_string();
    }
    let files_block = if changed_files.is_empty() {
        "No changed files detected.".to_string()
    } else {
        changed_files
            .into_iter()
            .map(|f| format!("- {}", f))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "# Merge Request Diff Context\n\nProject: {project}\nMR: !{iid} {title}\nSource branch: {source}\nTarget branch: {target}\n\n{source_note}\n\n## MR diff refs\n{diff_refs}\n\n## Diff stat\n{stat}\n\n## Changed files\n{files}\n\n## Patch\n```diff\n{patch}\n```\n{overflow_line}",
        project = project_name,
        iid = mr.iid,
        title = mr.title,
        source = mr.source_branch,
        target = mr.target_branch,
        source_note = source_note,
        diff_refs = diff_refs_line.unwrap_or("Unavailable (local git fallback)".to_string()),
        stat = if diff_stat.trim().is_empty() {
            "(no stat output)"
        } else {
            diff_stat.trim()
        },
        files = files_block,
        patch = diff_patch,
        overflow_line = overflow_note.unwrap_or("")
    )
}

/// Truncate a UTF-8 string to at most `max_bytes` without splitting a multibyte character.
fn truncate_utf8_string_in_place(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

fn format_comments_for_prompt(comments: &[crate::agents::gitlab::Comment]) -> String {
    comments
        .iter()
        .map(|c| c.format_for_prompt())
        .collect::<Vec<_>>()
        .join("\n")
}

struct CombinedMrFeedbackContextInput<'a> {
    project_name: &'a str,
    mr: &'a crate::agents::gitlab::MergeRequest,
    issue_context: &'a str,
    implementation_summary: &'a str,
    merge_conflict_status: &'a str,
    unresolved_comments_text: &'a str,
    plain_comments_text: &'a str,
    all_comments_text: &'a str,
    diff_context: &'a str,
}

fn build_combined_mr_feedback_context(input: CombinedMrFeedbackContextInput<'_>) -> String {
    format!(
        "# Merge Request Feedback + Diff Context\n\nProject: {project_name}\nMR: !{mr_iid} {mr_title}\nSource branch: {source_branch}\nTarget branch: {target_branch}\n\n{merge_conflict_status}\n\n## Original issue context\n{issue_context}\n\n## Original implementation summary\n{implementation_summary}\n\n## MR description\n{mr_description}\n\n## Unresolved MR comments to address\n{unresolved_comments_text}\n\n## Plain MR comments to consider\n{plain_comments_text}\n\n## Full MR comment history for context\n{all_comments_text}\n\n## MR diff context\n{diff_context}\n",
        project_name = input.project_name,
        mr_iid = input.mr.iid,
        mr_title = input.mr.title,
        source_branch = input.mr.source_branch,
        target_branch = input.mr.target_branch,
        merge_conflict_status = input.merge_conflict_status,
        issue_context = input.issue_context,
        implementation_summary = input.implementation_summary,
        mr_description = input.mr.description,
        unresolved_comments_text = if input.unresolved_comments_text.trim().is_empty() {
            "No unresolved comments.".to_string()
        } else {
            input.unresolved_comments_text.to_string()
        },
        plain_comments_text = if input.plain_comments_text.trim().is_empty() {
            "No plain MR comments.".to_string()
        } else {
            input.plain_comments_text.to_string()
        },
        all_comments_text = if input.all_comments_text.trim().is_empty() {
            "No MR comments.".to_string()
        } else {
            input.all_comments_text.to_string()
        },
        diff_context = input.diff_context
    )
}

fn build_merge_conflict_status_section(
    mr: &crate::agents::gitlab::MergeRequest,
    requires_conflict_resolution: bool,
    local_merge_clean: bool,
    git_repo: &GitRepo,
) -> Result<String> {
    let unmerged = git_repo.list_unmerged_paths()?;
    let marker_files = git_repo.list_conflict_marker_files()?;
    let target_sha = git_repo
        .remote_short_sha(&mr.target_branch)
        .unwrap_or_else(|_| "(unknown)".to_string());
    let source_sha = git_repo
        .remote_short_sha(&mr.source_branch)
        .unwrap_or_else(|_| "(unknown)".to_string());
    let mut lines = vec![
        "## Merge conflict status (verified by Potlatch — trust this section)".to_string(),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.target_branch, target_sha
        ),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.source_branch, source_sha
        ),
        format!(
            "- GitLab reports merge conflicts on this MR: {}",
            if mr.has_conflicts { "yes" } else { "no" }
        ),
        format!(
            "- Local merge of `origin/{}` into `origin/{}` at task start: {}",
            mr.target_branch,
            mr.source_branch,
            if local_merge_clean {
                "clean (no conflict markers introduced locally)"
            } else {
                "FAILED — conflict markers and/or unmerged paths are present in your workspace"
            }
        ),
        format!(
            "- Merge currently in progress in workspace: {}",
            if git_repo.is_merge_in_progress().unwrap_or(false) {
                "yes — resolve every unmerged file, then Potlatch will conclude the merge commit"
            } else {
                "no"
            }
        ),
        format!(
            "- Conflict resolution required this run: {}",
            if requires_conflict_resolution {
                "yes"
            } else {
                "no"
            }
        ),
    ];

    if unmerged.is_empty() {
        lines.push("- Unmerged paths in workspace: none".to_string());
    } else {
        lines.push("- Unmerged paths in workspace:".to_string());
        for path in &unmerged {
            lines.push(format!("  - {path}"));
        }
    }

    if marker_files.is_empty() {
        lines.push("- Files containing `<<<<<<<` conflict markers: none".to_string());
    } else {
        lines.push("- Files containing `<<<<<<<` conflict markers:".to_string());
        for path in &marker_files {
            lines.push(format!("  - {path}"));
        }
    }

    if requires_conflict_resolution {
        lines.push(
            "- After editing conflicted files, remove every conflict marker. Potlatch will `git add` resolved files and conclude the merge commit; you do not need to run git commands.".to_string(),
        );
    }

    Ok(lines.join("\n"))
}

fn feedback_discussions_may_be_resolved(
    agent_output: &AgentHandoff,
    implicit_from_actions: bool,
    conflicts_unresolved: bool,
) -> bool {
    if conflicts_unresolved {
        return false;
    }
    should_resolve_mr_feedback_discussions(agent_output, implicit_from_actions)
}

fn extract_no_change_resolution_reason(agent_output: &AgentHandoff) -> Option<String> {
    if let Some(reason) = &agent_output.reason {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            return Some(strip_markdown_formatting(trimmed));
        }
    }
    if let Some(pos) = agent_output.response.find("REASON:") {
        let raw = &agent_output.response[pos + 7..];
        let line = if let Some(end) = raw.find('\n') {
            raw[..end].trim()
        } else {
            raw.trim()
        };
        let cleaned = strip_markdown_formatting(line);
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    if let Some(s) = handoff_output(agent_output).and_then(|ho| {
        ho.get("changes_summary")
            .and_then(serde_json::Value::as_str)
    }) {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return Some(strip_markdown_formatting(trimmed));
        }
    }
    if let Some(pos) = agent_output.response.find("CHANGES_SUMMARY:") {
        let raw = &agent_output.response[pos + 16..];
        let line = if let Some(end) = raw.find('\n') {
            raw[..end].trim()
        } else {
            raw.trim()
        };
        let cleaned = strip_markdown_formatting(line);
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

fn format_issue_comments_for_worker_context(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let comments = match gitlab.get_issue_comments(issue_iid) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Worker: failed to fetch GitLab issue comments for #{}: {}",
                issue_iid, e
            );
            Vec::new()
        }
    };
    if comments.is_empty() {
        "_No comments on this issue yet._\n".to_string()
    } else {
        comments
            .iter()
            .map(|c| c.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

fn worker_issue_context_markdown(issue: &Issue, gitlab_comments_text: &str) -> String {
    format!(
        "# Issue Context\n\nIssue: #{} {}\n\n## Description\n{}\n\n## GitLab issue comments\n\n{}\n",
        issue.iid, issue.title, issue.description, gitlab_comments_text
    )
}

fn build_implementation_prompt(
    state: &AgentState,
    issue: &Issue,
    gitlab_comments_text: &str,
) -> Result<String> {
    let context_content = worker_issue_context_markdown(issue, gitlab_comments_text);
    // Write to disk for archival, but inject content into the prompt.
    let _context_path = write_task_context_file(
        &state.sessions_dir,
        &format!("{}-issue-{}.md", state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(false);
    let output_format = get_output_format();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are implementing a feature for a software project in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CONTEXT:
- The task context above is included in your prompt: it contains this issue's **description** and **every GitLab issue comment** at the time the task started. That is your primary written spec.
- Labels on the issue (e.g. priority) are visible in GitLab; infer scope from description + comments + `AGENTS.md`.

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. The task context is already included above. Do NOT use read on it — review it from your prompt, then start implementing.
3. Analyze the issue and comments carefully
4. Estimate the number of changed lines:
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
5. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the feature can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be implemented as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
6. If the issue is unclear or missing critical information that makes implementation impossible, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
7. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
8. If at any point you determine the issue simply cannot be implemented without additional human input that you cannot infer or assume (e.g. missing API credentials, undocumented external system dependencies, contradictory requirements), respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain precisely what input is needed and why you cannot proceed>
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the issue, REJECT it. Do not guess or produce speculative code.
- If you believe the implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

9. If the issue is clear, focused, and reasonably sized (or cannot be split), implement ONLY what is asked
10. Make all necessary code changes autonomously
11. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
12. {}

Proceed with the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_content,
        common_requirements,
        scope_rules,
        output_format
    );

    Ok(prompt)
}

fn build_continuation_prompt(
    state: &AgentState,
    issue: &Issue,
    gitlab_comments_text: &str,
) -> Result<String> {
    let context_content = worker_issue_context_markdown(issue, gitlab_comments_text);
    // Write to disk for archival, but inject content into the prompt.
    let _context_path = write_task_context_file(
        &state.sessions_dir,
        &format!("{}-issue-{}.md", &state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(true);
    let output_format = get_output_format();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are continuing work on an existing feature branch in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CONTEXT:
- The task context above contains this issue's **description** and **every GitLab issue comment** at task start.
- A branch for this issue already exists with previous work
- You are continuing the implementation from where it was left off
- Review the existing code changes in this branch
- Complete any remaining work needed to fully implement the issue

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. The task context is already included above. Do NOT use read on it — review it from your prompt.
3. Review the existing changes in the current branch
4. Analyze what has been done and what remains
5. Estimate total changed lines (including existing + remaining work):
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
6. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the remaining work can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be completed as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
7. If the issue is unclear or missing critical information that makes implementation impossible, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
8. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
9. If at any point you determine the remaining work simply cannot be completed without additional human input that you cannot infer or assume, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain precisely what input is needed and why you cannot proceed>
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the remaining work, REJECT it. Do not guess or produce speculative code.
- If you believe the total implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

10. If the issue is clear, focused, and reasonably sized (or cannot be split), continue the implementation
11. ONLY implement what the issue asks for, nothing more
12. Complete any remaining work autonomously
13. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
14. {}

Proceed with continuing the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_content,
        common_requirements,
        scope_rules,
        output_format
    );

    Ok(prompt)
}

fn get_common_requirements() -> &'static str {
    r#"CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You MUST delete, rename, move, or create files as needed — do not ask permission or suggest it
- You MUST NOT ask the user for input, confirmation, or decisions — decide autonomously
- You MUST NOT produce output that suggests actions for a human to take — YOU take those actions
- Leave staging, committing, pushing, and merge request creation to the system
- If information is missing, document what's needed in your response (do not ask interactively)
- If you are making code changes you MUST stick to AGENTS.md in the project strictly
- Read the issue comments carefully — they may contain guidance from the PMO agent on how to proceed. PMO guidance appears as a comment starting with **PMO guidance for the worker agent:** — treat the body of that comment as authoritative worker instructions and follow it exactly.
- Before finishing, update repo-root notes.md only when you have bullets that pass the NOTES.MD rules (see MANDATORY OUTPUT): not a recap of your MR, not generic best-practice slides, not meta about notes — if nothing qualifies, leave the file unchanged. Never paste notes.md into MR metadata or GitLab comments
- ACT QUICKLY. The task context is already in your prompt — do not read it from disk. After reading AGENTS.md, grep for the first symbol you need to change, read the surrounding lines, and make the edit. Do not write long analysis prose — each turn should produce a tool call that makes progress (a grep, a read, or an edit). If you catch yourself writing more than 2 sentences of "Let me analyze..." or "I notice that...", stop and make the edit instead. The first edit should happen within your first 5 tool calls."#
}

fn get_evidence_bound_scope_bullets() -> &'static str {
    r#"- Only modify code that is strongly supported by the issue title, issue description, GitLab issue comments, or current unresolved MR feedback. If a change is merely adjacent, speculative, weakly coupled, or "nice to have", do not make it.
- Treat the issue title, issue description, and GitLab issue comments as the strict boundary of allowed code changes.
- Every production code change must have a clear, direct link to those issue details or comments; if you cannot explain that link in one sentence, do not make the change.
- Do NOT add features, refactors, or integrations not described in the issue.
- Do NOT over-engineer: avoid new abstractions, compatibility layers, broad refactors, generalized frameworks, hypothetical future cases, unrelated edge cases, broad compatibility, "while here" cleanup, or behavior changes that are not directly required by the task evidence.
- The only acceptable unrelated edits are lint-only fixes or unit-test-only fixes needed to validate the requested implementation."#
}

fn get_feedback_scope_rules() -> String {
    format!(
        r#"FEEDBACK SCOPE RULES:
- Treat the current unresolved reviewer feedback as the PRIMARY request. The original issue, MR description, and older comments provide context, but they do not override the current unresolved feedback.
- Do NOT mutate, reinterpret, or weaken the current feedback to fit an old implementation choice or old commit. Change the code so it strictly complies with the reviewer feedback.
- Scope each change directly to the unresolved comment, its inline code location, the linked issue, or the MR description. Do not use feedback handling as an opportunity for broader cleanup, redesign, abstraction, feature expansion, or compatibility shims unless the reviewer explicitly asked for it.
{}"#,
        get_evidence_bound_scope_bullets()
    )
}

fn get_scope_rules(is_continuation: bool) -> String {
    let line_context = if is_continuation {
        "Review existing changes and estimate remaining work"
    } else {
        "Before starting implementation, estimate if the changes will be significantly larger than the limits below"
    };

    format!(
        r#"SCOPE RULES:
- ONLY implement what the issue specifically asks for, nothing more
{}
- CHANGE SIZE LIMITS (STRICT):
  * Non-test, non-generated code: ~500 changed lines maximum
  * Total changes including tests: ~1500 changed lines maximum
  * Do NOT count auto-generated code (files with "generated by", "auto-generated", "DO NOT EDIT" comments) toward either limit
- {}
- If non-test code changes would be substantially larger than ~500 lines, or total changes larger than ~1500 lines:
  * First, carefully evaluate if the feature can be split into smaller, independent pieces
  * If you are VERY SURE the feature CANNOT be split and MUST be {} as one atomic unit, you may proceed
  * Otherwise, respond with:
    CANNOT_IMPLEMENT
    NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
- If implementing the issue requires a large feature integration that is mainly unrelated to the task, respond with:
  CANNOT_IMPLEMENT
  NEEDS_SPLIT: <explain why the issue is too broad and how to split it>"#,
        get_evidence_bound_scope_bullets(),
        line_context,
        if is_continuation {
            "completed"
        } else {
            "implemented"
        }
    )
}

fn get_output_format() -> &'static str {
    r#"MANDATORY OUTPUT — call the `handoff` tool with your output fields. This is the primary output channel — Potlatch reads the tool's JSON, not text markers in your response.

Call `handoff` with the fields relevant to your outcome:

- `mr_title` (string): Short title (max 8-10 words) stating the main feature or fix. Focus on WHAT, not HOW or HOW MUCH. No markdown, no **, no backticks.
- `mr_description` (string): Full MR description in markdown with ## Goal, ## Implementation, ## Testing sections.
- `changes_summary` (string): A concise sentence summarizing the substance of changes made.
- `depends_on_issue` (integer): IID of a dependency issue that must close before this work can proceed. Only set when the dependency is real — your work cannot proceed until the other issue is closed. The system will park this issue until the dependency closes, then resume it automatically.
- `needs_split` (string): Reason the issue needs splitting into smaller issues.
- `needs_clarification` (string): What information is needed from a human to proceed.
- `cannot_implement` (boolean): Set to true when the issue cannot be implemented.
- `cannot_resolve` (boolean): Set to true when reviewer feedback cannot be resolved autonomously.
- `reason` (string): Explanation for cannot_implement, cannot_resolve, or no-code-changes.
- `public_comment` (string): Human-facing GitLab comment text (separate from MR description).
- `mark_discussions_resolved` (boolean): Whether to mark open review discussions as resolved.
- `post_plain_comment` (boolean): Whether to post a new plain MR comment.

All fields are optional — include only the ones relevant to your outcome. Call `handoff` exactly once when you're done.

TEXT MARKER FALLBACK — if for any reason you cannot call the `handoff` tool, you may use these text markers instead (the system parses them as a fallback):

MR_TITLE: <title>
MR_DESCRIPTION:
## Goal
<goal>
## Implementation
<approach>
## Testing
<testing>

Stable alternative:
MR_TITLE_BEGIN
<title text>
MR_TITLE_END
MR_DESCRIPTION_BEGIN
<full markdown description text>
MR_DESCRIPTION_END

DEPENDS_ON_ISSUE: #<N>

PUBLIC_COMMENT_BEGIN
<final public comment only>
PUBLIC_COMMENT_END

The `handoff` tool is preferred — use text markers only as a last resort.

Do NOT put PUBLIC_COMMENT text inside mr_description — the MR description must be plain documentation (goal, implementation, testing); reply text belongs in the `public_comment` field or a separate PUBLIC_COMMENT block.

NOTES.MD (agent-maintained in the repo — edit before you finish **only if** you earn real bullets):
- Open or create notes.md at the repository root. Append **0–3** new "- " lines this run (often **0**). Each line is **one** short sentence capturing a **genuine surprise, near-mistake, or emotional friction** from the run — something you almost got wrong or that wasted time — expressed so a stranger learns the *habit of noticing*, not the *contents of this MR*.

HARD REJECT (if a line violates any of these, delete it — do not append):
- Backticks, file or directory paths, dotted import paths, or shell snippets tied to this repo layout.
- Names of this project's classes, functions, modules, config keys, env vars, or third-party symbols (anything a reader would grep for in *this* tree).
- Documentation-style explanation of how this tree works — put that in MR description or code, not notes.
- Obvious restatements of AGENTS.md / the issue / standard practice, and hollow platitudes ("write tests", "read carefully").
- **Abstract recap of your implementation** dressed as advice: if the line is basically "what we changed" or "how we tested" in generic software-engineering words (single source of truth for identifiers, test through public API, stub optional deps, match CI cwd when running tests, etc.) **without** a sharp *I almost messed this up because…* insight, it is **rubbish** — delete it.
- **Meta about notes.md** (e.g. "keep notes short and transferable", "agent-maintained") — never; that parrots instructions, not experience.
- Anything that could appear unchanged on a generic "best practices" slide deck with no story of *what tripped you* — delete it.

SELF-CHECK before saving: (1) "Would this still help on a **different** repo without opening our source?" If no, drop. (2) "Could I have written this bullet **before** starting this task?" If yes, drop. If **no** line survives, append nothing this run.

BAD STYLE (examples of rubbish — do not imitate):
- In-repo code tours (paths, classes, long semicolon chains).
- "Timeless" bullets that are really your MR summary: composable naming over literals, property vs field assumptions, stub heavy imports, or environment wiring — unless each line names a **non-obvious failure mode you personally hit** in **one** concrete clause (still without paths or symbol names).

Stay concise; no secrets. That file is committed with your other changes. Never paste or quote any text from notes.md into MR_TITLE, MR_DESCRIPTION, PUBLIC_COMMENT, or anywhere on GitLab — those surfaces are for humans/reviewers only."#
}

fn extract_split_reason(agent_output: &AgentHandoff) -> String {
    if let Some(block) = extract_worker_public_comment(agent_output) {
        return block;
    }
    if let Some(reason) = handoff_output(agent_output)
        .and_then(|ho| ho.get("needs_split").and_then(serde_json::Value::as_str))
    {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(pos) = agent_output.response.find("NEEDS_SPLIT:") {
        let reason = &agent_output.response[pos + 12..];
        return reason.trim().to_string();
    }
    "This issue is too broad and requires large unrelated feature work. Please split it into smaller, focused issues with detailed descriptions.".to_string()
}

fn extract_clarification(agent_output: &AgentHandoff) -> String {
    if let Some(block) = extract_worker_public_comment(agent_output) {
        return block;
    }
    if let Some(clarification) = &agent_output.needs_clarification {
        let trimmed = clarification.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(pos) = agent_output.response.find("NEEDS_CLARIFICATION:") {
        let clarification = &agent_output.response[pos + 20..];
        if let Some(end) = clarification.find('\n') {
            return clarification[..end].trim().to_string();
        }
        return clarification.trim().to_string();
    }
    "This issue needs clarification. Please provide more details.".to_string()
}

/// Get a reference to the `handoff` tool's captured JSON from
/// `agent_output.structured_outputs`, or `None` if the tool wasn't called.
fn handoff_output(agent_output: &AgentHandoff) -> Option<&serde_json::Value> {
    agent_output
        .structured_outputs
        .as_ref()
        .and_then(|s| s.get("handoff"))
}

/// Extract the worker's public comment text. Checks the `handoff` tool's
/// `public_comment` field first, then falls back to the `PUBLIC_COMMENT_BEGIN…
/// END` text marker.
fn extract_worker_public_comment(agent_output: &AgentHandoff) -> Option<String> {
    if let Some(ho) = handoff_output(agent_output)
        && let Some(s) = ho.get("public_comment").and_then(serde_json::Value::as_str)
    {
        let t = s.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    extract_public_comment_block(&agent_output.response)
}

/// Apply the structured JSON the model emitted via the `handoff` tool,
/// populating the shared `AgentHandoff` fields that the worker and other
/// agents read (`decision`, `reason`, `needs_clarification`,
/// `depends_on_issue`). Worker-specific fields (`mr_title`, `mr_description`,
/// `changes_summary`, `needs_split`, `public_comment`, etc.) are read directly
/// from `structured_outputs["handoff"]` by the worker's own extractors — they
/// don't live on `AgentHandoff`.
fn apply_worker_handoff(mut handoff: AgentHandoff) -> AgentHandoff {
    let Some(ho) = handoff_output(&handoff).cloned() else {
        return handoff;
    };

    if let Some(s) = ho
        .get("needs_clarification")
        .and_then(serde_json::Value::as_str)
    {
        let t = s.trim();
        if !t.is_empty() {
            handoff.needs_clarification = Some(t.to_string());
        }
    }
    if let Some(s) = ho.get("reason").and_then(serde_json::Value::as_str) {
        let t = s.trim();
        if !t.is_empty() {
            handoff.reason = Some(t.to_string());
        }
    }
    if let Some(n) = ho
        .get("depends_on_issue")
        .and_then(serde_json::Value::as_u64)
        .filter(|n| *n > 0)
    {
        handoff.depends_on_issue = Some(n);
    }
    if ho
        .get("cannot_implement")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        handoff.decision = Some("cannot_implement".to_string());
    }
    if ho
        .get("cannot_resolve")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        handoff.decision = Some("cannot_resolve".to_string());
    }

    handoff
}

/// Extract a dependency issue IID from the agent's output. Checks the
/// structured `depends_on_issue` field first (populated by `apply_worker_handoff`),
/// then falls back to the `DEPENDS_ON_ISSUE:` text marker and prose patterns.
fn extract_depends_on_issue(agent_output: &AgentHandoff) -> Option<u64> {
    // Structured field first (populated by `apply_worker_handoff` from the
    // `handoff` tool's `depends_on_issue` field).
    if let Some(n) = agent_output.depends_on_issue {
        return Some(n);
    }

    let response = &agent_output.response;

    // Fallback: explicit marker `DEPENDS_ON_ISSUE: #N`.
    if let Some(pos) = response.find("DEPENDS_ON_ISSUE:") {
        let rest = &response[pos + "DEPENDS_ON_ISSUE:".len()..];
        let rest = rest.trim_start();
        let rest = rest.strip_prefix('#').unwrap_or(rest);
        let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num.parse::<u64>() {
            return Some(n);
        }
    }

    // Fallback: prose declarations like "blocked on #727", "depends on #727",
    // "cannot proceed until #727 is closed". The model often describes the
    // dependency without using the explicit marker, so scan for issue
    // references preceded by dependency keywords.
    extract_depends_on_issue_from_prose(response)
}

/// Scan prose for dependency phrases followed by `#<N>` issue references.
/// Matches patterns like "blocked on #727", "depends on #727",
/// "cannot proceed until #727", "waiting on #727", "blocked by #727".
fn extract_depends_on_issue_from_prose(text: &str) -> Option<u64> {
    let lower = text.to_lowercase();
    let keywords = [
        "blocked on",
        "blocked by",
        "depends on",
        "waiting on",
        "waiting for",
        "cannot proceed until",
        "cannot start until",
        "hard-blocked on",
        "blocked until",
    ];
    for kw in keywords {
        let mut search_from = 0;
        while let Some(pos) = lower[search_from..].find(kw) {
            let abs = search_from + pos + kw.len();
            let rest = &text[abs..];
            // Skip whitespace, then expect '#'.
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('#') {
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = num.parse::<u64>()
                    && n > 0
                {
                    return Some(n);
                }
            }
            search_from = abs;
        }
    }
    None
}

/// Prefix for labels that park an issue until a dependency issue is closed.
/// The full label is `waiting-on-issue:#N` where N is the dependency issue IID.
const WAITING_ON_ISSUE_LABEL_PREFIX: &str = "waiting-on-issue:#";

/// Build the `waiting-on-issue:#N` label for a dependency issue IID.
fn waiting_on_issue_label(issue_iid: u64) -> String {
    format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{issue_iid}")
}

/// Extract the dependency issue IID from a `waiting-on-issue:#N` issue label.
fn extract_waiting_on_issue_iid(labels: &[String]) -> Option<u64> {
    labels.iter().find_map(|l| {
        l.strip_prefix(WAITING_ON_ISSUE_LABEL_PREFIX)
            .and_then(|s| s.parse::<u64>().ok())
    })
}

/// Extract the MR title the agent emitted, falling back to the issue title
/// (not a generic placeholder like "Implementation changes") when the agent
/// didn't provide one. The issue title is always meaningful and specific to
/// the work, so it's a far better default than a placeholder that produces
/// a stream of indistinguishable MRs.
fn extract_mr_title(agent_output: &AgentHandoff, issue_title: &str) -> String {
    // Structured field first (from the `handoff` tool).
    if let Some(s) = handoff_output(agent_output)
        .and_then(|ho| ho.get("mr_title").and_then(serde_json::Value::as_str))
    {
        let cleaned = strip_markdown_formatting(s.trim());
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    if let Some(block) =
        extract_block_between_markers(&agent_output.response, "MR_TITLE_BEGIN", "MR_TITLE_END")
    {
        let cleaned = strip_markdown_formatting(block.trim());
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    // Try exact marker first
    if let Some(pos) = agent_output.response.find("MR_TITLE:") {
        let title_section = &agent_output.response[pos + 9..];
        let raw = if let Some(end) = title_section.find('\n') {
            title_section[..end].trim()
        } else {
            title_section.trim()
        };
        let cleaned = strip_markdown_formatting(raw);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }

    // Try common variations the agent might use
    for marker in &["Title:", "TITLE:", "## Title", "Commit message:"] {
        if let Some(pos) = agent_output.response.find(marker) {
            let section = &agent_output.response[pos + marker.len()..];
            let raw = if let Some(end) = section.find('\n') {
                section[..end].trim()
            } else {
                section.trim()
            };
            let cleaned = strip_markdown_formatting(raw);
            if !cleaned.is_empty() && cleaned.len() > 5 {
                return cleaned;
            }
        }
    }

    // Last resort: use the CHANGES_SUMMARY if present
    if let Some(pos) = agent_output.response.find("CHANGES_SUMMARY:") {
        let section = &agent_output.response[pos + 16..];
        let raw = if let Some(end) = section.find('\n') {
            section[..end].trim()
        } else {
            section.trim()
        };
        let cleaned = strip_markdown_formatting(raw);
        if !cleaned.is_empty() {
            return cleaned;
        }
    }

    // No title extracted from the agent output — fall back to the issue title,
    // which is always specific to the work. Never use a generic placeholder.
    issue_title.to_string()
}

/// Extract an explicit MR title the agent emitted (for metadata updates on
/// follow-up turns). Returns `None` when the agent didn't provide one, so the
/// caller can distinguish "no title provided" from "title provided". Unlike
/// [`extract_mr_title`], this does NOT fall back to CHANGES_SUMMARY or the
/// issue title — a metadata update should only overwrite the title when the
/// agent explicitly said to. Does NOT read from the `handoff` tool's
/// `mr_title` field — that is for MR creation, not metadata updates. Only
/// text markers (`MR_TITLE_BEGIN…END`, `MR_TITLE:`) are used here, so the
/// model must explicitly emit them when a reviewer asks for a title change.
fn extract_explicit_mr_title(agent_output: &AgentHandoff) -> Option<String> {
    if let Some(block) =
        extract_block_between_markers(&agent_output.response, "MR_TITLE_BEGIN", "MR_TITLE_END")
    {
        let cleaned = strip_markdown_formatting(block.trim());
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    if let Some(pos) = agent_output.response.find("MR_TITLE:") {
        let title_section = &agent_output.response[pos + 9..];
        let raw = if let Some(end) = title_section.find('\n') {
            title_section[..end].trim()
        } else {
            title_section.trim()
        };
        let cleaned = strip_markdown_formatting(raw);
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    None
}

fn strip_markdown_formatting(s: &str) -> String {
    let result = s
        .trim_start_matches('*')
        .trim_end_matches('*')
        .trim_start_matches('`')
        .trim_end_matches('`')
        .trim();
    result.to_string()
}

fn extract_block_between_markers(text: &str, begin: &str, end: &str) -> Option<String> {
    let start = text.find(begin)?;
    let body_start = start + begin.len();
    let rest = &text[body_start..];
    let end_rel = rest.find(end)?;
    let body = rest[..end_rel].trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_string())
    }
}

fn sanitize_mr_description_text(s: &str) -> String {
    let stripped = strip_public_comment_blocks(s);
    let filtered: Vec<&str> = stripped
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("CHANGES_SUMMARY:")
                && !t.starts_with("MARK_DISCUSSIONS_RESOLVED:")
                && !t.starts_with("POST_PLAIN_COMMENT:")
        })
        .collect();
    filtered.join("\n").trim().to_string()
}

fn extract_mr_description(agent_output: &AgentHandoff) -> String {
    // Structured field first (from the `handoff` tool).
    if let Some(s) = handoff_output(agent_output)
        .and_then(|ho| ho.get("mr_description").and_then(serde_json::Value::as_str))
    {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(trimmed);
        }
    }
    if let Some(block) = extract_block_between_markers(
        &agent_output.response,
        "MR_DESCRIPTION_BEGIN",
        "MR_DESCRIPTION_END",
    ) {
        return sanitize_mr_description_text(&block);
    }
    if let Some(pos) = agent_output.response.find("MR_DESCRIPTION:") {
        let desc_section = &agent_output.response[pos + 15..];
        let trimmed = desc_section.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(trimmed);
        }
    }

    if let Some(pos) = agent_output.response.find("IMPLEMENTATION_SUMMARY:") {
        let summary = &agent_output.response[pos + 23..];
        let trimmed = summary.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(&format!("## Implementation\n\n{}", trimmed));
        }
    }

    // Try to extract the last substantial paragraph as a summary
    let lines: Vec<&str> = agent_output.response.lines().collect();
    let last_chunk: Vec<&str> = lines
        .iter()
        .rev()
        .take(20)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .filter(|l| !l.trim().is_empty())
        .collect();
    if !last_chunk.is_empty() {
        return sanitize_mr_description_text(&format!("## Summary\n\n{}", last_chunk.join("\n")));
    }

    "Implementation completed.".to_string()
}

fn extract_explicit_mr_description(agent_output: &AgentHandoff) -> String {
    // Structured field first (from the `handoff` tool).
    if let Some(s) = handoff_output(agent_output)
        .and_then(|ho| ho.get("mr_description").and_then(serde_json::Value::as_str))
    {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(trimmed);
        }
    }
    if let Some(block) = extract_block_between_markers(
        &agent_output.response,
        "MR_DESCRIPTION_BEGIN",
        "MR_DESCRIPTION_END",
    ) {
        return sanitize_mr_description_text(&block);
    }
    if let Some(pos) = agent_output.response.find("MR_DESCRIPTION:") {
        let desc_section = &agent_output.response[pos + 15..];
        let trimmed = desc_section.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(trimmed);
        }
    }
    "Implementation completed.".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_issue_context_includes_comments_section() {
        let issue = Issue {
            iid: 7,
            title: "Add feature".to_string(),
            description: "Do the thing".to_string(),
            labels: vec![],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };
        let md = worker_issue_context_markdown(&issue, "- alice: hi");
        assert!(md.contains("## GitLab issue comments"));
        assert!(md.contains("- alice: hi"));
        assert!(md.contains("#7"));
        assert!(md.contains("Do the thing"));
    }

    #[test]
    fn test_should_skip_issue() {
        let mut issue = Issue {
            iid: 1,
            title: "[Draft] Test issue".to_string(),
            description: "Test".to_string(),
            labels: vec![],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };
        assert!(should_skip_issue(&issue));

        issue.title = "Draft: Test issue".to_string();
        assert!(should_skip_issue(&issue));

        issue.title = "Normal issue".to_string();
        issue.labels = vec![super::super::labels::DO_NOT_IMPLEMENT.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKING_ON_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![PMO_PROCESSED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![PMO_PENDING_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKER_PENDING_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKER_REVIEW_ONLY_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![];
        assert!(!should_skip_issue(&issue));
    }

    #[test]
    fn worker_pending_label_is_not_treated_as_skip_label_without_pending() {
        assert!(!has_worker_skip_label(&[WORKER_PENDING_LABEL.to_string()]));
        assert!(issue_has_worker_pending_label(&[
            WORKER_PENDING_LABEL.to_string()
        ]));
    }

    #[test]
    fn worker_review_only_label_is_not_resume_abandon_label() {
        assert!(issue_has_worker_review_only_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
        assert!(!has_worker_resume_abandon_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
        assert!(!has_worker_skip_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
    }

    #[test]
    fn worker_should_cancel_issue_processing_for_review_only_closed_or_pending() {
        let mut issue = Issue {
            iid: 9,
            title: "Test".into(),
            description: String::new(),
            state: "opened".into(),
            labels: vec![WORKER_REVIEW_ONLY_LABEL.to_string()],
            created_at: None,
            updated_at: None,
        };
        assert!(worker_should_cancel_issue_processing(&issue));

        issue.labels.clear();
        issue.state = "closed".into();
        assert!(worker_should_cancel_issue_processing(&issue));

        // `pending` label set by a human mid-run must cancel in-flight work.
        issue.state = "opened".into();
        issue.labels = vec![WORKER_PENDING_LABEL.to_string()];
        assert!(worker_should_cancel_issue_processing(&issue));

        // Opened with no blocking labels → keep working.
        issue.labels.clear();
        assert!(!worker_should_cancel_issue_processing(&issue));
    }

    #[test]
    fn truncate_utf8_string_in_place_does_not_split_multibyte_chars() {
        let mut patch = "α".repeat(60_000);
        patch.push_str(&"β".repeat(60_000));
        assert!(patch.len() > 120_000);
        truncate_utf8_string_in_place(&mut patch, 120_000);
        assert!(patch.len() <= 120_000);
        assert!(std::str::from_utf8(patch.as_bytes()).is_ok());
    }

    #[test]
    fn extract_issue_number_from_branch_for_mr_source() {
        assert_eq!(extract_issue_number_from_branch("issue-42").unwrap(), 42);
    }

    #[test]
    fn no_code_changes_retry_message_stays_internal() {
        let message = no_code_changes_retry_error_message(42);
        assert!(message.contains("retrying"));
        assert!(message.contains("without marking action-required"));
        assert!(!message.contains("may need more detail"));
        assert!(!message.contains("different approach"));
    }

    #[test]
    fn is_worker_agent_cancelled_matches_runtime_message() {
        let err = anyhow::anyhow!(WORKER_AGENT_CANCELLED_MSG);
        assert!(is_worker_agent_cancelled(&err));
        assert!(!is_worker_agent_cancelled(&anyhow::anyhow!(
            "other failure"
        )));
    }

    #[test]
    fn test_has_worker_resume_abandon_label() {
        assert!(!has_worker_resume_abandon_label(&[
            WORKING_ON_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            ACTION_REQUIRED_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            PMO_PROCESSED_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            PMO_PENDING_LABEL.to_string()
        ]));
        assert!(!has_worker_resume_abandon_label(&[
            "claimed:worker-0".to_string()
        ]));
    }

    #[test]
    fn test_has_worker_skip_label() {
        assert!(has_worker_skip_label(&[WORKING_ON_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[ACTION_REQUIRED_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[PMO_PROCESSED_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[PMO_PENDING_LABEL.to_string()]));
        assert!(!has_worker_skip_label(&["claimed:worker-0".to_string()]));
    }

    #[test]
    fn test_extract_explicit_mr_title_does_not_fall_back_to_changes_summary() {
        let output = AgentHandoff {
            response: "CHANGES_SUMMARY: Fix reviewer follow-up lint issue.".to_string(),
            ..Default::default()
        };
        // extract_explicit_mr_title returns None when no MR_TITLE marker is
        // present (it must not pick up CHANGES_SUMMARY).
        assert_eq!(extract_explicit_mr_title(&output), None);
        // extract_mr_title falls back to CHANGES_SUMMARY, then to the issue
        // title when no marker and no CHANGES_SUMMARY are present.
        assert_eq!(
            extract_mr_title(&output, "Issue title"),
            "Fix reviewer follow-up lint issue."
        );
    }

    #[test]
    fn extract_mr_title_falls_back_to_issue_title_when_no_marker() {
        // When the agent emits neither MR_TITLE nor CHANGES_SUMMARY, the MR
        // title falls back to the issue title — never a generic placeholder.
        let output = AgentHandoff {
            response: "I made some changes.".to_string(),
            ..Default::default()
        };
        assert_eq!(
            extract_mr_title(&output, "Add login rate limiting"),
            "Add login rate limiting"
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_includes_reason_without_code_changes() {
        let output = AgentHandoff {
            response: "REASON: Existing validation already covered this case.".to_string(),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&output, false, None),
            Some(
                "Resolved without code changes:\n\nExisting validation already covered this case."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_uses_changes_summary_when_changes_exist() {
        let output = AgentHandoff {
            response: "CHANGES_SUMMARY: Add missing null check in parser.".to_string(),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&output, true, None),
            Some("Addressed feedback:\n\nAdd missing null check in parser.".to_string())
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_includes_diff_highlights() {
        let output = AgentHandoff {
            response: "CHANGES_SUMMARY: Tighten input validation.".to_string(),
            ..Default::default()
        };
        let diff = "- src/validation.rs\n- 1 file changed, 4 insertions(+)";
        assert_eq!(
            build_feedback_resolution_reply(&output, true, Some(diff)),
            Some("Addressed feedback:\n\nTighten input validation.\n\nDiff highlights:\n- src/validation.rs\n- 1 file changed, 4 insertions(+)".to_string())
        );
    }

    #[test]
    fn build_feedback_resolution_reply_requires_reason_without_code_changes() {
        let output = AgentHandoff {
            response: "I'll inspect the reviewer feedback first.".to_string(),
            ..Default::default()
        };
        assert_eq!(build_feedback_resolution_reply(&output, false, None), None);
    }

    #[test]
    fn feedback_discussions_may_be_resolved_blocks_while_conflicts_remain() {
        let out = AgentHandoff {
            response: "MARK_DISCUSSIONS_RESOLVED: yes\n".to_string(),
            ..Default::default()
        };
        assert!(!feedback_discussions_may_be_resolved(&out, true, true));
        assert!(feedback_discussions_may_be_resolved(&out, true, false));
    }

    #[test]
    fn combined_mr_feedback_context_separates_unresolved_and_full_history() {
        let mr = crate::agents::gitlab::MergeRequest {
            iid: 287,
            title: "Add config".into(),
            description: "MR description".into(),
            source_branch: "issue-285".into(),
            target_branch: "main".into(),
            state: "opened".into(),
            sha: None,
            labels: None,
            has_conflicts: false,
        };
        let ctx = build_combined_mr_feedback_context(CombinedMrFeedbackContextInput {
            project_name: "project",
            mr: &mr,
            issue_context: "issue context",
            implementation_summary: "implementation summary",
            merge_conflict_status: "## Merge conflict status\n- GitLab reports merge conflicts on this MR: no",
            unresolved_comments_text: "- reviewer (discussion d1): fix this",
            plain_comments_text: "- reviewer (discussion d2): plain actionable note",
            all_comments_text: "- reviewer (discussion d1): fix this\n- maintainer (discussion d2): simple context",
            diff_context: "diff context",
        });

        assert!(ctx.contains("## Merge conflict status"));
        assert!(ctx.contains("## Unresolved MR comments to address"));
        assert!(ctx.contains("## Plain MR comments to consider"));
        assert!(ctx.contains("## Full MR comment history for context"));
        assert!(ctx.contains("fix this"));
        assert!(ctx.contains("plain actionable note"));
        assert!(ctx.contains("simple context"));
    }

    #[test]
    fn strip_worker_reply_boilerplate_removes_prefix_when_suffix_present() {
        assert_eq!(
            strip_worker_reply_boilerplate(
                "Resolved without code changes:\n\nAlready covered by tests."
            ),
            "Already covered by tests."
        );
        assert_eq!(
            strip_worker_reply_boilerplate("ADDRESSED FEEDBACK:\n\nFixed the null deref."),
            "Fixed the null deref."
        );
    }

    #[test]
    fn strip_worker_reply_boilerplate_keeps_prefix_when_nothing_follows() {
        assert_eq!(
            strip_worker_reply_boilerplate("Resolved without code changes:"),
            "Resolved without code changes:"
        );
        assert_eq!(
            strip_worker_reply_boilerplate("Resolved without code changes:\n  \n  "),
            "Resolved without code changes:"
        );
    }

    #[test]
    fn strip_worker_reply_boilerplate_keeps_all_lines_after_prefix() {
        let input = "Addressed feedback:\n\nI’ll run a full readonly review from local state.\nFixed null checks.\nError: S: [unavailable] Error";
        assert_eq!(
            strip_worker_reply_boilerplate(input),
            "I’ll run a full readonly review from local state.\nFixed null checks.\nError: S: [unavailable] Error"
        );
    }

    #[test]
    fn strip_worker_reply_boilerplate_uses_public_comment_block() {
        let input = "Addressed feedback:\n\nPUBLIC_COMMENT_BEGIN\nFinal reviewer reply.\nPUBLIC_COMMENT_END";
        assert_eq!(
            strip_worker_reply_boilerplate(input),
            "Final reviewer reply."
        );
    }

    #[test]
    fn should_resolve_mr_feedback_discussions_defaults_follow_implicit_actions() {
        let out = AgentHandoff::default();
        assert!(!should_resolve_mr_feedback_discussions(&out, false));
        assert!(should_resolve_mr_feedback_discussions(&out, true));
    }

    #[test]
    fn mark_discussions_resolved_parsed_from_response() {
        let out_no = AgentHandoff {
            response: "MARK_DISCUSSIONS_RESOLVED: no\n".to_string(),
            ..Default::default()
        };
        assert!(!should_resolve_mr_feedback_discussions(&out_no, true));
        let out_yes = AgentHandoff {
            response: "mark_discussions_resolved: YES\n".to_string(),
            ..Default::default()
        };
        assert!(should_resolve_mr_feedback_discussions(&out_yes, false));
    }

    #[test]
    fn plain_comment_posting_requires_explicit_marker() {
        let out_default = AgentHandoff {
            response: "PUBLIC_COMMENT_BEGIN\nNo further changes were needed.\nPUBLIC_COMMENT_END"
                .to_string(),
            ..Default::default()
        };
        assert!(!should_post_plain_comment(&out_default));

        let out_no = AgentHandoff {
            response: "POST_PLAIN_COMMENT: no\nPUBLIC_COMMENT_BEGIN\nNo further changes were needed.\nPUBLIC_COMMENT_END".to_string(),
            ..Default::default()
        };
        assert!(!should_post_plain_comment(&out_no));

        let out_yes = AgentHandoff {
            response: "post_plain_comment: YES\nPUBLIC_COMMENT_BEGIN\nPosted by request.\nPUBLIC_COMMENT_END".to_string(),
            ..Default::default()
        };
        assert!(should_post_plain_comment(&out_yes));
    }

    #[test]
    fn no_change_reply_text_requires_structured_reason_marker() {
        let out = AgentHandoff {
            response: "No new code changes were needed in this run. The branch already satisfies the requested behavior.".to_string(),
            ..Default::default()
        };
        assert_eq!(build_feedback_resolution_reply(&out, false, None), None);

        let explicit = AgentHandoff {
            response: "REASON: No new code changes were needed in this run. The branch already satisfies the requested behavior.".to_string(),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&explicit, false, None),
            Some("Resolved without code changes:\n\nNo new code changes were needed in this run. The branch already satisfies the requested behavior.".to_string())
        );
    }

    #[test]
    fn extract_no_change_resolution_reason_ignores_freeform_planning_text() {
        let out = AgentHandoff {
            response: "I'll help you address the reviewer feedback.\nLet me inspect files first."
                .to_string(),
            ..Default::default()
        };
        assert_eq!(extract_no_change_resolution_reason(&out), None);
    }

    #[test]
    fn extract_no_change_resolution_reason_prefers_structured_reason_marker() {
        let out = AgentHandoff {
            response: "Some analysis\nREASON: Property deletion already removes stored data on schema update."
                .to_string(),
            ..Default::default()
        };
        assert_eq!(
            extract_no_change_resolution_reason(&out),
            Some("Property deletion already removes stored data on schema update.".to_string())
        );
    }

    #[test]
    fn merge_request_surface_changed_detects_title_labels() {
        use crate::agents::gitlab::MergeRequest;
        let a = MergeRequest {
            iid: 1,
            title: "Old".into(),
            description: "d".into(),
            source_branch: "b".into(),
            target_branch: "m".into(),
            state: "opened".into(),
            sha: None,
            labels: Some(vec!["a".into()]),
            has_conflicts: false,
        };
        let mut b = a.clone();
        assert!(!merge_request_surface_changed(&a, &b));
        b.title = "New".into();
        assert!(merge_request_surface_changed(&a, &b));
        b.title = "Old".into();
        b.labels = Some(vec!["a".into(), "b".into()]);
        assert!(merge_request_surface_changed(&a, &b));
    }

    #[test]
    fn extract_mr_description_filters_control_markers() {
        let output = AgentHandoff {
            structured_outputs: Some(serde_json::json!({
                "handoff": {
                    "mr_description": "## Goal\nDescribe change.\nCHANGES_SUMMARY: noisy line\nMARK_DISCUSSIONS_RESOLVED: yes\nPOST_PLAIN_COMMENT: yes\n## Testing\ncargo test"
                }
            })),
            ..Default::default()
        };
        let desc = extract_mr_description(&output);
        assert!(!desc.contains("CHANGES_SUMMARY:"), "{desc}");
        assert!(!desc.contains("MARK_DISCUSSIONS_RESOLVED:"), "{desc}");
        assert!(!desc.contains("POST_PLAIN_COMMENT:"), "{desc}");
        assert!(desc.contains("## Goal"), "{desc}");
        assert!(desc.contains("## Testing"), "{desc}");
    }

    #[test]
    fn extract_mr_description_strips_public_comment_blocks() {
        let output = AgentHandoff {
            structured_outputs: Some(serde_json::json!({
                "handoff": {
                    "mr_description": "## Goal\npytest coverage.\nPUBLIC_COMMENT_BEGIN\nThanks for the review.\nPUBLIC_COMMENT_END\n## Testing\nuv run pytest"
                }
            })),
            ..Default::default()
        };
        let desc = extract_mr_description(&output);
        assert!(!desc.contains("PUBLIC_COMMENT_BEGIN"), "{desc}");
        assert!(!desc.contains("Thanks for the review"), "{desc}");
        assert!(desc.contains("## Goal"), "{desc}");
        assert!(desc.contains("uv run pytest"), "{desc}");
    }

    #[test]
    fn extract_explicit_mr_description_filters_control_markers() {
        let output = AgentHandoff {
            response:
                "MR_DESCRIPTION:\nSummary line\nCHANGES_SUMMARY: x\nMARK_DISCUSSIONS_RESOLVED: yes\n"
                    .to_string(),
            ..Default::default()
        };
        assert_eq!(extract_explicit_mr_description(&output), "Summary line");
    }

    #[test]
    fn extract_mr_title_prefers_block_markers() {
        let output = AgentHandoff {
            response: "MR_TITLE_BEGIN\nStable title\nMR_TITLE_END\nMR_TITLE: fallback".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_mr_title(&output, "Issue title"), "Stable title");
        assert_eq!(
            extract_explicit_mr_title(&output),
            Some("Stable title".into())
        );
    }

    #[test]
    fn extract_mr_description_prefers_block_markers() {
        let output = AgentHandoff {
            response:
                "MR_DESCRIPTION_BEGIN\n## Goal\nA\nCHANGES_SUMMARY: noisy\nMR_DESCRIPTION_END\nMR_DESCRIPTION:\nB"
                    .to_string(),
            ..Default::default()
        };
        assert_eq!(extract_mr_description(&output), "## Goal\nA");
        assert_eq!(extract_explicit_mr_description(&output), "## Goal\nA");
    }

    #[test]
    fn extract_depends_on_issue_parses_single_iid() {
        let out = AgentHandoff {
            response: "DEPENDS_ON_ISSUE: #42".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(42));
    }

    #[test]
    fn extract_depends_on_issue_parses_hashless_iid() {
        let out = AgentHandoff {
            response: "DEPENDS_ON_ISSUE: 42".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(42));
    }

    #[test]
    fn extract_depends_on_issue_takes_first_of_multiple() {
        let out = AgentHandoff {
            response: "DEPENDS_ON_ISSUE: #42, #43".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(42));
    }

    #[test]
    fn extract_depends_on_issue_parses_embedded_in_prose() {
        // The model often embeds the marker in a sentence rather than on its
        // own line, e.g. "Parked via DEPENDS_ON_ISSUE: #727 so this
        // auto-resumes once #727 merges." The parser must extract 727, not
        // fail on the trailing prose.
        let out = AgentHandoff {
            response: "Parked via DEPENDS_ON_ISSUE: #727 so this auto-resumes once #727 merges."
                .to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(727));
    }

    #[test]
    fn extract_depends_on_issue_parses_no_space_after_colon() {
        let out = AgentHandoff {
            response: "DEPENDS_ON_ISSUE:#42".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(42));
    }

    #[test]
    fn extract_depends_on_issue_returns_none_when_absent() {
        let out = AgentHandoff {
            response: "MR_TITLE: something\nMR_DESCRIPTION: done".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), None);
    }

    #[test]
    fn extract_depends_on_issue_returns_none_for_garbage() {
        let out = AgentHandoff {
            response: "DEPENDS_ON_ISSUE: #abc".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), None);
    }

    #[test]
    fn extract_depends_on_issue_fallback_blocked_on() {
        let out = AgentHandoff {
            response: "Issue #733 is hard-blocked on #727 and cannot be started yet.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(727));
    }

    #[test]
    fn extract_depends_on_issue_fallback_depends_on() {
        let out = AgentHandoff {
            response: "This work depends on #727 which is not yet closed.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(727));
    }

    #[test]
    fn extract_depends_on_issue_fallback_cannot_proceed_until() {
        let out = AgentHandoff {
            response: "I cannot proceed until #727 is closed; it cannot compile.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(727));
    }

    #[test]
    fn extract_depends_on_issue_fallback_blocked_by() {
        let out = AgentHandoff {
            response: "Implementation is blocked by #47 and needs its API.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(47));
    }

    #[test]
    fn extract_depends_on_issue_fallback_waiting_on() {
        let out = AgentHandoff {
            response: "Parked — waiting on #99 to merge first.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), Some(99));
    }

    #[test]
    fn extract_depends_on_issue_prose_fallback_ignores_unrelated_issue_refs() {
        // "See #42 for context" should NOT trigger — no dependency keyword.
        let out = AgentHandoff {
            response: "See #42 for background. The work is done.".to_string(),
            ..Default::default()
        };
        assert_eq!(extract_depends_on_issue(&out), None);
    }

    #[test]
    fn waiting_on_issue_label_format() {
        assert_eq!(waiting_on_issue_label(42), "waiting-on-issue:#42");
    }

    #[test]
    fn extract_waiting_on_issue_iid_finds_label() {
        let labels = vec![
            "in-progress".to_string(),
            "waiting-on-issue:#15".to_string(),
        ];
        assert_eq!(extract_waiting_on_issue_iid(&labels), Some(15));
    }

    #[test]
    fn extract_waiting_on_issue_iid_returns_none_without_label() {
        let labels = vec!["in-progress".to_string(), "priority::3".to_string()];
        assert_eq!(extract_waiting_on_issue_iid(&labels), None);
    }

    #[test]
    fn extract_waiting_on_issue_iid_ignores_malformed() {
        let labels = vec!["waiting-on-issue:#abc".to_string()];
        assert_eq!(extract_waiting_on_issue_iid(&labels), None);
    }
}
