use anyhow::{Context, Result};
use rand::RngExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

use super::{claim, extract_project_name, issue_in_scope, labels, pmo_cursor_ask};
use crate::acp::workspace_read::read_text_file_under_workspace;
use crate::agent::Agent;
use crate::config::PmoConfig;
use crate::cursor_mcp_config;
use crate::git::GitRepo;
use crate::gitlab::{self, GitLabClient, Issue};
use crate::mcp_coord::CoordinatorHandle;
use crate::mcp_coord::{AgentHandoff, HandoffSubIssue};

const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";

struct AgentState {
    sessions_dir: String,
    working_dir: String,
    agent_id: String,
    project_name: String,
}

impl AgentState {
    fn ensure_sessions_dir(&self) -> Result<()> {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        fs::create_dir_all(&ctx_dir).context("Failed to create .codepair-context directory")?;

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
            return;
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

fn interruptible_sleep(shutdown: &AtomicBool, duration: Duration) -> bool {
    let interval = Duration::from_millis(200);
    let mut remaining = duration;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return true;
        }
        if remaining.is_zero() {
            return false;
        }
        let sleep_time = remaining.min(interval);
        thread::sleep(sleep_time);
        remaining = remaining.saturating_sub(sleep_time);
    }
}

fn state_meta(agent: &Agent) -> String {
    let quits = agent.unexpected_quits_count();
    if quits == 0 {
        String::new()
    } else {
        format!(", {} unexpected quits", quits)
    }
}

pub fn run(
    repo_url: String,
    config: PmoConfig,
    instance_id: usize,
    shutdown: Arc<AtomicBool>,
    base_dir: String,
    coordinator: Option<CoordinatorHandle>,
    scope_label: String,
) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let agent_id = format!("pmo-{}", instance_id);
    let pmo_dir = super::work_dir(&base_dir, &project_name, &agent_id);
    let sessions_dir = super::sessions_dir(&base_dir, &project_name);

    let state = AgentState {
        sessions_dir: sessions_dir,
        working_dir: pmo_dir,
        agent_id: agent_id,
        project_name,
    };

    state.ensure_sessions_dir()?;

    let git_repo = GitRepo::new(state.working_dir.clone());
    let gitlab = GitLabClient::new(state.working_dir.clone());

    if coordinator.is_some() {
        cursor_mcp_config::refresh_mcp_mirror_best_effort(&state.working_dir);
    }

    let bridge = coordinator
        .as_ref()
        .map(|c| c.register_agent(&state.agent_id))
        .transpose()?;
    let agent = Agent::new(
        state.working_dir.clone(),
        config.model.clone(),
        Some(crate::agent::PREFERRED_SESSION_MODE_PMO),
        shutdown.clone(),
        state.agent_id.clone(),
        bridge,
    );

    let scope = crate::config::scope_label_filter(&scope_label);
    let mut claimed_issue_iid: Option<u64> = try_resume_pmo_state(&state, &gitlab, scope);

    if let Some(iid) = claimed_issue_iid {
        match (gitlab.get_issue(iid), gitlab.list_issues()) {
            (Ok(issue), Ok(issues)) => {
                if let Err(e) = refresh_pmo_issue_context_file(&state, &gitlab, &issue, &issues) {
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

    info!(
        "{}: Poll interval: {} seconds",
        &state.agent_id, config.poll_interval_secs
    );

    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        let jitter = rand::rng().random_range(0..5000);
        if interruptible_sleep(&shutdown, Duration::from_millis(jitter)) {
            break;
        }

        if let Err(e) = pmo_cycle(
            &state,
            &git_repo,
            &gitlab,
            &agent,
            &mut claimed_issue_iid,
            Arc::clone(&shutdown),
            scope,
            &config,
        ) {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            error!("{}: Cycle error: {}", &state.agent_id, e);
        }

        if interruptible_sleep(&shutdown, Duration::from_secs(config.poll_interval_secs)) {
            break;
        }
    }

    info!("{}: Shutting down, cleaning up...", &state.agent_id);

    if let Some(issue_iid) = claimed_issue_iid {
        info!(
            "{}: Preserving claim on issue #{} for restart",
            &state.agent_id, issue_iid
        );

        state.save_state(issue_iid);
    }

    info!("{}: Stopped", &state.agent_id);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pmo_cycle(
    state: &AgentState,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    claimed_issue_iid: &mut Option<u64>,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
    pmo_config: &PmoConfig,
) -> Result<()> {
    if agent.mcp_registered() {
        cursor_mcp_config::refresh_mcp_mirror_best_effort(&state.working_dir);
    }

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
                // Still waiting for human input — keep the on-disk context file fresh (new comments).
                match gitlab.list_issues() {
                    Ok(issues) => {
                        if let Err(e) =
                            refresh_pmo_issue_context_file(&state, gitlab, &issue, &issues)
                        {
                            warn!(
                                "{}: Could not refresh PMO context file while pmo-pending on #{}: {}",
                                &state.agent_id, held_iid, e
                            );
                        }
                    }
                    Err(e) => warn!(
                        "{}: Could not list issues to refresh context (pmo-pending #{}): {}",
                        &state.agent_id, held_iid, e
                    ),
                }

                debug!(
                    "{}: Issue #{} still pmo-pending, waiting for human input",
                    &state.agent_id, held_iid
                );

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
                    refresh_pmo_issue_context_file(&state, gitlab, &parent_issue, &issues)
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

    info!(
        "{}: Checking for issues requiring action...",
        &state.agent_id
    );

    let issues = gitlab.list_issues()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    let mut found_action_required = false;
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

        found_action_required = true;
        info!(
            "{}: Processing issue #{}: {}",
            &state.agent_id, issue.iid, issue.title
        );

        match process_action_required_issue(
            state,
            gitlab,
            agent,
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

    if !found_action_required {
        info!(
            "{}: No action-required issues found{}",
            &state.agent_id,
            state_meta(agent)
        );
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
    agent: &Agent,
    issue: &Issue,
    all_issues: &[Issue],
    scope_label: Option<&str>,
    shutdown: Arc<AtomicBool>,
    pmo_config: &PmoConfig,
) -> Result<bool> {
    let parent_priority = issue.priority();
    let context_path = refresh_pmo_issue_context_file(state, gitlab, issue, all_issues)?;
    let prompt = build_split_prompt(state, issue, &context_path, parent_priority)?;

    let ask_handler: Option<std::sync::Arc<dyn crate::acp::client::CursorAskQuestionHandler>> =
        if pmo_config.cursor_ask_via_gitlab {
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

    let agent_output = pmo_apply_cursor_plan_files(
        &state.working_dir,
        agent.run_with_cancel(&prompt, None, ask_handler)?,
    );

    // --- NEEDS_CLARIFICATION: PMO itself cannot decide, ask human ---
    if pmo_needs_clarification(&agent_output) {
        let question = extract_clarification_question(&agent_output);
        info!(
            "PMO: Issue #{} needs human clarification, marking pmo-pending",
            issue.iid
        );
        gitlab.add_issue_comment(
            issue.iid,
            &format!(
                "**PMO needs clarification before proceeding:**\n\n{}\n\n\
                 Please reply to this comment with the requested information. \
                 Once clarified, remove the `pmo-pending` label to let the PMO retry.",
                question
            ),
        )?;

        gitlab.add_issue_label(issue.iid, labels::PMO_PENDING)?;
        return Ok(true); // keep claim
    }

    // --- ALREADY_DONE: work is already implemented, close the issue ---
    // Require `ALREADY_DONE` then only whitespace before `REASON:` (avoids false positives from stray keywords).
    if is_pmo_already_done_response(&agent_output) {
        let reason = extract_already_done_reason(&agent_output);
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

    // --- GUIDE_WORKER: single focused retry instruction ---
    if pmo_guides_worker(&agent_output) {
        let guidance = extract_guidance(&agent_output);
        info!("PMO: Issue #{} needs guidance, not splitting", issue.iid);
        gitlab.add_issue_comment(
            issue.iid,
            &format!("**PMO guidance for the worker agent:**\n\n{}", guidance),
        )?;
        gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
        return Ok(false);
    }

    if pmo_no_split_needed(&agent_output) {
        let guidance = extract_guidance(&agent_output);
        let comment = if guidance.is_empty() {
            "PMO Agent reviewed this issue and determined it does not need splitting. \
             The worker should retry implementation."
                .to_string()
        } else {
            format!("**PMO guidance for the worker agent:**\n\n{}", guidance)
        };
        info!("PMO: Issue #{} does not need splitting", issue.iid);
        gitlab.add_issue_comment(issue.iid, &comment)?;
        gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
        return Ok(false);
    }

    // --- SPLIT: create sub-issues, close the parent as a task container ---
    let sub_issues = extract_sub_issues(&agent_output);

    if sub_issues.is_empty() {
        let preview = pmo_truncate_utf8_by_bytes(agent_output.response.trim(), 800);
        warn!(
            "PMO: No sub-issues extracted from agent output for issue #{} (response preview, {} bytes): {}",
            issue.iid,
            preview.len(),
            preview
        );

        gitlab.add_issue_comment(
            issue.iid,
            "PMO could not parse any sub-issues from the agent response. Manual intervention may be required.\n\n\
             **Use one of these formats:**\n\n\
             1) Text blocks (repeat per sub-issue):\n\
             ```text\n\
             SUB_ISSUE_1:\n\
             TITLE: Short title\n\
             PRIORITY: 1\n\
             DESCRIPTION:\n\
             Multi-line description.\n\
             ```\n\
             `PRIORITY` is optional (1–3). Headers are recognized with or without a trailing `:` and common markdown (list/`#`) prefixes.\n\n\
             2) Or a fenced JSON array:\n\
             ```json\n\
             [{\"title\":\"…\",\"description\":\"…\",\"priority\":1}]\n\
             ```",
        )?;

        gitlab.add_issue_label(issue.iid, PMO_PROCESSED_LABEL)?;
        return Ok(false);
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
                    "**ERROR: Codepair could not load GitLab issue comments.**\n\n\
                     PMO triage may be unreliable until this works (check `glab` auth, network, and that `repo_path` is the git clone root).\n\n\
                     ```text\n{e}\n```"
                ),
                0,
            )
        }
    }
}

/// Rewrites `pmo-issue-<iid>.md` under [`CODEPAIR_CONTEXT_DIR`]. Call whenever PMO holds or resumes
/// work on an issue (new claim, resumed claim, pmo-pending poll, pending split).
fn refresh_pmo_issue_context_file(
    state: &AgentState,
    gitlab: &GitLabClient,
    issue: &Issue,
    all_issues: &[Issue],
) -> Result<String> {
    let (comments_text, gitlab_note_count) = pmo_gitlab_comments_section(gitlab, issue.iid);
    let existing_issues_text = build_existing_issues_summary(issue.iid, all_issues);

    let generated_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "# PMO Triage Context\n\nProject: {project_name}\nIssue: #{iid} {title}\n\n## Issue description\n{description}\n\n## Comments and worker feedback\n{comments}\n\n## Existing open issues\n{existing}\n\n---\n_Codepair: PMO refreshed this file; {gitlab_note_count} GitLab note(s) in the section above; UNIX ts {generated_ts}._\n",
        project_name = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        description = issue.description,
        comments = comments_text,
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
Codepair wrote the path above as a markdown file: **full issue description**, **every GitLab issue comment** (including worker rejection / PMO notes), and **the list of other open issues**. That file is the authoritative written context for this triage.
- Use your **file-reading** capability on the absolute path and read it **end-to-end** before you decide the situation is unclear.
- The single line `ISSUE #…: title` in this prompt is **not** a substitute for the file; do not claim "no context" or choose NEEDS_CLARIFICATION only because you did not read the task context file.
Your job is to analyze the failure reason (from the file + repo when needed) and take the appropriate action.

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You do NOT write new production code — you inspect the existing project state, then write comments and create issue descriptions as needed
- The worker agent has FULL ACCESS to shell commands (rm, mv, git, etc.) and all build/test tools
- If the worker claimed it "cannot run commands" or "cannot delete files", that is WRONG — it CAN. Instruct it clearly.
- Before deciding GUIDE_WORKER or SPLIT, you MUST verify whether the issue is already implemented in the current project state when that is plausible from the issue, comments, or worker output. If the behavior/tests/code already exist, choose ALREADY_DONE so Codepair will close the issue and add a comment.
- In your final reply for this turn, make the outcome obvious in plain text (markers below). Codepair parses your message; there is no separate tool call for handoff.
- **Plan mode:** Cursor may save the plan as a `.md` file under the repo. Codepair reads that file after the turn and treats its contents like assistant text for parsing. A structured UI plan **without** the plain-text formats below is **not** enough — the saved plan file (or streamed text) **must** include `GUIDE_WORKER`, `SPLIT` + `SUB_ISSUE_N` blocks, `ALREADY_DONE`, or `NEEDS_CLARIFICATION` as required. Do not stop after only using plan UI widgets, and do not wait for user confirmation — this run is fully automated.
- When applicable, mirror structured intent clearly in prose so parsers can pick it up, e.g. state `decision`, `question` / clarification needs, `instructions` for worker guidance, `reason` for ALREADY_DONE, and numbered `SUB_ISSUE_N` blocks for SPLIT.
- For SPLIT, describe each planned sub-issue using the human-readable `SUB_ISSUE_N / TITLE / PRIORITY / DESCRIPTION` blocks in your reply (include acceptance criteria and dependencies in `DESCRIPTION` when relevant).

DECISION — choose EXACTLY ONE of the following:

1. GUIDE_WORKER — Use ONLY when ALL of these are true:
   a) The issue describes a SINGLE, focused task (not a list of modules/files/components)
   b) The worker failed due to a specific misunderstanding, wrong command, or simple technical obstacle
   c) The fix is ONE clear action (e.g. "use flag X instead of Y", "the config file is at path Z")
   If your guidance would enumerate 2+ independent modules, files, or components, you MUST choose SPLIT instead.
   Respond with:
   GUIDE_WORKER
   INSTRUCTIONS:
   <Brief, actionable guidance — MAX 3-5 sentences. State the single core action the worker must take.>
   Include `decision: guide_worker` and the same guidance under `INSTRUCTIONS:` in your reply text.

2. SPLIT — Use when ANY of these are true:
   - The issue is a "task container" describing a broad goal (e.g. "add tests for module X", "refactor all Y", "check full code for Z") — these ALWAYS need splitting into concrete sub-tasks
   - The issue involves work on 2+ independent modules, files, or components
   - The issue is too large (estimated >500 lines of non-test code, or >1500 lines total including tests; auto-generated code does not count)
   - The issue description or worker rejection lists multiple distinct things to do
   - Your guidance would need to enumerate 2+ independent items
   When splitting, the PARENT ISSUE will be CLOSED automatically as a task container. The sub-issues become the real tracked work.
   Respond with sub-issues in this format:

   SUB_ISSUE_1:
   TITLE: <concise title>
   PRIORITY: <1, 2, or 3>
   DESCRIPTION:
   <Detailed description of what needs to be implemented>
   <Include acceptance criteria>
   <Mention any dependencies on other sub-issues>

   SUB_ISSUE_2:
   TITLE: <concise title>
   PRIORITY: <1, 2, or 3>
   DESCRIPTION:
   <Detailed description>

   (continue for all sub-issues — each should target ~500 lines of non-test code, ~1500 total including tests; auto-generated code does not count)
   Include `decision: split` in your reply and the same sub-issues as `SUB_ISSUE_N` blocks.
   Example structured item:
   - `{{"title":"Fix failing TestPythonChunker tests","description":"Repair the failing tests in tests/parser/test_python_chunker.py. Acceptance criteria: all tests in that file pass. Dependencies: SUB_ISSUE_1.","priority":2}}`

   PRIORITY LEVELS:
   - 1 = Critical: blocking other work, security fix, core dependency that other sub-issues depend on
   - 2 = High: important feature, depended on by lower-priority sub-issues
   - 3 = Normal: independent work, enhancements, nice-to-haves
   The parent issue has priority {parent_priority}. Sub-issues that are dependencies for others should get higher priority (lower number). Independent leaf tasks can inherit the parent priority or be lower.

3. ALREADY_DONE — Use when the work described in the issue is ALREADY fully implemented in the codebase:
   - The worker's output or your analysis shows the feature/tests/code already exists
   - There is nothing left to implement — the issue is simply outdated or redundant
   - Prefer ALREADY_DONE over GUIDE_WORKER or SPLIT when the required behavior is already present in the repository as it exists now
   You MUST use this exact pattern so the PMO can parse it (only whitespace may appear between the two lines; no other text in between):
   ALREADY_DONE
   REASON: <Brief explanation of why this issue is already complete, referencing the existing code/files>
   (Same line is also valid: ALREADY_DONE  REASON: <explanation>)
   Include `decision: already_done` and `reason:` with the same explanation in your reply.

4. NEEDS_CLARIFICATION — Use when you CANNOT make a decision because:
   - After reading the **task context file** and (if needed) the repo, the issue is still too vague to determine scope or intent
   - The worker's rejection and the issue (as given in that file) still don't give enough to guide or split
   - You need specific information from a human (e.g. which modules to cover, what the acceptance criteria are)
   Do **not** use this option because you skipped reading the task context file.
   Respond with:
   NEEDS_CLARIFICATION
   QUESTION:
   <Specific question(s) you need answered before you can guide or split this issue. Be precise about what information is missing.>
   Include `decision: needs_clarification` and your `question:` in the reply text.

DUPLICATE / OVERLAP RULES (STRICT):
- Review the EXISTING OPEN ISSUES list above before creating any sub-issue.
- Do NOT create a sub-issue that duplicates or substantially overlaps with an existing open issue.
- If an existing OPEN (not IN-PROGRESS) issue covers part of the work, reference it (e.g. "See existing #42") instead of creating a new sub-issue for that part.
- If an existing IN-PROGRESS issue already covers it, simply skip that part entirely — do not create a sub-issue or reference.
- If ALL sub-issues would duplicate existing issues, choose GUIDE_WORKER instead and tell the worker which existing issues already cover the work.

INSTRUCTIONS:
1. Open and read the **entire** TASK CONTEXT FILE at the absolute path above (description, GitLab comments, existing issues). Do this first.
2. From that file, read the issue description and **all** comments — especially the worker's rejection reason.
3. Review the EXISTING OPEN ISSUES section in that same file to see what is already tracked.
4. TASK CONTAINER TEST: Does the issue describe a broad goal that involves multiple independent pieces of work (e.g. "add tests for all modules", "refactor X across the codebase", "check code for Y")? If YES → SPLIT. The parent issue is just a container; the real work is in the sub-issues.
5. GUIDANCE TEST: Is there ONE specific thing the worker misunderstood or did wrong? If YES → GUIDE_WORKER.
6. ENUMERATION TEST: If your guidance would list 2+ independent modules, files, or components → SPLIT, not GUIDE_WORKER.
7. CLARITY TEST: Only if the task context file plus (if needed) repo inspection still leaves intent unclear → NEEDS_CLARIFICATION.
8. COMPLETION TEST: Does the worker's output, the comments in the file, or your direct inspection of the current project state indicate the work is already fully implemented in the codebase? If YES → ALREADY_DONE.
9. Choose EXACTLY ONE of GUIDE_WORKER, SPLIT, ALREADY_DONE, or NEEDS_CLARIFICATION — never combine them.
10. When in doubt between GUIDE_WORKER and SPLIT, prefer SPLIT — it's better to create focused sub-issues than to give the worker a laundry list.

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubIssue {
    title: String,
    description: String,
    #[serde(default)]
    priority: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PendingSplit {
    parent_issue_iid: u64,
    parent_issue_title: String,
    #[serde(default)]
    parent_priority: u8,
    sub_issues: Vec<SubIssue>,
    created_issue_ids: Vec<u64>,
}

/// `ALREADY_DONE` followed only by whitespace (including newlines), then `REASON:`.
static PMO_ALREADY_DONE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"ALREADY_DONE\s+REASON:").expect("PMO ALREADY_DONE pattern"));

fn pmo_needs_clarification(output: &AgentHandoff) -> bool {
    output.needs_clarification.is_some()
        || output.question.is_some()
        || output
            .decision
            .as_deref()
            .is_some_and(|d| d.eq_ignore_ascii_case("needs_clarification"))
        || response_has_decision_marker(&output.response, "NEEDS_CLARIFICATION")
}

fn pmo_guides_worker(output: &AgentHandoff) -> bool {
    output.instructions.is_some()
        || output
            .decision
            .as_deref()
            .is_some_and(|d| d.eq_ignore_ascii_case("guide_worker"))
        || response_has_decision_marker(&output.response, "GUIDE_WORKER")
}

fn pmo_no_split_needed(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("no_split_needed"))
        || response_has_decision_marker(&output.response, "NO_SPLIT_NEEDED")
}

fn response_has_decision_marker(response: &str, marker: &str) -> bool {
    response.lines().map(str::trim).any(|line| line == marker)
}

fn is_pmo_already_done_response(output: &AgentHandoff) -> bool {
    output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("already_done"))
        || PMO_ALREADY_DONE_RE.is_match(&output.response)
}

fn extract_already_done_reason(agent_output: &AgentHandoff) -> String {
    if let Some(reason) = &agent_output.reason {
        let trimmed = reason.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    static EXTRACT_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)ALREADY_DONE\s+REASON:\s*(.+)\z").expect("PMO ALREADY_DONE extract")
    });
    if let Some(cap) = EXTRACT_RE
        .captures(&agent_output.response)
        .and_then(|c| c.get(1))
    {
        let trimmed = cap.as_str().trim();
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
    if let Some(pos) = agent_output.response.find("QUESTION:") {
        let rest = &agent_output.response[pos + 9..];
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "The PMO agent could not determine how to proceed with this issue. Please provide more details about the expected scope and acceptance criteria.".to_string()
}

fn extract_guidance(agent_output: &AgentHandoff) -> String {
    let raw = if let Some(instructions) = &agent_output.instructions {
        let trimmed = instructions.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            String::new()
        }
    } else if let Some(pos) = agent_output.response.find("INSTRUCTIONS:") {
        let rest = &agent_output.response[pos + 13..];
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            String::new()
        }
    } else if let Some(reason) = &agent_output.reason {
        reason.trim().to_string()
    } else if let Some(pos) = agent_output.response.find("REASON:") {
        let rest = &agent_output.response[pos + 7..];
        rest.trim().to_string()
    } else {
        return String::new();
    };

    // Cap guidance length — keep it brief and actionable
    if raw.len() > 500 {
        let truncated: String = raw.chars().take(500).collect();
        if let Some(last_period) = truncated.rfind('.') {
            truncated[..=last_period].to_string()
        } else {
            format!("{}...", truncated)
        }
    } else {
        raw
    }
}

/// Append contents of [`AgentHandoff::cursor_plan_paths`] (Cursor plan-mode `tool_call_update`) into
/// `handoff.response` so triage markers and `extract_sub_issues` see the plan file text.
fn pmo_apply_cursor_plan_files(working_dir: &str, mut handoff: AgentHandoff) -> AgentHandoff {
    if handoff.cursor_plan_paths.is_empty() {
        return handoff;
    }

    let root = path::Path::new(working_dir);
    for plan_path in std::mem::take(&mut handoff.cursor_plan_paths) {
        match read_text_file_under_workspace(root, &plan_path) {
            Ok(body) => {
                if !handoff.response.is_empty() {
                    handoff.response.push_str("\n\n");
                }

                handoff.response.push_str("=== PMO Cursor plan file ===\n");
                handoff.response.push_str(body.trim_end());
                handoff.response.push('\n');
            }

            Err(e) => {
                warn!(
                    target: "codepair::pmo_plan_file",
                    path = %plan_path,
                    err = %e,
                    "PMO could not read Cursor plan file from tool_call_update"
                );
            }
        }
    }

    handoff
}

fn combined_pmo_handoff_text(h: &AgentHandoff) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let r = h.response.trim();
    if !r.is_empty() {
        parts.push(r);
    }
    for s in [
        h.instructions.as_deref(),
        h.needs_split.as_deref(),
        h.feedback.as_deref(),
        h.decision.as_deref(),
        h.changes_summary.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        let t = s.trim();
        if !t.is_empty() {
            parts.push(t);
        }
    }
    parts.join("\n\n")
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

fn pmo_strip_ordered_list_prefix(s: &str) -> &str {
    let s = s.trim_start();
    let Some(digits_end) = s
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map(|(i, _)| i)
    else {
        return s;
    };
    if digits_end == 0 {
        return s;
    }
    if s[digits_end..].starts_with(". ") {
        s[digits_end + 2..].trim_start()
    } else {
        s
    }
}

/// Strip common markdown / list noise so `SUB_ISSUE_1` and `TITLE:` lines still match.
fn pmo_normalize_issue_line(line: &str) -> &str {
    let mut s = line.trim();
    while s.starts_with('#') {
        s = s[1..].trim_start();
    }
    while s.starts_with('>') {
        s = s[1..].trim_start();
    }
    s = s.trim_start();
    if let Some(rest) = s.strip_prefix("- ") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("* ") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("+ ") {
        s = rest;
    } else {
        s = pmo_strip_ordered_list_prefix(s);
    }
    if s.len() >= 2 && s.starts_with('`') && s.ends_with('`') {
        s = &s[1..s.len() - 1];
    }
    s.trim()
}

static PMO_SUB_ISSUE_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(SUB_ISSUE_\d+|SUB\s+ISSUE\s+\d+)\s*:?\s*(.*)$").expect("pmo SUB_ISSUE line")
});

enum PmoField<'a> {
    Title(&'a str),
    Priority(&'a str),
    Description(&'a str),
}

fn pmo_field_value_ci<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let line = line.trim_start();
    let kl = key.len();
    if line.len() < kl {
        return None;
    }
    if !line[..kl].eq_ignore_ascii_case(key) {
        return None;
    }
    let rest = line[kl..].trim_start();
    let rest = rest.strip_prefix(':')?.trim();
    Some(rest)
}

fn pmo_parse_field(line: &str) -> Option<PmoField<'_>> {
    if let Some(v) = pmo_field_value_ci(line, "TITLE") {
        return Some(PmoField::Title(v));
    }
    if let Some(v) = pmo_field_value_ci(line, "PRIORITY") {
        return Some(PmoField::Priority(v));
    }
    if let Some(v) = pmo_field_value_ci(line, "DESCRIPTION") {
        return Some(PmoField::Description(v));
    }
    None
}

fn extract_json_arrays_from_markdown_fences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut s = text;
    while let Some(start) = s.find("```") {
        let mut cur = &s[start + 3..];
        cur = cur.trim_start();
        if cur.len() >= 4 && cur[..4].eq_ignore_ascii_case("json") {
            cur = &cur[4..];
        }
        cur = cur.trim_start();
        let Some(end) = cur.find("```") else {
            break;
        };
        let body = cur[..end].trim();
        if body.starts_with('[') {
            out.push(body.to_string());
        }
        s = &cur[end + 3..];
    }
    out
}

fn try_parse_sub_issues_from_json_array(json: &str) -> Vec<SubIssue> {
    let Ok(vals) = serde_json::from_str::<Vec<Value>>(json) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for v in vals {
        let Some(obj) = v.as_object() else {
            continue;
        };
        let title = obj
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let description = obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let priority = obj
            .get("priority")
            .and_then(|p| {
                p.as_u64()
                    .map(|u| u as u8)
                    .or_else(|| p.as_str()?.parse().ok())
            })
            .filter(|p| (1..=3).contains(p));
        if title.is_empty() || description.is_empty() {
            continue;
        }
        out.push(SubIssue {
            title: title.to_string(),
            description: description.to_string(),
            priority,
        });
    }
    out
}

fn extract_sub_issues_from_text(text: &str) -> Vec<SubIssue> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sub_issues = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let normalized = pmo_normalize_issue_line(lines[i]);
        let Some(caps) = PMO_SUB_ISSUE_LINE.captures(normalized.trim()) else {
            i += 1;
            continue;
        };
        let tail = caps
            .get(2)
            .map(|m| m.as_str().trim())
            .filter(|t| !t.is_empty())
            .unwrap_or("");

        let mut title = String::new();
        let mut description = String::new();
        let mut priority: Option<u8> = None;
        let mut in_description = false;

        let mut pending: Option<String> = if tail.is_empty() {
            None
        } else if pmo_parse_field(tail).is_some() {
            Some(tail.to_string())
        } else {
            title = tail.to_string();
            None
        };

        i += 1;

        loop {
            let cur: std::borrow::Cow<'_, str> = if let Some(p) = pending.take() {
                std::borrow::Cow::Owned(p)
            } else if i < lines.len() {
                let nl = pmo_normalize_issue_line(lines[i]);
                if PMO_SUB_ISSUE_LINE.is_match(nl.trim()) {
                    break;
                }
                i += 1;
                std::borrow::Cow::Borrowed(nl)
            } else {
                break;
            };

            if let Some(field) = pmo_parse_field(&cur) {
                match field {
                    PmoField::Title(t) => {
                        title = t.to_string();
                        in_description = false;
                    }
                    PmoField::Priority(p) => {
                        if let Ok(v) = p.trim().parse::<u8>()
                            && (1..=3).contains(&v)
                        {
                            priority = Some(v);
                        }
                        in_description = false;
                    }
                    PmoField::Description(d) => {
                        in_description = true;
                        if !d.is_empty() {
                            if !description.is_empty() {
                                description.push('\n');
                            }
                            description.push_str(d);
                        }
                    }
                }
            } else if in_description && !cur.trim().is_empty() {
                if !description.is_empty() {
                    description.push('\n');
                }
                description.push_str(cur.trim());
            }
        }

        if !title.is_empty() && !description.is_empty() {
            sub_issues.push(SubIssue {
                title,
                description,
                priority,
            });
        }
    }
    sub_issues
}

fn extract_sub_issues(agent_output: &AgentHandoff) -> Vec<SubIssue> {
    if !agent_output.sub_issues.is_empty() {
        return agent_output
            .sub_issues
            .iter()
            .map(|item: &HandoffSubIssue| SubIssue {
                title: item.title.clone(),
                description: item.description.clone(),
                priority: item.priority,
            })
            .collect();
    }

    let text = combined_pmo_handoff_text(agent_output);
    let mut sub_issues = extract_sub_issues_from_text(&text);
    if sub_issues.is_empty() {
        let t = text.trim();
        if t.starts_with('[') {
            let parsed = try_parse_sub_issues_from_json_array(t);
            if !parsed.is_empty() {
                sub_issues = parsed;
            }
        }
    }
    if sub_issues.is_empty() {
        for block in extract_json_arrays_from_markdown_fences(&text) {
            let parsed = try_parse_sub_issues_from_json_array(&block);
            if !parsed.is_empty() {
                sub_issues = parsed;
                break;
            }
        }
    }

    debug!(
        "Extracted {} sub-issues from agent output",
        sub_issues.len()
    );
    sub_issues
}

fn save_pending_split(path: &str, pending: &PendingSplit) -> Result<()> {
    let json = serde_json::to_string_pretty(pending)?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)
            .context("Failed to create .codepair-context for pending split")?;
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

        match gitlab.create_issue(&sub_issue_title, &sub_issue.description) {
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
    fn test_extract_project_name() {
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project.git").unwrap(),
            "project"
        );
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

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string(), "codepair".to_string()];
        assert!(!should_process_issue(&issue, Some("other-scope")));
        assert!(should_process_issue(&issue, Some("codepair")));
    }

    #[test]
    fn test_extract_sub_issues() {
        let output = AgentHandoff {
            response: r#"
SUB_ISSUE_1:
TITLE: Add authentication module
PRIORITY: 1
DESCRIPTION:
Implement basic authentication with JWT tokens.
Include login and logout endpoints.

SUB_ISSUE_2:
TITLE: Add user management
PRIORITY: 2
DESCRIPTION:
Create user CRUD operations.
Add role-based access control.
        "#
            .to_string(),
            ..Default::default()
        };

        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 2);
        assert_eq!(sub_issues[0].title, "Add authentication module");
        assert!(sub_issues[0].description.contains("JWT tokens"));
        assert_eq!(sub_issues[0].priority, Some(1));
        assert_eq!(sub_issues[1].title, "Add user management");
        assert!(sub_issues[1].description.contains("CRUD"));
        assert_eq!(sub_issues[1].priority, Some(2));
    }

    #[test]
    fn test_extract_sub_issues_without_priority() {
        let output = AgentHandoff {
            response: r#"
SUB_ISSUE_1:
TITLE: Simple task
DESCRIPTION:
Do something simple.
        "#
            .to_string(),
            ..Default::default()
        };

        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 1);
        assert_eq!(sub_issues[0].priority, None);
    }

    #[test]
    fn test_extract_sub_issues_markdown_headers_and_lists() {
        let output = AgentHandoff {
            response: r#"
## SUB_ISSUE_1
- title: Auth API
- priority: 2
- description:
First line of desc.
Second line.

* SUB_ISSUE_2:
* TITLE: Cleanup
* DESCRIPTION: One-line description only.
"#
            .to_string(),
            ..Default::default()
        };
        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 2);
        assert_eq!(sub_issues[0].title, "Auth API");
        assert_eq!(sub_issues[0].priority, Some(2));
        assert!(sub_issues[0].description.contains("Second line"));
        assert_eq!(sub_issues[1].title, "Cleanup");
        assert_eq!(sub_issues[1].description, "One-line description only.");
    }

    #[test]
    fn test_extract_sub_issues_description_on_same_line() {
        let output = AgentHandoff {
            response: r#"
SUB_ISSUE_1:
TITLE: Task A
DESCRIPTION: All on one line.
"#
            .to_string(),
            ..Default::default()
        };
        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 1);
        assert_eq!(sub_issues[0].description, "All on one line.");
    }

    #[test]
    fn test_extract_sub_issues_from_json_fence() {
        let output = AgentHandoff {
            response: r#"
Here is the plan:

```json
[
  {"title": "API", "description": "Build REST API.", "priority": 1},
  {"title": "UI", "description": "Add dashboard."}
]
```
"#
            .to_string(),
            ..Default::default()
        };
        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 2);
        assert_eq!(sub_issues[0].priority, Some(1));
        assert_eq!(sub_issues[1].priority, None);
    }

    #[test]
    fn test_extract_sub_issues_from_instructions_when_response_empty() {
        let output = AgentHandoff {
            response: String::new(),
            instructions: Some(
                r#"
SUB_ISSUE_1:
TITLE: From instructions
DESCRIPTION:
Body here.
"#
                .to_string(),
            ),
            ..Default::default()
        };
        let sub_issues = extract_sub_issues(&output);
        assert_eq!(sub_issues.len(), 1);
        assert_eq!(sub_issues[0].title, "From instructions");
    }

    #[test]
    fn test_pmo_already_done_pattern() {
        assert!(is_pmo_already_done_response(&AgentHandoff {
            response: "Analysis complete.\nALREADY_DONE\nREASON: Feature exists in src/foo.rs."
                .to_string(),
            ..Default::default()
        }));
        assert!(is_pmo_already_done_response(&AgentHandoff {
            response: "ALREADY_DONE  REASON: All tests already pass.".to_string(),
            ..Default::default()
        }));
        assert!(!is_pmo_already_done_response(&AgentHandoff {
            response: "ALREADY_DONE\n\nSee above.\nREASON: wrong — non-whitespace between tokens"
                .to_string(),
            ..Default::default()
        }));
        assert!(!is_pmo_already_done_response(&AgentHandoff {
            response: "REASON: something\nALREADY_DONE".to_string(),
            ..Default::default()
        }));
        assert!(!is_pmo_already_done_response(&AgentHandoff {
            response: "ALREADY_DONE without reason header".to_string(),
            ..Default::default()
        }));

        let reason = extract_already_done_reason(&AgentHandoff {
            response: "noise\nALREADY_DONE\nREASON:\n\nImplemented in module X.".to_string(),
            ..Default::default()
        });
        assert!(reason.contains("module X"));
    }

    #[test]
    fn decision_markers_do_not_trigger_on_negated_mentions() {
        let output = AgentHandoff {
            response: "Decision: SPLIT\nNot NEEDS_CLARIFICATION (scope is clear).".to_string(),
            ..Default::default()
        };
        assert!(!pmo_needs_clarification(&output));
        assert!(!pmo_guides_worker(&output));
        assert!(!pmo_no_split_needed(&output));
    }

    #[test]
    fn pmo_apply_cursor_plan_files_merges_markdown_into_response() {
        let tmp =
            std::env::temp_dir().join(format!("codepair-pmo-plan-merge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("plans")).unwrap();
        let f = tmp.join("plans/triage.md");
        std::fs::write(&f, "SUB_ISSUE_1 TITLE: x\n").unwrap();
        let root = std::fs::canonicalize(&tmp).unwrap();
        let abs = std::fs::canonicalize(&f)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let mut h = AgentHandoff::default();
        h.cursor_plan_paths.push(abs);
        let out = super::pmo_apply_cursor_plan_files(&root.to_string_lossy(), h);
        assert!(out.response.contains("PMO Cursor plan file"));
        assert!(out.response.contains("SUB_ISSUE_1"));
        assert!(out.cursor_plan_paths.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
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
}
