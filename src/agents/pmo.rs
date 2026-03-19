use anyhow::{Context, Result};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{claim, extract_project_name};
use crate::agent::Agent;
use crate::config::PmoConfig;
use crate::git::GitRepo;
use crate::gitlab::{self, GitLabClient, Issue};

const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";
const PMO_PENDING_LABEL: &str = "pmo-pending";

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

pub fn run(
    repo_url: String,
    config: PmoConfig,
    instance_id: usize,
    shutdown: Arc<AtomicBool>,
    base_dir: String,
) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let agent_id = format!("pmo-{}", instance_id);
    let pmo_dir = super::work_dir(&base_dir, &project_name, &agent_id);

    let git_repo = GitRepo::new(pmo_dir.clone());
    let gitlab = GitLabClient::new(pmo_dir.clone());
    let agent = Agent::new(pmo_dir.clone(), config.model.clone(), shutdown.clone());

    let mut claimed_issue_iid: Option<u64> = try_resume_pmo_state(&agent_id, &pmo_dir, &gitlab);

    info!(
        "{}: Poll interval: {} seconds",
        agent_id, config.poll_interval_secs
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
            &agent_id,
            &project_name,
            &pmo_dir,
            &git_repo,
            &gitlab,
            &agent,
            &mut claimed_issue_iid,
            &shutdown,
        ) {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            error!("{}: Cycle error: {}", agent_id, e);
        }

        if interruptible_sleep(&shutdown, Duration::from_secs(config.poll_interval_secs)) {
            break;
        }
    }

    info!("{}: Shutting down, cleaning up...", agent_id);
    if let Some(issue_iid) = claimed_issue_iid {
        info!(
            "{}: Preserving claim on issue #{} for restart",
            agent_id, issue_iid
        );
        save_pmo_state(&pmo_dir, issue_iid);
    }
    info!("{}: Stopped", agent_id);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pmo_cycle(
    agent_id: &str,
    project_name: &str,
    pmo_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    claimed_issue_iid: &mut Option<u64>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;
    
    // Ensure we're on the latest upstream — hard reset if checkout fails
    if let Err(e) = git_repo.checkout_remote_branch(&default_branch) {
        warn!(
            "{}: Failed to checkout {}: {}, forcing reset",
            agent_id, default_branch, e
        );
        let _ = git_repo.reset_hard();
        git_repo.checkout_remote_branch(&default_branch)?;
    }

    // If we still hold a claim from a previous run, decide what to do.
    if let Some(held_iid) = *claimed_issue_iid {
        // Pending splits take priority — handled below
        let pending_file_check = pending_split_file(pmo_dir, agent_id);
        let has_pending = load_pending_split(&pending_file_check)?;
        if has_pending.is_some() {
            // Fall through to pending split handling
        } else if let Ok(issue) = gitlab.get_issue(held_iid) {
            if issue.labels.contains(&PMO_PENDING_LABEL.to_string()) {
                // Still waiting for human clarification — keep the claim, skip processing
                debug!(
                    "{}: Issue #{} still pmo-pending, waiting for human input",
                    agent_id, held_iid
                );
                // Don't release, don't process — just return and check again next cycle
                return Ok(());
            }
            // No longer pending (human removed the label) — release claim so it
            // can be re-processed as a fresh action-required issue.
            info!(
                "{}: Releasing claim on issue #{} from previous run",
                agent_id, held_iid
            );
            let _ = claim::release_claim(gitlab, held_iid, agent_id);
            *claimed_issue_iid = None;
            clear_pmo_state(pmo_dir);
        } else {
            // Can't fetch issue — release to be safe
            let _ = claim::release_claim(gitlab, held_iid, agent_id);
            *claimed_issue_iid = None;
            clear_pmo_state(pmo_dir);
        }
    }

    let pending_file = pending_split_file(pmo_dir, agent_id);
    if let Some(pending_split) = load_pending_split(&pending_file)? {
        info!(
            "{}: Resuming pending split for issue #{} ({} sub-issues remaining)",
            agent_id,
            pending_split.parent_issue_iid,
            pending_split.sub_issues.len()
        );

        match resume_split(&pending_file, gitlab, &pending_split) {
            Ok(_) => {
                info!(
                    "{}: Successfully completed pending split for issue #{}",
                    agent_id, pending_split.parent_issue_iid
                );
                delete_pending_split(&pending_file)?;
                // Release the claim from the split
                if let Some(held_iid) = *claimed_issue_iid {
                    let _ = claim::release_claim(gitlab, held_iid, agent_id);
                    *claimed_issue_iid = None;
                    clear_pmo_state(pmo_dir);
                }
            }
            Err(e) => {
                error!("{}: Failed to complete pending split: {}", agent_id, e);
            }
        }

        return Ok(());
    }

    info!("{}: Checking for issues requiring action...", agent_id);
    let issues = gitlab.list_issues()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    let mut found_action_required = false;
    for issue in &issues {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !should_process_issue(issue) {
            continue;
        }

        if claim::is_claimed(&issue.labels) {
            debug!(
                "{}: Issue #{} already claimed, skipping",
                agent_id, issue.iid
            );
            continue;
        }

        if !claim::try_claim_issue(gitlab, issue.iid, agent_id, shutdown)? {
            info!(
                "{}: Failed to claim issue #{}, skipping",
                agent_id, issue.iid
            );
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_claim(gitlab, issue.iid, agent_id);
            return Ok(());
        }

        *claimed_issue_iid = Some(issue.iid);
        save_pmo_state(pmo_dir, issue.iid);

        found_action_required = true;
        info!(
            "{}: Processing issue #{}: {}",
            agent_id, issue.iid, issue.title
        );

        match process_action_required_issue(
            agent_id,
            project_name,
            pmo_dir,
            gitlab,
            agent,
            issue,
            &issues,
        ) {
            Ok(keep_claim) => {
                info!("{}: Successfully processed issue #{}", agent_id, issue.iid);
                if keep_claim {
                    info!(
                        "{}: Keeping claim on issue #{} (pmo-pending)",
                        agent_id, issue.iid
                    );
                } else {
                    claim::release_claim(gitlab, issue.iid, agent_id)?;
                    *claimed_issue_iid = None;
                    clear_pmo_state(pmo_dir);
                }
            }
            Err(e) => {
                error!(
                    "{}: Failed to process issue #{}: {}",
                    agent_id, issue.iid, e
                );
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
                claim::release_claim(gitlab, issue.iid, agent_id)?;
                *claimed_issue_iid = None;
                clear_pmo_state(pmo_dir);
            }
        }

        break;
    }

    if !found_action_required {
        info!("{}: No action-required issues found", agent_id);
    }

    // Assign default priority to open issues that lack a priority label.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }
    assign_default_priority(agent_id, &issues, gitlab);

    // Close stale pmo-processed issues that have not been picked up for over 1 hour.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }
    close_stale_processed_issues(agent_id, &issues, gitlab);

    Ok(())
}

fn assign_default_priority(agent_id: &str, issues: &[Issue], gitlab: &GitLabClient) {
    for issue in issues {
        if issue.state != "opened" {
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

fn close_stale_processed_issues(agent_id: &str, issues: &[Issue], gitlab: &GitLabClient) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    for issue in issues {
        if issue.state != "opened" {
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

fn should_process_issue(issue: &Issue) -> bool {
    // Must have action-required label
    if !issue.labels.contains(&ACTION_REQUIRED_LABEL.to_string()) {
        return false;
    }

    // Skip if already processed by PMO
    if issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
        return false;
    }

    // Skip if PMO is waiting for human clarification
    if issue.labels.contains(&PMO_PENDING_LABEL.to_string()) {
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
fn process_action_required_issue(
    agent_id: &str,
    project_name: &str,
    pmo_dir: &str,
    gitlab: &GitLabClient,
    agent: &Agent,
    issue: &Issue,
    all_issues: &[Issue],
) -> Result<bool> {
    // Get all comments to understand the context
    let comments = gitlab.get_issue_comments(issue.iid).unwrap_or_default();

    let comments_text = if comments.is_empty() {
        "No comments yet.".to_string()
    } else {
        comments
            .iter()
            .map(|c| format!("- {}: {}", c.author, c.body))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let existing_issues_text = build_existing_issues_summary(issue.iid, all_issues);
    let parent_priority = issue.priority();

    let prompt = build_split_prompt(
        project_name,
        issue,
        &comments_text,
        &existing_issues_text,
        parent_priority,
    )?;

    let agent_output = agent.run(&prompt)?;

    // --- NEEDS_CLARIFICATION: PMO itself cannot decide, ask human ---
    if agent_output.contains("NEEDS_CLARIFICATION") {
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
        gitlab.add_issue_label(issue.iid, PMO_PENDING_LABEL)?;
        return Ok(true); // keep claim
    }

    // --- ALREADY_DONE: work is already implemented, close the issue ---
    if agent_output.contains("ALREADY_DONE") {
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
    if agent_output.contains("GUIDE_WORKER") {
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

    if agent_output.contains("NO_SPLIT_NEEDED") {
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
        warn!(
            "PMO: No sub-issues extracted from agent output for issue #{}",
            issue.iid
        );
        gitlab.add_issue_comment(
            issue.iid,
            "PMO Agent was unable to split this issue. Manual intervention may be required.",
        )?;
        gitlab.add_issue_label(issue.iid, PMO_PROCESSED_LABEL)?;
        return Ok(false);
    }

    let pending_file = pending_split_file(pmo_dir, agent_id);
    let pending_split = PendingSplit {
        parent_issue_iid: issue.iid,
        parent_issue_title: issue.title.clone(),
        parent_priority: issue.priority(),
        sub_issues: sub_issues.clone(),
        created_issue_ids: Vec::new(),
    };
    save_pending_split(&pending_file, &pending_split)?;

    match resume_split(&pending_file, gitlab, &pending_split) {
        Ok(_) => {
            info!(
                "{}: Successfully split issue #{} into sub-issues",
                agent_id, issue.iid
            );
            delete_pending_split(&pending_file)?;
        }
        Err(e) => {
            error!("{}: Failed to create all sub-issues: {}", agent_id, e);
            return Err(e);
        }
    }

    // Close the parent issue — it served as a task container, the real work
    // is now tracked in the sub-issues.
    info!(
        "{}: Closing parent issue #{} (task container)",
        agent_id, issue.iid
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

fn build_split_prompt(
    project_name: &str,
    issue: &Issue,
    comments: &str,
    existing_issues: &str,
    parent_priority: u8,
) -> Result<String> {
    let prompt = format!(
        r#"You are a Project Management Office (PMO) agent responsible for triaging issues that an automated worker agent could not implement.

PROJECT: {project}

ISSUE #{iid}: {title}

DESCRIPTION:
{description}

COMMENTS AND FEEDBACK (read carefully — the worker's rejection reason is here):
{comments}

EXISTING OPEN ISSUES (check for overlap before creating sub-issues):
{existing}

CONTEXT:
An automated worker agent attempted to implement this issue but was unable to complete it.
The COMMENTS section above contains the worker's explanation of why it failed.
Your job is to analyze the failure reason and take the appropriate action.

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You do NOT write any code — you only write comments and create issue descriptions
- The worker agent has FULL ACCESS to shell commands (rm, mv, git, etc.) and all build/test tools
- If the worker claimed it "cannot run commands" or "cannot delete files", that is WRONG — it CAN. Instruct it clearly.

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

   PRIORITY LEVELS:
   - 1 = Critical: blocking other work, security fix, core dependency that other sub-issues depend on
   - 2 = High: important feature, depended on by lower-priority sub-issues
   - 3 = Normal: independent work, enhancements, nice-to-haves
   The parent issue has priority {parent_priority}. Sub-issues that are dependencies for others should get higher priority (lower number). Independent leaf tasks can inherit the parent priority or be lower.

3. ALREADY_DONE — Use when the work described in the issue is ALREADY fully implemented in the codebase:
   - The worker's output or your analysis shows the feature/tests/code already exists
   - There is nothing left to implement — the issue is simply outdated or redundant
   Respond with:
   ALREADY_DONE
   REASON: <Brief explanation of why this issue is already complete, referencing the existing code/files>

4. NEEDS_CLARIFICATION — Use when you CANNOT make a decision because:
   - The issue description is too vague to determine scope or intent
   - The worker's rejection and the issue together don't give enough context to guide or split
   - You need specific information from a human (e.g. which modules to cover, what the acceptance criteria are)
   Respond with:
   NEEDS_CLARIFICATION
   QUESTION:
   <Specific question(s) you need answered before you can guide or split this issue. Be precise about what information is missing.>

DUPLICATE / OVERLAP RULES (STRICT):
- Review the EXISTING OPEN ISSUES list above before creating any sub-issue.
- Do NOT create a sub-issue that duplicates or substantially overlaps with an existing open issue.
- If an existing OPEN (not IN-PROGRESS) issue covers part of the work, reference it (e.g. "See existing #42") instead of creating a new sub-issue for that part.
- If an existing IN-PROGRESS issue already covers it, simply skip that part entirely — do not create a sub-issue or reference.
- If ALL sub-issues would duplicate existing issues, choose GUIDE_WORKER instead and tell the worker which existing issues already cover the work.

INSTRUCTIONS:
1. Read the original issue description carefully
2. Read ALL comments — especially the worker's rejection reason
3. Review the EXISTING OPEN ISSUES to understand what is already tracked
4. TASK CONTAINER TEST: Does the issue describe a broad goal that involves multiple independent pieces of work (e.g. "add tests for all modules", "refactor X across the codebase", "check code for Y")? If YES → SPLIT. The parent issue is just a container; the real work is in the sub-issues.
5. GUIDANCE TEST: Is there ONE specific thing the worker misunderstood or did wrong? If YES → GUIDE_WORKER.
6. ENUMERATION TEST: If your guidance would list 2+ independent modules, files, or components → SPLIT, not GUIDE_WORKER.
7. CLARITY TEST: If you cannot determine what to do because the issue is too vague → NEEDS_CLARIFICATION.
8. COMPLETION TEST: Does the worker's output or the comments indicate the work is already fully implemented in the codebase? If YES → ALREADY_DONE.
9. Choose EXACTLY ONE of GUIDE_WORKER, SPLIT, ALREADY_DONE, or NEEDS_CLARIFICATION — never combine them.
10. When in doubt between GUIDE_WORKER and SPLIT, prefer SPLIT — it's better to create focused sub-issues than to give the worker a laundry list.

Proceed with analyzing the issue autonomously.
"#,
        project = project_name,
        iid = issue.iid,
        title = issue.title,
        description = issue.description,
        comments = comments,
        existing = existing_issues,
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

fn pending_split_file(pmo_dir: &str, agent_id: &str) -> String {
    std::path::Path::new(pmo_dir)
        .join(format!("{}_pending_split.json", agent_id))
        .to_string_lossy()
        .into_owned()
}

fn extract_already_done_reason(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("REASON:") {
        let rest = &agent_output[pos + 7..];
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "The work described in this issue is already fully implemented in the codebase.".to_string()
}

fn extract_clarification_question(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("QUESTION:") {
        let rest = &agent_output[pos + 9..];
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "The PMO agent could not determine how to proceed with this issue. Please provide more details about the expected scope and acceptance criteria.".to_string()
}

fn extract_guidance(agent_output: &str) -> String {
    let raw = if let Some(pos) = agent_output.find("INSTRUCTIONS:") {
        let rest = &agent_output[pos + 13..];
        let trimmed = rest.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            String::new()
        }
    } else if let Some(pos) = agent_output.find("REASON:") {
        let rest = &agent_output[pos + 7..];
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

fn extract_sub_issues(agent_output: &str) -> Vec<SubIssue> {
    let mut sub_issues = Vec::new();
    let lines: Vec<&str> = agent_output.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Look for SUB_ISSUE_N:
        if line.starts_with("SUB_ISSUE_") && line.ends_with(':') {
            let mut title = String::new();
            let mut description = String::new();
            let mut priority: Option<u8> = None;
            let mut in_description = false;

            i += 1;

            while i < lines.len() {
                let current_line = lines[i].trim();

                if current_line.starts_with("SUB_ISSUE_") && current_line.ends_with(':') {
                    break;
                }

                if current_line.starts_with("TITLE:") {
                    title = current_line
                        .strip_prefix("TITLE:")
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    in_description = false;
                } else if current_line.starts_with("PRIORITY:") {
                    if let Ok(p) = current_line
                        .strip_prefix("PRIORITY:")
                        .unwrap_or("")
                        .trim()
                        .parse::<u8>()
                        && (1..=3).contains(&p)
                    {
                        priority = Some(p);
                    }
                    in_description = false;
                } else if current_line.starts_with("DESCRIPTION:") {
                    in_description = true;
                } else if in_description && !current_line.is_empty() {
                    if !description.is_empty() {
                        description.push('\n');
                    }
                    description.push_str(current_line);
                }

                i += 1;
            }

            if !title.is_empty() && !description.is_empty() {
                sub_issues.push(SubIssue {
                    title,
                    description,
                    priority,
                });
            }

            continue;
        }

        i += 1;
    }

    debug!(
        "Extracted {} sub-issues from agent output",
        sub_issues.len()
    );
    sub_issues
}

fn save_pending_split(path: &str, pending: &PendingSplit) -> Result<()> {
    let json = serde_json::to_string_pretty(pending)?;
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

fn pmo_state_path(pmo_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(pmo_dir).join("pmo_state.json")
}

fn save_pmo_state(pmo_dir: &str, issue_iid: u64) {
    let path = pmo_state_path(pmo_dir);
    let json = format!(r#"{{"claimed_issue_iid":{}}}"#, issue_iid);
    if let Err(e) = fs::write(&path, json) {
        warn!("Failed to save PMO state: {}", e);
    }
}

fn clear_pmo_state(pmo_dir: &str) {
    let path = pmo_state_path(pmo_dir);
    let _ = fs::remove_file(path);
}

fn try_resume_pmo_state(agent_id: &str, pmo_dir: &str, gitlab: &GitLabClient) -> Option<u64> {
    let path = pmo_state_path(pmo_dir);
    let content = fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&content).ok()?;
    let issue_iid = v.get("claimed_issue_iid")?.as_u64()?;

    let claim_label = format!("claimed:{}", agent_id);
    match gitlab.get_issue(issue_iid) {
        Ok(issue) => {
            if issue.state != "opened" {
                info!(
                    "{}: Previously claimed issue #{} is {}, discarding state",
                    agent_id, issue_iid, issue.state
                );
                clear_pmo_state(pmo_dir);
                return None;
            }
            if !issue.labels.contains(&claim_label) {
                info!(
                    "{}: Claim label missing from issue #{}, discarding state",
                    agent_id, issue_iid
                );
                clear_pmo_state(pmo_dir);
                return None;
            }
            info!("{}: Resumed claim on issue #{}", agent_id, issue_iid);
            Some(issue_iid)
        }
        Err(e) => {
            warn!(
                "{}: Failed to verify issue #{}: {}, discarding state",
                agent_id, issue_iid, e
            );
            clear_pmo_state(pmo_dir);
            None
        }
    }
}

fn resume_split(pending_file: &str, gitlab: &GitLabClient, pending: &PendingSplit) -> Result<()> {
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
            format!(
                "{} (Part {}/{})",
                pending.parent_issue_title,
                index + 1,
                total_sub_issues
            )
        } else {
            format!(
                "{} (Part {}/{})",
                sub_issue.title,
                index + 1,
                total_sub_issues
            )
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
        assert!(should_process_issue(&issue));

        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            PMO_PROCESSED_LABEL.to_string(),
        ];
        assert!(!should_process_issue(&issue));

        issue.labels = vec![];
        assert!(!should_process_issue(&issue));

        issue.title = "[Draft] Test".to_string();
        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(!should_process_issue(&issue));

        issue.title = "Normal issue".to_string();
        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            PMO_PENDING_LABEL.to_string(),
        ];
        assert!(!should_process_issue(&issue));
    }

    #[test]
    fn test_extract_sub_issues() {
        let output = r#"
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
        "#;

        let sub_issues = extract_sub_issues(output);
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
        let output = r#"
SUB_ISSUE_1:
TITLE: Simple task
DESCRIPTION:
Do something simple.
        "#;

        let sub_issues = extract_sub_issues(output);
        assert_eq!(sub_issues.len(), 1);
        assert_eq!(sub_issues[0].priority, None);
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
