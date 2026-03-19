use anyhow::{Context, Result};
use rand::RngExt;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{claim, extract_project_name};
use crate::agent::Agent;
use crate::config::WorkerConfig;
use crate::git::GitRepo;
use crate::gitlab::{GitLabClient, Issue};

const WORKING_ON_LABEL: &str = "in-progress";
const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";

/// The single issue a worker is pinned to for its full lifecycle.
struct ActiveIssue {
    issue_iid: u64,
    mr_iid: Option<u64>,
    branch_name: Option<String>,
    mr_created: bool,
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

pub fn run(
    repo_url: String,
    config: WorkerConfig,
    instance_id: usize,
    shutdown: Arc<AtomicBool>,
    base_dir: String,
) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let agent_id = format!("worker-{}", instance_id);
    let worker_dir = super::work_dir(&base_dir, &project_name, &agent_id);
    let sessions_dir = super::sessions_dir(&base_dir, &project_name);

    let git_repo = GitRepo::new(worker_dir.clone());
    let gitlab = GitLabClient::new(worker_dir.clone());
    let agent = Agent::new(worker_dir.clone(), config.model.clone(), shutdown.clone());

    // Try to resume an existing session from a previous run.
    // If the session file is missing (e.g. hard kill), fall back to scanning
    // GitLab issues for an orphaned claim label belonging to this worker.
    let mut active: Option<ActiveIssue> = try_resume_session(&agent_id, &sessions_dir, &gitlab)
        .or_else(|| find_claimed_issue(&agent_id, &gitlab));
    if let Some(ref a) = active {
        if let Some(mr_iid) = a.mr_iid {
            info!(
                "{}: Resumed issue #{} with MR !{}",
                agent_id, a.issue_iid, mr_iid
            );
        } else {
            info!("{}: Resumed issue #{} (no MR yet)", agent_id, a.issue_iid);
        }
    }

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

        if let Err(e) = worker_cycle(
            &agent_id,
            &project_name,
            &worker_dir,
            &sessions_dir,
            &git_repo,
            &gitlab,
            &agent,
            &mut active,
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
    cleanup_on_shutdown(&agent_id, &active, &sessions_dir, &git_repo);
    info!("{}: Stopped", agent_id);

    Ok(())
}

fn cleanup_on_shutdown(
    agent_id: &str,
    active: &Option<ActiveIssue>,
    sessions_dir: &str,
    git_repo: &GitRepo,
) {
    let Some(a) = active else { return };

    // Always keep the claim and session so this worker resumes on restart.
    // The session file (even with mr_iid=0) tells try_resume_session that
    // this worker owns the issue.
    info!(
        "{}: Preserving claim on issue #{} for restart (MR: {})",
        agent_id,
        a.issue_iid,
        a.mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
    );

    let mr_iid = a.mr_iid.unwrap_or(0);
    let _ = save_session(sessions_dir, a.issue_iid, mr_iid, agent_id);
    let _ = git_repo.reset_hard();
}

#[allow(clippy::too_many_arguments)]
fn worker_cycle(
    agent_id: &str,
    project_name: &str,
    worker_dir: &str,
    sessions_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    active: &mut Option<ActiveIssue>,
    shutdown: &AtomicBool,
) -> Result<()> {
    // If we have an active issue with an MR, watch the MR
    if let Some(a) = &*active
        && let Some(mr_iid) = a.mr_iid
    {
        // Check if the issue was closed externally (e.g. by PMO stale cleanup)
        if let Ok(issue) = gitlab.get_issue(a.issue_iid)
            && issue.state != "opened"
        {
            abandon_closed_issue(
                agent_id,
                a.issue_iid,
                Some(mr_iid),
                sessions_dir,
                git_repo,
                gitlab,
            );
            *active = None;
            return Ok(());
        }

        info!(
            "{}: Watching MR !{} for issue #{}",
            agent_id, mr_iid, a.issue_iid
        );

        match gitlab.get_merge_request(mr_iid) {
            Ok(mr) => {
                if mr.state == "merged" || mr.state == "closed" {
                    info!(
                        "{}: MR !{} is {}, releasing issue #{}",
                        agent_id, mr_iid, mr.state, a.issue_iid
                    );
                    let branch = format!("issue-{}", a.issue_iid);
                    let default_branch =
                        git_repo.get_default_branch().unwrap_or("main".to_string());
                    let _ = git_repo.reset_hard();
                    let _ = git_repo.checkout_remote_branch(&default_branch);
                    let _ = git_repo.delete_local_branch(&branch);
                    if mr.state == "merged" {
                        let _ = git_repo.delete_remote_branch(&branch);
                    }
                    let _ = claim::release_claim(gitlab, a.issue_iid, agent_id);
                    let _ = gitlab.remove_issue_label(a.issue_iid, WORKING_ON_LABEL);
                    cleanup_session_file(sessions_dir, a.issue_iid);
                    *active = None;
                    return Ok(());
                }

                match handle_mr_comments(project_name, sessions_dir, gitlab, git_repo, agent, &mr) {
                    Ok(true) => {
                        info!(
                            "{}: Issue #{} abandoned, MR !{} closed",
                            agent_id, a.issue_iid, mr_iid
                        );
                        let _ = claim::release_claim(gitlab, a.issue_iid, agent_id);
                        cleanup_session_file(sessions_dir, a.issue_iid);
                        *active = None;
                        return Ok(());
                    }
                    Err(e) => {
                        if shutdown.load(Ordering::SeqCst) {
                            return Ok(());
                        }
                        error!(
                            "{}: Failed to handle comments for MR !{}: {}",
                            agent_id, mr_iid, e
                        );
                    }
                    Ok(false) => {}
                }
            }
            Err(e) => {
                warn!("{}: Failed to check MR !{}: {}", agent_id, mr_iid, e);
            }
        }

        return Ok(());
    }

    // If we have an active issue without an MR, we were interrupted before
    // creating the MR. Re-attempt implementation from scratch.
    if let Some(ref a) = *active
        && a.mr_iid.is_none()
        && !a.mr_created
    {
        let issue_iid = a.issue_iid;

        // Check if the issue was closed externally
        if let Ok(issue) = gitlab.get_issue(issue_iid)
            && issue.state != "opened"
        {
            abandon_closed_issue(agent_id, issue_iid, None, sessions_dir, git_repo, gitlab);
            *active = None;
            return Ok(());
        }

        info!(
            "{}: Active issue #{} has no MR, re-attempting implementation",
            agent_id, issue_iid
        );

        match gitlab.get_issue(issue_iid) {
            Ok(issue) => {
                let mut current = ActiveIssue {
                    issue_iid,
                    mr_iid: None,
                    branch_name: None,
                    mr_created: false,
                };

                match process_issue(
                    agent_id,
                    project_name,
                    worker_dir,
                    sessions_dir,
                    git_repo,
                    gitlab,
                    agent,
                    &issue,
                    &mut current,
                ) {
                    Ok(_) => {
                        if current.mr_created {
                            *active = Some(current);
                        } else {
                            let _ = claim::release_claim(gitlab, issue_iid, agent_id);
                            cleanup_session_file(sessions_dir, issue_iid);
                            *active = None;
                        }
                    }
                    Err(e) => {
                        if shutdown.load(Ordering::SeqCst) {
                            *active = Some(current);
                            return Ok(());
                        }
                        error!(
                            "{}: Failed to re-process issue #{}: {}",
                            agent_id, issue_iid, e
                        );
                        let _ = claim::release_claim(gitlab, issue_iid, agent_id);
                        let _ = gitlab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                        cleanup_session_file(sessions_dir, issue_iid);
                        *active = None;
                        if let Some(ref branch) = current.branch_name {
                            let default_branch =
                                git_repo.get_default_branch().unwrap_or("main".to_string());
                            let _ = git_repo.reset_hard();
                            let _ = git_repo.checkout_remote_branch(&default_branch);
                            let _ = git_repo.delete_local_branch(branch);
                        }
                    }
                }

                return Ok(());
            }
            Err(e) => {
                warn!(
                    "{}: Failed to fetch issue #{} for re-attempt: {}, releasing",
                    agent_id, issue_iid, e
                );
                let _ = claim::release_claim(gitlab, issue_iid, agent_id);
                let _ = gitlab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                cleanup_session_file(sessions_dir, issue_iid);
                *active = None;
            }
        }
    }

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // No active issue — try to pick up an orphaned session first
    if active.is_none() {
        *active = try_adopt_orphaned_session(agent_id, sessions_dir, gitlab, shutdown);
        if let Some(a) = &*active {
            info!(
                "{}: Adopted orphaned issue #{} with MR !{}",
                agent_id,
                a.issue_iid,
                a.mr_iid.unwrap_or(0)
            );
            return Ok(());
        }
    }

    // No orphaned sessions — poll for new issues
    info!("{}: Polling for new issues...", agent_id);
    let issues = gitlab.list_issues()?;

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

        // Persist a pre-session immediately so that even a SIGKILL leaves
        // a record of which issue this worker owns. mr_iid=0 means no MR yet.
        let _ = save_session(sessions_dir, issue.iid, 0, agent_id);

        info!(
            "{}: Implementing issue #{}: {}",
            agent_id, issue.iid, issue.title
        );

        let mut current = ActiveIssue {
            issue_iid: issue.iid,
            mr_iid: None,
            branch_name: None,
            mr_created: false,
        };

        match process_issue(
            agent_id,
            project_name,
            worker_dir,
            sessions_dir,
            git_repo,
            gitlab,
            agent,
            &issue,
            &mut current,
        ) {
            Ok(_) => {
                if current.mr_created {
                    *active = Some(current);
                } else {
                    // Rejected or no MR — release claim
                    let _ = claim::release_claim(gitlab, issue.iid, agent_id);
                }
            }
            Err(e) => {
                if shutdown.load(Ordering::SeqCst) {
                    // Shutdown during processing — store as active for cleanup
                    *active = Some(current);
                    return Ok(());
                }
                error!(
                    "{}: Failed to process issue #{}: {}",
                    agent_id, issue.iid, e
                );
                let _ = claim::release_claim(gitlab, issue.iid, agent_id);
                let _ = gitlab.remove_issue_label(issue.iid, WORKING_ON_LABEL);
                if let Some(ref branch) = current.branch_name {
                    let default_branch =
                        git_repo.get_default_branch().unwrap_or("main".to_string());
                    let _ = git_repo.reset_hard();
                    let _ = git_repo.checkout_remote_branch(&default_branch);
                    let _ = git_repo.delete_local_branch(branch);
                }
            }
        }

        break;
    }

    if active.is_none() {
        info!("{}: Idle, no issues to work on", agent_id);
    }

    Ok(())
}

fn should_skip_issue(issue: &Issue) -> bool {
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return true;
    }

    if issue.labels.contains(&"do-not-implement".to_string()) {
        return true;
    }

    if issue.labels.contains(&WORKING_ON_LABEL.to_string()) {
        return true;
    }

    if issue.labels.contains(&ACTION_REQUIRED_LABEL.to_string()) {
        return true;
    }

    if issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
        return true;
    }

    if issue.labels.contains(&"pmo-pending".to_string()) {
        return true;
    }

    false
}

#[allow(clippy::too_many_arguments)]
fn process_issue(
    agent_id: &str,
    project_name: &str,
    worker_dir: &str,
    sessions_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    issue: &Issue,
    current: &mut ActiveIssue,
) -> Result<Option<u64>> {
    // Check if there's already an *open* MR for this issue
    if let Some(mr_iid) = find_open_mr_for_issue(gitlab, issue.iid) {
        info!(
            "Issue #{} already has open MR !{}, tracking it",
            issue.iid, mr_iid
        );
        current.mr_iid = Some(mr_iid);
        current.mr_created = true;
        let _ = gitlab.add_issue_label(issue.iid, WORKING_ON_LABEL);
        save_session(sessions_dir, issue.iid, mr_iid, agent_id)?;
        return Ok(Some(mr_iid));
    }

    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;
    let _ = git_repo.reset_hard();

    let branch_name = format!("issue-{}", issue.iid);

    let branch_existed = if git_repo.remote_branch_exists(&branch_name)? {
        info!(
            "Branch {} already exists on remote, checking if it's stale",
            branch_name
        );
        git_repo.checkout_remote_branch(&branch_name)?;

        // Check if the branch has any diff against the target — if not, it's
        // stale (content already merged). Delete and start fresh.
        if !git_repo.has_diff_against(&default_branch)? {
            warn!(
                "Branch {} has no diff against {}, discarding stale branch",
                branch_name, default_branch
            );
            let _ = git_repo.reset_hard();
            git_repo.checkout_remote_branch(&default_branch)?;
            let _ = git_repo.delete_local_branch(&branch_name);
            let _ = git_repo.delete_remote_branch(&branch_name);
            git_repo.create_branch_from(&branch_name, &default_branch)?;
            false
        } else if !git_repo.try_merge(&default_branch)? {
            warn!(
                "Branch {} has conflicts with {}, creating fresh branch instead",
                branch_name, default_branch
            );
            let _ = git_repo.reset_hard();
            git_repo.checkout_remote_branch(&default_branch)?;
            let _ = git_repo.delete_local_branch(&branch_name);
            git_repo.create_branch_from(&branch_name, &default_branch)?;
            false
        } else {
            true
        }
    } else {
        git_repo.create_branch_from(&branch_name, &default_branch)?;
        false
    };

    current.branch_name = Some(branch_name.clone());

    gitlab.add_issue_label(issue.iid, WORKING_ON_LABEL)?;

    let prompt = if branch_existed {
        build_continuation_prompt(project_name, worker_dir, issue)?
    } else {
        build_implementation_prompt(project_name, worker_dir, issue)?
    };

    let issue_iid_for_cancel = issue.iid;
    let cancel_check = || {
        gitlab
            .get_issue(issue_iid_for_cancel)
            .is_ok_and(|i| i.state != "opened")
    };
    let agent_output = agent.run_with_cancel(&prompt, Some(&cancel_check))?;

    if agent_output.contains("CANNOT_IMPLEMENT") {
        let reason = if agent_output.contains("NEEDS_SPLIT") {
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
        gitlab.add_issue_comment(issue.iid, &reason)?;

        // If the branch already had an open MR, close it
        if let Some(mr_iid) = find_open_mr_for_issue(gitlab, issue.iid) {
            gitlab.add_mr_comment(
                mr_iid,
                &format!(
                    "Closing this MR — the issue cannot be implemented:\n\n{}",
                    reason
                ),
            )?;
            let _ = gitlab.close_mr(mr_iid);
        }

        // Reset git to a clean state — keep remote branch for potential retry
        let default_branch = git_repo.get_default_branch().unwrap_or("main".to_string());
        let _ = git_repo.reset_hard();
        let _ = git_repo.checkout_remote_branch(&default_branch);
        let _ = git_repo.delete_local_branch(&branch_name);

        gitlab.remove_issue_label(issue.iid, WORKING_ON_LABEL)?;
        gitlab.add_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        current.branch_name = None;
        info!(
            "Issue #{} requires user action, labeled with '{}'",
            issue.iid, ACTION_REQUIRED_LABEL
        );
        return Ok(None);
    }

    // The agent may have already committed changes itself (it has full shell
    // access). Stage+commit any remaining uncommitted work, then check whether
    // the branch diverges from the base at all.
    git_repo.add_all()?;

    let mr_title = extract_mr_title(&agent_output);

    if git_repo.has_staged_changes()? {
        let commit_message = build_commit_message(&mr_title, issue.iid);
        git_repo.commit(&commit_message)?;
    }

    if !git_repo.has_diff_against(&default_branch)? {
        warn!(
            "Issue #{}: agent produced no code changes, rejecting",
            issue.iid
        );
        let reason = "The implementation produced no code changes. The issue may need more detail or a different approach.";
        gitlab.add_issue_comment(issue.iid, reason)?;
        let _ = git_repo.reset_hard();
        let _ = git_repo.checkout_remote_branch(&default_branch);
        let _ = git_repo.delete_local_branch(&branch_name);
        gitlab.remove_issue_label(issue.iid, WORKING_ON_LABEL)?;
        gitlab.add_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        current.branch_name = None;
        return Ok(None);
    }

    git_repo.push(&branch_name)?;
    let mr_description = format!(
        "Closes #{}\n\n{}",
        issue.iid,
        extract_mr_description(&agent_output)
    );

    let mr_iid = gitlab.create_merge_request(&branch_name, &mr_title, &mr_description)?;
    current.mr_iid = Some(mr_iid);
    current.mr_created = true;

    info!("Created MR !{} for issue #{}", mr_iid, issue.iid);

    let impl_summary = extract_mr_description(&agent_output);
    save_session_with_summary(sessions_dir, issue.iid, mr_iid, agent_id, &impl_summary)?;

    Ok(Some(mr_iid))
}

// ---------------------------------------------------------------------------
// MR comment handling
// ---------------------------------------------------------------------------

/// Returns `Ok(true)` if the agent decided the issue cannot be resolved and
/// the MR was closed + issue rejected.
fn handle_mr_comments(
    project_name: &str,
    sessions_dir: &str,
    gitlab: &GitLabClient,
    git_repo: &GitRepo,
    agent: &Agent,
    mr: &crate::gitlab::MergeRequest,
) -> Result<bool> {
    let unresolved_ids = gitlab.get_unresolved_discussion_ids(mr.iid)?;

    if unresolved_ids.is_empty() && !mr.has_conflicts {
        return Ok(false);
    }

    if !unresolved_ids.is_empty() {
        info!(
            "MR !{} has {} unresolved discussion(s) to address",
            mr.iid,
            unresolved_ids.len()
        );
    }
    if mr.has_conflicts {
        info!("MR !{} has merge conflicts to resolve", mr.iid);
    }

    let comments = gitlab.get_mr_comments(mr.iid)?;

    git_repo.fetch()?;
    let _ = git_repo.reset_hard();
    git_repo.checkout_remote_branch(&mr.source_branch)?;

    // Remember the remote HEAD so we can detect changes after the agent runs,
    // even if the agent disobeys and commits/pushes itself.
    let pre_agent_sha = git_repo.rev_parse(&format!("origin/{}", mr.source_branch))?;

    // Merge the latest target branch so the worker has up-to-date upstream code.
    let default_branch = git_repo.get_default_branch().unwrap_or("main".to_string());
    let merge_ok = git_repo.merge_no_abort(&default_branch)?;
    if !merge_ok {
        warn!(
            "MR !{}: source branch has conflicts with {}, worker agent will resolve them",
            mr.iid, default_branch
        );
    }

    let issue_number = extract_issue_number_from_branch(&mr.source_branch)?;
    let issue_context = load_issue_context(gitlab, issue_number)?;
    let implementation_summary = load_implementation_summary(sessions_dir, issue_number);

    let all_comments_text = comments
        .iter()
        .map(|c| format!("- {}: {}", c.author, c.body))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent with FULL shell access. You MUST execute all commands yourself. Never say you cannot run commands — you can and must.

You are addressing reviewer feedback on a merge request in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

ORIGINAL ISSUE CONTEXT:
{}

ORIGINAL IMPLEMENTATION:
{}

MR DESCRIPTION:
{}

ALL COMMENTS (full conversation):
{}

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system running with --trust mode (full shell access granted)
- You MUST execute all necessary commands yourself — there is no human to do anything for you
- You have FULL shell access: rm, mv, cp, mkdir, git, python, cargo, npm, etc.
- You MUST run tests, linters, build commands directly — do not suggest them, EXECUTE them
- You MUST delete, rename, or move files as needed — do not ask permission or suggest it
- You MUST NOT say "I cannot run commands" or "please run this" — YOU run everything
- You MUST NOT produce passive output suggesting a human take action — YOU take all actions
- Do NOT run `git add`, `git commit`, or `git push` — the system handles staging, committing, and pushing automatically after you finish
- Do NOT create merge requests or pull requests (e.g. via `glab mr create`, `gh pr create`, or any API call) — the system manages them automatically
- Review ALL comments to understand the full conversation
- Identify which feedback items still need to be addressed
- Address all unresolved feedback autonomously
- Make all necessary code changes to resolve the comments
- Keep the original issue requirements in mind while addressing feedback
- If the workspace has merge conflict markers (<<<<<<< / ======= / >>>>>>>), resolve ALL of them before doing anything else. Edit each conflicted file to keep the correct version.

INSTRUCTIONS:
1. First, check for merge conflicts: run `git status` and look for "Unmerged paths" or "both modified". If any exist, resolve ALL conflicts in every file before proceeding.
2. Review the original issue and what was implemented
3. Review ALL comments to understand the full conversation and context
4. Identify which feedback items are still unresolved
5. Make the necessary code changes to address all unresolved feedback
6. After making changes, RUN tests and linters to verify everything passes. If the reviewer asked you to run tests or fix linting — you MUST actually execute those commands (e.g. `cargo test`, `cargo clippy`, `python -m pytest`, `npm test`, etc.) and fix any failures.
7. If the reviewer asked you to delete, rename, or move files — do it directly with `rm`, `mv`, `mkdir`, etc.
8. Ensure changes align with both the original requirements and reviewer feedback
9. If the reviewer says code changes are too large (above ~1500 lines total or ~500 non-test lines), you have TWO options:
   a) Adjust your implementation to reduce changed lines — simplify, remove unnecessary changes, trim scope — then re-run tests
   b) If you cannot reasonably reduce the size, respond with CANNOT_RESOLVE so the issue is rejected and the problem is reported back
   Do NOT try to split the issue yourself — that is handled by the PMO agent, not you.
10. If you determine that the feedback cannot be resolved without additional human input (e.g. the requirements are ambiguous, the reviewer is asking for something outside the scope of the issue, or the necessary information is missing), respond with:
   CANNOT_RESOLVE
   REASON: <explain concisely why this cannot be resolved autonomously and what input is needed>
11. If the reviewer asked you to fix the MR title or description, include updated versions in your response:
   MR_TITLE: <SHORT title (max 8-10 words) stating the main feature or fix — no enumeration of details, no markdown>
   MR_DESCRIPTION:
   <full description with goal, implementation, and testing sections>
12. After addressing feedback, provide a summary:
   CHANGES_SUMMARY: <A concise sentence summarizing the substance of the changes made — this will be used as the git commit message, so it must convey the main idea of what was changed>

REMINDER: You are fully autonomous. Execute every command, test, and file operation yourself. Never output instructions for a human.

Proceed with addressing the feedback autonomously. Do not ask for any user input.
"#,
        project_name,
        mr.iid,
        mr.title,
        issue_context,
        implementation_summary,
        mr.description,
        all_comments_text
    );

    let cancel_check = || {
        gitlab
            .get_issue(issue_number)
            .is_ok_and(|i| i.state != "opened")
    };
    let agent_output = agent.run_with_cancel(&prompt, Some(&cancel_check))?;

    if agent_output.contains("CANNOT_RESOLVE") {
        let reason = extract_cannot_resolve_reason(&agent_output);
        warn!("MR !{} cannot be resolved autonomously: {}", mr.iid, reason);
        abandon_mr(gitlab, git_repo, mr, issue_number, sessions_dir, &reason)?;
        return Ok(true);
    }

    // If the agent provided updated MR_TITLE / MR_DESCRIPTION (e.g. the
    // reviewer asked for a better title), update the MR metadata.
    let new_title = extract_mr_title(&agent_output);
    let new_desc = extract_mr_description(&agent_output);
    if new_title != "Implementation changes" || new_desc != "Implementation completed." {
        let title = if new_title != "Implementation changes" {
            &new_title
        } else {
            &mr.title
        };
        let desc = if new_desc != "Implementation completed." {
            &new_desc
        } else {
            &mr.description
        };
        if let Err(e) = gitlab.update_mr_title_description(mr.iid, title, desc) {
            warn!("Failed to update MR !{} metadata: {}", mr.iid, e);
        } else {
            info!(
                "Updated MR !{} title/description from agent feedback",
                mr.iid
            );
        }
    }

    // Detect all changes the agent made: working tree, staged, or committed
    // (even if the agent disobeyed and ran git commit/push itself).
    // Re-fetch in case the agent pushed.
    git_repo.fetch()?;
    let has_new_changes = git_repo.has_changes_since(&pre_agent_sha)?;

    if has_new_changes {
        // Stage and commit any uncommitted leftovers
        git_repo.add_all()?;
        let summary_for_commit = extract_changes_summary(&agent_output);
        if git_repo.has_staged_changes()? {
            let commit_msg = build_commit_message(&summary_for_commit, issue_number);
            git_repo.commit(&commit_msg)?;
        }
    }

    let summary = extract_changes_summary(&agent_output);
    if has_new_changes {
        git_repo.push(&mr.source_branch)?;
        info!("Pushed changes addressing feedback for MR !{}", mr.iid);
    } else {
        info!(
            "Agent processed comments for MR !{} but made no code changes",
            mr.iid
        );
    }
    // Re-fetch unresolved discussions — the original list may have been empty
    // if we were triggered by has_conflicts alone. After pushing, resolve all
    // remaining open discussions (including conflict comments).
    let ids_to_resolve = if unresolved_ids.is_empty() {
        gitlab
            .get_unresolved_discussion_ids(mr.iid)
            .unwrap_or_default()
    } else {
        unresolved_ids
    };
    let reply_body = if has_new_changes {
        format!("Addressed feedback:\n\n{}", summary)
    } else {
        "Resolved".to_string()
    };
    for discussion_id in &ids_to_resolve {
        if let Err(e) = gitlab.reply_to_discussion(mr.iid, discussion_id, &reply_body) {
            warn!("Failed to reply to discussion {}: {}", discussion_id, e);
        }
        if let Err(e) = gitlab.resolve_discussion(mr.iid, discussion_id) {
            warn!("Failed to resolve discussion {}: {}", discussion_id, e);
        }
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// Session file management (shared directory)
// ---------------------------------------------------------------------------

fn session_file_path(sessions_dir: &str, issue_iid: u64) -> std::path::PathBuf {
    Path::new(sessions_dir).join(format!("issue_{}.json", issue_iid))
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionFile {
    issue_iid: u64,
    /// 0 means no MR created yet (issue claimed but implementation not done).
    mr_iid: u64,
    #[serde(default)]
    agent_id: Option<String>,
    implementation_summary: Option<String>,
}

fn save_session(sessions_dir: &str, issue_iid: u64, mr_iid: u64, agent_id: &str) -> Result<()> {
    let session = SessionFile {
        issue_iid,
        mr_iid,
        agent_id: Some(agent_id.to_string()),
        implementation_summary: None,
    };
    let path = session_file_path(sessions_dir, issue_iid);
    let json = serde_json::to_string_pretty(&session)?;
    fs::write(&path, json).context("Failed to write session file")?;
    debug!("Saved session for issue #{} -> MR !{}", issue_iid, mr_iid);
    Ok(())
}

fn save_session_with_summary(
    sessions_dir: &str,
    issue_iid: u64,
    mr_iid: u64,
    agent_id: &str,
    summary: &str,
) -> Result<()> {
    let session = SessionFile {
        issue_iid,
        mr_iid,
        agent_id: Some(agent_id.to_string()),
        implementation_summary: Some(summary.to_string()),
    };
    let path = session_file_path(sessions_dir, issue_iid);
    let json = serde_json::to_string_pretty(&session)?;
    fs::write(&path, json).context("Failed to write session file")?;
    Ok(())
}

fn load_session(sessions_dir: &str, issue_iid: u64) -> Option<SessionFile> {
    let path = session_file_path(sessions_dir, issue_iid);
    let content = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn cleanup_session_file(sessions_dir: &str, issue_iid: u64) {
    let path = session_file_path(sessions_dir, issue_iid);
    if fs::remove_file(&path).is_ok() {
        info!("Cleaned up session file for issue #{}", issue_iid);
    }
}

/// On startup, try to resume a session this worker previously owned.
/// Checks the stored agent_id first, then falls back to checking GitLab labels.
fn try_resume_session(
    agent_id: &str,
    sessions_dir: &str,
    gitlab: &GitLabClient,
) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", agent_id);

    let entries = match fs::read_dir(sessions_dir) {
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
        if !file_name.starts_with("issue_") || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix("issue_")
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };
        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = load_session(sessions_dir, issue_iid) else {
            continue;
        };

        // Fast path: session file records which agent owned it
        if let Some(ref stored_id) = session.agent_id {
            if stored_id == agent_id {
                let (mr_iid, mr_created) = if session.mr_iid > 0 {
                    (Some(session.mr_iid), true)
                } else {
                    // mr_iid == 0 means issue was claimed but no MR was created
                    // before shutdown. Check if an MR was created in the meantime.
                    match find_open_mr_for_issue(gitlab, issue_iid) {
                        Some(mr) => (Some(mr), true),
                        None => (None, false),
                    }
                };
                info!(
                    "{}: Found session file for issue #{} (MR: {}), resuming",
                    agent_id,
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
            // Session belongs to a different agent — skip
            continue;
        }

        // Fallback for old session files without agent_id: check GitLab labels
        match gitlab.get_issue(issue_iid) {
            Ok(issue) if issue.labels.contains(&claim_label) => {
                let (mr_iid, mr_created) = if session.mr_iid > 0 {
                    (Some(session.mr_iid), true)
                } else {
                    match find_open_mr_for_issue(gitlab, issue_iid) {
                        Some(mr) => (Some(mr), true),
                        None => (None, false),
                    }
                };
                info!(
                    "{}: Found unclaimed session for issue #{} with matching label, resuming",
                    agent_id, issue_iid
                );
                return Some(ActiveIssue {
                    issue_iid,
                    mr_iid,
                    branch_name: Some(format!("issue-{}", issue_iid)),
                    mr_created,
                });
            }
            Ok(_) => {
                debug!(
                    "{}: Session for issue #{} exists but claim label not found",
                    agent_id, issue_iid
                );
            }
            Err(e) => {
                warn!(
                    "{}: Failed to verify issue #{} on GitLab: {}, skipping",
                    agent_id, issue_iid, e
                );
            }
        }
    }

    None
}

/// Scan all open GitLab issues for this worker's claim label.
/// Used as a fallback when the session file is missing (e.g. hard kill / crash).
fn find_claimed_issue(agent_id: &str, gitlab: &GitLabClient) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", agent_id);

    let issues = match gitlab.list_issues() {
        Ok(i) => i,
        Err(e) => {
            warn!(
                "{}: Failed to scan issues for orphaned claims: {}",
                agent_id, e
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

        let (mr_iid, mr_created) = match find_open_mr_for_issue(gitlab, issue.iid) {
            Some(mr) => (Some(mr), true),
            None => (None, false),
        };

        info!(
            "{}: Found orphaned claim on issue #{} (MR: {}), adopting it",
            agent_id,
            issue.iid,
            mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
        );

        return Some(ActiveIssue {
            issue_iid: issue.iid,
            mr_iid,
            branch_name: Some(format!("issue-{}", issue.iid)),
            mr_created,
        });
    }

    None
}

/// Try to adopt an orphaned session file (issue has no claim label from any worker).
fn try_adopt_orphaned_session(
    agent_id: &str,
    sessions_dir: &str,
    gitlab: &GitLabClient,
    shutdown: &AtomicBool,
) -> Option<ActiveIssue> {
    let entries = fs::read_dir(sessions_dir).ok()?;
    for entry in entries.flatten() {
        let Ok(file_name) = entry.file_name().into_string() else {
            continue;
        };
        if !file_name.starts_with("issue_") || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix("issue_")
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };
        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = load_session(sessions_dir, issue_iid) else {
            continue;
        };

        // Skip sessions that belong to a specific agent — those should be
        // resumed by their owner via try_resume_session, not adopted.
        if session.agent_id.is_some() {
            continue;
        }

        let Ok(issue) = gitlab.get_issue(issue_iid) else {
            continue;
        };

        // Skip if already claimed by someone
        if claim::is_claimed(&issue.labels) {
            continue;
        }

        let (mr_iid, mr_created) = if session.mr_iid > 0 {
            // Check if the MR is still open
            if let Ok(mr) = gitlab.get_merge_request(session.mr_iid) {
                if mr.state == "merged" || mr.state == "closed" {
                    info!(
                        "{}: Orphaned session for issue #{} has {} MR !{}, cleaning up",
                        agent_id, issue_iid, mr.state, session.mr_iid
                    );
                    cleanup_session_file(sessions_dir, issue_iid);
                    let _ = gitlab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                    continue;
                }
            } else {
                continue;
            }
            (Some(session.mr_iid), true)
        } else {
            // No MR yet — check if one was created in the meantime
            match find_open_mr_for_issue(gitlab, issue_iid) {
                Some(mr) => (Some(mr), true),
                None => (None, false),
            }
        };

        // Try to claim this issue
        match claim::try_claim_issue(gitlab, issue_iid, agent_id, shutdown) {
            Ok(true) => {
                info!(
                    "{}: Adopted orphaned issue #{} (MR: {})",
                    agent_id,
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
    if let Ok(mrs) = gitlab.list_merge_requests() {
        for mr in mrs {
            if mr.source_branch == branch_name && mr.state == "opened" {
                return Some(mr.iid);
            }
        }
    }
    None
}

/// Clean up everything when the issue we're working on has been closed externally.
/// Closes any related MR, deletes branches, releases the claim, and removes session.
fn abandon_closed_issue(
    agent_id: &str,
    issue_iid: u64,
    mr_iid: Option<u64>,
    sessions_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
) {
    info!(
        "{}: Issue #{} was closed externally, abandoning work",
        agent_id, issue_iid
    );

    if let Some(mr) = mr_iid {
        let _ = gitlab.add_mr_comment(mr, "Closing this MR — the linked issue has been closed.");
        let _ = gitlab.close_mr(mr);
    }

    let branch = format!("issue-{}", issue_iid);
    let default_branch = git_repo.get_default_branch().unwrap_or("main".to_string());
    let _ = git_repo.reset_hard();
    let _ = git_repo.checkout_remote_branch(&default_branch);
    let _ = git_repo.delete_local_branch(&branch);
    let _ = git_repo.delete_remote_branch(&branch);
    let _ = claim::release_claim(gitlab, issue_iid, agent_id);
    let _ = gitlab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
    cleanup_session_file(sessions_dir, issue_iid);
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

fn load_implementation_summary(sessions_dir: &str, issue_number: u64) -> String {
    if let Some(session) = load_session(sessions_dir, issue_number)
        && let Some(summary) = session.implementation_summary
    {
        return summary;
    }
    "No previous implementation summary available.".to_string()
}

fn abandon_mr(
    gitlab: &GitLabClient,
    git_repo: &GitRepo,
    mr: &crate::gitlab::MergeRequest,
    issue_iid: u64,
    sessions_dir: &str,
    reason: &str,
) -> Result<()> {
    gitlab.add_mr_comment(
        mr.iid,
        &format!(
            "Closing this MR — the issue cannot be resolved autonomously:\n\n{}",
            reason
        ),
    )?;
    let _ = gitlab.close_mr(mr.iid);

    let default_branch = git_repo.get_default_branch().unwrap_or("main".to_string());
    let _ = git_repo.reset_hard();
    let _ = git_repo.checkout_remote_branch(&default_branch);
    let _ = git_repo.delete_local_branch(&mr.source_branch);

    let _ = gitlab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
    gitlab.add_issue_label(issue_iid, ACTION_REQUIRED_LABEL)?;
    gitlab.add_issue_comment(
        issue_iid,
        &format!(
            "This issue requires additional human input before it can be implemented:\n\n{}",
            reason
        ),
    )?;

    cleanup_session_file(sessions_dir, issue_iid);

    info!(
        "Abandoned MR !{} and rejected issue #{} with action-required",
        mr.iid, issue_iid
    );
    Ok(())
}

fn extract_cannot_resolve_reason(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("REASON:") {
        let reason = &agent_output[pos + 7..];
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

fn extract_changes_summary(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("CHANGES_SUMMARY:") {
        let raw = &agent_output[pos + 16..];
        let line = if let Some(end) = raw.find('\n') {
            raw[..end].trim()
        } else {
            raw.trim()
        };
        return strip_markdown_formatting(line);
    }
    "Changes made to address reviewer feedback.".to_string()
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

fn build_implementation_prompt(
    project_name: &str,
    worker_dir: &str,
    issue: &Issue,
) -> Result<String> {
    let agents_md = read_project_docs(worker_dir)?;
    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(false);
    let output_format = get_output_format();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent with FULL shell access. You MUST execute all commands yourself. Never say you cannot run commands — you can and must.

You are implementing a feature for a software project in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

DESCRIPTION:
{}

PROJECT DOCUMENTATION:
{}

{}

{}

INSTRUCTIONS:
1. Analyze the issue carefully
2. Estimate the number of changed lines:
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
3. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the feature can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be implemented as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
4. If the issue is unclear or missing critical information that makes implementation impossible, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
5. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
6. If at any point you determine the issue simply cannot be implemented without additional human input that you cannot infer or assume (e.g. missing API credentials, undocumented external system dependencies, contradictory requirements), respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain precisely what input is needed and why you cannot proceed>
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the issue, REJECT it. Do not guess or produce speculative code.
- If you believe the implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

7. If the issue is clear, focused, and reasonably sized (or cannot be split), implement ONLY what is asked
8. Make all necessary code changes autonomously
9. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
10. {}

REMINDER: You are fully autonomous. Execute every command, test, and file operation yourself. Never output instructions for a human.

Proceed with the implementation autonomously. Do not ask for any user input.
"#,
        project_name,
        issue.iid,
        issue.title,
        issue.description,
        agents_md,
        common_requirements,
        scope_rules,
        output_format
    );

    Ok(prompt)
}

fn build_continuation_prompt(
    project_name: &str,
    worker_dir: &str,
    issue: &Issue,
) -> Result<String> {
    let agents_md = read_project_docs(worker_dir)?;
    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(true);
    let output_format = get_output_format();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent with FULL shell access. You MUST execute all commands yourself. Never say you cannot run commands — you can and must.

You are continuing work on an existing feature branch in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

DESCRIPTION:
{}

PROJECT DOCUMENTATION:
{}

CONTEXT:
- A branch for this issue already exists with previous work
- You are continuing the implementation from where it was left off
- Review the existing code changes in this branch
- Complete any remaining work needed to fully implement the issue

{}

{}

INSTRUCTIONS:
1. Review the existing changes in the current branch
2. Analyze what has been done and what remains
3. Estimate total changed lines (including existing + remaining work):
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
4. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the remaining work can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be completed as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
5. If the issue is unclear or missing critical information that makes implementation impossible, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
6. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
7. If at any point you determine the remaining work simply cannot be completed without additional human input that you cannot infer or assume, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain precisely what input is needed and why you cannot proceed>
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the remaining work, REJECT it. Do not guess or produce speculative code.
- If you believe the total implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

8. If the issue is clear, focused, and reasonably sized (or cannot be split), continue the implementation
9. ONLY implement what the issue asks for, nothing more
10. Complete any remaining work autonomously
11. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
12. {}

REMINDER: You are fully autonomous. Execute every command, test, and file operation yourself. Never output instructions for a human.

Proceed with continuing the implementation autonomously. Do not ask for any user input.
"#,
        project_name,
        issue.iid,
        issue.title,
        issue.description,
        agents_md,
        common_requirements,
        scope_rules,
        output_format
    );

    Ok(prompt)
}

fn get_common_requirements() -> &'static str {
    r#"CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system running with --trust mode (full shell access granted)
- You MUST execute all necessary commands yourself — there is no human to do anything for you
- You have FULL shell access: rm, mv, cp, mkdir, cat, grep, sed, git, python, pip, cargo, npm, make, etc.
- You MUST run tests, linters, and build commands directly — do not suggest them, EXECUTE them
- You MUST delete, rename, move, or create files as needed — do not ask permission or suggest it
- You MUST NOT say "I cannot run commands", "I don't have permission", or "please run this command"
- You MUST NOT ask the user for input, confirmation, or decisions — decide autonomously
- You MUST NOT produce output that suggests actions for a human to take — YOU take those actions
- You MUST NOT be passive — if a file needs deleting, delete it; if a test needs running, run it
- Do NOT run `git add` or `git commit` — the system handles staging and committing automatically after you finish
- Do NOT run `git push` — the system handles pushing automatically
- Do NOT create merge requests or pull requests (e.g. via `glab mr create`, `gh pr create`, or any API call) — the system creates them automatically after you finish
- If information is missing, document what's needed in your response (do not ask interactively)
- If you are making code changes you MUST stick to AGENTS.md in the project strictly
- Read the issue comments carefully — they may contain guidance from the PMO agent on how to proceed"#
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
- Do NOT add features, refactors, or integrations not described in the issue
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
        line_context,
        if is_continuation {
            "completed"
        } else {
            "implemented"
        }
    )
}

fn get_output_format() -> &'static str {
    r#"MANDATORY OUTPUT — you MUST include these EXACT markers at the end of your response:

MR_TITLE: <SHORT title (max 8-10 words) stating the main feature or fix. Focus on WHAT, not HOW or HOW MUCH. Good: "Add unit tests for BaseProcessor". Bad: "Restore MySQL reporting tests, remove unrelated test files, and add 8 edge case tests to achieve 100% coverage". No markdown, no **, no backticks.>

MR_DESCRIPTION:
## Goal
<What is the goal of this MR? What problem does it solve?>

## Implementation
<How was it implemented? What approach was taken? What are the key changes?>

## Testing
<What testing was done or should be done?>

IMPORTANT: The MR_TITLE and MR_DESCRIPTION markers are REQUIRED. Without them, the system cannot create the merge request properly."#
}

fn read_project_docs(worker_dir: &str) -> Result<String> {
    let base = Path::new(worker_dir);
    let mut docs = String::new();

    if let Ok(content) = fs::read_to_string(base.join("AGENTS.md")) {
        docs.push_str("=== AGENTS.md ===\n");
        docs.push_str(&content);
        docs.push_str("\n\n");
    }

    if let Ok(content) = fs::read_to_string(base.join("README.md")) {
        docs.push_str("=== README.md ===\n");
        docs.push_str(&content);
        docs.push_str("\n\n");
    }

    if docs.is_empty() {
        docs = "No project documentation found.".to_string();
    }

    Ok(docs)
}

fn extract_split_reason(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("NEEDS_SPLIT:") {
        let reason = &agent_output[pos + 12..];
        return reason.trim().to_string();
    }
    "This issue is too broad and requires large unrelated feature work. Please split it into smaller, focused issues with detailed descriptions.".to_string()
}

fn extract_clarification(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("NEEDS_CLARIFICATION:") {
        let clarification = &agent_output[pos + 20..];
        if let Some(end) = clarification.find('\n') {
            return clarification[..end].trim().to_string();
        }
        return clarification.trim().to_string();
    }
    "This issue needs clarification. Please provide more details.".to_string()
}

fn extract_mr_title(agent_output: &str) -> String {
    // Try exact marker first
    if let Some(pos) = agent_output.find("MR_TITLE:") {
        let title_section = &agent_output[pos + 9..];
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
        if let Some(pos) = agent_output.find(marker) {
            let section = &agent_output[pos + marker.len()..];
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
    if let Some(pos) = agent_output.find("CHANGES_SUMMARY:") {
        let section = &agent_output[pos + 16..];
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

    "Implementation changes".to_string()
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

fn extract_mr_description(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("MR_DESCRIPTION:") {
        let desc_section = &agent_output[pos + 15..];
        let trimmed = desc_section.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Some(pos) = agent_output.find("IMPLEMENTATION_SUMMARY:") {
        let summary = &agent_output[pos + 23..];
        let trimmed = summary.trim();
        if !trimmed.is_empty() {
            return format!("## Implementation\n\n{}", trimmed);
        }
    }

    // Try to extract the last substantial paragraph as a summary
    let lines: Vec<&str> = agent_output.lines().collect();
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
        return format!("## Summary\n\n{}", last_chunk.join("\n"));
    }

    "Implementation completed.".to_string()
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
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project/").unwrap(),
            "project"
        );
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
        issue.labels = vec!["do-not-implement".to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKING_ON_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![PMO_PROCESSED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec!["pmo-pending".to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![];
        assert!(!should_skip_issue(&issue));
    }
}
