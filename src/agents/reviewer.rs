use anyhow::Result;
use rand::RngExt;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{claim, extract_project_name};
use crate::agent::Agent;
use crate::config::ReviewerConfig;
use crate::git::GitRepo;
use crate::gitlab::{GitLabClient, MergeRequest};

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
    config: ReviewerConfig,
    instance_id: usize,
    shutdown: Arc<AtomicBool>,
    base_dir: String,
) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let agent_id = format!("reviewer-{}", instance_id);
    let reviewer_dir = super::work_dir(&base_dir, &project_name, &agent_id);

    let git_repo = GitRepo::new(reviewer_dir.clone());
    let gitlab = GitLabClient::new(reviewer_dir.clone());
    let agent = Agent::new(reviewer_dir.clone(), config.model.clone(), shutdown.clone());

    let mut merged_mrs: HashSet<u64> = HashSet::new();
    let mut claimed_mr_iid: Option<u64> =
        try_resume_reviewer_state(&agent_id, &reviewer_dir, &gitlab);

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

        if let Err(e) = reviewer_cycle(
            &agent_id,
            &project_name,
            &reviewer_dir,
            &git_repo,
            &gitlab,
            &agent,
            &mut merged_mrs,
            &mut claimed_mr_iid,
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
    if let Some(mr_iid) = claimed_mr_iid {
        info!(
            "{}: Releasing claim on MR !{} before shutdown",
            agent_id, mr_iid
        );
        let _ = claim::release_mr_claim(&gitlab, mr_iid, &agent_id);
        clear_reviewer_state(&reviewer_dir);
    }
    let _ = git_repo.reset_hard();
    info!("{}: Stopped", agent_id);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reviewer_cycle(
    agent_id: &str,
    project_name: &str,
    reviewer_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    merged_mrs: &mut HashSet<u64>,
    claimed_mr_iid: &mut Option<u64>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }
    let _ = git_repo.reset_hard();
    git_repo.checkout_remote_branch(&default_branch)?;

    // If we still hold a claim from a previous cycle, release it now.
    // Mutual exclusivity during feedback is enforced by the
    // `has_unresolved_comments` check below — other reviewers will skip
    // MRs that still have open discussions.
    if let Some(held_iid) = claimed_mr_iid.take() {
        info!(
            "{}: Releasing held claim on MR !{} from previous cycle",
            agent_id, held_iid
        );
        let _ = claim::release_mr_claim(gitlab, held_iid, agent_id);
        clear_reviewer_state(reviewer_dir);
    }

    let mrs = gitlab.list_merge_requests()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    for mr in mrs {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if mr.state != "opened" {
            continue;
        }

        if claim::is_mr_claimed(&mr.labels) {
            debug!("{}: MR !{} already claimed, skipping", agent_id, mr.iid);
            continue;
        }

        if has_unresolved_comments(gitlab, mr.iid) {
            info!(
                "{}: MR !{} has unresolved comments, skipping",
                agent_id, mr.iid
            );
            continue;
        }

        if !claim::try_claim_mr(gitlab, mr.iid, agent_id, shutdown)? {
            info!("{}: Failed to claim MR !{}, skipping", agent_id, mr.iid);
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
            return Ok(());
        }

        *claimed_mr_iid = Some(mr.iid);
        save_reviewer_state(reviewer_dir, mr.iid);

        info!("{}: Reviewing MR !{}: {}", agent_id, mr.iid, mr.title);

        match review_merge_request(project_name, git_repo, gitlab, agent, &mr) {
            Ok(approved) => {
                if approved {
                    info!("{}: MR !{} approved and merged", agent_id, mr.iid);
                    merged_mrs.insert(mr.iid);
                    let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                    *claimed_mr_iid = None;
                    clear_reviewer_state(reviewer_dir);
                } else {
                    info!(
                        "{}: MR !{} reviewed with feedback, releasing claim",
                        agent_id, mr.iid
                    );
                    let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                    *claimed_mr_iid = None;
                    clear_reviewer_state(reviewer_dir);
                }
            }
            Err(e) => {
                error!("{}: Failed to review MR !{}: {}", agent_id, mr.iid, e);
                let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
                clear_reviewer_state(reviewer_dir);
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
            }
        }

        break;
    }

    info!("{}: {} MRs merged total", agent_id, merged_mrs.len());

    Ok(())
}

/// Check if the MR has any resolvable discussion threads that are still unresolved,
/// using the GitLab discussions API `resolved` / `resolvable` fields.
fn has_unresolved_comments(gitlab: &GitLabClient, mr_iid: u64) -> bool {
    match gitlab.get_unresolved_discussion_count(mr_iid) {
        Ok((unresolved, total)) => {
            if unresolved > 0 {
                info!(
                    "MR !{} has {}/{} unresolved discussion(s)",
                    mr_iid, unresolved, total
                );
                return true;
            }
            if total > 0 {
                info!(
                    "MR !{} all {} discussion(s) resolved, ready for re-review",
                    mr_iid, total
                );
            }
            false
        }
        Err(e) => {
            warn!(
                "Failed to fetch discussions for MR !{}: {}, skipping to be safe",
                mr_iid, e
            );
            true
        }
    }
}

fn review_merge_request(
    project_name: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    mr: &MergeRequest,
) -> Result<bool> {
    if has_bad_title_or_description(mr) {
        warn!(
            "MR !{} has a generic or missing title/description, requesting fix",
            mr.iid
        );
        let mut issues = Vec::new();
        if is_generic_title(&mr.title) {
            issues.push(format!(
                "The MR title `{}` is too generic. Please provide a concise, descriptive title that summarizes what the code changes actually do.",
                mr.title
            ));
        }
        if is_generic_description(&mr.description) {
            issues.push(
                "The MR description is missing or generic. Please provide a description that explains the goal, implementation approach, and testing."
                    .to_string(),
            );
        }
        gitlab.add_mr_discussion(
            mr.iid,
            &format!(
                "REQUEST_CHANGES — please fix the following before review:\n\n{}",
                issues.join("\n\n")
            ),
        )?;
        return Ok(false);
    }

    let _ = git_repo.reset_hard();
    git_repo.checkout_remote_branch(&mr.source_branch)?;

    let source_sha = git_repo.rev_parse("HEAD")?;
    let target_sha = git_repo.rev_parse(&format!("origin/{}", mr.target_branch))?;
    info!(
        "Reviewing MR !{}: {} ({}) -> {} ({})",
        mr.iid, mr.source_branch, source_sha, mr.target_branch, target_sha
    );

    if !git_repo.try_merge(&mr.target_branch)? {
        warn!(
            "MR !{} has merge conflicts with {}",
            mr.iid, mr.target_branch
        );
        gitlab.add_mr_discussion(
            mr.iid,
            &format!(
                "This MR has merge conflicts with `{}`. Please rebase or resolve conflicts before review can proceed.",
                mr.target_branch
            ),
        )?;
        git_repo.checkout_remote_branch(&mr.target_branch)?;
        return Ok(false);
    }

    let diff = git_repo.diff_against(&mr.target_branch)?;

    let prompt = build_review_prompt(project_name, gitlab, mr, &diff)?;

    let agent_output = agent.run(&prompt)?;

    git_repo.checkout_remote_branch(&mr.target_branch)?;

    if agent_output.contains("APPROVE") && agent_output.contains("LGTM") {
        info!("MR !{} approved by reviewer", mr.iid);

        match gitlab.merge_mr(mr.iid) {
            Ok(_) => {
                info!("Successfully merged MR !{}", mr.iid);
                return Ok(true);
            }
            Err(e) => {
                warn!("Failed to merge MR !{}: {}", mr.iid, e);
                gitlab.add_mr_discussion(
                    mr.iid,
                    &format!(
                        "Code is approved, but automatic merge failed (`{}`). \
                         This is likely due to merge conflicts with the target branch. \
                         Please rebase or resolve conflicts and push again.",
                        e
                    ),
                )?;
            }
        }
    } else if agent_output.contains("REQUEST_CHANGES") {
        info!("MR !{} needs changes", mr.iid);

        let feedback = extract_review_feedback(&agent_output);
        gitlab.add_mr_discussion(mr.iid, &feedback)?;
    }

    Ok(false)
}

fn build_review_prompt(
    project_name: &str,
    gitlab: &GitLabClient,
    mr: &MergeRequest,
    diff: &str,
) -> Result<String> {
    let comments = gitlab.get_mr_comments(mr.iid).unwrap_or_default();

    let comments_text = if comments.is_empty() {
        "No comments yet.".to_string()
    } else {
        comments
            .iter()
            .map(|c| format!("- {}: {}", c.author, c.body))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let diff_text = if diff.is_empty() {
        "No changes detected.".to_string()
    } else {
        diff.to_string()
    };

    let prompt = format!(
        r#"You are reviewing a merge request for a software project in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

DESCRIPTION:
{}

SOURCE BRANCH: {}
TARGET BRANCH: {}

DIFF (changes against {}):
{}

FULL COMMENT HISTORY:
{}

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You have FULL ACCESS to the local workspace, git, and all build/test tools
- You CAN and MUST execute git commands, run tests, run linters directly
- NEVER claim you cannot run commands - you have full access
- NEVER ask the user for input, confirmation, or decisions
- NEVER prompt for additional information interactively
- Make all review decisions autonomously based on the code and information provided
- Provide clear, actionable feedback in comments (do not ask questions)
- Review the full comment history to understand what feedback was already given and addressed
- Do NOT repeat feedback that has already been addressed
- You **MUST** refer to AGENTS.md for more info if exists

INSTRUCTIONS:
1. Review the full comment history to understand previous feedback and responses
2. The source branch has already been merged with the target branch locally - you are on the merged result
3. Review the code changes in the DIFF section thoroughly
4. Check if the implementation matches the stated goal
5. Run tests locally to verify they pass (do NOT rely on CI/CD)
6. Run linting locally to verify it passes (do NOT rely on CI/CD)
7. Check code quality, best practices, and potential issues
8. Only raise NEW issues not already covered in previous comments
9. Readability and maintainability must be ensured
10. Make autonomous decisions about approval or requesting changes

MR TITLE AND DESCRIPTION (STRICT — reject if violated):
- The MR title MUST be a concise, meaningful summary of the code changes. Reject if the title is generic (e.g. "Implementation changes", "Update", "Fix"), just an issue number, or contains markdown formatting like ** or backticks.
- The MR description MUST explain the goal, implementation approach, and testing. Reject if the description is empty, a single generic sentence (e.g. "Implementation completed."), or does not describe the actual changes.
- When rejecting for poor title/description, tell the worker exactly what is wrong and ask it to provide a proper MR_TITLE and MR_DESCRIPTION in its response.

CHANGE SIZE LIMITS (reject if exceeded):
- Non-test, non-generated code changes should be around ~500 lines. If substantially over, request the worker to split the MR.
- Total changes including tests should be around ~1500 lines. If substantially over, request a split.
- Auto-generated code (files containing comments like "generated by", "auto-generated", "DO NOT EDIT", or similar) should NOT be counted toward either limit and should NOT be reviewed for code quality. Skip reviewing auto-generated files entirely — only verify they are properly gitignored or legitimately needed.

TEST QUALITY (STRICT — reject if violated):
- Be very cautious about tests that appear to pass but do not actually test the main logic. Common red flags:
  * Empty test bodies or tests that only assert `true` / `assert!(true)`
  * Tests that hard-code the expected output instead of exercising the real function
  * Tests whose assertions are trivially satisfied regardless of whether the implementation is correct (e.g. checking a return type exists but not its value)
  * Tests that were mutated or weakened to make them pass (e.g. removing the core assertion, catching all exceptions and ignoring them, mocking the function under test itself)
  * Tests that test only a helper or stub but skip the main feature being implemented
- Every test MUST exercise the actual production code path it claims to cover. If a test does not meaningfully verify the behavior described in the issue, reject it and ask for a real test.

FILE HYGIENE (STRICT — reject if violated):
- Every file in the MR must be directly relevant to the final deliverable. Reject any file that is only useful during the development process but serves no purpose in the shipped result, such as scratch scripts, debug helpers, personal notes, TODO files, temporary test harnesses, or throwaway utilities.
- Files must be placed in the correct directory according to the project's conventions. A test file must live in the designated test directory, configuration files in the config directory, etc. If a file is in the wrong location, request it be moved before approving.
- Do NOT allow leftover artifacts: generated files that should be gitignored, editor config files, OS-specific metadata files (e.g. .DS_Store, Thumbs.db), or log files.
- If unsure whether a file belongs, check the project structure and AGENTS.md for conventions.

After your review, provide your decision:

If the MR is good to merge:
APPROVE
LGTM: <brief summary of what was reviewed>

If changes are needed:
REQUEST_CHANGES
FEEDBACK:
- <specific issue 1>
- <specific issue 2>
- <etc>

Proceed with the review autonomously. Do not ask for any user input.
"#,
        project_name,
        mr.iid,
        mr.title,
        mr.description,
        mr.source_branch,
        mr.target_branch,
        mr.target_branch,
        diff_text,
        comments_text
    );

    Ok(prompt)
}

const GENERIC_TITLES: &[&str] = &[
    "implementation changes",
    "implementation completed",
    "update",
    "changes",
    "fix",
    "fixes",
    "updates",
];

fn has_bad_title_or_description(mr: &MergeRequest) -> bool {
    is_generic_title(&mr.title) || is_generic_description(&mr.description)
}

fn is_generic_title(title: &str) -> bool {
    let lower = title.trim().to_lowercase();
    if lower.is_empty() {
        return true;
    }
    GENERIC_TITLES.contains(&lower.as_str())
}

fn is_generic_description(desc: &str) -> bool {
    let trimmed = desc.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    lower == "implementation completed."
        || lower == "implementation changes"
        || lower == "implementation changes."
}

fn extract_review_feedback(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("FEEDBACK:") {
        let feedback = &agent_output[pos + 9..];
        return feedback.trim().to_string();
    }

    if let Some(pos) = agent_output.find("REQUEST_CHANGES") {
        let feedback = &agent_output[pos + 15..];
        return format!("Changes requested:\n{}", feedback.trim());
    }

    "Please review the changes and address any issues.".to_string()
}

// ---------------------------------------------------------------------------
// Reviewer state persistence
// ---------------------------------------------------------------------------

fn reviewer_state_path(reviewer_dir: &str) -> std::path::PathBuf {
    Path::new(reviewer_dir).join("reviewer_state.json")
}

fn save_reviewer_state(reviewer_dir: &str, mr_iid: u64) {
    let path = reviewer_state_path(reviewer_dir);
    let json = format!("{{\"claimed_mr_iid\":{}}}", mr_iid);
    if let Err(e) = fs::write(&path, json) {
        warn!("Failed to save reviewer state: {}", e);
    }
}

fn clear_reviewer_state(reviewer_dir: &str) {
    let path = reviewer_state_path(reviewer_dir);
    let _ = fs::remove_file(&path);
}

fn try_resume_reviewer_state(
    agent_id: &str,
    reviewer_dir: &str,
    gitlab: &GitLabClient,
) -> Option<u64> {
    let path = reviewer_state_path(reviewer_dir);
    let content = fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&content).ok()?;
    let mr_iid = v.get("claimed_mr_iid")?.as_u64()?;

    // Verify the MR still exists, is open, and our claim label is still on it
    let claim_label = format!("claimed:{}", agent_id);
    match gitlab.get_merge_request(mr_iid) {
        Ok(mr) => {
            if mr.state != "opened" {
                info!(
                    "{}: Previously claimed MR !{} is {}, discarding state",
                    agent_id, mr_iid, mr.state
                );
                clear_reviewer_state(reviewer_dir);
                return None;
            }
            let has_claim = mr.labels.as_ref().is_some_and(|l| l.contains(&claim_label));
            if !has_claim {
                info!(
                    "{}: Claim label missing from MR !{}, discarding state",
                    agent_id, mr_iid
                );
                clear_reviewer_state(reviewer_dir);
                return None;
            }
            info!("{}: Resumed claim on MR !{}", agent_id, mr_iid);
            Some(mr_iid)
        }
        Err(e) => {
            warn!(
                "{}: Failed to verify MR !{}: {}, discarding state",
                agent_id, mr_iid, e
            );
            clear_reviewer_state(reviewer_dir);
            None
        }
    }
}
