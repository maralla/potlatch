use anyhow::Result;
use rand::RngExt;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{claim, extract_project_name};
use crate::agent::Agent;
use crate::config::ReviewerConfig;
use crate::git::GitRepo;
use crate::gitlab::{self, GitLabClient, MergeRequest};

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
    let mut claimed_mr_iid: Option<u64> = find_claimed_mr(&agent_id, &gitlab);

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
            "{}: Preserving claim on MR !{} for restart",
            agent_id, mr_iid
        );
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
    // Do not proceed to claim a new MR if the release fails.
    if let Some(held_iid) = *claimed_mr_iid {
        info!(
            "{}: Releasing held claim on MR !{} from previous cycle",
            agent_id, held_iid
        );
        match claim::release_mr_claim(gitlab, held_iid, agent_id) {
            Ok(_) => {
                *claimed_mr_iid = None;
            }
            Err(e) => {
                warn!(
                    "{}: Failed to release claim on MR !{}: {}, will retry next cycle",
                    agent_id, held_iid, e
                );
                return Ok(());
            }
        }
    }

    let mut mrs = gitlab.list_merge_requests()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // Sort MRs by the priority of their linked issue (lowest number first),
    // then by MR IID ascending (oldest MR first) for tie-breaking.
    let issues = gitlab.list_issues().unwrap_or_default();
    let priority_map: std::collections::HashMap<u64, u8> =
        issues.iter().map(|i| (i.iid, i.priority())).collect();
    mrs.sort_by(|a, b| {
        let pa = gitlab::issue_iid_from_branch(&a.source_branch)
            .and_then(|iid| priority_map.get(&iid).copied())
            .unwrap_or(gitlab::DEFAULT_PRIORITY);
        let pb = gitlab::issue_iid_from_branch(&b.source_branch)
            .and_then(|iid| priority_map.get(&iid).copied())
            .unwrap_or(gitlab::DEFAULT_PRIORITY);
        pa.cmp(&pb).then_with(|| a.iid.cmp(&b.iid))
    });

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

        info!("{}: Reviewing MR !{}: {}", agent_id, mr.iid, mr.title);

        match review_merge_request(project_name, reviewer_dir, git_repo, gitlab, agent, &mr) {
            Ok(approved) => {
                if approved {
                    info!("{}: MR !{} approved and merged", agent_id, mr.iid);
                    merged_mrs.insert(mr.iid);
                    let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                    *claimed_mr_iid = None;
                } else {
                    info!(
                        "{}: MR !{} reviewed with feedback, releasing claim",
                        agent_id, mr.iid
                    );
                    let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                    *claimed_mr_iid = None;
                }
            }
            Err(e) => {
                error!("{}: Failed to review MR !{}: {}", agent_id, mr.iid, e);
                let _ = claim::release_mr_claim(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
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
    reviewer_dir: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    mr: &MergeRequest,
) -> Result<bool> {
    // Check if the MR links to an issue via description first, then branch name
    let issue_iid = {
        let re = regex::Regex::new(r"(?i)closes?\s+#(\d+)").ok();
        re.and_then(|r| {
            r.captures(&mr.description)
                .and_then(|c| c.get(1)?.as_str().parse().ok())
        })
        .or_else(|| gitlab::issue_iid_from_branch(&mr.source_branch))
    };

    if issue_iid.is_none() {
        warn!(
            "MR !{} does not reference any issue, requesting fix",
            mr.iid
        );
        gitlab.add_mr_discussion(
            mr.iid,
            "This MR does not reference an issue. Please link it to the relevant issue by using a branch name like `issue-N` or adding `Closes #N` in the MR description.",
        )?;
        return Ok(false);
    }

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

    let prompt = build_review_prompt(project_name, gitlab, mr, &diff, issue_iid, reviewer_dir)?;

    let agent_output = agent.run(&prompt)?;

    git_repo.checkout_remote_branch(&mr.target_branch)?;

    if agent_output.contains("APPROVE") && agent_output.contains("LGTM") {
        info!("MR !{} approved by reviewer", mr.iid);

        // Re-check for unresolved discussions before merging — another reviewer
        // or the worker may have left new comments during the review.
        if has_unresolved_comments(gitlab, mr.iid) {
            warn!(
                "MR !{} approved but has unresolved discussions, skipping merge",
                mr.iid
            );
            return Ok(false);
        }

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
    issue_iid: Option<u64>,
    reviewer_dir: &str,
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

    let issue_context = if let Some(iid) = issue_iid {
        build_issue_context(gitlab, iid)
    } else {
        String::new()
    };

    let agents_md = load_agents_md(reviewer_dir);

    let prompt = format!(
        r#"You are reviewing a merge request for a software project in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

DESCRIPTION:
{}

SOURCE BRANCH: {}
TARGET BRANCH: {}
{}
PROJECT RULES (AGENTS.md — STRICT COMPLIANCE REQUIRED):
{}

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

INSTRUCTIONS:
1. Review the full comment history to understand previous feedback and responses
2. The source branch has already been merged with the target branch locally - you are on the merged result
3. Review the code changes in the DIFF section thoroughly
4. Check if the implementation matches the stated goal
5. COMPLETENESS CHECK (STRICT): Compare the DIFF against the LINKED ISSUE (title, description, and comments). Every requirement or item mentioned in the issue MUST be addressed in the implementation. If any part is missing or only partially implemented, list the missing items and REQUEST_CHANGES. This check is critical to avoid shipping incomplete features.
6. Run tests locally to verify they pass (do NOT rely on CI/CD)
7. Run linting locally to verify it passes (do NOT rely on CI/CD)
8. Check code quality, best practices, and potential issues
9. Only raise NEW issues not already covered in previous comments
10. Readability and maintainability must be ensured
11. Make autonomous decisions about approval or requesting changes

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

AGENTS.md COMPLIANCE (STRICT — reject if violated):
- The PROJECT RULES (AGENTS.md) section above contains the project's mandatory conventions and standards.
- You MUST check every code change in the DIFF against AGENTS.md rules. If the code violates any rule defined there (naming conventions, file structure, required patterns, forbidden patterns, testing requirements, etc.), you MUST reject and cite the specific rule being violated.
- AGENTS.md rules take precedence over general best practices when they conflict.
- If AGENTS.md specifies test locations, file naming, code style, or architecture patterns, verify the MR follows them exactly.

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
        issue_context,
        agents_md,
        mr.target_branch,
        diff_text,
        comments_text
    );

    Ok(prompt)
}

fn load_agents_md(reviewer_dir: &str) -> String {
    let path = std::path::Path::new(reviewer_dir).join("AGENTS.md");
    match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(_) => "No AGENTS.md found in the project.".to_string(),
    }
}

fn build_issue_context(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let mut ctx = String::new();

    match gitlab.get_issue(issue_iid) {
        Ok(issue) => {
            ctx.push_str(&format!(
                "\nLINKED ISSUE #{}: {}\n\nISSUE DESCRIPTION:\n{}\n",
                issue_iid, issue.title, issue.description
            ));
        }
        Err(e) => {
            warn!("Failed to fetch issue #{}: {}", issue_iid, e);
            return String::new();
        }
    }

    match gitlab.get_issue_comments(issue_iid) {
        Ok(comments) if !comments.is_empty() => {
            ctx.push_str("\nISSUE COMMENTS:\n");
            for c in &comments {
                ctx.push_str(&format!("- {}: {}\n", c.author, c.body));
            }
            ctx.push('\n');
        }
        _ => {}
    }

    ctx
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
// Reviewer claim recovery
// ---------------------------------------------------------------------------

/// Scan all open MRs for this reviewer's claim label.
/// The GitLab label is the single source of truth — no local state file needed.
fn find_claimed_mr(agent_id: &str, gitlab: &GitLabClient) -> Option<u64> {
    let claim_label = format!("claimed:{}", agent_id);

    match gitlab.list_merge_requests() {
        Ok(mrs) => {
            for mr in &mrs {
                if mr.state != "opened" {
                    continue;
                }
                let has_claim = mr
                    .labels
                    .as_ref()
                    .is_some_and(|l| l.contains(&claim_label));
                if has_claim {
                    info!(
                        "{}: Found existing claim on MR !{}, will release next cycle",
                        agent_id, mr.iid
                    );
                    return Some(mr.iid);
                }
            }
        }
        Err(e) => {
            warn!(
                "{}: Failed to scan MRs for existing claims: {}",
                agent_id, e
            );
        }
    }

    None
}
