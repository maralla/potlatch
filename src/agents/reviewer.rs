use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use tokio::time::{Duration, sleep};
use tracing::{error, info, warn};

use crate::agent::Agent;
use crate::config::ReviewerConfig;
use crate::git::GitRepo;
use crate::gitlab::{GitLabClient, MergeRequest};

pub async fn run(repo_url: String, config: ReviewerConfig) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let reviewer_dir = format!("{}-reviewer", project_name);

    let git_repo = GitRepo::new(reviewer_dir.clone());

    if !git_repo.exists() {
        git_repo.clone(&repo_url)?;
    }

    let gitlab = GitLabClient::new(reviewer_dir.clone());
    let agent = Agent::new(reviewer_dir.clone(), config.model.clone());

    let mut merged_mrs: HashSet<u64> = HashSet::new();

    info!(
        "Reviewer poll interval: {} seconds",
        config.poll_interval_secs
    );

    loop {
        if let Err(e) =
            reviewer_cycle(&project_name, &git_repo, &gitlab, &agent, &mut merged_mrs).await
        {
            error!("Reviewer cycle error: {}", e);
        }

        sleep(Duration::from_secs(config.poll_interval_secs)).await;
    }
}

async fn reviewer_cycle(
    project_name: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    merged_mrs: &mut HashSet<u64>,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;
    git_repo.checkout_remote_branch(&default_branch)?;

    let mrs = gitlab.list_merge_requests()?;

    for mr in mrs {
        if mr.state != "opened" {
            continue;
        }

        if has_unresolved_comments(gitlab, mr.iid) {
            info!("MR !{} has unresolved comments, skipping", mr.iid);
            continue;
        }

        info!("Reviewing MR !{}: {}", mr.iid, mr.title);

        match review_merge_request(project_name, git_repo, gitlab, agent, &mr).await {
            Ok(approved) => {
                if approved {
                    info!("MR !{} approved and merged", mr.iid);
                    merged_mrs.insert(mr.iid);
                } else {
                    info!("MR !{} reviewed with feedback", mr.iid);
                }
            }
            Err(e) => {
                error!("Failed to review MR !{}: {}", mr.iid, e);
            }
        }
    }

    info!("Reviewer status: {} MRs merged total", merged_mrs.len());

    Ok(())
}

/// Check if the MR has any discussion threads that haven't been addressed.
/// Groups comments by discussion and checks each one:
/// - No discussions → not unresolved (first review)
/// - ALL discussions have a last note containing "addressed feedback" or "resolved" → all resolved
/// - ANY discussion's last note doesn't → still unresolved, wait for worker
fn has_unresolved_comments(gitlab: &GitLabClient, mr_iid: u64) -> bool {
    match gitlab.get_mr_comments(mr_iid) {
        Ok(comments) => {
            if comments.is_empty() {
                return false;
            }

            // Group comments by discussion
            let mut discussions: HashMap<&str, Vec<&crate::gitlab::Comment>> = HashMap::new();
            for comment in &comments {
                discussions
                    .entry(&comment.discussion_id)
                    .or_default()
                    .push(comment);
            }

            // Every discussion must have its last note marked as resolved
            for notes in discussions.values() {
                if let Some(last_note) = notes.last() {
                    let body = last_note.body.to_lowercase();
                    if !body.contains("addressed feedback") && !body.contains("resolved") {
                        return true;
                    }
                }
            }

            info!(
                "MR !{} all {} discussion(s) resolved, ready for re-review",
                mr_iid,
                discussions.len()
            );
            false
        }
        Err(e) => {
            warn!(
                "Failed to fetch comments for MR !{}: {}, skipping to be safe",
                mr_iid, e
            );
            true
        }
    }
}

async fn review_merge_request(
    project_name: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    mr: &MergeRequest,
) -> Result<bool> {
    git_repo.checkout_remote_branch(&mr.source_branch)?;

    let source_sha = git_repo.rev_parse("HEAD")?;
    let target_sha = git_repo.rev_parse(&format!("origin/{}", mr.target_branch))?;
    info!(
        "Reviewing MR !{}: {} ({}) -> {} ({})",
        mr.iid, mr.source_branch, source_sha, mr.target_branch, target_sha
    );

    // Merge target into source to review the merged result
    if !git_repo.try_merge(&mr.target_branch)? {
        warn!(
            "MR !{} has merge conflicts with {}",
            mr.iid, mr.target_branch
        );
        gitlab.add_mr_comment(
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

    // Return to target branch after review
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
                gitlab.add_mr_comment(
                    mr.iid,
                    "Approved, but automatic merge failed. Please merge manually.",
                )?;
            }
        }
    } else if agent_output.contains("REQUEST_CHANGES") {
        info!("MR !{} needs changes", mr.iid);

        let feedback = extract_review_feedback(&agent_output);
        gitlab.add_mr_comment(mr.iid, &feedback)?;
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

INSTRUCTIONS:
1. Review the full comment history to understand previous feedback and responses
2. The source branch has already been merged with the target branch locally - you are on the merged result
3. Review the code changes in the DIFF section thoroughly
4. Check if the implementation matches the stated goal
5. Run tests locally to verify they pass (do NOT rely on CI/CD)
6. Run linting locally to verify it passes (do NOT rely on CI/CD)
7. Check code quality, best practices, and potential issues
8. Only raise NEW issues not already covered in previous comments
9. Make autonomous decisions about approval or requesting changes

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

fn extract_project_name(repo_url: &str) -> Result<String> {
    let parts: Vec<&str> = repo_url.trim_end_matches('/').split('/').collect();
    let name = parts
        .last()
        .context("Invalid repository URL")?
        .trim_end_matches(".git");
    Ok(name.to_string())
}
