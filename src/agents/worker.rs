use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs;
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info, warn};

use crate::agent::Agent;
use crate::config::WorkerConfig;
use crate::git::GitRepo;
use crate::gitlab::{GitLabClient, Issue};

const WORKING_ON_LABEL: &str = "in-progress";
const ACTION_REQUIRED_LABEL: &str = "action-required";

pub async fn run(repo_url: String, config: WorkerConfig) -> Result<()> {
    let project_name = extract_project_name(&repo_url)?;
    let worker_dir = format!("{}-worker", project_name);

    let git_repo = GitRepo::new(worker_dir.clone());

    if !git_repo.exists() {
        git_repo.clone(&repo_url)?;
    }

    let gitlab = GitLabClient::new(worker_dir.clone());
    let agent = Agent::new(worker_dir.clone(), config.model.clone());

    let mut processed_issues: HashSet<u64> = HashSet::new();
    let mut active_mrs: HashSet<u64> = HashSet::new();

    // Load existing MRs and their associated issues from session files on startup
    load_existing_mrs(&mut active_mrs, &mut processed_issues);

    info!(
        "Worker poll interval: {} seconds",
        config.poll_interval_secs
    );
    if !active_mrs.is_empty() {
        info!(
            "Resumed tracking {} active MR(s) for {} issue(s)",
            active_mrs.len(),
            processed_issues.len()
        );
    }

    loop {
        if let Err(e) = worker_cycle(
            &project_name,
            &git_repo,
            &gitlab,
            &agent,
            &mut processed_issues,
            &mut active_mrs,
        )
        .await
        {
            error!("Worker cycle error: {}", e);
        }

        sleep(Duration::from_secs(config.poll_interval_secs)).await;
    }
}

async fn worker_cycle(
    project_name: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    processed_issues: &mut HashSet<u64>,
    active_mrs: &mut HashSet<u64>,
) -> Result<()> {
    if !active_mrs.is_empty() {
        info!(
            "Checking {} active MR(s) for new comments...",
            active_mrs.len()
        );
    }
    check_active_mrs(
        project_name,
        gitlab,
        git_repo,
        agent,
        processed_issues,
        active_mrs,
    )
    .await?;

    info!("Polling for new issues...");
    let issues = gitlab.list_issues()?;

    let mut found_new_issue = false;
    for issue in issues {
        if should_skip_issue(&issue) {
            continue;
        }

        if processed_issues.contains(&issue.iid) {
            continue;
        }

        found_new_issue = true;
        info!("Implementing issue #{}: {}", issue.iid, issue.title);

        match process_issue(project_name, git_repo, gitlab, agent, &issue).await {
            Ok(mr_iid) => {
                processed_issues.insert(issue.iid);
                if let Some(iid) = mr_iid {
                    active_mrs.insert(iid);
                    info!(
                        "Worker status: now tracking MR !{} for issue #{}",
                        iid, issue.iid
                    );
                }
            }
            Err(e) => {
                error!("Failed to process issue #{}: {}", issue.iid, e);
            }
        }

        break;
    }

    if !found_new_issue {
        if active_mrs.is_empty() {
            info!("Worker idle: no new issues, no active MRs");
        } else {
            info!(
                "Worker waiting: no new issues, monitoring {} active MR(s)",
                active_mrs.len()
            );
        }
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

    false
}

async fn process_issue(
    project_name: &str,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    agent: &Agent,
    issue: &Issue,
) -> Result<Option<u64>> {
    // Check if MR already exists for this issue
    if let Some(mr_iid) = check_mr_exists_for_issue(gitlab, issue.iid) {
        info!(
            "Issue #{} already has MR !{}, skipping implementation",
            issue.iid, mr_iid
        );
        return Ok(Some(mr_iid));
    }

    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;

    let branch_name = format!("issue-{}", issue.iid);

    let branch_existed = if git_repo.branch_exists(&branch_name)? {
        info!(
            "Branch {} already exists, continuing work on it",
            branch_name
        );
        git_repo.checkout_branch(&branch_name)?;
        true
    } else {
        git_repo.create_branch_from(&branch_name, &default_branch)?;
        false
    };

    gitlab.add_issue_label(issue.iid, WORKING_ON_LABEL)?;

    let prompt = if branch_existed {
        build_continuation_prompt(project_name, issue)?
    } else {
        build_implementation_prompt(project_name, issue)?
    };

    let agent_output = match agent.run(&prompt) {
        Ok(output) => output,
        Err(e) => {
            error!("Agent failed for issue #{}: {}", issue.iid, e);
            gitlab.remove_issue_label(issue.iid, WORKING_ON_LABEL)?;
            return Ok(None);
        }
    };

    if agent_output.contains("CANNOT_IMPLEMENT") {
        if agent_output.contains("NEEDS_SPLIT") {
            warn!("Issue #{} is too broad, needs splitting", issue.iid);
            let split_reason = extract_split_reason(&agent_output);
            gitlab.add_issue_comment(
                issue.iid,
                &format!(
                    "This issue needs to be split into smaller, focused issues:\n\n{}",
                    split_reason
                ),
            )?;
        } else {
            warn!("Issue #{} needs clarification", issue.iid);
            let clarification = extract_clarification(&agent_output);
            gitlab.add_issue_comment(issue.iid, &clarification)?;
        }
        gitlab.remove_issue_label(issue.iid, WORKING_ON_LABEL)?;
        gitlab.add_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
        info!("Issue #{} requires user action, labeled with '{}'", issue.iid, ACTION_REQUIRED_LABEL);
        return Ok(None);
    }

    git_repo.add_all()?;

    let commit_message = if branch_existed {
        format!("Continue work on issue #{}: {}", issue.iid, issue.title)
    } else {
        format!("Implement issue #{}: {}", issue.iid, issue.title)
    };
    git_repo.commit(&commit_message)?;
    git_repo.push(&branch_name)?;

    // Extract MR title and description from agent output
    let mr_title = extract_mr_title(&agent_output);
    let mr_description = format!(
        "Closes #{}\n\n{}",
        issue.iid,
        extract_mr_description(&agent_output)
    );

    let mr_iid = gitlab.create_merge_request(&branch_name, &mr_title, &mr_description)?;

    info!("Created MR !{} for issue #{}", mr_iid, issue.iid);

    // Save session summary with implementation details
    save_session_summary_with_implementation(issue, mr_iid, &agent_output)?;

    Ok(Some(mr_iid))
}

async fn handle_mr_comments(
    project_name: &str,
    gitlab: &GitLabClient,
    git_repo: &GitRepo,
    agent: &Agent,
    mr: &crate::gitlab::MergeRequest,
) -> Result<()> {
    let comments = gitlab.get_mr_comments(mr.iid)?;

    if comments.is_empty() {
        return Ok(());
    }

    // Group comments by discussion, find unresolved discussions
    let mut discussions: std::collections::HashMap<&str, Vec<&crate::gitlab::Comment>> =
        std::collections::HashMap::new();
    for comment in &comments {
        discussions
            .entry(&comment.discussion_id)
            .or_default()
            .push(comment);
    }

    let unresolved_ids: Vec<&str> = discussions
        .iter()
        .filter(|(_, notes)| {
            if let Some(last) = notes.last() {
                let body = last.body.to_lowercase();
                !body.contains("addressed feedback") && !body.contains("resolved")
            } else {
                false
            }
        })
        .map(|(id, _)| *id)
        .collect();

    if unresolved_ids.is_empty() {
        return Ok(());
    }

    info!(
        "MR !{} has {} unresolved discussion(s) to address",
        mr.iid,
        unresolved_ids.len()
    );

    git_repo.fetch()?;
    git_repo.checkout_branch(&mr.source_branch)?;

    let issue_number = extract_issue_number_from_branch(&mr.source_branch)?;
    let issue_context = load_issue_context(gitlab, issue_number)?;
    let implementation_summary = load_implementation_summary(issue_number);

    // Full comment history for context
    let all_comments_text = comments
        .iter()
        .map(|c| format!("- {}: {}", c.author, c.body))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"You are addressing reviewer feedback on a merge request in a fully automated, non-interactive environment.

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
- This is a NON-INTERACTIVE automated system
- You have FULL ACCESS to the local workspace, git, and all build/test tools
- You CAN and MUST execute git commands, run tests, run linters directly
- NEVER claim you cannot run commands - you have full access
- NEVER ask the user for input, confirmation, or decisions
- Review ALL comments to understand the full conversation
- Identify which feedback items still need to be addressed
- Address all unresolved feedback autonomously
- Make all necessary code changes to resolve the comments
- Keep the original issue requirements in mind while addressing feedback

INSTRUCTIONS:
1. Review the original issue and what was implemented
2. Review ALL comments to understand the full conversation and context
3. Identify which feedback items are still unresolved
4. Make the necessary code changes to address all unresolved feedback
5. Ensure changes align with both the original requirements and reviewer feedback
6. After addressing feedback, provide a summary:
   CHANGES_SUMMARY: <describe what was changed to address the feedback>

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

    let agent_output = agent.run(&prompt)?;

    git_repo.add_all()?;
    let has_changes = git_repo.has_staged_changes()?;
    if has_changes {
        git_repo.commit(&format!("Address reviewer feedback on MR !{}", mr.iid))?;
        git_repo.push(&mr.source_branch)?;
        info!("Pushed changes addressing feedback for MR !{}", mr.iid);
    } else {
        info!(
            "Agent processed comments for MR !{} but made no code changes",
            mr.iid
        );
    }

    // Reply to every unresolved discussion
    let summary = extract_changes_summary(&agent_output);
    let reply_body = if has_changes {
        format!("Addressed feedback:\n\n{}", summary)
    } else {
        "Resolved".to_string()
    };
    for discussion_id in &unresolved_ids {
        if let Err(e) = gitlab.reply_to_discussion(mr.iid, discussion_id, &reply_body) {
            warn!("Failed to reply to discussion {}: {}", discussion_id, e);
        }
    }

    Ok(())
}

fn cleanup_session_file(source_branch: &str) {
    if let Ok(issue_number) = extract_issue_number_from_branch(source_branch) {
        let filename = format!("issue_{}.md", issue_number);
        if fs::remove_file(&filename).is_ok() {
            info!("Cleaned up session file {}", filename);
        }
    }
}

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

fn load_implementation_summary(issue_number: u64) -> String {
    let session_file = format!("issue_{}.md", issue_number);

    if let Ok(content) = fs::read_to_string(&session_file) {
        // Try to extract implementation summary from session file
        if let Some(pos) = content.find("Implementation:") {
            let summary = &content[pos..];
            if let Some(end) = summary.find("\n\n") {
                return summary[..end].to_string();
            }
            return summary.to_string();
        }

        // Return the whole session file if no specific summary found
        return format!("Previous session:\n{}", content);
    }

    "No previous implementation summary available.".to_string()
}

fn extract_changes_summary(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("CHANGES_SUMMARY:") {
        let summary = &agent_output[pos + 16..];
        return summary.trim().to_string();
    }

    "Changes made to address reviewer feedback.".to_string()
}

async fn check_active_mrs(
    project_name: &str,
    gitlab: &GitLabClient,
    git_repo: &GitRepo,
    agent: &Agent,
    _processed_issues: &mut HashSet<u64>,
    active_mrs: &mut HashSet<u64>,
) -> Result<()> {
    let mut to_remove = Vec::new();

    for &mr_iid in active_mrs.iter() {
        match gitlab.get_merge_request(mr_iid) {
            Ok(mr) => {
                if mr.state == "merged" || mr.state == "closed" {
                    info!("MR !{} is {}, removing from active list", mr_iid, mr.state);
                    to_remove.push(mr_iid);
                    cleanup_session_file(&mr.source_branch);
                    continue;
                }

                if let Err(e) = handle_mr_comments(project_name, gitlab, git_repo, agent, &mr).await
                {
                    error!("Failed to handle comments for MR !{}: {}", mr_iid, e);
                }
            }
            Err(e) => {
                warn!("Failed to check MR !{}: {}", mr_iid, e);
            }
        }
    }

    for mr_iid in to_remove {
        active_mrs.remove(&mr_iid);
    }

    Ok(())
}

fn build_implementation_prompt(project_name: &str, issue: &Issue) -> Result<String> {
    let agents_md = read_project_docs()?;
    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(false);
    let output_format = get_output_format();

    let prompt = format!(
        r#"You are implementing a feature for a software project in a fully automated, non-interactive environment.

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
2. Estimate the number of changed lines (excluding tests/generated code) this implementation will require
3. If the estimate is substantially larger than ~500 lines:
   - Evaluate if the feature can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be implemented as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
4. If the issue is unclear or missing critical information, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
5. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
6. If the issue is clear, focused, and reasonably sized (or cannot be split), implement ONLY what is asked
7. Make all necessary code changes autonomously
8. Make reasonable assumptions when minor details are unclear
9. {}

Proceed with the implementation autonomously. Do not ask for any user input.
"#,
        project_name, issue.iid, issue.title, issue.description, agents_md,
        common_requirements, scope_rules, output_format
    );

    Ok(prompt)
}

fn build_continuation_prompt(project_name: &str, issue: &Issue) -> Result<String> {
    let agents_md = read_project_docs()?;
    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(true);
    let output_format = get_output_format();

    let prompt = format!(
        r#"You are continuing work on an existing feature branch in a fully automated, non-interactive environment.

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
3. Estimate total changed lines (including existing + remaining work, excluding tests/generated code)
4. If the total estimate is substantially larger than ~500 lines:
   - Evaluate if the remaining work can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be completed as one unit, proceed with implementation
   - Otherwise, respond with:
     CANNOT_IMPLEMENT
     NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
5. If the issue is unclear or missing critical information, respond with:
   CANNOT_IMPLEMENT
   NEEDS_CLARIFICATION: <explain what information is needed and why>
6. If the issue requires large unrelated feature work, respond with:
   CANNOT_IMPLEMENT
   NEEDS_SPLIT: <explain how to split the issue>
7. If the issue is clear, focused, and reasonably sized (or cannot be split), continue the implementation
8. ONLY implement what the issue asks for, nothing more
9. Complete any remaining work autonomously
10. Make reasonable assumptions when minor details are unclear
11. {}

Proceed with continuing the implementation autonomously. Do not ask for any user input.
"#,
        project_name, issue.iid, issue.title, issue.description, agents_md,
        common_requirements, scope_rules, output_format
    );

    Ok(prompt)
}

fn get_common_requirements() -> &'static str {
    r#"CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You have FULL ACCESS to the local workspace, git, and all build/test tools
- You CAN and MUST execute git commands, run tests, run linters directly
- NEVER claim you cannot run commands - you have full access
- NEVER ask the user for input, confirmation, or decisions
- NEVER prompt for additional information interactively
- Make all decisions autonomously based on the information provided
- If information is missing, document what's needed in your response (do not ask interactively)
- If you are making code changes stick to AGENTS.md in the project strictly."#
}

fn get_scope_rules(is_continuation: bool) -> String {
    let line_context = if is_continuation {
        "Review existing changes and estimate remaining work"
    } else {
        "Before starting implementation, estimate if the changes will be significantly larger than 500 lines"
    };

    format!(
        r#"SCOPE RULES:
- ONLY implement what the issue specifically asks for, nothing more
- Do NOT add features, refactors, or integrations not described in the issue
- CRITICAL: The implementation should stay around 500 changed lines (excluding unit tests and generated code)
- {}
- If total changes would be substantially larger than ~500 lines (excluding tests/generated code):
  * First, carefully evaluate if the feature can be split into smaller, independent pieces
  * If you are VERY SURE the feature CANNOT be split and MUST be {} as one atomic unit, you may proceed
  * Otherwise, respond with:
    CANNOT_IMPLEMENT
    NEEDS_SPLIT: <explain the estimated line count and how to split into smaller issues>
- If implementing the issue requires a large feature integration that is mainly unrelated to the task, respond with:
  CANNOT_IMPLEMENT
  NEEDS_SPLIT: <explain why the issue is too broad and how to split it>"#,
        line_context,
        if is_continuation { "completed" } else { "implemented" }
    )
}

fn get_output_format() -> &'static str {
    r#"After implementation, provide the following information:
   
   MR_TITLE: <A concise title summarizing what this MR does>
   
   MR_DESCRIPTION:
   ## Goal
   <What is the goal of this MR? What problem does it solve?>
   
   ## Implementation
   <How was it implemented? What approach was taken? What are the key changes?>
   
   ## Testing
   <What testing was done or should be done?>"#
}

fn read_project_docs() -> Result<String> {
    let mut docs = String::new();

    if let Ok(content) = fs::read_to_string("AGENTS.md") {
        docs.push_str("=== AGENTS.md ===\n");
        docs.push_str(&content);
        docs.push_str("\n\n");
    }

    if let Ok(content) = fs::read_to_string("README.md") {
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
    if let Some(pos) = agent_output.find("MR_TITLE:") {
        let title_section = &agent_output[pos + 9..];
        // Get the first line after MR_TITLE:
        if let Some(end) = title_section.find('\n') {
            return title_section[..end].trim().to_string();
        }
        return title_section.trim().to_string();
    }

    // Fallback to a generic title
    "Implementation changes".to_string()
}

fn extract_mr_description(agent_output: &str) -> String {
    if let Some(pos) = agent_output.find("MR_DESCRIPTION:") {
        let desc_section = &agent_output[pos + 15..];
        return desc_section.trim().to_string();
    }

    // Fallback: try old format
    if let Some(pos) = agent_output.find("IMPLEMENTATION_SUMMARY:") {
        let summary = &agent_output[pos + 23..];
        return format!("## Implementation\n\n{}", summary.trim());
    }

    "Implementation completed.".to_string()
}

fn save_session_summary_with_implementation(
    issue: &Issue,
    mr_iid: u64,
    agent_output: &str,
) -> Result<()> {
    let implementation_summary = extract_mr_description(agent_output);

    let summary = format!(
        "# Issue #{}: {}\n\nMR: !{}\nStatus: MR Created\nTimestamp: {}\n\nImplementation:\n{}\n",
        issue.iid,
        issue.title,
        mr_iid,
        chrono::Utc::now().to_rfc3339(),
        implementation_summary
    );

    let filename = format!("issue_{}.md", issue.iid);
    fs::write(&filename, summary).context("Failed to write session summary")?;

    Ok(())
}

fn load_existing_mr(issue_iid: u64) -> Option<u64> {
    let filename = format!("issue_{}.md", issue_iid);
    if let Ok(content) = fs::read_to_string(&filename) {
        // Parse MR IID from the file
        let re = regex::Regex::new(r"MR: !(\d+)").unwrap();
        if let Some(caps) = re.captures(&content)
            && let Ok(mr_iid) = caps[1].parse::<u64>()
        {
            return Some(mr_iid);
        }
    }
    None
}

fn load_existing_mrs(active_mrs: &mut HashSet<u64>, processed_issues: &mut HashSet<u64>) {
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string()
                && file_name.starts_with("issue_")
                && file_name.ends_with(".md")
                && let Some(issue_str) = file_name
                    .strip_prefix("issue_")
                    .and_then(|s| s.strip_suffix(".md"))
                && let Ok(issue_iid) = issue_str.parse::<u64>()
                && let Some(mr_iid) = load_existing_mr(issue_iid)
            {
                active_mrs.insert(mr_iid);
                processed_issues.insert(issue_iid);
                debug!(
                    "Loaded MR !{} for issue #{} from session file",
                    mr_iid, issue_iid
                );
            }
        }
    }
}

fn check_mr_exists_for_issue(gitlab: &GitLabClient, issue_iid: u64) -> Option<u64> {
    // First check session file
    if let Some(mr_iid) = load_existing_mr(issue_iid) {
        // Verify the MR still exists and is open
        if let Ok(mr) = gitlab.get_merge_request(mr_iid)
            && mr.state == "opened"
        {
            info!(
                "Found existing open MR !{} for issue #{}",
                mr_iid, issue_iid
            );
            return Some(mr_iid);
        }
    }

    // Also check if there's an open MR for the branch
    let branch_name = format!("issue-{}", issue_iid);
    if let Ok(mrs) = gitlab.list_merge_requests() {
        for mr in mrs {
            if mr.source_branch == branch_name && mr.state == "opened" {
                info!(
                    "Found existing open MR !{} for branch {}",
                    mr.iid, branch_name
                );
                // Save it for future reference
                let _ = fs::write(
                    format!("issue_{}.md", issue_iid),
                    format!(
                        "# Issue #{}\n\nMR: !{}\nStatus: MR Found\nTimestamp: {}\n",
                        issue_iid,
                        mr.iid,
                        chrono::Utc::now().to_rfc3339()
                    ),
                );
                return Some(mr.iid);
            }
        }
    }

    None
}

fn extract_project_name(repo_url: &str) -> Result<String> {
    let parts: Vec<&str> = repo_url.trim_end_matches('/').split('/').collect();
    let name = parts
        .last()
        .context("Invalid repository URL")?
        .trim_end_matches(".git");
    Ok(name.to_string())
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

        issue.labels = vec![];
        assert!(!should_skip_issue(&issue));
    }
}
