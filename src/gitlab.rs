use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, info};

pub const PRIORITY_LABEL_PREFIX: &str = "priority::";
pub const DEFAULT_PRIORITY: u8 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub iid: u64,
    pub title: String,
    pub description: String,
    pub labels: Vec<String>,
    pub state: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

impl Issue {
    pub fn priority(&self) -> u8 {
        priority_from_labels(&self.labels)
    }
}

pub fn priority_from_labels(labels: &[String]) -> u8 {
    for label in labels {
        if let Some(num_str) = label.strip_prefix(PRIORITY_LABEL_PREFIX)
            && let Ok(p) = num_str.parse::<u8>()
            && (1..=3).contains(&p)
        {
            return p;
        }
    }
    DEFAULT_PRIORITY
}

pub fn priority_label(priority: u8) -> String {
    format!("{}{}", PRIORITY_LABEL_PREFIX, priority)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRequest {
    pub iid: u64,
    pub title: String,
    pub description: String,
    pub source_branch: String,
    pub target_branch: String,
    pub state: String,
    pub sha: Option<String>,
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub has_conflicts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub id: u64,
    pub body: String,
    pub author: String,
    pub discussion_id: String,
}

pub struct GitLabClient {
    repo_path: String,
}

impl GitLabClient {
    pub fn new(repo_path: String) -> Self {
        Self { repo_path }
    }

    pub fn list_issues(&self) -> Result<Vec<Issue>> {
        debug!("Fetching issues from GitLab");

        let output = Command::new("glab")
            .args(["issue", "list", "--output", "json"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to execute glab issue list")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue list failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let mut issues: Vec<Issue> =
            serde_json::from_slice(&output.stdout).context("Failed to parse issues JSON")?;

        sort_issues_by_priority(&mut issues);

        Ok(issues)
    }

    pub fn get_issue(&self, iid: u64) -> Result<Issue> {
        debug!("Fetching issue #{}", iid);

        let output = Command::new("glab")
            .args(["issue", "view", &iid.to_string(), "--output", "json"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to execute glab issue view")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue view failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let issue: Issue =
            serde_json::from_slice(&output.stdout).context("Failed to parse issue JSON")?;

        Ok(issue)
    }

    pub fn add_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Adding label '{}' to issue #{}", label, iid);

        let endpoint = format!("projects/:id/issues/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                &format!("add_labels={}", label),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to add label to issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to add label to issue: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn remove_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Removing label '{}' from issue #{}", label, iid);

        let endpoint = format!("projects/:id/issues/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                &format!("remove_labels={}", label),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to remove label from issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to remove label from issue: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn add_issue_comment(&self, iid: u64, comment: &str) -> Result<()> {
        debug!("Adding comment to issue #{}", iid);

        let output = Command::new("glab")
            .args(["issue", "note", &iid.to_string(), "--message", comment])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to add comment to issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue note failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn create_merge_request(
        &self,
        source_branch: &str,
        title: &str,
        description: &str,
    ) -> Result<u64> {
        debug!("Creating merge request from branch {}", source_branch);

        let output = Command::new("glab")
            .args([
                "mr",
                "create",
                "--source-branch",
                source_branch,
                "--title",
                title,
                "--description",
                description,
                "--yes",
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to create merge request")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr create failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);

        debug!("MR create stdout: {}", output_str);
        debug!("MR create stderr: {}", stderr_str);

        let mr_iid = self.extract_mr_iid(&output_str, &stderr_str)?;

        Ok(mr_iid)
    }

    pub fn list_merge_requests(&self) -> Result<Vec<MergeRequest>> {
        debug!("Fetching merge requests from GitLab");

        let output = Command::new("glab")
            .args([
                "api",
                "projects/:id/merge_requests?state=opened&per_page=100",
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to fetch merge requests via API")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to list merge requests: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let mrs: Vec<MergeRequest> = serde_json::from_slice(&output.stdout)
            .context("Failed to parse merge requests JSON")?;

        Ok(mrs)
    }

    pub fn get_merge_request(&self, iid: u64) -> Result<MergeRequest> {
        debug!("Fetching merge request !{}", iid);

        let endpoint = format!("projects/:id/merge_requests/{}", iid);
        let output = Command::new("glab")
            .args(["api", &endpoint])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to execute glab api for MR")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api MR failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let mr: MergeRequest =
            serde_json::from_slice(&output.stdout).context("Failed to parse merge request JSON")?;

        Ok(mr)
    }

    pub fn get_mr_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        debug!("Fetching discussions for merge request !{}", iid);

        let endpoint = format!(
            "projects/:id/merge_requests/{}/discussions?per_page=100",
            iid
        );
        let output = Command::new("glab")
            .args(["api", &endpoint])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to fetch MR discussions via API")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            debug!("Failed to get MR discussions: {}", stderr);
            return Ok(Vec::new());
        }

        let discussions: Vec<serde_json::Value> =
            serde_json::from_slice(&output.stdout).unwrap_or_default();

        let mut comments = Vec::new();
        for discussion in &discussions {
            let discussion_id = discussion["id"].as_str().unwrap_or("").to_string();
            if let Some(notes) = discussion["notes"].as_array() {
                for note in notes {
                    if note["system"].as_bool().unwrap_or(true) {
                        continue;
                    }
                    comments.push(Comment {
                        id: note["id"].as_u64().unwrap_or(0),
                        body: note["body"].as_str().unwrap_or("").to_string(),
                        author: note["author"]["username"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string(),
                        discussion_id: discussion_id.clone(),
                    });
                }
            }
        }

        Ok(comments)
    }

    fn fetch_discussions(&self, iid: u64) -> Result<Vec<serde_json::Value>> {
        let endpoint = format!(
            "projects/:id/merge_requests/{}/discussions?per_page=100",
            iid
        );
        let output = Command::new("glab")
            .args(["api", &endpoint])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to fetch MR discussions via API")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to fetch MR discussions: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(serde_json::from_slice(&output.stdout).unwrap_or_default())
    }

    /// Check whether a discussion has resolvable notes and whether any are unresolved.
    /// Falls back to note-level fields when discussion-level fields are absent,
    /// which is the case on older GitLab instances.
    fn discussion_resolution(discussion: &serde_json::Value) -> Option<(bool, bool)> {
        // Try discussion-level fields first
        if let Some(resolvable) = discussion["resolvable"].as_bool() {
            if resolvable {
                let resolved = discussion["resolved"].as_bool() == Some(true);
                return Some((true, resolved));
            }
            return None;
        }

        // Fall back to note-level fields
        let notes = discussion["notes"].as_array()?;
        let mut has_resolvable = false;
        let mut all_resolved = true;
        for note in notes {
            if note["system"].as_bool() == Some(true) {
                continue;
            }
            if note["resolvable"].as_bool() == Some(true) {
                has_resolvable = true;
                if note["resolved"].as_bool() != Some(true) {
                    all_resolved = false;
                }
            }
        }
        if has_resolvable {
            Some((true, all_resolved))
        } else {
            None
        }
    }

    /// Returns the discussion IDs of unresolved resolvable discussions on an MR.
    pub fn get_unresolved_discussion_ids(&self, iid: u64) -> Result<Vec<String>> {
        debug!(
            "Fetching unresolved discussion IDs for merge request !{}",
            iid
        );

        let discussions = self.fetch_discussions(iid)?;

        let mut ids = Vec::new();
        for discussion in &discussions {
            if let Some((true, false)) = Self::discussion_resolution(discussion)
                && let Some(id) = discussion["id"].as_str()
            {
                ids.push(id.to_string());
            }
        }

        Ok(ids)
    }

    /// Returns (unresolved_count, total_resolvable_count) for an MR's discussions.
    pub fn get_unresolved_discussion_count(&self, iid: u64) -> Result<(usize, usize)> {
        debug!("Checking discussion resolution state for MR !{}", iid);

        let discussions = self.fetch_discussions(iid)?;

        let mut total = 0;
        let mut unresolved = 0;
        for discussion in &discussions {
            if let Some((true, resolved)) = Self::discussion_resolution(discussion) {
                total += 1;
                if !resolved {
                    unresolved += 1;
                }
            }
        }

        Ok((unresolved, total))
    }

    pub fn resolve_discussion(&self, mr_iid: u64, discussion_id: &str) -> Result<()> {
        debug!("Resolving discussion {} on MR !{}", discussion_id, mr_iid);

        let endpoint = format!(
            "projects/:id/merge_requests/{}/discussions/{}",
            mr_iid, discussion_id
        );
        let output = Command::new("glab")
            .args(["api", &endpoint, "--method", "PUT", "-f", "resolved=true"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to resolve discussion")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to resolve discussion: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn reply_to_discussion(&self, mr_iid: u64, discussion_id: &str, body: &str) -> Result<()> {
        debug!("Replying to discussion {} on MR !{}", discussion_id, mr_iid);

        let endpoint = format!(
            "projects/:id/merge_requests/{}/discussions/{}/notes",
            mr_iid, discussion_id
        );
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "POST",
                "-f",
                &format!("body={}", body),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to reply to discussion")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to reply to discussion: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Creates a plain (non-resolvable) note on an MR.
    pub fn add_mr_comment(&self, iid: u64, comment: &str) -> Result<()> {
        debug!("Adding comment to merge request !{}", iid);

        let output = Command::new("glab")
            .args(["mr", "note", &iid.to_string(), "--message", comment])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to add comment to merge request")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr note failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Creates a resolvable discussion thread on an MR via the discussions API.
    pub fn add_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        debug!("Creating discussion thread on MR !{}", iid);

        let endpoint = format!("projects/:id/merge_requests/{}/discussions", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "POST",
                "-f",
                &format!("body={}", body),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to create MR discussion")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to create MR discussion: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn add_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Adding label '{}' to MR !{}", label, iid);

        let endpoint = format!("projects/:id/merge_requests/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                &format!("add_labels={}", label),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to add label to merge request")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to add label to MR: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn remove_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Removing label '{}' from MR !{}", label, iid);

        let endpoint = format!("projects/:id/merge_requests/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                &format!("remove_labels={}", label),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to remove label from MR")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to remove label from MR: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn merge_mr(&self, iid: u64) -> Result<()> {
        debug!("Merging merge request !{}", iid);

        let output = Command::new("glab")
            .args(["mr", "merge", &iid.to_string(), "--yes"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to merge merge request")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr merge failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn close_mr(&self, iid: u64) -> Result<()> {
        debug!("Closing merge request !{}", iid);

        let output = Command::new("glab")
            .args(["mr", "close", &iid.to_string()])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to close merge request")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr close failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn close_issue(&self, iid: u64) -> Result<()> {
        debug!("Closing issue #{}", iid);

        let endpoint = format!("projects/:id/issues/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                "state_event=close",
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to close issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to close issue #{}: {}",
                iid,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        info!("Closed issue #{}", iid);
        Ok(())
    }

    pub fn update_mr_title_description(
        &self,
        iid: u64,
        title: &str,
        description: &str,
    ) -> Result<()> {
        debug!("Updating MR !{} title and description", iid);

        let endpoint = format!("projects/:id/merge_requests/{}", iid);
        let output = Command::new("glab")
            .args([
                "api",
                &endpoint,
                "--method",
                "PUT",
                "-f",
                &format!("title={}", title),
                "-f",
                &format!("description={}", description),
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to update MR")?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to update MR !{}: {}",
                iid,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        info!("Updated MR !{} title and description", iid);
        Ok(())
    }

    fn extract_mr_iid(&self, stdout: &str, stderr: &str) -> Result<u64> {
        // Try multiple patterns to extract MR IID
        let patterns = vec![
            r"!(\d+)",                  // !123
            r"#(\d+)",                  // #123
            r"/merge_requests/(\d+)",   // URL format
            r"merge_requests/(\d+)",    // URL without leading slash
            r"MR\s+!?(\d+)",            // MR !123 or MR 123
            r"merge request\s+!?(\d+)", // merge request !123
            r"created.*?!(\d+)",        // created !123
        ];

        // Check both stdout and stderr
        let combined = format!("{}\n{}", stdout, stderr);

        for pattern in patterns {
            let re = regex::Regex::new(pattern).unwrap();
            if let Some(caps) = re.captures(&combined)
                && let Ok(iid) = caps[1].parse::<u64>()
            {
                debug!("Extracted MR IID {} using pattern: {}", iid, pattern);
                return Ok(iid);
            }
        }

        anyhow::bail!(
            "Could not extract MR IID from output.\nStdout: {}\nStderr: {}",
            stdout,
            stderr
        )
    }

    pub fn create_issue(&self, title: &str, description: &str) -> Result<u64> {
        debug!("Creating issue: {}", title);

        let output = Command::new("glab")
            .args([
                "issue",
                "create",
                "--title",
                title,
                "--description",
                description,
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to create issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue create failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);

        debug!("Issue create stdout: {}", output_str);
        debug!("Issue create stderr: {}", stderr_str);

        // Extract issue IID from output (similar to MR extraction)
        let combined = format!("{}\n{}", output_str, stderr_str);
        let patterns = vec![r"#(\d+)", r"issue/(\d+)", r"issues/(\d+)"];

        for pattern in patterns {
            let re = regex::Regex::new(pattern).unwrap();
            if let Some(caps) = re.captures(&combined)
                && let Ok(iid) = caps[1].parse::<u64>()
            {
                debug!("Extracted issue IID {}", iid);
                return Ok(iid);
            }
        }

        anyhow::bail!(
            "Could not extract issue IID from output.\nStdout: {}\nStderr: {}",
            output_str,
            stderr_str
        )
    }

    pub fn get_issue_comments(&self, issue_iid: u64) -> Result<Vec<Comment>> {
        debug!("Fetching comments for issue #{}", issue_iid);

        let output = Command::new("glab")
            .args([
                "issue",
                "note",
                "list",
                &issue_iid.to_string(),
                "--output",
                "json",
            ])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to fetch issue comments")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue note list failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[derive(Deserialize)]
        struct NoteResponse {
            id: u64,
            body: String,
            author: AuthorInfo,
        }

        #[derive(Deserialize)]
        struct AuthorInfo {
            username: String,
        }

        let notes: Vec<NoteResponse> = serde_json::from_slice(&output.stdout)
            .context("Failed to parse issue comments JSON")?;

        let comments = notes
            .into_iter()
            .map(|note| Comment {
                id: note.id,
                body: note.body,
                author: note.author.username,
                discussion_id: format!("issue_{}", note.id),
            })
            .collect();

        Ok(comments)
    }
}

/// Sort issues by priority (lowest number = highest priority) then by
/// `created_at` ascending (oldest first within the same priority level)
/// to prevent starvation.
pub fn sort_issues_by_priority(issues: &mut [Issue]) {
    issues.sort_by(|a, b| {
        let pa = a.priority();
        let pb = b.priority();
        pa.cmp(&pb).then_with(|| {
            let ca = a.created_at.as_deref().unwrap_or("");
            let cb = b.created_at.as_deref().unwrap_or("");
            ca.cmp(cb)
        })
    });
}

/// Extract an issue IID from a branch name following the `issue-N` convention.
pub fn issue_iid_from_branch(branch: &str) -> Option<u64> {
    branch.strip_prefix("issue-")?.parse().ok()
}
