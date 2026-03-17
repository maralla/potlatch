use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub iid: u64,
    pub title: String,
    pub description: String,
    pub labels: Vec<String>,
    pub state: String,
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

        let issues: Vec<Issue> =
            serde_json::from_slice(&output.stdout).context("Failed to parse issues JSON")?;

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

        let output = Command::new("glab")
            .args(["issue", "update", &iid.to_string(), "--label", label])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to add label to issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue update failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn remove_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Removing label '{}' from issue #{}", label, iid);

        let output = Command::new("glab")
            .args(["issue", "update", &iid.to_string(), "--unlabel", label])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to remove label from issue")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab issue update failed: {}",
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
            .args(["mr", "list", "--output", "json"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to execute glab mr list")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr list failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let mrs: Vec<MergeRequest> = serde_json::from_slice(&output.stdout)
            .context("Failed to parse merge requests JSON")?;

        Ok(mrs)
    }

    pub fn get_merge_request(&self, iid: u64) -> Result<MergeRequest> {
        debug!("Fetching merge request !{}", iid);

        let output = Command::new("glab")
            .args(["mr", "view", &iid.to_string(), "--output", "json"])
            .current_dir(&self.repo_path)
            .output()
            .context("Failed to execute glab mr view")?;

        if !output.status.success() {
            anyhow::bail!(
                "glab mr view failed: {}",
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
}
