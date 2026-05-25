use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::thread;
use std::time::Duration;
use tracing::{debug, info, warn};

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

#[derive(Debug, Clone)]
pub struct MergeRequestChangesSnapshot {
    pub files: Vec<String>,
    pub patch: String,
    pub overflow: bool,
    pub base_sha: Option<String>,
    pub start_sha: Option<String>,
    pub head_sha: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub id: u64,
    pub body: String,
    pub author: String,
    pub discussion_id: String,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub location_details: Option<String>,
}

impl Comment {
    pub fn format_for_prompt(&self) -> String {
        if let Some(location) = &self.location {
            if let Some(details) = &self.location_details {
                format!(
                    "- {} [{} | {}] (discussion {}): {}",
                    self.author, location, details, self.discussion_id, self.body
                )
            } else {
                format!(
                    "- {} [{}] (discussion {}): {}",
                    self.author, location, self.discussion_id, self.body
                )
            }
        } else {
            format!(
                "- {} (discussion {}): {}",
                self.author, self.discussion_id, self.body
            )
        }
    }
}

/// Issue note with fields needed to resolve **thread replies** (GitLab discussion).
#[derive(Debug, Clone, Deserialize)]
pub struct IssueThreadNote {
    pub id: u64,
    pub body: String,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub discussion_id: Option<String>,
    author: IssueThreadNoteAuthor,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct IssueThreadNoteAuthor {
    #[serde(default)]
    username: String,
}

impl IssueThreadNote {
    pub fn author_username(&self) -> &str {
        self.author.username.as_str()
    }
}

fn issue_thread_notes_as_comments(notes: Vec<IssueThreadNote>) -> Vec<Comment> {
    notes
        .into_iter()
        .map(|n| {
            let author = n.author_username().to_string();
            Comment {
                id: n.id,
                body: n.body,
                author,
                discussion_id: format!("issue_{}", n.id),
                location: None,
                location_details: None,
            }
        })
        .collect()
}

#[derive(Clone)]
pub struct GitLabClient {
    repo_path: String,
    host: String,
    project_id: u64,
}

fn parse_gitlab_repo(repo_url: &str) -> Result<(String, String)> {
    let repo_url = repo_url.trim();
    if let Some(rest) = repo_url.strip_prefix("git@") {
        let (host, path) = rest
            .split_once(':')
            .with_context(|| format!("invalid SSH gitlab_repo URL: {repo_url}"))?;
        let path = path.trim_end_matches('/').trim_end_matches(".git");
        anyhow::ensure!(!path.is_empty(), "missing project path in gitlab_repo");
        return Ok((host.to_string(), path.to_string()));
    }
    if repo_url.starts_with("http://") || repo_url.starts_with("https://") {
        let without_scheme = repo_url
            .split("//")
            .nth(1)
            .with_context(|| format!("invalid HTTPS gitlab_repo URL: {repo_url}"))?;
        let (host, path) = without_scheme
            .split_once('/')
            .with_context(|| format!("missing project path in gitlab_repo URL: {repo_url}"))?;
        let path = path.trim_end_matches('/').trim_end_matches(".git");
        anyhow::ensure!(!path.is_empty(), "missing project path in gitlab_repo");
        return Ok((host.to_string(), path.to_string()));
    }
    anyhow::bail!("unsupported gitlab_repo URL scheme: {repo_url}");
}

fn resolve_project_id(host: &str, project_path: &str) -> Result<u64> {
    let search_term = project_path.rsplit('/').next().unwrap_or(project_path);
    let endpoint = format!(
        "projects?search={}&membership=true&simple=true&per_page=50",
        search_term
    );
    let output = Command::new("glab")
        .args(["api", "--hostname", host, &endpoint])
        .output()
        .with_context(|| format!("Failed to resolve GitLab project id for {project_path}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "Failed to resolve GitLab project id for {}: {}",
            project_path,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    #[derive(Deserialize)]
    struct ProjectHit {
        id: u64,
        path_with_namespace: String,
    }
    let hits: Vec<ProjectHit> =
        serde_json::from_slice(&output.stdout).context("Failed to parse project search JSON")?;
    let want = project_path.to_ascii_lowercase();
    hits.into_iter()
        .find(|p| p.path_with_namespace.to_ascii_lowercase() == want)
        .map(|p| p.id)
        .with_context(|| format!("GitLab project not found for path {project_path} on {host}"))
}

fn configure_repo_glab(repo_path: &str, host: &str) -> Result<()> {
    use std::fs;
    use std::path::Path;
    let config_dir = Path::new(repo_path).join(".git/glab-cli");
    fs::create_dir_all(&config_dir).with_context(|| {
        format!(
            "Failed to create glab config directory {}",
            config_dir.display()
        )
    })?;
    fs::write(config_dir.join("config.yml"), format!("host: {host}\n"))
        .with_context(|| format!("Failed to write glab config under {}", config_dir.display()))?;
    Ok(())
}

fn mr_create_error_is_duplicate(err_msg: &str) -> bool {
    err_msg.to_ascii_lowercase().contains("already exists")
}

/// Returns `true` when the glab/API error string suggests a transient problem worth retrying.
fn mr_label_api_error_should_retry(err_msg: &str) -> bool {
    let m = err_msg.to_lowercase();
    // Permanent client / validation errors — repeating the request is unlikely to help.
    if m.contains("http 401")
        || m.contains("http 403")
        || m.contains("http 404")
        || m.contains("http 400")
        || m.contains("http 405")
        || m.contains("http 422")
        || m.contains("unauthorized")
        || m.contains("not found")
    {
        return false;
    }
    true
}

impl GitLabClient {
    pub fn new(repo_path: String, gitlab_repo: &str) -> Result<Self> {
        let (host, project_path) = parse_gitlab_repo(gitlab_repo)?;
        let project_id = resolve_project_id(&host, &project_path)?;
        configure_repo_glab(&repo_path, &host)?;
        info!(
            "GitLab client for {} on {} (project id {})",
            project_path, host, project_id
        );
        Ok(Self {
            repo_path,
            host,
            project_id,
        })
    }

    fn api_path(&self, tail: &str) -> String {
        format!(
            "projects/{}/{}",
            self.project_id,
            tail.trim_start_matches('/')
        )
    }

    fn run_api(&self, endpoint: &str, extra_args: &[&str]) -> Result<std::process::Output> {
        Command::new("glab")
            .arg("api")
            .arg("--hostname")
            .arg(&self.host)
            .arg(endpoint)
            .args(extra_args)
            .current_dir(&self.repo_path)
            .output()
            .with_context(|| format!("Failed to execute glab api {endpoint}"))
    }

    fn run_api_method(
        &self,
        tail: &str,
        method: &str,
        fields: &[(&str, &str)],
    ) -> Result<std::process::Output> {
        let endpoint = self.api_path(tail);
        let field_strings: Vec<String> = fields.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let mut cmd = Command::new("glab");
        cmd.arg("api")
            .arg("--hostname")
            .arg(&self.host)
            .arg(&endpoint)
            .args(["--method", method]);
        for field in &field_strings {
            cmd.args(["-f", field]);
        }
        cmd.current_dir(&self.repo_path)
            .output()
            .with_context(|| format!("Failed to execute glab api {method} {endpoint}"))
    }

    fn run_api_json(
        &self,
        tail: &str,
        method: &str,
        body: &impl Serialize,
    ) -> Result<std::process::Output> {
        let endpoint = self.api_path(tail);
        let json = serde_json::to_vec(body).context("Failed to serialize GitLab API JSON body")?;
        let mut child = Command::new("glab")
            .arg("api")
            .arg("--hostname")
            .arg(&self.host)
            .arg(&endpoint)
            .args([
                "--method",
                method,
                "-H",
                "Content-Type: application/json",
                "--input",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&self.repo_path)
            .spawn()
            .with_context(|| format!("Failed to spawn glab api {method} {endpoint}"))?;
        child
            .stdin
            .take()
            .context("Failed to open glab api stdin")?
            .write_all(&json)
            .context("Failed to write glab api JSON body")?;
        child
            .wait_with_output()
            .with_context(|| format!("Failed to execute glab api {method} {endpoint}"))
    }

    fn glab_api_error_message(output: &std::process::Output) -> String {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            stderr.into_owned()
        } else if stderr.trim().is_empty() {
            stdout.into_owned()
        } else {
            format!("{stderr}\n{stdout}")
        }
    }

    /// Returns the IID of an open MR for `source_branch`, if one exists.
    pub fn find_open_mr_by_source_branch(&self, source_branch: &str) -> Result<Option<u64>> {
        let endpoint = self.api_path(&format!(
            "merge_requests?source_branch={source_branch}&state=opened&per_page=1"
        ));
        let output = self.run_api(&endpoint, &[])?;
        if !output.status.success() {
            anyhow::bail!(
                "Failed to find merge request for branch {}: {}",
                source_branch,
                Self::glab_api_error_message(&output)
            );
        }
        #[derive(Deserialize)]
        struct MrHit {
            iid: u64,
        }
        let hits: Vec<MrHit> =
            serde_json::from_slice(&output.stdout).context("Failed to parse MR search JSON")?;
        Ok(hits.into_iter().next().map(|mr| mr.iid))
    }

    pub fn list_issues(&self) -> Result<Vec<Issue>> {
        debug!("Fetching issues from GitLab");
        const PER_PAGE: usize = 100;
        let mut page = 1usize;
        let mut issues = Vec::new();

        loop {
            let endpoint = self.api_path(&format!(
                "issues?state=opened&per_page={PER_PAGE}&page={page}"
            ));
            let output = self.run_api(&endpoint, &[])?;

            if !output.status.success() {
                anyhow::bail!(
                    "glab api issue list failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            let batch: Vec<Issue> =
                serde_json::from_slice(&output.stdout).context("Failed to parse issues JSON")?;

            let batch_len = batch.len();
            issues.extend(batch);

            if batch_len < PER_PAGE {
                break;
            }
            page += 1;
        }

        sort_issues_by_priority(&mut issues);

        Ok(issues)
    }

    pub fn get_issue(&self, iid: u64) -> Result<Issue> {
        debug!("Fetching issue #{}", iid);

        let endpoint = self.api_path(&format!("issues/{iid}"));
        let output = self.run_api(&endpoint, &[])?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api issue view failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let issue: Issue =
            serde_json::from_slice(&output.stdout).context("Failed to parse issue JSON")?;

        Ok(issue)
    }

    pub fn add_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Adding label '{}' to issue #{}", label, iid);

        let output =
            self.run_api_method(&format!("issues/{iid}"), "PUT", &[("add_labels", label)])?;

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

        let output =
            self.run_api_method(&format!("issues/{iid}"), "PUT", &[("remove_labels", label)])?;

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

        let output =
            self.run_api_method(&format!("issues/{iid}/notes"), "POST", &[("body", comment)])?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api issue note failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn create_merge_request(
        &self,
        source_branch: &str,
        target_branch: &str,
        title: &str,
        description: &str,
    ) -> Result<u64> {
        debug!(
            "Creating merge request from {} into {}",
            source_branch, target_branch
        );

        if let Some(existing) = self.find_open_mr_by_source_branch(source_branch)? {
            info!(
                "Open MR !{} already exists for branch {}, reusing it",
                existing, source_branch
            );
            return Ok(existing);
        }

        #[derive(Serialize)]
        struct CreateMrBody<'a> {
            source_branch: &'a str,
            target_branch: &'a str,
            title: &'a str,
            description: &'a str,
        }

        let body = CreateMrBody {
            source_branch,
            target_branch,
            title,
            description,
        };
        let output = self.run_api_json("merge_requests", "POST", &body)?;

        if !output.status.success() {
            let err = Self::glab_api_error_message(&output);
            if mr_create_error_is_duplicate(&err)
                && let Some(existing) = self.find_open_mr_by_source_branch(source_branch)?
            {
                info!(
                    "Open MR !{} already exists for branch {}, reusing it after create conflict",
                    existing, source_branch
                );
                return Ok(existing);
            }
            anyhow::bail!("glab api mr create failed: {err}");
        }

        #[derive(Deserialize)]
        struct CreatedMr {
            iid: u64,
        }
        let mr: CreatedMr =
            serde_json::from_slice(&output.stdout).context("Failed to parse created MR JSON")?;

        Ok(mr.iid)
    }

    pub fn list_merge_requests(&self) -> Result<Vec<MergeRequest>> {
        debug!("Fetching merge requests from GitLab");

        let endpoint = self.api_path("merge_requests?state=opened&per_page=100");
        let output = self.run_api(&endpoint, &[])?;

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

        let endpoint = self.api_path(&format!("merge_requests/{iid}"));
        let output = self.run_api(&endpoint, &[])?;

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

    /// Fetches the exact MR diff payload as produced by GitLab for this MR.
    /// This is preferred for agent context because it matches the MR view.
    pub fn get_merge_request_changes(&self, iid: u64) -> Result<MergeRequestChangesSnapshot> {
        debug!("Fetching merge request !{} changes", iid);

        let endpoint = self.api_path(&format!("merge_requests/{iid}/changes"));
        let output = self.run_api(&endpoint, &[])?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api MR changes failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let payload: serde_json::Value =
            serde_json::from_slice(&output.stdout).context("Failed to parse MR changes JSON")?;
        Ok(Self::parse_mr_changes_payload(&payload))
    }

    fn parse_mr_changes_payload(payload: &serde_json::Value) -> MergeRequestChangesSnapshot {
        let overflow = payload["overflow"].as_bool().unwrap_or(false);
        let changes = payload["changes"].as_array().cloned().unwrap_or_default();

        let mut files = Vec::new();
        let mut patch_parts = Vec::new();
        for change in changes {
            let new_path = change["new_path"].as_str().unwrap_or("").trim();
            let old_path = change["old_path"].as_str().unwrap_or("").trim();
            let file_label = match (old_path.is_empty(), new_path.is_empty()) {
                (false, false) if old_path != new_path => format!("{} -> {}", old_path, new_path),
                (_, false) => new_path.to_string(),
                (false, _) => old_path.to_string(),
                _ => "(unknown path)".to_string(),
            };
            files.push(file_label);

            let diff = change["diff"].as_str().unwrap_or("").trim_end();
            let old_header = if old_path.is_empty() {
                "dev/null"
            } else {
                old_path
            };
            let new_header = if new_path.is_empty() {
                "dev/null"
            } else {
                new_path
            };
            let section = if diff.is_empty() {
                format!("diff --git a/{old_header} b/{new_header}\n")
            } else {
                format!("diff --git a/{old_header} b/{new_header}\n{diff}\n")
            };
            patch_parts.push(section);
        }

        MergeRequestChangesSnapshot {
            files,
            patch: patch_parts.join("\n"),
            overflow,
            base_sha: payload
                .pointer("/diff_refs/base_sha")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            start_sha: payload
                .pointer("/diff_refs/start_sha")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            head_sha: payload
                .pointer("/diff_refs/head_sha")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }
    }

    pub fn get_mr_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        debug!("Fetching discussions for merge request !{}", iid);

        let endpoint = self.api_path(&format!("merge_requests/{iid}/discussions?per_page=100"));
        let output = self.run_api(&endpoint, &[])?;

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
            let discussion_location = Self::discussion_location(discussion);
            if let Some(notes) = discussion["notes"].as_array() {
                for note in notes {
                    if note["system"].as_bool().unwrap_or(true) {
                        continue;
                    }
                    let note_location =
                        Self::position_location(&note["position"]).or(discussion_location.clone());
                    let note_location_details = Self::position_details(&note["position"]);
                    comments.push(Comment {
                        id: note["id"].as_u64().unwrap_or(0),
                        body: note["body"].as_str().unwrap_or("").to_string(),
                        author: note["author"]["username"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string(),
                        discussion_id: discussion_id.clone(),
                        location: note_location,
                        location_details: note_location_details,
                    });
                }
            }
        }

        Ok(comments)
    }

    fn fetch_discussions(&self, iid: u64) -> Result<Vec<serde_json::Value>> {
        let endpoint = self.api_path(&format!("merge_requests/{iid}/discussions?per_page=100"));
        let output = self.run_api(&endpoint, &[])?;

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

    fn discussion_location(discussion: &serde_json::Value) -> Option<String> {
        Self::position_location(&discussion["position"]).or_else(|| {
            discussion["notes"].as_array().and_then(|notes| {
                notes
                    .iter()
                    .find_map(|note| Self::position_location(&note["position"]))
            })
        })
    }

    fn position_location(position: &serde_json::Value) -> Option<String> {
        if !position.is_object() {
            return None;
        }

        let new_path = position["new_path"].as_str().filter(|s| !s.is_empty());
        let old_path = position["old_path"].as_str().filter(|s| !s.is_empty());
        let path = new_path.or(old_path)?;

        if let Some(range) = position["line_range"].as_object() {
            let start_line = range
                .get("start")
                .and_then(|v| v["new_line"].as_u64().or_else(|| v["old_line"].as_u64()));
            let end_line = range
                .get("end")
                .and_then(|v| v["new_line"].as_u64().or_else(|| v["old_line"].as_u64()));
            match (start_line, end_line) {
                (Some(start), Some(end)) if start != end => {
                    return Some(format!("{}:{}-{}", path, start, end));
                }
                (Some(line), _) | (_, Some(line)) => {
                    return Some(format!("{}:{}", path, line));
                }
                _ => {}
            }
        }

        if let Some(line) = position["new_line"]
            .as_u64()
            .or_else(|| position["old_line"].as_u64())
        {
            return Some(format!("{}:{}", path, line));
        }

        Some(path.to_string())
    }

    fn position_details(position: &serde_json::Value) -> Option<String> {
        if !position.is_object() {
            return None;
        }
        let mut parts = Vec::new();
        if let Some(v) = position["position_type"].as_str()
            && !v.is_empty()
        {
            parts.push(format!("type={}", v));
        }
        if let Some(v) = position["new_line"].as_u64() {
            parts.push(format!("new_line={}", v));
        }
        if let Some(v) = position["old_line"].as_u64() {
            parts.push(format!("old_line={}", v));
        }
        if let Some(v) = position["line_code"].as_str()
            && !v.is_empty()
        {
            parts.push(format!("line_code={}", v));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(", "))
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

        let output = self.run_api_method(
            &format!("merge_requests/{mr_iid}/discussions/{discussion_id}"),
            "PUT",
            &[("resolved", "true")],
        )?;

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

        let output = self.run_api_method(
            &format!("merge_requests/{mr_iid}/discussions/{discussion_id}/notes"),
            "POST",
            &[("body", body)],
        )?;

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

        let output = self.run_api_method(
            &format!("merge_requests/{iid}/notes"),
            "POST",
            &[("body", comment)],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api mr note failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    fn create_mr_discussion(&self, iid: u64, body: &str) -> Result<String> {
        debug!("Creating discussion thread on MR !{}", iid);

        let output = self.run_api_method(
            &format!("merge_requests/{iid}/discussions"),
            "POST",
            &[("body", body)],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to create MR discussion: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let discussion: serde_json::Value =
            serde_json::from_slice(&output.stdout).context("Failed to parse MR discussion JSON")?;
        let discussion_id = discussion["id"]
            .as_str()
            .context("MR discussion response missing id")?;

        Ok(discussion_id.to_string())
    }

    /// Creates a resolvable discussion thread on an MR via the discussions API.
    pub fn add_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        let _ = self.create_mr_discussion(iid, body)?;
        Ok(())
    }

    /// Creates a resolvable discussion thread on an MR and immediately resolves it.
    pub fn add_resolved_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        let discussion_id = self.create_mr_discussion(iid, body)?;
        self.resolve_discussion(iid, &discussion_id)?;
        Ok(())
    }

    pub fn add_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Adding label '{}' to MR !{}", label, iid);

        let output = self.run_api_method(
            &format!("merge_requests/{iid}"),
            "PUT",
            &[("add_labels", label)],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to add label to MR: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Like [`Self::add_mr_label`], but on likely-transient failures (GitLab 5xx, rate limits, etc.)
    /// sleeps with exponential backoff (capped) and retries until success. Returns `Err` only when
    /// the error looks permanent (e.g. 401/403/404/400/422) so the caller can log and continue.
    pub fn add_mr_label_with_transient_retries(&self, iid: u64, label: &str) -> Result<()> {
        let mut attempt = 0u32;
        let mut delay = Duration::from_secs(1);
        const MAX_DELAY: Duration = Duration::from_secs(60);
        loop {
            attempt += 1;
            match self.add_mr_label(iid, label) {
                Ok(()) => {
                    if attempt > 1 {
                        info!(
                            "Added label {:?} to MR !{} after {} attempts",
                            label, iid, attempt
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    let msg = e.to_string();
                    if !mr_label_api_error_should_retry(&msg) {
                        return Err(e);
                    }
                    warn!(
                        "Transient failure adding label {:?} to MR !{} (attempt {}), retrying in {:?}: {}",
                        label, iid, attempt, delay, msg
                    );
                    thread::sleep(delay);
                    delay = (delay * 2).min(MAX_DELAY);
                }
            }
        }
    }

    pub fn remove_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        debug!("Removing label '{}' from MR !{}", label, iid);

        let output = self.run_api_method(
            &format!("merge_requests/{iid}"),
            "PUT",
            &[("remove_labels", label)],
        )?;

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

        let output = self.run_api_method(&format!("merge_requests/{iid}/merge"), "PUT", &[])?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api mr merge failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn close_mr(&self, iid: u64) -> Result<()> {
        debug!("Closing merge request !{}", iid);

        let output = self.run_api_method(
            &format!("merge_requests/{iid}"),
            "PUT",
            &[("state_event", "close")],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api mr close failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn close_issue(&self, iid: u64) -> Result<()> {
        debug!("Closing issue #{}", iid);

        let output =
            self.run_api_method(&format!("issues/{iid}"), "PUT", &[("state_event", "close")])?;

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

        let output = self.run_api_method(
            &format!("merge_requests/{iid}"),
            "PUT",
            &[("title", title), ("description", description)],
        )?;

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

    pub fn create_issue(&self, title: &str, description: &str) -> Result<u64> {
        debug!("Creating issue: {}", title);

        let output = self.run_api_method(
            "issues",
            "POST",
            &[("title", title), ("description", description)],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api issue create failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[derive(Deserialize)]
        struct CreatedIssue {
            iid: u64,
        }
        let issue: CreatedIssue =
            serde_json::from_slice(&output.stdout).context("Failed to parse created issue JSON")?;

        debug!("Created issue IID {}", issue.iid);
        Ok(issue.iid)
    }

    /// Issue notes (comments) from GitLab issue discussions API — same payload as
    /// [`Self::get_issue_thread_notes`], mapped to [`Comment`] for prompts.
    pub fn get_issue_comments(&self, issue_iid: u64) -> Result<Vec<Comment>> {
        debug!(
            "Fetching comments for issue #{} (glab api …/discussions)",
            issue_iid
        );
        Ok(issue_thread_notes_as_comments(
            self.fetch_issue_discussions_via_api(issue_iid)?,
        ))
    }

    /// Issue thread notes with GitLab discussion ids from the issue discussions API.
    pub fn get_issue_thread_notes(&self, issue_iid: u64) -> Result<Vec<IssueThreadNote>> {
        debug!(
            "Fetching thread notes for issue #{} (glab api …/discussions)",
            issue_iid
        );
        self.fetch_issue_discussions_via_api(issue_iid)
    }

    fn fetch_issue_discussions_via_api(&self, issue_iid: u64) -> Result<Vec<IssueThreadNote>> {
        let endpoint = self.api_path(&format!("issues/{issue_iid}/discussions"));
        let output = self.run_api(&endpoint, &[])?;

        if !output.status.success() {
            anyhow::bail!(
                "glab api issue discussions failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[derive(Deserialize)]
        struct DiscussionJson {
            id: String,
            notes: Vec<DiscussionNoteJson>,
        }

        #[derive(Deserialize)]
        struct DiscussionNoteJson {
            id: u64,
            #[serde(default)]
            body: String,
            #[serde(default)]
            system: bool,
            /// GitLab may omit or null this for some system/imported notes.
            #[serde(default)]
            author: Option<DiscussionNoteAuthorJson>,
        }

        #[derive(Deserialize)]
        struct DiscussionNoteAuthorJson {
            #[serde(default)]
            username: String,
        }

        let discussions: Vec<DiscussionJson> = serde_json::from_slice(&output.stdout)
            .with_context(|| {
                let preview =
                    String::from_utf8_lossy(&output.stdout[..output.stdout.len().min(400)]);
                format!("Failed to parse issue discussions JSON (stdout preview): {preview}")
            })?;

        let mut out = Vec::new();
        for d in discussions {
            for n in d.notes {
                let username = n
                    .author
                    .as_ref()
                    .map(|a| a.username.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("(unknown)")
                    .to_string();
                out.push(IssueThreadNote {
                    id: n.id,
                    body: n.body,
                    system: n.system,
                    discussion_id: Some(d.id.clone()),
                    author: IssueThreadNoteAuthor { username },
                });
            }
        }

        Ok(out)
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

static CLOSES_ISSUE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)closes?\s+#(\d+)").expect("CLOSES_ISSUE_RE"));

/// True if the MR description contains `Closes #iid` / `Close #iid` for this issue (case-insensitive).
pub fn mr_description_closes_issue(description: &str, issue_iid: u64) -> bool {
    CLOSES_ISSUE_RE.captures_iter(description).any(|cap| {
        cap.get(1)
            .and_then(|m| m.as_str().parse::<u64>().ok())
            .is_some_and(|n| n == issue_iid)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        GitLabClient, IssueThreadNote, MergeRequestChangesSnapshot, mr_create_error_is_duplicate,
        mr_description_closes_issue, mr_label_api_error_should_retry, parse_gitlab_repo,
    };
    use serde_json::json;

    #[test]
    fn parse_gitlab_repo_ssh_url() {
        let (host, path) =
            parse_gitlab_repo("git@git.example.com:platform/projects/data-worker.git")
                .unwrap();
        assert_eq!(host, "git.example.com");
        assert_eq!(path, "platform/projects/data-worker");
    }

    #[test]
    fn parse_gitlab_repo_https_url() {
        let (host, path) = parse_gitlab_repo("https://gitlab.com/group/sub/project.git").unwrap();
        assert_eq!(host, "gitlab.com");
        assert_eq!(path, "group/sub/project");
    }

    #[test]
    fn mr_create_error_is_duplicate_detects_gitlab_conflict_message() {
        assert!(mr_create_error_is_duplicate(
            "glab: map[message:[Another open merge request already exists for this source branch: !21]]"
        ));
        assert!(!mr_create_error_is_duplicate(
            "glab: HTTP 400\n{\"error\":\"target_branch is missing\"}"
        ));
    }

    #[test]
    fn parse_gitlab_repo_rejects_invalid_url() {
        assert!(parse_gitlab_repo("not-a-url").is_err());
    }

    #[test]
    fn discussion_location_prefers_inline_new_line() {
        let discussion = json!({
            "position": {
                "new_path": "src/lib.rs",
                "old_path": "src/lib.rs",
                "new_line": 42
            }
        });

        assert_eq!(
            GitLabClient::discussion_location(&discussion).as_deref(),
            Some("src/lib.rs:42")
        );
    }

    #[test]
    fn discussion_location_supports_line_ranges() {
        let discussion = json!({
            "position": {
                "new_path": "src/lib.rs",
                "line_range": {
                    "start": { "new_line": 10 },
                    "end": { "new_line": 14 }
                }
            }
        });

        assert_eq!(
            GitLabClient::discussion_location(&discussion).as_deref(),
            Some("src/lib.rs:10-14")
        );
    }

    #[test]
    fn parse_mr_changes_payload_handles_paths_and_patch() {
        let payload = json!({
            "overflow": false,
            "changes": [
                {
                    "old_path": "src/old.rs",
                    "new_path": "src/new.rs",
                    "diff": "@@ -1 +1 @@\n-old\n+new"
                },
                {
                    "old_path": "",
                    "new_path": "src/added.rs",
                    "diff": "@@ -0,0 +1 @@\n+line"
                }
            ]
        });

        let parsed: MergeRequestChangesSnapshot = GitLabClient::parse_mr_changes_payload(&payload);
        assert!(!parsed.overflow);
        assert_eq!(
            parsed.files,
            vec![
                "src/old.rs -> src/new.rs".to_string(),
                "src/added.rs".to_string()
            ]
        );
        assert!(
            parsed
                .patch
                .contains("diff --git a/src/old.rs b/src/new.rs")
        );
        assert!(parsed.patch.contains("@@ -1 +1 @@"));
        assert!(
            parsed
                .patch
                .contains("diff --git a/dev/null b/src/added.rs")
        );
        assert_eq!(parsed.base_sha, None);
        assert_eq!(parsed.start_sha, None);
        assert_eq!(parsed.head_sha, None);
    }

    #[test]
    fn parse_mr_changes_payload_preserves_overflow_flag() {
        let payload = json!({
            "overflow": true,
            "changes": []
        });
        let parsed = GitLabClient::parse_mr_changes_payload(&payload);
        assert!(parsed.overflow);
        assert!(parsed.files.is_empty());
        assert!(parsed.patch.is_empty());
    }

    #[test]
    fn parse_mr_changes_payload_reads_diff_refs() {
        let payload = json!({
            "overflow": false,
            "diff_refs": {
                "base_sha": "base123",
                "start_sha": "start123",
                "head_sha": "head123"
            },
            "changes": []
        });
        let parsed = GitLabClient::parse_mr_changes_payload(&payload);
        assert_eq!(parsed.base_sha.as_deref(), Some("base123"));
        assert_eq!(parsed.start_sha.as_deref(), Some("start123"));
        assert_eq!(parsed.head_sha.as_deref(), Some("head123"));
    }

    #[test]
    fn mr_label_api_error_retry_heuristic_matches_glab_500() {
        assert!(mr_label_api_error_should_retry(
            "Failed to add label to MR: glab: 500 Internal Server Error (HTTP 500)"
        ));
    }

    #[test]
    fn mr_label_api_error_retry_skips_permanent_http_codes() {
        assert!(!mr_label_api_error_should_retry(
            "Failed to add label to MR: glab: 404 Not Found (HTTP 404)"
        ));
        assert!(!mr_label_api_error_should_retry(
            "Failed to add label to MR: HTTP 403 Forbidden"
        ));
    }

    /// Matches a single note object inside GitLab issue discussions API JSON.
    #[test]
    fn issue_discussions_api_note_deserializes_to_issue_thread_note() {
        let j = json!({
            "id": 24087u64,
            "type": null,
            "body": "**PMO needs clarification**",
            "attachment": null,
            "author": {
                "id": 480,
                "username": "alice",
                "name": "Alice",
                "state": "active",
                "web_url": "https://example.com/alice"
            },
            "system": false,
            "noteable_id": 291,
            "noteable_type": "Issue",
            "noteable_iid": 24
        });
        let n: IssueThreadNote = serde_json::from_value(j).unwrap();
        assert_eq!(n.id, 24087);
        assert_eq!(n.author_username(), "alice");
        assert!(!n.system);
        assert!(n.discussion_id.is_none());
    }

    #[test]
    fn glab_issue_discussions_json_flattens_with_discussion_ids() {
        let j = json!([
            {
                "id": "disc-a",
                "notes": [
                    { "id": 1u64, "body": "root", "system": false, "author": { "username": "u1" } },
                    { "id": 2u64, "body": "reply", "system": false, "author": { "username": "u2" } }
                ]
            },
            {
                "id": "disc-b",
                "notes": [
                    { "id": 3u64, "body": "other", "system": false, "author": { "username": "u3" } }
                ]
            }
        ]);

        let discussions = j.as_array().unwrap();
        let mut flat: Vec<(&str, u64)> = Vec::new();
        for d in discussions {
            let id = d["id"].as_str().unwrap();
            for n in d["notes"].as_array().unwrap() {
                flat.push((id, n["id"].as_u64().unwrap()));
            }
        }
        assert_eq!(flat, vec![("disc-a", 1), ("disc-a", 2), ("disc-b", 3)]);
    }

    #[test]
    fn mr_description_closes_issue_matches_variants() {
        assert!(mr_description_closes_issue("Closes #42", 42));
        assert!(mr_description_closes_issue("closes #42", 42));
        assert!(mr_description_closes_issue("CLOSE #7 and more", 7));
        assert!(mr_description_closes_issue("Text\n\nCloses #99\n", 99));
        assert!(!mr_description_closes_issue("Closes #42", 43));
        assert!(!mr_description_closes_issue("Refs #42", 42));
    }
}
