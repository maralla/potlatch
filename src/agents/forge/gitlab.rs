use anyhow::{Context, Result};

use crate::core::retry::{NonRetryable, with_backoff_retries};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tracing::{debug, info};

#[cfg(test)]
pub(crate) use super::order_active_claim_labels;
use super::{
    Comment, Issue, IssueThreadNote, IssueThreadNoteAuthor, MergeRequest,
    MergeRequestChangesSnapshot, ResourceLabelEvent,
};
use super::{is_not_found, sort_issues_by_priority};

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
                discussion_resolvable: false,
                location: None,
                location_details: None,
            }
        })
        .collect()
}

#[derive(Clone)]
pub(crate) struct GitLabClient {
    repo_path: String,
    host: String,
    project_id: u64,
    shutdown: Arc<AtomicBool>,
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

fn resolve_project_id(host: &str, project_path: &str, shutdown: &AtomicBool) -> Result<u64> {
    let search_term = project_path.rsplit('/').next().unwrap_or(project_path);
    let endpoint = format!(
        "projects?search={}&membership=true&simple=true&per_page=50",
        search_term
    );
    // This runs once per agent at startup, concurrently across all GitLab agents
    // (worker/reviewer/pmo/ops/qa). It is the only glab call that wasn't retried,
    // so a single transient blip (TLS timeout, 502, rate-limit) while N agents race
    // to resolve the same project id would kill agent startup nondeterministically.
    // Wrap it in the same transient-retry policy every other glab call uses.
    let output = with_backoff_retries(
        shutdown,
        &format!("resolve GitLab project id for {project_path} on {host}"),
        || {
            let output = Command::new("glab")
                .args(["api", "--hostname", host, &endpoint])
                .output()
                .with_context(|| {
                    format!("Failed to resolve GitLab project id for {project_path}")
                })?;
            if !output.status.success() {
                anyhow::bail!(
                    "Failed to resolve GitLab project id for {}: {}",
                    project_path,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Ok(output)
        },
    )?;
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

fn compact_cli_output(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl GitLabClient {
    pub(crate) fn new(
        repo_path: String,
        gitlab_repo: &str,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Self> {
        let (host, project_path) = parse_gitlab_repo(gitlab_repo)?;
        let project_id = resolve_project_id(&host, &project_path, &shutdown)?;
        configure_repo_glab(&repo_path, &host)?;
        Ok(Self {
            repo_path,
            host,
            project_id,
            shutdown,
        })
    }

    /// Construct a client for characterization tests without resolving a
    /// real GitLab project (no network access). Values built this way must
    /// never reach a method that shells out to `glab` — they only exist so
    /// tests can construct role `AgentState`-style structs that embed a
    /// client field to exercise pure/file-only logic paths.
    #[cfg(test)]
    pub(crate) fn for_test(repo_path: impl Into<String>) -> Self {
        Self {
            repo_path: repo_path.into(),
            host: "gitlab.example.com".to_string(),
            project_id: 1,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    fn api_path(&self, tail: &str) -> String {
        format!(
            "projects/{}/{}",
            self.project_id,
            tail.trim_start_matches('/')
        )
    }

    fn run_api(&self, endpoint: &str, extra_args: &[&str]) -> Result<std::process::Output> {
        with_backoff_retries(&self.shutdown, &format!("glab api GET {endpoint}"), || {
            let output = self.run_api_once(endpoint, extra_args)?;
            if !output.status.success() {
                let message = Self::glab_api_error_message(&output);
                // A 404 is permanent (deleted issue/MR); don't retry forever.
                if Self::is_404(&message) {
                    return Err(
                        NonRetryable(format!("glab api GET {endpoint} failed: {message}")).into(),
                    );
                }
                anyhow::bail!("glab api GET {endpoint} failed: {message}");
            }
            Ok(output)
        })
    }

    /// Returns true if a glab api error message indicates HTTP 404.
    pub fn is_404(message: &str) -> bool {
        message.contains("404 Not found") || message.contains("HTTP 404")
    }

    fn run_api_once(&self, endpoint: &str, extra_args: &[&str]) -> Result<std::process::Output> {
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
            compact_cli_output(&stderr)
        } else if stderr.trim().is_empty() {
            compact_cli_output(&stdout)
        } else {
            compact_cli_output(&format!("{stderr}\n{stdout}"))
        }
    }

    /// Returns the IID of an open MR for `source_branch`, if one exists.
    pub fn find_open_mr_by_source_branch(&self, source_branch: &str) -> Result<Option<u64>> {
        let endpoint = self.api_path(&format!(
            "merge_requests?source_branch={source_branch}&state=opened&per_page=1"
        ));
        let output = self.run_api(&endpoint, &[])?;
        #[derive(Deserialize)]
        struct MrHit {
            iid: u64,
        }
        let hits: Vec<MrHit> =
            serde_json::from_slice(&output.stdout).context("Failed to parse MR search JSON")?;
        Ok(hits.into_iter().next().map(|mr| mr.iid))
    }

    /// Find merge requests by source branch in any state (open, closed, merged).
    /// Returns MR iids sorted by iid descending (most recent first). Used by the
    /// PMO to inspect closed MRs when a worker hands off an issue after failing
    /// to resolve reviewer feedback — the MR's comments and diff carry context
    /// the issue comments alone don't capture.
    pub fn find_mrs_by_source_branch(&self, source_branch: &str) -> Result<Vec<u64>> {
        let endpoint = self.api_path(&format!(
            "merge_requests?source_branch={source_branch}&per_page=20&order_by=updated_at&sort=desc"
        ));
        let output = self.run_api(&endpoint, &[])?;
        #[derive(Deserialize)]
        struct MrHit {
            iid: u64,
        }
        let hits: Vec<MrHit> =
            serde_json::from_slice(&output.stdout).context("Failed to parse MR search JSON")?;
        let mut iids: Vec<u64> = hits.into_iter().map(|h| h.iid).collect();
        iids.sort_by(|a, b| b.cmp(a));
        Ok(iids)
    }

    pub fn list_issues(&self) -> Result<Vec<Issue>> {
        with_backoff_retries(&self.shutdown, "listing open issues", || {
            self.list_issues_pages()
        })
    }

    fn list_issues_pages(&self) -> Result<Vec<Issue>> {
        debug!("Fetching issues from GitLab");
        const PER_PAGE: usize = 100;
        let mut page = 1usize;
        let mut issues = Vec::new();

        loop {
            let endpoint = self.api_path(&format!(
                "issues?state=opened&per_page={PER_PAGE}&page={page}"
            ));
            let output = self.run_api(&endpoint, &[])?;

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

    pub(crate) fn get_issue_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        self.get_resource_label_events(&format!("issues/{iid}/resource_label_events"))
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

        let mrs: Vec<MergeRequest> = serde_json::from_slice(&output.stdout)
            .context("Failed to parse merge requests JSON")?;

        Ok(mrs)
    }

    pub fn get_merge_request(&self, iid: u64) -> Result<MergeRequest> {
        debug!("Fetching merge request !{}", iid);

        let endpoint = self.api_path(&format!("merge_requests/{iid}"));
        let output = self.run_api(&endpoint, &[])?;

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

        let discussions = match self.fetch_discussions(iid) {
            Ok(discussions) => discussions,
            Err(e) => {
                debug!("Failed to get MR discussions: {}", e);
                return Ok(Vec::new());
            }
        };

        let mut comments = Vec::new();
        for discussion in &discussions {
            let discussion_id = discussion["id"].as_str().unwrap_or("").to_string();
            let discussion_resolvable = Self::discussion_resolution(discussion).is_some();
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
                        discussion_resolvable,
                        location: note_location,
                        location_details: note_location_details,
                    });
                }
            }
        }

        Ok(comments)
    }

    fn fetch_discussions(&self, iid: u64) -> Result<Vec<serde_json::Value>> {
        with_backoff_retries(
            &self.shutdown,
            &format!("fetching MR !{iid} discussions"),
            || self.fetch_discussions_pages(iid),
        )
    }

    fn fetch_discussions_pages(&self, iid: u64) -> Result<Vec<serde_json::Value>> {
        const PER_PAGE: usize = 100;
        let mut page = 1usize;
        let mut discussions = Vec::new();

        loop {
            let endpoint = self.api_path(&format!(
                "merge_requests/{iid}/discussions?per_page={PER_PAGE}&page={page}"
            ));
            let output = self.run_api(&endpoint, &[])?;

            let batch: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)
                .with_context(|| format!("Failed to parse MR discussions JSON page {page}"))?;
            let batch_len = batch.len();
            discussions.extend(batch);

            if batch_len < PER_PAGE {
                break;
            }
            page += 1;
        }

        Ok(discussions)
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

    /// Like [`Self::add_mr_label`], but retries indefinitely with capped exponential backoff.
    pub fn add_mr_label_with_retries(&self, iid: u64, label: &str) -> Result<()> {
        with_backoff_retries(
            &self.shutdown,
            &format!("adding label {label:?} to MR !{iid}"),
            || self.add_mr_label(iid, label),
        )
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

    pub(crate) fn get_mr_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        self.get_resource_label_events(&format!("merge_requests/{iid}/resource_label_events"))
    }

    fn get_resource_label_events(&self, resource_path: &str) -> Result<Vec<ResourceLabelEvent>> {
        const PER_PAGE: usize = 100;
        let mut page = 1usize;
        let mut events = Vec::new();

        loop {
            let endpoint =
                self.api_path(&format!("{resource_path}?per_page={PER_PAGE}&page={page}"));
            let output = self.run_api(&endpoint, &[])?;
            if !output.status.success() {
                anyhow::bail!(
                    "Failed to fetch resource label events: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let mut batch: Vec<ResourceLabelEvent> = serde_json::from_slice(&output.stdout)
                .context("Failed to parse resource label events")?;
            let batch_len = batch.len();
            events.append(&mut batch);
            if batch_len < PER_PAGE {
                break;
            }
            page += 1;
        }

        Ok(events)
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

    pub fn update_issue_description(&self, iid: u64, description: &str) -> Result<()> {
        debug!("Updating issue #{} description", iid);

        let output = self.run_api_method(
            &format!("issues/{iid}"),
            "PUT",
            &[("description", description)],
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to update issue #{} description: {}",
                iid,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        info!("Updated issue #{} description", iid);
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

impl super::ForgeClient for GitLabClient {
    fn list_issues(&self) -> Result<Vec<Issue>> {
        GitLabClient::list_issues(self)
    }
    fn get_issue(&self, iid: u64) -> Result<Issue> {
        GitLabClient::get_issue(self, iid)
    }
    fn create_issue(&self, title: &str, description: &str) -> Result<u64> {
        GitLabClient::create_issue(self, title, description)
    }
    fn close_issue(&self, iid: u64) -> Result<()> {
        GitLabClient::close_issue(self, iid)
    }
    fn update_issue_description(&self, iid: u64, description: &str) -> Result<()> {
        GitLabClient::update_issue_description(self, iid, description)
    }
    fn add_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        GitLabClient::add_issue_label(self, iid, label)
    }
    fn remove_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        GitLabClient::remove_issue_label(self, iid, label)
    }
    fn add_issue_comment(&self, iid: u64, comment: &str) -> Result<()> {
        GitLabClient::add_issue_comment(self, iid, comment)
    }
    fn get_issue_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        GitLabClient::get_issue_comments(self, iid)
    }
    fn get_issue_thread_notes(&self, iid: u64) -> Result<Vec<IssueThreadNote>> {
        GitLabClient::get_issue_thread_notes(self, iid)
    }
    fn list_merge_requests(&self) -> Result<Vec<MergeRequest>> {
        GitLabClient::list_merge_requests(self)
    }
    fn get_merge_request(&self, iid: u64) -> Result<MergeRequest> {
        GitLabClient::get_merge_request(self, iid)
    }
    fn find_open_mr_by_source_branch(&self, source_branch: &str) -> Result<Option<u64>> {
        GitLabClient::find_open_mr_by_source_branch(self, source_branch)
    }
    fn find_mrs_by_source_branch(&self, source_branch: &str) -> Result<Vec<u64>> {
        GitLabClient::find_mrs_by_source_branch(self, source_branch)
    }
    fn get_merge_request_changes(&self, iid: u64) -> Result<MergeRequestChangesSnapshot> {
        GitLabClient::get_merge_request_changes(self, iid)
    }
    fn create_merge_request(
        &self,
        source_branch: &str,
        target_branch: &str,
        title: &str,
        description: &str,
    ) -> Result<u64> {
        GitLabClient::create_merge_request(self, source_branch, target_branch, title, description)
    }
    fn merge_mr(&self, iid: u64) -> Result<()> {
        GitLabClient::merge_mr(self, iid)
    }
    fn close_mr(&self, iid: u64) -> Result<()> {
        GitLabClient::close_mr(self, iid)
    }
    fn update_mr_title_description(
        &self,
        iid: u64,
        title: Option<&str>,
        description: Option<&str>,
    ) -> Result<()> {
        GitLabClient::update_mr_title_description(
            self,
            iid,
            title.unwrap_or(""),
            description.unwrap_or(""),
        )
    }
    fn add_mr_label_with_retries(&self, iid: u64, label: &str) -> Result<()> {
        GitLabClient::add_mr_label_with_retries(self, iid, label)
    }
    fn remove_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        GitLabClient::remove_mr_label(self, iid, label)
    }
    fn add_mr_comment(&self, iid: u64, comment: &str) -> Result<()> {
        GitLabClient::add_mr_comment(self, iid, comment)
    }
    fn add_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        GitLabClient::add_mr_discussion(self, iid, body)
    }
    fn add_resolved_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        GitLabClient::add_resolved_mr_discussion(self, iid, body)
    }
    fn get_mr_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        GitLabClient::get_mr_comments(self, iid)
    }
    fn get_unresolved_discussion_ids(&self, iid: u64) -> Result<Vec<String>> {
        GitLabClient::get_unresolved_discussion_ids(self, iid)
    }
    fn get_unresolved_discussion_count(&self, iid: u64) -> Result<(usize, usize)> {
        GitLabClient::get_unresolved_discussion_count(self, iid)
    }
    fn resolve_discussion(&self, mr_iid: u64, discussion_id: &str) -> Result<()> {
        GitLabClient::resolve_discussion(self, mr_iid, discussion_id)
    }
    fn reply_to_discussion(&self, mr_iid: u64, discussion_id: &str, body: &str) -> Result<()> {
        GitLabClient::reply_to_discussion(self, mr_iid, discussion_id, body)
    }
    fn is_not_found(&self, err: &anyhow::Error) -> bool {
        is_not_found(err)
    }
    fn get_issue_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        GitLabClient::get_issue_label_events(self, iid)
    }
    fn get_mr_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        GitLabClient::get_mr_label_events(self, iid)
    }
}

#[cfg(test)]
mod tests {
    use super::super::mr_description_closes_issue;
    use super::{
        GitLabClient, IssueThreadNote, MergeRequestChangesSnapshot, ResourceLabelEvent,
        compact_cli_output, mr_create_error_is_duplicate, order_active_claim_labels,
        parse_gitlab_repo,
    };
    use serde_json::json;

    #[test]
    fn parse_gitlab_repo_ssh_url() {
        let (host, path) =
            parse_gitlab_repo("git@gitlab.example.com:group/tool/example-project.git").unwrap();
        assert_eq!(host, "gitlab.example.com");
        assert_eq!(path, "group/tool/example-project");
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
    fn compact_cli_output_removes_glab_table_spacing() {
        assert_eq!(
            compact_cli_output(
                "              ERROR                Get \"https://gitlab.example/api\": net/http: TLS handshake timeout.\n"
            ),
            "ERROR Get \"https://gitlab.example/api\": net/http: TLS handshake timeout."
        );
    }

    #[test]
    fn parse_gitlab_repo_rejects_invalid_url() {
        assert!(parse_gitlab_repo("not-a-url").is_err());
    }

    #[test]
    fn active_claims_are_ordered_by_their_latest_add_events() {
        let events: Vec<ResourceLabelEvent> = serde_json::from_value(json!([
            {
                "id": 9,
                "action": "add",
                "created_at": "2026-08-19T08:03:08Z",
                "label": {"name": "claimed:reviewer-0"}
            },
            {
                "id": 4,
                "action": "add",
                "created_at": "2026-08-19T08:03:07Z",
                "label": {"name": "claimed:reviewer-1"}
            }
        ]))
        .unwrap();

        assert_eq!(
            order_active_claim_labels(
                &events,
                &[
                    "claimed:reviewer-0".to_string(),
                    "claimed:reviewer-1".to_string()
                ]
            )
            .unwrap(),
            vec![
                "claimed:reviewer-1".to_string(),
                "claimed:reviewer-0".to_string()
            ]
        );
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

    #[test]
    fn is_404_detects_glab_not_found_messages() {
        assert!(GitLabClient::is_404(
            "glab: 404 Not found (HTTP 404) {\"message\":\"404 Not found\"}"
        ));
        assert!(GitLabClient::is_404("404 Not found"));
        assert!(GitLabClient::is_404(
            "HTTP 404 {\"message\":\"404 Not found\"}"
        ));
        // The truncated format seen in worker logs (e.g. "Failed to verify
        // issue #956 for session resume: ... Not found (HTTP 404)
        // {"message":"404 Not found"}, skipping") must also be detected so a
        // deleted issue's stale session is cleaned up rather than retried.
        assert!(GitLabClient::is_404(
            "glab api GET projects/26962/issues/956 failed: Not found (HTTP 404) {\"message\":\"404 Not found\"}"
        ));
        assert!(!GitLabClient::is_404("HTTP 500 Internal Server Error"));
        assert!(!GitLabClient::is_404(
            "glab: HTTP 400\n{\"error\":\"target_branch is missing\"}"
        ));
        assert!(!GitLabClient::is_404("connection refused"));
    }

    #[test]
    fn is_not_found_detects_non_retryable_in_error_chain() {
        use super::is_not_found;
        use crate::core::retry::NonRetryable;

        let err: anyhow::Error =
            NonRetryable("glab api GET .../issues/957 failed: 404".to_string()).into();
        assert!(is_not_found(&err));

        let err: anyhow::Error = anyhow::anyhow!("transient failure");
        assert!(!is_not_found(&err));

        // Wrapped in a context chain — should still detect it.
        let err: anyhow::Error =
            anyhow::Error::from(NonRetryable("404".to_string())).context("fetching issue");
        assert!(is_not_found(&err));
    }
}
