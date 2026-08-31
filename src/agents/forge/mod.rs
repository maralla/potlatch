//! Software forge abstraction.
//!
//! Agents (worker, reviewer, pmo, qa, ops) interact with the forge (GitLab,
//! GitHub) through the [`ForgeClient`] trait. Each provider implements it in
//! its own submodule ([`gitlab`], [`github`]); the agents are
//! provider-agnostic.
//!
//! The shared data types ([`Issue`], [`MergeRequest`], [`Comment`], etc.)
//! originated from the GitLab API shape and are reused across all providers.
//! Each provider maps its native types into these shared shapes.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;

pub mod github;
pub mod gitlab;

// ─── Shared types ─────────────────────────────────────────────────────────────

pub const PRIORITY_LABEL_PREFIX: &str = "priority::";
pub const DEFAULT_PRIORITY: u8 = 3;

/// Empty or whitespace-only `scope_label` means handle all items.
pub fn scope_label_filter(scope_label: &str) -> Option<&str> {
    let t = scope_label.trim();
    if t.is_empty() { None } else { Some(t) }
}

pub fn issue_in_scope(issue: &Issue, scope_label: Option<&str>) -> bool {
    issue_labels_in_scope(&issue.labels, scope_label)
}

/// Scope test for an issue known only by its labels, so a role that
/// observes an issue through its own snapshot type still applies exactly
/// the same rule as [`issue_in_scope`].
pub fn issue_labels_in_scope(labels: &[String], scope_label: Option<&str>) -> bool {
    match scope_label {
        None => true,
        Some(l) => labels.iter().any(|x| x == l),
    }
}

pub fn mr_in_scope(mr: &MergeRequest, scope_label: Option<&str>) -> bool {
    mr_labels_in_scope(mr.labels.as_deref(), scope_label)
}

/// Scope test for a merge request known only by its labels, so a role that
/// observes an MR through its own snapshot type still applies exactly the
/// same rule as [`mr_in_scope`].
pub fn mr_labels_in_scope(labels: Option<&[String]>, scope_label: Option<&str>) -> bool {
    if labels.is_some_and(|labels| {
        labels
            .iter()
            .any(|x| x == crate::agents::labels::NEED_AI_WORKER)
    }) {
        return true;
    }
    match scope_label {
        None => true,
        Some(l) => labels.is_some_and(|labels| labels.iter().any(|x| x == l)),
    }
}

const SPLIT_PARENT_PREFIX: &str = "This issue was split from parent issue #";

/// Append stable, human-readable split provenance to a child issue created
/// by splitting a parent.
pub fn with_split_parent(description: &str, parent_iid: u64) -> String {
    format!(
        "{}\n\n---\n\n{SPLIT_PARENT_PREFIX}{parent_iid}.",
        description.trim_end()
    )
}

/// Read the parent issue IID from split provenance, when present.
pub fn split_parent_iid(description: &str) -> Option<u64> {
    description.lines().find_map(|line| {
        line.trim()
            .strip_prefix(SPLIT_PARENT_PREFIX)?
            .strip_suffix('.')?
            .parse()
            .ok()
    })
}

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
    pub discussion_resolvable: bool,
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

/// Issue note with fields needed to resolve thread replies.
#[derive(Debug, Clone, Deserialize)]
pub struct IssueThreadNote {
    pub id: u64,
    pub body: String,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub discussion_id: Option<String>,
    pub author: IssueThreadNoteAuthor,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IssueThreadNoteAuthor {
    #[serde(default)]
    pub username: String,
}

impl IssueThreadNote {
    pub fn author_username(&self) -> &str {
        self.author.username.as_str()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResourceLabelEvent {
    pub id: u64,
    pub action: String,
    pub created_at: String,
    #[serde(default)]
    pub label: Option<ResourceLabelEventLabel>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResourceLabelEventLabel {
    pub name: String,
}

pub(crate) fn order_active_claim_labels(
    events: &[ResourceLabelEvent],
    active_claim_labels: &[String],
) -> Result<Vec<String>> {
    use anyhow::Context;
    let mut ordered = Vec::with_capacity(active_claim_labels.len());
    for label in active_claim_labels {
        let event = events
            .iter()
            .filter(|event| {
                event
                    .label
                    .as_ref()
                    .is_some_and(|event_label| event_label.name == *label)
            })
            .max_by_key(|event| event.id)
            .with_context(|| format!("active claim label {label:?} has no label event"))?;
        anyhow::ensure!(
            event.action == "add",
            "latest label event for active claim {label:?} is not an add"
        );
        let created_at = chrono::DateTime::parse_from_rfc3339(&event.created_at)
            .with_context(|| format!("invalid timestamp for claim label {label:?}"))?;
        ordered.push((created_at, event.id, label.clone()));
    }
    ordered.sort();
    Ok(ordered.into_iter().map(|(_, _, label)| label).collect())
}

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

pub fn issue_iid_from_branch(branch: &str) -> Option<u64> {
    branch.strip_prefix("issue-")?.parse().ok()
}

static CLOSES_ISSUE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)closes?\s+#(\d+)").expect("CLOSES_ISSUE_RE"));

/// True if the MR description contains `Closes #iid` / `Close #iid` for this
/// issue (case-insensitive). The `Closes #NNN` convention is shared across
/// GitLab and GitHub.
pub fn mr_description_closes_issue(description: &str, issue_iid: u64) -> bool {
    CLOSES_ISSUE_RE.captures_iter(description).any(|cap| {
        cap.get(1)
            .and_then(|m| m.as_str().parse::<u64>().ok())
            .is_some_and(|n| n == issue_iid)
    })
}

/// Returns true if `err` wraps a [`NonRetryable`] error (e.g. a permanent
/// `404 Not Found` for a deleted issue or MR). Callers that fetch a specific
/// resource by IID should check this and abandon the stale resource (e.g.
/// clear persisted claim state) instead of treating it as a transient failure.
pub fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.is::<crate::core::retry::NonRetryable>())
}

// ─── Trait ─────────────────────────────────────────────────────────────────────

/// Unified forge API. Implemented by [`gitlab::GitLabClient`] and
/// [`github::GitHubClient`]. Agents hold an `Arc<dyn ForgeClient>` and
/// are unaware of which forge they're talking to.
///
/// "Merge request" is the GitLab term; on GitHub the equivalent is a "pull
/// request". The trait uses "merge request" throughout for consistency; the
/// GitHub implementation maps PRs to the same [`MergeRequest`] type.
pub trait ForgeClient: Send + Sync {
    // ── Issues ──

    fn list_issues(&self) -> Result<Vec<Issue>>;
    fn get_issue(&self, iid: u64) -> Result<Issue>;
    fn create_issue(&self, title: &str, description: &str) -> Result<u64>;
    fn close_issue(&self, iid: u64) -> Result<()>;
    fn update_issue_description(&self, iid: u64, description: &str) -> Result<()>;

    // ── Issue labels ──

    fn add_issue_label(&self, iid: u64, label: &str) -> Result<()>;
    fn remove_issue_label(&self, iid: u64, label: &str) -> Result<()>;

    // ── Issue comments ──

    fn add_issue_comment(&self, iid: u64, comment: &str) -> Result<()>;
    fn get_issue_comments(&self, iid: u64) -> Result<Vec<Comment>>;
    fn get_issue_thread_notes(&self, iid: u64) -> Result<Vec<IssueThreadNote>>;

    // ── Merge requests / pull requests ──

    fn list_merge_requests(&self) -> Result<Vec<MergeRequest>>;
    fn get_merge_request(&self, iid: u64) -> Result<MergeRequest>;
    fn find_open_mr_by_source_branch(&self, source_branch: &str) -> Result<Option<u64>>;
    fn find_mrs_by_source_branch(&self, source_branch: &str) -> Result<Vec<u64>>;
    fn get_merge_request_changes(&self, iid: u64) -> Result<MergeRequestChangesSnapshot>;
    fn create_merge_request(
        &self,
        source_branch: &str,
        target_branch: &str,
        title: &str,
        description: &str,
    ) -> Result<u64>;
    fn merge_mr(&self, iid: u64) -> Result<()>;
    fn close_mr(&self, iid: u64) -> Result<()>;
    fn update_mr_title_description(
        &self,
        iid: u64,
        title: Option<&str>,
        description: Option<&str>,
    ) -> Result<()>;

    // ── MR labels ──

    fn add_mr_label_with_retries(&self, iid: u64, label: &str) -> Result<()>;
    fn remove_mr_label(&self, iid: u64, label: &str) -> Result<()>;

    // ── MR comments & discussions ──

    fn add_mr_comment(&self, iid: u64, comment: &str) -> Result<()>;
    fn add_mr_discussion(&self, iid: u64, body: &str) -> Result<()>;
    fn add_resolved_mr_discussion(&self, iid: u64, body: &str) -> Result<()>;
    fn get_mr_comments(&self, iid: u64) -> Result<Vec<Comment>>;
    fn get_unresolved_discussion_ids(&self, iid: u64) -> Result<Vec<String>>;
    fn get_unresolved_discussion_count(&self, iid: u64) -> Result<(usize, usize)>;
    fn resolve_discussion(&self, mr_iid: u64, discussion_id: &str) -> Result<()>;
    fn reply_to_discussion(&self, mr_iid: u64, discussion_id: &str, body: &str) -> Result<()>;

    // ── Utility ──

    fn is_not_found(&self, err: &anyhow::Error) -> bool;

    // ── Label events (for claim ordering) ──

    /// Label-event history for an issue, used to order active claims by when
    /// they were added. Providers without label-event history (GitHub scaffold)
    /// return an empty vec; the claim layer falls back to lexicographic order.
    fn get_issue_label_events(&self, _iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        Ok(Vec::new())
    }

    /// Label-event history for a merge request. See [`Self::get_issue_label_events`].
    fn get_mr_label_events(&self, _iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        Ok(Vec::new())
    }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

/// Construct the appropriate [`ForgeClient`] for `repo_url`.
///
/// GitHub URLs get a [`github::GitHubClient`]; everything else gets a
/// [`gitlab::GitLabClient`]. The concrete types stay private to this module —
/// callers receive an `Arc<dyn ForgeClient>` and never see which forge
/// they're talking to.
pub fn create_client(
    working_dir: &str,
    repo_url: &str,
    shutdown: Arc<AtomicBool>,
) -> Result<Arc<dyn ForgeClient>> {
    let url_lower = repo_url.to_lowercase();
    if url_lower.contains("github.com") {
        let client = github::GitHubClient::new(working_dir.to_string(), repo_url)?;
        return Ok(Arc::new(client));
    }
    let client = gitlab::GitLabClient::new(working_dir.to_string(), repo_url, shutdown)?;
    Ok(Arc::new(client))
}

/// Construct a no-network client for tests. Returns a GitLab-backed client
/// whose methods are never called — tests use it to satisfy
/// `Arc<dyn ForgeClient>` fields in agent-state structs while
/// exercising pure/file-only logic paths.
#[cfg(test)]
pub fn for_test_client(repo_path: impl Into<String>) -> Arc<dyn ForgeClient> {
    Arc::new(gitlab::GitLabClient::for_test(repo_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_issue(labels: Vec<&str>) -> Issue {
        Issue {
            iid: 1,
            title: "t".to_string(),
            description: "".to_string(),
            labels: labels.into_iter().map(String::from).collect(),
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        }
    }

    fn sample_mr(labels: Option<Vec<&str>>) -> MergeRequest {
        MergeRequest {
            iid: 1,
            title: "t".to_string(),
            description: "".to_string(),
            source_branch: "issue-1".to_string(),
            target_branch: "main".to_string(),
            state: "opened".to_string(),
            sha: None,
            labels: labels.map(|v| v.into_iter().map(String::from).collect()),
            has_conflicts: false,
        }
    }

    #[test]
    fn issue_in_scope_respects_label() {
        let issue = sample_issue(vec!["potlatch", "bug"]);
        assert!(issue_in_scope(&issue, None));
        assert!(issue_in_scope(&issue, Some("potlatch")));
        assert!(!issue_in_scope(&issue, Some("other")));
    }

    #[test]
    fn mr_in_scope_respects_label() {
        let mr = sample_mr(Some(vec!["potlatch"]));
        assert!(mr_in_scope(&mr, None));
        assert!(mr_in_scope(&mr, Some("potlatch")));
        assert!(!mr_in_scope(&mr, Some("other")));

        let no_labels = sample_mr(None);
        assert!(!mr_in_scope(&no_labels, Some("potlatch")));

        let ai_worker = sample_mr(Some(vec![crate::agents::labels::NEED_AI_WORKER]));
        assert!(mr_in_scope(&ai_worker, Some("other-scope")));
    }

    #[test]
    fn split_parent_provenance_round_trips_without_changing_the_child_scope() {
        let description = with_split_parent("Implement the parser.\n", 42);

        assert!(description.starts_with("Implement the parser.\n\n---"));
        assert_eq!(split_parent_iid(&description), Some(42));
        assert_eq!(split_parent_iid("Implement an unrelated issue."), None);
    }

    #[test]
    fn scope_label_filter_treats_blank_as_all() {
        assert_eq!(scope_label_filter(""), None);
        assert_eq!(scope_label_filter("  "), None);
        assert_eq!(scope_label_filter("potlatch"), Some("potlatch"));
    }
}
