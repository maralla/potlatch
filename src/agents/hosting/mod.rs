//! Unified code hosting abstraction.
//!
//! Agents (worker, reviewer, pmo, qa, ops) interact with the code hosting
//! platform (GitLab, GitHub) through the [`CodeHostingClient`] trait. Each
//! provider implements it in its own submodule ([`gitlab`], [`github`]);
//! the agents are provider-agnostic.
//!
//! The shared data types ([`Issue`], [`MergeRequest`], [`Comment`], etc.)
//! originated from the GitLab API shape and are reused across all providers.
//! Each provider maps its native types into these shared shapes.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod github;
pub mod gitlab;

// ─── Shared types ─────────────────────────────────────────────────────────────

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

/// Returns true if `err` wraps a [`NonRetryable`] error (e.g. a permanent
/// `404 Not Found` for a deleted issue or MR). Callers that fetch a specific
/// resource by IID should check this and abandon the stale resource (e.g.
/// clear persisted claim state) instead of treating it as a transient failure.
pub fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.is::<crate::core::retry::NonRetryable>())
}

// ─── Trait ─────────────────────────────────────────────────────────────────────

/// Unified code hosting API. Implemented by [`gitlab::GitLabClient`] and
/// [`github::GitHubClient`]. Agents hold an `Arc<dyn CodeHostingClient>` and
/// are unaware of which platform they're talking to.
///
/// "Merge request" is the GitLab term; on GitHub the equivalent is a "pull
/// request". The trait uses "merge request" throughout for consistency; the
/// GitHub implementation maps PRs to the same [`MergeRequest`] type.
pub trait CodeHostingClient: Send + Sync {
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
}
