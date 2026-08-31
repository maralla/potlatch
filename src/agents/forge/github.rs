//! GitHub forge client.
//!
//! Implements [`super::ForgeClient`] using the GitHub REST API (via
//! the `gh` CLI). Maps GitHub's pull requests to the shared [`super::MergeRequest`]
//! type and GitHub issues to [`super::Issue`].
//!
//! **Status:** trait implementation scaffold. The method bodies are not yet
//! implemented — each returns a "not yet implemented" error so the trait is
//! wired up but unusable until the real API calls are added.

use anyhow::Result;

use super::{
    Comment, ForgeClient, Issue, IssueThreadNote, MergeRequest, MergeRequestChangesSnapshot,
};

pub(crate) struct GitHubClient;

impl GitHubClient {
    pub(crate) fn new(_repo_path: String, _repo_url: &str) -> Result<Self> {
        Ok(Self)
    }
}

impl ForgeClient for GitHubClient {
    fn list_issues(&self) -> Result<Vec<Issue>> {
        unimplemented!("GitHub: list_issues")
    }
    fn get_issue(&self, _iid: u64) -> Result<Issue> {
        unimplemented!("GitHub: get_issue")
    }
    fn create_issue(&self, _title: &str, _description: &str) -> Result<u64> {
        unimplemented!("GitHub: create_issue")
    }
    fn close_issue(&self, _iid: u64) -> Result<()> {
        unimplemented!("GitHub: close_issue")
    }
    fn update_issue_description(&self, _iid: u64, _description: &str) -> Result<()> {
        unimplemented!("GitHub: update_issue_description")
    }
    fn add_issue_label(&self, _iid: u64, _label: &str) -> Result<()> {
        unimplemented!("GitHub: add_issue_label")
    }
    fn remove_issue_label(&self, _iid: u64, _label: &str) -> Result<()> {
        unimplemented!("GitHub: remove_issue_label")
    }
    fn add_issue_comment(&self, _iid: u64, _comment: &str) -> Result<()> {
        unimplemented!("GitHub: add_issue_comment")
    }
    fn get_issue_comments(&self, _iid: u64) -> Result<Vec<Comment>> {
        unimplemented!("GitHub: get_issue_comments")
    }
    fn get_issue_thread_notes(&self, _iid: u64) -> Result<Vec<IssueThreadNote>> {
        unimplemented!("GitHub: get_issue_thread_notes")
    }
    fn list_merge_requests(&self) -> Result<Vec<MergeRequest>> {
        unimplemented!("GitHub: list_merge_requests")
    }
    fn get_merge_request(&self, _iid: u64) -> Result<MergeRequest> {
        unimplemented!("GitHub: get_merge_request")
    }
    fn find_open_mr_by_source_branch(&self, _source_branch: &str) -> Result<Option<u64>> {
        unimplemented!("GitHub: find_open_mr_by_source_branch")
    }
    fn find_mrs_by_source_branch(&self, _source_branch: &str) -> Result<Vec<u64>> {
        unimplemented!("GitHub: find_mrs_by_source_branch")
    }
    fn get_merge_request_changes(&self, _iid: u64) -> Result<MergeRequestChangesSnapshot> {
        unimplemented!("GitHub: get_merge_request_changes")
    }
    fn create_merge_request(
        &self,
        _source_branch: &str,
        _target_branch: &str,
        _title: &str,
        _description: &str,
    ) -> Result<u64> {
        unimplemented!("GitHub: create_merge_request")
    }
    fn merge_mr(&self, _iid: u64) -> Result<()> {
        unimplemented!("GitHub: merge_mr")
    }
    fn close_mr(&self, _iid: u64) -> Result<()> {
        unimplemented!("GitHub: close_mr")
    }
    fn update_mr_title_description(
        &self,
        _iid: u64,
        _title: Option<&str>,
        _description: Option<&str>,
    ) -> Result<()> {
        unimplemented!("GitHub: update_mr_title_description")
    }
    fn add_mr_label_with_retries(&self, _iid: u64, _label: &str) -> Result<()> {
        unimplemented!("GitHub: add_mr_label_with_retries")
    }
    fn remove_mr_label(&self, _iid: u64, _label: &str) -> Result<()> {
        unimplemented!("GitHub: remove_mr_label")
    }
    fn add_mr_comment(&self, _iid: u64, _comment: &str) -> Result<()> {
        unimplemented!("GitHub: add_mr_comment")
    }
    fn add_mr_discussion(&self, _iid: u64, _body: &str) -> Result<()> {
        unimplemented!("GitHub: add_mr_discussion")
    }
    fn add_resolved_mr_discussion(&self, _iid: u64, _body: &str) -> Result<()> {
        unimplemented!("GitHub: add_resolved_mr_discussion")
    }
    fn get_mr_comments(&self, _iid: u64) -> Result<Vec<Comment>> {
        unimplemented!("GitHub: get_mr_comments")
    }
    fn get_unresolved_discussion_ids(&self, _iid: u64) -> Result<Vec<String>> {
        unimplemented!("GitHub: get_unresolved_discussion_ids")
    }
    fn get_unresolved_discussion_count(&self, _iid: u64) -> Result<(usize, usize)> {
        unimplemented!("GitHub: get_unresolved_discussion_count")
    }
    fn resolve_discussion(&self, _mr_iid: u64, _discussion_id: &str) -> Result<()> {
        unimplemented!("GitHub: resolve_discussion")
    }
    fn reply_to_discussion(&self, _mr_iid: u64, _discussion_id: &str, _body: &str) -> Result<()> {
        unimplemented!("GitHub: reply_to_discussion")
    }
    fn is_not_found(&self, _err: &anyhow::Error) -> bool {
        false
    }
}
