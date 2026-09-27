//! GitHub implementation of the shared forge API, using the `gh` CLI.

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use chrono::DateTime;
use serde_json::{Value, json};

use super::{
    Comment, ForgeClient, Issue, IssueThreadNote, IssueThreadNoteAuthor, MergeRequest,
    MergeRequestChangesSnapshot, ResourceLabelEvent, ResourceLabelEventLabel,
    sort_issues_by_priority,
};
use crate::core::retry::NonRetryable;

pub(crate) struct GitHubClient {
    repo_path: String,
    repo: String,
    owner: String,
    name: String,
}

fn parse_github_repo(url: &str) -> Result<(String, String)> {
    let path = if let Some(path) = url.strip_prefix("git@github.com:") {
        path
    } else {
        let parsed = reqwest::Url::parse(url).context("invalid GitHub repository URL")?;
        anyhow::ensure!(
            parsed.host_str() == Some("github.com"),
            "expected github.com repository URL"
        );
        let path = parsed.path().trim_matches('/').trim_end_matches(".git");
        return split_repo(path);
    };
    split_repo(path.trim_end_matches(".git"))
}

fn split_repo(path: &str) -> Result<(String, String)> {
    let (owner, name) = path
        .split_once('/')
        .context("GitHub URL needs owner/repository")?;
    anyhow::ensure!(
        !owner.is_empty() && !name.is_empty() && !name.contains('/'),
        "GitHub URL needs owner/repository"
    );
    Ok((owner.to_string(), name.to_string()))
}

fn string(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or_default().to_string()
}

fn labels(value: &Value) -> Vec<String> {
    value["labels"].as_array().map_or_else(Vec::new, |items| {
        items
            .iter()
            .map(|label| string(label, "name"))
            .filter(|s| !s.is_empty())
            .collect()
    })
}

fn issue(value: &Value) -> Result<Issue> {
    Ok(Issue {
        iid: value["number"]
            .as_u64()
            .context("GitHub issue has no number")?,
        title: string(value, "title"),
        description: string(value, "body"),
        labels: labels(value),
        state: if value["state"] == "open" {
            "opened"
        } else {
            "closed"
        }
        .to_string(),
        created_at: value["created_at"].as_str().map(str::to_string),
        updated_at: value["updated_at"].as_str().map(str::to_string),
    })
}

fn pull(value: &Value) -> Result<MergeRequest> {
    let state = if !value["merged_at"].is_null() {
        "merged"
    } else if value["state"] == "open" {
        "opened"
    } else {
        "closed"
    };
    Ok(MergeRequest {
        iid: value["number"]
            .as_u64()
            .context("GitHub pull request has no number")?,
        title: string(value, "title"),
        description: string(value, "body"),
        source_branch: string(&value["head"], "ref"),
        target_branch: string(&value["base"], "ref"),
        state: state.to_string(),
        sha: value["head"]["sha"].as_str().map(str::to_string),
        labels: Some(labels(value)),
        has_conflicts: value["mergeable"] == false || value["mergeable_state"] == "dirty",
    })
}

fn comment_id(id: u64, created_at: &str) -> u64 {
    DateTime::parse_from_rfc3339(created_at)
        .ok()
        .and_then(|date| u64::try_from(date.timestamp()).ok())
        .map_or(id, |seconds| {
            seconds.saturating_mul(1_000_000_000) + id % 1_000_000_000
        })
}

fn plain_comment(value: &Value) -> Comment {
    let id = value["id"].as_u64().unwrap_or_default();
    Comment {
        id: comment_id(id, value["created_at"].as_str().unwrap_or_default()),
        body: string(value, "body"),
        author: string(&value["user"], "login"),
        discussion_id: format!("comment_{id}"),
        discussion_resolvable: false,
        location: None,
        location_details: None,
    }
}

impl GitHubClient {
    pub(crate) fn new(repo_path: String, repo_url: &str) -> Result<Self> {
        let (owner, name) = parse_github_repo(repo_url)?;
        Ok(Self {
            repo_path,
            repo: format!("{owner}/{name}"),
            owner,
            name,
        })
    }

    fn endpoint(&self, tail: &str) -> String {
        format!("repos/{}/{}", self.repo, tail)
    }

    fn api(&self, method: &str, endpoint: &str, body: Option<&Value>) -> Result<Value> {
        let mut command = Command::new("gh");
        command
            .args(["api", "-X", method, endpoint])
            .current_dir(&self.repo_path);
        if body.is_some() {
            command.args(["--input", "-"]).stdin(Stdio::piped());
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to run gh api {method} {endpoint}"))?;
        if let Some(body) = body {
            child
                .stdin
                .take()
                .context("gh stdin unavailable")?
                .write_all(&serde_json::to_vec(body)?)?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let message = format!("gh api {method} {endpoint} failed: {detail}");
            if detail.contains("HTTP 404") || detail.contains("Not Found (HTTP 404)") {
                return Err(NonRetryable(message).into());
            }
            bail!("{message}");
        }
        if output.stdout.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&output.stdout)
            .with_context(|| format!("invalid JSON from gh api {method} {endpoint}"))
    }

    fn get(&self, tail: &str) -> Result<Value> {
        self.api("GET", &self.endpoint(tail), None)
    }
    fn post(&self, tail: &str, body: Value) -> Result<Value> {
        self.api("POST", &self.endpoint(tail), Some(&body))
    }
    fn patch(&self, tail: &str, body: Value) -> Result<Value> {
        self.api("PATCH", &self.endpoint(tail), Some(&body))
    }
    fn put(&self, tail: &str, body: Value) -> Result<Value> {
        self.api("PUT", &self.endpoint(tail), Some(&body))
    }
    fn delete(&self, tail: &str) -> Result<Value> {
        self.api("DELETE", &self.endpoint(tail), None)
    }

    fn pages(&self, tail: &str) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        for page in 1.. {
            let separator = if tail.contains('?') { '&' } else { '?' };
            let response = self.get(&format!("{tail}{separator}per_page=100&page={page}"))?;
            let items = response.as_array().context("expected GitHub API list")?;
            let count = items.len();
            all.extend(items.iter().cloned());
            if count < 100 {
                break;
            }
        }
        Ok(all)
    }

    fn graphql(&self, query: &str, variables: Value) -> Result<Value> {
        let body = json!({"query": query, "variables": variables});
        let response = self.api("POST", "graphql", Some(&body))?;
        if let Some(errors) = response["errors"].as_array()
            && !errors.is_empty()
        {
            bail!("GitHub GraphQL error: {errors:?}");
        }
        Ok(response["data"].clone())
    }

    fn review_threads(&self, iid: u64) -> Result<Vec<Value>> {
        const QUERY: &str = "query($owner:String!,$name:String!,$number:Int!,$cursor:String){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewThreads(first:100,after:$cursor){nodes{id isResolved comments(first:100){nodes{fullDatabaseId body createdAt author{login} path line originalLine} pageInfo{hasNextPage}}} pageInfo{hasNextPage endCursor}}}}}";
        let mut threads = Vec::new();
        let mut cursor = Value::Null;
        loop {
            let data = self.graphql(
                QUERY,
                json!({
                    "owner": self.owner, "name": self.name, "number": iid, "cursor": cursor
                }),
            )?;
            let connection = &data["repository"]["pullRequest"]["reviewThreads"];
            let nodes = connection["nodes"]
                .as_array()
                .context("missing GitHub review threads")?;
            anyhow::ensure!(
                nodes
                    .iter()
                    .all(|thread| thread["comments"]["pageInfo"]["hasNextPage"] != true),
                "GitHub review thread exceeds 100 comments"
            );
            threads.extend(nodes.iter().cloned());
            if connection["pageInfo"]["hasNextPage"] != true {
                break;
            }
            cursor = connection["pageInfo"]["endCursor"].clone();
            anyhow::ensure!(!cursor.is_null(), "missing GitHub review thread cursor");
        }
        Ok(threads)
    }

    fn add_label(&self, iid: u64, label: &str) -> Result<()> {
        let endpoint = format!("issues/{iid}/labels");
        if let Err(error) = self.post(&endpoint, json!({"labels": [label]})) {
            if !error.to_string().contains("HTTP 422") {
                return Err(error);
            }
            // GitHub requires labels to exist before assignment. A concurrent
            // agent may have created this label between the two calls.
            if let Err(create_error) =
                self.post("labels", json!({"name": label, "color": "ededed"}))
                && !create_error.to_string().contains("HTTP 422")
            {
                return Err(create_error);
            }
            self.post(&endpoint, json!({"labels": [label]}))?;
        }
        Ok(())
    }

    fn remove_label(&self, iid: u64, label: &str) -> Result<()> {
        let encoded = url_encode(label);
        match self.delete(&format!("issues/{iid}/labels/{encoded}")) {
            Ok(_) => Ok(()),
            Err(error) if super::is_not_found(&error) => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        Ok(self
            .pages(&format!("issues/{iid}/events"))?
            .iter()
            .filter_map(|event| {
                let action = match event["event"].as_str()? {
                    "labeled" => "add",
                    "unlabeled" => "remove",
                    _ => return None,
                };
                Some(ResourceLabelEvent {
                    id: event["id"].as_u64()?,
                    action: action.to_string(),
                    created_at: string(event, "created_at"),
                    label: Some(ResourceLabelEventLabel {
                        name: string(&event["label"], "name"),
                    }),
                })
            })
            .collect::<Vec<_>>())
    }
}

fn url_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                (byte as char).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

impl ForgeClient for GitHubClient {
    fn list_issues(&self) -> Result<Vec<Issue>> {
        let mut issues = self
            .pages("issues?state=open")?
            .iter()
            .filter(|value| value.get("pull_request").is_none())
            .map(issue)
            .collect::<Result<Vec<_>>>()?;
        sort_issues_by_priority(&mut issues);
        Ok(issues)
    }
    fn get_issue(&self, iid: u64) -> Result<Issue> {
        issue(&self.get(&format!("issues/{iid}"))?)
    }
    fn create_issue(&self, title: &str, description: &str) -> Result<u64> {
        self.post("issues", json!({"title": title, "body": description}))?["number"]
            .as_u64()
            .context("created GitHub issue has no number")
    }
    fn close_issue(&self, iid: u64) -> Result<()> {
        self.patch(&format!("issues/{iid}"), json!({"state": "closed"}))?;
        Ok(())
    }
    fn update_issue_description(&self, iid: u64, description: &str) -> Result<()> {
        self.patch(&format!("issues/{iid}"), json!({"body": description}))?;
        Ok(())
    }
    fn add_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        self.add_label(iid, label)
    }
    fn remove_issue_label(&self, iid: u64, label: &str) -> Result<()> {
        self.remove_label(iid, label)
    }
    fn add_issue_comment(&self, iid: u64, comment: &str) -> Result<()> {
        self.post(&format!("issues/{iid}/comments"), json!({"body": comment}))?;
        Ok(())
    }
    fn get_issue_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        Ok(self
            .pages(&format!("issues/{iid}/comments"))?
            .iter()
            .map(plain_comment)
            .collect())
    }
    fn get_issue_thread_notes(&self, iid: u64) -> Result<Vec<IssueThreadNote>> {
        Ok(self
            .pages(&format!("issues/{iid}/comments"))?
            .iter()
            .map(|value| IssueThreadNote {
                id: value["id"].as_u64().unwrap_or_default(),
                body: string(value, "body"),
                system: false,
                discussion_id: None,
                author: IssueThreadNoteAuthor {
                    username: string(&value["user"], "login"),
                },
            })
            .collect())
    }
    fn list_merge_requests(&self) -> Result<Vec<MergeRequest>> {
        self.pages("pulls?state=open")?.iter().map(pull).collect()
    }
    fn get_merge_request(&self, iid: u64) -> Result<MergeRequest> {
        pull(&self.get(&format!("pulls/{iid}"))?)
    }
    fn find_open_mr_by_source_branch(&self, source_branch: &str) -> Result<Option<u64>> {
        Ok(self
            .list_merge_requests()?
            .into_iter()
            .find(|mr| mr.source_branch == source_branch)
            .map(|mr| mr.iid))
    }
    fn find_mrs_by_source_branch(&self, source_branch: &str) -> Result<Vec<u64>> {
        Ok(self
            .pages("pulls?state=all")?
            .iter()
            .map(pull)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|mr| mr.source_branch == source_branch)
            .map(|mr| mr.iid)
            .collect())
    }
    fn get_merge_request_changes(&self, iid: u64) -> Result<MergeRequestChangesSnapshot> {
        let pr = self.get(&format!("pulls/{iid}"))?;
        let files = self.pages(&format!("pulls/{iid}/files"))?;
        let mut names = Vec::new();
        let mut patch = String::new();
        let mut overflow = false;
        for file in files {
            let name = string(&file, "filename");
            let old = file["previous_filename"].as_str().unwrap_or(&name);
            if old == name {
                names.push(name.clone());
            } else {
                names.push(format!("{old} -> {name}"));
            }
            patch.push_str(&format!("diff --git a/{old} b/{name}\n"));
            if let Some(diff) = file["patch"].as_str() {
                patch.push_str(diff);
                patch.push('\n');
            } else {
                overflow = true;
            }
        }
        Ok(MergeRequestChangesSnapshot {
            files: names,
            patch,
            overflow,
            base_sha: pr["base"]["sha"].as_str().map(str::to_string),
            start_sha: pr["base"]["sha"].as_str().map(str::to_string),
            head_sha: pr["head"]["sha"].as_str().map(str::to_string),
        })
    }
    fn create_merge_request(
        &self,
        source_branch: &str,
        target_branch: &str,
        title: &str,
        description: &str,
    ) -> Result<u64> {
        self.post(
            "pulls",
            json!({"head": source_branch, "base": target_branch,
            "title": title, "body": description}),
        )?["number"]
            .as_u64()
            .context("created GitHub pull request has no number")
    }
    fn merge_mr(&self, iid: u64) -> Result<()> {
        self.put(&format!("pulls/{iid}/merge"), json!({}))?;
        Ok(())
    }
    fn close_mr(&self, iid: u64) -> Result<()> {
        self.patch(&format!("pulls/{iid}"), json!({"state": "closed"}))?;
        Ok(())
    }
    fn update_mr_title_description(
        &self,
        iid: u64,
        title: Option<&str>,
        description: Option<&str>,
    ) -> Result<()> {
        let mut body = serde_json::Map::new();
        if let Some(title) = title {
            body.insert("title".into(), json!(title));
        }
        if let Some(description) = description {
            body.insert("body".into(), json!(description));
        }
        if !body.is_empty() {
            self.patch(&format!("pulls/{iid}"), Value::Object(body))?;
        }
        Ok(())
    }
    fn add_mr_label_with_retries(&self, iid: u64, label: &str) -> Result<()> {
        self.add_label(iid, label)
    }
    fn remove_mr_label(&self, iid: u64, label: &str) -> Result<()> {
        self.remove_label(iid, label)
    }
    fn add_mr_comment(&self, iid: u64, comment: &str) -> Result<()> {
        self.add_issue_comment(iid, comment)
    }
    fn add_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        match self.post(
            &format!("pulls/{iid}/reviews"),
            json!({"body": body, "event": "REQUEST_CHANGES"}),
        ) {
            Ok(_) => Ok(()),
            // GitHub rejects a formal review by the PR author. The issue
            // comment still reaches the worker and preserves the feedback.
            Err(error) if error.to_string().contains("HTTP 422") => self.add_mr_comment(iid, body),
            Err(error) => Err(error),
        }
    }
    fn add_resolved_mr_discussion(&self, iid: u64, body: &str) -> Result<()> {
        match self.post(
            &format!("pulls/{iid}/reviews"),
            json!({"body": body, "event": "APPROVE"}),
        ) {
            Ok(_) => Ok(()),
            // GitHub does not allow a PR author to approve their own PR.
            Err(error) if error.to_string().contains("HTTP 422") => self.add_mr_comment(iid, body),
            Err(error) => Err(error),
        }
    }
    fn get_mr_comments(&self, iid: u64) -> Result<Vec<Comment>> {
        let mut comments = self.get_issue_comments(iid)?;
        for review in self.pages(&format!("pulls/{iid}/reviews"))? {
            let body = string(&review, "body");
            if body.is_empty() {
                continue;
            }
            let id = review["id"].as_u64().unwrap_or_default();
            comments.push(Comment {
                id: comment_id(id, review["submitted_at"].as_str().unwrap_or_default()),
                body,
                author: string(&review["user"], "login"),
                discussion_id: format!("review_{id}"),
                discussion_resolvable: false,
                location: None,
                location_details: None,
            });
        }
        for thread in self.review_threads(iid)? {
            let thread_id = string(&thread, "id");
            if let Some(nodes) = thread["comments"]["nodes"].as_array() {
                for node in nodes {
                    let id = node["fullDatabaseId"]
                        .as_u64()
                        .or_else(|| {
                            node["fullDatabaseId"]
                                .as_str()
                                .and_then(|id| id.parse().ok())
                        })
                        .unwrap_or_default();
                    let path = string(node, "path");
                    let line = node["line"]
                        .as_u64()
                        .or_else(|| node["originalLine"].as_u64());
                    comments.push(Comment {
                        id: comment_id(id, node["createdAt"].as_str().unwrap_or_default()),
                        body: string(node, "body"),
                        author: string(&node["author"], "login"),
                        discussion_id: thread_id.clone(),
                        discussion_resolvable: true,
                        location: line.map(|line| format!("{path}:{line}")),
                        location_details: None,
                    });
                }
            }
        }
        comments.sort_by_key(|comment| comment.id);
        Ok(comments)
    }
    fn get_unresolved_discussion_ids(&self, iid: u64) -> Result<Vec<String>> {
        Ok(self
            .review_threads(iid)?
            .iter()
            .filter(|thread| thread["isResolved"] == false)
            .map(|thread| string(thread, "id"))
            .collect())
    }
    fn get_unresolved_discussion_count(&self, iid: u64) -> Result<(usize, usize)> {
        let threads = self.review_threads(iid)?;
        let unresolved = threads
            .iter()
            .filter(|thread| thread["isResolved"] == false)
            .count();
        Ok((unresolved, threads.len()))
    }
    fn resolve_discussion(&self, _mr_iid: u64, discussion_id: &str) -> Result<()> {
        const QUERY: &str =
            "mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{id}}}";
        self.graphql(QUERY, json!({"id": discussion_id}))?;
        Ok(())
    }
    fn reply_to_discussion(&self, _mr_iid: u64, discussion_id: &str, body: &str) -> Result<()> {
        const QUERY: &str = "mutation($id:ID!,$body:String!){addPullRequestReviewThreadReply(input:{pullRequestReviewThreadId:$id,body:$body}){comment{id}}}";
        self.graphql(QUERY, json!({"id": discussion_id, "body": body}))?;
        Ok(())
    }
    fn is_not_found(&self, err: &anyhow::Error) -> bool {
        super::is_not_found(err)
    }
    fn get_issue_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        self.label_events(iid)
    }
    fn get_mr_label_events(&self, iid: u64) -> Result<Vec<ResourceLabelEvent>> {
        self.label_events(iid)
    }
}

#[cfg(test)]
mod tests {
    use super::{issue, parse_github_repo, pull};
    use serde_json::json;

    #[test]
    fn parses_ssh_and_https_repositories() {
        assert_eq!(
            parse_github_repo("git@github.com:team/project.git").unwrap(),
            ("team".into(), "project".into())
        );
        assert_eq!(
            parse_github_repo("https://github.com/team/project.git").unwrap(),
            ("team".into(), "project".into())
        );
        assert!(parse_github_repo("https://notgithub.com/team/project").is_err());
    }

    #[test]
    fn maps_github_states_and_labels() {
        let issue = issue(&json!({"number": 4, "title": "Bug", "body": null,
            "state": "open", "labels": [{"name": "priority::1"}]}))
        .unwrap();
        assert_eq!(issue.state, "opened");
        assert_eq!(issue.priority(), 1);
        let pr = pull(
            &json!({"number": 5, "state": "closed", "merged_at": "2026-01-01T00:00:00Z",
            "head": {"ref": "issue-4"}, "base": {"ref": "main"}, "labels": []}),
        )
        .unwrap();
        assert_eq!(pr.state, "merged");
    }
}
