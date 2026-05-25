//! PMO: answer Cursor ACP `cursor/ask_question` via a GitLab **thread reply** to the posted question.
//!
//! While this process is running, PMO blocks until someone replies on that thread (or timeout/shutdown).
//! After a restart there is no special recovery: the question and reply are ordinary issue comments
//! in GitLab. Once `pmo-pending` is cleared (e.g. after an answer in the live session), the issue is
//! triaged like any other.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use rand::RngExt;
use serde_json::Value;
use tracing::{info, warn};

use crate::agents::gitlab::{GitLabClient, IssueThreadNote};
use crate::core::model::acp::client::{
    CursorAskQuestionHandler, headless_cursor_ask_question_reply,
};

use super::labels;

const POLL_INTERVAL: Duration = Duration::from_secs(4);

fn new_ask_id() -> String {
    let mut r = rand::rng();
    format!("{:016x}{:016x}", r.random::<u64>(), r.random::<u64>())
}

fn ask_marker_snippet(ask_id: &str) -> String {
    format!("<!-- codepair-pmo-acp-ask:{ask_id} -->")
}

fn option_entry_id(opt: &Value) -> Option<&str> {
    opt.get("id")
        .or_else(|| opt.get("optionId"))
        .or_else(|| opt.get("option_id"))
        .and_then(|v| v.as_str())
}

fn option_entry_label(opt: &Value) -> String {
    for k in ["label", "title", "name", "text"] {
        if let Some(s) = opt.get(k).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return s.to_string();
        }
    }
    option_entry_id(opt).unwrap_or("(option)").to_string()
}

fn extract_question_text(params: &Value) -> String {
    for k in ["question", "message", "text", "prompt", "title", "body"] {
        if let Some(s) = params.get(k).and_then(|v| v.as_str()) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    params
        .as_object()
        .map(|o| {
            let preview: serde_json::Map<String, Value> = o
                .iter()
                .filter(|(key, _)| *key != "sessionId" && *key != "session_id")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            serde_json::to_string_pretty(&preview).unwrap_or_else(|_| "{}".to_string())
        })
        .filter(|s| s != "{}")
        .unwrap_or_else(|| "_(no question text in params)_".to_string())
}

fn format_options_for_comment(params: &Value) -> String {
    if let Some(opts) = params.get("options").and_then(|v| v.as_array()) {
        if opts.is_empty() {
            return "_Reply with an option id, or a number like `0` for the first choice._\n"
                .to_string();
        }
        let mut lines = Vec::new();
        for (i, o) in opts.iter().enumerate() {
            let id = option_entry_id(o).unwrap_or("");
            let label = option_entry_label(o);
            if id.is_empty() {
                lines.push(format!("{}. {}", i + 1, label));
            } else {
                lines.push(format!("{}. `{}` — {}", i + 1, id, label));
            }
        }
        return lines.join("\n");
    }
    if let Some(choices) = params.get("choices").and_then(|v| v.as_array()) {
        choices
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let owned = c.to_string();
                let t = c
                    .as_str()
                    .or_else(|| c.get("text").and_then(|v| v.as_str()))
                    .unwrap_or(owned.as_str());
                format!("{}. {}", i, t)
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        "_Reply with a number (e.g. `0`) or free text._\n".to_string()
    }
}

fn build_issue_comment(ask_id: &str, params: &Value) -> String {
    let q = extract_question_text(params);
    let opts = format_options_for_comment(params);
    let marker = ask_marker_snippet(ask_id);
    format!(
        "{marker}\n\n\
         **Cursor agent question**\n\n\
         {q}\n\n\
         **Choices**\n\n\
         {opts}\n\n\
         ---\n\n\
         **Reply to this comment** (use GitLab’s *Reply* on this note so your answer stays in this thread). \
         Your reply text is the answer — usually one line: an option id, a number (`0` = first choice), or a short answer.\n\n\
         The `{}` label is set until Codepair forwards your reply to the running agent.",
        labels::PMO_PENDING
    )
}

/// First non-empty line of the reply body (trimmed), or empty string if none.
fn reply_body_as_choice(body: &str) -> String {
    body.trim()
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

fn find_root_note<'a>(notes: &'a [IssueThreadNote], ask_id: &str) -> Option<&'a IssueThreadNote> {
    let needle = ask_marker_snippet(ask_id);
    notes.iter().find(|n| n.body.contains(&needle))
}

/// First **direct** reply in the same discussion as `root` (GitLab thread), excluding system notes and new ask posts.
fn pick_direct_thread_reply<'a>(
    notes: &'a [IssueThreadNote],
    root: &'a IssueThreadNote,
) -> Option<&'a IssueThreadNote> {
    let mut candidates: Vec<&IssueThreadNote> = notes
        .iter()
        .filter(|n| {
            if n.id == root.id || n.system {
                return false;
            }
            if n.body.contains("codepair-pmo-acp-ask:") {
                return false;
            }
            same_discussion_or_sequential_fallback(n, root)
        })
        .collect();
    candidates.sort_by_key(|n| n.id);
    candidates.into_iter().next()
}

fn same_discussion_or_sequential_fallback(n: &IssueThreadNote, root: &IssueThreadNote) -> bool {
    match (&root.discussion_id, &n.discussion_id) {
        (Some(rd), Some(nd)) => rd == nd,
        _ => n.id > root.id,
    }
}

fn resolve_choice_to_acp_result(params: &Value, choice_raw: &str) -> Value {
    let choice_trim = choice_raw.trim();
    if choice_trim.is_empty() {
        return headless_cursor_ask_question_reply(params);
    }
    if let Ok(idx) = choice_trim.parse::<usize>() {
        if let Some(opts) = params.get("options").and_then(|v| v.as_array()) {
            if let Some(opt) = opts.get(idx)
                && let Some(id) = option_entry_id(opt)
            {
                return serde_json::json!({ "selectedOptionId": id });
            }
            if let Some(opt) = idx.checked_sub(1).and_then(|j| opts.get(j))
                && let Some(id) = option_entry_id(opt)
            {
                return serde_json::json!({ "selectedOptionId": id });
            }
        }
        if params.get("choices").and_then(|v| v.as_array()).is_some() {
            return serde_json::json!({ "selectedIndex": idx });
        }
        return serde_json::json!({ "selectedIndex": idx });
    }
    if let Some(opts) = params.get("options").and_then(|v| v.as_array()) {
        for opt in opts {
            if let Some(id) = option_entry_id(opt)
                && id == choice_trim
            {
                return serde_json::json!({ "selectedOptionId": id });
            }
        }
    }
    warn!(
        target: "codepair::acp_cursor",
        choice = %choice_trim,
        "PMO thread reply did not match a listed option; echoing as selectedOptionId"
    );
    serde_json::json!({ "selectedOptionId": choice_trim })
}

fn clear_pmo_pending_label(gitlab: &GitLabClient, issue_iid: u64) {
    let _ = gitlab.remove_issue_label(issue_iid, labels::PMO_PENDING);
}

/// Posts the question note, then **blocks** until a **thread reply** arrives (same process only).
pub struct GitLabIssueCursorAskHandler {
    issue_iid: u64,
    gitlab: GitLabClient,
    shutdown: Arc<AtomicBool>,
    wait_deadline: Option<Instant>,
}

impl GitLabIssueCursorAskHandler {
    pub fn new(
        issue_iid: u64,
        gitlab: GitLabClient,
        shutdown: Arc<AtomicBool>,
        timeout: Option<Duration>,
    ) -> Self {
        let wait_deadline = timeout.map(|d| Instant::now() + d);
        Self {
            issue_iid,
            gitlab,
            shutdown,
            wait_deadline,
        }
    }

    fn finish_with_reply(&self, params_for_resolve: &Value, reply: &IssueThreadNote) -> Value {
        let choice = reply_body_as_choice(&reply.body);
        let result = resolve_choice_to_acp_result(params_for_resolve, &choice);
        clear_pmo_pending_label(&self.gitlab, self.issue_iid);
        info!(
            target: "codepair::acp_cursor",
            issue_iid = self.issue_iid,
            note_id = reply.id,
            author = %reply.author_username(),
            "Using direct thread reply as cursor/ask_question answer"
        );
        result
    }
}

impl CursorAskQuestionHandler for GitLabIssueCursorAskHandler {
    fn handle_ask_question(&self, params: &Value) -> Value {
        let params_for_resolve = params.clone();
        let ask_id = new_ask_id();
        let comment = build_issue_comment(&ask_id, params);

        if let Err(e) = self.gitlab.add_issue_comment(self.issue_iid, &comment) {
            warn!(
                target: "codepair::acp_cursor",
                err = %e,
                "failed to post ask_question to GitLab; using headless fallback"
            );
            return headless_cursor_ask_question_reply(params);
        }

        if let Err(e) = self
            .gitlab
            .add_issue_label(self.issue_iid, labels::PMO_PENDING)
        {
            warn!(
                target: "codepair::acp_cursor",
                err = %e,
                "failed to add pmo-pending for ask_question"
            );
        }

        let notes = match self.gitlab.get_issue_thread_notes(self.issue_iid) {
            Ok(n) => n,
            Err(e) => {
                warn!(
                    target: "codepair::acp_cursor",
                    err = %e,
                    "failed to list thread notes after posting ask_question"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return headless_cursor_ask_question_reply(params);
            }
        };

        let Some(root) = find_root_note(&notes, &ask_id) else {
            warn!(
                target: "codepair::acp_cursor",
                ask_id = %ask_id,
                "could not find posted ask note by marker; using headless fallback"
            );
            clear_pmo_pending_label(&self.gitlab, self.issue_iid);
            return headless_cursor_ask_question_reply(params);
        };

        let root_note_id = root.id;

        info!(
            target: "codepair::acp_cursor",
            issue_iid = self.issue_iid,
            root_note_id = root.id,
            ask_id = %ask_id,
            "Posted cursor/ask_question; waiting for direct thread reply"
        );
        eprintln!(
            "codepair PMO: Posted question on issue #{} — **Reply to that GitLab comment** (thread). Waiting…",
            self.issue_iid
        );

        loop {
            if let Some(dl) = self.wait_deadline
                && Instant::now() >= dl
            {
                eprintln!(
                    "codepair PMO: GitLab thread wait timed out on issue #{}; using automatic choice.",
                    self.issue_iid
                );
                let _ = self.gitlab.add_issue_comment(
                    self.issue_iid,
                    "**PMO:** Timed out waiting for a **reply** to the Cursor question comment; proceeding with an automatic choice.",
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return headless_cursor_ask_question_reply(&params_for_resolve);
            }

            if self.shutdown.load(Ordering::SeqCst) {
                warn!(
                    target: "codepair::acp_cursor",
                    "shutdown during ask_question wait; headless fallback"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return headless_cursor_ask_question_reply(&params_for_resolve);
            }

            thread::sleep(POLL_INTERVAL);

            let notes = match self.gitlab.get_issue_thread_notes(self.issue_iid) {
                Ok(n) => n,
                Err(e) => {
                    warn!(target: "codepair::acp_cursor", err = %e, "poll thread notes failed");
                    continue;
                }
            };

            let Some(root) = notes.iter().find(|n| n.id == root_note_id) else {
                warn!(
                    target: "codepair::acp_cursor",
                    root_note_id,
                    "root note disappeared during wait"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return headless_cursor_ask_question_reply(&params_for_resolve);
            };

            if let Some(reply) = pick_direct_thread_reply(&notes, root) {
                let choice = reply_body_as_choice(&reply.body);
                if choice.is_empty() {
                    continue;
                }
                return self.finish_with_reply(&params_for_resolve, reply);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: u64, body: &str, system: bool, disc: Option<&str>) -> IssueThreadNote {
        let raw = format!(
            r#"{{"id":{id},"body":{},"system":{system},"discussion_id":{},"author":{{"username":"u"}}}}"#,
            serde_json::to_string(body).unwrap(),
            serde_json::to_string(&disc).unwrap()
        );
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn pick_reply_same_discussion() {
        let root = n(10, "<!-- codepair-pmo-acp-ask:abc -->", false, Some("d1"));
        let r1 = n(11, "opt-b", false, Some("d1"));
        let noise = n(9, "old", false, Some("d2"));
        let notes = vec![noise, root.clone(), r1.clone()];
        let root_ref = notes.iter().find(|x| x.id == 10).unwrap();
        let got = pick_direct_thread_reply(&notes, root_ref).unwrap();
        assert_eq!(got.id, 11);
        assert_eq!(reply_body_as_choice(&got.body), "opt-b");
    }

    #[test]
    fn pick_first_reply_when_sorted() {
        let root = n(5, "<!-- codepair-pmo-acp-ask:x -->", false, None);
        let r1 = n(6, "0", false, None);
        let notes = vec![root.clone(), r1.clone()];
        let root_ref = &notes[0];
        let got = pick_direct_thread_reply(&notes, root_ref).unwrap();
        assert_eq!(got.id, 6);
    }

    #[test]
    fn ignores_new_ask_in_thread() {
        let root = n(1, "<!-- codepair-pmo-acp-ask:a -->", false, Some("d"));
        let bad = n(2, "<!-- codepair-pmo-acp-ask:b -->", false, Some("d"));
        let notes = vec![root.clone(), bad];
        let root_ref = &notes[0];
        assert!(pick_direct_thread_reply(&notes, root_ref).is_none());
    }
}
