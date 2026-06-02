//! [`AcpHooks`] that records streaming assistant text from `session/update` notifications.
//!
//! Also parses [slash commands](https://agentclientprotocol.com/protocol/slash-commands) from
//! `available_commands_update` into [`StreamTextHooks::available_slash_command_names`] (not yet
//! consumed when building prompts; for future role-specific logic).
//!
//! Tracks [session modes](https://agentclientprotocol.com/protocol/session-modes) from
//! `session/new` (`modes`) and `current_mode_update` notifications (for future `session/set_mode`).
//!
//! ## Plans (Cursor plan mode)
//!
//! - Cursor’s CLI sends **extension RPCs** such as [`cursor/create_plan` and
//!   `cursor/ask_question`](https://cursor.com/docs/cli/acp) that expect a client response; a
//!   headless client must answer or plan mode can block waiting for approval.
//! - In **plan** mode, `session/update` may carry `tool_call_update` text such as
//!   `Plan saved to file://…`. Potlatch records those absolute paths on [`AgentHandoff::cursor_plan_paths`];
//!   the PMO role reads the files from the git workspace and merges their contents into the text it
//!   parses (split markers, sub-issues, etc.).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tracing::{debug, info, warn};

use super::client::{
    AcpHooks, CursorAskQuestionHandler, extract_cursor_create_plan_text,
    headless_agent_request_result, headless_cursor_create_plan_reply,
};
use super::types::SessionModeStateBrief;
use super::workspace_read::{read_text_file_under_workspace, slice_by_line_range};

/// Accumulates `agent_message_chunk` text; auto-approves tool permissions.
pub struct StreamTextHooks {
    buffer: Mutex<String>,
    /// Command `name` fields from the latest `available_commands_update` (session notification).
    available_slash_command_names: Mutex<Vec<String>>,
    /// Latest mode state: seeded from `session/new`, `current_mode_id` updated from notifications.
    session_modes: Mutex<Option<SessionModeStateBrief>>,
    /// When set, [`cursor/ask_question`](https://cursor.com/docs/cli/acp) is delegated here instead of [`headless_agent_request_result`].
    cursor_ask_question_handler: Mutex<Option<Arc<dyn CursorAskQuestionHandler>>>,
    /// Git clone root for ACP [`fs/read_text_file`](https://agentclientprotocol.com/protocol/file-system.md).
    workspace_root: Option<PathBuf>,
    /// Plan-mode `tool_call_update` paths (`Plan saved to file://…`), taken into [`AgentHandoff`] for PMO.
    cursor_plan_paths: Mutex<Vec<String>>,
    /// Plan-mode `cursor/create_plan` markdown body (ACP extension RPC).
    cursor_create_plan_text: Mutex<String>,
    /// Monotonic count of `session/update` notifications seen for this task.
    notification_seq: AtomicU64,
}

impl StreamTextHooks {
    pub fn new() -> Self {
        Self {
            buffer: Mutex::new(String::new()),
            available_slash_command_names: Mutex::new(Vec::new()),
            session_modes: Mutex::new(None),
            cursor_ask_question_handler: Mutex::new(None),
            workspace_root: None,
            cursor_plan_paths: Mutex::new(Vec::new()),
            cursor_create_plan_text: Mutex::new(String::new()),
            notification_seq: AtomicU64::new(0),
        }
    }

    /// Hooks that can serve [`fs/read_text_file`] for paths under this directory (the agent repo root).
    pub fn with_workspace(workspace_root: PathBuf) -> Self {
        Self {
            buffer: Mutex::new(String::new()),
            available_slash_command_names: Mutex::new(Vec::new()),
            session_modes: Mutex::new(None),
            cursor_ask_question_handler: Mutex::new(None),
            workspace_root: Some(workspace_root),
            cursor_plan_paths: Mutex::new(Vec::new()),
            cursor_create_plan_text: Mutex::new(String::new()),
            notification_seq: AtomicU64::new(0),
        }
    }

    fn handle_fs_read_text_file(&self, params: &Value) -> Value {
        let Some(ref root) = self.workspace_root else {
            warn!(
                target: "potlatch::acp_fs",
                "fs/read_text_file requested but no workspace_root configured"
            );
            return serde_json::json!({ "content": "" });
        };
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if path.is_empty() {
            warn!(target: "potlatch::acp_fs", "fs/read_text_file missing path");
            return serde_json::json!({ "content": "" });
        }
        let line = params.get("line").and_then(|v| v.as_u64());
        let limit = params.get("limit").and_then(|v| v.as_u64());
        match read_text_file_under_workspace(root, path) {
            Ok(mut text) => {
                if line.is_some() || limit.is_some() {
                    text = slice_by_line_range(&text, line, limit);
                }
                serde_json::json!({ "content": text })
            }
            Err(e) => {
                warn!(
                    target: "potlatch::acp_fs",
                    path = %path,
                    err = %e,
                    "fs/read_text_file failed"
                );
                serde_json::json!({ "content": format!("# (potlatch could not read file: {e})\n") })
            }
        }
    }

    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
        self.cursor_plan_paths.lock().unwrap().clear();
        self.cursor_create_plan_text.lock().unwrap().clear();
        self.notification_seq.store(0, Ordering::SeqCst);
    }

    /// Install or clear a [`CursorAskQuestionHandler`] for `cursor/ask_question`.
    pub fn set_cursor_ask_question_handler(
        &self,
        handler: Option<Arc<dyn CursorAskQuestionHandler>>,
    ) {
        *self.cursor_ask_question_handler.lock().unwrap() = handler;
    }

    pub fn take_text(&self) -> String {
        std::mem::take(&mut *self.buffer.lock().unwrap())
    }

    pub(crate) fn take_cursor_plan_paths(&self) -> Vec<String> {
        std::mem::take(&mut *self.cursor_plan_paths.lock().unwrap())
    }

    pub(crate) fn take_cursor_create_plan_text(&self) -> String {
        std::mem::take(&mut *self.cursor_create_plan_text.lock().unwrap())
    }

    pub fn has_cursor_plan_paths(&self) -> bool {
        !self.cursor_plan_paths.lock().unwrap().is_empty()
    }

    pub fn notification_seq(&self) -> u64 {
        self.notification_seq.load(Ordering::SeqCst)
    }

    /// Seed from `session/new` result [`super::types::NewSessionResult::modes`].
    pub(crate) fn seed_session_modes(&self, state: &SessionModeStateBrief) {
        *self.session_modes.lock().unwrap() = Some(state.clone());
    }

    fn apply_current_mode_update(&self, mode_id: String) {
        let mut g = self.session_modes.lock().unwrap();
        match g.as_mut() {
            Some(s) => {
                s.current_mode_id = mode_id;
            }
            None => {
                *g = Some(SessionModeStateBrief {
                    current_mode_id: mode_id,
                    available_modes: Vec::new(),
                });
            }
        }
    }

    /// After a successful client-initiated mode change (`session/set_mode` or `set_config_option`).
    pub(crate) fn sync_tracked_current_mode(&self, mode_id: &str) {
        self.apply_current_mode_update(mode_id.to_string());
    }

    fn session_current_mode_is_plan(&self) -> bool {
        self.session_modes
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|s| s.current_mode_id.eq_ignore_ascii_case("plan"))
    }

    /// When the session is in **plan** mode, record absolute paths from `tool_call_update` text
    /// (`Plan saved to file://…`). PMO reads these files after the prompt completes.
    fn record_cursor_saved_plan_paths(&self, params: &Value) {
        if !self.session_current_mode_is_plan() {
            return;
        }

        let Some(update) = params.get("update") else {
            return;
        };

        let paths = plan_saved_paths_from_tool_call_update(update);
        if paths.is_empty() {
            return;
        }

        let mut g = self.cursor_plan_paths.lock().unwrap();
        for p in paths {
            if !g.contains(&p) {
                g.push(p);
            }
        }
    }
}

#[cfg(test)]
impl StreamTextHooks {
    fn slash_command_names_snapshot(&self) -> Vec<String> {
        self.available_slash_command_names.lock().unwrap().clone()
    }

    fn session_modes_snapshot(&self) -> Option<SessionModeStateBrief> {
        self.session_modes.lock().unwrap().clone()
    }

    fn cursor_plan_paths_snapshot(&self) -> Vec<String> {
        self.cursor_plan_paths.lock().unwrap().clone()
    }
}

impl Default for StreamTextHooks {
    fn default() -> Self {
        Self::new()
    }
}

/// Best-effort extraction of streamed assistant text from `session/update` params.
pub fn extract_agent_message_chunk_text(params: &Value) -> Option<String> {
    let update = params.get("update")?;
    let kind = update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())?;
    if kind != "agent_message_chunk" && kind != "agentMessageChunk" {
        return None;
    }
    update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
}

/// Parses `session/update` params for [`available_commands_update`](https://agentclientprotocol.com/protocol/slash-commands).
pub fn extract_available_slash_command_names(params: &Value) -> Option<Vec<String>> {
    let update = params.get("update")?;
    let kind = update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())?;
    if kind != "available_commands_update" && kind != "availableCommandsUpdate" {
        return None;
    }
    let cmds = update
        .get("availableCommands")
        .or_else(|| update.get("available_commands"))?;
    let arr = cmds.as_array()?;
    let names: Vec<String> = arr
        .iter()
        .filter_map(|c| {
            c.get("name")
                .and_then(|n| n.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .collect();
    Some(names)
}

/// Decodes `%HH` sequences in a `file:` path segment (UTF-8, lossy on invalid sequences).
fn percent_decode_uri_path(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let h = &b[i + 1..i + 3];
            if h[0].is_ascii_hexdigit()
                && h[1].is_ascii_hexdigit()
                && let Ok(v) = u8::from_str_radix(std::str::from_utf8(h).unwrap_or(""), 16)
            {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Converts a `file:` URI to an absolute path string suitable for [`read_text_file_under_workspace`].
fn file_uri_to_absolute_path_str(uri: &str) -> Option<String> {
    let rest = uri.strip_prefix("file://")?;
    let path_part = if rest.starts_with('/') {
        rest
    } else {
        rest.find('/').map(|i| &rest[i..])?
    };
    if path_part.is_empty() {
        return None;
    }
    Some(percent_decode_uri_path(path_part))
}

/// Collects every `text` string nested under a `tool_call_update` payload (Cursor plan mode).
fn collect_tool_call_update_texts(update: &Value, out: &mut Vec<String>) {
    match update {
        Value::Object(map) => {
            if let Some(Value::String(t)) = map.get("text")
                && !t.is_empty()
            {
                out.push(t.clone());
            }
            for child in map.values() {
                collect_tool_call_update_texts(child, out);
            }
        }
        Value::Array(arr) => {
            for x in arr {
                collect_tool_call_update_texts(x, out);
            }
        }
        _ => {}
    }
}

fn session_update_kind(update: &Value) -> Option<&str> {
    update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())
}

fn is_tool_call_update(update: &Value) -> bool {
    session_update_kind(update).is_some_and(|k| {
        let norm: String = k
            .chars()
            .filter(|c| *c != '_' && *c != '-')
            .flat_map(|c| c.to_lowercase())
            .collect();
        norm == "toolcallupdate"
    })
}

/// Returns absolute filesystem paths from "Plan saved to file://…" lines (deduped).
fn plan_saved_paths_from_tool_call_update(update: &Value) -> Vec<String> {
    if !is_tool_call_update(update) {
        return Vec::new();
    }

    let mut texts = Vec::new();
    collect_tool_call_update_texts(update, &mut texts);

    let mut seen = HashSet::<String>::new();
    let mut paths = Vec::new();

    for t in texts {
        let lower = t.to_ascii_lowercase();
        if !lower.contains("plan saved") || !t.contains("file://") {
            continue;
        }

        let Some(start) = t.find("file://") else {
            continue;
        };

        let tail: String = t[start..]
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '"' && *c != ')')
            .collect();

        if let Some(p) = file_uri_to_absolute_path_str(&tail)
            && seen.insert(p.clone())
        {
            paths.push(p);
        }
    }

    paths
}

/// Parses `session/update` for [`current_mode_update`](https://agentclientprotocol.com/protocol/session-modes).
pub fn extract_current_mode_update(params: &Value) -> Option<String> {
    let update = params.get("update")?;
    let kind = update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())?;
    if kind != "current_mode_update" && kind != "currentModeUpdate" {
        return None;
    }
    update
        .get("modeId")
        .or_else(|| update.get("mode_id"))
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)
}

impl AcpHooks for StreamTextHooks {
    fn handle_agent_request(&self, method: &str, params: &Value, _id: &Value) -> Value {
        if method == "cursor/ask_question"
            && let Some(h) = self.cursor_ask_question_handler.lock().unwrap().clone()
        {
            return h.handle_ask_question(params);
        }
        if method == "cursor/create_plan" {
            if let Some(plan) = extract_cursor_create_plan_text(params) {
                info!(
                    target: "potlatch::acp_cursor",
                    plan_len = plan.len(),
                    "Captured cursor/create_plan markdown for PMO parsing"
                );
                *self.cursor_create_plan_text.lock().unwrap() = plan;
            }
            return headless_cursor_create_plan_reply();
        }
        if method.starts_with("cursor/") {
            debug!(
                target: "potlatch::acp_cursor",
                %method,
                "handling Cursor ACP extension request"
            );
        }
        if method == "fs/read_text_file" {
            return self.handle_fs_read_text_file(params);
        }

        headless_agent_request_result(method, params)
    }

    fn on_agent_notification(&self, method: &str, params: &Value) {
        if method != "session/update" {
            return;
        }
        self.notification_seq.fetch_add(1, Ordering::SeqCst);

        if let Some(names) = extract_available_slash_command_names(params) {
            debug!(
                target: "potlatch::acp_slash",
                "available_commands_update: {:?}",
                names
            );
            *self.available_slash_command_names.lock().unwrap() = names;
            return;
        }

        if let Some(mode_id) = extract_current_mode_update(params) {
            debug!(
                target: "potlatch::acp_modes",
                "current_mode_update: {}",
                mode_id
            );
            self.apply_current_mode_update(mode_id);
            return;
        }

        self.record_cursor_saved_plan_paths(params);

        if let Some(t) = extract_agent_message_chunk_text(params) {
            self.buffer.lock().unwrap().push_str(&t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hooks_capture_cursor_create_plan_markdown() {
        let h = StreamTextHooks::new();
        let params = json!({
            "toolCallId": "call_1",
            "name": "PMO triage",
            "plan": "GUIDE_WORKER\nINSTRUCTIONS:\nUse the existing loader."
        });
        let result = h.handle_agent_request("cursor/create_plan", &params, &json!(1));
        assert_eq!(result["outcome"]["outcome"], "accepted");
        assert!(h.take_cursor_create_plan_text().contains("GUIDE_WORKER"));
    }

    #[test]
    fn extracts_snake_case_chunk_kind() {
        let params = json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "hello" }
            }
        });
        assert_eq!(
            extract_agent_message_chunk_text(&params).as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn ignores_non_chunk_updates() {
        let params = json!({
            "update": { "sessionUpdate": "tool_call", "content": {} }
        });
        assert_eq!(extract_agent_message_chunk_text(&params), None);
    }

    #[test]
    fn extracts_available_slash_command_names_camel_case() {
        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [
                    { "name": "web", "description": "Search" },
                    { "name": "plan", "description": "Plan" }
                ]
            }
        });
        assert_eq!(
            extract_available_slash_command_names(&params),
            Some(vec!["web".to_string(), "plan".to_string()])
        );
    }

    #[test]
    fn hooks_record_slash_commands_without_touching_text_buffer() {
        let h = StreamTextHooks::new();
        let params = json!({
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [{ "name": "test" }]
            }
        });
        h.on_agent_notification("session/update", &params);
        assert_eq!(h.slash_command_names_snapshot(), vec!["test".to_string()]);
        assert!(h.take_text().is_empty());
    }

    #[test]
    fn extracts_current_mode_update() {
        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "current_mode_update",
                "modeId": "code"
            }
        });
        assert_eq!(
            extract_current_mode_update(&params).as_deref(),
            Some("code")
        );
    }

    #[test]
    fn hooks_apply_current_mode_update_preserves_available_list() {
        use super::super::types::{SessionModeEntry, SessionModeStateBrief};

        let h = StreamTextHooks::new();
        h.seed_session_modes(&SessionModeStateBrief {
            current_mode_id: "ask".into(),
            available_modes: vec![
                SessionModeEntry {
                    id: "ask".into(),
                    name: None,
                    description: None,
                },
                SessionModeEntry {
                    id: "code".into(),
                    name: None,
                    description: None,
                },
            ],
        });
        let params = json!({
            "update": {
                "sessionUpdate": "current_mode_update",
                "modeId": "code"
            }
        });
        h.on_agent_notification("session/update", &params);
        let snap = h.session_modes_snapshot().unwrap();
        assert_eq!(snap.current_mode_id, "code");
        assert_eq!(snap.available_modes.len(), 2);
        assert!(h.take_text().is_empty());
    }

    #[test]
    fn hooks_do_not_record_acp_plan_entries_in_buffer() {
        let h = StreamTextHooks::new();
        let plan = json!({
            "update": {
                "sessionUpdate": "plan",
                "entries": [
                    { "content": "Read context", "priority": "medium", "status": "pending" },
                    { "content": "Ship decision", "priority": "high", "status": "pending" }
                ]
            }
        });
        h.on_agent_notification("session/update", &plan);
        assert!(h.take_text().is_empty());
    }

    #[test]
    fn hooks_chunk_after_plan_entries_only_keeps_chunk_text() {
        let h = StreamTextHooks::new();
        let plan = json!({
            "update": {
                "sessionUpdate": "plan",
                "entries": [{ "content": "Step one", "status": "pending", "priority": "low" }]
            }
        });
        h.on_agent_notification("session/update", &plan);
        let chunk = json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "DECISION: SPLIT\n" }
            }
        });
        h.on_agent_notification("session/update", &chunk);
        let out = h.take_text();
        assert!(!out.contains("Step one"), "{}", out);
        assert!(out.contains("DECISION"));
    }

    #[test]
    fn hooks_accumulate_then_take() {
        let h = StreamTextHooks::new();
        let params = json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "a" }
            }
        });
        h.on_agent_notification("session/update", &params);
        h.on_agent_notification("session/update", &params);
        assert_eq!(h.take_text(), "aa");
        assert!(h.take_text().is_empty());
    }

    #[test]
    fn hooks_notification_seq_increments_on_session_updates() {
        let h = StreamTextHooks::new();
        assert_eq!(h.notification_seq(), 0);
        let params = json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "a" }
            }
        });
        h.on_agent_notification("session/update", &params);
        h.on_agent_notification("session/update", &params);
        assert_eq!(h.notification_seq(), 2);
        h.clear();
        assert_eq!(h.notification_seq(), 0);
    }

    #[test]
    fn plan_mode_tool_call_update_records_plan_file_path() {
        use super::super::types::{SessionModeEntry, SessionModeStateBrief};
        use std::fs;

        let tmp = std::env::temp_dir().join(format!("potlatch-plan-file-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join(".cursor/plans")).unwrap();
        let plan_path = tmp.join(".cursor/plans/PMO.plan.md");
        fs::write(&plan_path, "DECISION: SPLIT\nSUB_ISSUE_1 TITLE: A\n").unwrap();
        let ws = fs::canonicalize(&tmp).unwrap();
        let file_url = format!("file://{}", plan_path.display());

        let h = StreamTextHooks::with_workspace(ws);
        h.seed_session_modes(&SessionModeStateBrief {
            current_mode_id: "plan".into(),
            available_modes: vec![SessionModeEntry {
                id: "plan".into(),
                name: None,
                description: None,
            }],
        });
        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tool_x",
                "status": "in_progress",
                "content": [{
                    "type": "content",
                    "content": { "type": "text", "text": format!("Plan saved to {file_url}") }
                }]
            }
        });
        h.on_agent_notification("session/update", &params);
        assert!(h.take_text().is_empty());
        let paths = h.cursor_plan_paths_snapshot();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].ends_with("PMO.plan.md"), "{paths:?}");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn plan_file_injection_skipped_when_not_in_plan_mode() {
        use super::super::types::{SessionModeEntry, SessionModeStateBrief};
        use std::fs;

        let tmp = std::env::temp_dir().join(format!("potlatch-plan-skip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join(".cursor/plans")).unwrap();
        let plan_path = tmp.join(".cursor/plans/x.plan.md");
        fs::write(&plan_path, "SECRET").unwrap();
        let ws = fs::canonicalize(&tmp).unwrap();
        let file_url = format!("file://{}", plan_path.display());

        let h = StreamTextHooks::with_workspace(ws);
        h.seed_session_modes(&SessionModeStateBrief {
            current_mode_id: "code".into(),
            available_modes: vec![SessionModeEntry {
                id: "code".into(),
                name: None,
                description: None,
            }],
        });
        let params = json!({
            "update": {
                "sessionUpdate": "tool_call_update",
                "content": [{
                    "type": "content",
                    "content": { "type": "text", "text": format!("Plan saved to {file_url}") }
                }]
            }
        });
        h.on_agent_notification("session/update", &params);
        assert!(h.take_text().is_empty());
        assert!(h.cursor_plan_paths_snapshot().is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn percent_encoded_file_uri_decodes_for_plan_read() {
        use super::super::types::{SessionModeEntry, SessionModeStateBrief};
        use std::fs;

        let tmp = std::env::temp_dir().join(format!("potlatch-plan-pct-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join(".cursor/plans")).unwrap();
        let plan_path = tmp.join(".cursor/plans/PMO Issue.plan.md");
        fs::write(&plan_path, "FROM_ENCODED_PATH").unwrap();
        let ws = fs::canonicalize(&tmp).unwrap();
        let enc = ws.join(".cursor/plans/PMO%20Issue.plan.md");
        let file_url = format!("file://{}", enc.display());

        let h = StreamTextHooks::with_workspace(ws);
        h.seed_session_modes(&SessionModeStateBrief {
            current_mode_id: "plan".into(),
            available_modes: vec![SessionModeEntry {
                id: "plan".into(),
                name: None,
                description: None,
            }],
        });
        let params = json!({
            "update": {
                "sessionUpdate": "toolCallUpdate",
                "content": [{
                    "type": "content",
                    "content": { "type": "text", "text": format!("Plan saved to {file_url}") }
                }]
            }
        });
        h.on_agent_notification("session/update", &params);
        assert!(h.take_text().is_empty());
        let paths = h.cursor_plan_paths_snapshot();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].contains("PMO Issue.plan.md"), "{paths:?}");
        let _ = fs::remove_dir_all(&tmp);
    }
}
