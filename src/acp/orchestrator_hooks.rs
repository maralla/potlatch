//! [`AcpHooks`] that records streaming assistant text from `session/update` notifications.
//!
//! Also parses [slash commands](https://agentclientprotocol.com/protocol/slash-commands) from
//! `available_commands_update` into [`StreamTextHooks::available_slash_command_names`] (not yet
//! consumed when building prompts; for future role-specific logic).
//!
//! Tracks [session modes](https://agentclientprotocol.com/protocol/session-modes) from
//! `session/new` (`modes`) and `current_mode_update` notifications (for future `session/set_mode`).
//!
//! ## Plans (standard ACP vs Cursor)
//!
//! - Standard [`session/update` plan entries](https://agentclientprotocol.com/protocol/agent-plan)
//!   are notifications only; the client replaces its view of the plan and the turn continues until
//!   `session/prompt` completes ([prompt turn](https://agentclientprotocol.com/protocol/prompt-turn)).
//! - Cursor’s CLI additionally sends **extension RPCs** such as [`cursor/create_plan` and
//!   `cursor/ask_question`](https://cursor.com/docs/cli/acp) that expect a client response; a
//!   headless client must answer or plan mode can block waiting for approval.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tracing::{debug, warn};

use super::client::{AcpHooks, CursorAskQuestionHandler, headless_agent_request_result};
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
}

impl StreamTextHooks {
    pub fn new() -> Self {
        Self {
            buffer: Mutex::new(String::new()),
            available_slash_command_names: Mutex::new(Vec::new()),
            session_modes: Mutex::new(None),
            cursor_ask_question_handler: Mutex::new(None),
            workspace_root: None,
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
        }
    }

    fn handle_fs_read_text_file(&self, params: &Value) -> Value {
        let Some(ref root) = self.workspace_root else {
            warn!(
                target: "codepair::acp_fs",
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
            warn!(target: "codepair::acp_fs", "fs/read_text_file missing path");
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
                    target: "codepair::acp_fs",
                    path = %path,
                    err = %e,
                    "fs/read_text_file failed"
                );
                serde_json::json!({ "content": format!("# (codepair could not read file: {e})\n") })
            }
        }
    }

    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
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

    /// Seed from `session/new` result [`crate::acp::types::NewSessionResult::modes`].
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
}

#[cfg(test)]
impl StreamTextHooks {
    fn slash_command_names_snapshot(&self) -> Vec<String> {
        self.available_slash_command_names.lock().unwrap().clone()
    }

    fn session_modes_snapshot(&self) -> Option<SessionModeStateBrief> {
        self.session_modes.lock().unwrap().clone()
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

/// Parses `session/update` when the agent is in **plan** mode: structured todo entries, not
/// `agent_message_chunk`. Without this, plan-only turns leave an empty handoff `response`.
pub fn extract_plan_update_text(params: &Value) -> Option<String> {
    let update = params.get("update")?;
    let kind = update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())?;
    if !kind.eq_ignore_ascii_case("plan") {
        return None;
    }
    let entries = update.get("entries").and_then(|e| e.as_array())?;
    if entries.is_empty() {
        return None;
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push("=== Plan update (from agent plan mode) ===".into());
    for e in entries {
        let text = e
            .get("content")
            .or_else(|| e.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if text.is_empty() {
            continue;
        }
        let status = e
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let priority = e
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let status_pri = match (status.is_empty(), priority.is_empty()) {
            (true, true) => String::new(),
            (false, true) => format!("[{status}] "),
            (true, false) => format!("[{priority}] "),
            (false, false) => format!("[{status}; {priority}] "),
        };
        lines.push(format!("- {status_pri}{text}"));
    }
    if lines.len() <= 1 {
        return None;
    }
    Some(lines.join("\n"))
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
        if method == "fs/read_text_file" {
            return self.handle_fs_read_text_file(params);
        }

        headless_agent_request_result(method, params)
    }

    fn on_agent_notification(&self, method: &str, params: &Value) {
        if method != "session/update" {
            return;
        }
        if let Some(names) = extract_available_slash_command_names(params) {
            debug!(
                target: "codepair::acp_slash",
                "available_commands_update: {:?}",
                names
            );
            *self.available_slash_command_names.lock().unwrap() = names;
            return;
        }
        if let Some(mode_id) = extract_current_mode_update(params) {
            debug!(
                target: "codepair::acp_modes",
                "current_mode_update: {}",
                mode_id
            );
            self.apply_current_mode_update(mode_id);
            return;
        }
        if let Some(t) = extract_plan_update_text(params) {
            let mut buf = self.buffer.lock().unwrap();
            if !buf.is_empty() && !buf.ends_with('\n') {
                buf.push('\n');
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&t);
            buf.push('\n');
            return;
        }
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
        use crate::acp::types::{SessionModeEntry, SessionModeStateBrief};

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
    fn extracts_plan_update_entries() {
        let params = json!({
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "plan",
                "entries": [
                    { "content": "Read context", "priority": "medium", "status": "pending" },
                    { "content": "Ship decision", "priority": "high", "status": "pending" }
                ]
            }
        });
        let t = extract_plan_update_text(&params).expect("plan text");
        assert!(t.contains("Plan update"));
        assert!(t.contains("Read context"));
        assert!(t.contains("Ship decision"));
    }

    #[test]
    fn hooks_append_plan_updates_to_buffer() {
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
        assert!(out.contains("Step one"), "{}", out);
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
}
