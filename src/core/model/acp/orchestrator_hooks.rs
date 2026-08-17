//! [`AcpHooks`] that records streaming assistant text from `session/update` notifications.
//!
//! Also parses [slash commands](https://agentclientprotocol.com/protocol/slash-commands) from
//! `available_commands_update` into [`StreamTextHooks::available_slash_command_names`] (not yet
//! consumed when building prompts; for future role-specific logic).
//!
//! Vendor extension state (e.g. plan-mode tracking, ask-question dispatch) is
//! installed via [`StreamTextHooks::set_vendor_state`]. Generic code never
//! references vendor-specific names.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tracing::{debug, warn};

use super::backends::AcpVendorState;
use super::client::{AcpHooks, headless_agent_request_result};
use super::workspace_read::{read_text_file_under_workspace, slice_by_line_range};

/// Accumulates `agent_message_chunk` text; auto-approves tool permissions.
/// Vendor extension state is optional — installed by the runtime when a
/// vendor extension is active.
pub struct StreamTextHooks {
    buffer: Mutex<String>,
    /// Command `name` fields from the latest `available_commands_update` (session notification).
    available_slash_command_names: Mutex<Vec<String>>,
    /// Git clone root for ACP [`fs/read_text_file`](https://agentclientprotocol.com/protocol/file-system.md).
    workspace_root: Option<PathBuf>,
    /// Vendor extension state (plan paths, create_plan text, session modes, ask dispatch).
    /// `None` when no vendor extension is active.
    vendor_state: Mutex<Option<Arc<dyn AcpVendorState>>>,
}

impl StreamTextHooks {
    pub fn new() -> Self {
        Self {
            buffer: Mutex::new(String::new()),
            available_slash_command_names: Mutex::new(Vec::new()),
            workspace_root: None,
            vendor_state: Mutex::new(None),
        }
    }

    /// Hooks that can serve [`fs/read_text_file`] for paths under this directory (the agent repo root).
    pub fn with_workspace(workspace_root: PathBuf) -> Self {
        Self {
            buffer: Mutex::new(String::new()),
            available_slash_command_names: Mutex::new(Vec::new()),
            workspace_root: Some(workspace_root),
            vendor_state: Mutex::new(None),
        }
    }

    /// Install the vendor extension state (called by the runtime when a
    /// vendor extension is active).
    pub fn set_vendor_state(&self, state: Option<Arc<dyn AcpVendorState>>) {
        *self.vendor_state.lock().unwrap() = state;
    }

    /// Snapshot the vendor state Arc (for runtime use).
    pub(crate) fn vendor_state_snapshot(&self) -> Option<Arc<dyn AcpVendorState>> {
        self.vendor_state.lock().unwrap().clone()
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
        if let Some(ref vs) = *self.vendor_state.lock().unwrap() {
            vs.clear();
        }
    }

    pub fn take_text(&self) -> String {
        std::mem::take(&mut *self.buffer.lock().unwrap())
    }

    /// Seed session modes from `session/new` result.
    pub(crate) fn seed_session_modes(&self, modes: &super::types::SessionModeStateBrief) {
        if let Some(ref vs) = *self.vendor_state.lock().unwrap() {
            vs.seed_session_modes(modes);
        }
    }
}

#[cfg(test)]
impl StreamTextHooks {
    fn slash_command_names_snapshot(&self) -> Vec<String> {
        self.available_slash_command_names.lock().unwrap().clone()
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

impl AcpHooks for StreamTextHooks {
    fn handle_agent_request(&self, method: &str, params: &Value, id: &Value) -> Value {
        // Try the vendor extension state first — it handles vendor-specific
        // requests (e.g. create_plan, ask_question, update_todos).
        if let Some(ref vs) = *self.vendor_state.lock().unwrap()
            && let Some(result) = vs.handle_agent_request(method, params, id)
        {
            return result;
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
                target: "potlatch::acp_slash",
                "available_commands_update: {:?}",
                names
            );
            *self.available_slash_command_names.lock().unwrap() = names;
            return;
        }

        // Let the vendor state handle the notification (mode updates, plan
        // path recording, etc.). Returns true if consumed exclusively.
        if let Some(ref vs) = *self.vendor_state.lock().unwrap()
            && vs.on_session_update(params)
        {
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
}
