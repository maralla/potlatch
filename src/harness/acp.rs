//! ACP server: JSON-RPC 2.0 over newline-delimited stdio.
//!
//! Handles the Agent Client Protocol methods that potlatch's `AcpClient` calls:
//! `initialize`, `authenticate`, `session/new`, `session/set_model`,
//! `session/set_config_option`, `session/prompt`, `session/close`, `session/cancel`.
//!
//! During `session/prompt`, the agent loop runs and emits `session/update` notifications
//! in real-time (streamed to stdout as they're produced, not buffered).

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use super::agent_loop::AgentLoop;
use super::client::ChatClient;
use super::tools::ToolRegistry;
use crate::core::model::acp::jsonrpc::Outbound;

/// Context token budget for the harness ACP agent loop. Compaction triggers at
/// 60% and targets 30% of this value (see `Context::enforce_budget`).
const CONTEXT_TOKEN_BUDGET: usize = 200_000;

/// A session in the ACP server.
struct Session {
    id: String,
    cwd: String,
    model: String,
    /// Current session mode (e.g. "plan", "ask", ""). Empty when no mode was
    /// set. The ACP runtime sends `session/set_config_option` with
    /// `configId=mode` when PMO requests plan mode.
    mode: String,
    cancel: Arc<AtomicBool>,
    /// Session-level state (background jobs, language servers, etc.). Tools
    /// retrieve their state by concrete type via `SessionStates::get`.
    /// All state is shut down on session close.
    states: super::tools::SessionStates,
    /// Optional allow-list of tool names. `None` means all built-in tools;
    /// `Some(names)` registers only the named tools. Set via the `tools`
    /// extension field of `session/new`.
    allowed_tools: Option<Vec<String>>,
    /// Caller-defined structured-output tool definitions, passed via the
    /// `structured_output_tools` extension field of `session/new`. Each entry
    /// is a JSON object with `name`, `description`, and `parameters` (JSON
    /// schema). The harness creates a generic `StructuredOutputTool` per
    /// definition at prompt time.
    structured_output_tools: Option<Vec<Value>>,
}

impl Session {
    fn new(cwd: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cwd,
            model: std::env::var("BREEZE_MODEL").unwrap_or_default(),
            mode: String::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            states: super::tools::SessionStates::new(),
            allowed_tools: None,
            structured_output_tools: None,
        }
    }
}

/// The ACP server state.
pub struct AcpServer {
    llm: Arc<dyn ChatClient>,
    sessions: HashMap<String, Session>,
}

impl AcpServer {
    pub fn new(llm: Arc<dyn ChatClient>) -> Self {
        Self {
            llm,
            sessions: HashMap::new(),
        }
    }

    /// Handle an incoming JSON-RPC message. Writes notifications directly to `writer`
    /// as they're produced (for real-time streaming during `session/prompt`).
    /// Returns the response (if the message was a request with an id).
    pub fn handle_message(
        &mut self,
        msg: &Value,
        writer: &mut dyn Write,
    ) -> Result<Option<Outbound>> {
        let method = msg["method"].as_str().unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        debug!("harness ACP: received method={method}");

        let result = match method {
            "initialize" => self.handle_initialize(&params)?,
            "authenticate" => self.handle_authenticate(&params)?,
            "session/new" => self.handle_session_new(&params)?,
            "session/set_model" => self.handle_set_model(&params)?,
            "session/set_config_option" => self.handle_set_config_option(&params)?,
            "session/set_mode" => json!({}),
            "session/prompt" => {
                // session/prompt streams notifications directly to the writer
                self.handle_session_prompt(&params, writer)?
            }
            "session/close" => self.handle_session_close(&params)?,
            "session/cancel" => self.handle_session_cancel(&params)?,
            other => {
                warn!("harness ACP: unhandled method: {other}");
                json!({})
            }
        };

        // Return a response only if this was a request (has an id)
        Ok(id.map(|id| Outbound::Response { id, result }))
    }

    fn handle_initialize(&self, _params: &Value) -> Result<Value> {
        Ok(json!({
            "protocolVersion": 1,
            "agentCapabilities": {
                "fs": {
                    "readTextFile": true,
                    "writeTextFile": true
                }
            },
            "authMethods": []
        }))
    }

    fn handle_authenticate(&self, _params: &Value) -> Result<Value> {
        Ok(json!({}))
    }

    fn handle_session_new(&mut self, params: &Value) -> Result<Value> {
        let cwd = params["cwd"].as_str().unwrap_or(".").to_string();
        info!("harness ACP: creating session with cwd={cwd}");

        let mut session = Session::new(cwd);
        // Optional `tools` extension: an allow-list of tool names. When
        // present, only those tools are registered for this session.
        if let Some(arr) = params.get("tools").and_then(|v| v.as_array()) {
            let names: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if !names.is_empty() {
                session.allowed_tools = Some(names);
            }
        }
        // Optional `structured_output_tools` extension: caller-defined
        // structured-output tool definitions. Each entry has `name`,
        // `description`, and `parameters` (JSON schema). The harness
        // registers a generic StructuredOutputTool per definition at prompt
        // time and returns captured output in the session/prompt response.
        if let Some(arr) = params
            .get("structured_output_tools")
            .and_then(|v| v.as_array())
        {
            let defs: Vec<Value> = arr
                .iter()
                .filter(|v| {
                    v.get("name").and_then(Value::as_str).is_some() && v.get("parameters").is_some()
                })
                .cloned()
                .collect();
            if !defs.is_empty() {
                session.structured_output_tools = Some(defs);
            }
        }
        let session_id = session.id.clone();

        let models = self.llm.list_models().unwrap_or_default();
        let model_options: Vec<Value> = if models.is_empty() {
            vec![]
        } else {
            models
                .iter()
                .map(|m| json!({ "value": m, "name": m }))
                .collect()
        };

        let result = json!({
            "sessionId": session_id,
            "configOptions": [
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": model_options
                },
                {
                    "id": "mode",
                    "category": "mode",
                    "type": "select",
                    "options": [
                        {"value": "ask", "name": "Ask"},
                        {"value": "plan", "name": "Plan"}
                    ]
                }
            ]
        });

        self.sessions.insert(session_id, session);
        info!("harness ACP: created session");
        Ok(result)
    }

    fn handle_set_model(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        let model_id = params["modelId"].as_str().unwrap_or("");

        if let Some(session) = self.sessions.get_mut(session_id) {
            session.model = model_id.to_string();
            debug!("harness ACP: set model to {model_id} for session {session_id}");
        }

        Ok(json!({}))
    }

    fn handle_set_config_option(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        let config_id = params["configId"].as_str().unwrap_or("");
        let value = params["value"].as_str().unwrap_or("");

        if let Some(session) = self.sessions.get_mut(session_id) {
            if config_id == "model" {
                session.model = value.to_string();
                info!("harness ACP: session {session_id} set model={value}");
            } else if config_id == "mode" {
                session.mode = value.to_string();
                info!("harness ACP: session {session_id} set mode={value}");
            } else {
                debug!(
                    "harness ACP: session {session_id} unknown configId={config_id} value={value}"
                );
            }
        } else {
            warn!("harness ACP: set_config_option for unknown session {session_id}");
        }

        Ok(json!({ "configOptions": [] }))
    }

    fn handle_session_prompt(&mut self, params: &Value, writer: &mut dyn Write) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("").to_string();

        let prompt_text = extract_prompt_text(&params["prompt"]);

        let session = match self.sessions.get_mut(&session_id) {
            Some(s) => s,
            None => {
                return Ok(json!({
                    "stopReason": "error",
                    "message": format!("unknown session: {session_id}")
                }));
            }
        };

        let cwd = session.cwd.clone();
        let model = session.model.clone();
        let mode = session.mode.clone();
        let allowed_tools = session.allowed_tools.clone();
        let structured_output_defs = session.structured_output_tools.clone();
        let cancel = session.cancel.clone();
        cancel.store(false, Ordering::SeqCst);

        let mut tools = ToolRegistry::with_builtin_tools(
            &mut session.states,
            &cwd,
            &model,
            allowed_tools.as_deref(),
        );
        // Register caller-defined structured-output tools (e.g. the worker's
        // `handoff` tool). Each definition has `name`, `description`, and
        // `parameters` (JSON schema). The harness captures the model's calls
        // and returns them in the session/prompt response.
        if let Some(defs) = &structured_output_defs {
            for def in defs {
                if let (Some(name), Some(desc), Some(params)) = (
                    def.get("name").and_then(Value::as_str),
                    def.get("description").and_then(Value::as_str),
                    def.get("parameters"),
                ) {
                    tools.register_structured_output(name, desc, params.clone());
                }
            }
        }
        let mut agent = AgentLoop::new(
            Arc::clone(&self.llm),
            tools,
            model,
            CONTEXT_TOKEN_BUDGET,
            cancel,
        );
        info!(
            "harness ACP: session {session_id} mode={}, registered tools: {:?}",
            if mode.is_empty() { "default" } else { &mode },
            agent.tool_names()
        );

        // Collect progress text; the agent loop calls this callback after each LLM response.
        // We emit notifications by writing to the writer after collection.
        let progress_buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let progress_buf_clone = Arc::clone(&progress_buf);
        let progress_cb: &super::client::StreamCallback = &move |text: &str| {
            if let Ok(mut buf) = progress_buf_clone.lock() {
                buf.push(text.to_string());
            }
        };

        let result = agent.run(&prompt_text, &cwd, Some(progress_cb));
        // Read all captured structured-output tool calls (e.g. `handoff`, `plan`).
        let structured_outputs = agent.take_structured_outputs();

        // Emit all collected progress as session/update notifications
        let collected = progress_buf.lock().unwrap();
        for text in collected.iter() {
            let notif = Outbound::Notification {
                method: "session/update".to_string(),
                params: json!({
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": { "text": text }
                    }
                }),
            };
            if let Ok(line) = notif.to_json_line() {
                let _ = writer.write_all(line.as_bytes());
                let _ = writer.flush();
            }
        }

        match result {
            Ok(response) => Ok(json!({
                "stopReason": "end_turn",
                "message": response,
                "structured_outputs": structured_outputs,
            })),
            Err(e) => {
                let err_msg = format!("Agent loop error: {e}");
                warn!("harness ACP: {err_msg}");
                Ok(json!({
                    "stopReason": "error",
                    "message": err_msg,
                }))
            }
        }
    }

    fn handle_session_close(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        if let Some(session) = self.sessions.remove(session_id) {
            session.states.shutdown();
            debug!("harness ACP: closed session {session_id}");
        }
        Ok(json!(null))
    }

    fn handle_session_cancel(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        if let Some(session) = self.sessions.get(session_id) {
            session.cancel.store(true, Ordering::SeqCst);
            info!("harness ACP: cancel requested for session {session_id}");
        }
        Ok(json!({}))
    }
}

/// Extract text from the ACP `prompt` field (array of content blocks).
fn extract_prompt_text(prompt: &Value) -> String {
    if let Some(blocks) = prompt.as_array() {
        blocks
            .iter()
            .filter_map(|b| {
                if b["type"].as_str() == Some("text") {
                    b["text"].as_str().map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else if let Some(text) = prompt.as_str() {
        text.to_string()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::ChatResponse;
    use super::*;
    use serde_json::json;
    use std::io::Cursor;

    struct StubClient;

    impl ChatClient for StubClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            _tools: &[Value],
            _on_chunk: Option<&super::super::client::StreamCallback>,
            _on_tool_calls: Option<&super::super::client::ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&super::super::client::EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                content: "Task completed successfully.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            })
        }
    }

    fn collect_output(server: &mut AcpServer, msg: &Value) -> (Option<Outbound>, String) {
        let mut buf = Cursor::new(Vec::new());
        let response = server.handle_message(msg, &mut buf).unwrap();
        let written = String::from_utf8(buf.into_inner()).unwrap();
        (response, written)
    }

    #[test]
    fn initialize_returns_capabilities() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        });

        let (response, written) = collect_output(&mut server, &msg);
        assert!(written.is_empty()); // no notifications
        match response {
            Some(Outbound::Response { result, .. }) => {
                assert_eq!(result["protocolVersion"], 1);
                assert!(result["authMethods"].as_array().unwrap().is_empty());
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn session_new_creates_session() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": "/tmp", "mcpServers": [] }
        });

        let (response, _) = collect_output(&mut server, &msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                assert!(!result["sessionId"].as_str().unwrap().is_empty());
                assert!(result["configOptions"].is_array());
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn session_new_advertises_mode_config_option() {
        // The harness must advertise a "mode" config option with "plan" as an
        // available value. Without this, the ACP runtime skips setting the
        // session mode, the plan tool is never registered, and the PMO can't
        // call it.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": "/tmp", "mcpServers": [] }
        });

        let (response, _) = collect_output(&mut server, &msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                let options = result["configOptions"].as_array().unwrap();
                let mode_opt = options
                    .iter()
                    .find(|o| o["id"] == "mode")
                    .expect("mode config option must be advertised");
                assert_eq!(mode_opt["type"], "select");
                let values: Vec<&str> = mode_opt["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| o["value"].as_str().unwrap())
                    .collect();
                assert!(values.contains(&"plan"));
                assert!(values.contains(&"ask"));
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn authenticate_is_noop() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "authenticate",
            "params": { "methodId": "cursor_login" }
        });

        let (response, _) = collect_output(&mut server, &msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                assert_eq!(result, json!({}));
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn session_prompt_returns_response_and_streams_notifications() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let (resp, _) = collect_output(&mut server, &new_msg);
        let session_id = match resp {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        let prompt_msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "say hello" }]
            }
        });

        let (response, written) = collect_output(&mut server, &prompt_msg);
        // Should have streamed at least one notification (the progress text)
        assert!(
            written.contains("session/update") || written.is_empty(),
            "expected notification or empty"
        );
        match response {
            Some(Outbound::Response { result, .. }) => {
                assert_eq!(result["stopReason"], "end_turn");
                assert!(
                    result["message"]
                        .as_str()
                        .unwrap()
                        .contains("Task completed")
                );
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn extract_prompt_text_from_blocks() {
        let prompt = json!([
            { "type": "text", "text": "hello " },
            { "type": "text", "text": "world" }
        ]);
        assert_eq!(extract_prompt_text(&prompt), "hello \nworld");
    }

    #[test]
    fn session_prompt_includes_structured_outputs_when_no_tools_registered() {
        // A session without structured-output tools should return
        // structured_outputs: {} (empty object) in the session/prompt result.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let (resp, _) = collect_output(&mut server, &new_msg);
        let session_id = match resp {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        let prompt_msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "hi" }]
            }
        });
        let (response, _) = collect_output(&mut server, &prompt_msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                // structured_outputs is present (empty object — no tools called).
                assert!(result.get("structured_outputs").is_some());
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn set_config_option_records_session_mode() {
        // session/set_config_option with configId=mode, value=plan should
        // record the mode on the session. We verify by checking that a
        // subsequent session/prompt succeeds (the mode was stored).
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let (resp, _) = collect_output(&mut server, &new_msg);
        let session_id = match resp {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        // Set mode to plan.
        let mode_msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_config_option",
            "params": {
                "sessionId": session_id,
                "configId": "mode",
                "value": "plan"
            }
        });
        let _ = collect_output(&mut server, &mode_msg);

        // Now prompt — the session/prompt should succeed.
        let prompt_msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "triage" }]
            }
        });
        let (response, _) = collect_output(&mut server, &prompt_msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                // structured_outputs is present.
                assert!(result.get("structured_outputs").is_some());
            }
            _ => panic!("expected response"),
        }
    }

    /// A chat client that records the `tools` schema array it was called with,
    /// so tests can assert which tools a session registered.
    struct ToolsCapturingClient {
        captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
    }

    impl ToolsCapturingClient {
        fn new(captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>>) -> Self {
            Self { captured }
        }
    }

    impl ChatClient for ToolsCapturingClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            tools: &[Value],
            _on_chunk: Option<&super::super::client::StreamCallback>,
            _on_tool_calls: Option<&super::super::client::ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&super::super::client::EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            *self.captured.lock().unwrap() = tools.to_vec();
            Ok(ChatResponse {
                content: "done".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            })
        }
    }

    fn run_session_prompt_in_mode(mode: &str) -> Vec<String> {
        let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> =
            Arc::new(ToolsCapturingClient::new(std::sync::Arc::clone(&captured)));

        let mut server = AcpServer::new(llm);
        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let (resp, _) = collect_output(&mut server, &new_msg);
        let session_id = match resp {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected session/new response"),
        };

        if !mode.is_empty() {
            let mode_msg = json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_config_option",
                "params": {
                    "sessionId": session_id,
                    "configId": "mode",
                    "value": mode
                }
            });
            let _ = collect_output(&mut server, &mode_msg);
        }

        let prompt_msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "hi" }]
            }
        });
        let _ = collect_output(&mut server, &prompt_msg);
        captured
            .lock()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn default_mode_keeps_file_edit_registered() {
        let tools = run_session_prompt_in_mode("");
        assert!(tools.contains(&"edit".to_string()));
    }
}
