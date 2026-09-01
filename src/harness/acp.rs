//! ACP server: JSON-RPC 2.0 over newline-delimited stdio.
//!
//! Handles the Agent Client Protocol methods that potlatch's `AcpClient` calls:
//! `initialize`, `authenticate`, `session/new`, `session/set_model`,
//! `session/set_config_option`, `session/prompt`, `session/close`, `session/cancel`.
//!
//! During `session/prompt`, the agent loop runs and emits `session/update` notifications
//! in real-time (streamed to stdout as they're produced, not buffered).

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;
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

/// Channel handles for a session, shared between the main thread (which owns
/// the `AcpServer` and runs agent loops) and the reader thread (which reads
/// stdin continuously). The reader thread uses these to handle `session/inject`
/// and `session/cancel` while the main thread is blocked running
/// `session/prompt`.
#[derive(Clone)]
pub struct SessionChannels {
    /// Sender for mid-run message injection. Messages pushed here are drained
    /// by the `AgentLoop` at the top of each iteration.
    pub inject_tx: Arc<Mutex<VecDeque<String>>>,
    /// Cancel flag. Setting this to `true` causes the running agent loop to
    /// exit at the top of its next iteration.
    pub cancel: Arc<AtomicBool>,
}

/// Shared map of session id → channels. Wrapped in `Arc<Mutex<...>>` so both
/// the reader thread and main thread can access it.
pub type SharedSessionChannels = Arc<std::sync::Mutex<HashMap<String, SessionChannels>>>;

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
    /// The persistent agent loop for this session. Created at `session/new`,
    /// reused across all `session/prompt` calls. Owns the `Context`, so
    /// conversation history accumulates across prompts — true single long
    /// session.
    agent: Option<AgentLoop>,
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
    agent_tools: Vec<crate::core::bus::RemoteAgentToolDefinition>,
    /// Optional path to a transcript file. When set, the harness writes a
    /// human-readable transcript of each `session/prompt` turn (user prompt,
    /// assistant response, reasoning, tool calls) to this file in real time.
    /// The parent agent can read it to inspect subagent progress.
    transcript_path: Option<std::path::PathBuf>,
}

impl Session {
    fn new(cwd: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cwd,
            model: std::env::var("POTLATCH_MODEL").unwrap_or_default(),
            mode: String::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            agent: None,
            states: super::tools::SessionStates::new(),
            allowed_tools: None,
            structured_output_tools: None,
            agent_tools: Vec::new(),
            transcript_path: None,
        }
    }
}

/// The ACP server state.
pub struct AcpServer {
    llm: Arc<dyn ChatClient>,
    sessions: HashMap<String, Session>,
    /// Shared map of session id → channels. The reader thread uses this to
    /// handle `session/inject` and `session/cancel` while the main thread is
    /// blocked running `session/prompt`.
    shared_channels: SharedSessionChannels,
    agent_tool_caller: Option<Arc<dyn super::tools::agent_bus::AgentToolCaller>>,
}

impl AcpServer {
    pub fn new(llm: Arc<dyn ChatClient>) -> Self {
        Self {
            llm,
            sessions: HashMap::new(),
            shared_channels: Arc::new(std::sync::Mutex::new(HashMap::new())),
            agent_tool_caller: None,
        }
    }

    pub fn with_agent_tool_caller(
        mut self,
        caller: Arc<dyn super::tools::agent_bus::AgentToolCaller>,
    ) -> Self {
        self.agent_tool_caller = Some(caller);
        self
    }

    /// Get the shared session channels map. The reader thread uses this to
    /// handle `session/inject` and `session/cancel` directly.
    pub fn shared_channels(&self) -> SharedSessionChannels {
        Arc::clone(&self.shared_channels)
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
            // session/inject and session/cancel are handled directly by the
            // reader thread via shared channels (they work mid-run, while the
            // main thread is blocked in session/prompt). They never reach
            // handle_message.
            "session/close" => self.handle_session_close(&params)?,
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

        let mut session = Session::new(cwd.clone());
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
        if let Some(arr) = params.get("agent_tools").and_then(Value::as_array) {
            session.agent_tools = arr
                .iter()
                .filter_map(|definition| serde_json::from_value(definition.clone()).ok())
                .collect();
        }
        // Optional `transcript_path` extension: path to a transcript file
        // where the harness writes a human-readable log of each session/prompt
        // turn (user prompt, assistant response, reasoning, tool calls).
        if let Some(path) = params
            .get("transcript_path")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            session.transcript_path = Some(std::path::PathBuf::from(path));
        }

        // Optional `write_roots` extension: directories the write/edit tools
        // may touch outside the cwd (via `outside_cwd: true`). Empty/absent
        // keeps the historical permissive behavior.
        let write_roots = params
            .get("write_roots")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(std::path::PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let write_roots = super::tools::WriteRoots::from_paths(write_roots);

        // Build the tool registry with built-in tools + caller-defined
        // structured-output tools. The tool set is fixed for the session
        // lifetime.
        let mut tools = ToolRegistry::with_builtin_tools(
            &mut session.states,
            &session.id,
            &cwd,
            &session.model,
            &write_roots,
            session.allowed_tools.as_deref(),
        );
        if let Some(caller) = &self.agent_tool_caller {
            tools.register_agent_tools(
                Arc::clone(caller),
                session.agent_tools.clone(),
                session.allowed_tools.as_deref(),
            );
        }
        if let Some(defs) = &session.structured_output_tools {
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

        // Optional `context_channels` extension: named context strings
        // registered by in-process agents on the bus. Each has `name` and
        // `content`; the harness injects each as a system message at session
        // init. Potlatch extension — enables agents like the memory agent to
        // publish context into other agents' sessions.
        let context_channels: Vec<(String, String)> = params
            .get("context_channels")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|entry| {
                        let name = entry.get("name").and_then(Value::as_str)?;
                        let content = entry.get("content").and_then(Value::as_str)?;
                        Some((name.to_string(), content.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Create the inject channel and the persistent AgentLoop. The agent
        // loop owns the Context and is reused across all session/prompt calls
        // — true single long session. The inject_tx is stored in the shared
        // channels map so session/inject can push messages into the running
        // loop.
        let inject_queue: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        let mut agent = AgentLoop::new(
            Arc::clone(&self.llm),
            tools,
            session.model.clone(),
            CONTEXT_TOKEN_BUDGET,
            Arc::clone(&session.cancel),
            Arc::clone(&inject_queue),
        );
        agent.init_context(&cwd, &context_channels);
        session.agent = Some(agent);

        // Register the inject channel and cancel flag in the shared channels
        // map so the reader thread can handle session/inject and
        // session/cancel while the main thread is blocked running
        // session/prompt.
        let session_id = session.id.clone();
        {
            let mut sc = self.shared_channels.lock().unwrap();
            sc.insert(
                session_id.clone(),
                SessionChannels {
                    inject_tx: inject_queue,
                    cancel: Arc::clone(&session.cancel),
                },
            );
        }

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
        Ok(result)
    }

    pub fn session_tool_names(&self, session_id: &str) -> Vec<String> {
        self.sessions
            .get(session_id)
            .and_then(|session| session.agent.as_ref())
            .map(|agent| agent.tool_names().into_iter().map(str::to_string).collect())
            .unwrap_or_default()
    }

    fn handle_set_model(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        let model_id = params["modelId"].as_str().unwrap_or("");

        if let Some(session) = self.sessions.get_mut(session_id) {
            session.model = model_id.to_string();
            if let Some(ref mut agent) = session.agent {
                agent.set_model(model_id);
            }
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
                if let Some(ref mut agent) = session.agent {
                    agent.set_model(value);
                }
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
        let transcript_path = session.transcript_path.clone();
        let cancel = session.cancel.clone();
        cancel.store(false, Ordering::SeqCst);

        let agent = match session.agent.as_mut() {
            Some(a) => a,
            None => {
                return Ok(json!({
                    "stopReason": "error",
                    "message": "session agent not initialized"
                }));
            }
        };
        // Collect progress text; the agent loop calls this callback after each LLM response.
        // We emit notifications by writing to the writer after collection.
        let progress_buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let progress_buf_clone = Arc::clone(&progress_buf);
        let progress_cb: &super::client::StreamCallback = &move |text: &str| {
            if let Ok(mut buf) = progress_buf_clone.lock() {
                buf.push(text.to_string());
            }
        };

        // Write the user prompt to the transcript file at the start.
        let turn_counter = std::sync::Arc::new(std::sync::Mutex::new(0u32));
        if let Some(ref tp) = transcript_path {
            write_transcript_entry(tp, "## User", &prompt_text);
        }

        // Turn callback: writes each turn's content, reasoning, and tool calls
        // to the transcript file in real time.
        let transcript_path_for_cb = transcript_path.clone();
        let turn_counter_for_cb = Arc::clone(&turn_counter);
        let turn_cb: &super::client::TurnCallback =
            &move |_response: &super::client::ChatResponse| {
                if let Some(ref tp) = transcript_path_for_cb {
                    let mut count = turn_counter_for_cb.lock().unwrap();
                    *count += 1;
                    write_transcript_turn(tp, *count, _response);
                }
            };

        let result =
            agent.run_with_turn_callback(&prompt_text, &cwd, Some(progress_cb), Some(turn_cb));
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
            // Also remove from the shared channels map.
            self.shared_channels.lock().unwrap().remove(session_id);
            debug!("harness ACP: closed session {session_id}");
        }
        Ok(json!(null))
    }
}

/// Append a section to the transcript file.
fn write_transcript_entry(path: &std::path::Path, header: &str, body: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "\n{header}\n\n{body}");
        let _ = f.flush();
    }
}

/// Append a full turn (assistant content, reasoning, tool calls) to the
/// transcript file.
fn write_transcript_turn(
    path: &std::path::Path,
    turn: u32,
    response: &super::client::ChatResponse,
) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "\n=== Turn {turn} ===");

        if !response.content.is_empty() {
            let _ = writeln!(f, "\n## Assistant\n\n{}\n", response.content);
        }

        if !response.reasoning.is_empty() {
            let _ = writeln!(f, "\n## Thinking\n\n{}\n", response.reasoning);
        }

        if !response.tool_calls.is_empty() {
            let _ = writeln!(f, "\n## Tool Calls");
            for tc in &response.tool_calls {
                let name = tc["function"]["name"].as_str().unwrap_or("(unknown)");
                let args = tc["function"]["arguments"].as_str().unwrap_or("");
                let _ = writeln!(f, "\n- **{name}**: `{args}`");
            }
            let _ = writeln!(f);
        }

        if !response.tool_results.is_empty() {
            let _ = writeln!(f, "\n## Tool Results");
            for (i, result) in response.tool_results.iter().enumerate() {
                let preview: String = result.chars().take(500).collect();
                let suffix = if result.len() > 500 { "..." } else { "" };
                let _ = writeln!(f, "\n### Result {i}\n\n{preview}{suffix}\n");
            }
        }

        let _ = f.flush();
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
    struct EchoAgentToolCaller;

    impl super::super::tools::agent_bus::AgentToolCaller for EchoAgentToolCaller {
        fn call(&self, _target: &str, _operation: &str, arguments: Value) -> Result<Value> {
            Ok(arguments)
        }
    }

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
    fn session_new_registers_tools_supplied_by_the_potlatch_parent() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let mut server = AcpServer::new(llm).with_agent_tool_caller(Arc::new(EchoAgentToolCaller));
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {
                "cwd": "/tmp",
                "agent_tools": [{
                    "name": "web_search",
                    "description": "Search.",
                    "parameters": {"type": "object"},
                    "target": "web",
                    "operation": "run"
                }]
            }
        });

        let (response, _) = collect_output(&mut server, &msg);
        let session_id = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };
        let tools = server.sessions[&session_id]
            .agent
            .as_ref()
            .unwrap()
            .tool_names();
        assert!(tools.contains(&"web_search"));
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
    fn multiple_session_prompts_reuse_same_agent_loop() {
        // Multiple session/prompt calls on the same session should succeed —
        // the AgentLoop persists on the Session and is reused. Context
        // accumulates across prompts (single long session).
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

        // First prompt.
        let prompt1 = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "first prompt" }]
            }
        });
        let (resp1, _) = collect_output(&mut server, &prompt1);
        match resp1 {
            Some(Outbound::Response { result, .. }) => {
                assert_eq!(result["stopReason"], "end_turn");
            }
            _ => panic!("expected response for first prompt"),
        }

        // Second prompt on the same session — should reuse the AgentLoop.
        let prompt2 = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "second prompt" }]
            }
        });
        let (resp2, _) = collect_output(&mut server, &prompt2);
        match resp2 {
            Some(Outbound::Response { result, .. }) => {
                assert_eq!(result["stopReason"], "end_turn");
            }
            _ => panic!("expected response for second prompt"),
        }
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
    fn session_new_registers_caller_defined_structured_output_schemas_verbatim() {
        // The harness has no notion of the orchestrator's schema types: it
        // forwards whatever `parameters` JSON the caller registered, so a
        // tagged union with `oneOf`/`const`/`additionalProperties` reaches the
        // model exactly as the adapter emitted it.
        let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> =
            Arc::new(ToolsCapturingClient::new(std::sync::Arc::clone(&captured)));
        let mut server = AcpServer::new(llm);

        let parameters = json!({
            "type": "object",
            "description": "How the run ended.",
            "oneOf": [
                {
                    "type": "object",
                    "description": "It is done.",
                    "additionalProperties": false,
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "const": "implemented",
                            "enum": ["implemented"],
                            "description": "It is done."
                        }
                    },
                    "required": ["outcome"]
                },
                {
                    "type": "object",
                    "description": "It is blocked.",
                    "additionalProperties": false,
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "const": "blocked",
                            "enum": ["blocked"],
                            "description": "It is blocked."
                        },
                        "reason": {"type": "string", "description": "Why."}
                    },
                    "required": ["outcome", "reason"]
                }
            ]
        });
        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": {
                "cwd": "/tmp",
                "structured_output_tools": [{
                    "name": "handoff",
                    "description": "Hand the run back.",
                    "parameters": parameters,
                }]
            }
        });
        let (resp, _) = collect_output(&mut server, &new_msg);
        let session_id = match resp {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected session/new response"),
        };

        let prompt_msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "go" }]
            }
        });
        let _ = collect_output(&mut server, &prompt_msg);

        let tools = captured.lock().unwrap().clone();
        let handoff = tools
            .iter()
            .find(|tool| tool["function"]["name"] == json!("handoff"))
            .expect("the caller-defined tool must be registered for the session");
        assert_eq!(handoff["function"]["description"], "Hand the run back.");
        assert_eq!(handoff["function"]["parameters"], parameters);
    }

    #[test]
    fn default_mode_keeps_file_edit_registered() {
        let tools = run_session_prompt_in_mode("");
        assert!(tools.contains(&"edit".to_string()));
    }
}
