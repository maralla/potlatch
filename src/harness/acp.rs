//! ACP server: JSON-RPC 2.0 over newline-delimited stdio.
//!
//! Handles the Agent Client Protocol methods that potlatch's `AcpClient` calls:
//! `initialize`, `authenticate`, `session/new`, `session/set_model`,
//! `session/set_config_option`, `session/prompt`, `session/close`, `session/cancel`.
//!
//! During `session/prompt`, the agent loop runs and emits `session/update` notifications
//! in real-time (streamed to stdout as they're produced, not buffered).

use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use super::agent_loop::AgentLoop;
use super::client::{ChatClient, ChatResponse, StreamCallback, TurnCallback};
use super::parent::SharedOutput;
use super::session_store::{
    SessionRoots, agent_current_marker, clear_current_session, read_current_session,
    session_context_file, write_current_session,
};
use super::tools::ToolRegistry;
use super::tools::agent_bus::AgentToolCaller;
use crate::core::bus::RemoteAgentToolDefinition;
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
pub type SharedSessionChannels = Arc<Mutex<HashMap<String, SessionChannels>>>;

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
    /// session. Shared with the session's turn threads: the mutex serializes
    /// prompts to the same session while different sessions run concurrently.
    agent: Option<Arc<Mutex<AgentLoop>>>,
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
    agent_tools: Vec<RemoteAgentToolDefinition>,
    /// Optional path to a transcript file. When set, the harness writes a
    /// human-readable transcript of each `session/prompt` turn (user prompt,
    /// assistant response, reasoning, tool calls) to this file in real time.
    /// The parent agent can read it to inspect subagent progress.
    transcript_path: Option<PathBuf>,
    /// The orchestrator agent this session runs for (e.g. `worker-7`), from
    /// the `agent_id` extension of `session/new`. Drives the
    /// `<working-dir>/.potlatch/agents/<agent-id>/current` marker: written
    /// when this
    /// session starts, emptied when it finishes, so a recovered process
    /// resumes only interrupted sessions. Empty when the caller passes none —
    /// the session then has no marker and cannot be resumed.
    agent_id: String,
    /// Path to this session's persisted context file. `None` when the
    /// session has no sessions-directory identity (agent_id empty): a fresh
    /// context, never persisted, never resumed.
    context_path: Option<PathBuf>,
}

impl Session {
    fn new(cwd: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cwd,
            model: env::var("POTLATCH_MODEL").unwrap_or_default(),
            mode: String::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            agent: None,
            states: super::tools::SessionStates::new(),
            allowed_tools: None,
            structured_output_tools: None,
            agent_tools: Vec::new(),
            transcript_path: None,
            agent_id: String::new(),
            context_path: None,
        }
    }
}

/// The ACP server state.
pub struct AcpServer {
    llm: Arc<dyn ChatClient>,
    sessions: HashMap<String, Session>,
    /// Shared map of session id → channels. The reader thread uses this to
    /// handle `session/inject` and `session/cancel` while turn threads run.
    shared_channels: SharedSessionChannels,
    agent_tool_caller: Option<Arc<dyn AgentToolCaller>>,
    /// Where session data (run.log, context) and agent markers live.
    /// Overridable so tests never touch the real `~/.potlatch`.
    roots: SessionRoots,
    /// Shared stdout of the harness process. Turn threads write their
    /// notifications and responses here when they finish; the main thread
    /// writes synchronous responses. The internal mutex keeps concurrent
    /// writes line-atomic.
    output: SharedOutput,
}

impl AcpServer {
    /// Run the server writing to an explicit shared output (tests, and the
    /// `harness` command, which passes the process stdout).
    pub fn with_output(llm: Arc<dyn ChatClient>, output: SharedOutput) -> Self {
        Self {
            llm,
            sessions: HashMap::new(),
            shared_channels: Arc::new(Mutex::new(HashMap::new())),
            agent_tool_caller: None,
            roots: SessionRoots::real(),
            output,
        }
    }

    pub fn with_agent_tool_caller(mut self, caller: Arc<dyn AgentToolCaller>) -> Self {
        self.agent_tool_caller = Some(caller);
        self
    }

    /// Get the shared session channels map. The reader thread uses this to
    /// handle `session/inject` and `session/cancel` directly.
    pub fn shared_channels(&self) -> SharedSessionChannels {
        Arc::clone(&self.shared_channels)
    }

    /// Handle an incoming JSON-RPC message. Returns the immediate response
    /// (if the message was a request with an id that is answered
    /// synchronously). `session/prompt` turns run on their own thread and
    /// write their notifications and response to the server's shared output
    /// when the turn completes, so the caller receives `Ok(None)` for them.
    pub fn handle_message(&mut self, msg: &Value) -> Result<Option<Outbound>> {
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
            // The turn runs asynchronously: the turn thread writes the
            // notifications and the response to the shared output when it
            // finishes, so nothing is returned here. Sessions are independent
            // — prompts to different sessions run concurrently; prompts to
            // the same session queue behind the session's agent loop.
            "session/prompt" => {
                self.handle_session_prompt(&params, id)?;
                return Ok(None);
            }
            // session/inject and session/cancel are handled directly by the
            // reader thread via shared channels (they work mid-run). They
            // never reach handle_message.
            "session/close" => self.handle_session_close(&params)?,
            other => {
                warn!("harness ACP: unhandled method: {other}");
                json!({})
            }
        };

        // Return a response only if this was a request (has an id). The
        // server writes it to the shared output itself — turn threads write
        // their own responses there too, and the shared output's mutex keeps
        // concurrent lines from interleaving.
        let response = id.map(|id| Outbound::Response { id, result });
        if let Some(outbound) = &response
            && let Ok(line) = outbound.to_json_line()
        {
            write_line(&self.output, &line);
        }
        Ok(response)
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

        // Guarantee: a session/new on this connection replaces any session
        // that is still registered. The orchestrator closes the previous
        // task's session best-effort before rotating; when that close fails
        // (transport blip — the runtime logs and continues), the old session
        // would otherwise stay in `sessions` forever with its full agent
        // context and job table: nothing else ever evicts it, and the child
        // process outlives thousands of tasks. Running the close path here
        // bounds the leak to one stale session, reclaimed by the very
        // `session/new` that follows the failed close.
        //
        // Multiplexing clients (the subagent agent keeps many independent
        // subagent sessions alive on one connection) opt out via the
        // `multiplex` extension: they manage their sessions' lifetimes
        // explicitly through `session/close`.
        let multiplex = params
            .get("multiplex")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !multiplex {
            self.close_all_registered_sessions("superseded by session/new");
        }

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
            session.transcript_path = Some(PathBuf::from(path));
        }

        // Optional `agent_id` extension: the orchestrator agent this session
        // runs for (e.g. `worker-7`). It locates the session in
        // `<working-dir>/.potlatch/agents/<agent-id>/current` so a
        // recovered process can
        // find the interrupted session and reuse its persisted context.
        session.agent_id = params
            .get("agent_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_default();

        // A previous run of this agent may have crashed mid-task: its marker
        // still names the session it never finished. Adopt that session's id —
        // and with it the run.log/context directory — so this run resumes it;
        // the marker then names this run. Nothing to resume (no agent_id, no
        // marker, or a marker pointing at a session whose context is gone)
        // keeps the fresh id and starts an empty context.
        let sessions_root = self.roots.sessions.clone();
        if !session.agent_id.is_empty() {
            let marker = agent_current_marker(&self.roots.agents, &session.agent_id);
            if let Some(prev_id) = read_current_session(&marker)
                && prev_id != session.id
                && session_context_file(&sessions_root, &prev_id).is_file()
            {
                info!(
                    "harness ACP: agent {} resuming interrupted session {prev_id} \
                     (context recovered from disk)",
                    session.agent_id
                );
                session.id = prev_id;
            }
            session.context_path = Some(session_context_file(&sessions_root, &session.id));
            write_current_session(&marker, &session.id);
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
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let write_roots = super::tools::WriteRoots::from_paths(write_roots);

        // Build the tool registry with built-in tools + caller-defined
        // structured-output tools. The tool set is fixed for the session
        // lifetime.
        let mut tools = ToolRegistry::with_builtin_tools(
            &mut session.states,
            &cwd,
            &write_roots,
            session.allowed_tools.as_deref(),
        );
        if let Some(caller) = &self.agent_tool_caller {
            tools.register_agent_tools(
                Arc::clone(caller),
                session.agent_tools.clone(),
                session.allowed_tools.as_deref(),
                &session.id,
            );
        }
        if let Some(defs) = &session.structured_output_tools {
            for def in defs {
                if let (Some(name), Some(desc), Some(params)) = (
                    def.get("name").and_then(Value::as_str),
                    def.get("description").and_then(Value::as_str),
                    def.get("parameters"),
                ) {
                    let terminal = def
                        .get("terminal")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    tools.register_structured_output(name, desc, params.clone(), terminal);
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
        // loop. A resumed session restores its context from the previous
        // run's snapshot; a fresh one initializes from the system prompt and
        // context channels.
        let inject_queue: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        let mut agent = AgentLoop::new(
            Arc::clone(&self.llm),
            tools,
            session.model.clone(),
            CONTEXT_TOKEN_BUDGET,
            Arc::clone(&session.cancel),
            Arc::clone(&inject_queue),
        );
        if let Some(context_path) = session.context_path.clone() {
            if self.llm.preserves_reasoning() {
                agent.set_keep_reasoning(true);
            }
            agent.set_context_path(context_path);
            // restore_context falls back to init_context itself when no
            // transcript restores; initing again here would double the
            // system prompt.
            agent.restore_context(&cwd, &context_channels);
        } else {
            agent.init_context(&cwd, &context_channels);
        }
        session.agent = Some(Arc::new(Mutex::new(agent)));

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
            .map(|agent| {
                agent
                    .lock()
                    .unwrap()
                    .tool_names()
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn handle_set_model(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        let model_id = params["modelId"].as_str().unwrap_or("");

        if let Some(session) = self.sessions.get_mut(session_id) {
            session.model = model_id.to_string();
            if let Some(ref agent) = session.agent {
                agent.lock().unwrap().set_model(model_id);
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
                if let Some(ref agent) = session.agent {
                    agent.lock().unwrap().set_model(value);
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

    fn handle_session_prompt(&mut self, params: &Value, request_id: Option<Value>) -> Result<()> {
        let session_id = params["sessionId"].as_str().unwrap_or("").to_string();
        let prompt_text = extract_prompt_text(&params["prompt"]);

        // Gather everything the turn thread needs. The session stays in the
        // map and stays usable while the turn runs: the thread holds only
        // shared handles (agent loop, cancel flag, transcript path), so
        // control messages for OTHER sessions keep flowing and this session
        // can still be closed or cancelled mid-turn.
        let turn = {
            let Some(session) = self.sessions.get(&session_id) else {
                warn!("harness ACP: prompt for unknown session {session_id}");
                if let Some(id) = request_id {
                    let response = Outbound::Response {
                        id,
                        result: json!({
                            "stopReason": "error",
                            "message": format!("unknown session: {session_id}")
                        }),
                    };
                    write_line(&self.output, &response.to_json_line()?);
                }
                return Ok(());
            };
            let Some(ref agent) = session.agent else {
                warn!("harness ACP: prompt for session without agent {session_id}");
                if let Some(id) = request_id {
                    let response = Outbound::Response {
                        id,
                        result: json!({
                            "stopReason": "error",
                            "message": "session agent not initialized"
                        }),
                    };
                    write_line(&self.output, &response.to_json_line()?);
                }
                return Ok(());
            };
            PromptTurn {
                session_id: session_id.clone(),
                request_id,
                agent: Arc::clone(agent),
                cancel: Arc::clone(&session.cancel),
                cwd: session.cwd.clone(),
                transcript_path: session.transcript_path.clone(),
                output: self.output.clone(),
            }
        };

        thread::Builder::new()
            .name(format!("session-turn-{session_id}"))
            .spawn(move || turn.run(&prompt_text))
            .context("spawn session/prompt turn thread")?;
        Ok(())
    }

    fn handle_session_close(&mut self, params: &Value) -> Result<Value> {
        let session_id = params["sessionId"].as_str().unwrap_or("");
        self.close_registered_session(session_id);
        Ok(json!(null))
    }

    /// Close one registered session by id: shut down its states (kills
    /// background jobs, stops language servers) and drop it from the shared
    /// channels map. Unknown ids are a no-op — `session/close` for a session
    /// this child never knew (or already closed) must not fail the caller's
    /// rotation. The agent's `current` marker is emptied when it still names
    /// this session: a closed session is a finished task, and a recovered
    /// process must resume only interrupted work.
    fn close_registered_session(&mut self, session_id: &str) {
        if let Some(session) = self.sessions.remove(session_id) {
            if !session.agent_id.is_empty() {
                let marker = agent_current_marker(&self.roots.agents, &session.agent_id);
                clear_current_session(&marker, session_id);
            }
            // Stop a turn that may still be running on this session: the
            // agent loop exits at its next iteration, so the turn thread
            // ends promptly instead of finishing a turn nobody waits for.
            session.cancel.store(true, Ordering::SeqCst);
            session.states.shutdown();
            debug!("harness ACP: closed session {session_id}");
        }
        self.shared_channels.lock().unwrap().remove(session_id);
    }

    /// Close every registered session. Called from `session/new` so a failed
    /// best-effort close on the previous task's session cannot leave it
    /// registered forever; at most the sessions created since the last
    /// successful rotation are still present, and they are all stale by
    /// definition of being replaced.
    fn close_all_registered_sessions(&mut self, reason: &str) {
        let stale: Vec<String> = self.sessions.keys().cloned().collect();
        for session_id in stale {
            debug!("harness ACP: closing session {session_id} ({reason})");
            self.close_registered_session(&session_id);
        }
    }

    /// Tear down every registered session's runtime state without touching
    /// the recovery markers. Called when the harness process exits (its
    /// parent closed stdin): background shell jobs are killed and language
    /// servers stopped, so a graceful harness exit cannot leave the
    /// session's running commands behind as strays. Markers are left alone —
    /// whether an interrupted session is resumed is the marker's decision,
    /// not the exit's.
    pub fn shutdown_all_sessions(&mut self) {
        for (session_id, session) in self.sessions.drain() {
            session.cancel.store(true, Ordering::SeqCst);
            session.states.shutdown();
            self.shared_channels.lock().unwrap().remove(&session_id);
            debug!("harness ACP: shut down session {session_id} on exit");
        }
    }
}

/// Append a section to the transcript file.
fn write_transcript_entry(path: &Path, header: &str, body: &str) {
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "\n{header}\n\n{body}");
        let _ = f.flush();
    }
}

/// Append a full turn (assistant content, reasoning, tool calls) to the
/// transcript file.
/// One `session/prompt` turn, running on its own thread. Holds only shared
/// handles, so the session stays closable/cancellable mid-turn and control
/// traffic for other sessions is never blocked by this turn.
struct PromptTurn {
    session_id: String,
    request_id: Option<Value>,
    agent: Arc<Mutex<AgentLoop>>,
    cancel: Arc<AtomicBool>,
    cwd: String,
    transcript_path: Option<PathBuf>,
    output: SharedOutput,
}

impl PromptTurn {
    fn run(self, prompt_text: &str) {
        // Serialize prompts to this session: a second prompt queues here
        // until the running turn releases the agent loop. Different
        // sessions' turns run concurrently — the agent loop is per session.
        let mut agent = self.agent.lock().unwrap();
        // A stale cancel flag (session/cancel or close during a previous
        // turn) must not abort this new turn.
        self.cancel.store(false, Ordering::SeqCst);

        // Collect progress text; the agent loop calls this callback after
        // each LLM response. All collected progress is emitted as
        // notifications when the turn finishes.
        let progress_buf = Arc::new(Mutex::new(Vec::<String>::new()));
        let progress_buf_clone = Arc::clone(&progress_buf);
        let progress_cb: &StreamCallback = &move |text: &str| {
            if let Ok(mut buf) = progress_buf_clone.lock() {
                buf.push(text.to_string());
            }
        };

        // Write the user prompt to the transcript file at the start.
        if let Some(ref tp) = self.transcript_path {
            write_transcript_entry(tp, "## User", prompt_text);
        }

        // Turn callback: writes each turn's content, reasoning, and tool
        // calls to the transcript file in real time.
        let turn_counter = Arc::new(Mutex::new(0u32));
        let transcript_path_for_cb = self.transcript_path.clone();
        let turn_counter_for_cb = Arc::clone(&turn_counter);
        let turn_cb: &TurnCallback = &move |response: &ChatResponse| {
            if let Some(ref tp) = transcript_path_for_cb {
                let mut count = turn_counter_for_cb.lock().unwrap();
                *count += 1;
                write_transcript_turn(tp, *count, response);
            }
        };

        let result =
            agent.run_with_turn_callback(prompt_text, &self.cwd, Some(progress_cb), Some(turn_cb));
        // Read all captured structured-output tool calls (e.g. `handoff`, `plan`).
        let structured_outputs = agent.take_structured_outputs();
        drop(agent);

        // Emit all collected progress as session/update notifications. The
        // sessionId routes each notification to the right session on
        // multiplexing clients (the ACP session/update params include it).
        let collected = progress_buf.lock().unwrap();
        for text in collected.iter() {
            let notif = Outbound::Notification {
                method: "session/update".to_string(),
                params: json!({
                    "sessionId": &self.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": { "text": text }
                    }
                }),
            };
            if let Ok(line) = notif.to_json_line() {
                write_line(&self.output, &line);
            }
        }
        drop(collected);

        let result_json = match result {
            Ok(response) => json!({
                "stopReason": "end_turn",
                "message": response,
                "structured_outputs": structured_outputs,
            }),
            Err(e) => {
                let err_msg = format!("Agent loop error: {e}");
                warn!("harness ACP: {err_msg}");
                json!({
                    "stopReason": "error",
                    "message": err_msg,
                })
            }
        };

        // Answer the prompt request (when it had an id) now that the turn is
        // complete — the only response the caller ever sees.
        if let Some(id) = self.request_id
            && let Ok(line) = (Outbound::Response {
                id,
                result: result_json,
            })
            .to_json_line()
        {
            write_line(&self.output, &line);
        }
    }
}

/// Write one JSON-RPC line as a single atomic write. Turn threads, the main
/// thread, and the stdin reader thread share the process stdout; the
/// underlying mutex keeps concurrent lines from interleaving mid-write.
/// Callers pass newline-terminated lines (`Outbound::to_json_line`).
pub(crate) fn write_line(output: &SharedOutput, line: &str) {
    let mut output = output.clone();
    let _ = output.write_all(line.as_bytes());
    let _ = output.flush();
}

fn write_transcript_turn(path: &Path, turn: u32, response: &ChatResponse) {
    use std::io::Write;
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
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

    use super::super::context::Context;
    use super::*;
    use crate::harness::client::{EarlyToolExecCallback, ToolExecCallback, Usage};
    use crate::harness::tools::test_util;
    use serde_json::json;
    use std::io;
    use std::process;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    struct StubClient;
    struct EchoAgentToolCaller;

    impl AgentToolCaller for EchoAgentToolCaller {
        fn call(
            &self,
            _target: &str,
            _operation: &str,
            arguments: Value,
            _session_id: &str,
        ) -> Result<Value> {
            Ok(arguments)
        }
    }

    impl ChatClient for StubClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            _tools: &[Value],
            on_chunk: Option<&StreamCallback>,
            _on_tool_calls: Option<&ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            if let Some(cb) = on_chunk {
                cb("Task completed successfully.");
            }
            Ok(ChatResponse {
                content: "Task completed successfully.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            })
        }
    }

    /// A test-only writer that captures the harness output for inspection.
    struct BufferSink(Arc<Mutex<Vec<u8>>>);

    impl Write for BufferSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Build a server whose entire output (responses, notifications) lands
    /// in an inspectable buffer.
    fn buffered_server(llm: Arc<dyn ChatClient>) -> (AcpServer, Arc<Mutex<Vec<u8>>>) {
        let buffer: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let server = AcpServer::with_output(
            llm,
            SharedOutput::from_writer(Box::new(BufferSink(Arc::clone(&buffer)))),
        );
        (server, buffer)
    }

    /// Build a buffer-backed server against explicit filesystem roots (so
    /// session-store tests never touch the real `~/.potlatch`). Sets the
    /// private `roots` field directly — the test module is a child of this
    /// module.
    fn buffered_server_with_roots(
        llm: Arc<dyn ChatClient>,
        roots: SessionRoots,
    ) -> (AcpServer, Arc<Mutex<Vec<u8>>>) {
        let (mut server, buffer) = buffered_server(llm);
        server.roots = roots;
        (server, buffer)
    }

    fn written(buffer: &Arc<Mutex<Vec<u8>>>) -> String {
        String::from_utf8(buffer.lock().unwrap().clone()).unwrap()
    }

    /// Send a message and wait for its response line in the shared buffer.
    /// Prompt turns answer asynchronously from their turn thread, so the
    /// response is polled for. Fails the test when no matching response
    /// arrives within a few seconds.
    fn collect_output(
        server: &mut AcpServer,
        buffer: &Arc<Mutex<Vec<u8>>>,
        msg: &Value,
    ) -> Option<Outbound> {
        let start_len = buffer.lock().unwrap().len();
        server.handle_message(msg).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let written = written(buffer);
            for line in written[start_len..].lines() {
                let Ok(parsed) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                let is_response = parsed.get("method").is_none() && parsed.get("id").is_some();
                if is_response && parsed["id"] == msg["id"] {
                    return Some(Outbound::Response {
                        id: parsed["id"].clone(),
                        result: parsed.get("result").cloned().unwrap_or(Value::Null),
                    });
                }
            }
            if Instant::now() >= deadline {
                panic!(
                    "no response to {} id={:?} within timeout",
                    msg["method"], msg["id"]
                );
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn initialize_returns_capabilities() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        });

        let response = collect_output(&mut server, &buffer, &msg);
        let written = written(&buffer);
        eprintln!("BUFFER: {:?}", written);
        assert_eq!(written.lines().count(), 1);
        assert!(written.contains("protocolVersion"));
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
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": "/tmp", "mcpServers": [] }
        });

        let response = collect_output(&mut server, &buffer, &msg);
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
        let (server, buffer) = buffered_server(llm);
        let mut server = server.with_agent_tool_caller(Arc::new(EchoAgentToolCaller));
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

        let response = collect_output(&mut server, &buffer, &msg);
        let session_id = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };
        let tools = {
            let agent = server.sessions[&session_id].agent.as_ref().unwrap();
            let agent = agent.lock().unwrap();
            agent
                .tool_names()
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert!(tools.iter().any(|t| t == "web_search"));
    }

    #[test]
    fn session_new_advertises_mode_config_option() {
        // The harness must advertise a "mode" config option with "plan" as an
        // available value. Without this, the ACP runtime skips setting the
        // session mode, the plan tool is never registered, and the PMO can't
        // call it.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": { "cwd": "/tmp", "mcpServers": [] }
        });

        let response = collect_output(&mut server, &buffer, &msg);
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
    fn multiplexed_session_new_keeps_existing_sessions() {
        // The subagent agent keeps many independent sessions alive on one
        // harness connection: `multiplex: true` opts out of the supersede
        // eviction so a second session/new leaves the first session usable.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let new_session = |id: Value| {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "session/new",
                "params": { "cwd": "/tmp", "multiplex": true }
            })
        };

        let response = collect_output(&mut server, &buffer, &new_session(json!(1)));
        let first = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };
        let response = collect_output(&mut server, &buffer, &new_session(json!(2)));
        let second = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };
        assert_ne!(first, second);

        // Both sessions are alive: both accept prompts.
        for (id, sid) in [(3, &first), (4, &second)] {
            let prompt = json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{ "type": "text", "text": "hi" }]
                }
            });
            let response = collect_output(&mut server, &buffer, &prompt);
            match response {
                Some(Outbound::Response { result, .. }) => {
                    assert_eq!(result["stopReason"], "end_turn");
                }
                _ => panic!("expected response for {sid}"),
            }
        }
    }

    #[test]
    fn prompts_to_different_sessions_run_concurrently() {
        // The second prompt must not wait for the first turn to finish: the
        // first turn parks inside its LLM call until the second turn's
        // response has already arrived. If turns were serialized globally,
        // the second response could never be observed before the release.
        struct GatedClient {
            release: Arc<AtomicBool>,
        }
        impl ChatClient for GatedClient {
            fn chat(
                &self,
                _model: &str,
                messages: &[Value],
                _tools: &[Value],
                _on_chunk: Option<&StreamCallback>,
                _on_tool_calls: Option<&ToolExecCallback<'_>>,
                _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
            ) -> Result<ChatResponse> {
                let is_slow = messages
                    .iter()
                    .any(|m| serde_json::to_string(m).unwrap().contains("slow turn"));
                if is_slow {
                    while !self.release.load(Ordering::SeqCst) {
                        thread::sleep(Duration::from_millis(5));
                    }
                }
                Ok(ChatResponse {
                    content: if is_slow {
                        "slow done".into()
                    } else {
                        "quick done".into()
                    },
                    tool_calls: vec![],
                    finish_reason: "stop".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                })
            }
        }

        let release = Arc::new(AtomicBool::new(false));
        let llm: Arc<dyn ChatClient> = Arc::new(GatedClient {
            release: Arc::clone(&release),
        });
        let (mut server, buffer) = buffered_server(llm);

        let mut sids = Vec::new();
        for id in 1..=2 {
            let msg = json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "session/new",
                "params": { "cwd": "/tmp", "multiplex": true }
            });
            match collect_output(&mut server, &buffer, &msg) {
                Some(Outbound::Response { result, .. }) => {
                    sids.push(result["sessionId"].as_str().unwrap().to_string());
                }
                _ => panic!("expected response"),
            }
        }

        // First prompt parks inside its LLM call.
        let slow_prompt = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "session/prompt",
            "params": {
                "sessionId": sids[0],
                "prompt": [{ "type": "text", "text": "slow turn" }]
            }
        });
        server.handle_message(&slow_prompt).unwrap();

        // Second prompt on the OTHER session must complete while the first
        // is still parked.
        let quick_prompt = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "session/prompt",
            "params": {
                "sessionId": sids[1],
                "prompt": [{ "type": "text", "text": "quick turn" }]
            }
        });
        server.handle_message(&quick_prompt).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let quick_done = loop {
            if written(&buffer).contains("quick done") {
                break true;
            }
            if Instant::now() >= deadline {
                break false;
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert!(
            quick_done,
            "second session's prompt was blocked by the first"
        );

        // Release the slow turn; its response must still arrive.
        release.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(5);
        let slow_done = loop {
            if written(&buffer).contains("slow done") {
                break true;
            }
            if Instant::now() >= deadline {
                break false;
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert!(slow_done, "slow turn never completed");
    }

    #[test]
    fn progress_notifications_carry_the_session_id() {
        // Multiplexing clients route notifications by sessionId: every
        // session/update the harness emits must name its session.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let session_id = match collect_output(&mut server, &buffer, &msg) {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        let prompt = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "hi" }]
            }
        });
        collect_output(&mut server, &buffer, &prompt);

        let updates: Vec<Value> = written(&buffer)
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|msg| msg["method"] == "session/update")
            .collect();
        assert!(!updates.is_empty(), "expected at least one session/update");
        for update in updates {
            assert_eq!(update["params"]["sessionId"], session_id.as_str());
        }
    }

    #[test]
    fn background_jobs_span_the_session_and_die_on_close() {
        // A session's background jobs live from session/new to session/close:
        // a turn finishes without touching them (the model may keep polling
        // or acting on them in later turns), and session/close kills whatever
        // is still running so no daemon outlives the session's pipes.
        struct ScriptedClient {
            call: AtomicUsize,
            pid_file: PathBuf,
        }
        impl ChatClient for ScriptedClient {
            fn chat(
                &self,
                _model: &str,
                _messages: &[Value],
                _tools: &[Value],
                _on_chunk: Option<&StreamCallback>,
                _on_tool_calls: Option<&ToolExecCallback<'_>>,
                _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
            ) -> Result<ChatResponse> {
                let call = self.call.fetch_add(1, Ordering::SeqCst);
                if call == 0 {
                    return Ok(ChatResponse {
                        content: String::new(),
                        tool_calls: vec![json!({
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": format!(
                                    "{{\"command\":\"echo $$ > {}; sleep 30\",\"background\":true}}",
                                    self.pid_file.display()
                                )
                            }
                        })],
                        finish_reason: "tool_calls".into(),
                        usage: Usage::default(),
                        tool_results: vec![],
                        elapsed_ms: 0,
                        reasoning: String::new(),
                    });
                }
                // Before ending the turn, wait until the background job's
                // bash has written its pid, so the kill assertion below can
                // never race the spawn.
                let deadline = Instant::now() + Duration::from_secs(5);
                while !self.pid_file.is_file() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(10));
                }
                assert!(self.pid_file.is_file(), "background job never started");
                Ok(ChatResponse {
                    content: "done".into(),
                    tool_calls: vec![],
                    finish_reason: "stop".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                })
            }
        }

        let pid_file = env::temp_dir().join(format!("potlatch-job-span-{}", process::id()));
        let _ = fs::remove_file(&pid_file);
        let llm: Arc<dyn ChatClient> = Arc::new(ScriptedClient {
            call: AtomicUsize::new(0),
            pid_file: pid_file.clone(),
        });
        let (mut server, buffer) = buffered_server(llm);

        let response = collect_output(
            &mut server,
            &buffer,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": { "cwd": "/tmp", "mcpServers": [] }
            }),
        )
        .expect("session/new responds");
        let sid = match &response {
            Outbound::Response { result, .. } => result["sessionId"].as_str().unwrap().to_string(),
            _ => panic!("expected response"),
        };

        collect_output(
            &mut server,
            &buffer,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{ "type": "text", "text": "start a background job" }]
                }
            }),
        )
        .expect("session/prompt responds");

        // The turn finished, the job did not: the process must still be
        // running so a later turn can poll it.
        let pid = fs::read_to_string(&pid_file)
            .expect("background job wrote its pid")
            .trim()
            .parse::<i32>()
            .expect("pid");
        assert!(
            unsafe { libc::kill(pid, 0) } == 0,
            "background job must survive the turn that spawned it"
        );

        collect_output(
            &mut server,
            &buffer,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/close",
                "params": { "sessionId": sid }
            }),
        )
        .expect("session/close responds");

        // Session close tears the session's jobs down: poll until the
        // SIGKILL is observable.
        let mut gone = false;
        for _ in 0..100 {
            if unsafe { libc::kill(pid, 0) } != 0 {
                gone = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            gone,
            "session/close must kill the session's background jobs"
        );
        let _ = fs::remove_file(&pid_file);
    }

    #[test]
    fn harness_exit_kills_running_background_jobs() {
        // When the harness process exits (parent closed stdin), every
        // session's runtime state is torn down: a background job still
        // running must be killed instead of leaking as a stray process.
        // Same scripted setup as `background_jobs_span_the_session_and_die_on_close`.
        struct ScriptedClient {
            call: AtomicUsize,
            pid_file: PathBuf,
        }
        impl ChatClient for ScriptedClient {
            fn chat(
                &self,
                _model: &str,
                _messages: &[Value],
                _tools: &[Value],
                _on_chunk: Option<&StreamCallback>,
                _on_tool_calls: Option<&ToolExecCallback<'_>>,
                _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
            ) -> Result<ChatResponse> {
                let call = self.call.fetch_add(1, Ordering::SeqCst);
                if call == 0 {
                    return Ok(ChatResponse {
                        content: String::new(),
                        tool_calls: vec![json!({
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": format!(
                                    "{{\"command\":\"echo $$ > {}; sleep 30\",\"background\":true}}",
                                    self.pid_file.display()
                                )
                            }
                        })],
                        finish_reason: "tool_calls".into(),
                        usage: Usage::default(),
                        tool_results: vec![],
                        elapsed_ms: 0,
                        reasoning: String::new(),
                    });
                }
                let deadline = Instant::now() + std::time::Duration::from_secs(5);
                while !self.pid_file.is_file() && Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                assert!(self.pid_file.is_file(), "background job never started");
                Ok(ChatResponse {
                    content: "done".into(),
                    tool_calls: vec![],
                    finish_reason: "stop".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                })
            }
        }

        fn is_alive(pid: i32) -> bool {
            let rc = unsafe { libc::kill(pid, 0) };
            rc == 0
        }

        let pid_file =
            std::env::temp_dir().join(format!("potlatch-exit-jobs-{}", std::process::id()));
        let _ = std::fs::remove_file(&pid_file);
        let llm: Arc<dyn ChatClient> = Arc::new(ScriptedClient {
            call: AtomicUsize::new(0),
            pid_file: pid_file.clone(),
        });
        let (mut server, buffer) = buffered_server(llm);

        let response = collect_output(
            &mut server,
            &buffer,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": { "cwd": "/tmp", "mcpServers": [] }
            }),
        )
        .expect("session/new responds");
        let sid = match &response {
            Outbound::Response { result, .. } => result["sessionId"].as_str().unwrap().to_string(),
            _ => panic!("expected response"),
        };

        collect_output(
            &mut server,
            &buffer,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{ "type": "text", "text": "start a background job" }]
                }
            }),
        )
        .expect("session/prompt responds");

        let pid = std::fs::read_to_string(&pid_file)
            .expect("background job wrote its pid")
            .trim()
            .parse::<i32>()
            .expect("pid");
        assert!(is_alive(pid), "job must be alive before the harness exits");

        // The harness is going away (stdin EOF): the exit cleanup must kill
        // the session's running background job.
        server.shutdown_all_sessions();
        let mut gone = false;
        for _ in 0..100 {
            if !is_alive(pid) {
                gone = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(gone, "harness exit must kill running background jobs");
        let _ = std::fs::remove_file(&pid_file);
    }

    #[test]
    fn session_new_evicts_sessions_the_caller_failed_to_close() {
        // The orchestrator closes the previous task's session best-effort
        // before rotating. When that close never arrives (transport blip),
        // the old session must not stay registered forever: nothing else
        // ever evicts it, and this child serves thousands of tasks. The
        // session/new that follows the failed close reclaims it.
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let new_session = |id: Value| {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "session/new",
                "params": { "cwd": "/tmp", "mcpServers": [] }
            })
        };

        // First session — never closed (simulating the failed close).
        let response = collect_output(&mut server, &buffer, &new_session(json!(1)));
        let first = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        // Second session/new: must reclaim the first.
        let response = collect_output(&mut server, &buffer, &new_session(json!(2)));
        let second = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected response"),
        };

        assert_ne!(first, second);
        assert!(
            !server.sessions.contains_key(&first),
            "the superseded session must be evicted, not leaked"
        );
        assert!(server.sessions.contains_key(&second));
        assert!(
            server.shared_channels.lock().unwrap().get(&first).is_none(),
            "the superseded session's channels must be removed too"
        );
    }

    #[test]
    fn session_close_for_unknown_session_is_a_noop() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "session/close",
            "params": { "sessionId": "never-existed" }
        });
        let response = collect_output(&mut server, &buffer, &msg);
        assert!(matches!(response, Some(Outbound::Response { .. })));
    }

    #[test]
    fn authenticate_is_noop() {
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server(llm);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "authenticate",
            "params": { "methodId": "cursor_login" }
        });

        let response = collect_output(&mut server, &buffer, &msg);
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
        let (mut server, buffer) = buffered_server(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let resp = collect_output(&mut server, &buffer, &new_msg);
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

        let response = collect_output(&mut server, &buffer, &prompt_msg);
        let written = written(&buffer);
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
        let (mut server, buffer) = buffered_server(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let resp = collect_output(&mut server, &buffer, &new_msg);
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
        let resp1 = collect_output(&mut server, &buffer, &prompt1);
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
        let resp2 = collect_output(&mut server, &buffer, &prompt2);
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
        let (mut server, buffer) = buffered_server(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let resp = collect_output(&mut server, &buffer, &new_msg);
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
        let response = collect_output(&mut server, &buffer, &prompt_msg);
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
        let (mut server, buffer) = buffered_server(llm);

        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let resp = collect_output(&mut server, &buffer, &new_msg);
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
        let _ = collect_output(&mut server, &buffer, &mode_msg);

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
        let response = collect_output(&mut server, &buffer, &prompt_msg);
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
        captured: Arc<Mutex<Vec<Value>>>,
    }

    impl ToolsCapturingClient {
        fn new(captured: Arc<Mutex<Vec<Value>>>) -> Self {
            Self { captured }
        }
    }

    impl ChatClient for ToolsCapturingClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            tools: &[Value],
            _on_chunk: Option<&StreamCallback>,
            _on_tool_calls: Option<&ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            *self.captured.lock().unwrap() = tools.to_vec();
            Ok(ChatResponse {
                content: "done".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            })
        }
    }

    fn run_session_prompt_in_mode(mode: &str) -> Vec<String> {
        let captured: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> = Arc::new(ToolsCapturingClient::new(Arc::clone(&captured)));

        let (mut server, buffer) = buffered_server(llm);
        let new_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let resp = collect_output(&mut server, &buffer, &new_msg);
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
            let _ = collect_output(&mut server, &buffer, &mode_msg);
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
        let _ = collect_output(&mut server, &buffer, &prompt_msg);
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
        let captured: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> = Arc::new(ToolsCapturingClient::new(Arc::clone(&captured)));
        let (mut server, buffer) = buffered_server(llm);

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
        let resp = collect_output(&mut server, &buffer, &new_msg);
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
        let _ = collect_output(&mut server, &buffer, &prompt_msg);

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

    /// A chat client that captures the messages it was called with, so tests
    /// can assert what context a session restored or initialized.
    struct MessagesCapturingClient {
        captured: Arc<Mutex<Vec<Vec<Value>>>>,
    }

    impl ChatClient for MessagesCapturingClient {
        fn chat(
            &self,
            _model: &str,
            messages: &[Value],
            _tools: &[Value],
            _on_chunk: Option<&StreamCallback>,
            _on_tool_calls: Option<&ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            self.captured.lock().unwrap().push(messages.to_vec());
            Ok(ChatResponse {
                content: "ok".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            })
        }
    }

    fn test_roots(dir: &Path) -> SessionRoots {
        SessionRoots {
            sessions: dir.join("sessions"),
            agents: dir.join("agents"),
        }
    }

    fn session_new_with_agent(
        server: &mut AcpServer,
        buffer: &Arc<Mutex<Vec<u8>>>,
        id: Value,
        agent_id: &str,
    ) -> String {
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": "/tmp", "agent_id": agent_id }
        });
        let response = collect_output(server, buffer, &msg);
        match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected session/new response"),
        }
    }

    #[test]
    fn session_new_writes_the_current_marker_and_persists_context() {
        let dir = test_util::unique_test_dir();
        let captured: Arc<Mutex<Vec<Vec<Value>>>> = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> = Arc::new(MessagesCapturingClient {
            captured: captured.clone(),
        });
        let (mut server, buffer) = buffered_server_with_roots(llm, test_roots(dir.path()));

        let sid = session_new_with_agent(&mut server, &buffer, json!(1), "worker-7");
        let marker = dir.path().join("agents").join("worker-7").join("current");
        assert_eq!(read_current_session(&marker).as_deref(), Some(sid.as_str()));

        // Prompt once — the context (system + user prompt) is persisted.
        let prompt = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "implement the feature" }]
            }
        });
        let response = collect_output(&mut server, &buffer, &prompt);
        assert!(
            matches!(&response, Some(Outbound::Response { result, .. })
                if result.get("stopReason").and_then(Value::as_str).is_some_and(|s| s != "error")),
            "prompt should end without error, got: {:?}",
            response.map(|r| r.to_json_line().unwrap_or_default())
        );
        // The persisted transcript restores: system prompt, user prompt,
        // and the assistant turn — all three survive the markdown roundtrip.
        let context_file = dir.path().join("sessions").join(&sid).join("context");
        let transcript = fs::read_to_string(&context_file).expect("context persisted");
        assert_eq!(
            Context::restore_markdown(&transcript)
                .expect("transcript restores")
                .entries()
                .len(),
            3,
            "transcript:\n{transcript}"
        );
    }

    #[test]
    fn closing_a_session_empties_the_current_marker() {
        let dir = test_util::unique_test_dir();
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server_with_roots(llm, test_roots(dir.path()));

        let sid = session_new_with_agent(&mut server, &buffer, json!(1), "worker-7");
        let close = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/close",
            "params": { "sessionId": sid }
        });
        let response = collect_output(&mut server, &buffer, &close);
        assert!(matches!(response, Some(Outbound::Response { .. })));

        let marker = dir.path().join("agents").join("worker-7").join("current");
        assert_eq!(
            read_current_session(&marker),
            None,
            "a closed session is a finished task; the marker must be empty"
        );
    }

    #[test]
    fn a_recovered_agent_resumes_its_interrupted_session() {
        let dir = test_util::unique_test_dir();
        let captured: Arc<Mutex<Vec<Vec<Value>>>> = Arc::new(Mutex::new(Vec::new()));
        let llm: Arc<dyn ChatClient> = Arc::new(MessagesCapturingClient {
            captured: captured.clone(),
        });

        // First run: an agent session whose task never finished (no close).
        let (mut first, buffer) = buffered_server_with_roots(llm.clone(), test_roots(dir.path()));
        let sid = session_new_with_agent(&mut first, &buffer, json!(1), "worker-7");
        let prompt = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": sid,
                "prompt": [{ "type": "text", "text": "halfway through the task" }]
            }
        });
        let _ = collect_output(&mut first, &buffer, &prompt);
        drop(first); // crash: the session was never closed

        // Second run: a fresh server (recovered process) for the same agent.
        let (mut second, buffer) = buffered_server_with_roots(llm.clone(), test_roots(dir.path()));
        let resumed_sid = session_new_with_agent(&mut second, &buffer, json!(1), "worker-7");
        assert_eq!(
            resumed_sid, sid,
            "the interrupted session's id must be adopted, not replaced"
        );

        // The next prompt reuses the restored context: the previous run's
        // user prompt is already in the conversation.
        let prompt = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": resumed_sid,
                "prompt": [{ "type": "text", "text": "continue" }]
            }
        });
        let _ = collect_output(&mut second, &buffer, &prompt);
        let calls = captured.lock().unwrap();
        let last = calls.last().expect("prompt ran");
        let all_text: String = last
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect();
        assert!(
            all_text.contains("halfway through the task"),
            "resumed context must carry the previous run's prompt: {all_text}"
        );
    }

    #[test]
    fn an_empty_marker_starts_a_fresh_session() {
        let dir = test_util::unique_test_dir();
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server_with_roots(llm, test_roots(dir.path()));

        let first = session_new_with_agent(&mut server, &buffer, json!(1), "worker-7");
        let close = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/close",
            "params": { "sessionId": first }
        });
        let _ = collect_output(&mut server, &buffer, &close);

        // The marker is empty; the next session must not adopt the old id.
        let second = session_new_with_agent(&mut server, &buffer, json!(3), "worker-7");
        assert_ne!(first, second);
    }

    #[test]
    fn a_marker_without_a_context_file_starts_fresh() {
        let dir = test_util::unique_test_dir();
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);

        // A stale marker names a session whose context file is gone (e.g.
        // sessions directory wiped): resume must not be attempted.
        let marker = dir.path().join("agents").join("worker-7").join("current");
        fs::create_dir_all(marker.parent().unwrap()).unwrap();
        fs::write(&marker, "missing-session").unwrap();

        let (mut server, buffer) = buffered_server_with_roots(llm, test_roots(dir.path()));
        let sid = session_new_with_agent(&mut server, &buffer, json!(1), "worker-7");
        assert_ne!(sid, "missing-session");
        // The marker now names the new session.
        assert_eq!(read_current_session(&marker).as_deref(), Some(sid.as_str()));
    }

    #[test]
    fn sessions_without_agent_id_have_no_marker() {
        let dir = test_util::unique_test_dir();
        let llm: Arc<dyn ChatClient> = Arc::new(StubClient);
        let (mut server, buffer) = buffered_server_with_roots(llm, test_roots(dir.path()));

        let msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": { "cwd": "/tmp" }
        });
        let response = collect_output(&mut server, &buffer, &msg);
        let sid = match response {
            Some(Outbound::Response { result, .. }) => {
                result["sessionId"].as_str().unwrap().to_string()
            }
            _ => panic!("expected session/new response"),
        };
        let marker = dir.path().join("agents").join("worker-7").join("current");
        assert!(!marker.exists());
        let context_file = dir.path().join("sessions").join(&sid).join("context");
        assert!(!context_file.exists());
    }
}
