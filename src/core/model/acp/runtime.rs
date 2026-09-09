//! Configurable ACP server command (default: `agent acp`) runs as **one long-lived subprocess**
//! per Potlatch agent role. Each task
//! calls **`session/close`** (best effort) then **`session/new`** on the same stdio connection so
//! the model does not keep prior in-agent transcript; workflow continuity stays in Potlatch’s
//! state files and token use stays lower than reusing one session for every task.
//!
//! The structured-output contract is **per task**: [`AcpRuntime::run_task`] takes the tool
//! definitions for the task it is starting and registers them with the session it creates or
//! rotates, so a role can ask for a different shape on every call.
//! [`AcpRuntime::run_in_current_session`] sends a follow-up prompt into that same session without
//! rotating it — it is how the structured-output repair loop asks the model to fix its tool call
//! without making it redo the task.
//!
//! [ACP slash commands](https://agentclientprotocol.com/protocol/slash-commands): the agent may
//! send `available_commands_update`; [`StreamTextHooks`] records command names for future
//! role-specific prompt logic (not wired into prompts yet).
//!
//! [Session modes](https://agentclientprotocol.com/protocol/session-modes): callers may request
//! **`ask`** or **`plan`**. Modes apply only when the agent advertises them (config-option `mode`
//! value or non-empty legacy `availableModes`). Otherwise the agent default is kept;
//! `current_mode_update` keeps hooks in sync.

use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use tracing::{debug, info, warn};

use super::backends::{AcpVendorExtension, StructuredOutputBackend};
use super::capabilities::CapabilityProvider;
use super::client::AcpClient;
use super::orchestrator_hooks::StreamTextHooks;
use super::types::{
    ClientCapabilities, ClientFsCapabilities, DEFAULT_PROTOCOL_VERSION, ImplementationInfo,
    InitializeParams, NewSessionParams, NewSessionResult, PromptResult, mode_id_is_available,
    model_selector_for_session, select_option_allows_value,
};
use crate::core::agent::AgentHandoff;
use crate::core::bus::AgentBus;
use crate::core::config::build_acp_spawn_command;
use crate::core::config::uri::ModelUri;
use crate::paths::APP_NAME;

const TASK_CONTEXT_RESET_GUIDANCE: &str = r#"IMPORTANT CONTEXT HANDLING:
Treat this assignment as a fresh task. Do not rely on prior chat history or assumptions from earlier assignments unless this prompt explicitly refers to them. Use only the repository state, issue/MR context, and instructions present in this task.

"#;

/// How many times a task prompt is resent after a *transport* failure (the
/// ACP child exited, or `session/prompt` itself errored). Unrelated to
/// structured-output repair, which lives above this layer.
const MAX_TRANSPORT_RETRIES: u32 = 5;

struct AcpSession {
    client: Arc<AcpClient>,
    child: Child,
    session_id: String,
    hooks: Arc<StreamTextHooks>,
}

fn dispose_acp_session(s: AcpSession) {
    let _ = s.client.session_cancel(&s.session_id);
    let mut s = s;
    let _ = s.child.kill();
    let _ = s.child.wait();
    drop(s.client);
}

fn close_acp_session_best_effort(client: &AcpClient, session_id: &str) {
    match client.session_close(session_id) {
        Ok(()) => {}
        Err(e) => debug!(
            target: "potlatch::acp",
            "session/close failed for session {} (continuing with session/new): {}",
            session_id,
            e
        ),
    }
}

pub(crate) struct AcpRuntime {
    working_dir: String,
    model_uri: Option<String>,
    endpoint_model: Option<String>,
    acp_command: Vec<String>,
    acp_env: std::collections::HashMap<String, String>,
    /// Vendor ACP extension — None when the vendor has no extension.
    vendor_ext: Option<Arc<dyn AcpVendorExtension>>,
    /// Structured-output rendering/capture strategy selected by vendor.
    structured_output_backend: Arc<dyn StructuredOutputBackend>,
    /// Capability provider (set by the agent at construction).
    capability_provider: Mutex<Option<Arc<dyn CapabilityProvider>>>,
    agent_bus: Option<AgentBus>,
    /// Directories the harness's write/edit tools may touch outside the
    /// session cwd, forwarded to the harness in `session/new`.
    write_roots: Vec<String>,
    /// Structured-output definitions for the task currently in flight. The
    /// selected backend either passes them to the harness via `session/new` or
    /// renders them into the marker prompt. Set per task by
    /// [`AcpRuntime::run_task`] and retained across transport retries.
    structured_output_tools: Mutex<Vec<Value>>,
    shutdown: Arc<AtomicBool>,
    agent_id: String,
    acp: Mutex<Option<AcpSession>>,
    unexpected_quits_count: Arc<AtomicU64>,
}

impl AcpRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        working_dir: String,
        model_uri: Option<String>,
        endpoint_model: Option<String>,
        acp_command: Vec<String>,
        acp_env: std::collections::HashMap<String, String>,
        preferred_session_mode: Option<&'static str>,
        agent_bus: Option<AgentBus>,
        write_roots: Vec<String>,
        shutdown: Arc<AtomicBool>,
        agent_id: String,
    ) -> Self {
        let vendor_ext =
            super::backends::resolve_vendor_extension(model_uri.as_deref(), preferred_session_mode);
        let structured_output_backend =
            super::backends::resolve_structured_output_backend(model_uri.as_deref());
        Self {
            working_dir,
            model_uri,
            endpoint_model,
            acp_command,
            acp_env,
            structured_output_tools: Mutex::new(Vec::new()),
            shutdown,
            agent_id,
            acp: Mutex::new(None),
            unexpected_quits_count: Arc::new(AtomicU64::new(0)),
            vendor_ext,
            structured_output_backend,
            capability_provider: Mutex::new(None),
            agent_bus,
            write_roots,
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Set the capability provider (called by the agent at construction).
    pub fn set_capability_provider(&self, provider: Option<Arc<dyn CapabilityProvider>>) {
        *self.capability_provider.lock().unwrap() = provider;
    }

    /// Run one task: register this task's structured-output contract, ensure a
    /// fresh ACP session (`session/close` then `session/new` on the same child
    /// when the process is already running), and send `session/prompt`.
    /// Retries after transport failures keep the same session while the child
    /// stays up; if the child exits, a new process and session are created and
    /// the same contract is registered again.
    pub fn run_task(
        &self,
        prompt: &str,
        structured_output_tools: Vec<Value>,
        cancel_check: Option<&dyn Fn() -> bool>,
        follow_up_poll: Option<&dyn Fn() -> Vec<String>>,
    ) -> Result<AgentHandoff> {
        self.register_task_contract(structured_output_tools);
        let prompt = self.prompt_with_structured_output(prompt);
        let prompt = prepare_task_prompt(&prompt);

        if let Some(model_uri) = &self.model_uri {
            info!(
                "Running ACP agent {} model={:?} prompt_len={} (new session per task, same process)",
                self.agent_id(),
                model_uri,
                prompt.len()
            );
        } else {
            info!(
                "Running ACP agent {} prompt_len={} (new session per task, same process)",
                self.agent_id(),
                prompt.len()
            );
        }
        debug!("Agent task prompt: {}", prompt);

        self.rotate_acp_session_for_new_task()?;
        self.run_prompt_with_transport_retry(&prompt, cancel_check, follow_up_poll)
    }

    /// Send a follow-up prompt into the session the current task is already
    /// running in: no rotation, no contract re-registration, no restatement of
    /// the task. The model keeps its full conversation for this task, so the
    /// prompt only has to say what to fix. Used by structured-output repair.
    ///
    /// Transport failures are returned as-is rather than retried: a respawned
    /// child would have lost the task context this prompt depends on, which
    /// makes a retry here worse than letting the caller fail the task.
    pub fn run_in_current_session(
        &self,
        prompt: &str,
        cancel_check: Option<&dyn Fn() -> bool>,
        follow_up_poll: Option<&dyn Fn() -> Vec<String>>,
    ) -> Result<AgentHandoff> {
        let prompt = self.prompt_with_structured_output(prompt);
        debug!(
            "Agent {} follow-up prompt in current session: {}",
            self.agent_id(),
            prompt
        );
        self.run_prompt_once(&prompt, cancel_check, follow_up_poll)
    }

    /// Record the structured-output contract the task being started asks for.
    /// It stays registered for every session this task needs — including one
    /// recreated by a transport retry — until the next task replaces it.
    fn register_task_contract(&self, tools: Vec<Value>) {
        *self.structured_output_tools.lock().unwrap() = tools;
    }

    fn prompt_with_structured_output(&self, prompt: &str) -> String {
        let tools = self.structured_output_tools.lock().unwrap();
        self.structured_output_backend
            .prepare_prompt(prompt, &tools)
    }

    /// The backend-specific `structured_output_tools` value for `session/new`.
    /// The Potlatch harness receives tool definitions; marker vendors receive
    /// the same contract in their prompt and therefore expose no wire tools.
    fn session_structured_output_tools(&self) -> Option<Vec<Value>> {
        let tools = self.structured_output_tools.lock().unwrap().clone();
        self.structured_output_backend.session_tools(&tools)
    }

    fn session_agent_tools(&self) -> Option<Vec<Value>> {
        let is_potlatch = self
            .model_uri
            .as_deref()
            .and_then(|uri| ModelUri::parse(uri).ok())
            .is_some_and(|uri| uri.vendor == "potlatch");
        if !is_potlatch {
            return None;
        }
        let tools = self
            .agent_bus
            .as_ref()?
            .registered_tools(Duration::from_millis(250))
            .ok()?;
        (!tools.is_empty()).then(|| {
            tools
                .into_iter()
                .filter_map(|tool| serde_json::to_value(tool).ok())
                .collect()
        })
    }

    /// Context channels registered by in-process agents on the bus. Each is
    /// shipped to the potlatch harness via `session/new` and injected as a
    /// system message at session init. Potlatch extension — `None` for non-potlatch
    /// backends or when no channels are registered.
    fn session_context_channels(&self) -> Option<Vec<Value>> {
        let is_potlatch = self
            .model_uri
            .as_deref()
            .and_then(|uri| ModelUri::parse(uri).ok())
            .is_some_and(|uri| uri.vendor == "potlatch");
        if !is_potlatch {
            return None;
        }
        let channels = self.agent_bus.as_ref()?.context_channels().ok()?;
        (!channels.is_empty()).then(|| {
            channels
                .into_iter()
                .map(|ch| {
                    serde_json::json!({
                        "name": ch.name,
                        "content": ch.content,
                    })
                })
                .collect()
        })
    }

    fn run_prompt_with_transport_retry(
        &self,
        prompt: &str,
        cancel_check: Option<&dyn Fn() -> bool>,
        follow_up_poll: Option<&dyn Fn() -> Vec<String>>,
    ) -> Result<AgentHandoff> {
        let mut transport_retries: u32 = 0;
        loop {
            match self.run_prompt_once(prompt, cancel_check, follow_up_poll) {
                Ok(handoff) => return Ok(handoff),
                Err(error) if is_transport_failure(&error) => {
                    transport_retries += 1;
                    if transport_retries > MAX_TRANSPORT_RETRIES {
                        return Err(error);
                    }
                    warn!(
                        "ACP task failed for {} ({error}). Retry {transport_retries}/{MAX_TRANSPORT_RETRIES}",
                        self.agent_id(),
                    );
                    self.unexpected_quits_count.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(retry_backoff_for_unfinished_task(transport_retries));
                    self.ensure_acp_session_for_retry()?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// One `session/prompt` round trip on the current session, with the
    /// cancellation check and follow-up forwarding running while it is in
    /// flight. Captures from any earlier turn are cleared first, so what comes
    /// back belongs to this prompt only.
    ///
    /// A result reporting `stopReason: "error"` — the agent loop itself
    /// failed, e.g. the LLM endpoint stayed unavailable after the agent's own
    /// in-process retries — propagates as an error instead of a handoff: no
    /// repair prompt can fix an endpoint, so it must not be misread as a turn
    /// that merely skipped the required tool call.
    fn run_prompt_once(
        &self,
        prompt: &str,
        cancel_check: Option<&dyn Fn() -> bool>,
        follow_up_poll: Option<&dyn Fn() -> Vec<String>>,
    ) -> Result<AgentHandoff> {
        if self.shutdown.load(Ordering::SeqCst) {
            self.kill_child();
            anyhow::bail!("Agent interrupted by shutdown");
        }

        let (client, session_id, hooks) = {
            let g = self.acp.lock().unwrap();
            let s = g.as_ref().context("ACP session missing after ensure")?;
            (
                Arc::clone(&s.client),
                s.session_id.clone(),
                Arc::clone(&s.hooks),
            )
        };

        hooks.clear();
        if let Some(ref ext) = self.vendor_ext {
            let provider = self.capability_provider.lock().unwrap().clone();
            hooks.set_vendor_state(Some(ext.create_state(provider, self.agent_bus.clone())));
        }

        let prompt_owned = prompt.to_string();
        let (tx, rx) = std::sync::mpsc::channel::<Result<PromptResult, String>>();
        let client_for_thread = Arc::clone(&client);
        let session_id_for_thread = session_id.clone();
        thread::spawn(move || {
            let out = client_for_thread
                .session_prompt(&session_id_for_thread, &prompt_owned)
                .map_err(|e| e.to_string());
            let _ = tx.send(out);
        });

        let mut polls_since_cancel_check: u32 = 0;
        const CANCEL_CHECK_INTERVAL: u32 = 25;

        let handoff_result = loop {
            if self.shutdown.load(Ordering::SeqCst) {
                self.kill_child();
                break Err(anyhow::anyhow!("Agent interrupted by shutdown"));
            }
            polls_since_cancel_check += 1;
            if polls_since_cancel_check >= CANCEL_CHECK_INTERVAL {
                polls_since_cancel_check = 0;
                if let Some(check) = cancel_check
                    && check()
                {
                    warn!(
                        "Cancel check triggered, killing ACP agent {}",
                        self.agent_id()
                    );
                    self.kill_child();
                    break Err(anyhow::anyhow!("Agent cancelled by external condition"));
                }

                // Poll for follow-up messages and forward them to the
                // running session via the vendor extension (session/inject
                // on the potlatch harness backend; no-op on others).
                if let Some(poll) = follow_up_poll {
                    let msgs = poll();
                    if !msgs.is_empty()
                        && let Some(ref ext) = self.vendor_ext
                    {
                        ext.forward_followups(&client, &session_id, &msgs);
                    }
                }
            }

            if let Some(exit) = self.take_child_exit_status()? {
                self.kill_child();
                break Err(anyhow::anyhow!(
                    "agent {} exited during prompt ({})",
                    self.agent_id(),
                    exit
                ));
            }

            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok(pr)) => {
                    if let Some(message) = agent_loop_error_message(&pr) {
                        break Err(anyhow::anyhow!(
                            "ACP agent loop error: {}",
                            if message.trim().is_empty() {
                                "no detail reported"
                            } else {
                                &message
                            }
                        ));
                    }
                    self.wait_for_followup_if_vendor(&hooks, cancel_check)?;
                    break Ok(handoff_from_prompt_hooks(
                        &hooks,
                        pr,
                        self.vendor_ext.as_ref(),
                        Some(self.structured_output_backend.as_ref()),
                    ));
                }
                Ok(Err(e)) => break Err(anyhow::anyhow!("ACP session/prompt: {}", e)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(anyhow::anyhow!(
                        "ACP prompt thread died for {}",
                        self.agent_id()
                    ));
                }
            }
        };

        hooks.set_vendor_state(None);
        handoff_result
    }

    fn wait_for_followup_if_vendor(
        &self,
        hooks: &StreamTextHooks,
        cancel_check: Option<&dyn Fn() -> bool>,
    ) -> Result<()> {
        if let Some(ref ext) = self.vendor_ext
            && let Some(vs) = hooks.vendor_state_snapshot()
        {
            ext.wait_for_followup(vs.as_ref(), cancel_check, &self.shutdown)?;
        }
        Ok(())
    }

    fn kill_child(&self) {
        let mut g = self.acp.lock().unwrap();
        if let Some(s) = g.take() {
            self.notify_caller_session_closed(&s.session_id);
            dispose_acp_session(s);
        }
    }

    /// Notice that a caller task session retired: bus lifecycle listeners
    /// (e.g. the subagent agent) close the resources owned by that session.
    /// Best-effort by construction — the notice never blocks or fails, and a
    /// missed notice (no bus, no listener) is bounded by the subagent cap.
    fn notify_caller_session_closed(&self, session_id: &str) {
        if let Some(bus) = self.agent_bus.as_ref() {
            bus.notify_caller_session_closed(session_id);
        }
    }

    /// Applies preferred session mode via the vendor extension when active.
    fn try_apply_preferred_mode_if_vendor(
        &self,
        client: &AcpClient,
        session: &NewSessionResult,
        hooks: &StreamTextHooks,
    ) {
        if let Some(ref ext) = self.vendor_ext
            && let Some(vs) = hooks.vendor_state_snapshot()
        {
            ext.try_apply_preferred_mode(client, session, vs.as_ref());
        }
    }

    /// Spawn the configured ACP server command, attaches stdio, `initialize`,
    /// and vendor-specific `authenticate` — no `session/new` yet.
    fn spawn_acp_connection(&self) -> Result<(Arc<AcpClient>, Child, Arc<StreamTextHooks>)> {
        let program = self
            .acp_command
            .first()
            .map(String::as_str)
            .unwrap_or("agent");
        let mut cmd = build_acp_spawn_command(&self.acp_command, &self.acp_env)
            .with_context(|| format!("build ACP spawn command for {program}"))?;

        let mut child = cmd
            .current_dir(&self.working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn `{program}` ACP process"))?;

        let stderr = child.stderr.take();
        if let Some(err) = stderr {
            let aid = self.agent_id().to_string();
            thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(err);
                for line in reader.lines().map_while(|l| l.ok()) {
                    // Surfaced at warn level so auth-provider OAuth URLs and
                    // other user-actionable notices appear live in the TUI.
                    warn!(target: "potlatch::agent_stderr", agent_id = %aid, "stderr: {}", line);
                }
            });
        }

        let hooks = Arc::new(StreamTextHooks::with_workspace(PathBuf::from(
            self.working_dir.clone(),
        )));
        let (client, child) = AcpClient::from_child_stdio(child, hooks.clone())
            .context("attach ACP client to agent stdio")?;
        let client = Arc::new(client);
        if let Some(ref ext) = self.vendor_ext {
            let provider = self.capability_provider.lock().unwrap().clone();
            hooks.set_vendor_state(Some(ext.create_state(provider, self.agent_bus.clone())));
        }

        let init_result = client
            .initialize(&InitializeParams {
                protocol_version: DEFAULT_PROTOCOL_VERSION,
                client_capabilities: ClientCapabilities {
                    fs: ClientFsCapabilities {
                        read_text_file: true,
                        write_text_file: false,
                    },
                    terminal: false,
                },
                client_info: ImplementationInfo {
                    name: APP_NAME.into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
            })
            .context("ACP initialize")?;

        if let Some(ref ext) = self.vendor_ext {
            ext.authenticate(&client, &init_result)?;
        } else {
            debug!(
                target: "potlatch::acp",
                agent_id = %self.agent_id,
                "no vendor extension; skipping authenticate"
            );
        }

        Ok((client, child, hooks))
    }

    fn start_acp_session_on_connection(
        &self,
        client: &Arc<AcpClient>,
        hooks: &Arc<StreamTextHooks>,
    ) -> Result<String> {
        let cwd: PathBuf = std::fs::canonicalize(&self.working_dir)
            .unwrap_or_else(|_| PathBuf::from(&self.working_dir));
        let session = client
            .session_new(&NewSessionParams {
                cwd: cwd.to_string_lossy().into_owned(),
                agent_id: Some(self.agent_id.clone()),
                mcp_servers: vec![],
                structured_output_tools: self.session_structured_output_tools(),
                agent_tools: self.session_agent_tools(),
                context_channels: self.session_context_channels(),
                write_roots: (!self.write_roots.is_empty()).then(|| self.write_roots.clone()),
            })
            .context("ACP session/new")?;

        if let Some(ref modes) = session.modes {
            hooks.seed_session_modes(modes);
            if !mode_id_is_available(modes, &modes.current_mode_id) {
                debug!(
                    target: "potlatch::acp_modes",
                    agent_id = %self.agent_id,
                    current = %modes.current_mode_id,
                    "ACP session/new currentModeId not listed in availableModes",
                );
            }
            debug!(
                target: "potlatch::acp_modes",
                agent_id = %self.agent_id,
                current = %modes.current_mode_id,
                available = ?modes
                    .available_modes
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>(),
                "ACP session modes from session/new",
            );
        }

        self.try_apply_preferred_mode_if_vendor(client, &session, hooks);

        if let Some(model) = &self.endpoint_model {
            let mut applied = false;
            if let Some(cfg) = session.config_options.as_deref()
                && let Some(opt) = model_selector_for_session(cfg)
                && select_option_allows_value(opt, model)
            {
                match client.session_set_config_option(&session.session_id, &opt.id, model) {
                    Ok(_) => {
                        info!(
                            "ACP model set via session/set_config_option for agent {}",
                            self.agent_id
                        );
                        applied = true;
                    }
                    Err(e) => debug!(
                        "ACP session/set_config_option failed for {}, trying session/set_model: {}",
                        self.agent_id, e
                    ),
                }
            }
            if !applied {
                match client.session_set_model(&session.session_id, model) {
                    Ok(_) => info!(
                        "ACP model set via session/set_model for agent {}",
                        self.agent_id
                    ),
                    Err(e) => debug!(
                        "ACP session/set_model model={model:?} skipped for {}: {} (CLI --model still in effect)",
                        self.agent_id, e
                    ),
                }
            }
        }

        info!(
            "ACP session {} ready for agent {}",
            session.session_id, self.agent_id
        );

        Ok(session.session_id)
    }

    /// At the start of each task: new `session/new` on the existing child, or spawn if needed.
    fn rotate_acp_session_for_new_task(&self) -> Result<()> {
        let mut g = self.acp.lock().unwrap();
        let need_fresh_spawn = match g.as_mut() {
            None => true,
            Some(s) => match s.child.try_wait() {
                Ok(Some(_)) => {
                    if let Some(s) = g.take() {
                        self.notify_caller_session_closed(&s.session_id);
                        dispose_acp_session(s);
                    }
                    true
                }
                Ok(None) => false,
                Err(_) => {
                    if let Some(s) = g.take() {
                        self.notify_caller_session_closed(&s.session_id);
                        dispose_acp_session(s);
                    }
                    true
                }
            },
        };

        if need_fresh_spawn {
            let (client, child, hooks) = self.spawn_acp_connection()?;
            let session_id = self.start_acp_session_on_connection(&client, &hooks)?;
            *g = Some(AcpSession {
                client,
                child,
                session_id,
                hooks,
            });
            return Ok(());
        }

        let s = g
            .as_mut()
            .context("ACP session missing after child check")?;
        close_acp_session_best_effort(&s.client, &s.session_id);
        self.notify_caller_session_closed(&s.session_id);
        s.session_id = self.start_acp_session_on_connection(&s.client, &s.hooks)?;
        Ok(())
    }

    /// After a failed prompt while the child may have exited: respawn + `session/new` if needed.
    /// If the child is still running, the current session is left in place for the next attempt.
    fn ensure_acp_session_for_retry(&self) -> Result<()> {
        let mut g = self.acp.lock().unwrap();
        let need_spawn = match g.as_mut() {
            None => true,
            Some(s) => match s.child.try_wait() {
                Ok(Some(_)) => {
                    if let Some(s) = g.take() {
                        self.notify_caller_session_closed(&s.session_id);
                        dispose_acp_session(s);
                    }
                    true
                }
                Ok(None) => false,
                Err(_) => {
                    if let Some(s) = g.take() {
                        self.notify_caller_session_closed(&s.session_id);
                        dispose_acp_session(s);
                    }
                    true
                }
            },
        };

        if need_spawn {
            let (client, child, hooks) = self.spawn_acp_connection()?;
            let session_id = self.start_acp_session_on_connection(&client, &hooks)?;
            *g = Some(AcpSession {
                client,
                child,
                session_id,
                hooks,
            });
        }

        Ok(())
    }

    fn take_child_exit_status(&self) -> Result<Option<String>> {
        let mut g = self.acp.lock().unwrap();
        let Some(s) = g.as_mut() else {
            return Ok(Some("ACP child missing".to_string()));
        };
        match s.child.try_wait() {
            Ok(Some(status)) => Ok(Some(status.to_string())),
            Ok(None) => Ok(None),
            Err(err) => Ok(Some(format!("unable to query child status: {}", err))),
        }
    }
}

/// Whether an error from one prompt round trip is the transport giving out
/// (child gone, `session/prompt` failed) rather than a task-level outcome.
/// Only these are worth resending the same prompt for.
fn is_transport_failure(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.contains("exited during prompt") || message.contains("ACP session/prompt")
}

/// The error detail when a prompt result reports `stopReason: "error"` — the
/// agent loop itself failed (e.g. the LLM endpoint stayed unavailable after
/// the agent's own in-process retries) rather than completing a turn. No
/// repair prompt can fix that, so the caller must fail the task instead of
/// treating the result as a handoff whose structured output is merely
/// missing. Returns `None` for every other stop reason.
fn agent_loop_error_message(pr: &PromptResult) -> Option<String> {
    if !pr.stop_reason.eq_ignore_ascii_case("error") {
        return None;
    }
    Some(
        pr.extra
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_default(),
    )
}

fn collect_text_fragments_from_value(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_text_fragments_from_value(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                let t = text.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
            }
            for key in [
                "message", "output", "response", "content", "contents", "messages", "parts",
                "blocks", "items",
            ] {
                if let Some(child) = map.get(key) {
                    collect_text_fragments_from_value(child, out);
                }
            }
        }
        _ => {}
    }
}

fn final_text_from_prompt_extra(extra: &Map<String, Value>) -> Option<String> {
    for key in ["message", "output", "response", "text"] {
        if let Some(v) = extra.get(key)
            && let Some(s) = v.as_str()
        {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }

    let mut pieces: Vec<String> = Vec::new();
    for key in ["message", "output", "response", "content", "messages"] {
        if let Some(v) = extra.get(key) {
            collect_text_fragments_from_value(v, &mut pieces);
        }
    }
    pieces.dedup();
    let joined = pieces.join("\n\n").trim().to_string();
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn handoff_from_prompt_hooks(
    hooks: &StreamTextHooks,
    pr: PromptResult,
    vendor_ext: Option<&Arc<dyn AcpVendorExtension>>,
    structured_output_backend: Option<&dyn StructuredOutputBackend>,
) -> AgentHandoff {
    let stream = hooks.take_text();
    let final_text = final_text_from_prompt_extra(&pr.extra);

    // Prefer final prompt result text (`message` / `output`) over streamed chunks.
    // Streamed chunks can contain intermediate progress narration, while `extra` carries
    // the end-of-turn canonical answer that downstream parsers should consume.
    let mut response = if let Some(s) = final_text {
        s.to_string()
    } else {
        stream
    };

    // Let the vendor extension merge its data into the response.
    if let Some(ext) = vendor_ext
        && let Some(vs) = hooks.vendor_state_snapshot()
    {
        ext.process_response(vs.as_ref(), &mut response);
    }

    // Extract captured structured-output tool calls (e.g. `handoff`, `plan`).
    // A JSON object mapping tool name to captured args; `None` when no
    // structured-output tools were registered or the backend didn't include
    // the field.
    let structured_outputs = pr
        .extra
        .get("structured_outputs")
        .filter(|value| value.as_object().is_some_and(|outputs| !outputs.is_empty()))
        .cloned()
        .or_else(|| {
            structured_output_backend.and_then(|backend| backend.extract_outputs(&response))
        });

    AgentHandoff {
        response,
        structured_outputs,
    }
}

fn prepare_task_prompt(prompt: &str) -> String {
    format!("{}{}", TASK_CONTEXT_RESET_GUIDANCE, prompt)
}

fn retry_backoff_for_unfinished_task(attempt: u32) -> Duration {
    let capped = attempt.min(5) as u64;
    Duration::from_millis(capped * 200)
}

impl Drop for AcpRuntime {
    fn drop(&mut self) {
        let mut g = self.acp.lock().unwrap();
        if let Some(s) = g.take() {
            self.notify_caller_session_closed(&s.session_id);
            dispose_acp_session(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::AcpHooks;
    use super::*;
    use crate::core::bus::AgentToolDefinition;
    use serde_json::json;

    fn test_runtime_with_model(model_uri: Option<&str>) -> AcpRuntime {
        AcpRuntime::new(
            "/tmp/repo".into(),
            model_uri.map(str::to_string),
            None,
            vec!["true".into()],
            std::collections::HashMap::new(),
            None,
            None,
            Vec::new(),
            Arc::new(AtomicBool::new(false)),
            "worker-0".into(),
        )
    }

    fn test_runtime() -> AcpRuntime {
        test_runtime_with_model(None)
    }

    fn test_tool() -> AgentToolDefinition {
        AgentToolDefinition {
            name: "web_search".to_string(),
            description: "Search.".to_string(),
            parameters: json!({"type": "object"}),
            operation: "run".to_string(),
        }
    }

    #[test]
    fn potlatch_tasks_register_their_contract_for_the_sessions_they_need() {
        let runtime = test_runtime_with_model(Some("acp://potlatch/test"));
        assert_eq!(runtime.session_structured_output_tools(), None);

        let handoff = json!({"name": "handoff", "description": "d", "parameters": {}});
        runtime.register_task_contract(vec![handoff.clone()]);
        // Every session this task creates — including one recreated by a
        // transport retry, and the one a repair prompt keeps using — carries
        // the same contract, because nothing but a new task replaces it.
        assert_eq!(
            runtime.session_structured_output_tools(),
            Some(vec![handoff])
        );

        let review = json!({"name": "review", "description": "d", "parameters": {}});
        runtime.register_task_contract(vec![review.clone()]);
        assert_eq!(
            runtime.session_structured_output_tools(),
            Some(vec![review])
        );
    }

    #[test]
    fn default_vendor_keeps_contract_out_of_session_new() {
        let runtime = test_runtime();
        runtime.register_task_contract(vec![json!({
            "name": "handoff",
            "description": "d",
            "parameters": {}
        })]);

        assert_eq!(runtime.session_structured_output_tools(), None);
        assert!(
            runtime
                .prompt_with_structured_output("Do the task.")
                .contains("POTLATCH_STRUCTURED_OUTPUT_BEGIN")
        );
    }

    #[test]
    fn agent_tools_are_sent_only_to_the_potlatch_vendor() {
        let bus = AgentBus::new();
        let _inbox = bus.register("web", vec![test_tool()]).unwrap();
        let runtime_for = |model: &str| {
            AcpRuntime::new(
                "/tmp/repo".into(),
                Some(model.to_string()),
                None,
                vec!["true".into()],
                std::collections::HashMap::new(),
                None,
                Some(bus.clone()),
                Vec::new(),
                Arc::new(AtomicBool::new(false)),
                "worker-0".into(),
            )
        };

        assert_eq!(
            runtime_for("acp://potlatch/model")
                .session_agent_tools()
                .unwrap()
                .len(),
            1
        );
        assert!(
            runtime_for("acp://cursor/model")
                .session_agent_tools()
                .is_none()
        );
        assert!(runtime_for("model").session_agent_tools().is_none());
    }

    #[test]
    fn prepends_fresh_context_guidance_to_each_task() {
        let prompt = prepare_task_prompt("Implement issue #42.");
        assert!(prompt.starts_with(TASK_CONTEXT_RESET_GUIDANCE));
        assert!(prompt.ends_with("Implement issue #42."));
    }

    #[test]
    fn unfinished_task_retry_backoff_is_capped() {
        assert_eq!(
            retry_backoff_for_unfinished_task(1),
            Duration::from_millis(200)
        );
        assert_eq!(
            retry_backoff_for_unfinished_task(3),
            Duration::from_millis(600)
        );
        assert_eq!(
            retry_backoff_for_unfinished_task(99),
            Duration::from_millis(1000)
        );
    }

    #[test]
    fn only_transport_errors_are_worth_resending_the_prompt_for() {
        assert!(is_transport_failure(&anyhow::anyhow!(
            "agent worker-0 exited during prompt (signal: 9)"
        )));
        assert!(is_transport_failure(&anyhow::anyhow!(
            "ACP session/prompt: broken pipe"
        )));
        assert!(!is_transport_failure(&anyhow::anyhow!(
            "Agent cancelled by external condition"
        )));
        assert!(!is_transport_failure(&anyhow::anyhow!(
            "Agent interrupted by shutdown"
        )));
    }

    #[test]
    fn agent_loop_error_results_are_not_worth_resending_the_prompt_for() {
        // An agent loop error is the endpoint failing, not the transport:
        // resending the same prompt would re-hit the same dead endpoint.
        assert!(!is_transport_failure(&anyhow::anyhow!(
            "ACP agent loop error: LLM request failed (503 Service Unavailable)"
        )));
    }

    #[test]
    fn prompt_results_reporting_agent_loop_errors_yield_their_message() {
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "error",
            "message": "LLM request failed (503 Service Unavailable)"
        }))
        .unwrap();
        assert_eq!(
            agent_loop_error_message(&pr).as_deref(),
            Some("LLM request failed (503 Service Unavailable)")
        );

        let missing_detail: PromptResult =
            serde_json::from_value(json!({"stopReason": "error"})).unwrap();
        assert_eq!(
            agent_loop_error_message(&missing_detail).as_deref(),
            Some("")
        );

        let end_turn: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "done"
        }))
        .unwrap();
        assert_eq!(agent_loop_error_message(&end_turn), None);

        let aborted: PromptResult =
            serde_json::from_value(json!({"stopReason": "aborted"})).unwrap();
        assert_eq!(agent_loop_error_message(&aborted), None);
    }

    #[test]
    fn handoff_prefers_prompt_extra_over_stream_buffer() {
        let hooks = StreamTextHooks::new();
        let params = serde_json::json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "from stream" }
            }
        });
        hooks.on_agent_notification("session/update", &params);
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "from result"
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert_eq!(h.response, "from result");

        let hooks_m = StreamTextHooks::new();
        hooks_m.on_agent_notification("session/update", &params);
        let pr_m: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "SUB_ISSUE_1:\nTITLE: T\nDESCRIPTION:\nD"
        }))
        .unwrap();
        let hm = handoff_from_prompt_hooks(&hooks_m, pr_m, None, None);
        assert_eq!(hm.response, "SUB_ISSUE_1:\nTITLE: T\nDESCRIPTION:\nD");

        let hooks2 = StreamTextHooks::new();
        let pr2: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "only result"
        }))
        .unwrap();
        let h2 = handoff_from_prompt_hooks(&hooks2, pr2, None, None);
        assert_eq!(h2.response, "only result");

        let hooks3 = StreamTextHooks::new();
        hooks3.on_agent_notification("session/update", &params);
        let pr3: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn"
        }))
        .unwrap();
        let h3 = handoff_from_prompt_hooks(&hooks3, pr3, None, None);
        assert_eq!(h3.response, "from stream");
    }

    #[test]
    fn handoff_extracts_final_text_from_structured_content_blocks() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "content": [
                {"type": "text", "text": "SUB_ISSUE_1:"},
                {"type": "text", "text": "TITLE: Refactor queue"},
                {"type": "text", "text": "DESCRIPTION:\nDo the refactor."}
            ]
        }))
        .unwrap();
        let out = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert!(out.response.contains("SUB_ISSUE_1:"));
        assert!(out.response.contains("TITLE: Refactor queue"));
    }

    #[test]
    fn handoff_extracts_structured_outputs_from_prompt_extra() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "done",
            "structured_outputs": {"plan": {"decision": "split", "sub_issues": [{"title": "A"}]}}
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert_eq!(
            h.structured_outputs,
            Some(json!({"plan": {"decision": "split", "sub_issues": [{"title": "A"}]}}))
        );
    }

    #[test]
    fn handoff_extracts_default_backend_marker_output() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "done\nPOTLATCH_STRUCTURED_OUTPUT_BEGIN\n\
                {\"plan\":{\"decision\":\"split\",\"sub_issues\":[]}}\n\
                POTLATCH_STRUCTURED_OUTPUT_END"
        }))
        .unwrap();
        let backend = super::super::backends::resolve_structured_output_backend(None);
        let handoff = handoff_from_prompt_hooks(&hooks, pr, None, Some(backend.as_ref()));

        assert_eq!(
            handoff.structured_outputs,
            Some(json!({"plan": {"decision": "split", "sub_issues": []}}))
        );
    }

    #[test]
    fn handoff_structured_outputs_none_when_absent() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult =
            serde_json::from_value(json!({"stopReason": "end_turn", "message": "done"})).unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert!(h.structured_outputs.is_none());
    }

    #[test]
    fn handoff_structured_outputs_none_when_null() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "done",
            "structured_outputs": null
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert!(h.structured_outputs.is_none());
    }

    #[test]
    fn handoff_structured_outputs_none_when_empty() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "done",
            "structured_outputs": {}
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr, None, None);
        assert!(h.structured_outputs.is_none());
    }
}
