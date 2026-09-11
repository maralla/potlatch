//! Subagent agent: maintains ALL subagent sessions for the platform.
//!
//! Bus-served: exposes a `subagent` tool that other agents' harness sessions
//! call to spawn, poll, message, inject into, and kill subagents.
//!
//! One ACP client per vendor: the agent groups the configured
//! `provided_models` (full model URIs) by vendor and runs one child process
//! per vendor — the platform's own `potlatch harness` for the potlatch
//! vendor, the profile's `acp_command` for others. Every subagent is an
//! independent ACP session on its vendor's client (`session/new` with
//! `multiplex: true`), so N subagents cost one process per vendor, not N.
//! Sessions run concurrently; killing a subagent closes its session, which
//! tears down its context and background jobs.
//!
//! Transcripts land under `<current-directory>/.potlatch/<agent-name>/
//! transcripts/`, next to the project the platform runs in.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::bus::{AgentInbox, AgentToolDefinition, ContextChannelGuard, ContextProvider};
use crate::core::config::uri::ModelUri;
use crate::core::config::{
    AUTH_COMMAND_DIR_ENV, AUTH_COMMAND_ENV, AcpSpawnConfig, Config, POTLATCH_ACP_PROFILE,
    build_acp_spawn_command,
};
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;
use crate::paths::{APP_NAME, display_name};
use crate::util::dies_with_parent;

const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
const SUBAGENT_OPERATION: &str = "subagent";
const AGENT_NAME: &str = "subagent";

/// Context channel publishing the configured subagent models to callers'
/// sessions (rendered as a `## subagent models` system message).
const MODELS_CONTEXT_CHANNEL: &str = "subagent models";

/// Accumulated output kept per subagent session; past the cap the oldest
/// bytes are dropped (the newest output is what a poll must surface).
const MAX_OUTPUT: usize = 50_000;

/// Accumulated output up to this size is inlined in a poll result. Larger
/// output is not: the caller's tool-result cap would behead it anyway, so
/// the poll directs the model to the transcript file — the durable full
/// record — instead. The threshold keeps header + output under the
/// caller's tool-result window.
const POLL_INLINE_MAX_BYTES: usize = 3_000;

/// How long a control-plane request (session/new, session/close, ...) may
/// take before the tool call fails. Bus callers enforce 45s themselves.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SubagentAgentSettings {
    /// Model ids callers may request for subagent sessions. Published to
    /// callers via the tool description and a context channel; every spawn
    /// must pass one of them.
    #[serde(default)]
    provided_models: Vec<String>,
    /// Upper bound on live (non-killed) subagent sessions. Absent = no cap:
    /// the shared harness process and the caller-retirement reaping bound
    /// the footprint, so unlimited is safe by default. Set it to keep a
    /// tighter leash on concurrent subagent work.
    #[serde(default)]
    max_live_subagents: Option<usize>,
}

pub struct SubagentAgent {
    runtime: AgentRuntime,
    inbox: AgentInbox,
    hub: SubagentHub,
    /// Retired caller task sessions: each id names a caller whose subagents
    /// must be closed.
    lifecycle: mpsc::Receiver<String>,
    /// Keeps the models context channel registered while the agent lives.
    _models_context: ContextChannelGuard,
}

impl CoreAgent for SubagentAgent {
    type Settings = SubagentAgentSettings;
    const FIXED_INSTANCES: Option<usize> = Some(1);

    fn name() -> &'static str {
        AGENT_NAME
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: REQUEST_TASK,
            interval: Duration::ZERO,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 0,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        ensure!(task_id == REQUEST_TASK, "unknown subagent task {task_id:?}");
        // Retired caller task sessions first: their subagents are dead
        // weight — nobody will ever poll them again. Callers that never
        // spawned any produce no report and no log line.
        while let Ok(caller) = self.lifecycle.try_recv() {
            if let Some(report) = self.hub.close_caller_sessions(&caller) {
                info!("{report}");
            }
        }
        if let Some(request) = self.inbox.recv_timeout(INBOX_WAIT)? {
            let payload = request.payload.clone();
            let result = self
                .hub
                .dispatch(&payload, request.session_id.as_deref())
                .map(Value::String);
            request.respond(result);
        }
        Ok(())
    }

    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
        let bus = ctx
            .workflow
            .bus
            .as_ref()
            .context("subagent agent requires the cross-agent bus")?;

        // The models callers may pick from: published with the tool (so the
        // model sees the menu when calling it) and as a context channel (so
        // it is visible in every caller's session context).
        let models = ctx.settings.provided_models.clone();
        ensure!(
            !models.is_empty(),
            "[agent.subagent] requires a non-empty `provided_models` list — the models \
             callers may request for subagent sessions"
        );
        if let Some(max) = ctx.settings.max_live_subagents
            && max == 0
        {
            anyhow::bail!("[agent.subagent] `max_live_subagents` must be at least 1");
        }
        let max_live_subagents = ctx.settings.max_live_subagents;
        let inbox = bus.register(
            Self::name(),
            vec![subagent_tool_definition(&models, max_live_subagents)],
        )?;
        let models_for_context = models.clone();
        let models_context = bus.register_context(
            MODELS_CONTEXT_CHANNEL,
            Arc::new(move || models_context_content(&models_for_context)) as ContextProvider,
        );
        let lifecycle = bus.register_session_lifecycle_listener()?;

        // One ACP client per vendor in `provided_models` (full model URIs):
        // the potlatch vendor runs the platform's own harness binary; every
        // other vendor runs its profile's `acp_command`. No model of its own
        // — callers pick one per spawn.
        let config = &ctx.workflow.config;
        let base_dir = ctx.workflow.base_dir.clone();
        let hub = SubagentHub::spawn_children(&base_dir, config, &models, max_live_subagents)?;

        info!("subagent agent ready (harness multiplexing subagents, base dir {base_dir})");
        Ok(Self {
            runtime: ctx.runtime.clone(),
            inbox,
            hub,
            lifecycle,
            _models_context: models_context,
        })
    }
}

/// The context-channel content publishing the configured subagent models:
/// rendered as a `## subagent models` system message in callers' sessions.
fn models_context_content(models: &[String]) -> String {
    format!(
        "When spawning a subagent with the `subagent` tool, the `model` argument is required \
         and must be one of: {}.",
        models.join(", ")
    )
}

/// The `subagent` tool other agents call through the bus. One tool, one
/// operation; the argument shape dispatches (spawn / poll / message /
/// inject / kill), matching the tool schema the model sees. The configured
/// `models` are embedded so the caller knows what it may pass.
fn subagent_tool_definition(
    models: &[String],
    max_live_subagents: Option<usize>,
) -> AgentToolDefinition {
    // The capacity sentence depends on the operator's configuration: a cap
    // must be visible to the caller (spawns start failing at it); without a
    // cap, hygiene is the caller's job — killing is what releases resources.
    let capacity = match max_live_subagents {
        Some(max) => format!(
            "At most {max} subagents can be live at once: when the cap is reached, spawn \
             fails until you kill a finished one."
        ),
        None => "There is no hard limit on how many subagents can be live at once.".to_string(),
    };
    AgentToolDefinition {
        name: "subagent".to_string(),
        description: format!(
            "Spawn a subagent — a separate {d} harness session with its own LLM context, \
             run by the platform's subagent agent. The subagent runs asynchronously in the \
             background. Returns a subagent_id immediately; poll with subagent_id to get \
             accumulated output (short output is returned in place; large output is not \
             inlined — read the reported transcript file instead). Use for parallel \
             exploration, independent research tasks, or \
             dividing complex work. The subagent has no context from the parent session — \
             provide everything it needs in the prompt. The `model` argument is required when \
             spawning and must be one of the configured subagent models: {models}. Send \
             follow-up messages to a subagent with subagent_id + message (a new turn; context \
             persists), or redirect a running one mid-turn with subagent_id + inject. \
             {capacity} Kill a subagent (subagent_id + kill) once you are done with it: \
             killing releases its context and background jobs. Subagents you leave behind \
             are closed when your task session ends.",
            d = display_name(),
            models = models.join(", "),
            capacity = capacity
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task prompt for the subagent. Be specific — the subagent has no context from the parent session."
                },
                "model": {
                    "type": "string",
                    "description": format!(
                        "Required when spawning a subagent. One of the configured subagent models: {}.",
                        models.join(", ")
                    )
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names the subagent can use (e.g. ['read','grep','shell']). Omit for all tools."
                },
                "subagent_id": {
                    "type": "string",
                    "description": "A subagent id. Use with message to send a follow-up, with inject for mid-run redirection, with kill to terminate, or alone to poll for accumulated output."
                },
                "message": {
                    "type": "string",
                    "description": "Send a follow-up message to the subagent as a new turn. Context persists across turns. Poll with subagent_id to read the response."
                },
                "inject": {
                    "type": "string",
                    "description": "Inject a message into a running subagent's context mid-turn — it redirects the agent without starting a new turn. Poll with subagent_id to read the response."
                },
                "kill": {
                    "type": "boolean",
                    "description": "When true with subagent_id, terminate the subagent and release its resources."
                }
            },
            "required": []
        }),
        operation: SUBAGENT_OPERATION.to_string(),
    }
}

/// The status of a subagent session's current (or last) turn.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    /// A turn is in flight.
    Running,
    /// The last turn finished normally.
    Done,
    /// The last turn failed.
    Error(String),
    /// The subagent was killed: its harness session is closed and its
    /// resources released; only a tombstone remains.
    Killed,
}

/// One subagent session: an ACP session inside the ACP client of its
/// vendor.
struct SubagentSession {
    /// The vendor whose ACP client serves this session.
    vendor: String,
    /// The ACP session id inside that client. `None` once killed — the
    /// tombstone then only remembers the status so later polls answer
    /// sanely.
    session_id: Option<String>,
    /// The calling harness session that spawned this subagent (the harness
    /// injects its session id into every bus tool payload). All of a
    /// caller's subagents are closed when its task session retires.
    owner: Option<String>,
    started_at: Instant,
    /// Transcript file — only for ACP clients that speak the platform's
    /// transcript extension (the potlatch harness).
    transcript_path: Option<PathBuf>,
    /// Accumulated output of all turns so far (capped at [`MAX_OUTPUT`]).
    output: Mutex<String>,
    status: Mutex<Status>,
}

/// State shared between the hub (dispatch) and the harness reader thread.
struct HubShared {
    sessions: Mutex<HashMap<String, SubagentSession>>,
    next_session_num: AtomicU64,
}

impl HubShared {
    /// Route a `session/update` notification's text to the session it names
    /// (the ACP client includes the sessionId in the notification params).
    fn append_output(&self, vendor: &str, session_id: &str, text: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions
            .values_mut()
            .find(|s| s.vendor == vendor && s.session_id.as_deref() == Some(session_id))
        {
            append_capped(&mut session.output.lock().unwrap(), text);
        }
    }

    /// A `session/prompt` response arrived: record the turn's output and
    /// terminal status for the subagent it belongs to.
    fn complete_turn(&self, subagent_id: &str, response: &Value) {
        let mut sessions = self.sessions.lock().unwrap();
        let Some(session) = sessions.get_mut(subagent_id) else {
            debug!("subagent hub: turn completion for unknown subagent {subagent_id}");
            return;
        };
        let status = if let Some(err) = response.get("error") {
            Status::Error(format!("harness error: {err}"))
        } else {
            let result = response.get("result").cloned().unwrap_or(Value::Null);
            if let Some(message) = result.get("message").and_then(Value::as_str)
                && !message.is_empty()
            {
                append_capped(&mut session.output.lock().unwrap(), message);
            }
            match result.get("stopReason").and_then(Value::as_str) {
                Some("error") => Status::Error(
                    result
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                        .to_string(),
                ),
                _ => Status::Done,
            }
        };
        *session.status.lock().unwrap() = status;
    }
}

/// Request demultiplexing state shared between the transport (writer side)
/// and the stdout reader thread.
#[derive(Default)]
struct Demux {
    /// Control-plane request id → responder.
    pending: Mutex<HashMap<u64, mpsc::SyncSender<Result<Value, String>>>>,
    /// Fired prompt request id → subagent id.
    prompts: Mutex<HashMap<u64, String>>,
}

impl Demux {
    fn register_pending(&self, id: u64, sender: mpsc::SyncSender<Result<Value, String>>) {
        self.pending.lock().unwrap().insert(id, sender);
    }

    fn take_pending(&self, id: u64) -> Option<mpsc::SyncSender<Result<Value, String>>> {
        self.pending.lock().unwrap().remove(&id)
    }

    fn register_prompt(&self, id: u64, subagent_id: String) {
        self.prompts.lock().unwrap().insert(id, subagent_id);
    }

    fn take_prompt(&self, id: u64) -> Option<String> {
        self.prompts.lock().unwrap().remove(&id)
    }
}

/// How the hub talks to one ACP client (one child process per vendor).
/// Production spawns the child processes; tests substitute scripted fakes.
trait HarnessTransport: Send + Sync + 'static {
    /// Send a control-plane request and wait for its response.
    fn request(&self, method: &str, params: Value) -> Result<Value>;
    /// Fire a turn: send `session/prompt` without waiting for the response;
    /// the reader routes the response back to the subagent id.
    fn fire_prompt(&self, subagent_id: &str, session_id: &str, prompt: &str) -> Result<()>;
    /// End the client. The hub closes live sessions first; this releases
    /// the process itself.
    fn shutdown(&self);
    /// Whether the client speaks the platform's ACP extensions (multiplex
    /// `session/new`, transcript files, `session/inject`). Only the
    /// potlatch harness does; other vendors get the plain protocol.
    fn supports_extensions(&self) -> bool {
        false
    }
}

impl<T: HarnessTransport + ?Sized> HarnessTransport for Arc<T> {
    fn request(&self, method: &str, params: Value) -> Result<Value> {
        (**self).request(method, params)
    }

    fn fire_prompt(&self, subagent_id: &str, session_id: &str, prompt: &str) -> Result<()> {
        (**self).fire_prompt(subagent_id, session_id, prompt)
    }

    fn shutdown(&self) {
        (**self).shutdown();
    }

    fn supports_extensions(&self) -> bool {
        (**self).supports_extensions()
    }
}

/// The hub: one ACP client per vendor, many subagent sessions.
struct SubagentHub {
    /// ACP clients keyed by vendor — one child process per vendor present in
    /// `provided_models`.
    children: HashMap<String, Arc<dyn HarnessTransport>>,
    shared: Arc<HubShared>,
    /// The provided models (full URIs) callers may request; every spawn must
    /// pass one.
    provided_models: Vec<String>,
    /// Live-subagent cap configured by the operator. `None` = unlimited.
    max_live_subagents: Option<usize>,
    /// Bare alias → real model name (from the vendor profile's endpoint
    /// entries). A spawn addressed by an alias sends the real name to the
    /// backend via `session/set_model`.
    model_aliases: HashMap<String, String>,
    base_dir: String,
}

impl SubagentHub {
    /// Spawn the production ACP clients: one child per vendor found in
    /// `provided_models` (full model URIs). Each child serves that vendor's
    /// subagent sessions.
    fn spawn_children(
        base_dir: &str,
        config: &Config,
        provided_models: &[String],
        max_live_subagents: Option<usize>,
    ) -> Result<Self> {
        let shared = Arc::new(HubShared {
            sessions: Mutex::new(HashMap::new()),
            next_session_num: AtomicU64::new(1),
        });

        // Group the provided models by vendor — one client each.
        let mut by_vendor: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for model in provided_models {
            let uri = ModelUri::parse(model)
                .with_context(|| format!("invalid provided model '{model}'"))?;
            by_vendor.entry(uri.vendor).or_default().push(model.clone());
        }

        let mut children = HashMap::new();
        let mut model_aliases = HashMap::new();
        for (vendor, models) in &by_vendor {
            let acp = config.resolve_client_spawn(vendor, models)?;
            let demux = Arc::new(Demux::default());
            let (transport, stdout) =
                AcpChildProcess::spawn(base_dir, &acp, demux.clone(), vendor.clone())?;
            // The reader must be running before the handshake: it reads the
            // child's stdout and delivers the initialize response to the
            // waiting request.
            let shared_for_reader = Arc::clone(&shared);
            let vendor_for_reader = vendor.clone();
            thread::Builder::new()
                .name(format!("subagent-reader-{vendor}"))
                .spawn(move || run_reader(stdout, demux, shared_for_reader, vendor_for_reader))
                .with_context(|| format!("spawn reader thread for vendor {vendor}"))?;
            if let Err(error) = transport.request("initialize", json!({})) {
                transport.shutdown();
                return Err(error)
                    .with_context(|| format!("harness ACP initialize failed for vendor {vendor}"));
            }
            children.insert(vendor.clone(), transport as Arc<dyn HarnessTransport>);
            model_aliases.extend(acp.model_aliases);
        }

        Ok(Self {
            children,
            shared,
            provided_models: provided_models.to_vec(),
            max_live_subagents,
            model_aliases,
            base_dir: base_dir.to_string(),
        })
    }

    /// Construct around existing transports (tests), keyed by vendor.
    #[cfg(test)]
    fn with_children(
        children: HashMap<String, Arc<dyn HarnessTransport>>,
        provided_models: Vec<String>,
        max_live_subagents: Option<usize>,
        base_dir: &str,
    ) -> Self {
        Self::with_children_and_aliases(
            children,
            provided_models,
            max_live_subagents,
            HashMap::new(),
            base_dir,
        )
    }

    /// Construct around existing transports (tests) with an alias map.
    #[cfg(test)]
    fn with_children_and_aliases(
        children: HashMap<String, Arc<dyn HarnessTransport>>,
        provided_models: Vec<String>,
        max_live_subagents: Option<usize>,
        model_aliases: HashMap<String, String>,
        base_dir: &str,
    ) -> Self {
        Self {
            children,
            shared: Arc::new(HubShared {
                sessions: Mutex::new(HashMap::new()),
                next_session_num: AtomicU64::new(1),
            }),
            provided_models,
            max_live_subagents,
            model_aliases,
            base_dir: base_dir.to_string(),
        }
    }

    /// The wire `modelId` for a `session/set_model` call: an aliased model
    /// segment is rewritten to the profile entry's real model name; the query
    /// suffix (e.g. `?effort=high`) is preserved either way.
    fn wire_model_id(&self, model_segment: &str) -> String {
        match model_segment.split_once('?') {
            Some((bare, query)) => {
                let wire = self
                    .model_aliases
                    .get(bare)
                    .map(String::as_str)
                    .unwrap_or(bare);
                format!("{wire}?{query}")
            }
            None => self
                .model_aliases
                .get(model_segment)
                .map(String::as_str)
                .unwrap_or(model_segment)
                .to_string(),
        }
    }

    /// The ACP client serving `vendor`'s subagent sessions.
    fn child_for(&self, vendor: &str) -> Result<Arc<dyn HarnessTransport>> {
        self.children
            .get(vendor)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no ACP client configured for vendor '{vendor}'"))
    }

    /// Dispatch a `subagent` tool call: spawn, poll, message, inject, or
    /// kill, mirroring the tool's argument shape. `owner` is the calling
    /// harness session (bus request metadata) — recorded at spawn so the
    /// caller's subagents can be closed together when its task retires.
    fn dispatch(&self, payload: &Value, owner: Option<&str>) -> Result<String> {
        // An empty `subagent_id` counts as absent: models sometimes fill
        // every schema property with defaults, and an empty id would route
        // a fresh spawn request to a poll of a nonexistent subagent.
        let subagent_id = payload["subagent_id"]
            .as_str()
            .map(str::trim)
            .filter(|id| !id.is_empty());
        let kill = payload["kill"].as_bool().unwrap_or(false);
        let message = payload["message"].as_str().filter(|s| !s.is_empty());
        let inject = payload["inject"].as_str().filter(|s| !s.is_empty());

        if let Some(id) = subagent_id {
            if let Some(msg) = message {
                return self.message(id, msg);
            }
            if let Some(msg) = inject {
                return self.inject(id, msg);
            }
            if kill {
                return self.kill(id);
            }
            return self.poll(id);
        }

        let prompt = payload["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'prompt' argument"))?;
        let model = payload["model"].as_str().unwrap_or_default();
        let tools: Option<Vec<String>> = payload["tools"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty());
        self.spawn(prompt, model, tools.as_deref(), owner.map(str::to_string))
    }

    /// Start a subagent: create an ACP session in the shared harness, set
    /// its model, fire the prompt, and return the subagent id immediately.
    /// The model is required: callers pick from the configured
    /// `provided_models` list (published with the tool and in their
    /// session context).
    fn spawn(
        &self,
        prompt: &str,
        model: &str,
        tools: Option<&[String]>,
        owner: Option<String>,
    ) -> Result<String> {
        self.enforce_cap()?;
        ensure!(
            !model.is_empty(),
            "the 'model' argument is required when spawning a subagent; use one of: {}",
            self.provided_models.join(", ")
        );
        ensure!(
            self.provided_models.iter().any(|m| m == model),
            "unknown subagent model '{model}'; use one of: {}",
            self.provided_models.join(", ")
        );
        let uri =
            ModelUri::parse(model).with_context(|| format!("invalid subagent model '{model}'"))?;
        let child = self.child_for(&uri.vendor)?;

        let num = self.shared.next_session_num.fetch_add(1, Ordering::Relaxed);
        let id = format!("subagent-{num}");
        // Transcript files are a potlatch-harness extension; other vendors
        // run without one.
        let transcript_path = if child.supports_extensions() {
            let path = transcript_path(&self.base_dir, num);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("create subagent transcript directory {}", parent.display())
                })?;
            }
            fs::write(&path, "")
                .with_context(|| format!("initialize subagent transcript {}", path.display()))?;
            Some(path)
        } else {
            None
        };

        let mut params = json!({
            "cwd": self.base_dir,
            "mcpServers": [],
        });
        if child.supports_extensions() {
            // Many independent subagent sessions share this connection:
            // a session/new must not supersede the others.
            params["multiplex"] = json!(true);
            if let Some(path) = &transcript_path {
                params["transcript_path"] = json!(path.display().to_string());
            }
        }
        if let Some(tools) = tools {
            params["tools"] = json!(tools);
        }
        let result = child.request("session/new", params)?;
        let session_id = result["sessionId"]
            .as_str()
            .context("session/new response missing sessionId")?
            .to_string();

        child.request(
            "session/set_model",
            json!({
                "sessionId": session_id,
                // An aliased model is rewritten to the profile entry's real
                // model name; the query suffix (e.g. `?effort=high`) is
                // preserved either way.
                "modelId": self.wire_model_id(uri.endpoint_model_name()),
            }),
        )?;

        // Register before firing the prompt so notifications from the first
        // turn land in the table.
        self.shared.sessions.lock().unwrap().insert(
            id.clone(),
            SubagentSession {
                vendor: uri.vendor.clone(),
                session_id: Some(session_id.clone()),
                owner,
                started_at: Instant::now(),
                transcript_path,
                output: Mutex::new(String::new()),
                status: Mutex::new(Status::Running),
            },
        );
        if let Err(error) = child.fire_prompt(&id, &session_id, prompt) {
            if let Some(session) = self.shared.sessions.lock().unwrap().get_mut(&id) {
                *session.status.lock().unwrap() =
                    Status::Error(format!("prompt delivery failed: {error}"));
            }
            return Err(error);
        }

        Ok(format!(
            "Subagent started: {id}\nprompt: {}",
            prompt.chars().take(200).collect::<String>()
        ))
    }

    /// Enforce the operator's live-subagent cap, if one is configured:
    /// drop killed tombstones first (they hold no harness resources), then
    /// reject the spawn while the cap is still reached. `None` = unlimited.
    fn enforce_cap(&self) -> Result<()> {
        let Some(max) = self.max_live_subagents else {
            return Ok(());
        };
        let mut sessions = self.shared.sessions.lock().unwrap();
        sessions.retain(|_, session| *session.status.lock().unwrap() != Status::Killed);
        let live: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.session_id.is_some())
            .map(|(id, _)| id.clone())
            .collect();
        if live.len() >= max {
            anyhow::bail!(
                "subagent limit reached ({max} live: {}); kill a finished \
                 subagent before spawning another",
                live.join(", ")
            );
        }
        Ok(())
    }

    /// Poll a subagent: accumulated output and current status.
    fn poll(&self, subagent_id: &str) -> Result<String> {
        let (status, output, elapsed, transcript) = {
            let sessions = self.shared.sessions.lock().unwrap();
            let Some(session) = sessions.get(subagent_id) else {
                bail!("unknown subagent id: {subagent_id}");
            };
            (
                session.status.lock().unwrap().clone(),
                session.output.lock().unwrap().clone(),
                session.started_at.elapsed(),
                session.transcript_path.clone(),
            )
        };

        let status_line = match &status {
            Status::Running => "running".to_string(),
            Status::Done => "done".to_string(),
            Status::Error(_) => "error".to_string(),
            Status::Killed => "killed".to_string(),
        };
        let mut result = format!(
            "subagent: {subagent_id}\nstatus: {status_line}\nelapsed: {:.1}s",
            elapsed.as_secs_f64()
        );
        if let Some(transcript) = &transcript {
            result.push_str(&format!("\ntranscript: {}", transcript.display()));
        }
        if let Status::Error(err) = &status {
            result.push_str(&format!("\nerror: {err}"));
        }
        if !output.is_empty() {
            if output.len() <= POLL_INLINE_MAX_BYTES {
                result.push_str("\noutput:\n");
                result.push_str(&output);
            } else if let Some(transcript) = &transcript {
                // Large output stays out of band: the transcript file holds
                // the full durable record, and the model reads it with its
                // own tools (paged, only the parts it needs). The transcript
                // can be long, so also report where the final result lies.
                let location = std::fs::read_to_string(transcript)
                    .ok()
                    .and_then(|ref text| final_response_location(text))
                    .map(|(line, turn)| match turn {
                        0 => format!("the final response starts at line {line}"),
                        n => format!("the final response is turn {n}, starting at line {line}"),
                    })
                    .unwrap_or_else(|| {
                        "the final response is the last '## Assistant' section".to_string()
                    });
                result.push_str(&format!(
                    "\noutput: {} bytes accumulated — too large to inline here; \
                     read the transcript file for the full content: {}\n{}",
                    output.len(),
                    transcript.display(),
                    location
                ));
            } else {
                // No transcript (non-harness vendor): fall back to the tail,
                // the newest output is what a poll is for.
                let mut start = output.len() - POLL_INLINE_MAX_BYTES;
                while start > 0 && !output.is_char_boundary(start) {
                    start -= 1;
                }
                result.push_str(&format!(
                    "\noutput (newest {} of {} bytes):\n{}",
                    output.len() - start,
                    output.len(),
                    &output[start..]
                ));
            }
        }

        if result.len() > MAX_OUTPUT {
            let cut = truncate_at_char_boundary(&result, MAX_OUTPUT);
            result = format!(
                "{cut}\n\n[...output truncated, {} total chars...]",
                result.len()
            );
        }
        Ok(result)
    }

    /// Send a follow-up message to a running or finished subagent as a new
    /// turn. Context persists across turns inside the shared harness.
    fn message(&self, subagent_id: &str, message: &str) -> Result<String> {
        let (vendor, session_id) = self.live_routing(subagent_id)?;
        let child = self.child_for(&vendor)?;
        // Mark running before firing: a fast turn may complete before the
        // fire call returns, and Done must not be overwritten afterwards.
        if let Some(session) = self.shared.sessions.lock().unwrap().get_mut(subagent_id) {
            *session.status.lock().unwrap() = Status::Running;
        }
        if let Err(error) = child.fire_prompt(subagent_id, &session_id, message) {
            if let Some(session) = self.shared.sessions.lock().unwrap().get_mut(subagent_id) {
                *session.status.lock().unwrap() =
                    Status::Error(format!("message delivery failed: {error}"));
            }
            return Err(error);
        }
        Ok(subagent_id.to_string())
    }

    /// Inject a message into a running subagent's context mid-turn.
    fn inject(&self, subagent_id: &str, message: &str) -> Result<String> {
        let (vendor, session_id) = self.live_routing(subagent_id)?;
        let result = self.child_for(&vendor)?.request(
            "session/inject",
            json!({ "sessionId": session_id, "message": message }),
        )?;
        if let Some(err) = result.get("error").and_then(Value::as_str) {
            bail!("subagent {subagent_id} inject failed: {err}");
        }
        Ok(subagent_id.to_string())
    }

    /// Kill a subagent: close its harness session (which tears down its
    /// context and background jobs) and leave only a tombstone behind.
    fn kill(&self, subagent_id: &str) -> Result<String> {
        let (vendor, session_id) = {
            let mut sessions = self.shared.sessions.lock().unwrap();
            let Some(session) = sessions.get_mut(subagent_id) else {
                bail!("unknown subagent id: {subagent_id}");
            };
            let session_id = session.session_id.take();
            *session.status.lock().unwrap() = Status::Killed;
            // The output has nowhere to go once the session is closed; free
            // it instead of keeping it in the tombstone.
            session.output.lock().unwrap().clear();
            (session.vendor.clone(), session_id)
        };
        if let Some(session_id) = session_id {
            // Best effort: the client may already be gone.
            let _ = self
                .child_for(&vendor)?
                .request("session/close", json!({ "sessionId": session_id }));
        }
        Ok(format!("subagent: {subagent_id}\nstatus: killed"))
    }

    /// Close every live subagent spawned by `owner`. Called when the
    /// caller's task session retires: its task is over, so the subagents it
    /// spawned will never be polled again.
    fn close_caller_sessions(&self, owner: &str) -> Option<String> {
        let ids: Vec<String> = {
            let sessions = self.shared.sessions.lock().unwrap();
            sessions
                .iter()
                .filter(|(_, s)| s.owner.as_deref() == Some(owner) && s.session_id.is_some())
                .map(|(id, _)| id.clone())
                .collect()
        };
        for id in &ids {
            let _ = self.kill(id);
        }
        // The caller's task is over: its killed tombstones have no future
        // reader either — drop them so nothing accumulates across tasks.
        let dropped = {
            let mut sessions = self.shared.sessions.lock().unwrap();
            let before = sessions.len();
            sessions.retain(|_, s| s.owner.as_deref() != Some(owner));
            before - sessions.len()
        };
        // Most caller sessions never spawned a subagent: no report, no log
        // line for them.
        if ids.is_empty() && dropped == 0 {
            return None;
        }
        Some(format!(
            "closed {} subagent(s) of caller session {owner} ({} entries reclaimed)",
            ids.len(),
            dropped
        ))
    }

    /// The vendor and live ACP session id of a tracked subagent.
    fn live_routing(&self, subagent_id: &str) -> Result<(String, String)> {
        let sessions = self.shared.sessions.lock().unwrap();
        let Some(session) = sessions.get(subagent_id) else {
            bail!("unknown subagent id: {subagent_id}");
        };
        let session_id = session.session_id.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "subagent {subagent_id} has exited (killed); spawn a new subagent instead"
            )
        })?;
        Ok((session.vendor.clone(), session_id))
    }
}

impl Drop for SubagentHub {
    fn drop(&mut self) {
        for (_, child) in self.children.drain() {
            child.shutdown();
        }
    }
}

/// `<current-directory>/.potlatch/<agent-name>/transcripts/subagent-<n>.log`
fn transcript_path(base_dir: &str, num: u64) -> PathBuf {
    Path::new(base_dir)
        .join(format!(".{APP_NAME}"))
        .join(AGENT_NAME)
        .join("transcripts")
        .join(format!("subagent-{num}.log"))
}

/// Append `text` to the buffer, keeping at most [`MAX_OUTPUT`] bytes (the
/// tail — the newest output is what a poll must surface) and discarding the
/// oldest, on char boundaries.
fn append_capped(buf: &mut String, text: &str) {
    buf.push_str(text);
    if buf.len() <= MAX_OUTPUT {
        return;
    }
    const DROPPED_MARKER: &str = "[...earlier output dropped...]\n";
    let keep = MAX_OUTPUT - DROPPED_MARKER.len();
    let mut start = buf.len() - keep;
    while start > 0 && !buf.is_char_boundary(start) {
        start -= 1;
    }
    *buf = format!("{DROPPED_MARKER}{}", &buf[start..]);
}

fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}

/// Locate the final response in a subagent transcript: the last
/// `## Assistant` section (the harness writes one only for turns that
/// produced text, so the last one always carries content) plus the
/// `=== Turn N ===` marker enclosing it. Returns the 1-based line number of
/// the `## Assistant` header and the turn number (0 when the section sits
/// before any marker). Poll results point here so the model can read just
/// the final section of a long transcript instead of scanning it all.
fn final_response_location(transcript: &str) -> Option<(usize, u32)> {
    let mut last: Option<(usize, u32)> = None;
    let mut current_turn: Option<u32> = None;
    for (idx, line) in transcript.lines().enumerate() {
        if let Some(rest) = line.strip_prefix("=== Turn ") {
            if let Ok(n) = rest.trim_end_matches(" ===").trim().parse::<u32>() {
                current_turn = Some(n);
            }
        } else if line.trim() == "## Assistant" {
            last = Some((idx + 1, current_turn.unwrap_or(0)));
        }
    }
    last
}

/// Extract text from a `session/update` notification's
/// `agent_message_chunk` content.
fn extract_update_text(msg: &Value) -> Option<&str> {
    let update = msg.get("params")?.get("update")?;
    let kind = update.get("sessionUpdate").and_then(Value::as_str)?;
    if kind != "agent_message_chunk" {
        return None;
    }
    update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
}

/// Reader thread body: demultiplex the harness child's stdout. Responses
/// answer either a pending control-plane request or a fired prompt;
/// `session/update` notifications are routed to their session by the
/// sessionId the harness includes in the params.
fn run_reader(mut reader: impl BufRead, demux: Arc<Demux>, shared: Arc<HubShared>, vendor: String) {
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                warn!("subagent hub: harness stdout read error: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                debug!("subagent hub: skipping invalid JSON from harness: {e}");
                continue;
            }
        };

        if msg.get("method").is_some() {
            if msg.get("method").and_then(Value::as_str) == Some("session/update") {
                if let (Some(session_id), Some(text)) = (
                    msg.pointer("/params/sessionId").and_then(Value::as_str),
                    extract_update_text(&msg),
                ) && !text.is_empty()
                {
                    shared.append_output(&vendor, session_id, text);
                }
            } else {
                // Agent→client requests are not expected from the harness;
                // drop them (a missing permission reply is treated as deny).
                debug!("subagent hub: dropping harness message: {}", msg["method"]);
            }
            continue;
        }

        let Some(id) = msg.get("id").and_then(Value::as_u64).or_else(|| {
            msg.get("id")
                .and_then(Value::as_str)
                .and_then(|s| s.parse().ok())
        }) else {
            continue;
        };

        if let Some(sender) = demux.take_pending(id) {
            let result = if let Some(err) = msg.get("error") {
                Err(err.to_string())
            } else {
                Ok(msg.get("result").cloned().unwrap_or(Value::Null))
            };
            let _ = sender.send(result);
            continue;
        }

        if let Some(subagent_id) = demux.take_prompt(id) {
            shared.complete_turn(&subagent_id, &msg);
            continue;
        }

        debug!("subagent hub: unmatched harness response id={id}");
    }

    // The harness is gone (EOF or read error): nothing will ever complete
    // the in-flight turns. Mark running sessions failed so polls report the
    // truth instead of running forever.
    let mut sessions = shared.sessions.lock().unwrap();
    for session in sessions.values_mut() {
        let mut status = session.status.lock().unwrap();
        if *status == Status::Running {
            *status = Status::Error("harness exited".to_string());
        }
    }
}

/// The production transport: ONE `potlatch harness` child process.
/// An ACP client child process (one per vendor).
struct AcpChildProcess {
    vendor: String,
    demux: Arc<Demux>,
    next_request_id: AtomicU64,
    stdin: Mutex<Option<ChildStdin>>,
    child: Mutex<Option<Child>>,
}

impl AcpChildProcess {
    /// Spawn the child. Returns the transport and the child's stdout for
    /// the reader thread — the caller starts the reader and then runs the
    /// ACP handshake (the reader is what delivers the initialize response).
    /// A failure here kills the child: a client without a hub idles forever.
    fn spawn(
        base_dir: &str,
        acp: &AcpSpawnConfig,
        demux: Arc<Demux>,
        vendor: String,
    ) -> Result<(Arc<Self>, BufReader<std::process::ChildStdout>)> {
        let mut env = acp.env.clone();
        if let Some(argv) = &acp.auth_command {
            let serialized =
                serde_json::to_string(argv).expect("serializing a Vec<String> cannot fail");
            env.insert(AUTH_COMMAND_ENV.to_string(), serialized);
            if let Some(config_dir) = &acp.config_dir {
                env.insert(
                    AUTH_COMMAND_DIR_ENV.to_string(),
                    config_dir.display().to_string(),
                );
            }
        }
        let mut cmd = build_acp_spawn_command(&acp.command, &env)
            .with_context(|| format!("build harness spawn command {:?}", acp.command))?;
        cmd.current_dir(base_dir);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawned from the hub's own long-lived thread: register the child to
        // die with the orchestrator process.
        dies_with_parent(&mut cmd);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn harness subprocess {:?}", acp.command))?;

        if let Some(err) = child.stderr.take() {
            thread::spawn(move || {
                let reader = BufReader::new(err);
                for line in reader.lines().map_while(|l| l.ok()) {
                    warn!(target: "potlatch::subagent_client", "ACP client stderr: {line}");
                }
            });
        }

        let stdin = child
            .stdin
            .take()
            .context("harness child missing stdin pipe")?;
        let stdout = child
            .stdout
            .take()
            .context("harness child missing stdout pipe")?;

        let transport = Arc::new(Self {
            vendor,
            demux,
            next_request_id: AtomicU64::new(1),
            stdin: Mutex::new(Some(stdin)),
            child: Mutex::new(Some(child)),
        });

        Ok((transport, BufReader::new(stdout)))
    }

    fn write_request(&self, id: u64, method: &str, params: Value) -> Result<()> {
        use std::io::Write;
        let line = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut s = serde_json::to_string(&line).context("serialize ACP request")?;
        s.push('\n');
        let mut stdin = self.stdin.lock().unwrap();
        let stdin = stdin
            .as_mut()
            .context("harness stdin already closed (child shut down)")?;
        stdin
            .write_all(s.as_bytes())
            .with_context(|| format!("write ACP request `{method}`"))?;
        stdin.flush()?;
        Ok(())
    }
}

impl HarnessTransport for AcpChildProcess {
    fn supports_extensions(&self) -> bool {
        // The platform's own harness understands multiplex `session/new`,
        // transcript files, and `session/inject`.
        self.vendor == POTLATCH_ACP_PROFILE
    }

    fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::sync_channel(1);
        self.demux.register_pending(id, sender);
        if let Err(error) = self.write_request(id, method, params) {
            self.demux.take_pending(id);
            return Err(error);
        }
        match receiver.recv_timeout(REQUEST_TIMEOUT) {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(anyhow::Error::msg(error)),
            Err(_) => {
                self.demux.take_pending(id);
                Err(anyhow::anyhow!("harness request `{method}` timed out"))
            }
        }
    }

    fn fire_prompt(&self, subagent_id: &str, session_id: &str, prompt: &str) -> Result<()> {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.demux.register_prompt(id, subagent_id.to_string());
        if let Err(error) = self.write_request(
            id,
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": prompt }],
            }),
        ) {
            self.demux.take_prompt(id);
            return Err(error);
        }
        Ok(())
    }

    fn shutdown(&self) {
        // Dropping stdin signals EOF: the harness finishes its work and its
        // main loop exits. Force-kill only if it does not exit promptly.
        self.stdin.lock().unwrap().take();
        let mut child = self.child.lock().unwrap();
        if let Some(child) = child.as_mut() {
            for _ in 0..50 {
                if !matches!(child.try_wait(), Ok(None)) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// A scripted transport: records every call, answers control-plane
    /// requests with canned responses, and never delivers prompt responses
    /// (tests drive turn completion through `HubShared::complete_turn`, the
    /// same entry point the production reader thread uses).
    #[derive(Default)]
    struct FakeTransport {
        requests: Mutex<Vec<(String, Value)>>,
        prompts: Mutex<Vec<(String, String, String)>>,
        shutdowns: AtomicU64,
        session_counter: AtomicU64,
        fail_requests: AtomicBool,
        /// Whether this fake speaks the platform's ACP extensions.
        extensions: AtomicBool,
    }

    impl FakeTransport {
        fn recorded_requests(&self, method: &str) -> Vec<Value> {
            self.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|(m, _)| m == method)
                .map(|(_, p)| p.clone())
                .collect()
        }
    }

    impl HarnessTransport for FakeTransport {
        fn request(&self, method: &str, params: Value) -> Result<Value> {
            self.requests
                .lock()
                .unwrap()
                .push((method.to_string(), params));
            if self.fail_requests.load(Ordering::SeqCst) {
                bail!("harness unreachable");
            }
            if method == "session/new" {
                let n = self.session_counter.fetch_add(1, Ordering::SeqCst) + 1;
                return Ok(json!({ "sessionId": format!("harness-session-{n}") }));
            }
            Ok(json!({}))
        }

        fn fire_prompt(&self, subagent_id: &str, session_id: &str, prompt: &str) -> Result<()> {
            self.prompts.lock().unwrap().push((
                subagent_id.to_string(),
                session_id.to_string(),
                prompt.to_string(),
            ));
            Ok(())
        }

        fn shutdown(&self) {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
        }

        fn supports_extensions(&self) -> bool {
            self.extensions.load(Ordering::SeqCst)
        }
    }

    fn test_hub() -> (SubagentHub, HashMap<String, Arc<FakeTransport>>, PathBuf) {
        // Unique per call: tests run in parallel in one process, and the
        // hub's spawn writes (truncates) the transcript path under this dir.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let base_dir = std::env::temp_dir().join(format!(
            "potlatch-subagent-agent-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        let mut fakes: HashMap<String, Arc<FakeTransport>> = HashMap::new();
        let potlatch = Arc::new(FakeTransport {
            extensions: AtomicBool::new(true),
            ..Default::default()
        });
        fakes.insert("potlatch".to_string(), potlatch);
        let hub = SubagentHub::with_children(
            fakes
                .iter()
                .map(|(vendor, fake)| {
                    (
                        vendor.clone(),
                        Arc::clone(fake) as Arc<dyn HarnessTransport>,
                    )
                })
                .collect(),
            vec![
                "acp://potlatch/model1?effort=high".to_string(),
                "acp://potlatch/model2".to_string(),
            ],
            None,
            &base_dir.display().to_string(),
        );
        (hub, fakes, base_dir)
    }

    /// A hub with the operator's live-subagent cap configured.
    fn capped_hub(max: usize) -> (SubagentHub, Arc<FakeTransport>, PathBuf) {
        let base_dir = std::env::temp_dir().join(format!(
            "potlatch-subagent-agent-capped-test-{}-{max}",
            std::process::id()
        ));
        let transport = Arc::new(FakeTransport {
            extensions: AtomicBool::new(true),
            ..Default::default()
        });
        let hub = SubagentHub::with_children(
            [(
                "potlatch".to_string(),
                Arc::clone(&transport) as Arc<dyn HarnessTransport>,
            )]
            .into_iter()
            .collect(),
            vec!["acp://potlatch/model1?effort=high".to_string()],
            Some(max),
            &base_dir.display().to_string(),
        );
        (hub, transport, base_dir)
    }

    fn spawn_one(hub: &SubagentHub, prompt: &str) -> String {
        let report = hub
            .dispatch(
                &json!({
                    "prompt": prompt,
                    "model": "acp://potlatch/model1?effort=high",
                    "tools": ["shell"]
                }),
                None,
            )
            .unwrap();
        report
            .strip_prefix("Subagent started: ")
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_string()
    }

    #[test]
    fn spawn_rewrites_an_aliased_model_to_the_real_name() {
        // A provided model may be an endpoint alias: the caller addresses the
        // subagent by the alias, but the backend receives the entry's real
        // model name (query suffix preserved).
        let transport = Arc::new(FakeTransport::default());
        let hub = SubagentHub::with_children_and_aliases(
            HashMap::from([(
                "potlatch".to_string(),
                Arc::clone(&transport) as Arc<dyn HarnessTransport>,
            )]),
            vec!["acp://potlatch/model-x?effort=high".to_string()],
            None,
            HashMap::from([("model-x".to_string(), "model1".to_string())]),
            ".",
        );

        hub.dispatch(
            &json!({
                "prompt": "explore",
                "model": "acp://potlatch/model-x?effort=high"
            }),
            None,
        )
        .unwrap();

        assert_eq!(
            transport.recorded_requests("session/set_model")[0]["modelId"],
            "model1?effort=high"
        );
    }

    #[test]
    fn spawn_creates_a_multiplexed_session_and_fires_the_prompt() {
        let (hub, fakes, base_dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let id = spawn_one(&hub, "explore the repo");

        // One harness child, one session per subagent, multiplexed.
        let session_news = transport.recorded_requests("session/new");
        assert_eq!(session_news.len(), 1);
        assert_eq!(session_news[0]["multiplex"], true);
        assert_eq!(session_news[0]["cwd"], base_dir.display().to_string());
        assert_eq!(session_news[0]["tools"], json!(["shell"]));
        // The model defaults to the agent's configured endpoint model.
        assert_eq!(
            transport.recorded_requests("session/set_model")[0]["modelId"],
            "model1?effort=high"
        );
        // The prompt was fired against the created session.
        assert_eq!(
            transport.prompts.lock().unwrap()[0],
            (
                id.clone(),
                "harness-session-1".to_string(),
                "explore the repo".to_string()
            )
        );

        // The transcript lives under <base>/.potlatch/subagent/transcripts/.
        let report = hub.dispatch(&json!({ "subagent_id": id }), None).unwrap();
        let expected = base_dir
            .join(format!(".{APP_NAME}"))
            .join(AGENT_NAME)
            .join("transcripts")
            .join("subagent-1.log");
        assert!(report.contains(&expected.display().to_string()), "{report}");
        assert!(expected.is_file());
    }

    #[test]
    fn poll_inlines_short_output_and_points_at_the_transcript_for_large() {
        let (hub, _fakes, _dir) = test_hub();
        let id = spawn_one(&hub, "task");

        // Short output: inlined in place. The first fake session is
        // harness-session-1 (spawn_one is the hub's first spawn).
        hub.shared.append_output(
            "potlatch",
            "harness-session-1",
            "short finding: the sink is nil-safe",
        );
        let report = hub.dispatch(&json!({ "subagent_id": &id }), None).unwrap();
        assert!(
            report.contains("short finding: the sink is nil-safe"),
            "{report}"
        );

        // Large output: not inlined; the model is directed to the transcript,
        // and told where in it the final response lies.
        let transcript_path = {
            let sessions = hub.shared.sessions.lock().unwrap();
            sessions[&id]
                .transcript_path
                .clone()
                .expect("transcript path")
        };
        let transcript_text = "\n=== Turn 1 ===\n\n## Assistant\n\nfirst response\n\n## Thinking\n\nearly thoughts\n\n=== Turn 2 ===\n\n## Tool Calls\n\n- **shell**: `ls`\n\n=== Turn 3 ===\n\n## Assistant\n\nfinal report: the sink is sound\n";
        std::fs::write(&transcript_path, transcript_text).unwrap();
        hub.shared
            .append_output("potlatch", "harness-session-1", &"x".repeat(20_000));
        let report = hub.dispatch(&json!({ "subagent_id": &id }), None).unwrap();
        assert!(
            !report.contains("xxxxx"),
            "large output must not be inlined: {}",
            &report[..report.len().min(400)]
        );
        assert!(
            report.contains("too large to inline") && report.contains("transcript file"),
            "{report}"
        );
        let expected_line = transcript_text
            .lines()
            .enumerate()
            .filter(|(_, l)| l.trim() == "## Assistant")
            .last()
            .map(|(i, _)| i + 1)
            .unwrap();
        assert!(
            report.contains(&format!(
                "the final response is turn 3, starting at line {expected_line}"
            )),
            "{report}"
        );
    }

    #[test]
    fn final_response_location_finds_the_last_assistant_section() {
        let transcript = "\n=== Turn 1 ===\n\n## Assistant\n\nfirst\n\n=== Turn 2 ===\n\n## Tool Calls\n\n- **shell**: `ls`\n\n=== Turn 3 ===\n\n## Assistant\n\nfinal\n";
        let (line, turn) = final_response_location(transcript).unwrap();
        assert_eq!(turn, 3);
        // 1-based line of the last '## Assistant' header.
        assert_eq!(
            transcript.lines().nth(line - 1).unwrap().trim(),
            "## Assistant"
        );

        // No assistant section at all: None.
        assert_eq!(
            final_response_location("\n=== Turn 1 ===\n\n## Tool Calls\n"),
            None
        );
    }

    #[test]
    fn poll_reports_running_until_the_turn_completes() {
        let (hub, _fakes, _dir) = test_hub();
        let id = spawn_one(&hub, "task");

        let report = hub.dispatch(&json!({ "subagent_id": id }), None).unwrap();
        assert!(report.contains("status: running"), "{report}");

        // The reader thread routes the prompt response by request id; tests
        // drive the same entry point directly.
        hub.shared.complete_turn(
            "subagent-1",
            &json!({
                "jsonrpc": "2.0",
                "id": 99,
                "result": { "stopReason": "end_turn", "message": "found 3 bugs" }
            }),
        );

        let report = hub
            .dispatch(&json!({ "subagent_id": "subagent-1" }), None)
            .unwrap();
        assert!(report.contains("status: done"), "{report}");
        assert!(report.contains("found 3 bugs"), "{report}");
    }

    #[test]
    fn failed_turn_reports_the_error() {
        let (hub, _fakes, _dir) = test_hub();
        spawn_one(&hub, "task");

        hub.shared.complete_turn(
            "subagent-1",
            &json!({
                "jsonrpc": "2.0",
                "id": 99,
                "result": { "stopReason": "error", "message": "Agent loop error: boom" }
            }),
        );
        let report = hub
            .dispatch(&json!({ "subagent_id": "subagent-1" }), None)
            .unwrap();
        assert!(report.contains("status: error"), "{report}");
        assert!(report.contains("Agent loop error: boom"), "{report}");
    }

    #[test]
    fn message_starts_a_new_turn_and_context_persists() {
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let id = spawn_one(&hub, "first task");

        hub.dispatch(
            &json!({ "subagent_id": &id, "message": "now check tests" }),
            None,
        )
        .unwrap();
        let prompts = transport.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[1].0, id);
        assert_eq!(prompts[1].2, "now check tests");
        // Same harness session: context persists across turns.
        assert_eq!(prompts[1].1, prompts[0].1);
    }

    #[test]
    fn inject_sends_session_inject_to_the_harness_session() {
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let id = spawn_one(&hub, "task");

        hub.dispatch(
            &json!({ "subagent_id": &id, "inject": "stop and reconsider" }),
            None,
        )
        .unwrap();
        let injects = transport.recorded_requests("session/inject");
        assert_eq!(injects.len(), 1);
        assert_eq!(injects[0]["sessionId"], "harness-session-1");
        assert_eq!(injects[0]["message"], "stop and reconsider");
    }

    #[test]
    fn kill_closes_the_harness_session_and_leaves_a_tombstone() {
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let id = spawn_one(&hub, "task");

        let report = hub
            .dispatch(&json!({ "subagent_id": &id, "kill": true }), None)
            .unwrap();
        assert!(report.contains("status: killed"), "{report}");
        // The harness session was closed: its context and jobs are released.
        assert_eq!(transport.recorded_requests("session/close").len(), 1);

        // The tombstone still answers polls, and messages are rejected.
        let report = hub.dispatch(&json!({ "subagent_id": &id }), None).unwrap();
        assert!(report.contains("status: killed"), "{report}");
        let err = hub
            .dispatch(&json!({ "subagent_id": &id, "message": "hello" }), None)
            .unwrap_err();
        assert!(err.to_string().contains("has exited"), "{err}");
    }

    #[test]
    fn spawn_is_rejected_when_the_configured_cap_is_reached() {
        // A cap only exists when the operator configures one; it must be
        // visible in the error and freeable by killing a finished subagent.
        let (hub, _transport, _dir) = capped_hub(2);
        let mut ids = Vec::new();
        for i in 0..2 {
            ids.push(spawn_one(&hub, &format!("task {i}")));
        }

        let err = hub
            .dispatch(
                &json!({ "prompt": "one too many", "model": "model1?effort=high" }),
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("subagent limit reached (2 live"),
            "{err}"
        );
        assert!(err.to_string().contains("kill a finished"), "{err}");

        // Killing one frees a slot (tombstones do not count against the cap).
        hub.dispatch(&json!({ "subagent_id": &ids[0], "kill": true }), None)
            .unwrap();
        let id = spawn_one(&hub, "fits again");
        assert_eq!(id, "subagent-3");
    }

    #[test]
    fn without_a_cap_spawning_is_unlimited() {
        // The default (no `max_live_subagents` configured) has no cap: the
        // shared harness process, bounded per-session buffers, and the
        // caller-retirement reaping bound the footprint instead.
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let mut ids = Vec::new();
        for i in 0..12 {
            ids.push(spawn_one(&hub, &format!("task {i}")));
        }
        assert_eq!(ids.len(), 12);
        assert_eq!(transport.prompts.lock().unwrap().len(), 12);

        // Distinct harness sessions, one shared child: every prompt fired
        // against its own session id.
        let prompts = transport.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 12);
        let mut session_ids: Vec<_> = prompts
            .iter()
            .map(|(_, session_id, _)| session_id.clone())
            .collect();
        session_ids.sort();
        session_ids.dedup();
        assert_eq!(session_ids.len(), 12);
    }

    #[test]
    fn models_route_to_their_vendors_client() {
        // provided_models carry the vendor: each vendor gets its own ACP
        // client, and a spawn is served by the client of its model's vendor.
        // Non-harness vendors run without the platform's extensions — no
        // multiplex flag, no transcript.
        let base_dir = std::env::temp_dir().join(format!(
            "potlatch-subagent-agent-vendors-test-{}",
            std::process::id()
        ));
        let potlatch = Arc::new(FakeTransport {
            extensions: AtomicBool::new(true),
            ..Default::default()
        });
        let cursor = Arc::new(FakeTransport::default());
        let mut children: HashMap<String, Arc<dyn HarnessTransport>> = HashMap::new();
        children.insert("potlatch".to_string(), potlatch.clone());
        children.insert("cursor".to_string(), cursor.clone());
        let hub = SubagentHub::with_children(
            children,
            vec![
                "acp://potlatch/model1?effort=high".to_string(),
                "acp://cursor/composer-2".to_string(),
            ],
            None,
            &base_dir.display().to_string(),
        );

        let potlatch_id = hub
            .dispatch(
                &json!({
                    "prompt": "on potlatch",
                    "model": "acp://potlatch/model1?effort=high"
                }),
                None,
            )
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .strip_prefix("Subagent started: ")
            .unwrap()
            .to_string();
        let cursor_id = hub
            .dispatch(
                &json!({
                    "prompt": "on cursor",
                    "model": "acp://cursor/composer-2"
                }),
                None,
            )
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .strip_prefix("Subagent started: ")
            .unwrap()
            .to_string();

        // Each client saw exactly its own vendor's session.
        let potlatch_new = potlatch.recorded_requests("session/new");
        assert_eq!(potlatch_new.len(), 1);
        assert_eq!(potlatch_new[0]["multiplex"], true);
        assert!(potlatch_new[0].get("transcript_path").is_some());
        let cursor_new = cursor.recorded_requests("session/new");
        assert_eq!(cursor_new.len(), 1);
        assert_eq!(cursor_new[0].get("multiplex"), None);
        assert_eq!(cursor_new[0].get("transcript_path"), None);

        // The set_model payload is the model name (vendor prefix stripped).
        assert_eq!(
            potlatch.recorded_requests("session/set_model")[0]["modelId"],
            "model1?effort=high"
        );
        assert_eq!(
            cursor.recorded_requests("session/set_model")[0]["modelId"],
            "composer-2"
        );

        // Prompts landed on the right clients.
        assert_eq!(potlatch.prompts.lock().unwrap()[0].0, potlatch_id);
        assert_eq!(cursor.prompts.lock().unwrap()[0].0, cursor_id);

        // The potlatch subagent has a transcript; the cursor one does not.
        let report = hub
            .dispatch(&json!({ "subagent_id": &potlatch_id }), None)
            .unwrap();
        assert!(report.contains("transcript:"), "{report}");
        let report = hub
            .dispatch(&json!({ "subagent_id": &cursor_id }), None)
            .unwrap();
        assert!(!report.contains("transcript:"), "{report}");

        // Vendor routing holds for follow-ups: a message goes to the same
        // client's session.
        hub.dispatch(
            &json!({ "subagent_id": &cursor_id, "message": "continue" }),
            None,
        )
        .unwrap();
        assert_eq!(cursor.prompts.lock().unwrap().len(), 2);
        assert_eq!(potlatch.prompts.lock().unwrap().len(), 1);
    }

    #[test]
    fn dispatch_routes_by_argument_shape() {
        let (hub, _fakes, _dir) = test_hub();

        // No prompt and no subagent id: a caller error.
        let err = hub.dispatch(&json!({}), None).unwrap_err();
        assert!(err.to_string().contains("missing 'prompt'"), "{err}");

        // Unknown ids are caller errors.
        let err = hub
            .dispatch(&json!({ "subagent_id": "subagent-99" }), None)
            .unwrap_err();
        assert!(err.to_string().contains("unknown subagent id"), "{err}");
        let err = hub
            .dispatch(&json!({ "subagent_id": "subagent-99", "kill": true }), None)
            .unwrap_err();
        assert!(err.to_string().contains("unknown subagent id"), "{err}");
    }

    #[test]
    fn dispatch_treats_an_empty_subagent_id_as_absent() {
        // Models sometimes fill every schema property with defaults: an
        // empty `subagent_id` alongside a `prompt` must count as absent and
        // spawn — not route to a poll of a nonexistent subagent (observed
        // as `unknown subagent id: ` spins on the Responses endpoint).
        let (hub, _fakes, _dir) = test_hub();
        let report = hub
            .dispatch(
                &json!({
                    "prompt": "explore the repo",
                    "model": "acp://potlatch/model1?effort=high",
                    "subagent_id": "",
                    "message": "",
                    "kill": false
                }),
                None,
            )
            .unwrap();
        assert!(
            report.starts_with("Subagent started: "),
            "an empty subagent_id must fall through to spawn: {report}"
        );

        // A whitespace-only id counts as empty too.
        let report = hub
            .dispatch(
                &json!({
                    "prompt": "explore again",
                    "model": "acp://potlatch/model1?effort=high",
                    "subagent_id": "   "
                }),
                None,
            )
            .unwrap();
        assert!(report.starts_with("Subagent started: "), "{report}");
    }

    #[test]
    fn spawn_requires_a_configured_model() {
        // The `model` argument is mandatory on spawn and must be one of the
        // configured models; the error always names the menu.
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();

        let err = hub
            .dispatch(&json!({ "prompt": "no model", "tools": ["shell"] }), None)
            .unwrap_err();
        assert!(
            err.to_string().contains("'model' argument is required"),
            "{err}"
        );
        assert!(err.to_string().contains("model1?effort=high"), "{err}");

        let err = hub
            .dispatch(
                &json!({ "prompt": "bad model", "model": "gpt-9", "tools": ["shell"] }),
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown subagent model 'gpt-9'"),
            "{err}"
        );
        assert!(err.to_string().contains("model2"), "{err}");

        // Nothing was spawned for the rejected requests.
        assert!(transport.requests.lock().unwrap().is_empty());
        assert!(transport.prompts.lock().unwrap().is_empty());

        // A configured model (with query suffix) is accepted.
        let id = spawn_one(&hub, "task");
        assert_eq!(id, "subagent-1");
        assert_eq!(
            transport.recorded_requests("session/set_model")[0]["modelId"],
            "model1?effort=high"
        );
    }

    #[test]
    fn models_context_content_names_the_menu_and_the_requirement() {
        let content =
            models_context_content(&["model1?effort=high".to_string(), "model2".to_string()]);
        assert!(
            content.contains("`model` argument is required"),
            "{content}"
        );
        assert!(content.contains("model1?effort=high, model2"), "{content}");
    }

    #[test]
    fn spawn_records_the_calling_session_as_owner() {
        // The harness tags every bus tool call with its session id; the bus
        // lifts it into the request metadata, and the hub remembers it so
        // the caller's subagents can be closed together.
        let (hub, _fakes, _dir) = test_hub();
        hub.dispatch(
            &json!({
                "prompt": "task",
                "model": "acp://potlatch/model1?effort=high"
            }),
            Some("caller-session-7"),
        )
        .unwrap();

        let sessions = hub.shared.sessions.lock().unwrap();
        assert_eq!(
            sessions["subagent-1"].owner.as_deref(),
            Some("caller-session-7")
        );
    }

    #[test]
    fn caller_closed_closes_that_callers_live_subagents_only() {
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        for (name, caller) in [("a", "caller-7"), ("b", "caller-7"), ("c", "caller-8")] {
            hub.dispatch(
                &json!({
                    "prompt": name,
                    "model": "acp://potlatch/model1?effort=high"
                }),
                Some(caller),
            )
            .unwrap();
        }

        // The caller's task session retired (the agent drains the bus
        // lifecycle channel into this call).
        let report = hub.close_caller_sessions("caller-7").unwrap();
        assert!(
            report.contains("closed 2 subagent(s)") && report.contains("2 entries reclaimed"),
            "{report}"
        );

        // Exactly caller-7's two sessions were closed in the harness.
        assert_eq!(transport.recorded_requests("session/close").len(), 2);
        // The caller's task is over: its entries were fully reclaimed — polls
        // for them are caller errors now.
        let err = hub
            .dispatch(&json!({ "subagent_id": "subagent-1" }), None)
            .unwrap_err();
        assert!(err.to_string().contains("unknown subagent id"), "{err}");
        // Another caller's subagent is untouched: still running.
        let report = hub
            .dispatch(&json!({ "subagent_id": "subagent-3" }), None)
            .unwrap();
        assert!(report.contains("status: running"), "{report}");
    }

    #[test]
    fn close_caller_sessions_is_quiet_when_there_is_nothing_to_close() {
        // Most caller sessions never spawn a subagent: their retirement
        // notice produces no report (and no UI log line).
        let (hub, _transport, _dir) = test_hub();
        assert_eq!(hub.close_caller_sessions("caller-none"), None);
    }

    #[test]
    fn close_caller_sessions_reclaims_killed_tombstones_quietly() {
        // A caller that spawned and killed its subagents leaves tombstones;
        // retirement reclaims them and still reports the accounting.
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        let id = hub
            .dispatch(
                &json!({
                    "prompt": "task",
                    "model": "acp://potlatch/model1?effort=high"
                }),
                Some("caller-7"),
            )
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .strip_prefix("Subagent started: ")
            .unwrap()
            .to_string();
        hub.dispatch(&json!({ "subagent_id": &id, "kill": true }), None)
            .unwrap();
        assert_eq!(transport.recorded_requests("session/close").len(), 1);

        let report = hub.close_caller_sessions("caller-7").unwrap();
        assert!(
            report.contains("closed 0 subagent(s)") && report.contains("1 entries reclaimed"),
            "{report}"
        );
    }

    #[test]
    fn transcript_path_is_under_the_project_potlatch_directory() {
        let path = transcript_path("/home/user/project", 7);
        let expected = Path::new("/home/user/project")
            .join(format!(".{APP_NAME}"))
            .join(AGENT_NAME)
            .join("transcripts")
            .join("subagent-7.log");
        assert_eq!(path, expected);
    }

    #[test]
    fn append_capped_keeps_the_newest_output() {
        let mut buf = String::new();
        append_capped(&mut buf, &"x".repeat(MAX_OUTPUT * 3));
        assert_eq!(buf.len(), MAX_OUTPUT);
        // The tail survives: the newest append is at the end, the head is
        // dropped with a marker.
        append_capped(&mut buf, "more");
        assert_eq!(buf.len(), MAX_OUTPUT);
        assert!(buf.ends_with("more"));
        assert!(buf.starts_with("[...earlier output dropped...]"));
    }

    #[test]
    fn append_capped_truncates_on_a_char_boundary() {
        // The cap lands mid-codepoint: the boundary search moves backward so
        // the kept tail never splits a codepoint.
        let mut buf = "a".repeat(MAX_OUTPUT - 1);
        append_capped(&mut buf, "éé");
        assert_eq!(buf.len(), MAX_OUTPUT);
        assert!(buf.is_char_boundary(buf.len()));
        assert!(buf.ends_with("éé"));
    }

    #[test]
    fn extract_update_text_parses_agent_message_chunks() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "text": "hello world" }
                }
            }
        });
        assert_eq!(extract_update_text(&msg).unwrap(), "hello world");

        let other = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": { "update": { "sessionUpdate": "tool_call", "content": {} } }
        });
        assert!(extract_update_text(&other).is_none());
    }

    #[test]
    fn harness_exit_fails_running_turns() {
        // When the shared harness child dies, nothing will ever complete the
        // in-flight turns: the reader marks running sessions failed so polls
        // report the truth instead of running forever.
        let (hub, _fakes, _dir) = test_hub();
        let id = spawn_one(&hub, "task");

        // Simulate the reader's exit path for a dead harness.
        let shared = Arc::clone(&hub.shared);
        let mut sessions = shared.sessions.lock().unwrap();
        for session in sessions.values_mut() {
            let mut status = session.status.lock().unwrap();
            if *status == Status::Running {
                *status = Status::Error("harness exited".to_string());
            }
        }
        drop(sessions);

        let report = hub.dispatch(&json!({ "subagent_id": id }), None).unwrap();
        assert!(report.contains("status: error"), "{report}");
        assert!(report.contains("harness exited"), "{report}");
    }

    #[test]
    fn hub_shutdown_ends_the_transport() {
        let (hub, fakes, _dir) = test_hub();
        let transport = fakes["potlatch"].clone();
        spawn_one(&hub, "task");
        drop(hub);
        // The hub owns the harness: dropping it releases the process.
        assert_eq!(transport.shutdowns.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn subagent_tool_definition_has_the_model_facing_schema() {
        let def = subagent_tool_definition(
            &["model1?effort=high".to_string(), "model2".to_string()],
            None,
        );
        assert_eq!(def.name, "subagent");
        assert_eq!(def.operation, SUBAGENT_OPERATION);
        // The configured models are embedded so the caller knows the menu.
        assert!(def.description.contains("model1?effort=high"));
        assert!(def.description.contains("model2"));
        assert!(def.description.contains("required when spawning"));
        // Without a configured cap the description promises no hard limit;
        // with one, the number is visible to the caller.
        assert!(def.description.contains("no hard limit"));
        let capped = subagent_tool_definition(&["model1".to_string()], Some(3));
        assert!(
            capped.description.contains("At most 3 subagents"),
            "{}",
            capped.description
        );
        let props = def.parameters["properties"].as_object().unwrap();
        assert!(props.contains_key("prompt"));
        assert!(props.contains_key("model"));
        assert!(props.contains_key("tools"));
        assert!(props.contains_key("subagent_id"));
        assert!(props.contains_key("message"));
        assert!(props.contains_key("inject"));
        assert!(props.contains_key("kill"));
    }
}
