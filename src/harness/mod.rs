//! Potlatch harness: a frontier-tier coding agent ACP server.
//!
//! Replaces `agent-local acp` — speaks ACP (JSON-RPC over stdio) with potlatch's
//! existing `AcpClient`, but runs its own tool-calling loop against a self-hosted
//! OpenAI-compatible LLM endpoint. No Cursor cloud dependency.
//!
//! Config arrives via env vars (`POTLATCH_BASE_URL`, `POTLATCH_API_KEY`) and the ACP
//! protocol (`session/set_model`, `session/new` with cwd).
//!
//! Per-session logs are written to `~/.potlatch/sessions/<session-id>/run.log`.

pub mod acp;
pub mod agent_loop;
pub mod auth_provider;
pub mod client;
pub mod context;
mod parent;
pub mod prompt;
pub(crate) mod session_store;
pub mod todo;
pub mod tools;

use std::fs;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::core::model::acp::jsonrpc::Outbound;
use crate::paths::sessions_dir;

/// A writer that wraps an `Option<File>` behind a `Mutex`, implementing `io::Write`.
/// When the inner file is `None`, writes are silently dropped.
struct SwappableWriter {
    inner: Mutex<Option<fs::File>>,
}

/// Newtype wrapper for `MakeWriter` impl (avoids orphan rule).
struct SwappableWriterMaker(Arc<SwappableWriter>);

impl SwappableWriter {
    fn new(file: Option<fs::File>) -> Self {
        Self {
            inner: Mutex::new(file),
        }
    }

    fn swap(&self, file: Option<fs::File>) {
        *self.inner.lock().unwrap() = file;
    }
}

impl Write for &SwappableWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut guard = self.inner.lock().unwrap();
        match guard.as_mut() {
            Some(file) => file.write(buf),
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        match guard.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SwappableWriterMaker {
    type Writer = SwappableWriterWriter;

    fn make_writer(&'a self) -> Self::Writer {
        SwappableWriterWriter {
            inner: Arc::clone(&self.0),
        }
    }
}

/// Owned writer wrapper for `MakeWriter`.
struct SwappableWriterWriter {
    inner: Arc<SwappableWriter>,
}

impl Write for SwappableWriterWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut guard = self.inner.inner.lock().unwrap();
        match guard.as_mut() {
            Some(file) => file.write(buf),
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut guard = self.inner.inner.lock().unwrap();
        match guard.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

/// Initialize file-based logging for the harness.
/// Starts logging to a temporary file; switches to `~/.potlatch/sessions/<session-id>/run.log`
/// when a session is created (see [`set_session_log`]).
fn init_logging() -> Arc<SwappableWriter> {
    let temp_path =
        std::env::temp_dir().join(format!("potlatch-harness-{}.log", std::process::id()));
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&temp_path)
        .ok();

    let writer = Arc::new(SwappableWriter::new(file));

    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::DEBUG.into())
        .from_env_lossy()
        .add_directive("hyper_util=warn".parse().expect("valid directive"));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(SwappableWriterMaker(Arc::clone(&writer)))
        .with_ansi(false)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    writer
}

/// Switch the log output to `~/.potlatch/sessions/<session-id>/run.log`.
///
/// The full log for one session lives in that session's directory, alongside
/// the persisted context (see [`session_store`]).
fn set_session_log(writer: &SwappableWriter, session_id: &str) {
    let _ = fs::create_dir_all(logging_dir().join(session_id));

    let log_path = session_store::session_run_log(&logging_dir(), session_id);
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    if file.is_some() {
        tracing::info!("switching session log to {}", log_path.display());
    }
    writer.swap(file);
}

pub(crate) fn logging_dir() -> PathBuf {
    sessions_dir()
}

/// Entry point for `potlatch harness`. Reads JSON-RPC from stdin, writes to stdout.
///
/// Uses a reader thread + main thread architecture:
/// - **Reader thread**: reads stdin continuously, parses JSON-RPC. Routes
///   `session/inject` and `session/cancel` directly via shared channels (so
///   they work mid-run while the main thread is blocked in `session/prompt`).
///   All other messages go to the main thread via an mpsc channel.
/// - **Main thread**: owns the `AcpServer`, processes messages from the
///   mpsc channel. `session/prompt` blocks until the agent loop finishes;
///   queued messages wait in the channel.
///
/// Notifications (e.g. `session/update` during `session/prompt`) are written
/// directly to stdout in real-time as they're produced, not buffered.
pub fn run_acp_server() -> Result<()> {
    let log_writer = init_logging();
    tracing::info!("potlatch harness starting (pid={})", std::process::id());

    let base_url = std::env::var("POTLATCH_BASE_URL").unwrap_or_else(|_| {
        eprintln!("POTLATCH_BASE_URL is not set");
        std::process::exit(1);
    });
    let api_key = std::env::var("POTLATCH_API_KEY").unwrap_or_else(|_| "EMPTY".into());

    // Optional endpoint auth provider: an arbitrary command (configured per
    // endpoint via `auth_provider` and forwarded by the parent in the env)
    // whose final stdout line is the headers JSON document. Its headers are
    // applied to every request to the endpoint and cached until expiration.
    // The parent also forwards the directory holding potlatch.toml; the
    // provider command runs from there, so `./auth-tool.py` (or any relative
    // path) resolves against the config, not the agent's repo checkout.
    let auth_provider = match std::env::var(crate::core::config::AUTH_COMMAND_ENV) {
        Ok(raw) if !raw.trim().is_empty() => {
            let argv: Vec<String> = serde_json::from_str(&raw).with_context(|| {
                format!(
                    "{} must be a JSON array of command arguments",
                    crate::core::config::AUTH_COMMAND_ENV
                )
            })?;
            if argv.is_empty() {
                anyhow::bail!(
                    "{} must not be empty",
                    crate::core::config::AUTH_COMMAND_ENV
                );
            }
            let working_dir = std::env::var(crate::core::config::AUTH_COMMAND_DIR_ENV)
                .ok()
                .map(PathBuf::from);
            Some(auth_provider::AuthProvider::new(argv, working_dir))
        }
        _ => None,
    };
    if let Some(auth) = &auth_provider {
        tracing::info!("auth provider configured: {}", auth.program());
    }

    let llm_client = Arc::new(client::OpenAiClient::with_auth_provider(
        base_url,
        api_key,
        auth_provider.map(Arc::new),
    ));
    let shared_stdout = parent::SharedOutput::stdout();
    let parent_rpc = Arc::new(parent::ParentRpc::new(shared_stdout.clone()));
    let mut server = acp::AcpServer::with_output(llm_client, shared_stdout.clone())
        .with_agent_tool_caller(parent_rpc.clone());

    // Shared channels map: session_id → inject_tx + cancel flag.
    // The reader thread uses this to handle session/inject and session/cancel
    // directly, bypassing the main thread (which may be blocked in
    // session/prompt).
    let shared_channels = server.shared_channels();

    // mpsc channel: reader thread → main thread for all non-inject/non-cancel
    // messages. Messages queue here when the main thread is blocked in
    // session/prompt.
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<Value>();

    // Spawn the reader thread.
    let shared_stdout_reader = shared_stdout.clone();
    let parent_rpc_reader = Arc::clone(&parent_rpc);
    let msg_tx_clone = msg_tx.clone();
    std::thread::Builder::new()
        .name("acp-reader".into())
        .spawn(move || {
            run_reader_thread(
                shared_channels,
                shared_stdout_reader,
                parent_rpc_reader,
                msg_tx_clone,
            );
        })
        .context("spawn ACP reader thread")?;

    // Main thread: process messages from the reader thread.
    drop(msg_tx); // Close our copy so msg_rx closes when the reader thread exits.
    while let Ok(msg) = msg_rx.recv() {
        let method = msg["method"].as_str().unwrap_or("(unknown)").to_string();
        tracing::info!("ACP request: {method}");

        // Responses (and turn notifications) are written by the server to the
        // shared stdout it owns; the main thread only reacts to session/new
        // by switching to the per-session log file.
        let response = server.handle_message(&msg)?;

        // When a task session is created, switch to the per-session log
        // file. Multiplexed sessions (subagent agents keep many alive on one
        // connection) stay on the process log: switching per session/new
        // would send concurrent sessions' logs into whichever file was set
        // last.
        if method == "session/new"
            && msg["params"]["multiplex"] != true
            && let Some(Outbound::Response { result, .. }) = &response
            && let Some(sid) = result.get("sessionId").and_then(|v| v.as_str())
        {
            set_session_log(&log_writer, sid);
            tracing::info!(
                "harness ACP: session {} created, registered tools: {:?}",
                sid,
                server.session_tool_names(sid)
            );
        }
    }

    Ok(())
}

/// Reader thread body: reads stdin continuously, parses JSON-RPC, and routes
/// messages. `session/inject` and `session/cancel` are handled directly via
/// shared channels (they work mid-run). All other messages go to the main
/// thread via the mpsc channel.
fn run_reader_thread(
    shared_channels: acp::SharedSessionChannels,
    shared_stdout: parent::SharedOutput,
    parent_rpc: Arc<parent::ParentRpc>,
    msg_tx: std::sync::mpsc::Sender<Value>,
) {
    let stdin = std::io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("ACP reader: stdin read error: {e}");
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("invalid JSON from stdin: {e}");
                eprintln!("harness: invalid JSON: {e}");
                continue;
            }
        };

        let method = msg["method"].as_str().unwrap_or("");
        let id = msg.get("id").cloned();
        if parent_rpc.handle_response(&msg) {
            continue;
        }

        // Route session/inject directly via shared channels. This works
        // mid-run: the inject_tx pushes to the agent loop's inject channel,
        // which is drained at the top of the next iteration.
        if method == "session/inject" {
            let session_id = msg["params"]["sessionId"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let message = msg["params"]["message"].as_str().unwrap_or("").to_string();
            let result = {
                let sc = shared_channels.lock().unwrap();
                if let Some(chans) = sc.get(&session_id) {
                    chans.inject_tx.lock().unwrap().push_back(message);
                    tracing::debug!("ACP reader: injected message into session {session_id}");
                    json!({})
                } else {
                    tracing::warn!("ACP reader: inject for unknown session {session_id}");
                    json!({ "error": "unknown session" })
                }
            };
            // Write the response to stdout.
            if let Some(ref id) = id {
                let resp = Outbound::Response {
                    id: id.clone(),
                    result,
                };
                if let Ok(line) = resp.to_json_line() {
                    acp::write_line(&shared_stdout, &line);
                }
            }
            continue;
        }

        // Route session/cancel directly via shared channels. This works
        // mid-run: the cancel flag is checked at the top of each iteration.
        if method == "session/cancel" {
            let session_id = msg["params"]["sessionId"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let result = {
                let sc = shared_channels.lock().unwrap();
                if let Some(chans) = sc.get(&session_id) {
                    chans
                        .cancel
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    tracing::info!("ACP reader: cancel requested for session {session_id}");
                    json!({})
                } else {
                    tracing::warn!("ACP reader: cancel for unknown session {session_id}");
                    json!({ "error": "unknown session" })
                }
            };
            // Write the response to stdout.
            if let Some(ref id) = id {
                let resp = Outbound::Response {
                    id: id.clone(),
                    result,
                };
                if let Ok(line) = resp.to_json_line() {
                    acp::write_line(&shared_stdout, &line);
                }
            }
            continue;
        }

        // All other messages go to the main thread. If the main thread is
        // blocked in session/prompt, the message queues in the channel.
        if msg_tx.send(msg).is_err() {
            tracing::warn!("ACP reader: main thread channel closed, exiting");
            break;
        }
    }
}
