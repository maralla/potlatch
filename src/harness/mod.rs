//! Potlatch harness: a frontier-tier coding agent ACP server.
//!
//! Replaces `agent-local acp` — speaks ACP (JSON-RPC over stdio) with potlatch's
//! existing `AcpClient`, but runs its own tool-calling loop against a self-hosted
//! OpenAI-compatible LLM endpoint. No Cursor cloud dependency.
//!
//! Config arrives via env vars (`BREEZE_BASE_URL`, `BREEZE_API_KEY`) and the ACP
//! protocol (`session/set_model`, `session/new` with cwd).
//!
//! Per-session logs are written to `~/.potlatch/sessions/<session-id>.log`.

pub mod acp;
pub mod agent_loop;
pub mod client;
pub mod context;
mod parent;
pub mod prompt;
pub mod todo;
pub mod tools;

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::core::model::acp::jsonrpc::Outbound;

/// A writer that wraps an `Option<File>` behind a `Mutex`, implementing `io::Write`.
/// When the inner file is `None`, writes are silently dropped.
struct SwappableWriter {
    inner: Mutex<Option<std::fs::File>>,
}

/// Newtype wrapper for `MakeWriter` impl (avoids orphan rule).
struct SwappableWriterMaker(Arc<SwappableWriter>);

impl SwappableWriter {
    fn new(file: Option<std::fs::File>) -> Self {
        Self {
            inner: Mutex::new(file),
        }
    }

    fn swap(&self, file: Option<std::fs::File>) {
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
/// Starts logging to a temporary file; switches to `~/.potlatch/sessions/<session-id>.log`
/// when a session is created (see [`set_session_log`]).
fn init_logging() -> Arc<SwappableWriter> {
    let temp_path = std::env::temp_dir().join(format!("potlatch-harness-{}.log", std::process::id()));
    let file = std::fs::OpenOptions::new()
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

/// Switch the log output to `~/.potlatch/sessions/<session-id>.log`.
fn set_session_log(writer: &SwappableWriter, session_id: &str) {
    let sessions_dir = logging_dir();
    let _ = std::fs::create_dir_all(&sessions_dir);

    let log_path = sessions_dir.join(format!("{session_id}.log"));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    if file.is_some() {
        tracing::info!("switching session log to {}", log_path.display());
    }
    writer.swap(file);
}

pub(crate) fn logging_dir() -> std::path::PathBuf {
    home_dir().join(".potlatch").join("sessions")
}

fn home_dir() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home);
    }
    std::path::PathBuf::from(".")
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

    let base_url = std::env::var("BREEZE_BASE_URL")
        .context("BREEZE_BASE_URL env var is required for potlatch harness")?;
    let api_key = std::env::var("BREEZE_API_KEY").unwrap_or_else(|_| "EMPTY".into());

    let llm_client = Arc::new(client::OpenAiClient::new(base_url, api_key));
    let shared_stdout = parent::SharedOutput::stdout();
    let parent_rpc = Arc::new(parent::ParentRpc::new(shared_stdout.clone()));
    let mut server = acp::AcpServer::new(llm_client).with_agent_tool_caller(parent_rpc.clone());

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
    let shared_stdout_main = shared_stdout.clone();
    while let Ok(msg) = msg_rx.recv() {
        let method = msg["method"].as_str().unwrap_or("(unknown)");
        tracing::info!("ACP request: {method}");

        // When a session is created, switch to per-session log file
        if method == "session/new" {
            let mut out = shared_stdout_main.clone();
            let response = server.handle_message(&msg, &mut out)?;
            if let Some(ref resp) = response {
                let line = resp.to_json_line().context("serialize response")?;
                out.write_all(line.as_bytes())?;
                out.flush()?;

                if let Outbound::Response { result, .. } = resp
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
            continue;
        }

        let mut out = shared_stdout_main.clone();
        let response = server.handle_message(&msg, &mut out)?;
        if let Some(resp) = response {
            let line = resp.to_json_line().context("serialize response")?;
            out.write_all(line.as_bytes())?;
            out.flush()?;
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
    mut shared_stdout: parent::SharedOutput,
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
                    let _ = shared_stdout.write_all(line.as_bytes());
                    let _ = shared_stdout.flush();
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
                    let _ = shared_stdout.write_all(line.as_bytes());
                    let _ = shared_stdout.flush();
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
