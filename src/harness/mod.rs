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
pub mod prompt;
pub mod tools;

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde_json::Value;

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

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(SwappableWriterMaker(Arc::clone(&writer)))
        .with_ansi(false)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    writer
}

/// Switch the log output to `~/.potlatch/sessions/<session-id>.log`.
fn set_session_log(writer: &SwappableWriter, session_id: &str) {
    let sessions_dir = home_dir().join(".potlatch").join("sessions");
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

fn home_dir() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home);
    }
    std::path::PathBuf::from(".")
}

/// Entry point for `potlatch harness`. Reads JSON-RPC from stdin, writes to stdout.
///
/// Notifications (e.g. `session/update` during `session/prompt`) are written directly
/// to stdout in real-time as they're produced, not buffered.
pub fn run_acp_server() -> Result<()> {
    let log_writer = init_logging();
    tracing::info!("potlatch harness starting (pid={})", std::process::id());

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let base_url = std::env::var("BREEZE_BASE_URL")
        .context("BREEZE_BASE_URL env var is required for potlatch harness")?;
    let api_key = std::env::var("BREEZE_API_KEY").unwrap_or_else(|_| "EMPTY".into());

    let llm_client = Arc::new(client::OpenAiClient::new(base_url, api_key));
    let mut server = acp::AcpServer::new(llm_client);

    let reader = stdin.lock();
    for line in reader.lines() {
        let line = line.context("read stdin line")?;
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

        let method = msg["method"].as_str().unwrap_or("(unknown)");
        tracing::info!("ACP request: {method}");

        // When a session is created, switch to per-session log file
        if method == "session/new" {
            let response = server.handle_message(&msg, &mut out)?;
            if let Some(ref resp) = response {
                let line = resp.to_json_line().context("serialize response")?;
                out.write_all(line.as_bytes())?;
                out.flush()?;

                if let Outbound::Response { result, .. } = resp
                    && let Some(sid) = result.get("sessionId").and_then(|v| v.as_str())
                {
                    set_session_log(&log_writer, sid);
                }
            }
            continue;
        }

        let response = server.handle_message(&msg, &mut out)?;
        if let Some(resp) = response {
            let line = resp.to_json_line().context("serialize response")?;
            out.write_all(line.as_bytes())?;
            out.flush()?;
        }
    }

    Ok(())
}
