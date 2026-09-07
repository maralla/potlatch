//! Subagent tool: spawn `potlatch harness` as a child process and drive it via
//! ACP JSON-RPC over stdio. The subagent runs asynchronously — the parent gets
//! a `subagent_id` immediately and polls for accumulated output. Subagents
//! support multi-turn conversations: the parent can send follow-up messages
//! via `session/inject`, which are injected into the running agent loop's
//! context before the next model call. The harness persists context across
//! all prompts and injections — a single long session.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::paths::display_name;

use super::Tool;

const MAX_OUTPUT: usize = 50_000;

/// A running subagent session: a `potlatch harness` child process plus the
/// shared output buffer the reader thread appends to.
struct Subagent {
    child: Child,
    /// The ACP driver, wrapped in `Arc<Mutex>` so both the main thread and
    /// reader thread can send messages. Owns the stdin handle.
    driver: Arc<Mutex<AcpDriver>>,
    /// The session id assigned by the child harness during `session/new`.
    session_id: String,
    started_at: Instant,
    /// Path to the transcript file written by the child harness. The parent
    /// agent can read this file to inspect the subagent's full conversation
    /// (prompt, response, reasoning, tool calls) in real time.
    transcript_path: std::path::PathBuf,
    /// Accumulated output from `session/update` notifications and the final
    /// `session/prompt` response. Shared with the reader thread.
    output_buf: Arc<Mutex<String>>,
    /// Set once the reader thread observes EOF or an error (process exited).
    done: Arc<AtomicBool>,
    /// Set when the subagent was killed via `kill` (forceful).
    killed: Arc<AtomicBool>,
    /// Set when the parent sent `session/close` (graceful shutdown).
    closed: Arc<AtomicBool>,
    /// Populated when the reader thread hits an error or the process exits
    /// unexpectedly (not via `session/close` or `kill`).
    error: Arc<Mutex<Option<String>>>,
}

impl Subagent {
    /// Try to reap the child without blocking; update internal state on exit.
    fn try_reap(&mut self) -> Option<i32> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code().unwrap_or(-1)),
            Ok(None) => None,
            Err(_) => Some(-1),
        }
    }
}

/// Shared table of running subagents, keyed by id. Mirrors `JobTable`.
/// All methods take `&self` and lock internally.
pub struct SubagentTable {
    subagents: Mutex<HashMap<String, Subagent>>,
    next_id: AtomicU64,
}

impl SubagentTable {
    pub fn new() -> Self {
        Self {
            subagents: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Spawn a `potlatch harness` subprocess, drive the ACP handshake, send the
    /// prompt, and return the subagent id. The reader thread collects
    /// `session/update` notifications and all responses asynchronously.
    pub fn spawn(
        &self,
        prompt: &str,
        model: &str,
        tools: Option<&[String]>,
        cwd: &str,
        parent_session_id: &str,
    ) -> Result<String> {
        // Keep transcripts with Potlatch's session logs, never in the repository.
        let subagent_num = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("subagent-{subagent_num}");
        let logging_dir = crate::harness::logging_dir();
        let transcript_path = subagent_transcript_path(
            &logging_dir,
            parent_session_id,
            std::process::id(),
            subagent_num,
        );
        std::fs::create_dir_all(
            transcript_path
                .parent()
                .context("subagent transcript path has no parent")?,
        )
        .context("create Potlatch logging directory for subagent transcript")?;
        std::fs::write(&transcript_path, "")
            .context("initialize subagent transcript in Potlatch logging directory")?;

        let exe =
            std::env::current_exe().context("locate current potlatch executable for subagent")?;

        let mut cmd = Command::new(exe);
        cmd.arg("harness");
        cmd.current_dir(cwd);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::inherit());

        let mut child = cmd.spawn().context("spawn `potlatch harness` subprocess")?;
        let stdin = child
            .stdin
            .take()
            .context("subagent child missing stdin pipe")?;
        let stdout = child
            .stdout
            .take()
            .context("subagent child missing stdout pipe")?;

        let mut driver = AcpDriver::new(stdin, stdout);

        // Synchronous ACP handshake.
        driver.send_request("initialize", json!({}))?;

        let transcript_path_str = transcript_path.to_string_lossy().to_string();
        let mut session_params = json!({ "cwd": cwd, "mcpServers": [] });
        if let Some(names) = tools {
            session_params["tools"] = json!(names);
        }
        session_params["transcript_path"] = json!(transcript_path_str);
        let new_result = driver.send_request("session/new", session_params)?;
        let session_id = new_result["sessionId"]
            .as_str()
            .context("session/new response missing sessionId")?
            .to_string();

        if !model.is_empty() {
            driver.send_request(
                "session/set_model",
                json!({ "sessionId": session_id, "modelId": model }),
            )?;
        }

        let output_buf = Arc::new(Mutex::new(String::new()));
        let done = Arc::new(AtomicBool::new(false));
        let killed = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None::<String>));

        // Fire the first prompt (fire-and-forget) and hand the stdout reader
        // to a background thread. The reader thread is persistent — it
        // accumulates all output from all turns and only exits on EOF/error.
        driver.send_request_raw(
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": prompt }],
            }),
        )?;

        let reader_stdout = driver
            .stdout
            .take()
            .context("reader stdout already taken")?;
        let output_buf_r = Arc::clone(&output_buf);
        let done_r = Arc::clone(&done);
        let killed_r = Arc::clone(&killed);
        let closed_r = Arc::clone(&closed);
        let error_r = Arc::clone(&error);
        thread::Builder::new()
            .name("subagent-reader".into())
            .spawn(move || {
                run_reader(
                    reader_stdout,
                    output_buf_r,
                    done_r,
                    killed_r,
                    closed_r,
                    error_r,
                );
            })
            .context("spawn subagent reader thread")?;

        // Wrap the driver in Arc<Mutex> so both send_message and kill can
        // access it. The stdin handle stays alive inside the driver.
        let driver = Arc::new(Mutex::new(driver));
        let subagent = Subagent {
            child,
            driver,
            session_id: session_id.clone(),
            started_at: Instant::now(),
            transcript_path,
            output_buf,
            done,
            killed,
            closed,
            error,
        };
        self.subagents.lock().unwrap().insert(id.clone(), subagent);
        Ok(id)
    }

    /// Poll a subagent: return accumulated output and current status.
    pub fn poll(&self, subagent_id: &str) -> Result<String> {
        let mut subagents = self.subagents.lock().unwrap();
        let sub = subagents
            .get_mut(subagent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown subagent id: {subagent_id}"))?;

        // Reap the child if the reader thread signaled completion.
        if sub.done.load(Ordering::SeqCst)
            && let Some(code) = sub.try_reap()
            && code != 0
            && !sub.killed.load(Ordering::SeqCst)
            && !sub.closed.load(Ordering::SeqCst)
            && sub.error.lock().unwrap().is_none()
        {
            *sub.error.lock().unwrap() = Some(format!("subagent exited with code {code}"));
        }

        // Detect unexpected exit: done but neither closed nor killed.
        if sub.done.load(Ordering::SeqCst)
            && !sub.killed.load(Ordering::SeqCst)
            && !sub.closed.load(Ordering::SeqCst)
            && sub.error.lock().unwrap().is_none()
        {
            *sub.error.lock().unwrap() = Some("unexpected exit".into());
        }

        let running = !sub.done.load(Ordering::SeqCst) && sub.try_reap().is_none();
        let killed = sub.killed.load(Ordering::SeqCst);
        let closed = sub.closed.load(Ordering::SeqCst);
        let status = if killed {
            "killed"
        } else if closed {
            "closed"
        } else if running {
            "running"
        } else {
            "done"
        };

        let output = sub.output_buf.lock().unwrap().clone();
        let error = sub.error.lock().unwrap().clone();
        let elapsed = sub.started_at.elapsed();
        let transcript_path = sub.transcript_path.clone();

        let mut result = format!(
            "subagent: {subagent_id}\nstatus: {status}\nelapsed: {:.1}s",
            elapsed.as_secs_f64()
        );
        result.push_str(&format!("\ntranscript: {}", transcript_path.display()));
        if let Some(ref err) = error {
            result.push_str(&format!("\nerror: {err}"));
        }
        if !output.is_empty() {
            result.push_str("\noutput:\n");
            result.push_str(&output);
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

    /// Send a follow-up message to a running subagent via `session/prompt`.
    /// Fire-and-forget: sends the prompt and returns immediately. If the
    /// subagent is still running a previous prompt, this one queues behind it
    /// (the harness processes `session/prompt` calls sequentially). If the
    /// subagent has finished its previous turn, this starts a new turn. Context
    /// persists across all prompts — true single long session.
    pub fn send_message(&self, subagent_id: &str, message: &str) -> Result<String> {
        let mut subagents = self.subagents.lock().unwrap();
        let sub = subagents
            .get_mut(subagent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown subagent id: {subagent_id}"))?;

        if sub.done.load(Ordering::SeqCst) {
            anyhow::bail!("subagent {subagent_id} has exited");
        }

        // Write the user message to the transcript file.
        write_transcript_entry(&sub.transcript_path, "## User", message);

        // Send session/prompt via the driver (fire-and-forget). The harness
        // queues prompts sequentially and shares context across all of them.
        let driver = Arc::clone(&sub.driver);
        let session_id = sub.session_id.clone();
        drop(subagents);

        let mut driver = driver.lock().unwrap();
        driver.send_request_raw(
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": message }],
            }),
        )?;

        Ok(subagent_id.to_string())
    }

    /// Inject a message into a running subagent's context via `session/inject`.
    /// Fire-and-forget: pushes the message into the child harness's inject
    /// channel and returns immediately. The agent loop drains it at the top of
    /// the next iteration and adds it to context before the next model call.
    /// Unlike `send_message`, this does NOT start a new turn — it redirects
    /// the currently running turn. If the agent has already stopped, the
    /// message sits in the channel until the next `session/prompt` drains it.
    pub fn send_inject(&self, subagent_id: &str, message: &str) -> Result<String> {
        let mut subagents = self.subagents.lock().unwrap();
        let sub = subagents
            .get_mut(subagent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown subagent id: {subagent_id}"))?;

        if sub.done.load(Ordering::SeqCst) {
            anyhow::bail!("subagent {subagent_id} has exited");
        }

        // Write the user message to the transcript file.
        write_transcript_entry(&sub.transcript_path, "## User (inject)", message);

        // Send session/inject via the driver (fire-and-forget).
        let driver = Arc::clone(&sub.driver);
        let session_id = sub.session_id.clone();
        drop(subagents);

        let mut driver = driver.lock().unwrap();
        driver.send_request_raw(
            "session/inject",
            json!({
                "sessionId": session_id,
                "message": message,
            }),
        )?;

        Ok(subagent_id.to_string())
    }

    /// Gracefully close a subagent: send `session/close`, set the `closed`
    /// flag, and reap the child. The child harness shuts down cleanly.
    pub fn close(&self, subagent_id: &str) -> Result<String> {
        let mut subagents = self.subagents.lock().unwrap();
        let sub = subagents
            .get_mut(subagent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown subagent id: {subagent_id}"))?;

        if sub.done.load(Ordering::SeqCst) {
            drop(subagents);
            return self.poll(subagent_id);
        }

        sub.closed.store(true, Ordering::SeqCst);
        let driver = Arc::clone(&sub.driver);
        let session_id = sub.session_id.clone();
        drop(subagents);

        // Send session/close (best effort — the child may have already exited).
        let _ = driver
            .lock()
            .unwrap()
            .send_request_raw("session/close", json!({ "sessionId": session_id }));

        self.poll(subagent_id)
    }

    /// Kill a subagent. Drops stdin (EOF), sends nothing more, and reaps the
    /// child. Marks it killed so `poll` reports `killed` rather than `done`.
    pub fn kill(&self, subagent_id: &str) -> Result<String> {
        let mut subagents = self.subagents.lock().unwrap();
        let sub = subagents
            .get_mut(subagent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown subagent id: {subagent_id}"))?;

        if sub.done.load(Ordering::SeqCst) {
            drop(subagents);
            return self.poll(subagent_id);
        }

        sub.killed.store(true, Ordering::SeqCst);
        // Drop stdin to signal EOF by taking it from the driver, then kill.
        {
            let mut driver = sub.driver.lock().unwrap();
            driver.take_stdin();
        }
        let _ = sub.child.kill();
        drop(subagents);
        self.poll(subagent_id)
    }

    /// Close all running subagents gracefully. Called on session close.
    pub fn close_all(&self) {
        let ids: Vec<String> = self.subagents.lock().unwrap().keys().cloned().collect();
        for id in ids {
            let _ = self.close(&id);
        }
    }
}

fn subagent_transcript_path(
    logging_dir: &std::path::Path,
    session_id: &str,
    process_id: u32,
    subagent_num: u64,
) -> std::path::PathBuf {
    logging_dir
        .join("transcripts")
        .join(session_id)
        .join(format!("subagent-{process_id}-{subagent_num}.log"))
}

impl Default for SubagentTable {
    fn default() -> Self {
        Self::new()
    }
}

impl super::SessionState for SubagentTable {
    fn shutdown(&self) {
        self.close_all();
    }
}

/// Minimal inline ACP JSON-RPC driver for the synchronous handshake phase.
/// The stdout reader is moved into the reader thread after the handshake; the
/// stdin stays with the driver (wrapped in `Arc<Mutex>` on the `Subagent`).
struct AcpDriver {
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<std::process::ChildStdout>>,
    next_id: u64,
}

impl AcpDriver {
    fn new(stdin: ChildStdin, stdout: std::process::ChildStdout) -> Self {
        Self {
            stdin: Some(stdin),
            stdout: Some(BufReader::new(stdout)),
            next_id: 1,
        }
    }

    /// Take the stdin handle (drops it, signaling EOF to the child).
    fn take_stdin(&mut self) {
        self.stdin.take();
    }

    /// Send a request and read lines until the matching response arrives.
    /// `session/update` notifications emitted during the synchronous phase
    /// (none expected before the prompt) are skipped.
    fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.send_request_raw(method, params)?;
        self.wait_for_response(id)
    }

    /// Send a request line and return its id without waiting for the response.
    fn send_request_raw(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let line = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut s = serde_json::to_string(&line).context("serialize ACP request")?;
        s.push('\n');
        let stdin = self
            .stdin
            .as_mut()
            .context("subagent stdin already closed")?;
        stdin
            .write_all(s.as_bytes())
            .with_context(|| format!("write ACP request `{method}`"))?;
        stdin.flush()?;
        debug!("subagent: sent `{method}` (id={id})");
        Ok(id)
    }

    /// Read lines from stdout until a response with the matching id arrives.
    /// Notifications and agent→client requests are ignored during the
    /// handshake (the harness does not issue any before the prompt).
    fn wait_for_response(&mut self, id: u64) -> Result<Value> {
        let stdout = self
            .stdout
            .as_mut()
            .context("subagent stdout already moved to reader thread")?;
        loop {
            let mut line = String::new();
            let n = stdout.read_line(&mut line).context("read ACP response")?;
            if n == 0 {
                anyhow::bail!("subagent: EOF before response to id {id}");
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let msg: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    warn!("subagent: skipping invalid JSON: {e}");
                    continue;
                }
            };
            if msg.get("method").is_some() {
                // Notification or agent→client request — not expected during
                // the handshake. Drop it.
                debug!("subagent: ignoring handshake-phase message: {trimmed}");
                continue;
            }
            let resp_id = msg.get("id").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            });
            if resp_id == Some(id) {
                if let Some(err) = msg.get("error") {
                    anyhow::bail!("subagent: ACP error for id {id}: {err}");
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
}

/// Reader thread body: drain the subagent's stdout, append `session/update`
/// text chunks and response messages to the shared buffer. The reader is
/// persistent — it loops forever, accumulating all output from all turns.
/// It only exits on EOF or error (process exited). `done` means the process
/// is gone, not that a specific prompt finished.
fn run_reader(
    mut reader: impl BufRead,
    output_buf: Arc<Mutex<String>>,
    done: Arc<AtomicBool>,
    killed: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
) {
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                *error.lock().unwrap() = Some(format!("reader error: {e}"));
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
                debug!("subagent reader: skipping invalid JSON: {e}");
                continue;
            }
        };

        // Agent→client request (e.g. session/request_permission). Drop it;
        // the harness treats a missing permission reply as a deny.
        if msg.get("method").is_some()
            && msg.get("id").is_some()
            && msg.get("id") != Some(&Value::Null)
        {
            debug!(
                "subagent reader: dropping agent→client request: {}",
                msg["method"]
            );
            continue;
        }

        // Notification: session/update with agent_message_chunk text.
        if msg.get("method").and_then(|v| v.as_str()) == Some("session/update") {
            if let Some(text) = extract_update_text(&msg) {
                output_buf.lock().unwrap().push_str(&text);
            }
            continue;
        }

        // Response (no `method` field). Append the result message to the
        // output buffer if non-empty. Don't mark done — the reader stays
        // alive for subsequent prompts and injections.
        if msg.get("method").is_none() {
            if let Some(message) = msg
                .get("result")
                .and_then(|r| r.get("message"))
                .and_then(|m| m.as_str())
                && !message.is_empty()
            {
                output_buf.lock().unwrap().push_str(message);
            }
            continue;
        }
    }

    // Stream ended (EOF or error). Mark done. If neither killed nor closed,
    // this is an unexpected exit — the error will be set by `poll`.
    if !killed.load(Ordering::SeqCst)
        && !closed.load(Ordering::SeqCst)
        && error.lock().unwrap().is_none()
    {
        *error.lock().unwrap() = Some("unexpected exit".into());
    }
    done.store(true, Ordering::SeqCst);
}

/// Extract text from a `session/update` notification's
/// `agent_message_chunk` content.
fn extract_update_text(msg: &Value) -> Option<String> {
    let update = msg.get("params")?.get("update")?;
    let kind = update.get("sessionUpdate").and_then(|v| v.as_str())?;
    if kind != "agent_message_chunk" {
        return None;
    }
    let text = update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())?;
    Some(text.to_string())
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

pub struct SubagentTool {
    table: Arc<SubagentTable>,
    model: String,
    session_id: String,
}

impl SubagentTool {
    /// Construct with session-scoped state. Creates a `SubagentTable` in
    /// `SessionStates` if not already present (first prompt), then retrieves
    /// it so subagents survive across prompts and are killed on session close.
    pub fn new(
        states: &mut super::SessionStates,
        session_id: &str,
        _cwd: &str,
        model: &str,
    ) -> Self {
        if states.get::<SubagentTable>().is_none() {
            states.insert(Arc::new(SubagentTable::new()));
        }
        Self {
            table: states.get::<SubagentTable>().unwrap(),
            model: model.to_string(),
            session_id: session_id.to_string(),
        }
    }

    /// Construct with an externally-owned table (for tests).
    #[cfg(test)]
    pub fn with_table(table: Arc<SubagentTable>, model: &str) -> Self {
        Self {
            table,
            model: model.to_string(),
            session_id: "test-session".to_string(),
        }
    }
}

impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "subagent"
    }

    fn schema(&self) -> Value {
        json!({
            "description": format!("Spawn a subagent — a separate {d} harness instance with its own LLM session. The subagent runs asynchronously in the background. Returns a subagent_id immediately; poll with subagent_id to get accumulated output. Use for parallel exploration, independent research tasks, or dividing complex work. The subagent has no context from the parent session — provide everything it needs in the prompt. Send follow-up messages to a running subagent with subagent_id + message (mid-run injection into the agent's context).", d = display_name()),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the subagent. Be specific — the subagent has no context from the parent session."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model for the subagent (e.g. 'model1?thinking=true'). Defaults to the parent's model."
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the subagent can use (e.g. ['read','grep','shell']). Omit for all tools."
                    },
                    "subagent_id": {
                        "type": "string",
                        "description": "A running subagent id. Use with message to send a follow-up, with kill to terminate, or alone to poll for accumulated output."
                    },
                    "message": {
                        "type": "string",
                        "description": "Send a follow-up message to a running subagent as a new session/prompt turn. Returns the subagent_id immediately. If the subagent is still running a previous turn, this one queues behind it. Context persists across all turns. Poll with subagent_id to read the response."
                    },
                    "inject": {
                        "type": "string",
                        "description": "Inject a message into a running subagent's context mid-run via session/inject. The message is added before the next model call in the current turn — it redirects the agent without starting a new turn. Returns the subagent_id immediately. If the agent has already stopped, the message waits for the next session/prompt. Poll with subagent_id to read the response."
                    },
                    "kill": {
                        "type": "boolean",
                        "description": "When true with subagent_id, terminate the subagent."
                    }
                },
                "required": []
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let subagent_id = args["subagent_id"].as_str();
        let kill = args["kill"].as_bool().unwrap_or(false);
        let message = args["message"].as_str().filter(|s| !s.is_empty());
        let inject = args["inject"].as_str().filter(|s| !s.is_empty());

        if let Some(id) = subagent_id {
            if let Some(msg) = message {
                return self.table.send_message(id, msg);
            }
            if let Some(msg) = inject {
                return self.table.send_inject(id, msg);
            }
            if kill {
                return self.table.kill(id);
            }
            return self.table.poll(id);
        }

        let prompt = args["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'prompt' argument"))?;
        let model = args["model"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.model);
        let tools: Option<Vec<String>> = args["tools"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .filter(|v: &Vec<String>| !v.is_empty());

        let id = self
            .table
            .spawn(prompt, model, tools.as_deref(), cwd, &self.session_id)?;
        Ok(format!(
            "Subagent started: {id}\nprompt: {}",
            prompt.chars().take(200).collect::<String>()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::tools::test_util;

    #[test]
    fn transcript_path_uses_the_potlatch_logging_directory() {
        let logging_dir = std::path::Path::new("/home/test/.potlatch/sessions");

        assert_eq!(
            subagent_transcript_path(logging_dir, "session-abc", 42, 7),
            logging_dir.join("transcripts/session-abc/subagent-42-7.log")
        );
    }

    #[test]
    fn extract_update_text_parses_agent_message_chunk() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "text": "hello world" }
                }
            }
        });
        assert_eq!(extract_update_text(&msg).unwrap(), "hello world");
    }

    #[test]
    fn extract_update_text_ignores_other_update_types() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "tool_call",
                    "content": {}
                }
            }
        });
        assert!(extract_update_text(&msg).is_none());
    }

    #[test]
    fn extract_update_text_returns_none_for_non_update() {
        let msg = json!({ "jsonrpc": "2.0", "method": "session/prompt", "params": {} });
        assert!(extract_update_text(&msg).is_none());
    }

    #[test]
    fn execute_missing_prompt_returns_error() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let args = json!({});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing 'prompt'"));
    }

    #[test]
    fn execute_polls_unknown_id_returns_error() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let args = json!({"subagent_id": "nope"});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown subagent"));
    }

    #[test]
    fn execute_kill_unknown_id_returns_error() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let args = json!({"subagent_id": "nope", "kill": true});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown subagent"));
    }

    #[test]
    fn schema_has_required_fields() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let schema = tool.schema();
        assert!(schema["description"].as_str().unwrap().contains("subagent"));
        let props = schema["parameters"]["properties"].as_object().unwrap();
        assert!(props.contains_key("prompt"));
        assert!(props.contains_key("model"));
        assert!(props.contains_key("tools"));
        assert!(props.contains_key("subagent_id"));
        assert!(props.contains_key("message"));
        assert!(props.contains_key("inject"));
        assert!(props.contains_key("kill"));
    }

    #[test]
    fn execute_send_message_unknown_id_returns_error() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let args = json!({"subagent_id": "nope", "message": "hello"});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown subagent"));
    }

    #[test]
    fn execute_inject_unknown_id_returns_error() {
        let tool = SubagentTool::with_table(Arc::new(SubagentTable::new()), "m");
        let args = json!({"subagent_id": "nope", "inject": "hello"});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown subagent"));
    }

    /// Spawn a subagent that runs `echo hi` via the shell tool and verifies we
    /// can poll its output. This is an integration test that requires the
    /// `potlatch` binary to be runnable (we use the current executable).
    #[test]
    fn spawn_and_poll_real_subagent() {
        // The test harness sets up a real LLM client via env vars. Skip if no
        // POTLATCH_BASE_URL is configured — the subagent needs an LLM to run.
        if std::env::var("POTLATCH_BASE_URL").is_err() {
            eprintln!("skipping spawn_and_poll_real_subagent: no POTLATCH_BASE_URL");
            return;
        }

        let dir = test_util::unique_test_dir();
        let table = Arc::new(SubagentTable::new());
        let tool = SubagentTool::with_table(Arc::clone(&table), "");

        let args = json!({
            "prompt": "Use the shell tool to run: echo hi-from-subagent",
            "tools": ["shell"]
        });
        let spawn_result = tool.execute(&args, dir.as_str()).unwrap();
        let id = spawn_result
            .strip_prefix("Subagent started: ")
            .unwrap()
            .lines()
            .next()
            .unwrap();
        assert!(!id.is_empty());

        // Poll until done or timeout.
        let mut found = false;
        for _ in 0..120 {
            let poll = tool
                .execute(&json!({"subagent_id": id}), dir.as_str())
                .unwrap();
            if poll.contains("status: done") {
                assert!(
                    poll.contains("hi-from-subagent"),
                    "expected subagent output to contain 'hi-from-subagent', got: {poll}"
                );
                found = true;
                break;
            }
            if poll.contains("status: killed") || poll.contains("error:") {
                panic!("subagent failed: {poll}");
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        assert!(found, "subagent did not complete in time");
    }
}
