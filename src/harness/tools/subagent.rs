//! Subagent tool: spawn `potlatch harness` as a child process and drive it via
//! ACP JSON-RPC over stdio. The subagent runs asynchronously — the parent gets
//! a `subagent_id` immediately and polls for accumulated output. Subagents are
//! one-shot: each runs a single `session/prompt` and terminates when it
//! finishes. The harness does not preserve conversation history across
//! `session/prompt` calls, so follow-up messages would start fresh anyway.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{debug, warn};

use super::Tool;

const MAX_OUTPUT: usize = 50_000;

/// A running subagent session: a `potlatch harness` child process plus the
/// shared output buffer the reader thread appends to.
struct Subagent {
    child: Child,
    /// Kept alive so the child's stdin pipe stays open until the subagent is
    /// dropped or killed; dropping it signals EOF to the child.
    stdin: Option<ChildStdin>,
    started_at: Instant,
    /// Accumulated output from `session/update` notifications and the final
    /// `session/prompt` response. Shared with the reader thread.
    output_buf: Arc<Mutex<String>>,
    /// Set once the reader thread observes the `session/prompt` response or
    /// the process exits.
    done: Arc<std::sync::atomic::AtomicBool>,
    /// Set when the subagent was killed via `kill`/`kill_all`.
    killed: Arc<std::sync::atomic::AtomicBool>,
    /// Populated when the reader thread hits an error or the process exits
    /// with a non-zero code.
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
    /// `session/update` notifications and the final response asynchronously.
    pub fn spawn(
        &self,
        prompt: &str,
        model: &str,
        tools: Option<&[String]>,
        cwd: &str,
    ) -> Result<String> {
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
        let session_params = if let Some(names) = tools {
            json!({ "cwd": cwd, "mcpServers": [], "tools": names })
        } else {
            json!({ "cwd": cwd, "mcpServers": [] })
        };
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
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let killed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None::<String>));

        // Fire the prompt and hand the stdout reader to a background thread.
        // The prompt request id is tracked so the reader recognizes the final
        // response and marks the subagent done.
        let prompt_id = driver.send_request_raw(
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
        let error_r = Arc::clone(&error);
        thread::Builder::new()
            .name("subagent-reader".into())
            .spawn(move || {
                run_reader(
                    reader_stdout,
                    prompt_id,
                    output_buf_r,
                    done_r,
                    killed_r,
                    error_r,
                );
            })
            .context("spawn subagent reader thread")?;

        // Keep the stdin handle alive on the Subagent so we can close it on
        // kill (dropping stdin signals EOF to the child).
        let id = format!("subagent-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let subagent = Subagent {
            child,
            stdin: Some(driver.stdin),
            started_at: Instant::now(),
            output_buf,
            done,
            killed,
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
            && sub.error.lock().unwrap().is_none()
        {
            *sub.error.lock().unwrap() = Some(format!("subagent exited with code {code}"));
        }

        let running = !sub.done.load(Ordering::SeqCst) && sub.try_reap().is_none();
        let killed = sub.killed.load(Ordering::SeqCst);
        let status = if killed {
            "killed"
        } else if running {
            "running"
        } else {
            "done"
        };

        let output = sub.output_buf.lock().unwrap().clone();
        let error = sub.error.lock().unwrap().clone();
        let elapsed = sub.started_at.elapsed();

        let mut result = format!(
            "subagent: {subagent_id}\nstatus: {status}\nelapsed: {:.1}s",
            elapsed.as_secs_f64()
        );
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
        // Drop stdin to signal EOF, then kill the process group to be sure.
        sub.stdin.take();
        let _ = sub.child.kill();
        drop(subagents);
        self.poll(subagent_id)
    }

    /// Kill all running subagents. Called on session close.
    pub fn kill_all(&self) {
        let ids: Vec<String> = self.subagents.lock().unwrap().keys().cloned().collect();
        for id in ids {
            let _ = self.kill(&id);
        }
    }
}

impl Default for SubagentTable {
    fn default() -> Self {
        Self::new()
    }
}

impl super::SessionState for SubagentTable {
    fn shutdown(&self) {
        self.kill_all();
    }
}

/// Minimal inline ACP JSON-RPC driver for the synchronous handshake phase.
/// The stdout reader is moved into the reader thread after the handshake; the
/// stdin stays with the driver (and is handed back to the `Subagent`).
struct AcpDriver {
    stdin: ChildStdin,
    stdout: Option<BufReader<std::process::ChildStdout>>,
    next_id: u64,
}

impl AcpDriver {
    fn new(stdin: ChildStdin, stdout: std::process::ChildStdout) -> Self {
        Self {
            stdin,
            stdout: Some(BufReader::new(stdout)),
            next_id: 1,
        }
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
        self.stdin
            .write_all(s.as_bytes())
            .with_context(|| format!("write ACP request `{method}`"))?;
        self.stdin.flush()?;
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
/// text chunks to the shared buffer, and mark done when the `session/prompt`
/// response (matching `prompt_id`) arrives or the stream ends.
fn run_reader(
    mut reader: impl BufRead,
    prompt_id: u64,
    output_buf: Arc<Mutex<String>>,
    done: Arc<std::sync::atomic::AtomicBool>,
    killed: Arc<std::sync::atomic::AtomicBool>,
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

        // Agent→client request (e.g. session/request_permission). Reply with
        // an auto-allow so the subagent keeps running.
        if msg.get("method").is_some()
            && msg.get("id").is_some()
            && msg.get("id") != Some(&Value::Null)
        {
            // We can't write back here (stdin is owned by the Subagent), so
            // just drop it; the harness treats a missing permission reply as
            // a deny, which is acceptable for a one-shot subagent.
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

        // Response to our session/prompt request.
        if msg.get("method").is_none() {
            let resp_id = msg.get("id").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            });
            if resp_id == Some(prompt_id) {
                if let Some(message) = msg
                    .get("result")
                    .and_then(|r| r.get("message"))
                    .and_then(|m| m.as_str())
                    && !message.is_empty()
                {
                    output_buf.lock().unwrap().push_str(message);
                }
                done.store(true, Ordering::SeqCst);
                return;
            }
        }
    }

    // Stream ended without a matching response. If we weren't killed, record
    // an error; otherwise just mark done.
    if !killed.load(Ordering::SeqCst) && error.lock().unwrap().is_none() {
        *error.lock().unwrap() = Some("subagent stream ended before response".into());
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

pub struct SubagentTool {
    table: Arc<SubagentTable>,
    model: String,
}

impl SubagentTool {
    /// Construct with session-scoped state. Creates a `SubagentTable` in
    /// `SessionStates` if not already present (first prompt), then retrieves
    /// it so subagents survive across prompts and are killed on session close.
    pub fn new(states: &mut super::SessionStates, _cwd: &str, model: &str) -> Self {
        if states.get::<SubagentTable>().is_none() {
            states.insert(Arc::new(SubagentTable::new()));
        }
        Self {
            table: states.get::<SubagentTable>().unwrap(),
            model: model.to_string(),
        }
    }

    /// Construct with an externally-owned table (for tests).
    pub fn with_table(table: Arc<SubagentTable>, model: &str) -> Self {
        Self {
            table,
            model: model.to_string(),
        }
    }
}

impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "subagent"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Spawn a subagent — a separate potlatch harness instance with its own LLM session. The subagent runs asynchronously in the background. Returns a subagent_id immediately; poll with subagent_id to get accumulated output. Use for parallel exploration, independent research tasks, or dividing complex work. The subagent has no context from the parent session — provide everything it needs in the prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the subagent. Be specific — the subagent has no context from the parent session."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model for the subagent (e.g. 'model1-fp8?thinking=true'). Defaults to the parent's model."
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the subagent can use (e.g. ['read','grep','shell']). Omit for all tools."
                    },
                    "subagent_id": {
                        "type": "string",
                        "description": "Poll a running subagent. Returns accumulated output and status."
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

        if let Some(id) = subagent_id {
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

        let id = self.table.spawn(prompt, model, tools.as_deref(), cwd)?;
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
        assert!(props.contains_key("kill"));
    }

    /// Spawn a subagent that runs `echo hi` via the shell tool and verifies we
    /// can poll its output. This is an integration test that requires the
    /// `potlatch` binary to be runnable (we use the current executable).
    #[test]
    fn spawn_and_poll_real_subagent() {
        // The test harness sets up a real LLM client via env vars. Skip if no
        // BREEZE_BASE_URL is configured — the subagent needs an LLM to run.
        if std::env::var("BREEZE_BASE_URL").is_err() {
            eprintln!("skipping spawn_and_poll_real_subagent: no BREEZE_BASE_URL");
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
