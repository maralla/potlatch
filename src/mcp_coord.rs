//! MCP HTTP coordinator: role threads register `agent_id`s so Cursor can open MCP against
//! [`crate::mcp_http`]. Tasks are **not** delivered here anymore (orchestrator uses ACP); this layer
//! still implements `tools/call` for `codepair/wait_for_next_task` for tests and legacy clients.
//!
//! Streamable HTTP POST with newline-delimited JSON-RPC. Agent identity: `X-Codepair-Agent-ID`
//! header (optional fallback: `params.clientInfo.agent_id` on `initialize`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::debug;

pub(crate) const HEARTBEAT_TASK_TOKEN: &str = "__CODEPAIR_HEARTBEAT__";
pub(crate) const HEARTBEAT_RESPONSE_TOKEN: &str = "__CODEPAIR_HEARTBEAT_ACK__";
pub(crate) const HEARTBEAT_INSTRUCTION: &str = "Heartbeat only. This is not real work. Immediately call codepair/wait_for_next_task again with response __CODEPAIR_HEARTBEAT_ACK__. Repeated heartbeats are normal; keep ACKing until a real task arrives.";
#[cfg(test)]
const WAIT_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(50);
#[cfg(not(test))]
const WAIT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(45);

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffSubIssue {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentHandoff {
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub mr_title: Option<String>,
    #[serde(default)]
    pub mr_description: Option<String>,
    #[serde(default)]
    pub changes_summary: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub needs_split: Option<String>,
    #[serde(default)]
    pub needs_clarification: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub lgtm: Option<String>,
    #[serde(default)]
    pub sub_issues: Vec<HandoffSubIssue>,
    /// Absolute paths from Cursor plan-mode `tool_call_update` ("Plan saved to file://…"); PMO reads these files after ACP.
    #[serde(default)]
    pub cursor_plan_paths: Vec<String>,
}

struct AgentSlot {
    /// Test builds expose this via [`CoordinatorHandle::submit_mcp_task`]. Release builds leak the
    /// sender with [`std::mem::forget`] so the receiver stays connected without a dead-code field.
    #[cfg(test)]
    mcp_task_tx: std::sync::mpsc::Sender<String>,
    /// Receiver for tasks delivered to the MCP client (`codepair/wait_for_next_task`).
    /// Stays in the coordinator; HTTP handlers lock and `recv` (see design: one consumer).
    mcp_task_rx: Arc<Mutex<Option<std::sync::mpsc::Receiver<String>>>>,
    pending_task: Arc<Mutex<Option<String>>>,
    waiter_generation: Arc<AtomicU64>,
    result_tx: std::sync::mpsc::Sender<AgentHandoff>,
}

impl AgentSlot {
    fn new() -> (Self, std::sync::mpsc::Receiver<AgentHandoff>) {
        let (task_tx, task_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        #[cfg(not(test))]
        std::mem::forget(task_tx);

        let slot = AgentSlot {
            #[cfg(test)]
            mcp_task_tx: task_tx,
            mcp_task_rx: Arc::new(Mutex::new(Some(task_rx))),
            pending_task: Arc::new(Mutex::new(None)),
            waiter_generation: Arc::new(AtomicU64::new(0)),
            result_tx,
        };
        (slot, result_rx)
    }
}

struct CoordinatorInner {
    agents: HashMap<String, AgentSlot>,
}

/// Handle shared by role threads: submit tasks and read agent-reported results.
#[derive(Clone)]
pub struct CoordinatorHandle {
    inner: Arc<Mutex<CoordinatorInner>>,
    pub(crate) shutdown: Arc<AtomicBool>,
}

/// Per-role-instance bridge returned from [`CoordinatorHandle::register_agent`].
pub struct RoleAgentBridge {
    pub agent_id: String,
    result_rx: std::sync::mpsc::Receiver<AgentHandoff>,
    handle: CoordinatorHandle,
}

impl RoleAgentBridge {
    /// Block until the agent reports completion (via the `response` argument of `wait_for_next_task`).
    // Used from `#[cfg(test)]` and by external callers; the binary lib surface triggers `dead_code`.
    #[allow(dead_code)]
    pub fn recv_result(&self) -> Result<String> {
        self.result_rx
            .recv()
            .map(|h| h.response)
            .map_err(|_| anyhow::anyhow!("agent {} result channel closed", self.agent_id))
    }

    /// Non-blocking try: completion if already reported.
    #[allow(dead_code)] // Public API; role threads use [`recv_result_timeout`] in the hot path.
    pub fn try_recv_result(&self) -> Option<String> {
        self.result_rx.try_recv().ok().map(|h| h.response)
    }
}

#[cfg(test)]
impl RoleAgentBridge {
    pub fn recv_handoff_timeout(&self, timeout: Duration) -> Result<Option<AgentHandoff>> {
        use std::sync::mpsc::RecvTimeoutError;
        match self.result_rx.recv_timeout(timeout) {
            Ok(h) => Ok(Some(h)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(anyhow::anyhow!(
                "agent {} result channel closed",
                self.agent_id
            )),
        }
    }
}

impl Drop for RoleAgentBridge {
    fn drop(&mut self) {
        let _ = self.handle.unregister_agent(&self.agent_id);
    }
}

impl CoordinatorHandle {
    /// For [`crate::mcp_http`]: build a handle that shares state with the accept loop.
    pub(crate) fn new_pair(shutdown: Arc<AtomicBool>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CoordinatorInner {
                agents: HashMap::new(),
            })),
            shutdown,
        }
    }

    /// Register an `agent_id` **before** the external MCP client connects.
    /// Returns a bridge for the role thread to send tasks and recv results.
    pub fn register_agent(&self, agent_id: &str) -> Result<RoleAgentBridge> {
        if self.shutdown.load(Ordering::SeqCst) {
            anyhow::bail!("MCP coordinator is shutting down");
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.agents.contains_key(agent_id) {
            anyhow::bail!("agent_id {} already registered", agent_id);
        }
        let (slot, result_rx) = AgentSlot::new();
        inner.agents.insert(agent_id.to_string(), slot);
        Ok(RoleAgentBridge {
            agent_id: agent_id.to_string(),
            result_rx,
            handle: self.clone(),
        })
    }

    fn unregister_agent(&self, agent_id: &str) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.agents.remove(agent_id);
        Ok(())
    }

    fn result_sender(&self, agent_id: &str) -> Result<std::sync::mpsc::Sender<AgentHandoff>> {
        let inner = self.inner.lock().unwrap();
        Ok(inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?
            .result_tx
            .clone())
    }

    fn mcp_task_cell(
        &self,
        agent_id: &str,
    ) -> Result<Arc<Mutex<Option<std::sync::mpsc::Receiver<String>>>>> {
        let inner = self.inner.lock().unwrap();
        Ok(inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?
            .mcp_task_rx
            .clone())
    }

    fn pending_task_cell(&self, agent_id: &str) -> Result<Arc<Mutex<Option<String>>>> {
        let inner = self.inner.lock().unwrap();
        Ok(inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?
            .pending_task
            .clone())
    }

    fn next_waiter_generation(&self, agent_id: &str) -> Result<u64> {
        let inner = self.inner.lock().unwrap();
        let generation = inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?
            .waiter_generation
            .clone();
        Ok(generation.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn is_current_waiter(&self, agent_id: &str, generation: u64) -> Result<bool> {
        let inner = self.inner.lock().unwrap();
        let current = inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?
            .waiter_generation
            .load(Ordering::SeqCst);
        Ok(current == generation)
    }

    pub(crate) fn restore_pending_task(&self, agent_id: &str, task: String) {
        let Ok(cell) = self.pending_task_cell(agent_id) else {
            return;
        };
        let Ok(mut guard) = cell.lock() else {
            return;
        };
        if guard.is_none() {
            *guard = Some(task);
        }
    }

    fn has_agent(&self, agent_id: &str) -> bool {
        self.inner.lock().unwrap().agents.contains_key(agent_id)
    }
}

#[cfg(test)]
impl CoordinatorHandle {
    pub(crate) fn submit_mcp_task(&self, agent_id: &str, prompt: String) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let slot = inner
            .agents
            .get(agent_id)
            .with_context(|| format!("unknown agent_id {}", agent_id))?;
        slot.mcp_task_tx.send(prompt).map_err(|_| {
            anyhow::anyhow!("agent {} disconnected or MCP not consuming tasks", agent_id)
        })
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

pub(crate) fn rpc_error(id: Option<&Value>, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.cloned().unwrap_or(Value::Null),
        "error": { "code": code, "message": message }
    })
}

pub(crate) fn rpc_result(id: &Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn string_arg(msg: &Value, path: &str) -> Option<String> {
    msg.pointer(path)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn sub_issues_arg(msg: &Value) -> Vec<HandoffSubIssue> {
    msg.pointer("/params/arguments/sub_issues")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value::<HandoffSubIssue>(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn handoff_from_tool_call(msg: &Value) -> AgentHandoff {
    AgentHandoff {
        response: string_arg(msg, "/params/arguments/response").unwrap_or_default(),
        decision: string_arg(msg, "/params/arguments/decision"),
        feedback: string_arg(msg, "/params/arguments/feedback"),
        mr_title: string_arg(msg, "/params/arguments/mr_title"),
        mr_description: string_arg(msg, "/params/arguments/mr_description"),
        changes_summary: string_arg(msg, "/params/arguments/changes_summary"),
        reason: string_arg(msg, "/params/arguments/reason"),
        needs_split: string_arg(msg, "/params/arguments/needs_split"),
        needs_clarification: string_arg(msg, "/params/arguments/needs_clarification"),
        instructions: string_arg(msg, "/params/arguments/instructions"),
        question: string_arg(msg, "/params/arguments/question"),
        lgtm: string_arg(msg, "/params/arguments/lgtm"),
        sub_issues: sub_issues_arg(msg),
        cursor_plan_paths: Vec::new(),
    }
}

/// Outcome of handling one JSON-RPC envelope from an MCP HTTP POST.
#[derive(Debug)]
pub enum McpDispatchOutcome {
    /// Return 200 with this JSON body.
    Json(Value),
    /// Return 200 with this JSON body and consider this task delivered only if the HTTP response succeeds.
    JsonWithHeldTask {
        json: Value,
        agent_id: String,
        task: String,
    },
    /// Notification: no JSON body (HTTP 204).
    NoContent,
}

/// Process one JSON-RPC 2.0 object from an MCP client.
pub fn dispatch_mcp_json_rpc(
    msg: &Value,
    coord: &CoordinatorHandle,
    header_agent_id: Option<&str>,
    shutdown: &AtomicBool,
) -> McpDispatchOutcome {
    let method = msg.get("method").and_then(|m| m.as_str());
    let id = msg.get("id").cloned();

    if method == Some("notifications/initialized") {
        debug!("MCP notifications/initialized");
        return McpDispatchOutcome::NoContent;
    }

    let Some(method) = method else {
        return McpDispatchOutcome::Json(rpc_error(None, -32600, "Invalid Request"));
    };

    if method == "initialize" {
        let result = json!({
            "protocolVersion": "2025-03-26",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "codepair-mcp", "version": env!("CARGO_PKG_VERSION") }
        });
        return McpDispatchOutcome::Json(rpc_result(&id.unwrap_or(Value::Null), result));
    }

    if method == "tools/list" {
        // Tasks use ACP (`agent acp` + `session/prompt`); MCP HTTP stays for workspace tooling only.
        let tools = json!({ "tools": [] });
        return McpDispatchOutcome::Json(rpc_result(&id.unwrap_or(Value::Null), tools));
    }

    if method == "tools/call" {
        let Some(aid) = header_agent_id.filter(|s| !s.is_empty()) else {
            return McpDispatchOutcome::Json(rpc_error(
                id.as_ref(),
                -32002,
                "missing agent_id (provide via URL path /mcp/<id> or X-Codepair-Agent-ID header)",
            ));
        };

        if !coord.has_agent(aid) {
            return McpDispatchOutcome::Json(rpc_error(id.as_ref(), -32002, "unknown agent_id"));
        }

        let name = msg
            .pointer("/params/name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if name == "codepair/wait_for_next_task" {
            let cell = match coord.mcp_task_cell(aid) {
                Ok(c) => c,
                Err(e) => {
                    return McpDispatchOutcome::Json(rpc_error(
                        id.as_ref(),
                        -32003,
                        &format!("{}", e),
                    ));
                }
            };
            let pending = match coord.pending_task_cell(aid) {
                Ok(c) => c,
                Err(e) => {
                    return McpDispatchOutcome::Json(rpc_error(
                        id.as_ref(),
                        -32003,
                        &format!("{}", e),
                    ));
                }
            };
            let waiter_generation = match coord.next_waiter_generation(aid) {
                Ok(g) => g,
                Err(e) => {
                    return McpDispatchOutcome::Json(rpc_error(
                        id.as_ref(),
                        -32003,
                        &format!("{}", e),
                    ));
                }
            };

            let handoff = handoff_from_tool_call(msg);
            if handoff.response != HEARTBEAT_RESPONSE_TOKEN {
                let rtx = match coord.result_sender(aid) {
                    Ok(t) => t,
                    Err(e) => {
                        return McpDispatchOutcome::Json(rpc_error(
                            id.as_ref(),
                            -32003,
                            &format!("{}", e),
                        ));
                    }
                };
                let _ = rtx.send(handoff);
            }

            let mut guard = match cell.lock() {
                Ok(g) => g,
                Err(_) => {
                    return McpDispatchOutcome::Json(rpc_error(
                        id.as_ref(),
                        -32003,
                        "task receiver lock poisoned",
                    ));
                }
            };
            let Some(rx) = guard.as_mut() else {
                return McpDispatchOutcome::Json(rpc_error(
                    id.as_ref(),
                    -32003,
                    "no task receiver for this agent",
                ));
            };

            let last_heartbeat = Instant::now();
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    return McpDispatchOutcome::Json(rpc_error(
                        id.as_ref(),
                        -32001,
                        "shutting down",
                    ));
                }
                match coord.is_current_waiter(aid, waiter_generation) {
                    Ok(true) => {}
                    Ok(false) => {
                        return McpDispatchOutcome::Json(rpc_error(
                            id.as_ref(),
                            -32005,
                            "superseded by newer wait_for_next_task",
                        ));
                    }
                    Err(e) => {
                        return McpDispatchOutcome::Json(rpc_error(
                            id.as_ref(),
                            -32003,
                            &format!("{}", e),
                        ));
                    }
                }

                if let Ok(mut pending_guard) = pending.lock()
                    && let Some(task) = pending_guard.take()
                {
                    let payload = json!({ "task": task }).to_string();
                    let result = json!({
                        "content": [{ "type": "text", "text": payload }],
                        "isError": false
                    });
                    return McpDispatchOutcome::JsonWithHeldTask {
                        json: rpc_result(&id.clone().unwrap_or(Value::Null), result),
                        agent_id: aid.to_string(),
                        task,
                    };
                }

                match rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(task) => {
                        let payload = json!({ "task": task.clone() }).to_string();
                        let result = json!({
                            "content": [{ "type": "text", "text": payload }],
                            "isError": false
                        });
                        if matches!(coord.is_current_waiter(aid, waiter_generation), Ok(false)) {
                            coord.restore_pending_task(aid, task);
                            return McpDispatchOutcome::Json(rpc_error(
                                id.as_ref(),
                                -32005,
                                "superseded by newer wait_for_next_task",
                            ));
                        }
                        return McpDispatchOutcome::JsonWithHeldTask {
                            json: rpc_result(&id.clone().unwrap_or(Value::Null), result),
                            agent_id: aid.to_string(),
                            task,
                        };
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if last_heartbeat.elapsed() >= WAIT_HEARTBEAT_INTERVAL {
                            let payload = json!({
                                "task": HEARTBEAT_TASK_TOKEN,
                                "heartbeat": true,
                                "instruction": HEARTBEAT_INSTRUCTION
                            })
                            .to_string();
                            let result = json!({
                                "content": [{ "type": "text", "text": payload }],
                                "isError": false
                            });
                            return McpDispatchOutcome::Json(rpc_result(
                                &id.clone().unwrap_or(Value::Null),
                                result,
                            ));
                        }
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        return McpDispatchOutcome::Json(rpc_error(
                            id.as_ref(),
                            -32004,
                            "task channel disconnected",
                        ));
                    }
                }
            }
        }

        return McpDispatchOutcome::Json(rpc_error(id.as_ref(), -32601, "unknown tool"));
    }

    McpDispatchOutcome::Json(rpc_error(id.as_ref(), -32601, "method not found"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_http::spawn_http_mcp_server;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    fn http_post_json(port: u16, agent_id: &str, body: &str) -> Result<(u16, String)> {
        let req = format!(
            "POST /mcp/{agent_id} HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {body}",
            body.len()
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port))?;
        stream.write_all(req.as_bytes())?;
        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line)?;
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line == "\n" {
                break;
            }
            let l = line.trim_end_matches(['\r', '\n']);
            let lower = l.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("content-length:") {
                content_length = rest.trim().parse().ok();
            }
        }
        let mut body_out = String::new();
        if let Some(n) = content_length {
            let mut buf = vec![0u8; n];
            reader.read_exact(&mut buf)?;
            body_out = String::from_utf8_lossy(&buf).into_owned();
        } else {
            reader.read_to_string(&mut body_out)?;
        }
        Ok((status_code, body_out))
    }

    #[test]
    fn mcp_task_and_result_roundtrip_http() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge = coord.register_agent("worker-0")?;

        let port_c = port;
        let client = thread::spawn(move || -> Result<()> {
            let init = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": { "name": "test", "agent_id": "worker-0" }
                }
            }))?;
            let (_st, _body) = http_post_json(port_c, "worker-0", &init)?;

            let list = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }))?;
            let (_st2, _body2) = http_post_json(port_c, "worker-0", &list)?;

            let wait = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "codepair/wait_for_next_task",
                    "arguments": { "response": "" }
                }
            }))?;
            let (_st3, b3) = http_post_json(port_c, "worker-0", &wait)?;
            let resp: Value = serde_json::from_str(&b3)?;
            let text = resp
                .pointer("/result/content/0/text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            assert!(text.contains("do the thing"), "{}", text);

            let done = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "codepair/wait_for_next_task",
                    "arguments": { "response": "finished" }
                }
            }))?;
            let (_st4, _body4) = http_post_json(port_c, "worker-0", &done)?;

            Ok(())
        });

        assert_eq!(bridge.recv_result()?, "");
        coord.submit_mcp_task("worker-0", "do the thing".to_string())?;
        assert_eq!(bridge.recv_result()?, "finished");
        coord.submit_mcp_task("worker-0", "next".to_string())?;

        client.join().expect("join").expect("client");

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(300));
        Ok(())
    }

    #[test]
    fn wait_for_next_task_returns_heartbeat_while_idle() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge = coord.register_agent("worker-0")?;

        let wait = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": { "response": "" }
            }
        }))?;
        let (_status, body) = http_post_json(port, "worker-0", &wait)?;

        assert_eq!(bridge.recv_result()?, "");
        assert!(body.contains(HEARTBEAT_TASK_TOKEN), "{}", body);
        assert!(body.contains("Repeated heartbeats are normal"), "{}", body);

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    #[test]
    fn heartbeat_response_is_not_forwarded_as_completion() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge = coord.register_agent("worker-0")?;

        let wait = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": { "response": HEARTBEAT_RESPONSE_TOKEN }
            }
        }))?;
        let (_status, body) = http_post_json(port, "worker-0", &wait)?;

        assert!(body.contains(HEARTBEAT_TASK_TOKEN), "{}", body);
        assert!(body.contains("Repeated heartbeats are normal"), "{}", body);
        assert_eq!(
            bridge.recv_handoff_timeout(Duration::from_millis(100))?,
            None
        );

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    #[test]
    fn structured_tool_arguments_are_forwarded() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge = coord.register_agent("worker-0")?;

        let call = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": {
                    "response": "finished",
                    "decision": "request_changes",
                    "feedback": "Please fix this.",
                    "changes_summary": "Adjust reviewer handling.",
                    "mr_title": "Keep reviewer metadata stable"
                }
            }
        }))?;

        let port_c = port;
        let client = thread::spawn(move || -> Result<()> {
            let (_status, _body) = http_post_json(port_c, "worker-0", &call)?;
            Ok(())
        });

        let handoff = bridge
            .recv_handoff_timeout(Duration::from_millis(200))?
            .expect("handoff");
        assert_eq!(handoff.response, "finished");
        assert_eq!(handoff.decision.as_deref(), Some("request_changes"));
        assert_eq!(handoff.feedback.as_deref(), Some("Please fix this."));
        assert_eq!(
            handoff.changes_summary.as_deref(),
            Some("Adjust reviewer handling.")
        );
        assert_eq!(
            handoff.mr_title.as_deref(),
            Some("Keep reviewer metadata stable")
        );

        shutdown.store(true, Ordering::SeqCst);
        let _ = client.join();
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    #[test]
    fn task_is_held_after_heartbeat_until_next_wait_call() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge = coord.register_agent("worker-0")?;

        let wait = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": { "response": "" }
            }
        }))?;
        let (_status1, body1) = http_post_json(port, "worker-0", &wait)?;
        assert_eq!(bridge.recv_result()?, "");
        assert!(body1.contains(HEARTBEAT_TASK_TOKEN), "{}", body1);

        coord.submit_mcp_task("worker-0", "real task".to_string())?;

        let wait_again = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": { "response": HEARTBEAT_RESPONSE_TOKEN }
            }
        }))?;
        let (_status2, body2) = http_post_json(port, "worker-0", &wait_again)?;
        assert!(body2.contains("real task"), "{}", body2);

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    #[test]
    fn concurrent_same_role_instances_keep_task_routing_isolated() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let bridge0 = coord.register_agent("worker-0")?;
        let bridge1 = coord.register_agent("worker-1")?;

        let wait = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codepair/wait_for_next_task",
                "arguments": { "response": "" }
            }
        }))?;

        let wait0 = wait.clone();
        let waiter0 = thread::spawn(move || -> Result<String> {
            let (_status, body) = http_post_json(port, "worker-0", &wait0)?;
            Ok(body)
        });

        let wait1 = wait;
        let waiter1 = thread::spawn(move || -> Result<String> {
            let (_status, body) = http_post_json(port, "worker-1", &wait1)?;
            Ok(body)
        });

        assert_eq!(bridge0.recv_result()?, "");
        assert_eq!(bridge1.recv_result()?, "");

        coord.submit_mcp_task("worker-0", "task for worker-0".to_string())?;
        coord.submit_mcp_task("worker-1", "task for worker-1".to_string())?;

        let body0 = waiter0.join().expect("worker-0 waiter")?;
        let body1 = waiter1.join().expect("worker-1 waiter")?;

        assert!(body0.contains("task for worker-0"), "{}", body0);
        assert!(!body0.contains("task for worker-1"), "{}", body0);
        assert!(body1.contains("task for worker-1"), "{}", body1);
        assert!(!body1.contains("task for worker-0"), "{}", body1);

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(100));
        Ok(())
    }
}
