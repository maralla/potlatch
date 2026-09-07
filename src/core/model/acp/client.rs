//! ACP client: JSON-RPC over newline-delimited stdio with a background reader.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::Child;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{debug, warn};

use super::jsonrpc::{
    JsonRpcError, Outbound, is_incoming_notification, is_incoming_request, is_response_to_request,
    jsonrpc_response_id_as_u64, parse_response_result,
};
use super::types::{
    AuthenticateParams, InitializeParams, InitializeResult, NewSessionParams, NewSessionResult,
    PromptResult, text_prompt,
};

type PendingTx = std::sync::mpsc::Sender<Result<Value, JsonRpcError>>;
type PendingMap = HashMap<u64, PendingTx>;
type SharedPending = Arc<Mutex<PendingMap>>;

/// Handle agent-initiated JSON-RPC (permission prompts, etc.).
pub trait AcpHooks: Send + Sync {
    /// The agent sent a request that expects a JSON `result` on the wire.
    fn handle_agent_request(&self, method: &str, params: &Value, id: &Value) -> Value;

    /// The agent sent a notification (no response).
    fn on_agent_notification(&self, method: &str, params: &Value);
}

fn permission_option_id(opt: &Value) -> Option<&str> {
    opt.get("optionId")
        .or_else(|| opt.get("option_id"))
        .and_then(|v| v.as_str())
}

fn permission_options_array(params: &Value) -> Option<&Vec<Value>> {
    params
        .get("options")
        .or_else(|| params.get("permissionOptions"))
        .and_then(|v| v.as_array())
}

/// Picks an `optionId` from [`session/request_permission`](https://agentclientprotocol.com/protocol/tool-calls#requesting-permission)
/// for non-interactive clients. Prefer `allow_always`, then `allow_once`, then the first option.
///
/// [Cursor’s minimal ACP example](https://cursor.com/docs/cli/acp) uses literal ids `allow-once` /
/// `allow-always`; structured prompts (e.g. mode switches) use opaque ids — the response must
/// echo one of the ids from `params.options`, not the `kind` field alone.
fn pick_auto_permission_option_id(params: &Value) -> String {
    let Some(options) = permission_options_array(params) else {
        return "allow-once".to_string();
    };
    if options.is_empty() {
        return "allow-once".to_string();
    }
    for kind in ["allow_always", "allow_once"] {
        for opt in options {
            if opt.get("kind").and_then(|v| v.as_str()) == Some(kind)
                && let Some(id) = permission_option_id(opt)
            {
                return id.to_string();
            }
        }
    }
    permission_option_id(&options[0])
        .unwrap_or("allow-once")
        .to_string()
}

/// JSON-RPC `result` for agent→client requests when no interactive UI is available.
///
/// Handles permission auto-selection for non-interactive clients.
/// Vendor-specific extension requests (e.g. `cursor/`) are handled by the
/// vendor state on `StreamTextHooks`, not here.
pub fn headless_agent_request_result(method: &str, params: &Value) -> Value {
    if method == "session/request_permission" {
        let option_id = pick_auto_permission_option_id(params);
        return json!({
            "outcome": {
                "outcome": "selected",
                "optionId": option_id
            }
        });
    }

    Value::Object(serde_json::Map::new())
}

pub struct AcpClient {
    /// Shared with the reader thread so closing drops the last `Sender` and stops the writer.
    write_shared: Arc<Mutex<Option<std::sync::mpsc::Sender<String>>>>,
    next_id: Arc<AtomicU64>,
    pending: SharedPending,
    stop: Arc<AtomicBool>,
    reader_join: Mutex<Option<JoinHandle<Result<()>>>>,
    writer_join: Mutex<Option<JoinHandle<()>>>,
}

impl AcpClient {
    /// Connect using stdio-style streams (e.g. `ChildStdout` / `ChildStdin`).
    pub fn from_read_write<R, W>(reader: R, writer: W, hooks: Arc<dyn AcpHooks>) -> Result<Self>
    where
        R: std::io::Read + Send + 'static,
        W: Write + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let (write_tx, write_rx) = std::sync::mpsc::channel::<String>();
        let write_shared = Arc::new(Mutex::new(Some(write_tx)));
        let pending: SharedPending = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(AtomicU64::new(1));

        let writer_join = thread::Builder::new()
            .name("acp-writer".into())
            .spawn(move || {
                let mut w = writer;
                while let Ok(line) = write_rx.recv() {
                    if w.write_all(line.as_bytes()).is_err() {
                        break;
                    }
                    let _ = w.flush();
                }
            })
            .context("spawn acp writer")?;

        let pending_r = Arc::clone(&pending);
        let write_for_reader = Arc::clone(&write_shared);
        let stop_r = Arc::clone(&stop);
        let reader_join = thread::Builder::new()
            .name("acp-reader".into())
            .spawn(move || {
                Self::reader_loop(
                    BufReader::new(reader),
                    hooks,
                    pending_r,
                    write_for_reader,
                    stop_r,
                )
            })
            .context("spawn acp reader")?;

        Ok(Self {
            write_shared,
            next_id,
            pending,
            stop,
            reader_join: Mutex::new(Some(reader_join)),
            writer_join: Mutex::new(Some(writer_join)),
        })
    }

    /// Attach to a subprocess that speaks ACP on inherited stdin/stdout (e.g. `agent acp`).
    ///
    /// Takes ownership of the child's stdin/stdout pipes; keep the returned [`Child`] alive
    /// so the process stays running (stderr may remain inherited).
    pub fn from_child_stdio(mut child: Child, hooks: Arc<dyn AcpHooks>) -> Result<(Self, Child)> {
        let stdin = child
            .stdin
            .take()
            .context("ACP child missing stdin pipe (use Stdio::piped())")?;
        let stdout = child
            .stdout
            .take()
            .context("ACP child missing stdout pipe (use Stdio::piped())")?;
        let client = Self::from_read_write(stdout, stdin, hooks)?;
        Ok((client, child))
    }

    fn reader_loop(
        mut reader: impl BufRead,
        hooks: Arc<dyn AcpHooks>,
        pending: SharedPending,
        write_shared: Arc<Mutex<Option<std::sync::mpsc::Sender<String>>>>,
        stop: Arc<AtomicBool>,
    ) -> Result<()> {
        loop {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => anyhow::bail!("ACP read_line: {}", e),
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let msg: Value = serde_json::from_str(trimmed).with_context(|| {
                format!(
                    "ACP invalid JSON: {}",
                    trimmed.chars().take(200).collect::<String>()
                )
            })?;

            if is_response_to_request(&msg, true) {
                let id = msg
                    .get("id")
                    .and_then(jsonrpc_response_id_as_u64)
                    .context("ACP response id (expected u64 or decimal string)")?;
                let res = parse_response_result(&msg);
                let tx = pending.lock().unwrap().remove(&id);
                if let Some(tx) = tx {
                    let _ = tx.send(res);
                } else {
                    warn!("ACP orphan response for id {}", id);
                }
                continue;
            }

            if is_incoming_request(&msg) {
                let method = msg["method"].as_str().unwrap_or("");
                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                debug!("ACP agent→client request {}", method);

                let result = hooks.handle_agent_request(method, &params, &id);

                let line = Outbound::Response { id, result }
                    .to_json_line()
                    .context("serialize ACP response")?;

                let wg = write_shared.lock().unwrap();

                let Some(ref tx) = *wg else {
                    anyhow::bail!("ACP writer closed");
                };

                if tx.send(line).is_err() {
                    anyhow::bail!("ACP writer disconnected");
                }

                continue;
            }

            if is_incoming_notification(&msg) {
                let method = msg["method"].as_str().unwrap_or("");
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                hooks.on_agent_notification(method, &params);
                continue;
            }

            warn!(
                "ACP unhandled message: {}",
                trimmed.chars().take(120).collect::<String>()
            );
        }
        Ok(())
    }

    fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let mut g = self.pending.lock().unwrap();
            g.insert(id, tx);
        }
        let line = Outbound::Request {
            id,
            method: method.to_string(),
            params,
        }
        .to_json_line()
        .context("serialize ACP request")?;

        let send_out = (|| -> Result<Value> {
            {
                let wg = self.write_shared.lock().unwrap();
                let Some(ref tx) = *wg else {
                    anyhow::bail!("ACP writer channel closed");
                };
                tx.send(line)
                    .map_err(|_| anyhow::anyhow!("ACP writer channel closed"))?;
            }
            match rx.recv() {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(anyhow::anyhow!("ACP RPC error {}: {}", e.code, e.message)),
                Err(_) => Err(anyhow::anyhow!("ACP reader stopped before response")),
            }
        })();

        self.pending.lock().unwrap().remove(&id);
        send_out
    }

    /// [`session/close`](https://agentclientprotocol.com/rfds/session-close) (ACP RFD; agents may
    /// advertise `session.close` in initialize capabilities). Ends the session on a still-running
    /// agent process so a new `session/new` can be opened on the same stdio link. Callers should
    /// treat failures as non-fatal if the agent does not implement this yet.
    pub fn session_close(&self, session_id: &str) -> Result<()> {
        let params = json!({ "sessionId": session_id });
        let _ = self.send_request("session/close", params)?;
        Ok(())
    }

    /// `session/cancel` notification.
    pub fn session_cancel(&self, session_id: &str) -> Result<()> {
        let line = Outbound::Notification {
            method: "session/cancel".to_string(),
            params: json!({ "sessionId": session_id }),
        }
        .to_json_line()
        .context("serialize session/cancel")?;
        let wg = self.write_shared.lock().unwrap();
        let Some(ref tx) = *wg else {
            anyhow::bail!("ACP writer channel closed");
        };
        tx.send(line)
            .map_err(|_| anyhow::anyhow!("ACP writer channel closed"))?;
        Ok(())
    }

    /// `session/inject` notification: push a follow-up message into the running
    /// session's context before the next model call. Potlatch-harness extension;
    /// backends that don't implement it silently ignore the notification.
    pub fn session_inject(&self, session_id: &str, message: &str) -> Result<()> {
        let line = Outbound::Notification {
            method: "session/inject".to_string(),
            params: json!({ "sessionId": session_id, "message": message }),
        }
        .to_json_line()
        .context("serialize session/inject")?;
        let wg = self.write_shared.lock().unwrap();
        let Some(ref tx) = *wg else {
            anyhow::bail!("ACP writer channel closed");
        };
        tx.send(line)
            .map_err(|_| anyhow::anyhow!("ACP writer channel closed"))?;
        Ok(())
    }

    pub fn initialize(&self, params: &InitializeParams) -> Result<InitializeResult> {
        let v = serde_json::to_value(params).context("initialize params")?;
        let r = self.send_request("initialize", v)?;
        serde_json::from_value(r).context("deserialize initialize result")
    }

    pub fn authenticate(&self, params: &AuthenticateParams) -> Result<()> {
        let v = serde_json::to_value(params).context("authenticate params")?;
        let _ = self.send_request("authenticate", v)?;
        Ok(())
    }

    pub fn authenticate_cursor_login(&self) -> Result<()> {
        self.authenticate(&AuthenticateParams {
            method_id: "cursor_login".to_string(),
        })
    }

    pub fn session_new(&self, params: &NewSessionParams) -> Result<NewSessionResult> {
        let v = serde_json::to_value(params).context("session/new params")?;
        let r = self.send_request("session/new", v)?;
        serde_json::from_value(r).context("deserialize session/new result")
    }

    /// [`session/set_config_option`](https://agentclientprotocol.com/protocol/session-config-options):
    /// `config_id` is the option's `id`; `value` must appear in that option's `options` from
    /// `session/new` (see [`crate::core::model::acp::types::select_option_allows_value`]).
    pub fn session_set_config_option(
        &self,
        session_id: &str,
        config_id: &str,
        value: &str,
    ) -> Result<Value> {
        let params = json!({
            "sessionId": session_id,
            "configId": config_id,
            "value": value,
        });
        self.send_request("session/set_config_option", params)
    }

    /// Experimental ACP `session/set_model` (`sessionId` + `modelId`). Used when the agent did
    /// not advertise a model in [`NewSessionResult::config_options`] or `set_config_option` failed.
    pub fn session_set_model(&self, session_id: &str, model_id: &str) -> Result<Value> {
        let params = json!({
            "sessionId": session_id,
            "modelId": model_id,
        });
        self.send_request("session/set_model", params)
    }

    /// [`session/set_mode`](https://agentclientprotocol.com/protocol/session-modes): `mode_id` must
    /// be one of the agent's `availableModes` from `session/new` when that list is non-empty.
    pub fn session_set_mode(&self, session_id: &str, mode_id: &str) -> Result<Value> {
        let params = json!({
            "sessionId": session_id,
            "modeId": mode_id,
        });
        self.send_request("session/set_mode", params)
    }

    pub fn session_prompt(&self, session_id: &str, prompt_text: &str) -> Result<PromptResult> {
        let params = text_prompt(session_id, prompt_text);
        let r = self.send_request("session/prompt", params)?;
        PromptResult::from_value(&r).context("deserialize session/prompt result")
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(mut g) = self.write_shared.lock() {
            *g = None;
        }
        if let Ok(mut wj) = self.writer_join.lock()
            && let Some(h) = wj.take()
        {
            let _ = h.join();
        }
        if let Ok(mut rj) = self.reader_join.lock()
            && let Some(h) = rj.take()
        {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Write};
    use std::thread;
    use std::time::Duration;

    use super::super::transport::{LineTransport, line_channel_pair};
    use super::super::types::{
        ClientCapabilities, ClientFsCapabilities, DEFAULT_PROTOCOL_VERSION, ImplementationInfo,
        model_selector_for_session, select_option_allows_value,
    };

    #[derive(Debug, Default, Clone, Copy)]
    struct AutoAllowPermissions;

    impl AcpHooks for AutoAllowPermissions {
        fn handle_agent_request(&self, method: &str, params: &Value, _id: &Value) -> Value {
            headless_agent_request_result(method, params)
        }

        fn on_agent_notification(&self, _method: &str, _params: &Value) {}
    }

    #[test]
    fn headless_permission_prefers_allow_always_kind() {
        let r = headless_agent_request_result(
            "session/request_permission",
            &json!({
                "sessionId": "s",
                "toolCall": {},
                "options": [
                    { "optionId": "code-mode", "kind": "allow_always" },
                    { "optionId": "ask-mode", "kind": "allow_once" }
                ]
            }),
        );
        assert_eq!(r["outcome"]["optionId"], "code-mode");
    }

    #[test]
    fn headless_permission_empty_options_uses_allow_once_literal() {
        let r = headless_agent_request_result(
            "session/request_permission",
            &json!({ "sessionId": "s", "toolCall": {}, "options": [] }),
        );
        assert_eq!(r["outcome"]["optionId"], "allow-once");
    }

    fn from_line_transport(
        transport: LineTransport,
        hooks: Arc<dyn AcpHooks>,
    ) -> Result<AcpClient> {
        AcpClient::from_read_write(transport.reader, transport.writer, hooks)
    }

    fn shutdown_client(client: AcpClient) -> Result<()> {
        drop(client);
        Ok(())
    }

    fn minimal_init_result(id: &Value) -> String {
        let v = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {},
                "authMethods": []
            }
        });
        format!("{}\n", serde_json::to_string(&v).unwrap())
    }

    fn simple_result(id: &Value, body: Value) -> String {
        let v = json!({ "jsonrpc": "2.0", "id": id, "result": body });
        format!("{}\n", serde_json::to_string(&v).unwrap())
    }

    /// Scripted agent: answers initialize / authenticate / session/new / session/prompt.
    fn run_scripted_agent(agent: LineTransport) -> thread::JoinHandle<Result<()>> {
        thread::spawn(move || {
            let mut reader = BufReader::new(agent.reader);
            let mut writer = agent.writer;
            macro_rules! read_req {
                () => {{
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        anyhow::bail!("agent: unexpected EOF");
                    }
                    serde_json::from_str::<Value>(line.trim()).context("agent parse")?
                }};
            }

            let m = read_req!();
            assert_eq!(m["method"], "initialize");
            write!(writer, "{}", minimal_init_result(&m["id"]))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "authenticate");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/new");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "sessionId": "test-session" }))
            )?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/prompt");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "stopReason": "end_turn" }))
            )?;
            writer.flush()?;

            Ok(())
        })
    }

    #[test]
    fn from_child_stdio_connects_piped_process() -> Result<()> {
        let child = std::process::Command::new("true")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("spawn /bin/true")?;
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let (client, mut child) = AcpClient::from_child_stdio(child, hooks)?;
        shutdown_client(client)?;
        let _ = child.wait();
        Ok(())
    }

    #[test]
    fn full_session_against_scripted_agent() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let agent_thr = run_scripted_agent(agent_tp);
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let client = from_line_transport(client_tp, hooks)?;

        let init = client.initialize(&InitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            client_capabilities: ClientCapabilities {
                fs: ClientFsCapabilities {
                    read_text_file: false,
                    write_text_file: false,
                },
                terminal: false,
            },
            client_info: ImplementationInfo {
                name: "potlatch-test".into(),
                version: "0.0.1".into(),
            },
        })?;
        assert_eq!(init.protocol_version, json!(1));

        client.authenticate_cursor_login()?;

        let sid = client
            .session_new(&NewSessionParams {
                cwd: "/tmp".into(),
                agent_id: None,
                mcp_servers: vec![],
                structured_output_tools: None,
                agent_tools: None,
                context_channels: None,
                write_roots: None,
            })?
            .session_id;
        assert_eq!(sid, "test-session");

        let pr = client.session_prompt(&sid, "ping")?;
        assert_eq!(pr.stop_reason, "end_turn");

        shutdown_client(client)?;
        agent_thr.join().expect("agent join").expect("agent");
        Ok(())
    }

    fn run_scripted_agent_with_set_model(agent: LineTransport) -> thread::JoinHandle<Result<()>> {
        thread::spawn(move || {
            let mut reader = BufReader::new(agent.reader);
            let mut writer = agent.writer;
            macro_rules! read_req {
                () => {{
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        anyhow::bail!("agent: unexpected EOF");
                    }
                    serde_json::from_str::<Value>(line.trim()).context("agent parse")?
                }};
            }

            let m = read_req!();
            assert_eq!(m["method"], "initialize");
            write!(writer, "{}", minimal_init_result(&m["id"]))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "authenticate");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/new");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "sessionId": "test-session" }))
            )?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/set_model");
            assert_eq!(m["params"]["sessionId"], "test-session");
            assert_eq!(m["params"]["modelId"], "composer-2");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/prompt");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "stopReason": "end_turn" }))
            )?;
            writer.flush()?;

            Ok(())
        })
    }

    #[test]
    fn session_set_model_roundtrip() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let agent_thr = run_scripted_agent_with_set_model(agent_tp);
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let client = from_line_transport(client_tp, hooks)?;

        client.initialize(&InitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            client_capabilities: ClientCapabilities {
                fs: ClientFsCapabilities {
                    read_text_file: false,
                    write_text_file: false,
                },
                terminal: false,
            },
            client_info: ImplementationInfo {
                name: "potlatch-test".into(),
                version: "0.0.1".into(),
            },
        })?;
        client.authenticate_cursor_login()?;

        let sid = client
            .session_new(&NewSessionParams {
                cwd: "/tmp".into(),
                agent_id: None,
                mcp_servers: vec![],
                structured_output_tools: None,
                agent_tools: None,
                context_channels: None,
                write_roots: None,
            })?
            .session_id;
        client.session_set_model(&sid, "composer-2")?;

        let pr = client.session_prompt(&sid, "ping")?;
        assert_eq!(pr.stop_reason, "end_turn");

        shutdown_client(client)?;
        agent_thr.join().expect("agent join").expect("agent");
        Ok(())
    }

    fn run_scripted_agent_with_set_mode(agent: LineTransport) -> thread::JoinHandle<Result<()>> {
        thread::spawn(move || {
            let mut reader = BufReader::new(agent.reader);
            let mut writer = agent.writer;
            macro_rules! read_req {
                () => {{
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        anyhow::bail!("agent: unexpected EOF");
                    }
                    serde_json::from_str::<Value>(line.trim()).context("agent parse")?
                }};
            }

            let m = read_req!();
            assert_eq!(m["method"], "initialize");
            write!(writer, "{}", minimal_init_result(&m["id"]))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "authenticate");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/new");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "sessionId": "test-session" }))
            )?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/set_mode");
            assert_eq!(m["params"]["sessionId"], "test-session");
            assert_eq!(m["params"]["modeId"], "code");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/prompt");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "stopReason": "end_turn" }))
            )?;
            writer.flush()?;

            Ok(())
        })
    }

    #[test]
    fn session_set_mode_roundtrip() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let agent_thr = run_scripted_agent_with_set_mode(agent_tp);
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let client = from_line_transport(client_tp, hooks)?;

        client.initialize(&InitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            client_capabilities: ClientCapabilities {
                fs: ClientFsCapabilities {
                    read_text_file: false,
                    write_text_file: false,
                },
                terminal: false,
            },
            client_info: ImplementationInfo {
                name: "potlatch-test".into(),
                version: "0.0.1".into(),
            },
        })?;
        client.authenticate_cursor_login()?;

        let sid = client
            .session_new(&NewSessionParams {
                cwd: "/tmp".into(),
                agent_id: None,
                mcp_servers: vec![],
                structured_output_tools: None,
                agent_tools: None,
                context_channels: None,
                write_roots: None,
            })?
            .session_id;
        client.session_set_mode(&sid, "code")?;

        let pr = client.session_prompt(&sid, "ping")?;
        assert_eq!(pr.stop_reason, "end_turn");

        shutdown_client(client)?;
        agent_thr.join().expect("agent join").expect("agent");
        Ok(())
    }

    fn run_scripted_agent_config_option_then_prompt(
        agent: LineTransport,
    ) -> thread::JoinHandle<Result<()>> {
        thread::spawn(move || {
            let mut reader = BufReader::new(agent.reader);
            let mut writer = agent.writer;
            macro_rules! read_req {
                () => {{
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        anyhow::bail!("agent: unexpected EOF");
                    }
                    serde_json::from_str::<Value>(line.trim()).context("agent parse")?
                }};
            }

            let m = read_req!();
            assert_eq!(m["method"], "initialize");
            write!(writer, "{}", minimal_init_result(&m["id"]))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "authenticate");
            write!(writer, "{}", simple_result(&m["id"], json!({})))?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/new");
            write!(
                writer,
                "{}",
                simple_result(
                    &m["id"],
                    json!({
                        "sessionId": "test-session",
                        "configOptions": [{
                            "id": "model",
                            "category": "model",
                            "type": "select",
                            "options": [
                                { "value": "composer-2", "name": "Composer 2" }
                            ]
                        }]
                    }),
                )
            )?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/set_config_option");
            assert_eq!(m["params"]["sessionId"], "test-session");
            assert_eq!(m["params"]["configId"], "model");
            assert_eq!(m["params"]["value"], "composer-2");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "configOptions": [] }))
            )?;
            writer.flush()?;

            let m = read_req!();
            assert_eq!(m["method"], "session/prompt");
            write!(
                writer,
                "{}",
                simple_result(&m["id"], json!({ "stopReason": "end_turn" }))
            )?;
            writer.flush()?;

            Ok(())
        })
    }

    #[test]
    fn session_set_config_option_roundtrip_matches_acp_spec() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let agent_thr = run_scripted_agent_config_option_then_prompt(agent_tp);
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let client = from_line_transport(client_tp, hooks)?;

        client.initialize(&InitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            client_capabilities: ClientCapabilities {
                fs: ClientFsCapabilities {
                    read_text_file: false,
                    write_text_file: false,
                },
                terminal: false,
            },
            client_info: ImplementationInfo {
                name: "potlatch-test".into(),
                version: "0.0.1".into(),
            },
        })?;
        client.authenticate_cursor_login()?;

        let session = client.session_new(&NewSessionParams {
            cwd: "/tmp".into(),
            agent_id: None,
            mcp_servers: vec![],
            structured_output_tools: None,
            agent_tools: None,
            context_channels: None,
            write_roots: None,
        })?;
        assert!(
            session
                .config_options
                .as_ref()
                .is_some_and(|c| !c.is_empty())
        );

        let sid = session.session_id;
        let opt = model_selector_for_session(session.config_options.as_deref().unwrap())
            .expect("model option");
        assert!(select_option_allows_value(opt, "composer-2"));
        client.session_set_config_option(&sid, &opt.id, "composer-2")?;

        let pr = client.session_prompt(&sid, "ping")?;
        assert_eq!(pr.stop_reason, "end_turn");

        shutdown_client(client)?;
        agent_thr.join().expect("agent join").expect("agent");
        Ok(())
    }

    #[test]
    fn permission_request_echoes_structured_option_id() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();

        let agent = thread::spawn(move || -> Result<()> {
            let mut reader = BufReader::new(agent_tp.reader);
            let mut writer = agent_tp.writer;
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let m: Value = serde_json::from_str(line.trim())?;
            assert_eq!(m["method"], "initialize");
            write!(writer, "{}", minimal_init_result(&m["id"]))?;
            writer.flush()?;

            let perm = json!({
                "jsonrpc": "2.0",
                "id": 9001,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "s",
                    "toolCall": {
                        "toolCallId": "tc1",
                        "title": "run",
                        "kind": "other",
                        "status": "pending"
                    },
                    "options": [{
                        "optionId": "run-once-xyz",
                        "name": "Run once",
                        "kind": "allow_once"
                    }]
                }
            });
            writeln!(writer, "{}", serde_json::to_string(&perm)?)?;
            writer.flush()?;

            line.clear();
            reader.read_line(&mut line)?;
            let resp: Value = serde_json::from_str(line.trim())?;
            assert_eq!(resp["id"], 9001);
            assert_eq!(resp["result"]["outcome"]["optionId"], "run-once-xyz");
            let _ = done_tx.send(());
            Ok(())
        });

        let client = from_line_transport(client_tp, hooks)?;
        let _ = client.initialize(&InitializeParams {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            client_capabilities: ClientCapabilities::default(),
            client_info: ImplementationInfo {
                name: "t".into(),
                version: "1".into(),
            },
        })?;

        done_rx
            .recv_timeout(Duration::from_secs(2))
            .context("timed out waiting for permission round-trip")?;
        shutdown_client(client)?;
        agent.join().expect("agent")?;
        Ok(())
    }

    #[test]
    fn session_close_roundtrip() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let agent = thread::spawn(move || -> Result<()> {
            let mut reader = BufReader::new(agent_tp.reader);
            let mut writer = agent_tp.writer;
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let m: Value = serde_json::from_str(line.trim())?;
            assert_eq!(m["method"], "session/close");
            assert_eq!(m["params"]["sessionId"], "sid-1");
            write!(writer, "{}", simple_result(&m["id"], json!(null)))?;
            writer.flush()?;
            Ok(())
        });
        let client = from_line_transport(client_tp, hooks)?;
        client.session_close("sid-1")?;
        shutdown_client(client)?;
        agent.join().expect("agent join").expect("agent");
        Ok(())
    }

    #[test]
    fn session_cancel_is_notification_without_id() -> Result<()> {
        let (client_tp, agent_tp) = line_channel_pair();
        let hooks: Arc<dyn AcpHooks> = Arc::new(AutoAllowPermissions);
        let agent = thread::spawn(move || -> Result<()> {
            let LineTransport { reader, writer } = agent_tp;
            drop(writer);
            let mut reader = BufReader::new(reader);
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let m: Value = serde_json::from_str(line.trim())?;
            assert_eq!(m["method"], "session/cancel");
            assert_eq!(m["params"]["sessionId"], "abc");
            Ok(())
        });
        let client = from_line_transport(client_tp, hooks)?;
        client.session_cancel("abc")?;
        agent.join().expect("a")?;
        shutdown_client(client)?;
        Ok(())
    }
}
