//! Harness-to-parent requests over the existing ACP stdio connection.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use super::tools::agent_bus::AgentToolCaller;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

type Pending = Arc<Mutex<HashMap<u64, mpsc::SyncSender<Result<Value, String>>>>>;

#[derive(Clone)]
pub struct SharedOutput(Arc<Mutex<Box<dyn Write + Send>>>);

impl SharedOutput {
    pub fn stdout() -> Self {
        Self(Arc::new(Mutex::new(Box::new(std::io::stdout()))))
    }

    /// Wrap an explicit writer (tests inspect what the harness wrote).
    #[cfg(test)]
    pub fn from_writer(writer: Box<dyn Write + Send>) -> Self {
        Self(Arc::new(Mutex::new(writer)))
    }
}

impl Write for SharedOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("ACP stdout lock poisoned"))?
            .write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("ACP stdout lock poisoned"))?
            .flush()
    }
}

pub struct ParentRpc {
    output: SharedOutput,
    next_id: AtomicU64,
    pending: Pending,
}

impl ParentRpc {
    pub fn new(output: SharedOutput) -> Self {
        Self {
            output,
            next_id: AtomicU64::new(1),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = mpsc::sync_channel(1);
        self.pending
            .lock()
            .map_err(|_| anyhow::anyhow!("parent RPC pending lock poisoned"))?
            .insert(id, sender);
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let send_result = (|| {
            let mut output = self.output.clone();
            serde_json::to_writer(&mut output, &message).context("encode parent RPC request")?;
            output
                .write_all(b"\n")
                .and_then(|_| output.flush())
                .context("send parent RPC request")
        })();
        if let Err(error) = send_result {
            self.remove_pending(id);
            return Err(error);
        }
        // Must exceed the parent-side bus request timeout (45s): if this
        // wait fires first, the parent's eventual reply arrives with no
        // pending entry and gets echoed back onto the ACP stream as a junk
        // response (surfacing as orphan-response warnings on the parent).
        match receiver.recv_timeout(Duration::from_secs(60)) {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(anyhow::Error::msg(error)),
            Err(error) => {
                self.remove_pending(id);
                Err(error).context("parent RPC request timed out")
            }
        }
    }

    pub fn handle_response(&self, message: &Value) -> bool {
        if message.get("method").is_some() {
            return false;
        }
        let Some(id) = message.get("id").and_then(Value::as_u64) else {
            return false;
        };
        let sender = self
            .pending
            .lock()
            .ok()
            .and_then(|mut pending| pending.remove(&id));
        let Some(sender) = sender else {
            return false;
        };
        let result = if let Some(error) = message.get("error") {
            Err(error.to_string())
        } else if let Some(result) = message.get("result") {
            Ok(result.clone())
        } else {
            Err("parent RPC response has neither result nor error".to_string())
        };
        let _ = sender.send(result);
        true
    }

    fn remove_pending(&self, id: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(&id);
        }
    }
}

impl AgentToolCaller for ParentRpc {
    fn call(
        &self,
        target: &str,
        operation: &str,
        arguments: Value,
        session_id: &str,
    ) -> Result<Value> {
        let response = self.request(
            "potlatch/agent_tool_call",
            json!({
                "target": target,
                "operation": operation,
                "arguments": arguments,
                "session_id": session_id,
            }),
        )?;
        if let Some(error) = response.get("error").and_then(Value::as_str) {
            bail!("{error}");
        }
        response
            .get("result")
            .cloned()
            .context("parent agent tool response has no result")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn response_is_delivered_to_the_waiting_request() {
        let buffer = Buffer::default();
        let output = SharedOutput(Arc::new(Mutex::new(Box::new(buffer.clone()))));
        let rpc = Arc::new(ParentRpc::new(output));
        let request_rpc = Arc::clone(&rpc);
        let request = std::thread::spawn(move || request_rpc.request("example", json!({})));
        while buffer.0.lock().unwrap().is_empty() {
            std::thread::yield_now();
        }
        assert!(rpc.handle_response(&json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}})));
        assert_eq!(request.join().unwrap().unwrap()["ok"], true);
    }
}
