//! Potlatch harness ACP vendor extension.
//!
//! Active when the model URI vendor is `potlatch` (the ACP backend is the
//! potlatch harness itself, not the Cursor CLI). The harness needs no auth,
//! no mode tracking, and no plan-path handling — its only vendor extension
//! is [`session/inject`](super::super::client::AcpClient::session_inject) for
//! mid-task follow-up message forwarding.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;
use tracing::debug;

use super::super::capabilities::CapabilityProvider;
use super::super::client::AcpClient;
use super::super::types::{InitializeResult, NewSessionResult, SessionModeStateBrief};
use super::{AcpVendorExtension, AcpVendorState, StructuredOutputBackend};

/// Per-session vendor state for the potlatch backend. The harness sends no
/// `cursor/*` requests and no mode updates that need tracking, so this is a
/// no-op implementation.
struct PotlatchVendorState {
    agent_bus: Option<crate::core::bus::AgentBus>,
}

impl AcpVendorState for PotlatchVendorState {
    fn handle_agent_request(&self, method: &str, params: &Value, _id: &Value) -> Option<Value> {
        if method != "potlatch/agent_tool_call" {
            return None;
        }
        let result = (|| {
            let bus = self
                .agent_bus
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("cross-agent bus is unavailable"))?;
            let target = params
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("agent tool target is required"))?;
            let operation = params
                .get("operation")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("agent tool operation is required"))?;
            let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
            bus.request(
                target,
                operation.to_string(),
                arguments,
                Duration::from_secs(45),
            )
        })();
        Some(match result {
            Ok(value) => serde_json::json!({ "result": value }),
            Err(error) => serde_json::json!({ "error": format!("{error:#}") }),
        })
    }

    fn on_session_update(&self, _params: &Value) -> bool {
        false
    }

    fn clear(&self) {}

    fn seed_session_modes(&self, _modes: &SessionModeStateBrief) {}

    fn sync_current_mode(&self, _mode_id: &str) {}

    fn notification_seq(&self) -> u64 {
        0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Potlatch harness ACP vendor extension. Auto-detected from the model URI
/// vendor (`potlatch`). The only non-noop behavior is mid-task follow-up
/// forwarding via `session/inject`.
pub struct PotlatchExtension;

/// The Potlatch harness receives structured-output contracts as real tools in
/// `session/new`; no marker instructions are added to role prompts.
pub(super) struct PotlatchStructuredOutputBackend;

impl StructuredOutputBackend for PotlatchStructuredOutputBackend {
    fn prepare_prompt(&self, prompt: &str, _tools: &[Value]) -> String {
        prompt.to_string()
    }

    fn session_tools(&self, tools: &[Value]) -> Option<Vec<Value>> {
        (!tools.is_empty()).then(|| tools.to_vec())
    }

    fn extract_outputs(&self, _response: &str) -> Option<Value> {
        None
    }
}

impl PotlatchExtension {
    /// Create a Potlatch extension if the model URI vendor is `potlatch`.
    pub fn new(model_uri: Option<&str>) -> Option<Self> {
        let is_potlatch = model_uri
            .and_then(|uri| crate::core::config::uri::ModelUri::parse(uri).ok())
            .is_some_and(|u| u.vendor == "potlatch");

        if !is_potlatch {
            return None;
        }

        Some(Self)
    }
}

impl AcpVendorExtension for PotlatchExtension {
    fn create_state(
        &self,
        _provider: Option<Arc<dyn CapabilityProvider>>,
        agent_bus: Option<crate::core::bus::AgentBus>,
    ) -> Arc<dyn AcpVendorState> {
        Arc::new(PotlatchVendorState { agent_bus })
    }

    fn authenticate(&self, _client: &AcpClient, _init: &InitializeResult) -> Result<()> {
        Ok(())
    }

    fn try_apply_preferred_mode(
        &self,
        _client: &AcpClient,
        _session: &NewSessionResult,
        _state: &dyn AcpVendorState,
    ) {
    }

    fn wait_for_followup(
        &self,
        _state: &dyn AcpVendorState,
        _cancel_check: Option<&dyn Fn() -> bool>,
        _shutdown: &AtomicBool,
    ) -> Result<()> {
        Ok(())
    }

    fn process_response(&self, _state: &dyn AcpVendorState, _response: &mut String) {}

    fn forward_followups(&self, client: &AcpClient, session_id: &str, messages: &[String]) {
        for msg in messages {
            if let Err(e) = client.session_inject(session_id, msg) {
                debug!(
                    target: "potlatch::acp",
                    err = %e,
                    "session/inject failed (backend may not support it)"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structured_output_uses_session_tools_without_changing_prompt() {
        let tools = vec![json!({
            "name": "plan",
            "description": "Plan result.",
            "parameters": {"type": "object"}
        })];

        assert_eq!(
            PotlatchStructuredOutputBackend.prepare_prompt("Triage.", &tools),
            "Triage."
        );
        assert_eq!(
            PotlatchStructuredOutputBackend.session_tools(&tools),
            Some(tools)
        );
        assert_eq!(
            PotlatchStructuredOutputBackend.extract_outputs("anything"),
            None
        );
    }

    #[test]
    fn potlatch_vendor_routes_agent_tool_calls_to_the_in_process_bus() {
        let bus = crate::core::bus::AgentBus::new();
        let inbox = bus.register("web", vec![]).unwrap();
        let worker = std::thread::spawn(move || {
            let request = inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
            let payload = request.payload.clone();
            request.respond(Ok(payload));
        });
        let state = PotlatchExtension.create_state(None, Some(bus));
        let response = state
            .handle_agent_request(
                "potlatch/agent_tool_call",
                &json!({
                    "target": "web",
                    "operation": "run",
                    "arguments": {"query": "rust"}
                }),
                &json!(1),
            )
            .unwrap();
        assert_eq!(response["result"]["query"], "rust");
        worker.join().unwrap();
    }

    #[test]
    fn potlatch_vendor_does_not_claim_other_agent_requests() {
        let state = PotlatchExtension.create_state(None, None);
        assert!(
            state
                .handle_agent_request("cursor/ask_question", &json!({}), &json!(1))
                .is_none()
        );
    }
}
