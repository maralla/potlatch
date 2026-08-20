//! ACP backend abstraction and implementations.
//!
//! Vendors (e.g. Cursor CLI, potlatch harness) provide protocol extensions beyond
//! the standard ACP spec. This module defines traits that the generic ACP
//! runtime calls through at each lifecycle stage, so vendor-specific method names
//! (e.g. `cursor/ask_question`, `session/inject`) never appear in generic code.
//!
//! Vendor selection lives here; generic runtime code sees only the traits.

pub mod potlatch;
pub mod cursor;
pub mod default;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use serde_json::Value;

use super::capabilities::CapabilityProvider;
use super::client::AcpClient;
use super::types::{NewSessionResult, SessionModeStateBrief};

/// Backend-specific presentation and capture of a neutral structured-output
/// contract. Backends that understand Potlatch's ACP extension receive real
/// tools; every other ACP vendor gets a marker envelope generated from the
/// same schema.
pub(super) trait StructuredOutputBackend: Send + Sync {
    /// Add this vendor's structured-output instructions to a task/repair
    /// prompt. Tool-capable vendors leave the role prompt untouched.
    fn prepare_prompt(&self, prompt: &str, tools: &[Value]) -> String;

    /// Tool definitions to expose through `session/new`, if supported.
    fn session_tools(&self, tools: &[Value]) -> Option<Vec<Value>>;

    /// Recover structured values from the final response. Tool-capable
    /// vendors return them through the protocol instead.
    fn extract_outputs(&self, response: &str) -> Option<Value>;
}

/// Potlatch's harness implements the structured-output tool extension. Standard
/// ACP vendors use the portable marker renderer.
pub(super) fn resolve_structured_output_backend(
    model_uri: Option<&str>,
) -> Arc<dyn StructuredOutputBackend> {
    let uses_tools = model_uri
        .and_then(|uri| crate::core::config::uri::ModelUri::parse(uri).ok())
        .is_some_and(|uri| uri.vendor == "potlatch");
    if uses_tools {
        Arc::new(potlatch::PotlatchStructuredOutputBackend)
    } else {
        Arc::new(default::DefaultBackend)
    }
}

/// Per-session vendor state, stored on `StreamTextHooks`.
///
/// All methods are called by the runtime/hooks through the vendor extension —
/// never by agent code.
pub trait AcpVendorState: Send + Sync {
    /// Handle an agent→client request. Returns `Some(result)` if the method
    /// is a vendor extension that this state handles, `None` to let the
    /// generic hooks handle it.
    fn handle_agent_request(&self, method: &str, params: &Value, id: &Value) -> Option<Value>;

    /// Handle a `session/update` notification. Returns `true` if consumed
    /// exclusively (the generic hooks skip further processing).
    fn on_session_update(&self, params: &Value) -> bool;

    /// Clear all state for a new task.
    fn clear(&self);

    /// Seed session modes from `session/new` result.
    fn seed_session_modes(&self, modes: &SessionModeStateBrief);

    /// Sync the current mode after a client-initiated mode change.
    fn sync_current_mode(&self, mode_id: &str);

    /// Monotonic notification sequence count (for followup change detection).
    fn notification_seq(&self) -> u64;

    /// Downcast to `Any` for vendor-specific access (e.g. extracting plan
    /// text/paths in `process_response`). Only called by the vendor extension
    /// that created the state — never by generic code.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Vendor extension for the ACP runtime.
///
/// Created at runtime construction when the model vendor matches. The runtime
/// calls through these methods at each lifecycle stage. Agents never call
/// this directly — the vendor extension is fully transparent.
pub trait AcpVendorExtension: Send + Sync {
    /// Create per-session vendor state (stored on `StreamTextHooks`).
    /// Called during spawn. The vendor stores the capability provider and
    /// wires its methods to the vendor-specific protocol.
    fn create_state(
        &self,
        provider: Option<Arc<dyn CapabilityProvider>>,
        agent_bus: Option<crate::core::bus::AgentBus>,
    ) -> Arc<dyn AcpVendorState>;

    /// Authenticate after `initialize` if the vendor requires it.
    /// Called during spawn, after `initialize`, before `session/new`.
    fn authenticate(&self, client: &AcpClient, init: &super::types::InitializeResult)
    -> Result<()>;

    /// Apply preferred session mode after `session/new`.
    /// Called during session start.
    fn try_apply_preferred_mode(
        &self,
        client: &AcpClient,
        session: &NewSessionResult,
        state: &dyn AcpVendorState,
    );

    /// Wait for followup updates after a prompt completes (e.g. plan paths).
    /// Called after the prompt response is received.
    fn wait_for_followup(
        &self,
        state: &dyn AcpVendorState,
        cancel_check: Option<&dyn Fn() -> bool>,
        shutdown: &AtomicBool,
    ) -> Result<()>;

    /// Process the prompt result into vendor-specific response additions.
    /// Merges vendor data (e.g. plan text, plan paths) into the response
    /// string. Called during handoff construction.
    fn process_response(&self, state: &dyn AcpVendorState, response: &mut String);

    /// Forward follow-up messages to the running session mid-task.
    /// Called by the runtime at the polling cadence during `session/prompt`.
    /// The potlatch extension sends `session/inject`; other vendors no-op.
    fn forward_followups(&self, _client: &AcpClient, _session_id: &str, _messages: &[String]) {}
}

/// Resolve the vendor extension for a model URI.
///
/// This is the single dispatch point: the runtime calls this with the model URI
/// and receives the matching extension (or `None` when the vendor has no
/// extension). Each vendor module owns its own detection; the runtime never
/// references vendor names.
///
/// `preferred_session_mode` is forwarded to extensions that use it (Cursor).
pub(super) fn resolve_vendor_extension(
    model_uri: Option<&str>,
    preferred_session_mode: Option<&'static str>,
) -> Option<Arc<dyn AcpVendorExtension>> {
    if let Some(mut ext) = cursor::CursorExtension::new(model_uri) {
        if let Some(mode) = preferred_session_mode {
            ext.set_preferred_session_mode(Some(mode));
        }
        return Some(Arc::new(ext));
    }
    if let Some(ext) = potlatch::PotlatchExtension::new(model_uri) {
        return Some(Arc::new(ext));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn potlatch_uses_tools_and_other_vendors_use_markers() {
        let tools = vec![json!({
            "name": "result",
            "description": "Result.",
            "parameters": {"type": "object"}
        })];

        let potlatch = resolve_structured_output_backend(Some("acp://potlatch/test"));
        assert_eq!(potlatch.session_tools(&tools), Some(tools.clone()));
        assert_eq!(potlatch.prepare_prompt("Task.", &tools), "Task.");

        for uri in [
            None,
            Some("acp://cursor/composer-2"),
            Some("acp://other/model"),
        ] {
            let backend = resolve_structured_output_backend(uri);
            assert_eq!(backend.session_tools(&tools), None);
            assert!(
                backend
                    .prepare_prompt("Task.", &tools)
                    .contains("BREEZE_STRUCTURED_OUTPUT_BEGIN")
            );
        }
    }
}
