//! Vendor extension abstraction for the ACP runtime.
//!
//! Vendors (e.g. Cursor CLI) provide protocol extensions beyond the standard
//! ACP spec. This module defines traits that the generic ACP runtime calls
//! through at each lifecycle stage, so vendor-specific method names (e.g.
//! `cursor/ask_question`) never appear in generic code.
//!
//! The only implementation lives in [`super::cursor`].

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use serde_json::Value;

use super::capabilities::CapabilityProvider;
use super::client::AcpClient;
use super::types::{NewSessionResult, SessionModeStateBrief};

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
}
