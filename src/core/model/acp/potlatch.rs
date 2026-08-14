//! Potlatch harness ACP vendor extension.
//!
//! Active when the model URI vendor is `potlatch` (the ACP backend is the
//! potlatch harness itself, not the Cursor CLI). The harness needs no auth,
//! no mode tracking, and no plan-path handling — its only vendor extension
//! is [`session/inject`](super::client::AcpClient::session_inject) for
//! mid-task follow-up message forwarding.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use serde_json::Value;
use tracing::debug;

use super::capabilities::CapabilityProvider;
use super::client::AcpClient;
use super::types::{InitializeResult, NewSessionResult, SessionModeStateBrief};
use super::vendor::{AcpVendorExtension, AcpVendorState};

/// Per-session vendor state for the potlatch backend. The harness sends no
/// `cursor/*` requests and no mode updates that need tracking, so this is a
/// no-op implementation.
struct NoopVendorState;

impl AcpVendorState for NoopVendorState {
    fn handle_agent_request(&self, _method: &str, _params: &Value, _id: &Value) -> Option<Value> {
        None
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
    ) -> Arc<dyn AcpVendorState> {
        Arc::new(NoopVendorState)
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
