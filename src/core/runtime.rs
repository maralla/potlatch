//! In-memory, per-instance agent runtime: identity, the shared process-wide
//! shutdown flag, and observable health.
//!
//! This is GENERAL core: zero Git/GitLab/workspace/session-file assumptions,
//! and nothing here touches disk or network. Health lives only in memory —
//! it resets when the process restarts — and is intentionally never
//! persisted.

use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Observable lifecycle/health state of one agent instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthState {
    /// Being (re)constructed; `from_spawn` is running or about to run.
    Starting,
    /// Constructed and waiting for the next scheduled cycle.
    Idle,
    /// Actively running the named periodic task.
    Busy(String),
    /// The last periodic cycle failed; the agent keeps running.
    Degraded,
    /// The instance stopped (error, panic, or unexpected return) and is
    /// waiting under the supervisor's backoff before restarting.
    Backoff,
    /// Running `on_shutdown`.
    Stopping,
    /// Fully stopped; this instance will not run again.
    Stopped,
}

impl fmt::Display for HealthState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthState::Starting => f.write_str("starting"),
            HealthState::Idle => f.write_str("idle"),
            HealthState::Busy(task) => write!(f, "busy({task})"),
            HealthState::Degraded => f.write_str("degraded"),
            HealthState::Backoff => f.write_str("backoff"),
            HealthState::Stopping => f.write_str("stopping"),
            HealthState::Stopped => f.write_str("stopped"),
        }
    }
}

/// A point-in-time, observable copy of one agent instance's health.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthSnapshot {
    pub state: HealthState,
    /// Periodic cycles that have failed since the last successful cycle.
    /// Reset to zero by a successful cycle.
    pub consecutive_failures: u32,
    /// Number of times the supervisor has restarted this instance's
    /// `CoreAgent` value. Never reset; cumulative for the instance's whole
    /// process lifetime.
    pub restart_count: u32,
    /// The most recent error message, from either a periodic cycle failure
    /// or a supervisor-level restart. `None` until the first failure.
    pub last_error: Option<String>,
}

#[derive(Debug)]
struct HealthInner {
    state: HealthState,
    consecutive_failures: u32,
    restart_count: u32,
    last_error: Option<String>,
}

impl Default for HealthInner {
    fn default() -> Self {
        Self {
            state: HealthState::Starting,
            consecutive_failures: 0,
            restart_count: 0,
            last_error: None,
        }
    }
}

/// Shared, in-memory runtime for one planned agent instance: identity, the
/// process-wide shutdown flag, and observable health.
///
/// One [`AgentRuntime`] backs the whole supervised lifetime of a planned
/// instance (see [`crate::core::workflow`]), so it survives across
/// supervisor restarts of the underlying [`crate::core::agent::CoreAgent`]
/// value — counters and the last error are cumulative for the instance, not
/// reset per construction attempt. Cloning shares the same underlying
/// state; there is no persistence and no network access anywhere in this
/// type.
#[derive(Clone)]
pub struct AgentRuntime {
    agent_id: Arc<str>,
    shutdown: Arc<AtomicBool>,
    health: Arc<Mutex<HealthInner>>,
}

impl AgentRuntime {
    /// Create a runtime for `agent_id`, backed by the process-wide shared
    /// `shutdown` flag. Health starts in [`HealthState::Starting`].
    pub fn new(agent_id: impl Into<Arc<str>>, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            agent_id: agent_id.into(),
            shutdown,
            health: Arc::new(Mutex::new(HealthInner::default())),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn shutdown(&self) -> &Arc<AtomicBool> {
        &self.shutdown
    }

    /// Take an observable snapshot of current health.
    pub fn health(&self) -> HealthSnapshot {
        let inner = self.lock();
        HealthSnapshot {
            state: inner.state.clone(),
            consecutive_failures: inner.consecutive_failures,
            restart_count: inner.restart_count,
            last_error: inner.last_error.clone(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HealthInner> {
        self.health
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set_state(&self, state: HealthState) {
        self.lock().state = state;
    }

    pub fn mark_starting(&self) {
        self.set_state(HealthState::Starting);
    }

    pub fn mark_idle(&self) {
        self.set_state(HealthState::Idle);
    }

    pub fn mark_busy(&self, task_id: impl Into<String>) {
        self.set_state(HealthState::Busy(task_id.into()));
    }

    pub fn mark_stopping(&self) {
        self.set_state(HealthState::Stopping);
    }

    pub fn mark_stopped(&self) {
        self.set_state(HealthState::Stopped);
    }

    /// Record a successful periodic cycle: resets consecutive failures and
    /// returns to idle. Does not touch `restart_count` or `last_error`.
    pub fn record_cycle_success(&self) {
        let mut inner = self.lock();
        inner.consecutive_failures = 0;
        inner.state = HealthState::Idle;
    }

    /// Record a periodic cycle failure: the scheduler keeps running the
    /// agent (this is not a restart), but health is now degraded.
    pub fn record_cycle_failure(&self, error: impl Into<String>) {
        let mut inner = self.lock();
        inner.consecutive_failures += 1;
        inner.last_error = Some(error.into());
        inner.state = HealthState::Degraded;
    }

    /// Record a supervisor-level restart: the instance's construction or
    /// run failed, panicked, or returned unexpectedly, and the supervisor
    /// is about to back off before trying again.
    pub fn record_restart(&self, error: impl Into<String>) {
        let mut inner = self.lock();
        inner.restart_count += 1;
        inner.last_error = Some(error.into());
        inner.state = HealthState::Backoff;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    fn runtime() -> AgentRuntime {
        AgentRuntime::new("agent-test-0", Arc::new(AtomicBool::new(false)))
    }

    #[test]
    fn new_runtime_starts_in_starting_state_with_zeroed_counters() {
        let runtime = runtime();
        let health = runtime.health();
        assert_eq!(health.state, HealthState::Starting);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.restart_count, 0);
        assert_eq!(health.last_error, None);
    }

    #[test]
    fn agent_id_and_shutdown_are_exposed() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = AgentRuntime::new("worker-3", Arc::clone(&shutdown));
        assert_eq!(runtime.agent_id(), "worker-3");
        assert!(Arc::ptr_eq(runtime.shutdown(), &shutdown));
    }

    #[test]
    fn mark_methods_set_the_expected_state() {
        let runtime = runtime();
        runtime.mark_idle();
        assert_eq!(runtime.health().state, HealthState::Idle);
        runtime.mark_busy("poll");
        assert_eq!(
            runtime.health().state,
            HealthState::Busy("poll".to_string())
        );
        runtime.mark_stopping();
        assert_eq!(runtime.health().state, HealthState::Stopping);
        runtime.mark_stopped();
        assert_eq!(runtime.health().state, HealthState::Stopped);
        runtime.mark_starting();
        assert_eq!(runtime.health().state, HealthState::Starting);
    }

    #[test]
    fn cycle_failure_increments_consecutive_failures_and_marks_degraded() {
        let runtime = runtime();
        runtime.record_cycle_failure("boom");
        let health = runtime.health();
        assert_eq!(health.state, HealthState::Degraded);
        assert_eq!(health.consecutive_failures, 1);
        assert_eq!(health.last_error.as_deref(), Some("boom"));
        assert_eq!(health.restart_count, 0);

        runtime.record_cycle_failure("boom again");
        assert_eq!(runtime.health().consecutive_failures, 2);
    }

    #[test]
    fn cycle_success_resets_consecutive_failures_and_marks_idle() {
        let runtime = runtime();
        runtime.record_cycle_failure("boom");
        runtime.record_cycle_success();
        let health = runtime.health();
        assert_eq!(health.state, HealthState::Idle);
        assert_eq!(health.consecutive_failures, 0);
        // Success does not erase the historical last error.
        assert_eq!(health.last_error.as_deref(), Some("boom"));
    }

    #[test]
    fn restart_increments_restart_count_and_marks_backoff() {
        let runtime = runtime();
        runtime.record_restart("crashed");
        let health = runtime.health();
        assert_eq!(health.state, HealthState::Backoff);
        assert_eq!(health.restart_count, 1);
        assert_eq!(health.last_error.as_deref(), Some("crashed"));

        runtime.record_restart("crashed again");
        let health = runtime.health();
        assert_eq!(health.restart_count, 2);
        assert_eq!(health.last_error.as_deref(), Some("crashed again"));
    }

    #[test]
    fn cloned_runtime_shares_the_same_health_and_shutdown() {
        let runtime = runtime();
        let clone = runtime.clone();

        runtime.record_cycle_failure("boom");
        assert_eq!(clone.health().consecutive_failures, 1);

        clone.shutdown().store(true, Ordering::SeqCst);
        assert!(runtime.shutdown().load(Ordering::SeqCst));
    }

    #[test]
    fn health_state_display_matches_variant() {
        assert_eq!(HealthState::Starting.to_string(), "starting");
        assert_eq!(HealthState::Idle.to_string(), "idle");
        assert_eq!(
            HealthState::Busy("poll".to_string()).to_string(),
            "busy(poll)"
        );
        assert_eq!(HealthState::Degraded.to_string(), "degraded");
        assert_eq!(HealthState::Backoff.to_string(), "backoff");
        assert_eq!(HealthState::Stopping.to_string(), "stopping");
        assert_eq!(HealthState::Stopped.to_string(), "stopped");
    }
}
