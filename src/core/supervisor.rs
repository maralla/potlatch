//! Supervises one planned [`CoreAgent`] instance.
//!
//! One supervisor runs per planned instance (see
//! [`crate::core::workflow::Workflow::run`]), on its own thread. It
//! constructs and runs the agent, catching startup errors, lifecycle
//! errors, unexpected returns, and panics, then restarts indefinitely with
//! shutdown-aware exponential backoff while global shutdown has not been
//! requested — and never restarts once shutdown fires.
//!
//! Exactly one [`AgentRuntime`] backs the whole supervised lifetime of an
//! instance, so failure/restart counters and health are cumulative across
//! restarts, not reset per construction attempt.
//!
//! GENERAL core: zero filesystem/network/Git assumptions. This module only
//! touches process-local atomics and the in-memory [`AgentRuntime`] health
//! cell.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use tracing::{error, warn};

use crate::core::agent::CoreAgent;
use crate::core::retry::{INITIAL_DELAY, MAX_DELAY};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::{AgentSpawnContext, WorkflowContext};

/// Supervise one planned instance of agent `A` for the remaining lifetime
/// of the process.
pub(crate) fn supervise<A>(workflow: WorkflowContext, instance_id: usize) -> Result<()>
where
    A: CoreAgent<SpawnContext = AgentSpawnContext>,
{
    let agent_id = format!("{}-{instance_id}", A::name());
    let shutdown = Arc::clone(&workflow.shutdown);
    let runtime = AgentRuntime::new(agent_id, Arc::clone(&shutdown));
    let attempt_runtime = runtime.clone();

    run_supervised(
        &runtime,
        &shutdown,
        INITIAL_DELAY,
        MAX_DELAY,
        crate::util::sleep,
        move || {
            let ctx = AgentSpawnContext {
                workflow: workflow.clone_for_spawn(),
                instance_id,
                runtime: attempt_runtime.clone(),
            };
            A::run_from(ctx)
        },
    )
}

/// Core supervision loop, parameterized over the restart attempt and the
/// shutdown-aware backoff wait so it can be exercised deterministically in
/// tests without real sleeps or a live [`CoreAgent`].
fn run_supervised<F, W>(
    runtime: &AgentRuntime,
    shutdown: &AtomicBool,
    initial_delay: Duration,
    max_delay: Duration,
    mut wait: W,
    mut attempt: F,
) -> Result<()>
where
    F: FnMut() -> Result<()>,
    W: FnMut(&AtomicBool, Duration) -> bool,
{
    let mut delay = initial_delay;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            runtime.mark_stopped();
            return Ok(());
        }

        runtime.mark_starting();
        let outcome = panic::catch_unwind(AssertUnwindSafe(&mut attempt));

        // Never restart once shutdown has fired, even if it fired during
        // (rather than before) this attempt.
        if shutdown.load(Ordering::SeqCst) {
            runtime.mark_stopped();
            return Ok(());
        }

        runtime.record_restart(describe_outcome(runtime.agent_id(), outcome));
        log_health(runtime);

        if wait(shutdown, delay) {
            runtime.mark_stopped();
            return Ok(());
        }
        delay = (delay * 2).min(max_delay);
    }
}

type AttemptOutcome = std::thread::Result<Result<()>>;

/// Turn one supervised attempt's outcome into a loggable, storable error
/// message, logging along the way. An `Ok(())` return before shutdown was
/// requested is itself unexpected — the agent should only ever stop because
/// `shutdown` was set — so it is treated the same as an error.
fn describe_outcome(agent_id: &str, outcome: AttemptOutcome) -> String {
    match outcome {
        Ok(Ok(())) => {
            warn!("{agent_id}: exited before shutdown was requested, restarting");
            "agent exited before shutdown was requested".to_string()
        }
        Ok(Err(error)) => {
            error!("{agent_id}: {error:#}");
            format!("{error:#}")
        }
        Err(panic_payload) => {
            // `panic_payload.as_ref()` matters here: taking `&panic_payload`
            // (a `&Box<dyn Any + Send>`) would coerce to a trait object over
            // the `Box` itself rather than its contents, since `Box<dyn Any
            // + Send>` is itself `'static` and satisfies `Any`. `as_ref()`
            // derefs through the box first, exposing the actual payload.
            let message = panic_message(panic_payload.as_ref());
            error!("{agent_id}: panicked: {message}");
            format!("panicked: {message}")
        }
    }
}

/// Log an observable snapshot of health right after a restart is recorded,
/// so operators can see cumulative restart/failure counts without any
/// persisted state.
fn log_health(runtime: &AgentRuntime) {
    let health = runtime.health();
    warn!(
        "{}: health={} restart_count={} consecutive_failures={} last_error={}",
        runtime.agent_id(),
        health.state,
        health.restart_count,
        health.consecutive_failures,
        health.last_error.as_deref().unwrap_or("none"),
    );
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use anyhow::anyhow;

    use super::*;
    use crate::core::runtime::HealthState;

    fn test_runtime(shutdown: &Arc<AtomicBool>) -> AgentRuntime {
        AgentRuntime::new("test-0", Arc::clone(shutdown))
    }

    #[test]
    fn does_not_restart_or_attempt_when_shutdown_is_already_set() {
        let shutdown = Arc::new(AtomicBool::new(true));
        let runtime = test_runtime(&shutdown);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_closure = Arc::clone(&attempts);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            |_, _| unreachable!("must not wait when already shut down"),
            move || {
                attempts_for_closure.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        );

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
        assert_eq!(runtime.health().state, HealthState::Stopped);
        assert_eq!(runtime.health().restart_count, 0);
    }

    #[test]
    fn restarts_on_startup_error_and_records_last_error() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_closure = Arc::clone(&attempts);
        let shutdown_for_wait = Arc::clone(&shutdown);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            move |_, _| {
                shutdown_for_wait.store(true, Ordering::SeqCst);
                false
            },
            move || {
                attempts_for_closure.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!("boom"))
            },
        );

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let health = runtime.health();
        assert_eq!(health.restart_count, 1);
        assert_eq!(health.last_error.as_deref(), Some("boom"));
    }

    #[test]
    fn catches_panics_and_restarts() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_closure = Arc::clone(&attempts);
        let shutdown_for_wait = Arc::clone(&shutdown);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            move |_, _| {
                shutdown_for_wait.store(true, Ordering::SeqCst);
                false
            },
            move || -> Result<()> {
                attempts_for_closure.fetch_add(1, Ordering::SeqCst);
                panic!("kaboom");
            },
        );

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let health = runtime.health();
        assert_eq!(health.restart_count, 1);
        assert!(health.last_error.unwrap().contains("kaboom"));
    }

    #[test]
    fn restarts_on_unexpected_ok_return() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let shutdown_for_wait = Arc::clone(&shutdown);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            move |_, _| {
                shutdown_for_wait.store(true, Ordering::SeqCst);
                false
            },
            || Ok(()),
        );

        assert!(result.is_ok());
        assert_eq!(runtime.health().restart_count, 1);
        assert!(
            runtime
                .health()
                .last_error
                .unwrap()
                .contains("before shutdown")
        );
    }

    #[test]
    fn never_restarts_once_shutdown_fires_during_the_attempt() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let shutdown_for_attempt = Arc::clone(&shutdown);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            |_, _| unreachable!("must not wait; shutdown already fired mid-attempt"),
            move || {
                shutdown_for_attempt.store(true, Ordering::SeqCst);
                Err(anyhow!("boom"))
            },
        );

        assert!(result.is_ok());
        assert_eq!(runtime.health().restart_count, 0);
        assert_eq!(runtime.health().state, HealthState::Stopped);
    }

    #[test]
    fn backoff_delay_doubles_and_caps_at_the_configured_maximum() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let delays = Arc::new(Mutex::new(Vec::new()));
        let delays_for_wait = Arc::clone(&delays);
        let shutdown_for_wait = Arc::clone(&shutdown);

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(4),
            move |_, delay| {
                let mut recorded = delays_for_wait.lock().unwrap();
                recorded.push(delay);
                if recorded.len() == 4 {
                    shutdown_for_wait.store(true, Ordering::SeqCst);
                }
                false
            },
            || Err(anyhow!("boom")),
        );

        assert!(result.is_ok());
        assert_eq!(
            *delays.lock().unwrap(),
            vec![
                Duration::from_millis(1),
                Duration::from_millis(2),
                Duration::from_millis(4),
                Duration::from_millis(4),
            ]
        );
        assert_eq!(runtime.health().restart_count, 4);
    }

    #[test]
    fn health_transitions_through_starting_backoff_and_stopped() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let runtime = test_runtime(&shutdown);
        let observed_during_wait = Arc::new(Mutex::new(None));
        let observed_for_wait = Arc::clone(&observed_during_wait);
        let runtime_for_wait = runtime.clone();
        let shutdown_for_wait = Arc::clone(&shutdown);
        let seen_starting = Arc::new(AtomicUsize::new(0));
        let seen_starting_for_attempt = Arc::clone(&seen_starting);
        let runtime_for_attempt = runtime.clone();

        let result = run_supervised(
            &runtime,
            &shutdown,
            Duration::from_millis(1),
            Duration::from_millis(10),
            move |_, _| {
                *observed_for_wait.lock().unwrap() = Some(runtime_for_wait.health().state.clone());
                shutdown_for_wait.store(true, Ordering::SeqCst);
                false
            },
            move || {
                if runtime_for_attempt.health().state == HealthState::Starting {
                    seen_starting_for_attempt.fetch_add(1, Ordering::SeqCst);
                }
                Err(anyhow!("boom"))
            },
        );

        assert!(result.is_ok());
        assert_eq!(seen_starting.load(Ordering::SeqCst), 1);
        assert_eq!(
            *observed_during_wait.lock().unwrap(),
            Some(HealthState::Backoff)
        );
        assert_eq!(runtime.health().state, HealthState::Stopped);
    }

    // ------------------------------------------------------------------
    // End-to-end wiring through `supervise::<A>` with a real `CoreAgent`.
    // Uses production backoff constants, so it is designed to never need a
    // restart (self-stopping on the first tick) to keep the test fast and
    // deterministic.
    // ------------------------------------------------------------------

    struct SelfStoppingAgent {
        runtime: AgentRuntime,
    }

    static SELF_STOPPING_FROM_SPAWN_CALLS: AtomicUsize = AtomicUsize::new(0);
    static SELF_STOPPING_ON_SHUTDOWN_CALLS: AtomicUsize = AtomicUsize::new(0);

    impl CoreAgent for SelfStoppingAgent {
        type SpawnContext = AgentSpawnContext;

        fn name() -> &'static str {
            "self-stopping-supervisor-test"
        }

        fn runtime(&self) -> &AgentRuntime {
            &self.runtime
        }

        fn periodic_tasks(&self) -> Vec<crate::core::periodic::PeriodicTaskSpec> {
            vec![crate::core::periodic::PeriodicTaskSpec {
                id: "tick",
                interval: Duration::ZERO,
                jitter: crate::core::periodic::JitterPolicy::BeforeEachCycle,
                jitter_max_ms: 0,
                autostart: true,
            }]
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            // Stop immediately so the supervisor never needs to restart
            // (and never waits out a real production backoff delay).
            self.runtime.shutdown().store(true, Ordering::SeqCst);
            Ok(())
        }

        fn from_spawn(ctx: Self::SpawnContext) -> Result<Self> {
            SELF_STOPPING_FROM_SPAWN_CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(Self {
                runtime: ctx.runtime,
            })
        }

        fn on_shutdown(&mut self) {
            SELF_STOPPING_ON_SHUTDOWN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
    }

    // Both scenarios share process-wide statics, so they run inside a single
    // test to avoid a cross-test race on the counters.
    #[test]
    fn supervise_wires_construction_and_shutdown_through_the_real_core_agent_trait() {
        use crate::core::activity::NoopActivityReporter;
        use crate::core::config::Config;

        fn workflow_with(shutdown: Arc<AtomicBool>) -> WorkflowContext {
            WorkflowContext {
                config: Arc::new(Config::from_toml_str("").unwrap()),
                base_dir: std::env::temp_dir().to_string_lossy().into_owned(),
                shutdown,
                activity: Arc::new(NoopActivityReporter),
            }
        }

        // Already shut down: never constructs the agent, no on_shutdown.
        let already_shutdown = Arc::new(AtomicBool::new(true));
        let before_spawns = SELF_STOPPING_FROM_SPAWN_CALLS.load(Ordering::SeqCst);
        assert!(supervise::<SelfStoppingAgent>(workflow_with(already_shutdown), 0).is_ok());
        assert_eq!(
            SELF_STOPPING_FROM_SPAWN_CALLS.load(Ordering::SeqCst),
            before_spawns,
            "must not construct once shutdown is already set"
        );

        // Not shut down: constructs exactly once, runs, and shuts down
        // exactly once, without ever needing a restart.
        let shutdown = Arc::new(AtomicBool::new(false));
        let before_spawns = SELF_STOPPING_FROM_SPAWN_CALLS.load(Ordering::SeqCst);
        let before_shutdowns = SELF_STOPPING_ON_SHUTDOWN_CALLS.load(Ordering::SeqCst);
        assert!(supervise::<SelfStoppingAgent>(workflow_with(Arc::clone(&shutdown)), 1).is_ok());
        assert_eq!(
            SELF_STOPPING_FROM_SPAWN_CALLS.load(Ordering::SeqCst) - before_spawns,
            1
        );
        assert_eq!(
            SELF_STOPPING_ON_SHUTDOWN_CALLS.load(Ordering::SeqCst) - before_shutdowns,
            1
        );
        assert!(shutdown.load(Ordering::SeqCst));
    }
}
