use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use rand::RngExt;

use crate::core::agent::CoreAgent;
use crate::util::sleep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterPolicy {
    BeforeEachCycle,
}

fn apply_jitter(jitter_max_ms: u64) -> Duration {
    if jitter_max_ms == 0 {
        return Duration::ZERO;
    }
    let ms = rand::rng().random_range(0..jitter_max_ms);
    Duration::from_millis(ms)
}

#[derive(Debug, Clone)]
pub struct PeriodicTaskSpec {
    pub id: &'static str,
    pub interval: Duration,
    pub jitter: JitterPolicy,
    /// Upper bound (exclusive) for random jitter milliseconds.
    pub jitter_max_ms: u64,
    pub autostart: bool,
}

impl PeriodicTaskSpec {
    /// Create an automatically started polling task with per-cycle jitter.
    pub fn polling(id: &'static str, interval: Duration) -> Self {
        Self {
            id,
            interval,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 5000,
            autostart: true,
        }
    }
}

pub fn run_periodic_scheduler<A: CoreAgent>(
    agent: &mut A,
    tasks: &[PeriodicTaskSpec],
    shutdown: &AtomicBool,
) -> Result<()> {
    if tasks.is_empty() {
        loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            if sleep(shutdown, Duration::from_millis(200)) {
                break;
            }
        }
        return Ok(());
    }

    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }

        for task in tasks {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            if task.jitter == JitterPolicy::BeforeEachCycle {
                let pre = apply_jitter(task.jitter_max_ms);
                if !pre.is_zero() && sleep(shutdown, pre) {
                    break;
                }
            }

            agent.runtime().mark_busy(task.id);
            match agent.run_periodic_task(task.id) {
                Ok(()) => agent.runtime().record_cycle_success(),
                // A failure caused by shutdown itself isn't a real health
                // signal — the scheduler is about to exit anyway — so only
                // degrade and log when shutdown wasn't already requested.
                Err(e) if !shutdown.load(Ordering::SeqCst) => {
                    agent.runtime().record_cycle_failure(e.to_string());
                    tracing::error!(
                        "{} periodic task {} failed: {}",
                        agent.agent_id(),
                        task.id,
                        e
                    );
                }
                Err(_) => {}
            }

            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            if sleep(shutdown, task.interval) {
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    use anyhow::{Result, anyhow};

    use super::*;
    use crate::core::runtime::{AgentRuntime, HealthSnapshot, HealthState};

    struct SchedulerAgent {
        shutdown: Arc<AtomicBool>,
        runtime: AgentRuntime,
        calls: Arc<AtomicUsize>,
        fail_first: bool,
        /// Health snapshot observed at the start of each `run_periodic_task`
        /// call, in call order.
        observed_health: Arc<Mutex<Vec<HealthSnapshot>>>,
    }

    impl SchedulerAgent {
        fn new(shutdown: Arc<AtomicBool>, calls: Arc<AtomicUsize>, fail_first: bool) -> Self {
            Self {
                runtime: AgentRuntime::new("scheduler-test-0", Arc::clone(&shutdown)),
                shutdown,
                calls,
                fail_first,
                observed_health: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl CoreAgent for SchedulerAgent {
        type SpawnContext = ();

        fn name() -> &'static str {
            "scheduler-test"
        }

        fn runtime(&self) -> &AgentRuntime {
            &self.runtime
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            self.observed_health
                .lock()
                .unwrap()
                .push(self.runtime.health());
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call > 0 || !self.fail_first {
                self.shutdown.store(true, Ordering::SeqCst);
            }
            if call == 0 && self.fail_first {
                Err(anyhow!("expected cycle failure"))
            } else {
                Ok(())
            }
        }

        fn from_spawn(_ctx: Self::SpawnContext) -> Result<Self> {
            unreachable!("scheduler tests construct the agent directly")
        }

        fn on_shutdown(&mut self) {}
    }

    fn task(jitter_max_ms: u64) -> PeriodicTaskSpec {
        PeriodicTaskSpec {
            jitter_max_ms,
            interval: Duration::ZERO,
            ..PeriodicTaskSpec::polling("poll", Duration::ZERO)
        }
    }

    #[test]
    fn polling_task_uses_standard_scheduler_defaults() {
        let task = PeriodicTaskSpec::polling("poll", Duration::from_secs(3));

        assert_eq!(task.id, "poll");
        assert_eq!(task.interval, Duration::from_secs(3));
        assert_eq!(task.jitter, JitterPolicy::BeforeEachCycle);
        assert_eq!(task.jitter_max_ms, 5000);
        assert!(task.autostart);
    }

    #[test]
    fn scheduler_does_not_run_tasks_after_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), false);

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn scheduler_continues_after_task_error_until_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), true);

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn zero_jitter_runs_without_random_range_panic() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), false);

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn task_runs_with_health_marked_busy() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), false);
        let observed = Arc::clone(&agent.observed_health);

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].state, HealthState::Busy("poll".to_string()));
    }

    #[test]
    fn successful_cycle_resets_consecutive_failures_and_returns_to_idle() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), false);
        let runtime = agent.runtime.clone();

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        let health = runtime.health();
        assert_eq!(health.state, HealthState::Idle);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn failed_cycle_is_recorded_as_degraded_before_a_later_success_resets_it() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent::new(Arc::clone(&shutdown), Arc::clone(&calls), true);
        let observed = Arc::clone(&agent.observed_health);
        let runtime = agent.runtime.clone();

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        // Snapshot taken at the start of the second call reflects the first
        // call's recorded failure.
        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[1].consecutive_failures, 1);
        assert_eq!(
            observed[1].last_error.as_deref(),
            Some("expected cycle failure")
        );

        // The scheduler kept running the agent after the failure (this is
        // not a restart) and the later success reset the failure counter.
        let health = runtime.health();
        assert_eq!(health.state, HealthState::Idle);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.last_error.as_deref(), Some("expected cycle failure"));
    }
}
