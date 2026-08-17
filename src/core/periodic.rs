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

            if let Err(e) = agent.run_periodic_task(task.id)
                && !shutdown.load(Ordering::SeqCst)
            {
                tracing::error!(
                    "{} periodic task {} failed: {}",
                    agent.agent_id(),
                    task.id,
                    e
                );
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
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use anyhow::{Result, anyhow};

    use super::*;
    struct SchedulerAgent {
        shutdown: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
        fail_first: bool,
    }

    impl CoreAgent for SchedulerAgent {
        type SpawnContext = ();

        fn name() -> &'static str {
            "scheduler-test"
        }

        fn agent_id(&self) -> &str {
            "scheduler-test-0"
        }

        fn shutdown(&self) -> &Arc<AtomicBool> {
            &self.shutdown
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
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
        let mut agent = SchedulerAgent {
            shutdown: Arc::clone(&shutdown),
            calls: Arc::clone(&calls),
            fail_first: false,
        };

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn scheduler_continues_after_task_error_until_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent {
            shutdown: Arc::clone(&shutdown),
            calls: Arc::clone(&calls),
            fail_first: true,
        };

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn zero_jitter_runs_without_random_range_panic() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut agent = SchedulerAgent {
            shutdown: Arc::clone(&shutdown),
            calls: Arc::clone(&calls),
            fail_first: false,
        };

        run_periodic_scheduler(&mut agent, &[task(0)], &shutdown).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
