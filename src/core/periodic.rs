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

            if let Err(e) = agent.run_periodic_task(task.id) {
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
