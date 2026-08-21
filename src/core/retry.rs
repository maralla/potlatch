use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use tracing::{info, warn};

/// Shared with [`crate::core::supervisor`], which restarts stopped agents
/// with the same shutdown-aware exponential backoff.
pub(crate) const INITIAL_DELAY: Duration = Duration::from_secs(1);
pub(crate) const MAX_DELAY: Duration = Duration::from_secs(3 * 60 * 60);
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Error returned when shutdown cancels a retry loop or backoff wait.
#[derive(Debug)]
pub struct RetryCancelled;

impl fmt::Display for RetryCancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retry cancelled by shutdown")
    }
}

impl std::error::Error for RetryCancelled {}

/// Non-retryable error for a permanent failure (e.g. HTTP 404 for a deleted
/// GitLab resource). [`with_backoff_retries`] returns this immediately
/// instead of looping with backoff. Wraps a human-readable detail message.
#[derive(Debug)]
pub struct NonRetryable(pub String);

impl fmt::Display for NonRetryable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NonRetryable {}

/// Retry an operation indefinitely with capped exponential backoff.
pub fn with_backoff_retries<T, F>(shutdown: &AtomicBool, context: &str, operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    with_backoff_retries_using(
        shutdown,
        context,
        operation,
        INITIAL_DELAY,
        MAX_DELAY,
        shutdown_aware_wait,
    )
}

fn shutdown_aware_wait(shutdown: &AtomicBool, duration: Duration) -> Result<()> {
    let started = std::time::Instant::now();
    while !shutdown.load(Ordering::SeqCst) {
        let remaining = duration.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Ok(());
        }
        std::thread::sleep(remaining.min(SHUTDOWN_POLL_INTERVAL));
    }
    Err(RetryCancelled.into())
}

fn with_backoff_retries_using<T, F, W>(
    shutdown: &AtomicBool,
    context: &str,
    mut operation: F,
    initial_delay: Duration,
    max_delay: Duration,
    mut wait: W,
) -> Result<T>
where
    F: FnMut() -> Result<T>,
    W: FnMut(&AtomicBool, Duration) -> Result<()>,
{
    let mut attempt = 0u32;
    let mut delay = initial_delay;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return Err(RetryCancelled.into());
        }
        attempt += 1;
        match operation() {
            Ok(value) => {
                if attempt > 1 {
                    info!("{} succeeded after {} attempts", context, attempt);
                }
                return Ok(value);
            }
            Err(error) if error.downcast_ref::<RetryCancelled>().is_some() => return Err(error),
            Err(error) if error.downcast_ref::<NonRetryable>().is_some() => return Err(error),
            Err(error) => {
                if shutdown.load(Ordering::SeqCst) {
                    return Err(RetryCancelled.into());
                }
                warn!(
                    "{} failed (attempt {}), retrying in {:?}: {}",
                    context, attempt, delay, error
                );
                wait(shutdown, delay)?;
                delay = (delay * 2).min(max_delay);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn production_backoff_defaults_are_stable() {
        assert_eq!(INITIAL_DELAY, Duration::from_secs(1));
        assert_eq!(MAX_DELAY, Duration::from_secs(3 * 60 * 60));
    }

    #[test]
    fn backoff_retries_until_success_and_caps_delay() {
        let shutdown = AtomicBool::new(false);
        let mut attempts = 0;
        let mut delays = Vec::new();
        let result = with_backoff_retries_using(
            &shutdown,
            "test operation",
            || {
                attempts += 1;
                if attempts < 5 {
                    anyhow::bail!("temporary failure");
                }
                Ok("done")
            },
            Duration::from_secs(1),
            Duration::from_secs(2),
            |_, delay| {
                delays.push(delay);
                Ok(())
            },
        );

        assert_eq!(result.unwrap(), "done");
        assert_eq!(
            delays,
            vec![
                Duration::from_secs(1),
                Duration::from_secs(2),
                Duration::from_secs(2),
                Duration::from_secs(2)
            ]
        );
    }

    #[test]
    fn cancellation_interrupts_backoff_wait() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&shutdown);
        let started = std::time::Instant::now();
        let handle = std::thread::spawn(move || {
            with_backoff_retries_using(
                &signal,
                "test operation",
                || -> Result<()> { anyhow::bail!("temporary failure") },
                Duration::from_secs(30),
                Duration::from_secs(30),
                shutdown_aware_wait,
            )
        });

        std::thread::sleep(Duration::from_millis(30));
        shutdown.store(true, Ordering::SeqCst);
        let error = handle.join().unwrap().unwrap_err();

        assert!(error.downcast_ref::<RetryCancelled>().is_some());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn cancellation_before_first_attempt_skips_operation() {
        let shutdown = AtomicBool::new(true);
        let mut called = false;

        let error = with_backoff_retries(&shutdown, "test operation", || {
            called = true;
            Ok(())
        })
        .unwrap_err();

        assert!(!called);
        assert!(error.downcast_ref::<RetryCancelled>().is_some());
    }

    #[test]
    fn non_retryable_error_returns_immediately_without_retrying() {
        let shutdown = AtomicBool::new(false);
        let mut attempts = 0;

        let result: anyhow::Result<()> = with_backoff_retries(&shutdown, "test operation", || {
            attempts += 1;
            Err(NonRetryable("permanent failure".to_string()).into())
        });
        let error = result.unwrap_err();

        assert_eq!(attempts, 1, "NonRetryable should not be retried");
        assert!(error.downcast_ref::<NonRetryable>().is_some());
    }
}
