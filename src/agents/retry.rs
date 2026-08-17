use std::thread;
use std::time::Duration;

use anyhow::Result;
use tracing::{info, warn};

pub(crate) fn with_backoff_retries<T, F>(context: &str, operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    with_backoff_retries_using(
        context,
        operation,
        Duration::from_secs(1),
        Duration::from_secs(3 * 60 * 60),
        thread::sleep,
    )
}

fn with_backoff_retries_using<T, F, S>(
    context: &str,
    mut operation: F,
    initial_delay: Duration,
    max_delay: Duration,
    mut sleep: S,
) -> Result<T>
where
    F: FnMut() -> Result<T>,
    S: FnMut(Duration),
{
    let mut attempt = 0u32;
    let mut delay = initial_delay;
    loop {
        attempt += 1;
        match operation() {
            Ok(value) => {
                if attempt > 1 {
                    info!("{} succeeded after {} attempts", context, attempt);
                }
                return Ok(value);
            }
            Err(e) => {
                warn!(
                    "{} failed (attempt {}), retrying in {:?}: {}",
                    context, attempt, delay, e
                );
                sleep(delay);
                delay = (delay * 2).min(max_delay);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_retries_until_success() {
        let mut attempts = 0;
        let mut delays = Vec::new();
        let result = with_backoff_retries_using(
            "test operation",
            || {
                attempts += 1;
                if attempts < 3 {
                    anyhow::bail!("temporary failure");
                }
                Ok("done")
            },
            Duration::from_secs(1),
            Duration::from_secs(60),
            |delay| delays.push(delay),
        );

        assert_eq!(result.unwrap(), "done");
        assert_eq!(attempts, 3);
        assert_eq!(delays, vec![Duration::from_secs(1), Duration::from_secs(2)]);
    }

    #[test]
    fn backoff_caps_delay_while_retrying_until_success() {
        let mut attempts = 0;
        let mut delays = Vec::new();
        let result = with_backoff_retries_using(
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
            |delay| delays.push(delay),
        );

        assert_eq!(result.unwrap(), "done");
        assert_eq!(attempts, 5);
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
}
