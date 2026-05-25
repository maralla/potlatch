use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Sleep up to `duration`, checking `shutdown` every 200ms.
/// Returns `true` if shutdown was signaled before the wait finished.
pub fn sleep(shutdown: &AtomicBool, duration: Duration) -> bool {
    let interval = Duration::from_millis(200);
    let mut remaining = duration;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return true;
        }
        if remaining.is_zero() {
            return false;
        }
        let sleep_time = remaining.min(interval);
        thread::sleep(sleep_time);
        remaining = remaining.saturating_sub(sleep_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn sleep_respects_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let s = shutdown.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            s.store(true, Ordering::SeqCst);
        });
        assert!(sleep(&shutdown, Duration::from_secs(10)));
    }
}
