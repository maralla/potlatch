use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

#[cfg(target_os = "linux")]
use std::io::Error;
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::process;

/// Register `cmd` to be killed (SIGKILL) when the spawning process dies
/// (Linux `PR_SET_PDEATHSIG`), so a child that outlives its owner — a shell
/// command still running while its harness is killed — cannot accumulate as
/// a stray running process. Also guards the registration race: a child whose
/// parent died between fork and registration exits immediately.
///
/// Linux semantics note: the signal fires when the spawning **thread** dies,
/// so this is only for children whose intended lifetime is bounded by the
/// thread that spawns them (foreground commands, long-lived ACP subprocesses
/// spawned from dedicated supervisor threads) — never for state meant to
/// outlive that thread.
///
/// The backstop exists on Linux only: macOS and Windows have no equivalent
/// primitive available to the child, and there the graceful shutdown paths
/// (stdin EOF cleanup, graceful dispose) remain the only reaping mechanism.
pub fn dies_with_parent(cmd: &mut Command) {
    parent_death_signal(cmd);
}

#[cfg(target_os = "linux")]
fn parent_death_signal(cmd: &mut Command) {
    let parent_pid = process::id();
    // Safety: `pre_exec` runs the closure in the forked child before exec.
    // Both calls are async-signal-safe: `prctl` is a direct syscall, and
    // `_exit` is the sanctioned immediate exit for a pre-exec child.
    unsafe {
        cmd.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(Error::last_os_error());
            }
            if libc::getppid() as u32 != parent_pid {
                // The parent died between fork and registration: the child
                // was re-parented, so the signal would never arrive.
                libc::_exit(1);
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn parent_death_signal(_cmd: &mut Command) {}

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
    use std::sync::mpsc;
    use std::time::Instant;

    #[test]
    fn sleep_respects_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let s = shutdown.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            s.store(true, Ordering::SeqCst);
        });
        assert!(sleep(&shutdown, Duration::from_secs(10)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dies_with_parent_kills_the_child_when_the_spawning_thread_exits() {
        use std::os::unix::process::ExitStatusExt;

        // PR_SET_PDEATHSIG fires when the spawning thread dies: register and
        // spawn on a short-lived thread, then observe the SIGKILL from here.
        let (sender, receiver) = mpsc::channel();
        let spawner = thread::spawn(move || {
            let mut cmd = Command::new("sleep");
            cmd.arg("30");
            dies_with_parent(&mut cmd);
            let child = cmd.spawn().expect("spawn sleep");
            let _ = sender.send(child.id());
            child
        });
        let mut child = spawner.join().expect("spawner thread");
        let _ = receiver.try_recv();

        let deadline = Instant::now() + Duration::from_secs(5);
        let status = loop {
            match child.try_wait().expect("poll child") {
                Some(status) => break status,
                None => {
                    assert!(
                        Instant::now() < deadline,
                        "child must be killed when the spawning thread exits"
                    );
                    thread::sleep(Duration::from_millis(20));
                }
            }
        };
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }
}
