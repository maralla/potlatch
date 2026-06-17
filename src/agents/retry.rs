use std::thread;
use std::time::Duration;

use anyhow::Result;
use tracing::{debug, info};

/// Returns `true` when an error string suggests a transient network/API problem worth retrying.
pub(crate) fn transient_error_should_retry(err_msg: &str) -> bool {
    let m = err_msg.to_lowercase();
    // Permanent client / validation errors — repeating the request is unlikely to help.
    if m.contains("http 401")
        || m.contains("http 403")
        || m.contains("http 404")
        || m.contains("http 400")
        || m.contains("http 405")
        || m.contains("http 422")
        || m.contains("unauthorized")
        || m.contains("authentication failed")
        || m.contains("invalid ref")
        || m.contains("unknown revision")
        || m.contains("nothing to commit")
        || m.contains("already exists")
        || m.contains("cannot lock ref")
        || m.contains("repository not found")
    {
        return false;
    }
    true
}

pub(crate) fn with_transient_retries<T, F>(context: &str, mut operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let mut attempt = 0u32;
    let mut delay = Duration::from_secs(1);
    const MAX_DELAY: Duration = Duration::from_secs(60);
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
                let msg = e.to_string();
                if !transient_error_should_retry(&msg) {
                    return Err(e);
                }
                debug!(
                    "Transient failure {} (attempt {}), retrying in {:?}: {}",
                    context, attempt, delay, msg
                );
                thread::sleep(delay);
                delay = (delay * 2).min(MAX_DELAY);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_error_retry_heuristic_matches_glab_500() {
        assert!(transient_error_should_retry(
            "Failed to add label to MR: glab: 500 Internal Server Error (HTTP 500)"
        ));
    }

    #[test]
    fn transient_error_retry_heuristic_matches_tls_handshake_timeout() {
        assert!(transient_error_should_retry(
            "glab api issue discussions failed: ERROR Get \"https://gitlab.example/api\": net/http: TLS handshake timeout."
        ));
    }

    #[test]
    fn transient_error_retry_heuristic_matches_git_connection_closed() {
        assert!(transient_error_should_retry(
            "Git fetch failed: Connection closed by 192.0.2.1 port 22 fatal: Could not read from remote repository. Please make sure you have the correct access rights and the repository exists."
        ));
    }

    #[test]
    fn transient_error_retry_skips_permanent_http_codes() {
        assert!(!transient_error_should_retry(
            "Failed to add label to MR: glab: 404 Not Found (HTTP 404)"
        ));
        assert!(!transient_error_should_retry(
            "Failed to add label to MR: HTTP 403 Forbidden"
        ));
    }

    #[test]
    fn transient_error_retry_skips_git_auth_failures() {
        assert!(!transient_error_should_retry(
            "Git fetch failed: fatal: Authentication failed for 'https://gitlab.example/group/project.git/'"
        ));
    }
}
