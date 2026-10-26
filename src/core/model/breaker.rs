//! Process-global, per-model-endpoint failure breaker.
//!
//! Every model run talks to exactly one model endpoint (the ACP child's
//! endpoint binding). When that access keeps failing, retrying with
//! exponential backoff is worth a while — but an endpoint that stays down
//! must not keep the agent fleet churning: every failed run drops the claims
//! and labels it just added, respawns children, and burns dial timeouts,
//! only to fail again on the next cycle.
//!
//! The breaker counts **consecutive failures globally per endpoint**: all
//! agents in this process share one counter per endpoint binding, so a dead
//! endpoint is given up on after [`MAX_ENDPOINT_RETRIES`] retries no matter
//! which agent burned them. Once given up, the endpoint stays given up for
//! the lifetime of the process — every later model run against it fails
//! immediately, without spawning a child or dialing the endpoint. Restarting
//! the process (after the endpoint is fixed) resets everything.

use std::collections::HashMap;
use std::error;
use std::fmt;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use tracing::error;

/// How many consecutive failed *retries* an endpoint tolerates before it is
/// given up on: the initial attempt plus up to this many retries with
/// exponential backoff. The counter is global per endpoint — failures from
/// every agent in the process add up, and any success resets it.
pub(crate) const MAX_ENDPOINT_RETRIES: u32 = 10;

/// Backoff before the first retry; doubles per consecutive failure.
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
/// Backoff cap, so the tenth retry does not wait hours.
const MAX_BACKOFF: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Default)]
struct EndpointState {
    consecutive_failures: u32,
    gave_up: bool,
    last_error: Option<String>,
}

static ENDPOINTS: LazyLock<Mutex<HashMap<String, EndpointState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The error every model run against a given-up endpoint returns — including
/// the failure that tripped the breaker, so operators see why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelEndpointGaveUp {
    /// Human-readable endpoint label (the base URL when known).
    pub endpoint: String,
    /// Consecutive failures recorded when the endpoint was given up.
    pub failures: u32,
    /// The error that landed the final blow.
    pub last_error: String,
}

impl fmt::Display for ModelEndpointGaveUp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model endpoint {} gave up after {} consecutive failures \
             (last error: {}); failing fast until restart",
            self.endpoint, self.failures, self.last_error
        )
    }
}

impl error::Error for ModelEndpointGaveUp {}

/// Record one failed model run against the endpoint. Returns `true` when the
/// endpoint has now been given up on — because this failure crossed the
/// threshold, or because it had already been given up earlier.
pub(crate) fn record_failure(endpoint_key: &str, endpoint_label: &str, error: &str) -> bool {
    let mut endpoints = ENDPOINTS.lock().unwrap();
    let state = endpoints.entry(endpoint_key.to_string()).or_default();
    state.consecutive_failures += 1;
    state.last_error = Some(error.to_string());
    if state.gave_up {
        return true;
    }
    if state.consecutive_failures > MAX_ENDPOINT_RETRIES {
        state.gave_up = true;
        error!(
            "model endpoint {} is chronically failing: {} consecutive failures \
             (last error: {}); giving up — no further model runs will be \
             attempted against it until restart",
            endpoint_label, state.consecutive_failures, error
        );
        return true;
    }
    false
}

/// Record a successful model run: the endpoint is reachable again, so the
/// consecutive-failure counter resets. Never revives a given-up endpoint.
pub(crate) fn record_success(endpoint_key: &str) {
    let mut endpoints = ENDPOINTS.lock().unwrap();
    if let Some(state) = endpoints.get_mut(endpoint_key)
        && !state.gave_up
    {
        state.consecutive_failures = 0;
    }
}

/// The typed fail-fast error when the endpoint was given up on, `None` while
/// it is still trusted.
pub(crate) fn gave_up(endpoint_key: &str, endpoint_label: &str) -> Option<ModelEndpointGaveUp> {
    let endpoints = ENDPOINTS.lock().unwrap();
    let state = endpoints.get(endpoint_key)?;
    state.gave_up.then(|| ModelEndpointGaveUp {
        endpoint: endpoint_label.to_string(),
        failures: state.consecutive_failures,
        last_error: state.last_error.clone().unwrap_or_default(),
    })
}

/// Backoff before retry number `retry` (1-indexed): exponential from
/// [`INITIAL_BACKOFF`], capped at [`MAX_BACKOFF`].
pub(crate) fn backoff_delay(retry: u32) -> Duration {
    let exponent = retry.saturating_sub(1).min(20);
    INITIAL_BACKOFF
        .checked_mul(1u32 << exponent)
        .unwrap_or(MAX_BACKOFF)
        .min(MAX_BACKOFF)
}

/// The breaker key and human-readable label for a model endpoint binding.
///
/// The key is the ACP child's full spawn identity — command plus resolved
/// env — because the child's env IS its endpoint binding: two agents share a
/// failure budget iff they would spawn identical children. The label is the
/// first URL found in that env (the endpoint base URL), falling back to the
/// model URI and then the command, so logs name the endpoint readably
/// without ever printing api keys.
pub(crate) fn endpoint_identity(
    command: &[String],
    env: &HashMap<String, String>,
    model_uri: Option<&str>,
) -> (String, String) {
    let mut pairs: Vec<(&String, &String)> = env.iter().collect();
    pairs.sort();
    let key = format!("{command:?}|{pairs:?}");
    let label = pairs
        .iter()
        .map(|(_, value)| value.as_str())
        .find(|value| value.starts_with("http://") || value.starts_with("https://"))
        .map(str::to_string)
        .or_else(|| model_uri.map(str::to_string))
        .unwrap_or_else(|| {
            command
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown endpoint".to_string())
        });
    (key, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry is process-global and give-up is permanent, so every
    /// test works on a key derived from its (unique) test name.
    fn key(name: &str) -> String {
        format!("breaker-test:{name}")
    }

    #[test]
    fn endpoint_gives_up_after_max_retries_and_stays_given_up() {
        let key = &key("gives-up");

        for _ in 0..MAX_ENDPOINT_RETRIES {
            assert!(!record_failure(key, "http://ep.example", "boom"));
            assert!(gave_up(key, "http://ep.example").is_none());
        }
        // Every retry the endpoint granted has failed: the next failure
        // gives up on it.
        assert!(record_failure(key, "http://ep.example", "final blow"));

        let error = gave_up(key, "http://ep.example").unwrap();
        assert_eq!(error.endpoint, "http://ep.example");
        assert_eq!(error.failures, MAX_ENDPOINT_RETRIES + 1);
        assert_eq!(error.last_error, "final blow");

        // Giving up is permanent: success does not revive the endpoint, and
        // further failures keep reporting it as given up.
        record_success(key);
        assert!(gave_up(key, "http://ep.example").is_some());
        assert!(record_failure(key, "http://ep.example", "again"));
    }

    #[test]
    fn success_resets_the_consecutive_counter() {
        let key = &key("success-resets");

        for _ in 0..(MAX_ENDPOINT_RETRIES / 2) {
            assert!(!record_failure(key, "http://ep.example", "boom"));
        }
        record_success(key);
        for _ in 0..MAX_ENDPOINT_RETRIES {
            assert!(!record_failure(key, "http://ep.example", "boom"));
            record_success(key);
        }
        assert!(gave_up(key, "http://ep.example").is_none());

        // A full run of retries without a success in between still trips.
        for _ in 0..MAX_ENDPOINT_RETRIES {
            assert!(!record_failure(key, "http://ep.example", "boom"));
        }
        assert!(record_failure(key, "http://ep.example", "boom"));
        assert!(gave_up(key, "http://ep.example").is_some());
    }

    #[test]
    fn endpoints_count_failures_independently() {
        let first = &key("independent-first");
        let second = &key("independent-second");

        for _ in 0..MAX_ENDPOINT_RETRIES {
            assert!(!record_failure(second, "http://second.example", "boom"));
        }
        assert!(record_failure(second, "http://second.example", "boom"));
        assert!(gave_up(second, "http://second.example").is_some());
        assert!(gave_up(first, "http://first.example").is_none());
        assert!(!record_failure(first, "http://first.example", "boom"));
    }

    #[test]
    fn backoff_doubles_per_retry_and_caps() {
        let delays: Vec<u64> = (1..=MAX_ENDPOINT_RETRIES)
            .map(|retry| backoff_delay(retry).as_secs())
            .collect();
        assert_eq!(
            delays,
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_BACKOFF.as_secs()]
        );
        assert_eq!(backoff_delay(0), INITIAL_BACKOFF);
        assert_eq!(backoff_delay(u32::MAX), MAX_BACKOFF);
    }

    #[test]
    fn identity_keys_on_the_spawn_and_labels_the_endpoint_url() {
        let command = vec!["agent".to_string(), "acp".to_string()];
        let mut env = HashMap::new();
        env.insert("API_KEY".to_string(), "secret".to_string());
        env.insert(
            "BASE_URL".to_string(),
            "http://endpoint1.example/v1".to_string(),
        );

        let (key, label) = endpoint_identity(&command, &env, Some("acp://vendor/model1"));
        assert_eq!(label, "http://endpoint1.example/v1");

        // The key covers everything that determines the child: the same
        // spawn under a different api key is a different endpoint binding.
        let mut other_env = env.clone();
        other_env.insert("API_KEY".to_string(), "other".to_string());
        let (other_key, _) = endpoint_identity(&command, &other_env, Some("acp://vendor/model1"));
        assert_ne!(key, other_key);

        // The key is order-independent; the label ignores api keys entirely.
        let (rekeyed, relabel) = endpoint_identity(&command, &env, Some("acp://vendor/model1"));
        assert_eq!(key, rekeyed);
        assert_eq!(label, relabel);

        // Without a URL env, the model URI names the endpoint.
        let (_, uri_label) = endpoint_identity(&command, &HashMap::new(), Some("acp://vendor/m"));
        assert_eq!(uri_label, "acp://vendor/m");

        // With nothing at all, the command does.
        let (_, command_label) = endpoint_identity(&command, &HashMap::new(), None);
        assert_eq!(command_label, "agent");
    }
}
