//! Agent-layer claim leases: atomically claim a GitLab issue or MR using a
//! label-based claim-and-verify protocol, and hand back an explicit lease
//! object representing held ownership.
//!
//! ## Protocol
//! 1. Add a unique claim label (`claimed:<agent_id>`) to the resource.
//! 2. Wait for the settle time, to let concurrent claims from other
//!    instances propagate through the GitLab API.
//! 3. Re-fetch the resource's labels and check for competing claims.
//! 4. Repeat the wait+re-fetch once more, to guard against propagation
//!    delay on the first read.
//! 5. If only our claim label is present, we won.
//! 6. If more than one claim label is present, a deterministic tiebreak
//!    (lexicographically smallest label wins) decides the winner.
//!
//! A shutdown signal observed during either settle wait interrupts the
//! attempt. Losing, being interrupted, or failing to read the resource back
//! all clean up the same way: the claim label we just added is removed
//! before returning.
//!
//! ## Leases
//! A won (or recovered) claim is returned as a [`ClaimLease`]. The only way
//! to relinquish one is to call [`ClaimLease::release`] (remove the label
//! now) or [`ClaimLease::preserve`] (leave the label in place; some later
//! process — this same instance on a future cycle, or a restarted instance
//! via [`ClaimLease::recover`] — is responsible for eventually releasing
//! it). Dropping a lease without calling either does **no** network or
//! filesystem work — `Drop` only ever logs that ownership was left
//! unresolved, so a bug like a missing release/preserve is visible without
//! ever risking a duplicate or unwanted side effect from `Drop` itself.
//!
//! ## Restart recovery
//! An instance that restarts (crash, redeploy, supervisor restart) never
//! trusts a local record of what it used to hold: [`ClaimLease::recover`]
//! only ever constructs a lease from a label list the caller just fetched
//! live from GitLab, so recovered ownership always reflects the resource's
//! current, real state rather than possibly-stale local bookkeeping.
//!
//! ## Ports
//! [`ClaimPort`] is deliberately narrow — add/remove/read one resource's
//! labels — rather than a stand-in for the entire GitLab API, so tests can
//! exercise the claim protocol against an in-memory fake instead of a real
//! (or fully mocked) [`GitLabClient`].

use std::fmt;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::agents::gitlab::GitLabClient;
use crate::util::sleep;

/// Settle time in seconds. After adding a claim label, we wait this long
/// before verifying, to allow concurrent claims from other instances to
/// propagate through the GitLab API.
const CLAIM_SETTLE_SECS: u64 = 5;

/// Number of settle-then-verify rounds `acquire` performs before it is
/// satisfied no contender showed up.
const CLAIM_SETTLE_ROUNDS: u32 = 2;

const CLAIM_LABEL_PREFIX: &str = "claimed:";

pub(crate) fn claim_label(agent_id: &str) -> String {
    format!("{CLAIM_LABEL_PREFIX}{agent_id}")
}

/// A claimable GitLab resource, identified by its typed kind and IID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ClaimResource {
    Issue(u64),
    MergeRequest(u64),
}

impl ClaimResource {
    pub(crate) fn iid(self) -> u64 {
        match self {
            ClaimResource::Issue(iid) | ClaimResource::MergeRequest(iid) => iid,
        }
    }
}

impl fmt::Display for ClaimResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaimResource::Issue(iid) => write!(f, "issue #{iid}"),
            ClaimResource::MergeRequest(iid) => write!(f, "MR !{iid}"),
        }
    }
}

/// The narrow surface `claim` needs from a GitLab-like backend: add/remove
/// one label and read a resource's current labels. Deliberately not a
/// stand-in for the whole GitLab API — see the module docs.
pub(crate) trait ClaimPort {
    fn add_label(&self, resource: ClaimResource, label: &str) -> Result<()>;
    fn remove_label(&self, resource: ClaimResource, label: &str) -> Result<()>;
    fn labels(&self, resource: ClaimResource) -> Result<Vec<String>>;
}

impl ClaimPort for GitLabClient {
    fn add_label(&self, resource: ClaimResource, label: &str) -> Result<()> {
        match resource {
            ClaimResource::Issue(iid) => self.add_issue_label(iid, label),
            ClaimResource::MergeRequest(iid) => self.add_mr_label_with_retries(iid, label),
        }
    }

    fn remove_label(&self, resource: ClaimResource, label: &str) -> Result<()> {
        match resource {
            ClaimResource::Issue(iid) => self.remove_issue_label(iid, label),
            ClaimResource::MergeRequest(iid) => self.remove_mr_label(iid, label),
        }
    }

    fn labels(&self, resource: ClaimResource) -> Result<Vec<String>> {
        match resource {
            ClaimResource::Issue(iid) => Ok(self.get_issue(iid)?.labels),
            ClaimResource::MergeRequest(iid) => {
                Ok(self.get_merge_request(iid)?.labels.unwrap_or_default())
            }
        }
    }
}

/// Explicit ownership of a won or recovered claim.
///
/// Must be resolved with [`ClaimLease::release`] or [`ClaimLease::preserve`]
/// — dropping one without resolving it does no network/filesystem work, it
/// only reports (via `tracing`) that ownership was left dangling.
pub(crate) struct ClaimLease {
    resource: ClaimResource,
    label: String,
    agent_id: String,
    settled: bool,
}

impl ClaimLease {
    fn new(resource: ClaimResource, label: String, agent_id: &str) -> Self {
        Self {
            resource,
            label,
            agent_id: agent_id.to_string(),
            settled: false,
        }
    }

    /// Reconstruct a lease for a claim this agent already holds — but only
    /// from a label list the caller just fetched live from GitLab, never
    /// from local/persisted bookkeeping alone. Returns `None` when this
    /// agent's claim label is not present in `live_labels`.
    pub(crate) fn recover(
        resource: ClaimResource,
        agent_id: &str,
        live_labels: &[String],
    ) -> Option<Self> {
        let label = claim_label(agent_id);
        live_labels
            .iter()
            .any(|l| l == &label)
            .then(|| Self::new(resource, label, agent_id))
    }

    pub(crate) fn resource(&self) -> ClaimResource {
        self.resource
    }

    /// Explicitly relinquish the claim: removes the label from GitLab.
    /// Consumes the lease either way, so `Drop` never re-reports it.
    pub(crate) fn release(self, port: &dyn ClaimPort) -> Result<()> {
        let mut this = self;
        this.settled = true;
        debug!("{}: Releasing claim on {}", this.agent_id, this.resource);
        port.remove_label(this.resource, &this.label)
    }

    /// Attempt to release without consuming the lease, so a failed attempt
    /// (e.g. a transient GitLab API error) can be retried later against the
    /// same lease rather than losing track of held ownership. On success
    /// the lease is marked settled, exactly as [`ClaimLease::release`]
    /// would; on failure it is left unsettled, still owned by the caller.
    pub(crate) fn try_release(&mut self, port: &dyn ClaimPort) -> Result<()> {
        debug!("{}: Releasing claim on {}", self.agent_id, self.resource);
        port.remove_label(self.resource, &self.label)?;
        self.settled = true;
        Ok(())
    }

    /// Explicitly keep the claim label in place: some later process — this
    /// same instance on a future cycle, or a restarted instance via
    /// [`ClaimLease::recover`] — is responsible for eventually releasing
    /// it. Does no network/filesystem work.
    pub(crate) fn preserve(self) {
        let mut this = self;
        this.settled = true;
    }
}

impl Drop for ClaimLease {
    fn drop(&mut self) {
        if !self.settled {
            warn!(
                "{}: Claim lease on {} dropped without an explicit release or preserve; leaving the label as-is",
                self.agent_id, self.resource
            );
        }
    }
}

/// Result of a claim attempt.
pub(crate) enum ClaimAcquireOutcome {
    /// This agent won the claim; the [`ClaimLease`] must be released or
    /// preserved.
    Won(ClaimLease),
    /// Another agent's claim label won the deterministic tiebreak; our own
    /// claim label has already been removed.
    Lost,
    /// A shutdown signal fired while waiting for the claim to settle; our
    /// own claim label has already been removed.
    Interrupted,
}

/// Attempt to atomically claim `resource` using the claim-and-verify
/// protocol with double-check (see module docs).
pub(crate) fn acquire(
    port: &dyn ClaimPort,
    resource: ClaimResource,
    agent_id: &str,
    shutdown: &AtomicBool,
) -> Result<ClaimAcquireOutcome> {
    acquire_with(port, resource, agent_id, shutdown, sleep)
}

/// Core of [`acquire`], parameterized over the shutdown-aware settle wait so
/// it can be exercised deterministically in tests without a real sleep.
fn acquire_with(
    port: &dyn ClaimPort,
    resource: ClaimResource,
    agent_id: &str,
    shutdown: &AtomicBool,
    mut wait_settle: impl FnMut(&AtomicBool, Duration) -> bool,
) -> Result<ClaimAcquireOutcome> {
    let label = claim_label(agent_id);
    debug!("{agent_id}: Attempting to claim {resource}");

    port.add_label(resource, &label)?;

    let outcome = settle_and_check(port, resource, agent_id, &label, shutdown, &mut wait_settle);

    // Anything other than a win means we must not keep the label we just
    // added: contention losses clean up in `resolve_contention` too, but a
    // shutdown interruption or a failed settle-read never reach it, so the
    // cleanup lives here where every non-`Won` path funnels through.
    if !matches!(outcome, Ok(ClaimAcquireOutcome::Won(_))) {
        let _ = port.remove_label(resource, &label);
    }

    outcome
}

fn settle_and_check(
    port: &dyn ClaimPort,
    resource: ClaimResource,
    agent_id: &str,
    label: &str,
    shutdown: &AtomicBool,
    wait_settle: &mut impl FnMut(&AtomicBool, Duration) -> bool,
) -> Result<ClaimAcquireOutcome> {
    for _ in 0..CLAIM_SETTLE_ROUNDS {
        if wait_settle(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
            info!("{agent_id}: Shutdown while claiming {resource}, backing off");
            return Ok(ClaimAcquireOutcome::Interrupted);
        }

        let live_labels = port.labels(resource)?;
        let claim_labels: Vec<&String> = live_labels
            .iter()
            .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
            .collect();

        if claim_labels.len() > 1 {
            return Ok(resolve_contention(resource, agent_id, label, &claim_labels));
        }
    }

    info!("{agent_id}: Successfully claimed {resource}");
    Ok(ClaimAcquireOutcome::Won(ClaimLease::new(
        resource,
        label.to_string(),
        agent_id,
    )))
}

/// Deterministic tiebreak: the lexicographically smallest claim label wins.
/// Pure — used after the double-settle re-check finds more than one claim
/// label, so the deadlock between two concurrently-claiming instances
/// always resolves the same way regardless of which instance observes
/// contention first.
fn claim_tiebreak_wins(own_label: &str, claim_labels: &[&String]) -> bool {
    let winner = claim_labels
        .iter()
        .min()
        .expect("claim_labels is non-empty");
    winner.as_str() == own_label
}

fn resolve_contention(
    resource: ClaimResource,
    agent_id: &str,
    label: &str,
    claim_labels: &[&String],
) -> ClaimAcquireOutcome {
    if claim_tiebreak_wins(label, claim_labels) {
        info!(
            "{agent_id}: Won claim tiebreaker for {resource} against {} other(s)",
            claim_labels.len() - 1
        );
        return ClaimAcquireOutcome::Won(ClaimLease::new(resource, label.to_string(), agent_id));
    }

    warn!(
        "{agent_id}: Lost claim for {resource} to {}, backing off",
        claim_labels.iter().min().unwrap()
    );
    ClaimAcquireOutcome::Lost
}

/// Remove `agent_id`'s claim label from `resource` without a held
/// [`ClaimLease`] — for call sites that only ever tracked a resource's IID
/// across cycles (GitLab's label is the single source of truth for those),
/// never an in-memory lease value.
pub(crate) fn release(port: &dyn ClaimPort, resource: ClaimResource, agent_id: &str) -> Result<()> {
    let label = claim_label(agent_id);
    debug!("{agent_id}: Releasing claim on {resource}");
    port.remove_label(resource, &label)
}

/// Check if a resource is already claimed by any instance.
pub(crate) fn is_claimed(labels: &[String]) -> bool {
    labels.iter().any(|l| l.starts_with(CLAIM_LABEL_PREFIX))
}

/// Check if an MR is already claimed by any instance.
pub(crate) fn is_mr_claimed(labels: &Option<Vec<String>>) -> bool {
    labels
        .as_ref()
        .map(|l| l.iter().any(|l| l.starts_with(CLAIM_LABEL_PREFIX)))
        .unwrap_or(false)
}

/// Returns true when this agent's claim label is present on the MR.
pub(crate) fn has_our_mr_claim(labels: &Option<Vec<String>>, agent_id: &str) -> bool {
    let claim_label = claim_label(agent_id);
    labels
        .as_ref()
        .is_some_and(|labels| labels.iter().any(|label| label == &claim_label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;

    // -----------------------------------------------------------------
    // Fake claim port + fake settle/sleeper: a narrow in-memory double for
    // `ClaimPort`, deliberately not a general GitLab mock. Shared (via a
    // shared reference) between multiple simulated `acquire_with` calls so
    // tests can characterize contention/order-independence without threads
    // or a real GitLab backend.
    // -----------------------------------------------------------------

    #[derive(Default)]
    struct FakeClaimPort {
        labels: RefCell<HashMap<ClaimResource, Vec<String>>>,
        add_calls: Cell<u32>,
        remove_calls: Cell<u32>,
        label_reads: Cell<u32>,
        fail_reads_from_call: Cell<Option<u32>>,
    }

    impl FakeClaimPort {
        fn new() -> Self {
            Self::default()
        }

        fn with_labels(resource: ClaimResource, labels: &[&str]) -> Self {
            let port = Self::new();
            port.labels
                .borrow_mut()
                .insert(resource, labels.iter().map(|l| l.to_string()).collect());
            port
        }

        /// Make the Nth (1-indexed) and every subsequent `labels()` call
        /// fail, to exercise cleanup-on-read-failure.
        fn fail_reads_from(&self, call: u32) {
            self.fail_reads_from_call.set(Some(call));
        }

        fn current_labels(&self, resource: ClaimResource) -> Vec<String> {
            self.labels
                .borrow()
                .get(&resource)
                .cloned()
                .unwrap_or_default()
        }

        fn inject_label(&self, resource: ClaimResource, label: &str) {
            self.labels
                .borrow_mut()
                .entry(resource)
                .or_default()
                .push(label.to_string());
        }
    }

    impl ClaimPort for FakeClaimPort {
        fn add_label(&self, resource: ClaimResource, label: &str) -> Result<()> {
            self.add_calls.set(self.add_calls.get() + 1);
            self.labels
                .borrow_mut()
                .entry(resource)
                .or_default()
                .push(label.to_string());
            Ok(())
        }

        fn remove_label(&self, resource: ClaimResource, label: &str) -> Result<()> {
            self.remove_calls.set(self.remove_calls.get() + 1);
            if let Some(labels) = self.labels.borrow_mut().get_mut(&resource) {
                labels.retain(|l| l != label);
            }
            Ok(())
        }

        fn labels(&self, resource: ClaimResource) -> Result<Vec<String>> {
            let read_number = self.label_reads.get() + 1;
            self.label_reads.set(read_number);
            if let Some(fail_from) = self.fail_reads_from_call.get()
                && read_number >= fail_from
            {
                anyhow::bail!("simulated read failure");
            }
            Ok(self.current_labels(resource))
        }
    }

    /// A sleeper that never reports a shutdown — settle waits proceed
    /// instantly and always "complete".
    fn never_interrupts(_shutdown: &AtomicBool, _duration: Duration) -> bool {
        false
    }

    /// A sleeper that reports shutdown starting at the given (1-indexed)
    /// call number.
    fn interrupt_at(call: u32) -> impl FnMut(&AtomicBool, Duration) -> bool {
        let mut calls = 0u32;
        move |_: &AtomicBool, _: Duration| {
            calls += 1;
            calls >= call
        }
    }

    fn shutdown_flag() -> AtomicBool {
        AtomicBool::new(false)
    }

    // --- Acquisition: happy path, contention, order-independence ---

    #[test]
    fn acquire_wins_uncontested_claim() {
        let port = FakeClaimPort::new();
        let resource = ClaimResource::Issue(1);
        let shutdown = shutdown_flag();

        let outcome =
            acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap();

        let lease = match outcome {
            ClaimAcquireOutcome::Won(lease) => lease,
            _ => panic!("expected Won"),
        };
        assert_eq!(lease.resource(), resource);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
        lease.release(&port).unwrap();
    }

    #[test]
    fn acquire_resolves_contention_by_lexicographic_tiebreak() {
        let resource = ClaimResource::MergeRequest(7);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        // Simulate a competing instance's claim label appearing before the
        // first settle-read completes.
        let outcome = acquire_with(&port, resource, "worker-1", &shutdown, {
            let port = &port;
            let mut calls = 0u32;
            move |_: &AtomicBool, _: Duration| {
                calls += 1;
                if calls == 1 {
                    port.inject_label(resource, "claimed:worker-0");
                }
                false
            }
        })
        .unwrap();

        // "claimed:worker-0" < "claimed:worker-1" lexicographically, so
        // worker-1 loses and its own label is cleaned up.
        assert!(matches!(outcome, ClaimAcquireOutcome::Lost));
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
    }

    #[test]
    fn acquire_order_independence_the_smallest_label_always_wins() {
        // Two acquisitions race for the same resource on a shared port; the
        // outcome must not depend on which one's settle-read observes
        // contention first.
        for (first, second) in [("worker-0", "worker-1"), ("worker-1", "worker-0")] {
            let resource = ClaimResource::Issue(42);
            let port = FakeClaimPort::new();
            let shutdown = shutdown_flag();

            port.add_label(resource, &claim_label(first)).unwrap();
            port.add_label(resource, &claim_label(second)).unwrap();

            let outcome_first =
                acquire_with(&port, resource, first, &shutdown, never_interrupts).unwrap();
            let outcome_second =
                acquire_with(&port, resource, second, &shutdown, never_interrupts).unwrap();

            let winner_is_first = matches!(outcome_first, ClaimAcquireOutcome::Won(_));
            let winner_is_second = matches!(outcome_second, ClaimAcquireOutcome::Won(_));
            assert_ne!(winner_is_first, winner_is_second);

            let expected_winner = [first, second].into_iter().min().unwrap();
            assert_eq!(winner_is_first, first == expected_winner);
        }
    }

    // --- Cancellation / interruption ---

    #[test]
    fn acquire_reports_interrupted_and_cleans_up_on_shutdown_during_first_settle() {
        let resource = ClaimResource::Issue(9);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let outcome =
            acquire_with(&port, resource, "worker-0", &shutdown, interrupt_at(1)).unwrap();

        assert!(matches!(outcome, ClaimAcquireOutcome::Interrupted));
        assert!(port.current_labels(resource).is_empty());
    }

    #[test]
    fn acquire_reports_interrupted_and_cleans_up_on_shutdown_during_second_settle() {
        let resource = ClaimResource::MergeRequest(3);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let outcome =
            acquire_with(&port, resource, "worker-0", &shutdown, interrupt_at(2)).unwrap();

        assert!(matches!(outcome, ClaimAcquireOutcome::Interrupted));
        assert!(port.current_labels(resource).is_empty());
    }

    // --- Cleanup on read/settle failure ---

    #[test]
    fn acquire_cleans_up_the_claim_label_when_the_settle_read_fails() {
        let resource = ClaimResource::Issue(5);
        let port = FakeClaimPort::new();
        port.fail_reads_from(1);
        let shutdown = shutdown_flag();

        let result = acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts);

        assert!(result.is_err());
        assert!(port.current_labels(resource).is_empty());
    }

    #[test]
    fn acquire_cleans_up_the_claim_label_when_the_second_settle_read_fails() {
        let resource = ClaimResource::Issue(6);
        let port = FakeClaimPort::new();
        port.fail_reads_from(2);
        let shutdown = shutdown_flag();

        let result = acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts);

        assert!(result.is_err());
        assert!(port.current_labels(resource).is_empty());
    }

    // --- Explicit release / preserve ---

    #[test]
    fn explicit_release_removes_the_label_and_settles_the_lease() {
        let resource = ClaimResource::Issue(11);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let lease =
            match acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap() {
                ClaimAcquireOutcome::Won(lease) => lease,
                _ => panic!("expected Won"),
            };

        lease.release(&port).unwrap();
        assert!(port.current_labels(resource).is_empty());
    }

    #[test]
    fn explicit_preserve_keeps_the_label_and_does_no_port_work() {
        let resource = ClaimResource::Issue(12);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let lease =
            match acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap() {
                ClaimAcquireOutcome::Won(lease) => lease,
                _ => panic!("expected Won"),
            };

        let remove_calls_before = port.remove_calls.get();
        lease.preserve();
        assert_eq!(port.remove_calls.get(), remove_calls_before);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
    }

    // --- Drop performs no side effects ---

    #[test]
    fn dropping_an_unresolved_lease_does_no_network_or_filesystem_work() {
        let resource = ClaimResource::Issue(13);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let lease =
            match acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap() {
                ClaimAcquireOutcome::Won(lease) => lease,
                _ => panic!("expected Won"),
            };

        let add_calls_before = port.add_calls.get();
        let remove_calls_before = port.remove_calls.get();
        let read_calls_before = port.label_reads.get();

        drop(lease);

        assert_eq!(port.add_calls.get(), add_calls_before);
        assert_eq!(port.remove_calls.get(), remove_calls_before);
        assert_eq!(port.label_reads.get(), read_calls_before);
        // The label is left exactly as it was — Drop did not touch it.
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
    }

    #[test]
    fn dropping_a_released_or_preserved_lease_does_not_report_again() {
        // Not directly observable from outside the module, but exercised
        // here for documentation: release/preserve must mark the lease
        // settled so a subsequent Drop is a no-op warning-wise. We assert
        // the field directly since this test lives inside the module.
        let resource = ClaimResource::Issue(14);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();

        let lease =
            match acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap() {
                ClaimAcquireOutcome::Won(lease) => lease,
                _ => panic!("expected Won"),
            };
        lease.preserve();

        let lease2 = ClaimLease::recover(resource, "worker-0", &port.current_labels(resource))
            .expect("claim label is present");
        assert!(!lease2.settled);
        lease2.release(&port).unwrap();
    }

    // --- Recovered lease validation ---

    #[test]
    fn recover_returns_a_lease_only_when_the_live_label_is_present() {
        let resource = ClaimResource::Issue(20);

        let live_labels = vec!["claimed:worker-0".to_string(), "priority::1".to_string()];
        let lease = ClaimLease::recover(resource, "worker-0", &live_labels)
            .expect("our claim label is present");
        assert_eq!(lease.resource(), resource);
        lease.preserve();

        assert!(ClaimLease::recover(resource, "worker-1", &live_labels).is_none());
        assert!(ClaimLease::recover(resource, "worker-0", &[]).is_none());
    }

    #[test]
    fn recovered_lease_behaves_like_a_normal_lease() {
        let resource = ClaimResource::MergeRequest(21);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0"]);

        let lease = ClaimLease::recover(resource, "worker-0", &port.current_labels(resource))
            .expect("our claim label is present");
        lease.release(&port).unwrap();

        assert!(port.current_labels(resource).is_empty());
    }

    // --- Pure helpers ---

    #[test]
    fn has_our_mr_claim_detects_matching_label() {
        let labels = Some(vec![
            "reviewer-approved".to_string(),
            "claimed:reviewer-0".to_string(),
        ]);
        assert!(has_our_mr_claim(&labels, "reviewer-0"));
        assert!(!has_our_mr_claim(&labels, "reviewer-1"));
    }

    #[test]
    fn has_our_mr_claim_ignores_other_agents() {
        let labels = Some(vec!["claimed:reviewer-1".to_string()]);
        assert!(!has_our_mr_claim(&labels, "reviewer-0"));
        assert!(is_mr_claimed(&labels));
    }

    #[test]
    fn claim_label_formats_with_prefix() {
        assert_eq!(claim_label("worker-3"), "claimed:worker-3");
    }

    #[test]
    fn is_claimed_detects_any_claim_prefix() {
        assert!(!is_claimed(&["priority::1".to_string()]));
        assert!(is_claimed(&["claimed:worker-0".to_string()]));
    }

    // --- Tiebreak ordering: lexicographically smallest claim label wins.
    // This is the exact resolution rule `acquire` uses after the
    // double-settle re-check finds more than one claim label. It is
    // characterized directly, without a claim port, so the deadlock
    // between two concurrently-claiming instances always resolves the same
    // way regardless of which instance observes contention first.

    #[test]
    fn claim_tiebreak_wins_when_own_label_is_lexicographically_smallest() {
        let a = "claimed:worker-0".to_string();
        let b = "claimed:worker-1".to_string();
        assert!(claim_tiebreak_wins(&a, &[&a, &b]));
        assert!(claim_tiebreak_wins(&a, &[&b, &a]));
    }

    #[test]
    fn claim_tiebreak_loses_when_another_label_is_smaller() {
        let a = "claimed:worker-0".to_string();
        let b = "claimed:worker-1".to_string();
        assert!(!claim_tiebreak_wins(&b, &[&a, &b]));
    }

    #[test]
    fn claim_tiebreak_wins_alone_with_no_contenders() {
        let a = "claimed:worker-0".to_string();
        assert!(claim_tiebreak_wins(&a, &[&a]));
    }

    #[test]
    fn claim_tiebreak_resolves_three_way_contention_deterministically() {
        let a = "claimed:worker-0".to_string();
        let b = "claimed:worker-1".to_string();
        let c = "claimed:worker-2".to_string();
        // Order of the slice must not affect the outcome.
        assert!(claim_tiebreak_wins(&a, &[&c, &a, &b]));
        assert!(claim_tiebreak_wins(&a, &[&b, &c, &a]));
        assert!(!claim_tiebreak_wins(&b, &[&a, &b, &c]));
        assert!(!claim_tiebreak_wins(&c, &[&a, &b, &c]));
    }

    // --- Free-function release (no held lease value) ---

    #[test]
    fn release_removes_the_agents_claim_label() {
        let resource = ClaimResource::Issue(30);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0", "priority::1"]);

        release(&port, resource, "worker-0").unwrap();

        assert_eq!(
            port.current_labels(resource),
            vec!["priority::1".to_string()]
        );
    }
}
