//! Agent-layer claim leases: atomically claim a GitLab issue or MR using a
//! label-based claim-and-verify protocol, and hand back an explicit lease
//! object representing held ownership.
//!
//! ## Protocol
//! 1. Refuse resources that already expose a claim label.
//! 2. Add this agent's claim label (`claimed:<agent_id>`) to the resource.
//! 3. Wait for the settle time, to let concurrent claims from other
//!    instances propagate through the GitLab API.
//! 4. Re-fetch the resource's labels, require our exact label to be visible,
//!    and check for competing claims.
//! 5. Repeat the wait+re-fetch once more, to guard against propagation
//!    delay on the first read.
//! 6. If more than one claim label is present, the earliest active GitLab
//!    label-add event wins; label order is only a backend fallback.
//! 7. A winner keeps verifying through every settle round before proceeding.
//!
//! A shutdown signal observed during either settle wait interrupts the
//! attempt. Losing, being interrupted, or failing to read the resource back
//! all clean up the same way: the claim label we just added is removed
//! before returning.
//!
//! ## Leases
//! A won (or recovered) claim is returned as a [`ClaimLease`]. The only way
//! to relinquish one is to call [`ClaimLease::try_release`] (remove the label
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

use anyhow::{Context, Result};
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

    fn ordered_claim_labels(
        &self,
        _resource: ClaimResource,
        active_claim_labels: &[String],
    ) -> Result<Vec<String>> {
        Ok(fallback_claim_order(active_claim_labels))
    }
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

    fn ordered_claim_labels(
        &self,
        resource: ClaimResource,
        active_claim_labels: &[String],
    ) -> Result<Vec<String>> {
        let events = match resource {
            ClaimResource::Issue(iid) => self.get_issue_label_events(iid)?,
            ClaimResource::MergeRequest(iid) => self.get_mr_label_events(iid)?,
        };
        crate::agents::gitlab::order_active_claim_labels(&events, active_claim_labels)
            .with_context(|| format!("could not order active claims on {resource}"))
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
        let claim_labels: Vec<_> = live_labels
            .iter()
            .filter(|label| label.starts_with(CLAIM_LABEL_PREFIX))
            .collect();
        (claim_labels.len() == 1 && claim_labels[0] == &label)
            .then(|| Self::new(resource, label, agent_id))
    }

    pub(crate) fn resource(&self) -> ClaimResource {
        self.resource
    }

    /// Attempt to release without consuming the lease, so a failed attempt
    /// (e.g. a transient GitLab API error) can be retried later against the
    /// same lease rather than losing track of held ownership. On success
    /// the lease is marked settled; on failure it is left unsettled, still
    /// owned by the caller.
    pub(crate) fn try_release(&mut self, port: &dyn ClaimPort) -> Result<()> {
        debug!("{}: Releasing claim on {}", self.agent_id, self.resource);
        remove_claim_label(port, self.resource, &self.label)?;
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

    if port
        .labels(resource)?
        .iter()
        .any(|label| label.starts_with(CLAIM_LABEL_PREFIX))
    {
        debug!("{agent_id}: {resource} already has a claim, backing off");
        return Ok(ClaimAcquireOutcome::Lost);
    }

    if let Err(add_error) = port.add_label(resource, &label) {
        let labels = port.labels(resource).with_context(|| {
            format!(
                "failed to add claim {label:?} to {resource} ({add_error}); verification also failed"
            )
        })?;
        if !labels.iter().any(|candidate| candidate == &label) {
            return Err(add_error)
                .with_context(|| format!("claim {label:?} was not added to {resource}"));
        }
        info!("Claim {label:?} is present on {resource} after an ambiguous add failure");
    }

    let outcome = settle_and_check(port, resource, agent_id, &label, shutdown, &mut wait_settle);

    // Anything other than a win means we must not keep the label we just
    // added. Never hide cleanup failure: the remaining label still protects
    // the resource, and startup recovery can reclaim it, but callers must
    // know the acquisition did not finish cleanly.
    if !matches!(outcome, Ok(ClaimAcquireOutcome::Won(_))) {
        remove_claim_label(port, resource, &label)
            .context("failed to clean up an unsuccessful claim attempt")?;
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
    let mut own_label_visible = false;
    for _ in 0..CLAIM_SETTLE_ROUNDS {
        if wait_settle(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
            info!("{agent_id}: Shutdown while claiming {resource}, backing off");
            return Ok(ClaimAcquireOutcome::Interrupted);
        }

        let live_labels = port.labels(resource)?;
        let claim_labels: Vec<String> = live_labels
            .iter()
            .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
            .cloned()
            .collect();
        own_label_visible = claim_labels.iter().any(|candidate| candidate == label);

        if own_label_visible && claim_labels.len() > 1 {
            let ordered = port.ordered_claim_labels(resource, &claim_labels)?;
            if ordered.first().is_none_or(|winner| winner != label) {
                warn!(
                    "{agent_id}: Lost claim for {resource} to {}, backing off",
                    ordered
                        .first()
                        .map(String::as_str)
                        .unwrap_or("unknown owner")
                );
                return Ok(ClaimAcquireOutcome::Lost);
            }

            // Keep our winning label visible through every settle round.
            // Returning here would let a fast operation release the label
            // before a slower contender performs its own verification,
            // allowing both contenders to believe they won.
            info!(
                "{agent_id}: Won claim tiebreaker for {resource} against {} other(s); continuing verification",
                claim_labels.len() - 1
            );
        }
    }

    if !own_label_visible {
        warn!("{agent_id}: Claim label never became visible on {resource}, backing off");
        return Ok(ClaimAcquireOutcome::Lost);
    }

    info!("{agent_id}: Successfully claimed {resource}");
    Ok(ClaimAcquireOutcome::Won(ClaimLease::new(
        resource,
        label.to_string(),
        agent_id,
    )))
}

fn fallback_claim_order(active_claim_labels: &[String]) -> Vec<String> {
    let mut labels = active_claim_labels.to_vec();
    labels.sort();
    labels
}

/// Remove `agent_id`'s claim label from `resource` without a held
/// [`ClaimLease`] — for call sites that only ever tracked a resource's IID
/// across cycles (GitLab's label is the single source of truth for those),
/// never an in-memory lease value.
pub(crate) fn release(port: &dyn ClaimPort, resource: ClaimResource, agent_id: &str) -> Result<()> {
    let label = claim_label(agent_id);
    debug!("{agent_id}: Releasing claim on {resource}");
    remove_claim_label(port, resource, &label)
}

fn remove_claim_label(port: &dyn ClaimPort, resource: ClaimResource, label: &str) -> Result<()> {
    match port.remove_label(resource, label) {
        Ok(()) => Ok(()),
        Err(remove_error) => {
            let labels = port.labels(resource).with_context(|| {
                format!(
                    "failed to remove claim {label:?} from {resource} ({remove_error}); verification also failed"
                )
            })?;
            if labels.iter().any(|candidate| candidate == label) {
                Err(remove_error).with_context(|| {
                    format!("claim {label:?} is still present on {resource} after removal failed")
                })
            } else {
                info!(
                    "Claim {label:?} is absent from {resource} after an ambiguous removal failure"
                );
                Ok(())
            }
        }
    }
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
        fail_add: Cell<bool>,
        apply_failed_add: Cell<bool>,
        remove_calls: Cell<u32>,
        label_reads: Cell<u32>,
        fail_reads_from_call: Cell<Option<u32>>,
        fail_remove: Cell<bool>,
        apply_failed_remove: Cell<bool>,
        claim_order: RefCell<Option<Vec<String>>>,
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

        fn fail_remove(&self, apply_removal: bool) {
            self.fail_remove.set(true);
            self.apply_failed_remove.set(apply_removal);
        }

        fn fail_add(&self, apply_addition: bool) {
            self.fail_add.set(true);
            self.apply_failed_add.set(apply_addition);
        }

        fn set_claim_order(&self, labels: &[&str]) {
            *self.claim_order.borrow_mut() =
                Some(labels.iter().map(|label| label.to_string()).collect());
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
            if !self.fail_add.get() || self.apply_failed_add.get() {
                self.labels
                    .borrow_mut()
                    .entry(resource)
                    .or_default()
                    .push(label.to_string());
            }
            if self.fail_add.get() {
                anyhow::bail!("simulated add failure");
            }
            Ok(())
        }

        fn remove_label(&self, resource: ClaimResource, label: &str) -> Result<()> {
            self.remove_calls.set(self.remove_calls.get() + 1);
            if (!self.fail_remove.get() || self.apply_failed_remove.get())
                && let Some(labels) = self.labels.borrow_mut().get_mut(&resource)
            {
                labels.retain(|l| l != label);
            }
            if self.fail_remove.get() {
                anyhow::bail!("simulated remove failure");
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

        fn ordered_claim_labels(
            &self,
            _resource: ClaimResource,
            active_claim_labels: &[String],
        ) -> Result<Vec<String>> {
            Ok(self
                .claim_order
                .borrow()
                .clone()
                .unwrap_or_else(|| fallback_claim_order(active_claim_labels)))
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

        let mut lease = match outcome {
            ClaimAcquireOutcome::Won(lease) => lease,
            _ => panic!("expected Won"),
        };
        assert_eq!(lease.resource(), resource);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
        lease.try_release(&port).unwrap();
    }

    #[test]
    fn ambiguous_add_continues_when_verification_finds_the_label_present() {
        let port = FakeClaimPort::new();
        port.fail_add(true);
        let resource = ClaimResource::Issue(31);
        let shutdown = shutdown_flag();

        let outcome =
            acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap();
        assert!(matches!(outcome, ClaimAcquireOutcome::Won(_)));
    }

    #[test]
    fn failed_add_stops_when_verification_finds_no_label() {
        let port = FakeClaimPort::new();
        port.fail_add(false);
        let resource = ClaimResource::Issue(32);
        let shutdown = shutdown_flag();

        assert!(acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).is_err());
        assert!(port.current_labels(resource).is_empty());
    }

    #[test]
    fn acquire_loses_when_its_own_label_never_becomes_visible() {
        let port = FakeClaimPort::new();
        let resource = ClaimResource::Issue(2);
        let shutdown = shutdown_flag();

        let outcome = acquire_with(&port, resource, "worker-0", &shutdown, {
            let port = &port;
            move |_: &AtomicBool, _: Duration| {
                port.remove_label(resource, "claimed:worker-0").unwrap();
                false
            }
        })
        .unwrap();

        assert!(matches!(outcome, ClaimAcquireOutcome::Lost));
        assert!(port.current_labels(resource).is_empty());
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
    fn unsuccessful_acquisition_reports_claim_cleanup_failure() {
        let resource = ClaimResource::MergeRequest(10);
        let port = FakeClaimPort::new();
        port.fail_remove(false);
        let shutdown = shutdown_flag();

        let result = acquire_with(&port, resource, "worker-1", &shutdown, {
            let port = &port;
            let mut calls = 0u32;
            move |_: &AtomicBool, _: Duration| {
                calls += 1;
                if calls == 1 {
                    port.inject_label(resource, "claimed:worker-0");
                }
                false
            }
        });

        assert!(result.is_err());
        assert!(
            port.current_labels(resource)
                .contains(&"claimed:worker-1".to_string())
        );
    }

    #[test]
    fn earlier_label_event_wins_even_when_its_label_sorts_later() {
        let resource = ClaimResource::MergeRequest(9);
        let port = FakeClaimPort::new();
        port.set_claim_order(&["claimed:worker-1", "claimed:worker-0"]);
        let shutdown = shutdown_flag();

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

        let mut lease = match outcome {
            ClaimAcquireOutcome::Won(lease) => lease,
            _ => panic!("earliest active claim should win"),
        };
        assert_eq!(port.current_labels(resource).len(), 2);
        lease.try_release(&port).unwrap();
    }

    #[test]
    fn tiebreak_winner_keeps_its_label_through_all_settle_rounds() {
        let resource = ClaimResource::MergeRequest(8);
        let port = FakeClaimPort::new();
        let shutdown = shutdown_flag();
        let waits = Cell::new(0u32);

        let outcome = acquire_with(&port, resource, "worker-0", &shutdown, {
            let port = &port;
            let waits = &waits;
            move |_: &AtomicBool, _: Duration| {
                let call = waits.get() + 1;
                waits.set(call);
                if call == 1 {
                    port.inject_label(resource, "claimed:worker-1");
                } else {
                    port.remove_label(resource, "claimed:worker-1").unwrap();
                }
                false
            }
        })
        .unwrap();

        let mut lease = match outcome {
            ClaimAcquireOutcome::Won(lease) => lease,
            _ => panic!("expected tiebreak winner to retain the claim"),
        };
        assert_eq!(waits.get(), CLAIM_SETTLE_ROUNDS);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
        lease.try_release(&port).unwrap();
    }

    #[test]
    fn acquire_does_not_join_contention_when_a_claim_is_already_visible() {
        let resource = ClaimResource::Issue(42);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0"]);
        let shutdown = shutdown_flag();
        let waits = Cell::new(0u32);

        let outcome = acquire_with(
            &port,
            resource,
            "worker-1",
            &shutdown,
            |_: &AtomicBool, _: Duration| {
                waits.set(waits.get() + 1);
                false
            },
        )
        .unwrap();

        assert!(matches!(outcome, ClaimAcquireOutcome::Lost));
        assert_eq!(waits.get(), 0);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
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

        let mut lease =
            match acquire_with(&port, resource, "worker-0", &shutdown, never_interrupts).unwrap() {
                ClaimAcquireOutcome::Won(lease) => lease,
                _ => panic!("expected Won"),
            };

        lease.try_release(&port).unwrap();
        assert!(port.current_labels(resource).is_empty());
    }

    #[test]
    fn failed_try_release_keeps_the_lease_retryable_while_the_label_remains() {
        let resource = ClaimResource::Issue(15);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0"]);
        port.fail_remove(false);
        let mut lease =
            ClaimLease::recover(resource, "worker-0", &port.current_labels(resource)).unwrap();

        assert!(lease.try_release(&port).is_err());
        assert!(!lease.settled);
        assert_eq!(port.current_labels(resource), vec!["claimed:worker-0"]);
    }

    #[test]
    fn ambiguous_remove_is_success_when_verification_finds_the_label_absent() {
        let resource = ClaimResource::Issue(16);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0"]);
        port.fail_remove(true);
        let mut lease =
            ClaimLease::recover(resource, "worker-0", &port.current_labels(resource)).unwrap();

        lease.try_release(&port).unwrap();
        assert!(lease.settled);
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

        let mut lease2 = ClaimLease::recover(resource, "worker-0", &port.current_labels(resource))
            .expect("claim label is present");
        assert!(!lease2.settled);
        lease2.try_release(&port).unwrap();
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
        assert!(
            ClaimLease::recover(
                resource,
                "worker-0",
                &[
                    "claimed:worker-0".to_string(),
                    "claimed:worker-1".to_string()
                ]
            )
            .is_none()
        );
    }

    #[test]
    fn recovered_lease_behaves_like_a_normal_lease() {
        let resource = ClaimResource::MergeRequest(21);
        let port = FakeClaimPort::with_labels(resource, &["claimed:worker-0"]);

        let mut lease = ClaimLease::recover(resource, "worker-0", &port.current_labels(resource))
            .expect("our claim label is present");
        lease.try_release(&port).unwrap();

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

    #[test]
    fn fallback_claim_order_is_deterministic() {
        assert_eq!(
            fallback_claim_order(&[
                "claimed:worker-2".to_string(),
                "claimed:worker-0".to_string(),
                "claimed:worker-1".to_string(),
            ]),
            vec![
                "claimed:worker-0".to_string(),
                "claimed:worker-1".to_string(),
                "claimed:worker-2".to_string(),
            ]
        );
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
