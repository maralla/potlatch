use anyhow::Result;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::agents::gitlab::GitLabClient;
use crate::util::sleep;

/// Settle time in seconds. After adding a claim label, we wait this long
/// before verifying, to allow concurrent claims from other instances to
/// propagate through the GitLab API.
const CLAIM_SETTLE_SECS: u64 = 5;

const CLAIM_LABEL_PREFIX: &str = "claimed:";

pub fn claim_label(agent_id: &str) -> String {
    format!("{CLAIM_LABEL_PREFIX}{agent_id}")
}

/// Attempt to atomically claim a task (issue or MR) using the
/// claim-and-verify protocol with double-check.
///
/// Protocol:
/// 1. Add a unique claim label (`claimed:<agent_id>`) to the resource
/// 2. Wait for the settle time to let concurrent claims propagate
/// 3. Re-fetch the resource labels and check for competing claims
/// 4. Wait again and re-fetch a second time to guard against API propagation delay
/// 5. If only our claim label exists -> we won, return true
/// 6. If multiple claim labels exist -> deterministic tiebreaker (lexicographically smallest wins)
/// 7. Losers remove their claim label and return false
pub fn try_claim_issue(
    gitlab: &GitLabClient,
    issue_iid: u64,
    agent_id: &str,
    shutdown: &AtomicBool,
) -> Result<bool> {
    let claim_label = claim_label(agent_id);

    debug!("{}: Attempting to claim issue #{}", agent_id, issue_iid);

    gitlab.add_issue_label(issue_iid, &claim_label)?;

    if sleep(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
        let _ = gitlab.remove_issue_label(issue_iid, &claim_label);
        anyhow::bail!("Shutdown during claim settle for issue #{}", issue_iid);
    }

    let issue = gitlab.get_issue(issue_iid)?;
    let claim_labels: Vec<&String> = issue
        .labels
        .iter()
        .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
        .collect();

    if claim_labels.len() > 1 {
        return resolve_contention(gitlab, issue_iid, agent_id, &claim_label, &claim_labels);
    }

    if sleep(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
        let _ = gitlab.remove_issue_label(issue_iid, &claim_label);
        anyhow::bail!("Shutdown during claim settle for issue #{}", issue_iid);
    }

    let issue = gitlab.get_issue(issue_iid)?;
    let claim_labels: Vec<&String> = issue
        .labels
        .iter()
        .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
        .collect();

    if claim_labels.len() > 1 {
        return resolve_contention(gitlab, issue_iid, agent_id, &claim_label, &claim_labels);
    }

    info!("{}: Successfully claimed issue #{}", agent_id, issue_iid);
    Ok(true)
}

fn resolve_contention(
    gitlab: &GitLabClient,
    issue_iid: u64,
    agent_id: &str,
    claim_label: &str,
    claim_labels: &[&String],
) -> Result<bool> {
    let winner = claim_labels.iter().min().unwrap();

    if winner.as_str() == claim_label {
        info!(
            "{}: Won claim tiebreaker for issue #{} against {} other(s)",
            agent_id,
            issue_iid,
            claim_labels.len() - 1
        );
        return Ok(true);
    }

    warn!(
        "{}: Lost claim for issue #{} to {}, backing off",
        agent_id, issue_iid, winner
    );
    gitlab.remove_issue_label(issue_iid, claim_label)?;

    Ok(false)
}

/// Release a claim on an issue by removing the claim label.
pub fn release_claim(gitlab: &GitLabClient, issue_iid: u64, agent_id: &str) -> Result<()> {
    let claim_label = claim_label(agent_id);
    debug!("{}: Releasing claim on issue #{}", agent_id, issue_iid);
    gitlab.remove_issue_label(issue_iid, &claim_label)?;
    Ok(())
}

/// Check if an issue is already claimed by any instance.
pub fn is_claimed(labels: &[String]) -> bool {
    labels.iter().any(|l| l.starts_with(CLAIM_LABEL_PREFIX))
}

/// Attempt to atomically claim an MR using the claim-and-verify protocol with double-check.
pub fn try_claim_mr(
    gitlab: &GitLabClient,
    mr_iid: u64,
    agent_id: &str,
    shutdown: &AtomicBool,
) -> Result<bool> {
    let claim_label = claim_label(agent_id);

    debug!("{}: Attempting to claim MR !{}", agent_id, mr_iid);

    gitlab.add_mr_label_with_retries(mr_iid, &claim_label)?;

    let claim_result = (|| -> Result<bool> {
        if sleep(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
            anyhow::bail!("Shutdown during claim settle for MR !{}", mr_iid);
        }

        let mr = gitlab.get_merge_request(mr_iid)?;
        let mr_labels = mr.labels.as_deref().unwrap_or(&[]);
        let claim_labels: Vec<&String> = mr_labels
            .iter()
            .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
            .collect();

        if claim_labels.len() > 1 {
            return resolve_mr_contention(gitlab, mr_iid, agent_id, &claim_label, &claim_labels);
        }

        if sleep(shutdown, Duration::from_secs(CLAIM_SETTLE_SECS)) {
            anyhow::bail!("Shutdown during claim settle for MR !{}", mr_iid);
        }

        let mr = gitlab.get_merge_request(mr_iid)?;
        let mr_labels = mr.labels.as_deref().unwrap_or(&[]);
        let claim_labels: Vec<&String> = mr_labels
            .iter()
            .filter(|l| l.starts_with(CLAIM_LABEL_PREFIX))
            .collect();

        if claim_labels.len() > 1 {
            return resolve_mr_contention(gitlab, mr_iid, agent_id, &claim_label, &claim_labels);
        }

        info!("{}: Successfully claimed MR !{}", agent_id, mr_iid);
        Ok(true)
    })();

    match claim_result {
        Ok(won) => Ok(won),
        Err(e) => {
            let _ = gitlab.remove_mr_label(mr_iid, &claim_label);
            Err(e)
        }
    }
}

fn resolve_mr_contention(
    gitlab: &GitLabClient,
    mr_iid: u64,
    agent_id: &str,
    claim_label: &str,
    claim_labels: &[&String],
) -> Result<bool> {
    let winner = claim_labels.iter().min().unwrap();

    if winner.as_str() == claim_label {
        info!(
            "{}: Won claim tiebreaker for MR !{} against {} other(s)",
            agent_id,
            mr_iid,
            claim_labels.len() - 1
        );
        return Ok(true);
    }

    warn!(
        "{}: Lost claim for MR !{} to {}, backing off",
        agent_id, mr_iid, winner
    );
    gitlab.remove_mr_label(mr_iid, claim_label)?;

    Ok(false)
}

/// Release a claim on an MR by removing the claim label.
pub fn release_mr_claim(gitlab: &GitLabClient, mr_iid: u64, agent_id: &str) -> Result<()> {
    let claim_label = claim_label(agent_id);
    debug!("{}: Releasing claim on MR !{}", agent_id, mr_iid);
    gitlab.remove_mr_label(mr_iid, &claim_label)?;
    Ok(())
}

/// Check if an MR is already claimed by any instance.
pub fn is_mr_claimed(labels: &Option<Vec<String>>) -> bool {
    labels
        .as_ref()
        .map(|l| l.iter().any(|l| l.starts_with(CLAIM_LABEL_PREFIX)))
        .unwrap_or(false)
}

/// Returns true when this agent's claim label is present on the MR.
pub fn has_our_mr_claim(labels: &Option<Vec<String>>, agent_id: &str) -> bool {
    let claim_label = claim_label(agent_id);
    labels
        .as_ref()
        .is_some_and(|labels| labels.iter().any(|label| label == &claim_label))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
