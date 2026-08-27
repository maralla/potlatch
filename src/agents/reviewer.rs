use anyhow::Result;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::claim::{ClaimAcquireOutcome, ClaimLease, ClaimResource};
use super::{claim, mr_in_scope, write_task_context_file};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient, Issue, MergeRequest};
use crate::agents::workspace::{GitLabAgentBootstrap, GitLabAgentRuntime, gitlab_banner};
use crate::core::agent::schema::tagged;
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, compat, structured_output};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;

const REVIEWER_APPROVED_LABEL: &str = "reviewer-approved";
const NEED_AI_WORKER_LABEL: &str = "need-ai-worker";

/// The reviewer's typed structured-output contract: a tagged union on
/// `decision`, so each outcome carries exactly the fields it needs and cannot
/// carry the other outcome's. The model calls the `review` tool; core
/// validates the captured JSON against [`ReviewerOutput::schema`] and
/// deserializes it (see [`AgentModel::complete_typed`]). Core converts the
/// declarative schema to whatever wire format the model backend expects —
/// this file never builds backend JSON directly.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerOutput {
    Approve {
        summary: Option<String>,
    },
    RequestChanges {
        feedback: String,
        public_comment: Option<String>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApproveWire {
    #[serde(default)]
    summary: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestChangesWire {
    feedback: String,
    #[serde(default)]
    public_comment: Option<String>,
}

impl<'de> Deserialize<'de> for ReviewerOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (decision, fields) = tagged::parts(deserializer, "decision")?;
        match decision.as_str() {
            "approve" => tagged::branch(fields).map(|wire: ApproveWire| Self::Approve {
                summary: wire.summary,
            }),
            "request_changes" => {
                tagged::branch(fields).map(|wire: RequestChangesWire| Self::RequestChanges {
                    feedback: wire.feedback,
                    public_comment: wire.public_comment,
                })
            }
            decision => Err(serde::de::Error::unknown_variant(
                decision,
                &["approve", "request_changes"],
            )),
        }
    }
}

structured_output! {
    impl ReviewerOutput {
        tool_name: "review";
        tool_description: "The final merge-request review decision. Approval summaries must stay on one line.";
        schema: one_of(
            "decision",
            "Your review decision. Pick exactly one and send only that decision's fields.",
            {
                "approve" => (
                    "The merge request is ready to merge as-is.",
                    object({
                        optional summary: string(
                            "Optional one-line note. The posted comment is always just 'LGTM', so this is only for the log."
                        ),
                    })
                ),
                "request_changes" => (
                    "The merge request needs work before it can merge.",
                    object({
                        required feedback: string(
                            "Specific issues that must be addressed, one bullet per line. Posted as discussion threads."
                        ),
                        optional public_comment: string(
                            "Human-facing comment text (separate from feedback). Use for explanations, context, or recommendations that don't require code changes."
                        ),
                    })
                ),
            }
        );
    /// Tolerated: a decision spelled with different case or padding
    /// (`"APPROVE"`, `" approve "`).
        normalize(value) {
            compat::normalize_tag(value, "decision");
        }
    }
}

/// The model-generated brief of a merge request, used as the description of an
/// issue created when the MR has no linked issue.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MrBriefOutput {
    description: String,
}

structured_output! {
    impl MrBriefOutput {
        tool_name: "mr_brief";
        tool_description: "A brief issue description summarizing what the merge request changes.";
        schema: object("A concise issue description for the merge request.", {
            required description: string(
                "A concise issue description (2-5 sentences) summarizing what the merge request changes: the goal, the approach, and the affected area. Base it on the MR title, description, and diff. Do not include review verdicts or implementation steps."
            ),
        });
    }
}

#[derive(Debug, Clone)]
struct ReviewerConfig {
    poll_interval: Duration,
    merge_when_approved: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ReviewerAgentSettings {
    #[serde(
        default = "default_reviewer_poll_interval",
        deserialize_with = "crate::core::config::duration::deserialize"
    )]
    poll_interval: Duration,
    poll_interval_secs: Option<u64>,
    #[serde(default = "default_merge_when_approved")]
    merge_when_approved: bool,
}

fn default_reviewer_poll_interval() -> Duration {
    Duration::from_secs(120)
}

fn default_merge_when_approved() -> bool {
    true
}

pub(crate) struct ReviewerAgent {
    runtime: GitLabAgentRuntime,
    config: ReviewerConfig,
    merged_mrs: HashSet<u64>,
    claimed_mr: Option<ClaimLease>,
}

impl CoreAgent for ReviewerAgent {
    type Settings = ReviewerAgentSettings;

    fn name() -> &'static str {
        "reviewer"
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime.core
    }

    fn banner(config: &Config, banner: &mut Banner) {
        gitlab_banner(config, banner);
    }

    fn validate_settings(
        config: &Config,
        _section: &crate::core::config::AgentSection,
        settings: &Self::Settings,
    ) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_gitlab_repo()?;
        anyhow::ensure!(
            settings.poll_interval_secs.is_none(),
            "poll_interval_secs was replaced by poll_interval for [agent.reviewer]"
        );
        Ok(())
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec::polling(
            "gitlab_poll",
            self.config.poll_interval,
        )]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "gitlab_poll" => {
                let scope = crate::agents::scope_label_filter(&self.runtime.scope_label);
                let model = &self.runtime.model;
                let shutdown = Arc::clone(model.shutdown());
                reviewer_cycle(
                    &self.runtime.agent_id,
                    &self.runtime.project_name,
                    &self.runtime.working_dir,
                    &self.runtime.sessions_dir,
                    &self.config,
                    &self.runtime.git_repo,
                    &self.runtime.gitlab,
                    model,
                    &mut self.merged_mrs,
                    &mut self.claimed_mr,
                    &shutdown,
                    scope,
                )
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = GitLabAgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        let settings = ctx.settings;
        let config = ReviewerConfig {
            poll_interval: settings.poll_interval,
            merge_when_approved: settings.merge_when_approved,
        };
        let scope = crate::agents::scope_label_filter(&runtime.scope_label);
        let claimed_mr = find_claimed_mr(&runtime.agent_id, &runtime.gitlab, scope);
        Ok(Self {
            runtime,
            config,
            merged_mrs: HashSet::new(),
            claimed_mr,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.runtime.agent_id);
        if let Some(lease) = self.claimed_mr.take() {
            info!(
                "{}: Preserving claim on MR !{} for restart",
                self.runtime.agent_id,
                lease.resource().iid()
            );
            // GitLab's claim label is the source of truth across restarts;
            // no local state to persist (see `find_claimed_mr`).
            lease.preserve();
        }
        let _ = self.runtime.git_repo.reset_hard();
        info!("{}: Stopped", self.runtime.agent_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewOutcome {
    Merged,
    ApprovedWithoutMerge,
    NeedsChanges,
}

// ---------------------------------------------------------------------------
// Reviewer role port
// ---------------------------------------------------------------------------

/// Immutable snapshot of the merge-request fields the reviewer's decisions
/// read. Role-local on purpose: the reviewer machine never sees a general
/// GitLab object, only the fields its own decisions are allowed to look at,
/// and it can never mutate what it observed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MrObservation {
    iid: u64,
    title: String,
    description: String,
    source_branch: String,
    target_branch: String,
    state: String,
    labels: Option<Vec<String>>,
}

impl MrObservation {
    fn from_merge_request(mr: &MergeRequest) -> Self {
        Self {
            iid: mr.iid,
            title: mr.title.clone(),
            description: mr.description.clone(),
            source_branch: mr.source_branch.clone(),
            target_branch: mr.target_branch.clone(),
            state: mr.state.clone(),
            labels: mr.labels.clone(),
        }
    }

    fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    fn has_label(&self, label: &str) -> bool {
        mr_labels_has(self.labels(), label)
    }

    fn in_scope(&self, scope_label: Option<&str>) -> bool {
        super::mr_labels_in_scope(self.labels(), scope_label)
    }
}

/// Resolvable discussion counts for one MR, as the discussions API reports
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiscussionCounts {
    unresolved: usize,
    total: usize,
}

impl DiscussionCounts {
    fn any_unresolved(self) -> bool {
        self.unresolved > 0
    }
}

/// Everything gathered about the MR the reviewer claimed and passes to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReviewSubject {
    mr: MrObservation,
    issue_iid: Option<u64>,
    is_need_ai_worker_mr: bool,
    diff_stat: String,
    changed_files: Vec<String>,
}

/// Result of trying to acquire one MR claim. The lease itself remains owned by
/// the port so it can preserve the cross-cycle release semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimAttempt {
    Won,
    Lost,
    Interrupted,
}

/// The narrow, typed surface needed by the reviewer workflows.
trait ReviewerPort {
    fn shutdown_requested(&self) -> bool;
    fn default_branch(&self) -> Result<String>;
    fn fetch_remote(&mut self) -> Result<()>;
    /// Best effort by contract.
    fn reset_worktree(&mut self);
    fn checkout_branch(&mut self, branch: &str) -> Result<()>;
    fn recover_claim(&mut self);
    /// A failure retains the lease so the next cycle retries the same release.
    fn try_release_held_claim(&mut self) -> Result<()>;
    /// Warns on release failure and always drops the lease.
    fn release_held_claim(&mut self);
    fn release_stale_claim_label(&mut self, mr_iid: u64);
    fn merge_requests(&self) -> Result<Vec<MrObservation>>;
    /// Best effort: unavailable issue priorities are represented by an empty list.
    fn issue_priorities(&self) -> Vec<(u64, u8)>;
    fn unresolved_discussions(&self, mr_iid: u64) -> Result<DiscussionCounts>;
    fn acquire_claim(&mut self, mr_iid: u64) -> Result<ClaimAttempt>;
    fn branch_tips(&self, target_branch: &str) -> Result<(String, String)>;
    fn merge_target_into_worktree(&mut self, target_branch: &str) -> Result<bool>;
    fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)>;
    fn invoke_review_model(&mut self, subject: &ReviewSubject) -> Result<ReviewerOutput>;
    fn post_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()>;
    fn post_resolved_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()>;
    fn create_issue(&mut self, title: &str, description: &str) -> Result<u64>;
    fn generate_issue_brief(
        &mut self,
        mr: &MrObservation,
        diff_stat: &str,
        changed_files: &[String],
    ) -> Result<String>;
    fn add_approved_label(&mut self, mr_iid: u64) -> Result<()>;
    fn merge_merge_request(&mut self, mr_iid: u64) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Pure reviewer decisions
// ---------------------------------------------------------------------------

/// Why the scan skips a candidate MR, or what it must do before screening
/// it further. Mirrors the filter chain at the top of the candidate loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateDecision {
    Skip,
    /// Our own claim label is still on the MR from an interrupted cycle:
    /// drop it, then keep screening this same MR.
    ReleaseOurStaleClaim,
    Screen,
}

fn decide_candidate(
    mr: &MrObservation,
    agent_id: &str,
    scope_label: Option<&str>,
) -> CandidateDecision {
    if mr.state != "opened"
        || !mr.in_scope(scope_label)
        || mr.has_label(REVIEWER_APPROVED_LABEL)
        || mr.title.starts_with("Draft:")
    {
        return CandidateDecision::Skip;
    }
    if claim::has_our_mr_claim(&mr.labels, agent_id) {
        return CandidateDecision::ReleaseOurStaleClaim;
    }
    if claim::is_mr_claimed(&mr.labels) {
        return CandidateDecision::Skip;
    }
    CandidateDecision::Screen
}

/// Order candidate MRs the way the reviewer picks work up: `need-ai-worker`
/// MRs first, then by the priority of the linked issue (lowest number
/// first), then oldest MR first.
fn sort_review_candidates(candidates: &mut [MrObservation], priorities: &[(u64, u8)]) {
    let priority_map: std::collections::HashMap<u64, u8> = priorities.iter().copied().collect();
    candidates.sort_by(|a, b| {
        let aa = a.has_label(NEED_AI_WORKER_LABEL);
        let bb = b.has_label(NEED_AI_WORKER_LABEL);
        if aa != bb {
            return bb.cmp(&aa);
        }
        let pa = gitlab::issue_iid_from_branch(&a.source_branch)
            .and_then(|iid| priority_map.get(&iid).copied())
            .unwrap_or(gitlab::DEFAULT_PRIORITY);
        let pb = gitlab::issue_iid_from_branch(&b.source_branch)
            .and_then(|iid| priority_map.get(&iid).copied())
            .unwrap_or(gitlab::DEFAULT_PRIORITY);
        pa.cmp(&pb).then_with(|| a.iid.cmp(&b.iid))
    });
}

/// The issue this MR closes, read from `Closes #N` in the description and
/// falling back to the `issue-N` branch convention.
fn linked_issue_iid(mr: &MrObservation) -> Option<u64> {
    let re = regex::Regex::new(r"(?i)closes?\s+#(\d+)").ok();
    re.and_then(|r| {
        r.captures(&mr.description)
            .and_then(|c| c.get(1)?.as_str().parse().ok())
    })
    .or_else(|| gitlab::issue_iid_from_branch(&mr.source_branch))
}

/// Whether the MR can be reviewed at all, or must be handed straight back
/// with a discussion. Pure — no GitLab call — so the pre-review gates keep
/// their exact wording and precedence without a live client.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PreReviewGate {
    MissingIssueLink,
    BadMetadata(String),
    Proceed,
}

fn decide_pre_review_gate(mr: &MrObservation, subject: &ReviewSubject) -> PreReviewGate {
    if subject.issue_iid.is_none() && !subject.is_need_ai_worker_mr {
        return PreReviewGate::MissingIssueLink;
    }
    if has_bad_title_or_description(&mr.title, &mr.description) {
        let mut issues = Vec::new();
        if is_generic_title(&mr.title) {
            issues.push(format!(
                "The MR title `{}` is too generic. Please provide a concise, descriptive title that summarizes what the code changes actually do.",
                mr.title
            ));
        }
        if is_generic_description(&mr.description) {
            issues.push(
                "The MR description is missing or generic. Please provide a description that explains the goal, implementation approach, and testing."
                    .to_string(),
            );
        }
        return PreReviewGate::BadMetadata(issues.join("\n\n"));
    }
    PreReviewGate::Proceed
}

const MISSING_ISSUE_LINK_BODY: &str = "This MR does not reference an issue. Please link it to the relevant issue by using a branch name like `issue-N` or adding `Closes #N` in the MR description.";

fn merge_conflict_body(target_branch: &str) -> String {
    format!(
        "This MR has merge conflicts with `{target_branch}`. Please rebase or resolve conflicts before review can proceed."
    )
}

fn merge_failed_body(error: &anyhow::Error) -> String {
    format!(
        "Code is approved, but automatic merge failed (`{error}`). \
         This is likely due to merge conflicts with the target branch. \
         Please rebase or resolve conflicts and push again."
    )
}

// ---------------------------------------------------------------------------
// Imperative reviewer workflows
// ---------------------------------------------------------------------------

fn run_reviewer_cycle(
    agent_id: &str,
    scope_label: Option<&str>,
    merge_when_approved: bool,
    port: &mut dyn ReviewerPort,
) -> Result<Option<u64>> {
    let default_branch = port.default_branch()?;
    port.fetch_remote()?;
    // Shutdown boundary 1: after the remote refresh.
    if port.shutdown_requested() {
        return Ok(None);
    }

    port.reset_worktree();
    port.checkout_branch(&default_branch)?;
    port.recover_claim();
    if let Err(error) = port.try_release_held_claim() {
        warn!("{agent_id}: Failed to release held claim: {error}, will retry next cycle");
        return Ok(None);
    }

    let mut candidates = port.merge_requests()?;
    // Shutdown boundary 2: after reading merge requests.
    if port.shutdown_requested() {
        return Ok(None);
    }
    sort_review_candidates(&mut candidates, &port.issue_priorities());

    for mr in candidates {
        // Shutdown boundary 3: before screening each candidate.
        if port.shutdown_requested() {
            return Ok(None);
        }
        match decide_candidate(&mr, agent_id, scope_label) {
            CandidateDecision::Skip => {
                if mr.state == "opened"
                    && mr.in_scope(scope_label)
                    && mr.has_label(REVIEWER_APPROVED_LABEL)
                {
                    debug!(
                        "{agent_id}: MR !{} already marked {}, skipping",
                        mr.iid, REVIEWER_APPROVED_LABEL
                    );
                } else if claim::is_mr_claimed(&mr.labels) {
                    debug!("{agent_id}: MR !{} already claimed, skipping", mr.iid);
                }
                continue;
            }
            CandidateDecision::ReleaseOurStaleClaim => {
                warn!(
                    "{agent_id}: MR !{} still has our claim label, releasing before retry",
                    mr.iid
                );
                port.release_stale_claim_label(mr.iid);
            }
            CandidateDecision::Screen => {}
        }

        match port.unresolved_discussions(mr.iid) {
            Ok(counts) if counts.any_unresolved() => continue,
            Ok(_) => {}
            Err(error) => {
                warn!(
                    "{agent_id}: Could not check discussions for MR !{}: {error}, skipping this cycle",
                    mr.iid
                );
                continue;
            }
        }

        match port.acquire_claim(mr.iid)? {
            ClaimAttempt::Lost => {
                info!("{agent_id}: Failed to claim MR !{}, skipping", mr.iid);
                continue;
            }
            ClaimAttempt::Interrupted => return Ok(None),
            ClaimAttempt::Won => {}
        }

        // Exactly one won claim is processed per cycle. Boundary 4 is after
        // acquisition, and a shutdown here still releases the freshly won claim.
        if port.shutdown_requested() {
            port.release_held_claim();
            return Ok(None);
        }

        info!("{agent_id}: Reviewing MR !{}: {}", mr.iid, mr.title);
        let outcome = match review_claimed_merge_request(agent_id, &mr, merge_when_approved, port) {
            Ok(outcome) => Some(outcome),
            Err(error) => {
                if !port.shutdown_requested() {
                    error!("{agent_id}: Failed to review MR !{}: {error:#}", mr.iid);
                }
                None
            }
        };

        match outcome {
            Some(ReviewOutcome::Merged) => info!("{agent_id}: MR !{} approved and merged", mr.iid),
            Some(ReviewOutcome::ApprovedWithoutMerge) => info!(
                "{agent_id}: MR !{} approved without merge, releasing claim",
                mr.iid
            ),
            Some(ReviewOutcome::NeedsChanges) => info!(
                "{agent_id}: MR !{} reviewed with feedback, releasing claim",
                mr.iid
            ),
            None => {}
        }
        port.release_held_claim();
        return Ok(matches!(outcome, Some(ReviewOutcome::Merged)).then_some(mr.iid));
    }

    Ok(None)
}

fn review_claimed_merge_request(
    agent_id: &str,
    mr: &MrObservation,
    merge_when_approved: bool,
    port: &mut dyn ReviewerPort,
) -> Result<ReviewOutcome> {
    let mut subject = ReviewSubject {
        mr: mr.clone(),
        issue_iid: linked_issue_iid(mr),
        is_need_ai_worker_mr: mr.has_label(NEED_AI_WORKER_LABEL),
        diff_stat: String::new(),
        changed_files: Vec::new(),
    };
    match decide_pre_review_gate(mr, &subject) {
        PreReviewGate::MissingIssueLink => {
            // The MR has no linked issue. Proceed with the review — after the
            // diff is computed below, a brief is generated and an issue is
            // created so the review has a real issue context.
        }
        PreReviewGate::BadMetadata(body) => {
            warn!(
                "MR !{} has a generic or missing title/description, requesting fix",
                mr.iid
            );
            port.post_discussion(mr.iid, &body)?;
            return Ok(ReviewOutcome::NeedsChanges);
        }
        PreReviewGate::Proceed => {}
    }

    port.reset_worktree();
    port.checkout_branch(&mr.source_branch)?;
    let (source_sha, target_sha) = port.branch_tips(&mr.target_branch)?;
    info!(
        "MR !{} diff: {} ({}) -> {} ({})",
        mr.iid, mr.source_branch, source_sha, mr.target_branch, target_sha
    );

    if !port.merge_target_into_worktree(&mr.target_branch)? {
        warn!(
            "MR !{} has merge conflicts with {}",
            mr.iid, mr.target_branch
        );
        port.post_discussion(mr.iid, &merge_conflict_body(&mr.target_branch))?;
        port.checkout_branch(&mr.target_branch)?;
        return Ok(ReviewOutcome::NeedsChanges);
    }

    (subject.diff_stat, subject.changed_files) = port.local_diff(&mr.target_branch)?;

    // If the MR has no linked issue, generate a brief from the diff and create
    // one so the review has a real issue context.
    if subject.issue_iid.is_none() && !subject.is_need_ai_worker_mr {
        match port.generate_issue_brief(mr, &subject.diff_stat, &subject.changed_files) {
            Ok(brief) => match port.create_issue(&mr.title, &brief) {
                Ok(issue_iid) => {
                    info!(
                        "MR !{} has no linked issue; created issue #{} with a model-generated brief",
                        mr.iid, issue_iid
                    );
                    subject.issue_iid = Some(issue_iid);
                }
                Err(error) => {
                    warn!(
                        "MR !{} does not reference any issue and creating one failed: {error}; requesting fix",
                        mr.iid
                    );
                    port.post_discussion(mr.iid, MISSING_ISSUE_LINK_BODY)?;
                    port.checkout_branch(&mr.target_branch)?;
                    return Ok(ReviewOutcome::NeedsChanges);
                }
            },
            Err(error) => {
                warn!(
                    "MR !{} does not reference any issue and generating a brief failed: {error}; requesting fix",
                    mr.iid
                );
                port.post_discussion(mr.iid, MISSING_ISSUE_LINK_BODY)?;
                port.checkout_branch(&mr.target_branch)?;
                return Ok(ReviewOutcome::NeedsChanges);
            }
        }
    }

    let decision = port.invoke_review_model(&subject)?;
    info!("{agent_id}: Reviewer agent finished MR !{}", mr.iid);
    // The model inspects the merged source worktree; restore the target before
    // posting feedback, approving, or attempting the server-side merge.
    port.checkout_branch(&mr.target_branch)?;

    match decision {
        ReviewerOutput::RequestChanges {
            feedback,
            public_comment,
        } => {
            info!("MR !{} needs changes", mr.iid);
            let body = extract_review_feedback(Some(&feedback), public_comment.as_deref());
            port.post_discussion(mr.iid, &body)?;
            Ok(ReviewOutcome::NeedsChanges)
        }
        ReviewerOutput::Approve { .. } => {
            info!("MR !{} approved by reviewer", mr.iid);
            let counts = port.unresolved_discussions(mr.iid).map_err(|error| {
                error.context(format!(
                    "Could not verify discussions are resolved for MR !{} before merge",
                    mr.iid
                ))
            })?;
            match plan_approval(counts.any_unresolved(), merge_when_approved) {
                ApprovalPlan::SkipMergeUnresolvedDiscussions => {
                    warn!(
                        "MR !{} approved but has unresolved discussions, skipping merge",
                        mr.iid
                    );
                    Ok(ReviewOutcome::NeedsChanges)
                }
                ApprovalPlan::AttemptMerge => match port.merge_merge_request(mr.iid) {
                    Ok(()) => {
                        info!("Successfully merged MR !{}", mr.iid);
                        Ok(ReviewOutcome::Merged)
                    }
                    Err(error) => {
                        warn!("Failed to merge MR !{}: {error}", mr.iid);
                        port.post_discussion(mr.iid, &merge_failed_body(&error))?;
                        Ok(ReviewOutcome::NeedsChanges)
                    }
                },
                ApprovalPlan::ApproveWithoutMerge => {
                    port.post_resolved_discussion(mr.iid, &extract_approval_message())?;
                    port.add_approved_label(mr.iid)?;
                    Ok(ReviewOutcome::ApprovedWithoutMerge)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Live reviewer port
// ---------------------------------------------------------------------------

struct LiveReviewerPort<'a> {
    agent_id: &'a str,
    project_name: &'a str,
    sessions_dir: &'a str,
    git_repo: &'a GitRepo,
    gitlab: &'a GitLabClient,
    model: &'a AgentModel,
    claimed_mr: &'a mut Option<ClaimLease>,
    shutdown: &'a AtomicBool,
    scope_label: Option<&'a str>,
}

impl ReviewerPort for LiveReviewerPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
    fn default_branch(&self) -> Result<String> {
        self.git_repo.get_default_branch()
    }
    fn fetch_remote(&mut self) -> Result<()> {
        self.git_repo.fetch()
    }
    fn reset_worktree(&mut self) {
        let _ = self.git_repo.reset_hard();
    }
    fn checkout_branch(&mut self, branch: &str) -> Result<()> {
        self.git_repo.checkout_remote_branch(branch)
    }
    fn recover_claim(&mut self) {
        if self.claimed_mr.is_none() {
            *self.claimed_mr = find_claimed_mr(self.agent_id, self.gitlab, self.scope_label);
        }
    }
    fn try_release_held_claim(&mut self) -> Result<()> {
        let Some(lease) = self.claimed_mr.as_mut() else {
            return Ok(());
        };
        let iid = lease.resource().iid();
        info!(
            "{}: Releasing held claim on MR !{} from previous cycle",
            self.agent_id, iid
        );
        lease.try_release(self.gitlab)?;
        *self.claimed_mr = None;
        Ok(())
    }
    fn release_held_claim(&mut self) {
        release_claimed_mr(self.claimed_mr, self.gitlab, self.agent_id);
    }
    fn release_stale_claim_label(&mut self, mr_iid: u64) {
        release_mr_claim_or_warn(self.gitlab, mr_iid, self.agent_id);
    }
    fn merge_requests(&self) -> Result<Vec<MrObservation>> {
        Ok(self
            .gitlab
            .list_merge_requests()?
            .iter()
            .map(MrObservation::from_merge_request)
            .collect())
    }
    fn issue_priorities(&self) -> Vec<(u64, u8)> {
        self.gitlab
            .list_issues()
            .unwrap_or_default()
            .iter()
            .map(|issue| (issue.iid, issue.priority()))
            .collect()
    }
    fn unresolved_discussions(&self, mr_iid: u64) -> Result<DiscussionCounts> {
        let (unresolved, total) = self
            .gitlab
            .get_unresolved_discussion_count(mr_iid)
            .map_err(|error| {
                error.context(format!("Failed to fetch discussions for MR !{mr_iid}"))
            })?;
        if unresolved == 0 && total > 0 {
            info!(
                "MR !{} all {} discussion(s) resolved, ready for re-review",
                mr_iid, total
            );
        }
        Ok(DiscussionCounts { unresolved, total })
    }
    fn acquire_claim(&mut self, mr_iid: u64) -> Result<ClaimAttempt> {
        match claim::acquire(
            self.gitlab,
            ClaimResource::MergeRequest(mr_iid),
            self.agent_id,
            self.shutdown,
        )? {
            ClaimAcquireOutcome::Won(lease) => {
                *self.claimed_mr = Some(lease);
                Ok(ClaimAttempt::Won)
            }
            ClaimAcquireOutcome::Lost => Ok(ClaimAttempt::Lost),
            ClaimAcquireOutcome::Interrupted => Ok(ClaimAttempt::Interrupted),
        }
    }
    fn branch_tips(&self, target_branch: &str) -> Result<(String, String)> {
        Ok((
            self.git_repo.rev_parse("HEAD")?,
            self.git_repo
                .rev_parse(&format!("origin/{target_branch}"))?,
        ))
    }
    fn merge_target_into_worktree(&mut self, target_branch: &str) -> Result<bool> {
        self.git_repo.try_merge(target_branch)
    }
    fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)> {
        Ok((
            self.git_repo.diff_stat_against(target_branch)?,
            self.git_repo.changed_files_against(target_branch)?,
        ))
    }
    fn invoke_review_model(&mut self, subject: &ReviewSubject) -> Result<ReviewerOutput> {
        let prompt = build_review_prompt(ReviewPromptInput {
            project_name: self.project_name,
            gitlab: self.gitlab,
            mr: &subject.mr,
            diff_stat: &subject.diff_stat,
            changed_files: &subject.changed_files,
            issue_iid: subject.issue_iid,
            is_need_ai_worker_mr: subject.is_need_ai_worker_mr,
            sessions_dir: self.sessions_dir,
        })?;
        Ok(self
            .model
            .complete_typed::<ReviewerOutput>(
                &prompt,
                &InvokeOptions {
                    activity_label: Some(format!(
                        "{} reviewing MR !{}",
                        self.model.agent_id(),
                        subject.mr.iid
                    )),
                    ..InvokeOptions::default()
                },
            )?
            .output)
    }
    fn generate_issue_brief(
        &mut self,
        mr: &MrObservation,
        diff_stat: &str,
        changed_files: &[String],
    ) -> Result<String> {
        let prompt = format!(
            r#"You are writing a brief issue description for a merge request that has no linked issue. Summarize what the MR changes so the issue can stand on its own.

MR title: {title}
MR description: {description}
MR source branch: {source_branch}
MR target branch: {target_branch}

Diff stat:
{diff_stat}

Changed files:
{changed_files}

Inspect the actual code changes in the repository (the source branch is checked out and merged with the target). Write a concise issue description (2-5 sentences) covering the goal, the approach, and the affected area. Do not include review verdicts or implementation steps. Return the description as plain markdown."#,
            title = mr.title,
            description = mr.description,
            source_branch = mr.source_branch,
            target_branch = mr.target_branch,
            diff_stat = diff_stat,
            changed_files = changed_files.join("\n"),
        );
        Ok(self
            .model
            .complete_typed::<MrBriefOutput>(
                &prompt,
                &InvokeOptions {
                    activity_label: Some(format!(
                        "{} briefing MR !{}",
                        self.model.agent_id(),
                        mr.iid
                    )),
                    ..InvokeOptions::default()
                },
            )?
            .output
            .description)
    }
    fn post_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()> {
        self.gitlab.add_mr_discussion(mr_iid, body)
    }
    fn post_resolved_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()> {
        self.gitlab.add_resolved_mr_discussion(mr_iid, body)
    }
    fn create_issue(&mut self, title: &str, description: &str) -> Result<u64> {
        self.gitlab.create_issue(title, description)
    }
    fn add_approved_label(&mut self, mr_iid: u64) -> Result<()> {
        self.gitlab
            .add_mr_label_with_retries(mr_iid, REVIEWER_APPROVED_LABEL)
    }
    fn merge_merge_request(&mut self, mr_iid: u64) -> Result<()> {
        self.gitlab.merge_mr(mr_iid)
    }
}

#[allow(clippy::too_many_arguments)]
fn reviewer_cycle(
    agent_id: &str,
    project_name: &str,
    _reviewer_dir: &str,
    sessions_dir: &str,
    config: &ReviewerConfig,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    model: &AgentModel,
    merged_mrs: &mut HashSet<u64>,
    claimed_mr: &mut Option<ClaimLease>,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut port = LiveReviewerPort {
        agent_id,
        project_name,
        sessions_dir,
        git_repo,
        gitlab,
        model,
        claimed_mr,
        shutdown,
        scope_label,
    };
    if let Some(mr_iid) =
        run_reviewer_cycle(agent_id, scope_label, config.merge_when_approved, &mut port)?
    {
        merged_mrs.insert(mr_iid);
    }
    Ok(())
}

fn release_mr_claim_or_warn(gitlab: &GitLabClient, mr_iid: u64, agent_id: &str) {
    if let Err(e) = claim::release(gitlab, ClaimResource::MergeRequest(mr_iid), agent_id) {
        warn!(
            "{}: Failed to release claim on MR !{}: {}",
            agent_id, mr_iid, e
        );
    }
}

/// Release the currently-held `claimed_mr` lease (if any). A failed removal
/// keeps the lease in memory so the next cycle can retry it.
fn release_claimed_mr(claimed_mr: &mut Option<ClaimLease>, gitlab: &GitLabClient, agent_id: &str) {
    let Some(lease) = claimed_mr.as_mut() else {
        return;
    };
    let mr_iid = lease.resource().iid();
    if let Err(e) = lease.try_release(gitlab) {
        warn!(
            "{}: Failed to release claim on MR !{}: {}",
            agent_id, mr_iid, e
        );
        return;
    }
    *claimed_mr = None;
}

/// What to do with an `Approve` decision, given whether the MR still has
/// unresolved discussions and whether the role is configured to merge on
/// approval. Pure — no GitLab call — so the approve/merge ordering policy
/// can be characterized without a live client. Unresolved discussions
/// always take priority over merging, and label/discussion side effects
/// only differ by whether a merge is attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalPlan {
    SkipMergeUnresolvedDiscussions,
    AttemptMerge,
    ApproveWithoutMerge,
}

fn plan_approval(has_unresolved: bool, merge_when_approved: bool) -> ApprovalPlan {
    if has_unresolved {
        ApprovalPlan::SkipMergeUnresolvedDiscussions
    } else if merge_when_approved {
        ApprovalPlan::AttemptMerge
    } else {
        ApprovalPlan::ApproveWithoutMerge
    }
}

struct ReviewPromptInput<'a> {
    project_name: &'a str,
    gitlab: &'a GitLabClient,
    mr: &'a MrObservation,
    diff_stat: &'a str,
    changed_files: &'a [String],
    issue_iid: Option<u64>,
    is_need_ai_worker_mr: bool,
    sessions_dir: &'a str,
}

fn build_review_prompt(input: ReviewPromptInput<'_>) -> Result<String> {
    let comments = input
        .gitlab
        .get_mr_comments(input.mr.iid)
        .unwrap_or_default();

    let comments_text = if comments.is_empty() {
        "No comments yet.".to_string()
    } else {
        comments
            .iter()
            .map(|c| c.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n")
    };

    let diff_stat_text = if input.diff_stat.is_empty() {
        "No diff stat detected.".to_string()
    } else {
        input.diff_stat.to_string()
    };

    let changed_files_text = if input.changed_files.is_empty() {
        "No changed files detected.".to_string()
    } else {
        input.changed_files.join("\n")
    };

    let issue_context = if let Some(iid) = input.issue_iid {
        build_issue_context(input.gitlab, iid)
    } else {
        String::new()
    };
    let context_path = write_task_context_file(
        input.sessions_dir,
        &format!("reviewer-mr-{}.md", input.mr.iid),
        &format!(
            "# Review Context\n\nProject: {project_name}\nMR: !{mr_iid} {mr_title}\nSource branch: {source_branch}\nTarget branch: {target_branch}\n\n## MR description\n{mr_description}\n\n## Linked issue context\n{issue_context}\n\n## Diff summary against {target_branch}\n{diff_stat}\n\n## Changed files\n{changed_files}\n\n## Comment history\n{comments}\n",
            project_name = input.project_name,
            mr_iid = input.mr.iid,
            mr_title = input.mr.title,
            source_branch = input.mr.source_branch,
            target_branch = input.mr.target_branch,
            mr_description = input.mr.description,
            issue_context = if issue_context.is_empty() {
                "No linked issue context.".to_string()
            } else {
                issue_context
            },
            diff_stat = diff_stat_text,
            changed_files = changed_files_text,
            comments = comments_text
        ),
    )?;

    let completeness_line = if input.is_need_ai_worker_mr {
        "9. COMPLETENESS CHECK (STRICT): For `need-ai-worker` MRs, evaluate completeness against the current MR title, MR description, diff, and comment history (do not require linked issue context). Treat later comments as updates to the requested work. If the current scope implied by those sources is missing or partial, list missing items and request changes.".to_string()
    } else {
        "9. COMPLETENESS CHECK (STRICT): Compare the actual local diff and changed files against the CURRENT linked issue requirements: issue title, issue description, issue comments, MR description, and MR comment history. Later comments may clarify, narrow, expand, or supersede earlier issue text. Every current requirement MUST be addressed in the implementation, but do not request changes for an older constraint that later comments removed, changed, or accepted as intentionally out of scope. If any current requirement is missing or only partially implemented, list missing items and request changes. This check is critical to avoid shipping incomplete features.".to_string()
    };
    let prompt = format!(
        r#"You are reviewing a merge request for a software project in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

SOURCE BRANCH: {}
TARGET BRANCH: {}
TASK CONTEXT FILE:
{}

CRITICAL REQUIREMENTS:
- Provide clear, actionable feedback in comments (do not ask questions)
- Review the full comment history to understand what feedback was already given and addressed
- Do NOT repeat feedback that has already been addressed
- Treat comments after the original issue description as requirement updates when they clarify, narrow, expand, or supersede earlier constraints
- Do NOT request changes for outdated requirements from the original issue when later issue or MR comments clearly changed the accepted scope

COMMENT STYLE (STRICT — for requesting changes and any posted feedback):
- Do NOT start with a long paragraph of hollow praise or thanks that only restates the diff or issue number (e.g. listing routes, files, or "aligns with #N" without adding a review decision). That adds no value and wastes the reader's time.
- Lead with what matters: **what must change before merge**, or **why you approve**. Use a direct lead-in such as `Request before merge:` or `Blocking:` when the MR must not merge until the item is addressed.
- Only request MR description updates after you have read the full `## MR description` section in the task context file (including everything after any `Closes #N` line). Do **not** treat an opening `Closes #N` as “description is only the closing line” when the rest of that section documents the work. If it already states goal, implementation approach, and verification, do not ask to expand the description.
- Comments must use reader-facing wording only. Do NOT mention internal field names such as `decision`, `feedback`, or `public_comment`. For example, say "Update the MR description to include the actual verification and testing performed", not "Update MR_DESCRIPTION with the actual verification/testing performed."
- Keep the comment focused: one short optional line of genuine substance is OK, but **never** pad with a multi-sentence "thanks for the thorough coverage" preface that duplicates the diff.

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before starting the review. Treat it as authoritative project policy.
2. Before inspecting or judging code, perform any repository setup or pre-review steps required by `AGENTS.md` (for example, updating submodules when the project policy says to do so). If a required setup command fails, request changes and include the failure as blocking review feedback.
3. Read the task context file above before starting the review. For description quality, rely on the full text under `## MR description` there (do not judge from the MR title line alone).
4. Review the full comment history to understand previous feedback, worker responses, and scope updates after the original issue was written
5. The source branch has already been merged with the target branch locally - you are on the merged result
6. Inspect the actual code changes in the merged local repository state instead of relying only on the summaries above.
7. Review the code changes thoroughly using the local repository state
8. Check if the implementation matches the current stated goal after considering the issue description, issue comments, MR description, and MR comment history
{}
10. Run tests locally to verify they pass (do NOT rely on CI/CD)
11. Run linting locally to verify it passes (do NOT rely on CI/CD)
12. Check code quality, best practices, and potential issues
13. Only raise NEW issues not already covered in previous comments
14. Readability and maintainability must be ensured

MR TITLE AND DESCRIPTION (STRICT — reject if violated):
- The MR title MUST be a concise, meaningful summary of the code changes. Reject if the title is generic (e.g. "Implementation changes", "Update", "Fix"), just an issue number, or contains markdown formatting like ** or backticks.
- The MR description MUST explain the goal, implementation approach, and testing. Judge using the full `## MR description` text in the task context file (not the MR title line alone). Reject only if that text is empty, a single generic sentence (e.g. "Implementation completed."), or does not describe the actual changes. If it includes substantive detail (sections like Goal / Implementation / Testing, or equivalent prose), it satisfies this requirement even when the first line is only `Closes #N` or similar.
- When rejecting for poor title/description, tell the worker exactly what is wrong using public MR terminology, such as "Please update the MR title..." or "Please update the MR description...". Do not ask for internal fields like `MR_TITLE` or `MR_DESCRIPTION`.

CHANGE SIZE LIMITS (reject if exceeded):
- Non-test, non-generated code changes should be around ~500 lines. If substantially over, request the worker to split the MR.
- Total changes including tests should be around ~1500 lines. If substantially over, request a split.
- If the linked issue has the `review-only` label, ignore these change size limits entirely. Allow the MR to be huge and do NOT request splitting, reducing scope, or smaller follow-up MRs because of size.
- Auto-generated code (files containing comments like "generated by", "auto-generated", "DO NOT EDIT", or similar) should NOT be counted toward either limit and should NOT be reviewed for code quality. Skip reviewing auto-generated files entirely — only verify they are properly gitignored or legitimately needed.

TEST QUALITY (STRICT — reject if violated):
- Be very cautious about tests that appear to pass but do not actually test the main logic. Common red flags:
  * Empty test bodies or tests that only assert `true` / `assert!(true)`
  * Tests that hard-code the expected output instead of exercising the real function
  * Tests whose assertions are trivially satisfied regardless of whether the implementation is correct (e.g. checking a return type exists but not its value)
  * Tests that were mutated or weakened to make them pass (e.g. removing the core assertion, catching all exceptions and ignoring them, mocking the function under test itself)
  * Tests that test only a helper or stub but skip the main feature being implemented
- Every test MUST exercise the actual production code path it claims to cover. If a test does not meaningfully verify the behavior described in the issue, reject it and ask for a real test.

AGENTS.md COMPLIANCE (STRICT — reject if violated):
- You MUST check every code change in the DIFF against the local `AGENTS.md` file. If the code violates any rule defined there (naming conventions, file structure, required patterns, forbidden patterns, testing requirements, etc.), you MUST reject and cite the specific rule being violated.
- AGENTS.md rules take precedence over general best practices when they conflict.
- If AGENTS.md specifies test locations, file naming, code style, or architecture patterns, verify the MR follows them exactly.

FILE HYGIENE (STRICT — reject if violated):
- Every file in the MR must be directly relevant to the final deliverable. Reject any file that is only useful during the development process but serves no purpose in the shipped result, such as scratch scripts, debug helpers, personal notes, TODO files, temporary test harnesses, or throwaway utilities.
- Files must be placed in the correct directory according to the project's conventions. A test file must live in the designated test directory, configuration files in the config directory, etc. If a file is in the wrong location, request it be moved before approving.
- Do NOT allow leftover artifacts: generated files that should be gitignored, editor config files, OS-specific metadata files (e.g. .DS_Store, Thumbs.db), or log files.
- If unsure whether a file belongs, check the project structure and AGENTS.md for conventions.
"#,
        input.project_name,
        input.mr.iid,
        input.mr.title,
        input.mr.source_branch,
        input.mr.target_branch,
        context_path,
        completeness_line
    );

    Ok(prompt)
}

fn build_issue_context(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let mut ctx = String::new();

    match gitlab.get_issue(issue_iid) {
        Ok(issue) => {
            ctx.push_str(&format_issue_context_header(issue_iid, &issue));
        }
        Err(e) => {
            warn!("Failed to fetch issue #{}: {}", issue_iid, e);
            return String::new();
        }
    }

    match gitlab.get_issue_comments(issue_iid) {
        Ok(comments) if !comments.is_empty() => {
            ctx.push_str("\nISSUE COMMENTS (read as possible updates to requirements):\n");
            for c in &comments {
                ctx.push_str(&format!("- {}: {}\n", c.author, c.body));
            }
            ctx.push('\n');
        }
        _ => {}
    }

    ctx
}

fn format_issue_context_header(issue_iid: u64, issue: &Issue) -> String {
    let labels = if issue.labels.is_empty() {
        "none".to_string()
    } else {
        issue.labels.join(", ")
    };

    format!(
        "\nLINKED ISSUE #{}: {}\nISSUE LABELS: {}\n\nISSUE DESCRIPTION:\n{}\n",
        issue_iid, issue.title, labels, issue.description
    )
}

const GENERIC_TITLES: &[&str] = &[
    "implementation changes",
    "implementation completed",
    "update",
    "changes",
    "fix",
    "fixes",
    "updates",
];

fn has_bad_title_or_description(title: &str, description: &str) -> bool {
    is_generic_title(title) || is_generic_description(description)
}

fn is_generic_title(title: &str) -> bool {
    let lower = title.trim().to_lowercase();
    if lower.is_empty() {
        return true;
    }
    GENERIC_TITLES.contains(&lower.as_str())
}

fn is_generic_description(desc: &str) -> bool {
    let trimmed = desc.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    lower == "implementation completed."
        || lower == "implementation changes"
        || lower == "implementation changes."
}

/// Resolved approval thread on GitLab: always just "LGTM" — no description.
fn extract_approval_message() -> String {
    "LGTM".to_string()
}

/// Removes a leading `REQUEST_CHANGES` marker (and `—` / `:`).
fn strip_request_changes_prefix(text: &str) -> String {
    let s = text.trim();
    let lower = s.to_lowercase();
    const PREFIX: &str = "request_changes";
    if !lower.starts_with(PREFIX) {
        return s.to_string();
    }
    let mut rest = s[PREFIX.len()..].trim_start();
    loop {
        let trimmed = rest.trim_start();
        if let Some(r) = trimmed.strip_prefix('—') {
            rest = r.trim_start();
            continue;
        }
        if let Some(r) = trimmed.strip_prefix('-') {
            rest = r.trim_start();
            continue;
        }
        if let Some(r) = trimmed.strip_prefix(':') {
            rest = r.trim_start();
            continue;
        }
        break;
    }
    rest.trim().to_string()
}

/// Strips the common “please fix the following before review” intro (and punctuation after it).
fn strip_review_boilerplate(text: &str) -> String {
    let mut s = text.trim();
    const BOILERPLATE: &[u8] = b"please fix the following before review";
    loop {
        let t = s.trim_start();
        let b = t.as_bytes();
        if b.len() < BOILERPLATE.len() || !b[..BOILERPLATE.len()].eq_ignore_ascii_case(BOILERPLATE)
        {
            break;
        }
        let after = &t[BOILERPLATE.len()..];
        s = after.trim_start_matches(|c: char| {
            c == ':' || c == '.' || c == '—' || c == '-' || c.is_whitespace()
        });
    }
    s.trim().to_string()
}

fn normalize_review_comment_body(text: &str) -> String {
    let stripped = strip_review_boilerplate(&strip_request_changes_prefix(text));
    stripped.trim().to_string()
}

/// Builds the GitLab discussion body for a `request_changes` decision from
/// the typed `feedback`/`public_comment` fields, falling back to a generic
/// message when the model left both empty.
fn extract_review_feedback(feedback: Option<&str>, public_comment: Option<&str>) -> String {
    if let Some(feedback) = feedback {
        let out = normalize_review_comment_body(feedback.trim());
        if !out.is_empty() {
            return out;
        }
    }
    if let Some(comment) = public_comment {
        let out = normalize_review_comment_body(comment.trim());
        if !out.is_empty() {
            return out;
        }
    }
    "Please review the changes and address any issues.".to_string()
}

fn mr_labels_has(labels: Option<&[String]>, label: &str) -> bool {
    labels.is_some_and(|labels| labels.iter().any(|existing| existing == label))
}

// ---------------------------------------------------------------------------
// Reviewer claim recovery
// ---------------------------------------------------------------------------

/// Scan all open MRs for this reviewer's claim label.
///
/// The GitLab label is the single source of truth — no local state file
/// needed — so a recovered claim only ever becomes a [`ClaimLease`] once
/// this scan finds the label live on GitLab, per [`ClaimLease::recover`].
fn find_claimed_mr(
    agent_id: &str,
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) -> Option<ClaimLease> {
    match gitlab.list_merge_requests() {
        Ok(mrs) => {
            for mr in &mrs {
                if mr.state != "opened" || !mr_in_scope(mr, scope_label) {
                    continue;
                }
                let live_labels = mr.labels.as_deref().unwrap_or(&[]);
                if let Some(lease) =
                    ClaimLease::recover(ClaimResource::MergeRequest(mr.iid), agent_id, live_labels)
                {
                    info!(
                        "{}: Found existing claim on MR !{}, will release next cycle",
                        agent_id, mr.iid
                    );
                    return Some(lease);
                }
            }
        }
        Err(e) => {
            warn!(
                "{}: Failed to scan MRs for existing claims: {}",
                agent_id, e
            );
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::conformance;
    use serde_json::json;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    #[test]
    fn extract_approval_message_is_always_short_lgtm() {
        assert_eq!(extract_approval_message(), "LGTM");
    }

    // -----------------------------------------------------------------
    // Imperative reviewer workflows driven through a recording typed port.
    // Traces are intentionally plain strings; payloads are captured separately.
    // -----------------------------------------------------------------

    const TEST_AGENT: &str = "reviewer-1";

    fn observation(iid: u64, labels: Option<Vec<&str>>) -> MrObservation {
        MrObservation {
            iid,
            title: format!("Fix the retry backoff cap for issue {iid}"),
            description: format!("Closes #{iid}\n\n## Goal\nCap the backoff."),
            source_branch: format!("issue-{iid}"),
            target_branch: "main".to_string(),
            state: "opened".to_string(),
            labels: labels.map(|labels| labels.into_iter().map(String::from).collect()),
        }
    }

    struct FakeReviewerPort {
        trace: RefCell<Vec<String>>,
        shutdown_answers: RefCell<VecDeque<bool>>,
        merge_requests: Vec<MrObservation>,
        issue_priorities: Vec<(u64, u8)>,
        discussion_counts: RefCell<VecDeque<DiscussionCounts>>,
        claim_attempts: RefCell<VecDeque<ClaimAttempt>>,
        review_output: ReviewerOutput,
        local_merge_clean: bool,
        recoverable_claim: Option<u64>,
        held_claim: Option<u64>,
        failures: Vec<String>,
        discussions: Vec<(u64, String)>,
        resolved_discussions: Vec<(u64, String)>,
        reviewed_subjects: Vec<ReviewSubject>,
        created_issues: RefCell<Vec<(String, String)>>,
    }

    impl FakeReviewerPort {
        fn new(merge_requests: Vec<MrObservation>) -> Self {
            Self {
                trace: RefCell::new(Vec::new()),
                shutdown_answers: RefCell::new(VecDeque::new()),
                merge_requests,
                issue_priorities: Vec::new(),
                discussion_counts: RefCell::new(VecDeque::new()),
                claim_attempts: RefCell::new(VecDeque::new()),
                review_output: ReviewerOutput::Approve { summary: None },
                local_merge_clean: true,
                recoverable_claim: None,
                held_claim: None,
                failures: Vec::new(),
                discussions: Vec::new(),
                resolved_discussions: Vec::new(),
                reviewed_subjects: Vec::new(),
                created_issues: RefCell::new(Vec::new()),
            }
        }

        fn trace(&self, entry: impl Into<String>) {
            self.trace.borrow_mut().push(entry.into());
        }
        fn fails(&self, operation: &str) -> Result<()> {
            if self.failures.iter().any(|failure| failure == operation) {
                anyhow::bail!("injected failure in {operation}");
            }
            Ok(())
        }
        fn deciding(mut self, output: ReviewerOutput) -> Self {
            self.review_output = output;
            self
        }
        fn with_discussions(self, counts: &[(usize, usize)]) -> Self {
            self.discussion_counts.borrow_mut().extend(
                counts
                    .iter()
                    .map(|&(unresolved, total)| DiscussionCounts { unresolved, total }),
            );
            self
        }
        fn with_shutdowns(self, answers: &[bool]) -> Self {
            self.shutdown_answers.borrow_mut().extend(answers);
            self
        }
        fn with_claims(self, attempts: &[ClaimAttempt]) -> Self {
            self.claim_attempts.borrow_mut().extend(attempts);
            self
        }
        fn with_priorities(mut self, priorities: &[(u64, u8)]) -> Self {
            self.issue_priorities = priorities.to_vec();
            self
        }
        fn failing(mut self, operation: &str) -> Self {
            self.failures.push(operation.to_string());
            self
        }
    }

    impl ReviewerPort for FakeReviewerPort {
        fn shutdown_requested(&self) -> bool {
            self.trace("shutdown");
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }
        fn default_branch(&self) -> Result<String> {
            self.trace("default_branch");
            Ok("main".into())
        }
        fn fetch_remote(&mut self) -> Result<()> {
            self.trace("fetch");
            self.fails("fetch")
        }
        fn reset_worktree(&mut self) {
            self.trace("reset");
        }
        fn checkout_branch(&mut self, branch: &str) -> Result<()> {
            self.trace(format!("checkout:{branch}"));
            self.fails("checkout")
        }
        fn recover_claim(&mut self) {
            if self.held_claim.is_none() {
                self.trace("recover_claim");
                self.held_claim = self.recoverable_claim;
            }
        }
        fn try_release_held_claim(&mut self) -> Result<()> {
            if self.held_claim.is_none() {
                return Ok(());
            }
            self.trace("release_previous");
            self.fails("release_previous")?;
            self.held_claim = None;
            Ok(())
        }
        fn release_held_claim(&mut self) {
            self.trace("release_final");
            self.held_claim = None;
        }
        fn release_stale_claim_label(&mut self, mr_iid: u64) {
            self.trace(format!("release_stale:{mr_iid}"));
        }
        fn merge_requests(&self) -> Result<Vec<MrObservation>> {
            self.trace("merge_requests");
            Ok(self.merge_requests.clone())
        }
        fn issue_priorities(&self) -> Vec<(u64, u8)> {
            self.trace("issue_priorities");
            self.issue_priorities.clone()
        }
        fn unresolved_discussions(&self, mr_iid: u64) -> Result<DiscussionCounts> {
            self.trace(format!("discussions:{mr_iid}"));
            self.fails(&format!("discussions:{mr_iid}"))?;
            Ok(self
                .discussion_counts
                .borrow_mut()
                .pop_front()
                .unwrap_or(DiscussionCounts {
                    unresolved: 0,
                    total: 0,
                }))
        }
        fn acquire_claim(&mut self, mr_iid: u64) -> Result<ClaimAttempt> {
            self.trace(format!("claim:{mr_iid}"));
            let attempt = self
                .claim_attempts
                .borrow_mut()
                .pop_front()
                .unwrap_or(ClaimAttempt::Won);
            if attempt == ClaimAttempt::Won {
                self.held_claim = Some(mr_iid);
            }
            Ok(attempt)
        }
        fn branch_tips(&self, target_branch: &str) -> Result<(String, String)> {
            self.trace(format!("tips:{target_branch}"));
            Ok(("source-sha".into(), "target-sha".into()))
        }
        fn merge_target_into_worktree(&mut self, target_branch: &str) -> Result<bool> {
            self.trace(format!("merge_target:{target_branch}"));
            Ok(self.local_merge_clean)
        }
        fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)> {
            self.trace(format!("diff:{target_branch}"));
            Ok((" src/lib.rs | 2 +-".into(), vec!["src/lib.rs".into()]))
        }
        fn invoke_review_model(&mut self, subject: &ReviewSubject) -> Result<ReviewerOutput> {
            self.trace(format!("model:{}", subject.mr.iid));
            self.reviewed_subjects.push(subject.clone());
            self.fails("model")?;
            Ok(self.review_output.clone())
        }
        fn post_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()> {
            self.trace(format!("post:{mr_iid}"));
            self.discussions.push((mr_iid, body.into()));
            Ok(())
        }
        fn post_resolved_discussion(&mut self, mr_iid: u64, body: &str) -> Result<()> {
            self.trace(format!("post_resolved:{mr_iid}"));
            self.resolved_discussions.push((mr_iid, body.into()));
            Ok(())
        }
        fn create_issue(&mut self, title: &str, description: &str) -> Result<u64> {
            self.trace(format!("create_issue:{title}"));
            self.fails("create_issue")?;
            self.created_issues
                .borrow_mut()
                .push((title.to_string(), description.to_string()));
            Ok(self.merge_requests.len() as u64 + 1000)
        }
        fn generate_issue_brief(
            &mut self,
            mr: &MrObservation,
            _diff_stat: &str,
            _changed_files: &[String],
        ) -> Result<String> {
            self.trace(format!("brief:{}", mr.iid));
            self.fails("brief")?;
            Ok(format!("Brief for MR !{}: {}", mr.iid, mr.title))
        }
        fn add_approved_label(&mut self, mr_iid: u64) -> Result<()> {
            self.trace(format!("label:{mr_iid}"));
            Ok(())
        }
        fn merge_merge_request(&mut self, mr_iid: u64) -> Result<()> {
            self.trace(format!("merge_mr:{mr_iid}"));
            self.fails("merge_mr")
        }
    }

    fn run(port: &mut FakeReviewerPort, merge_when_approved: bool) -> Result<Option<u64>> {
        run_reviewer_cycle(TEST_AGENT, None, merge_when_approved, port)
    }

    fn startup() -> Vec<String> {
        [
            "default_branch",
            "fetch",
            "shutdown",
            "reset",
            "checkout:main",
            "recover_claim",
            "merge_requests",
            "shutdown",
            "issue_priorities",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn through_model(iid: u64) -> Vec<String> {
        [
            "shutdown",
            &format!("discussions:{iid}"),
            &format!("claim:{iid}"),
            "shutdown",
            "reset",
            &format!("checkout:issue-{iid}"),
            "tips:main",
            "merge_target:main",
            "diff:main",
            &format!("model:{iid}"),
            "checkout:main",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn reviewer_cycle_approves_and_merges_one_merge_request_in_order() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        assert_eq!(run(&mut port, true).unwrap(), Some(7));
        let mut expected = startup();
        expected.extend(through_model(7));
        expected.extend(["discussions:7", "merge_mr:7", "release_final"].map(String::from));
        assert_eq!(*port.trace.borrow(), expected);
        assert_eq!(port.reviewed_subjects[0].diff_stat, " src/lib.rs | 2 +-");
        assert_eq!(port.held_claim, None);
    }

    #[test]
    fn reviewer_cycle_approves_without_merge_in_post_then_label_order() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        assert_eq!(run(&mut port, false).unwrap(), None);
        assert_eq!(
            &port.trace.borrow()[port.trace.borrow().len() - 4..],
            [
                "discussions:7",
                "post_resolved:7",
                "label:7",
                "release_final"
            ]
        );
        assert_eq!(port.resolved_discussions, vec![(7, "LGTM".into())]);
    }

    #[test]
    fn reviewer_cycle_does_not_merge_when_discussions_appear_during_review() {
        let mut port =
            FakeReviewerPort::new(vec![observation(7, None)]).with_discussions(&[(0, 0), (1, 1)]);
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert!(!port.trace.borrow().contains(&"merge_mr:7".into()));
        assert_eq!(port.trace.borrow().last().unwrap(), "release_final");
    }

    #[test]
    fn reviewer_cycle_posts_requested_changes_and_releases_claim() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).deciding(
            ReviewerOutput::RequestChanges {
                feedback: "- Add a test".into(),
                public_comment: None,
            },
        );
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert_eq!(port.discussions, vec![(7, "- Add a test".into())]);
        assert_eq!(
            &port.trace.borrow()[port.trace.borrow().len() - 2..],
            ["post:7", "release_final"]
        );
    }

    #[test]
    fn reviewer_cycle_handles_conflict_then_checks_out_target() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        port.local_merge_clean = false;
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert!(
            port.discussions[0]
                .1
                .contains("merge conflicts with `main`")
        );
        assert_eq!(
            &port.trace.borrow()[port.trace.borrow().len() - 3..],
            ["post:7", "checkout:main", "release_final"]
        );
        assert!(port.reviewed_subjects.is_empty());
    }

    #[test]
    fn reviewer_cycle_merge_failure_posts_feedback_before_release() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).failing("merge_mr");
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert!(
            port.discussions[0]
                .1
                .starts_with("Code is approved, but automatic merge failed")
        );
        assert_eq!(
            &port.trace.borrow()[port.trace.borrow().len() - 3..],
            ["merge_mr:7", "post:7", "release_final"]
        );
    }

    #[test]
    fn reviewer_cycle_model_failure_observes_shutdown_then_releases() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).failing("model");
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert_eq!(
            &port.trace.borrow()[port.trace.borrow().len() - 3..],
            ["model:7", "shutdown", "release_final"]
        );
        assert_eq!(port.held_claim, None);
    }

    #[test]
    fn reviewer_cycle_preserves_failed_previous_claim_release() {
        let mut port =
            FakeReviewerPort::new(vec![observation(7, None)]).failing("release_previous");
        port.held_claim = Some(4);
        assert_eq!(run(&mut port, true).unwrap(), None);
        assert_eq!(
            *port.trace.borrow(),
            [
                "default_branch",
                "fetch",
                "shutdown",
                "reset",
                "checkout:main",
                "release_previous"
            ]
        );
        assert_eq!(port.held_claim, Some(4));
    }

    #[test]
    fn reviewer_cycle_releases_recovered_claim_before_listing_work() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        port.recoverable_claim = Some(4);
        run(&mut port, true).unwrap();
        let trace = port.trace.borrow();
        let recovered = trace
            .iter()
            .position(|entry| entry == "recover_claim")
            .unwrap();
        assert_eq!(
            &trace[recovered..recovered + 3],
            ["recover_claim", "release_previous", "merge_requests"]
        );
    }

    #[test]
    fn reviewer_cycle_honors_all_four_shutdown_boundaries() {
        for stop_at in 0..4 {
            let mut answers = vec![false; stop_at];
            answers.push(true);
            let mut port =
                FakeReviewerPort::new(vec![observation(7, None)]).with_shutdowns(&answers);
            run(&mut port, true).unwrap();
            assert_eq!(
                port.trace
                    .borrow()
                    .iter()
                    .filter(|entry| *entry == "shutdown")
                    .count(),
                stop_at + 1
            );
            if stop_at == 3 {
                assert_eq!(port.trace.borrow().last().unwrap(), "release_final");
            }
        }
    }

    #[test]
    fn reviewer_cycle_skips_unreadable_candidate_and_claims_at_most_one() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)])
            .failing("discussions:7");
        assert_eq!(run(&mut port, true).unwrap(), Some(9));
        assert!(!port.trace.borrow().contains(&"claim:7".into()));
        assert_eq!(
            port.trace
                .borrow()
                .iter()
                .filter(|entry| entry.starts_with("model:"))
                .count(),
            1
        );
    }

    #[test]
    fn reviewer_cycle_lost_claim_moves_on_and_interrupted_stops() {
        let mut lost = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)])
            .with_claims(&[ClaimAttempt::Lost, ClaimAttempt::Won]);
        assert_eq!(run(&mut lost, true).unwrap(), Some(9));
        let mut interrupted = FakeReviewerPort::new(vec![observation(7, None)])
            .with_claims(&[ClaimAttempt::Interrupted]);
        assert_eq!(run(&mut interrupted, true).unwrap(), None);
        assert_eq!(interrupted.trace.borrow().last().unwrap(), "claim:7");
    }

    #[test]
    fn reviewer_cycle_filters_sorts_and_releases_own_stale_claim() {
        let mine = format!("claimed:{TEST_AGENT}");
        let mut port = FakeReviewerPort::new(vec![
            observation(5, Some(vec![REVIEWER_APPROVED_LABEL])),
            observation(6, Some(vec!["claimed:other"])),
            observation(7, Some(vec![mine.as_str(), NEED_AI_WORKER_LABEL])),
            observation(9, None),
        ])
        .with_priorities(&[(7, 1), (9, 3)]);
        assert_eq!(run(&mut port, true).unwrap(), Some(7));
        assert!(!port.trace.borrow().contains(&"discussions:5".into()));
        assert!(!port.trace.borrow().contains(&"discussions:6".into()));
        let trace = port.trace.borrow();
        let released = trace
            .iter()
            .position(|entry| entry == "release_stale:7")
            .unwrap();
        assert_eq!(
            &trace[released..released + 2],
            ["release_stale:7", "discussions:7"]
        );
        assert!(!trace.contains(&"claim:9".into()));
    }

    #[test]
    fn reviewer_cycle_rejects_gate_failures_before_git_or_model() {
        let mut mr = observation(7, None);
        mr.title = "Update".into();
        let mut port = FakeReviewerPort::new(vec![mr]);
        run(&mut port, true).unwrap();
        assert!(
            port.discussions[0]
                .1
                .contains("title `Update` is too generic")
        );
        assert_eq!(
            port.trace
                .borrow()
                .iter()
                .filter(|entry| *entry == "reset")
                .count(),
            1
        );
        assert!(port.reviewed_subjects.is_empty());
    }

    // -----------------------------------------------------------------
    // Pure reviewer decisions
    // -----------------------------------------------------------------

    #[test]
    fn decide_candidate_screens_only_open_unclaimed_unapproved_mrs() {
        assert_eq!(
            decide_candidate(&observation(7, None), TEST_AGENT, None),
            CandidateDecision::Screen
        );

        let mut closed = observation(7, None);
        closed.state = "merged".to_string();
        assert_eq!(
            decide_candidate(&closed, TEST_AGENT, None),
            CandidateDecision::Skip
        );

        assert_eq!(
            decide_candidate(
                &observation(7, Some(vec![REVIEWER_APPROVED_LABEL])),
                TEST_AGENT,
                None
            ),
            CandidateDecision::Skip
        );
        assert_eq!(
            decide_candidate(
                &observation(7, Some(vec!["claimed:other"])),
                TEST_AGENT,
                None
            ),
            CandidateDecision::Skip
        );
        let mine = format!("claimed:{TEST_AGENT}");
        assert_eq!(
            decide_candidate(&observation(7, Some(vec![mine.as_str()])), TEST_AGENT, None),
            CandidateDecision::ReleaseOurStaleClaim
        );
        assert_eq!(
            decide_candidate(&observation(7, None), TEST_AGENT, Some("potlatch")),
            CandidateDecision::Skip
        );
    }

    #[test]
    fn decide_pre_review_gate_requires_an_issue_link_unless_labeled_for_the_worker() {
        let mut unlinked = observation(7, None);
        unlinked.description = "No link here.".to_string();
        unlinked.source_branch = "feature".to_string();

        let subject = ReviewSubject {
            mr: unlinked.clone(),
            issue_iid: None,
            is_need_ai_worker_mr: false,
            diff_stat: String::new(),
            changed_files: Vec::new(),
        };
        assert_eq!(
            decide_pre_review_gate(&unlinked, &subject),
            PreReviewGate::MissingIssueLink
        );

        let labeled = ReviewSubject {
            is_need_ai_worker_mr: true,
            ..subject
        };
        assert_eq!(
            decide_pre_review_gate(&unlinked, &labeled),
            PreReviewGate::Proceed
        );
    }

    #[test]
    fn linked_issue_iid_prefers_the_closes_keyword_over_the_branch_name() {
        let mut mr = observation(7, None);
        mr.description = "Closes #42".to_string();
        assert_eq!(linked_issue_iid(&mr), Some(42));

        mr.description = "No keyword".to_string();
        assert_eq!(linked_issue_iid(&mr), Some(7));

        mr.source_branch = "feature".to_string();
        assert_eq!(linked_issue_iid(&mr), None);
    }

    #[test]
    fn sort_review_candidates_orders_labeled_then_priority_then_oldest() {
        let mut candidates = vec![
            observation(9, None),
            observation(7, None),
            observation(11, Some(vec![NEED_AI_WORKER_LABEL])),
        ];
        sort_review_candidates(&mut candidates, &[(7, 3), (9, 1)]);
        let order: Vec<u64> = candidates.iter().map(|mr| mr.iid).collect();
        assert_eq!(order, vec![11, 9, 7]);
    }

    // -----------------------------------------------------------------
    // Approve/merge/request-changes ordering. Unresolved discussions
    // always take priority over merging; whether a merge is attempted is
    // otherwise driven solely by `merge_when_approved`.
    // -----------------------------------------------------------------

    #[test]
    fn plan_approval_skips_merge_when_discussions_are_unresolved() {
        assert_eq!(
            plan_approval(true, true),
            ApprovalPlan::SkipMergeUnresolvedDiscussions
        );
        assert_eq!(
            plan_approval(true, false),
            ApprovalPlan::SkipMergeUnresolvedDiscussions
        );
    }

    #[test]
    fn plan_approval_attempts_merge_when_resolved_and_configured() {
        assert_eq!(plan_approval(false, true), ApprovalPlan::AttemptMerge);
    }

    #[test]
    fn plan_approval_skips_merge_attempt_when_not_configured() {
        assert_eq!(
            plan_approval(false, false),
            ApprovalPlan::ApproveWithoutMerge
        );
    }

    // -----------------------------------------------------------------
    // `review_merge_request` characterization: reject-before-review gates.
    // These run before the checkout/diff/model-invoke steps, so a bad MR
    // never reaches the model.
    // -----------------------------------------------------------------

    #[test]
    fn is_generic_title_flags_known_placeholders_and_blank() {
        assert!(is_generic_title(""));
        assert!(is_generic_title("   "));
        assert!(is_generic_title("Update"));
        assert!(is_generic_title("FIX"));
        assert!(!is_generic_title("Fix null pointer in session resume"));
    }

    #[test]
    fn is_generic_description_flags_known_placeholders_and_blank() {
        assert!(is_generic_description(""));
        assert!(is_generic_description("Implementation completed."));
        assert!(is_generic_description("implementation changes"));
        assert!(!is_generic_description(
            "## Goal\nFix the retry backoff cap."
        ));
    }

    fn mr_with(title: &str, description: &str) -> MergeRequest {
        MergeRequest {
            iid: 1,
            title: title.to_string(),
            description: description.to_string(),
            source_branch: "issue-1".to_string(),
            target_branch: "main".to_string(),
            state: "opened".to_string(),
            sha: None,
            labels: None,
            has_conflicts: false,
        }
    }

    #[test]
    fn has_bad_title_or_description_flags_either_field() {
        let generic_title = mr_with("Update", "Real description");
        assert!(has_bad_title_or_description(
            &generic_title.title,
            &generic_title.description
        ));
        let generic_description = mr_with("Real title", "Implementation completed.");
        assert!(has_bad_title_or_description(
            &generic_description.title,
            &generic_description.description
        ));
        let good = mr_with("Fix session resume bug", "## Goal\nFix it.");
        assert!(!has_bad_title_or_description(
            &good.title,
            &good.description
        ));
    }

    #[test]
    fn mr_has_label_checks_present_labels_only() {
        let mut mr = mr_with("Fix session resume bug", "## Goal\nFix it.");
        mr.labels = Some(vec!["reviewer-approved".to_string()]);
        assert!(mr_labels_has(mr.labels.as_deref(), "reviewer-approved"));
        assert!(!mr_labels_has(mr.labels.as_deref(), "need-ai-worker"));
        mr.labels = None;
        assert!(!mr_labels_has(mr.labels.as_deref(), "reviewer-approved"));
    }

    #[test]
    fn extract_review_feedback_uses_structured_feedback() {
        let feedback = extract_review_feedback(
            Some("- Fix the error handling in main.go\n- Add test for edge case"),
            None,
        );
        assert!(feedback.contains("Fix the error handling"));
        assert!(feedback.contains("Add test for edge case"));
    }

    #[test]
    fn reviewer_contract_passes_the_shared_conformance_suite() {
        conformance::assert_contract::<ReviewerOutput>();
    }

    #[test]
    fn reviewer_output_accepts_approve_decision() {
        assert_eq!(
            conformance::assert_accepts::<ReviewerOutput>(json!({"decision": "approve"})),
            ReviewerOutput::Approve { summary: None }
        );
        assert_eq!(
            conformance::assert_accepts::<ReviewerOutput>(
                json!({"decision": "approve", "summary": "clean"})
            ),
            ReviewerOutput::Approve {
                summary: Some("clean".into())
            }
        );
    }

    #[test]
    fn reviewer_output_decision_is_case_insensitive() {
        assert_eq!(
            conformance::assert_accepts::<ReviewerOutput>(json!({"decision": " APPROVE "})),
            ReviewerOutput::Approve { summary: None }
        );
    }

    #[test]
    fn reviewer_output_accepts_request_changes_decision() {
        assert_eq!(
            conformance::assert_accepts::<ReviewerOutput>(json!({
                "decision": "request_changes",
                "feedback": "- Fix X"
            })),
            ReviewerOutput::RequestChanges {
                feedback: "- Fix X".into(),
                public_comment: None
            }
        );
    }

    #[test]
    fn reviewer_output_rejects_unknown_decision() {
        let error = conformance::assert_rejects::<ReviewerOutput>(json!({"decision": "maybe"}));
        assert!(error.starts_with("$.decision: expected one of"), "{error}");
    }

    #[test]
    fn reviewer_output_requires_feedback_when_requesting_changes() {
        let error = conformance::assert_rejects::<ReviewerOutput>(
            json!({"decision": "request_changes", "public_comment": "nice work"}),
        );
        assert_eq!(error, "$.feedback: required property is missing");
    }

    #[test]
    fn reviewer_output_rejects_fields_from_the_other_decision() {
        let error = conformance::assert_rejects::<ReviewerOutput>(
            json!({"decision": "approve", "feedback": "- Fix X"}),
        );
        assert!(
            error.starts_with("$.feedback: unexpected property"),
            "{error}"
        );
    }

    #[test]
    fn issue_context_header_includes_labels_for_review_only_exception() {
        let issue = Issue {
            iid: 42,
            title: "Audit large migration".to_string(),
            description: "Review a large existing MR.".to_string(),
            labels: vec!["review-only".to_string(), "priority::2".to_string()],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };

        let header = format_issue_context_header(issue.iid, &issue);

        assert!(header.contains("ISSUE LABELS: review-only, priority::2"));
        assert!(header.contains("LINKED ISSUE #42: Audit large migration"));
    }

    #[test]
    fn strip_request_changes_prefix_removes_marker_only() {
        assert_eq!(
            strip_request_changes_prefix(
                "REQUEST_CHANGES — please fix the following before review:\n\nThe MR title is bad."
            ),
            "please fix the following before review:\n\nThe MR title is bad."
        );
    }

    #[test]
    fn strip_review_boilerplate_removes_intro_line() {
        assert_eq!(
            strip_review_boilerplate(
                "Please fix the following before review:\n\nThe MR title `x` is too generic."
            ),
            "The MR title `x` is too generic."
        );
    }

    #[test]
    fn normalize_review_comment_body_strips_marker_and_boilerplate() {
        assert_eq!(
            normalize_review_comment_body(
                "REQUEST_CHANGES — please fix the following before review:\n\nThe MR title is bad."
            ),
            "The MR title is bad."
        );
    }

    #[test]
    fn extract_review_feedback_strips_request_changes_and_boilerplate() {
        // Even structured `feedback` text gets normalized in case the model
        // habitually prepends the old marker vocabulary or boilerplate.
        assert_eq!(
            extract_review_feedback(
                Some(
                    "REQUEST_CHANGES — please fix the following before review:\n\nThe MR title `x` is too generic."
                ),
                None
            ),
            "The MR title `x` is too generic."
        );
    }

    #[test]
    fn extract_review_feedback_falls_back_to_public_comment() {
        assert_eq!(
            extract_review_feedback(None, Some("Please add one integration test.")),
            "Please add one integration test."
        );
    }

    #[test]
    fn extract_review_feedback_falls_back_to_generic_message_when_empty() {
        assert_eq!(
            extract_review_feedback(None, None),
            "Please review the changes and address any issues."
        );
    }
}
