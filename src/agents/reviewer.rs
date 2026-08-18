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
use crate::core::agent::{
    InvokeOptions, ObjectSchema, OneOfSchema, Schema, StructuredOutput, compat,
};
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

impl StructuredOutput for ReviewerOutput {
    fn tool_name() -> &'static str {
        "review"
    }

    fn tool_description() -> &'static str {
        "The final merge-request review decision. Approval summaries must stay on one line."
    }

    fn schema() -> Schema {
        Schema::one_of(
            OneOfSchema::new(
                "decision",
                "Your review decision. Pick exactly one and send only that decision's fields.",
            )
            .variant(
                "approve",
                "The merge request is ready to merge as-is.",
                ObjectSchema::new().property(
                    "summary",
                    Schema::string(
                        "Optional one-line note. The posted GitLab comment is always just 'LGTM', so this is only for the log.",
                    ),
                ),
            )
            .variant(
                "request_changes",
                "The merge request needs work before it can merge.",
                ObjectSchema::new()
                    .required_property(
                        "feedback",
                        Schema::string(
                            "Specific issues that must be addressed, one bullet per line. Posted as GitLab discussion threads.",
                        ),
                    )
                    .property(
                        "public_comment",
                        Schema::string(
                            "Human-facing GitLab comment text (separate from feedback). Use for explanations, context, or recommendations that don't require code changes.",
                        ),
                    ),
            ),
        )
    }

    /// Tolerated: a decision spelled with different case or padding
    /// (`"APPROVE"`, `" approve "`).
    fn normalize(value: &mut serde_json::Value) {
        compat::normalize_tag(value, "decision");
    }
}

#[derive(Debug, Clone)]
struct ReviewerConfig {
    poll_interval_secs: u64,
    merge_when_approved: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ReviewerAgentSettings {
    #[serde(default = "default_reviewer_poll_interval")]
    poll_interval_secs: u64,
    #[serde(default = "default_merge_when_approved")]
    merge_when_approved: bool,
}

fn default_reviewer_poll_interval() -> u64 {
    120
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
        _settings: &Self::Settings,
    ) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_gitlab_repo()?;
        Ok(())
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec::polling(
            "gitlab_poll",
            Duration::from_secs(self.config.poll_interval_secs),
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
            poll_interval_secs: settings.poll_interval_secs,
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

/// One question the reviewer machine asks before it decides anything. Every
/// read the cycle performs is one of these, so a recorded trace shows the
/// observations in the order the machine needed them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerQuery {
    ShutdownRequested,
    DefaultBranch,
    MergeRequests,
    IssuePriorities,
    UnresolvedDiscussions {
        mr_iid: u64,
    },
    /// Tips of the checked-out source (`HEAD`) and of `origin/<target>`.
    BranchTips {
        target_branch: String,
    },
    LocalDiff {
        target_branch: String,
    },
}

/// The answer to one [`ReviewerQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerFact {
    ShutdownRequested(bool),
    DefaultBranch(String),
    MergeRequests(Vec<MrObservation>),
    IssuePriorities(Vec<(u64, u8)>),
    UnresolvedDiscussions(DiscussionCounts),
    BranchTips {
        source_sha: String,
        target_sha: String,
    },
    LocalDiff {
        diff_stat: String,
        changed_files: Vec<String>,
    },
}

/// Everything the machine has gathered about the MR it claimed, and which
/// the model invocation needs. Immutable once assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReviewSubject {
    mr: MrObservation,
    issue_iid: Option<u64>,
    is_need_ai_worker_mr: bool,
    diff_stat: String,
    changed_files: Vec<String>,
}

/// A single side effect (or the single model invocation) the reviewer
/// machine asks the port to perform. Every variant is one step: the machine
/// never hands over a batch, so the recorded order of these *is* the
/// reviewer's mutation order.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerAction {
    FetchRemote,
    /// Best effort, exactly like the bare `let _ = reset_hard()` it replaces.
    ResetWorktree,
    CheckoutBranch {
        branch: String,
    },
    /// Scan GitLab for a claim label this reviewer left behind.
    RecoverClaim,
    /// Release the claim held from a previous cycle, keeping the lease when
    /// the release fails so the same claim is retried next cycle.
    TryReleaseHeldClaim,
    /// Release the held claim, warning on failure and always dropping it.
    ReleaseHeldClaim,
    /// Drop our own stale claim label found on a candidate MR (best effort).
    ReleaseStaleClaimLabel {
        mr_iid: u64,
    },
    AcquireClaim {
        mr_iid: u64,
    },
    MergeTargetIntoWorktree {
        target_branch: String,
    },
    InvokeReviewModel(Box<ReviewSubject>),
    PostDiscussion {
        mr_iid: u64,
        body: String,
    },
    PostResolvedDiscussion {
        mr_iid: u64,
        body: String,
    },
    AddApprovedLabel {
        mr_iid: u64,
    },
    MergeMergeRequest {
        mr_iid: u64,
    },
}

/// Result of a claim attempt, mirroring [`ClaimAcquireOutcome`] without
/// carrying the lease itself — the lease lives in the port, which is what
/// owns claim side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimAttempt {
    Won(u64),
    Lost,
    Interrupted,
}

/// What the port reports back after executing one [`ReviewerAction`].
enum ReviewerOutcome {
    /// The action completed and has nothing to report.
    Done,
    /// The action failed. Whether that aborts the cycle or ends the review
    /// of one MR is decided by the stage that asked for it, mirroring which
    /// call sites used `?` and which were wrapped in a `match`.
    Failed(anyhow::Error),
    ClaimRecovered(Option<u64>),
    Claim(ClaimAttempt),
    LocalMerge {
        clean: bool,
    },
    Reviewed(ReviewerOutput),
}

/// The narrow surface the reviewer cycle needs. Object-safe and role-local:
/// it is the reviewer's own observe/execute vocabulary, not a stand-in for
/// the GitLab API, git, or the model backend (see [`super::claim::ClaimPort`]
/// for the same reasoning at claim granularity).
trait ReviewerPort {
    fn shutdown_requested(&self) -> bool;
    fn default_branch(&self) -> Result<String>;
    fn merge_requests(&self) -> Result<Vec<MrObservation>>;
    /// Best effort by contract: the cycle sorts by issue priority and
    /// treats an unavailable issue list as "no priorities known".
    fn issue_priorities(&self) -> Vec<(u64, u8)>;
    fn unresolved_discussions(&self, mr_iid: u64) -> Result<DiscussionCounts>;
    fn branch_tips(&self, target_branch: &str) -> Result<(String, String)>;
    fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)>;
    fn execute(&mut self, action: &ReviewerAction) -> ReviewerOutcome;
}

/// One turn of the driver loop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerStep {
    Observe(ReviewerQuery),
    Act(ReviewerAction),
    Finish,
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
    if mr.state != "opened" || !mr.in_scope(scope_label) || mr.has_label(REVIEWER_APPROVED_LABEL) {
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
// Reviewer state machine
// ---------------------------------------------------------------------------

/// Where the reviewer cycle is. Each variant names the single next
/// observation, action, or pure transition, so [`ReviewerMachine::next_step`]
/// is a function of this plus what the machine already learned — never of
/// the world.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewerStage {
    ObserveDefaultBranch,
    FetchRemote,
    ShutdownAfterFetch,
    ResetWorktree,
    CheckoutDefaultBranch,
    RecoverClaimIfUnheld,
    ReleaseClaimFromPreviousCycle,
    ObserveMergeRequests,
    ShutdownAfterMergeRequests,
    ObserveIssuePriorities,
    NextCandidate,
    ShutdownBeforeCandidate,
    ScreenCandidate,
    ReleaseStaleClaim,
    ObserveCandidateDiscussions,
    AcquireCandidateClaim,
    ShutdownAfterClaim,
    ReviewGate,
    PostGateFeedback(String),
    ReviewResetWorktree,
    CheckoutSourceBranch,
    ObserveBranchTips,
    MergeTargetIntoWorktree,
    PostConflictFeedback,
    CheckoutTargetAfterConflict,
    ObserveLocalDiff,
    InvokeReviewModel,
    CheckoutTargetAfterReview,
    ApplyReviewDecision,
    ObserveApprovalDiscussions,
    MergeMergeRequest,
    PostMergeFailedFeedback(String),
    PostApprovalComment,
    AddApprovedLabel,
    PostRequestChanges(String),
    ShutdownAfterReviewError(String),
    ReleaseClaimAfterReview,
    Finish,
}

/// How the review of the claimed MR ended: the three [`ReviewOutcome`]s,
/// plus the failure the cycle used to catch by matching on the old
/// `review_merge_request`'s `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewCompletion {
    Outcome(ReviewOutcome),
    Failed,
}

/// The reviewer's orchestration state. Holds only plain data — no GitLab
/// client, no git repo, no lease — so every decision it makes is a pure
/// function of what it has observed so far.
struct ReviewerMachine<'a> {
    agent_id: &'a str,
    scope_label: Option<&'a str>,
    merge_when_approved: bool,
    stage: ReviewerStage,
    default_branch: String,
    held_claim: Option<u64>,
    candidates: std::collections::VecDeque<MrObservation>,
    candidate: Option<MrObservation>,
    subject: Option<ReviewSubject>,
    decision: Option<ReviewerOutput>,
    completion: Option<ReviewCompletion>,
}

impl<'a> ReviewerMachine<'a> {
    fn new(agent_id: &'a str, scope_label: Option<&'a str>, merge_when_approved: bool) -> Self {
        Self {
            agent_id,
            scope_label,
            merge_when_approved,
            stage: ReviewerStage::ObserveDefaultBranch,
            default_branch: String::new(),
            held_claim: None,
            candidates: std::collections::VecDeque::new(),
            candidate: None,
            subject: None,
            decision: None,
            completion: None,
        }
    }

    /// Start a cycle already holding a claim from a previous cycle (or from
    /// startup recovery), so the machine releases it before claiming
    /// anything new.
    fn with_held_claim(mut self, mr_iid: Option<u64>) -> Self {
        self.held_claim = mr_iid;
        self
    }

    fn subject(&self) -> &ReviewSubject {
        self.subject
            .as_ref()
            .expect("review stages run only after the subject is assembled")
    }

    fn reviewed_mr_iid(&self) -> u64 {
        self.subject().mr.iid
    }

    /// The MR this cycle merged, if the review ended in a merge — the only
    /// thing the caller needs from the finished machine.
    fn merged_mr_iid(&self) -> Option<u64> {
        matches!(
            self.completion,
            Some(ReviewCompletion::Outcome(ReviewOutcome::Merged))
        )
        .then(|| self.subject().mr.iid)
    }

    /// The single next thing to do. Pure: it only resolves stages that need
    /// no port interaction (skips, sorting, gate decisions) before handing
    /// back an observation or an action.
    fn next_step(&mut self) -> ReviewerStep {
        loop {
            match &self.stage {
                ReviewerStage::ObserveDefaultBranch => {
                    return ReviewerStep::Observe(ReviewerQuery::DefaultBranch);
                }
                ReviewerStage::FetchRemote => {
                    return ReviewerStep::Act(ReviewerAction::FetchRemote);
                }
                ReviewerStage::ShutdownAfterFetch
                | ReviewerStage::ShutdownAfterMergeRequests
                | ReviewerStage::ShutdownBeforeCandidate
                | ReviewerStage::ShutdownAfterClaim
                | ReviewerStage::ShutdownAfterReviewError(_) => {
                    return ReviewerStep::Observe(ReviewerQuery::ShutdownRequested);
                }
                ReviewerStage::ResetWorktree | ReviewerStage::ReviewResetWorktree => {
                    return ReviewerStep::Act(ReviewerAction::ResetWorktree);
                }
                ReviewerStage::CheckoutDefaultBranch => {
                    return ReviewerStep::Act(ReviewerAction::CheckoutBranch {
                        branch: self.default_branch.clone(),
                    });
                }
                ReviewerStage::RecoverClaimIfUnheld => {
                    if self.held_claim.is_some() {
                        self.stage = ReviewerStage::ReleaseClaimFromPreviousCycle;
                        continue;
                    }
                    return ReviewerStep::Act(ReviewerAction::RecoverClaim);
                }
                ReviewerStage::ReleaseClaimFromPreviousCycle => {
                    if self.held_claim.is_none() {
                        self.stage = ReviewerStage::ObserveMergeRequests;
                        continue;
                    }
                    return ReviewerStep::Act(ReviewerAction::TryReleaseHeldClaim);
                }
                ReviewerStage::ObserveMergeRequests => {
                    return ReviewerStep::Observe(ReviewerQuery::MergeRequests);
                }
                ReviewerStage::ObserveIssuePriorities => {
                    return ReviewerStep::Observe(ReviewerQuery::IssuePriorities);
                }
                ReviewerStage::NextCandidate => match self.candidates.pop_front() {
                    Some(mr) => {
                        self.candidate = Some(mr);
                        self.stage = ReviewerStage::ShutdownBeforeCandidate;
                    }
                    None => self.stage = ReviewerStage::Finish,
                },
                ReviewerStage::ScreenCandidate => {
                    let mr = self
                        .candidate
                        .clone()
                        .expect("a candidate is set before it is screened");
                    match decide_candidate(&mr, self.agent_id, self.scope_label) {
                        CandidateDecision::Skip => {
                            if mr.state == "opened"
                                && mr.in_scope(self.scope_label)
                                && mr.has_label(REVIEWER_APPROVED_LABEL)
                            {
                                debug!(
                                    "{}: MR !{} already marked {}, skipping",
                                    self.agent_id, mr.iid, REVIEWER_APPROVED_LABEL
                                );
                            } else if claim::is_mr_claimed(&mr.labels) {
                                debug!(
                                    "{}: MR !{} already claimed, skipping",
                                    self.agent_id, mr.iid
                                );
                            }
                            self.stage = ReviewerStage::NextCandidate;
                        }
                        CandidateDecision::ReleaseOurStaleClaim => {
                            warn!(
                                "{}: MR !{} still has our claim label, releasing before retry",
                                self.agent_id, mr.iid
                            );
                            self.stage = ReviewerStage::ReleaseStaleClaim;
                        }
                        CandidateDecision::Screen => {
                            self.stage = ReviewerStage::ObserveCandidateDiscussions;
                        }
                    }
                }
                ReviewerStage::ReleaseStaleClaim => {
                    return ReviewerStep::Act(ReviewerAction::ReleaseStaleClaimLabel {
                        mr_iid: self.candidate_iid(),
                    });
                }
                ReviewerStage::ObserveCandidateDiscussions => {
                    return ReviewerStep::Observe(ReviewerQuery::UnresolvedDiscussions {
                        mr_iid: self.candidate_iid(),
                    });
                }
                ReviewerStage::AcquireCandidateClaim => {
                    return ReviewerStep::Act(ReviewerAction::AcquireClaim {
                        mr_iid: self.candidate_iid(),
                    });
                }
                ReviewerStage::ReviewGate => {
                    let mr = self
                        .candidate
                        .clone()
                        .expect("a candidate is claimed before it is reviewed");
                    let subject = ReviewSubject {
                        issue_iid: linked_issue_iid(&mr),
                        is_need_ai_worker_mr: mr.has_label(NEED_AI_WORKER_LABEL),
                        diff_stat: String::new(),
                        changed_files: Vec::new(),
                        mr,
                    };
                    let gate = decide_pre_review_gate(&subject.mr, &subject);
                    self.subject = Some(subject);
                    self.stage = match gate {
                        PreReviewGate::MissingIssueLink => {
                            warn!(
                                "MR !{} does not reference any issue, requesting fix",
                                self.reviewed_mr_iid()
                            );
                            ReviewerStage::PostGateFeedback(MISSING_ISSUE_LINK_BODY.to_string())
                        }
                        PreReviewGate::BadMetadata(body) => {
                            warn!(
                                "MR !{} has a generic or missing title/description, requesting fix",
                                self.reviewed_mr_iid()
                            );
                            ReviewerStage::PostGateFeedback(body)
                        }
                        PreReviewGate::Proceed => ReviewerStage::ReviewResetWorktree,
                    };
                }
                ReviewerStage::PostGateFeedback(body)
                | ReviewerStage::PostRequestChanges(body)
                | ReviewerStage::PostMergeFailedFeedback(body) => {
                    return ReviewerStep::Act(ReviewerAction::PostDiscussion {
                        mr_iid: self.reviewed_mr_iid(),
                        body: body.clone(),
                    });
                }
                ReviewerStage::CheckoutSourceBranch => {
                    return ReviewerStep::Act(ReviewerAction::CheckoutBranch {
                        branch: self.subject().mr.source_branch.clone(),
                    });
                }
                ReviewerStage::ObserveBranchTips => {
                    return ReviewerStep::Observe(ReviewerQuery::BranchTips {
                        target_branch: self.subject().mr.target_branch.clone(),
                    });
                }
                ReviewerStage::MergeTargetIntoWorktree => {
                    return ReviewerStep::Act(ReviewerAction::MergeTargetIntoWorktree {
                        target_branch: self.subject().mr.target_branch.clone(),
                    });
                }
                ReviewerStage::PostConflictFeedback => {
                    return ReviewerStep::Act(ReviewerAction::PostDiscussion {
                        mr_iid: self.reviewed_mr_iid(),
                        body: merge_conflict_body(&self.subject().mr.target_branch),
                    });
                }
                ReviewerStage::CheckoutTargetAfterConflict
                | ReviewerStage::CheckoutTargetAfterReview => {
                    return ReviewerStep::Act(ReviewerAction::CheckoutBranch {
                        branch: self.subject().mr.target_branch.clone(),
                    });
                }
                ReviewerStage::ObserveLocalDiff => {
                    return ReviewerStep::Observe(ReviewerQuery::LocalDiff {
                        target_branch: self.subject().mr.target_branch.clone(),
                    });
                }
                ReviewerStage::InvokeReviewModel => {
                    return ReviewerStep::Act(ReviewerAction::InvokeReviewModel(Box::new(
                        self.subject().clone(),
                    )));
                }
                ReviewerStage::ApplyReviewDecision => {
                    let mr_iid = self.reviewed_mr_iid();
                    match self
                        .decision
                        .take()
                        .expect("a review decision is recorded before it is applied")
                    {
                        ReviewerOutput::Approve { .. } => {
                            info!("MR !{} approved by reviewer", mr_iid);
                            self.stage = ReviewerStage::ObserveApprovalDiscussions;
                        }
                        ReviewerOutput::RequestChanges {
                            feedback,
                            public_comment,
                        } => {
                            info!("MR !{} needs changes", mr_iid);
                            let body =
                                extract_review_feedback(Some(&feedback), public_comment.as_deref());
                            self.stage = ReviewerStage::PostRequestChanges(body);
                        }
                    }
                }
                ReviewerStage::ObserveApprovalDiscussions => {
                    return ReviewerStep::Observe(ReviewerQuery::UnresolvedDiscussions {
                        mr_iid: self.reviewed_mr_iid(),
                    });
                }
                ReviewerStage::MergeMergeRequest => {
                    return ReviewerStep::Act(ReviewerAction::MergeMergeRequest {
                        mr_iid: self.reviewed_mr_iid(),
                    });
                }
                ReviewerStage::PostApprovalComment => {
                    return ReviewerStep::Act(ReviewerAction::PostResolvedDiscussion {
                        mr_iid: self.reviewed_mr_iid(),
                        body: extract_approval_message(),
                    });
                }
                ReviewerStage::AddApprovedLabel => {
                    return ReviewerStep::Act(ReviewerAction::AddApprovedLabel {
                        mr_iid: self.reviewed_mr_iid(),
                    });
                }
                ReviewerStage::ReleaseClaimAfterReview => {
                    return ReviewerStep::Act(ReviewerAction::ReleaseHeldClaim);
                }
                ReviewerStage::Finish => return ReviewerStep::Finish,
            }
        }
    }

    fn candidate_iid(&self) -> u64 {
        self.candidate
            .as_ref()
            .expect("a candidate is set before it is acted on")
            .iid
    }

    /// Feed back the answer to the observation the machine just asked for.
    /// `Err` here means the cycle itself fails, exactly where the original
    /// code used `?` on a read.
    fn apply_fact(&mut self, fact: Result<ReviewerFact>) -> Result<()> {
        match (&self.stage, fact) {
            (ReviewerStage::ObserveDefaultBranch, Ok(ReviewerFact::DefaultBranch(branch))) => {
                self.default_branch = branch;
                self.stage = ReviewerStage::FetchRemote;
            }
            (ReviewerStage::ShutdownAfterFetch, Ok(ReviewerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    ReviewerStage::Finish
                } else {
                    ReviewerStage::ResetWorktree
                };
            }
            (ReviewerStage::ObserveMergeRequests, Ok(ReviewerFact::MergeRequests(mrs))) => {
                self.candidates = mrs.into();
                self.stage = ReviewerStage::ShutdownAfterMergeRequests;
            }
            (
                ReviewerStage::ShutdownAfterMergeRequests,
                Ok(ReviewerFact::ShutdownRequested(stop)),
            ) => {
                self.stage = if stop {
                    ReviewerStage::Finish
                } else {
                    ReviewerStage::ObserveIssuePriorities
                };
            }
            (ReviewerStage::ObserveIssuePriorities, Ok(ReviewerFact::IssuePriorities(prios))) => {
                let mut candidates: Vec<MrObservation> =
                    std::mem::take(&mut self.candidates).into();
                sort_review_candidates(&mut candidates, &prios);
                self.candidates = candidates.into();
                self.stage = ReviewerStage::NextCandidate;
            }
            (ReviewerStage::ShutdownBeforeCandidate, Ok(ReviewerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    ReviewerStage::Finish
                } else {
                    ReviewerStage::ScreenCandidate
                };
            }
            (
                ReviewerStage::ObserveCandidateDiscussions,
                Ok(ReviewerFact::UnresolvedDiscussions(counts)),
            ) => {
                self.stage = if counts.any_unresolved() {
                    ReviewerStage::NextCandidate
                } else {
                    ReviewerStage::AcquireCandidateClaim
                };
            }
            (ReviewerStage::ObserveCandidateDiscussions, Err(e)) => {
                warn!(
                    "{}: Could not check discussions for MR !{}: {}, skipping this cycle",
                    self.agent_id,
                    self.candidate_iid(),
                    e
                );
                self.stage = ReviewerStage::NextCandidate;
            }
            (ReviewerStage::ShutdownAfterClaim, Ok(ReviewerFact::ShutdownRequested(stop))) => {
                if stop {
                    self.stage = ReviewerStage::ReleaseClaimAfterReview;
                } else {
                    let mr = self
                        .candidate
                        .as_ref()
                        .expect("a candidate is claimed before it is reviewed");
                    info!("{}: Reviewing MR !{}: {}", self.agent_id, mr.iid, mr.title);
                    self.stage = ReviewerStage::ReviewGate;
                }
            }
            (
                ReviewerStage::ObserveBranchTips,
                Ok(ReviewerFact::BranchTips {
                    source_sha,
                    target_sha,
                }),
            ) => {
                let mr = &self.subject().mr;
                info!(
                    "MR !{} diff: {} ({}) -> {} ({})",
                    mr.iid, mr.source_branch, source_sha, mr.target_branch, target_sha
                );
                self.stage = ReviewerStage::MergeTargetIntoWorktree;
            }
            (
                ReviewerStage::ObserveLocalDiff,
                Ok(ReviewerFact::LocalDiff {
                    diff_stat,
                    changed_files,
                }),
            ) => {
                let subject = self
                    .subject
                    .as_mut()
                    .expect("review stages run only after the subject is assembled");
                subject.diff_stat = diff_stat;
                subject.changed_files = changed_files;
                self.stage = ReviewerStage::InvokeReviewModel;
            }
            (
                ReviewerStage::ObserveApprovalDiscussions,
                Ok(ReviewerFact::UnresolvedDiscussions(counts)),
            ) => {
                let mr_iid = self.reviewed_mr_iid();
                self.stage = match plan_approval(counts.any_unresolved(), self.merge_when_approved)
                {
                    ApprovalPlan::SkipMergeUnresolvedDiscussions => {
                        warn!(
                            "MR !{} approved but has unresolved discussions, skipping merge",
                            mr_iid
                        );
                        self.complete_review(ReviewCompletion::Outcome(ReviewOutcome::NeedsChanges))
                    }
                    ApprovalPlan::AttemptMerge => ReviewerStage::MergeMergeRequest,
                    ApprovalPlan::ApproveWithoutMerge => ReviewerStage::PostApprovalComment,
                };
            }
            (ReviewerStage::ObserveApprovalDiscussions, Err(e)) => {
                let mr_iid = self.reviewed_mr_iid();
                self.stage = self.fail_review(e.context(format!(
                    "Could not verify discussions are resolved for MR !{mr_iid} before merge"
                )));
            }
            (
                ReviewerStage::ShutdownAfterReviewError(message),
                Ok(ReviewerFact::ShutdownRequested(stop)),
            ) => {
                if !stop {
                    error!(
                        "{}: Failed to review MR !{}: {}",
                        self.agent_id,
                        self.reviewed_mr_iid(),
                        message
                    );
                }
                self.stage = ReviewerStage::ReleaseClaimAfterReview;
            }
            (stage, Ok(fact)) => {
                anyhow::bail!("reviewer port answered {stage:?} with {fact:?}");
            }
            (_, Err(e)) => return Err(e),
        }
        Ok(())
    }

    /// Feed back the outcome of the action the machine just asked for.
    fn apply_outcome(&mut self, outcome: ReviewerOutcome) -> Result<()> {
        match (&self.stage, outcome) {
            (ReviewerStage::FetchRemote, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::ShutdownAfterFetch;
            }
            (ReviewerStage::ResetWorktree, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::CheckoutDefaultBranch;
            }
            (ReviewerStage::CheckoutDefaultBranch, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::RecoverClaimIfUnheld;
            }
            (ReviewerStage::RecoverClaimIfUnheld, ReviewerOutcome::ClaimRecovered(held)) => {
                self.held_claim = held;
                self.stage = ReviewerStage::ReleaseClaimFromPreviousCycle;
            }
            (ReviewerStage::ReleaseClaimFromPreviousCycle, ReviewerOutcome::Done) => {
                self.held_claim = None;
                self.stage = ReviewerStage::ObserveMergeRequests;
            }
            // A failed release keeps the lease so the same claim is retried
            // next cycle instead of being re-derived from a fresh scan.
            (ReviewerStage::ReleaseClaimFromPreviousCycle, ReviewerOutcome::Failed(_)) => {
                self.stage = ReviewerStage::Finish;
            }
            (ReviewerStage::ReleaseStaleClaim, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::ObserveCandidateDiscussions;
            }
            (ReviewerStage::AcquireCandidateClaim, ReviewerOutcome::Claim(attempt)) => {
                match attempt {
                    ClaimAttempt::Won(mr_iid) => {
                        self.held_claim = Some(mr_iid);
                        self.stage = ReviewerStage::ShutdownAfterClaim;
                    }
                    ClaimAttempt::Lost => {
                        info!(
                            "{}: Failed to claim MR !{}, skipping",
                            self.agent_id,
                            self.candidate_iid()
                        );
                        self.stage = ReviewerStage::NextCandidate;
                    }
                    ClaimAttempt::Interrupted => self.stage = ReviewerStage::Finish,
                }
            }
            (ReviewerStage::ReviewResetWorktree, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::CheckoutSourceBranch;
            }
            (ReviewerStage::CheckoutSourceBranch, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::ObserveBranchTips;
            }
            (ReviewerStage::MergeTargetIntoWorktree, ReviewerOutcome::LocalMerge { clean }) => {
                if clean {
                    self.stage = ReviewerStage::ObserveLocalDiff;
                } else {
                    let mr = &self.subject().mr;
                    warn!(
                        "MR !{} has merge conflicts with {}",
                        mr.iid, mr.target_branch
                    );
                    self.stage = ReviewerStage::PostConflictFeedback;
                }
            }
            (ReviewerStage::PostConflictFeedback, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::CheckoutTargetAfterConflict;
            }
            (ReviewerStage::CheckoutTargetAfterConflict, ReviewerOutcome::Done)
            | (ReviewerStage::PostGateFeedback(_), ReviewerOutcome::Done)
            | (ReviewerStage::PostRequestChanges(_), ReviewerOutcome::Done)
            | (ReviewerStage::PostMergeFailedFeedback(_), ReviewerOutcome::Done) => {
                self.stage =
                    self.complete_review(ReviewCompletion::Outcome(ReviewOutcome::NeedsChanges));
            }
            (ReviewerStage::InvokeReviewModel, ReviewerOutcome::Reviewed(output)) => {
                info!(
                    "{}: Reviewer agent finished MR !{}",
                    self.agent_id,
                    self.reviewed_mr_iid()
                );
                self.decision = Some(output);
                self.stage = ReviewerStage::CheckoutTargetAfterReview;
            }
            (ReviewerStage::CheckoutTargetAfterReview, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::ApplyReviewDecision;
            }
            (ReviewerStage::MergeMergeRequest, ReviewerOutcome::Done) => {
                info!("Successfully merged MR !{}", self.reviewed_mr_iid());
                self.stage = self.complete_review(ReviewCompletion::Outcome(ReviewOutcome::Merged));
            }
            (ReviewerStage::MergeMergeRequest, ReviewerOutcome::Failed(e)) => {
                warn!("Failed to merge MR !{}: {}", self.reviewed_mr_iid(), e);
                self.stage = ReviewerStage::PostMergeFailedFeedback(merge_failed_body(&e));
            }
            (ReviewerStage::PostApprovalComment, ReviewerOutcome::Done) => {
                self.stage = ReviewerStage::AddApprovedLabel;
            }
            (ReviewerStage::AddApprovedLabel, ReviewerOutcome::Done) => {
                self.stage = self.complete_review(ReviewCompletion::Outcome(
                    ReviewOutcome::ApprovedWithoutMerge,
                ));
            }
            (ReviewerStage::ReleaseClaimAfterReview, ReviewerOutcome::Done) => {
                self.held_claim = None;
                self.stage = ReviewerStage::Finish;
            }
            // Reads and writes the cycle itself performed with `?` abort it.
            (
                ReviewerStage::FetchRemote
                | ReviewerStage::CheckoutDefaultBranch
                | ReviewerStage::AcquireCandidateClaim,
                ReviewerOutcome::Failed(e),
            ) => return Err(e),
            // Anything failing inside the review of one MR ends that review,
            // just like the `Err` arm of the old `review_merge_request` match.
            (_, ReviewerOutcome::Failed(e)) => {
                self.stage = self.fail_review(e);
            }
            (stage, _) => {
                anyhow::bail!("reviewer port reported an unexpected outcome for {stage:?}");
            }
        }
        Ok(())
    }

    fn complete_review(&mut self, completion: ReviewCompletion) -> ReviewerStage {
        let mr_iid = self.reviewed_mr_iid();
        match &completion {
            ReviewCompletion::Outcome(ReviewOutcome::Merged) => {
                info!("{}: MR !{} approved and merged", self.agent_id, mr_iid);
            }
            ReviewCompletion::Outcome(ReviewOutcome::ApprovedWithoutMerge) => {
                info!(
                    "{}: MR !{} approved without merge, releasing claim",
                    self.agent_id, mr_iid
                );
            }
            ReviewCompletion::Outcome(ReviewOutcome::NeedsChanges) => {
                info!(
                    "{}: MR !{} reviewed with feedback, releasing claim",
                    self.agent_id, mr_iid
                );
            }
            ReviewCompletion::Failed => {}
        }
        self.completion = Some(completion);
        ReviewerStage::ReleaseClaimAfterReview
    }

    fn fail_review(&mut self, error: anyhow::Error) -> ReviewerStage {
        self.completion = Some(ReviewCompletion::Failed);
        ReviewerStage::ShutdownAfterReviewError(format!("{error:#}"))
    }
}

/// Ask the port one question.
fn observe_reviewer(port: &dyn ReviewerPort, query: &ReviewerQuery) -> Result<ReviewerFact> {
    Ok(match query {
        ReviewerQuery::ShutdownRequested => {
            ReviewerFact::ShutdownRequested(port.shutdown_requested())
        }
        ReviewerQuery::DefaultBranch => ReviewerFact::DefaultBranch(port.default_branch()?),
        ReviewerQuery::MergeRequests => ReviewerFact::MergeRequests(port.merge_requests()?),
        ReviewerQuery::IssuePriorities => ReviewerFact::IssuePriorities(port.issue_priorities()),
        ReviewerQuery::UnresolvedDiscussions { mr_iid } => {
            ReviewerFact::UnresolvedDiscussions(port.unresolved_discussions(*mr_iid)?)
        }
        ReviewerQuery::BranchTips { target_branch } => {
            let (source_sha, target_sha) = port.branch_tips(target_branch)?;
            ReviewerFact::BranchTips {
                source_sha,
                target_sha,
            }
        }
        ReviewerQuery::LocalDiff { target_branch } => {
            let (diff_stat, changed_files) = port.local_diff(target_branch)?;
            ReviewerFact::LocalDiff {
                diff_stat,
                changed_files,
            }
        }
    })
}

/// Run the reviewer machine to completion: observe, decide one step,
/// execute it, feed the result back.
fn drive_reviewer(machine: &mut ReviewerMachine, port: &mut dyn ReviewerPort) -> Result<()> {
    loop {
        match machine.next_step() {
            ReviewerStep::Observe(query) => {
                let fact = observe_reviewer(port, &query);
                machine.apply_fact(fact)?;
            }
            ReviewerStep::Act(action) => {
                let outcome = port.execute(&action);
                machine.apply_outcome(outcome)?;
            }
            ReviewerStep::Finish => return Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Live reviewer port
// ---------------------------------------------------------------------------

/// The reviewer port backed by the real runtime: this file's only place
/// where a reviewer decision meets git, GitLab, or the model.
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

fn required(result: Result<()>) -> ReviewerOutcome {
    match result {
        Ok(()) => ReviewerOutcome::Done,
        Err(e) => ReviewerOutcome::Failed(e),
    }
}

impl ReviewerPort for LiveReviewerPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn default_branch(&self) -> Result<String> {
        self.git_repo.get_default_branch()
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
            .map_err(|e| e.context(format!("Failed to fetch discussions for MR !{mr_iid}")))?;
        if unresolved == 0 && total > 0 {
            info!(
                "MR !{} all {} discussion(s) resolved, ready for re-review",
                mr_iid, total
            );
        }
        Ok(DiscussionCounts { unresolved, total })
    }

    fn branch_tips(&self, target_branch: &str) -> Result<(String, String)> {
        let source_sha = self.git_repo.rev_parse("HEAD")?;
        let target_sha = self
            .git_repo
            .rev_parse(&format!("origin/{target_branch}"))?;
        Ok((source_sha, target_sha))
    }

    fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)> {
        let diff_stat = self.git_repo.diff_stat_against(target_branch)?;
        let changed_files = self.git_repo.changed_files_against(target_branch)?;
        Ok((diff_stat, changed_files))
    }

    fn execute(&mut self, action: &ReviewerAction) -> ReviewerOutcome {
        match action {
            ReviewerAction::FetchRemote => required(self.git_repo.fetch()),
            ReviewerAction::ResetWorktree => {
                let _ = self.git_repo.reset_hard();
                ReviewerOutcome::Done
            }
            ReviewerAction::CheckoutBranch { branch } => {
                required(self.git_repo.checkout_remote_branch(branch))
            }
            ReviewerAction::RecoverClaim => {
                *self.claimed_mr = find_claimed_mr(self.agent_id, self.gitlab, self.scope_label);
                ReviewerOutcome::ClaimRecovered(
                    self.claimed_mr.as_ref().map(|lease| lease.resource().iid()),
                )
            }
            ReviewerAction::TryReleaseHeldClaim => {
                let Some(lease) = self.claimed_mr.as_mut() else {
                    return ReviewerOutcome::Done;
                };
                let held_iid = lease.resource().iid();
                info!(
                    "{}: Releasing held claim on MR !{} from previous cycle",
                    self.agent_id, held_iid
                );
                match lease.try_release(self.gitlab) {
                    Ok(()) => {
                        *self.claimed_mr = None;
                        ReviewerOutcome::Done
                    }
                    Err(e) => {
                        warn!(
                            "{}: Failed to release claim on MR !{}: {}, will retry next cycle",
                            self.agent_id, held_iid, e
                        );
                        ReviewerOutcome::Failed(e)
                    }
                }
            }
            ReviewerAction::ReleaseHeldClaim => {
                release_claimed_mr(self.claimed_mr, self.gitlab, self.agent_id);
                ReviewerOutcome::Done
            }
            ReviewerAction::ReleaseStaleClaimLabel { mr_iid } => {
                release_mr_claim_or_warn(self.gitlab, *mr_iid, self.agent_id);
                ReviewerOutcome::Done
            }
            ReviewerAction::AcquireClaim { mr_iid } => {
                match claim::acquire(
                    self.gitlab,
                    ClaimResource::MergeRequest(*mr_iid),
                    self.agent_id,
                    self.shutdown,
                ) {
                    Ok(ClaimAcquireOutcome::Won(lease)) => {
                        *self.claimed_mr = Some(lease);
                        ReviewerOutcome::Claim(ClaimAttempt::Won(*mr_iid))
                    }
                    Ok(ClaimAcquireOutcome::Lost) => ReviewerOutcome::Claim(ClaimAttempt::Lost),
                    Ok(ClaimAcquireOutcome::Interrupted) => {
                        ReviewerOutcome::Claim(ClaimAttempt::Interrupted)
                    }
                    Err(e) => ReviewerOutcome::Failed(e),
                }
            }
            ReviewerAction::MergeTargetIntoWorktree { target_branch } => {
                match self.git_repo.try_merge(target_branch) {
                    Ok(clean) => ReviewerOutcome::LocalMerge { clean },
                    Err(e) => ReviewerOutcome::Failed(e),
                }
            }
            ReviewerAction::InvokeReviewModel(subject) => self.invoke_review_model(subject),
            ReviewerAction::PostDiscussion { mr_iid, body } => {
                required(self.gitlab.add_mr_discussion(*mr_iid, body))
            }
            ReviewerAction::PostResolvedDiscussion { mr_iid, body } => {
                required(self.gitlab.add_resolved_mr_discussion(*mr_iid, body))
            }
            ReviewerAction::AddApprovedLabel { mr_iid } => required(
                self.gitlab
                    .add_mr_label_with_retries(*mr_iid, REVIEWER_APPROVED_LABEL),
            ),
            ReviewerAction::MergeMergeRequest { mr_iid } => required(self.gitlab.merge_mr(*mr_iid)),
        }
    }
}

impl LiveReviewerPort<'_> {
    fn invoke_review_model(&self, subject: &ReviewSubject) -> ReviewerOutcome {
        let prompt = match build_review_prompt(ReviewPromptInput {
            project_name: self.project_name,
            gitlab: self.gitlab,
            mr: &subject.mr,
            diff_stat: &subject.diff_stat,
            changed_files: &subject.changed_files,
            issue_iid: subject.issue_iid,
            is_need_ai_worker_mr: subject.is_need_ai_worker_mr,
            sessions_dir: self.sessions_dir,
        }) {
            Ok(prompt) => prompt,
            Err(e) => return ReviewerOutcome::Failed(e),
        };
        match self.model.complete_typed::<ReviewerOutput>(
            &prompt,
            &InvokeOptions {
                activity_label: Some(format!(
                    "{} reviewing MR !{}",
                    self.model.agent_id(),
                    subject.mr.iid
                )),
                ..InvokeOptions::default()
            },
        ) {
            Ok(completion) => ReviewerOutcome::Reviewed(completion.output),
            Err(e) => ReviewerOutcome::Failed(e),
        }
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
    let held_claim = claimed_mr.as_ref().map(|lease| lease.resource().iid());
    let mut machine = ReviewerMachine::new(agent_id, scope_label, config.merge_when_approved)
        .with_held_claim(held_claim);
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
    let result = drive_reviewer(&mut machine, &mut port);
    if let Some(mr_iid) = machine.merged_mr_iid() {
        merged_mrs.insert(mr_iid);
    }
    result
}

fn release_mr_claim_or_warn(gitlab: &GitLabClient, mr_iid: u64, agent_id: &str) {
    if let Err(e) = claim::release(gitlab, ClaimResource::MergeRequest(mr_iid), agent_id) {
        warn!(
            "{}: Failed to release claim on MR !{}: {}",
            agent_id, mr_iid, e
        );
    }
}

/// Release the currently-held `claimed_mr` lease (if any), warning on
/// failure. Always consumes the field, mirroring the pre-lease behavior of
/// unconditionally clearing `claimed_mr_iid` after a release attempt.
fn release_claimed_mr(claimed_mr: &mut Option<ClaimLease>, gitlab: &GitLabClient, agent_id: &str) {
    let Some(lease) = claimed_mr.take() else {
        return;
    };
    let mr_iid = lease.resource().iid();
    if let Err(e) = lease.release(gitlab) {
        warn!(
            "{}: Failed to release claim on MR !{}: {}",
            agent_id, mr_iid, e
        );
    }
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
- This is a NON-INTERACTIVE automated system
- NEVER ask the user for input, confirmation, or decisions
- NEVER prompt for additional information interactively
- Make all review decisions autonomously based on the code and information provided
- Provide clear, actionable feedback in comments (do not ask questions)
- Review the full comment history to understand what feedback was already given and addressed
- Do NOT repeat feedback that has already been addressed
- Treat comments after the original issue description as requirement updates when they clarify, narrow, expand, or supersede earlier constraints
- Do NOT request changes for outdated requirements from the original issue when later issue or MR comments clearly changed the accepted scope

GITLAB COMMENT STYLE (STRICT — for requesting changes and any posted feedback):
- Do NOT start with a long paragraph of hollow praise or thanks that only restates the diff or issue number (e.g. listing routes, files, or "aligns with #N" without adding a review decision). That adds no value and wastes the reader's time.
- Lead with what matters: **what must change before merge**, or **why you approve**. Use a direct lead-in such as `Request before merge:` or `Blocking:` when the MR must not merge until the item is addressed.
- For approvals, the posted GitLab comment is "LGTM" by default (no summary or description).
- Only request MR description updates after you have read the full `## MR description` section in the task context file (including everything after any `Closes #N` line). Do **not** treat an opening `Closes #N` as “description is only the closing line” when the rest of that section documents the work. If it already states goal, implementation approach, and verification, do not ask to expand the description.
- Public GitLab comments must use reader-facing wording only. Do NOT mention internal field names such as `decision`, `feedback`, or `public_comment`. For example, say "Please update the MR description to include the actual verification and testing performed", not "Update MR_DESCRIPTION with the actual verification/testing performed."
- Keep the public comment focused: one short optional line of genuine substance is OK, but **never** pad with a multi-sentence "thanks for the thorough coverage" preface that duplicates the diff.

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
15. Make autonomous decisions about approval or requesting changes

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

Proceed with the review autonomously. Do not ask for any user input.
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
    // Reviewer state machine driven against a recording fake port. Every
    // observation and mutation the cycle performs lands in one ordered
    // trace, so a full flow can be replayed — and its exact mutation
    // order asserted — without git, GitLab, or a model.
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
            labels: labels.map(|l| l.into_iter().map(String::from).collect()),
        }
    }

    /// A recording reviewer port. Answers are scripted per observation kind
    /// and failures are injected by naming the exact step that should fail,
    /// so tests read as "this world, then this trace".
    struct FakeReviewerPort {
        trace: RefCell<Vec<ReviewerStep>>,
        shutdown_answers: RefCell<VecDeque<bool>>,
        default_branch: String,
        merge_requests: Vec<MrObservation>,
        issue_priorities: Vec<(u64, u8)>,
        discussion_counts: RefCell<VecDeque<DiscussionCounts>>,
        local_merge_clean: bool,
        review_output: ReviewerOutput,
        claim_attempts: RefCell<VecDeque<ClaimAttempt>>,
        recoverable_claim: Option<u64>,
        held_claim: Option<u64>,
        failing_queries: Vec<ReviewerQuery>,
        failing_actions: Vec<ReviewerAction>,
    }

    impl FakeReviewerPort {
        fn new(merge_requests: Vec<MrObservation>) -> Self {
            Self {
                trace: RefCell::new(Vec::new()),
                shutdown_answers: RefCell::new(VecDeque::new()),
                default_branch: "main".to_string(),
                merge_requests,
                issue_priorities: Vec::new(),
                discussion_counts: RefCell::new(VecDeque::new()),
                local_merge_clean: true,
                review_output: ReviewerOutput::Approve { summary: None },
                claim_attempts: RefCell::new(VecDeque::new()),
                recoverable_claim: None,
                held_claim: None,
                failing_queries: Vec::new(),
                failing_actions: Vec::new(),
            }
        }

        fn deciding(mut self, output: ReviewerOutput) -> Self {
            self.review_output = output;
            self
        }

        fn with_discussion_counts(self, counts: &[(usize, usize)]) -> Self {
            self.discussion_counts.borrow_mut().extend(
                counts
                    .iter()
                    .map(|&(unresolved, total)| DiscussionCounts { unresolved, total }),
            );
            self
        }

        fn with_shutdown_answers(self, answers: &[bool]) -> Self {
            self.shutdown_answers.borrow_mut().extend(answers);
            self
        }

        fn with_claim_attempts(self, attempts: &[ClaimAttempt]) -> Self {
            self.claim_attempts.borrow_mut().extend(attempts);
            self
        }

        fn with_priorities(mut self, priorities: &[(u64, u8)]) -> Self {
            self.issue_priorities = priorities.to_vec();
            self
        }

        fn conflicting(mut self) -> Self {
            self.local_merge_clean = false;
            self
        }

        fn holding_claim(mut self, mr_iid: u64) -> Self {
            self.held_claim = Some(mr_iid);
            self
        }

        fn recovering_claim(mut self, mr_iid: u64) -> Self {
            self.recoverable_claim = Some(mr_iid);
            self
        }

        fn failing_query(mut self, query: ReviewerQuery) -> Self {
            self.failing_queries.push(query);
            self
        }

        fn failing_action(mut self, action: ReviewerAction) -> Self {
            self.failing_actions.push(action);
            self
        }

        fn record(&self, step: ReviewerStep) {
            self.trace.borrow_mut().push(step);
        }

        fn observe(&self, query: ReviewerQuery) -> Result<()> {
            self.record(ReviewerStep::Observe(query.clone()));
            if self.failing_queries.contains(&query) {
                anyhow::bail!("injected failure observing {query:?}");
            }
            Ok(())
        }

        fn next_discussion_counts(&self) -> DiscussionCounts {
            self.discussion_counts
                .borrow_mut()
                .pop_front()
                .unwrap_or(DiscussionCounts {
                    unresolved: 0,
                    total: 0,
                })
        }
    }

    impl ReviewerPort for FakeReviewerPort {
        fn shutdown_requested(&self) -> bool {
            self.record(ReviewerStep::Observe(ReviewerQuery::ShutdownRequested));
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }

        fn default_branch(&self) -> Result<String> {
            self.observe(ReviewerQuery::DefaultBranch)?;
            Ok(self.default_branch.clone())
        }

        fn merge_requests(&self) -> Result<Vec<MrObservation>> {
            self.observe(ReviewerQuery::MergeRequests)?;
            Ok(self.merge_requests.clone())
        }

        fn issue_priorities(&self) -> Vec<(u64, u8)> {
            self.record(ReviewerStep::Observe(ReviewerQuery::IssuePriorities));
            self.issue_priorities.clone()
        }

        fn unresolved_discussions(&self, mr_iid: u64) -> Result<DiscussionCounts> {
            self.observe(ReviewerQuery::UnresolvedDiscussions { mr_iid })?;
            Ok(self.next_discussion_counts())
        }

        fn branch_tips(&self, target_branch: &str) -> Result<(String, String)> {
            self.observe(ReviewerQuery::BranchTips {
                target_branch: target_branch.to_string(),
            })?;
            Ok(("source-sha".to_string(), "target-sha".to_string()))
        }

        fn local_diff(&self, target_branch: &str) -> Result<(String, Vec<String>)> {
            self.observe(ReviewerQuery::LocalDiff {
                target_branch: target_branch.to_string(),
            })?;
            Ok((
                " src/lib.rs | 2 +-".to_string(),
                vec!["src/lib.rs".to_string()],
            ))
        }

        fn execute(&mut self, action: &ReviewerAction) -> ReviewerOutcome {
            self.record(ReviewerStep::Act(action.clone()));
            if self.failing_actions.contains(action) {
                return ReviewerOutcome::Failed(anyhow::anyhow!(
                    "injected failure executing {action:?}"
                ));
            }
            match action {
                ReviewerAction::RecoverClaim => {
                    self.held_claim = self.recoverable_claim;
                    ReviewerOutcome::ClaimRecovered(self.held_claim)
                }
                ReviewerAction::TryReleaseHeldClaim | ReviewerAction::ReleaseHeldClaim => {
                    self.held_claim = None;
                    ReviewerOutcome::Done
                }
                ReviewerAction::AcquireClaim { mr_iid } => {
                    let attempt = self
                        .claim_attempts
                        .borrow_mut()
                        .pop_front()
                        .unwrap_or(ClaimAttempt::Won(*mr_iid));
                    if let ClaimAttempt::Won(iid) = attempt {
                        self.held_claim = Some(iid);
                    }
                    ReviewerOutcome::Claim(attempt)
                }
                ReviewerAction::MergeTargetIntoWorktree { .. } => ReviewerOutcome::LocalMerge {
                    clean: self.local_merge_clean,
                },
                ReviewerAction::InvokeReviewModel(_) => {
                    ReviewerOutcome::Reviewed(self.review_output.clone())
                }
                _ => ReviewerOutcome::Done,
            }
        }
    }

    struct FakeRun {
        result: Result<()>,
        trace: Vec<ReviewerStep>,
        completion: Option<ReviewCompletion>,
        merged: Option<u64>,
        held_claim: Option<u64>,
    }

    fn run_reviewer(port: &mut FakeReviewerPort, merge_when_approved: bool) -> FakeRun {
        let held = port.held_claim;
        let mut machine =
            ReviewerMachine::new(TEST_AGENT, None, merge_when_approved).with_held_claim(held);
        let result = drive_reviewer(&mut machine, port);
        FakeRun {
            result,
            trace: port.trace.borrow().clone(),
            completion: machine.completion,
            merged: machine
                .subject
                .as_ref()
                .and_then(|_| machine.merged_mr_iid()),
            held_claim: port.held_claim,
        }
    }

    fn observe(query: ReviewerQuery) -> ReviewerStep {
        ReviewerStep::Observe(query)
    }

    fn act(action: ReviewerAction) -> ReviewerStep {
        ReviewerStep::Act(action)
    }

    fn shutdown() -> ReviewerStep {
        observe(ReviewerQuery::ShutdownRequested)
    }

    fn checkout(branch: &str) -> ReviewerStep {
        act(ReviewerAction::CheckoutBranch {
            branch: branch.to_string(),
        })
    }

    /// The startup steps every cycle performs before it looks at any MR.
    fn startup_steps() -> Vec<ReviewerStep> {
        vec![
            observe(ReviewerQuery::DefaultBranch),
            act(ReviewerAction::FetchRemote),
            shutdown(),
            act(ReviewerAction::ResetWorktree),
            checkout("main"),
            act(ReviewerAction::RecoverClaim),
            observe(ReviewerQuery::MergeRequests),
            shutdown(),
            observe(ReviewerQuery::IssuePriorities),
        ]
    }

    /// The steps that take a screened candidate all the way to the model's
    /// decision, for an MR whose local merge is clean.
    fn steps_up_to_review_decision(iid: u64) -> Vec<ReviewerStep> {
        vec![
            shutdown(),
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: iid }),
            act(ReviewerAction::AcquireClaim { mr_iid: iid }),
            shutdown(),
            act(ReviewerAction::ResetWorktree),
            checkout(&format!("issue-{iid}")),
            observe(ReviewerQuery::BranchTips {
                target_branch: "main".to_string(),
            }),
            act(ReviewerAction::MergeTargetIntoWorktree {
                target_branch: "main".to_string(),
            }),
            observe(ReviewerQuery::LocalDiff {
                target_branch: "main".to_string(),
            }),
            act(ReviewerAction::InvokeReviewModel(Box::new(ReviewSubject {
                mr: observation(iid, None),
                issue_iid: Some(iid),
                is_need_ai_worker_mr: false,
                diff_stat: " src/lib.rs | 2 +-".to_string(),
                changed_files: vec!["src/lib.rs".to_string()],
            }))),
            checkout("main"),
        ]
    }

    #[test]
    fn reviewer_cycle_approves_and_merges_one_merge_request_in_order() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let mut expected = startup_steps();
        expected.extend(steps_up_to_review_decision(7));
        expected.extend([
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 }),
            act(ReviewerAction::MergeMergeRequest { mr_iid: 7 }),
            act(ReviewerAction::ReleaseHeldClaim),
        ]);
        assert_eq!(run.trace, expected);
        assert_eq!(
            run.completion,
            Some(ReviewCompletion::Outcome(ReviewOutcome::Merged))
        );
        assert_eq!(run.merged, Some(7));
        assert_eq!(run.held_claim, None);
    }

    #[test]
    fn reviewer_cycle_approves_without_merge_by_posting_lgtm_then_labeling() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]);
        let run = run_reviewer(&mut port, false);

        assert!(run.result.is_ok());
        let mut expected = startup_steps();
        expected.extend(steps_up_to_review_decision(7));
        expected.extend([
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 }),
            act(ReviewerAction::PostResolvedDiscussion {
                mr_iid: 7,
                body: "LGTM".to_string(),
            }),
            act(ReviewerAction::AddApprovedLabel { mr_iid: 7 }),
            act(ReviewerAction::ReleaseHeldClaim),
        ]);
        assert_eq!(run.trace, expected);
        assert_eq!(
            run.completion,
            Some(ReviewCompletion::Outcome(
                ReviewOutcome::ApprovedWithoutMerge
            ))
        );
        assert_eq!(run.merged, None);
    }

    #[test]
    fn reviewer_cycle_skips_the_merge_when_discussions_appear_during_review() {
        // First count (candidate screening) is clean, the re-check before
        // merging finds a new unresolved thread.
        let mut port = FakeReviewerPort::new(vec![observation(7, None)])
            .with_discussion_counts(&[(0, 0), (1, 1)]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert!(
            !run.trace
                .contains(&act(ReviewerAction::MergeMergeRequest { mr_iid: 7 }))
        );
        assert_eq!(
            run.trace.last(),
            Some(&act(ReviewerAction::ReleaseHeldClaim))
        );
        assert_eq!(
            run.completion,
            Some(ReviewCompletion::Outcome(ReviewOutcome::NeedsChanges))
        );
    }

    #[test]
    fn reviewer_cycle_posts_requested_changes_as_one_discussion() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).deciding(
            ReviewerOutput::RequestChanges {
                feedback: "- Add a test for the backoff cap".to_string(),
                public_comment: None,
            },
        );
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let mut expected = startup_steps();
        expected.extend(steps_up_to_review_decision(7));
        expected.extend([
            act(ReviewerAction::PostDiscussion {
                mr_iid: 7,
                body: "- Add a test for the backoff cap".to_string(),
            }),
            act(ReviewerAction::ReleaseHeldClaim),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn reviewer_cycle_rejects_an_mr_without_an_issue_link_before_touching_git() {
        let mut unlinked = observation(7, None);
        unlinked.description = "## Goal\nSomething unrelated.".to_string();
        unlinked.source_branch = "feature-branch".to_string();
        let mut port = FakeReviewerPort::new(vec![unlinked]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let mut expected = startup_steps();
        expected.extend([
            shutdown(),
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 }),
            act(ReviewerAction::AcquireClaim { mr_iid: 7 }),
            shutdown(),
            act(ReviewerAction::PostDiscussion {
                mr_iid: 7,
                body: MISSING_ISSUE_LINK_BODY.to_string(),
            }),
            act(ReviewerAction::ReleaseHeldClaim),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn reviewer_cycle_rejects_a_generic_title_before_invoking_the_model() {
        let mut generic = observation(7, None);
        generic.title = "Update".to_string();
        let mut port = FakeReviewerPort::new(vec![generic]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let posted = run
            .trace
            .iter()
            .filter_map(|step| match step {
                ReviewerStep::Act(ReviewerAction::PostDiscussion { body, .. }) => {
                    Some(body.clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(posted.len(), 1);
        assert!(posted[0].contains("The MR title `Update` is too generic"));
        assert!(!run.trace.iter().any(|step| matches!(
            step,
            ReviewerStep::Act(ReviewerAction::InvokeReviewModel(_))
        )));
    }

    #[test]
    fn reviewer_cycle_reports_conflicts_and_returns_the_worktree_to_the_target() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).conflicting();
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let mut expected = startup_steps();
        expected.extend([
            shutdown(),
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 }),
            act(ReviewerAction::AcquireClaim { mr_iid: 7 }),
            shutdown(),
            act(ReviewerAction::ResetWorktree),
            checkout("issue-7"),
            observe(ReviewerQuery::BranchTips {
                target_branch: "main".to_string(),
            }),
            act(ReviewerAction::MergeTargetIntoWorktree {
                target_branch: "main".to_string(),
            }),
            act(ReviewerAction::PostDiscussion {
                mr_iid: 7,
                body: merge_conflict_body("main"),
            }),
            checkout("main"),
            act(ReviewerAction::ReleaseHeldClaim),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn reviewer_cycle_explains_a_failed_merge_on_the_merge_request() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)])
            .failing_action(ReviewerAction::MergeMergeRequest { mr_iid: 7 });
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let tail = &run.trace[run.trace.len() - 3..];
        assert_eq!(
            tail[0],
            act(ReviewerAction::MergeMergeRequest { mr_iid: 7 })
        );
        match &tail[1] {
            ReviewerStep::Act(ReviewerAction::PostDiscussion { mr_iid, body }) => {
                assert_eq!(*mr_iid, 7);
                assert!(body.starts_with("Code is approved, but automatic merge failed"));
            }
            other => panic!("expected a merge-failure discussion, got {other:?}"),
        }
        assert_eq!(tail[2], act(ReviewerAction::ReleaseHeldClaim));
        assert_eq!(
            run.completion,
            Some(ReviewCompletion::Outcome(ReviewOutcome::NeedsChanges))
        );
    }

    #[test]
    fn reviewer_cycle_releases_the_claim_when_the_model_fails() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).failing_action(
            ReviewerAction::InvokeReviewModel(Box::new(ReviewSubject {
                mr: observation(7, None),
                issue_iid: Some(7),
                is_need_ai_worker_mr: false,
                diff_stat: " src/lib.rs | 2 +-".to_string(),
                changed_files: vec!["src/lib.rs".to_string()],
            })),
        );
        let run = run_reviewer(&mut port, true);

        // A failed review ends the cycle cleanly: the claim is released and
        // the poll does not fail, exactly like the old `Err` arm.
        assert!(run.result.is_ok());
        assert_eq!(run.completion, Some(ReviewCompletion::Failed));
        assert_eq!(
            &run.trace[run.trace.len() - 2..],
            &[shutdown(), act(ReviewerAction::ReleaseHeldClaim)]
        );
        assert_eq!(run.held_claim, None);
    }

    #[test]
    fn reviewer_cycle_keeps_a_held_claim_when_releasing_it_fails() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)])
            .holding_claim(4)
            .failing_action(ReviewerAction::TryReleaseHeldClaim);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                observe(ReviewerQuery::DefaultBranch),
                act(ReviewerAction::FetchRemote),
                shutdown(),
                act(ReviewerAction::ResetWorktree),
                checkout("main"),
                act(ReviewerAction::TryReleaseHeldClaim),
            ]
        );
        assert_eq!(run.held_claim, Some(4));
    }

    #[test]
    fn reviewer_cycle_releases_a_recovered_claim_before_claiming_new_work() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)]).recovering_claim(4);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let recover = run
            .trace
            .iter()
            .position(|step| step == &act(ReviewerAction::RecoverClaim))
            .expect("the cycle scans for an orphaned claim");
        assert_eq!(
            run.trace[recover + 1],
            act(ReviewerAction::TryReleaseHeldClaim)
        );
        assert_eq!(
            run.trace[recover + 2],
            observe(ReviewerQuery::MergeRequests)
        );
    }

    #[test]
    fn reviewer_cycle_stops_at_shutdown_right_after_fetching() {
        let mut port =
            FakeReviewerPort::new(vec![observation(7, None)]).with_shutdown_answers(&[true]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                observe(ReviewerQuery::DefaultBranch),
                act(ReviewerAction::FetchRemote),
                shutdown(),
            ]
        );
    }

    #[test]
    fn reviewer_cycle_releases_the_claim_when_shutdown_lands_after_claiming() {
        // Shutdown is quiet through the two startup checks and the
        // per-candidate check, then fires on the check that follows a won
        // claim.
        let mut port = FakeReviewerPort::new(vec![observation(7, None)])
            .with_shutdown_answers(&[false, false, false, true]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&act(ReviewerAction::ReleaseHeldClaim))
        );
        assert!(!run.trace.iter().any(|step| matches!(
            step,
            ReviewerStep::Act(ReviewerAction::InvokeReviewModel(_))
        )));
        assert_eq!(run.held_claim, None);
    }

    #[test]
    fn reviewer_cycle_fails_the_poll_when_a_required_git_step_fails() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None)])
            .failing_action(ReviewerAction::FetchRemote);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_err());
        assert_eq!(
            run.trace,
            vec![
                observe(ReviewerQuery::DefaultBranch),
                act(ReviewerAction::FetchRemote),
            ]
        );
    }

    #[test]
    fn reviewer_cycle_skips_a_candidate_whose_discussions_cannot_be_read() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)])
            .failing_query(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 });
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert!(
            !run.trace
                .contains(&act(ReviewerAction::AcquireClaim { mr_iid: 7 }))
        );
        assert!(
            run.trace
                .contains(&act(ReviewerAction::AcquireClaim { mr_iid: 9 }))
        );
    }

    #[test]
    fn reviewer_cycle_moves_on_when_a_claim_is_lost_and_stops_when_interrupted() {
        let mut lost = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)])
            .with_claim_attempts(&[ClaimAttempt::Lost, ClaimAttempt::Won(9)]);
        let run = run_reviewer(&mut lost, true);
        assert!(run.result.is_ok());
        assert_eq!(run.merged, Some(9));

        let mut interrupted = FakeReviewerPort::new(vec![observation(7, None)])
            .with_claim_attempts(&[ClaimAttempt::Interrupted]);
        let run = run_reviewer(&mut interrupted, true);
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&act(ReviewerAction::AcquireClaim { mr_iid: 7 }))
        );
    }

    #[test]
    fn reviewer_cycle_reviews_only_the_first_candidate_it_claims() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert!(
            !run.trace
                .contains(&act(ReviewerAction::AcquireClaim { mr_iid: 9 }))
        );
    }

    #[test]
    fn reviewer_cycle_drops_its_own_stale_claim_label_before_screening() {
        let mine = format!("claimed:{TEST_AGENT}");
        let mut port = FakeReviewerPort::new(vec![observation(7, Some(vec![mine.as_str()]))]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        let released = run
            .trace
            .iter()
            .position(|step| step == &act(ReviewerAction::ReleaseStaleClaimLabel { mr_iid: 7 }))
            .expect("our stale claim label is dropped");
        assert_eq!(
            run.trace[released + 1],
            observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: 7 })
        );
    }

    #[test]
    fn reviewer_cycle_skips_approved_and_foreign_claimed_candidates() {
        let mut port = FakeReviewerPort::new(vec![
            observation(5, Some(vec![REVIEWER_APPROVED_LABEL])),
            observation(6, Some(vec!["claimed:other-reviewer"])),
            observation(7, None),
        ]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(run.merged, Some(7));
        for skipped in [5, 6] {
            assert!(
                !run.trace.iter().any(|step| step
                    == &observe(ReviewerQuery::UnresolvedDiscussions { mr_iid: skipped }))
            );
        }
    }

    #[test]
    fn reviewer_cycle_takes_the_highest_priority_candidate_first() {
        let mut port = FakeReviewerPort::new(vec![observation(7, None), observation(9, None)])
            .with_priorities(&[(7, 3), (9, 1)]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(run.merged, Some(9));
    }

    #[test]
    fn reviewer_cycle_takes_a_need_ai_worker_candidate_before_priority() {
        let mut port = FakeReviewerPort::new(vec![
            observation(7, None),
            observation(9, Some(vec![NEED_AI_WORKER_LABEL])),
        ])
        .with_priorities(&[(7, 1), (9, 3)]);
        let run = run_reviewer(&mut port, true);

        assert!(run.result.is_ok());
        assert_eq!(run.merged, Some(9));
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
