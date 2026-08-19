use anyhow::{Context, Result};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use super::claim::{ClaimAcquireOutcome, ClaimLease, ClaimResource};
use super::{claim, issue_in_scope, labels, strip_internal_markers, with_split_parent};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient, Issue, IssueThreadNote};
use crate::agents::workspace::{GitLabAgentBootstrap, GitLabAgentRuntime, gitlab_banner};
use crate::core::agent::schema::tagged;
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, compat, structured_output};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::model::acp::capabilities::{AskAnswer, AskQuestion, CapabilityProvider};
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;

const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";

/// One sub-issue as the model described it via the `plan` tool's `sub_issues`
/// array, before the empty-title/description defensive filtering in
/// [`normalize_sub_issues`] is applied.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RawSubIssue {
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    priority: Option<u64>,
    #[serde(default)]
    depends_on: usize,
}

/// A sub-issue to create for a `split` decision. Owned by PMO — this is the
/// only role that creates sub-issues, so the type has no reason to live in
/// core.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct PmoSubIssue {
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    priority: Option<u8>,
    #[serde(default)]
    depends_on: usize,
}

/// Drop sub-issues with an empty title or description (the model
/// occasionally emits a placeholder entry), and clamp `priority` to the
/// valid 1..=3 range instead of erroring on an out-of-range value.
fn normalize_sub_issues(raw: Vec<RawSubIssue>) -> Vec<PmoSubIssue> {
    let mut old_to_new = vec![None; raw.len() + 1];
    let mut normalized = Vec::new();

    for (old_index, issue) in raw.into_iter().enumerate() {
        let title = issue.title.trim().to_string();
        let description = issue.description.trim().to_string();
        if title.is_empty() || description.is_empty() {
            continue;
        }
        let priority = issue
            .priority
            .filter(|priority| (1..=3).contains(priority))
            .map(|priority| priority as u8);
        old_to_new[old_index + 1] = Some(normalized.len() + 1);
        normalized.push(PmoSubIssue {
            title,
            description,
            priority,
            depends_on: issue.depends_on,
        });
    }

    for issue in &mut normalized {
        issue.depends_on = old_to_new
            .get(issue.depends_on)
            .copied()
            .flatten()
            .unwrap_or(0);
    }

    normalized
}

/// The PMO's typed structured-output contract: a tagged union on `decision`,
/// so each triage outcome carries exactly the fields it needs. The model
/// calls the `plan` tool; core validates the captured JSON against
/// [`PmoOutput::schema`] and deserializes it (see
/// [`AgentModel::complete_typed`]).
#[derive(Debug, Clone, PartialEq)]
enum PmoOutput {
    GuideWorker { instructions: String },
    ProposePlan { plan_text: String },
    KeepPlan,
    Split { sub_issues: Vec<RawSubIssue> },
    AlreadyDone { reason: String },
    NeedsClarification { question: String },
    WaitForDependency { dependency_issue_iid: u64 },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GuideWorkerWire {
    instructions: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposePlanWire {
    plan_text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepPlanWire {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitWire {
    sub_issues: Vec<RawSubIssue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AlreadyDoneWire {
    reason: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NeedsClarificationWire {
    question: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WaitForDependencyWire {
    dependency_issue_iid: u64,
}

/// Legacy names the model has been seen using for the dependency IID.
const DEPENDENCY_IID_ALIASES: &[&str] = &[
    "dependency_iid",
    "depends_on_issue",
    "blocked_by",
    "dependency",
];

const PMO_DECISIONS: &[&str] = &[
    "guide_worker",
    "propose_plan",
    "keep_plan",
    "split",
    "already_done",
    "needs_clarification",
    "wait_for_dependency",
];

impl<'de> Deserialize<'de> for PmoOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (decision, fields) = tagged::parts(deserializer, "decision")?;
        match decision.as_str() {
            "guide_worker" => {
                tagged::branch(fields).map(|wire: GuideWorkerWire| Self::GuideWorker {
                    instructions: wire.instructions,
                })
            }
            "propose_plan" => {
                tagged::branch(fields).map(|wire: ProposePlanWire| Self::ProposePlan {
                    plan_text: wire.plan_text,
                })
            }
            "keep_plan" => tagged::branch(fields).map(|_: KeepPlanWire| Self::KeepPlan),
            "split" => tagged::branch(fields).map(|wire: SplitWire| Self::Split {
                sub_issues: wire.sub_issues,
            }),
            "already_done" => {
                tagged::branch(fields).map(|wire: AlreadyDoneWire| Self::AlreadyDone {
                    reason: wire.reason,
                })
            }
            "needs_clarification" => tagged::branch(fields).map(|wire: NeedsClarificationWire| {
                Self::NeedsClarification {
                    question: wire.question,
                }
            }),
            "wait_for_dependency" => {
                tagged::branch(fields).map(|wire: WaitForDependencyWire| Self::WaitForDependency {
                    dependency_issue_iid: wire.dependency_issue_iid,
                })
            }
            decision => Err(serde::de::Error::unknown_variant(decision, PMO_DECISIONS)),
        }
    }
}

structured_output! {
    impl PmoOutput {
        tool_name: "plan";
        tool_description: "The PMO triage result for an issue the worker could not complete.";
        schema: one_of(
            "decision",
            "Your triage decision. Pick exactly one and send only that decision's fields.",
            {
                "guide_worker" => (
                    "The issue is workable as-is; the worker just needs one focused instruction.",
                    object({
                        required instructions: string(
                            "3-5 sentences, one clear action for the worker. Posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable."
                        ),
                    })
                ),
                "propose_plan" => (
                    "The issue needs a detailed implementation plan in its description before work proceeds or while an existing PMO plan is being refined.",
                    object({
                        required plan_text: string(
                            "A repository-informed implementation plan. For an issue without a bound merge request, provide the complete rewritten issue description. For an issue already bound to an MR, provide only the proposed changes and refinements relative to the existing issue description and prior comments; the system posts them as a new comment without replacing the description."
                        ),
                    })
                ),
                "keep_plan" => (
                    "The issue is already pmo-pending and its current plan is clear and implementation-ready. Human feedback does not require any change to the issue description or a new comment.",
                    object({})
                ),
                "split" => (
                    "The issue is too broad and must become several smaller issues.",
                    object({
                        required sub_issues: array(
                            "The sub-issues to create, in the order they should be worked.",
                            object("One sub-issue to create.", {
                                required title: string("Concise sub-issue title."),
                                required description: string(
                                    "Scope and acceptance criteria. If this sub-issue depends on another, reference it by title."
                                ),
                                optional priority: integer_enum(
                                    "Priority: 1 (critical/blocking), 2 (high/depended-on), 3 (normal/independent).",
                                    &[1, 2, 3]
                                ),
                                optional depends_on: integer(
                                    "1-based index of another sub-issue this one depends on (omit or 0 if none)."
                                ),
                            })
                        ),
                    })
                ),
                "already_done" => (
                    "The codebase already satisfies the issue, so it should be closed.",
                    object({
                        required reason: string(
                            "Why the codebase already satisfies the issue."
                        ),
                    })
                ),
                "needs_clarification" => (
                    "A human must answer something before the work can be scoped.",
                    object({
                        required question: string(
                            "Specific questions for a human. Posted as a GitLab comment."
                        ),
                    })
                ),
                "wait_for_dependency" => (
                    "The issue is blocked by another open issue and must be parked.",
                    object({
                        required dependency_issue_iid: integer(
                            "The IID (number) of the existing open issue this issue depends on and must wait for. Must be a positive integer."
                        ),
                    })
                ),
            }
        );
    /// Tolerated: a decision spelled with different case, the dependency IID
    /// under one of its legacy names or sent as `"#727"`, and a sub-issue
    /// priority outside 1-3 (dropped, so the sub-issue inherits the parent's).
        normalize(value) {
            compat::normalize_tag(value, "decision");
            for alias in DEPENDENCY_IID_ALIASES {
                compat::rename_property(value, alias, "dependency_issue_iid");
            }
            compat::normalize_iid(value, "dependency_issue_iid");
            compat::each_in_array(value, "sub_issues", |sub_issue| {
                compat::drop_integer_outside(sub_issue, "priority", &[1, 2, 3]);
            });
        }
    }
}

#[derive(Debug, Clone)]
struct PmoConfig {
    poll_interval_secs: u64,
    ask_via_gitlab: bool,
    ask_gitlab_timeout_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct PmoAgentSettings {
    #[serde(default = "default_pmo_poll_interval")]
    poll_interval_secs: u64,
    #[serde(default)]
    ask_via_gitlab: bool,
    #[serde(default = "default_ask_gitlab_timeout_secs")]
    ask_gitlab_timeout_secs: u64,
}

fn default_pmo_poll_interval() -> u64 {
    180
}

fn default_ask_gitlab_timeout_secs() -> u64 {
    600
}

/// A borrowing view over the [`GitLabAgentRuntime`] identity/path fields
/// the PMO cycle needs. Built fresh from `&GitLabAgentRuntime` at each use
/// site rather than stored, so PMO never keeps a second copy of these
/// fields — and, since it is never stored alongside the runtime it borrows
/// from, it can't become self-referential.
struct AgentState<'a> {
    sessions_dir: &'a str,
    agent_id: &'a str,
    project_name: &'a str,
}

#[derive(Serialize, Deserialize)]
struct PersistedPmoState {
    claimed_issue_iid: u64,
    #[serde(default)]
    last_seen_comment_id: u64,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &GitLabAgentRuntime) -> AgentState<'_> {
        AgentState {
            sessions_dir: &runtime.sessions_dir,
            agent_id: &runtime.agent_id,
            project_name: &runtime.project_name,
        }
    }

    fn ensure_sessions_dir(&self) -> Result<()> {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        fs::create_dir_all(ctx_dir).context("Failed to create .potlatch-context directory")?;

        Ok(())
    }

    fn state_path(&self) -> path::PathBuf {
        path::Path::new(&self.sessions_dir).join(format!("{}_state.json", &self.agent_id))
    }

    fn save_state(&self, issue_iid: u64) {
        let last_seen_comment_id = self.last_seen_comment_id(issue_iid);
        self.save_state_with_comment_cursor(issue_iid, last_seen_comment_id);
    }

    fn save_state_with_comment_cursor(&self, issue_iid: u64, last_seen_comment_id: u64) {
        let store = crate::agents::state::StateStore::new(self.state_path());
        if let Err(e) = store.save(&PersistedPmoState {
            claimed_issue_iid: issue_iid,
            last_seen_comment_id,
        }) {
            warn!("Failed to save PMO state: {}", e);
        }
    }

    fn last_seen_comment_id(&self, issue_iid: u64) -> u64 {
        let store = crate::agents::state::StateStore::new(self.state_path());
        store
            .load()
            .ok()
            .flatten()
            .filter(|state: &PersistedPmoState| state.claimed_issue_iid == issue_iid)
            .map(|state| state.last_seen_comment_id)
            .unwrap_or(0)
    }

    fn clear_state(&self) {
        let store: crate::agents::state::StateStore<PersistedPmoState> =
            crate::agents::state::StateStore::new(self.state_path());
        let _ = store.remove();
    }

    fn context_path(&self, id: u64) -> path::PathBuf {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        let file_name = format!("{}-issue-{}.md", &self.agent_id, id);
        ctx_dir.join(&file_name)
    }

    fn task_path(&self) -> String {
        let base = path::Path::new(&self.sessions_dir);
        base.join(format!("{}_pending_split.json", &self.agent_id))
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) struct PmoAgent {
    runtime: GitLabAgentRuntime,
    config: PmoConfig,
    claimed_issue: Option<ClaimLease>,
}

impl CoreAgent for PmoAgent {
    type Settings = PmoAgentSettings;

    fn name() -> &'static str {
        "pmo"
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
                let state = AgentState::from_runtime(&self.runtime);
                pmo_cycle(
                    &state,
                    &self.runtime.git_repo,
                    &self.runtime.gitlab,
                    model,
                    &mut self.claimed_issue,
                    Arc::clone(&shutdown),
                    scope,
                    &self.config,
                )
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = GitLabAgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        let settings = ctx.settings;
        let config = PmoConfig {
            poll_interval_secs: settings.poll_interval_secs,
            ask_via_gitlab: settings.ask_via_gitlab,
            ask_gitlab_timeout_secs: settings.ask_gitlab_timeout_secs,
        };
        let scope = crate::agents::scope_label_filter(&runtime.scope_label);
        let claimed_issue = {
            let state = AgentState::from_runtime(&runtime);
            state.ensure_sessions_dir()?;
            let claimed_issue = try_resume_pmo_state(&state, &runtime.gitlab, scope)
                .or_else(|| find_claimed_pmo_issue(&state, &runtime.gitlab, scope));
            if let Some(ref lease) = claimed_issue {
                let iid = lease.resource().iid();
                match (runtime.gitlab.get_issue(iid), runtime.gitlab.list_issues()) {
                    (Ok(issue), Ok(issues)) => {
                        if let Err(e) =
                            refresh_pmo_issue_context_file(&state, &runtime.gitlab, &issue, &issues)
                        {
                            warn!(
                                "{}: Could not refresh PMO context file after resuming claim on #{}: {}",
                                &state.agent_id, iid, e
                            );
                        }
                    }
                    (Err(e), _) => warn!(
                        "{}: Could not fetch issue #{} to refresh context after resume: {}",
                        &state.agent_id, iid, e
                    ),
                    (_, Err(e)) => warn!(
                        "{}: Could not list issues to refresh context after resume: {}",
                        &state.agent_id, e
                    ),
                }
            }
            claimed_issue
        };
        Ok(Self {
            runtime,
            config,
            claimed_issue,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.runtime.agent_id);
        if let Some(lease) = self.claimed_issue.take() {
            let issue_iid = lease.resource().iid();
            info!(
                "{}: Preserving claim on issue #{} for restart",
                self.runtime.agent_id, issue_iid
            );
            AgentState::from_runtime(&self.runtime).save_state(issue_iid);
            // GitLab's claim label plus the persisted state file are the
            // source of truth across restarts (see `try_resume_pmo_state`).
            lease.preserve();
        }
        info!("{}: Stopped", self.runtime.agent_id);
    }
}

// ---------------------------------------------------------------------------
// PMO role port and imperative workflow
// ---------------------------------------------------------------------------

/// Immutable issue fields used by PMO policy. The workflow deliberately does
/// not receive GitLab's mutable/API-facing issue type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PmoIssueObservation {
    iid: u64,
    title: String,
    description: String,
    labels: Vec<String>,
    state: String,
    created_at: Option<String>,
    updated_at: Option<String>,
}

impl From<&Issue> for PmoIssueObservation {
    fn from(issue: &Issue) -> Self {
        Self {
            iid: issue.iid,
            title: issue.title.clone(),
            description: issue.description.clone(),
            labels: issue.labels.clone(),
            state: issue.state.clone(),
            created_at: issue.created_at.clone(),
            updated_at: issue.updated_at.clone(),
        }
    }
}

impl PmoIssueObservation {
    fn priority(&self) -> u8 {
        gitlab::priority_from_labels(&self.labels)
    }

    fn as_issue(&self) -> Issue {
        Issue {
            iid: self.iid,
            title: self.title.clone(),
            description: self.description.clone(),
            labels: self.labels.clone(),
            state: self.state.clone(),
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PmoClaimOutcome {
    Won,
    Lost,
    Interrupted,
}

trait PmoPort {
    fn default_branch(&self) -> Result<String>;
    fn shutdown_requested(&self) -> bool;
    fn fetch_repository(&mut self) -> Result<()>;
    fn checkout_default_branch(&mut self, branch: &str) -> Result<()>;
    fn reset_worktree(&mut self) -> Result<()>;
    fn pending_split(&self) -> Result<Option<PendingSplit>>;
    fn issue(&self, issue_iid: u64) -> Option<PmoIssueObservation>;
    fn issues(&self) -> Result<Vec<PmoIssueObservation>>;
    fn new_human_comments(&self, issue_iid: u64) -> bool;
    fn current_epoch(&self) -> u64;
    fn acquire_claim(&mut self, issue_iid: u64) -> Result<PmoClaimOutcome>;
    /// Releases and drops the live lease. Some call sites intentionally ignore failure.
    fn release_claim(&mut self) -> Result<()>;
    fn save_claim_state(&mut self, issue_iid: u64);
    fn clear_claim_state(&mut self);
    fn prepare_issue_context(
        &mut self,
        issue: &PmoIssueObservation,
        all_issues: &[PmoIssueObservation],
    ) -> Result<String>;
    fn bound_merge_request(&self, issue_iid: u64) -> Result<Option<u64>>;
    fn invoke_plan(
        &mut self,
        issue: &PmoIssueObservation,
        context_path: &str,
        bound_mr_iid: Option<u64>,
    ) -> Result<PmoOutput>;
    fn update_issue_description(&mut self, issue_iid: u64, body: &str) -> Result<()>;
    fn add_issue_comment(&mut self, issue_iid: u64, body: &str) -> Result<()>;
    fn add_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()>;
    fn remove_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()>;
    fn close_issue(&mut self, issue_iid: u64) -> Result<()>;
    fn save_split_checkpoint(&mut self, pending: &PendingSplit) -> Result<()>;
    fn delete_split_checkpoint(&mut self) -> Result<()>;
    fn create_child(&mut self, title: &str, description: &str) -> Result<u64>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecisionDisposition {
    ReleaseClaim,
    KeepClaim,
}

#[derive(Debug)]
enum PmoDecisionError {
    Recoverable(anyhow::Error),
    KeepPending(anyhow::Error),
    Immediate(anyhow::Error),
}

fn issue_in_pmo_scope(issue: &PmoIssueObservation, scope_label: Option<&str>) -> bool {
    scope_label.is_none_or(|label| issue.labels.iter().any(|candidate| candidate == label))
}

fn apply_pmo_decision(
    issue: &PmoIssueObservation,
    decision: PmoOutput,
    bound_mr_iid: Option<u64>,
    scope_label: Option<&str>,
    port: &mut dyn PmoPort,
) -> std::result::Result<DecisionDisposition, PmoDecisionError> {
    let iid = issue.iid;
    match decision {
        PmoOutput::NeedsClarification { question } => {
            let question = strip_internal_markers(&clarification_question_or_default(&question));
            port.add_issue_comment(
                iid,
                &format!(
                    "**PMO needs clarification before proceeding:**\n\n{question}\n\nPlease reply to this comment with the requested information. The PMO will refine the plan based on your feedback. Remove the `pmo-pending` label when you are satisfied with the plan to let the PMO proceed."
                ),
            )
            .map_err(PmoDecisionError::Recoverable)?;
            port.add_issue_label(iid, labels::PMO_PENDING)
                .map_err(PmoDecisionError::Recoverable)?;
            Ok(DecisionDisposition::KeepClaim)
        }
        PmoOutput::ProposePlan { plan_text } => {
            let plan = strip_internal_markers(plan_text.trim());
            if plan.trim().is_empty() {
                return Err(PmoDecisionError::Recoverable(anyhow::anyhow!(
                    "PMO PROPOSE_PLAN output for issue #{iid} had no usable implementation plan; retrying later"
                )));
            }
            port.add_issue_label(iid, labels::PMO_PENDING)
                .map_err(PmoDecisionError::Recoverable)?;
            if let Some(mr_iid) = bound_mr_iid {
                port.add_issue_comment(
                    iid,
                    &format!(
                        "**PMO plan refinement for merge request !{mr_iid}:**\n\n{plan}\n\n_The existing issue description was preserved because this issue is already associated with an MR._"
                    ),
                )
                .map_err(PmoDecisionError::KeepPending)?;
            } else {
                port.update_issue_description(iid, &plan)
                    .map_err(PmoDecisionError::KeepPending)?;
            }
            Ok(DecisionDisposition::KeepClaim)
        }
        PmoOutput::KeepPlan => {
            if !issue
                .labels
                .iter()
                .any(|label| label == labels::PMO_PENDING)
            {
                return Err(PmoDecisionError::Recoverable(anyhow::anyhow!(
                    "PMO KEEP_PLAN output is only valid for an existing pmo-pending issue"
                )));
            }
            Ok(DecisionDisposition::KeepClaim)
        }
        PmoOutput::AlreadyDone { reason } => {
            let reason = strip_internal_markers(&already_done_reason_or_default(&reason));
            port.add_issue_comment(
                iid,
                &format!("**PMO: Closing — this work is already implemented.**\n\n{reason}"),
            )
            .map_err(PmoDecisionError::Recoverable)?;
            let _ = port.remove_issue_label(iid, ACTION_REQUIRED_LABEL);
            let _ = port.remove_issue_label(iid, PMO_PROCESSED_LABEL);
            port.close_issue(iid)
                .map_err(PmoDecisionError::Recoverable)?;
            Ok(DecisionDisposition::ReleaseClaim)
        }
        PmoOutput::WaitForDependency {
            dependency_issue_iid,
        } => {
            let _ = port.add_issue_label(iid, &format!("waiting-on-issue:#{dependency_issue_iid}"));
            let _ = port.remove_issue_label(iid, ACTION_REQUIRED_LABEL);
            let _ = port.remove_issue_label(iid, PMO_PROCESSED_LABEL);
            port.add_issue_comment(
                iid,
                &format!(
                    "**PMO: Parking — this issue depends on issue #{dependency_issue_iid} which is still open.**\n\nThe worker will skip this issue until #{dependency_issue_iid} is closed, then resume automatically."
                ),
            )
            .map_err(PmoDecisionError::Recoverable)?;
            Ok(DecisionDisposition::ReleaseClaim)
        }
        PmoOutput::GuideWorker { instructions } => {
            let guidance = strip_internal_markers(&guidance_or_empty(&instructions));
            if guidance.trim().is_empty() {
                return Err(PmoDecisionError::Recoverable(anyhow::anyhow!(
                    "PMO GUIDE_WORKER output for issue #{iid} had no usable guidance; retrying later"
                )));
            }
            port.add_issue_comment(iid, &format_pmo_guidance_comment(&guidance))
                .map_err(PmoDecisionError::Recoverable)?;
            port.remove_issue_label(iid, ACTION_REQUIRED_LABEL)
                .map_err(PmoDecisionError::Recoverable)?;
            let _ = port.remove_issue_label(iid, PMO_PROCESSED_LABEL);
            Ok(DecisionDisposition::ReleaseClaim)
        }
        PmoOutput::Split { sub_issues } => {
            let sub_issues = normalize_sub_issues(sub_issues);
            if sub_issues.is_empty() {
                return Err(PmoDecisionError::Recoverable(anyhow::anyhow!(
                    "PMO split output for issue #{iid} was not machine-readable; retrying later"
                )));
            }
            let mut pending = PendingSplit {
                parent_issue_iid: iid,
                parent_issue_title: issue.title.clone(),
                parent_priority: issue.priority(),
                sub_issues,
                created_issue_ids: Vec::new(),
            };
            port.save_split_checkpoint(&pending)
                .map_err(PmoDecisionError::Immediate)?;
            resume_split(&mut pending, scope_label, port).map_err(PmoDecisionError::Immediate)?;
            Ok(DecisionDisposition::ReleaseClaim)
        }
    }
}

fn process_pmo_issue(
    issue: &PmoIssueObservation,
    issues: &[PmoIssueObservation],
    scope_label: Option<&str>,
    port: &mut dyn PmoPort,
) -> std::result::Result<DecisionDisposition, PmoDecisionError> {
    let path = port
        .prepare_issue_context(issue, issues)
        .map_err(PmoDecisionError::Recoverable)?;
    let bound_mr_iid = port.bound_merge_request(issue.iid).map_err(|error| {
        if issue
            .labels
            .iter()
            .any(|label| label == labels::PMO_PENDING)
        {
            PmoDecisionError::KeepPending(error)
        } else {
            PmoDecisionError::Recoverable(error)
        }
    })?;
    let decision = port
        .invoke_plan(issue, &path, bound_mr_iid)
        .map_err(PmoDecisionError::Recoverable)?;
    apply_pmo_decision(issue, decision, bound_mr_iid, scope_label, port)
}

fn resume_split(
    pending: &mut PendingSplit,
    scope_label: Option<&str>,
    port: &mut dyn PmoPort,
) -> Result<()> {
    while pending.created_issue_ids.len() < pending.sub_issues.len() {
        let index = pending.created_issue_ids.len();
        if sub_issue_already_created(pending, index) {
            break;
        }
        let child = &pending.sub_issues[index];
        let title = if child.title.is_empty() {
            &pending.parent_issue_title
        } else {
            &child.title
        };
        let description = with_split_parent(&child.description, pending.parent_issue_iid);
        let child_iid = port.create_child(title, &description)?;
        let priority = child.priority.unwrap_or(pending.parent_priority);
        let _ = port.add_issue_label(child_iid, &gitlab::priority_label(priority));
        if let Some(label) = scope_label {
            let _ = port.add_issue_label(child_iid, label);
        }
        if child.depends_on != 0 {
            let _ = port.add_issue_label(child_iid, labels::DO_NOT_IMPLEMENT);
        }
        let mut checkpoint = pending.clone();
        checkpoint.created_issue_ids.push(child_iid);
        port.save_split_checkpoint(&checkpoint)?;
        pending.created_issue_ids.push(child_iid);
    }

    for (index, child) in pending.sub_issues.iter().enumerate() {
        if child.depends_on == 0 || child.depends_on > pending.created_issue_ids.len() {
            continue;
        }
        let child_iid = pending.created_issue_ids[index];
        let dependency_iid = pending.created_issue_ids[child.depends_on - 1];
        let _ = port.remove_issue_label(child_iid, labels::DO_NOT_IMPLEMENT);
        let _ = port.add_issue_label(child_iid, &format!("waiting-on-issue:#{dependency_iid}"));
        let _ = port.add_issue_comment(
            child_iid,
            &format!(
                "This sub-issue depends on #{dependency_iid} and will become actionable after that issue is closed."
            ),
        );
    }

    let links = pending
        .created_issue_ids
        .iter()
        .map(|iid| format!("- #{iid}"))
        .collect::<Vec<_>>()
        .join("\n");
    port.add_issue_comment(
        pending.parent_issue_iid,
        &format!(
            "This issue has been split into {} smaller sub-issues by the PMO Agent:\n\n{}\n\nEach sub-issue is designed to stay around ~500 lines of non-test code (~1500 total including tests). Auto-generated code is excluded from these limits.",
            pending.created_issue_ids.len(),
            links
        ),
    )?;
    port.remove_issue_label(pending.parent_issue_iid, ACTION_REQUIRED_LABEL)?;
    port.add_issue_label(pending.parent_issue_iid, PMO_PROCESSED_LABEL)?;
    port.delete_split_checkpoint()?;
    let _ = port.close_issue(pending.parent_issue_iid);
    Ok(())
}

fn run_pmo_maintenance(
    issues: &[PmoIssueObservation],
    scope_label: Option<&str>,
    port: &mut dyn PmoPort,
) {
    if port.shutdown_requested() {
        return;
    }
    for issue in issues {
        if issue.state == "opened"
            && issue_in_pmo_scope(issue, scope_label)
            && !issue
                .labels
                .iter()
                .any(|label| label.starts_with(gitlab::PRIORITY_LABEL_PREFIX))
        {
            let _ =
                port.add_issue_label(issue.iid, &gitlab::priority_label(gitlab::DEFAULT_PRIORITY));
        }
    }
    if port.shutdown_requested() {
        return;
    }
    let now = port.current_epoch();
    for issue in issues {
        if issue.state == "opened"
            && issue_in_pmo_scope(issue, scope_label)
            && issue
                .labels
                .iter()
                .any(|label| label == PMO_PROCESSED_LABEL)
            && issue
                .updated_at
                .as_deref()
                .and_then(parse_iso8601_to_epoch)
                .is_some_and(|updated| now.saturating_sub(updated) >= STALE_THRESHOLD_SECS)
        {
            let _ = port.add_issue_comment(
                issue.iid,
                "Closing this issue — it has been marked as `pmo-processed` for over 1 hour with no further activity.",
            );
            let _ = port.close_issue(issue.iid);
        }
    }
}

fn handle_processing_failure(
    port: &mut dyn PmoPort,
    issues: &[PmoIssueObservation],
    scope_label: Option<&str>,
) -> Result<()> {
    if port.shutdown_requested() {
        return Ok(());
    }
    port.release_claim()?;
    port.clear_claim_state();
    run_pmo_maintenance(issues, scope_label, port);
    Ok(())
}

fn run_pmo_cycle(
    agent_id: &str,
    scope_label: Option<&str>,
    held_issue_iid: Option<u64>,
    port: &mut dyn PmoPort,
) -> Result<()> {
    let default_branch = port.default_branch()?;
    port.fetch_repository()?;
    if port.checkout_default_branch(&default_branch).is_err() {
        port.reset_worktree()?;
        port.checkout_default_branch(&default_branch)
            .context("failed to checkout default branch after resetting worktree")?;
    }

    if let Some(mut pending) = port.pending_split()? {
        if let Some(parent) = port.issue(pending.parent_issue_iid) {
            let issues = port.issues()?;
            let _ = port.prepare_issue_context(&parent, &issues);
        }
        resume_split(&mut pending, scope_label, port)?;
        // Recovery can legitimately have a checkpoint but no live lease.
        if port.release_claim().is_ok() {
            port.clear_claim_state();
        }
        return Ok(());
    }

    if let Some(iid) = held_issue_iid {
        match port.issue(iid) {
            Some(issue)
                if issue_in_pmo_scope(&issue, scope_label)
                    && issue
                        .labels
                        .iter()
                        .any(|label| label == labels::PMO_PENDING) =>
            {
                let issues = port.issues()?;
                let _ = port.prepare_issue_context(&issue, &issues);
                if !port.new_human_comments(iid) {
                    return Ok(());
                }
                port.save_claim_state(iid);
                match process_pmo_issue(&issue, &issues, scope_label, port) {
                    Ok(DecisionDisposition::KeepClaim) => return Ok(()),
                    Ok(DecisionDisposition::ReleaseClaim) => {
                        port.release_claim()?;
                        port.clear_claim_state();
                        return Ok(());
                    }
                    Err(PmoDecisionError::Recoverable(error)) => {
                        warn!("{agent_id}: PMO refinement failed for issue #{iid}: {error}");
                        return handle_processing_failure(port, &issues, scope_label);
                    }
                    Err(PmoDecisionError::KeepPending(error)) => {
                        warn!(
                            "{agent_id}: PMO could not rewrite the plan for issue #{iid}; keeping it pmo-pending: {error}"
                        );
                        return Ok(());
                    }
                    Err(PmoDecisionError::Immediate(error)) => return Err(error),
                }
            }
            _ => {
                if port.release_claim().is_ok() {
                    port.clear_claim_state();
                }
            }
        }
    }

    let issues = port.issues()?;
    if port.shutdown_requested() {
        return Ok(());
    }

    for issue in &issues {
        if !should_process_issue(&issue.as_issue(), scope_label)
            || issue
                .labels
                .iter()
                .any(|label| label.starts_with("claimed:"))
        {
            continue;
        }
        if port.shutdown_requested() {
            return Ok(());
        }
        match port.acquire_claim(issue.iid)? {
            PmoClaimOutcome::Lost => continue,
            PmoClaimOutcome::Interrupted => return Ok(()),
            PmoClaimOutcome::Won => {}
        }
        if port.shutdown_requested() {
            let _ = port.release_claim();
            return Ok(());
        }

        port.save_claim_state(issue.iid);
        match process_pmo_issue(issue, &issues, scope_label, port) {
            Ok(DecisionDisposition::KeepClaim) => return Ok(()),
            Ok(DecisionDisposition::ReleaseClaim) => {
                port.release_claim()?;
                port.clear_claim_state();
                run_pmo_maintenance(&issues, scope_label, port);
                return Ok(());
            }
            Err(PmoDecisionError::Recoverable(error)) => {
                warn!(
                    "{agent_id}: required PMO processing failed for issue #{}: {error}",
                    issue.iid
                );
                return handle_processing_failure(port, &issues, scope_label);
            }
            Err(PmoDecisionError::KeepPending(error)) => {
                warn!(
                    "{agent_id}: PMO could not rewrite the plan for issue #{}; keeping it pmo-pending: {error}",
                    issue.iid
                );
                return Ok(());
            }
            Err(PmoDecisionError::Immediate(error)) => return Err(error),
        }
    }

    run_pmo_maintenance(&issues, scope_label, port);
    Ok(())
}

struct LivePmoPort<'a> {
    state: &'a AgentState<'a>,
    git_repo: &'a GitRepo,
    gitlab: &'a GitLabClient,
    model: &'a AgentModel,
    claimed_issue: &'a mut Option<ClaimLease>,
    shutdown: Arc<AtomicBool>,
    config: &'a PmoConfig,
}

impl PmoPort for LivePmoPort<'_> {
    fn default_branch(&self) -> Result<String> {
        self.git_repo.get_default_branch()
    }

    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn fetch_repository(&mut self) -> Result<()> {
        self.git_repo.fetch()
    }

    fn checkout_default_branch(&mut self, branch: &str) -> Result<()> {
        self.git_repo.checkout_remote_branch(branch)
    }

    fn reset_worktree(&mut self) -> Result<()> {
        self.git_repo.reset_hard()
    }

    fn pending_split(&self) -> Result<Option<PendingSplit>> {
        load_pending_split(&self.state.task_path())
    }

    fn issue(&self, issue_iid: u64) -> Option<PmoIssueObservation> {
        self.gitlab
            .get_issue(issue_iid)
            .ok()
            .as_ref()
            .map(PmoIssueObservation::from)
    }

    fn issues(&self) -> Result<Vec<PmoIssueObservation>> {
        Ok(self
            .gitlab
            .list_issues()?
            .iter()
            .map(PmoIssueObservation::from)
            .collect())
    }

    fn new_human_comments(&self, issue_iid: u64) -> bool {
        let Ok(comments) = self.gitlab.get_issue_comments(issue_iid) else {
            return false;
        };
        has_new_human_comments(
            &comments,
            self.state.last_seen_comment_id(issue_iid),
            self.state.agent_id,
        )
    }

    fn current_epoch(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn acquire_claim(&mut self, issue_iid: u64) -> Result<PmoClaimOutcome> {
        match claim::acquire(
            self.gitlab,
            ClaimResource::Issue(issue_iid),
            self.state.agent_id,
            self.shutdown.as_ref(),
        )? {
            ClaimAcquireOutcome::Won(lease) => {
                *self.claimed_issue = Some(lease);
                Ok(PmoClaimOutcome::Won)
            }
            ClaimAcquireOutcome::Lost => Ok(PmoClaimOutcome::Lost),
            ClaimAcquireOutcome::Interrupted => Ok(PmoClaimOutcome::Interrupted),
        }
    }

    fn release_claim(&mut self) -> Result<()> {
        let Some(lease) = self.claimed_issue.as_mut() else {
            return Ok(());
        };
        lease.try_release(self.gitlab)?;
        *self.claimed_issue = None;
        Ok(())
    }

    fn save_claim_state(&mut self, issue_iid: u64) {
        let last_seen_comment_id = self
            .gitlab
            .get_issue_comments(issue_iid)
            .ok()
            .and_then(|comments| comments.iter().map(|comment| comment.id).max())
            .unwrap_or_else(|| self.state.last_seen_comment_id(issue_iid));
        self.state
            .save_state_with_comment_cursor(issue_iid, last_seen_comment_id);
    }

    fn clear_claim_state(&mut self) {
        self.state.clear_state();
    }

    fn prepare_issue_context(
        &mut self,
        issue: &PmoIssueObservation,
        all_issues: &[PmoIssueObservation],
    ) -> Result<String> {
        let issue = issue.as_issue();
        let all_issues: Vec<Issue> = all_issues
            .iter()
            .map(PmoIssueObservation::as_issue)
            .collect();
        refresh_pmo_issue_context_file(self.state, self.gitlab, &issue, &all_issues)
    }

    fn bound_merge_request(&self, issue_iid: u64) -> Result<Option<u64>> {
        let branch_name = format!("issue-{issue_iid}");
        Ok(self
            .gitlab
            .find_mrs_by_source_branch(&branch_name)?
            .into_iter()
            .next())
    }

    fn invoke_plan(
        &mut self,
        issue: &PmoIssueObservation,
        context_path: &str,
        bound_mr_iid: Option<u64>,
    ) -> Result<PmoOutput> {
        let prompt = build_split_prompt(
            self.state,
            &issue.as_issue(),
            context_path,
            issue.priority(),
            bound_mr_iid,
        )?;
        let provider: Option<Arc<dyn CapabilityProvider>> = if self.config.ask_via_gitlab {
            let timeout = (self.config.ask_gitlab_timeout_secs > 0)
                .then(|| Duration::from_secs(self.config.ask_gitlab_timeout_secs));
            Some(Arc::new(GitLabIssueAskHandler::new(
                issue.iid,
                self.gitlab.clone(),
                Arc::clone(&self.shutdown),
                timeout,
            )))
        } else {
            None
        };
        self.model.set_capability_provider(provider);
        let result = self.model.complete_typed::<PmoOutput>(
            &prompt,
            &InvokeOptions {
                activity_label: Some(format!(
                    "{} triaging issue #{}",
                    self.state.agent_id, issue.iid
                )),
                ..InvokeOptions::default()
            },
        );
        self.model.set_capability_provider(None);
        Ok(result?.output)
    }

    fn update_issue_description(&mut self, issue_iid: u64, body: &str) -> Result<()> {
        self.gitlab.update_issue_description(issue_iid, body)
    }

    fn add_issue_comment(&mut self, issue_iid: u64, body: &str) -> Result<()> {
        self.gitlab.add_issue_comment(issue_iid, body)
    }

    fn add_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()> {
        self.gitlab.add_issue_label(issue_iid, label)
    }

    fn remove_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()> {
        self.gitlab.remove_issue_label(issue_iid, label)
    }

    fn close_issue(&mut self, issue_iid: u64) -> Result<()> {
        self.gitlab.close_issue(issue_iid)
    }

    fn save_split_checkpoint(&mut self, pending: &PendingSplit) -> Result<()> {
        save_pending_split(&self.state.task_path(), pending)
    }

    fn delete_split_checkpoint(&mut self) -> Result<()> {
        delete_pending_split(&self.state.task_path())
    }

    fn create_child(&mut self, title: &str, description: &str) -> Result<u64> {
        self.gitlab.create_issue(title, description)
    }
}

#[allow(clippy::too_many_arguments)]
fn pmo_cycle(
    state: &AgentState,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    model: &AgentModel,
    claimed_issue: &mut Option<ClaimLease>,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
    pmo_config: &PmoConfig,
) -> Result<()> {
    let held_issue_iid = claimed_issue.as_ref().map(|lease| lease.resource().iid());
    let mut port = LivePmoPort {
        state,
        git_repo,
        gitlab,
        model,
        claimed_issue,
        shutdown: Arc::clone(&shutdown),
        config: pmo_config,
    };
    run_pmo_cycle(state.agent_id, scope_label, held_issue_iid, &mut port)
}

const STALE_THRESHOLD_SECS: u64 = 3600; // 1 hour

fn parse_iso8601_to_epoch(s: &str) -> Option<u64> {
    // GitLab returns timestamps like "2026-03-18T05:30:00.000Z" or "2026-03-18T05:30:00+00:00"
    // Parse manually to avoid pulling in a datetime crate.
    let s = s.trim().trim_end_matches('Z');
    let s = if let Some(pos) = s.rfind('+') {
        // Strip timezone offset like +00:00
        if pos > 10 { &s[..pos] } else { s }
    } else if let Some(pos) = s.rfind('-') {
        // Could be timezone offset like -05:00, but also date separator
        // Only strip if it looks like a timezone (position > 10, i.e. after the date part)
        if pos > 18 { &s[..pos] } else { s }
    } else {
        s
    };
    // Strip fractional seconds
    let s = if let Some(pos) = s.find('.') {
        &s[..pos]
    } else {
        s
    };
    // Now s should be "YYYY-MM-DDTHH:MM:SS"
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 {
        return None;
    }
    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
    let time_parts: Vec<u64> = parts[1].split(':').filter_map(|p| p.parse().ok()).collect();
    if date_parts.len() != 3 || time_parts.len() != 3 {
        return None;
    }
    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, min, sec) = (time_parts[0], time_parts[1], time_parts[2]);

    // Days from year 0 to start of month (non-leap)
    const DAYS_BEFORE_MONTH: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if !(1..=12).contains(&month) {
        return None;
    }
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let leap_extra = if is_leap && month > 2 { 1 } else { 0 };

    let days_since_epoch = (year - 1970) * 365 + (year - 1969) / 4 - (year - 1901) / 100
        + (year - 1601) / 400
        + DAYS_BEFORE_MONTH[(month - 1) as usize]
        + leap_extra
        + day
        - 1;

    Some(days_since_epoch * 86400 + hour * 3600 + min * 60 + sec)
}

fn should_process_issue(issue: &Issue, scope_label: Option<&str>) -> bool {
    if !issue_in_scope(issue, scope_label) {
        return false;
    }

    // Must have action-required label
    if !issue.labels.contains(&ACTION_REQUIRED_LABEL.to_string()) {
        return false;
    }

    // Skip if already processed by PMO
    if issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
        return false;
    }

    // Skip if PMO is waiting for human clarification
    if issue.labels.contains(&labels::PMO_PENDING.to_string()) {
        return false;
    }

    // Skip drafts
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return false;
    }

    true
}

fn build_existing_issues_summary(current_iid: u64, all_issues: &[Issue]) -> String {
    let mut issues: Vec<&Issue> = all_issues
        .iter()
        .filter(|issue| issue.iid != current_iid && issue.state == "opened")
        .collect();
    issues.sort_by(|a, b| {
        a.priority()
            .cmp(&b.priority())
            .then_with(|| b.created_at.cmp(&a.created_at))
            .then_with(|| b.iid.cmp(&a.iid))
    });

    let mut lines = Vec::new();
    for issue in issues {
        let in_progress = issue.labels.contains(&"in-progress".to_string())
            || issue.labels.iter().any(|l| l.starts_with("claimed:"));
        let status = if in_progress { "IN-PROGRESS" } else { "OPEN" };
        let p = issue.priority();
        lines.push(format!(
            "- #{} [{}] (priority {p}): {}",
            issue.iid, status, issue.title
        ));
    }
    if lines.is_empty() {
        "No other open issues.".to_string()
    } else {
        lines.join("\n")
    }
}

/// Markdown for the "Comments" section of `pmo-issue-*.md` via [`GitLabClient::get_issue_comments`].
fn pmo_gitlab_comments_section(gitlab: &GitLabClient, issue_iid: u64) -> (String, usize) {
    match gitlab.get_issue_comments(issue_iid) {
        Ok(comments) => {
            let n = comments.len();
            let text = if comments.is_empty() {
                "No comments yet.".to_string()
            } else {
                comments
                    .iter()
                    .map(|c| c.format_for_prompt())
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            (text, n)
        }
        Err(e) => {
            warn!(
                "PMO: failed to fetch GitLab issue comments for #{}: {}",
                issue_iid, e
            );
            (
                format!(
                    "**ERROR: Potlatch could not load GitLab issue comments.**\n\n\
                     PMO triage may be unreliable until this works (check `glab` auth, network, and that `repo_path` is the git clone root).\n\n\
                     ```text\n{e}\n```"
                ),
                0,
            )
        }
    }
}

/// Rewrites `pmo-issue-<iid>.md` under [`BREEZE_CONTEXT_DIR`]. Call whenever PMO holds or resumes
/// work on an issue (new claim, resumed claim, pmo-pending poll, pending split).
fn refresh_pmo_issue_context_file(
    state: &AgentState,
    gitlab: &GitLabClient,
    issue: &Issue,
    all_issues: &[Issue],
) -> Result<String> {
    let (comments_text, gitlab_note_count) = pmo_gitlab_comments_section(gitlab, issue.iid);
    let existing_issues_text = build_existing_issues_summary(issue.iid, all_issues);
    let closed_mr_text = pmo_closed_mr_section(gitlab, issue.iid);

    let generated_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "# PMO Triage Context\n\nProject: {project_name}\nIssue: #{iid} {title}\n\n## Issue description\n{description}\n\n## Comments and worker feedback\n{comments}\n\n## Closed merge request context\n{closed_mr}\n\n## Existing open issues\n{existing}\n\n---\n_Potlatch: PMO refreshed this file; {gitlab_note_count} GitLab note(s) in the comments section; UNIX ts {generated_ts}._\n",
        project_name = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        description = issue.description,
        comments = comments_text,
        closed_mr = closed_mr_text,
        existing = existing_issues_text,
        gitlab_note_count = gitlab_note_count,
        generated_ts = generated_ts,
    );

    let path = state.context_path(issue.iid);
    fs::write(&path, &body)
        .with_context(|| format!("Failed to write PMO context file {}", &state.agent_id,))?;

    let context_path = fs::canonicalize(&path)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();

    Ok(context_path)
}

/// Build the "Closed merge request context" section for the PMO context file.
///
/// When a worker hands off an issue to PMO after failing to resolve reviewer
/// feedback, it closes the MR and posts a reason as an issue comment. But the
/// MR's own comment threads (reviewer feedback, worker replies) and the diff
/// carry context the issue comment alone doesn't capture. This section fetches
/// the most recent closed MR for the issue (keyed by the `issue-<iid>` branch
/// convention) and includes its comments and a truncated diff.
///
/// Returns "No closed MRs found for this issue." when there are none (the
/// common case for issues that were never assigned to a worker).
fn pmo_closed_mr_section(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let branch_name = format!("issue-{issue_iid}");
    let mr_iids = match gitlab.find_mrs_by_source_branch(&branch_name) {
        Ok(iids) => iids,
        Err(e) => {
            warn!("PMO: failed to search MRs for issue #{issue_iid} branch {branch_name}: {e}");
            return format!(
                "**ERROR: Potlatch could not search merge requests for this issue.**\n\n```\n{e}\n```"
            );
        }
    };

    // Find the most recent non-open MR (closed or merged). Open MRs are
    // excluded — they're handled by the worker/reviewer flow, not PMO.
    let mr_iid = mr_iids
        .iter()
        .find(|&&iid| match gitlab.get_merge_request(iid) {
            Ok(mr) => mr.state != "opened",
            Err(_) => false,
        })
        .copied();

    let Some(mr_iid) = mr_iid else {
        return "No closed MRs found for this issue.".to_string();
    };

    let mr = match gitlab.get_merge_request(mr_iid) {
        Ok(mr) => mr,
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid}: {e}");
            return format!(
                "**ERROR: Potlatch could not load merge request !{mr_iid}.**\n\n```\n{e}\n```"
            );
        }
    };

    let comments_text = match gitlab.get_mr_comments(mr_iid) {
        Ok(comments) if comments.is_empty() => "No MR comments.".to_string(),
        Ok(comments) => comments
            .iter()
            .map(|c| c.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n"),
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid} comments: {e}");
            format!("**ERROR: could not load MR comments: {e}**")
        }
    };

    let diff_text = match gitlab.get_merge_request_changes(mr_iid) {
        Ok(snapshot) if snapshot.patch.is_empty() => "No diff available.".to_string(),
        Ok(snapshot) => {
            let files_list = if snapshot.files.is_empty() {
                String::new()
            } else {
                format!(
                    "\n\nChanged files ({}):\n{}\n",
                    snapshot.files.len(),
                    snapshot
                        .files
                        .iter()
                        .take(50)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            };
            let patch = truncate_diff_for_pmo(&snapshot.patch);
            if snapshot.overflow {
                format!(
                    "Diff (truncated — too large for full inclusion):\n```\n{patch}\n```{files_list}\n\n_Note: the full diff exceeds the context limit. See the MR in GitLab for the complete changes._"
                )
            } else {
                format!("Diff:\n```\n{patch}\n```{files_list}")
            }
        }
        Err(e) => {
            warn!("PMO: failed to fetch MR !{mr_iid} diff: {e}");
            format!("**ERROR: could not load MR diff: {e}**")
        }
    };

    format!(
        "MR: !{mr_iid} {mr_title}\nState: {mr_state}\nSource branch: {source_branch} → Target: {target_branch}\n\n### MR comments\n{comments}\n\n### MR diff\n{diff}",
        mr_iid = mr.iid,
        mr_title = mr.title,
        mr_state = mr.state,
        source_branch = mr.source_branch,
        target_branch = mr.target_branch,
        comments = comments_text,
        diff = diff_text,
    )
}

/// Truncate a diff patch for inclusion in the PMO context file. Keeps the
/// first 8000 characters (enough to see the shape of changes without bloating
/// the context) with a marker when truncated.
fn truncate_diff_for_pmo(patch: &str) -> String {
    const MAX_DIFF_CHARS: usize = 8_000;
    if patch.len() <= MAX_DIFF_CHARS {
        return patch.to_string();
    }
    let mut end = MAX_DIFF_CHARS;
    while !patch.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!(
        "{}\n\n[...diff truncated at {} chars, {}/{} bytes shown...]",
        &patch[..end],
        MAX_DIFF_CHARS,
        end,
        patch.len()
    )
}

fn build_split_prompt(
    state: &AgentState,
    issue: &Issue,
    context_path: &str,
    parent_priority: u8,
    bound_mr_iid: Option<u64>,
) -> Result<String> {
    let planning_state = if issue
        .labels
        .iter()
        .any(|label| label == labels::PMO_PENDING)
    {
        "This issue is already pmo-pending. Human comments are refinement feedback: produce the requested plan update rather than only worker guidance or an unrelated clarification question."
    } else {
        "This issue is not currently in PMO plan refinement."
    };
    let plan_destination = match bound_mr_iid {
        Some(mr_iid) => format!(
            "This issue is already bound to merge request !{mr_iid}. Do not rewrite or restate the full issue description. Provide only the plan changes and refinements relative to the existing description and previous comments; they will be posted as a new issue comment."
        ),
        None => "This issue is not bound to a merge request. A proposed plan must be the complete rewritten issue description.".to_string(),
    };
    let prompt = format!(
        r#"You are a Project Management Office (PMO) agent responsible for triaging issues that an automated worker agent could not implement.

PROJECT: {project}

ISSUE #{iid}: {title}  _(summary only — not sufficient by itself)_

PMO PLANNING STATE:
{planning_state}
{plan_destination}

TASK CONTEXT FILE (you MUST open and read this path on disk — it has the full picture):
{context_path}

CONTEXT:
An automated worker agent attempted to implement this issue but was unable to complete it.
Potlatch wrote the path above as a markdown file: **full issue description**, **every GitLab issue comment** (including worker rejection / PMO notes), **closed merge request context** (MR comments, reviewer feedback, and diff from the worker's closed MR, when one exists), and **the list of other open issues**. That file is the authoritative written context for this triage.
- Use your **file-reading** capability on the absolute path and read it **end-to-end** before you decide the situation is unclear.
- The single line `ISSUE #…: title` in this prompt is **not** a substitute for the file; do not claim "no context" merely because you did not read the task context file.
- The **Closed merge request context** section is especially important when the worker closed an MR after failing to resolve reviewer feedback — the MR comments and diff show what the reviewer asked for and what the worker tried.
Your job is to analyze the failure reason (from the file + repo when needed) and take the appropriate action.

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You do NOT write new production code — you inspect the existing project state, then write comments and create issue descriptions as needed
- The worker agent has FULL ACCESS to shell commands (rm, mv, git, etc.) and all build/test tools
- If the worker claimed it "cannot run commands" or "cannot delete files", that is WRONG — it CAN. Instruct it clearly.
- Before giving guidance or decomposing the work, verify whether the required behavior, tests, or code already exist when the issue or comments make that plausible. If the repository already satisfies the issue, report it as complete so Potlatch can close it.

TRIAGE POLICY:
- Give focused worker guidance only for a single focused task blocked by one specific misunderstanding, wrong command, or simple technical obstacle. The guidance should be one clear action.
- Decompose broad task containers, work spanning multiple independent modules/files/components, lists of distinct tasks, or work estimated above roughly 500 non-test lines or 1500 total lines. Auto-generated code does not count. Prefer focused sub-issues over a laundry-list instruction.
- Do not invent a speculative decomposition. If the context remains too vague to define concrete sub-issues after reading the file and inspecting the repository as needed, provide your best current plan and the specific questions a human must answer.
- If the repository already fully implements the requested behavior, report that fact instead of guiding or decomposing.
- Park work behind an existing open issue only when that issue is a real build-order prerequisite. Merely related or parallel work is not a dependency.
- Produce one triage result; do not combine alternatives.

PLAN PROPOSAL POLICY:
- Propose a plan when the issue is a coherent task and the repository provides enough information to write an implementation-ready plan, but the current issue description lacks the concrete scope and steps needed for reliable execution.
- When no MR is bound, a proposed plan is the complete rewritten issue description. Preserve the original requirements and add repository-informed code areas, ordered implementation steps, acceptance criteria, tests, assumptions, and open questions.
- When an MR is already bound, preserve the issue description and propose only changes relative to the existing description and previous comments. The changes are posted in a new comment; do not produce a full rewritten description.
- During pmo-pending refinement, if the current plan is already clear and implementation-ready and the latest human comment does not require a real plan change, explicitly keep the current plan unchanged. In that case Potlatch must not rewrite the description or add another comment.
- Proposing or refining a plan keeps the issue pmo-pending so human comments can trigger another refinement cycle.
- Do not use worker guidance as a substitute for a missing plan. Use clarification only when missing human information prevents you from writing a useful plan.

DECOMPOSITION POLICY:
- The parent is a task container and will be closed after its sub-issues are created.
- Keep each sub-issue focused around the same approximate size limits as above.
- The parent priority is {parent_priority}. Give blockers, security fixes, and shared prerequisites higher priority than independent leaf work.
- Record dependencies only for real build order. Potlatch parks dependent sub-issues until their prerequisites close.
- Review the existing-open-issues section before proposing sub-issues. Never duplicate or substantially overlap existing work.
- If an existing open but inactive issue covers part of the work, reference it instead of creating a duplicate. If an in-progress issue covers it, skip that part. If existing issues cover everything, guide the worker to those issues instead of creating more.

CLARIFICATION POLICY:
- Ask for human input only after reading the entire task context and inspecting the repository when needed.
- Include the best current scope and approach with the specific unresolved questions. On later triage cycles, refine that plan from the human replies.
- The pending label remains until a human removes it; during refinement, state your recommendation and questions rather than taking an unapproved final action.

INSTRUCTIONS:
1. Open and read the **entire** TASK CONTEXT FILE at the absolute path above (description, GitLab comments, closed MR context, existing issues). Do this first.
2. From that file, read the issue description and **all** comments — especially the worker's rejection reason. If there is a **Closed merge request context** section, read the MR comments and diff to understand what the reviewer asked for and what the worker tried.
3. Review the EXISTING OPEN ISSUES section in that same file to see what is already tracked.
4. Verify whether the repository already satisfies the issue.
5. Apply the triage, decomposition, dependency, duplicate, and clarification policies above.
Proceed with analyzing the issue autonomously.
"#,
        project = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        planning_state = planning_state,
        plan_destination = plan_destination,
        context_path = context_path,
        parent_priority = parent_priority,
    );

    Ok(prompt)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PendingSplit {
    parent_issue_iid: u64,
    parent_issue_title: String,
    #[serde(default)]
    parent_priority: u8,
    sub_issues: Vec<PmoSubIssue>,
    created_issue_ids: Vec<u64>,
}

// --- Field extractors (typed `PmoOutput` fields only) ---

fn already_done_reason_or_default(reason: &str) -> String {
    let reason = reason.trim();
    if reason.is_empty() {
        "The work described in this issue is already fully implemented in the codebase.".to_string()
    } else {
        reason.to_string()
    }
}

fn clarification_question_or_default(question: &str) -> String {
    let question = question.trim();
    if question.is_empty() {
        "The PMO agent could not determine how to proceed with this issue. Please provide more details about the expected scope and acceptance criteria.".to_string()
    } else {
        question.to_string()
    }
}

fn guidance_or_empty(instructions: &str) -> String {
    let instructions = instructions.trim();
    if instructions.is_empty() {
        String::new()
    } else {
        cap_guidance_length(instructions)
    }
}

/// Whether a human added a comment after the PMO's persisted observation
/// cursor. PMO-generated planning comments are ignored even when GitLab
/// reports the shared service-account username instead of the agent id.
fn has_new_human_comments(
    comments: &[gitlab::Comment],
    last_seen_comment_id: u64,
    pmo_agent_id: &str,
) -> bool {
    comments.iter().any(|comment| {
        comment.id > last_seen_comment_id
            && !comment.author.is_empty()
            && comment.author != pmo_agent_id
            && !comment.body.starts_with("**PMO needs clarification")
            && !comment.body.starts_with("**PMO plan refinement")
    })
}

/// Cap guidance length — keep it brief and actionable. Truncates at the last
/// sentence boundary within the limit, falling back to an ellipsis suffix.
fn cap_guidance_length(raw: &str) -> String {
    const MAX: usize = 500;
    if raw.len() <= MAX {
        return raw.to_string();
    }
    let truncated: String = raw.chars().take(MAX).collect();
    if let Some(last_period) = truncated.rfind('.') {
        truncated[..=last_period].to_string()
    } else {
        format!("{}...", truncated)
    }
}

/// Format the worker-facing PMO guidance as a GitLab issue comment. The
/// guidance body is posted as plain text (no markers) under a header that
/// gives reviewers context. Both humans and the worker agent read it from
/// the comment stream.
fn format_pmo_guidance_comment(guidance: &str) -> String {
    let trimmed = guidance.trim();
    format!("**PMO guidance for the worker agent:**\n\n{trimmed}")
}

fn save_pending_split(path: &str, pending: &PendingSplit) -> Result<()> {
    crate::agents::state::StateStore::new(path)
        .save(pending)
        .context("Failed to save pending split file")?;
    info!(
        "PMO: Saved pending split for issue #{} with {} sub-issues to {}",
        pending.parent_issue_iid,
        pending.sub_issues.len(),
        path
    );
    Ok(())
}

fn load_pending_split(path: &str) -> Result<Option<PendingSplit>> {
    // Strict: unlike the resumable claim state below, a corrupt or
    // unsupported pending-split checkpoint is quarantined (never losing
    // bytes) but still surfaced as an error — resuming a split from bad
    // checkpoint data would risk re-creating or losing sub-issues.
    crate::agents::state::StateStore::new(path)
        .load()
        .context("Failed to load pending split file")
}

fn delete_pending_split(path: &str) -> Result<()> {
    let store: crate::agents::state::StateStore<PendingSplit> =
        crate::agents::state::StateStore::new(path);
    let existed = store.path().exists();
    store
        .remove()
        .context("Failed to delete pending split file")?;
    if existed {
        info!("PMO: Deleted pending split file {}", path);
    }
    Ok(())
}

fn try_resume_pmo_state(
    state: &AgentState,
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) -> Option<ClaimLease> {
    // Tolerant: invalid persisted PMO claim state has historically been
    // ignored, warn and treat as "nothing to resume" rather than failing.
    let store: crate::agents::state::StateStore<PersistedPmoState> =
        crate::agents::state::StateStore::new(state.state_path());
    let persisted = match store.load() {
        Ok(Some(persisted)) => persisted,
        Ok(None) => return None,
        Err(error) => {
            warn!(
                "{}: Failed to load persisted PMO state: {:#}",
                &state.agent_id, error
            );
            return None;
        }
    };
    let issue_iid = persisted.claimed_issue_iid;

    match gitlab.get_issue(issue_iid) {
        Ok(issue) => {
            if issue.state != "opened" {
                info!(
                    "{}: Previously claimed issue #{} is {}, discarding state",
                    &state.agent_id, issue_iid, issue.state
                );

                state.clear_state();

                return None;
            }
            if !issue_in_scope(&issue, scope_label) {
                info!(
                    "{}: Resumed issue #{} outside scope label {:?}, discarding state",
                    &state.agent_id, issue_iid, scope_label
                );

                state.clear_state();
                return None;
            }
            // Only ever resumed once the claim label has just been
            // verified live on GitLab above — never from the persisted
            // state file alone.
            let Some(lease) = ClaimLease::recover(
                ClaimResource::Issue(issue_iid),
                state.agent_id,
                &issue.labels,
            ) else {
                info!(
                    "{}: Claim label missing from issue #{}, discarding state",
                    &state.agent_id, issue_iid
                );

                state.clear_state();

                return None;
            };

            info!("{}: Resumed claim on issue #{}", &state.agent_id, issue_iid);

            Some(lease)
        }
        Err(e) => {
            warn!(
                "{}: Failed to verify issue #{}: {}, retaining state and scanning live claims",
                &state.agent_id, issue_iid, e
            );
            find_claimed_pmo_issue(state, gitlab, scope_label)
        }
    }
}

fn find_claimed_pmo_issue(
    state: &AgentState,
    gitlab: &GitLabClient,
    _scope_label: Option<&str>,
) -> Option<ClaimLease> {
    let mut issues = match gitlab.list_issues() {
        Ok(issues) => issues,
        Err(error) => {
            warn!(
                "{}: Failed to scan for an orphaned PMO claim: {}",
                state.agent_id, error
            );
            return None;
        }
    };
    issues.sort_by_key(|issue| issue.iid);
    let mut recovered = Vec::new();
    for issue in issues {
        if issue.state != "opened" {
            continue;
        }
        if let Some(lease) = ClaimLease::recover(
            ClaimResource::Issue(issue.iid),
            state.agent_id,
            &issue.labels,
        ) {
            recovered.push(lease);
        }
    }
    let (lease, recovered) = split_recovered_claims(recovered)?;
    let issue_iid = lease.resource().iid();
    for mut extra in recovered {
        let iid = extra.resource().iid();
        if let Err(error) = extra.try_release(gitlab) {
            warn!(
                "{}: Failed to release extra orphaned claim on issue #{}: {}",
                state.agent_id, iid, error
            );
            extra.preserve();
        }
    }
    info!(
        "{}: Recovered orphaned claim on issue #{} from GitLab",
        state.agent_id, issue_iid
    );
    state.save_state(issue_iid);
    Some(lease)
}

fn split_recovered_claims(mut claims: Vec<ClaimLease>) -> Option<(ClaimLease, Vec<ClaimLease>)> {
    if claims.is_empty() {
        return None;
    }
    let primary = claims.remove(0);
    Some((primary, claims))
}

/// Checkpoint resume decision: index `index` of `pending.sub_issues` was
/// already created in a prior (possibly crashed) run when it is covered by
/// the persisted `created_issue_ids` prefix. Pure — characterizes the
/// split's checkpoint-then-create resumability without any GitLab call:
/// `resume_split` always skips exactly this prefix, in order, and never
/// re-creates a sub-issue the checkpoint already recorded.
fn sub_issue_already_created(pending: &PendingSplit, index: usize) -> bool {
    index < pending.created_issue_ids.len()
}

// ---------------------------------------------------------------------------
// GitLab-based ask question handler (CapabilityProvider::ask)
// ---------------------------------------------------------------------------

const ASK_POLL_INTERVAL: Duration = Duration::from_secs(4);

fn new_ask_id() -> String {
    let mut r = rand::rng();
    format!("{:016x}{:016x}", r.random::<u64>(), r.random::<u64>())
}

fn ask_marker_snippet(ask_id: &str) -> String {
    format!("<!-- potlatch-pmo-acp-ask:{ask_id} -->")
}

/// Renders the question's choices for the GitLab comment body.
fn format_choices_for_comment(question: &AskQuestion) -> String {
    if question.choices.is_empty() {
        return "_Reply with an option id, or a number like `0` for the first choice._\n"
            .to_string();
    }
    let mut lines = Vec::new();
    for (i, c) in question.choices.iter().enumerate() {
        if c.id.is_empty() {
            lines.push(format!("{}. {}", i + 1, c.label));
        } else {
            lines.push(format!("{}. `{}` — {}", i + 1, c.id, c.label));
        }
    }
    lines.join("\n")
}

fn build_issue_comment(ask_id: &str, question: &AskQuestion) -> String {
    let q = &question.text;
    let opts = format_choices_for_comment(question);
    let marker = ask_marker_snippet(ask_id);
    format!(
        "{marker}\n\n\
         **Agent question**\n\n\
         {q}\n\n\
         **Choices**\n\n\
         {opts}\n\n\
         ---\n\n\
         **Reply to this comment** (use GitLab’s *Reply* on this note so your answer stays in this thread). \
         Your reply text is the answer — usually one line: an option id, a number (`0` = first choice), or a short answer.\n\n\
         The `{}` label is set until Potlatch forwards your reply to the running agent.",
        labels::PMO_PENDING
    )
}

/// First non-empty line of the reply body (trimmed), or empty string if none.
fn reply_body_as_choice(body: &str) -> String {
    body.trim()
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

fn find_root_note<'a>(notes: &'a [IssueThreadNote], ask_id: &str) -> Option<&'a IssueThreadNote> {
    let needle = ask_marker_snippet(ask_id);
    notes.iter().find(|n| n.body.contains(&needle))
}

/// First **direct** reply in the same discussion as `root` (GitLab thread), excluding system notes and new ask posts.
fn pick_direct_thread_reply<'a>(
    notes: &'a [IssueThreadNote],
    root: &'a IssueThreadNote,
) -> Option<&'a IssueThreadNote> {
    let mut candidates: Vec<&IssueThreadNote> = notes
        .iter()
        .filter(|n| {
            if n.id == root.id || n.system {
                return false;
            }
            if n.body.contains("potlatch-pmo-acp-ask:") {
                return false;
            }
            same_discussion_or_sequential_fallback(n, root)
        })
        .collect();
    candidates.sort_by_key(|n| n.id);
    candidates.into_iter().next()
}

fn same_discussion_or_sequential_fallback(n: &IssueThreadNote, root: &IssueThreadNote) -> bool {
    match (&root.discussion_id, &n.discussion_id) {
        (Some(rd), Some(nd)) => rd == nd,
        _ => n.id > root.id,
    }
}

/// Resolve a human reply to a neutral [`AskAnswer`].
///
/// - empty → [`AskAnswer::Auto`] (let the vendor pick its default)
/// - numeric index → [`AskAnswer::Choice`] with the matching choice id (1-based; `0` = first)
/// - matching choice id → [`AskAnswer::Choice`]
/// - otherwise → [`AskAnswer::FreeText`]
fn resolve_reply_to_answer(question: &AskQuestion, choice_raw: &str) -> AskAnswer {
    let choice_trim = choice_raw.trim();
    if choice_trim.is_empty() {
        return AskAnswer::Auto;
    }
    if let Ok(idx) = choice_trim.parse::<usize>() {
        let selected = if idx == 0 {
            question.choices.first()
        } else {
            question.choices.get(idx - 1)
        };
        if let Some(opt) = selected
            && !opt.id.is_empty()
        {
            return AskAnswer::Choice(opt.id.clone());
        }
        // No matching choice by index: fall through to free-text.
    }
    for opt in &question.choices {
        if !opt.id.is_empty() && opt.id == choice_trim {
            return AskAnswer::Choice(opt.id.clone());
        }
    }
    warn!(
        target: "potlatch::pmo_ask",
        choice = %choice_trim,
        "PMO thread reply did not match a listed option; echoing as free text"
    );
    AskAnswer::FreeText(choice_trim.to_string())
}

fn clear_pmo_pending_label(gitlab: &GitLabClient, issue_iid: u64) {
    let _ = gitlab.remove_issue_label(issue_iid, labels::PMO_PENDING);
}

/// Posts the question note, then **blocks** until a **thread reply** arrives (same process only).
pub struct GitLabIssueAskHandler {
    issue_iid: u64,
    gitlab: GitLabClient,
    shutdown: Arc<AtomicBool>,
    wait_deadline: Option<Instant>,
}

impl GitLabIssueAskHandler {
    pub fn new(
        issue_iid: u64,
        gitlab: GitLabClient,
        shutdown: Arc<AtomicBool>,
        timeout: Option<Duration>,
    ) -> Self {
        let wait_deadline = timeout.map(|d| Instant::now() + d);
        Self {
            issue_iid,
            gitlab,
            shutdown,
            wait_deadline,
        }
    }

    fn finish_with_reply(&self, question: &AskQuestion, reply: &IssueThreadNote) -> AskAnswer {
        let choice = reply_body_as_choice(&reply.body);
        let answer = resolve_reply_to_answer(question, &choice);
        clear_pmo_pending_label(&self.gitlab, self.issue_iid);
        info!(
            target: "potlatch::pmo_ask",
            issue_iid = self.issue_iid,
            note_id = reply.id,
            author = %reply.author_username(),
            "Using direct thread reply as ask answer"
        );
        answer
    }
}

impl CapabilityProvider for GitLabIssueAskHandler {
    fn ask(&self, question: &AskQuestion) -> AskAnswer {
        let ask_id = new_ask_id();
        let comment = build_issue_comment(&ask_id, question);

        if let Err(e) = self.gitlab.add_issue_comment(self.issue_iid, &comment) {
            warn!(
                target: "potlatch::pmo_ask",
                err = %e,
                "failed to post ask question to GitLab; using automatic answer"
            );
            return AskAnswer::Auto;
        }

        if let Err(e) = self
            .gitlab
            .add_issue_label(self.issue_iid, labels::PMO_PENDING)
        {
            warn!(
                target: "potlatch::pmo_ask",
                err = %e,
                "failed to add pmo-pending for ask question"
            );
        }

        let notes = match self.gitlab.get_issue_thread_notes(self.issue_iid) {
            Ok(n) => n,
            Err(e) => {
                warn!(
                    target: "potlatch::pmo_ask",
                    err = %e,
                    "failed to list thread notes after posting ask question"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return AskAnswer::Auto;
            }
        };

        let Some(root) = find_root_note(&notes, &ask_id) else {
            warn!(
                target: "potlatch::pmo_ask",
                ask_id = %ask_id,
                "could not find posted ask note by marker; using automatic answer"
            );
            clear_pmo_pending_label(&self.gitlab, self.issue_iid);
            return AskAnswer::Auto;
        };

        let root_note_id = root.id;

        info!(
            target: "potlatch::pmo_ask",
            issue_iid = self.issue_iid,
            root_note_id = root.id,
            ask_id = %ask_id,
            "Posted ask question; waiting for direct thread reply"
        );
        eprintln!(
            "potlatch PMO: Posted question on issue #{} — **Reply to that GitLab comment** (thread). Waiting…",
            self.issue_iid
        );

        loop {
            if let Some(dl) = self.wait_deadline
                && Instant::now() >= dl
            {
                eprintln!(
                    "potlatch PMO: GitLab thread wait timed out on issue #{}; using automatic answer.",
                    self.issue_iid
                );
                let _ = self.gitlab.add_issue_comment(
                    self.issue_iid,
                    "**PMO:** Timed out waiting for a **reply** to the question comment; proceeding with an automatic choice.",
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return AskAnswer::Auto;
            }

            if self.shutdown.load(Ordering::SeqCst) {
                warn!(
                    target: "potlatch::pmo_ask",
                    "shutdown during ask wait; using automatic answer"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return AskAnswer::Auto;
            }

            thread::sleep(ASK_POLL_INTERVAL);

            let notes = match self.gitlab.get_issue_thread_notes(self.issue_iid) {
                Ok(n) => n,
                Err(e) => {
                    warn!(target: "potlatch::pmo_ask", err = %e, "poll thread notes failed");
                    continue;
                }
            };

            let Some(root) = notes.iter().find(|n| n.id == root_note_id) else {
                warn!(
                    target: "potlatch::pmo_ask",
                    root_note_id,
                    "root note disappeared during wait"
                );
                clear_pmo_pending_label(&self.gitlab, self.issue_iid);
                return AskAnswer::Auto;
            };

            if let Some(reply) = pick_direct_thread_reply(&notes, root) {
                let choice = reply_body_as_choice(&reply.body);
                if choice.is_empty() {
                    continue;
                }
                return self.finish_with_reply(question, reply);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::conformance;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    struct FakePmoPort {
        trace: RefCell<Vec<String>>,
        shutdown: RefCell<VecDeque<bool>>,
        pending: Option<PendingSplit>,
        issues: Vec<PmoIssueObservation>,
        plan: PmoOutput,
        next_iid: Cell<u64>,
        failures: Vec<&'static str>,
        claim_outcomes: RefCell<VecDeque<PmoClaimOutcome>>,
        checkout_failures: Cell<usize>,
        new_comments: bool,
        bound_mr_iid: Option<u64>,
        descriptions: Vec<(u64, String)>,
        comments: Vec<(u64, String)>,
        children: Vec<(String, String)>,
    }

    impl FakePmoPort {
        fn split_plan() -> PmoOutput {
            PmoOutput::Split {
                sub_issues: vec![
                    RawSubIssue {
                        title: "Foundation".into(),
                        description: "Build the foundation".into(),
                        priority: Some(1),
                        depends_on: 0,
                    },
                    RawSubIssue {
                        title: "Integration".into(),
                        description: "Integrate the foundation".into(),
                        priority: None,
                        depends_on: 1,
                    },
                ],
            }
        }

        fn issue(iid: u64) -> PmoIssueObservation {
            PmoIssueObservation {
                iid,
                title: "Broad parent".into(),
                description: "Split this work".into(),
                labels: vec![
                    ACTION_REQUIRED_LABEL.into(),
                    "priority::2".into(),
                    "scope::test".into(),
                ],
                state: "opened".into(),
                created_at: None,
                updated_at: None,
            }
        }

        fn successful() -> Self {
            Self {
                trace: RefCell::new(Vec::new()),
                shutdown: RefCell::new(VecDeque::new()),
                pending: None,
                issues: vec![Self::issue(10)],
                plan: Self::split_plan(),
                next_iid: Cell::new(101),
                failures: Vec::new(),
                claim_outcomes: RefCell::new(VecDeque::new()),
                checkout_failures: Cell::new(0),
                new_comments: true,
                bound_mr_iid: None,
                descriptions: Vec::new(),
                comments: Vec::new(),
                children: Vec::new(),
            }
        }

        fn record(&self, event: impl Into<String>) {
            self.trace.borrow_mut().push(event.into());
        }

        fn fail(&self, name: &'static str) -> Result<()> {
            if self.failures.contains(&name) {
                anyhow::bail!("injected {name} failure");
            }
            Ok(())
        }

        fn child_label_failure(&self, issue_iid: u64) -> bool {
            issue_iid != 10 && self.failures.contains(&"child_label")
        }
    }

    impl PmoPort for FakePmoPort {
        fn default_branch(&self) -> Result<String> {
            self.record("observe:default_branch");
            Ok("main".into())
        }

        fn shutdown_requested(&self) -> bool {
            self.record("observe:shutdown");
            self.shutdown.borrow_mut().pop_front().unwrap_or(false)
        }

        fn fetch_repository(&mut self) -> Result<()> {
            self.record("act:fetch");
            self.fail("fetch")
        }

        fn checkout_default_branch(&mut self, branch: &str) -> Result<()> {
            self.record(format!("act:checkout:{branch}"));
            let remaining = self.checkout_failures.get();
            if remaining > 0 {
                self.checkout_failures.set(remaining - 1);
                anyhow::bail!("injected checkout failure");
            }
            self.fail("checkout")
        }

        fn reset_worktree(&mut self) -> Result<()> {
            self.record("act:reset");
            self.fail("reset")
        }

        fn pending_split(&self) -> Result<Option<PendingSplit>> {
            self.record("observe:pending");
            Ok(self.pending.clone())
        }

        fn issue(&self, issue_iid: u64) -> Option<PmoIssueObservation> {
            self.record(format!("observe:issue:{issue_iid}"));
            self.issues
                .iter()
                .find(|issue| issue.iid == issue_iid)
                .cloned()
        }

        fn issues(&self) -> Result<Vec<PmoIssueObservation>> {
            self.record("observe:issues");
            Ok(self.issues.clone())
        }

        fn new_human_comments(&self, issue_iid: u64) -> bool {
            self.record(format!("observe:comments:{issue_iid}"));
            self.new_comments
        }

        fn current_epoch(&self) -> u64 {
            self.record("observe:epoch");
            1_800_000_000
        }

        fn acquire_claim(&mut self, issue_iid: u64) -> Result<PmoClaimOutcome> {
            self.record(format!("act:claim:{issue_iid}"));
            self.fail("claim")?;
            Ok(self
                .claim_outcomes
                .borrow_mut()
                .pop_front()
                .unwrap_or(PmoClaimOutcome::Won))
        }

        fn release_claim(&mut self) -> Result<()> {
            self.record("act:release");
            self.fail("release")
        }

        fn save_claim_state(&mut self, issue_iid: u64) {
            self.record(format!("act:save_claim:{issue_iid}"));
        }

        fn clear_claim_state(&mut self) {
            self.record("act:clear_claim");
        }

        fn prepare_issue_context(
            &mut self,
            issue: &PmoIssueObservation,
            _all_issues: &[PmoIssueObservation],
        ) -> Result<String> {
            self.record(format!("act:context:{}", issue.iid));
            self.fail("context")?;
            Ok("/sessions/pmo-issue.md".into())
        }

        fn bound_merge_request(&self, issue_iid: u64) -> Result<Option<u64>> {
            self.record(format!("observe:bound_mr:{issue_iid}"));
            self.fail("bound_mr")?;
            Ok(self.bound_mr_iid)
        }

        fn invoke_plan(
            &mut self,
            issue: &PmoIssueObservation,
            _context_path: &str,
            _bound_mr_iid: Option<u64>,
        ) -> Result<PmoOutput> {
            self.record(format!("act:model:{}", issue.iid));
            self.fail("model")?;
            Ok(self.plan.clone())
        }

        fn update_issue_description(&mut self, issue_iid: u64, body: &str) -> Result<()> {
            self.record(format!("act:description:{issue_iid}"));
            self.descriptions.push((issue_iid, body.to_string()));
            self.fail("description")
        }

        fn add_issue_comment(&mut self, issue_iid: u64, body: &str) -> Result<()> {
            let kind = if body.starts_with("This issue has been split") {
                "parent_comment"
            } else if body.starts_with("This sub-issue depends") {
                "dependency_comment"
            } else {
                "comment"
            };
            self.record(format!("act:{kind}:{issue_iid}"));
            self.comments.push((issue_iid, body.to_string()));
            self.fail(kind)
        }

        fn add_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()> {
            self.record(format!("act:add:{issue_iid}:{label}"));
            if self.child_label_failure(issue_iid) {
                anyhow::bail!("injected child label failure");
            }
            self.fail("add_label")
        }

        fn remove_issue_label(&mut self, issue_iid: u64, label: &str) -> Result<()> {
            self.record(format!("act:remove:{issue_iid}:{label}"));
            if self.child_label_failure(issue_iid) {
                anyhow::bail!("injected child label failure");
            }
            self.fail("remove_label")
        }

        fn close_issue(&mut self, issue_iid: u64) -> Result<()> {
            self.record(format!("act:close:{issue_iid}"));
            self.fail("close")
        }

        fn save_split_checkpoint(&mut self, pending: &PendingSplit) -> Result<()> {
            self.record(format!("act:checkpoint:{:?}", pending.created_issue_ids));
            self.fail("checkpoint")
        }

        fn delete_split_checkpoint(&mut self) -> Result<()> {
            self.record("act:delete_checkpoint");
            self.fail("delete_checkpoint")
        }

        fn create_child(&mut self, title: &str, description: &str) -> Result<u64> {
            self.record(format!("act:create:{title}"));
            self.fail("create")?;
            self.children
                .push((title.to_string(), description.to_string()));
            let iid = self.next_iid.get();
            self.next_iid.set(iid + 1);
            Ok(iid)
        }
    }

    fn run_fake(port: &mut FakePmoPort, held_issue_iid: Option<u64>) -> Result<()> {
        run_pmo_cycle("pmo-0", Some("scope::test"), held_issue_iid, port)
    }

    #[test]
    fn pmo_cycle_records_full_split_order_and_generated_iids() {
        let mut port = FakePmoPort::successful();
        run_fake(&mut port, None).unwrap();
        assert_eq!(
            *port.trace.borrow(),
            vec![
                "observe:default_branch",
                "act:fetch",
                "act:checkout:main",
                "observe:pending",
                "observe:issues",
                "observe:shutdown",
                "observe:shutdown",
                "act:claim:10",
                "observe:shutdown",
                "act:save_claim:10",
                "act:context:10",
                "observe:bound_mr:10",
                "act:model:10",
                "act:checkpoint:[]",
                "act:create:Foundation",
                "act:add:101:priority::1",
                "act:add:101:scope::test",
                "act:checkpoint:[101]",
                "act:create:Integration",
                "act:add:102:priority::2",
                "act:add:102:scope::test",
                "act:add:102:do-not-implement",
                "act:checkpoint:[101, 102]",
                "act:remove:102:do-not-implement",
                "act:add:102:waiting-on-issue:#101",
                "act:dependency_comment:102",
                "act:parent_comment:10",
                "act:remove:10:action-required",
                "act:add:10:pmo-processed",
                "act:delete_checkpoint",
                "act:close:10",
                "act:release",
                "act:clear_claim",
                "observe:shutdown",
                "observe:shutdown",
                "observe:epoch",
            ]
        );
        assert_eq!(port.children.len(), 2);
        assert!(port.children.iter().all(|(_, description)| {
            description.contains("This issue was split from parent issue #10.")
        }));
    }

    #[test]
    fn pmo_cycle_resumes_from_generated_iid_checkpoint_without_recreating_prefix() {
        let mut port = FakePmoPort::successful();
        let mut pending = match FakePmoPort::split_plan() {
            PmoOutput::Split { sub_issues } => PendingSplit {
                parent_issue_iid: 10,
                parent_issue_title: "Broad parent".into(),
                parent_priority: 2,
                sub_issues: normalize_sub_issues(sub_issues),
                created_issue_ids: vec![501],
            },
            _ => unreachable!(),
        };
        pending.sub_issues[1].depends_on = 1;
        port.pending = Some(pending);
        port.next_iid.set(502);

        run_fake(&mut port, Some(10)).unwrap();
        let trace = port.trace.borrow();
        assert!(!trace.iter().any(|event| event == "act:create:Foundation"));
        assert!(trace.iter().any(|event| event == "act:create:Integration"));
        assert!(
            trace
                .iter()
                .any(|event| event == "act:add:502:waiting-on-issue:#501")
        );
        assert_eq!(trace.last().map(String::as_str), Some("act:clear_claim"));
    }

    #[test]
    fn pmo_cycle_aborts_required_split_failures_but_continues_best_effort_labels() {
        let mut required = FakePmoPort::successful();
        required.failures = vec!["checkpoint"];
        assert!(run_fake(&mut required, None).is_err());

        let mut best_effort = FakePmoPort::successful();
        best_effort.failures = vec!["child_label"];
        run_fake(&mut best_effort, None).unwrap();
        assert!(
            best_effort
                .trace
                .borrow()
                .contains(&"act:delete_checkpoint".to_string())
        );
    }

    #[test]
    fn pmo_cycle_honors_shutdown_before_claim_and_after_claim() {
        let mut before = FakePmoPort::successful();
        before.shutdown.borrow_mut().push_back(true);
        run_fake(&mut before, None).unwrap();
        assert!(
            !before
                .trace
                .borrow()
                .iter()
                .any(|event| event.starts_with("act:claim"))
        );

        let mut after = FakePmoPort::successful();
        after.shutdown.borrow_mut().extend([false, false, true]);
        run_fake(&mut after, None).unwrap();
        assert!(after.trace.borrow().contains(&"act:release".to_string()));
        assert!(!after.trace.borrow().contains(&"act:model:10".to_string()));
    }

    #[test]
    fn pmo_decision_branches_preserve_order_policy_and_payloads() {
        let issue = FakePmoPort::issue(10);

        let mut clarification = FakePmoPort::successful();
        let disposition = apply_pmo_decision(
            &issue,
            PmoOutput::NeedsClarification {
                question: "Which API?".into(),
            },
            None,
            Some("scope::test"),
            &mut clarification,
        )
        .unwrap();
        assert_eq!(disposition, DecisionDisposition::KeepClaim);
        assert_eq!(
            *clarification.trace.borrow(),
            vec!["act:comment:10", "act:add:10:pmo-pending",]
        );
        assert!(clarification.descriptions.is_empty());
        assert!(clarification.comments[0].1.contains("Which API?"));

        let mut planning = FakePmoPort::successful();
        let disposition = apply_pmo_decision(
            &issue,
            PmoOutput::ProposePlan {
                plan_text: "## Implementation plan\n\nUpdate the parser and its tests.".into(),
            },
            None,
            Some("scope::test"),
            &mut planning,
        )
        .unwrap();
        assert_eq!(disposition, DecisionDisposition::KeepClaim);
        assert_eq!(
            *planning.trace.borrow(),
            vec!["act:add:10:pmo-pending", "act:description:10",]
        );
        assert_eq!(
            planning.descriptions,
            vec![(
                10,
                "## Implementation plan\n\nUpdate the parser and its tests.".into()
            )]
        );
        assert!(planning.comments.is_empty());

        let mut unchanged_issue = issue.clone();
        unchanged_issue.labels.push(labels::PMO_PENDING.into());
        let mut unchanged = FakePmoPort::successful();
        let disposition = apply_pmo_decision(
            &unchanged_issue,
            PmoOutput::KeepPlan,
            None,
            Some("scope::test"),
            &mut unchanged,
        )
        .unwrap();
        assert_eq!(disposition, DecisionDisposition::KeepClaim);
        assert!(unchanged.trace.borrow().is_empty());
        assert!(unchanged.comments.is_empty());
        assert!(unchanged.descriptions.is_empty());

        let mut invalid_unchanged = FakePmoPort::successful();
        assert!(matches!(
            apply_pmo_decision(
                &issue,
                PmoOutput::KeepPlan,
                None,
                Some("scope::test"),
                &mut invalid_unchanged,
            ),
            Err(PmoDecisionError::Recoverable(_))
        ));

        let mut done = FakePmoPort::successful();
        apply_pmo_decision(
            &issue,
            PmoOutput::AlreadyDone {
                reason: "Already shipped".into(),
            },
            None,
            Some("scope::test"),
            &mut done,
        )
        .unwrap();
        assert_eq!(
            *done.trace.borrow(),
            vec![
                "act:comment:10",
                "act:remove:10:action-required",
                "act:remove:10:pmo-processed",
                "act:close:10",
            ]
        );
        assert!(done.comments[0].1.contains("Already shipped"));

        let mut waiting = FakePmoPort::successful();
        apply_pmo_decision(
            &issue,
            PmoOutput::WaitForDependency {
                dependency_issue_iid: 77,
            },
            None,
            Some("scope::test"),
            &mut waiting,
        )
        .unwrap();
        assert_eq!(
            *waiting.trace.borrow(),
            vec![
                "act:add:10:waiting-on-issue:#77",
                "act:remove:10:action-required",
                "act:remove:10:pmo-processed",
                "act:comment:10",
            ]
        );
        assert!(waiting.comments[0].1.contains("#77"));

        let mut guidance = FakePmoPort::successful();
        apply_pmo_decision(
            &issue,
            PmoOutput::GuideWorker {
                instructions: "Implement the parser first.".into(),
            },
            None,
            Some("scope::test"),
            &mut guidance,
        )
        .unwrap();
        assert_eq!(
            *guidance.trace.borrow(),
            vec![
                "act:comment:10",
                "act:remove:10:action-required",
                "act:remove:10:pmo-processed",
            ]
        );
        assert!(
            guidance.comments[0]
                .1
                .contains("Implement the parser first.")
        );
    }

    #[test]
    fn propose_plan_rejects_empty_content_before_mutating_the_issue() {
        let issue = FakePmoPort::issue(10);
        let mut port = FakePmoPort::successful();

        let result = apply_pmo_decision(
            &issue,
            PmoOutput::ProposePlan {
                plan_text: " \n\t ".into(),
            },
            None,
            Some("scope::test"),
            &mut port,
        );

        assert!(matches!(result, Err(PmoDecisionError::Recoverable(_))));
        assert!(port.trace.borrow().is_empty());
        assert!(port.descriptions.is_empty());
    }

    #[test]
    fn propose_plan_keeps_fresh_and_refined_issues_claimed_and_pending() {
        let plan = "## Plan\n\n1. Update `src/parser.rs`.\n2. Add parser tests.";

        let mut fresh = FakePmoPort::successful();
        fresh.plan = PmoOutput::ProposePlan {
            plan_text: plan.into(),
        };
        run_fake(&mut fresh, None).unwrap();
        let trace = fresh.trace.borrow();
        assert!(trace.contains(&"act:add:10:pmo-pending".to_string()));
        assert!(trace.contains(&"act:description:10".to_string()));
        assert!(!trace.contains(&"act:release".to_string()));
        assert!(!trace.contains(&"act:clear_claim".to_string()));
        drop(trace);
        assert_eq!(fresh.descriptions, vec![(10, plan.into())]);
        assert!(fresh.comments.is_empty());

        let mut refinement = FakePmoPort::successful();
        refinement.issues[0].labels.push(labels::PMO_PENDING.into());
        refinement.bound_mr_iid = Some(88);
        refinement.plan = PmoOutput::ProposePlan {
            plan_text: "Change the parser validation step to cover empty arrays.".into(),
        };
        run_fake(&mut refinement, Some(10)).unwrap();
        let trace = refinement.trace.borrow();
        assert!(trace.contains(&"observe:comments:10".to_string()));
        assert!(trace.contains(&"observe:bound_mr:10".to_string()));
        assert!(trace.contains(&"act:add:10:pmo-pending".to_string()));
        assert!(trace.contains(&"act:comment:10".to_string()));
        assert!(!trace.contains(&"act:description:10".to_string()));
        assert!(!trace.contains(&"act:release".to_string()));
        assert!(!trace.contains(&"act:clear_claim".to_string()));
        drop(trace);
        assert!(refinement.descriptions.is_empty());
        assert!(refinement.comments[0].1.contains("merge request !88"));
        assert!(
            refinement.comments[0]
                .1
                .contains("Change the parser validation step")
        );
    }

    #[test]
    fn propose_plan_description_failure_preserves_pending_claim_state() {
        let mut port = FakePmoPort::successful();
        port.plan = PmoOutput::ProposePlan {
            plan_text: "## Plan\n\nImplement and test the change.".into(),
        };
        port.failures = vec!["description"];

        run_fake(&mut port, None).unwrap();

        let trace = port.trace.borrow();
        let pending = trace
            .iter()
            .position(|event| event == "act:add:10:pmo-pending")
            .unwrap();
        let save = trace
            .iter()
            .rposition(|event| event == "act:save_claim:10")
            .unwrap();
        let description = trace
            .iter()
            .position(|event| event == "act:description:10")
            .unwrap();
        assert!(save < pending && pending < description);
        assert!(!trace.contains(&"act:release".to_string()));
        assert!(!trace.contains(&"act:clear_claim".to_string()));
    }

    #[test]
    fn pending_issue_is_not_reprocessed_without_a_new_human_comment() {
        let mut port = FakePmoPort::successful();
        port.issues[0].labels.push(labels::PMO_PENDING.into());
        port.new_comments = false;
        port.plan = PmoOutput::ProposePlan {
            plan_text: "This must not be applied.".into(),
        };

        run_fake(&mut port, Some(10)).unwrap();

        let trace = port.trace.borrow();
        assert!(trace.contains(&"observe:comments:10".to_string()));
        assert!(!trace.contains(&"act:model:10".to_string()));
        assert!(!trace.contains(&"act:description:10".to_string()));
        assert!(!trace.contains(&"act:release".to_string()));
        assert!(!trace.contains(&"act:clear_claim".to_string()));
    }

    #[test]
    fn clear_pending_plan_feedback_can_finish_without_gitlab_mutations() {
        let mut port = FakePmoPort::successful();
        port.issues[0].labels.push(labels::PMO_PENDING.into());
        port.new_comments = true;
        port.plan = PmoOutput::KeepPlan;

        run_fake(&mut port, Some(10)).unwrap();

        let trace = port.trace.borrow();
        assert!(trace.contains(&"observe:comments:10".to_string()));
        assert!(trace.contains(&"act:model:10".to_string()));
        assert!(!trace.iter().any(|event| {
            event.starts_with("act:description:")
                || event.starts_with("act:comment:")
                || event.starts_with("act:add:")
                || event.starts_with("act:remove:")
        }));
        assert!(!trace.contains(&"act:release".to_string()));
        assert!(!trace.contains(&"act:clear_claim".to_string()));
    }

    #[test]
    fn comment_cursor_ignores_old_and_pmo_comments_but_detects_later_human_feedback() {
        let comment = |id, author: &str, body: &str| gitlab::Comment {
            id,
            author: author.into(),
            body: body.into(),
            discussion_id: format!("discussion-{id}"),
            discussion_resolvable: false,
            location: None,
            location_details: None,
        };
        let comments = vec![
            comment(10, "alice", "Original discussion"),
            comment(
                11,
                "service-account",
                "**PMO plan refinement for merge request !5:**\n\nUpdated plan",
            ),
            comment(12, "bob", "Please change the validation step"),
        ];

        assert!(has_new_human_comments(&comments, 10, "pmo-0"));
        assert!(!has_new_human_comments(&comments[..2], 10, "pmo-0"));
        assert!(!has_new_human_comments(&comments, 12, "pmo-0"));
    }

    #[test]
    fn pmo_cycle_retries_checkout_and_stops_after_first_won_claim() {
        let mut port = FakePmoPort::successful();
        port.checkout_failures.set(1);
        port.issues.push(FakePmoPort::issue(11));
        port.claim_outcomes
            .borrow_mut()
            .extend([PmoClaimOutcome::Lost, PmoClaimOutcome::Won]);
        port.plan = PmoOutput::GuideWorker {
            instructions: "Proceed carefully.".into(),
        };

        run_fake(&mut port, None).unwrap();
        let trace = port.trace.borrow();
        assert_eq!(
            &trace[..5],
            [
                "observe:default_branch",
                "act:fetch",
                "act:checkout:main",
                "act:reset",
                "act:checkout:main",
            ]
        );
        assert!(
            trace
                .windows(2)
                .any(|events| events == ["act:claim:10", "observe:shutdown"])
        );
        assert!(trace.contains(&"act:claim:11".to_string()));
        assert_eq!(
            trace
                .iter()
                .filter(|event| event.as_str() == "act:model:11")
                .count(),
            1
        );
        assert!(!trace.iter().any(|event| event == "act:model:10"));
    }

    #[test]
    fn pmo_cycle_preserves_claim_on_shutdown_failure_and_tolerates_recovery_release_failure() {
        let mut shutdown_failure = FakePmoPort::successful();
        shutdown_failure.failures = vec!["model"];
        shutdown_failure
            .shutdown
            .borrow_mut()
            .extend([false, false, false, true]);
        run_fake(&mut shutdown_failure, None).unwrap();
        assert!(
            !shutdown_failure
                .trace
                .borrow()
                .contains(&"act:release".to_string())
        );
        assert!(
            !shutdown_failure
                .trace
                .borrow()
                .contains(&"act:clear_claim".to_string())
        );

        let mut recovered = FakePmoPort::successful();
        recovered.pending = Some(match FakePmoPort::split_plan() {
            PmoOutput::Split { sub_issues } => PendingSplit {
                parent_issue_iid: 10,
                parent_issue_title: "Broad parent".into(),
                parent_priority: 2,
                sub_issues: normalize_sub_issues(sub_issues),
                created_issue_ids: vec![101, 102],
            },
            _ => unreachable!(),
        });
        recovered.failures = vec!["release"];
        run_fake(&mut recovered, None).unwrap();
        assert_eq!(
            recovered.trace.borrow().last().map(String::as_str),
            Some("act:release")
        );
        assert!(!recovered.trace.borrow().contains(&"act:clear_claim".into()));
    }

    #[test]
    fn pmo_cycle_preserves_irregular_release_failure_policies_and_interruption() {
        let mut interrupted = FakePmoPort::successful();
        interrupted
            .claim_outcomes
            .borrow_mut()
            .push_back(PmoClaimOutcome::Interrupted);
        run_fake(&mut interrupted, None).unwrap();
        let trace = interrupted.trace.borrow();
        assert!(trace.contains(&"act:claim:10".to_string()));
        assert!(!trace.contains(&"act:model:10".to_string()));
        assert!(!trace.contains(&"act:release".to_string()));
        drop(trace);

        let mut invalid_held = FakePmoPort::successful();
        invalid_held.failures = vec!["release"];
        invalid_held.shutdown.borrow_mut().push_back(true);
        run_fake(&mut invalid_held, Some(10)).unwrap();
        assert!(invalid_held.trace.borrow().contains(&"act:release".into()));
        assert!(
            !invalid_held
                .trace
                .borrow()
                .contains(&"act:clear_claim".into())
        );

        let mut fresh_completion = FakePmoPort::successful();
        fresh_completion.plan = PmoOutput::GuideWorker {
            instructions: "Continue with the focused implementation.".into(),
        };
        fresh_completion.failures = vec!["release"];
        assert!(run_fake(&mut fresh_completion, None).is_err());
        assert!(
            !fresh_completion
                .trace
                .borrow()
                .contains(&"act:clear_claim".to_string())
        );
    }

    // -----------------------------------------------------------------
    // Split checkpoint/create/label/finalize ordering (`resume_split`).
    // -----------------------------------------------------------------

    fn sample_pending(created_issue_ids: Vec<u64>) -> PendingSplit {
        PendingSplit {
            parent_issue_iid: 100,
            parent_issue_title: "Parent issue".to_string(),
            parent_priority: 2,
            sub_issues: vec![
                PmoSubIssue {
                    title: "Sub A".to_string(),
                    description: "desc A".to_string(),
                    priority: None,
                    depends_on: 0,
                },
                PmoSubIssue {
                    title: "Sub B".to_string(),
                    description: "desc B".to_string(),
                    priority: None,
                    depends_on: 0,
                },
                PmoSubIssue {
                    title: "Sub C".to_string(),
                    description: "desc C".to_string(),
                    priority: None,
                    depends_on: 0,
                },
            ],
            created_issue_ids,
        }
    }

    #[test]
    fn sub_issue_already_created_covers_exactly_the_checkpointed_prefix() {
        let pending = sample_pending(vec![201, 202]);
        assert!(sub_issue_already_created(&pending, 0));
        assert!(sub_issue_already_created(&pending, 1));
        assert!(!sub_issue_already_created(&pending, 2));
    }

    #[test]
    fn sub_issue_already_created_recreates_nothing_from_a_fresh_split() {
        let pending = sample_pending(vec![]);
        assert!(!sub_issue_already_created(&pending, 0));
        assert!(!sub_issue_already_created(&pending, 1));
        assert!(!sub_issue_already_created(&pending, 2));
    }

    #[test]
    fn sub_issue_already_created_treats_a_fully_completed_checkpoint_as_done() {
        let pending = sample_pending(vec![201, 202, 203]);
        for index in 0..pending.sub_issues.len() {
            assert!(sub_issue_already_created(&pending, index));
        }
    }

    // Checkpoint persistence round trip: real temp files, no GitLab call.
    // `resume_split` relies on this surviving a crash between sub-issue
    // creations — the checkpoint records exactly the `created_issue_ids`
    // prefix so a restart resumes instead of re-creating issues.

    fn pending_split_test_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "potlatch-pmo-pending-split-{name}-{}",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn load_pending_split_returns_none_when_no_checkpoint_exists() {
        let path = pending_split_test_path("missing");
        assert!(load_pending_split(&path).unwrap().is_none());
    }

    #[test]
    fn pending_split_checkpoint_round_trips_through_save_and_load() {
        let path = pending_split_test_path("roundtrip");
        let pending = sample_pending(vec![201]);

        save_pending_split(&path, &pending).unwrap();
        let loaded = load_pending_split(&path).unwrap().unwrap();
        assert_eq!(loaded.parent_issue_iid, pending.parent_issue_iid);
        assert_eq!(loaded.created_issue_ids, pending.created_issue_ids);
        assert_eq!(loaded.sub_issues, pending.sub_issues);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn pending_split_checkpoint_is_updated_incrementally_as_issues_are_created() {
        let path = pending_split_test_path("incremental");
        let mut pending = sample_pending(vec![]);
        save_pending_split(&path, &pending).unwrap();

        // Simulate the per-created-issue checkpoint write inside the
        // `resume_split` loop: each successful `create_issue` immediately
        // persists the updated `created_issue_ids` prefix before moving on.
        pending.created_issue_ids.push(301);
        save_pending_split(&path, &pending).unwrap();
        assert_eq!(
            load_pending_split(&path)
                .unwrap()
                .unwrap()
                .created_issue_ids,
            vec![301]
        );

        pending.created_issue_ids.push(302);
        save_pending_split(&path, &pending).unwrap();
        assert_eq!(
            load_pending_split(&path)
                .unwrap()
                .unwrap()
                .created_issue_ids,
            vec![301, 302]
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn delete_pending_split_finalizes_by_removing_the_checkpoint() {
        let path = pending_split_test_path("finalize");
        let pending = sample_pending(vec![301, 302, 303]);
        save_pending_split(&path, &pending).unwrap();
        assert!(load_pending_split(&path).unwrap().is_some());

        delete_pending_split(&path).unwrap();
        assert!(load_pending_split(&path).unwrap().is_none());
        // Deleting an already-absent checkpoint is idempotent.
        delete_pending_split(&path).unwrap();
    }

    #[test]
    fn pending_split_checkpoint_round_trips_through_the_v1_envelope() {
        let path = pending_split_test_path("v1-envelope");
        let pending = sample_pending(vec![201]);
        save_pending_split(&path, &pending).unwrap();

        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["parent_issue_iid"], 100);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_pending_split_reads_the_legacy_unversioned_format_and_migrates_it() {
        let path = pending_split_test_path("legacy");
        let pending = sample_pending(vec![301]);
        // The bare pre-envelope payload written by older builds.
        fs::write(&path, serde_json::to_vec(&pending).unwrap()).unwrap();

        let loaded = load_pending_split(&path).unwrap().unwrap();
        assert_eq!(loaded.parent_issue_iid, pending.parent_issue_iid);
        assert_eq!(loaded.created_issue_ids, pending.created_issue_ids);

        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);

        let _ = fs::remove_file(&path);
    }

    /// A dedicated, per-test directory for the quarantine tests below (unlike
    /// `pending_split_test_path`, which places its file directly in the
    /// shared system temp dir — fine for name-based lookups, but not safe
    /// to `read_dir` and scan for stray quarantine files from other tests).
    fn pending_split_quarantine_test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "potlatch-pmo-pending-split-quarantine-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn load_pending_split_is_strict_and_quarantines_malformed_json() {
        let dir = pending_split_quarantine_test_dir("corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json").to_string_lossy().into_owned();
        fs::write(&path, b"not valid json").unwrap();

        let error = load_pending_split(&path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to load pending split file")
        );

        // Strict: quarantined (bytes preserved) rather than silently reset —
        // resuming from bad checkpoint data risks re-creating sub-issues.
        assert!(!path::Path::new(&path).exists());
        let quarantined: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("quarantined"))
            .collect();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(fs::read(quarantined[0].path()).unwrap(), b"not valid json");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_pending_split_is_strict_and_quarantines_an_unsupported_version() {
        let dir = pending_split_quarantine_test_dir("unsupported-version");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json").to_string_lossy().into_owned();
        fs::write(&path, br#"{"version":3,"state":{}}"#).unwrap();

        let error = load_pending_split(&path).unwrap_err();
        assert!(error.chain().any(|e| {
            e.to_string()
                .contains("Unsupported state envelope version 3")
        }));
        assert!(!path::Path::new(&path).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // PMO claim state persistence (`save_state`/`clear_state`/
    // `try_resume_pmo_state`): tolerant policy. Invalid persisted state
    // has historically been ignored (warn + treat as "nothing to
    // resume") rather than failing PMO startup. `GitLabClient::for_test`
    // is a network-free constructor, and every case below returns before
    // `try_resume_pmo_state` would ever reach a real GitLab call.
    // -----------------------------------------------------------------

    fn pmo_test_agent_state<'a>(sessions_dir: &'a str, agent_id: &'a str) -> AgentState<'a> {
        AgentState {
            sessions_dir,
            agent_id,
            project_name: "test-project",
        }
    }

    fn pmo_state_test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "potlatch-pmo-claim-state-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn pmo_claim_state_round_trips_through_save_state_and_the_v1_envelope() {
        let dir = pmo_state_test_dir("roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = pmo_test_agent_state(&sessions_dir, "pmo-0");

        state.save_state_with_comment_cursor(55, 99);
        let on_disk: serde_json::Value =
            serde_json::from_slice(&fs::read(state.state_path()).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["claimed_issue_iid"], 55);
        assert_eq!(on_disk["state"]["last_seen_comment_id"], 99);
        assert_eq!(state.last_seen_comment_id(55), 99);

        state.clear_state();
        assert!(!state.state_path().exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pmo_claim_state_reads_the_legacy_unversioned_format_and_migrates_it() {
        let dir = pmo_state_test_dir("legacy");
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = pmo_test_agent_state(&sessions_dir, "pmo-1");
        let path = state.state_path();

        // The bare pre-envelope payload written by older builds.
        fs::write(&path, br#"{"claimed_issue_iid":77}"#).unwrap();

        let store: crate::agents::state::StateStore<PersistedPmoState> =
            crate::agents::state::StateStore::new(&path);
        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded.claimed_issue_iid, 77);
        assert_eq!(loaded.last_seen_comment_id, 0);

        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pending_comment_cursor_stays_with_the_original_pmo_claim_state() {
        let dir = pmo_state_test_dir("pending-comment-cursor");
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let first = pmo_test_agent_state(&sessions_dir, "pmo-0");
        let second = pmo_test_agent_state(&sessions_dir, "pmo-1");

        first.save_state_with_comment_cursor(55, 99);

        assert_eq!(first.last_seen_comment_id(55), 99);
        assert_eq!(second.last_seen_comment_id(55), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn try_resume_pmo_state_tolerates_a_missing_state_file() {
        let dir = pmo_state_test_dir("missing");
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = pmo_test_agent_state(&sessions_dir, "pmo-2");
        let gitlab = GitLabClient::for_test("/tmp/unused-repo");

        assert!(try_resume_pmo_state(&state, &gitlab, None).is_none());
    }

    #[test]
    fn empty_orphan_claim_scan_has_no_primary_claim() {
        assert!(split_recovered_claims(Vec::new()).is_none());
    }

    #[test]
    fn try_resume_pmo_state_tolerates_corrupt_state_by_warning_and_returning_none() {
        let dir = pmo_state_test_dir("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = pmo_test_agent_state(&sessions_dir, "pmo-3");
        fs::write(state.state_path(), b"not valid json").unwrap();
        let gitlab = GitLabClient::for_test("/tmp/unused-repo");

        assert!(try_resume_pmo_state(&state, &gitlab, None).is_none());

        // Quarantined beside the original rather than deleted outright.
        assert!(!state.state_path().exists());
        let quarantined: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("quarantined"))
            .collect();
        assert_eq!(quarantined.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn try_resume_pmo_state_tolerates_an_unsupported_envelope_version() {
        let dir = pmo_state_test_dir("unsupported-version");
        fs::create_dir_all(&dir).unwrap();
        let sessions_dir = dir.to_string_lossy().into_owned();
        let state = pmo_test_agent_state(&sessions_dir, "pmo-4");
        fs::write(state.state_path(), br#"{"version":4,"state":{}}"#).unwrap();
        let gitlab = GitLabClient::for_test("/tmp/unused-repo");

        assert!(try_resume_pmo_state(&state, &gitlab, None).is_none());
        assert!(!state.state_path().exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_issues_summary_puts_newest_first_within_priority() {
        let issue = |iid, priority, created_at: &str, state: &str| Issue {
            iid,
            title: format!("Issue {iid}"),
            description: String::new(),
            labels: vec![format!("priority::{priority}")],
            state: state.into(),
            created_at: Some(created_at.into()),
            updated_at: None,
        };
        let issues = vec![
            issue(10, 1, "2026-01-01T00:00:00Z", "opened"),
            issue(20, 1, "2026-02-01T00:00:00Z", "opened"),
            issue(30, 2, "2026-03-01T00:00:00Z", "opened"),
            issue(40, 1, "2026-04-01T00:00:00Z", "closed"),
            issue(50, 1, "2026-05-01T00:00:00Z", "opened"),
        ];

        let summary = build_existing_issues_summary(50, &issues);
        let newest_priority_one = summary.find("#20").unwrap();
        let oldest_priority_one = summary.find("#10").unwrap();
        let priority_two = summary.find("#30").unwrap();

        assert!(newest_priority_one < oldest_priority_one);
        assert!(oldest_priority_one < priority_two);
        assert!(!summary.contains("#40"));
        assert!(!summary.contains("#50"));
    }

    #[test]
    fn build_split_prompt_contains_policy_without_structured_output_markers() {
        let state = AgentState {
            sessions_dir: "/tmp",
            agent_id: "pmo-test",
            project_name: "test-proj",
        };
        let issue = Issue {
            iid: 42,
            title: "Worker could not complete".into(),
            description: "Details".into(),
            labels: vec![],
            state: "opened".into(),
            created_at: None,
            updated_at: None,
        };
        let prompt = build_split_prompt(&state, &issue, "/abs/pmo-issue-42.md", 2, None).unwrap();
        // Structured-output presentation belongs to the backend vendor, not
        // the role prompt.
        assert!(!prompt.contains("`plan` tool"));
        assert!(!prompt.contains("output contract"));
        assert!(!prompt.contains("The tool's parameters are"));
        assert!(!prompt.contains("JSON shape above"));
        for marker in [
            "guide_worker",
            "propose_plan",
            "keep_plan",
            "already_done",
            "needs_clarification",
            "wait_for_dependency",
            "depends_on",
            "plan_text",
            "dependency_issue_iid",
            "SUB_ISSUE_N:",
            "GUIDE_WORKER",
            "ALREADY_DONE",
            "NEEDS_CLARIFICATION",
            "WAIT_FOR_DEPENDENCY",
            "PUBLIC_COMMENT_BEGIN",
            "text-marker",
        ] {
            assert!(!prompt.contains(marker), "unexpected marker {marker:?}");
        }
        // No text-marker fallback.
        assert!(!prompt.contains("SUB_ISSUE_N:"));
        // Reasoning guidance is intact.
        assert!(prompt.contains("broad task containers"));
        assert!(prompt.contains("Never duplicate or substantially overlap existing work"));
        assert!(prompt.contains("repository already fully implements"));
        assert!(prompt.contains("complete rewritten issue description"));
        assert!(prompt.contains("keeps the issue pmo-pending"));
        assert!(prompt.contains("explicitly keep the current plan unchanged"));
        assert!(prompt.contains("must not rewrite the description or add another comment"));
    }

    #[test]
    fn build_split_prompt_treats_human_comments_as_plan_refinement_while_pending() {
        let state = AgentState {
            sessions_dir: "/tmp",
            agent_id: "pmo-test",
            project_name: "test-proj",
        };
        let issue = Issue {
            iid: 42,
            title: "Refine parser design".into(),
            description: "## Existing plan".into(),
            labels: vec![labels::PMO_PENDING.into()],
            state: "opened".into(),
            created_at: None,
            updated_at: None,
        };

        let prompt =
            build_split_prompt(&state, &issue, "/abs/pmo-issue-42.md", 2, Some(77)).unwrap();

        assert!(prompt.contains("Human comments are refinement feedback"));
        assert!(prompt.contains("already bound to merge request !77"));
        assert!(prompt.contains("only the plan changes and refinements"));
        assert!(prompt.contains("posted as a new issue comment"));
        assert!(prompt.contains("do not produce a full rewritten description"));
    }

    #[test]
    fn pmo_contract_passes_the_shared_conformance_suite() {
        conformance::assert_contract::<PmoOutput>();
    }

    #[test]
    fn pmo_output_deserializes_split_with_sub_issues() {
        let output = conformance::assert_accepts::<PmoOutput>(serde_json::json!({
            "decision": "split",
            "sub_issues": [
                {"title": "First", "description": "Do the first thing", "priority": 1},
                {"title": "Second", "description": "Do the second thing", "priority": 2, "depends_on": 1}
            ]
        }));
        let PmoOutput::Split { sub_issues } = output else {
            panic!("expected Split");
        };
        let normalized = normalize_sub_issues(sub_issues);
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].title, "First");
        assert_eq!(normalized[0].priority, Some(1));
        assert_eq!(normalized[0].depends_on, 0);
        assert_eq!(normalized[1].title, "Second");
        assert_eq!(normalized[1].priority, Some(2));
        assert_eq!(normalized[1].depends_on, 1);
    }

    #[test]
    fn normalize_sub_issues_remaps_dependencies_after_filtering() {
        let normalized = normalize_sub_issues(vec![
            RawSubIssue {
                title: String::new(),
                description: "placeholder".into(),
                ..Default::default()
            },
            RawSubIssue {
                title: "First".into(),
                description: "one".into(),
                ..Default::default()
            },
            RawSubIssue {
                title: "Second".into(),
                description: "two".into(),
                depends_on: 2,
                ..Default::default()
            },
        ]);

        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[1].depends_on, 1);
    }

    #[test]
    fn pmo_output_deserializes_guide_worker_instructions() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "guide_worker",
                "instructions": "Use flag --foo instead of --bar."
            })),
            PmoOutput::GuideWorker {
                instructions: "Use flag --foo instead of --bar.".into()
            }
        );
    }

    #[test]
    fn pmo_output_decision_is_case_insensitive() {
        let output = conformance::assert_accepts::<PmoOutput>(serde_json::json!({
            "decision": " GUIDE_WORKER ",
            "instructions": "Proceed."
        }));
        assert!(matches!(output, PmoOutput::GuideWorker { .. }));
    }

    #[test]
    fn pmo_output_deserializes_already_done_reason() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "already_done",
                "reason": "The feature exists in src/lib.rs."
            })),
            PmoOutput::AlreadyDone {
                reason: "The feature exists in src/lib.rs.".into()
            }
        );
    }

    #[test]
    fn pmo_output_deserializes_needs_clarification_question() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "needs_clarification",
                "question": "Which modules should be covered?"
            })),
            PmoOutput::NeedsClarification {
                question: "Which modules should be covered?".into()
            }
        );
    }

    #[test]
    fn pmo_output_deserializes_wait_for_dependency_iid() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "wait_for_dependency",
                "dependency_issue_iid": 47
            })),
            PmoOutput::WaitForDependency {
                dependency_issue_iid: 47
            }
        );
    }

    #[test]
    fn pmo_output_rejects_zero_dependency_issue_iid() {
        let error = conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "wait_for_dependency",
            "dependency_issue_iid": 0
        }));
        assert_eq!(
            error,
            "$.dependency_issue_iid: required property is missing"
        );
    }

    #[test]
    fn pmo_output_accepts_string_dependency_issue_iid() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "wait_for_dependency",
                "dependency_issue_iid": "#727"
            })),
            PmoOutput::WaitForDependency {
                dependency_issue_iid: 727
            }
        );
    }

    #[test]
    fn pmo_output_accepts_alternative_dependency_field_names() {
        for key in DEPENDENCY_IID_ALIASES {
            let mut captured = serde_json::json!({"decision": "wait_for_dependency"});
            captured[*key] = serde_json::json!(727);
            assert_eq!(
                conformance::assert_accepts::<PmoOutput>(captured),
                PmoOutput::WaitForDependency {
                    dependency_issue_iid: 727
                },
                "failed for alternative field name `{key}`"
            );
        }
    }

    #[test]
    fn pmo_output_rejects_unknown_decision() {
        let error = conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "bogus"
        }));
        assert!(error.starts_with("$.decision: expected one of"), "{error}");
    }

    #[test]
    fn pmo_output_requires_each_decision_branch_field() {
        assert_eq!(
            conformance::assert_rejects::<PmoOutput>(serde_json::json!({
                "decision": "guide_worker"
            })),
            "$.instructions: required property is missing"
        );
        assert_eq!(
            conformance::assert_rejects::<PmoOutput>(serde_json::json!({"decision": "split"})),
            "$.sub_issues: required property is missing"
        );
        assert_eq!(
            conformance::assert_rejects::<PmoOutput>(serde_json::json!({
                "decision": "already_done"
            })),
            "$.reason: required property is missing"
        );
        assert_eq!(
            conformance::assert_rejects::<PmoOutput>(serde_json::json!({
                "decision": "needs_clarification"
            })),
            "$.question: required property is missing"
        );
    }

    #[test]
    fn pmo_output_rejects_fields_from_another_decision() {
        let error = conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "already_done",
            "reason": "done",
            "sub_issues": []
        }));
        assert!(
            error.starts_with("$.sub_issues: unexpected property"),
            "{error}"
        );
    }

    #[test]
    fn pmo_output_reports_the_offending_sub_issue_by_index() {
        let error = conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "split",
            "sub_issues": [
                {"title": "First", "description": "one"},
                {"title": "Second"}
            ]
        }));
        assert_eq!(
            error,
            "$.sub_issues[1].description: required property is missing"
        );
    }

    #[test]
    fn pmo_output_drops_an_out_of_range_sub_issue_priority() {
        let output = conformance::assert_accepts::<PmoOutput>(serde_json::json!({
            "decision": "split",
            "sub_issues": [{"title": "First", "description": "one", "priority": 9}]
        }));
        let PmoOutput::Split { sub_issues } = output else {
            panic!("expected Split");
        };
        assert_eq!(sub_issues[0].priority, None);
    }

    #[test]
    fn normalize_sub_issues_skips_empty_titles() {
        let raw = vec![
            RawSubIssue {
                title: "".into(),
                description: "no title".into(),
                ..Default::default()
            },
            RawSubIssue {
                title: "Valid".into(),
                description: "has title".into(),
                ..Default::default()
            },
        ];
        let normalized = normalize_sub_issues(raw);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].title, "Valid");
    }

    #[test]
    fn normalize_sub_issues_skips_empty_descriptions() {
        let raw = vec![
            RawSubIssue {
                title: "No desc".into(),
                description: "".into(),
                ..Default::default()
            },
            RawSubIssue {
                title: "Valid".into(),
                description: "has desc".into(),
                ..Default::default()
            },
        ];
        let normalized = normalize_sub_issues(raw);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].title, "Valid");
    }

    #[test]
    fn test_should_process_issue() {
        let mut issue = Issue {
            iid: 1,
            title: "Test issue".to_string(),
            description: "Test".to_string(),
            labels: vec![ACTION_REQUIRED_LABEL.to_string()],
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        };
        assert!(should_process_issue(&issue, None));

        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            PMO_PROCESSED_LABEL.to_string(),
        ];
        assert!(!should_process_issue(&issue, None));

        issue.labels = vec![];
        assert!(!should_process_issue(&issue, None));

        issue.title = "[Draft] Test".to_string();
        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(!should_process_issue(&issue, None));

        issue.title = "Normal issue".to_string();
        issue.labels = vec![
            ACTION_REQUIRED_LABEL.to_string(),
            labels::PMO_PENDING.to_string(),
        ];
        assert!(!should_process_issue(&issue, None));

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string(), "potlatch".to_string()];
        assert!(!should_process_issue(&issue, Some("other-scope")));
        assert!(should_process_issue(&issue, Some("potlatch")));
    }

    #[test]
    fn guidance_or_empty_returns_instructions() {
        assert_eq!(
            guidance_or_empty("Use the existing config loader."),
            "Use the existing config loader."
        );
    }

    #[test]
    fn guidance_or_empty_returns_empty_when_instructions_are_blank() {
        assert!(guidance_or_empty("   ").is_empty());
    }

    #[test]
    fn guidance_or_empty_truncates_long_instructions() {
        let long = "Do this. ".repeat(80);
        let extracted = guidance_or_empty(&long);
        assert!(extracted.len() <= 502);
        assert!(extracted.starts_with("Do this."));
    }

    #[test]
    fn format_pmo_guidance_comment_includes_header_and_body() {
        let comment = format_pmo_guidance_comment("Use --foo instead of --bar.");
        assert!(comment.contains("**PMO guidance for the worker agent:**"));
        assert!(comment.contains("Use --foo instead of --bar."));
        // No internal markers — the comment is plain human-facing text.
        assert!(!comment.contains("PMO_GUIDANCE_BEGIN"));
        assert!(!comment.contains("PMO_GUIDANCE_END"));
    }

    #[test]
    fn format_pmo_guidance_comment_preserves_multiline_body() {
        let body = "Step one: do X.\nStep two: do Y.";
        let comment = format_pmo_guidance_comment(body);
        assert!(comment.contains(body));
    }

    #[test]
    fn format_pmo_guidance_comment_includes_header_even_when_body_is_whitespace() {
        let comment = format_pmo_guidance_comment("   ");
        assert!(comment.contains("**PMO guidance for the worker agent:**"));
        assert!(!comment.contains("PMO_GUIDANCE_BEGIN"));
    }

    #[test]
    fn already_done_reason_or_default_uses_field() {
        assert!(already_done_reason_or_default("Implemented in module X.").contains("module X"));
    }

    #[test]
    fn already_done_reason_or_default_falls_back_when_blank() {
        assert!(!already_done_reason_or_default("  ").is_empty());
    }

    #[test]
    fn clarification_question_or_default_uses_field() {
        assert_eq!(
            clarification_question_or_default("Which modules?"),
            "Which modules?"
        );
    }

    #[test]
    fn clarification_question_or_default_falls_back_when_blank() {
        assert!(!clarification_question_or_default("").is_empty());
    }

    #[test]
    fn pmo_output_deserializes_propose_plan() {
        let output = conformance::assert_accepts::<PmoOutput>(serde_json::json!({
            "decision": "propose_plan",
            "plan_text": "## Plan Draft\n\n1. Implement X\n2. Test X\n\nOpen question: which config?"
        }));
        match output {
            PmoOutput::ProposePlan { plan_text } => {
                assert!(plan_text.contains("## Plan Draft"));
                assert!(plan_text.contains("Implement X"));
            }
            other => panic!("expected ProposePlan, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_deserializes_keep_plan_without_payload() {
        assert_eq!(
            conformance::assert_accepts::<PmoOutput>(serde_json::json!({
                "decision": "keep_plan"
            })),
            PmoOutput::KeepPlan
        );
        conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "keep_plan",
            "reason": "No change needed"
        }));
    }

    #[test]
    fn pmo_output_requires_plan_text_for_propose_plan() {
        conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "propose_plan"
        }));
        conformance::assert_rejects::<PmoOutput>(serde_json::json!({
            "decision": "needs_clarification",
            "question": "Which config?",
            "plan_text": "This belongs only to propose_plan."
        }));
    }

    #[test]
    fn test_priority_from_labels() {
        assert_eq!(
            gitlab::priority_from_labels(&["priority::1".to_string()]),
            1
        );
        assert_eq!(
            gitlab::priority_from_labels(&["priority::2".to_string(), "in-progress".to_string()]),
            2
        );
        assert_eq!(
            gitlab::priority_from_labels(&["in-progress".to_string()]),
            3
        );
    }

    #[test]
    fn truncate_diff_for_pmo_keeps_short_diffs_unchanged() {
        let diff = "diff --git a/file.go b/file.go\n+hello\n";
        assert_eq!(truncate_diff_for_pmo(diff), diff);
    }

    #[test]
    fn truncate_diff_for_pmo_truncates_long_diffs_with_marker() {
        let diff = "x".repeat(10_000);
        let result = truncate_diff_for_pmo(&diff);
        assert!(result.contains("[...diff truncated at"));
        assert!(result.len() < diff.len());
        // Should still start with the original content.
        assert!(result.starts_with('x'));
    }

    #[test]
    fn truncate_diff_for_pmo_respects_char_boundary() {
        // Multi-byte UTF-8 characters must not be split.
        let diff = format!("{}\n", "α".repeat(4_000)); // each α is 2 bytes
        let result = truncate_diff_for_pmo(&diff);
        // Result is valid UTF-8 (no panic), and either truncated or full.
        assert!(result.contains('α'));
    }

    fn ask_thread_note(id: u64, body: &str, system: bool, disc: Option<&str>) -> IssueThreadNote {
        let raw = format!(
            r#"{{"id":{id},"body":{},"system":{system},"discussion_id":{},"author":{{"username":"u"}}}}"#,
            serde_json::to_string(body).unwrap(),
            serde_json::to_string(&disc).unwrap()
        );
        serde_json::from_str(&raw).unwrap()
    }

    fn mode_question() -> AskQuestion {
        use crate::core::model::acp::capabilities::AskChoice;
        AskQuestion {
            text: "Choose a mode".into(),
            choices: vec![
                AskChoice {
                    id: "guide".into(),
                    label: "Guide worker".into(),
                },
                AskChoice {
                    id: "split".into(),
                    label: "Split issue".into(),
                },
            ],
        }
    }

    #[test]
    fn ask_pick_reply_same_discussion() {
        let root = ask_thread_note(10, "<!-- potlatch-pmo-acp-ask:abc -->", false, Some("d1"));
        let r1 = ask_thread_note(11, "opt-b", false, Some("d1"));
        let noise = ask_thread_note(9, "old", false, Some("d2"));
        let notes = vec![noise, root.clone(), r1.clone()];
        let root_ref = notes.iter().find(|x| x.id == 10).unwrap();
        let got = pick_direct_thread_reply(&notes, root_ref).unwrap();
        assert_eq!(got.id, 11);
        assert_eq!(reply_body_as_choice(&got.body), "opt-b");
    }

    #[test]
    fn ask_pick_first_reply_when_sorted() {
        let root = ask_thread_note(5, "<!-- potlatch-pmo-acp-ask:x -->", false, None);
        let r1 = ask_thread_note(6, "0", false, None);
        let notes = vec![root.clone(), r1.clone()];
        let root_ref = &notes[0];
        let got = pick_direct_thread_reply(&notes, root_ref).unwrap();
        assert_eq!(got.id, 6);
    }

    #[test]
    fn ask_ignores_new_ask_in_thread() {
        let root = ask_thread_note(1, "<!-- potlatch-pmo-acp-ask:a -->", false, Some("d"));
        let bad = ask_thread_note(2, "<!-- potlatch-pmo-acp-ask:b -->", false, Some("d"));
        let notes = vec![root.clone(), bad];
        let root_ref = &notes[0];
        assert!(pick_direct_thread_reply(&notes, root_ref).is_none());
    }

    #[test]
    fn ask_resolves_reply_to_answer_by_number_index_and_id() {
        let q = mode_question();

        let by_number = resolve_reply_to_answer(&q, "2");
        assert!(matches!(by_number, AskAnswer::Choice(ref id) if id == "split"));

        let by_zero = resolve_reply_to_answer(&q, "0");
        assert!(matches!(by_zero, AskAnswer::Choice(ref id) if id == "guide"));

        let by_one = resolve_reply_to_answer(&q, "1");
        assert!(matches!(by_one, AskAnswer::Choice(ref id) if id == "guide"));

        let by_id = resolve_reply_to_answer(&q, "guide");
        assert!(matches!(by_id, AskAnswer::Choice(ref id) if id == "guide"));
    }

    #[test]
    fn ask_resolves_empty_reply_to_auto_and_unmatched_to_free_text() {
        let q = mode_question();

        assert!(matches!(resolve_reply_to_answer(&q, ""), AskAnswer::Auto));
        assert!(matches!(
            resolve_reply_to_answer(&q, "   "),
            AskAnswer::Auto
        ));

        let free = resolve_reply_to_answer(&q, "something else");
        assert!(matches!(free, AskAnswer::FreeText(ref t) if t == "something else"));
    }

    #[test]
    fn ask_comment_uses_question_text_and_choices() {
        let q = mode_question();
        let comment = build_issue_comment("askid", &q);
        assert!(comment.contains("Choose a mode"));
        assert!(comment.contains("`guide`"));
        assert!(comment.contains("Split issue"));
        assert!(comment.contains("**Agent question**"));
        assert!(!comment.contains("Cursor"));
    }
}
