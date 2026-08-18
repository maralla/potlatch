use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::claim::{ClaimAcquireOutcome, ClaimLease, ClaimResource};
use super::{
    claim, issue_in_scope, strip_internal_markers, strip_public_comment_blocks,
    write_task_context_file,
};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{GitLabClient, Issue};
use crate::agents::workspace::{GitLabAgentBootstrap, GitLabAgentRuntime, gitlab_banner};
use crate::core::agent::schema::tagged;
use crate::core::agent::{
    AgentModel, CoreAgent, InvokeOptions, ModelPreferences, ObjectSchema, OneOfSchema, Schema,
    StructuredOutput, compat,
};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;

const WORKING_ON_LABEL: &str = "in-progress";
/// Root-level file updated by Potlatch after each successful worker run (impl or MR feedback).
const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";
const PMO_PENDING_LABEL: &str = "pmo-pending";
const NEED_AI_WORKER_LABEL: &str = "need-ai-worker";
/// Human/workflow pause: worker skips the issue (no close) and releases its hold until removed.
const WORKER_PENDING_LABEL: &str = "pending";
/// Reviewer-only workflow: worker skips and stops tracking while reviewer may still process the MR.
const WORKER_REVIEW_ONLY_LABEL: &str = "review-only";
/// ACP runtime message when `cancel_check` returns true.
const WORKER_AGENT_CANCELLED_MSG: &str = "Agent cancelled by external condition";
const WORKER_MISSING_OUTPUT_NUDGE: &str = "Continue this implementation in the current session. \
Your previous turns did not produce the required structured result. Do not restart or merely \
explain the task: finish the work, then submit the result using the backend-provided structured \
output format.";
const WORKER_NO_CHANGES_NUDGE: &str = "Continue this implementation in the current session. \
Your previous result claimed completion, but the repository has no code changes for this issue. \
Inspect the current workspace, make the required implementation and tests, then submit an updated \
structured result. Do not merely repeat the previous answer.";

/// The name of the structured-output tool both worker contracts use. An
/// implementation run and a feedback run are different tasks with different
/// outcomes, but from the model's side they are the same gesture: hand the
/// run's result back to Potlatch.
const HANDOFF_TOOL: &str = "handoff";

/// Metadata for a merge request the worker actually produced. Every field is
/// optional because the worker's fallbacks (issue title, placeholder
/// description) are better defaults than forcing the model to invent text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImplementedMetadata {
    #[serde(default)]
    mr_title: Option<String>,
    #[serde(default)]
    mr_description: Option<String>,
    #[serde(default)]
    changes_summary: Option<String>,
}

/// Why the worker is handing the run back to a human instead of finishing it.
/// The discriminator says which kind of blockage it is; `reason` carries the
/// explanation, and `public_comment` overrides the text posted to GitLab.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockedOutcome {
    reason: String,
    #[serde(default)]
    public_comment: Option<String>,
}

/// Everything a feedback run can report once it has addressed (or decided not
/// to change anything for) the reviewer's comments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeedbackResolution {
    #[serde(default)]
    mr_title: Option<String>,
    #[serde(default)]
    mr_description: Option<String>,
    #[serde(default)]
    changes_summary: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    public_comment: Option<String>,
    #[serde(default)]
    mark_discussions_resolved: Option<bool>,
    #[serde(default)]
    post_plain_comment: bool,
}

/// The outcome of a worker *implementation* run, as a tagged union on
/// `outcome`. The model calls the `handoff` tool; core validates the captured
/// JSON against [`WorkerImplementationOutput::schema`] and deserializes it
/// (see [`AgentModel::complete_typed`]). Because the outcomes are branches
/// rather than independent flags, contradictory combinations — "I implemented
/// it *and* it needs splitting", "here is an existing MR *and* a dependency"
/// — cannot be expressed at all.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerImplementationOutput {
    /// The work is done in the checked-out branch; here is its MR metadata.
    Implemented(ImplementedMetadata),
    /// An already-open MR implements this issue; adopt it instead of
    /// creating a new one.
    ExistingMr { existing_mr_iid: u64 },
    /// The work is hard-blocked on another issue closing first.
    WaitDependency { depends_on_issue: u64 },
    /// The issue is too broad and must be split before it can be worked on.
    NeedsSplit(BlockedOutcome),
    /// The issue is missing information only a human can supply.
    NeedsClarification(BlockedOutcome),
    /// The issue cannot be implemented as specified, for some other reason.
    CannotImplement(BlockedOutcome),
}

/// The outcome of a worker *MR feedback* run, as a tagged union on `outcome`.
/// Same tool name as [`WorkerImplementationOutput`], different contract:
/// a feedback run cannot request a split or declare a dependency, and only a
/// feedback run controls discussion resolution and plain comments.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerFeedbackOutput {
    /// The reviewer's feedback was handled (in code, in MR metadata, or by
    /// explaining that the branch already satisfies it).
    Addressed(FeedbackResolution),
    /// The feedback cannot be resolved autonomously; abandon the MR.
    CannotResolve(BlockedOutcome),
}

impl<'de> Deserialize<'de> for WorkerImplementationOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (outcome, fields) = tagged::parts(deserializer, "outcome")?;
        match outcome.as_str() {
            "implemented" => tagged::branch(fields).map(Self::Implemented),
            "existing_mr" => tagged::branch(fields).map(|wire: ExistingMrWire| Self::ExistingMr {
                existing_mr_iid: wire.existing_mr_iid,
            }),
            "wait_dependency" => {
                tagged::branch(fields).map(|wire: WaitDependencyWire| Self::WaitDependency {
                    depends_on_issue: wire.depends_on_issue,
                })
            }
            "needs_split" => tagged::branch(fields).map(Self::NeedsSplit),
            "needs_clarification" => tagged::branch(fields).map(Self::NeedsClarification),
            "cannot_implement" => tagged::branch(fields).map(Self::CannotImplement),
            outcome => Err(serde::de::Error::unknown_variant(
                outcome,
                &[
                    "implemented",
                    "existing_mr",
                    "wait_dependency",
                    "needs_split",
                    "needs_clarification",
                    "cannot_implement",
                ],
            )),
        }
    }
}

impl<'de> Deserialize<'de> for WorkerFeedbackOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (outcome, fields) = tagged::parts(deserializer, "outcome")?;
        match outcome.as_str() {
            "addressed" => tagged::branch(fields).map(Self::Addressed),
            "cannot_resolve" => tagged::branch(fields).map(Self::CannotResolve),
            outcome => Err(serde::de::Error::unknown_variant(
                outcome,
                &["addressed", "cannot_resolve"],
            )),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExistingMrWire {
    existing_mr_iid: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WaitDependencyWire {
    depends_on_issue: u64,
}

/// The MR metadata properties shared by the implementation contract's
/// `implemented` branch and the feedback contract's `addressed` branch.
fn mr_metadata_properties(schema: ObjectSchema) -> ObjectSchema {
    schema
        .property(
            "mr_title",
            Schema::string(
                "Short MR title (max 8-10 words). Focus on WHAT, not HOW. No markdown. Omit to keep the current title.",
            ),
        )
        .property(
            "mr_description",
            Schema::string(
                "Full MR description in markdown with ## Goal, ## Implementation, ## Testing sections.",
            ),
        )
        .property(
            "changes_summary",
            Schema::string(
                "A concise sentence summarizing the substance of the changes you made (used as the commit message).",
            ),
        )
}

/// The `reason` + `public_comment` pair every blocked branch carries.
fn blocked_properties(schema: ObjectSchema, reason: &str) -> ObjectSchema {
    schema
        .required_property("reason", Schema::string(reason.to_string()))
        .property(
            "public_comment",
            Schema::string(
                "Human-facing GitLab comment text to post instead of `reason`. Omit to post `reason` as-is.",
            ),
        )
}

impl StructuredOutput for WorkerImplementationOutput {
    fn tool_name() -> &'static str {
        HANDOFF_TOOL
    }

    fn tool_description() -> &'static str {
        "The final outcome of this worker implementation run."
    }

    fn schema() -> Schema {
        Schema::one_of(
            OneOfSchema::new(
                "outcome",
                "How the implementation run ended. Pick exactly one and send only that outcome's fields.",
            )
            .variant(
                "implemented",
                "You made the code changes; Potlatch commits, pushes, and opens the merge request.",
                mr_metadata_properties(ObjectSchema::new()),
            )
            .variant(
                "existing_mr",
                "You found an already-open merge request that implements this issue; Potlatch tracks it instead of opening a new one.",
                ObjectSchema::new().required_property(
                    "existing_mr_iid",
                    Schema::integer("IID of the existing open merge request."),
                ),
            )
            .variant(
                "wait_dependency",
                "The work is hard-blocked until another issue closes; Potlatch parks this issue and resumes it automatically.",
                ObjectSchema::new().required_property(
                    "depends_on_issue",
                    Schema::integer(
                        "IID of the issue that must close before this work can proceed.",
                    ),
                ),
            )
            .variant(
                "needs_split",
                "The issue is too broad for one merge request and must be split first.",
                blocked_properties(
                    ObjectSchema::new(),
                    "The estimated line count and how to split the issue into smaller, focused issues.",
                ),
            )
            .variant(
                "needs_clarification",
                "The issue is missing information you cannot infer; a human must answer before you can proceed.",
                blocked_properties(
                    ObjectSchema::new(),
                    "Precisely what information is needed and why you cannot proceed without it.",
                ),
            )
            .variant(
                "cannot_implement",
                "The issue cannot be implemented as specified for some other reason (contradictory requirements, no resource-safe approach).",
                blocked_properties(
                    ObjectSchema::new(),
                    "Why the issue cannot be implemented as specified.",
                ),
            ),
        )
    }

    /// Tolerated: an outcome spelled with different case or padding, and an
    /// IID sent as `"#7"` / `"!12"` instead of a number.
    fn normalize(value: &mut serde_json::Value) {
        compat::normalize_tag(value, "outcome");
        compat::normalize_iid(value, "existing_mr_iid");
        compat::normalize_iid(value, "depends_on_issue");
    }
}

impl StructuredOutput for WorkerFeedbackOutput {
    fn tool_name() -> &'static str {
        HANDOFF_TOOL
    }

    fn tool_description() -> &'static str {
        "The final outcome of this merge-request feedback run."
    }

    fn schema() -> Schema {
        Schema::one_of(
            OneOfSchema::new(
                "outcome",
                "How the feedback run ended. Pick exactly one and send only that outcome's fields.",
            )
            .variant(
                "addressed",
                "You handled the reviewer feedback — in code, in merge request metadata, or by explaining that the branch already satisfies it.",
                mr_metadata_properties(ObjectSchema::new())
                    .property(
                        "reason",
                        Schema::string(
                            "Why no code change was needed, when you resolved the feedback without touching the branch.",
                        ),
                    )
                    .property(
                        "public_comment",
                        Schema::string(
                            "The exact concise human-facing reply to post on review threads. It must match the committed code/MR metadata and contain only final comment text: no progress updates, command output, validation section, test/lint lists, or unrelated backlog notes.",
                        ),
                    )
                    .property(
                        "mark_discussions_resolved",
                        Schema::boolean(
                            "Whether Potlatch may mark open review discussions resolved after posting your reply. True only when the request is fully fixed in code/MR metadata, or the branch was verified to already satisfy it and the public comment explains how. For merge-conflict feedback, true only after a pushed branch merges cleanly with no conflict markers. False for partial progress, disagreement, or anything still needing review. Omit to let Potlatch infer from branch changes; set explicitly for metadata-only fixes.",
                        ),
                    )
                    .property(
                        "post_plain_comment",
                        Schema::boolean(
                            "Whether to post a new plain (non-resolvable) merge request comment with `public_comment`. True only when a plain MR comment needs a new public reply; omit or use false when no reply is needed or it would only repeat that no changes were necessary.",
                        ),
                    ),
            )
            .variant(
                "cannot_resolve",
                "The feedback cannot be resolved autonomously; Potlatch abandons the merge request and reports back.",
                blocked_properties(
                    ObjectSchema::new(),
                    "Why the feedback cannot be resolved and what human input is needed.",
                ),
            ),
        )
    }

    /// Tolerated: an outcome spelled with different case or padding, and the
    /// comment-control booleans sent as `"true"`/`"false"` strings.
    fn normalize(value: &mut serde_json::Value) {
        compat::normalize_tag(value, "outcome");
        compat::normalize_bool(value, "mark_discussions_resolved");
        compat::normalize_bool(value, "post_plain_comment");
    }
}

#[derive(Debug, Clone)]
struct WorkerConfig {
    poll_interval_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct WorkerAgentSettings {
    #[serde(default = "default_worker_poll_interval")]
    poll_interval_secs: u64,
}

fn default_worker_poll_interval() -> u64 {
    60
}

/// The single issue a worker is pinned to for its full lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveIssue {
    issue_iid: u64,
    mr_iid: Option<u64>,
    branch_name: Option<String>,
    mr_created: bool,
}

/// A borrowing view over the [`GitLabAgentRuntime`] fields the worker cycle
/// needs. Built fresh from `&GitLabAgentRuntime` at each use site rather
/// than stored, so the worker never owns a second `GitRepo`/`GitLabClient`
/// — and, since it is never stored alongside the runtime it borrows from,
/// it can't become self-referential.
struct AgentState<'a> {
    project_name: &'a str,
    agent_id: &'a str,
    sessions_dir: &'a str,

    git_repo: &'a GitRepo,
    glab: &'a GitLabClient,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &GitLabAgentRuntime) -> AgentState<'_> {
        AgentState {
            project_name: &runtime.project_name,
            agent_id: &runtime.agent_id,
            sessions_dir: &runtime.sessions_dir,
            git_repo: &runtime.git_repo,
            glab: &runtime.gitlab,
        }
    }

    fn session_file_path(&self, issue_iid: u64) -> std::path::PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_issue_{}.json", &self.agent_id, issue_iid))
    }

    fn session_store(&self, issue_iid: u64) -> crate::agents::state::StateStore<SessionFile> {
        crate::agents::state::StateStore::new(self.session_file_path(issue_iid))
    }

    fn load_session(&self, issue_iid: u64) -> Option<SessionFile> {
        // Session corruption has historically been treated as a missing
        // session: tolerant, warn and fall back rather than failing the
        // worker cycle over a persistence problem.
        match self.session_store(issue_iid).load() {
            Ok(session) => session,
            Err(error) => {
                warn!(
                    "{}: Failed to load session for issue #{}: {:#}",
                    &self.agent_id, issue_iid, error
                );
                None
            }
        }
    }

    fn load_implementation_summary(&self, issue_number: u64) -> String {
        if let Some(session) = self.load_session(issue_number)
            && let Some(summary) = session.implementation_summary
        {
            return summary;
        }
        "No previous implementation summary available.".to_string()
    }

    fn cleanup_session(&self, issue_iid: u64) {
        let store = self.session_store(issue_iid);
        if store.path().exists() && store.remove().is_ok() {
            info!(
                "{}: Cleaned up session file for issue #{}",
                &self.agent_id, issue_iid
            );
        }
    }

    fn save_session(&self, issue_iid: u64, mr_iid: u64) -> Result<()> {
        let session = SessionFile {
            issue_iid,
            mr_iid,
            agent_id: Some(self.agent_id.to_string()),
            implementation_summary: None,
        };

        self.session_store(issue_iid)
            .save(&session)
            .context("Failed to write session file")?;
        Ok(())
    }

    fn save_session_with_summary(&self, issue_iid: u64, mr_iid: u64, summary: &str) -> Result<()> {
        let session = SessionFile {
            issue_iid,
            mr_iid,
            agent_id: Some(self.agent_id.to_string()),
            implementation_summary: Some(summary.to_string()),
        };

        self.session_store(issue_iid)
            .save(&session)
            .context("Failed to write session file")?;
        Ok(())
    }

    fn release_worker_hold_pending_gitlab_only(&self, issue_iid: u64) {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (no close)",
            &self.agent_id, issue_iid, WORKER_PENDING_LABEL
        );

        let _ = claim::release(self.glab, ClaimResource::Issue(issue_iid), self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
        self.cleanup_session(issue_iid);
    }

    fn release_worker_hold_review_only(&self, issue_iid: u64) {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (review only)",
            &self.agent_id, issue_iid, WORKER_REVIEW_ONLY_LABEL
        );

        self.clear_resumed_issue_state(issue_iid);
    }

    fn clear_resumed_issue_state(&self, issue_iid: u64) {
        let default_branch = self
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());

        let branch = format!("issue-{}", issue_iid);

        let _ = self.git_repo.reset_hard();
        let _ = self.git_repo.checkout_remote_branch(&default_branch);
        let _ = self.git_repo.delete_local_branch(&branch);

        let _ = claim::release(self.glab, ClaimResource::Issue(issue_iid), self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
    }

    fn abandon_closed_issue(&self, issue_iid: u64, mr_iid: Option<u64>) {
        info!(
            "{}: Issue #{} was closed externally, abandoning work",
            &self.agent_id, issue_iid
        );

        if let Some(mr) = mr_iid {
            let _ = self
                .glab
                .add_mr_comment(mr, "Closing this MR — the linked issue has been closed.");
            let _ = self.glab.close_mr(mr);
        }

        let branch = format!("issue-{}", issue_iid);
        let default_branch = self
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());

        let _ = self.git_repo.reset_hard();
        let _ = self.git_repo.checkout_remote_branch(&default_branch);
        let _ = self.git_repo.delete_local_branch(&branch);
        self.git_repo.delete_remote_branch_best_effort(&branch);
        let _ = claim::release(self.glab, ClaimResource::Issue(issue_iid), self.agent_id);
        let _ = self.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
    }

    fn cleanup_on_shutdown(&self, active: &Option<ActiveIssue>) {
        let Some(a) = active else { return };

        // Always keep the claim and session so this worker resumes on restart.
        // The session file (even with mr_iid=0) tells try_resume_session that
        // this worker owns the issue.
        info!(
            "{}: Preserving claim on issue #{} for restart (MR: {})",
            &self.agent_id,
            a.issue_iid,
            a.mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
        );

        let mr_iid = a.mr_iid.unwrap_or(0);
        let _ = self.save_session(a.issue_iid, mr_iid);
        let _ = self.git_repo.reset_hard();
    }
}

pub(crate) struct WorkerAgent {
    runtime: GitLabAgentRuntime,
    config: WorkerConfig,
    active: Option<ActiveIssue>,
}

impl CoreAgent for WorkerAgent {
    type Settings = WorkerAgentSettings;

    fn name() -> &'static str {
        "worker"
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
                worker_cycle(&state, model, &mut self.active, &shutdown, scope)
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = GitLabAgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        let settings = ctx.settings;
        let config = WorkerConfig {
            poll_interval_secs: settings.poll_interval_secs,
        };
        let scope = crate::agents::scope_label_filter(&runtime.scope_label);
        let active = {
            let state = AgentState::from_runtime(&runtime);
            let mut active =
                try_resume_session(&state, scope).or_else(|| find_claimed_issue(&state, scope));
            active = clear_resumed_issue_if_ignored(&state, active, scope);
            if let Some(ref a) = active {
                if let Some(mr_iid) = a.mr_iid {
                    info!(
                        "{}: Resumed issue #{} with MR !{}",
                        &state.agent_id, a.issue_iid, mr_iid
                    );
                } else {
                    info!(
                        "{}: Resumed issue #{} (no MR yet)",
                        &state.agent_id, a.issue_iid
                    );
                }
            }
            active
        };
        Ok(Self {
            runtime,
            config,
            active,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.runtime.agent_id);
        AgentState::from_runtime(&self.runtime).cleanup_on_shutdown(&self.active);
        info!("{}: Stopped", self.runtime.agent_id);
    }
}

fn clear_resumed_issue_if_ignored(
    state: &AgentState,
    active: Option<ActiveIssue>,
    scope_label: Option<&str>,
) -> Option<ActiveIssue> {
    let active_issue = active?;

    let issue = match state.glab.get_issue(active_issue.issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            warn!(
                "{}: Failed to verify resumed issue #{}: {}, dropping resume state",
                &state.agent_id, active_issue.issue_iid, e
            );

            state.clear_resumed_issue_state(active_issue.issue_iid);

            return None;
        }
    };

    if !issue_in_scope(&issue, scope_label) {
        info!(
            "{}: Resumed issue #{} is outside scope label {:?}, dropping resume state",
            &state.agent_id, issue.iid, scope_label
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    if issue.state != "opened" {
        info!(
            "{}: Resumed issue #{} is {}, dropping resume state",
            &state.agent_id, issue.iid, issue.state
        );
        state.abandon_closed_issue(issue.iid, active_issue.mr_iid);
        return None;
    }

    if issue_has_worker_pending_label(&issue.labels) {
        info!(
            "{}: Resumed issue #{} has `{}` label; releasing worker hold",
            &state.agent_id, issue.iid, WORKER_PENDING_LABEL
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    if issue_has_worker_review_only_label(&issue.labels) {
        state.release_worker_hold_review_only(issue.iid);
        return None;
    }

    if has_worker_resume_abandon_label(&issue.labels) {
        info!(
            "{}: Resumed issue #{} has blocking labels {:?}, abandoning resume state",
            &state.agent_id, issue.iid, issue.labels
        );

        state.clear_resumed_issue_state(issue.iid);
        return None;
    }

    Some(active_issue)
}

// ---------------------------------------------------------------------------
// Worker routing port
// ---------------------------------------------------------------------------

/// Immutable snapshot of the issue fields worker decisions read. Role-local
/// on purpose: the worker machine never holds a general GitLab object, and
/// it cannot mutate what it observed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IssueObservation {
    iid: u64,
    title: String,
    description: String,
    state: String,
    labels: Vec<String>,
}

impl IssueObservation {
    fn from_issue(issue: &Issue) -> Self {
        Self {
            iid: issue.iid,
            title: issue.title.clone(),
            description: issue.description.clone(),
            state: issue.state.clone(),
            labels: issue.labels.clone(),
        }
    }

    fn is_open(&self) -> bool {
        self.state == "opened"
    }

    fn in_scope(&self, scope_label: Option<&str>) -> bool {
        super::issue_labels_in_scope(&self.labels, scope_label)
    }
}

/// The only thing routing needs to know about a merge request it is
/// watching: whether it is still open.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MrStatusObservation {
    iid: u64,
    state: String,
}

impl MrStatusObservation {
    fn is_finished(&self) -> bool {
        self.state == "merged" || self.state == "closed"
    }

    fn is_merged(&self) -> bool {
        self.state == "merged"
    }
}

/// One question the routing machine asks before it decides anything.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerQuery {
    ShutdownRequested,
    Issue {
        issue_iid: u64,
    },
    MergeRequestStatus {
        mr_iid: u64,
    },
    Issues,
    /// The default branch, falling back to `main` — a read the worker never
    /// fails a cycle over.
    DefaultBranchOrMain,
}

/// The answer to one [`WorkerQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerFact {
    ShutdownRequested(bool),
    Issue(IssueObservation),
    MergeRequestStatus(MrStatusObservation),
    Issues(Vec<IssueObservation>),
    DefaultBranchOrMain(String),
}

/// A single side effect the routing machine asks the port to perform.
///
/// Composite variants (`ClearIssueState`, `AbandonClosedIssue`,
/// `ReleaseReviewOnlyHold`, `RunImplementation`, `RunFeedback`,
/// `AdoptOrphanedSession`, `HandleNeedAiWorkerMr`, `ResolveCancelledIssue`)
/// stand for one existing deep helper each: the helper stays the executor
/// and keeps its own internal mutation order, while the decision to run it
/// at this point in the cycle is the machine's.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerAction {
    ResetWorktree,
    CheckoutBranch {
        branch: String,
    },
    DeleteLocalBranch {
        branch: String,
    },
    DeleteRemoteBranch {
        branch: String,
    },
    ReleaseIssueClaim {
        issue_iid: u64,
    },
    RemoveWorkingOnLabel {
        issue_iid: u64,
    },
    RemoveIssueLabel {
        issue_iid: u64,
        label: String,
    },
    CleanupSession {
        issue_iid: u64,
    },
    SaveSession {
        issue_iid: u64,
        mr_iid: u64,
    },
    CloseIssue {
        issue_iid: u64,
    },
    AcquireIssueClaim {
        issue_iid: u64,
    },
    /// Hand the just-won claim over to `active`/the session file: GitLab's
    /// label stays in place across cycles and restarts.
    PreserveIssueClaim {
        issue_iid: u64,
    },
    /// Give the just-won claim straight back (shutdown landed).
    ReleaseAcquiredClaim {
        issue_iid: u64,
    },
    ClearIssueState {
        issue_iid: u64,
    },
    ReleaseReviewOnlyHold {
        issue_iid: u64,
    },
    AbandonClosedIssue {
        issue_iid: u64,
        mr_iid: Option<u64>,
    },
    AdoptOrphanedSession,
    HandleNeedAiWorkerMr,
    RunImplementation {
        issue: Box<IssueObservation>,
    },
    RunFeedback {
        mr_iid: u64,
        linked_issue_iid: Option<u64>,
        comments_only: bool,
    },
    ResolveCancelledIssue {
        issue_iid: u64,
    },
    /// Re-read the issue to see whether the worker may keep tracking it,
    /// releasing a `review-only` hold when it may not.
    CheckIssueTrackable {
        issue_iid: u64,
    },
}

/// Result of a claim attempt, mirroring [`ClaimAcquireOutcome`] without the
/// lease: the lease itself lives in the port, which owns claim effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IssueClaimAttempt {
    Won,
    Lost,
    Interrupted,
}

/// What the port reports after executing one [`WorkerAction`].
enum WorkerOutcome {
    Done,
    Failed(anyhow::Error),
    Claim(IssueClaimAttempt),
    Adopted(Option<ActiveIssue>),
    /// `true` when a labeled MR was handled this cycle.
    HandledNeedAiWorkerMr(bool),
    /// An implementation run: the in-flight issue as the run left it, plus
    /// the error when the run failed. Both are needed, because the cleanup
    /// the cycle performs after a failure depends on how far the run got.
    Implementation {
        current: ActiveIssue,
        error: Option<anyhow::Error>,
    },
    /// A feedback run; `abandoned` is the old `Ok(true)`: the MR was closed
    /// and the issue handed back.
    Feedback {
        abandoned: bool,
    },
    /// Whether an external cancel signal was handled as an intentional stop.
    CancelHandled(bool),
    Trackable(bool),
}

/// The narrow surface the worker's routing cycle needs. Object-safe and
/// role-local: the worker's own observe/execute vocabulary rather than a
/// general GitLab, git, session-store, or model interface.
trait WorkerRoutingPort {
    fn shutdown_requested(&self) -> bool;
    fn issue(&self, issue_iid: u64) -> Result<IssueObservation>;
    fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation>;
    fn issues(&self) -> Result<Vec<IssueObservation>>;
    fn default_branch_or_main(&self) -> String;
    fn execute(&mut self, action: &WorkerAction) -> WorkerOutcome;
}

/// One turn of the routing driver loop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerStep {
    Observe(WorkerQuery),
    Act(WorkerAction),
    Finish,
}

// ---------------------------------------------------------------------------
// Pure routing decisions
// ---------------------------------------------------------------------------

/// Whether the worker still owns the issue it is pinned to, and how it lets
/// go when it does not. Pure — no GitLab call — so the release rules and
/// their precedence can be characterized without a live client. Shared by
/// the MR-watch path and the no-MR re-attempt path, which release on
/// exactly the same conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IssueHold {
    Keep,
    AbandonClosed,
    ClearState(ClearReason),
    ReleaseReviewOnly,
}

/// Why the worker dropped its local state for an issue. Only the log
/// differs between these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClearReason {
    OutOfScope,
    Pending,
}

fn decide_issue_hold(issue: &IssueObservation, scope_label: Option<&str>) -> IssueHold {
    if !issue.is_open() {
        return IssueHold::AbandonClosed;
    }
    if !issue.in_scope(scope_label) {
        return IssueHold::ClearState(ClearReason::OutOfScope);
    }
    if issue_has_worker_pending_label(&issue.labels) {
        return IssueHold::ClearState(ClearReason::Pending);
    }
    if issue_has_worker_review_only_label(&issue.labels) {
        return IssueHold::ReleaseReviewOnly;
    }
    IssueHold::Keep
}

/// What the poll does with one candidate issue. Pure: mirrors the filter
/// chain in the polling loop, including that the dependency check runs
/// before the claim check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateScreening {
    Skip,
    WaitingOnIssue(u64),
    Claimable,
    AlreadyClaimed,
}

fn screen_issue_candidate(
    issue: &IssueObservation,
    scope_label: Option<&str>,
) -> CandidateScreening {
    if should_skip_issue(issue) || !issue.in_scope(scope_label) {
        return CandidateScreening::Skip;
    }
    if let Some(dep_issue_iid) = extract_waiting_on_issue_iid(&issue.labels) {
        return CandidateScreening::WaitingOnIssue(dep_issue_iid);
    }
    if claim::is_claimed(&issue.labels) {
        return CandidateScreening::AlreadyClaimed;
    }
    CandidateScreening::Claimable
}

/// Whether a dependency issue still parks the candidate. A dependency that
/// GitLab no longer has (404) is treated as resolved: the label is dropped
/// and the candidate proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyDecision {
    Resolved,
    StillWaiting,
    Unknown,
}

fn decide_dependency(dependency: Result<&IssueObservation, &anyhow::Error>) -> DependencyDecision {
    match dependency {
        Ok(dep) if dep.state == "closed" => DependencyDecision::Resolved,
        Ok(_) => DependencyDecision::StillWaiting,
        Err(e) if e.to_string().contains("404") => DependencyDecision::Resolved,
        Err(_) => DependencyDecision::Unknown,
    }
}

/// The ordered worktree and GitLab writes that release an issue whose merge
/// request finished. The order never changes; a merge additionally deletes
/// the remote branch and closes the issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinishedMrStep {
    ObserveDefaultBranch,
    ResetWorktree,
    CheckoutDefaultBranch,
    DeleteLocalBranch,
    DeleteRemoteBranch,
    ReleaseClaim,
    RemoveWorkingOnLabel,
    CleanupSession,
    CloseIssue,
}

impl FinishedMrStep {
    fn next(self, merged: bool) -> Option<Self> {
        let next = match self {
            Self::ObserveDefaultBranch => Self::ResetWorktree,
            Self::ResetWorktree => Self::CheckoutDefaultBranch,
            Self::CheckoutDefaultBranch => Self::DeleteLocalBranch,
            Self::DeleteLocalBranch if merged => Self::DeleteRemoteBranch,
            Self::DeleteLocalBranch | Self::DeleteRemoteBranch => Self::ReleaseClaim,
            Self::ReleaseClaim => Self::RemoveWorkingOnLabel,
            Self::RemoveWorkingOnLabel => Self::CleanupSession,
            Self::CleanupSession if merged => Self::CloseIssue,
            Self::CleanupSession | Self::CloseIssue => return None,
        };
        Some(next)
    }
}

/// Which releases follow an implementation run that produced no merge
/// request, or failed. The order of the steps never changes; the plan only
/// says which of them apply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CleanupPlan {
    issue_iid: u64,
    remove_working_label: bool,
    cleanup_session: bool,
    branch: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupStep {
    ReleaseClaim,
    RemoveWorkingOnLabel,
    CleanupSession,
    ObserveDefaultBranch,
    ResetWorktree,
    CheckoutDefaultBranch,
    DeleteLocalBranch,
}

impl CleanupPlan {
    fn first(&self) -> CleanupStep {
        CleanupStep::ReleaseClaim
    }

    fn next(&self, after: CleanupStep) -> Option<CleanupStep> {
        let mut step = after;
        loop {
            step = match step {
                CleanupStep::ReleaseClaim => CleanupStep::RemoveWorkingOnLabel,
                CleanupStep::RemoveWorkingOnLabel => CleanupStep::CleanupSession,
                CleanupStep::CleanupSession => CleanupStep::ObserveDefaultBranch,
                CleanupStep::ObserveDefaultBranch => CleanupStep::ResetWorktree,
                CleanupStep::ResetWorktree => CleanupStep::CheckoutDefaultBranch,
                CleanupStep::CheckoutDefaultBranch => CleanupStep::DeleteLocalBranch,
                CleanupStep::DeleteLocalBranch => return None,
            };
            let applies = match step {
                CleanupStep::RemoveWorkingOnLabel => self.remove_working_label,
                CleanupStep::CleanupSession => self.cleanup_session,
                CleanupStep::ObserveDefaultBranch
                | CleanupStep::ResetWorktree
                | CleanupStep::CheckoutDefaultBranch
                | CleanupStep::DeleteLocalBranch => self.branch.is_some(),
                CleanupStep::ReleaseClaim => true,
            };
            if applies {
                return Some(step);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker routing state machine
// ---------------------------------------------------------------------------

/// Where the routing cycle is. Each variant names the single next
/// observation, action, or pure transition, so
/// [`WorkerRoutingMachine::next_step`] is a function of this plus what the
/// machine already learned — never of the world.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerStage {
    // The active issue's merge request.
    ObserveActiveIssue,
    ObserveActiveMr,
    ClearIssueStateThen(u64),
    AbandonClosedIssueThen(u64, Option<u64>),
    ReleaseReviewOnlyHoldThen(u64),
    ReleaseFinishedMr(FinishedMrStep),
    RunActiveFeedback,
    ReleaseClaimAfterAbandon,
    CleanupSessionAfterAbandon,
    ShutdownAfterFeedbackError(String),
    ResolveCancelledActiveIssue(String),
    // The active issue that never reached a merge request.
    ObserveIssueBeforeReattempt,
    ObserveIssueForReattempt,
    RunReattemptImplementation,
    CheckTrackableAfterReattempt,
    ShutdownAfterImplementationError(String),
    ResolveCancelledImplementation(String),
    Cleanup(CleanupStep),
    // Looking for new work.
    ShutdownBeforePolling,
    AdoptOrphanedSession,
    HandleNeedAiWorkerMr,
    ObserveIssues,
    ShutdownAfterIssues,
    NextCandidate,
    ShutdownBeforeCandidate,
    ScreenCandidate,
    ObserveDependencyIssue(u64),
    RemoveDependencyLabel(u64),
    AcquireCandidateClaim,
    ShutdownAfterCandidateClaim,
    ReleaseCandidateClaimAtShutdown,
    PreserveCandidateClaim,
    SaveCandidatePreSession,
    RunCandidateImplementation,
    CheckTrackableAfterCandidate,
    Finish,
}

/// What the machine does once the cleanup sequence it is running finishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterCleanup {
    LookForNewWork,
    Finish,
}

/// Which of the two implementation paths is in flight. They release
/// differently after a failed or fruitless run, and log differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImplementationPhase {
    /// The active issue never reached a merge request, so the worker is
    /// implementing it again.
    Reattempt,
    /// A freshly claimed issue from the poll.
    Candidate,
}

/// The worker's routing state. Plain data only — no GitLab client, no git
/// repo, no lease — so every decision is a pure function of what the
/// machine has observed.
struct WorkerRoutingMachine<'a> {
    agent_id: &'a str,
    scope_label: Option<&'a str>,
    stage: WorkerStage,
    active: Option<ActiveIssue>,
    /// The issue an implementation run is being performed for, as the run
    /// left it.
    current: Option<ActiveIssue>,
    default_branch: String,
    candidates: std::collections::VecDeque<IssueObservation>,
    candidate: Option<IssueObservation>,
    cleanup: Option<(CleanupPlan, AfterCleanup)>,
    finished_mr_merged: bool,
    implementation_phase: ImplementationPhase,
}

impl<'a> WorkerRoutingMachine<'a> {
    fn new(agent_id: &'a str, scope_label: Option<&'a str>, active: Option<ActiveIssue>) -> Self {
        let stage = Self::entry_stage(active.as_ref());
        Self {
            agent_id,
            scope_label,
            stage,
            active,
            current: None,
            default_branch: String::new(),
            candidates: std::collections::VecDeque::new(),
            candidate: None,
            cleanup: None,
            finished_mr_merged: false,
            implementation_phase: ImplementationPhase::Candidate,
        }
    }

    /// Where a cycle starts: watching the active MR, re-attempting an
    /// implementation that never produced one, or looking for new work.
    fn entry_stage(active: Option<&ActiveIssue>) -> WorkerStage {
        match active {
            Some(a) if a.mr_iid.is_some() => WorkerStage::ObserveActiveIssue,
            Some(a) if !a.mr_created => WorkerStage::ObserveIssueBeforeReattempt,
            _ => WorkerStage::ShutdownBeforePolling,
        }
    }

    fn active_issue(&self) -> &ActiveIssue {
        self.active
            .as_ref()
            .expect("active stages run only while an issue is active")
    }

    fn active_iid(&self) -> u64 {
        self.active_issue().issue_iid
    }

    fn candidate(&self) -> &IssueObservation {
        self.candidate
            .as_ref()
            .expect("candidate stages run only after a candidate is picked")
    }

    /// Drop the active issue and continue to the new-work search, which is
    /// what every release path in the active phases does.
    fn release_active(&mut self) -> WorkerStage {
        self.active = None;
        WorkerStage::ShutdownBeforePolling
    }

    fn start_cleanup(&mut self, plan: CleanupPlan, after: AfterCleanup) -> WorkerStage {
        let first = plan.first();
        self.cleanup = Some((plan, after));
        WorkerStage::Cleanup(first)
    }

    fn advance_cleanup(&mut self, after_step: CleanupStep) -> WorkerStage {
        let (plan, after) = self
            .cleanup
            .as_ref()
            .expect("cleanup stages run only while a plan is set");
        match plan.next(after_step) {
            Some(step) => WorkerStage::Cleanup(step),
            None => {
                let after = *after;
                self.cleanup = None;
                match after {
                    AfterCleanup::LookForNewWork => WorkerStage::ShutdownBeforePolling,
                    AfterCleanup::Finish => WorkerStage::Finish,
                }
            }
        }
    }

    fn cleanup_plan(&self) -> &CleanupPlan {
        &self
            .cleanup
            .as_ref()
            .expect("cleanup stages run only while a plan is set")
            .0
    }

    /// The single next thing to do. Pure: resolves the stages that need no
    /// port interaction before handing back an observation or an action.
    fn next_step(&mut self) -> WorkerStep {
        loop {
            match self.stage.clone() {
                WorkerStage::ObserveActiveIssue | WorkerStage::ObserveIssueBeforeReattempt => {
                    return WorkerStep::Observe(WorkerQuery::Issue {
                        issue_iid: self.active_iid(),
                    });
                }
                WorkerStage::ObserveIssueForReattempt => {
                    return WorkerStep::Observe(WorkerQuery::Issue {
                        issue_iid: self.active_iid(),
                    });
                }
                WorkerStage::ObserveActiveMr => {
                    let mr_iid = self
                        .active_issue()
                        .mr_iid
                        .expect("the MR watch runs only for an active issue with an MR");
                    return WorkerStep::Observe(WorkerQuery::MergeRequestStatus { mr_iid });
                }
                WorkerStage::ClearIssueStateThen(issue_iid) => {
                    return WorkerStep::Act(WorkerAction::ClearIssueState { issue_iid });
                }
                WorkerStage::AbandonClosedIssueThen(issue_iid, mr_iid) => {
                    return WorkerStep::Act(WorkerAction::AbandonClosedIssue { issue_iid, mr_iid });
                }
                WorkerStage::ReleaseReviewOnlyHoldThen(issue_iid) => {
                    return WorkerStep::Act(WorkerAction::ReleaseReviewOnlyHold { issue_iid });
                }
                WorkerStage::ReleaseFinishedMr(step) => {
                    return self.finished_mr_step(step);
                }
                WorkerStage::RunActiveFeedback => {
                    let active = self.active_issue();
                    return WorkerStep::Act(WorkerAction::RunFeedback {
                        mr_iid: active
                            .mr_iid
                            .expect("the feedback run needs the active issue's MR"),
                        linked_issue_iid: Some(active.issue_iid),
                        comments_only: false,
                    });
                }
                WorkerStage::ReleaseClaimAfterAbandon => {
                    return WorkerStep::Act(WorkerAction::ReleaseIssueClaim {
                        issue_iid: self.active_iid(),
                    });
                }
                WorkerStage::CleanupSessionAfterAbandon => {
                    return WorkerStep::Act(WorkerAction::CleanupSession {
                        issue_iid: self.active_iid(),
                    });
                }
                WorkerStage::ShutdownAfterFeedbackError(_)
                | WorkerStage::ShutdownAfterImplementationError(_)
                | WorkerStage::ShutdownBeforePolling
                | WorkerStage::ShutdownAfterIssues
                | WorkerStage::ShutdownBeforeCandidate
                | WorkerStage::ShutdownAfterCandidateClaim => {
                    return WorkerStep::Observe(WorkerQuery::ShutdownRequested);
                }
                WorkerStage::ResolveCancelledActiveIssue(_) => {
                    return WorkerStep::Act(WorkerAction::ResolveCancelledIssue {
                        issue_iid: self.active_iid(),
                    });
                }
                WorkerStage::ResolveCancelledImplementation(_) => {
                    return WorkerStep::Act(WorkerAction::ResolveCancelledIssue {
                        issue_iid: self.implementation_iid(),
                    });
                }
                WorkerStage::RunReattemptImplementation => {
                    let issue = self
                        .candidate
                        .clone()
                        .expect("the re-attempt runs against a freshly observed issue");
                    self.implementation_phase = ImplementationPhase::Reattempt;
                    return WorkerStep::Act(WorkerAction::RunImplementation {
                        issue: Box::new(issue),
                    });
                }
                WorkerStage::CheckTrackableAfterReattempt
                | WorkerStage::CheckTrackableAfterCandidate => {
                    return WorkerStep::Act(WorkerAction::CheckIssueTrackable {
                        issue_iid: self.implementation_iid(),
                    });
                }
                WorkerStage::Cleanup(step) => return self.cleanup_step(step),
                WorkerStage::AdoptOrphanedSession => {
                    return WorkerStep::Act(WorkerAction::AdoptOrphanedSession);
                }
                WorkerStage::HandleNeedAiWorkerMr => {
                    return WorkerStep::Act(WorkerAction::HandleNeedAiWorkerMr);
                }
                WorkerStage::ObserveIssues => return WorkerStep::Observe(WorkerQuery::Issues),
                WorkerStage::NextCandidate => match self.candidates.pop_front() {
                    Some(issue) => {
                        self.candidate = Some(issue);
                        self.stage = WorkerStage::ShutdownBeforeCandidate;
                    }
                    None => self.stage = WorkerStage::Finish,
                },
                WorkerStage::ScreenCandidate => {
                    let issue = self.candidate().clone();
                    self.stage = match screen_issue_candidate(&issue, self.scope_label) {
                        CandidateScreening::Skip => WorkerStage::NextCandidate,
                        CandidateScreening::AlreadyClaimed => {
                            debug!(
                                "{}: Issue #{} already claimed, skipping",
                                self.agent_id, issue.iid
                            );
                            WorkerStage::NextCandidate
                        }
                        CandidateScreening::WaitingOnIssue(dep) => {
                            WorkerStage::ObserveDependencyIssue(dep)
                        }
                        CandidateScreening::Claimable => WorkerStage::AcquireCandidateClaim,
                    };
                }
                WorkerStage::ObserveDependencyIssue(dep_issue_iid) => {
                    return WorkerStep::Observe(WorkerQuery::Issue {
                        issue_iid: dep_issue_iid,
                    });
                }
                WorkerStage::RemoveDependencyLabel(dep_issue_iid) => {
                    return WorkerStep::Act(WorkerAction::RemoveIssueLabel {
                        issue_iid: self.candidate().iid,
                        label: format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{dep_issue_iid}"),
                    });
                }
                WorkerStage::AcquireCandidateClaim => {
                    return WorkerStep::Act(WorkerAction::AcquireIssueClaim {
                        issue_iid: self.candidate().iid,
                    });
                }
                WorkerStage::ReleaseCandidateClaimAtShutdown => {
                    return WorkerStep::Act(WorkerAction::ReleaseAcquiredClaim {
                        issue_iid: self.candidate().iid,
                    });
                }
                WorkerStage::PreserveCandidateClaim => {
                    return WorkerStep::Act(WorkerAction::PreserveIssueClaim {
                        issue_iid: self.candidate().iid,
                    });
                }
                WorkerStage::SaveCandidatePreSession => {
                    return WorkerStep::Act(WorkerAction::SaveSession {
                        issue_iid: self.candidate().iid,
                        mr_iid: 0,
                    });
                }
                WorkerStage::RunCandidateImplementation => {
                    let issue = self.candidate().clone();
                    info!(
                        "{}: Implementing issue #{}: {}",
                        self.agent_id, issue.iid, issue.title
                    );
                    self.implementation_phase = ImplementationPhase::Candidate;
                    return WorkerStep::Act(WorkerAction::RunImplementation {
                        issue: Box::new(issue),
                    });
                }
                WorkerStage::Finish => return WorkerStep::Finish,
            }
        }
    }

    /// The issue an implementation run is in flight for — the in-flight
    /// `current` while it exists, otherwise the candidate it started from.
    fn implementation_iid(&self) -> u64 {
        self.current
            .as_ref()
            .map(|current| current.issue_iid)
            .unwrap_or_else(|| self.candidate().iid)
    }

    fn finished_mr_step(&self, step: FinishedMrStep) -> WorkerStep {
        let active = self.active_issue();
        let issue_iid = active.issue_iid;
        let branch = format!("issue-{issue_iid}");
        match step {
            FinishedMrStep::ObserveDefaultBranch => {
                WorkerStep::Observe(WorkerQuery::DefaultBranchOrMain)
            }
            FinishedMrStep::ResetWorktree => WorkerStep::Act(WorkerAction::ResetWorktree),
            FinishedMrStep::CheckoutDefaultBranch => {
                WorkerStep::Act(WorkerAction::CheckoutBranch {
                    branch: self.default_branch.clone(),
                })
            }
            FinishedMrStep::DeleteLocalBranch => {
                WorkerStep::Act(WorkerAction::DeleteLocalBranch { branch })
            }
            FinishedMrStep::DeleteRemoteBranch => {
                WorkerStep::Act(WorkerAction::DeleteRemoteBranch { branch })
            }
            FinishedMrStep::ReleaseClaim => {
                WorkerStep::Act(WorkerAction::ReleaseIssueClaim { issue_iid })
            }
            FinishedMrStep::RemoveWorkingOnLabel => {
                WorkerStep::Act(WorkerAction::RemoveWorkingOnLabel { issue_iid })
            }
            FinishedMrStep::CleanupSession => {
                WorkerStep::Act(WorkerAction::CleanupSession { issue_iid })
            }
            FinishedMrStep::CloseIssue => WorkerStep::Act(WorkerAction::CloseIssue { issue_iid }),
        }
    }

    fn cleanup_step(&self, step: CleanupStep) -> WorkerStep {
        let plan = self.cleanup_plan();
        let issue_iid = plan.issue_iid;
        match step {
            CleanupStep::ReleaseClaim => {
                WorkerStep::Act(WorkerAction::ReleaseIssueClaim { issue_iid })
            }
            CleanupStep::RemoveWorkingOnLabel => {
                WorkerStep::Act(WorkerAction::RemoveWorkingOnLabel { issue_iid })
            }
            CleanupStep::CleanupSession => {
                WorkerStep::Act(WorkerAction::CleanupSession { issue_iid })
            }
            CleanupStep::ObserveDefaultBranch => {
                WorkerStep::Observe(WorkerQuery::DefaultBranchOrMain)
            }
            CleanupStep::ResetWorktree => WorkerStep::Act(WorkerAction::ResetWorktree),
            CleanupStep::CheckoutDefaultBranch => WorkerStep::Act(WorkerAction::CheckoutBranch {
                branch: self.default_branch.clone(),
            }),
            CleanupStep::DeleteLocalBranch => WorkerStep::Act(WorkerAction::DeleteLocalBranch {
                branch: plan
                    .branch
                    .clone()
                    .expect("branch cleanup steps run only when the plan has a branch"),
            }),
        }
    }

    /// Feed back the answer to the observation the machine just asked for.
    /// `Err` here fails the cycle, exactly where the original code used `?`.
    fn apply_fact(&mut self, fact: Result<WorkerFact>) -> Result<()> {
        match (self.stage.clone(), fact) {
            (WorkerStage::ObserveActiveIssue, observed) => {
                self.stage = self.apply_active_issue_hold(observed, true)?;
            }
            (WorkerStage::ObserveIssueBeforeReattempt, observed) => {
                let next = self.apply_active_issue_hold(observed, false)?;
                self.stage = if matches!(next, WorkerStage::Finish) {
                    // `Keep` for the no-MR path means: re-read the issue and
                    // implement it again from scratch.
                    let issue_iid = self.active_iid();
                    info!(
                        "{}: Active issue #{} has no MR, re-attempting implementation",
                        self.agent_id, issue_iid
                    );
                    WorkerStage::ObserveIssueForReattempt
                } else {
                    next
                };
            }
            (WorkerStage::ObserveIssueForReattempt, Ok(WorkerFact::Issue(issue))) => {
                self.candidate = Some(issue);
                self.current = None;
                self.stage = WorkerStage::RunReattemptImplementation;
            }
            (WorkerStage::ObserveIssueForReattempt, Err(e)) => {
                let issue_iid = self.active_iid();
                warn!(
                    "{}: Failed to fetch issue #{} for re-attempt: {}, releasing",
                    self.agent_id, issue_iid, e
                );
                self.active = None;
                self.stage = self.start_cleanup(
                    CleanupPlan {
                        issue_iid,
                        remove_working_label: true,
                        cleanup_session: true,
                        branch: None,
                    },
                    AfterCleanup::LookForNewWork,
                );
            }
            (WorkerStage::ObserveActiveMr, Ok(WorkerFact::MergeRequestStatus(mr))) => {
                if mr.is_finished() {
                    info!(
                        "{}: MR !{} is {}, releasing issue #{}",
                        self.agent_id,
                        mr.iid,
                        mr.state,
                        self.active_iid()
                    );
                    self.finished_mr_merged = mr.is_merged();
                    self.stage =
                        WorkerStage::ReleaseFinishedMr(FinishedMrStep::ObserveDefaultBranch);
                } else {
                    self.stage = WorkerStage::RunActiveFeedback;
                }
            }
            (WorkerStage::ObserveActiveMr, Err(e)) => {
                let mr_iid = self.active_issue().mr_iid.unwrap_or(0);
                warn!("{}: Failed to check MR !{}: {}", self.agent_id, mr_iid, e);
                // The issue stays active, so the cycle ends here.
                self.stage = WorkerStage::Finish;
            }
            (
                WorkerStage::ReleaseFinishedMr(FinishedMrStep::ObserveDefaultBranch),
                Ok(WorkerFact::DefaultBranchOrMain(branch)),
            ) => {
                self.default_branch = branch;
                self.stage = self.advance_finished_mr(FinishedMrStep::ObserveDefaultBranch);
            }
            (
                WorkerStage::Cleanup(CleanupStep::ObserveDefaultBranch),
                Ok(WorkerFact::DefaultBranchOrMain(branch)),
            ) => {
                self.default_branch = branch;
                self.stage = self.advance_cleanup(CleanupStep::ObserveDefaultBranch);
            }
            (
                WorkerStage::ShutdownAfterFeedbackError(message),
                Ok(WorkerFact::ShutdownRequested(stop)),
            ) => {
                if stop {
                    self.stage = WorkerStage::Finish;
                } else if message.contains(WORKER_AGENT_CANCELLED_MSG) {
                    self.stage = WorkerStage::ResolveCancelledActiveIssue(message);
                } else {
                    error!(
                        "{}: Failed to handle comments for MR !{}: {}",
                        self.agent_id,
                        self.active_issue().mr_iid.unwrap_or(0),
                        message
                    );
                    // The issue is still active, so the cycle ends here.
                    self.stage = WorkerStage::Finish;
                }
            }
            (
                WorkerStage::ShutdownAfterImplementationError(message),
                Ok(WorkerFact::ShutdownRequested(stop)),
            ) => {
                self.stage = self.apply_implementation_error(&message, stop);
            }
            (WorkerStage::ShutdownBeforePolling, Ok(WorkerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    WorkerStage::Finish
                } else if self.active.is_none() {
                    WorkerStage::AdoptOrphanedSession
                } else {
                    WorkerStage::HandleNeedAiWorkerMr
                };
            }
            (WorkerStage::ObserveIssues, Ok(WorkerFact::Issues(issues))) => {
                self.candidates = issues.into();
                self.stage = WorkerStage::ShutdownAfterIssues;
            }
            (WorkerStage::ShutdownAfterIssues, Ok(WorkerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    WorkerStage::Finish
                } else {
                    WorkerStage::NextCandidate
                };
            }
            (WorkerStage::ShutdownBeforeCandidate, Ok(WorkerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    WorkerStage::Finish
                } else {
                    WorkerStage::ScreenCandidate
                };
            }
            (WorkerStage::ObserveDependencyIssue(dep_issue_iid), observed) => {
                let candidate_iid = self.candidate().iid;
                let as_ref = match &observed {
                    Ok(WorkerFact::Issue(issue)) => Ok(issue),
                    Ok(fact) => {
                        anyhow::bail!("worker port answered a dependency observation with {fact:?}")
                    }
                    Err(e) => Err(e),
                };
                self.stage = match decide_dependency(as_ref) {
                    DependencyDecision::Resolved => {
                        match &observed {
                            Ok(_) => info!(
                                "{}: Issue #{} dependency issue #{} closed, resuming",
                                self.agent_id, candidate_iid, dep_issue_iid
                            ),
                            Err(_) => info!(
                                "{}: Issue #{} dependency issue #{} not found (deleted or never existed), dropping dependency label and resuming",
                                self.agent_id, candidate_iid, dep_issue_iid
                            ),
                        }
                        WorkerStage::RemoveDependencyLabel(dep_issue_iid)
                    }
                    DependencyDecision::StillWaiting => {
                        debug!(
                            "{}: Issue #{} waiting on issue #{} (not yet closed), skipping",
                            self.agent_id, candidate_iid, dep_issue_iid
                        );
                        WorkerStage::NextCandidate
                    }
                    DependencyDecision::Unknown => {
                        warn!(
                            "{}: Issue #{} waiting on issue #{} — failed to check dependency state: {}, skipping this cycle",
                            self.agent_id,
                            candidate_iid,
                            dep_issue_iid,
                            observed.err().map(|e| e.to_string()).unwrap_or_default()
                        );
                        WorkerStage::NextCandidate
                    }
                };
            }
            (WorkerStage::ShutdownAfterCandidateClaim, Ok(WorkerFact::ShutdownRequested(stop))) => {
                self.stage = if stop {
                    WorkerStage::ReleaseCandidateClaimAtShutdown
                } else {
                    WorkerStage::PreserveCandidateClaim
                };
            }
            (stage, Ok(fact)) => {
                anyhow::bail!("worker port answered {stage:?} with {fact:?}");
            }
            (_, Err(e)) => return Err(e),
        }
        Ok(())
    }

    /// Apply the shared release rules for the issue the worker is pinned
    /// to. `Keep` maps to [`WorkerStage::Finish`] for the caller to
    /// reinterpret, since the two active phases continue differently.
    fn apply_active_issue_hold(
        &mut self,
        observed: Result<WorkerFact>,
        watching_mr: bool,
    ) -> Result<WorkerStage> {
        let issue_iid = self.active_iid();
        let mr_iid = self.active_issue().mr_iid;
        let issue = match observed {
            Ok(WorkerFact::Issue(issue)) => issue,
            Ok(fact) => anyhow::bail!("worker port answered an issue observation with {fact:?}"),
            Err(e) => {
                if watching_mr {
                    warn!(
                        "{}: Failed to verify active issue #{}: {}, releasing worker state",
                        self.agent_id, issue_iid, e
                    );
                } else {
                    warn!(
                        "{}: Failed to verify active issue #{}: {}, releasing",
                        self.agent_id, issue_iid, e
                    );
                }
                self.active = None;
                return Ok(WorkerStage::ClearIssueStateThen(issue_iid));
            }
        };

        Ok(match decide_issue_hold(&issue, self.scope_label) {
            IssueHold::Keep => {
                if watching_mr {
                    WorkerStage::ObserveActiveMr
                } else {
                    WorkerStage::Finish
                }
            }
            IssueHold::AbandonClosed => {
                self.active = None;
                WorkerStage::AbandonClosedIssueThen(
                    issue_iid,
                    if watching_mr { mr_iid } else { None },
                )
            }
            IssueHold::ClearState(reason) => {
                match reason {
                    ClearReason::OutOfScope if watching_mr => info!(
                        "{}: Issue #{} left scope label {:?}, releasing worker state",
                        self.agent_id, issue_iid, self.scope_label
                    ),
                    ClearReason::OutOfScope => info!(
                        "{}: Active issue #{} left scope label {:?}, releasing",
                        self.agent_id, issue_iid, self.scope_label
                    ),
                    ClearReason::Pending if watching_mr => info!(
                        "{}: Issue #{} has `{}` — stopping MR watch (issue stays open)",
                        self.agent_id, issue_iid, WORKER_PENDING_LABEL
                    ),
                    ClearReason::Pending => info!(
                        "{}: Active issue #{} has `{}` — yielding (issue stays open)",
                        self.agent_id, issue_iid, WORKER_PENDING_LABEL
                    ),
                }
                self.active = None;
                WorkerStage::ClearIssueStateThen(issue_iid)
            }
            IssueHold::ReleaseReviewOnly => {
                self.active = None;
                WorkerStage::ReleaseReviewOnlyHoldThen(issue_iid)
            }
        })
    }

    fn advance_finished_mr(&mut self, after_step: FinishedMrStep) -> WorkerStage {
        match after_step.next(self.finished_mr_merged) {
            Some(step) => WorkerStage::ReleaseFinishedMr(step),
            None => self.release_active(),
        }
    }

    /// How a failed implementation run ends. Shutdown keeps the issue
    /// active so the shutdown hook can persist it; an external cancel is
    /// resolved by the cancel helper; anything else is logged and released.
    fn apply_implementation_error(&mut self, message: &str, shutting_down: bool) -> WorkerStage {
        if shutting_down {
            // Keep the in-flight issue active so the shutdown hook can
            // persist the claim and session for the next start.
            self.active = self.current.clone();
            return WorkerStage::Finish;
        }
        if message.contains(WORKER_AGENT_CANCELLED_MSG) {
            return WorkerStage::ResolveCancelledImplementation(message.to_string());
        }
        self.release_after_failed_implementation(message)
    }

    fn release_after_failed_implementation(&mut self, message: &str) -> WorkerStage {
        let current = self
            .current
            .clone()
            .expect("an implementation run records its in-flight issue");
        match self.implementation_phase {
            ImplementationPhase::Reattempt => {
                error!(
                    "{}: Failed to re-process issue #{}: {}",
                    self.agent_id, current.issue_iid, message
                );
                self.active = None;
                self.start_cleanup(
                    CleanupPlan {
                        issue_iid: current.issue_iid,
                        remove_working_label: true,
                        cleanup_session: true,
                        branch: current.branch_name.clone(),
                    },
                    AfterCleanup::Finish,
                )
            }
            ImplementationPhase::Candidate => {
                error!(
                    "{}: Failed to process issue #{}: {}",
                    self.agent_id, current.issue_iid, message
                );
                self.start_cleanup(
                    CleanupPlan {
                        issue_iid: current.issue_iid,
                        remove_working_label: true,
                        cleanup_session: false,
                        branch: current.branch_name.clone(),
                    },
                    AfterCleanup::Finish,
                )
            }
        }
    }

    /// Feed back the outcome of the action the machine just asked for.
    fn apply_outcome(&mut self, outcome: WorkerOutcome) -> Result<()> {
        match (self.stage.clone(), outcome) {
            (
                WorkerStage::ClearIssueStateThen(_)
                | WorkerStage::AbandonClosedIssueThen(_, _)
                | WorkerStage::ReleaseReviewOnlyHoldThen(_),
                WorkerOutcome::Done,
            ) => {
                self.stage = WorkerStage::ShutdownBeforePolling;
            }
            (WorkerStage::ReleaseFinishedMr(step), WorkerOutcome::Done) => {
                self.stage = self.advance_finished_mr(step);
            }
            (WorkerStage::RunActiveFeedback, WorkerOutcome::Feedback { abandoned }) => {
                if abandoned {
                    let active = self.active_issue();
                    info!(
                        "{}: Issue #{} abandoned, MR !{} closed",
                        self.agent_id,
                        active.issue_iid,
                        active.mr_iid.unwrap_or(0)
                    );
                    self.stage = WorkerStage::ReleaseClaimAfterAbandon;
                } else {
                    // The issue stays active, so the cycle ends here.
                    self.stage = WorkerStage::Finish;
                }
            }
            (WorkerStage::RunActiveFeedback, WorkerOutcome::Failed(e)) => {
                self.stage = WorkerStage::ShutdownAfterFeedbackError(format!("{e:#}"));
            }
            (WorkerStage::ReleaseClaimAfterAbandon, WorkerOutcome::Done) => {
                self.stage = WorkerStage::CleanupSessionAfterAbandon;
            }
            (WorkerStage::CleanupSessionAfterAbandon, WorkerOutcome::Done) => {
                self.stage = self.release_active();
            }
            (
                WorkerStage::ResolveCancelledActiveIssue(message),
                WorkerOutcome::CancelHandled(handled),
            ) => {
                if handled {
                    self.stage = self.release_active_and_finish();
                } else {
                    error!(
                        "{}: Failed to handle comments for MR !{}: {}",
                        self.agent_id,
                        self.active_issue().mr_iid.unwrap_or(0),
                        message
                    );
                    self.stage = WorkerStage::Finish;
                }
            }
            (
                WorkerStage::ResolveCancelledImplementation(message),
                WorkerOutcome::CancelHandled(handled),
            ) => {
                self.stage = if handled {
                    self.active = None;
                    WorkerStage::Finish
                } else {
                    self.release_after_failed_implementation(&message)
                };
            }
            (
                WorkerStage::RunReattemptImplementation | WorkerStage::RunCandidateImplementation,
                WorkerOutcome::Implementation { current, error },
            ) => {
                let mr_created = current.mr_created;
                self.current = Some(current);
                self.stage = match error {
                    Some(e) => WorkerStage::ShutdownAfterImplementationError(format!("{e:#}")),
                    None if mr_created => match self.implementation_phase {
                        ImplementationPhase::Reattempt => WorkerStage::CheckTrackableAfterReattempt,
                        ImplementationPhase::Candidate => WorkerStage::CheckTrackableAfterCandidate,
                    },
                    None => {
                        let issue_iid = self.implementation_iid();
                        let cleanup_session =
                            matches!(self.implementation_phase, ImplementationPhase::Reattempt);
                        self.active = None;
                        self.start_cleanup(
                            CleanupPlan {
                                issue_iid,
                                remove_working_label: false,
                                cleanup_session,
                                branch: None,
                            },
                            AfterCleanup::Finish,
                        )
                    }
                };
            }
            (
                WorkerStage::CheckTrackableAfterReattempt
                | WorkerStage::CheckTrackableAfterCandidate,
                WorkerOutcome::Trackable(trackable),
            ) => {
                self.active = trackable.then(|| {
                    self.current
                        .clone()
                        .expect("an implementation run records its in-flight issue")
                });
                self.stage = WorkerStage::Finish;
            }
            (WorkerStage::Cleanup(step), WorkerOutcome::Done) => {
                self.stage = self.advance_cleanup(step);
            }
            (WorkerStage::AdoptOrphanedSession, WorkerOutcome::Adopted(adopted)) => {
                self.stage = match adopted {
                    Some(active) => {
                        info!(
                            "{}: Adopted orphaned issue #{} with MR !{}",
                            self.agent_id,
                            active.issue_iid,
                            active.mr_iid.unwrap_or(0)
                        );
                        self.active = Some(active);
                        WorkerStage::Finish
                    }
                    None => WorkerStage::HandleNeedAiWorkerMr,
                };
            }
            (WorkerStage::HandleNeedAiWorkerMr, WorkerOutcome::HandledNeedAiWorkerMr(handled)) => {
                self.stage = if handled {
                    WorkerStage::Finish
                } else {
                    WorkerStage::ObserveIssues
                };
            }
            (WorkerStage::RemoveDependencyLabel(_), WorkerOutcome::Done) => {
                let candidate = self.candidate();
                self.stage = if claim::is_claimed(&candidate.labels) {
                    debug!(
                        "{}: Issue #{} already claimed, skipping",
                        self.agent_id, candidate.iid
                    );
                    WorkerStage::NextCandidate
                } else {
                    WorkerStage::AcquireCandidateClaim
                };
            }
            (WorkerStage::AcquireCandidateClaim, WorkerOutcome::Claim(attempt)) => {
                self.stage = match attempt {
                    IssueClaimAttempt::Won => WorkerStage::ShutdownAfterCandidateClaim,
                    IssueClaimAttempt::Lost => {
                        info!(
                            "{}: Failed to claim issue #{}, skipping",
                            self.agent_id,
                            self.candidate().iid
                        );
                        WorkerStage::NextCandidate
                    }
                    IssueClaimAttempt::Interrupted => WorkerStage::Finish,
                };
            }
            (WorkerStage::ReleaseCandidateClaimAtShutdown, WorkerOutcome::Done) => {
                self.stage = WorkerStage::Finish;
            }
            (WorkerStage::PreserveCandidateClaim, WorkerOutcome::Done) => {
                self.stage = WorkerStage::SaveCandidatePreSession;
            }
            (WorkerStage::SaveCandidatePreSession, WorkerOutcome::Done) => {
                self.stage = WorkerStage::RunCandidateImplementation;
            }
            // The reads and writes the cycle performed with `?` abort it.
            (
                WorkerStage::HandleNeedAiWorkerMr | WorkerStage::AcquireCandidateClaim,
                WorkerOutcome::Failed(e),
            ) => return Err(e),
            (stage, _) => {
                anyhow::bail!("worker port reported an unexpected outcome for {stage:?}");
            }
        }
        Ok(())
    }

    /// Drop the active issue and end the cycle.
    fn release_active_and_finish(&mut self) -> WorkerStage {
        self.active = None;
        WorkerStage::Finish
    }
}

/// Ask the routing port one question.
fn observe_worker(port: &dyn WorkerRoutingPort, query: &WorkerQuery) -> Result<WorkerFact> {
    Ok(match query {
        WorkerQuery::ShutdownRequested => WorkerFact::ShutdownRequested(port.shutdown_requested()),
        WorkerQuery::Issue { issue_iid } => WorkerFact::Issue(port.issue(*issue_iid)?),
        WorkerQuery::MergeRequestStatus { mr_iid } => {
            WorkerFact::MergeRequestStatus(port.merge_request_status(*mr_iid)?)
        }
        WorkerQuery::Issues => WorkerFact::Issues(port.issues()?),
        WorkerQuery::DefaultBranchOrMain => {
            WorkerFact::DefaultBranchOrMain(port.default_branch_or_main())
        }
    })
}

/// Run the worker's routing machine to completion: observe, decide one
/// step, execute it, feed the result back.
fn drive_worker_routing(
    machine: &mut WorkerRoutingMachine,
    port: &mut dyn WorkerRoutingPort,
) -> Result<()> {
    loop {
        match machine.next_step() {
            WorkerStep::Observe(query) => {
                let fact = observe_worker(port, &query);
                machine.apply_fact(fact)?;
            }
            WorkerStep::Act(action) => {
                let outcome = port.execute(&action);
                machine.apply_outcome(outcome)?;
            }
            WorkerStep::Finish => return Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Live worker routing port
// ---------------------------------------------------------------------------

/// The routing port backed by the real runtime: the worker's only place
/// where a routing decision meets git, GitLab, the session store, or the
/// model.
struct LiveWorkerRoutingPort<'a> {
    state: &'a AgentState<'a>,
    model: &'a AgentModel,
    shutdown: &'a AtomicBool,
    scope_label: Option<&'a str>,
    /// The claim won for the candidate currently being screened, until the
    /// machine preserves or releases it.
    candidate_lease: Option<ClaimLease>,
}

impl WorkerRoutingPort for LiveWorkerRoutingPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn issue(&self, issue_iid: u64) -> Result<IssueObservation> {
        Ok(IssueObservation::from_issue(
            &self.state.glab.get_issue(issue_iid)?,
        ))
    }

    fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation> {
        let mr = self.state.glab.get_merge_request(mr_iid)?;
        Ok(MrStatusObservation {
            iid: mr.iid,
            state: mr.state,
        })
    }

    fn issues(&self) -> Result<Vec<IssueObservation>> {
        Ok(self
            .state
            .glab
            .list_issues()?
            .iter()
            .map(IssueObservation::from_issue)
            .collect())
    }

    fn default_branch_or_main(&self) -> String {
        self.state
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string())
    }

    fn execute(&mut self, action: &WorkerAction) -> WorkerOutcome {
        let state = self.state;
        match action {
            WorkerAction::ResetWorktree => {
                let _ = state.git_repo.reset_hard();
                WorkerOutcome::Done
            }
            WorkerAction::CheckoutBranch { branch } => {
                let _ = state.git_repo.checkout_remote_branch(branch);
                WorkerOutcome::Done
            }
            WorkerAction::DeleteLocalBranch { branch } => {
                let _ = state.git_repo.delete_local_branch(branch);
                WorkerOutcome::Done
            }
            WorkerAction::DeleteRemoteBranch { branch } => {
                state.git_repo.delete_remote_branch_best_effort(branch);
                WorkerOutcome::Done
            }
            WorkerAction::ReleaseIssueClaim { issue_iid } => {
                let _ =
                    claim::release(state.glab, ClaimResource::Issue(*issue_iid), state.agent_id);
                WorkerOutcome::Done
            }
            WorkerAction::RemoveWorkingOnLabel { issue_iid } => {
                let _ = state.glab.remove_issue_label(*issue_iid, WORKING_ON_LABEL);
                WorkerOutcome::Done
            }
            WorkerAction::RemoveIssueLabel { issue_iid, label } => {
                let _ = state.glab.remove_issue_label(*issue_iid, label);
                WorkerOutcome::Done
            }
            WorkerAction::CleanupSession { issue_iid } => {
                state.cleanup_session(*issue_iid);
                WorkerOutcome::Done
            }
            WorkerAction::SaveSession { issue_iid, mr_iid } => {
                let _ = state.save_session(*issue_iid, *mr_iid);
                WorkerOutcome::Done
            }
            WorkerAction::CloseIssue { issue_iid } => {
                close_issue_best_effort(state.glab, *issue_iid);
                WorkerOutcome::Done
            }
            WorkerAction::AcquireIssueClaim { issue_iid } => match claim::acquire(
                state.glab,
                ClaimResource::Issue(*issue_iid),
                state.agent_id,
                self.shutdown,
            ) {
                Ok(ClaimAcquireOutcome::Won(lease)) => {
                    self.candidate_lease = Some(lease);
                    WorkerOutcome::Claim(IssueClaimAttempt::Won)
                }
                Ok(ClaimAcquireOutcome::Lost) => WorkerOutcome::Claim(IssueClaimAttempt::Lost),
                Ok(ClaimAcquireOutcome::Interrupted) => {
                    WorkerOutcome::Claim(IssueClaimAttempt::Interrupted)
                }
                Err(e) => WorkerOutcome::Failed(e),
            },
            WorkerAction::PreserveIssueClaim { .. } => {
                // Ownership is tracked by issue IID from here on (in
                // `active` and the session file): GitLab's claim label is
                // the source of truth across cycles and restarts.
                if let Some(lease) = self.candidate_lease.take() {
                    lease.preserve();
                }
                WorkerOutcome::Done
            }
            WorkerAction::ReleaseAcquiredClaim { .. } => {
                if let Some(lease) = self.candidate_lease.take() {
                    let _ = lease.release(state.glab);
                }
                WorkerOutcome::Done
            }
            WorkerAction::ClearIssueState { issue_iid } => {
                state.clear_resumed_issue_state(*issue_iid);
                WorkerOutcome::Done
            }
            WorkerAction::ReleaseReviewOnlyHold { issue_iid } => {
                state.release_worker_hold_review_only(*issue_iid);
                WorkerOutcome::Done
            }
            WorkerAction::AbandonClosedIssue { issue_iid, mr_iid } => {
                state.abandon_closed_issue(*issue_iid, *mr_iid);
                WorkerOutcome::Done
            }
            WorkerAction::AdoptOrphanedSession => WorkerOutcome::Adopted(
                try_adopt_orphaned_session(state, self.shutdown, self.scope_label),
            ),
            WorkerAction::HandleNeedAiWorkerMr => {
                match try_handle_need_ai_worker_mr(
                    state,
                    self.model,
                    self.shutdown,
                    self.scope_label,
                ) {
                    Ok(handled) => WorkerOutcome::HandledNeedAiWorkerMr(handled),
                    Err(e) => WorkerOutcome::Failed(e),
                }
            }
            WorkerAction::RunImplementation { issue } => {
                let mut current = ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: None,
                    branch_name: None,
                    mr_created: false,
                };
                let error =
                    process_issue(state, self.model, issue, &mut current, self.scope_label).err();
                WorkerOutcome::Implementation { current, error }
            }
            WorkerAction::RunFeedback {
                mr_iid,
                linked_issue_iid,
                comments_only,
            } => match handle_mr_comments(
                state,
                self.model,
                *mr_iid,
                *linked_issue_iid,
                *comments_only,
            ) {
                Ok(abandoned) => WorkerOutcome::Feedback { abandoned },
                Err(e) => WorkerOutcome::Failed(e),
            },
            WorkerAction::ResolveCancelledIssue { issue_iid } => {
                let error = anyhow::anyhow!("{}", WORKER_AGENT_CANCELLED_MSG);
                WorkerOutcome::CancelHandled(handle_worker_issue_processing_cancelled(
                    state, *issue_iid, &error,
                ))
            }
            WorkerAction::CheckIssueTrackable { issue_iid } => {
                WorkerOutcome::Trackable(should_track_worker_issue(state, *issue_iid))
            }
        }
    }
}

/// The worker's routing cycle: pick up where the last cycle left off, or
/// find new work. Observes, decides one step, executes it, feeds the result
/// back — see [`WorkerRoutingMachine`].
fn worker_cycle(
    state: &AgentState,
    model: &AgentModel,
    active: &mut Option<ActiveIssue>,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut machine = WorkerRoutingMachine::new(state.agent_id, scope_label, active.take());
    let mut port = LiveWorkerRoutingPort {
        state,
        model,
        shutdown,
        scope_label,
        candidate_lease: None,
    };
    let result = drive_worker_routing(&mut machine, &mut port);
    *active = machine.active.take();
    result
}

fn mr_has_label(mr: &crate::agents::gitlab::MergeRequest, label: &str) -> bool {
    mr.labels
        .as_ref()
        .is_some_and(|ls| ls.iter().any(|l| l.eq_ignore_ascii_case(label)))
}

/// Handle open MRs labeled `need-ai-worker` as worker tasks:
/// claim MR directly, address unresolved discussions, then release claim.
fn try_handle_need_ai_worker_mr(
    state: &AgentState,
    model: &AgentModel,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<bool> {
    let mut mrs = state.glab.list_merge_requests()?;
    mrs.sort_by_key(|mr| mr.iid);
    for mr in mrs {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(false);
        }
        if mr.state != "opened" {
            continue;
        }
        if !mr_has_label(&mr, NEED_AI_WORKER_LABEL) {
            continue;
        }
        if !super::mr_in_scope(&mr, scope_label) {
            continue;
        }
        if claim::is_mr_claimed(&mr.labels) {
            continue;
        }
        let unresolved = state.glab.get_unresolved_discussion_ids(mr.iid)?;
        if unresolved.is_empty() {
            continue;
        }
        let lease = match claim::acquire(
            state.glab,
            ClaimResource::MergeRequest(mr.iid),
            state.agent_id,
            shutdown,
        )? {
            ClaimAcquireOutcome::Won(lease) => lease,
            ClaimAcquireOutcome::Lost => continue,
            ClaimAcquireOutcome::Interrupted => return Ok(false),
        };
        if shutdown.load(Ordering::SeqCst) {
            let _ = lease.release(state.glab);
            return Ok(false);
        }
        info!(
            "{}: Handling labeled MR !{} (`{}`) with {} unresolved discussion(s)",
            &state.agent_id,
            mr.iid,
            NEED_AI_WORKER_LABEL,
            unresolved.len()
        );
        let result = handle_mr_comments(state, model, mr.iid, None, true);
        let _ = lease.release(state.glab);
        result?;
        return Ok(true);
    }
    Ok(false)
}

fn should_skip_issue(issue: &IssueObservation) -> bool {
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return true;
    }

    if issue
        .labels
        .contains(&super::labels::DO_NOT_IMPLEMENT.to_string())
    {
        return true;
    }

    if issue_has_worker_pending_label(&issue.labels) {
        return true;
    }

    if issue_has_worker_review_only_label(&issue.labels) {
        return true;
    }

    if has_worker_skip_label(&issue.labels) {
        return true;
    }

    false
}

fn issue_has_worker_pending_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_PENDING_LABEL.to_string())
}

fn issue_has_worker_review_only_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_REVIEW_ONLY_LABEL.to_string())
}

fn worker_should_cancel_issue_processing(issue: &IssueObservation) -> bool {
    issue.state != "opened"
        || issue_has_worker_review_only_label(&issue.labels)
        || issue_has_worker_pending_label(&issue.labels)
}

fn worker_issue_cancel_check(
    glab: GitLabClient,
    issue_iid: u64,
) -> Arc<dyn Fn() -> bool + Send + Sync> {
    Arc::new(move || {
        glab.get_issue(issue_iid).ok().is_some_and(|issue| {
            worker_should_cancel_issue_processing(&IssueObservation::from_issue(&issue))
        })
    })
}

fn is_worker_agent_cancelled(err: &anyhow::Error) -> bool {
    err.to_string().contains(WORKER_AGENT_CANCELLED_MSG)
}

/// Stop in-flight work when the issue was closed, switched to review-only,
/// or marked `pending` by a human. Returns true when the error was handled
/// as an intentional stop.
fn handle_worker_issue_processing_cancelled(
    state: &AgentState,
    issue_iid: u64,
    err: &anyhow::Error,
) -> bool {
    if !is_worker_agent_cancelled(err) {
        return false;
    }

    match state.glab.get_issue(issue_iid) {
        Ok(issue) if issue_has_worker_pending_label(&issue.labels) => {
            info!(
                "{}: Issue #{} marked `{}` mid-run — releasing worker hold (issue stays open)",
                &state.agent_id, issue_iid, WORKER_PENDING_LABEL
            );
            // Leave the issue open and the `pending` label in place; just drop
            // our claim and session so another worker can pick it up once a
            // human removes `pending`.
            let _ = claim::release(state.glab, ClaimResource::Issue(issue_iid), state.agent_id);
            let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
            state.cleanup_session(issue_iid);
            true
        }
        Ok(issue) if issue_has_worker_review_only_label(&issue.labels) => {
            state.release_worker_hold_review_only(issue_iid);
            true
        }
        Ok(issue) if issue.state != "opened" => {
            state.abandon_closed_issue(issue_iid, None);
            true
        }
        Ok(_) => {
            info!(
                "{}: Stopped work on issue #{} after external cancel signal",
                &state.agent_id, issue_iid
            );
            let _ = claim::release(state.glab, ClaimResource::Issue(issue_iid), state.agent_id);
            let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
            state.cleanup_session(issue_iid);
            true
        }
        Err(e) => {
            warn!(
                "{}: Cancelled while working on issue #{} but failed to re-fetch issue: {}",
                &state.agent_id, issue_iid, e
            );
            let _ = claim::release(state.glab, ClaimResource::Issue(issue_iid), state.agent_id);
            state.cleanup_session(issue_iid);
            true
        }
    }
}

fn stop_worker_issue_if_review_only(state: &AgentState, issue_iid: u64) -> bool {
    let Ok(issue) = state.glab.get_issue(issue_iid) else {
        return false;
    };
    if !issue_has_worker_review_only_label(&issue.labels) {
        return false;
    }
    state.release_worker_hold_review_only(issue_iid);
    true
}

fn should_track_worker_issue(state: &AgentState, issue_iid: u64) -> bool {
    if state
        .glab
        .get_issue(issue_iid)
        .is_ok_and(|issue| issue_has_worker_review_only_label(&issue.labels))
    {
        state.release_worker_hold_review_only(issue_iid);
        return false;
    }

    true
}

fn close_issue_best_effort(gitlab: &GitLabClient, issue_iid: u64) {
    if let Err(e) = gitlab.close_issue(issue_iid) {
        debug!(
            "Issue #{}: could not close after merged MR (may already be closed): {}",
            issue_iid, e
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosesLinkedMr {
    Open(u64),
    Merged,
}

/// MRs whose description contains `Closes #issue_iid` (case-insensitive), preferring an open MR.
fn closes_keyword_mr_status(gitlab: &GitLabClient, issue_iid: u64) -> Option<ClosesLinkedMr> {
    let mrs = gitlab.list_merge_requests().ok()?;
    let mut opens: Vec<u64> = Vec::new();
    let mut any_merged = false;
    for mr in mrs {
        if !crate::agents::gitlab::mr_description_closes_issue(&mr.description, issue_iid) {
            continue;
        }
        match mr.state.as_str() {
            "opened" => opens.push(mr.iid),
            "merged" => any_merged = true,
            _ => {}
        }
    }
    if let Some(&best) = opens.iter().min() {
        return Some(ClosesLinkedMr::Open(best));
    }
    if any_merged {
        return Some(ClosesLinkedMr::Merged);
    }
    None
}

#[derive(Debug, Clone, Copy)]
enum ResolvedTrackedMr {
    Track(u64),
    MergedCloseIssue,
    None,
}

/// Prefer an open MR linked via `Closes #issue`, then session MR if still open, then branch `issue-N`.
fn resolve_tracked_mr_for_worker_issue(
    gitlab: &GitLabClient,
    issue_iid: u64,
    session_mr_iid: u64,
) -> ResolvedTrackedMr {
    match closes_keyword_mr_status(gitlab, issue_iid) {
        Some(ClosesLinkedMr::Open(id)) => return ResolvedTrackedMr::Track(id),
        Some(ClosesLinkedMr::Merged) => return ResolvedTrackedMr::MergedCloseIssue,
        None => {}
    }
    if session_mr_iid > 0
        && let Ok(mr) = gitlab.get_merge_request(session_mr_iid)
        && mr.state == "opened"
    {
        return ResolvedTrackedMr::Track(session_mr_iid);
    }
    if let Some(id) = find_open_mr_for_issue(gitlab, issue_iid) {
        return ResolvedTrackedMr::Track(id);
    }
    ResolvedTrackedMr::None
}

fn has_worker_resume_abandon_label(labels: &[String]) -> bool {
    labels.contains(&ACTION_REQUIRED_LABEL.to_string())
        || labels.contains(&PMO_PROCESSED_LABEL.to_string())
        || labels.contains(&PMO_PENDING_LABEL.to_string())
}

fn has_worker_skip_label(labels: &[String]) -> bool {
    labels.contains(&WORKING_ON_LABEL.to_string()) || has_worker_resume_abandon_label(labels)
}

// ---------------------------------------------------------------------------
// Implementation progression port
// ---------------------------------------------------------------------------

/// One question the implementation progression asks before it decides
/// anything. Reads only — every write is an [`ImplAction`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImplQuery {
    /// The merge request linked to the issue by a `Closes #` keyword.
    ClosesLinkedMr,
    /// An open merge request on the issue's own branch.
    OpenMrForIssue,
    /// The default branch; a read the run cannot proceed without.
    DefaultBranch,
    /// The default branch, falling back to `main` — used by the release
    /// paths, which never fail the run over it.
    DefaultBranchOrMain,
    RemoteBranchExists,
    DiffAgainstDefault,
    StagedChanges,
    /// The state of a merge request the model claims already implements the
    /// issue; `None` when it could not be fetched at all.
    MergeRequestState {
        mr_iid: u64,
    },
    DependencyClosed {
        issue_iid: u64,
    },
    IssueComments,
}

/// The answer to one [`ImplQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImplFact {
    ClosesLinkedMr(Option<ClosesLinkedMr>),
    OpenMrForIssue(Option<u64>),
    DefaultBranch(String),
    DefaultBranchOrMain(String),
    RemoteBranchExists(bool),
    DiffAgainstDefault(bool),
    StagedChanges(bool),
    MergeRequestState(Option<String>),
    DependencyClosed(bool),
    IssueComments(String),
}

/// One side effect in the implementation progression: the workspace
/// preparation, the model invocation, and every GitLab write are all
/// actions the machine asks for one at a time.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImplAction {
    /// Release the issue if a human asked for review-only in the meantime.
    StopIfReviewOnly,
    AddWorkingOnLabel,
    /// The same label write, where the original code failed the run on it.
    RequireWorkingOnLabel,
    RemoveWorkingOnLabel,
    AddIssueLabel {
        label: String,
    },
    AddIssueComment {
        body: String,
    },
    ReleaseIssueClaim,
    CleanupSession,
    CloseIssue,
    SaveSession {
        mr_iid: u64,
    },
    RequireSaveSession {
        mr_iid: u64,
    },
    RequireSaveSessionWithSummary {
        mr_iid: u64,
        summary: String,
    },
    FetchRemote,
    ResetWorktree,
    CheckoutBranch {
        branch: String,
    },
    /// The same checkout on a release path, where a failure is tolerated.
    CheckoutBranchBestEffort {
        branch: String,
    },
    DeleteLocalBranch {
        branch: String,
    },
    DeleteRemoteBranch {
        branch: String,
    },
    CreateBranchFrom {
        branch: String,
        base: String,
    },
    MergeBaseIntoBranch {
        base: String,
    },
    /// Workspace preparation for the model: render the prompt and write the
    /// task context file next to the session.
    BuildPrompt {
        continuation: bool,
        comments: String,
    },
    InvokeImplementationModel {
        prompt: String,
    },
    /// Continue the same model session after a nominally successful handoff
    /// produced no repository changes.
    NudgeImplementationModel,
    StageAll,
    Commit {
        message: String,
    },
    PushBranch {
        branch: String,
    },
    CreateMergeRequest {
        branch: String,
        base: String,
        title: String,
        description: String,
    },
    AddMrScopeLabel {
        mr_iid: u64,
    },
    HandIssueBackToHumans {
        branch: String,
        reason: String,
    },
}

/// How one model invocation ended.
enum ImplModelResult {
    Output(Box<WorkerImplementationOutput>),
    /// An external cancel that the cancel helper resolved as an
    /// intentional stop.
    Cancelled,
}

/// What the port reports after executing one [`ImplAction`].
enum ImplOutcome {
    Done,
    Failed(anyhow::Error),
    /// [`ImplAction::StopIfReviewOnly`]: whether the run must stop.
    Stopped(bool),
    /// [`ImplAction::MergeBaseIntoBranch`]: whether the merge was clean.
    Merged(bool),
    Prompt(String),
    Model(ImplModelResult),
    MergeRequestCreated(u64),
}

/// The narrow surface one implementation run needs: the worktree, the
/// issue's merge requests, the session store, and the model. Object-safe
/// and role-local.
trait ImplementationPort {
    fn closes_linked_mr(&self) -> Option<ClosesLinkedMr>;
    fn open_mr_for_issue(&self) -> Option<u64>;
    fn default_branch(&self) -> Result<String>;
    fn default_branch_or_main(&self) -> String;
    fn remote_branch_exists(&self, branch: &str) -> Result<bool>;
    fn has_diff_against(&self, base: &str) -> Result<bool>;
    fn has_staged_changes(&self) -> Result<bool>;
    fn merge_request_state(&self, mr_iid: u64) -> Option<String>;
    fn dependency_closed(&self, issue_iid: u64) -> bool;
    fn issue_comments(&self) -> String;
    fn execute(&mut self, action: &ImplAction) -> ImplOutcome;
}

/// One turn of the implementation driver loop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImplStep {
    Observe(ImplQuery),
    Act(ImplAction),
    Finish,
}

/// Where an implementation run is. The variants spell out the fixed order:
/// adopt an existing merge request if there is one, then prepare the
/// worktree, then invoke the model, then commit, push, and open the MR.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ImplStage {
    StopBeforeDiscovery,
    ObserveClosesLinkedMr,
    LabelClosesMr(u64),
    SaveClosesMrSession(u64),
    CloseIssueForMergedMr,
    CleanupSessionForMergedMr,
    ObserveOpenMrForIssue,
    LabelOpenMr(u64),
    SaveOpenMrSession(u64),
    ObserveDefaultBranch,
    FetchRemote,
    ResetBeforePreparation,
    ObserveRemoteBranch,
    CheckoutExistingBranch,
    ObserveBranchDiff,
    ResetStaleBranch,
    CheckoutDefaultAfterStale,
    DeleteLocalStaleBranch,
    DeleteRemoteStaleBranch,
    CreateBranchAfterStale,
    MergeDefaultIntoBranch,
    ResetConflictedBranch,
    CheckoutDefaultAfterConflict,
    DeleteLocalConflictedBranch,
    CreateBranchAfterConflict,
    CreateBranch,
    StopAfterPreparation,
    RequireWorkingOnLabel,
    ObserveIssueComments,
    BuildPrompt,
    InvokeModel,
    NudgeModelAfterNoChanges,
    ObserveExistingMrState(u64),
    ResetForExistingMr(u64),
    ObserveDefaultForExistingMr(u64),
    CheckoutDefaultForExistingMr(u64),
    DeleteLocalBranchForExistingMr(u64),
    SaveExistingMrSession(u64),
    LabelExistingMr(u64),
    HandBackToHumans(String),
    ObserveDependencyState(u64),
    ParkLabelDependency(u64),
    ParkRemoveWorkingOnLabel(u64),
    ParkComment(u64),
    ParkReleaseClaim(u64),
    ParkCleanupSession(u64),
    ParkObserveDefaultBranch(u64),
    ParkResetWorktree(u64),
    ParkCheckoutDefault(u64),
    ParkDeleteLocalBranch(u64),
    StageAll,
    ObserveStagedChanges,
    CommitChanges,
    ObserveDiffBeforePush,
    PushBranch,
    CreateMergeRequest,
    AddScopeLabel(u64),
    SaveSessionWithSummary(u64),
    Finish,
}

/// The state of one implementation run. Plain data: no git repo, no GitLab
/// client, no model, so every transition is a pure function of what the
/// machine already observed.
struct ImplementationMachine<'a> {
    agent_id: &'a str,
    scope_label: Option<&'a str>,
    issue: IssueObservation,
    stage: ImplStage,
    branch_name: String,
    default_branch: String,
    branch_existed: bool,
    metadata: ImplementedMetadata,
    /// The rendered issue comments, held only between the observation that
    /// read them and the prompt build that consumes them.
    comments: String,
    prompt: String,
    /// The merge request this run ends up tracking, if any — the old
    /// `Ok(Some(mr_iid))`.
    tracked_mr: Option<u64>,
    /// Mirrors the `current` the cycle cleans up from: the branch the run
    /// left behind and whether it produced a merge request.
    left_branch: Option<String>,
    mr_created: bool,
}

impl<'a> ImplementationMachine<'a> {
    fn new(agent_id: &'a str, scope_label: Option<&'a str>, issue: IssueObservation) -> Self {
        let branch_name = format!("issue-{}", issue.iid);
        Self {
            agent_id,
            scope_label,
            issue,
            stage: ImplStage::StopBeforeDiscovery,
            branch_name,
            default_branch: String::new(),
            branch_existed: false,
            metadata: ImplementedMetadata::default(),
            comments: String::new(),
            prompt: String::new(),
            tracked_mr: None,
            left_branch: None,
            mr_created: false,
        }
    }

    fn mr_title(&self) -> String {
        extract_mr_title(self.metadata.mr_title.as_deref(), &self.issue.title)
    }

    /// The single next thing to do; resolves the stages that need no port
    /// interaction on the way.
    fn next_step(&mut self) -> ImplStep {
        match self.stage.clone() {
            ImplStage::StopBeforeDiscovery | ImplStage::StopAfterPreparation => {
                ImplStep::Act(ImplAction::StopIfReviewOnly)
            }
            ImplStage::ObserveClosesLinkedMr => ImplStep::Observe(ImplQuery::ClosesLinkedMr),
            ImplStage::LabelClosesMr(_) | ImplStage::LabelOpenMr(_) => {
                ImplStep::Act(ImplAction::AddWorkingOnLabel)
            }
            ImplStage::SaveClosesMrSession(mr_iid) => {
                ImplStep::Act(ImplAction::SaveSession { mr_iid })
            }
            ImplStage::CloseIssueForMergedMr => ImplStep::Act(ImplAction::CloseIssue),
            ImplStage::CleanupSessionForMergedMr | ImplStage::ParkCleanupSession(_) => {
                ImplStep::Act(ImplAction::CleanupSession)
            }
            ImplStage::ObserveOpenMrForIssue => ImplStep::Observe(ImplQuery::OpenMrForIssue),
            ImplStage::SaveOpenMrSession(mr_iid) | ImplStage::SaveExistingMrSession(mr_iid) => {
                ImplStep::Act(ImplAction::RequireSaveSession { mr_iid })
            }
            ImplStage::ObserveDefaultBranch => ImplStep::Observe(ImplQuery::DefaultBranch),
            ImplStage::FetchRemote => ImplStep::Act(ImplAction::FetchRemote),
            ImplStage::ResetBeforePreparation
            | ImplStage::ResetStaleBranch
            | ImplStage::ResetConflictedBranch
            | ImplStage::ResetForExistingMr(_)
            | ImplStage::ParkResetWorktree(_) => ImplStep::Act(ImplAction::ResetWorktree),
            ImplStage::ObserveRemoteBranch => ImplStep::Observe(ImplQuery::RemoteBranchExists),
            ImplStage::CheckoutExistingBranch => ImplStep::Act(ImplAction::CheckoutBranch {
                branch: self.branch_name.clone(),
            }),
            ImplStage::ObserveBranchDiff | ImplStage::ObserveDiffBeforePush => {
                ImplStep::Observe(ImplQuery::DiffAgainstDefault)
            }
            ImplStage::CheckoutDefaultAfterStale | ImplStage::CheckoutDefaultAfterConflict => {
                ImplStep::Act(ImplAction::CheckoutBranch {
                    branch: self.default_branch.clone(),
                })
            }
            ImplStage::CheckoutDefaultForExistingMr(_) | ImplStage::ParkCheckoutDefault(_) => {
                ImplStep::Act(ImplAction::CheckoutBranchBestEffort {
                    branch: self.default_branch.clone(),
                })
            }
            ImplStage::DeleteLocalStaleBranch
            | ImplStage::DeleteLocalConflictedBranch
            | ImplStage::DeleteLocalBranchForExistingMr(_)
            | ImplStage::ParkDeleteLocalBranch(_) => ImplStep::Act(ImplAction::DeleteLocalBranch {
                branch: self.branch_name.clone(),
            }),
            ImplStage::DeleteRemoteStaleBranch => ImplStep::Act(ImplAction::DeleteRemoteBranch {
                branch: self.branch_name.clone(),
            }),
            ImplStage::CreateBranchAfterStale
            | ImplStage::CreateBranchAfterConflict
            | ImplStage::CreateBranch => ImplStep::Act(ImplAction::CreateBranchFrom {
                branch: self.branch_name.clone(),
                base: self.default_branch.clone(),
            }),
            ImplStage::MergeDefaultIntoBranch => ImplStep::Act(ImplAction::MergeBaseIntoBranch {
                base: self.default_branch.clone(),
            }),
            ImplStage::RequireWorkingOnLabel => ImplStep::Act(ImplAction::RequireWorkingOnLabel),
            ImplStage::ObserveIssueComments => ImplStep::Observe(ImplQuery::IssueComments),
            ImplStage::BuildPrompt => ImplStep::Act(ImplAction::BuildPrompt {
                continuation: self.branch_existed,
                comments: self.comments.clone(),
            }),
            ImplStage::InvokeModel => ImplStep::Act(ImplAction::InvokeImplementationModel {
                prompt: self.prompt.clone(),
            }),
            ImplStage::NudgeModelAfterNoChanges => {
                ImplStep::Act(ImplAction::NudgeImplementationModel)
            }
            ImplStage::ObserveExistingMrState(mr_iid) => {
                ImplStep::Observe(ImplQuery::MergeRequestState { mr_iid })
            }
            ImplStage::ObserveDefaultForExistingMr(_) | ImplStage::ParkObserveDefaultBranch(_) => {
                ImplStep::Observe(ImplQuery::DefaultBranchOrMain)
            }
            ImplStage::LabelExistingMr(_) => ImplStep::Act(ImplAction::AddWorkingOnLabel),
            ImplStage::HandBackToHumans(reason) => {
                ImplStep::Act(ImplAction::HandIssueBackToHumans {
                    branch: self.branch_name.clone(),
                    reason,
                })
            }
            ImplStage::ObserveDependencyState(issue_iid) => {
                ImplStep::Observe(ImplQuery::DependencyClosed { issue_iid })
            }
            ImplStage::ParkLabelDependency(dep_issue_iid) => {
                ImplStep::Act(ImplAction::AddIssueLabel {
                    label: waiting_on_issue_label(dep_issue_iid),
                })
            }
            ImplStage::ParkRemoveWorkingOnLabel(_) => {
                ImplStep::Act(ImplAction::RemoveWorkingOnLabel)
            }
            ImplStage::ParkComment(dep_issue_iid) => ImplStep::Act(ImplAction::AddIssueComment {
                body: format!(
                    "Implementation cannot proceed until issue #{} is closed. \
                     Parking this issue until the dependency resolves.",
                    dep_issue_iid
                ),
            }),
            ImplStage::ParkReleaseClaim(_) => ImplStep::Act(ImplAction::ReleaseIssueClaim),
            ImplStage::StageAll => ImplStep::Act(ImplAction::StageAll),
            ImplStage::ObserveStagedChanges => ImplStep::Observe(ImplQuery::StagedChanges),
            ImplStage::CommitChanges => ImplStep::Act(ImplAction::Commit {
                message: build_commit_message(&self.mr_title(), self.issue.iid),
            }),
            ImplStage::PushBranch => ImplStep::Act(ImplAction::PushBranch {
                branch: self.branch_name.clone(),
            }),
            ImplStage::CreateMergeRequest => ImplStep::Act(ImplAction::CreateMergeRequest {
                branch: self.branch_name.clone(),
                base: self.default_branch.clone(),
                title: self.mr_title(),
                description: format!(
                    "Closes #{}\n\n{}",
                    self.issue.iid,
                    extract_mr_description(self.metadata.mr_description.as_deref())
                ),
            }),
            ImplStage::AddScopeLabel(mr_iid) => {
                ImplStep::Act(ImplAction::AddMrScopeLabel { mr_iid })
            }
            ImplStage::SaveSessionWithSummary(mr_iid) => {
                ImplStep::Act(ImplAction::RequireSaveSessionWithSummary {
                    mr_iid,
                    summary: extract_mr_description(self.metadata.mr_description.as_deref()),
                })
            }
            ImplStage::Finish => ImplStep::Finish,
        }
    }

    fn apply_fact(&mut self, fact: Result<ImplFact>) -> Result<()> {
        match (self.stage.clone(), fact?) {
            (ImplStage::ObserveClosesLinkedMr, ImplFact::ClosesLinkedMr(linked)) => {
                self.stage = match linked {
                    Some(ClosesLinkedMr::Open(mr_iid)) => {
                        info!(
                            "Issue #{} has open MR !{} (linked via Closes #{}), tracking it",
                            self.issue.iid, mr_iid, self.issue.iid
                        );
                        self.tracked_mr = Some(mr_iid);
                        self.mr_created = true;
                        ImplStage::LabelClosesMr(mr_iid)
                    }
                    Some(ClosesLinkedMr::Merged) => {
                        info!(
                            "Issue #{}: merged MR already references it via Closes #; closing issue",
                            self.issue.iid
                        );
                        ImplStage::CloseIssueForMergedMr
                    }
                    None => ImplStage::ObserveOpenMrForIssue,
                };
            }
            (ImplStage::ObserveOpenMrForIssue, ImplFact::OpenMrForIssue(found)) => {
                self.stage = match found {
                    Some(mr_iid) => {
                        info!(
                            "Issue #{} already has open MR !{}, tracking it",
                            self.issue.iid, mr_iid
                        );
                        self.tracked_mr = Some(mr_iid);
                        self.mr_created = true;
                        ImplStage::LabelOpenMr(mr_iid)
                    }
                    None => ImplStage::ObserveDefaultBranch,
                };
            }
            (ImplStage::ObserveDefaultBranch, ImplFact::DefaultBranch(branch)) => {
                self.default_branch = branch;
                self.stage = ImplStage::FetchRemote;
            }
            (ImplStage::ObserveRemoteBranch, ImplFact::RemoteBranchExists(exists)) => {
                self.stage = if exists {
                    info!(
                        "Branch {} already exists on remote, checking if it's stale",
                        self.branch_name
                    );
                    ImplStage::CheckoutExistingBranch
                } else {
                    ImplStage::CreateBranch
                };
            }
            (ImplStage::ObserveBranchDiff, ImplFact::DiffAgainstDefault(has_diff)) => {
                self.stage = if has_diff {
                    ImplStage::MergeDefaultIntoBranch
                } else {
                    warn!(
                        "Branch {} has no diff against {}, discarding stale branch",
                        self.branch_name, self.default_branch
                    );
                    ImplStage::ResetStaleBranch
                };
            }
            (ImplStage::ObserveIssueComments, ImplFact::IssueComments(comments)) => {
                self.comments = comments;
                self.stage = ImplStage::BuildPrompt;
            }
            (ImplStage::ObserveExistingMrState(mr_iid), ImplFact::MergeRequestState(state)) => {
                self.stage = match state.as_deref() {
                    Some("opened") => {
                        info!(
                            "Issue #{}: model identified existing MR !{} as the implementation; tracking it",
                            self.issue.iid, mr_iid
                        );
                        ImplStage::ResetForExistingMr(mr_iid)
                    }
                    Some(other) => {
                        warn!(
                            "Issue #{}: model identified MR !{} but it is not open (state={}); proceeding with new MR",
                            self.issue.iid, mr_iid, other
                        );
                        ImplStage::StageAll
                    }
                    None => {
                        warn!(
                            "Issue #{}: model identified MR !{} but it could not be fetched; proceeding with new MR",
                            self.issue.iid, mr_iid
                        );
                        ImplStage::StageAll
                    }
                };
            }
            (
                ImplStage::ObserveDefaultForExistingMr(mr_iid),
                ImplFact::DefaultBranchOrMain(branch),
            ) => {
                self.default_branch = branch;
                self.stage = ImplStage::CheckoutDefaultForExistingMr(mr_iid);
            }
            (ImplStage::ParkObserveDefaultBranch(dep), ImplFact::DefaultBranchOrMain(branch)) => {
                self.default_branch = branch;
                self.stage = ImplStage::ParkResetWorktree(dep);
            }
            (ImplStage::ObserveDependencyState(dep), ImplFact::DependencyClosed(closed)) => {
                self.stage = if closed {
                    ImplStage::StageAll
                } else {
                    ImplStage::ParkLabelDependency(dep)
                };
            }
            (ImplStage::ObserveStagedChanges, ImplFact::StagedChanges(staged)) => {
                self.stage = if staged {
                    ImplStage::CommitChanges
                } else {
                    ImplStage::ObserveDiffBeforePush
                };
            }
            (ImplStage::ObserveDiffBeforePush, ImplFact::DiffAgainstDefault(has_diff)) => {
                if !has_diff {
                    warn!(
                        "Issue #{}: agent produced no code changes, nudging the current session",
                        self.issue.iid
                    );
                    self.stage = ImplStage::NudgeModelAfterNoChanges;
                    return Ok(());
                }
                self.stage = ImplStage::PushBranch;
            }
            (stage, fact) => {
                anyhow::bail!("implementation port answered {stage:?} with {fact:?}");
            }
        }
        Ok(())
    }

    fn apply_outcome(&mut self, outcome: ImplOutcome) -> Result<()> {
        match (self.stage.clone(), outcome) {
            (ImplStage::StopBeforeDiscovery, ImplOutcome::Stopped(stop)) => {
                self.stage = if stop {
                    ImplStage::Finish
                } else {
                    ImplStage::ObserveClosesLinkedMr
                };
            }
            (ImplStage::StopAfterPreparation, ImplOutcome::Stopped(stop)) => {
                self.stage = if stop {
                    ImplStage::Finish
                } else {
                    ImplStage::RequireWorkingOnLabel
                };
            }
            (ImplStage::LabelClosesMr(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::SaveClosesMrSession(mr_iid);
            }
            (ImplStage::SaveClosesMrSession(_), ImplOutcome::Done) => {
                self.stage = ImplStage::Finish;
            }
            (ImplStage::CloseIssueForMergedMr, ImplOutcome::Done) => {
                self.stage = ImplStage::CleanupSessionForMergedMr;
            }
            (ImplStage::CleanupSessionForMergedMr, ImplOutcome::Done) => {
                self.stage = ImplStage::Finish;
            }
            (ImplStage::LabelOpenMr(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::SaveOpenMrSession(mr_iid);
            }
            (ImplStage::SaveOpenMrSession(_), ImplOutcome::Done) => {
                self.stage = ImplStage::Finish;
            }
            (ImplStage::FetchRemote, ImplOutcome::Done) => {
                self.stage = ImplStage::ResetBeforePreparation;
            }
            (ImplStage::ResetBeforePreparation, ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveRemoteBranch;
            }
            (ImplStage::CheckoutExistingBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveBranchDiff;
            }
            (ImplStage::ResetStaleBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::CheckoutDefaultAfterStale;
            }
            (ImplStage::CheckoutDefaultAfterStale, ImplOutcome::Done) => {
                self.stage = ImplStage::DeleteLocalStaleBranch;
            }
            (ImplStage::DeleteLocalStaleBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::DeleteRemoteStaleBranch;
            }
            (ImplStage::DeleteRemoteStaleBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::CreateBranchAfterStale;
            }
            (ImplStage::MergeDefaultIntoBranch, ImplOutcome::Merged(clean)) => {
                self.stage = if clean {
                    self.branch_existed = true;
                    self.enter_preparation_done()
                } else {
                    warn!(
                        "Branch {} has conflicts with {}, creating fresh branch instead",
                        self.branch_name, self.default_branch
                    );
                    ImplStage::ResetConflictedBranch
                };
            }
            (ImplStage::ResetConflictedBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::CheckoutDefaultAfterConflict;
            }
            (ImplStage::CheckoutDefaultAfterConflict, ImplOutcome::Done) => {
                self.stage = ImplStage::DeleteLocalConflictedBranch;
            }
            (ImplStage::DeleteLocalConflictedBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::CreateBranchAfterConflict;
            }
            (
                ImplStage::CreateBranchAfterStale
                | ImplStage::CreateBranchAfterConflict
                | ImplStage::CreateBranch,
                ImplOutcome::Done,
            ) => {
                self.branch_existed = false;
                self.stage = self.enter_preparation_done();
            }
            (ImplStage::RequireWorkingOnLabel, ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveIssueComments;
            }
            (ImplStage::BuildPrompt, ImplOutcome::Prompt(prompt)) => {
                self.prompt = prompt;
                self.stage = ImplStage::InvokeModel;
            }
            (
                ImplStage::InvokeModel | ImplStage::NudgeModelAfterNoChanges,
                ImplOutcome::Model(result),
            ) => {
                self.stage = match result {
                    ImplModelResult::Cancelled => ImplStage::Finish,
                    ImplModelResult::Output(output) => {
                        info!(
                            "{}: Worker agent finished issue #{}",
                            self.agent_id, self.issue.iid
                        );
                        self.classify_model_output(&output)
                    }
                };
            }
            (ImplStage::ResetForExistingMr(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveDefaultForExistingMr(mr_iid);
            }
            (ImplStage::CheckoutDefaultForExistingMr(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::DeleteLocalBranchForExistingMr(mr_iid);
            }
            (ImplStage::DeleteLocalBranchForExistingMr(mr_iid), ImplOutcome::Done) => {
                self.tracked_mr = Some(mr_iid);
                self.mr_created = true;
                self.stage = ImplStage::SaveExistingMrSession(mr_iid);
            }
            (ImplStage::SaveExistingMrSession(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::LabelExistingMr(mr_iid);
            }
            (ImplStage::LabelExistingMr(_), ImplOutcome::Done) => {
                self.stage = ImplStage::Finish;
            }
            (ImplStage::HandBackToHumans(_), ImplOutcome::Done) => {
                self.left_branch = None;
                self.stage = ImplStage::Finish;
            }
            (ImplStage::ParkLabelDependency(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkRemoveWorkingOnLabel(dep);
            }
            (ImplStage::ParkRemoveWorkingOnLabel(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkComment(dep);
            }
            (ImplStage::ParkComment(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkReleaseClaim(dep);
            }
            (ImplStage::ParkReleaseClaim(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkCleanupSession(dep);
            }
            (ImplStage::ParkCleanupSession(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkObserveDefaultBranch(dep);
            }
            (ImplStage::ParkResetWorktree(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkCheckoutDefault(dep);
            }
            (ImplStage::ParkCheckoutDefault(dep), ImplOutcome::Done) => {
                self.stage = ImplStage::ParkDeleteLocalBranch(dep);
            }
            (ImplStage::ParkDeleteLocalBranch(dep), ImplOutcome::Done) => {
                info!(
                    "{}: Issue #{} parked waiting on issue #{} (dependency open), released claim",
                    self.agent_id, self.issue.iid, dep
                );
                self.mr_created = false;
                self.left_branch = None;
                self.stage = ImplStage::Finish;
            }
            (ImplStage::StageAll, ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveStagedChanges;
            }
            (ImplStage::CommitChanges, ImplOutcome::Done) => {
                self.stage = ImplStage::ObserveDiffBeforePush;
            }
            (ImplStage::PushBranch, ImplOutcome::Done) => {
                self.stage = ImplStage::CreateMergeRequest;
            }
            (ImplStage::CreateMergeRequest, ImplOutcome::MergeRequestCreated(mr_iid)) => {
                self.tracked_mr = Some(mr_iid);
                self.mr_created = true;
                info!("Created MR !{} for issue #{}", mr_iid, self.issue.iid);
                self.stage = if self.scope_label.is_some() {
                    ImplStage::AddScopeLabel(mr_iid)
                } else {
                    ImplStage::SaveSessionWithSummary(mr_iid)
                };
            }
            (ImplStage::AddScopeLabel(mr_iid), ImplOutcome::Done) => {
                self.stage = ImplStage::SaveSessionWithSummary(mr_iid);
            }
            (ImplStage::SaveSessionWithSummary(_), ImplOutcome::Done) => {
                self.stage = ImplStage::Finish;
            }
            (_, ImplOutcome::Failed(e)) => return Err(e),
            (stage, _) => {
                anyhow::bail!("implementation port reported an unexpected outcome for {stage:?}");
            }
        }
        Ok(())
    }

    /// The branch is ready: record it as the branch this run may have to
    /// clean up, then re-check the review-only hold.
    fn enter_preparation_done(&mut self) -> ImplStage {
        self.left_branch = Some(self.branch_name.clone());
        ImplStage::StopAfterPreparation
    }

    /// Which of the handoff branches the run took. The branches that
    /// resolve the issue on their own end the run; the rest fall through to
    /// the ordinary commit/push/open-MR path.
    fn classify_model_output(&mut self, output: &WorkerImplementationOutput) -> ImplStage {
        match output {
            WorkerImplementationOutput::Implemented(metadata) => {
                self.metadata = metadata.clone();
                ImplStage::StageAll
            }
            WorkerImplementationOutput::ExistingMr { existing_mr_iid } => {
                ImplStage::ObserveExistingMrState(*existing_mr_iid)
            }
            WorkerImplementationOutput::NeedsSplit(blocked) => {
                warn!("Issue #{} is too broad, needs splitting", self.issue.iid);
                ImplStage::HandBackToHumans(format!(
                    "This issue needs to be split into smaller, focused issues:\n\n{}",
                    extract_split_reason(blocked)
                ))
            }
            WorkerImplementationOutput::NeedsClarification(blocked) => {
                warn!("Issue #{} needs clarification", self.issue.iid);
                ImplStage::HandBackToHumans(extract_clarification(blocked))
            }
            WorkerImplementationOutput::CannotImplement(blocked) => {
                warn!("Issue #{} cannot be implemented", self.issue.iid);
                ImplStage::HandBackToHumans(extract_cannot_implement_reason(blocked))
            }
            WorkerImplementationOutput::WaitDependency { depends_on_issue } => {
                ImplStage::ObserveDependencyState(*depends_on_issue)
            }
        }
    }
}

/// Ask the implementation port one question.
fn observe_implementation(
    port: &dyn ImplementationPort,
    query: &ImplQuery,
    machine: &ImplementationMachine,
) -> Result<ImplFact> {
    Ok(match query {
        ImplQuery::ClosesLinkedMr => ImplFact::ClosesLinkedMr(port.closes_linked_mr()),
        ImplQuery::OpenMrForIssue => ImplFact::OpenMrForIssue(port.open_mr_for_issue()),
        ImplQuery::DefaultBranch => ImplFact::DefaultBranch(port.default_branch()?),
        ImplQuery::DefaultBranchOrMain => {
            ImplFact::DefaultBranchOrMain(port.default_branch_or_main())
        }
        ImplQuery::RemoteBranchExists => {
            ImplFact::RemoteBranchExists(port.remote_branch_exists(&machine.branch_name)?)
        }
        ImplQuery::DiffAgainstDefault => {
            ImplFact::DiffAgainstDefault(port.has_diff_against(&machine.default_branch)?)
        }
        ImplQuery::StagedChanges => ImplFact::StagedChanges(port.has_staged_changes()?),
        ImplQuery::MergeRequestState { mr_iid } => {
            ImplFact::MergeRequestState(port.merge_request_state(*mr_iid))
        }
        ImplQuery::DependencyClosed { issue_iid } => {
            ImplFact::DependencyClosed(port.dependency_closed(*issue_iid))
        }
        ImplQuery::IssueComments => ImplFact::IssueComments(port.issue_comments()),
    })
}

/// Run one implementation to completion: observe, decide one step, execute
/// it, feed the result back.
fn drive_implementation(
    machine: &mut ImplementationMachine,
    port: &mut dyn ImplementationPort,
) -> Result<()> {
    loop {
        match machine.next_step() {
            ImplStep::Observe(query) => {
                let fact = observe_implementation(port, &query, machine);
                machine.apply_fact(fact)?;
            }
            ImplStep::Act(action) => {
                let outcome = port.execute(&action);
                machine.apply_outcome(outcome)?;
            }
            ImplStep::Finish => return Ok(()),
        }
    }
}

/// The implementation progression backed by the real runtime.
struct LiveImplementationPort<'a> {
    state: &'a AgentState<'a>,
    model: &'a AgentModel,
    issue: &'a IssueObservation,
    scope_label: Option<&'a str>,
}

impl LiveImplementationPort<'_> {
    fn invoke_implementation_model(&self, initial_prompt: Option<&str>) -> Result<ImplModelResult> {
        let issue_iid = self.issue.iid;
        let options = InvokeOptions {
            cancel_check: Some(worker_issue_cancel_check(
                self.state.glab.clone(),
                issue_iid,
            )),
            follow_up_poll: None,
            activity_label: Some(format!(
                "{} implementing issue #{}",
                self.state.agent_id, issue_iid
            )),
        };
        let mut completion = match initial_prompt {
            Some(prompt) => self
                .model
                .complete_typed::<WorkerImplementationOutput>(prompt, &options),
            None => self
                .model
                .continue_typed::<WorkerImplementationOutput>(WORKER_NO_CHANGES_NUDGE, &options),
        };

        loop {
            match completion {
                Ok(completion) => {
                    return Ok(ImplModelResult::Output(Box::new(completion.output)));
                }
                Err(error)
                    if handle_worker_issue_processing_cancelled(self.state, issue_iid, &error) =>
                {
                    return Ok(ImplModelResult::Cancelled);
                }
                Err(error) if AgentModel::structured_output_retries_exhausted(&error) => {
                    warn!(
                        "{}: issue #{} still has no valid structured result; nudging the current session",
                        self.state.agent_id, issue_iid
                    );
                    completion = self.model.continue_typed::<WorkerImplementationOutput>(
                        WORKER_MISSING_OUTPUT_NUDGE,
                        &options,
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl ImplementationPort for LiveImplementationPort<'_> {
    fn closes_linked_mr(&self) -> Option<ClosesLinkedMr> {
        closes_keyword_mr_status(self.state.glab, self.issue.iid)
    }

    fn open_mr_for_issue(&self) -> Option<u64> {
        find_open_mr_for_issue(self.state.glab, self.issue.iid)
    }

    fn default_branch(&self) -> Result<String> {
        self.state.git_repo.get_default_branch()
    }

    fn default_branch_or_main(&self) -> String {
        self.state
            .git_repo
            .get_default_branch()
            .unwrap_or_else(|_| "main".to_string())
    }

    fn remote_branch_exists(&self, branch: &str) -> Result<bool> {
        self.state.git_repo.remote_branch_exists(branch)
    }

    fn has_diff_against(&self, base: &str) -> Result<bool> {
        self.state.git_repo.has_diff_against(base)
    }

    fn has_staged_changes(&self) -> Result<bool> {
        self.state.git_repo.has_staged_changes()
    }

    fn merge_request_state(&self, mr_iid: u64) -> Option<String> {
        self.state
            .glab
            .get_merge_request(mr_iid)
            .ok()
            .map(|mr| mr.state)
    }

    fn dependency_closed(&self, issue_iid: u64) -> bool {
        self.state
            .glab
            .get_issue(issue_iid)
            .map(|dep| dep.state == "closed")
            .unwrap_or(false)
    }

    fn issue_comments(&self) -> String {
        format_issue_comments_for_worker_context(self.state.glab, self.issue.iid)
    }

    fn execute(&mut self, action: &ImplAction) -> ImplOutcome {
        let state = self.state;
        let issue_iid = self.issue.iid;
        match action {
            ImplAction::StopIfReviewOnly => {
                ImplOutcome::Stopped(stop_worker_issue_if_review_only(state, issue_iid))
            }
            ImplAction::AddWorkingOnLabel => {
                let _ = state.glab.add_issue_label(issue_iid, WORKING_ON_LABEL);
                ImplOutcome::Done
            }
            ImplAction::RequireWorkingOnLabel => {
                match state.glab.add_issue_label(issue_iid, WORKING_ON_LABEL) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::RemoveWorkingOnLabel => {
                let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                ImplOutcome::Done
            }
            ImplAction::AddIssueLabel { label } => {
                let _ = state.glab.add_issue_label(issue_iid, label);
                ImplOutcome::Done
            }
            ImplAction::AddIssueComment { body } => {
                let _ = state.glab.add_issue_comment(issue_iid, body);
                ImplOutcome::Done
            }
            ImplAction::ReleaseIssueClaim => {
                let _ = claim::release(state.glab, ClaimResource::Issue(issue_iid), state.agent_id);
                ImplOutcome::Done
            }
            ImplAction::CleanupSession => {
                state.cleanup_session(issue_iid);
                ImplOutcome::Done
            }
            ImplAction::CloseIssue => {
                close_issue_best_effort(state.glab, issue_iid);
                ImplOutcome::Done
            }
            ImplAction::SaveSession { mr_iid } => {
                let _ = state.save_session(issue_iid, *mr_iid);
                ImplOutcome::Done
            }
            ImplAction::RequireSaveSession { mr_iid } => {
                match state.save_session(issue_iid, *mr_iid) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::RequireSaveSessionWithSummary { mr_iid, summary } => {
                match state.save_session_with_summary(issue_iid, *mr_iid, summary) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::FetchRemote => match state.git_repo.fetch() {
                Ok(()) => ImplOutcome::Done,
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::ResetWorktree => {
                let _ = state.git_repo.reset_hard();
                ImplOutcome::Done
            }
            ImplAction::CheckoutBranch { branch } => {
                match state.git_repo.checkout_remote_branch(branch) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::CheckoutBranchBestEffort { branch } => {
                let _ = state.git_repo.checkout_remote_branch(branch);
                ImplOutcome::Done
            }
            ImplAction::DeleteLocalBranch { branch } => {
                let _ = state.git_repo.delete_local_branch(branch);
                ImplOutcome::Done
            }
            ImplAction::DeleteRemoteBranch { branch } => {
                let _ = state.git_repo.delete_remote_branch(branch);
                ImplOutcome::Done
            }
            ImplAction::CreateBranchFrom { branch, base } => {
                match state.git_repo.create_branch_from(branch, base) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::MergeBaseIntoBranch { base } => match state.git_repo.try_merge(base) {
                Ok(clean) => ImplOutcome::Merged(clean),
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::BuildPrompt {
                continuation,
                comments,
            } => {
                let built = if *continuation {
                    build_continuation_prompt(state, self.issue, comments)
                } else {
                    build_implementation_prompt(state, self.issue, comments)
                };
                match built {
                    Ok(prompt) => ImplOutcome::Prompt(prompt),
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
            ImplAction::InvokeImplementationModel { prompt } => {
                match self.invoke_implementation_model(Some(prompt)) {
                    Ok(result) => ImplOutcome::Model(result),
                    Err(error) => ImplOutcome::Failed(error),
                }
            }
            ImplAction::NudgeImplementationModel => match self.invoke_implementation_model(None) {
                Ok(result) => ImplOutcome::Model(result),
                Err(error) => ImplOutcome::Failed(error),
            },
            ImplAction::StageAll => match state.git_repo.add_all() {
                Ok(()) => ImplOutcome::Done,
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::Commit { message } => match state.git_repo.commit(message) {
                Ok(()) => ImplOutcome::Done,
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::PushBranch { branch } => match state.git_repo.push(branch) {
                Ok(()) => ImplOutcome::Done,
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::CreateMergeRequest {
                branch,
                base,
                title,
                description,
            } => match state
                .glab
                .create_merge_request(branch, base, title, description)
            {
                Ok(mr_iid) => ImplOutcome::MergeRequestCreated(mr_iid),
                Err(e) => ImplOutcome::Failed(e),
            },
            ImplAction::AddMrScopeLabel { mr_iid } => {
                if let Some(label) = self.scope_label
                    && let Err(e) = state.glab.add_mr_label_with_retries(*mr_iid, label)
                {
                    warn!(
                        "{}: Failed to add scope label {:?} to MR !{} (permanent error): {}",
                        state.agent_id, label, mr_iid, e
                    );
                }
                ImplOutcome::Done
            }
            ImplAction::HandIssueBackToHumans { branch, reason } => {
                match hand_issue_back_to_humans(state, issue_iid, branch, reason) {
                    Ok(()) => ImplOutcome::Done,
                    Err(e) => ImplOutcome::Failed(e),
                }
            }
        }
    }
}

/// Implement one issue: adopt an existing merge request if the issue
/// already has one, otherwise prepare the branch, invoke the model, and
/// turn what it produced into a merge request. Observes, decides one step,
/// executes it, feeds the result back — see [`ImplementationMachine`].
///
/// `current` is updated as the run progresses, because the cycle's cleanup
/// after a failure depends on how far the run got.
fn process_issue(
    state: &AgentState,
    model: &AgentModel,
    issue: &IssueObservation,
    current: &mut ActiveIssue,
    scope_label: Option<&str>,
) -> Result<Option<u64>> {
    let mut machine = ImplementationMachine::new(state.agent_id, scope_label, issue.clone());
    let mut port = LiveImplementationPort {
        state,
        model,
        issue,
        scope_label,
    };
    let result = drive_implementation(&mut machine, &mut port);
    current.mr_iid = machine.tracked_mr;
    current.mr_created = machine.mr_created;
    current.branch_name = machine.left_branch.clone();
    result.map(|()| machine.tracked_mr)
}

/// Post `reason` on the issue, close any MR the branch already had, reset the
/// worktree, and swap `in-progress` for `action-required` so a human picks the
/// issue up. Shared by the three implementation outcomes that hand the issue
/// back instead of producing a merge request.
fn hand_issue_back_to_humans(
    state: &AgentState,
    issue_iid: u64,
    branch_name: &str,
    reason: &str,
) -> Result<()> {
    state.glab.add_issue_comment(issue_iid, reason)?;

    if let Some(mr_iid) = find_open_mr_for_issue(state.glab, issue_iid) {
        state.glab.add_mr_comment(
            mr_iid,
            &format!(
                "Closing this MR — the issue cannot be implemented:\n\n{}",
                reason
            ),
        )?;
        let _ = state.glab.close_mr(mr_iid);
    }

    // Reset git to a clean state — keep remote branch for potential retry.
    let default_branch = state
        .git_repo
        .get_default_branch()
        .unwrap_or("main".to_string());
    let _ = state.git_repo.reset_hard();
    let _ = state.git_repo.checkout_remote_branch(&default_branch);
    let _ = state.git_repo.delete_local_branch(branch_name);

    state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL)?;
    state
        .glab
        .add_issue_label(issue_iid, ACTION_REQUIRED_LABEL)?;
    info!(
        "Issue #{} requires user action, labeled with '{}'",
        issue_iid, ACTION_REQUIRED_LABEL
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// MR comment handling
// ---------------------------------------------------------------------------

/// Collect new MR comments since `last_seen_id` as follow-up messages.
///
/// Returns formatted messages for each comment with an id greater than
/// `last_seen_id`, and updates `last_seen_id` to the highest id seen. Pure /
/// network-free so it can be unit-tested without a GitLab client.
fn collect_new_follow_ups(
    comments: &[crate::agents::gitlab::Comment],
    last_seen_id: &mut u64,
    mr_iid: u64,
) -> Vec<String> {
    let mut new_msgs = Vec::new();
    for c in comments {
        if c.id > *last_seen_id {
            new_msgs.push(format!(
                "**New comment from @{} on MR !{} (thread {}):**\n\n{}",
                c.author, mr_iid, c.discussion_id, c.body
            ));
        }
    }
    if let Some(max_id) = comments.iter().map(|c| c.id).max() {
        *last_seen_id = max_id;
    }
    new_msgs
}

/// A pending MR title/description write, computed purely from the agent's
/// output and the MR's current metadata.
struct MrMetadataUpdate {
    title: String,
    description: String,
}

/// Decide whether the agent's feedback-run output changes the MR title
/// and/or description, and if so what the resulting metadata should be.
/// Mirrors the update gate in [`handle_mr_comments`] exactly: an update is
/// only planned when the agent's title differs from the current title, or
/// its description is neither the default placeholder nor identical to the
/// current description. Pure — no GitLab call — so this metadata-update
/// decision can be characterized without a live client.
fn plan_mr_metadata_update(
    current_title: &str,
    current_description: &str,
    resolution: &FeedbackResolution,
) -> Option<MrMetadataUpdate> {
    let new_title = extract_explicit_mr_title(resolution.mr_title.as_deref());
    let new_desc = extract_mr_description(resolution.mr_description.as_deref());
    let title_changed = new_title.as_deref().is_some_and(|t| t != current_title);
    let desc_changed = new_desc != "Implementation completed." && new_desc != current_description;
    if !title_changed && !desc_changed {
        return None;
    }
    let title = if title_changed {
        new_title.unwrap_or_else(|| current_title.to_string())
    } else {
        current_title.to_string()
    };
    let description = if desc_changed {
        new_desc
    } else {
        current_description.to_string()
    };
    Some(MrMetadataUpdate { title, description })
}

/// Returns `Ok(true)` if the agent decided the issue cannot be resolved and
/// the MR was closed + issue rejected.
fn handle_mr_comments(
    state: &AgentState,
    model: &AgentModel,
    mr_iid: u64,
    linked_issue_iid: Option<u64>,
    comments_only_mode: bool,
) -> Result<bool> {
    let latest_mr = state.glab.get_merge_request(mr_iid)?;
    let unresolved_ids = state.glab.get_unresolved_discussion_ids(latest_mr.iid)?;
    let all_comments = state.glab.get_mr_comments(latest_mr.iid)?;
    let plain_comments: Vec<_> = all_comments
        .iter()
        .filter(|c| !c.discussion_resolvable)
        .cloned()
        .collect();

    if unresolved_ids.is_empty()
        && plain_comments.is_empty()
        && (comments_only_mode || !latest_mr.has_conflicts)
    {
        return Ok(false);
    }

    if !unresolved_ids.is_empty() {
        info!(
            "MR !{} has {} unresolved discussion(s) to address",
            latest_mr.iid,
            unresolved_ids.len()
        );
    }

    if latest_mr.has_conflicts {
        info!("MR !{} has merge conflicts to resolve", latest_mr.iid);
    }

    let unresolved_comments = if unresolved_ids.is_empty() {
        Vec::new()
    } else {
        let unresolved_set: HashSet<&str> = unresolved_ids.iter().map(String::as_str).collect();
        all_comments
            .iter()
            .filter(|c| unresolved_set.contains(c.discussion_id.as_str()))
            .cloned()
            .collect()
    };

    state.git_repo.fetch_branches(&[
        latest_mr.target_branch.as_str(),
        latest_mr.source_branch.as_str(),
    ])?;
    let _ = state.git_repo.reset_hard();

    state
        .git_repo
        .checkout_remote_branch(&latest_mr.source_branch)?;

    // Remember the remote HEAD so we can detect changes after the agent runs,
    // even if the agent disobeys and commits/pushes itself.
    let pre_agent_sha = latest_mr
        .sha
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or(
            state
                .git_repo
                .rev_parse(&format!("origin/{}", latest_mr.source_branch))?,
        );

    // Build a diff snapshot from the latest source/target state before we merge
    // target into source locally for conflict handling.
    let diff_context_content =
        build_mr_diff_context(state.project_name, &latest_mr, state.git_repo, state.glab);

    // Merge the latest target branch so the worker has up-to-date upstream code.
    let local_merge_clean = state.git_repo.merge_no_abort(&latest_mr.target_branch)?;
    if !local_merge_clean {
        warn!(
            "MR !{}: source branch has conflicts with {}, worker agent will resolve them",
            latest_mr.iid, latest_mr.target_branch
        );
    }
    let requires_conflict_resolution = latest_mr.has_conflicts || !local_merge_clean;
    let merge_conflict_status = build_merge_conflict_status_section(
        &latest_mr,
        requires_conflict_resolution,
        local_merge_clean,
        state.git_repo,
    )?;

    let issue_number = linked_issue_iid
        .or_else(|| extract_issue_number_from_branch(&latest_mr.source_branch).ok());
    if let Some(issue_iid) = issue_number
        && stop_worker_issue_if_review_only(state, issue_iid)
    {
        return Ok(false);
    }
    let issue_context = issue_number
        .map(|n| load_issue_context(state.glab, n))
        .transpose()?
        .unwrap_or_else(|| "No linked issue context available for this MR.".to_string());
    let implementation_summary = issue_number
        .map(|n| state.load_implementation_summary(n))
        .unwrap_or_else(|| "No previous implementation summary available.".to_string());

    let unresolved_comments_text = format_comments_for_prompt(&unresolved_comments);
    let plain_comments_text = format_comments_for_prompt(&plain_comments);
    let all_comments_text = format_comments_for_prompt(&all_comments);

    let combined_context_content =
        build_combined_mr_feedback_context(CombinedMrFeedbackContextInput {
            project_name: state.project_name,
            mr: &latest_mr,
            issue_context: &issue_context,
            implementation_summary: &implementation_summary,
            merge_conflict_status: &merge_conflict_status,
            unresolved_comments_text: &unresolved_comments_text,
            plain_comments_text: &plain_comments_text,
            all_comments_text: &all_comments_text,
            diff_context: &diff_context_content,
        });
    // Write the context to disk for archival/debugging, but inject the content
    // directly into the prompt so the model doesn't waste turns reading it back.
    let _combined_context_path = write_task_context_file(
        state.sessions_dir,
        &format!(
            "{}-mr-feedback-and-diff-{}.md",
            &state.agent_id, latest_mr.iid
        ),
        &combined_context_content,
    )?;

    let feedback_scope_rules = get_feedback_scope_rules();
    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are addressing reviewer feedback on a merge request in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You MUST delete, rename, or move files as needed — do not ask permission or suggest it
- Leave staging, committing, pushing, and merge request creation to the system
- The task context above includes all comments and the diff — review it, then act directly
- Address all unresolved thread feedback and actionable plain MR comments autonomously
- Make all necessary code changes to resolve the comments
- Keep the original issue requirements in mind while addressing feedback
- If the workspace has merge conflict markers (<<<<<<< / ======= / >>>>>>>), resolve ALL of them before doing anything else. Edit each conflicted file to keep the correct version.
- The **Merge conflict status** section in the task context above is verified by Potlatch. Do NOT claim conflicts are fixed unless that section would be clean after your edits and you commit/push the resolution.
- Potlatch will refuse to mark review threads resolved while GitLab still reports merge conflicts or conflict markers remain in the branch.
- Potlatch already fetched `origin/{}` and merged it into your workspace when conflicts were reported. Edit the listed conflicted files, remove all conflict markers, and leave committing/pushing to Potlatch. Do not claim the conflict is fixed until the **Merge conflict status** section shows a clean merge with the fetched target tip.

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly.
2. The task context is already included above. Review the "Unresolved MR comments to address", "Plain MR comments to consider", and "Full MR comment history for context" sections. Do NOT use read on the task context — it's already in your prompt.
3. Use inline comment locations (`path:line` or `path:start-end`) from the comments to find the corresponding code and make targeted fixes. Grep for the relevant symbol, read only the surrounding lines, then edit.
4. First, check for merge conflicts using the **Merge conflict status** section and your workspace. If any exist, resolve ALL conflicts in every file, commit the resolution, and verify the target branch merges cleanly before claiming completion.
5. Review the original issue and what was implemented
6. Review ALL comments to understand the full conversation and context, including simple comments that do not require resolution.
7. Identify which feedback items still need action. Treat comments in "Unresolved MR comments to address" as actionable threaded feedback. Also consider comments in "Plain MR comments to consider" actionable when they ask for changes, but remember they are plain MR comments and cannot be marked resolved. Use the full comment history only for context, clarification, and avoiding stale assumptions.
8. Make the necessary code changes to address all unresolved threaded feedback and any actionable plain MR comments
9. If the reviewer asked you to delete, rename, or move files, make those file changes.
10. Ensure changes align with both the original requirements and reviewer feedback
11. If the reviewer says code changes are too large (above ~1500 lines total or ~500 non-test lines), you have TWO options:
   a) Adjust your implementation to reduce changed lines — simplify, remove unnecessary changes, trim scope
   b) If you cannot reasonably reduce the size, report that the feedback cannot be resolved autonomously
   Do NOT try to split the issue yourself — that is handled by the PMO agent, not you.
12. If the feedback cannot be resolved without additional human input (for example ambiguous requirements, out-of-scope requests, or missing information), report that clearly and identify the needed input.
13. Keep the MR title stable unless the reviewer explicitly asks for a title fix or the current title is clearly wrong for the whole MR.
14. Report only changes and metadata updates actually completed in this run; never imply a concern was fixed when the final branch does not fix it.
15. Before you finish, edit repo-root notes.md only if you can add lines that pass the **NOTES.MD** rules in your main worker instructions (same as implementation runs): **no** backticks, **no** file paths, **no** repo-specific symbol names, **no** code tours — and **no** bullets that merely **summarize what you did** this run in "timeless" wording (that still belongs in the MR, not notes). **No** lines about how to write notes or what notes are for. If nothing meets that bar, leave notes.md unchanged. Never copy notes.md into MR metadata or GitLab comments.

Proceed with addressing the feedback autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        latest_mr.iid,
        latest_mr.title,
        combined_context_content,
        latest_mr.target_branch,
        feedback_scope_rules
    );

    // Follow-up poll: while the agent works on this MR's feedback, watch for
    // new comments added to the MR and forward them into the running session
    // as follow-up context (via session/inject on the potlatch harness backend).
    let seen_comment_id =
        std::sync::Mutex::new(all_comments.iter().map(|c| c.id).max().unwrap_or(0));
    let glab_for_poll = state.glab.clone();
    let mr_iid_for_poll = latest_mr.iid;
    let follow_up_poll: Arc<dyn Fn() -> Vec<String> + Send + Sync> = Arc::new(move || {
        let Ok(comments) = glab_for_poll.get_mr_comments(mr_iid_for_poll) else {
            return Vec::new();
        };
        let mut last = seen_comment_id.lock().unwrap();
        collect_new_follow_ups(&comments, &mut last, mr_iid_for_poll)
    });

    let agent_output = if let Some(issue_iid) = issue_number {
        info!(
            "{}: Worker agent addressing MR !{} feedback for issue #{}",
            &state.agent_id, latest_mr.iid, issue_iid
        );
        let output = match model.complete_typed::<WorkerFeedbackOutput>(
            &prompt,
            &InvokeOptions {
                cancel_check: Some(worker_issue_cancel_check(state.glab.clone(), issue_iid)),
                follow_up_poll: Some(follow_up_poll.clone()),
                activity_label: Some(format!(
                    "{} addressing MR !{} feedback",
                    &state.agent_id, latest_mr.iid
                )),
            },
        ) {
            Ok(output) => output,
            Err(e) if handle_worker_issue_processing_cancelled(state, issue_iid, &e) => {
                return Ok(false);
            }
            Err(e) => return Err(e),
        };
        info!(
            "{}: Worker agent finished MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        output
    } else {
        info!(
            "{}: Worker agent addressing MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        let output = model.complete_typed::<WorkerFeedbackOutput>(
            &prompt,
            &InvokeOptions {
                cancel_check: None,
                follow_up_poll: Some(follow_up_poll.clone()),
                activity_label: Some(format!(
                    "{} addressing MR !{} feedback",
                    &state.agent_id, latest_mr.iid
                )),
            },
        )?;
        info!(
            "{}: Worker agent finished MR !{} feedback",
            &state.agent_id, latest_mr.iid
        );
        output
    };

    let resolution = match &agent_output.output {
        WorkerFeedbackOutput::Addressed(resolution) => resolution.clone(),
        WorkerFeedbackOutput::CannotResolve(blocked) => {
            let reason = extract_cannot_resolve_reason(blocked);
            warn!(
                "MR !{} cannot be resolved autonomously: {}",
                latest_mr.iid, reason
            );

            if let Some(issue_number) = issue_number {
                abandon_mr(state, &latest_mr, issue_number, &reason)?;
            } else {
                state.glab.add_mr_comment(
                    latest_mr.iid,
                    &format!(
                        "Cannot resolve this MR feedback autonomously:\n\n{}",
                        reason
                    ),
                )?;
                let _ = state.glab.close_mr(latest_mr.iid);
            }

            return Ok(true);
        }
    };

    // From here the run is a driven progression: metadata, then
    // commit/push, then the conflict recheck, then the GitLab replies. The
    // decisions live in `FeedbackMachine`; the writes live in the port.
    let mut machine = FeedbackMachine::new(FeedbackTailInput {
        mr_iid: latest_mr.iid,
        source_branch: latest_mr.source_branch.clone(),
        target_branch: latest_mr.target_branch.clone(),
        pre_agent_sha,
        requires_conflict_resolution,
        unresolved_ids,
        plain_comments_present: !plain_comments.is_empty(),
        issue_iid: issue_number,
        surface_before: MrSurfaceObservation::from_mr(&latest_mr),
        resolution,
    });
    let mut port = LiveFeedbackTailPort {
        git_repo: state.git_repo,
        glab: state.glab,
        mr_iid: latest_mr.iid,
        source_branch: latest_mr.source_branch.clone(),
        target_branch: latest_mr.target_branch.clone(),
    };
    drive_feedback(&mut machine, &mut port)?;

    Ok(false)
}

/// One GitLab write in the feedback-reply tail of [`handle_mr_comments`],
/// in the exact order [`plan_feedback_reply_steps`] emits them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackReplyStep {
    Reply { discussion_id: String },
    Resolve { discussion_id: String },
    PostPlainComment,
}

/// Decide the ordered sequence of discussion replies, discussion resolves,
/// and the plain-comment reply that [`handle_mr_comments`] performs after
/// pushing feedback changes. Pure — takes only the already-computed gating
/// booleans — so the reply-before-resolve ordering and the plain-comment
/// gating can be locked down without a live GitLab client.
///
/// Mirrors current behavior exactly: each unresolved discussion gets a
/// reply immediately followed by a resolve (only when `resolve_discussions`
/// is true), in `ids_to_resolve` order; the plain-comment reply, if any, is
/// always emitted last. Nothing is emitted when there is no reply body.
fn plan_feedback_reply_steps(
    ids_to_resolve: &[String],
    reply_body_present: bool,
    resolve_discussions: bool,
    plain_comments_present: bool,
    should_post_plain_comment: bool,
) -> Vec<FeedbackReplyStep> {
    let mut steps = Vec::new();
    if reply_body_present {
        for discussion_id in ids_to_resolve {
            steps.push(FeedbackReplyStep::Reply {
                discussion_id: discussion_id.clone(),
            });
            if resolve_discussions {
                steps.push(FeedbackReplyStep::Resolve {
                    discussion_id: discussion_id.clone(),
                });
            }
        }
        if plain_comments_present && should_post_plain_comment {
            steps.push(FeedbackReplyStep::PostPlainComment);
        }
    }
    steps
}

// ---------------------------------------------------------------------------
// Feedback progression port
// ---------------------------------------------------------------------------

/// The MR fields the feedback progression compares before and after the
/// model run: a metadata-only edit must not imply that feedback was
/// addressed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MrSurfaceObservation {
    title: String,
    description: String,
    labels: Option<Vec<String>>,
    has_conflicts: bool,
}

impl MrSurfaceObservation {
    fn from_mr(mr: &crate::agents::gitlab::MergeRequest) -> Self {
        Self {
            title: mr.title.clone(),
            description: mr.description.clone(),
            labels: mr.labels.clone(),
            has_conflicts: mr.has_conflicts,
        }
    }

    /// True when title, description, or labels differ (e.g. metadata edit,
    /// label added or removed).
    fn differs_from(&self, other: &Self) -> bool {
        self.title.trim() != other.title.trim()
            || self.description.trim() != other.description.trim()
            || self.labels != other.labels
    }
}

/// Everything the pre-model half of a feedback run already learned, as one
/// immutable snapshot the progression decides from.
#[derive(Debug, Clone)]
struct FeedbackTailInput {
    mr_iid: u64,
    source_branch: String,
    target_branch: String,
    pre_agent_sha: String,
    requires_conflict_resolution: bool,
    /// The discussions that were unresolved before the model ran; empty
    /// when the run was triggered by conflicts alone.
    unresolved_ids: Vec<String>,
    plain_comments_present: bool,
    issue_iid: Option<u64>,
    surface_before: MrSurfaceObservation,
    resolution: FeedbackResolution,
}

impl FeedbackTailInput {
    fn commit_message(&self) -> String {
        build_commit_message(
            &extract_changes_summary(self.resolution.changes_summary.as_deref()),
            self.issue_iid.unwrap_or(0),
        )
    }

    fn merge_commit_message(&self) -> String {
        build_commit_message(
            &format!(
                "Merge origin/{} into {}",
                self.target_branch, self.source_branch
            ),
            self.issue_iid.unwrap_or(0),
        )
    }
}

/// One question the feedback progression asks about the worktree, the
/// branch, or the merge request.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackQuery {
    ChangesSinceModelRun,
    MergeInProgress,
    MergeConflictsPresent,
    StagedChanges,
    UpToDateWithTarget,
    DiffHighlights,
    MergeRequestSurface,
    OriginHead,
    UnresolvedDiscussionIds,
}

/// The answer to one [`FeedbackQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackFact {
    ChangesSinceModelRun(bool),
    MergeInProgress(bool),
    MergeConflictsPresent(bool),
    StagedChanges(bool),
    UpToDateWithTarget(bool),
    DiffHighlights(Option<String>),
    MergeRequestSurface(MrSurfaceObservation),
    OriginHead(String),
    UnresolvedDiscussionIds(Vec<String>),
}

/// One side effect in the feedback progression. Everything from the
/// metadata write to the last GitLab reply.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackAction {
    UpdateMrMetadata { title: String, description: String },
    FetchBranches,
    StageResolvedConflicts,
    StageAll,
    Commit { message: String },
    CompleteMergeIfReady { message: String },
    PushSourceBranch,
    ReplyToDiscussion { discussion_id: String, body: String },
    ResolveDiscussion { discussion_id: String },
    PostPlainComment { body: String },
}

/// What the port reports after executing one [`FeedbackAction`].
enum FeedbackOutcome {
    Done,
    Failed(anyhow::Error),
    /// [`FeedbackAction::StageResolvedConflicts`]: whether anything was
    /// staged.
    Staged(bool),
    /// [`FeedbackAction::CompleteMergeIfReady`]: whether the in-progress
    /// merge was concluded.
    MergeCompleted(bool),
}

/// The narrow surface the feedback progression needs: the worktree, the
/// branch tips, and the MR's discussions. Object-safe and role-local.
trait FeedbackTailPort {
    fn has_changes_since(&self, base_ref: &str) -> Result<bool>;
    fn merge_in_progress(&self) -> Result<bool>;
    fn merge_conflicts_present(&self) -> Result<bool>;
    fn has_staged_changes(&self) -> Result<bool>;
    fn up_to_date_with_target(&self, target_branch: &str) -> Result<bool>;
    /// Best effort: no highlights is a valid answer, never a failed run.
    fn diff_highlights(&self, base_ref: &str) -> Option<String>;
    fn merge_request_surface(&self, mr_iid: u64) -> Result<MrSurfaceObservation>;
    /// Best effort: falls back to the pre-run SHA when the tip cannot be
    /// read.
    fn origin_head(&self, source_branch: &str) -> Option<String>;
    /// Best effort: an unreadable list is treated as empty.
    fn unresolved_discussion_ids(&self, mr_iid: u64) -> Vec<String>;
    fn execute(&mut self, action: &FeedbackAction) -> FeedbackOutcome;
}

/// One turn of the feedback driver loop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackStep {
    Observe(FeedbackQuery),
    Act(FeedbackAction),
    Finish,
}

/// Where the feedback progression is. The stage names spell out the
/// required order: metadata, then commit/push, then the conflict recheck,
/// then the GitLab replies.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FeedbackStage {
    UpdateMetadata,
    FetchAfterModelRun,
    ObserveChangesAfterModelRun,
    ObserveMergeInProgress,
    StageResolvedConflicts,
    StageAllForMerge,
    ObserveChangesAfterStaging,
    StageAllForCommit,
    ObserveStagedForCommit,
    CommitChanges,
    CompleteMerge,
    ObserveConflictsAfterMerge,
    FetchForConflictRecheck,
    ObserveUpToDateWithTarget,
    ObserveChangesAfterRecheck,
    StageAllForMergeCommit,
    ObserveStagedForMergeCommit,
    CommitMergeResolution,
    ObserveConflictsBeforeReplies,
    ObserveDiffHighlights,
    PushChanges,
    ObserveMergeRequestSurface,
    ObserveOriginHead,
    ObserveUnresolvedIds,
    Reply(usize),
    Finish,
}

/// The state of one feedback progression. Plain data: no git repo, no
/// GitLab client, no model.
struct FeedbackMachine {
    input: FeedbackTailInput,
    stage: FeedbackStage,
    has_new_changes: bool,
    conflicts_unresolved: bool,
    diff_highlights: Option<String>,
    branch_tip_changed: bool,
    surface_changed: bool,
    ids_to_resolve: Vec<String>,
    reply_body: Option<String>,
    reply_steps: Vec<FeedbackReplyStep>,
}

impl FeedbackMachine {
    fn new(input: FeedbackTailInput) -> Self {
        Self {
            input,
            stage: FeedbackStage::UpdateMetadata,
            has_new_changes: false,
            conflicts_unresolved: false,
            diff_highlights: None,
            branch_tip_changed: false,
            surface_changed: false,
            ids_to_resolve: Vec::new(),
            reply_body: None,
            reply_steps: Vec::new(),
        }
    }

    /// The single next thing to do; resolves the stages that need no port
    /// interaction on the way.
    fn next_step(&mut self) -> FeedbackStep {
        loop {
            match self.stage.clone() {
                FeedbackStage::UpdateMetadata => {
                    match plan_mr_metadata_update(
                        &self.input.surface_before.title,
                        &self.input.surface_before.description,
                        &self.input.resolution,
                    ) {
                        Some(update) => {
                            return FeedbackStep::Act(FeedbackAction::UpdateMrMetadata {
                                title: update.title,
                                description: update.description,
                            });
                        }
                        None => self.stage = FeedbackStage::FetchAfterModelRun,
                    }
                }
                FeedbackStage::FetchAfterModelRun | FeedbackStage::FetchForConflictRecheck => {
                    return FeedbackStep::Act(FeedbackAction::FetchBranches);
                }
                FeedbackStage::ObserveChangesAfterModelRun
                | FeedbackStage::ObserveChangesAfterStaging
                | FeedbackStage::ObserveChangesAfterRecheck => {
                    return FeedbackStep::Observe(FeedbackQuery::ChangesSinceModelRun);
                }
                FeedbackStage::ObserveMergeInProgress => {
                    return FeedbackStep::Observe(FeedbackQuery::MergeInProgress);
                }
                FeedbackStage::StageResolvedConflicts => {
                    return FeedbackStep::Act(FeedbackAction::StageResolvedConflicts);
                }
                FeedbackStage::StageAllForMerge
                | FeedbackStage::StageAllForCommit
                | FeedbackStage::StageAllForMergeCommit => {
                    return FeedbackStep::Act(FeedbackAction::StageAll);
                }
                FeedbackStage::ObserveStagedForCommit
                | FeedbackStage::ObserveStagedForMergeCommit => {
                    return FeedbackStep::Observe(FeedbackQuery::StagedChanges);
                }
                FeedbackStage::CommitChanges => {
                    return FeedbackStep::Act(FeedbackAction::Commit {
                        message: self.input.commit_message(),
                    });
                }
                FeedbackStage::CommitMergeResolution => {
                    return FeedbackStep::Act(FeedbackAction::Commit {
                        message: self.input.merge_commit_message(),
                    });
                }
                FeedbackStage::CompleteMerge => {
                    return FeedbackStep::Act(FeedbackAction::CompleteMergeIfReady {
                        message: self.input.merge_commit_message(),
                    });
                }
                FeedbackStage::ObserveConflictsAfterMerge
                | FeedbackStage::ObserveConflictsBeforeReplies => {
                    return FeedbackStep::Observe(FeedbackQuery::MergeConflictsPresent);
                }
                FeedbackStage::ObserveUpToDateWithTarget => {
                    return FeedbackStep::Observe(FeedbackQuery::UpToDateWithTarget);
                }
                FeedbackStage::ObserveDiffHighlights => {
                    return FeedbackStep::Observe(FeedbackQuery::DiffHighlights);
                }
                FeedbackStage::PushChanges => {
                    return FeedbackStep::Act(FeedbackAction::PushSourceBranch);
                }
                FeedbackStage::ObserveMergeRequestSurface => {
                    return FeedbackStep::Observe(FeedbackQuery::MergeRequestSurface);
                }
                FeedbackStage::ObserveOriginHead => {
                    return FeedbackStep::Observe(FeedbackQuery::OriginHead);
                }
                FeedbackStage::ObserveUnresolvedIds => {
                    return FeedbackStep::Observe(FeedbackQuery::UnresolvedDiscussionIds);
                }
                FeedbackStage::Reply(index) => match self.reply_steps.get(index).cloned() {
                    None => self.stage = FeedbackStage::Finish,
                    Some(step) => {
                        self.stage = FeedbackStage::Reply(index + 1);
                        let body = self
                            .reply_body
                            .clone()
                            .expect("reply steps are only planned with a reply body");
                        return FeedbackStep::Act(match step {
                            FeedbackReplyStep::Reply { discussion_id } => {
                                FeedbackAction::ReplyToDiscussion {
                                    discussion_id,
                                    body,
                                }
                            }
                            FeedbackReplyStep::Resolve { discussion_id } => {
                                FeedbackAction::ResolveDiscussion { discussion_id }
                            }
                            FeedbackReplyStep::PostPlainComment => {
                                FeedbackAction::PostPlainComment { body }
                            }
                        });
                    }
                },
                FeedbackStage::Finish => return FeedbackStep::Finish,
            }
        }
    }

    fn apply_fact(&mut self, fact: Result<FeedbackFact>) -> Result<()> {
        match (self.stage.clone(), fact?) {
            (
                FeedbackStage::ObserveChangesAfterModelRun,
                FeedbackFact::ChangesSinceModelRun(changed),
            ) => {
                self.has_new_changes = changed;
                self.stage = FeedbackStage::ObserveMergeInProgress;
            }
            (FeedbackStage::ObserveMergeInProgress, FeedbackFact::MergeInProgress(in_progress)) => {
                self.stage = if in_progress {
                    FeedbackStage::StageResolvedConflicts
                } else if self.has_new_changes {
                    FeedbackStage::StageAllForCommit
                } else {
                    FeedbackStage::CompleteMerge
                };
            }
            (
                FeedbackStage::ObserveChangesAfterStaging,
                FeedbackFact::ChangesSinceModelRun(changed),
            ) => {
                self.has_new_changes = changed;
                self.stage = FeedbackStage::CompleteMerge;
            }
            (FeedbackStage::ObserveStagedForCommit, FeedbackFact::StagedChanges(staged)) => {
                self.stage = if staged {
                    FeedbackStage::CommitChanges
                } else {
                    FeedbackStage::CompleteMerge
                };
            }
            (
                FeedbackStage::ObserveConflictsAfterMerge,
                FeedbackFact::MergeConflictsPresent(present),
            ) => {
                self.conflicts_unresolved = present;
                self.stage = if self.input.requires_conflict_resolution {
                    FeedbackStage::FetchForConflictRecheck
                } else {
                    self.after_conflict_recheck()
                };
            }
            (FeedbackStage::ObserveUpToDateWithTarget, FeedbackFact::UpToDateWithTarget(ok)) => {
                self.stage = if ok {
                    FeedbackStage::ObserveChangesAfterRecheck
                } else {
                    self.conflicts_unresolved = true;
                    warn!(
                        "MR !{}: branch still does not merge cleanly with origin/{} (fetched latest target and source)",
                        self.input.mr_iid, self.input.target_branch
                    );
                    self.after_conflict_recheck()
                };
            }
            (
                FeedbackStage::ObserveChangesAfterRecheck,
                FeedbackFact::ChangesSinceModelRun(changed),
            ) => {
                self.stage = if changed {
                    self.has_new_changes = true;
                    FeedbackStage::StageAllForMergeCommit
                } else {
                    self.after_conflict_recheck()
                };
            }
            (FeedbackStage::ObserveStagedForMergeCommit, FeedbackFact::StagedChanges(staged)) => {
                self.stage = if staged {
                    FeedbackStage::CommitMergeResolution
                } else {
                    self.after_conflict_recheck()
                };
            }
            (
                FeedbackStage::ObserveConflictsBeforeReplies,
                FeedbackFact::MergeConflictsPresent(present),
            ) => {
                self.conflicts_unresolved = present;
                self.stage = self.after_conflicts_known();
            }
            (FeedbackStage::ObserveDiffHighlights, FeedbackFact::DiffHighlights(highlights)) => {
                self.diff_highlights = highlights;
                self.stage = self.after_diff_highlights();
            }
            (
                FeedbackStage::ObserveMergeRequestSurface,
                FeedbackFact::MergeRequestSurface(surface),
            ) => {
                if surface.has_conflicts {
                    self.conflicts_unresolved = true;
                    warn!(
                        "MR !{}: GitLab still reports merge conflicts after worker run",
                        self.input.mr_iid
                    );
                }
                self.surface_changed = self.input.surface_before.differs_from(&surface);
                self.stage = FeedbackStage::ObserveOriginHead;
            }
            (FeedbackStage::ObserveOriginHead, FeedbackFact::OriginHead(head)) => {
                self.branch_tip_changed = head.trim() != self.input.pre_agent_sha.trim();
                if self.surface_changed && !self.implicit_resolve_discussions() {
                    info!(
                        "MR !{} metadata changed without branch updates; discussions will remain open unless explicitly requested",
                        self.input.mr_iid
                    );
                }
                self.stage = if self.input.unresolved_ids.is_empty() {
                    FeedbackStage::ObserveUnresolvedIds
                } else {
                    self.ids_to_resolve = self.input.unresolved_ids.clone();
                    self.plan_replies()?
                };
            }
            (FeedbackStage::ObserveUnresolvedIds, FeedbackFact::UnresolvedDiscussionIds(ids)) => {
                self.ids_to_resolve = ids;
                self.stage = self.plan_replies()?;
            }
            (stage, fact) => {
                anyhow::bail!("feedback port answered {stage:?} with {fact:?}");
            }
        }
        Ok(())
    }

    fn apply_outcome(&mut self, outcome: FeedbackOutcome) -> Result<()> {
        match (self.stage.clone(), outcome) {
            // The metadata write is best effort; the port logs its failure.
            (FeedbackStage::UpdateMetadata, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::FetchAfterModelRun;
            }
            (FeedbackStage::FetchAfterModelRun, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::ObserveChangesAfterModelRun;
            }
            (FeedbackStage::StageResolvedConflicts, FeedbackOutcome::Staged(staged)) => {
                if staged {
                    info!(
                        "MR !{}: staged merge-conflict files with no remaining conflict markers",
                        self.input.mr_iid
                    );
                }
                self.stage = FeedbackStage::StageAllForMerge;
            }
            (FeedbackStage::StageAllForMerge, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::ObserveChangesAfterStaging;
            }
            (FeedbackStage::StageAllForCommit, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::ObserveStagedForCommit;
            }
            (FeedbackStage::CommitChanges, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::CompleteMerge;
            }
            (FeedbackStage::CompleteMerge, FeedbackOutcome::MergeCompleted(completed)) => {
                if completed {
                    self.has_new_changes = true;
                    info!(
                        "MR !{}: concluded in-progress merge with origin/{}",
                        self.input.mr_iid, self.input.target_branch
                    );
                }
                self.stage = FeedbackStage::ObserveConflictsAfterMerge;
            }
            (FeedbackStage::FetchForConflictRecheck, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::ObserveUpToDateWithTarget;
            }
            (FeedbackStage::StageAllForMergeCommit, FeedbackOutcome::Done) => {
                self.stage = FeedbackStage::ObserveStagedForMergeCommit;
            }
            (FeedbackStage::CommitMergeResolution, FeedbackOutcome::Done) => {
                self.stage = self.after_conflict_recheck();
            }
            (FeedbackStage::PushChanges, FeedbackOutcome::Done) => {
                info!(
                    "Pushed changes addressing feedback for MR !{}",
                    self.input.mr_iid
                );
                self.stage = FeedbackStage::ObserveMergeRequestSurface;
            }
            // Every GitLab reply, resolve, and plain comment is best
            // effort; the port logs what it could not post.
            (FeedbackStage::Reply(_), FeedbackOutcome::Done) => {}
            (_, FeedbackOutcome::Failed(e)) => return Err(e),
            (stage, _) => {
                anyhow::bail!("feedback port reported an unexpected outcome for {stage:?}");
            }
        }
        Ok(())
    }

    /// Where the progression goes once the conflict recheck is done: one
    /// more conflict probe unless conflicts are already known to remain.
    fn after_conflict_recheck(&self) -> FeedbackStage {
        if self.conflicts_unresolved {
            self.after_conflicts_known()
        } else {
            FeedbackStage::ObserveConflictsBeforeReplies
        }
    }

    fn after_conflicts_known(&self) -> FeedbackStage {
        if self.has_new_changes {
            FeedbackStage::ObserveDiffHighlights
        } else {
            self.after_diff_highlights()
        }
    }

    /// The push decision: conflicts hold the branch back, no changes means
    /// nothing to push.
    fn after_diff_highlights(&self) -> FeedbackStage {
        if self.has_new_changes && self.conflicts_unresolved {
            warn!(
                "MR !{}: not pushing — merge conflicts with origin/{} are still unresolved",
                self.input.mr_iid, self.input.target_branch
            );
            FeedbackStage::ObserveMergeRequestSurface
        } else if self.has_new_changes {
            FeedbackStage::PushChanges
        } else {
            info!(
                "Agent processed comments for MR !{} but made no code changes",
                self.input.mr_iid
            );
            FeedbackStage::ObserveMergeRequestSurface
        }
    }

    fn implicit_resolve_discussions(&self) -> bool {
        self.has_new_changes || self.branch_tip_changed
    }

    /// Decide the reply body and the ordered GitLab writes. Fails the run
    /// when the worker produced neither changes nor an explanation, which
    /// is what the pre-machine code did.
    fn plan_replies(&mut self) -> Result<FeedbackStage> {
        let should_post_plain_comment = self.input.resolution.post_plain_comment;
        let needs_reply_body = !self.ids_to_resolve.is_empty()
            || (self.input.plain_comments_present && should_post_plain_comment);

        if self.input.requires_conflict_resolution && self.conflicts_unresolved {
            info!(
                "MR !{}: merge conflicts with origin/{} remain; skipping GitLab replies until the branch merges cleanly",
                self.input.mr_iid, self.input.target_branch
            );
            return Ok(FeedbackStage::Finish);
        }

        self.reply_body = if needs_reply_body {
            let reply_raw = if let Some(block) =
                extract_worker_public_comment(self.input.resolution.public_comment.as_deref())
            {
                block
            } else if let Some(reply) = build_feedback_resolution_reply(
                &self.input.resolution,
                self.has_new_changes,
                self.diff_highlights.as_deref(),
            ) {
                reply
            } else {
                return Err(anyhow::anyhow!(
                    "worker produced no source changes and no feedback reply for MR !{}",
                    self.input.mr_iid
                ));
            };
            Some(strip_worker_reply_boilerplate(&reply_raw))
        } else {
            None
        };

        let resolve_discussions = feedback_discussions_may_be_resolved(
            &self.input.resolution,
            self.implicit_resolve_discussions(),
            self.conflicts_unresolved,
        );
        if self.conflicts_unresolved
            && self.input.resolution.mark_discussions_resolved == Some(true)
        {
            warn!(
                "MR !{}: ignoring agent request to mark discussions resolved while merge conflicts remain",
                self.input.mr_iid
            );
        }
        if !resolve_discussions && !self.ids_to_resolve.is_empty() {
            info!(
                "MR !{}: posting feedback replies without resolving discussions (no mark_discussions_resolved signal and no implicit resolving actions)",
                self.input.mr_iid
            );
        }

        self.reply_steps = plan_feedback_reply_steps(
            &self.ids_to_resolve,
            self.reply_body.is_some(),
            resolve_discussions,
            self.input.plain_comments_present,
            should_post_plain_comment,
        );
        Ok(FeedbackStage::Reply(0))
    }
}

/// Ask the feedback port one question.
fn observe_feedback(
    port: &dyn FeedbackTailPort,
    query: &FeedbackQuery,
    input: &FeedbackTailInput,
) -> Result<FeedbackFact> {
    Ok(match query {
        FeedbackQuery::ChangesSinceModelRun => {
            FeedbackFact::ChangesSinceModelRun(port.has_changes_since(&input.pre_agent_sha)?)
        }
        FeedbackQuery::MergeInProgress => FeedbackFact::MergeInProgress(port.merge_in_progress()?),
        FeedbackQuery::MergeConflictsPresent => {
            FeedbackFact::MergeConflictsPresent(port.merge_conflicts_present()?)
        }
        FeedbackQuery::StagedChanges => FeedbackFact::StagedChanges(port.has_staged_changes()?),
        FeedbackQuery::UpToDateWithTarget => {
            FeedbackFact::UpToDateWithTarget(port.up_to_date_with_target(&input.target_branch)?)
        }
        FeedbackQuery::DiffHighlights => {
            FeedbackFact::DiffHighlights(port.diff_highlights(&input.pre_agent_sha))
        }
        FeedbackQuery::MergeRequestSurface => {
            FeedbackFact::MergeRequestSurface(port.merge_request_surface(input.mr_iid)?)
        }
        FeedbackQuery::OriginHead => FeedbackFact::OriginHead(
            port.origin_head(&input.source_branch)
                .unwrap_or_else(|| input.pre_agent_sha.clone()),
        ),
        FeedbackQuery::UnresolvedDiscussionIds => {
            FeedbackFact::UnresolvedDiscussionIds(port.unresolved_discussion_ids(input.mr_iid))
        }
    })
}

/// Run the feedback progression to completion: observe, decide one step,
/// execute it, feed the result back.
fn drive_feedback(machine: &mut FeedbackMachine, port: &mut dyn FeedbackTailPort) -> Result<()> {
    loop {
        match machine.next_step() {
            FeedbackStep::Observe(query) => {
                let fact = observe_feedback(port, &query, &machine.input);
                machine.apply_fact(fact)?;
            }
            FeedbackStep::Act(action) => {
                let outcome = port.execute(&action);
                machine.apply_outcome(outcome)?;
            }
            FeedbackStep::Finish => return Ok(()),
        }
    }
}

/// The feedback progression backed by the real worktree and GitLab client.
struct LiveFeedbackTailPort<'a> {
    git_repo: &'a GitRepo,
    glab: &'a GitLabClient,
    mr_iid: u64,
    source_branch: String,
    target_branch: String,
}

impl FeedbackTailPort for LiveFeedbackTailPort<'_> {
    fn has_changes_since(&self, base_ref: &str) -> Result<bool> {
        self.git_repo.has_changes_since(base_ref)
    }

    fn merge_in_progress(&self) -> Result<bool> {
        self.git_repo.is_merge_in_progress()
    }

    fn merge_conflicts_present(&self) -> Result<bool> {
        self.git_repo.merge_conflicts_present()
    }

    fn has_staged_changes(&self) -> Result<bool> {
        self.git_repo.has_staged_changes()
    }

    fn up_to_date_with_target(&self, target_branch: &str) -> Result<bool> {
        self.git_repo.verify_up_to_date_with_target(target_branch)
    }

    fn diff_highlights(&self, base_ref: &str) -> Option<String> {
        build_diff_highlights_since(self.git_repo, base_ref)
    }

    fn merge_request_surface(&self, mr_iid: u64) -> Result<MrSurfaceObservation> {
        Ok(MrSurfaceObservation::from_mr(
            &self.glab.get_merge_request(mr_iid)?,
        ))
    }

    fn origin_head(&self, source_branch: &str) -> Option<String> {
        self.git_repo
            .rev_parse(&format!("origin/{source_branch}"))
            .ok()
    }

    fn unresolved_discussion_ids(&self, mr_iid: u64) -> Vec<String> {
        self.glab
            .get_unresolved_discussion_ids(mr_iid)
            .unwrap_or_default()
    }

    fn execute(&mut self, action: &FeedbackAction) -> FeedbackOutcome {
        match action {
            FeedbackAction::UpdateMrMetadata { title, description } => {
                if let Err(e) =
                    self.glab
                        .update_mr_title_description(self.mr_iid, title, description)
                {
                    warn!("Failed to update MR !{} metadata: {}", self.mr_iid, e);
                } else {
                    info!(
                        "Updated MR !{} title/description from agent feedback",
                        self.mr_iid
                    );
                }
                FeedbackOutcome::Done
            }
            FeedbackAction::FetchBranches => match self
                .git_repo
                .fetch_branches(&[self.target_branch.as_str(), self.source_branch.as_str()])
            {
                Ok(()) => FeedbackOutcome::Done,
                Err(e) => FeedbackOutcome::Failed(e),
            },
            FeedbackAction::StageResolvedConflicts => {
                match self.git_repo.stage_resolved_unmerged_paths() {
                    Ok(staged) => FeedbackOutcome::Staged(staged),
                    Err(e) => FeedbackOutcome::Failed(e),
                }
            }
            FeedbackAction::StageAll => match self.git_repo.add_all() {
                Ok(()) => FeedbackOutcome::Done,
                Err(e) => FeedbackOutcome::Failed(e),
            },
            FeedbackAction::Commit { message } => match self.git_repo.commit(message) {
                Ok(()) => FeedbackOutcome::Done,
                Err(e) => FeedbackOutcome::Failed(e),
            },
            FeedbackAction::CompleteMergeIfReady { message } => {
                match self.git_repo.complete_merge_if_ready(message) {
                    Ok(completed) => FeedbackOutcome::MergeCompleted(completed),
                    Err(e) => FeedbackOutcome::Failed(e),
                }
            }
            FeedbackAction::PushSourceBranch => match self.git_repo.push(&self.source_branch) {
                Ok(()) => FeedbackOutcome::Done,
                Err(e) => FeedbackOutcome::Failed(e),
            },
            FeedbackAction::ReplyToDiscussion {
                discussion_id,
                body,
            } => {
                if let Err(e) = self
                    .glab
                    .reply_to_discussion(self.mr_iid, discussion_id, body)
                {
                    warn!("Failed to reply to discussion {}: {}", discussion_id, e);
                }
                FeedbackOutcome::Done
            }
            FeedbackAction::ResolveDiscussion { discussion_id } => {
                if let Err(e) = self.glab.resolve_discussion(self.mr_iid, discussion_id) {
                    warn!("Failed to resolve discussion {}: {}", discussion_id, e);
                }
                FeedbackOutcome::Done
            }
            FeedbackAction::PostPlainComment { body } => {
                if let Err(e) = self.glab.add_mr_comment(self.mr_iid, body) {
                    warn!(
                        "Failed to post MR !{} reply for plain comments: {}",
                        self.mr_iid, e
                    );
                }
                FeedbackOutcome::Done
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Session file management (shared directory)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionFile {
    issue_iid: u64,
    /// 0 means no MR created yet (issue claimed but implementation not done).
    mr_iid: u64,
    #[serde(default)]
    agent_id: Option<String>,
    implementation_summary: Option<String>,
}

/// On startup, try to resume a session this worker previously owned.
/// A stored agent_id is only valid while the issue still has this worker's claim.
fn try_resume_session(state: &AgentState, scope_label: Option<&str>) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", &state.agent_id);
    let issue_prefix = format!("{}_issue_", &state.agent_id);

    let entries = match fs::read_dir(state.sessions_dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("Failed to read sessions directory: {}", e);
            return None;
        }
    };

    for entry in entries.flatten() {
        let Ok(file_name) = entry.file_name().into_string() else {
            continue;
        };
        if !file_name.starts_with(&issue_prefix) || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix(&issue_prefix)
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };

        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = state.load_session(issue_iid) else {
            continue;
        };

        // Fast path: session file records which agent owned it
        if let Some(ref stored_id) = session.agent_id {
            if stored_id == state.agent_id {
                let issue = match state.glab.get_issue(issue_iid) {
                    Ok(i) => i,
                    Err(e) => {
                        warn!(
                            "{}: Failed to verify issue #{} for session resume: {}, skipping",
                            &state.agent_id, issue_iid, e
                        );
                        continue;
                    }
                };

                if issue.state != "opened" {
                    let mr_iid = (session.mr_iid > 0).then_some(session.mr_iid);
                    state.abandon_closed_issue(issue_iid, mr_iid);
                    continue;
                }

                // Restart recovery only ever trusts a lease reconstructed
                // from labels just fetched live from GitLab above — never
                // the session file's own bookkeeping.
                let Some(lease) = ClaimLease::recover(
                    ClaimResource::Issue(issue_iid),
                    state.agent_id,
                    &issue.labels,
                ) else {
                    info!(
                        "{}: Session for issue #{} has no matching claim label, discarding stale session",
                        &state.agent_id, issue_iid
                    );
                    state.cleanup_session(issue_iid);
                    continue;
                };
                // Tracked by IID in `ActiveIssue` from here on, like every
                // other long-lived worker claim.
                lease.preserve();

                if !issue_in_scope(&issue, scope_label) {
                    continue;
                }

                if issue_has_worker_pending_label(&issue.labels) {
                    state.release_worker_hold_pending_gitlab_only(issue_iid);
                    continue;
                }

                if issue_has_worker_review_only_label(&issue.labels) {
                    state.release_worker_hold_review_only(issue_iid);
                    continue;
                }

                match resolve_tracked_mr_for_worker_issue(state.glab, issue_iid, session.mr_iid) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(state.glab, issue_iid);

                        let _ = claim::release(
                            state.glab,
                            ClaimResource::Issue(issue_iid),
                            state.agent_id,
                        );
                        let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

                        state.cleanup_session(issue_iid);
                        continue;
                    }
                    ResolvedTrackedMr::Track(mr_iid) => {
                        info!(
                            "{}: Found session file for issue #{} (MR: !{}), resuming",
                            &state.agent_id, issue_iid, mr_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: Some(mr_iid),
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: true,
                        });
                    }
                    ResolvedTrackedMr::None => {
                        info!(
                            "{}: Found session file for issue #{} (no MR yet), resuming",
                            &state.agent_id, issue_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: None,
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: false,
                        });
                    }
                }
            }
            // Session belongs to a different agent — skip
            continue;
        }

        // Fallback for old session files without agent_id: check GitLab labels
        match state.glab.get_issue(issue_iid) {
            Ok(issue) if issue.state != "opened" => {
                let mr_iid = (session.mr_iid > 0).then_some(session.mr_iid);
                state.abandon_closed_issue(issue_iid, mr_iid);
            }
            Ok(issue)
                if issue.labels.contains(&claim_label) && issue_in_scope(&issue, scope_label) =>
            {
                if issue_has_worker_pending_label(&issue.labels) {
                    state.release_worker_hold_pending_gitlab_only(issue_iid);
                    continue;
                }

                if issue_has_worker_review_only_label(&issue.labels) {
                    state.release_worker_hold_review_only(issue_iid);
                    continue;
                }

                match resolve_tracked_mr_for_worker_issue(state.glab, issue_iid, session.mr_iid) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(state.glab, issue_iid);

                        let _ = claim::release(
                            state.glab,
                            ClaimResource::Issue(issue_iid),
                            state.agent_id,
                        );
                        let _ = &state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);

                        state.cleanup_session(issue_iid);
                        continue;
                    }
                    ResolvedTrackedMr::Track(mr_iid) => {
                        info!(
                            "{}: Found unclaimed session for issue #{} with matching label, resuming (MR !{})",
                            &state.agent_id, issue_iid, mr_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: Some(mr_iid),
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: true,
                        });
                    }
                    ResolvedTrackedMr::None => {
                        info!(
                            "{}: Found unclaimed session for issue #{} with matching label, resuming",
                            &state.agent_id, issue_iid
                        );

                        return Some(ActiveIssue {
                            issue_iid,
                            mr_iid: None,
                            branch_name: Some(format!("issue-{}", issue_iid)),
                            mr_created: false,
                        });
                    }
                }
            }

            Ok(_) => {
                debug!(
                    "{}: Session for issue #{} exists but claim label not found",
                    &state.agent_id, issue_iid
                );
            }
            Err(e) => {
                warn!(
                    "{}: Failed to verify issue #{} on GitLab: {}, skipping",
                    &state.agent_id, issue_iid, e
                );
            }
        }
    }

    None
}

/// Scan all open GitLab issues for this worker's claim label.
/// Used as a fallback when the session file is missing (e.g. hard kill / crash).
fn find_claimed_issue(state: &AgentState, scope_label: Option<&str>) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", &state.agent_id);

    let issues = match state.glab.list_issues() {
        Ok(i) => i,
        Err(e) => {
            warn!(
                "{}: Failed to scan issues for orphaned claims: {}",
                &state.agent_id, e
            );

            return None;
        }
    };

    for issue in &issues {
        if issue.state != "opened" {
            continue;
        }

        if !issue.labels.contains(&claim_label) {
            continue;
        }

        if !issue_in_scope(issue, scope_label) {
            continue;
        }

        if issue_has_worker_pending_label(&issue.labels) {
            state.release_worker_hold_pending_gitlab_only(issue.iid);
            continue;
        }

        if issue_has_worker_review_only_label(&issue.labels) {
            state.release_worker_hold_review_only(issue.iid);
            continue;
        }

        match resolve_tracked_mr_for_worker_issue(state.glab, issue.iid, 0) {
            ResolvedTrackedMr::MergedCloseIssue => {
                close_issue_best_effort(state.glab, issue.iid);

                let _ = claim::release(state.glab, ClaimResource::Issue(issue.iid), state.agent_id);
                let _ = state.glab.remove_issue_label(issue.iid, WORKING_ON_LABEL);

                state.cleanup_session(issue.iid);
                continue;
            }
            ResolvedTrackedMr::Track(mr_iid) => {
                info!(
                    "{}: Found orphaned claim on issue #{} (MR: !{}), adopting it",
                    &state.agent_id, issue.iid, mr_iid
                );

                return Some(ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: Some(mr_iid),
                    branch_name: Some(format!("issue-{}", issue.iid)),
                    mr_created: true,
                });
            }
            ResolvedTrackedMr::None => {
                info!(
                    "{}: Found orphaned claim on issue #{} (no MR), adopting it",
                    &state.agent_id, issue.iid
                );

                return Some(ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: None,
                    branch_name: Some(format!("issue-{}", issue.iid)),
                    mr_created: false,
                });
            }
        }
    }

    None
}

/// Try to adopt an orphaned session file (issue has no claim label from any worker).
fn try_adopt_orphaned_session(
    state: &AgentState,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Option<ActiveIssue> {
    let entries = fs::read_dir(state.sessions_dir).ok()?;

    let issue_prefix = format!("{}_issue_", &state.agent_id);

    for entry in entries.flatten() {
        let Ok(file_name) = entry.file_name().into_string() else {
            continue;
        };

        if !file_name.starts_with(&issue_prefix) || !file_name.ends_with(".json") {
            continue;
        }

        let Some(issue_str) = file_name
            .strip_prefix(&issue_prefix)
            .and_then(|s| s.strip_suffix(".json"))
        else {
            continue;
        };

        let Ok(issue_iid) = issue_str.parse::<u64>() else {
            continue;
        };

        let Some(session) = state.load_session(issue_iid) else {
            continue;
        };

        // Skip sessions that belong to a specific agent — those should be
        // resumed by their owner via try_resume_session, not adopted.
        if session.agent_id.is_some() {
            continue;
        }

        let Ok(issue) = state.glab.get_issue(issue_iid) else {
            continue;
        };

        if issue.state != "opened" {
            state.cleanup_session(issue_iid);
            continue;
        }

        if !issue_in_scope(&issue, scope_label) {
            continue;
        }

        if issue_has_worker_pending_label(&issue.labels) {
            continue;
        }

        if issue_has_worker_review_only_label(&issue.labels) {
            continue;
        }

        // Skip if already claimed by someone
        if claim::is_claimed(&issue.labels) {
            continue;
        }

        let (mr_iid, mr_created) = if session.mr_iid > 0 {
            // Check if the MR is still open
            if let Ok(mr) = state.glab.get_merge_request(session.mr_iid) {
                if mr.state == "merged" || mr.state == "closed" {
                    info!(
                        "{}: Orphaned session for issue #{} has {} MR !{}, cleaning up",
                        &state.agent_id, issue_iid, mr.state, session.mr_iid
                    );

                    state.cleanup_session(issue_iid);
                    let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                    continue;
                }
            } else {
                continue;
            }
            (Some(session.mr_iid), true)
        } else {
            // No MR yet — check if one was created in the meantime
            match find_open_mr_for_issue(state.glab, issue_iid) {
                Some(mr) => (Some(mr), true),
                None => (None, false),
            }
        };

        // Try to claim this issue
        match claim::acquire(
            state.glab,
            ClaimResource::Issue(issue_iid),
            state.agent_id,
            shutdown,
        ) {
            Ok(ClaimAcquireOutcome::Won(lease)) => {
                info!(
                    "{}: Adopted orphaned issue #{} (MR: {})",
                    &state.agent_id,
                    issue_iid,
                    mr_iid.map_or("none".to_string(), |id| format!("!{}", id))
                );

                // Tracked by IID in `ActiveIssue` from here on, the same as
                // every other long-lived worker claim.
                lease.preserve();
                return Some(ActiveIssue {
                    issue_iid,
                    mr_iid,
                    branch_name: Some(format!("issue-{}", issue_iid)),
                    mr_created,
                });
            }
            _ => continue,
        }
    }

    None
}

// ---------------------------------------------------------------------------
// MR lookup
// ---------------------------------------------------------------------------

/// Find an *open* MR for an issue. Closed/merged MRs are ignored.
fn find_open_mr_for_issue(gitlab: &GitLabClient, issue_iid: u64) -> Option<u64> {
    let branch_name = format!("issue-{}", issue_iid);
    gitlab
        .find_open_mr_by_source_branch(&branch_name)
        .ok()
        .flatten()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_issue_number_from_branch(branch_name: &str) -> Result<u64> {
    if let Some(num_str) = branch_name.strip_prefix("issue-") {
        num_str
            .parse::<u64>()
            .context("Failed to parse issue number from branch name")
    } else {
        anyhow::bail!("Branch name does not match issue-<number> format")
    }
}

fn load_issue_context(gitlab: &GitLabClient, issue_number: u64) -> Result<String> {
    match gitlab.get_issue(issue_number) {
        Ok(issue) => Ok(format!(
            "Issue #{}: {}\n\nDescription:\n{}\n\nLabels: {}",
            issue.iid,
            issue.title,
            issue.description,
            issue.labels.join(", ")
        )),
        Err(_) => Ok(format!("Issue #{} (details not available)", issue_number)),
    }
}

fn abandon_mr(
    state: &AgentState,
    mr: &crate::agents::gitlab::MergeRequest,
    issue_iid: u64,
    reason: &str,
) -> Result<()> {
    state.glab.add_mr_comment(
        mr.iid,
        &format!(
            "Closing this MR — the issue cannot be resolved autonomously:\n\n{}",
            reason
        ),
    )?;
    let _ = state.glab.close_mr(mr.iid);

    let default_branch = state
        .git_repo
        .get_default_branch()
        .unwrap_or("main".to_string());
    let _ = state.git_repo.reset_hard();
    let _ = state.git_repo.checkout_remote_branch(&default_branch);
    let _ = state.git_repo.delete_local_branch(&mr.source_branch);

    let _ = state.glab.remove_issue_label(issue_iid, WORKING_ON_LABEL);
    state
        .glab
        .add_issue_label(issue_iid, ACTION_REQUIRED_LABEL)?;
    state.glab.add_issue_comment(
        issue_iid,
        &format!(
            "This issue requires additional human input before it can be implemented:\n\n{}",
            reason
        ),
    )?;

    state.cleanup_session(issue_iid);

    info!(
        "Abandoned MR !{} and rejected issue #{} with action-required",
        mr.iid, issue_iid
    );
    Ok(())
}

/// The text to post for a blocked outcome: the model's `public_comment` if it
/// wrote one, otherwise its `reason`, otherwise `default_text` (the schema
/// requires `reason`, but not that it be non-blank).
fn blocked_outcome_text(blocked: &BlockedOutcome, default_text: &str) -> String {
    if let Some(block) = extract_worker_public_comment(blocked.public_comment.as_deref()) {
        return block;
    }
    let trimmed = blocked.reason.trim();
    if !trimmed.is_empty() {
        return strip_internal_markers(trimmed);
    }
    default_text.to_string()
}

fn extract_cannot_resolve_reason(blocked: &BlockedOutcome) -> String {
    blocked_outcome_text(
        blocked,
        "The implementation cannot proceed without additional human input.",
    )
}

fn build_commit_message(title: &str, issue_iid: u64) -> String {
    let first_line = title.lines().next().unwrap_or(title).trim();
    // Truncate to a reasonable commit title length
    let title_truncated = if first_line.len() > 72 {
        format!("{}...", &first_line[..69])
    } else {
        first_line.to_string()
    };
    if issue_iid > 0 {
        format!("{}\n\nRefs #{}", title_truncated, issue_iid)
    } else {
        title_truncated
    }
}

fn extract_changes_summary(changes_summary: Option<&str>) -> String {
    if let Some(s) = changes_summary {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return strip_markdown_formatting(trimmed);
        }
    }
    "Changes made to address reviewer feedback.".to_string()
}

/// Strips leading `Resolved without code changes:` / `Addressed feedback:` from the posted reply.
/// If there is no substantive text after that prefix, returns the original string unchanged.
fn strip_worker_reply_boilerplate(text: &str) -> String {
    fn strip_diff_highlights_block(s: &str) -> String {
        let mut out: Vec<&str> = Vec::new();
        for line in s.lines() {
            if line.trim().eq_ignore_ascii_case("Diff highlights:") {
                break;
            }
            out.push(line);
        }
        out.join("\n").trim().to_string()
    }

    let trimmed = text.trim();
    const PREFIXES: &[&[u8]] = &[b"resolved without code changes:", b"addressed feedback:"];
    for prefix in PREFIXES {
        let b = trimmed.as_bytes();
        if b.len() >= prefix.len() && b[..prefix.len()].eq_ignore_ascii_case(prefix) {
            let suffix = trimmed[prefix.len()..].trim();
            if suffix.is_empty() {
                return trimmed.to_string();
            }
            let cleaned = strip_diff_highlights_block(suffix);
            let base = if cleaned.is_empty() { suffix } else { &cleaned };
            let sanitized = base.trim();
            return if sanitized.is_empty() {
                "Addressed the requested feedback.".to_string()
            } else {
                sanitized.to_string()
            };
        }
    }
    let cleaned = strip_diff_highlights_block(trimmed);
    let base = if cleaned.is_empty() {
        trimmed
    } else {
        &cleaned
    };
    let sanitized = base.trim();
    if sanitized.is_empty() {
        "Addressed the requested feedback.".to_string()
    } else {
        sanitized.to_string()
    }
}

/// Whether to call GitLab `resolve` on discussions after posting the worker reply.
fn should_resolve_mr_feedback_discussions(
    resolution: &FeedbackResolution,
    implicit_from_actions: bool,
) -> bool {
    resolution
        .mark_discussions_resolved
        .unwrap_or(implicit_from_actions)
}

fn build_feedback_resolution_reply(
    resolution: &FeedbackResolution,
    has_new_changes: bool,
    diff_highlights: Option<&str>,
) -> Option<String> {
    if has_new_changes {
        let summary = extract_changes_summary(resolution.changes_summary.as_deref());
        if let Some(diff) = diff_highlights
            && !diff.trim().is_empty()
        {
            return Some(format!(
                "Addressed feedback:\n\n{}\n\nDiff highlights:\n{}",
                summary, diff
            ));
        }
        return Some(format!("Addressed feedback:\n\n{}", summary));
    }
    extract_no_change_resolution_reason(resolution)
        .map(|reason| format!("Resolved without code changes:\n\n{}", reason))
}

fn build_diff_highlights_since(git_repo: &GitRepo, base_ref: &str) -> Option<String> {
    let files = git_repo.changed_files_since(base_ref).ok()?;
    if files.is_empty() {
        return None;
    }
    let shortstat = git_repo.diff_shortstat_since(base_ref).unwrap_or_default();

    const MAX_FILES: usize = 8;
    let mut lines: Vec<String> = files
        .iter()
        .take(MAX_FILES)
        .map(|f| format!("- {}", f))
        .collect();
    if files.len() > MAX_FILES {
        lines.push(format!("- ... and {} more files", files.len() - MAX_FILES));
    }
    if !shortstat.trim().is_empty() {
        lines.push(format!("- {}", shortstat.trim()));
    }
    Some(lines.join("\n"))
}

fn build_mr_diff_context(
    project_name: &str,
    mr: &crate::agents::gitlab::MergeRequest,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
) -> String {
    const MAX_DIFF_CHARS: usize = 120_000;
    const MAX_FILES: usize = 200;

    let mut source_note = "Source: GitLab MR changes API (matches MR diff view).".to_string();
    let (diff_stat, changed_files, mut diff_patch, overflow_note, diff_refs_line) =
        match gitlab.get_merge_request_changes(mr.iid) {
            Ok(snapshot) => {
                let mut files = snapshot.files;
                if files.len() > MAX_FILES {
                    files.truncate(MAX_FILES);
                }
                let stat = if files.is_empty() {
                    "(no changed files in MR changes payload)".to_string()
                } else {
                    format!("{} files changed", files.len())
                };
                let overflow_note = if snapshot.overflow {
                    Some("GitLab reported diff overflow; parts of the MR diff may be omitted.")
                } else {
                    None
                };
                let diff_refs = format!(
                    "Base commit: {}\nStart commit: {}\nHead commit: {}",
                    snapshot.base_sha.as_deref().unwrap_or("(unknown)"),
                    snapshot.start_sha.as_deref().unwrap_or("(unknown)"),
                    snapshot.head_sha.as_deref().unwrap_or("(unknown)")
                );
                (stat, files, snapshot.patch, overflow_note, Some(diff_refs))
            }
            Err(e) => {
                source_note = format!(
                    "Source: local git fallback (failed to read GitLab MR changes API: {}).",
                    e
                );
                (
                    git_repo
                        .diff_stat_against(&mr.target_branch)
                        .unwrap_or_else(|err| format!("(failed to compute diff stat: {})", err)),
                    git_repo
                        .changed_files_against(&mr.target_branch)
                        .unwrap_or_default(),
                    git_repo
                        .diff_patch_against(&mr.target_branch)
                        .unwrap_or_else(|err| format!("(failed to compute diff patch: {})", err)),
                    None,
                    None,
                )
            }
        };

    if diff_patch.len() > MAX_DIFF_CHARS {
        truncate_utf8_string_in_place(&mut diff_patch, MAX_DIFF_CHARS);
        diff_patch.push_str(
            "\n\n[diff truncated by potlatch: patch exceeded size limit; full diff may contain additional context]\n",
        );
    }
    if diff_patch.trim().is_empty() {
        diff_patch = "(No patch diff available.)".to_string();
    }
    let files_block = if changed_files.is_empty() {
        "No changed files detected.".to_string()
    } else {
        changed_files
            .into_iter()
            .map(|f| format!("- {}", f))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "# Merge Request Diff Context\n\nProject: {project}\nMR: !{iid} {title}\nSource branch: {source}\nTarget branch: {target}\n\n{source_note}\n\n## MR diff refs\n{diff_refs}\n\n## Diff stat\n{stat}\n\n## Changed files\n{files}\n\n## Patch\n```diff\n{patch}\n```\n{overflow_line}",
        project = project_name,
        iid = mr.iid,
        title = mr.title,
        source = mr.source_branch,
        target = mr.target_branch,
        source_note = source_note,
        diff_refs = diff_refs_line.unwrap_or("Unavailable (local git fallback)".to_string()),
        stat = if diff_stat.trim().is_empty() {
            "(no stat output)"
        } else {
            diff_stat.trim()
        },
        files = files_block,
        patch = diff_patch,
        overflow_line = overflow_note.unwrap_or("")
    )
}

/// Truncate a UTF-8 string to at most `max_bytes` without splitting a multibyte character.
fn truncate_utf8_string_in_place(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

fn format_comments_for_prompt(comments: &[crate::agents::gitlab::Comment]) -> String {
    comments
        .iter()
        .map(|c| c.format_for_prompt())
        .collect::<Vec<_>>()
        .join("\n")
}

struct CombinedMrFeedbackContextInput<'a> {
    project_name: &'a str,
    mr: &'a crate::agents::gitlab::MergeRequest,
    issue_context: &'a str,
    implementation_summary: &'a str,
    merge_conflict_status: &'a str,
    unresolved_comments_text: &'a str,
    plain_comments_text: &'a str,
    all_comments_text: &'a str,
    diff_context: &'a str,
}

fn build_combined_mr_feedback_context(input: CombinedMrFeedbackContextInput<'_>) -> String {
    format!(
        "# Merge Request Feedback + Diff Context\n\nProject: {project_name}\nMR: !{mr_iid} {mr_title}\nSource branch: {source_branch}\nTarget branch: {target_branch}\n\n{merge_conflict_status}\n\n## Original issue context\n{issue_context}\n\n## Original implementation summary\n{implementation_summary}\n\n## MR description\n{mr_description}\n\n## Unresolved MR comments to address\n{unresolved_comments_text}\n\n## Plain MR comments to consider\n{plain_comments_text}\n\n## Full MR comment history for context\n{all_comments_text}\n\n## MR diff context\n{diff_context}\n",
        project_name = input.project_name,
        mr_iid = input.mr.iid,
        mr_title = input.mr.title,
        source_branch = input.mr.source_branch,
        target_branch = input.mr.target_branch,
        merge_conflict_status = input.merge_conflict_status,
        issue_context = input.issue_context,
        implementation_summary = input.implementation_summary,
        mr_description = input.mr.description,
        unresolved_comments_text = if input.unresolved_comments_text.trim().is_empty() {
            "No unresolved comments.".to_string()
        } else {
            input.unresolved_comments_text.to_string()
        },
        plain_comments_text = if input.plain_comments_text.trim().is_empty() {
            "No plain MR comments.".to_string()
        } else {
            input.plain_comments_text.to_string()
        },
        all_comments_text = if input.all_comments_text.trim().is_empty() {
            "No MR comments.".to_string()
        } else {
            input.all_comments_text.to_string()
        },
        diff_context = input.diff_context
    )
}

fn build_merge_conflict_status_section(
    mr: &crate::agents::gitlab::MergeRequest,
    requires_conflict_resolution: bool,
    local_merge_clean: bool,
    git_repo: &GitRepo,
) -> Result<String> {
    let unmerged = git_repo.list_unmerged_paths()?;
    let marker_files = git_repo.list_conflict_marker_files()?;
    let target_sha = git_repo
        .remote_short_sha(&mr.target_branch)
        .unwrap_or_else(|_| "(unknown)".to_string());
    let source_sha = git_repo
        .remote_short_sha(&mr.source_branch)
        .unwrap_or_else(|_| "(unknown)".to_string());
    let mut lines = vec![
        "## Merge conflict status (verified by Potlatch — trust this section)".to_string(),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.target_branch, target_sha
        ),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.source_branch, source_sha
        ),
        format!(
            "- GitLab reports merge conflicts on this MR: {}",
            if mr.has_conflicts { "yes" } else { "no" }
        ),
        format!(
            "- Local merge of `origin/{}` into `origin/{}` at task start: {}",
            mr.target_branch,
            mr.source_branch,
            if local_merge_clean {
                "clean (no conflict markers introduced locally)"
            } else {
                "FAILED — conflict markers and/or unmerged paths are present in your workspace"
            }
        ),
        format!(
            "- Merge currently in progress in workspace: {}",
            if git_repo.is_merge_in_progress().unwrap_or(false) {
                "yes — resolve every unmerged file, then Potlatch will conclude the merge commit"
            } else {
                "no"
            }
        ),
        format!(
            "- Conflict resolution required this run: {}",
            if requires_conflict_resolution {
                "yes"
            } else {
                "no"
            }
        ),
    ];

    if unmerged.is_empty() {
        lines.push("- Unmerged paths in workspace: none".to_string());
    } else {
        lines.push("- Unmerged paths in workspace:".to_string());
        for path in &unmerged {
            lines.push(format!("  - {path}"));
        }
    }

    if marker_files.is_empty() {
        lines.push("- Files containing `<<<<<<<` conflict markers: none".to_string());
    } else {
        lines.push("- Files containing `<<<<<<<` conflict markers:".to_string());
        for path in &marker_files {
            lines.push(format!("  - {path}"));
        }
    }

    if requires_conflict_resolution {
        lines.push(
            "- After editing conflicted files, remove every conflict marker. Potlatch will `git add` resolved files and conclude the merge commit; you do not need to run git commands.".to_string(),
        );
    }

    Ok(lines.join("\n"))
}

fn feedback_discussions_may_be_resolved(
    resolution: &FeedbackResolution,
    implicit_from_actions: bool,
    conflicts_unresolved: bool,
) -> bool {
    if conflicts_unresolved {
        return false;
    }
    should_resolve_mr_feedback_discussions(resolution, implicit_from_actions)
}

/// The reason to post for a "resolved without code changes" reply: the
/// `reason` field if the model explained itself, otherwise `changes_summary`
/// (the model sometimes describes a no-op resolution there instead).
fn extract_no_change_resolution_reason(resolution: &FeedbackResolution) -> Option<String> {
    [&resolution.reason, &resolution.changes_summary]
        .into_iter()
        .flatten()
        .map(|text| text.trim())
        .find(|text| !text.is_empty())
        .map(|text| strip_markdown_formatting(&strip_internal_markers(text)))
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

fn format_issue_comments_for_worker_context(gitlab: &GitLabClient, issue_iid: u64) -> String {
    let comments = match gitlab.get_issue_comments(issue_iid) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Worker: failed to fetch GitLab issue comments for #{}: {}",
                issue_iid, e
            );
            Vec::new()
        }
    };
    if comments.is_empty() {
        "_No comments on this issue yet._\n".to_string()
    } else {
        comments
            .iter()
            .map(|c| c.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

fn worker_issue_context_markdown(issue: &IssueObservation, gitlab_comments_text: &str) -> String {
    format!(
        "# Issue Context\n\nIssue: #{} {}\n\n## Description\n{}\n\n## GitLab issue comments\n\n{}\n",
        issue.iid, issue.title, issue.description, gitlab_comments_text
    )
}

fn build_implementation_prompt(
    state: &AgentState,
    issue: &IssueObservation,
    gitlab_comments_text: &str,
) -> Result<String> {
    let context_content = worker_issue_context_markdown(issue, gitlab_comments_text);
    // Write to disk for archival, but inject content into the prompt.
    let _context_path = write_task_context_file(
        state.sessions_dir,
        &format!("{}-issue-{}.md", state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(false);
    let notes_rules = get_notes_rules();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are implementing a feature for a software project in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CONTEXT:
- The task context above is included in your prompt: it contains this issue's **description** and **every GitLab issue comment** at the time the task started. That is your primary written spec.
- Labels on the issue (e.g. priority) are visible in GitLab; infer scope from description + comments + `AGENTS.md`.

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. The task context is already included above. Do NOT use read on it — review it from your prompt, then start implementing.
3. Analyze the issue and comments carefully
4. Estimate the number of changed lines:
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
5. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the feature can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be implemented as one unit, proceed with implementation
   - Otherwise, report that the issue needs splitting and explain the estimated line count and decomposition
6. If the issue is unclear or missing critical information that makes implementation impossible, report that clarification is needed and explain what information is missing and why.
7. If the issue requires large unrelated feature work, report that the issue needs splitting and explain the decomposition.
8. If at any point you determine the issue simply cannot be implemented without additional human input that you cannot infer or assume (e.g. missing API credentials, undocumented external system dependencies, contradictory requirements), report that clarification is needed and explain precisely what input is required.
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the issue, REJECT it. Do not guess or produce speculative code.
- If you believe the implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

9. If the issue is clear, focused, and reasonably sized (or cannot be split), implement ONLY what is asked
10. Make all necessary code changes autonomously
11. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
12. {}

Proceed with the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_content,
        common_requirements,
        scope_rules,
        notes_rules
    );

    Ok(prompt)
}

fn build_continuation_prompt(
    state: &AgentState,
    issue: &IssueObservation,
    gitlab_comments_text: &str,
) -> Result<String> {
    let context_content = worker_issue_context_markdown(issue, gitlab_comments_text);
    // Write to disk for archival, but inject content into the prompt.
    let _context_path = write_task_context_file(
        state.sessions_dir,
        &format!("{}-issue-{}.md", &state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(true);
    let notes_rules = get_notes_rules();

    let prompt = format!(
        r#"SYSTEM: You are an autonomous coding agent.

You are continuing work on an existing feature branch in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT (already included below — do NOT read it from disk):
{}

CONTEXT:
- The task context above contains this issue's **description** and **every GitLab issue comment** at task start.
- A branch for this issue already exists with previous work
- You are continuing the implementation from where it was left off
- Review the existing code changes in this branch
- Complete any remaining work needed to fully implement the issue

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. The task context is already included above. Do NOT use read on it — review it from your prompt.
3. Review the existing changes in the current branch
4. Analyze what has been done and what remains
5. Estimate total changed lines (including existing + remaining work):
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
6. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the remaining work can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be completed as one unit, proceed with implementation
   - Otherwise, report that the issue needs splitting and explain the estimated line count and decomposition
7. If the issue is unclear or missing critical information that makes implementation impossible, report that clarification is needed and explain what information is missing and why.
8. If the issue requires large unrelated feature work, report that the issue needs splitting and explain the decomposition.
9. If at any point you determine the remaining work cannot be completed without additional human input that you cannot infer or assume, report that clarification is needed and explain precisely what input is required.
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the remaining work, REJECT it. Do not guess or produce speculative code.
- If you believe the total implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

10. If the issue is clear, focused, and reasonably sized (or cannot be split), continue the implementation
11. ONLY implement what the issue asks for, nothing more
12. Complete any remaining work autonomously
13. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
14. {}

Proceed with continuing the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_content,
        common_requirements,
        scope_rules,
        notes_rules
    );

    Ok(prompt)
}

fn get_common_requirements() -> &'static str {
    r#"CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You MUST delete, rename, move, or create files as needed — do not ask permission or suggest it
- You MUST NOT ask the user for input, confirmation, or decisions — decide autonomously
- You MUST NOT produce output that suggests actions for a human to take — YOU take those actions
- Leave staging, committing, pushing, and merge request creation to the system
- If information is missing, document what's needed in your response (do not ask interactively)
- If you are making code changes you MUST stick to AGENTS.md in the project strictly
- Read the issue comments carefully — they may contain guidance from the PMO agent on how to proceed. PMO guidance appears as a comment starting with **PMO guidance for the worker agent:** — treat the body of that comment as authoritative worker instructions and follow it exactly.
- Before finishing, update repo-root notes.md only when you have bullets that pass the NOTES.MD rules below: not a recap of your MR, not generic best-practice slides, not meta about notes — if nothing qualifies, leave the file unchanged. Never paste notes.md into MR metadata or GitLab comments

NO WORKAROUNDS — STRICTLY PROHIBITED:
- NEVER apply a workaround, hack, or shortcut to make code "work" without addressing the root cause.
- The ONLY exception is an explicit instruction in a code comment or doc comment within the existing codebase that says to use a specific approach. In that case, follow the comment's instruction exactly.
- If the correct fix is unclear or too large, reject the issue rather than shipping a workaround.

RESOURCE AWARENESS — MANDATORY:
- Before committing to an implementation approach, evaluate its resource footprint: memory, CPU, disk I/O, file descriptors, and goroutine/thread usage. An approach that has the potential to exhaust machine resources is UNACCEPTABLE, even if it produces correct output.
- Specifically avoid: unbounded buffering (loading entire files/datasets into memory), O(n^2) or worse algorithms on large inputs, spawning unbounded goroutines/threads without a semaphore, holding large data in memory across iterations, redundant re-reads of large files, or creating temp files without cleanup.
- If the correct, resource-safe implementation is too large for a single MR, report that the issue needs splitting and explain the resource concern.
- If you are unsure whether your approach is resource-safe under production-scale inputs, report that it cannot be implemented safely and explain the concern. Do not ship code that might OOM, hang, or exhaust file descriptors on real data."#
}

fn get_evidence_bound_scope_bullets() -> &'static str {
    r#"- Only modify code that is strongly supported by the issue title, issue description, GitLab issue comments, or current unresolved MR feedback. If a change is merely adjacent, speculative, weakly coupled, or "nice to have", do not make it.
- Treat the issue title, issue description, and GitLab issue comments as the strict boundary of allowed code changes.
- Every production code change must have a clear, direct link to those issue details or comments; if you cannot explain that link in one sentence, do not make the change.
- Do NOT add features, refactors, or integrations not described in the issue.
- Do NOT over-engineer: avoid new abstractions, compatibility layers, broad refactors, generalized frameworks, hypothetical future cases, unrelated edge cases, broad compatibility, "while here" cleanup, or behavior changes that are not directly required by the task evidence.
- The only acceptable unrelated edits are lint-only fixes or unit-test-only fixes needed to validate the requested implementation."#
}

fn get_feedback_scope_rules() -> String {
    format!(
        r#"FEEDBACK SCOPE RULES:
- Treat the current unresolved reviewer feedback as the PRIMARY request. The original issue, MR description, and older comments provide context, but they do not override the current unresolved feedback.
- Do NOT mutate, reinterpret, or weaken the current feedback to fit an old implementation choice or old commit. Change the code so it strictly complies with the reviewer feedback.
- Scope each change directly to the unresolved comment, its inline code location, the linked issue, or the MR description. Do not use feedback handling as an opportunity for broader cleanup, redesign, abstraction, feature expansion, or compatibility shims unless the reviewer explicitly asked for it.
{}"#,
        get_evidence_bound_scope_bullets()
    )
}

fn get_scope_rules(is_continuation: bool) -> String {
    let line_context = if is_continuation {
        "Review existing changes and estimate remaining work"
    } else {
        "Before starting implementation, estimate if the changes will be significantly larger than the limits below"
    };

    format!(
        r#"SCOPE RULES:
- ONLY implement what the issue specifically asks for, nothing more
{}
- CHANGE SIZE LIMITS (STRICT):
  * Non-test, non-generated code: ~500 changed lines maximum
  * Total changes including tests: ~1500 changed lines maximum
  * Do NOT count auto-generated code (files with "generated by", "auto-generated", "DO NOT EDIT" comments) toward either limit
- {}
- If non-test code changes would be substantially larger than ~500 lines, or total changes larger than ~1500 lines:
  * First, carefully evaluate if the feature can be split into smaller, independent pieces
  * If you are VERY SURE the feature CANNOT be split and MUST be {} as one atomic unit, you may proceed
  * Otherwise, report that the issue needs splitting and explain the estimated line count and decomposition
- If implementing the issue requires a large feature integration that is mainly unrelated to the task, report that the issue needs splitting and explain why it is too broad."#,
        get_evidence_bound_scope_bullets(),
        line_context,
        if is_continuation {
            "completed"
        } else {
            "implemented"
        }
    )
}

fn get_notes_rules() -> &'static str {
    r#"NOTES.MD (agent-maintained in the repo — edit before you finish **only if** you earn real bullets):
- Open or create notes.md at the repository root. Append **0–3** new "- " lines this run (often **0**). Each line is **one** short sentence capturing a **genuine surprise, near-mistake, or emotional friction** from the run — something you almost got wrong or that wasted time — expressed so a stranger learns the *habit of noticing*, not the *contents of this MR*.

HARD REJECT (if a line violates any of these, delete it — do not append):
- Backticks, file or directory paths, dotted import paths, or shell snippets tied to this repo layout.
- Names of this project's classes, functions, modules, config keys, env vars, or third-party symbols (anything a reader would grep for in *this* tree).
- Documentation-style explanation of how this tree works — put that in MR description or code, not notes.
- Obvious restatements of AGENTS.md / the issue / standard practice, and hollow platitudes ("write tests", "read carefully").
- **Abstract recap of your implementation** dressed as advice: if the line is basically "what we changed" or "how we tested" in generic software-engineering words (single source of truth for identifiers, test through public API, stub optional deps, match CI cwd when running tests, etc.) **without** a sharp *I almost messed this up because…* insight, it is **rubbish** — delete it.
- **Meta about notes.md** (e.g. "keep notes short and transferable", "agent-maintained") — never; that parrots instructions, not experience.
- Anything that could appear unchanged on a generic "best practices" slide deck with no story of *what tripped you* — delete it.

SELF-CHECK before saving: (1) "Would this still help on a **different** repo without opening our source?" If no, drop. (2) "Could I have written this bullet **before** starting this task?" If yes, drop. If **no** line survives, append nothing this run.

BAD STYLE (examples of rubbish — do not imitate):
- In-repo code tours (paths, classes, long semicolon chains).
- "Timeless" bullets that are really your MR summary: composable naming over literals, property vs field assumptions, stub heavy imports, or environment wiring — unless each line names a **non-obvious failure mode you personally hit** in **one** concrete clause (still without paths or symbol names).

Stay concise; no secrets. That file is committed with your other changes. Never paste or quote any text from notes.md into MR metadata, issue comments, or merge request comments."#
}

fn extract_split_reason(blocked: &BlockedOutcome) -> String {
    blocked_outcome_text(
        blocked,
        "This issue is too broad and requires large unrelated feature work. Please split it into smaller, focused issues with detailed descriptions.",
    )
}

fn extract_clarification(blocked: &BlockedOutcome) -> String {
    blocked_outcome_text(
        blocked,
        "This issue needs clarification. Please provide more details.",
    )
}

fn extract_cannot_implement_reason(blocked: &BlockedOutcome) -> String {
    blocked_outcome_text(
        blocked,
        "This issue cannot be implemented as specified. Please provide more details.",
    )
}

/// Clean the worker's public comment text from a `handoff` branch's
/// `public_comment` field. Stray internal markers are stripped as
/// defense-in-depth before this text reaches a GitLab surface.
fn extract_worker_public_comment(public_comment: Option<&str>) -> Option<String> {
    let s = public_comment?.trim();
    if s.is_empty() {
        return None;
    }
    Some(strip_internal_markers(s))
}

/// Prefix for labels that park an issue until a dependency issue is closed.
/// The full label is `waiting-on-issue:#N` where N is the dependency issue IID.
const WAITING_ON_ISSUE_LABEL_PREFIX: &str = "waiting-on-issue:#";

/// Build the `waiting-on-issue:#N` label for a dependency issue IID.
fn waiting_on_issue_label(issue_iid: u64) -> String {
    format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{issue_iid}")
}

/// Extract the dependency issue IID from a `waiting-on-issue:#N` issue label.
fn extract_waiting_on_issue_iid(labels: &[String]) -> Option<u64> {
    labels.iter().find_map(|l| {
        l.strip_prefix(WAITING_ON_ISSUE_LABEL_PREFIX)
            .and_then(|s| s.parse::<u64>().ok())
    })
}

/// Extract the MR title the agent emitted, falling back to the issue title
/// (not a generic placeholder like "Implementation changes") when the agent
/// didn't provide one. The issue title is always meaningful and specific to
/// the work, so it's a far better default than a placeholder that produces
/// a stream of indistinguishable MRs.
fn extract_mr_title(mr_title: Option<&str>, issue_title: &str) -> String {
    if let Some(s) = mr_title {
        let cleaned = strip_markdown_formatting(s.trim());
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    // No title provided by the agent — fall back to the issue title, which
    // is always specific to the work. Never use a generic placeholder.
    issue_title.to_string()
}

/// Extract an explicit MR title the agent emitted (for metadata updates on
/// follow-up turns). Returns `None` when the agent didn't provide one, so the
/// caller can distinguish "no title provided" from "title provided". Unlike
/// [`extract_mr_title`], this does NOT fall back to the issue title — a
/// metadata update should only overwrite the title when the agent explicitly
/// said to.
fn extract_explicit_mr_title(mr_title: Option<&str>) -> Option<String> {
    let cleaned = strip_markdown_formatting(mr_title?.trim());
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn strip_markdown_formatting(s: &str) -> String {
    let result = s
        .trim_start_matches('*')
        .trim_end_matches('*')
        .trim_start_matches('`')
        .trim_end_matches('`')
        .trim();
    result.to_string()
}

/// Defense-in-depth cleanup for `mr_description` text: strips any stray
/// public-comment blocks and internal field-name-looking lines the model
/// might echo into the description despite the typed contract.
fn sanitize_mr_description_text(s: &str) -> String {
    let stripped = strip_public_comment_blocks(s);
    let filtered: Vec<&str> = stripped
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("CHANGES_SUMMARY:")
                && !t.starts_with("MARK_DISCUSSIONS_RESOLVED:")
                && !t.starts_with("POST_PLAIN_COMMENT:")
        })
        .collect();
    filtered.join("\n").trim().to_string()
}

/// Extract the MR description the agent emitted via the `handoff` tool's
/// `mr_description` field, falling back to a generic placeholder when absent.
fn extract_mr_description(mr_description: Option<&str>) -> String {
    if let Some(s) = mr_description {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return sanitize_mr_description_text(trimmed);
        }
    }
    "Implementation completed.".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::conformance;

    #[test]
    fn worker_notes_rules_do_not_redeclare_the_structured_output_contract() {
        let rules = get_notes_rules();

        assert!(rules.contains("NOTES.MD"));
        assert!(!rules.contains("handoff"));
        assert!(!rules.contains("outcome"));
        assert!(!rules.contains("output contract"));
    }

    // -----------------------------------------------------------------
    // Session persistence: tolerant policy. Corrupt/unsupported session
    // files have historically been treated as "no session" rather than
    // failing the worker cycle; `GitRepo::new`/`GitLabClient::for_test` are
    // file/network-free constructors so this exercises the real
    // `AgentState` session methods without touching git or GitLab.
    // -----------------------------------------------------------------

    /// Owns the resources a test [`AgentState`] borrows from, standing in
    /// for the [`GitLabAgentRuntime`] fields the worker cycle needs.
    struct TestRuntime {
        project_name: String,
        agent_id: String,
        sessions_dir: String,
        git_repo: GitRepo,
        glab: GitLabClient,
    }

    fn test_runtime(sessions_dir: &str, agent_id: &str) -> TestRuntime {
        TestRuntime {
            project_name: "test-project".to_string(),
            agent_id: agent_id.to_string(),
            sessions_dir: sessions_dir.to_string(),
            git_repo: GitRepo::new(
                std::env::temp_dir().to_string_lossy().into_owned(),
                Arc::new(AtomicBool::new(false)),
            ),
            glab: GitLabClient::for_test("/tmp/unused-repo"),
        }
    }

    fn test_agent_state(rt: &TestRuntime) -> AgentState<'_> {
        AgentState {
            project_name: &rt.project_name,
            agent_id: &rt.agent_id,
            sessions_dir: &rt.sessions_dir,
            git_repo: &rt.git_repo,
            glab: &rt.glab,
        }
    }

    fn worker_session_test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "potlatch-worker-session-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn load_session_returns_none_when_no_file_exists_yet() {
        let dir = worker_session_test_dir("missing");
        let rt = test_runtime(&dir.to_string_lossy(), "worker-0");
        let state = test_agent_state(&rt);
        assert!(state.load_session(1).is_none());
    }

    #[test]
    fn save_then_load_session_round_trips_through_the_v1_envelope() {
        let dir = worker_session_test_dir("roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "worker-1");
        let state = test_agent_state(&rt);

        state.save_session(42, 7).unwrap();
        let session = state.load_session(42).unwrap();
        assert_eq!(session.issue_iid, 42);
        assert_eq!(session.mr_iid, 7);
        assert_eq!(session.agent_id.as_deref(), Some("worker-1"));

        let on_disk: serde_json::Value =
            serde_json::from_slice(&fs::read(state.session_file_path(42)).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["issue_iid"], 42);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_session_reads_the_legacy_unversioned_format_and_migrates_it() {
        let dir = worker_session_test_dir("legacy");
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "worker-2");
        let state = test_agent_state(&rt);
        let path = state.session_file_path(9);

        // The bare pre-envelope payload written by older builds.
        fs::write(
            &path,
            serde_json::to_vec(&SessionFile {
                issue_iid: 9,
                mr_iid: 3,
                agent_id: Some("worker-2".to_string()),
                implementation_summary: Some("done".to_string()),
            })
            .unwrap(),
        )
        .unwrap();

        let session = state.load_session(9).unwrap();
        assert_eq!(session.mr_iid, 3);
        assert_eq!(session.implementation_summary.as_deref(), Some("done"));

        // Transparently migrated to the v1 envelope on disk.
        let on_disk: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_session_tolerates_corrupt_files_by_warning_and_returning_none() {
        let dir = worker_session_test_dir("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "worker-3");
        let state = test_agent_state(&rt);
        let path = state.session_file_path(5);
        fs::write(&path, b"not json").unwrap();

        assert!(state.load_session(5).is_none());
        // Quarantined beside the original rather than deleted outright.
        assert!(!path.exists());
        let quarantined: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains("quarantined"))
            .collect();
        assert_eq!(quarantined.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_session_tolerates_an_unsupported_envelope_version() {
        let dir = worker_session_test_dir("unsupported-version");
        fs::create_dir_all(&dir).unwrap();
        let rt = test_runtime(&dir.to_string_lossy(), "worker-4");
        let state = test_agent_state(&rt);
        let path = state.session_file_path(6);
        fs::write(&path, br#"{"version":7,"state":{}}"#).unwrap();

        assert!(state.load_session(6).is_none());
        assert!(!path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // `handle_mr_comments` characterization: metadata-update decision.
    // -----------------------------------------------------------------

    #[test]
    fn plan_mr_metadata_update_is_none_when_nothing_changed() {
        let output = FeedbackResolution {
            mr_title: Some("Same title".to_string()),
            mr_description: Some("Same description".to_string()),
            ..Default::default()
        };
        assert!(plan_mr_metadata_update("Same title", "Same description", &output).is_none());
    }

    #[test]
    fn plan_mr_metadata_update_is_none_for_default_placeholder_description() {
        // "Implementation completed." is the extractor's fallback default,
        // not an agent-authored description, so it must never trigger a write.
        let output = FeedbackResolution::default();
        assert!(plan_mr_metadata_update("Title", "Old description", &output).is_none());
    }

    #[test]
    fn plan_mr_metadata_update_changes_title_only_keeps_current_description() {
        let output = FeedbackResolution {
            mr_title: Some("New title".to_string()),
            ..Default::default()
        };
        let update = plan_mr_metadata_update("Old title", "Old description", &output).unwrap();
        assert_eq!(update.title, "New title");
        assert_eq!(update.description, "Old description");
    }

    #[test]
    fn plan_mr_metadata_update_changes_description_only_keeps_current_title() {
        let output = FeedbackResolution {
            mr_description: Some("New description".to_string()),
            ..Default::default()
        };
        let update = plan_mr_metadata_update("Old title", "Old description", &output).unwrap();
        assert_eq!(update.title, "Old title");
        assert_eq!(update.description, "New description");
    }

    #[test]
    fn plan_mr_metadata_update_changes_both_when_both_differ() {
        let output = FeedbackResolution {
            mr_title: Some("New title".to_string()),
            mr_description: Some("New description".to_string()),
            ..Default::default()
        };
        let update = plan_mr_metadata_update("Old title", "Old description", &output).unwrap();
        assert_eq!(update.title, "New title");
        assert_eq!(update.description, "New description");
    }

    // -----------------------------------------------------------------
    // `handle_mr_comments` characterization: reply/resolve/plain-comment
    // ordering tail.
    // -----------------------------------------------------------------

    #[test]
    fn plan_feedback_reply_steps_is_empty_without_a_reply_body() {
        let steps = plan_feedback_reply_steps(
            &["d1".to_string()],
            false, // no reply body
            true,
            true,
            true,
        );
        assert!(steps.is_empty());
    }

    #[test]
    fn plan_feedback_reply_steps_replies_before_resolving_each_discussion() {
        let ids = vec!["d1".to_string(), "d2".to_string()];
        let steps = plan_feedback_reply_steps(&ids, true, true, false, false);
        assert_eq!(
            steps,
            vec![
                FeedbackReplyStep::Reply {
                    discussion_id: "d1".to_string()
                },
                FeedbackReplyStep::Resolve {
                    discussion_id: "d1".to_string()
                },
                FeedbackReplyStep::Reply {
                    discussion_id: "d2".to_string()
                },
                FeedbackReplyStep::Resolve {
                    discussion_id: "d2".to_string()
                },
            ]
        );
    }

    #[test]
    fn plan_feedback_reply_steps_replies_without_resolving_when_not_permitted() {
        let ids = vec!["d1".to_string()];
        let steps = plan_feedback_reply_steps(&ids, true, false, false, false);
        assert_eq!(
            steps,
            vec![FeedbackReplyStep::Reply {
                discussion_id: "d1".to_string()
            }]
        );
    }

    #[test]
    fn plan_feedback_reply_steps_appends_plain_comment_last() {
        let ids = vec!["d1".to_string()];
        let steps = plan_feedback_reply_steps(&ids, true, true, true, true);
        assert_eq!(
            steps,
            vec![
                FeedbackReplyStep::Reply {
                    discussion_id: "d1".to_string()
                },
                FeedbackReplyStep::Resolve {
                    discussion_id: "d1".to_string()
                },
                FeedbackReplyStep::PostPlainComment,
            ]
        );
    }

    #[test]
    fn plan_feedback_reply_steps_plain_comment_only_when_no_discussions() {
        let steps = plan_feedback_reply_steps(&[], true, true, true, true);
        assert_eq!(steps, vec![FeedbackReplyStep::PostPlainComment]);
    }

    #[test]
    fn plan_feedback_reply_steps_omits_plain_comment_when_not_requested() {
        let steps = plan_feedback_reply_steps(&[], true, true, true, false);
        assert!(steps.is_empty());
    }

    #[test]
    fn plan_feedback_reply_steps_omits_plain_comment_when_none_present() {
        let steps = plan_feedback_reply_steps(&[], true, true, false, true);
        assert!(steps.is_empty());
    }

    // -----------------------------------------------------------------
    // Git-mutation ordering used by `handle_mr_comments`: commit only when
    // something is staged, push only once committed, and a real `rev_parse`
    // comparison detects whether the remote branch tip moved. Exercised
    // against a real temporary git repository and a real local bare
    // "origin" remote — no fakes, no network.
    // -----------------------------------------------------------------

    fn run_git(repo: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .expect("git command");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    struct TestRepoPair {
        dir: std::path::PathBuf,
    }

    impl Drop for TestRepoPair {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    /// Sets up a bare "origin" plus a clone with an initial commit on `main`,
    /// mirroring the fetch/checkout preconditions `handle_mr_comments` relies
    /// on before it decides whether to commit and push.
    fn setup_origin_and_clone(name: &str) -> (TestRepoPair, GitRepo) {
        let base = std::env::temp_dir().join(format!(
            "potlatch-worker-git-{name}-{}-{}",
            std::process::id(),
            name.len()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let origin = base.join("origin.git");
        let clone = base.join("clone");

        run_git(
            &base,
            &["init", "--bare", "-b", "main", origin.to_str().unwrap()],
        );
        run_git(
            &base,
            &["clone", origin.to_str().unwrap(), clone.to_str().unwrap()],
        );
        run_git(&clone, &["config", "user.email", "test@example.com"]);
        run_git(&clone, &["config", "user.name", "test"]);
        fs::write(clone.join("file.txt"), "base\n").unwrap();
        run_git(&clone, &["add", "file.txt"]);
        run_git(&clone, &["commit", "-m", "base"]);
        run_git(&clone, &["push", "-u", "origin", "main"]);

        let repo = GitRepo::new(
            clone.to_string_lossy().into_owned(),
            Arc::new(AtomicBool::new(false)),
        );
        (TestRepoPair { dir: base }, repo)
    }

    #[test]
    fn commit_is_skipped_when_nothing_is_staged() {
        let (_guard, repo) = setup_origin_and_clone("no-staged-changes");
        // No working-tree edits were made, matching `has_new_changes == false`.
        repo.add_all().unwrap();
        assert!(!repo.has_staged_changes().unwrap());
    }

    #[test]
    fn commit_then_push_advances_the_remote_branch_tip() {
        let (_guard, repo) = setup_origin_and_clone("commit-then-push");
        let pre_sha = repo.rev_parse("origin/main").unwrap();

        fs::write(
            std::path::Path::new(&repo.path).join("file.txt"),
            "changed\n",
        )
        .unwrap();
        repo.add_all().unwrap();
        assert!(
            repo.has_staged_changes().unwrap(),
            "edit must be staged before commit is attempted"
        );
        repo.commit("feedback: address review comments").unwrap();
        repo.push("main").unwrap();

        repo.fetch_branches(&["main"]).unwrap();
        let post_sha = repo
            .rev_parse("origin/main")
            .unwrap_or_else(|_| pre_sha.clone());
        assert_ne!(
            pre_sha.trim(),
            post_sha.trim(),
            "branch tip must advance only after a real commit was pushed"
        );
    }

    // -----------------------------------------------------------------
    // The two `handoff` contracts. An implementation run and a feedback
    // run share the tool name but not the schema, and each branch of
    // either union carries only its own fields — so the combinations the
    // old flat struct allowed ("implemented *and* needs splitting") are
    // now unrepresentable rather than merely unhandled.
    // -----------------------------------------------------------------

    #[test]
    fn worker_contracts_pass_the_shared_conformance_suite() {
        conformance::assert_contract::<WorkerImplementationOutput>();
        conformance::assert_contract::<WorkerFeedbackOutput>();
    }

    #[test]
    fn both_worker_contracts_are_handed_off_through_the_same_tool() {
        assert_eq!(
            WorkerImplementationOutput::tool_definition().name,
            HANDOFF_TOOL
        );
        assert_eq!(WorkerFeedbackOutput::tool_definition().name, HANDOFF_TOOL);
    }

    #[test]
    fn implementation_output_accepts_implemented_metadata() {
        let output = conformance::assert_accepts::<WorkerImplementationOutput>(serde_json::json!({
            "outcome": "implemented",
            "mr_title": "Add feature",
            "mr_description": "## Goal\nDo it",
            "changes_summary": "Added the feature"
        }));
        assert_eq!(
            output,
            WorkerImplementationOutput::Implemented(ImplementedMetadata {
                mr_title: Some("Add feature".to_string()),
                mr_description: Some("## Goal\nDo it".to_string()),
                changes_summary: Some("Added the feature".to_string()),
            })
        );
    }

    #[test]
    fn implementation_output_requires_an_outcome() {
        let error = conformance::assert_rejects::<WorkerImplementationOutput>(serde_json::json!({
            "mr_title": "Add feature"
        }));
        assert!(
            error.starts_with("$.outcome: required discriminator is missing"),
            "{error}"
        );
    }

    #[test]
    fn implementation_output_rejects_contradictory_branch_fields() {
        let error = conformance::assert_rejects::<WorkerImplementationOutput>(serde_json::json!({
            "outcome": "implemented",
            "mr_title": "Add feature",
            "depends_on_issue": 7
        }));
        assert!(
            error.starts_with("$.depends_on_issue: unexpected property"),
            "{error}"
        );
    }

    #[test]
    fn implementation_output_tolerates_prefixed_string_iids() {
        assert_eq!(
            conformance::assert_accepts::<WorkerImplementationOutput>(serde_json::json!({
                "outcome": "existing_mr",
                "existing_mr_iid": "!12"
            })),
            WorkerImplementationOutput::ExistingMr {
                existing_mr_iid: 12
            }
        );
        assert_eq!(
            conformance::assert_accepts::<WorkerImplementationOutput>(serde_json::json!({
                "outcome": "wait_dependency",
                "depends_on_issue": "#7"
            })),
            WorkerImplementationOutput::WaitDependency {
                depends_on_issue: 7
            }
        );
    }

    #[test]
    fn implementation_output_treats_a_zero_iid_as_no_iid() {
        // Normalization drops it, so the branch's own required rule is what
        // reports the problem — "0" never reaches the GitLab calls.
        assert_eq!(
            conformance::assert_rejects::<WorkerImplementationOutput>(serde_json::json!({
                "outcome": "wait_dependency",
                "depends_on_issue": 0
            })),
            "$.depends_on_issue: required property is missing"
        );
    }

    #[test]
    fn implementation_output_requires_a_reason_on_every_blocked_branch() {
        for outcome in ["needs_split", "needs_clarification", "cannot_implement"] {
            assert_eq!(
                conformance::assert_rejects::<WorkerImplementationOutput>(serde_json::json!({
                    "outcome": outcome
                })),
                "$.reason: required property is missing",
                "{outcome}"
            );
        }
    }

    #[test]
    fn implementation_output_normalizes_outcome_case_and_padding() {
        assert_eq!(
            conformance::assert_accepts::<WorkerImplementationOutput>(serde_json::json!({
                "outcome": " Needs_Split ",
                "reason": "too large"
            })),
            WorkerImplementationOutput::NeedsSplit(BlockedOutcome {
                reason: "too large".to_string(),
                public_comment: None,
            })
        );
    }

    #[test]
    fn feedback_output_accepts_an_addressed_run() {
        assert_eq!(
            conformance::assert_accepts::<WorkerFeedbackOutput>(serde_json::json!({
                "outcome": "addressed",
                "changes_summary": "Fixed the null check",
                "public_comment": "Done.",
                "mark_discussions_resolved": true,
                "post_plain_comment": false
            })),
            WorkerFeedbackOutput::Addressed(FeedbackResolution {
                changes_summary: Some("Fixed the null check".to_string()),
                public_comment: Some("Done.".to_string()),
                mark_discussions_resolved: Some(true),
                ..Default::default()
            })
        );
    }

    #[test]
    fn feedback_output_tolerates_stringified_comment_controls() {
        let output = conformance::assert_accepts::<WorkerFeedbackOutput>(serde_json::json!({
            "outcome": "addressed",
            "mark_discussions_resolved": "true",
            "post_plain_comment": "False"
        }));
        let WorkerFeedbackOutput::Addressed(resolution) = output else {
            panic!("expected an addressed outcome");
        };
        assert_eq!(resolution.mark_discussions_resolved, Some(true));
        assert!(!resolution.post_plain_comment);
    }

    #[test]
    fn feedback_output_requires_a_reason_to_give_up() {
        assert_eq!(
            conformance::assert_rejects::<WorkerFeedbackOutput>(serde_json::json!({
                "outcome": "cannot_resolve"
            })),
            "$.reason: required property is missing"
        );
    }

    #[test]
    fn feedback_output_rejects_implementation_only_outcomes() {
        let error = conformance::assert_rejects::<WorkerFeedbackOutput>(serde_json::json!({
            "outcome": "needs_split",
            "reason": "too large"
        }));
        assert_eq!(
            error,
            "$.outcome: expected one of [\"addressed\", \"cannot_resolve\"], got \"needs_split\""
        );
    }

    #[test]
    fn validation_rejects_invalid_top_level_settings_before_spawn() {
        let config = Config::from_toml_str(
            r#"
            gitlab_repo = 42
            [agent.worker]
            instances = 1
            "#,
        )
        .unwrap();
        let section = config.agent("worker").unwrap();

        let error =
            crate::core::agent::validate_agent_config::<WorkerAgent>(&config, section).unwrap_err();

        assert!(format!("{error:#}").contains("Failed to parse agent settings"));
    }

    /// A blocked branch payload with the given reason and no public comment.
    fn blocked(reason: &str) -> BlockedOutcome {
        BlockedOutcome {
            reason: reason.to_string(),
            public_comment: None,
        }
    }

    #[test]
    fn extract_cannot_resolve_reason_prefers_public_comment_then_reason_then_default() {
        assert_eq!(
            extract_cannot_resolve_reason(&BlockedOutcome {
                reason: "internal reason".to_string(),
                public_comment: Some("Explained to the reviewer.".to_string()),
            }),
            "Explained to the reviewer."
        );
        assert_eq!(
            extract_cannot_resolve_reason(&blocked("Missing credentials.")),
            "Missing credentials."
        );
        assert_eq!(
            extract_cannot_resolve_reason(&blocked("   ")),
            "The implementation cannot proceed without additional human input."
        );
    }

    #[test]
    fn extract_split_reason_prefers_public_comment_then_reason_then_default() {
        assert_eq!(
            extract_split_reason(&blocked("split reason")),
            "split reason"
        );
        assert_eq!(
            extract_split_reason(&BlockedOutcome {
                reason: "split reason".to_string(),
                public_comment: Some("Splitting this up, here is why.".to_string()),
            }),
            "Splitting this up, here is why."
        );
        assert!(extract_split_reason(&blocked("")).contains("too broad"));
    }

    #[test]
    fn extract_clarification_prefers_public_comment_then_reason_then_default() {
        assert_eq!(
            extract_clarification(&blocked("what auth scheme?")),
            "what auth scheme?"
        );
        assert!(extract_clarification(&blocked("")).contains("clarification"));
    }

    #[test]
    fn extract_cannot_implement_reason_falls_back_to_its_own_default() {
        assert_eq!(
            extract_cannot_implement_reason(&blocked("contradictory requirements")),
            "contradictory requirements"
        );
        assert!(extract_cannot_implement_reason(&blocked("")).contains("cannot be implemented"));
    }

    #[test]
    fn extract_worker_public_comment_strips_stray_internal_markers() {
        let comment = extract_worker_public_comment(Some(
            "PUBLIC_COMMENT_BEGIN\nHidden.\nPUBLIC_COMMENT_END\nVisible.",
        ))
        .unwrap();
        assert!(!comment.contains("PUBLIC_COMMENT_BEGIN"));
        assert!(comment.contains("Visible."));
    }

    #[test]
    fn extract_worker_public_comment_returns_none_when_absent_or_blank() {
        assert_eq!(extract_worker_public_comment(None), None);
        assert_eq!(extract_worker_public_comment(Some("   ")), None);
    }

    #[test]
    fn worker_issue_context_includes_comments_section() {
        let issue = IssueObservation {
            iid: 7,
            title: "Add feature".to_string(),
            description: "Do the thing".to_string(),
            labels: vec![],
            state: "opened".to_string(),
        };
        let md = worker_issue_context_markdown(&issue, "- alice: hi");
        assert!(md.contains("## GitLab issue comments"));
        assert!(md.contains("- alice: hi"));
        assert!(md.contains("#7"));
        assert!(md.contains("Do the thing"));
    }

    #[test]
    fn test_should_skip_issue() {
        let mut issue = IssueObservation {
            iid: 1,
            title: "[Draft] Test issue".to_string(),
            description: "Test".to_string(),
            labels: vec![],
            state: "opened".to_string(),
        };
        assert!(should_skip_issue(&issue));

        issue.title = "Draft: Test issue".to_string();
        assert!(should_skip_issue(&issue));

        issue.title = "Normal issue".to_string();
        issue.labels = vec![super::super::labels::DO_NOT_IMPLEMENT.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKING_ON_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![ACTION_REQUIRED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![PMO_PROCESSED_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![PMO_PENDING_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKER_PENDING_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![WORKER_REVIEW_ONLY_LABEL.to_string()];
        assert!(should_skip_issue(&issue));

        issue.labels = vec![];
        assert!(!should_skip_issue(&issue));
    }

    #[test]
    fn worker_pending_label_is_not_treated_as_skip_label_without_pending() {
        assert!(!has_worker_skip_label(&[WORKER_PENDING_LABEL.to_string()]));
        assert!(issue_has_worker_pending_label(&[
            WORKER_PENDING_LABEL.to_string()
        ]));
    }

    #[test]
    fn worker_review_only_label_is_not_resume_abandon_label() {
        assert!(issue_has_worker_review_only_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
        assert!(!has_worker_resume_abandon_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
        assert!(!has_worker_skip_label(&[
            WORKER_REVIEW_ONLY_LABEL.to_string()
        ]));
    }

    #[test]
    fn worker_should_cancel_issue_processing_for_review_only_closed_or_pending() {
        let mut issue = IssueObservation {
            iid: 9,
            title: "Test".into(),
            description: String::new(),
            state: "opened".into(),
            labels: vec![WORKER_REVIEW_ONLY_LABEL.to_string()],
        };
        assert!(worker_should_cancel_issue_processing(&issue));

        issue.labels.clear();
        issue.state = "closed".into();
        assert!(worker_should_cancel_issue_processing(&issue));

        // `pending` label set by a human mid-run must cancel in-flight work.
        issue.state = "opened".into();
        issue.labels = vec![WORKER_PENDING_LABEL.to_string()];
        assert!(worker_should_cancel_issue_processing(&issue));

        // Opened with no blocking labels → keep working.
        issue.labels.clear();
        assert!(!worker_should_cancel_issue_processing(&issue));
    }

    #[test]
    fn truncate_utf8_string_in_place_does_not_split_multibyte_chars() {
        let mut patch = "α".repeat(60_000);
        patch.push_str(&"β".repeat(60_000));
        assert!(patch.len() > 120_000);
        truncate_utf8_string_in_place(&mut patch, 120_000);
        assert!(patch.len() <= 120_000);
        assert!(std::str::from_utf8(patch.as_bytes()).is_ok());
    }

    #[test]
    fn extract_issue_number_from_branch_for_mr_source() {
        assert_eq!(extract_issue_number_from_branch("issue-42").unwrap(), 42);
    }

    #[test]
    fn is_worker_agent_cancelled_matches_runtime_message() {
        let err = anyhow::anyhow!(WORKER_AGENT_CANCELLED_MSG);
        assert!(is_worker_agent_cancelled(&err));
        assert!(!is_worker_agent_cancelled(&anyhow::anyhow!(
            "other failure"
        )));
    }

    #[test]
    fn test_has_worker_resume_abandon_label() {
        assert!(!has_worker_resume_abandon_label(&[
            WORKING_ON_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            ACTION_REQUIRED_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            PMO_PROCESSED_LABEL.to_string()
        ]));
        assert!(has_worker_resume_abandon_label(&[
            PMO_PENDING_LABEL.to_string()
        ]));
        assert!(!has_worker_resume_abandon_label(&[
            "claimed:worker-0".to_string()
        ]));
    }

    #[test]
    fn test_has_worker_skip_label() {
        assert!(has_worker_skip_label(&[WORKING_ON_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[ACTION_REQUIRED_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[PMO_PROCESSED_LABEL.to_string()]));
        assert!(has_worker_skip_label(&[PMO_PENDING_LABEL.to_string()]));
        assert!(!has_worker_skip_label(&["claimed:worker-0".to_string()]));
    }

    #[test]
    fn worker_session_resume_requires_its_live_claim_label() {
        let resource = ClaimResource::Issue(4);
        let recovered = ClaimLease::recover(
            resource,
            "worker-4",
            &["claimed:worker-4".to_string(), WORKING_ON_LABEL.to_string()],
        );
        assert!(recovered.is_some());
        recovered.unwrap().preserve();

        assert!(
            ClaimLease::recover(
                resource,
                "worker-4",
                &["claimed:worker-3".to_string(), WORKING_ON_LABEL.to_string()],
            )
            .is_none()
        );
        assert!(ClaimLease::recover(resource, "worker-4", &[]).is_none());
    }

    #[test]
    fn extract_explicit_mr_title_returns_none_when_field_absent() {
        assert_eq!(extract_explicit_mr_title(None), None);
    }

    #[test]
    fn extract_explicit_mr_title_reads_handoff_structured_field() {
        assert_eq!(
            extract_explicit_mr_title(Some("Add comment chunk truncation docs")),
            Some("Add comment chunk truncation docs".into())
        );
    }

    #[test]
    fn extract_mr_title_falls_back_to_issue_title_when_absent() {
        // When the agent doesn't set mr_title, the MR title falls back to
        // the issue title — never a generic placeholder.
        assert_eq!(
            extract_mr_title(None, "Add login rate limiting"),
            "Add login rate limiting"
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_includes_reason_without_code_changes() {
        let output = FeedbackResolution {
            reason: Some("Existing validation already covered this case.".to_string()),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&output, false, None),
            Some(
                "Resolved without code changes:\n\nExisting validation already covered this case."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_uses_changes_summary_when_changes_exist() {
        let output = FeedbackResolution {
            changes_summary: Some("Add missing null check in parser.".to_string()),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&output, true, None),
            Some("Addressed feedback:\n\nAdd missing null check in parser.".to_string())
        );
    }

    #[test]
    fn test_build_feedback_resolution_reply_includes_diff_highlights() {
        let output = FeedbackResolution {
            changes_summary: Some("Tighten input validation.".to_string()),
            ..Default::default()
        };
        let diff = "- src/validation.rs\n- 1 file changed, 4 insertions(+)";
        assert_eq!(
            build_feedback_resolution_reply(&output, true, Some(diff)),
            Some("Addressed feedback:\n\nTighten input validation.\n\nDiff highlights:\n- src/validation.rs\n- 1 file changed, 4 insertions(+)".to_string())
        );
    }

    #[test]
    fn build_feedback_resolution_reply_requires_reason_without_code_changes() {
        let output = FeedbackResolution::default();
        assert_eq!(build_feedback_resolution_reply(&output, false, None), None);
    }

    #[test]
    fn feedback_discussions_may_be_resolved_blocks_while_conflicts_remain() {
        let out = FeedbackResolution {
            mark_discussions_resolved: Some(true),
            ..Default::default()
        };
        assert!(!feedback_discussions_may_be_resolved(&out, true, true));
        assert!(feedback_discussions_may_be_resolved(&out, true, false));
    }

    #[test]
    fn combined_mr_feedback_context_separates_unresolved_and_full_history() {
        let mr = crate::agents::gitlab::MergeRequest {
            iid: 287,
            title: "Add config".into(),
            description: "MR description".into(),
            source_branch: "issue-285".into(),
            target_branch: "main".into(),
            state: "opened".into(),
            sha: None,
            labels: None,
            has_conflicts: false,
        };
        let ctx = build_combined_mr_feedback_context(CombinedMrFeedbackContextInput {
            project_name: "project",
            mr: &mr,
            issue_context: "issue context",
            implementation_summary: "implementation summary",
            merge_conflict_status: "## Merge conflict status\n- GitLab reports merge conflicts on this MR: no",
            unresolved_comments_text: "- reviewer (discussion d1): fix this",
            plain_comments_text: "- reviewer (discussion d2): plain actionable note",
            all_comments_text: "- reviewer (discussion d1): fix this\n- maintainer (discussion d2): simple context",
            diff_context: "diff context",
        });

        assert!(ctx.contains("## Merge conflict status"));
        assert!(ctx.contains("## Unresolved MR comments to address"));
        assert!(ctx.contains("## Plain MR comments to consider"));
        assert!(ctx.contains("## Full MR comment history for context"));
        assert!(ctx.contains("fix this"));
        assert!(ctx.contains("plain actionable note"));
        assert!(ctx.contains("simple context"));
    }

    #[test]
    fn strip_worker_reply_boilerplate_removes_prefix_when_suffix_present() {
        assert_eq!(
            strip_worker_reply_boilerplate(
                "Resolved without code changes:\n\nAlready covered by tests."
            ),
            "Already covered by tests."
        );
        assert_eq!(
            strip_worker_reply_boilerplate("ADDRESSED FEEDBACK:\n\nFixed the null deref."),
            "Fixed the null deref."
        );
    }

    #[test]
    fn strip_worker_reply_boilerplate_keeps_prefix_when_nothing_follows() {
        assert_eq!(
            strip_worker_reply_boilerplate("Resolved without code changes:"),
            "Resolved without code changes:"
        );
        assert_eq!(
            strip_worker_reply_boilerplate("Resolved without code changes:\n  \n  "),
            "Resolved without code changes:"
        );
    }

    #[test]
    fn strip_worker_reply_boilerplate_keeps_all_lines_after_prefix() {
        let input = "Addressed feedback:\n\nI’ll run a full readonly review from local state.\nFixed null checks.\nError: S: [unavailable] Error";
        assert_eq!(
            strip_worker_reply_boilerplate(input),
            "I’ll run a full readonly review from local state.\nFixed null checks.\nError: S: [unavailable] Error"
        );
    }

    #[test]
    fn should_resolve_mr_feedback_discussions_defaults_follow_implicit_actions() {
        let out = FeedbackResolution::default();
        assert!(!should_resolve_mr_feedback_discussions(&out, false));
        assert!(should_resolve_mr_feedback_discussions(&out, true));
    }

    #[test]
    fn mark_discussions_resolved_reads_structured_field() {
        let out_no = FeedbackResolution {
            mark_discussions_resolved: Some(false),
            ..Default::default()
        };
        assert!(!should_resolve_mr_feedback_discussions(&out_no, true));
        let out_yes = FeedbackResolution {
            mark_discussions_resolved: Some(true),
            ..Default::default()
        };
        assert!(should_resolve_mr_feedback_discussions(&out_yes, false));
    }

    #[test]
    fn plain_comment_posting_requires_explicit_field() {
        // A public comment on its own never posts a plain MR comment; the
        // model has to ask for it.
        let default = FeedbackResolution {
            public_comment: Some("No further changes were needed.".to_string()),
            ..Default::default()
        };
        assert!(!default.post_plain_comment);

        let requested = FeedbackResolution {
            post_plain_comment: true,
            public_comment: Some("Posted by request.".to_string()),
            ..Default::default()
        };
        assert!(requested.post_plain_comment);
    }

    #[test]
    fn no_change_reply_requires_explicit_resolve_field() {
        let out = FeedbackResolution {
            public_comment: Some("No new code changes were needed in this run.".to_string()),
            ..Default::default()
        };
        assert!(!should_resolve_mr_feedback_discussions(&out, false));

        let explicit = FeedbackResolution {
            mark_discussions_resolved: Some(true),
            public_comment: Some("No new code changes were needed in this run.".to_string()),
            ..Default::default()
        };
        assert!(should_resolve_mr_feedback_discussions(&explicit, false));
    }

    #[test]
    fn no_change_reply_text_requires_structured_reason_field() {
        let out = FeedbackResolution::default();
        assert_eq!(build_feedback_resolution_reply(&out, false, None), None);

        let explicit = FeedbackResolution {
            reason: Some(
                "No new code changes were needed in this run. The branch already satisfies the requested behavior."
                    .to_string(),
            ),
            ..Default::default()
        };
        assert_eq!(
            build_feedback_resolution_reply(&explicit, false, None),
            Some("Resolved without code changes:\n\nNo new code changes were needed in this run. The branch already satisfies the requested behavior.".to_string())
        );
    }

    #[test]
    fn extract_no_change_resolution_reason_returns_none_when_absent() {
        let out = FeedbackResolution::default();
        assert_eq!(extract_no_change_resolution_reason(&out), None);
    }

    #[test]
    fn extract_no_change_resolution_reason_prefers_reason_over_changes_summary() {
        let out = FeedbackResolution {
            reason: Some(
                "Property deletion already removes stored data on schema update.".to_string(),
            ),
            changes_summary: Some("unrelated summary".to_string()),
            ..Default::default()
        };
        assert_eq!(
            extract_no_change_resolution_reason(&out),
            Some("Property deletion already removes stored data on schema update.".to_string())
        );
    }

    #[test]
    fn extract_no_change_resolution_reason_falls_back_to_changes_summary() {
        let out = FeedbackResolution {
            changes_summary: Some("Resolved via metadata update only.".to_string()),
            ..Default::default()
        };
        assert_eq!(
            extract_no_change_resolution_reason(&out),
            Some("Resolved via metadata update only.".to_string())
        );
    }

    #[test]
    fn merge_request_surface_changed_detects_title_labels() {
        use crate::agents::gitlab::MergeRequest;
        let a = MergeRequest {
            iid: 1,
            title: "Old".into(),
            description: "d".into(),
            source_branch: "b".into(),
            target_branch: "m".into(),
            state: "opened".into(),
            sha: None,
            labels: Some(vec!["a".into()]),
            has_conflicts: false,
        };
        let before = MrSurfaceObservation::from_mr(&a);
        let mut b = a.clone();
        assert!(!before.differs_from(&MrSurfaceObservation::from_mr(&b)));
        b.title = "New".into();
        assert!(before.differs_from(&MrSurfaceObservation::from_mr(&b)));
        b.title = "Old".into();
        b.labels = Some(vec!["a".into(), "b".into()]);
        assert!(before.differs_from(&MrSurfaceObservation::from_mr(&b)));
    }

    #[test]
    fn extract_mr_description_strips_control_marker_lines() {
        let desc = extract_mr_description(Some(
            "## Goal\nDescribe change.\nCHANGES_SUMMARY: noisy line\nMARK_DISCUSSIONS_RESOLVED: yes\nPOST_PLAIN_COMMENT: yes\n## Testing\ncargo test",
        ));
        assert!(!desc.contains("CHANGES_SUMMARY:"), "{desc}");
        assert!(!desc.contains("MARK_DISCUSSIONS_RESOLVED:"), "{desc}");
        assert!(!desc.contains("POST_PLAIN_COMMENT:"), "{desc}");
        assert!(desc.contains("## Goal"), "{desc}");
        assert!(desc.contains("## Testing"), "{desc}");
    }

    #[test]
    fn extract_mr_description_strips_public_comment_blocks() {
        let desc = extract_mr_description(Some(
            "## Goal\npytest coverage.\nPUBLIC_COMMENT_BEGIN\nThanks for the review.\nPUBLIC_COMMENT_END\n## Testing\nuv run pytest",
        ));
        assert!(!desc.contains("PUBLIC_COMMENT_BEGIN"), "{desc}");
        assert!(!desc.contains("Thanks for the review"), "{desc}");
        assert!(desc.contains("## Goal"), "{desc}");
        assert!(desc.contains("uv run pytest"), "{desc}");
    }

    #[test]
    fn extract_mr_description_defaults_when_field_absent() {
        assert_eq!(extract_mr_description(None), "Implementation completed.");
    }

    #[test]
    fn extract_mr_title_reads_structured_field() {
        assert_eq!(
            extract_mr_title(Some("Stable title"), "Issue title"),
            "Stable title"
        );
        assert_eq!(
            extract_explicit_mr_title(Some("Stable title")),
            Some("Stable title".into())
        );
    }

    #[test]
    fn waiting_on_issue_label_format() {
        assert_eq!(waiting_on_issue_label(42), "waiting-on-issue:#42");
    }

    #[test]
    fn extract_waiting_on_issue_iid_finds_label() {
        let labels = vec![
            "in-progress".to_string(),
            "waiting-on-issue:#15".to_string(),
        ];
        assert_eq!(extract_waiting_on_issue_iid(&labels), Some(15));
    }

    #[test]
    fn extract_waiting_on_issue_iid_returns_none_without_label() {
        let labels = vec!["in-progress".to_string(), "priority::3".to_string()];
        assert_eq!(extract_waiting_on_issue_iid(&labels), None);
    }

    #[test]
    fn extract_waiting_on_issue_iid_ignores_malformed() {
        let labels = vec!["waiting-on-issue:#abc".to_string()];
        assert_eq!(extract_waiting_on_issue_iid(&labels), None);
    }

    fn mr_comment(
        id: u64,
        author: &str,
        discussion_id: &str,
        body: &str,
    ) -> crate::agents::gitlab::Comment {
        crate::agents::gitlab::Comment {
            id,
            body: body.to_string(),
            author: author.to_string(),
            discussion_id: discussion_id.to_string(),
            discussion_resolvable: false,
            location: None,
            location_details: None,
        }
    }

    #[test]
    fn collect_new_follow_ups_returns_only_new_comments() {
        let comments = vec![
            mr_comment(10, "alice", "d1", "old comment"),
            mr_comment(15, "bob", "d2", "new comment"),
            mr_comment(20, "carol", "d1", "another new one"),
        ];
        let mut last_seen = 10u64;
        let msgs = collect_new_follow_ups(&comments, &mut last_seen, 42);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].contains("@bob"));
        assert!(msgs[0].contains("MR !42"));
        assert!(msgs[0].contains("new comment"));
        assert!(msgs[1].contains("@carol"));
        assert!(msgs[1].contains("another new one"));
        assert_eq!(last_seen, 20);
    }

    #[test]
    fn collect_new_follow_ups_skips_all_when_seen_is_max() {
        let comments = vec![
            mr_comment(5, "alice", "d1", "old"),
            mr_comment(5, "bob", "d2", "also old"),
        ];
        let mut last_seen = 5u64;
        let msgs = collect_new_follow_ups(&comments, &mut last_seen, 1);
        assert!(msgs.is_empty());
        assert_eq!(last_seen, 5);
    }

    #[test]
    fn collect_new_follow_ups_handles_empty_comments() {
        let mut last_seen = 3u64;
        let msgs = collect_new_follow_ups(&[], &mut last_seen, 1);
        assert!(msgs.is_empty());
        assert_eq!(last_seen, 3);
    }

    #[test]
    fn an_existing_mr_is_only_adopted_from_its_own_branch() {
        // Adoption is a whole outcome now, so the IID cannot ride along on an
        // ordinary implementation handoff and quietly redirect the run.
        assert_eq!(
            conformance::assert_accepts::<WorkerImplementationOutput>(serde_json::json!({
                "outcome": "existing_mr",
                "existing_mr_iid": 42
            })),
            WorkerImplementationOutput::ExistingMr {
                existing_mr_iid: 42
            }
        );
        let error = conformance::assert_rejects::<WorkerImplementationOutput>(serde_json::json!({
            "outcome": "implemented",
            "mr_title": "Fix bug",
            "existing_mr_iid": 42
        }));
        assert!(
            error.starts_with("$.existing_mr_iid: unexpected property"),
            "{error}"
        );
    }

    // -----------------------------------------------------------------
    // Routing driver traces against a recording fake port. The machine is
    // plain data, so a whole cycle can be characterized as an ordered
    // list of observations and actions — including the failure paths,
    // which are the ones that are hardest to reach against live GitLab.
    // -----------------------------------------------------------------

    const ROUTING_AGENT: &str = "worker-1";

    fn issue_observation(iid: u64, labels: &[&str]) -> IssueObservation {
        IssueObservation {
            iid,
            title: format!("Cap the retry backoff for issue {iid}"),
            description: "## Goal\nCap the backoff.".to_string(),
            state: "opened".to_string(),
            labels: labels.iter().map(|l| l.to_string()).collect(),
        }
    }

    /// What one scripted implementation run does: whether it produced an
    /// MR, which branch it left behind, and how it ended.
    #[derive(Clone)]
    struct ImplementationScript {
        mr_created: bool,
        mr_iid: Option<u64>,
        branch_name: Option<String>,
        error: Option<String>,
    }

    impl ImplementationScript {
        fn succeeded(mr_iid: u64) -> Self {
            Self {
                mr_created: true,
                mr_iid: Some(mr_iid),
                branch_name: Some(format!("issue-{mr_iid}")),
                error: None,
            }
        }

        fn no_mr() -> Self {
            Self {
                mr_created: false,
                mr_iid: None,
                branch_name: None,
                error: None,
            }
        }

        fn failed(branch: Option<&str>, message: &str) -> Self {
            Self {
                mr_created: false,
                mr_iid: None,
                branch_name: branch.map(str::to_string),
                error: Some(message.to_string()),
            }
        }
    }

    /// A recording routing port. The world is scripted per observation
    /// kind; failures are injected by naming the exact query or action
    /// that should fail, so each test reads as "this world, then this
    /// trace".
    struct FakeWorkerPort {
        trace: std::cell::RefCell<Vec<WorkerStep>>,
        shutdown_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        issues: Vec<IssueObservation>,
        known_issues: Vec<IssueObservation>,
        mr_states: Vec<(u64, String)>,
        default_branch: String,
        claim_attempts: std::cell::RefCell<std::collections::VecDeque<IssueClaimAttempt>>,
        adopted: Option<ActiveIssue>,
        handled_labeled_mr: bool,
        implementation: std::cell::RefCell<std::collections::VecDeque<ImplementationScript>>,
        feedback_abandoned: bool,
        feedback_error: Option<String>,
        cancel_handled: bool,
        trackable: bool,
        failing_queries: Vec<WorkerQuery>,
        failing_actions: Vec<WorkerAction>,
    }

    impl FakeWorkerPort {
        fn new() -> Self {
            Self {
                trace: std::cell::RefCell::new(Vec::new()),
                shutdown_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                issues: Vec::new(),
                known_issues: Vec::new(),
                mr_states: Vec::new(),
                default_branch: "main".to_string(),
                claim_attempts: std::cell::RefCell::new(std::collections::VecDeque::new()),
                adopted: None,
                handled_labeled_mr: false,
                implementation: std::cell::RefCell::new(std::collections::VecDeque::new()),
                feedback_abandoned: false,
                feedback_error: None,
                cancel_handled: false,
                trackable: true,
                failing_queries: Vec::new(),
                failing_actions: Vec::new(),
            }
        }

        /// The issues the poll lists; they are answerable by IID too.
        fn listing(mut self, issues: &[IssueObservation]) -> Self {
            self.issues = issues.to_vec();
            self.known_issues.extend(issues.iter().cloned());
            self
        }

        fn knowing(mut self, issues: &[IssueObservation]) -> Self {
            self.known_issues.extend(issues.iter().cloned());
            self
        }

        fn with_mr_state(mut self, mr_iid: u64, state: &str) -> Self {
            self.mr_states.push((mr_iid, state.to_string()));
            self
        }

        fn with_shutdown_answers(self, answers: &[bool]) -> Self {
            self.shutdown_answers.borrow_mut().extend(answers);
            self
        }

        fn with_claim_attempts(self, attempts: &[IssueClaimAttempt]) -> Self {
            self.claim_attempts.borrow_mut().extend(attempts);
            self
        }

        fn implementing(self, scripts: &[ImplementationScript]) -> Self {
            self.implementation
                .borrow_mut()
                .extend(scripts.iter().cloned());
            self
        }

        fn adopting(mut self, active: ActiveIssue) -> Self {
            self.adopted = Some(active);
            self
        }

        fn handling_labeled_mr(mut self) -> Self {
            self.handled_labeled_mr = true;
            self
        }

        fn abandoning_feedback(mut self) -> Self {
            self.feedback_abandoned = true;
            self
        }

        fn failing_feedback(mut self, message: &str) -> Self {
            self.feedback_error = Some(message.to_string());
            self
        }

        fn handling_cancel(mut self) -> Self {
            self.cancel_handled = true;
            self
        }

        fn untrackable(mut self) -> Self {
            self.trackable = false;
            self
        }

        fn failing_query(mut self, query: WorkerQuery) -> Self {
            self.failing_queries.push(query);
            self
        }

        fn failing_action(mut self, action: WorkerAction) -> Self {
            self.failing_actions.push(action);
            self
        }

        fn record(&self, step: WorkerStep) {
            self.trace.borrow_mut().push(step);
        }

        fn observe(&self, query: WorkerQuery) -> Result<()> {
            self.record(WorkerStep::Observe(query.clone()));
            if self.failing_queries.contains(&query) {
                anyhow::bail!("injected failure observing {query:?}");
            }
            Ok(())
        }
    }

    impl WorkerRoutingPort for FakeWorkerPort {
        fn shutdown_requested(&self) -> bool {
            self.record(WorkerStep::Observe(WorkerQuery::ShutdownRequested));
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }

        fn issue(&self, issue_iid: u64) -> Result<IssueObservation> {
            self.observe(WorkerQuery::Issue { issue_iid })?;
            self.known_issues
                .iter()
                .find(|issue| issue.iid == issue_iid)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("404 Issue Not Found: #{issue_iid}"))
        }

        fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation> {
            self.observe(WorkerQuery::MergeRequestStatus { mr_iid })?;
            let state = self
                .mr_states
                .iter()
                .find(|(iid, _)| *iid == mr_iid)
                .map(|(_, state)| state.clone())
                .unwrap_or_else(|| "opened".to_string());
            Ok(MrStatusObservation { iid: mr_iid, state })
        }

        fn issues(&self) -> Result<Vec<IssueObservation>> {
            self.observe(WorkerQuery::Issues)?;
            Ok(self.issues.clone())
        }

        fn default_branch_or_main(&self) -> String {
            self.record(WorkerStep::Observe(WorkerQuery::DefaultBranchOrMain));
            self.default_branch.clone()
        }

        fn execute(&mut self, action: &WorkerAction) -> WorkerOutcome {
            self.record(WorkerStep::Act(action.clone()));
            if self.failing_actions.contains(action) {
                return WorkerOutcome::Failed(anyhow::anyhow!(
                    "injected failure executing {action:?}"
                ));
            }
            match action {
                WorkerAction::AcquireIssueClaim { .. } => WorkerOutcome::Claim(
                    self.claim_attempts
                        .borrow_mut()
                        .pop_front()
                        .unwrap_or(IssueClaimAttempt::Won),
                ),
                WorkerAction::AdoptOrphanedSession => WorkerOutcome::Adopted(self.adopted.clone()),
                WorkerAction::HandleNeedAiWorkerMr => {
                    WorkerOutcome::HandledNeedAiWorkerMr(self.handled_labeled_mr)
                }
                WorkerAction::RunImplementation { issue } => {
                    let script = self
                        .implementation
                        .borrow_mut()
                        .pop_front()
                        .unwrap_or_else(ImplementationScript::no_mr);
                    WorkerOutcome::Implementation {
                        current: ActiveIssue {
                            issue_iid: issue.iid,
                            mr_iid: script.mr_iid,
                            branch_name: script.branch_name.clone(),
                            mr_created: script.mr_created,
                        },
                        error: script.error.as_ref().map(|e| anyhow::anyhow!("{e}")),
                    }
                }
                WorkerAction::RunFeedback { .. } => match &self.feedback_error {
                    Some(message) => WorkerOutcome::Failed(anyhow::anyhow!("{message}")),
                    None => WorkerOutcome::Feedback {
                        abandoned: self.feedback_abandoned,
                    },
                },
                WorkerAction::ResolveCancelledIssue { .. } => {
                    WorkerOutcome::CancelHandled(self.cancel_handled)
                }
                WorkerAction::CheckIssueTrackable { .. } => {
                    WorkerOutcome::Trackable(self.trackable)
                }
                _ => WorkerOutcome::Done,
            }
        }
    }

    struct FakeWorkerRun {
        result: Result<()>,
        trace: Vec<WorkerStep>,
        active: Option<ActiveIssue>,
    }

    fn run_worker_routing(port: &mut FakeWorkerPort, active: Option<ActiveIssue>) -> FakeWorkerRun {
        let mut machine = WorkerRoutingMachine::new(ROUTING_AGENT, None, active);
        let result = drive_worker_routing(&mut machine, port);
        FakeWorkerRun {
            result,
            trace: port.trace.borrow().clone(),
            active: machine.active.take(),
        }
    }

    fn w_observe(query: WorkerQuery) -> WorkerStep {
        WorkerStep::Observe(query)
    }

    fn w_act(action: WorkerAction) -> WorkerStep {
        WorkerStep::Act(action)
    }

    fn w_shutdown() -> WorkerStep {
        w_observe(WorkerQuery::ShutdownRequested)
    }

    fn active_with_mr(issue_iid: u64, mr_iid: u64) -> ActiveIssue {
        ActiveIssue {
            issue_iid,
            mr_iid: Some(mr_iid),
            branch_name: Some(format!("issue-{issue_iid}")),
            mr_created: true,
        }
    }

    fn active_without_mr(issue_iid: u64) -> ActiveIssue {
        ActiveIssue {
            issue_iid,
            mr_iid: None,
            branch_name: None,
            mr_created: false,
        }
    }

    /// The steps a cycle with no active issue performs before it looks at
    /// any candidate.
    fn polling_steps() -> Vec<WorkerStep> {
        vec![
            w_shutdown(),
            w_act(WorkerAction::AdoptOrphanedSession),
            w_act(WorkerAction::HandleNeedAiWorkerMr),
            w_observe(WorkerQuery::Issues),
            w_shutdown(),
        ]
    }

    #[test]
    fn worker_cycle_claims_a_candidate_then_implements_and_tracks_it_in_order() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        let mut expected = polling_steps();
        expected.extend([
            w_shutdown(),
            w_act(WorkerAction::AcquireIssueClaim { issue_iid: 7 }),
            w_shutdown(),
            w_act(WorkerAction::PreserveIssueClaim { issue_iid: 7 }),
            w_act(WorkerAction::SaveSession {
                issue_iid: 7,
                mr_iid: 0,
            }),
            w_act(WorkerAction::RunImplementation {
                issue: Box::new(candidate),
            }),
            w_act(WorkerAction::CheckIssueTrackable { issue_iid: 7 }),
        ]);
        assert_eq!(run.trace, expected);
        assert_eq!(run.active, Some(active_with_mr(7, 7)));
    }

    #[test]
    fn worker_cycle_releases_only_the_claim_when_the_run_produced_no_merge_request() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .implementing(&[ImplementationScript::no_mr()]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }))
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_cleans_the_branch_up_after_a_failed_candidate_run() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "boom")]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        // A failed candidate run keeps the session file (only the
        // re-attempt path drops it) and cleans the worktree last.
        assert_eq!(
            &run.trace[run.trace.len() - 7..],
            &[
                w_shutdown(),
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::RemoveWorkingOnLabel { issue_iid: 7 }),
                w_observe(WorkerQuery::DefaultBranchOrMain),
                w_act(WorkerAction::ResetWorktree),
                w_act(WorkerAction::CheckoutBranch {
                    branch: "main".to_string(),
                }),
                w_act(WorkerAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_keeps_the_issue_active_when_shutdown_lands_mid_implementation() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "interrupted")])
            .with_shutdown_answers(&[false, false, false, false, true]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(run.trace.last(), Some(&w_shutdown()));
        // The shutdown hook needs the in-flight issue to persist the claim.
        assert_eq!(
            run.active,
            Some(ActiveIssue {
                issue_iid: 7,
                mr_iid: None,
                branch_name: Some("issue-7".to_string()),
                mr_created: false,
            })
        );
    }

    #[test]
    fn worker_cycle_resolves_an_externally_cancelled_candidate_run_without_cleanup() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .implementing(&[ImplementationScript::failed(
                Some("issue-7"),
                WORKER_AGENT_CANCELLED_MSG,
            )])
            .handling_cancel();
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_act(WorkerAction::ResolveCancelledIssue { issue_iid: 7 }))
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_falls_back_to_cleanup_when_a_cancel_turns_out_to_be_a_real_failure() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new().listing(&[candidate]).implementing(&[
            ImplementationScript::failed(None, WORKER_AGENT_CANCELLED_MSG),
        ]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(
            &run.trace[run.trace.len() - 3..],
            &[
                w_act(WorkerAction::ResolveCancelledIssue { issue_iid: 7 }),
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::RemoveWorkingOnLabel { issue_iid: 7 }),
            ]
        );
    }

    #[test]
    fn worker_cycle_gives_a_won_claim_straight_back_when_shutdown_lands() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .with_shutdown_answers(&[false, false, false, true]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(
            &run.trace[run.trace.len() - 2..],
            &[
                w_shutdown(),
                w_act(WorkerAction::ReleaseAcquiredClaim { issue_iid: 7 }),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_moves_on_when_a_candidate_claim_is_lost_and_stops_when_interrupted() {
        let mut lost = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[]), issue_observation(9, &[])])
            .with_claim_attempts(&[IssueClaimAttempt::Lost, IssueClaimAttempt::Won])
            .implementing(&[ImplementationScript::succeeded(9)]);
        let run = run_worker_routing(&mut lost, None);
        assert!(run.result.is_ok());
        assert_eq!(run.active, Some(active_with_mr(9, 9)));

        let mut interrupted = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[]), issue_observation(9, &[])])
            .with_claim_attempts(&[IssueClaimAttempt::Interrupted]);
        let run = run_worker_routing(&mut interrupted, None);
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_act(WorkerAction::AcquireIssueClaim { issue_iid: 7 }))
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_skips_candidates_that_are_out_of_reach_without_touching_gitlab() {
        let mut port = FakeWorkerPort::new().listing(&[
            issue_observation(1, &[WORKING_ON_LABEL]),
            issue_observation(2, &[&format!("claimed:{}", "worker-2")]),
            issue_observation(3, &[ACTION_REQUIRED_LABEL]),
        ]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        // Screening is pure, so the whole poll is the listing steps plus
        // one shutdown check per candidate.
        let mut expected = polling_steps();
        expected.extend([w_shutdown(), w_shutdown(), w_shutdown()]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn worker_cycle_drops_a_resolved_dependency_label_before_claiming() {
        let candidate = issue_observation(7, &[&waiting_on_issue_label(4)]);
        let mut closed_dependency = issue_observation(4, &[]);
        closed_dependency.state = "closed".to_string();
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .knowing(&[closed_dependency])
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        let mut expected = polling_steps();
        expected.extend([
            w_shutdown(),
            w_observe(WorkerQuery::Issue { issue_iid: 4 }),
            w_act(WorkerAction::RemoveIssueLabel {
                issue_iid: 7,
                label: waiting_on_issue_label(4),
            }),
            w_act(WorkerAction::AcquireIssueClaim { issue_iid: 7 }),
            w_shutdown(),
            w_act(WorkerAction::PreserveIssueClaim { issue_iid: 7 }),
            w_act(WorkerAction::SaveSession {
                issue_iid: 7,
                mr_iid: 0,
            }),
            w_act(WorkerAction::RunImplementation {
                issue: Box::new(issue_observation(7, &[&waiting_on_issue_label(4)])),
            }),
            w_act(WorkerAction::CheckIssueTrackable { issue_iid: 7 }),
        ]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn worker_cycle_leaves_a_candidate_parked_while_its_dependency_is_open() {
        let candidate = issue_observation(7, &[&waiting_on_issue_label(4)]);
        let mut port = FakeWorkerPort::new()
            .listing(&[candidate])
            .knowing(&[issue_observation(4, &[])]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        let mut expected = polling_steps();
        expected.extend([w_shutdown(), w_observe(WorkerQuery::Issue { issue_iid: 4 })]);
        assert_eq!(run.trace, expected);
    }

    #[test]
    fn worker_cycle_treats_a_missing_dependency_as_resolved_but_an_unreadable_one_as_parked() {
        let candidate = issue_observation(7, &[&waiting_on_issue_label(4)]);
        let mut missing = FakeWorkerPort::new().listing(std::slice::from_ref(&candidate));
        let run = run_worker_routing(&mut missing, None);
        assert!(run.result.is_ok());
        assert!(run.trace.contains(&w_act(WorkerAction::RemoveIssueLabel {
            issue_iid: 7,
            label: waiting_on_issue_label(4),
        })));

        let mut unreadable = FakeWorkerPort::new()
            .listing(&[candidate])
            .knowing(&[issue_observation(4, &[])])
            .failing_query(WorkerQuery::Issue { issue_iid: 4 });
        let run = run_worker_routing(&mut unreadable, None);
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_observe(WorkerQuery::Issue { issue_iid: 4 }))
        );
    }

    #[test]
    fn worker_cycle_stops_at_an_adopted_orphan_and_at_a_handled_labeled_mr() {
        let mut adopting = FakeWorkerPort::new().adopting(active_with_mr(7, 12));
        let run = run_worker_routing(&mut adopting, None);
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![w_shutdown(), w_act(WorkerAction::AdoptOrphanedSession)]
        );
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        let mut labeled = FakeWorkerPort::new().handling_labeled_mr();
        let run = run_worker_routing(&mut labeled, None);
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                w_shutdown(),
                w_act(WorkerAction::AdoptOrphanedSession),
                w_act(WorkerAction::HandleNeedAiWorkerMr),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_fails_the_cycle_when_a_required_read_or_write_fails() {
        let mut listing = FakeWorkerPort::new().failing_query(WorkerQuery::Issues);
        assert!(run_worker_routing(&mut listing, None).result.is_err());

        let mut labeled_mr =
            FakeWorkerPort::new().failing_action(WorkerAction::HandleNeedAiWorkerMr);
        assert!(run_worker_routing(&mut labeled_mr, None).result.is_err());

        let mut claiming = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[])])
            .failing_action(WorkerAction::AcquireIssueClaim { issue_iid: 7 });
        assert!(run_worker_routing(&mut claiming, None).result.is_err());
    }

    #[test]
    fn worker_cycle_ends_the_release_of_a_merged_mr_by_closing_the_issue() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .with_mr_state(12, "merged")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_observe(WorkerQuery::MergeRequestStatus { mr_iid: 12 }),
                w_observe(WorkerQuery::DefaultBranchOrMain),
                w_act(WorkerAction::ResetWorktree),
                w_act(WorkerAction::CheckoutBranch {
                    branch: "main".to_string(),
                }),
                w_act(WorkerAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
                w_act(WorkerAction::DeleteRemoteBranch {
                    branch: "issue-7".to_string(),
                }),
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::RemoveWorkingOnLabel { issue_iid: 7 }),
                w_act(WorkerAction::CleanupSession { issue_iid: 7 }),
                w_act(WorkerAction::CloseIssue { issue_iid: 7 }),
                w_shutdown(),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_keeps_the_remote_branch_and_the_issue_when_the_mr_was_closed() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .with_mr_state(12, "closed")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));

        assert!(run.result.is_ok());
        assert!(
            !run.trace.contains(&w_act(WorkerAction::DeleteRemoteBranch {
                branch: "issue-7".to_string(),
            }))
        );
        assert!(
            !run.trace
                .contains(&w_act(WorkerAction::CloseIssue { issue_iid: 7 }))
        );
        assert_eq!(
            &run.trace[run.trace.len() - 2..],
            &[
                w_act(WorkerAction::CleanupSession { issue_iid: 7 }),
                w_shutdown()
            ]
        );
    }

    #[test]
    fn worker_cycle_runs_feedback_for_an_open_mr_and_keeps_the_issue_active() {
        let mut port = FakeWorkerPort::new().knowing(&[issue_observation(7, &[WORKING_ON_LABEL])]);
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_observe(WorkerQuery::MergeRequestStatus { mr_iid: 12 }),
                w_act(WorkerAction::RunFeedback {
                    mr_iid: 12,
                    linked_issue_iid: Some(7),
                    comments_only: false,
                }),
            ]
        );
        assert_eq!(run.active, Some(active_with_mr(7, 12)));
    }

    #[test]
    fn worker_cycle_releases_the_issue_when_feedback_abandoned_the_merge_request() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .abandoning_feedback()
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));

        assert!(run.result.is_ok());
        assert_eq!(
            &run.trace[run.trace.len() - 3..],
            &[
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::CleanupSession { issue_iid: 7 }),
                w_shutdown(),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_handles_a_failed_feedback_run_by_shutdown_cancel_or_logging() {
        // Shutdown: keep the issue active and stop.
        let mut shutting_down = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_feedback("boom")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut shutting_down, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(run.trace.last(), Some(&w_shutdown()));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        // An external cancel is resolved and releases the issue.
        let mut cancelled = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_feedback(WORKER_AGENT_CANCELLED_MSG)
            .handling_cancel();
        let run = run_worker_routing(&mut cancelled, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_act(WorkerAction::ResolveCancelledIssue { issue_iid: 7 }))
        );
        assert!(run.active.is_none());

        // Anything else is logged; the issue stays active for the next cycle.
        let mut failed = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_feedback("boom");
        let run = run_worker_routing(&mut failed, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(run.trace.last(), Some(&w_shutdown()));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));
    }

    #[test]
    fn worker_cycle_ends_the_mr_watch_when_the_mr_cannot_be_read() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_query(WorkerQuery::MergeRequestStatus { mr_iid: 12 });
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace.last(),
            Some(&w_observe(WorkerQuery::MergeRequestStatus { mr_iid: 12 }))
        );
        assert_eq!(run.active, Some(active_with_mr(7, 12)));
    }

    #[test]
    fn worker_cycle_releases_a_watched_issue_that_left_the_worker_behind() {
        // Closed externally.
        let mut closed_issue = issue_observation(7, &[WORKING_ON_LABEL]);
        closed_issue.state = "closed".to_string();
        let mut closed = FakeWorkerPort::new()
            .knowing(&[closed_issue])
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut closed, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_act(WorkerAction::AbandonClosedIssue {
                    issue_iid: 7,
                    mr_iid: Some(12),
                }),
                w_shutdown(),
            ]
        );
        assert!(run.active.is_none());

        // A human parked it with the pending label.
        let mut pending = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKER_PENDING_LABEL])])
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut pending, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace[1],
            w_act(WorkerAction::ClearIssueState { issue_iid: 7 })
        );
        assert!(run.active.is_none());

        // A human asked for review-only.
        let mut review_only = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKER_REVIEW_ONLY_LABEL])])
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut review_only, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace[1],
            w_act(WorkerAction::ReleaseReviewOnlyHold { issue_iid: 7 })
        );
        assert!(run.active.is_none());

        // The issue itself could not be read.
        let mut unreadable = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_query(WorkerQuery::Issue { issue_iid: 7 })
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut unreadable, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(
            run.trace[1],
            w_act(WorkerAction::ClearIssueState { issue_iid: 7 })
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_re_attempts_an_active_issue_that_never_reached_an_mr() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut port, Some(active_without_mr(7)));

        assert!(run.result.is_ok());
        assert_eq!(
            run.trace,
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_act(WorkerAction::RunImplementation {
                    issue: Box::new(issue_observation(7, &[WORKING_ON_LABEL])),
                }),
                w_act(WorkerAction::CheckIssueTrackable { issue_iid: 7 }),
            ]
        );
        assert_eq!(run.active, Some(active_with_mr(7, 7)));
    }

    #[test]
    fn worker_cycle_drops_the_issue_when_a_re_attempt_is_not_trackable() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .implementing(&[ImplementationScript::succeeded(7)])
            .untrackable();
        let run = run_worker_routing(&mut port, Some(active_without_mr(7)));

        assert!(run.result.is_ok());
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_drops_the_session_after_a_failed_re_attempt() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "boom")]);
        let run = run_worker_routing(&mut port, Some(active_without_mr(7)));

        assert!(run.result.is_ok());
        assert_eq!(
            &run.trace[run.trace.len() - 8..],
            &[
                w_shutdown(),
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::RemoveWorkingOnLabel { issue_iid: 7 }),
                w_act(WorkerAction::CleanupSession { issue_iid: 7 }),
                w_observe(WorkerQuery::DefaultBranchOrMain),
                w_act(WorkerAction::ResetWorktree),
                w_act(WorkerAction::CheckoutBranch {
                    branch: "main".to_string(),
                }),
                w_act(WorkerAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_releases_a_re_attempt_whose_issue_disappeared_then_looks_for_new_work() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing_query(WorkerQuery::Issue { issue_iid: 7 })
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, Some(active_without_mr(7)));

        assert!(run.result.is_ok());
        // The first read is the hold check, which releases through
        // `ClearIssueState` rather than the fetch-failure cleanup.
        assert_eq!(
            run.trace,
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_act(WorkerAction::ClearIssueState { issue_iid: 7 }),
                w_shutdown(),
            ]
        );
        assert!(run.active.is_none());
    }

    #[test]
    fn worker_cycle_cleans_up_when_the_re_attempt_read_fails_after_the_hold_check() {
        struct FailSecondIssueRead {
            inner: FakeWorkerPort,
            reads: std::cell::Cell<usize>,
        }

        impl WorkerRoutingPort for FailSecondIssueRead {
            fn shutdown_requested(&self) -> bool {
                self.inner.shutdown_requested()
            }

            fn issue(&self, issue_iid: u64) -> Result<IssueObservation> {
                let read = self.reads.get();
                self.reads.set(read + 1);
                let observed = self.inner.issue(issue_iid)?;
                if read == 0 {
                    Ok(observed)
                } else {
                    anyhow::bail!("injected failure on the re-attempt read")
                }
            }

            fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation> {
                self.inner.merge_request_status(mr_iid)
            }

            fn issues(&self) -> Result<Vec<IssueObservation>> {
                self.inner.issues()
            }

            fn default_branch_or_main(&self) -> String {
                self.inner.default_branch_or_main()
            }

            fn execute(&mut self, action: &WorkerAction) -> WorkerOutcome {
                self.inner.execute(action)
            }
        }

        let mut port = FailSecondIssueRead {
            inner: FakeWorkerPort::new()
                .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
                .with_shutdown_answers(&[true]),
            reads: std::cell::Cell::new(0),
        };
        let mut machine =
            WorkerRoutingMachine::new(ROUTING_AGENT, None, Some(active_without_mr(7)));
        let result = drive_worker_routing(&mut machine, &mut port);

        assert!(result.is_ok());
        assert_eq!(
            port.inner.trace.borrow().clone(),
            vec![
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_observe(WorkerQuery::Issue { issue_iid: 7 }),
                w_act(WorkerAction::ReleaseIssueClaim { issue_iid: 7 }),
                w_act(WorkerAction::RemoveWorkingOnLabel { issue_iid: 7 }),
                w_act(WorkerAction::CleanupSession { issue_iid: 7 }),
                w_shutdown(),
            ]
        );
        assert!(machine.active.is_none());
    }

    // -----------------------------------------------------------------
    // Implementation progression traces. Workspace preparation and the
    // model invocation are actions, so a whole implementation run — down
    // to which branch of the handoff union it took — is an ordered trace.
    // -----------------------------------------------------------------

    /// A recording implementation port. Answers are scripted per query;
    /// failures are injected by naming the query or action that fails.
    struct FakeImplPort {
        trace: std::cell::RefCell<Vec<ImplStep>>,
        closes_linked: Option<ClosesLinkedMr>,
        open_mr: Option<u64>,
        default_branch: String,
        remote_branch_exists: bool,
        diff_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        staged: bool,
        merge_clean: bool,
        existing_mr_state: Option<String>,
        dependency_closed: bool,
        model_cancelled: bool,
        model_output: WorkerImplementationOutput,
        created_mr_iid: u64,
        stop_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        failing_queries: Vec<ImplQuery>,
        failing_actions: Vec<ImplAction>,
    }

    impl FakeImplPort {
        fn new() -> Self {
            Self {
                trace: std::cell::RefCell::new(Vec::new()),
                closes_linked: None,
                open_mr: None,
                default_branch: "main".to_string(),
                remote_branch_exists: false,
                diff_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                staged: true,
                merge_clean: true,
                existing_mr_state: Some("opened".to_string()),
                dependency_closed: false,
                model_cancelled: false,
                model_output: WorkerImplementationOutput::Implemented(ImplementedMetadata {
                    mr_title: Some("Cap the retry backoff".to_string()),
                    mr_description: Some("Capped the backoff at 30s.".to_string()),
                    changes_summary: None,
                }),
                created_mr_iid: 12,
                stop_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                failing_queries: Vec::new(),
                failing_actions: Vec::new(),
            }
        }

        fn with_closes_linked(mut self, linked: ClosesLinkedMr) -> Self {
            self.closes_linked = Some(linked);
            self
        }

        fn with_open_mr(mut self, mr_iid: u64) -> Self {
            self.open_mr = Some(mr_iid);
            self
        }

        fn with_remote_branch(mut self) -> Self {
            self.remote_branch_exists = true;
            self
        }

        fn with_diff_answers(self, answers: &[bool]) -> Self {
            self.diff_answers.borrow_mut().extend(answers);
            self
        }

        fn nothing_staged(mut self) -> Self {
            self.staged = false;
            self
        }

        fn conflicting_merge(mut self) -> Self {
            self.merge_clean = false;
            self
        }

        fn deciding(mut self, output: WorkerImplementationOutput) -> Self {
            self.model_output = output;
            self
        }

        fn cancelling_model(mut self) -> Self {
            self.model_cancelled = true;
            self
        }

        fn with_existing_mr_state(mut self, state: Option<&str>) -> Self {
            self.existing_mr_state = state.map(str::to_string);
            self
        }

        fn with_closed_dependency(mut self) -> Self {
            self.dependency_closed = true;
            self
        }

        fn stopping_at(self, answers: &[bool]) -> Self {
            self.stop_answers.borrow_mut().extend(answers);
            self
        }

        fn failing_query(mut self, query: ImplQuery) -> Self {
            self.failing_queries.push(query);
            self
        }

        fn failing_action(mut self, action: ImplAction) -> Self {
            self.failing_actions.push(action);
            self
        }

        fn record(&self, step: ImplStep) {
            self.trace.borrow_mut().push(step);
        }

        fn observe(&self, query: ImplQuery) -> Result<()> {
            self.record(ImplStep::Observe(query.clone()));
            if self.failing_queries.contains(&query) {
                anyhow::bail!("injected failure observing {query:?}");
            }
            Ok(())
        }
    }

    impl ImplementationPort for FakeImplPort {
        fn closes_linked_mr(&self) -> Option<ClosesLinkedMr> {
            self.record(ImplStep::Observe(ImplQuery::ClosesLinkedMr));
            self.closes_linked
        }

        fn open_mr_for_issue(&self) -> Option<u64> {
            self.record(ImplStep::Observe(ImplQuery::OpenMrForIssue));
            self.open_mr
        }

        fn default_branch(&self) -> Result<String> {
            self.observe(ImplQuery::DefaultBranch)?;
            Ok(self.default_branch.clone())
        }

        fn default_branch_or_main(&self) -> String {
            self.record(ImplStep::Observe(ImplQuery::DefaultBranchOrMain));
            self.default_branch.clone()
        }

        fn remote_branch_exists(&self, _branch: &str) -> Result<bool> {
            self.observe(ImplQuery::RemoteBranchExists)?;
            Ok(self.remote_branch_exists)
        }

        fn has_diff_against(&self, _base: &str) -> Result<bool> {
            self.observe(ImplQuery::DiffAgainstDefault)?;
            Ok(self.diff_answers.borrow_mut().pop_front().unwrap_or(true))
        }

        fn has_staged_changes(&self) -> Result<bool> {
            self.observe(ImplQuery::StagedChanges)?;
            Ok(self.staged)
        }

        fn merge_request_state(&self, mr_iid: u64) -> Option<String> {
            self.record(ImplStep::Observe(ImplQuery::MergeRequestState { mr_iid }));
            self.existing_mr_state.clone()
        }

        fn dependency_closed(&self, issue_iid: u64) -> bool {
            self.record(ImplStep::Observe(ImplQuery::DependencyClosed { issue_iid }));
            self.dependency_closed
        }

        fn issue_comments(&self) -> String {
            self.record(ImplStep::Observe(ImplQuery::IssueComments));
            "- alice: please cap it".to_string()
        }

        fn execute(&mut self, action: &ImplAction) -> ImplOutcome {
            self.record(ImplStep::Act(action.clone()));
            if self.failing_actions.contains(action) {
                return ImplOutcome::Failed(anyhow::anyhow!(
                    "injected failure executing {action:?}"
                ));
            }
            match action {
                ImplAction::StopIfReviewOnly => ImplOutcome::Stopped(
                    self.stop_answers.borrow_mut().pop_front().unwrap_or(false),
                ),
                ImplAction::MergeBaseIntoBranch { .. } => ImplOutcome::Merged(self.merge_clean),
                ImplAction::BuildPrompt { continuation, .. } => {
                    ImplOutcome::Prompt(format!("prompt(continuation={continuation})"))
                }
                ImplAction::InvokeImplementationModel { .. }
                | ImplAction::NudgeImplementationModel => {
                    ImplOutcome::Model(if self.model_cancelled {
                        ImplModelResult::Cancelled
                    } else {
                        ImplModelResult::Output(Box::new(self.model_output.clone()))
                    })
                }
                ImplAction::CreateMergeRequest { .. } => {
                    ImplOutcome::MergeRequestCreated(self.created_mr_iid)
                }
                _ => ImplOutcome::Done,
            }
        }
    }

    struct FakeImplRun {
        result: Result<Option<u64>>,
        trace: Vec<ImplStep>,
        mr_created: bool,
        branch: Option<String>,
    }

    fn run_implementation(port: &mut FakeImplPort, scope_label: Option<&str>) -> FakeImplRun {
        let mut machine =
            ImplementationMachine::new(ROUTING_AGENT, scope_label, issue_observation(7, &[]));
        let result = drive_implementation(&mut machine, port);
        FakeImplRun {
            result: result.map(|()| machine.tracked_mr),
            trace: port.trace.borrow().clone(),
            mr_created: machine.mr_created,
            branch: machine.left_branch.clone(),
        }
    }

    fn i_observe(query: ImplQuery) -> ImplStep {
        ImplStep::Observe(query)
    }

    fn i_act(action: ImplAction) -> ImplStep {
        ImplStep::Act(action)
    }

    /// The steps that take a fresh issue from discovery to the model's
    /// answer, for an issue whose branch does not exist yet.
    fn steps_up_to_model(continuation: bool) -> Vec<ImplStep> {
        vec![
            i_act(ImplAction::StopIfReviewOnly),
            i_observe(ImplQuery::ClosesLinkedMr),
            i_observe(ImplQuery::OpenMrForIssue),
            i_observe(ImplQuery::DefaultBranch),
            i_act(ImplAction::FetchRemote),
            i_act(ImplAction::ResetWorktree),
            i_observe(ImplQuery::RemoteBranchExists),
            i_act(ImplAction::CreateBranchFrom {
                branch: "issue-7".to_string(),
                base: "main".to_string(),
            }),
            i_act(ImplAction::StopIfReviewOnly),
            i_act(ImplAction::RequireWorkingOnLabel),
            i_observe(ImplQuery::IssueComments),
            i_act(ImplAction::BuildPrompt {
                continuation,
                comments: "- alice: please cap it".to_string(),
            }),
            i_act(ImplAction::InvokeImplementationModel {
                prompt: format!("prompt(continuation={continuation})"),
            }),
        ]
    }

    #[test]
    fn implementation_prepares_the_branch_invokes_the_model_then_opens_the_merge_request() {
        let mut port = FakeImplPort::new();
        let run = run_implementation(&mut port, Some("team:core"));

        assert_eq!(run.result.unwrap(), Some(12));
        let mut expected = steps_up_to_model(false);
        expected.extend([
            i_act(ImplAction::StageAll),
            i_observe(ImplQuery::StagedChanges),
            i_act(ImplAction::Commit {
                message: build_commit_message("Cap the retry backoff", 7),
            }),
            i_observe(ImplQuery::DiffAgainstDefault),
            i_act(ImplAction::PushBranch {
                branch: "issue-7".to_string(),
            }),
            i_act(ImplAction::CreateMergeRequest {
                branch: "issue-7".to_string(),
                base: "main".to_string(),
                title: "Cap the retry backoff".to_string(),
                description: "Closes #7\n\nCapped the backoff at 30s.".to_string(),
            }),
            i_act(ImplAction::AddMrScopeLabel { mr_iid: 12 }),
            i_act(ImplAction::RequireSaveSessionWithSummary {
                mr_iid: 12,
                summary: "Capped the backoff at 30s.".to_string(),
            }),
        ]);
        assert_eq!(run.trace, expected);
        assert!(run.mr_created);
        assert_eq!(run.branch, Some("issue-7".to_string()));
    }

    #[test]
    fn implementation_skips_the_scope_label_when_the_worker_has_no_scope() {
        let mut port = FakeImplPort::new();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert!(
            !run.trace
                .iter()
                .any(|step| matches!(step, ImplStep::Act(ImplAction::AddMrScopeLabel { .. })))
        );
    }

    #[test]
    fn implementation_adopts_a_merge_request_linked_by_a_closes_keyword() {
        let mut port = FakeImplPort::new().with_closes_linked(ClosesLinkedMr::Open(9));
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(9));
        assert_eq!(
            run.trace,
            vec![
                i_act(ImplAction::StopIfReviewOnly),
                i_observe(ImplQuery::ClosesLinkedMr),
                i_act(ImplAction::AddWorkingOnLabel),
                i_act(ImplAction::SaveSession { mr_iid: 9 }),
            ]
        );
        assert!(run.mr_created);
        assert!(run.branch.is_none());
    }

    #[test]
    fn implementation_closes_an_issue_a_merged_merge_request_already_covered() {
        let mut port = FakeImplPort::new().with_closes_linked(ClosesLinkedMr::Merged);
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), None);
        assert_eq!(
            run.trace,
            vec![
                i_act(ImplAction::StopIfReviewOnly),
                i_observe(ImplQuery::ClosesLinkedMr),
                i_act(ImplAction::CloseIssue),
                i_act(ImplAction::CleanupSession),
            ]
        );
        assert!(!run.mr_created);
    }

    #[test]
    fn implementation_adopts_an_open_merge_request_on_the_issue_branch() {
        let mut port = FakeImplPort::new().with_open_mr(9);
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(9));
        assert_eq!(
            run.trace,
            vec![
                i_act(ImplAction::StopIfReviewOnly),
                i_observe(ImplQuery::ClosesLinkedMr),
                i_observe(ImplQuery::OpenMrForIssue),
                i_act(ImplAction::AddWorkingOnLabel),
                i_act(ImplAction::RequireSaveSession { mr_iid: 9 }),
            ]
        );
    }

    #[test]
    fn implementation_continues_an_existing_branch_that_merges_cleanly() {
        let mut port = FakeImplPort::new().with_remote_branch();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            &run.trace[6..10],
            &[
                i_observe(ImplQuery::RemoteBranchExists),
                i_act(ImplAction::CheckoutBranch {
                    branch: "issue-7".to_string(),
                }),
                i_observe(ImplQuery::DiffAgainstDefault),
                i_act(ImplAction::MergeBaseIntoBranch {
                    base: "main".to_string(),
                }),
            ]
        );
        // A branch that already has work gets the continuation prompt.
        assert!(run.trace.contains(&i_act(ImplAction::BuildPrompt {
            continuation: true,
            comments: "- alice: please cap it".to_string(),
        })));
    }

    #[test]
    fn implementation_discards_a_stale_branch_and_starts_over() {
        let mut port = FakeImplPort::new()
            .with_remote_branch()
            .with_diff_answers(&[false, true]);
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            &run.trace[8..14],
            &[
                i_observe(ImplQuery::DiffAgainstDefault),
                i_act(ImplAction::ResetWorktree),
                i_act(ImplAction::CheckoutBranch {
                    branch: "main".to_string(),
                }),
                i_act(ImplAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
                i_act(ImplAction::DeleteRemoteBranch {
                    branch: "issue-7".to_string(),
                }),
                i_act(ImplAction::CreateBranchFrom {
                    branch: "issue-7".to_string(),
                    base: "main".to_string(),
                }),
            ]
        );
        assert!(run.trace.contains(&i_act(ImplAction::BuildPrompt {
            continuation: false,
            comments: "- alice: please cap it".to_string(),
        })));
    }

    #[test]
    fn implementation_recreates_a_branch_that_conflicts_with_the_default_branch() {
        let mut port = FakeImplPort::new().with_remote_branch().conflicting_merge();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            &run.trace[9..14],
            &[
                i_act(ImplAction::MergeBaseIntoBranch {
                    base: "main".to_string(),
                }),
                i_act(ImplAction::ResetWorktree),
                i_act(ImplAction::CheckoutBranch {
                    branch: "main".to_string(),
                }),
                i_act(ImplAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
                i_act(ImplAction::CreateBranchFrom {
                    branch: "issue-7".to_string(),
                    base: "main".to_string(),
                }),
            ]
        );
        // The recreated branch is fresh, so no remote branch is deleted.
        assert!(
            !run.trace
                .iter()
                .any(|step| matches!(step, ImplStep::Act(ImplAction::DeleteRemoteBranch { .. })))
        );
    }

    #[test]
    fn implementation_stops_when_a_human_asks_for_review_only() {
        let mut before = FakeImplPort::new().stopping_at(&[true]);
        let run = run_implementation(&mut before, None);
        assert_eq!(run.result.unwrap(), None);
        assert_eq!(run.trace, vec![i_act(ImplAction::StopIfReviewOnly)]);
        assert!(run.branch.is_none());

        let mut after_prep = FakeImplPort::new().stopping_at(&[false, true]);
        let run = run_implementation(&mut after_prep, None);
        assert_eq!(run.result.unwrap(), None);
        assert_eq!(run.trace.last(), Some(&i_act(ImplAction::StopIfReviewOnly)));
        // The branch was already created, so the cycle has to clean it up.
        assert_eq!(run.branch, Some("issue-7".to_string()));
    }

    #[test]
    fn implementation_ends_quietly_when_the_model_run_was_cancelled() {
        let mut port = FakeImplPort::new().cancelling_model();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), None);
        assert_eq!(run.trace, steps_up_to_model(false));
        assert!(!run.mr_created);
    }

    #[test]
    fn implementation_hands_the_issue_back_and_forgets_the_branch_when_blocked() {
        for output in [
            WorkerImplementationOutput::NeedsSplit(BlockedOutcome {
                reason: "Two features in one issue".to_string(),
                public_comment: None,
            }),
            WorkerImplementationOutput::NeedsClarification(BlockedOutcome {
                reason: "Which backoff cap?".to_string(),
                public_comment: None,
            }),
            WorkerImplementationOutput::CannotImplement(BlockedOutcome {
                reason: "The API does not exist".to_string(),
                public_comment: None,
            }),
        ] {
            let mut port = FakeImplPort::new().deciding(output);
            let run = run_implementation(&mut port, None);

            assert_eq!(run.result.unwrap(), None);
            assert!(matches!(
                run.trace.last(),
                Some(ImplStep::Act(ImplAction::HandIssueBackToHumans { .. }))
            ));
            // The hand-back already reset the worktree, so the cycle must
            // not delete the branch a second time.
            assert!(run.branch.is_none());
            assert!(!run.mr_created);
        }
    }

    #[test]
    fn implementation_parks_an_issue_whose_dependency_is_still_open() {
        let mut port = FakeImplPort::new().deciding(WorkerImplementationOutput::WaitDependency {
            depends_on_issue: 4,
        });
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), None);
        let tail = &run.trace[run.trace.len() - 10..];
        assert_eq!(
            tail,
            &[
                i_observe(ImplQuery::DependencyClosed { issue_iid: 4 }),
                i_act(ImplAction::AddIssueLabel {
                    label: waiting_on_issue_label(4),
                }),
                i_act(ImplAction::RemoveWorkingOnLabel),
                i_act(ImplAction::AddIssueComment {
                    body: "Implementation cannot proceed until issue #4 is closed. \
                         Parking this issue until the dependency resolves."
                        .to_string(),
                }),
                i_act(ImplAction::ReleaseIssueClaim),
                i_act(ImplAction::CleanupSession),
                i_observe(ImplQuery::DefaultBranchOrMain),
                i_act(ImplAction::ResetWorktree),
                i_act(ImplAction::CheckoutBranchBestEffort {
                    branch: "main".to_string(),
                }),
                i_act(ImplAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
            ]
        );
        assert!(run.branch.is_none());
        assert!(!run.mr_created);
    }

    #[test]
    fn implementation_proceeds_normally_when_the_declared_dependency_is_already_closed() {
        let mut port = FakeImplPort::new()
            .deciding(WorkerImplementationOutput::WaitDependency {
                depends_on_issue: 4,
            })
            .with_closed_dependency();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        // No MR metadata came back with a dependency handoff, so the title
        // falls back to the issue title.
        assert!(run.trace.contains(&i_act(ImplAction::CreateMergeRequest {
            branch: "issue-7".to_string(),
            base: "main".to_string(),
            title: "Cap the retry backoff for issue 7".to_string(),
            description: "Closes #7\n\nImplementation completed.".to_string(),
        })));
    }

    #[test]
    fn implementation_adopts_the_merge_request_the_model_pointed_at() {
        let mut port = FakeImplPort::new()
            .deciding(WorkerImplementationOutput::ExistingMr { existing_mr_iid: 9 });
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(9));
        assert_eq!(
            &run.trace[run.trace.len() - 7..],
            &[
                i_observe(ImplQuery::MergeRequestState { mr_iid: 9 }),
                i_act(ImplAction::ResetWorktree),
                i_observe(ImplQuery::DefaultBranchOrMain),
                i_act(ImplAction::CheckoutBranchBestEffort {
                    branch: "main".to_string(),
                }),
                i_act(ImplAction::DeleteLocalBranch {
                    branch: "issue-7".to_string(),
                }),
                i_act(ImplAction::RequireSaveSession { mr_iid: 9 }),
                i_act(ImplAction::AddWorkingOnLabel),
            ]
        );
        assert!(run.mr_created);
    }

    #[test]
    fn implementation_opens_its_own_merge_request_when_the_pointed_at_one_is_unusable() {
        for state in [Some("merged"), None] {
            let mut port = FakeImplPort::new()
                .deciding(WorkerImplementationOutput::ExistingMr { existing_mr_iid: 9 })
                .with_existing_mr_state(state);
            let run = run_implementation(&mut port, None);

            assert_eq!(run.result.unwrap(), Some(12));
            assert!(run.trace.contains(&i_act(ImplAction::PushBranch {
                branch: "issue-7".to_string(),
            })));
        }
    }

    #[test]
    fn implementation_skips_the_commit_when_the_agent_committed_its_own_work() {
        let mut port = FakeImplPort::new().nothing_staged();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert!(
            !run.trace
                .iter()
                .any(|step| matches!(step, ImplStep::Act(ImplAction::Commit { .. })))
        );
    }

    #[test]
    fn implementation_nudges_the_same_session_when_the_agent_produced_no_code_changes() {
        let mut port = FakeImplPort::new()
            .nothing_staged()
            .with_diff_answers(&[false, true]);
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert!(
            run.trace
                .contains(&i_act(ImplAction::NudgeImplementationModel))
        );
        assert_eq!(
            run.trace
                .iter()
                .filter(|step| { matches!(step, ImplStep::Observe(ImplQuery::DiffAgainstDefault)) })
                .count(),
            2
        );
        assert_eq!(run.branch, Some("issue-7".to_string()));
    }

    #[test]
    fn implementation_fails_the_run_when_a_required_step_fails() {
        let mut branching = FakeImplPort::new().failing_action(ImplAction::CreateBranchFrom {
            branch: "issue-7".to_string(),
            base: "main".to_string(),
        });
        let run = run_implementation(&mut branching, None);
        assert!(run.result.is_err());
        // The branch was never created, so there is nothing to clean up.
        assert!(run.branch.is_none());

        let mut pushing = FakeImplPort::new().failing_action(ImplAction::PushBranch {
            branch: "issue-7".to_string(),
        });
        let run = run_implementation(&mut pushing, None);
        assert!(run.result.is_err());
        assert_eq!(run.branch, Some("issue-7".to_string()));

        let mut labeling = FakeImplPort::new().failing_action(ImplAction::RequireWorkingOnLabel);
        assert!(run_implementation(&mut labeling, None).result.is_err());

        let mut reading = FakeImplPort::new().failing_query(ImplQuery::DefaultBranch);
        assert!(run_implementation(&mut reading, None).result.is_err());
    }

    // -----------------------------------------------------------------
    // Feedback progression traces. The order the worker performs the
    // post-model writes in is load-bearing (metadata, then commit/push,
    // then the conflict recheck, then reply before resolve), so it is
    // characterized end to end against a recording port.
    // -----------------------------------------------------------------

    fn feedback_surface(title: &str, has_conflicts: bool) -> MrSurfaceObservation {
        MrSurfaceObservation {
            title: title.to_string(),
            description: "Closes #7".to_string(),
            labels: Some(vec!["ai".to_string()]),
            has_conflicts,
        }
    }

    fn feedback_input(
        unresolved_ids: &[&str],
        resolution: FeedbackResolution,
    ) -> FeedbackTailInput {
        FeedbackTailInput {
            mr_iid: 12,
            source_branch: "issue-7".to_string(),
            target_branch: "main".to_string(),
            pre_agent_sha: "sha-before".to_string(),
            requires_conflict_resolution: false,
            unresolved_ids: unresolved_ids.iter().map(|id| id.to_string()).collect(),
            plain_comments_present: false,
            issue_iid: Some(7),
            surface_before: feedback_surface("Cap the retry backoff", false),
            resolution,
        }
    }

    fn addressed(summary: &str) -> FeedbackResolution {
        FeedbackResolution {
            changes_summary: Some(summary.to_string()),
            ..Default::default()
        }
    }

    /// A recording feedback port. Every answer is scripted; failures are
    /// injected by naming the query or action that should fail.
    struct FakeFeedbackPort {
        trace: std::cell::RefCell<Vec<FeedbackStep>>,
        changes_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        conflict_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        staged_answers: std::cell::RefCell<std::collections::VecDeque<bool>>,
        merge_in_progress: bool,
        staged_conflicts: bool,
        merge_completed: bool,
        up_to_date: bool,
        highlights: Option<String>,
        surface_after: MrSurfaceObservation,
        origin_head: Option<String>,
        refetched_ids: Vec<String>,
        failing_queries: Vec<FeedbackQuery>,
        failing_actions: Vec<FeedbackAction>,
    }

    impl FakeFeedbackPort {
        fn new() -> Self {
            Self {
                trace: std::cell::RefCell::new(Vec::new()),
                changes_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                conflict_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                staged_answers: std::cell::RefCell::new(std::collections::VecDeque::new()),
                merge_in_progress: false,
                staged_conflicts: false,
                merge_completed: false,
                up_to_date: true,
                highlights: Some("- src/lib.rs\n- 1 file changed".to_string()),
                surface_after: feedback_surface("Cap the retry backoff", false),
                origin_head: None,
                refetched_ids: Vec::new(),
                failing_queries: Vec::new(),
                failing_actions: Vec::new(),
            }
        }

        fn with_changes(self, answers: &[bool]) -> Self {
            self.changes_answers.borrow_mut().extend(answers);
            self
        }

        fn with_conflicts(self, answers: &[bool]) -> Self {
            self.conflict_answers.borrow_mut().extend(answers);
            self
        }

        fn with_staged(self, answers: &[bool]) -> Self {
            self.staged_answers.borrow_mut().extend(answers);
            self
        }

        fn merging(mut self, staged_conflicts: bool) -> Self {
            self.merge_in_progress = true;
            self.staged_conflicts = staged_conflicts;
            self
        }

        fn completing_merge(mut self) -> Self {
            self.merge_completed = true;
            self
        }

        fn behind_target(mut self) -> Self {
            self.up_to_date = false;
            self
        }

        fn with_surface_after(mut self, surface: MrSurfaceObservation) -> Self {
            self.surface_after = surface;
            self
        }

        fn with_origin_head(mut self, head: &str) -> Self {
            self.origin_head = Some(head.to_string());
            self
        }

        fn refetching_ids(mut self, ids: &[&str]) -> Self {
            self.refetched_ids = ids.iter().map(|id| id.to_string()).collect();
            self
        }

        fn failing_query(mut self, query: FeedbackQuery) -> Self {
            self.failing_queries.push(query);
            self
        }

        fn failing_action(mut self, action: FeedbackAction) -> Self {
            self.failing_actions.push(action);
            self
        }

        fn record(&self, step: FeedbackStep) {
            self.trace.borrow_mut().push(step);
        }

        fn observe(&self, query: FeedbackQuery) -> Result<()> {
            self.record(FeedbackStep::Observe(query.clone()));
            if self.failing_queries.contains(&query) {
                anyhow::bail!("injected failure observing {query:?}");
            }
            Ok(())
        }

        fn next(
            queue: &std::cell::RefCell<std::collections::VecDeque<bool>>,
            fallback: bool,
        ) -> bool {
            queue.borrow_mut().pop_front().unwrap_or(fallback)
        }
    }

    impl FeedbackTailPort for FakeFeedbackPort {
        fn has_changes_since(&self, _base_ref: &str) -> Result<bool> {
            self.observe(FeedbackQuery::ChangesSinceModelRun)?;
            Ok(Self::next(&self.changes_answers, false))
        }

        fn merge_in_progress(&self) -> Result<bool> {
            self.observe(FeedbackQuery::MergeInProgress)?;
            Ok(self.merge_in_progress)
        }

        fn merge_conflicts_present(&self) -> Result<bool> {
            self.observe(FeedbackQuery::MergeConflictsPresent)?;
            Ok(Self::next(&self.conflict_answers, false))
        }

        fn has_staged_changes(&self) -> Result<bool> {
            self.observe(FeedbackQuery::StagedChanges)?;
            Ok(Self::next(&self.staged_answers, true))
        }

        fn up_to_date_with_target(&self, _target_branch: &str) -> Result<bool> {
            self.observe(FeedbackQuery::UpToDateWithTarget)?;
            Ok(self.up_to_date)
        }

        fn diff_highlights(&self, _base_ref: &str) -> Option<String> {
            self.record(FeedbackStep::Observe(FeedbackQuery::DiffHighlights));
            self.highlights.clone()
        }

        fn merge_request_surface(&self, _mr_iid: u64) -> Result<MrSurfaceObservation> {
            self.observe(FeedbackQuery::MergeRequestSurface)?;
            Ok(self.surface_after.clone())
        }

        fn origin_head(&self, _source_branch: &str) -> Option<String> {
            self.record(FeedbackStep::Observe(FeedbackQuery::OriginHead));
            self.origin_head.clone()
        }

        fn unresolved_discussion_ids(&self, _mr_iid: u64) -> Vec<String> {
            self.record(FeedbackStep::Observe(
                FeedbackQuery::UnresolvedDiscussionIds,
            ));
            self.refetched_ids.clone()
        }

        fn execute(&mut self, action: &FeedbackAction) -> FeedbackOutcome {
            self.record(FeedbackStep::Act(action.clone()));
            if self.failing_actions.contains(action) {
                return FeedbackOutcome::Failed(anyhow::anyhow!(
                    "injected failure executing {action:?}"
                ));
            }
            match action {
                FeedbackAction::StageResolvedConflicts => {
                    FeedbackOutcome::Staged(self.staged_conflicts)
                }
                FeedbackAction::CompleteMergeIfReady { .. } => {
                    FeedbackOutcome::MergeCompleted(self.merge_completed)
                }
                _ => FeedbackOutcome::Done,
            }
        }
    }

    fn run_feedback(
        port: &mut FakeFeedbackPort,
        input: FeedbackTailInput,
    ) -> (Result<()>, Vec<FeedbackStep>) {
        let mut machine = FeedbackMachine::new(input);
        let result = drive_feedback(&mut machine, port);
        (result, port.trace.borrow().clone())
    }

    fn f_observe(query: FeedbackQuery) -> FeedbackStep {
        FeedbackStep::Observe(query)
    }

    fn f_act(action: FeedbackAction) -> FeedbackStep {
        FeedbackStep::Act(action)
    }

    #[test]
    fn feedback_commits_pushes_then_replies_and_resolves_each_discussion_in_order() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1", "d2"], addressed("Capped the backoff")),
        );

        assert!(result.is_ok());
        // `strip_worker_reply_boilerplate` trims the prefix and the diff
        // block back off the generated reply before it is posted.
        let body = "Capped the backoff";
        assert_eq!(
            trace,
            vec![
                f_act(FeedbackAction::FetchBranches),
                f_observe(FeedbackQuery::ChangesSinceModelRun),
                f_observe(FeedbackQuery::MergeInProgress),
                f_act(FeedbackAction::StageAll),
                f_observe(FeedbackQuery::StagedChanges),
                f_act(FeedbackAction::Commit {
                    message: build_commit_message("Capped the backoff", 7),
                }),
                f_act(FeedbackAction::CompleteMergeIfReady {
                    message: build_commit_message("Merge origin/main into issue-7", 7),
                }),
                f_observe(FeedbackQuery::MergeConflictsPresent),
                f_observe(FeedbackQuery::MergeConflictsPresent),
                f_observe(FeedbackQuery::DiffHighlights),
                f_act(FeedbackAction::PushSourceBranch),
                f_observe(FeedbackQuery::MergeRequestSurface),
                f_observe(FeedbackQuery::OriginHead),
                f_act(FeedbackAction::ReplyToDiscussion {
                    discussion_id: "d1".to_string(),
                    body: body.to_string(),
                }),
                f_act(FeedbackAction::ResolveDiscussion {
                    discussion_id: "d1".to_string(),
                }),
                f_act(FeedbackAction::ReplyToDiscussion {
                    discussion_id: "d2".to_string(),
                    body: body.to_string(),
                }),
                f_act(FeedbackAction::ResolveDiscussion {
                    discussion_id: "d2".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn feedback_updates_mr_metadata_before_it_touches_the_branch() {
        let resolution = FeedbackResolution {
            mr_title: Some("Cap the retry backoff at 30s".to_string()),
            changes_summary: Some("Capped the backoff".to_string()),
            ..Default::default()
        };
        let mut port = FakeFeedbackPort::new().with_changes(&[true]);
        let (result, trace) = run_feedback(&mut port, feedback_input(&["d1"], resolution));

        assert!(result.is_ok());
        assert_eq!(
            &trace[..2],
            &[
                f_act(FeedbackAction::UpdateMrMetadata {
                    title: "Cap the retry backoff at 30s".to_string(),
                    description: "Closes #7".to_string(),
                }),
                f_act(FeedbackAction::FetchBranches),
            ]
        );
    }

    #[test]
    fn feedback_stages_a_conflict_resolution_before_it_looks_for_changes_again() {
        let mut port = FakeFeedbackPort::new()
            .merging(true)
            .with_changes(&[false, true])
            .completing_merge()
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1"], addressed("Resolved conflicts")),
        );

        assert!(result.is_ok());
        assert_eq!(
            &trace[..7],
            &[
                f_act(FeedbackAction::FetchBranches),
                f_observe(FeedbackQuery::ChangesSinceModelRun),
                f_observe(FeedbackQuery::MergeInProgress),
                f_act(FeedbackAction::StageResolvedConflicts),
                f_act(FeedbackAction::StageAll),
                f_observe(FeedbackQuery::ChangesSinceModelRun),
                f_act(FeedbackAction::CompleteMergeIfReady {
                    message: build_commit_message("Merge origin/main into issue-7", 7),
                }),
            ]
        );
        assert!(trace.contains(&f_act(FeedbackAction::PushSourceBranch)));
    }

    #[test]
    fn feedback_pushes_nothing_and_posts_nothing_while_conflicts_remain() {
        let mut input = feedback_input(&["d1"], addressed("Tried to resolve conflicts"));
        input.requires_conflict_resolution = true;
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .behind_target();
        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        assert_eq!(
            trace,
            vec![
                f_act(FeedbackAction::FetchBranches),
                f_observe(FeedbackQuery::ChangesSinceModelRun),
                f_observe(FeedbackQuery::MergeInProgress),
                f_act(FeedbackAction::StageAll),
                f_observe(FeedbackQuery::StagedChanges),
                f_act(FeedbackAction::Commit {
                    message: build_commit_message("Tried to resolve conflicts", 7),
                }),
                f_act(FeedbackAction::CompleteMergeIfReady {
                    message: build_commit_message("Merge origin/main into issue-7", 7),
                }),
                f_observe(FeedbackQuery::MergeConflictsPresent),
                f_act(FeedbackAction::FetchBranches),
                f_observe(FeedbackQuery::UpToDateWithTarget),
                f_observe(FeedbackQuery::DiffHighlights),
                f_observe(FeedbackQuery::MergeRequestSurface),
                f_observe(FeedbackQuery::OriginHead),
            ]
        );
    }

    #[test]
    fn feedback_treats_gitlab_reported_conflicts_as_unresolved_after_the_run() {
        let mut input = feedback_input(&["d1"], addressed("Capped the backoff"));
        input.requires_conflict_resolution = true;
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true, false])
            .with_surface_after(feedback_surface("Cap the retry backoff", true));
        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        // GitLab's own conflict flag is read after the push, so the replies
        // are skipped even though the local worktree looked clean.
        assert!(trace.contains(&f_act(FeedbackAction::PushSourceBranch)));
        assert_eq!(trace.last(), Some(&f_observe(FeedbackQuery::OriginHead)));
    }

    #[test]
    fn feedback_refetches_discussions_when_the_run_was_triggered_by_conflicts_alone() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_origin_head("sha-after")
            .refetching_ids(&["conflict-thread"]);
        let (result, trace) =
            run_feedback(&mut port, feedback_input(&[], addressed("Merged main")));

        assert!(result.is_ok());
        assert_eq!(
            &trace[trace.len() - 3..],
            &[
                f_observe(FeedbackQuery::UnresolvedDiscussionIds),
                f_act(FeedbackAction::ReplyToDiscussion {
                    discussion_id: "conflict-thread".to_string(),
                    body: "Merged main".to_string(),
                }),
                f_act(FeedbackAction::ResolveDiscussion {
                    discussion_id: "conflict-thread".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn feedback_without_branch_changes_replies_without_resolving() {
        let resolution = FeedbackResolution {
            reason: Some("The branch already handles this case".to_string()),
            ..Default::default()
        };
        let mut port = FakeFeedbackPort::new()
            .with_surface_after(feedback_surface("Cap the retry backoff at 30s", false));
        let (result, trace) = run_feedback(&mut port, feedback_input(&["d1"], resolution));

        assert!(result.is_ok());
        assert_eq!(
            trace,
            vec![
                f_act(FeedbackAction::FetchBranches),
                f_observe(FeedbackQuery::ChangesSinceModelRun),
                f_observe(FeedbackQuery::MergeInProgress),
                f_act(FeedbackAction::CompleteMergeIfReady {
                    message: build_commit_message("Merge origin/main into issue-7", 7),
                }),
                f_observe(FeedbackQuery::MergeConflictsPresent),
                f_observe(FeedbackQuery::MergeConflictsPresent),
                f_observe(FeedbackQuery::MergeRequestSurface),
                f_observe(FeedbackQuery::OriginHead),
                f_act(FeedbackAction::ReplyToDiscussion {
                    discussion_id: "d1".to_string(),
                    body: "The branch already handles this case".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn feedback_resolves_without_branch_changes_when_the_agent_says_so_explicitly() {
        let resolution = FeedbackResolution {
            reason: Some("Already handled".to_string()),
            mark_discussions_resolved: Some(true),
            ..Default::default()
        };
        let mut port = FakeFeedbackPort::new();
        let (result, trace) = run_feedback(&mut port, feedback_input(&["d1"], resolution));

        assert!(result.is_ok());
        assert_eq!(
            trace.last(),
            Some(&f_act(FeedbackAction::ResolveDiscussion {
                discussion_id: "d1".to_string(),
            }))
        );
    }

    #[test]
    fn feedback_posts_the_plain_comment_reply_last() {
        let resolution = FeedbackResolution {
            changes_summary: Some("Capped the backoff".to_string()),
            post_plain_comment: true,
            ..Default::default()
        };
        let mut input = feedback_input(&["d1"], resolution);
        input.plain_comments_present = true;
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        let body = "Capped the backoff";
        assert_eq!(
            &trace[trace.len() - 3..],
            &[
                f_act(FeedbackAction::ReplyToDiscussion {
                    discussion_id: "d1".to_string(),
                    body: body.to_string(),
                }),
                f_act(FeedbackAction::ResolveDiscussion {
                    discussion_id: "d1".to_string(),
                }),
                f_act(FeedbackAction::PostPlainComment {
                    body: body.to_string(),
                }),
            ]
        );
    }

    #[test]
    fn feedback_skips_the_commit_when_the_agent_left_nothing_staged() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_staged(&[false])
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1"], addressed("Committed it itself")),
        );

        assert!(result.is_ok());
        assert!(
            !trace
                .iter()
                .any(|step| matches!(step, FeedbackStep::Act(FeedbackAction::Commit { .. })))
        );
        // The agent's own commit still counts as a change to push.
        assert!(trace.contains(&f_act(FeedbackAction::PushSourceBranch)));
    }

    #[test]
    fn feedback_holds_the_push_back_when_the_worktree_still_has_conflict_markers() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_conflicts(&[true]);
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1"], addressed("Half-resolved")),
        );

        assert!(result.is_ok());
        assert!(!trace.contains(&f_act(FeedbackAction::PushSourceBranch)));
        // The conflict probe answered "conflicts", so the second probe is
        // skipped and the replies go out without resolving anything.
        assert_eq!(
            trace
                .iter()
                .filter(|step| **step == f_observe(FeedbackQuery::MergeConflictsPresent))
                .count(),
            1
        );
        assert_eq!(
            trace.last(),
            Some(&f_act(FeedbackAction::ReplyToDiscussion {
                discussion_id: "d1".to_string(),
                body: "Half-resolved".to_string(),
            }))
        );
    }

    #[test]
    fn feedback_fails_when_the_worker_produced_neither_changes_nor_an_explanation() {
        let mut port = FakeFeedbackPort::new();
        let (result, _) = run_feedback(
            &mut port,
            feedback_input(&["d1"], FeedbackResolution::default()),
        );

        let error = result.expect_err("a silent no-op run must fail the feedback cycle");
        assert!(error.to_string().contains("no feedback reply"), "{error}");
    }

    #[test]
    fn feedback_fails_the_run_when_a_required_git_step_fails() {
        let mut fetching = FakeFeedbackPort::new().failing_action(FeedbackAction::FetchBranches);
        let (result, _) = run_feedback(&mut fetching, feedback_input(&["d1"], addressed("x")));
        assert!(result.is_err());

        let mut pushing = FakeFeedbackPort::new()
            .with_changes(&[true])
            .failing_action(FeedbackAction::PushSourceBranch);
        let (result, _) = run_feedback(&mut pushing, feedback_input(&["d1"], addressed("x")));
        assert!(result.is_err());

        let mut probing = FakeFeedbackPort::new()
            .with_changes(&[true])
            .failing_query(FeedbackQuery::MergeConflictsPresent);
        let (result, _) = run_feedback(&mut probing, feedback_input(&["d1"], addressed("x")));
        assert!(result.is_err());
    }

    #[test]
    fn worker_cycle_stops_before_polling_when_shutdown_was_requested() {
        let mut port = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[])])
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(run.trace, vec![w_shutdown()]);
    }

    #[test]
    fn worker_cycle_stops_between_the_listing_and_the_first_candidate() {
        let mut port = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[])])
            .with_shutdown_answers(&[false, true]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(run.trace, polling_steps());
    }
}
