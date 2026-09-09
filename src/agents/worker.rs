use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::claim::{self, ClaimAcquireOutcome, ClaimLease, ClaimResource};
use super::labels::DO_NOT_IMPLEMENT;
use super::state::StateStore;
use crate::agents::artifact::write_task_context_file;
use crate::agents::forge::{
    self, Comment, ForgeClient, Issue, MergeRequest, is_not_found, issue_in_scope,
    mr_description_closes_issue, scope_label_filter, split_parent_iid,
};
use crate::agents::git::GitRepo;
use crate::agents::workspace::{AgentBootstrap, AgentWorkspace, repo_banner};
use crate::core::agent::schema::tagged;
use crate::core::agent::{
    AgentModel, CoreAgent, InvokeOptions, ModelPreferences, ObjectSchema, Schema, compat,
    structured_output,
};
use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::PeriodicTaskSpec;
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;
use crate::paths::display_name;

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
/// Prefix for labels that park an issue until a dependency issue is closed.
/// The full label is `waiting-on-issue:#N` where N is the dependency issue IID.
const WAITING_ON_ISSUE_LABEL_PREFIX: &str = "waiting-on-issue:#";
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
                "Human-facing comment text to post instead of `reason`. Omit to post `reason` as-is.",
            ),
        )
}

structured_output! {
    impl WorkerImplementationOutput {
        tool_name: HANDOFF_TOOL;
        tool_description: "The final outcome of this worker implementation run.";
        schema: one_of(
            "outcome",
            "How the implementation run ended. Pick exactly one and send only that outcome's fields.",
            {
                "implemented" => (
                    format!("You made the code changes; {} commits, pushes, and opens the merge request.", display_name()),
                    fields(mr_metadata_properties(ObjectSchema::new()))
                ),
                "existing_mr" => (
                    format!("You found an already-open merge request that implements this issue; {} tracks it instead of opening a new one.", display_name()),
                    object({
                        required existing_mr_iid: integer(
                            "IID of the existing open merge request."
                        ),
                    })
                ),
                "wait_dependency" => (
                    format!("The work is hard-blocked until another issue closes; {} parks this issue and resumes it automatically.", display_name()),
                    object({
                        required depends_on_issue: integer(
                            "IID of the issue that must close before this work can proceed."
                        ),
                    })
                ),
                "needs_split" => (
                    "The issue is too broad for one merge request and must be split first.",
                    fields(blocked_properties(
                        ObjectSchema::new(),
                        "The estimated line count and how to split the issue into smaller, focused issues."
                    ))
                ),
                "needs_clarification" => (
                    "The issue is missing information you cannot infer; a human must answer before you can proceed.",
                    fields(blocked_properties(
                        ObjectSchema::new(),
                        "Precisely what information is needed and why you cannot proceed without it."
                    ))
                ),
                "cannot_implement" => (
                    "The issue cannot be implemented as specified for some other reason (contradictory requirements, no resource-safe approach).",
                    fields(blocked_properties(
                        ObjectSchema::new(),
                        "Why the issue cannot be implemented as specified."
                    ))
                ),
            }
        );
    /// Tolerated: an outcome spelled with different case or padding, and an
    /// IID sent as `"#7"` / `"!12"` instead of a number.
        normalize(value) {
            compat::normalize_tag(value, "outcome");
            compat::normalize_iid(value, "existing_mr_iid");
            compat::normalize_iid(value, "depends_on_issue");
        }
    }
}

structured_output! {
    impl WorkerFeedbackOutput {
        tool_name: HANDOFF_TOOL;
        tool_description: "The final outcome of this merge-request feedback run.";
        schema: one_of(
            "outcome",
            "How the feedback run ended. Pick exactly one and send only that outcome's fields.",
            {
                "addressed" => (
                    "You handled the reviewer feedback — in code, in merge request metadata, or by explaining that the branch already satisfies it.",
                    fields(mr_metadata_properties(ObjectSchema::new())
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
                            format!("Whether {d} may mark open review discussions resolved after posting your reply. True only when the request is fully fixed in code/MR metadata, or the branch was verified to already satisfy it and the public comment explains how. For merge-conflict feedback, true only after a pushed branch merges cleanly with no conflict markers. False for partial progress, disagreement, or anything still needing review. Omit to let {d} infer from branch changes; set explicitly for metadata-only fixes.",
                            d = display_name()),
                        ),
                    )
                    .property(
                        "post_plain_comment",
                        Schema::boolean(
                            "Whether to post a new plain (non-resolvable) merge request comment with `public_comment`. True only when a plain MR comment needs a new public reply; omit or use false when no reply is needed or it would only repeat that no changes were necessary.",
                        ),
                    ))
                ),
                "cannot_resolve" => (
                    format!("The feedback cannot be resolved autonomously; {d} abandons the merge request and reports back.",
                    d = display_name()),
                    fields(blocked_properties(
                        ObjectSchema::new(),
                        "Why the feedback cannot be resolved and what human input is needed."
                    ))
                ),
            }
        );
    /// Tolerated: an outcome spelled with different case or padding, and the
    /// comment-control booleans sent as `"true"`/`"false"` strings.
        normalize(value) {
            compat::normalize_tag(value, "outcome");
            compat::normalize_bool(value, "mark_discussions_resolved");
            compat::normalize_bool(value, "post_plain_comment");
        }
    }
}

#[derive(Debug, Clone)]
struct WorkerConfig {
    poll_interval: Duration,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct WorkerAgentSettings {
    #[serde(
        default = "default_worker_poll_interval",
        deserialize_with = "crate::core::config::duration::deserialize"
    )]
    poll_interval: Duration,
    poll_interval_secs: Option<u64>,
}

fn default_worker_poll_interval() -> Duration {
    Duration::from_secs(60)
}

/// The single issue a worker is pinned to for its full lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveIssue {
    issue_iid: u64,
    mr_iid: Option<u64>,
    branch_name: Option<String>,
    mr_created: bool,
}

/// A borrowing view over the [`AgentWorkspace`] fields the worker cycle
/// needs. Built fresh from `&AgentWorkspace` at each use site rather
/// than stored, so the worker never owns a second `GitRepo`/forge client
/// — and, since it is never stored alongside the runtime it borrows from,
/// it can't become self-referential.
struct AgentState<'a> {
    project_name: &'a str,
    agent_id: &'a str,
    sessions_dir: &'a str,

    git_repo: &'a GitRepo,
    forge: &'a Arc<dyn ForgeClient>,
}

impl AgentState<'_> {
    fn from_runtime(runtime: &AgentWorkspace) -> AgentState<'_> {
        AgentState {
            project_name: &runtime.project_name,
            agent_id: &runtime.agent_id,
            sessions_dir: &runtime.sessions_dir,
            git_repo: &runtime.git_repo,
            forge: &runtime.forge,
        }
    }

    fn session_file_path(&self, issue_iid: u64) -> std::path::PathBuf {
        Path::new(&self.sessions_dir).join(format!("{}_issue_{}.json", &self.agent_id, issue_iid))
    }

    fn session_store(&self, issue_iid: u64) -> StateStore<SessionFile> {
        StateStore::new(self.session_file_path(issue_iid))
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

    fn release_worker_hold_pending_gitlab_only(&self, issue_iid: u64) -> bool {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (no close)",
            &self.agent_id, issue_iid, WORKER_PENDING_LABEL
        );

        if let Err(error) = claim::release(
            self.forge.as_ref(),
            ClaimResource::Issue(issue_iid),
            self.agent_id,
        ) {
            warn!(
                "{}: Failed to release claim on issue #{}: {}",
                self.agent_id, issue_iid, error
            );
            return false;
        }
        let _ = self.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);
        self.cleanup_session(issue_iid);
        true
    }

    fn release_worker_hold_review_only(&self, issue_iid: u64) -> bool {
        info!(
            "{}: Issue #{} has `{}` — releasing claim and session (review only)",
            &self.agent_id, issue_iid, WORKER_REVIEW_ONLY_LABEL
        );

        self.clear_resumed_issue_state(issue_iid)
    }

    fn clear_resumed_issue_state(&self, issue_iid: u64) -> bool {
        if let Err(error) = claim::release(
            self.forge.as_ref(),
            ClaimResource::Issue(issue_iid),
            self.agent_id,
        ) {
            warn!(
                "{}: Failed to release claim on issue #{}: {}",
                self.agent_id, issue_iid, error
            );
            return false;
        }
        let default_branch = self
            .git_repo
            .get_default_branch()
            .unwrap_or("main".to_string());

        let branch = format!("issue-{}", issue_iid);

        let _ = self.git_repo.reset_hard();
        let _ = self.git_repo.checkout_remote_branch(&default_branch);
        let _ = self.git_repo.delete_local_branch(&branch);

        let _ = self.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
        true
    }

    fn abandon_closed_issue(&self, issue_iid: u64, mr_iid: Option<u64>) -> bool {
        info!(
            "{}: Issue #{} was closed externally, abandoning work",
            &self.agent_id, issue_iid
        );

        if let Err(error) = claim::release(
            self.forge.as_ref(),
            ClaimResource::Issue(issue_iid),
            self.agent_id,
        ) {
            warn!(
                "{}: Failed to release claim on closed issue #{}: {}",
                self.agent_id, issue_iid, error
            );
            return false;
        }

        if let Some(mr) = mr_iid {
            let _ = self
                .forge
                .add_mr_comment(mr, "Closing this MR — the linked issue has been closed.");
            let _ = self.forge.close_mr(mr);
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
        let _ = self.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);

        self.cleanup_session(issue_iid);
        true
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
    runtime: AgentWorkspace,
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
        repo_banner(config, banner);
    }

    fn validate_settings(
        config: &Config,
        _section: &AgentSection,
        settings: &Self::Settings,
    ) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_repo_url()?;
        anyhow::ensure!(
            settings.poll_interval_secs.is_none(),
            "poll_interval_secs was replaced by poll_interval for [agent.worker]"
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
                let scope = scope_label_filter(&self.runtime.scope_label);
                let model = &self.runtime.model;
                let shutdown = Arc::clone(model.shutdown());
                let state = AgentState::from_runtime(&self.runtime);
                worker_cycle(&state, model, &mut self.active, &shutdown, scope)
            }
            _ => Ok(()),
        }
    }

    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
        let runtime = AgentBootstrap::new(&ctx, ModelPreferences::default()).build()?;
        let settings = ctx.settings;
        let config = WorkerConfig {
            poll_interval: settings.poll_interval,
        };
        let scope = scope_label_filter(&runtime.scope_label);
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

    let issue = match state.forge.get_issue(active_issue.issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            if is_not_found(&e) {
                info!(
                    "{}: Resumed issue #{} no longer exists (404), dropping resume state",
                    &state.agent_id, active_issue.issue_iid
                );
                state.abandon_closed_issue(active_issue.issue_iid, active_issue.mr_iid);
                return None;
            }
            warn!(
                "{}: Failed to verify resumed issue #{}: {}, retaining resume state for retry",
                &state.agent_id, active_issue.issue_iid, e
            );
            return Some(active_issue);
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
        forge::issue_labels_in_scope(&self.labels, scope_label)
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

/// Result of a claim attempt, mirroring [`ClaimAcquireOutcome`] without the
/// lease: the lease itself lives in the port, which owns claim effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IssueClaimAttempt {
    Won,
    Lost,
    Interrupted,
}

/// The narrow, typed surface used by the worker routing workflows.
trait WorkerRoutingPort {
    fn shutdown_requested(&self) -> bool;
    fn issue(&self, issue_iid: u64) -> Result<IssueObservation>;
    fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation>;
    fn issues(&self) -> Result<Vec<IssueObservation>>;
    fn default_branch_or_main(&self) -> String;
    fn reset_worktree(&mut self);
    fn checkout_branch(&mut self, branch: &str);
    fn delete_local_branch(&mut self, branch: &str);
    fn delete_remote_branch(&mut self, branch: &str);
    fn release_issue_claim(&mut self, issue_iid: u64) -> bool;
    fn remove_working_on_label(&mut self, issue_iid: u64);
    fn remove_issue_label(&mut self, issue_iid: u64, label: &str);
    fn cleanup_session(&mut self, issue_iid: u64);
    fn save_session(&mut self, issue_iid: u64, mr_iid: u64);
    fn close_issue(&mut self, issue_iid: u64);
    fn acquire_issue_claim(&mut self, issue_iid: u64) -> Result<IssueClaimAttempt>;
    fn preserve_issue_claim(&mut self, issue_iid: u64);
    fn release_acquired_claim(&mut self, issue_iid: u64);
    fn clear_issue_state(&mut self, issue_iid: u64) -> bool;
    fn release_review_only_hold(&mut self, issue_iid: u64) -> bool;
    fn abandon_closed_issue(&mut self, issue_iid: u64, mr_iid: Option<u64>) -> bool;
    fn adopt_orphaned_session(&mut self) -> Option<ActiveIssue>;
    fn handle_need_ai_worker_mr(&mut self) -> Result<bool>;
    fn run_implementation(
        &mut self,
        issue: &IssueObservation,
    ) -> (ActiveIssue, Option<anyhow::Error>);
    fn run_feedback(
        &mut self,
        mr_iid: u64,
        linked_issue_iid: Option<u64>,
        comments_only: bool,
    ) -> Result<bool>;
    fn resolve_cancelled_issue(&mut self, issue_iid: u64) -> bool;
    fn issue_trackable(&mut self, issue_iid: u64) -> bool;
}

// ---------------------------------------------------------------------------
// Pure routing decisions
// ---------------------------------------------------------------------------

/// Whether the worker still owns the issue it is pinned to, and how it lets
/// go when it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IssueHold {
    Keep,
    AbandonClosed,
    ClearState(ClearReason),
    ReleaseReviewOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClearReason {
    OutOfScope,
    Pending,
    DoNotImplement,
}

fn decide_issue_hold(issue: &IssueObservation, scope_label: Option<&str>) -> IssueHold {
    if !issue.is_open() {
        return IssueHold::AbandonClosed;
    }
    if !issue.in_scope(scope_label) {
        return IssueHold::ClearState(ClearReason::OutOfScope);
    }
    if issue_has_do_not_implement_label(&issue.labels) {
        return IssueHold::ClearState(ClearReason::DoNotImplement);
    }
    if issue_has_worker_pending_label(&issue.labels) {
        return IssueHold::ClearState(ClearReason::Pending);
    }
    if issue_has_worker_review_only_label(&issue.labels) {
        return IssueHold::ReleaseReviewOnly;
    }
    IssueHold::Keep
}

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

fn claimed_issue_is_still_eligible(issue: &IssueObservation, scope_label: Option<&str>) -> bool {
    issue.state == "opened" && issue.in_scope(scope_label) && !should_skip_issue(issue)
}

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

fn apply_active_issue_hold(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    scope_label: Option<&str>,
    active: &mut Option<ActiveIssue>,
    watching_mr: bool,
) -> bool {
    let tracked = active
        .as_ref()
        .expect("active issue validation requires an active issue")
        .clone();
    let issue = match port.issue(tracked.issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            if watching_mr {
                warn!(
                    "{}: Failed to verify active issue #{}: {}, retaining worker state for retry",
                    agent_id, tracked.issue_iid, e
                );
            } else {
                warn!(
                    "{}: Failed to verify active issue #{}: {}, retaining claim for retry",
                    agent_id, tracked.issue_iid, e
                );
            }
            return false;
        }
    };

    match decide_issue_hold(&issue, scope_label) {
        IssueHold::Keep => true,
        IssueHold::AbandonClosed => {
            if port.abandon_closed_issue(
                tracked.issue_iid,
                watching_mr.then_some(tracked.mr_iid).flatten(),
            ) {
                *active = None;
            }
            false
        }
        IssueHold::ClearState(reason) => {
            match reason {
                ClearReason::OutOfScope if watching_mr => info!(
                    "{}: Issue #{} left scope label {:?}, releasing worker state",
                    agent_id, tracked.issue_iid, scope_label
                ),
                ClearReason::OutOfScope => info!(
                    "{}: Active issue #{} left scope label {:?}, releasing",
                    agent_id, tracked.issue_iid, scope_label
                ),
                ClearReason::Pending if watching_mr => info!(
                    "{}: Issue #{} has `{}` — stopping MR watch (issue stays open)",
                    agent_id, tracked.issue_iid, WORKER_PENDING_LABEL
                ),
                ClearReason::Pending => info!(
                    "{}: Active issue #{} has `{}` — yielding (issue stays open)",
                    agent_id, tracked.issue_iid, WORKER_PENDING_LABEL
                ),
                ClearReason::DoNotImplement => info!(
                    "{}: Active issue #{} has `{}` — releasing without implementation",
                    agent_id, tracked.issue_iid, DO_NOT_IMPLEMENT
                ),
            }
            if port.clear_issue_state(tracked.issue_iid) {
                *active = None;
            }
            false
        }
        IssueHold::ReleaseReviewOnly => {
            if port.release_review_only_hold(tracked.issue_iid) {
                *active = None;
            }
            false
        }
    }
}

fn cleanup_finished_mr(port: &mut dyn WorkerRoutingPort, issue_iid: u64, merged: bool) -> bool {
    if !port.release_issue_claim(issue_iid) {
        return false;
    }
    let branch = format!("issue-{issue_iid}");
    let default_branch = port.default_branch_or_main();
    port.reset_worktree();
    port.checkout_branch(&default_branch);
    port.delete_local_branch(&branch);
    if merged {
        port.delete_remote_branch(&branch);
    }
    port.remove_working_on_label(issue_iid);
    port.cleanup_session(issue_iid);
    if merged {
        port.close_issue(issue_iid);
    }
    true
}

fn cleanup_implementation(
    port: &mut dyn WorkerRoutingPort,
    issue_iid: u64,
    remove_working_label: bool,
    cleanup_session: bool,
    branch: Option<&str>,
) -> bool {
    if !port.release_issue_claim(issue_iid) {
        return false;
    }
    if remove_working_label {
        port.remove_working_on_label(issue_iid);
    }
    if cleanup_session {
        port.cleanup_session(issue_iid);
    }
    if let Some(branch) = branch {
        let default_branch = port.default_branch_or_main();
        port.reset_worktree();
        port.checkout_branch(&default_branch);
        port.delete_local_branch(branch);
    }
    true
}

/// Returns whether routing should continue by looking for new work.
fn handle_active_mr(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    scope_label: Option<&str>,
    active: &mut Option<ActiveIssue>,
) -> Result<bool> {
    if !apply_active_issue_hold(port, agent_id, scope_label, active, true) {
        return Ok(true);
    }
    let tracked = active.as_ref().unwrap().clone();
    let mr_iid = tracked.mr_iid.expect("active MR workflow requires an MR");
    let mr = match port.merge_request_status(mr_iid) {
        Ok(mr) => mr,
        Err(e) => {
            warn!("{}: Failed to check MR !{}: {}", agent_id, mr_iid, e);
            return Ok(false);
        }
    };
    if mr.is_finished() {
        info!(
            "{}: MR !{} is {}, releasing issue #{}",
            agent_id, mr.iid, mr.state, tracked.issue_iid
        );
        if cleanup_finished_mr(port, tracked.issue_iid, mr.is_merged()) {
            *active = None;
            return Ok(true);
        }
        return Ok(false);
    }

    match port.run_feedback(mr_iid, Some(tracked.issue_iid), false) {
        Ok(false) => Ok(false),
        Ok(true) => {
            info!(
                "{}: Issue #{} abandoned, MR !{} closed",
                agent_id, tracked.issue_iid, mr_iid
            );
            if port.release_issue_claim(tracked.issue_iid) {
                port.cleanup_session(tracked.issue_iid);
                *active = None;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(e) => {
            let message = format!("{e:#}");
            if port.shutdown_requested() {
                return Ok(false);
            }
            if message.contains(WORKER_AGENT_CANCELLED_MSG)
                && port.resolve_cancelled_issue(tracked.issue_iid)
            {
                *active = None;
                return Ok(false);
            }
            error!(
                "{}: Failed to handle comments for MR !{}: {}",
                agent_id, mr_iid, message
            );
            Ok(false)
        }
    }
}

fn finish_implementation(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    active: &mut Option<ActiveIssue>,
    issue: &IssueObservation,
    reattempt: bool,
) {
    let (current, run_error) = port.run_implementation(issue);
    if let Some(e) = run_error {
        let message = format!("{e:#}");
        if port.shutdown_requested() {
            *active = Some(current);
            return;
        }
        if message.contains(WORKER_AGENT_CANCELLED_MSG)
            && port.resolve_cancelled_issue(current.issue_iid)
        {
            *active = None;
            return;
        }
        if reattempt {
            error!(
                "{}: Failed to re-process issue #{}: {}",
                agent_id, current.issue_iid, message
            );
        } else {
            error!(
                "{}: Failed to process issue #{}: {}",
                agent_id, current.issue_iid, message
            );
        }
        if cleanup_implementation(
            port,
            current.issue_iid,
            true,
            reattempt,
            current.branch_name.as_deref(),
        ) {
            *active = None;
        } else {
            *active = Some(current);
        }
        return;
    }

    if current.mr_created {
        *active = port.issue_trackable(current.issue_iid).then_some(current);
    } else if cleanup_implementation(port, issue.iid, false, reattempt, None) {
        *active = None;
    } else {
        *active = Some(current);
    }
}

/// Returns whether routing should continue by looking for new work.
fn handle_active_reattempt(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    scope_label: Option<&str>,
    active: &mut Option<ActiveIssue>,
) -> bool {
    if !apply_active_issue_hold(port, agent_id, scope_label, active, false) {
        return true;
    }
    let issue_iid = active.as_ref().unwrap().issue_iid;
    info!(
        "{}: Active issue #{} has no MR, re-attempting implementation",
        agent_id, issue_iid
    );
    let issue = match port.issue(issue_iid) {
        Ok(issue) => issue,
        Err(e) => {
            warn!(
                "{}: Failed to fetch issue #{} for re-attempt: {}, retaining claim for retry",
                agent_id, issue_iid, e
            );
            return false;
        }
    };
    finish_implementation(port, agent_id, active, &issue, true);
    false
}

fn poll_for_work(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    scope_label: Option<&str>,
    active: &mut Option<ActiveIssue>,
) -> Result<()> {
    if port.shutdown_requested() {
        return Ok(());
    }
    if active.is_none()
        && let Some(adopted) = port.adopt_orphaned_session()
    {
        info!(
            "{}: Adopted orphaned issue #{} with MR !{}",
            agent_id,
            adopted.issue_iid,
            adopted.mr_iid.unwrap_or(0)
        );
        *active = Some(adopted);
        return Ok(());
    }
    if port.handle_need_ai_worker_mr()? {
        return Ok(());
    }
    let issues = port.issues()?;
    if port.shutdown_requested() {
        return Ok(());
    }

    for issue in issues {
        if port.shutdown_requested() {
            return Ok(());
        }
        match screen_issue_candidate(&issue, scope_label) {
            CandidateScreening::Skip => continue,
            CandidateScreening::AlreadyClaimed => {
                debug!(
                    "{}: Issue #{} already claimed, skipping",
                    agent_id, issue.iid
                );
                continue;
            }
            CandidateScreening::WaitingOnIssue(dep_issue_iid) => {
                let dependency = port.issue(dep_issue_iid);
                match decide_dependency(dependency.as_ref()) {
                    DependencyDecision::Resolved => {
                        if dependency.is_ok() {
                            info!(
                                "{}: Issue #{} dependency issue #{} closed, resuming",
                                agent_id, issue.iid, dep_issue_iid
                            );
                        } else {
                            info!(
                                "{}: Issue #{} dependency issue #{} not found (deleted or never existed), dropping dependency label and resuming",
                                agent_id, issue.iid, dep_issue_iid
                            );
                        }
                        port.remove_issue_label(
                            issue.iid,
                            &format!("{WAITING_ON_ISSUE_LABEL_PREFIX}{dep_issue_iid}"),
                        );
                        if claim::is_claimed(&issue.labels) {
                            debug!(
                                "{}: Issue #{} already claimed, skipping",
                                agent_id, issue.iid
                            );
                            continue;
                        }
                    }
                    DependencyDecision::StillWaiting => {
                        debug!(
                            "{}: Issue #{} waiting on issue #{} (not yet closed), skipping",
                            agent_id, issue.iid, dep_issue_iid
                        );
                        continue;
                    }
                    DependencyDecision::Unknown => {
                        warn!(
                            "{}: Issue #{} waiting on issue #{} — failed to check dependency state: {}, skipping this cycle",
                            agent_id,
                            issue.iid,
                            dep_issue_iid,
                            dependency.err().map(|e| e.to_string()).unwrap_or_default()
                        );
                        continue;
                    }
                }
            }
            CandidateScreening::Claimable => {}
        }

        match port.acquire_issue_claim(issue.iid)? {
            IssueClaimAttempt::Lost => {
                info!(
                    "{}: Failed to claim issue #{}, skipping",
                    agent_id, issue.iid
                );
                continue;
            }
            IssueClaimAttempt::Interrupted => return Ok(()),
            IssueClaimAttempt::Won => {}
        }
        if port.shutdown_requested() {
            port.release_acquired_claim(issue.iid);
            return Ok(());
        }
        let claimed_issue = match port.issue(issue.iid) {
            Ok(claimed_issue) if claimed_issue_is_still_eligible(&claimed_issue, scope_label) => {
                claimed_issue
            }
            Ok(_) => {
                info!(
                    "{}: Issue #{} became ineligible while being claimed, releasing it",
                    agent_id, issue.iid
                );
                port.release_acquired_claim(issue.iid);
                continue;
            }
            Err(error) => {
                warn!(
                    "{}: Failed to revalidate issue #{} after claiming: {}, releasing it",
                    agent_id, issue.iid, error
                );
                port.release_acquired_claim(issue.iid);
                continue;
            }
        };
        port.preserve_issue_claim(issue.iid);
        port.save_session(issue.iid, 0);
        info!(
            "{}: Implementing issue #{}: {}",
            agent_id, claimed_issue.iid, claimed_issue.title
        );
        finish_implementation(port, agent_id, active, &claimed_issue, false);
        return Ok(());
    }
    Ok(())
}

fn run_worker_routing_cycle(
    port: &mut dyn WorkerRoutingPort,
    agent_id: &str,
    scope_label: Option<&str>,
    active: &mut Option<ActiveIssue>,
) -> Result<()> {
    let continue_polling = match active.as_ref() {
        Some(active_issue) if active_issue.mr_iid.is_some() => {
            handle_active_mr(port, agent_id, scope_label, active)?
        }
        Some(active_issue) if !active_issue.mr_created => {
            handle_active_reattempt(port, agent_id, scope_label, active)
        }
        _ => true,
    };
    if continue_polling {
        poll_for_work(port, agent_id, scope_label, active)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Live worker routing port
// ---------------------------------------------------------------------------

struct LiveWorkerRoutingPort<'a> {
    state: &'a AgentState<'a>,
    model: &'a AgentModel,
    shutdown: &'a AtomicBool,
    scope_label: Option<&'a str>,
    candidate_lease: Option<ClaimLease>,
}

impl WorkerRoutingPort for LiveWorkerRoutingPort<'_> {
    fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    fn issue(&self, issue_iid: u64) -> Result<IssueObservation> {
        Ok(IssueObservation::from_issue(
            &self.state.forge.get_issue(issue_iid)?,
        ))
    }

    fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation> {
        let mr = self.state.forge.get_merge_request(mr_iid)?;
        Ok(MrStatusObservation {
            iid: mr.iid,
            state: mr.state,
        })
    }

    fn issues(&self) -> Result<Vec<IssueObservation>> {
        Ok(self
            .state
            .forge
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

    fn reset_worktree(&mut self) {
        let _ = self.state.git_repo.reset_hard();
    }
    fn checkout_branch(&mut self, branch: &str) {
        let _ = self.state.git_repo.checkout_remote_branch(branch);
    }
    fn delete_local_branch(&mut self, branch: &str) {
        let _ = self.state.git_repo.delete_local_branch(branch);
    }
    fn delete_remote_branch(&mut self, branch: &str) {
        self.state.git_repo.delete_remote_branch_best_effort(branch);
    }
    fn release_issue_claim(&mut self, issue_iid: u64) -> bool {
        claim::release(
            self.state.forge.as_ref(),
            ClaimResource::Issue(issue_iid),
            self.state.agent_id,
        )
        .map_err(|error| {
            warn!(
                "{}: Failed to release claim on issue #{}: {}",
                self.state.agent_id, issue_iid, error
            );
        })
        .is_ok()
    }
    fn remove_working_on_label(&mut self, issue_iid: u64) {
        let _ = self
            .state
            .forge
            .remove_issue_label(issue_iid, WORKING_ON_LABEL);
    }
    fn remove_issue_label(&mut self, issue_iid: u64, label: &str) {
        let _ = self.state.forge.remove_issue_label(issue_iid, label);
    }
    fn cleanup_session(&mut self, issue_iid: u64) {
        self.state.cleanup_session(issue_iid);
    }
    fn save_session(&mut self, issue_iid: u64, mr_iid: u64) {
        let _ = self.state.save_session(issue_iid, mr_iid);
    }
    fn close_issue(&mut self, issue_iid: u64) {
        close_issue_best_effort(self.state.forge.as_ref(), issue_iid);
    }

    fn acquire_issue_claim(&mut self, issue_iid: u64) -> Result<IssueClaimAttempt> {
        Ok(
            match claim::acquire(
                self.state.forge.as_ref(),
                ClaimResource::Issue(issue_iid),
                self.state.agent_id,
                self.shutdown,
            )? {
                ClaimAcquireOutcome::Won(lease) => {
                    self.candidate_lease = Some(lease);
                    IssueClaimAttempt::Won
                }
                ClaimAcquireOutcome::Lost => IssueClaimAttempt::Lost,
                ClaimAcquireOutcome::Interrupted => IssueClaimAttempt::Interrupted,
            },
        )
    }

    fn preserve_issue_claim(&mut self, _issue_iid: u64) {
        if let Some(lease) = self.candidate_lease.take() {
            lease.preserve();
        }
    }
    fn release_acquired_claim(&mut self, _issue_iid: u64) {
        let Some(lease) = self.candidate_lease.as_mut() else {
            return;
        };
        if lease.try_release(self.state.forge.as_ref()).is_ok() {
            self.candidate_lease = None;
        } else if let Some(lease) = self.candidate_lease.take() {
            lease.preserve();
        }
    }
    fn clear_issue_state(&mut self, issue_iid: u64) -> bool {
        self.state.clear_resumed_issue_state(issue_iid)
    }
    fn release_review_only_hold(&mut self, issue_iid: u64) -> bool {
        self.state.release_worker_hold_review_only(issue_iid)
    }
    fn abandon_closed_issue(&mut self, issue_iid: u64, mr_iid: Option<u64>) -> bool {
        self.state.abandon_closed_issue(issue_iid, mr_iid)
    }
    fn adopt_orphaned_session(&mut self) -> Option<ActiveIssue> {
        try_adopt_orphaned_session(self.state, self.shutdown, self.scope_label)
    }
    fn handle_need_ai_worker_mr(&mut self) -> Result<bool> {
        try_handle_need_ai_worker_mr(self.state, self.model, self.shutdown, self.scope_label)
    }
    fn run_implementation(
        &mut self,
        issue: &IssueObservation,
    ) -> (ActiveIssue, Option<anyhow::Error>) {
        let mut current = ActiveIssue {
            issue_iid: issue.iid,
            mr_iid: None,
            branch_name: None,
            mr_created: false,
        };
        let error = process_issue(
            self.state,
            self.model,
            issue,
            &mut current,
            self.scope_label,
        )
        .err();
        (current, error)
    }
    fn run_feedback(
        &mut self,
        mr_iid: u64,
        linked_issue_iid: Option<u64>,
        comments_only: bool,
    ) -> Result<bool> {
        handle_mr_comments(
            self.state,
            self.model,
            mr_iid,
            linked_issue_iid,
            comments_only,
        )
    }
    fn resolve_cancelled_issue(&mut self, issue_iid: u64) -> bool {
        handle_worker_issue_processing_cancelled(
            self.state,
            issue_iid,
            &anyhow::anyhow!("{}", WORKER_AGENT_CANCELLED_MSG),
        )
    }
    fn issue_trackable(&mut self, issue_iid: u64) -> bool {
        should_track_worker_issue(self.state, issue_iid)
    }
}

/// The worker's routing cycle: resume active work or imperatively look for one
/// new candidate.
fn worker_cycle(
    state: &AgentState,
    model: &AgentModel,
    active: &mut Option<ActiveIssue>,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut port = LiveWorkerRoutingPort {
        state,
        model,
        shutdown,
        scope_label,
        candidate_lease: None,
    };
    run_worker_routing_cycle(&mut port, state.agent_id, scope_label, active)
}

fn mr_has_label(mr: &MergeRequest, label: &str) -> bool {
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
    let mut mrs = state.forge.list_merge_requests()?;
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
        if !forge::mr_in_scope(&mr, scope_label) {
            continue;
        }
        let recovered_lease = mr.labels.as_deref().and_then(|labels| {
            ClaimLease::recover(ClaimResource::MergeRequest(mr.iid), state.agent_id, labels)
        });
        if claim::is_mr_claimed(&mr.labels) && recovered_lease.is_none() {
            continue;
        }
        let unresolved = state.forge.get_unresolved_discussion_ids(mr.iid)?;
        if unresolved.is_empty() {
            continue;
        }
        let mut lease = match recovered_lease {
            Some(lease) => lease,
            None => match claim::acquire(
                state.forge.as_ref(),
                ClaimResource::MergeRequest(mr.iid),
                state.agent_id,
                shutdown,
            )? {
                ClaimAcquireOutcome::Won(lease) => lease,
                ClaimAcquireOutcome::Lost => continue,
                ClaimAcquireOutcome::Interrupted => return Ok(false),
            },
        };
        if shutdown.load(Ordering::SeqCst) {
            if lease.try_release(state.forge.as_ref()).is_err() {
                lease.preserve();
            }
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
        let release_result = lease.try_release(state.forge.as_ref());
        if release_result.is_err() {
            lease.preserve();
        }
        result?;
        release_result?;
        return Ok(true);
    }
    Ok(false)
}

fn should_skip_issue(issue: &IssueObservation) -> bool {
    if issue.title.starts_with("[Draft]") || issue.title.starts_with("Draft:") {
        return true;
    }

    if issue_has_do_not_implement_label(&issue.labels) {
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

fn issue_has_do_not_implement_label(labels: &[String]) -> bool {
    labels.iter().any(|label| label == DO_NOT_IMPLEMENT)
}

fn issue_has_worker_pending_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_PENDING_LABEL.to_string())
}

fn issue_has_worker_review_only_label(labels: &[String]) -> bool {
    labels.contains(&WORKER_REVIEW_ONLY_LABEL.to_string())
}

fn worker_should_cancel_issue_processing(issue: &IssueObservation) -> bool {
    issue.state != "opened"
        || issue_has_do_not_implement_label(&issue.labels)
        || issue_has_worker_review_only_label(&issue.labels)
        || issue_has_worker_pending_label(&issue.labels)
}

fn worker_issue_cancel_check(
    forge: Arc<dyn ForgeClient>,
    issue_iid: u64,
) -> Arc<dyn Fn() -> bool + Send + Sync> {
    Arc::new(move || {
        forge.get_issue(issue_iid).ok().is_some_and(|issue| {
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

    match state.forge.get_issue(issue_iid) {
        Ok(issue) if issue_has_worker_pending_label(&issue.labels) => {
            info!(
                "{}: Issue #{} marked `{}` mid-run — releasing worker hold (issue stays open)",
                &state.agent_id, issue_iid, WORKER_PENDING_LABEL
            );
            // Leave the issue open and the `pending` label in place; just drop
            // our claim and session so another worker can pick it up once a
            // human removes `pending`.
            let _ = claim::release(
                state.forge.as_ref(),
                ClaimResource::Issue(issue_iid),
                state.agent_id,
            );
            let _ = state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);
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
            let _ = claim::release(
                state.forge.as_ref(),
                ClaimResource::Issue(issue_iid),
                state.agent_id,
            );
            let _ = state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);
            state.cleanup_session(issue_iid);
            true
        }
        Err(e) => {
            warn!(
                "{}: Cancelled while working on issue #{} but failed to re-fetch issue: {}",
                &state.agent_id, issue_iid, e
            );
            let _ = claim::release(
                state.forge.as_ref(),
                ClaimResource::Issue(issue_iid),
                state.agent_id,
            );
            state.cleanup_session(issue_iid);
            true
        }
    }
}

fn stop_worker_issue_if_review_only(state: &AgentState, issue_iid: u64) -> bool {
    let Ok(issue) = state.forge.get_issue(issue_iid) else {
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
        .forge
        .get_issue(issue_iid)
        .is_ok_and(|issue| issue_has_worker_review_only_label(&issue.labels))
    {
        state.release_worker_hold_review_only(issue_iid);
        return false;
    }

    true
}

fn close_issue_best_effort(forge: &dyn ForgeClient, issue_iid: u64) {
    if let Err(e) = forge.close_issue(issue_iid) {
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
fn closes_keyword_mr_status(forge: &dyn ForgeClient, issue_iid: u64) -> Option<ClosesLinkedMr> {
    let mrs = forge.list_merge_requests().ok()?;
    let mut opens: Vec<u64> = Vec::new();
    let mut any_merged = false;
    for mr in mrs {
        if !mr_description_closes_issue(&mr.description, issue_iid) {
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
    forge: &dyn ForgeClient,
    issue_iid: u64,
    session_mr_iid: u64,
) -> ResolvedTrackedMr {
    match closes_keyword_mr_status(forge, issue_iid) {
        Some(ClosesLinkedMr::Open(id)) => return ResolvedTrackedMr::Track(id),
        Some(ClosesLinkedMr::Merged) => return ResolvedTrackedMr::MergedCloseIssue,
        None => {}
    }
    if session_mr_iid > 0
        && let Ok(mr) = forge.get_merge_request(session_mr_iid)
        && mr.state == "opened"
    {
        return ResolvedTrackedMr::Track(session_mr_iid);
    }
    if let Some(id) = find_open_mr_for_issue(forge, issue_iid) {
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

/// How one model invocation ended.
enum ImplModelResult {
    Output(Box<WorkerImplementationOutput>),
    /// An external cancel that the cancel helper resolved as an
    /// intentional stop.
    Cancelled,
}

/// The narrow surface one implementation run needs: the worktree, the
/// issue's merge requests, the session store, and the model. Object-safe
/// and role-local.
trait ImplementationPort {
    fn closes_linked_mr(&self) -> Option<ClosesLinkedMr>;
    fn open_mr_for_issue(&self) -> Option<u64>;
    fn stop_if_review_only(&mut self) -> bool;
    fn add_working_on_label(&mut self);
    fn require_working_on_label(&mut self) -> Result<()>;
    fn remove_working_on_label(&mut self);
    fn add_issue_label(&mut self, label: &str);
    fn add_issue_comment(&mut self, body: &str);
    fn release_issue_claim(&mut self) -> Result<()>;
    fn cleanup_session(&mut self);
    fn close_issue(&mut self);
    fn save_session(&mut self, mr_iid: u64);
    fn require_save_session(&mut self, mr_iid: u64) -> Result<()>;
    fn require_save_session_with_summary(&mut self, mr_iid: u64, summary: &str) -> Result<()>;
    fn default_branch(&self) -> Result<String>;
    fn default_branch_or_main(&self) -> String;
    fn fetch_remote(&mut self) -> Result<()>;
    fn reset_worktree(&mut self);
    fn remote_branch_exists(&self, branch: &str) -> Result<bool>;
    fn checkout_branch(&mut self, branch: &str) -> Result<()>;
    fn checkout_branch_best_effort(&mut self, branch: &str);
    fn delete_local_branch(&mut self, branch: &str);
    fn delete_remote_branch(&mut self, branch: &str);
    fn create_branch_from(&mut self, branch: &str, base: &str) -> Result<()>;
    fn merge_base_into_branch(&mut self, base: &str) -> Result<bool>;
    fn has_diff_against(&self, base: &str) -> Result<bool>;
    fn has_staged_changes(&self) -> Result<bool>;
    fn merge_request_state(&self, mr_iid: u64) -> Option<String>;
    fn dependency_closed(&self, issue_iid: u64) -> bool;
    fn issue_comments(&self) -> String;
    fn build_prompt(&mut self, continuation: bool, comments: &str) -> Result<String>;
    fn invoke_implementation_model(&mut self, prompt: &str) -> Result<ImplModelResult>;
    fn nudge_implementation_model(&mut self) -> Result<ImplModelResult>;
    fn stage_all(&mut self) -> Result<()>;
    fn commit(&mut self, message: &str) -> Result<()>;
    fn push_branch(&mut self, branch: &str) -> Result<()>;
    fn create_merge_request(
        &mut self,
        branch: &str,
        base: &str,
        title: &str,
        description: &str,
    ) -> Result<u64>;
    fn add_mr_scope_label(&mut self, mr_iid: u64);
    fn hand_issue_back_to_humans(&mut self, branch: &str, reason: &str) -> Result<()>;
}

#[derive(Default)]
struct ImplementationCycleResult {
    tracked_mr: Option<u64>,
    mr_created: bool,
    left_branch: Option<String>,
}

fn run_implementation_cycle(
    port: &mut dyn ImplementationPort,
    agent_id: &str,
    scope_label: Option<&str>,
    issue: &IssueObservation,
    result: &mut ImplementationCycleResult,
) -> Result<()> {
    let work = crate::ui::WorkTimer::start();
    if port.stop_if_review_only() {
        return Ok(());
    }

    match port.closes_linked_mr() {
        Some(ClosesLinkedMr::Open(mr_iid)) => {
            info!(
                "Issue #{} has open MR !{} (linked via Closes #{}), tracking it",
                issue.iid, mr_iid, issue.iid
            );
            result.tracked_mr = Some(mr_iid);
            result.mr_created = true;
            port.add_working_on_label();
            port.save_session(mr_iid);
            return Ok(());
        }
        Some(ClosesLinkedMr::Merged) => {
            info!(
                "Issue #{}: merged MR already references it via Closes #; closing issue",
                issue.iid
            );
            port.close_issue();
            port.cleanup_session();
            return Ok(());
        }
        None => {}
    }

    if let Some(mr_iid) = port.open_mr_for_issue() {
        info!(
            "Issue #{} already has open MR !{}, tracking it",
            issue.iid, mr_iid
        );
        result.tracked_mr = Some(mr_iid);
        result.mr_created = true;
        port.add_working_on_label();
        port.require_save_session(mr_iid)?;
        return Ok(());
    }

    let branch_name = format!("issue-{}", issue.iid);
    let default_branch = port.default_branch()?;
    port.fetch_remote()?;
    port.reset_worktree();
    let mut branch_existed = false;
    if port.remote_branch_exists(&branch_name)? {
        info!(
            "Branch {} already exists on remote, checking if it's stale",
            branch_name
        );
        port.checkout_branch(&branch_name)?;
        if port.has_diff_against(&default_branch)? {
            if port.merge_base_into_branch(&default_branch)? {
                branch_existed = true;
            } else {
                warn!(
                    "Branch {} has conflicts with {}, creating fresh branch instead",
                    branch_name, default_branch
                );
                port.reset_worktree();
                port.checkout_branch(&default_branch)?;
                port.delete_local_branch(&branch_name);
                port.create_branch_from(&branch_name, &default_branch)?;
            }
        } else {
            warn!(
                "Branch {} has no diff against {}, discarding stale branch",
                branch_name, default_branch
            );
            port.reset_worktree();
            port.checkout_branch(&default_branch)?;
            port.delete_local_branch(&branch_name);
            port.delete_remote_branch(&branch_name);
            port.create_branch_from(&branch_name, &default_branch)?;
        }
    } else {
        port.create_branch_from(&branch_name, &default_branch)?;
    }
    result.left_branch = Some(branch_name.clone());

    if port.stop_if_review_only() {
        return Ok(());
    }
    port.require_working_on_label()?;
    let comments = port.issue_comments();
    let prompt = port.build_prompt(branch_existed, &comments)?;
    let mut model_result = port.invoke_implementation_model(&prompt)?;
    let mut metadata = ImplementedMetadata::default();

    loop {
        let output = match model_result {
            ImplModelResult::Cancelled => return Ok(()),
            ImplModelResult::Output(output) => {
                info!(
                    "{}: Worker agent finished issue #{} (implemented for {})",
                    agent_id,
                    issue.iid,
                    crate::ui::format_work_duration(work.elapsed_seconds())
                );
                output
            }
        };

        match *output {
            WorkerImplementationOutput::Implemented(output_metadata) => {
                metadata = output_metadata;
            }
            WorkerImplementationOutput::ExistingMr { existing_mr_iid } => {
                match port.merge_request_state(existing_mr_iid).as_deref() {
                    Some("opened") => {
                        info!(
                            "Issue #{}: model identified existing MR !{} as the implementation; tracking it",
                            issue.iid, existing_mr_iid
                        );
                        port.reset_worktree();
                        let release_branch = port.default_branch_or_main();
                        port.checkout_branch_best_effort(&release_branch);
                        port.delete_local_branch(&branch_name);
                        result.tracked_mr = Some(existing_mr_iid);
                        result.mr_created = true;
                        port.require_save_session(existing_mr_iid)?;
                        port.add_working_on_label();
                        return Ok(());
                    }
                    Some(state) => warn!(
                        "Issue #{}: model identified MR !{} but it is not open (state={}); proceeding with new MR",
                        issue.iid, existing_mr_iid, state
                    ),
                    None => warn!(
                        "Issue #{}: model identified MR !{} but it could not be fetched; proceeding with new MR",
                        issue.iid, existing_mr_iid
                    ),
                }
            }
            WorkerImplementationOutput::NeedsSplit(blocked) => {
                warn!("Issue #{} is too broad, needs splitting", issue.iid);
                let reason = format!(
                    "This issue needs to be split into smaller, focused issues:\n\n{}",
                    extract_split_reason(&blocked)
                );
                port.hand_issue_back_to_humans(&branch_name, &reason)?;
                result.left_branch = None;
                return Ok(());
            }
            WorkerImplementationOutput::NeedsClarification(blocked) => {
                warn!("Issue #{} needs clarification", issue.iid);
                let reason = extract_clarification(&blocked);
                port.hand_issue_back_to_humans(&branch_name, &reason)?;
                result.left_branch = None;
                return Ok(());
            }
            WorkerImplementationOutput::CannotImplement(blocked) => {
                warn!("Issue #{} cannot be implemented", issue.iid);
                let reason = extract_cannot_implement_reason(&blocked);
                port.hand_issue_back_to_humans(&branch_name, &reason)?;
                result.left_branch = None;
                return Ok(());
            }
            WorkerImplementationOutput::WaitDependency { depends_on_issue } => {
                if !port.dependency_closed(depends_on_issue) {
                    port.add_issue_label(&waiting_on_issue_label(depends_on_issue));
                    port.remove_working_on_label();
                    port.add_issue_comment(&format!(
                        "Implementation cannot proceed until issue #{} is closed. \
                         Parking this issue until the dependency resolves.",
                        depends_on_issue
                    ));
                    port.release_issue_claim()?;
                    port.cleanup_session();
                    let release_branch = port.default_branch_or_main();
                    port.reset_worktree();
                    port.checkout_branch_best_effort(&release_branch);
                    port.delete_local_branch(&branch_name);
                    info!(
                        "{}: Issue #{} parked waiting on issue #{} (dependency open), released claim",
                        agent_id, issue.iid, depends_on_issue
                    );
                    result.mr_created = false;
                    result.left_branch = None;
                    return Ok(());
                }
            }
        }

        port.stage_all()?;
        if port.has_staged_changes()? {
            let title = extract_mr_title(metadata.mr_title.as_deref(), &issue.title);
            port.commit(&build_commit_message(&title, issue.iid))?;
        }
        if !port.has_diff_against(&default_branch)? {
            warn!(
                "Issue #{}: agent produced no code changes, nudging the current session",
                issue.iid
            );
            model_result = port.nudge_implementation_model()?;
            continue;
        }

        let title = extract_mr_title(metadata.mr_title.as_deref(), &issue.title);
        let summary = extract_mr_description(metadata.mr_description.as_deref());
        port.push_branch(&branch_name)?;
        let mr_iid = port.create_merge_request(
            &branch_name,
            &default_branch,
            &title,
            &format!("Closes #{}\n\n{}", issue.iid, summary),
        )?;
        result.tracked_mr = Some(mr_iid);
        result.mr_created = true;
        info!("Created MR !{} for issue #{}", mr_iid, issue.iid);
        if scope_label.is_some() {
            port.add_mr_scope_label(mr_iid);
        }
        port.require_save_session_with_summary(mr_iid, &summary)?;
        return Ok(());
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
                self.state.forge.clone(),
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
        closes_keyword_mr_status(self.state.forge.as_ref(), self.issue.iid)
    }

    fn open_mr_for_issue(&self) -> Option<u64> {
        find_open_mr_for_issue(self.state.forge.as_ref(), self.issue.iid)
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
            .forge
            .get_merge_request(mr_iid)
            .ok()
            .map(|mr| mr.state)
    }

    fn dependency_closed(&self, issue_iid: u64) -> bool {
        self.state
            .forge
            .get_issue(issue_iid)
            .map(|dep| dep.state == "closed")
            .unwrap_or(false)
    }

    fn issue_comments(&self) -> String {
        format_issue_comments_for_worker_context(self.state.forge.as_ref(), self.issue.iid)
    }

    fn stop_if_review_only(&mut self) -> bool {
        stop_worker_issue_if_review_only(self.state, self.issue.iid)
    }

    fn add_working_on_label(&mut self) {
        let _ = self
            .state
            .forge
            .add_issue_label(self.issue.iid, WORKING_ON_LABEL);
    }

    fn require_working_on_label(&mut self) -> Result<()> {
        self.state
            .forge
            .add_issue_label(self.issue.iid, WORKING_ON_LABEL)
    }

    fn remove_working_on_label(&mut self) {
        let _ = self
            .state
            .forge
            .remove_issue_label(self.issue.iid, WORKING_ON_LABEL);
    }

    fn add_issue_label(&mut self, label: &str) {
        let _ = self.state.forge.add_issue_label(self.issue.iid, label);
    }

    fn add_issue_comment(&mut self, body: &str) {
        let _ = self.state.forge.add_issue_comment(self.issue.iid, body);
    }

    fn release_issue_claim(&mut self) -> Result<()> {
        claim::release(
            self.state.forge.as_ref(),
            ClaimResource::Issue(self.issue.iid),
            self.state.agent_id,
        )
    }

    fn cleanup_session(&mut self) {
        self.state.cleanup_session(self.issue.iid);
    }

    fn close_issue(&mut self) {
        close_issue_best_effort(self.state.forge.as_ref(), self.issue.iid);
    }

    fn save_session(&mut self, mr_iid: u64) {
        let _ = self.state.save_session(self.issue.iid, mr_iid);
    }

    fn require_save_session(&mut self, mr_iid: u64) -> Result<()> {
        self.state.save_session(self.issue.iid, mr_iid)
    }

    fn require_save_session_with_summary(&mut self, mr_iid: u64, summary: &str) -> Result<()> {
        self.state
            .save_session_with_summary(self.issue.iid, mr_iid, summary)
    }

    fn fetch_remote(&mut self) -> Result<()> {
        self.state.git_repo.fetch()
    }

    fn reset_worktree(&mut self) {
        let _ = self.state.git_repo.reset_hard();
    }

    fn checkout_branch(&mut self, branch: &str) -> Result<()> {
        self.state.git_repo.checkout_remote_branch(branch)
    }

    fn checkout_branch_best_effort(&mut self, branch: &str) {
        let _ = self.state.git_repo.checkout_remote_branch(branch);
    }

    fn delete_local_branch(&mut self, branch: &str) {
        let _ = self.state.git_repo.delete_local_branch(branch);
    }

    fn delete_remote_branch(&mut self, branch: &str) {
        let _ = self.state.git_repo.delete_remote_branch(branch);
    }

    fn create_branch_from(&mut self, branch: &str, base: &str) -> Result<()> {
        self.state.git_repo.create_branch_from(branch, base)
    }

    fn merge_base_into_branch(&mut self, base: &str) -> Result<bool> {
        self.state.git_repo.try_merge(base)
    }

    fn build_prompt(&mut self, continuation: bool, comments: &str) -> Result<String> {
        if continuation {
            build_continuation_prompt(self.state, self.issue, comments)
        } else {
            build_implementation_prompt(self.state, self.issue, comments)
        }
    }

    fn invoke_implementation_model(&mut self, prompt: &str) -> Result<ImplModelResult> {
        LiveImplementationPort::invoke_implementation_model(self, Some(prompt))
    }

    fn nudge_implementation_model(&mut self) -> Result<ImplModelResult> {
        LiveImplementationPort::invoke_implementation_model(self, None)
    }

    fn stage_all(&mut self) -> Result<()> {
        self.state.git_repo.add_all()
    }

    fn commit(&mut self, message: &str) -> Result<()> {
        self.state.git_repo.commit(message)
    }

    fn push_branch(&mut self, branch: &str) -> Result<()> {
        self.state.git_repo.push(branch)
    }

    fn create_merge_request(
        &mut self,
        branch: &str,
        base: &str,
        title: &str,
        description: &str,
    ) -> Result<u64> {
        self.state
            .forge
            .create_merge_request(branch, base, title, description)
    }

    fn add_mr_scope_label(&mut self, mr_iid: u64) {
        if let Some(label) = self.scope_label
            && let Err(e) = self.state.forge.add_mr_label_with_retries(mr_iid, label)
        {
            warn!(
                "{}: Failed to add scope label {:?} to MR !{} (permanent error): {}",
                self.state.agent_id, label, mr_iid, e
            );
        }
    }

    fn hand_issue_back_to_humans(&mut self, branch: &str, reason: &str) -> Result<()> {
        hand_issue_back_to_humans(self.state, self.issue.iid, branch, reason)
    }
}

/// Implement one issue: adopt an existing merge request if the issue
/// already has one, otherwise prepare the branch, invoke the model, and
/// turn what it produced into a merge request.
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
    let mut port = LiveImplementationPort {
        state,
        model,
        issue,
        scope_label,
    };
    let mut cycle = ImplementationCycleResult::default();
    let result =
        run_implementation_cycle(&mut port, state.agent_id, scope_label, issue, &mut cycle);
    current.mr_iid = cycle.tracked_mr;
    current.mr_created = cycle.mr_created;
    current.branch_name = cycle.left_branch;
    result.map(|()| cycle.tracked_mr)
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
    state.forge.add_issue_comment(issue_iid, reason)?;

    if let Some(mr_iid) = find_open_mr_for_issue(state.forge.as_ref(), issue_iid) {
        state.forge.add_mr_comment(
            mr_iid,
            &format!(
                "Closing this MR — the issue cannot be implemented:\n\n{}",
                reason
            ),
        )?;
        let _ = state.forge.close_mr(mr_iid);
    }

    // Reset git to a clean state — keep remote branch for potential retry.
    let default_branch = state
        .git_repo
        .get_default_branch()
        .unwrap_or("main".to_string());
    let _ = state.git_repo.reset_hard();
    let _ = state.git_repo.checkout_remote_branch(&default_branch);
    let _ = state.git_repo.delete_local_branch(branch_name);

    state
        .forge
        .remove_issue_label(issue_iid, WORKING_ON_LABEL)?;
    state
        .forge
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

/// Collect new replies to the discussions currently being handled.
///
/// New top-level comments and replies to unrelated discussions advance the
/// observation cursor but are not injected into the active model session.
fn collect_new_follow_ups(
    comments: &[Comment],
    last_seen_id: &mut u64,
    mr_iid: u64,
    handled_discussion_ids: &HashSet<String>,
) -> Vec<String> {
    let mut new_msgs = Vec::new();
    for c in comments {
        if c.id > *last_seen_id && handled_discussion_ids.contains(&c.discussion_id) {
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
    let work = crate::ui::WorkTimer::start();
    let latest_mr = state.forge.get_merge_request(mr_iid)?;
    let unresolved_ids = state.forge.get_unresolved_discussion_ids(latest_mr.iid)?;
    let all_comments = state.forge.get_mr_comments(latest_mr.iid)?;
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
    let diff_context_content = build_mr_diff_context(
        state.project_name,
        &latest_mr,
        state.git_repo,
        state.forge.as_ref(),
    );

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
        .map(|n| load_issue_context(state.forge.as_ref(), n))
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
    // The task context (issue, summary, comments, full diff) goes to disk;
    // the prompt references it by path — the model reads what it needs.
    let combined_context_path = write_task_context_file(
        state.sessions_dir,
        &format!(
            "{}-mr-feedback-and-diff-{}.md",
            &state.agent_id, latest_mr.iid
        ),
        &combined_context_content,
    )?;

    let feedback_scope_rules = get_feedback_scope_rules();
    let prompt = format!(
        r#"SYSTEM: You are addressing reviewer feedback on a merge request in a fully automated, non-interactive environment.

PROJECT: {}

MERGE REQUEST !{}: {}

TASK CONTEXT FILE (open and read it first — it has the full picture):
{}

CRITICAL REQUIREMENTS:
- Leave staging, committing, pushing, and merge request creation to the system
- The task context file includes all comments and the diff — read it, then act directly
- Address all unresolved thread feedback and actionable plain MR comments autonomously
- Make all necessary code changes to resolve the comments
- Keep the original issue requirements in mind while addressing feedback
- If the workspace has merge conflict markers (<<<<<<< / ======= / >>>>>>>), resolve ALL of them before doing anything else. Edit each conflicted file to keep the correct version.
- The **Merge conflict status** section in the task context file is verified by {system}. Do NOT claim conflicts are fixed unless that section would be clean after your edits and you commit/push the resolution.
- `origin/{}` already fetched and merged it into your workspace when conflicts were reported. Edit the listed conflicted files, remove all conflict markers, and leave committing/pushing to {system}. Do not claim the conflict is fixed until the **Merge conflict status** section shows a clean merge with the fetched target tip.

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly.
2. Open and read the ENTIRE task context file at the path above before anything else. Review the "Unresolved MR comments to address", "Plain MR comments to consider", and "Full MR comment history for context" sections in it.
3. Use inline comment locations (`path:line` or `path:start-end`) from the comments to find the corresponding code and make targeted fixes. Grep for the relevant symbol, read only the surrounding lines, then edit.
4. First, check for merge conflicts using the **Merge conflict status** section and your workspace. If any exist, resolve ALL conflicts in every file, commit the resolution, and verify the target branch merges cleanly before claiming completion.
5. Focus on the **content** of each comment. Comments are always made by a reviewer (a user) — do not investigate who the author is, cross-reference their git history, or research their past commits or other MRs. The comment text is the instruction; act on it directly.
6. Identify which feedback items still need action. Treat comments in "Unresolved MR comments to address" as actionable threaded feedback. Also consider comments in "Plain MR comments to consider" actionable when they ask for changes, but remember they are plain MR comments and cannot be marked resolved.
7. Make the necessary code changes to address all unresolved threaded feedback and any actionable plain MR comments.
8. If the reviewer asked you to delete, rename, or move files, make those file changes.
9. Ensure changes align with both the original requirements and reviewer feedback.
10. If the reviewer says code changes are too large (above ~1500 lines total or ~500 non-test lines), you have TWO options:
   a) Adjust your implementation to reduce changed lines — simplify, remove unnecessary changes, trim scope
   b) If you cannot reasonably reduce the size, report that the feedback cannot be resolved autonomously
   Do NOT try to split the issue yourself — that is handled by the PMO agent, not you.
11. If the feedback cannot be resolved without additional human input (for example ambiguous requirements, out-of-scope requests, or missing information), report that clearly and identify the needed input.
12. Keep the MR title stable unless the reviewer explicitly asks for a title fix or the current title is clearly wrong for the whole MR.
13. Report only changes and metadata updates actually completed in this run; never imply a concern was fixed when the final branch does not fix it.
14. Before you finish, edit repo-root notes.md only if you can add lines that pass the **NOTES.MD** rules in your main worker instructions (same as implementation runs): **no** backticks, **no** file paths, **no** repo-specific symbol names, **no** code tours — and **no** bullets that merely **summarize what you did** this run in "timeless" wording (that still belongs in the MR, not notes). **No** lines about how to write notes or what notes are for. If nothing meets that bar, leave notes.md unchanged. Never copy notes.md into MR metadata or issue comments.
"#,
        &state.project_name,
        latest_mr.iid,
        latest_mr.title,
        combined_context_path,
        latest_mr.target_branch,
        feedback_scope_rules,
        system = display_name(),
    );

    // Follow-up poll: while the agent works on this MR's feedback, watch for
    // new comments added to the MR and forward them into the running session
    // as follow-up context (via session/inject on the potlatch harness backend).
    let seen_comment_id =
        std::sync::Mutex::new(all_comments.iter().map(|c| c.id).max().unwrap_or(0));
    let handled_discussion_ids: HashSet<String> = unresolved_ids.iter().cloned().collect();
    let glab_for_poll = state.forge.clone();
    let mr_iid_for_poll = latest_mr.iid;
    let follow_up_poll: Arc<dyn Fn() -> Vec<String> + Send + Sync> = Arc::new(move || {
        let Ok(comments) = glab_for_poll.get_mr_comments(mr_iid_for_poll) else {
            return Vec::new();
        };
        let mut last = seen_comment_id.lock().unwrap();
        collect_new_follow_ups(
            &comments,
            &mut last,
            mr_iid_for_poll,
            &handled_discussion_ids,
        )
    });

    let agent_output = if let Some(issue_iid) = issue_number {
        info!(
            "{}: Worker agent addressing MR !{} feedback for issue #{}",
            &state.agent_id, latest_mr.iid, issue_iid
        );
        let output = match model.complete_typed::<WorkerFeedbackOutput>(
            &prompt,
            &InvokeOptions {
                cancel_check: Some(worker_issue_cancel_check(state.forge.clone(), issue_iid)),
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
            "{}: Worker agent finished MR !{} feedback (addressed for {})",
            &state.agent_id,
            latest_mr.iid,
            crate::ui::format_work_duration(work.elapsed_seconds())
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
            "{}: Worker agent finished MR !{} feedback (addressed for {})",
            &state.agent_id,
            latest_mr.iid,
            crate::ui::format_work_duration(work.elapsed_seconds())
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
                state.forge.add_mr_comment(
                    latest_mr.iid,
                    &format!(
                        "Cannot resolve this MR feedback autonomously:\n\n{}",
                        reason
                    ),
                )?;
                let _ = state.forge.close_mr(latest_mr.iid);
            }

            return Ok(true);
        }
    };

    let input = FeedbackTailInput {
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
    };
    let mut port = LiveFeedbackTailPort {
        git_repo: state.git_repo,
        forge: state.forge.as_ref(),
        mr_iid: latest_mr.iid,
        source_branch: latest_mr.source_branch.clone(),
        target_branch: latest_mr.target_branch.clone(),
    };
    run_feedback_tail(&mut port, &input)?;

    Ok(false)
}

/// The MR fields the feedback tail compares before and after the model run:
/// a metadata-only edit must not imply that feedback was addressed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MrSurfaceObservation {
    title: String,
    description: String,
    labels: Option<Vec<String>>,
    has_conflicts: bool,
}

impl MrSurfaceObservation {
    fn from_mr(mr: &MergeRequest) -> Self {
        Self {
            title: mr.title.clone(),
            description: mr.description.clone(),
            labels: mr.labels.clone(),
            has_conflicts: mr.has_conflicts,
        }
    }

    fn differs_from(&self, other: &Self) -> bool {
        self.title.trim() != other.title.trim()
            || self.description.trim() != other.description.trim()
            || self.labels != other.labels
    }
}

/// Everything the pre-model half of a feedback run already learned.
#[derive(Debug, Clone)]
struct FeedbackTailInput {
    mr_iid: u64,
    source_branch: String,
    target_branch: String,
    pre_agent_sha: String,
    requires_conflict_resolution: bool,
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

/// Direct operations needed by the imperative feedback tail. Git operations
/// return errors; metadata and GitLab discussion writes are best effort.
trait FeedbackTailPort {
    fn update_mr_metadata(&mut self, title: &str, description: &str);
    fn fetch_branches(&mut self) -> Result<()>;
    fn has_changes_since(&self, base_ref: &str) -> Result<bool>;
    fn merge_in_progress(&self) -> Result<bool>;
    fn stage_resolved_conflicts(&mut self) -> Result<bool>;
    fn stage_all(&mut self) -> Result<()>;
    fn has_staged_changes(&self) -> Result<bool>;
    fn commit(&mut self, message: &str) -> Result<()>;
    fn complete_merge_if_ready(&mut self, message: &str) -> Result<bool>;
    fn merge_conflicts_present(&self) -> Result<bool>;
    fn up_to_date_with_target(&self, target_branch: &str) -> Result<bool>;
    fn diff_highlights(&self, base_ref: &str) -> Option<String>;
    fn push_source_branch(&mut self) -> Result<()>;
    fn merge_request_surface(&self, mr_iid: u64) -> Result<MrSurfaceObservation>;
    fn origin_head(&self, source_branch: &str) -> Option<String>;
    fn unresolved_discussion_ids(&self, mr_iid: u64) -> Vec<String>;
    fn reply_to_discussion(&mut self, discussion_id: &str, body: &str);
    fn resolve_discussion(&mut self, discussion_id: &str);
    fn post_plain_comment(&mut self, body: &str);
}

/// Finish a feedback run directly: metadata first, required git operations,
/// conflict gates, then best-effort GitLab replies.
fn run_feedback_tail(port: &mut dyn FeedbackTailPort, input: &FeedbackTailInput) -> Result<()> {
    if let Some(update) = plan_mr_metadata_update(
        &input.surface_before.title,
        &input.surface_before.description,
        &input.resolution,
    ) {
        port.update_mr_metadata(&update.title, &update.description);
    }

    port.fetch_branches()?;
    let mut has_new_changes = port.has_changes_since(&input.pre_agent_sha)?;

    if port.merge_in_progress()? {
        if port.stage_resolved_conflicts()? {
            info!(
                "MR !{}: staged merge-conflict files with no remaining conflict markers",
                input.mr_iid
            );
        }
        port.stage_all()?;
        has_new_changes = port.has_changes_since(&input.pre_agent_sha)?;
    } else if has_new_changes {
        port.stage_all()?;
        if port.has_staged_changes()? {
            port.commit(&input.commit_message())?;
        }
    }

    if port.complete_merge_if_ready(&input.merge_commit_message())? {
        has_new_changes = true;
        info!(
            "MR !{}: concluded in-progress merge with origin/{}",
            input.mr_iid, input.target_branch
        );
    }

    let mut conflicts_unresolved = port.merge_conflicts_present()?;
    if input.requires_conflict_resolution {
        port.fetch_branches()?;
        if port.up_to_date_with_target(&input.target_branch)? {
            if port.has_changes_since(&input.pre_agent_sha)? {
                has_new_changes = true;
                port.stage_all()?;
                if port.has_staged_changes()? {
                    port.commit(&input.merge_commit_message())?;
                }
            }
        } else {
            conflicts_unresolved = true;
            warn!(
                "MR !{}: branch still does not merge cleanly with origin/{} (fetched latest target and source)",
                input.mr_iid, input.target_branch
            );
        }
    }
    if !conflicts_unresolved {
        conflicts_unresolved = port.merge_conflicts_present()?;
    }

    let diff_highlights = has_new_changes
        .then(|| port.diff_highlights(&input.pre_agent_sha))
        .flatten();
    if has_new_changes && conflicts_unresolved {
        warn!(
            "MR !{}: not pushing — merge conflicts with origin/{} are still unresolved",
            input.mr_iid, input.target_branch
        );
    } else if has_new_changes {
        port.push_source_branch()?;
        info!(
            "Pushed changes addressing feedback for MR !{}",
            input.mr_iid
        );
    } else {
        info!(
            "Agent processed comments for MR !{} but made no code changes",
            input.mr_iid
        );
    }

    let surface_after = port.merge_request_surface(input.mr_iid)?;
    if surface_after.has_conflicts {
        conflicts_unresolved = true;
        warn!(
            "MR !{}: merge conflicts still reported after worker run",
            input.mr_iid
        );
    }
    let surface_changed = input.surface_before.differs_from(&surface_after);
    let origin_head = port
        .origin_head(&input.source_branch)
        .unwrap_or_else(|| input.pre_agent_sha.clone());
    let branch_tip_changed = origin_head.trim() != input.pre_agent_sha.trim();
    let implicit_resolve_discussions = has_new_changes || branch_tip_changed;
    if surface_changed && !implicit_resolve_discussions {
        info!(
            "MR !{} metadata changed without branch updates; discussions will remain open unless explicitly requested",
            input.mr_iid
        );
    }

    let ids_to_resolve = if input.unresolved_ids.is_empty() {
        port.unresolved_discussion_ids(input.mr_iid)
    } else {
        input.unresolved_ids.clone()
    };
    let should_post_plain_comment = input.resolution.post_plain_comment;
    let needs_reply_body =
        !ids_to_resolve.is_empty() || (input.plain_comments_present && should_post_plain_comment);

    if input.requires_conflict_resolution && conflicts_unresolved {
        info!(
            "MR !{}: merge conflicts with origin/{} remain; skipping replies until the branch merges cleanly",
            input.mr_iid, input.target_branch
        );
        return Ok(());
    }

    let reply_body = if needs_reply_body {
        let reply_raw = if let Some(block) =
            extract_worker_public_comment(input.resolution.public_comment.as_deref())
        {
            block
        } else if let Some(reply) = build_feedback_resolution_reply(
            &input.resolution,
            has_new_changes,
            diff_highlights.as_deref(),
        ) {
            reply
        } else {
            return Err(anyhow::anyhow!(
                "worker produced no source changes and no feedback reply for MR !{}",
                input.mr_iid
            ));
        };
        Some(strip_worker_reply_boilerplate(&reply_raw))
    } else {
        None
    };

    let resolve_discussions = feedback_discussions_may_be_resolved(
        &input.resolution,
        implicit_resolve_discussions,
        conflicts_unresolved,
    );
    if conflicts_unresolved && input.resolution.mark_discussions_resolved == Some(true) {
        warn!(
            "MR !{}: ignoring agent request to mark discussions resolved while merge conflicts remain",
            input.mr_iid
        );
    }
    if !resolve_discussions && !ids_to_resolve.is_empty() {
        info!(
            "MR !{}: posting feedback replies without resolving discussions (no mark_discussions_resolved signal and no implicit resolving actions)",
            input.mr_iid
        );
    }

    if let Some(body) = reply_body.as_deref() {
        for discussion_id in &ids_to_resolve {
            port.reply_to_discussion(discussion_id, body);
            if resolve_discussions {
                port.resolve_discussion(discussion_id);
            }
        }
        if input.plain_comments_present && should_post_plain_comment {
            port.post_plain_comment(body);
        }
    }
    Ok(())
}

struct LiveFeedbackTailPort<'a> {
    git_repo: &'a GitRepo,
    forge: &'a dyn ForgeClient,
    mr_iid: u64,
    source_branch: String,
    target_branch: String,
}

impl FeedbackTailPort for LiveFeedbackTailPort<'_> {
    fn update_mr_metadata(&mut self, title: &str, description: &str) {
        if let Err(e) =
            self.forge
                .update_mr_title_description(self.mr_iid, Some(title), Some(description))
        {
            warn!("Failed to update MR !{} metadata: {}", self.mr_iid, e);
        } else {
            info!(
                "Updated MR !{} title/description from agent feedback",
                self.mr_iid
            );
        }
    }

    fn fetch_branches(&mut self) -> Result<()> {
        self.git_repo
            .fetch_branches(&[self.target_branch.as_str(), self.source_branch.as_str()])
    }

    fn has_changes_since(&self, base_ref: &str) -> Result<bool> {
        self.git_repo.has_changes_since(base_ref)
    }

    fn merge_in_progress(&self) -> Result<bool> {
        self.git_repo.is_merge_in_progress()
    }

    fn stage_resolved_conflicts(&mut self) -> Result<bool> {
        self.git_repo.stage_resolved_unmerged_paths()
    }

    fn stage_all(&mut self) -> Result<()> {
        self.git_repo.add_all()
    }

    fn has_staged_changes(&self) -> Result<bool> {
        self.git_repo.has_staged_changes()
    }

    fn commit(&mut self, message: &str) -> Result<()> {
        self.git_repo.commit(message)
    }

    fn complete_merge_if_ready(&mut self, message: &str) -> Result<bool> {
        self.git_repo.complete_merge_if_ready(message)
    }

    fn merge_conflicts_present(&self) -> Result<bool> {
        self.git_repo.merge_conflicts_present()
    }

    fn up_to_date_with_target(&self, target_branch: &str) -> Result<bool> {
        self.git_repo.verify_up_to_date_with_target(target_branch)
    }

    fn diff_highlights(&self, base_ref: &str) -> Option<String> {
        build_diff_highlights_since(self.git_repo, base_ref)
    }

    fn push_source_branch(&mut self) -> Result<()> {
        self.git_repo.push(&self.source_branch)
    }

    fn merge_request_surface(&self, mr_iid: u64) -> Result<MrSurfaceObservation> {
        Ok(MrSurfaceObservation::from_mr(
            &self.forge.get_merge_request(mr_iid)?,
        ))
    }

    fn origin_head(&self, source_branch: &str) -> Option<String> {
        self.git_repo
            .rev_parse(&format!("origin/{source_branch}"))
            .ok()
    }

    fn unresolved_discussion_ids(&self, mr_iid: u64) -> Vec<String> {
        self.forge
            .get_unresolved_discussion_ids(mr_iid)
            .unwrap_or_default()
    }

    fn reply_to_discussion(&mut self, discussion_id: &str, body: &str) {
        if let Err(e) = self
            .forge
            .reply_to_discussion(self.mr_iid, discussion_id, body)
        {
            warn!("Failed to reply to discussion {}: {}", discussion_id, e);
        }
    }

    fn resolve_discussion(&mut self, discussion_id: &str) {
        if let Err(e) = self.forge.resolve_discussion(self.mr_iid, discussion_id) {
            warn!("Failed to resolve discussion {}: {}", discussion_id, e);
        }
    }

    fn post_plain_comment(&mut self, body: &str) {
        if let Err(e) = self.forge.add_mr_comment(self.mr_iid, body) {
            warn!(
                "Failed to post MR !{} reply for plain comments: {}",
                self.mr_iid, e
            );
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
                let issue = match state.forge.get_issue(issue_iid) {
                    Ok(i) => i,
                    Err(e) => {
                        if state.forge.is_not_found(&e) {
                            // The issue no longer exists (deleted or
                            // moved). Drop the stale session so we stop
                            // retrying a 404 on every cycle.
                            info!(
                                "{}: Issue #{} no longer exists (404), \
                                 discarding stale session",
                                &state.agent_id, issue_iid
                            );
                            state.cleanup_session(issue_iid);
                        } else {
                            warn!(
                                "{}: Failed to verify issue #{} for session resume: {}, skipping",
                                &state.agent_id, issue_iid, e
                            );
                        }
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

                match resolve_tracked_mr_for_worker_issue(
                    state.forge.as_ref(),
                    issue_iid,
                    session.mr_iid,
                ) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(state.forge.as_ref(), issue_iid);

                        let _ = claim::release(
                            state.forge.as_ref(),
                            ClaimResource::Issue(issue_iid),
                            state.agent_id,
                        );
                        let _ = state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);

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
        match state.forge.get_issue(issue_iid) {
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

                match resolve_tracked_mr_for_worker_issue(
                    state.forge.as_ref(),
                    issue_iid,
                    session.mr_iid,
                ) {
                    ResolvedTrackedMr::MergedCloseIssue => {
                        close_issue_best_effort(state.forge.as_ref(), issue_iid);

                        let _ = claim::release(
                            state.forge.as_ref(),
                            ClaimResource::Issue(issue_iid),
                            state.agent_id,
                        );
                        let _ = &state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);

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
                if state.forge.is_not_found(&e) {
                    info!(
                        "{}: Issue #{} no longer exists (404), \
                         discarding stale session",
                        &state.agent_id, issue_iid
                    );
                    state.cleanup_session(issue_iid);
                } else {
                    warn!(
                        "{}: Failed to verify issue #{}: {}, skipping",
                        &state.agent_id, issue_iid, e
                    );
                }
            }
        }
    }

    None
}

/// Scan all open GitLab issues for this worker's claim label.
/// Used as a fallback when the session file is missing (e.g. hard kill / crash).
fn find_claimed_issue(state: &AgentState, scope_label: Option<&str>) -> Option<ActiveIssue> {
    let claim_label = format!("claimed:{}", &state.agent_id);

    let issues = match state.forge.list_issues() {
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

        match resolve_tracked_mr_for_worker_issue(state.forge.as_ref(), issue.iid, 0) {
            ResolvedTrackedMr::MergedCloseIssue => {
                close_issue_best_effort(state.forge.as_ref(), issue.iid);

                let _ = claim::release(
                    state.forge.as_ref(),
                    ClaimResource::Issue(issue.iid),
                    state.agent_id,
                );
                let _ = state.forge.remove_issue_label(issue.iid, WORKING_ON_LABEL);

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

        let Ok(issue) = state.forge.get_issue(issue_iid) else {
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
            if let Ok(mr) = state.forge.get_merge_request(session.mr_iid) {
                if mr.state == "merged" || mr.state == "closed" {
                    info!(
                        "{}: Orphaned session for issue #{} has {} MR !{}, cleaning up",
                        &state.agent_id, issue_iid, mr.state, session.mr_iid
                    );

                    state.cleanup_session(issue_iid);
                    let _ = state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);
                    continue;
                }
            } else {
                continue;
            }
            (Some(session.mr_iid), true)
        } else {
            // No MR yet — check if one was created in the meantime
            match find_open_mr_for_issue(state.forge.as_ref(), issue_iid) {
                Some(mr) => (Some(mr), true),
                None => (None, false),
            }
        };

        // Try to claim this issue
        match claim::acquire(
            state.forge.as_ref(),
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
fn find_open_mr_for_issue(forge: &dyn ForgeClient, issue_iid: u64) -> Option<u64> {
    let branch_name = format!("issue-{}", issue_iid);
    forge
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

fn load_issue_context(forge: &dyn ForgeClient, issue_number: u64) -> Result<String> {
    match forge.get_issue(issue_number) {
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

fn abandon_mr(state: &AgentState, mr: &MergeRequest, issue_iid: u64, reason: &str) -> Result<()> {
    state.forge.add_mr_comment(
        mr.iid,
        &format!(
            "Closing this MR — the issue cannot be resolved autonomously:\n\n{}",
            reason
        ),
    )?;
    let _ = state.forge.close_mr(mr.iid);

    let default_branch = state
        .git_repo
        .get_default_branch()
        .unwrap_or("main".to_string());
    let _ = state.git_repo.reset_hard();
    let _ = state.git_repo.checkout_remote_branch(&default_branch);
    let _ = state.git_repo.delete_local_branch(&mr.source_branch);

    let _ = state.forge.remove_issue_label(issue_iid, WORKING_ON_LABEL);
    state
        .forge
        .add_issue_label(issue_iid, ACTION_REQUIRED_LABEL)?;
    state.forge.add_issue_comment(
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
        return trimmed.to_string();
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
    mr: &MergeRequest,
    git_repo: &GitRepo,
    forge: &dyn ForgeClient,
) -> String {
    const MAX_DIFF_CHARS: usize = 120_000;
    const MAX_FILES: usize = 200;

    let mut source_note = "Source: MR changes API (matches MR diff view).".to_string();
    let (diff_stat, changed_files, mut diff_patch, overflow_note, diff_refs_line) =
        match forge.get_merge_request_changes(mr.iid) {
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
                    Some("The MR diff overflowed; parts of the diff may be omitted.")
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
                    "Source: local git fallback (failed to read MR changes API: {}).",
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

fn format_comments_for_prompt(comments: &[Comment]) -> String {
    comments
        .iter()
        .map(|c| c.format_for_prompt())
        .collect::<Vec<_>>()
        .join("\n")
}

struct CombinedMrFeedbackContextInput<'a> {
    project_name: &'a str,
    mr: &'a MergeRequest,
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
    mr: &MergeRequest,
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
        format!(
            "## Merge conflict status (verified by {d} — trust this section)",
            d = display_name()
        ),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.target_branch, target_sha
        ),
        format!(
            "- Fetched `origin/{}` at commit: {}",
            mr.source_branch, source_sha
        ),
        format!(
            "- Merge conflicts reported on this MR: {}",
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
                format!(
                    "yes — resolve every unmerged file, then {d} will conclude the merge commit",
                    d = display_name()
                )
            } else {
                "no".to_string()
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
            format!("- After editing conflicted files, remove every conflict marker. {d} will `git add` resolved files and conclude the merge commit; you do not need to run git commands.", d = display_name()),
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
        .map(strip_markdown_formatting)
}

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

fn format_issue_comments_for_worker_context(forge: &dyn ForgeClient, issue_iid: u64) -> String {
    let comments = match forge.get_issue_comments(issue_iid) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Worker: failed to fetch issue comments for #{}: {}",
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

fn split_parent_context(
    forge: &dyn ForgeClient,
    issue: &IssueObservation,
) -> Result<Option<String>> {
    let Some(parent_iid) = split_parent_iid(&issue.description) else {
        return Ok(None);
    };
    let parent = forge
        .get_issue(parent_iid)
        .with_context(|| format!("failed to load parent issue #{parent_iid} for split child"))?;
    let comments = format_issue_comments_for_worker_context(forge, parent_iid);
    Ok(Some(format!(
        "## Original parent issue context\n\nIssue: #{} {}\n\n### Description\n{}\n\n### Issue comments\n\n{}",
        parent.iid, parent.title, parent.description, comments
    )))
}

fn worker_issue_context_markdown(
    issue: &IssueObservation,
    comments_text: &str,
    parent_context: Option<&str>,
) -> String {
    let parent_context = parent_context
        .map(|context| format!("\n\n{context}"))
        .unwrap_or_default();
    format!(
        "# Issue Context\n\nIssue: #{} {}\n\n## Description\n{}\n\n## Issue comments\n\n{}{}\n",
        issue.iid, issue.title, issue.description, comments_text, parent_context
    )
}

fn build_implementation_prompt(
    state: &AgentState,
    issue: &IssueObservation,
    comments_text: &str,
) -> Result<String> {
    let parent_context = split_parent_context(state.forge.as_ref(), issue)?;
    let context_content =
        worker_issue_context_markdown(issue, comments_text, parent_context.as_deref());
    // The task context (issue description + comments) goes to disk; the
    // prompt references it by path — the model reads what it needs.
    let context_path = write_task_context_file(
        state.sessions_dir,
        &format!("{}-issue-{}.md", state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(false);
    let notes_rules = get_notes_rules();

    let prompt = format!(
        r#"SYSTEM: You are implementing a feature for a software project in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT FILE (open and read it first — it has the full picture):
{}

CONTEXT:
- The task context file contains this issue's **description** and **every issue comment** at the time the task started. That is your primary written spec.
- Labels on the issue (e.g. priority) are visible in the task context file; infer scope from description + comments + `AGENTS.md`.

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. Open and read the ENTIRE task context file at the path above (description and every issue comment) before anything else.
3. Analyze the issue and comments carefully
4. Before writing any new code, investigate the codebase for existing mechanisms that already do what you need (see REUSE FIRST above). Grep for the shape of what you need; read what you find; reuse or extend it. Only create something new when the search genuinely comes up empty — and say what you searched.
5. Estimate the number of changed lines:
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
6. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the feature can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be implemented as one unit, proceed with implementation
   - Otherwise, report that the issue needs splitting and explain the estimated line count and decomposition
7. If the issue is unclear or missing critical information that makes implementation impossible, report that clarification is needed and explain what information is missing and why.
8. If the issue requires large unrelated feature work, report that the issue needs splitting and explain the decomposition.
9. If at any point you determine the issue simply cannot be implemented without additional human input that you cannot infer or assume (e.g. missing API credentials, undocumented external system dependencies, contradictory requirements), report that clarification is needed and explain precisely what input is required.
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the issue, REJECT it. Do not guess or produce speculative code.
- If you believe the implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

10. If the issue is clear, focused, and reasonably sized (or cannot be split), implement ONLY what is asked
11. Make all necessary code changes autonomously
12. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
13. {}

Proceed with the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_path,
        common_requirements,
        scope_rules,
        notes_rules
    );

    Ok(prompt)
}

fn build_continuation_prompt(
    state: &AgentState,
    issue: &IssueObservation,
    comments_text: &str,
) -> Result<String> {
    let parent_context = split_parent_context(state.forge.as_ref(), issue)?;
    let context_content =
        worker_issue_context_markdown(issue, comments_text, parent_context.as_deref());
    // The task context (issue description + comments) goes to disk; the
    // prompt references it by path — the model reads what it needs.
    let context_path = write_task_context_file(
        state.sessions_dir,
        &format!("{}-issue-{}.md", &state.agent_id, issue.iid),
        &context_content,
    )?;

    let common_requirements = get_common_requirements();
    let scope_rules = get_scope_rules(true);
    let notes_rules = get_notes_rules();

    let prompt = format!(
        r#"SYSTEM: You are continuing work on an existing feature branch in a fully automated, non-interactive environment.

PROJECT: {}

ISSUE #{}: {}

TASK CONTEXT FILE (open and read it first — it has the full picture):
{}

CONTEXT:
- The task context file contains this issue's **description** and **every issue comment** at task start.
- A branch for this issue already exists with previous work
- You are continuing the implementation from where it was left off
- Review the existing code changes in this branch
- Complete any remaining work needed to fully implement the issue

{}

{}

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before making any changes. Follow it strictly for implementation, tests, linting, and documentation rules.
2. Open and read the ENTIRE task context file at the path above (description and every issue comment) before anything else.
3. Review the existing changes in the current branch
4. Analyze what has been done and what remains
5. Before writing any new code, investigate the codebase for existing mechanisms that already do what you need (see REUSE FIRST above) — including the work already on this branch. Reuse or extend what exists; only create something new when the search genuinely comes up empty, and say what you searched.
6. Estimate total changed lines (including existing + remaining work):
   - Non-test, non-generated code: should stay around ~500 lines
   - Total including tests: should stay around ~1500 lines
   - Do NOT count auto-generated files (with "generated by", "auto-generated", "DO NOT EDIT" comments)
7. If non-test code exceeds ~500 lines or total exceeds ~1500 lines:
   - Evaluate if the remaining work can be split into smaller, independent pieces
   - If you are VERY SURE it CANNOT be split and MUST be completed as one unit, proceed with implementation
   - Otherwise, report that the issue needs splitting and explain the estimated line count and decomposition
8. If the issue is unclear or missing critical information that makes implementation impossible, report that clarification is needed and explain what information is missing and why.
9. If the issue requires large unrelated feature work, report that the issue needs splitting and explain the decomposition.
10. If at any point you determine the remaining work cannot be completed without additional human input that you cannot infer or assume, report that clarification is needed and explain precisely what input is required.
IMPORTANT — When in doubt, REJECT:
- If you are unsure how to implement the remaining work, REJECT it. Do not guess or produce speculative code.
- If you believe the total implementation would be huge or complex beyond what a single focused MR should contain, REJECT it.
- NEVER produce a result with no actual code changes. If you cannot make meaningful code changes, you MUST reject the issue instead.
- It is always better to reject an issue and explain why than to produce a bad, incomplete, or oversized implementation.

11. If the issue is clear, focused, and reasonably sized (or cannot be split), continue the implementation
12. ONLY implement what the issue asks for, nothing more
13. Complete any remaining work autonomously
14. Make reasonable assumptions only when minor details are unclear and the overall approach is certain
15. {}

Proceed with continuing the implementation autonomously. Do not ask for any user input.
"#,
        &state.project_name,
        issue.iid,
        issue.title,
        context_path,
        common_requirements,
        scope_rules,
        notes_rules
    );

    Ok(prompt)
}

fn get_common_requirements() -> &'static str {
    r#"CRITICAL REQUIREMENTS:
- Leave staging, committing, pushing, and merge request creation to the system
- If information is missing, document what's needed in your response (do not ask interactively)
- If you are making code changes you MUST stick to AGENTS.md in the project strictly
- Read the issue comments carefully — they may contain guidance from the PMO agent on how to proceed. PMO guidance appears as a comment starting with **PMO guidance for the worker agent:** — treat the body of that comment as authoritative worker instructions and follow it exactly.
- Before finishing, update repo-root notes.md only when you have bullets that pass the NOTES.MD rules below: not a recap of your MR, not generic best-practice slides, not meta about notes — if nothing qualifies, leave the file unchanged. Never paste notes.md into MR metadata or issue comments

REUSE FIRST — DUPLICATION IS THE ENEMY:
- Before writing any new function, type, helper, constant, wrapper, or module, investigate whether the codebase already provides the mechanism you need: grep for it, read it, and use it.
- A mechanism that already exists — even one you must adapt slightly or call with different arguments — is nearly always better than a new parallel implementation. Extend the existing one rather than adding a sibling next to it.
- Writing a new version of something that already exists is a defect, not a preference: it doubles the surface that must be tested, reviewed, and later fixed in two places. If you find yourself about to create something that resembles an existing thing in shape or purpose, STOP, find that thing, and reuse it.
- If you cannot find an existing mechanism after genuinely searching, say so in your result (what you searched for and where) before introducing a new one — this keeps the search honest.
- This rule outranks convenience: a slightly awkward call into existing code beats a clean new duplicate.

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
    r#"- Only modify code that is strongly supported by the issue title, issue description, issue comments, or current unresolved MR feedback. If a change is merely adjacent, speculative, weakly coupled, or "nice to have", do not make it.
- Treat the issue title, issue description, and issue comments as the strict boundary of allowed code changes.
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
/// `public_comment` field: blank text means "no comment".
fn extract_worker_public_comment(public_comment: Option<&str>) -> Option<String> {
    let s = public_comment?.trim();
    if s.is_empty() {
        return None;
    }
    Some(s.to_string())
}

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

/// Extract the MR description the agent emitted via the `handoff` tool's
/// `mr_description` field, falling back to a generic placeholder when absent.
fn extract_mr_description(mr_description: Option<&str>) -> String {
    if let Some(s) = mr_description {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "Implementation completed.".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::forge;
    use crate::agents::labels::DO_NOT_IMPLEMENT;
    use crate::core::agent::StructuredOutput;
    use crate::core::agent::schema::conformance;
    use crate::core::agent::validate_agent_config;

    // -----------------------------------------------------------------
    // Session persistence: tolerant policy. Corrupt/unsupported session
    // files have historically been treated as "no session" rather than
    // failing the worker cycle; `GitRepo::new`/`forge::for_test_client`
    // are file/network-free constructors so this exercises the real
    // `AgentState` session methods without touching git or GitLab.
    // -----------------------------------------------------------------

    /// Owns the resources a test [`AgentState`] borrows from, standing in
    /// for the [`AgentWorkspace`] fields the worker cycle needs.
    struct TestRuntime {
        project_name: String,
        agent_id: String,
        sessions_dir: String,
        git_repo: GitRepo,
        forge: Arc<dyn ForgeClient>,
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
            forge: forge::for_test_client("/tmp/unused-repo"),
        }
    }

    fn test_agent_state(rt: &TestRuntime) -> AgentState<'_> {
        AgentState {
            project_name: &rt.project_name,
            agent_id: &rt.agent_id,
            sessions_dir: &rt.sessions_dir,
            git_repo: &rt.git_repo,
            forge: &rt.forge,
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

        let error = validate_agent_config::<WorkerAgent>(&config, section).unwrap_err();

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
    }

    #[test]
    fn extract_clarification_prefers_public_comment_then_reason_then_default() {
        assert_eq!(
            extract_clarification(&blocked("what auth scheme?")),
            "what auth scheme?"
        );
    }

    #[test]
    fn extract_cannot_implement_reason_falls_back_to_its_own_default() {
        assert_eq!(
            extract_cannot_implement_reason(&blocked("contradictory requirements")),
            "contradictory requirements"
        );
    }

    #[test]
    fn extract_worker_public_comment_returns_none_when_absent_or_blank() {
        assert_eq!(extract_worker_public_comment(None), None);
        assert_eq!(extract_worker_public_comment(Some("   ")), None);
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
        issue.labels = vec![DO_NOT_IMPLEMENT.to_string()];
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
    fn worker_should_cancel_issue_processing_for_blocking_changes() {
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

        issue.labels = vec![DO_NOT_IMPLEMENT.to_string()];
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
        use MergeRequest;
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

    fn mr_comment(id: u64, author: &str, discussion_id: &str, body: &str) -> Comment {
        Comment {
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
    fn collect_new_follow_ups_returns_replies_to_any_active_discussion() {
        let comments = vec![
            mr_comment(10, "alice", "d1", "old comment"),
            mr_comment(15, "bob", "d2", "reply to another active discussion"),
            mr_comment(20, "carol", "d1", "another new one"),
            mr_comment(25, "dave", "d3", "new top-level comment"),
        ];
        let mut last_seen = 10u64;
        let handled = HashSet::from(["d1".to_string(), "d2".to_string()]);
        let msgs = collect_new_follow_ups(&comments, &mut last_seen, 42, &handled);
        assert_eq!(msgs.len(), 2);
        assert_eq!(last_seen, 25);
    }

    #[test]
    fn collect_new_follow_ups_skips_all_when_seen_is_max() {
        let comments = vec![
            mr_comment(5, "alice", "d1", "old"),
            mr_comment(5, "bob", "d2", "also old"),
        ];
        let mut last_seen = 5u64;
        let handled = HashSet::from(["d1".to_string(), "d2".to_string()]);
        let msgs = collect_new_follow_ups(&comments, &mut last_seen, 1, &handled);
        assert!(msgs.is_empty());
        assert_eq!(last_seen, 5);
    }

    #[test]
    fn collect_new_follow_ups_handles_empty_comments() {
        let mut last_seen = 3u64;
        let msgs = collect_new_follow_ups(&[], &mut last_seen, 1, &HashSet::new());
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
    // Direct routing workflow traces.
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

    struct FakeWorkerPort {
        trace: std::cell::RefCell<Vec<String>>,
        implementation_payloads: std::cell::RefCell<Vec<IssueObservation>>,
        feedback_payloads: std::cell::RefCell<Vec<(u64, Option<u64>, bool)>>,
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
        failures: Vec<String>,
        issue_reads: std::cell::RefCell<std::collections::HashMap<u64, usize>>,
        fail_issue_read: Option<(u64, usize)>,
    }

    impl FakeWorkerPort {
        fn new() -> Self {
            Self {
                trace: Default::default(),
                implementation_payloads: Default::default(),
                feedback_payloads: Default::default(),
                shutdown_answers: Default::default(),
                issues: vec![],
                known_issues: vec![],
                mr_states: vec![],
                default_branch: "main".to_string(),
                claim_attempts: Default::default(),
                adopted: None,
                handled_labeled_mr: false,
                implementation: Default::default(),
                feedback_abandoned: false,
                feedback_error: None,
                cancel_handled: false,
                trackable: true,
                failures: vec![],
                issue_reads: Default::default(),
                fail_issue_read: None,
            }
        }

        fn listing(mut self, issues: &[IssueObservation]) -> Self {
            self.issues = issues.to_vec();
            self.known_issues.extend(issues.iter().cloned());
            self
        }
        fn knowing(mut self, issues: &[IssueObservation]) -> Self {
            self.known_issues.extend(issues.iter().cloned());
            self
        }
        fn with_mr_state(mut self, iid: u64, state: &str) -> Self {
            self.mr_states.push((iid, state.to_string()));
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
        fn failing(mut self, operation: &str) -> Self {
            self.failures.push(operation.to_string());
            self
        }
        fn failing_issue_read(mut self, iid: u64, read: usize) -> Self {
            self.fail_issue_read = Some((iid, read));
            self
        }

        fn record(&self, operation: impl Into<String>) {
            self.trace.borrow_mut().push(operation.into());
        }
        fn required(&self, operation: impl Into<String>) -> Result<()> {
            let operation = operation.into();
            self.record(operation.clone());
            if self.failures.contains(&operation) {
                anyhow::bail!("injected failure: {operation}");
            }
            Ok(())
        }
    }

    impl WorkerRoutingPort for FakeWorkerPort {
        fn shutdown_requested(&self) -> bool {
            self.record("shutdown");
            self.shutdown_answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(false)
        }
        fn issue(&self, issue_iid: u64) -> Result<IssueObservation> {
            let operation = format!("issue:{issue_iid}");
            self.required(operation)?;
            let mut reads = self.issue_reads.borrow_mut();
            let read = reads.entry(issue_iid).or_default();
            let current = *read;
            *read += 1;
            if self.fail_issue_read == Some((issue_iid, current)) {
                anyhow::bail!("injected failure on issue read {current}");
            }
            self.known_issues
                .iter()
                .find(|i| i.iid == issue_iid)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("404 Issue Not Found: #{issue_iid}"))
        }
        fn merge_request_status(&self, mr_iid: u64) -> Result<MrStatusObservation> {
            self.required(format!("mr_status:{mr_iid}"))?;
            let state = self
                .mr_states
                .iter()
                .find(|(iid, _)| *iid == mr_iid)
                .map(|(_, state)| state.clone())
                .unwrap_or_else(|| "opened".to_string());
            Ok(MrStatusObservation { iid: mr_iid, state })
        }
        fn issues(&self) -> Result<Vec<IssueObservation>> {
            self.required("issues")?;
            Ok(self.issues.clone())
        }
        fn default_branch_or_main(&self) -> String {
            self.record("default_branch");
            self.default_branch.clone()
        }
        fn reset_worktree(&mut self) {
            self.record("reset_worktree");
        }
        fn checkout_branch(&mut self, branch: &str) {
            self.record(format!("checkout:{branch}"));
        }
        fn delete_local_branch(&mut self, branch: &str) {
            self.record(format!("delete_local:{branch}"));
        }
        fn delete_remote_branch(&mut self, branch: &str) {
            self.record(format!("delete_remote:{branch}"));
        }
        fn release_issue_claim(&mut self, iid: u64) -> bool {
            self.record(format!("release_claim:{iid}"));
            !self.failures.contains(&format!("release_claim:{iid}"))
        }
        fn remove_working_on_label(&mut self, iid: u64) {
            self.record(format!("remove_working:{iid}"));
        }
        fn remove_issue_label(&mut self, iid: u64, label: &str) {
            self.record(format!("remove_label:{iid}:{label}"));
        }
        fn cleanup_session(&mut self, iid: u64) {
            self.record(format!("cleanup_session:{iid}"));
        }
        fn save_session(&mut self, iid: u64, mr: u64) {
            self.record(format!("save_session:{iid}:{mr}"));
        }
        fn close_issue(&mut self, iid: u64) {
            self.record(format!("close_issue:{iid}"));
        }
        fn acquire_issue_claim(&mut self, iid: u64) -> Result<IssueClaimAttempt> {
            self.required(format!("acquire_claim:{iid}"))?;
            Ok(self
                .claim_attempts
                .borrow_mut()
                .pop_front()
                .unwrap_or(IssueClaimAttempt::Won))
        }
        fn preserve_issue_claim(&mut self, iid: u64) {
            self.record(format!("preserve_claim:{iid}"));
        }
        fn release_acquired_claim(&mut self, iid: u64) {
            self.record(format!("release_acquired:{iid}"));
        }
        fn clear_issue_state(&mut self, iid: u64) -> bool {
            self.record(format!("clear_state:{iid}"));
            true
        }
        fn release_review_only_hold(&mut self, iid: u64) -> bool {
            self.record(format!("release_review_only:{iid}"));
            true
        }
        fn abandon_closed_issue(&mut self, iid: u64, mr: Option<u64>) -> bool {
            self.record(format!("abandon_closed:{iid}:{mr:?}"));
            true
        }
        fn adopt_orphaned_session(&mut self) -> Option<ActiveIssue> {
            self.record("adopt_orphan");
            self.adopted.clone()
        }
        fn handle_need_ai_worker_mr(&mut self) -> Result<bool> {
            self.required("handle_labeled_mr")?;
            Ok(self.handled_labeled_mr)
        }
        fn run_implementation(
            &mut self,
            issue: &IssueObservation,
        ) -> (ActiveIssue, Option<anyhow::Error>) {
            self.record(format!("run_implementation:{}", issue.iid));
            self.implementation_payloads
                .borrow_mut()
                .push(issue.clone());
            let script = self
                .implementation
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(ImplementationScript::no_mr);
            (
                ActiveIssue {
                    issue_iid: issue.iid,
                    mr_iid: script.mr_iid,
                    branch_name: script.branch_name,
                    mr_created: script.mr_created,
                },
                script.error.map(|e| anyhow::anyhow!(e)),
            )
        }
        fn run_feedback(
            &mut self,
            mr: u64,
            issue: Option<u64>,
            comments_only: bool,
        ) -> Result<bool> {
            self.record(format!("run_feedback:{mr}:{issue:?}:{comments_only}"));
            self.feedback_payloads
                .borrow_mut()
                .push((mr, issue, comments_only));
            match &self.feedback_error {
                Some(e) => Err(anyhow::anyhow!(e.clone())),
                None => Ok(self.feedback_abandoned),
            }
        }
        fn resolve_cancelled_issue(&mut self, iid: u64) -> bool {
            self.record(format!("resolve_cancel:{iid}"));
            self.cancel_handled
        }
        fn issue_trackable(&mut self, iid: u64) -> bool {
            self.record(format!("trackable:{iid}"));
            self.trackable
        }
    }

    struct FakeWorkerRun {
        result: Result<()>,
        trace: Vec<String>,
        active: Option<ActiveIssue>,
    }

    fn run_worker_routing(
        port: &mut FakeWorkerPort,
        mut active: Option<ActiveIssue>,
    ) -> FakeWorkerRun {
        let result = run_worker_routing_cycle(port, ROUTING_AGENT, None, &mut active);
        FakeWorkerRun {
            result,
            trace: port.trace.borrow().clone(),
            active,
        }
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
    fn polling_steps() -> Vec<String> {
        [
            "shutdown",
            "adopt_orphan",
            "handle_labeled_mr",
            "issues",
            "shutdown",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    #[test]
    fn worker_cycle_claims_a_candidate_then_implements_and_tracks_it_in_order() {
        let candidate = issue_observation(7, &[]);
        let mut port = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut port, None);
        let mut expected = polling_steps();
        expected.extend(
            [
                "shutdown",
                "acquire_claim:7",
                "shutdown",
                "issue:7",
                "preserve_claim:7",
                "save_session:7:0",
                "run_implementation:7",
                "trackable:7",
            ]
            .into_iter()
            .map(str::to_string),
        );
        assert!(run.result.is_ok());
        assert_eq!(run.trace, expected);
        assert_eq!(run.active, Some(active_with_mr(7, 7)));
        assert_eq!(
            port.implementation_payloads.borrow().as_slice(),
            &[candidate]
        );
    }

    #[test]
    fn worker_releases_a_claim_when_do_not_implement_appears_after_screening() {
        let candidate = issue_observation(7, &[]);
        let blocked = issue_observation(7, &[DO_NOT_IMPLEMENT]);
        let mut port = FakeWorkerPort::new()
            .knowing(&[blocked])
            .listing(&[candidate]);

        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert!(run.active.is_none());
        assert!(run.trace.ends_with(&strings(&[
            "shutdown",
            "acquire_claim:7",
            "shutdown",
            "issue:7",
            "release_acquired:7",
        ])));
        assert!(
            !run.trace
                .iter()
                .any(|event| event.starts_with("run_implementation:"))
        );
    }

    #[test]
    fn worker_cycle_preserves_candidate_cleanup_and_shutdown_policies() {
        let candidate = issue_observation(7, &[]);
        let mut no_mr = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::no_mr()]);
        let run = run_worker_routing(&mut no_mr, None);
        assert_eq!(
            run.trace.last().map(String::as_str),
            Some("release_claim:7")
        );
        assert!(run.active.is_none());

        let mut failed = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "boom")]);
        let run = run_worker_routing(&mut failed, None);
        assert_eq!(
            &run.trace[run.trace.len() - 7..],
            &strings(&[
                "shutdown",
                "release_claim:7",
                "remove_working:7",
                "default_branch",
                "reset_worktree",
                "checkout:main",
                "delete_local:issue-7"
            ])
        );
        assert!(!run.trace.contains(&"cleanup_session:7".to_string()));

        let mut shutdown = FakeWorkerPort::new()
            .listing(&[candidate])
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "interrupted")])
            .with_shutdown_answers(&[false, false, false, false, true]);
        let run = run_worker_routing(&mut shutdown, None);
        assert_eq!(run.trace.last().map(String::as_str), Some("shutdown"));
        assert_eq!(run.active.unwrap().branch_name.as_deref(), Some("issue-7"));
    }

    #[test]
    fn worker_cycle_handles_candidate_cancel_and_claim_outcomes() {
        let candidate = issue_observation(7, &[]);
        let mut cancelled = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::failed(
                Some("issue-7"),
                WORKER_AGENT_CANCELLED_MSG,
            )])
            .handling_cancel();
        let run = run_worker_routing(&mut cancelled, None);
        assert_eq!(
            run.trace.last().map(String::as_str),
            Some("resolve_cancel:7")
        );
        assert!(run.active.is_none());

        let mut real_failure = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .implementing(&[ImplementationScript::failed(
                None,
                WORKER_AGENT_CANCELLED_MSG,
            )]);
        let run = run_worker_routing(&mut real_failure, None);
        assert_eq!(
            &run.trace[run.trace.len() - 3..],
            &strings(&["resolve_cancel:7", "release_claim:7", "remove_working:7"])
        );

        let mut shutdown = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .with_shutdown_answers(&[false, false, false, true]);
        let run = run_worker_routing(&mut shutdown, None);
        assert_eq!(
            &run.trace[run.trace.len() - 2..],
            &strings(&["shutdown", "release_acquired:7"])
        );

        let mut lost = FakeWorkerPort::new()
            .listing(&[candidate, issue_observation(9, &[])])
            .with_claim_attempts(&[IssueClaimAttempt::Lost, IssueClaimAttempt::Won])
            .implementing(&[ImplementationScript::succeeded(9)]);
        assert_eq!(
            run_worker_routing(&mut lost, None).active,
            Some(active_with_mr(9, 9))
        );

        let mut interrupted = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[]), issue_observation(9, &[])])
            .with_claim_attempts(&[IssueClaimAttempt::Interrupted]);
        let run = run_worker_routing(&mut interrupted, None);
        assert_eq!(
            run.trace.last().map(String::as_str),
            Some("acquire_claim:7")
        );
    }

    #[test]
    fn worker_cycle_screens_candidates_and_dependencies_in_order() {
        let mut skipped = FakeWorkerPort::new().listing(&[
            issue_observation(1, &[WORKING_ON_LABEL]),
            issue_observation(2, &["claimed:worker-2"]),
            issue_observation(3, &[ACTION_REQUIRED_LABEL]),
        ]);
        let run = run_worker_routing(&mut skipped, None);
        let mut expected = polling_steps();
        expected.extend(strings(&["shutdown", "shutdown", "shutdown"]));
        assert_eq!(run.trace, expected);

        let candidate = issue_observation(7, &[&waiting_on_issue_label(4)]);
        let mut closed_dep = issue_observation(4, &[]);
        closed_dep.state = "closed".to_string();
        let mut resumed = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .knowing(&[closed_dep])
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut resumed, None);
        assert!(run.trace.windows(3).any(|w| w
            == strings(&[
                "issue:4",
                "remove_label:7:waiting-on-issue:#4",
                "acquire_claim:7"
            ])));

        let mut waiting = FakeWorkerPort::new()
            .listing(std::slice::from_ref(&candidate))
            .knowing(&[issue_observation(4, &[])]);
        let run = run_worker_routing(&mut waiting, None);
        assert_eq!(run.trace.last().map(String::as_str), Some("issue:4"));

        let mut missing = FakeWorkerPort::new().listing(std::slice::from_ref(&candidate));
        let run = run_worker_routing(&mut missing, None);
        assert!(
            run.trace
                .contains(&"remove_label:7:waiting-on-issue:#4".to_string())
        );

        let mut unreadable = FakeWorkerPort::new()
            .listing(&[candidate])
            .knowing(&[issue_observation(4, &[])])
            .failing("issue:4");
        let run = run_worker_routing(&mut unreadable, None);
        assert!(run.result.is_ok());
        assert_eq!(run.trace.last().map(String::as_str), Some("issue:4"));
    }

    #[test]
    fn worker_cycle_adopts_or_handles_labeled_mr_before_listing() {
        let mut adopting = FakeWorkerPort::new().adopting(active_with_mr(7, 12));
        let run = run_worker_routing(&mut adopting, None);
        assert_eq!(run.trace, strings(&["shutdown", "adopt_orphan"]));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        let mut labeled = FakeWorkerPort::new().handling_labeled_mr();
        let run = run_worker_routing(&mut labeled, None);
        assert_eq!(
            run.trace,
            strings(&["shutdown", "adopt_orphan", "handle_labeled_mr"])
        );
    }

    #[test]
    fn worker_cycle_propagates_required_read_and_write_failures() {
        let mut listing = FakeWorkerPort::new().failing("issues");
        assert!(run_worker_routing(&mut listing, None).result.is_err());
        let mut labeled = FakeWorkerPort::new().failing("handle_labeled_mr");
        assert!(run_worker_routing(&mut labeled, None).result.is_err());
        let mut claim = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[])])
            .failing("acquire_claim:7");
        assert!(run_worker_routing(&mut claim, None).result.is_err());
    }

    #[test]
    fn worker_cycle_cleans_finished_merge_requests_in_exact_order() {
        let mut merged = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .with_mr_state(12, "merged")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut merged, Some(active_with_mr(7, 12)));
        assert_eq!(
            run.trace,
            strings(&[
                "issue:7",
                "mr_status:12",
                "release_claim:7",
                "default_branch",
                "reset_worktree",
                "checkout:main",
                "delete_local:issue-7",
                "delete_remote:issue-7",
                "remove_working:7",
                "cleanup_session:7",
                "close_issue:7",
                "shutdown",
            ])
        );
        assert!(run.active.is_none());

        let mut release_failure = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .with_mr_state(12, "merged")
            .failing("release_claim:7");
        let run = run_worker_routing(&mut release_failure, Some(active_with_mr(7, 12)));
        assert_eq!(
            run.trace,
            strings(&["issue:7", "mr_status:12", "release_claim:7"])
        );
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        let mut closed = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .with_mr_state(12, "closed")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut closed, Some(active_with_mr(7, 12)));
        assert!(!run.trace.contains(&"delete_remote:issue-7".to_string()));
        assert!(!run.trace.contains(&"close_issue:7".to_string()));
        assert_eq!(
            &run.trace[run.trace.len() - 2..],
            &strings(&["cleanup_session:7", "shutdown"])
        );
    }

    #[test]
    fn worker_cycle_runs_feedback_and_preserves_transient_failures() {
        let issue = issue_observation(7, &[WORKING_ON_LABEL]);
        let mut open = FakeWorkerPort::new().knowing(std::slice::from_ref(&issue));
        let run = run_worker_routing(&mut open, Some(active_with_mr(7, 12)));
        assert_eq!(
            run.trace,
            strings(&["issue:7", "mr_status:12", "run_feedback:12:Some(7):false"])
        );
        assert_eq!(
            open.feedback_payloads.borrow().as_slice(),
            &[(12, Some(7), false)]
        );
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        let mut abandoned = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .abandoning_feedback()
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut abandoned, Some(active_with_mr(7, 12)));
        assert_eq!(
            &run.trace[run.trace.len() - 3..],
            &strings(&["release_claim:7", "cleanup_session:7", "shutdown"])
        );
        assert!(run.active.is_none());

        let mut shutdown = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .failing_feedback("boom")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut shutdown, Some(active_with_mr(7, 12)));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));

        let mut cancelled = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .failing_feedback(WORKER_AGENT_CANCELLED_MSG)
            .handling_cancel();
        let run = run_worker_routing(&mut cancelled, Some(active_with_mr(7, 12)));
        assert_eq!(
            run.trace.last().map(String::as_str),
            Some("resolve_cancel:7")
        );
        assert!(run.active.is_none());

        let mut transient = FakeWorkerPort::new()
            .knowing(&[issue])
            .failing_feedback("boom");
        assert_eq!(
            run_worker_routing(&mut transient, Some(active_with_mr(7, 12))).active,
            Some(active_with_mr(7, 12))
        );
    }

    #[test]
    fn worker_cycle_retains_active_mr_when_status_read_fails() {
        let mut port = FakeWorkerPort::new()
            .knowing(&[issue_observation(7, &[WORKING_ON_LABEL])])
            .failing("mr_status:12");
        let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));
        assert!(run.result.is_ok());
        assert_eq!(run.trace.last().map(String::as_str), Some("mr_status:12"));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));
    }

    #[test]
    fn worker_cycle_releases_active_issues_that_are_no_longer_eligible() {
        let mut closed_issue = issue_observation(7, &[WORKING_ON_LABEL]);
        closed_issue.state = "closed".to_string();
        let mut closed = FakeWorkerPort::new()
            .knowing(&[closed_issue])
            .with_shutdown_answers(&[true]);
        assert_eq!(
            run_worker_routing(&mut closed, Some(active_with_mr(7, 12))).trace,
            strings(&["issue:7", "abandon_closed:7:Some(12)", "shutdown"])
        );

        for (label, action) in [
            (WORKER_PENDING_LABEL, "clear_state:7"),
            (DO_NOT_IMPLEMENT, "clear_state:7"),
            (WORKER_REVIEW_ONLY_LABEL, "release_review_only:7"),
        ] {
            let mut port = FakeWorkerPort::new()
                .knowing(&[issue_observation(7, &[label])])
                .with_shutdown_answers(&[true]);
            let run = run_worker_routing(&mut port, Some(active_with_mr(7, 12)));
            assert_eq!(run.trace[1], action);
            assert!(run.active.is_none());
        }
        let mut unreadable = FakeWorkerPort::new()
            .failing("issue:7")
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut unreadable, Some(active_with_mr(7, 12)));
        assert_eq!(run.trace, strings(&["issue:7", "shutdown"]));
        assert_eq!(run.active, Some(active_with_mr(7, 12)));
    }

    #[test]
    fn worker_cycle_retries_active_issue_and_deletes_reattempt_session() {
        let issue = issue_observation(7, &[WORKING_ON_LABEL]);
        let mut success = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .implementing(&[ImplementationScript::succeeded(7)]);
        let run = run_worker_routing(&mut success, Some(active_without_mr(7)));
        assert_eq!(
            run.trace,
            strings(&["issue:7", "issue:7", "run_implementation:7", "trackable:7"])
        );
        assert_eq!(run.active, Some(active_with_mr(7, 7)));

        let mut untrackable = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .implementing(&[ImplementationScript::succeeded(7)])
            .untrackable();
        assert!(
            run_worker_routing(&mut untrackable, Some(active_without_mr(7)))
                .active
                .is_none()
        );

        let mut failed = FakeWorkerPort::new()
            .knowing(std::slice::from_ref(&issue))
            .implementing(&[ImplementationScript::failed(Some("issue-7"), "boom")]);
        let run = run_worker_routing(&mut failed, Some(active_without_mr(7)));
        assert_eq!(
            &run.trace[run.trace.len() - 8..],
            &strings(&[
                "shutdown",
                "release_claim:7",
                "remove_working:7",
                "cleanup_session:7",
                "default_branch",
                "reset_worktree",
                "checkout:main",
                "delete_local:issue-7"
            ])
        );

        let mut second_read = FakeWorkerPort::new()
            .knowing(&[issue])
            .failing_issue_read(7, 1)
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut second_read, Some(active_without_mr(7)));
        assert_eq!(run.trace, strings(&["issue:7", "issue:7"]));
        assert_eq!(run.active, Some(active_without_mr(7)));
    }

    // -----------------------------------------------------------------
    // Implementation progression traces.
    // -----------------------------------------------------------------

    struct FakeImplPort {
        trace: std::cell::RefCell<Vec<String>>,
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
        failing_operations: Vec<&'static str>,
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
                failing_operations: Vec::new(),
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

        fn failing(mut self, operation: &'static str) -> Self {
            self.failing_operations.push(operation);
            self
        }

        fn record(&self, operation: impl Into<String>) {
            self.trace.borrow_mut().push(operation.into());
        }

        fn required(&self, operation: &'static str, payload: impl std::fmt::Display) -> Result<()> {
            self.record(format!("{operation}({payload})"));
            if self.failing_operations.contains(&operation) {
                anyhow::bail!("injected failure in {operation}");
            }
            Ok(())
        }
    }

    impl ImplementationPort for FakeImplPort {
        fn closes_linked_mr(&self) -> Option<ClosesLinkedMr> {
            self.record("closes_linked_mr");
            self.closes_linked
        }

        fn open_mr_for_issue(&self) -> Option<u64> {
            self.record("open_mr_for_issue");
            self.open_mr
        }

        fn stop_if_review_only(&mut self) -> bool {
            self.record("stop_if_review_only");
            self.stop_answers.borrow_mut().pop_front().unwrap_or(false)
        }

        fn add_working_on_label(&mut self) {
            self.record("add_working_on_label");
        }

        fn require_working_on_label(&mut self) -> Result<()> {
            self.required("require_working_on_label", "")
        }

        fn remove_working_on_label(&mut self) {
            self.record("remove_working_on_label");
        }

        fn add_issue_label(&mut self, label: &str) {
            self.record(format!("add_issue_label({label})"));
        }

        fn add_issue_comment(&mut self, body: &str) {
            self.record(format!("add_issue_comment({body})"));
        }

        fn release_issue_claim(&mut self) -> Result<()> {
            self.record("release_issue_claim");
            Ok(())
        }

        fn cleanup_session(&mut self) {
            self.record("cleanup_session");
        }

        fn close_issue(&mut self) {
            self.record("close_issue");
        }

        fn save_session(&mut self, mr_iid: u64) {
            self.record(format!("save_session({mr_iid})"));
        }

        fn require_save_session(&mut self, mr_iid: u64) -> Result<()> {
            self.required("require_save_session", mr_iid)
        }

        fn require_save_session_with_summary(&mut self, mr_iid: u64, summary: &str) -> Result<()> {
            self.required(
                "require_save_session_with_summary",
                format!("{mr_iid},{summary}"),
            )
        }

        fn default_branch(&self) -> Result<String> {
            self.required("default_branch", "")?;
            Ok(self.default_branch.clone())
        }

        fn default_branch_or_main(&self) -> String {
            self.record("default_branch_or_main");
            self.default_branch.clone()
        }

        fn fetch_remote(&mut self) -> Result<()> {
            self.required("fetch_remote", "")
        }

        fn reset_worktree(&mut self) {
            self.record("reset_worktree");
        }

        fn remote_branch_exists(&self, branch: &str) -> Result<bool> {
            self.required("remote_branch_exists", branch)?;
            Ok(self.remote_branch_exists)
        }

        fn checkout_branch(&mut self, branch: &str) -> Result<()> {
            self.required("checkout_branch", branch)
        }

        fn checkout_branch_best_effort(&mut self, branch: &str) {
            self.record(format!("checkout_branch_best_effort({branch})"));
        }

        fn delete_local_branch(&mut self, branch: &str) {
            self.record(format!("delete_local_branch({branch})"));
        }

        fn delete_remote_branch(&mut self, branch: &str) {
            self.record(format!("delete_remote_branch({branch})"));
        }

        fn create_branch_from(&mut self, branch: &str, base: &str) -> Result<()> {
            self.required("create_branch_from", format!("{branch},{base}"))
        }

        fn merge_base_into_branch(&mut self, base: &str) -> Result<bool> {
            self.required("merge_base_into_branch", base)?;
            Ok(self.merge_clean)
        }

        fn has_diff_against(&self, base: &str) -> Result<bool> {
            self.required("has_diff_against", base)?;
            Ok(self.diff_answers.borrow_mut().pop_front().unwrap_or(true))
        }

        fn has_staged_changes(&self) -> Result<bool> {
            self.required("has_staged_changes", "")?;
            Ok(self.staged)
        }

        fn merge_request_state(&self, mr_iid: u64) -> Option<String> {
            self.record(format!("merge_request_state({mr_iid})"));
            self.existing_mr_state.clone()
        }

        fn dependency_closed(&self, issue_iid: u64) -> bool {
            self.record(format!("dependency_closed({issue_iid})"));
            self.dependency_closed
        }

        fn issue_comments(&self) -> String {
            self.record("issue_comments");
            "- alice: please cap it".to_string()
        }

        fn build_prompt(&mut self, continuation: bool, comments: &str) -> Result<String> {
            self.required("build_prompt", format!("{continuation},{comments}"))?;
            Ok(format!("prompt(continuation={continuation})"))
        }

        fn invoke_implementation_model(&mut self, prompt: &str) -> Result<ImplModelResult> {
            self.required("invoke_implementation_model", prompt)?;
            Ok(if self.model_cancelled {
                ImplModelResult::Cancelled
            } else {
                ImplModelResult::Output(Box::new(self.model_output.clone()))
            })
        }

        fn nudge_implementation_model(&mut self) -> Result<ImplModelResult> {
            self.required("nudge_implementation_model", "")?;
            Ok(ImplModelResult::Output(Box::new(self.model_output.clone())))
        }

        fn stage_all(&mut self) -> Result<()> {
            self.required("stage_all", "")
        }

        fn commit(&mut self, message: &str) -> Result<()> {
            self.required("commit", message)
        }

        fn push_branch(&mut self, branch: &str) -> Result<()> {
            self.required("push_branch", branch)
        }

        fn create_merge_request(
            &mut self,
            branch: &str,
            base: &str,
            title: &str,
            description: &str,
        ) -> Result<u64> {
            self.required(
                "create_merge_request",
                format!("{branch},{base},{title},{description}"),
            )?;
            Ok(self.created_mr_iid)
        }

        fn add_mr_scope_label(&mut self, mr_iid: u64) {
            self.record(format!("add_mr_scope_label({mr_iid})"));
        }

        fn hand_issue_back_to_humans(&mut self, branch: &str, reason: &str) -> Result<()> {
            self.required("hand_issue_back_to_humans", format!("{branch},{reason}"))
        }
    }

    struct FakeImplRun {
        result: Result<Option<u64>>,
        trace: Vec<String>,
        mr_created: bool,
        branch: Option<String>,
    }

    fn run_implementation(port: &mut FakeImplPort, scope_label: Option<&str>) -> FakeImplRun {
        let mut cycle = ImplementationCycleResult::default();
        let result = run_implementation_cycle(
            port,
            ROUTING_AGENT,
            scope_label,
            &issue_observation(7, &[]),
            &mut cycle,
        );
        FakeImplRun {
            result: result.map(|()| cycle.tracked_mr),
            trace: port.trace.borrow().clone(),
            mr_created: cycle.mr_created,
            branch: cycle.left_branch,
        }
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn implementation_prepares_the_branch_invokes_the_model_then_opens_the_merge_request() {
        let mut port = FakeImplPort::new();
        let run = run_implementation(&mut port, Some("team:core"));

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            run.trace,
            strings(&[
                "stop_if_review_only",
                "closes_linked_mr",
                "open_mr_for_issue",
                "default_branch()",
                "fetch_remote()",
                "reset_worktree",
                "remote_branch_exists(issue-7)",
                "create_branch_from(issue-7,main)",
                "stop_if_review_only",
                "require_working_on_label()",
                "issue_comments",
                "build_prompt(false,- alice: please cap it)",
                "invoke_implementation_model(prompt(continuation=false))",
                "stage_all()",
                "has_staged_changes()",
                "commit(Cap the retry backoff\n\nRefs #7)",
                "has_diff_against(main)",
                "push_branch(issue-7)",
                "create_merge_request(issue-7,main,Cap the retry backoff,Closes #7\n\nCapped the backoff at 30s.)",
                "add_mr_scope_label(12)",
                "require_save_session_with_summary(12,Capped the backoff at 30s.)",
            ])
        );
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
                .any(|step| step.starts_with("add_mr_scope_label("))
        );
    }

    #[test]
    fn implementation_adopts_a_merge_request_linked_by_a_closes_keyword() {
        let mut port = FakeImplPort::new().with_closes_linked(ClosesLinkedMr::Open(9));
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(9));
        assert_eq!(
            run.trace,
            strings(&[
                "stop_if_review_only",
                "closes_linked_mr",
                "add_working_on_label",
                "save_session(9)",
            ])
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
            strings(&[
                "stop_if_review_only",
                "closes_linked_mr",
                "close_issue",
                "cleanup_session",
            ])
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
            strings(&[
                "stop_if_review_only",
                "closes_linked_mr",
                "open_mr_for_issue",
                "add_working_on_label",
                "require_save_session(9)",
            ])
        );
    }

    #[test]
    fn implementation_continues_an_existing_branch_that_merges_cleanly() {
        let mut port = FakeImplPort::new().with_remote_branch();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            &run.trace[6..10],
            strings(&[
                "remote_branch_exists(issue-7)",
                "checkout_branch(issue-7)",
                "has_diff_against(main)",
                "merge_base_into_branch(main)",
            ])
        );
        assert!(
            run.trace
                .contains(&"build_prompt(true,- alice: please cap it)".to_string())
        );
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
            strings(&[
                "has_diff_against(main)",
                "reset_worktree",
                "checkout_branch(main)",
                "delete_local_branch(issue-7)",
                "delete_remote_branch(issue-7)",
                "create_branch_from(issue-7,main)",
            ])
        );
        assert!(
            run.trace
                .contains(&"build_prompt(false,- alice: please cap it)".to_string())
        );
    }

    #[test]
    fn implementation_recreates_a_branch_that_conflicts_with_the_default_branch() {
        let mut port = FakeImplPort::new().with_remote_branch().conflicting_merge();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert_eq!(
            &run.trace[9..14],
            strings(&[
                "merge_base_into_branch(main)",
                "reset_worktree",
                "checkout_branch(main)",
                "delete_local_branch(issue-7)",
                "create_branch_from(issue-7,main)",
            ])
        );
        assert!(
            !run.trace
                .iter()
                .any(|step| step.starts_with("delete_remote_branch("))
        );
    }

    #[test]
    fn implementation_stops_when_a_human_asks_for_review_only() {
        let mut before = FakeImplPort::new().stopping_at(&[true]);
        let run = run_implementation(&mut before, None);
        assert_eq!(run.result.unwrap(), None);
        assert_eq!(run.trace, strings(&["stop_if_review_only"]));
        assert!(run.branch.is_none());

        let mut after_prep = FakeImplPort::new().stopping_at(&[false, true]);
        let run = run_implementation(&mut after_prep, None);
        assert_eq!(run.result.unwrap(), None);
        assert_eq!(run.trace.last().unwrap(), "stop_if_review_only");
        assert_eq!(run.branch, Some("issue-7".to_string()));
    }

    #[test]
    fn implementation_ends_quietly_when_the_model_run_was_cancelled() {
        let mut port = FakeImplPort::new().cancelling_model();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), None);
        assert_eq!(
            run.trace.last().unwrap(),
            "invoke_implementation_model(prompt(continuation=false))"
        );
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
            assert!(
                run.trace
                    .last()
                    .unwrap()
                    .starts_with("hand_issue_back_to_humans(")
            );
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
            strings(&[
                "dependency_closed(4)",
                "add_issue_label(waiting-on-issue:#4)",
                "remove_working_on_label",
                "add_issue_comment(Implementation cannot proceed until issue #4 is closed. Parking this issue until the dependency resolves.)",
                "release_issue_claim",
                "cleanup_session",
                "default_branch_or_main",
                "reset_worktree",
                "checkout_branch_best_effort(main)",
                "delete_local_branch(issue-7)",
            ])
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
        assert!(run.trace.contains(
            &"create_merge_request(issue-7,main,Cap the retry backoff for issue 7,Closes #7\n\nImplementation completed.)"
                .to_string()
        ));
    }

    #[test]
    fn implementation_adopts_the_merge_request_the_model_pointed_at() {
        let mut port = FakeImplPort::new()
            .deciding(WorkerImplementationOutput::ExistingMr { existing_mr_iid: 9 });
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(9));
        assert_eq!(
            &run.trace[run.trace.len() - 7..],
            strings(&[
                "merge_request_state(9)",
                "reset_worktree",
                "default_branch_or_main",
                "checkout_branch_best_effort(main)",
                "delete_local_branch(issue-7)",
                "require_save_session(9)",
                "add_working_on_label",
            ])
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
            assert!(run.trace.contains(&"push_branch(issue-7)".to_string()));
        }
    }

    #[test]
    fn implementation_skips_the_commit_when_the_agent_committed_its_own_work() {
        let mut port = FakeImplPort::new().nothing_staged();
        let run = run_implementation(&mut port, None);

        assert_eq!(run.result.unwrap(), Some(12));
        assert!(!run.trace.iter().any(|step| step.starts_with("commit(")));
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
                .contains(&"nudge_implementation_model()".to_string())
        );
        assert_eq!(
            run.trace
                .iter()
                .filter(|step| step.starts_with("has_diff_against("))
                .count(),
            2
        );
        assert_eq!(run.branch, Some("issue-7".to_string()));
    }

    #[test]
    fn implementation_fails_the_run_when_a_required_step_fails() {
        let mut branching = FakeImplPort::new().failing("create_branch_from");
        let run = run_implementation(&mut branching, None);
        assert!(run.result.is_err());
        assert!(run.branch.is_none());

        let mut pushing = FakeImplPort::new().failing("push_branch");
        let run = run_implementation(&mut pushing, None);
        assert!(run.result.is_err());

        let mut labeling = FakeImplPort::new().failing("require_working_on_label");
        assert!(run_implementation(&mut labeling, None).result.is_err());

        let mut reading = FakeImplPort::new().failing("default_branch");
        assert!(run_implementation(&mut reading, None).result.is_err());
    }

    // -----------------------------------------------------------------
    // Direct feedback-tail traces. Payloads are captured as strings/tuples;
    // the fake deliberately has no replacement protocol enums.
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

    struct FakeFeedbackPort {
        trace: std::cell::RefCell<Vec<String>>,
        metadata: Vec<(String, String)>,
        replies: Vec<(String, String)>,
        resolutions: Vec<String>,
        plain_replies: Vec<String>,
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
        failing_operations: Vec<String>,
    }

    impl FakeFeedbackPort {
        fn new() -> Self {
            Self {
                trace: std::cell::RefCell::new(Vec::new()),
                metadata: Vec::new(),
                replies: Vec::new(),
                resolutions: Vec::new(),
                plain_replies: Vec::new(),
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
                failing_operations: Vec::new(),
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

        fn failing(mut self, operation: &str) -> Self {
            self.failing_operations.push(operation.to_string());
            self
        }

        fn record(&self, operation: impl Into<String>) {
            self.trace.borrow_mut().push(operation.into());
        }

        fn required(&self, operation: &str) -> Result<()> {
            self.record(operation);
            if self
                .failing_operations
                .iter()
                .any(|failed| failed == operation)
            {
                anyhow::bail!("injected failure in {operation}");
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
        fn update_mr_metadata(&mut self, title: &str, description: &str) {
            self.record(format!("update_mr_metadata({title}|{description})"));
            self.metadata
                .push((title.to_string(), description.to_string()));
        }

        fn fetch_branches(&mut self) -> Result<()> {
            self.required("fetch_branches")
        }

        fn has_changes_since(&self, base_ref: &str) -> Result<bool> {
            self.required(&format!("has_changes_since({base_ref})"))?;
            Ok(Self::next(&self.changes_answers, false))
        }

        fn merge_in_progress(&self) -> Result<bool> {
            self.required("merge_in_progress")?;
            Ok(self.merge_in_progress)
        }

        fn stage_resolved_conflicts(&mut self) -> Result<bool> {
            self.required("stage_resolved_conflicts")?;
            Ok(self.staged_conflicts)
        }

        fn stage_all(&mut self) -> Result<()> {
            self.required("stage_all")
        }

        fn has_staged_changes(&self) -> Result<bool> {
            self.required("has_staged_changes")?;
            Ok(Self::next(&self.staged_answers, true))
        }

        fn commit(&mut self, message: &str) -> Result<()> {
            self.required(&format!("commit({message})"))
        }

        fn complete_merge_if_ready(&mut self, message: &str) -> Result<bool> {
            self.required(&format!("complete_merge_if_ready({message})"))?;
            Ok(self.merge_completed)
        }

        fn merge_conflicts_present(&self) -> Result<bool> {
            self.required("merge_conflicts_present")?;
            Ok(Self::next(&self.conflict_answers, false))
        }

        fn up_to_date_with_target(&self, target_branch: &str) -> Result<bool> {
            self.required(&format!("up_to_date_with_target({target_branch})"))?;
            Ok(self.up_to_date)
        }

        fn diff_highlights(&self, base_ref: &str) -> Option<String> {
            self.record(format!("diff_highlights({base_ref})"));
            self.highlights.clone()
        }

        fn push_source_branch(&mut self) -> Result<()> {
            self.required("push_source_branch")
        }

        fn merge_request_surface(&self, mr_iid: u64) -> Result<MrSurfaceObservation> {
            self.required(&format!("merge_request_surface({mr_iid})"))?;
            Ok(self.surface_after.clone())
        }

        fn origin_head(&self, source_branch: &str) -> Option<String> {
            self.record(format!("origin_head({source_branch})"));
            self.origin_head.clone()
        }

        fn unresolved_discussion_ids(&self, mr_iid: u64) -> Vec<String> {
            self.record(format!("unresolved_discussion_ids({mr_iid})"));
            self.refetched_ids.clone()
        }

        fn reply_to_discussion(&mut self, discussion_id: &str, body: &str) {
            self.record(format!("reply({discussion_id}|{body})"));
            self.replies
                .push((discussion_id.to_string(), body.to_string()));
        }

        fn resolve_discussion(&mut self, discussion_id: &str) {
            self.record(format!("resolve({discussion_id})"));
            self.resolutions.push(discussion_id.to_string());
        }

        fn post_plain_comment(&mut self, body: &str) {
            self.record(format!("plain_comment({body})"));
            self.plain_replies.push(body.to_string());
        }
    }

    fn run_feedback(
        port: &mut FakeFeedbackPort,
        input: FeedbackTailInput,
    ) -> (Result<()>, Vec<String>) {
        let result = run_feedback_tail(port, &input);
        let trace = port.trace.borrow().clone();
        (result, trace)
    }

    #[test]
    fn feedback_commits_pushes_then_replies_and_resolves_in_order() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1", "d2"], addressed("Capped the backoff")),
        );

        assert!(result.is_ok());
        let commit = format!("commit({})", build_commit_message("Capped the backoff", 7));
        let push = trace
            .iter()
            .position(|step| step == "push_source_branch")
            .unwrap();
        let reply = trace
            .iter()
            .position(|step| step == "reply(d1|Capped the backoff)")
            .unwrap();
        assert!(trace.iter().position(|step| step == &commit).unwrap() < push);
        assert!(push < reply);
        assert_eq!(
            &trace[reply..],
            &[
                "reply(d1|Capped the backoff)",
                "resolve(d1)",
                "reply(d2|Capped the backoff)",
                "resolve(d2)",
            ]
        );
        assert_eq!(
            port.replies,
            vec![
                ("d1".to_string(), "Capped the backoff".to_string()),
                ("d2".to_string(), "Capped the backoff".to_string()),
            ]
        );
    }

    #[test]
    fn feedback_updates_metadata_before_touching_the_branch() {
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
                "update_mr_metadata(Cap the retry backoff at 30s|Closes #7)",
                "fetch_branches",
            ]
        );
        assert_eq!(
            port.metadata,
            vec![(
                "Cap the retry backoff at 30s".to_string(),
                "Closes #7".to_string(),
            )]
        );
    }

    #[test]
    fn feedback_stages_conflict_resolution_before_rechecking_changes() {
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
            &trace[..6],
            &[
                "fetch_branches",
                "has_changes_since(sha-before)",
                "merge_in_progress",
                "stage_resolved_conflicts",
                "stage_all",
                "has_changes_since(sha-before)",
            ]
        );
        assert!(trace.contains(&"push_source_branch".to_string()));
    }

    #[test]
    fn feedback_pushes_and_posts_nothing_while_required_conflicts_remain() {
        let mut input = feedback_input(&["d1"], addressed("Tried to resolve conflicts"));
        input.requires_conflict_resolution = true;
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .behind_target();
        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        assert!(!trace.contains(&"push_source_branch".to_string()));
        assert!(port.replies.is_empty());
        assert!(port.resolutions.is_empty());
        assert!(trace.contains(&"up_to_date_with_target(main)".to_string()));
    }

    #[test]
    fn feedback_treats_gitlab_reported_conflicts_as_unresolved() {
        let mut input = feedback_input(&["d1"], addressed("Capped the backoff"));
        input.requires_conflict_resolution = true;
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true, false])
            .with_surface_after(feedback_surface("Cap the retry backoff", true));
        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        assert!(trace.contains(&"push_source_branch".to_string()));
        assert!(port.replies.is_empty());
        assert!(port.resolutions.is_empty());
    }

    #[test]
    fn feedback_refetches_discussions_when_triggered_by_conflicts_alone() {
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
                "unresolved_discussion_ids(12)",
                "reply(conflict-thread|Merged main)",
                "resolve(conflict-thread)",
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
            trace.last().map(String::as_str),
            Some("reply(d1|The branch already handles this case)")
        );
        assert!(port.resolutions.is_empty());
    }

    #[test]
    fn feedback_resolves_without_branch_changes_when_explicitly_requested() {
        let resolution = FeedbackResolution {
            reason: Some("Already handled".to_string()),
            mark_discussions_resolved: Some(true),
            ..Default::default()
        };
        let mut port = FakeFeedbackPort::new();
        let (result, trace) = run_feedback(&mut port, feedback_input(&["d1"], resolution));

        assert!(result.is_ok());
        assert_eq!(trace.last().map(String::as_str), Some("resolve(d1)"));
    }

    #[test]
    fn feedback_posts_plain_comment_last() {
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
        assert_eq!(
            &trace[trace.len() - 3..],
            &[
                "reply(d1|Capped the backoff)",
                "resolve(d1)",
                "plain_comment(Capped the backoff)",
            ]
        );
        assert_eq!(port.plain_replies, vec!["Capped the backoff"]);
    }

    #[test]
    fn feedback_posts_plain_comment_without_discussions() {
        let resolution = FeedbackResolution {
            public_comment: Some("Acknowledged the plain comment.".to_string()),
            post_plain_comment: true,
            ..Default::default()
        };
        let mut input = feedback_input(&[], resolution);
        input.plain_comments_present = true;
        let mut port = FakeFeedbackPort::new();

        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        assert_eq!(port.plain_replies, vec!["Acknowledged the plain comment."]);
        assert_eq!(
            trace.last().map(String::as_str),
            Some("plain_comment(Acknowledged the plain comment.)")
        );
        assert!(port.replies.is_empty());
        assert!(port.resolutions.is_empty());
    }

    #[test]
    fn feedback_omits_plain_comment_when_not_requested() {
        let resolution = FeedbackResolution {
            public_comment: Some("No public reply requested.".to_string()),
            ..Default::default()
        };
        let mut input = feedback_input(&[], resolution);
        input.plain_comments_present = true;
        let mut port = FakeFeedbackPort::new();

        let (result, trace) = run_feedback(&mut port, input);

        assert!(result.is_ok());
        assert!(port.plain_replies.is_empty());
        assert!(!trace.iter().any(|step| step.starts_with("plain_comment(")));
    }

    #[test]
    fn feedback_omits_plain_comment_when_none_are_present() {
        let resolution = FeedbackResolution {
            public_comment: Some("Nothing to reply to.".to_string()),
            post_plain_comment: true,
            ..Default::default()
        };
        let mut port = FakeFeedbackPort::new();

        let (result, trace) = run_feedback(&mut port, feedback_input(&[], resolution));

        assert!(result.is_ok());
        assert!(port.plain_replies.is_empty());
        assert!(!trace.iter().any(|step| step.starts_with("plain_comment(")));
    }

    #[test]
    fn feedback_skips_commit_when_nothing_is_staged() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_staged(&[false])
            .with_origin_head("sha-after");
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1"], addressed("Committed it itself")),
        );

        assert!(result.is_ok());
        assert!(!trace.iter().any(|step| step.starts_with("commit(")));
        assert!(trace.contains(&"push_source_branch".to_string()));
    }

    #[test]
    fn feedback_holds_push_when_conflict_markers_remain() {
        let mut port = FakeFeedbackPort::new()
            .with_changes(&[true])
            .with_conflicts(&[true]);
        let (result, trace) = run_feedback(
            &mut port,
            feedback_input(&["d1"], addressed("Half-resolved")),
        );

        assert!(result.is_ok());
        assert!(!trace.contains(&"push_source_branch".to_string()));
        assert_eq!(
            trace
                .iter()
                .filter(|step| step.as_str() == "merge_conflicts_present")
                .count(),
            1
        );
        assert_eq!(
            trace.last().map(String::as_str),
            Some("reply(d1|Half-resolved)")
        );
    }

    #[test]
    fn feedback_fails_on_silent_no_op() {
        let mut port = FakeFeedbackPort::new();
        let (result, _) = run_feedback(
            &mut port,
            feedback_input(&["d1"], FeedbackResolution::default()),
        );

        let error = result.expect_err("a silent no-op run must fail the feedback tail");
        assert!(error.to_string().contains("no feedback reply"), "{error}");
    }

    #[test]
    fn feedback_fails_when_required_git_operations_fail() {
        let mut fetching = FakeFeedbackPort::new().failing("fetch_branches");
        assert!(
            run_feedback(&mut fetching, feedback_input(&["d1"], addressed("x")))
                .0
                .is_err()
        );

        let mut pushing = FakeFeedbackPort::new()
            .with_changes(&[true])
            .failing("push_source_branch");
        assert!(
            run_feedback(&mut pushing, feedback_input(&["d1"], addressed("x")))
                .0
                .is_err()
        );

        let mut probing = FakeFeedbackPort::new()
            .with_changes(&[true])
            .failing("merge_conflicts_present");
        assert!(
            run_feedback(&mut probing, feedback_input(&["d1"], addressed("x")))
                .0
                .is_err()
        );
    }

    #[test]
    fn worker_cycle_stops_before_polling_when_shutdown_was_requested() {
        let mut port = FakeWorkerPort::new()
            .listing(&[issue_observation(7, &[])])
            .with_shutdown_answers(&[true]);
        let run = run_worker_routing(&mut port, None);

        assert!(run.result.is_ok());
        assert_eq!(run.trace, strings(&["shutdown"]));
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
