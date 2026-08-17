use anyhow::{Context, Result};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

use super::{claim, issue_in_scope, labels, strip_internal_markers};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient, Issue, IssueThreadNote};
use crate::agents::workspace::{GitLabAgentBootstrap, gitlab_banner};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, ObjectSchema, SchemaField, StructuredOutput};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::model::acp::capabilities::{AskAnswer, AskQuestion, CapabilityProvider};
use crate::core::periodic::PeriodicTaskSpec;

const ACTION_REQUIRED_LABEL: &str = "action-required";
const PMO_PROCESSED_LABEL: &str = "pmo-processed";

/// One sub-issue as the model described it via the `plan` tool's `sub_issues`
/// array, before the empty-title/description defensive filtering in
/// [`normalize_sub_issues`] is applied.
#[derive(Debug, Clone, Default, Deserialize)]
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

/// Parse a JSON value as an issue IID, accepting numbers, numeric strings,
/// and strings with a leading `#` (e.g. `727`, `"727"`, `"#727"`). Returns
/// `None` for zero or non-numeric values.
fn parse_iid_value(val: &Value) -> Option<u64> {
    let n = val.as_u64().or_else(|| {
        let s = val.as_str()?;
        let s = s.trim().trim_start_matches('#').trim();
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u64>().ok()
    });
    n.filter(|n| *n > 0)
}

/// The model may name the dependency field differently, or send the IID as
/// a string (e.g. `"#727"`). Accept any of a set of plausible keys and parse
/// leading digits; zero/unparseable values become `None` (see
/// [`PmoOutput::WaitForDependency`]).
fn deserialize_dependency_iid<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(parse_iid_value(&value))
}

/// The PMO's typed structured-output contract. The model calls the `plan`
/// tool with its triage decision; core deserializes the captured JSON into
/// this type (see [`AgentModel::complete_typed`]).
#[derive(Debug, Clone)]
enum PmoOutput {
    GuideWorker {
        instructions: Option<String>,
    },
    Split {
        sub_issues: Vec<RawSubIssue>,
    },
    AlreadyDone {
        reason: Option<String>,
    },
    NeedsClarification {
        question: Option<String>,
        plan_text: Option<String>,
    },
    WaitForDependency {
        dependency_issue_iid: Option<u64>,
    },
}

#[derive(Deserialize)]
struct PmoOutputWire {
    decision: String,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    sub_issues: Vec<RawSubIssue>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    plan_text: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_dependency_iid",
        alias = "dependency_iid",
        alias = "depends_on_issue",
        alias = "blocked_by",
        alias = "dependency"
    )]
    dependency_issue_iid: Option<u64>,
}

impl<'de> Deserialize<'de> for PmoOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PmoOutputWire::deserialize(deserializer)?;
        match wire.decision.trim().to_ascii_lowercase().as_str() {
            "guide_worker" => Ok(Self::GuideWorker {
                instructions: wire.instructions,
            }),
            "split" => Ok(Self::Split {
                sub_issues: wire.sub_issues,
            }),
            "already_done" => Ok(Self::AlreadyDone {
                reason: wire.reason,
            }),
            "needs_clarification" => Ok(Self::NeedsClarification {
                question: wire.question,
                plan_text: wire.plan_text,
            }),
            "wait_for_dependency" => Ok(Self::WaitForDependency {
                dependency_issue_iid: wire.dependency_issue_iid,
            }),
            decision => Err(serde::de::Error::unknown_variant(
                decision,
                &[
                    "guide_worker",
                    "split",
                    "already_done",
                    "needs_clarification",
                    "wait_for_dependency",
                ],
            )),
        }
    }
}

impl StructuredOutput for PmoOutput {
    fn tool_name() -> &'static str {
        "plan"
    }

    fn tool_description() -> &'static str {
        "Emit your triage decision as structured JSON. This is the primary output channel — Potlatch reads the tool's JSON, not your streamed text. Call this exactly once with your decision and the fields relevant to it."
    }

    fn schema() -> ObjectSchema {
        ObjectSchema::new()
            .property(
                "decision",
                SchemaField::string_enum(
                    "Your triage decision. Must be exactly one of: \"guide_worker\", \"split\", \"already_done\", \"needs_clarification\", \"wait_for_dependency\".",
                    &[
                        "guide_worker",
                        "split",
                        "already_done",
                        "needs_clarification",
                        "wait_for_dependency",
                    ],
                ),
            )
            .property(
                "instructions",
                SchemaField::string(
                    "For guide_worker: 3-5 sentences, one clear action for the worker. Posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable.",
                ),
            )
            .property(
                "sub_issues",
                SchemaField::array(
                    "For split: the sub-issues to create. Each must have a title and description.",
                    SchemaField::object(
                        ObjectSchema::new()
                            .property("title", SchemaField::string("Concise sub-issue title."))
                            .property(
                                "description",
                                SchemaField::string(
                                    "Scope and acceptance criteria. If this sub-issue depends on another, reference it by title.",
                                ),
                            )
                            .property(
                                "priority",
                                SchemaField::integer_enum(
                                    "Priority: 1 (critical/blocking), 2 (high/depended-on), 3 (normal/independent).",
                                    &[1, 2, 3],
                                ),
                            )
                            .property(
                                "depends_on",
                                SchemaField::integer(
                                    "1-based index of another sub-issue this one depends on (omit or 0 if none).",
                                ),
                            )
                            .required("title")
                            .required("description"),
                    ),
                ),
            )
            .property(
                "reason",
                SchemaField::string("For already_done: why the codebase already satisfies the issue."),
            )
            .property(
                "question",
                SchemaField::string(
                    "For needs_clarification: specific questions for a human. Posted as a GitLab comment.",
                ),
            )
            .property(
                "plan_text",
                SchemaField::string(
                    "For needs_clarification: your current best plan for this issue. The system will update the issue description with this text so humans can see and refine your proposed approach. Write a structured plan including scope, proposed approach, and any open questions. Each refinement cycle overwrites the description with an improved version.",
                ),
            )
            .property(
                "dependency_issue_iid",
                SchemaField::integer(
                    "For wait_for_dependency: the IID (number) of the existing open issue this issue depends on and must wait for. Must be a positive integer.",
                ),
            )
            .required("decision")
    }
}

#[derive(Debug, Clone)]
struct PmoConfig {
    poll_interval_secs: u64,
    ask_via_gitlab: bool,
    ask_gitlab_timeout_secs: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PmoAgentSettings {
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

impl PmoAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        raw.clone()
            .try_into()
            .context("pmo agent settings from config")
    }
}

struct AgentState {
    sessions_dir: String,
    agent_id: String,
    project_name: String,
}

#[derive(Serialize, Deserialize)]
struct PersistedPmoState {
    claimed_issue_iid: u64,
}

impl AgentState {
    fn ensure_sessions_dir(&self) -> Result<()> {
        let ctx_dir = path::Path::new(&self.sessions_dir);
        fs::create_dir_all(ctx_dir).context("Failed to create .potlatch-context directory")?;

        Ok(())
    }

    fn state_path(&self) -> path::PathBuf {
        path::Path::new(&self.sessions_dir).join(format!("{}_state.json", &self.agent_id))
    }

    fn claim_label(&self) -> String {
        format!("claimed:{}", &self.agent_id)
    }

    fn save_state(&self, issue_iid: u64) {
        let store = crate::core::state::StateStore::new(self.state_path());
        if let Err(e) = store.save(&PersistedPmoState {
            claimed_issue_iid: issue_iid,
        }) {
            warn!("Failed to save PMO state: {}", e);
        }
    }

    fn clear_state(&self) {
        let store: crate::core::state::StateStore<PersistedPmoState> =
            crate::core::state::StateStore::new(self.state_path());
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
    state: AgentState,
    git_repo: GitRepo,
    gitlab: GitLabClient,
    model: AgentModel,
    config: PmoConfig,
    scope_label: String,
    claimed_issue_iid: Option<u64>,
}

impl CoreAgent for PmoAgent {
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        "pmo"
    }

    fn agent_id(&self) -> &str {
        &self.state.agent_id
    }

    fn shutdown(&self) -> &Arc<AtomicBool> {
        self.model.shutdown()
    }

    fn banner(config: &Config, banner: &mut Banner) {
        gitlab_banner(config, banner);
    }

    fn validate_config(config: &Config, section: &crate::core::config::AgentSection) -> Result<()> {
        super::settings::AgentSettings::from_config(config)?.require_gitlab_repo()?;
        PmoAgentSettings::from_raw(&section.raw)?;
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
                let scope = crate::agents::scope_label_filter(&self.scope_label);
                let model = &self.model;
                let shutdown = Arc::clone(model.shutdown());
                pmo_cycle(
                    &self.state,
                    &self.git_repo,
                    &self.gitlab,
                    model,
                    &mut self.claimed_issue_iid,
                    Arc::clone(&shutdown),
                    scope,
                    &self.config,
                )
            }
            _ => Ok(()),
        }
    }

    fn from_spawn(ctx: crate::core::workflow::AgentSpawnContext) -> Result<Self> {
        let section = ctx
            .workflow
            .config
            .agent("pmo")
            .context("[agent.pmo] section required")?;
        let settings = PmoAgentSettings::from_raw(&section.raw)?;
        let runtime = GitLabAgentBootstrap::new(
            &ctx,
            "pmo",
            ModelPreferences {
                structured_output_tools: Some(vec![PmoOutput::tool_definition()]),
                ..ModelPreferences::default()
            },
        )
        .build()?;
        let state = AgentState {
            sessions_dir: runtime.sessions_dir,
            agent_id: runtime.agent_id,
            project_name: runtime.project_name,
        };
        state.ensure_sessions_dir()?;
        let config = PmoConfig {
            poll_interval_secs: settings.poll_interval_secs,
            ask_via_gitlab: settings.ask_via_gitlab,
            ask_gitlab_timeout_secs: settings.ask_gitlab_timeout_secs,
        };
        let scope = crate::agents::scope_label_filter(&runtime.scope_label);
        let claimed_issue_iid = try_resume_pmo_state(&state, &runtime.gitlab, scope);
        if let Some(iid) = claimed_issue_iid {
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
        Ok(Self {
            state,
            git_repo: runtime.git_repo,
            gitlab: runtime.gitlab,
            model: runtime.model,
            config,
            scope_label: runtime.scope_label,
            claimed_issue_iid,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.state.agent_id);
        if let Some(issue_iid) = self.claimed_issue_iid {
            info!(
                "{}: Preserving claim on issue #{} for restart",
                self.state.agent_id, issue_iid
            );
            self.state.save_state(issue_iid);
        }
        info!("{}: Stopped", self.state.agent_id);
    }
}

#[allow(clippy::too_many_arguments)]
fn pmo_cycle(
    state: &AgentState,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    model: &AgentModel,
    claimed_issue_iid: &mut Option<u64>,
    shutdown: Arc<AtomicBool>,
    scope_label: Option<&str>,
    pmo_config: &PmoConfig,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;

    // Ensure we're on the latest upstream — hard reset if checkout fails
    if let Err(e) = git_repo.checkout_remote_branch(&default_branch) {
        warn!(
            "{}: Failed to checkout {}: {}, forcing reset",
            &state.agent_id, default_branch, e
        );

        let _ = git_repo.reset_hard();
        git_repo.checkout_remote_branch(&default_branch)?;
    }

    // If we still hold a claim from a previous run, decide what to do.
    if let Some(held_iid) = *claimed_issue_iid {
        // Pending splits take priority — handled below
        let pending_file_check = state.task_path();
        let has_pending = load_pending_split(&pending_file_check)?;
        if has_pending.is_some() {
            // Fall through to pending split handling
        } else if let Ok(issue) = gitlab.get_issue(held_iid) {
            if !issue_in_scope(&issue, scope_label) {
                info!(
                    "{}: Held issue #{} left scope label {:?}, releasing",
                    &state.agent_id, held_iid, scope_label
                );

                let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                *claimed_issue_iid = None;

                state.clear_state();
            } else if issue.labels.contains(&labels::PMO_PENDING.to_string()) {
                // PMO is in plan-refinement mode: the issue description holds
                // the PMO's draft plan, and the comment thread has the Q&A.
                // Re-triage only when there are new human comments since the
                // last PMO triage — otherwise just wait.
                let issues = gitlab.list_issues();
                match issues {
                    Ok(issues) => {
                        if let Err(e) =
                            refresh_pmo_issue_context_file(state, gitlab, &issue, &issues)
                        {
                            warn!(
                                "{}: Could not refresh PMO context file while pmo-pending on #{}: {}",
                                &state.agent_id, held_iid, e
                            );
                        }

                        if has_new_comments_since_last_pmo_comment(gitlab, &issue, &state.agent_id)
                        {
                            info!(
                                "{}: Issue #{} has new comments, re-triaging pmo-pending refinement",
                                &state.agent_id, held_iid
                            );
                            // Re-run triage with the refreshed context. The PMO
                            // may refine the plan (needs_clarification again) or
                            // reach a final decision. Either way, pmo-pending
                            // stays — only a human removes it.
                            match process_action_required_issue(
                                state,
                                gitlab,
                                model,
                                &issue,
                                &issues,
                                scope_label,
                                Arc::clone(&shutdown),
                                pmo_config,
                            ) {
                                Ok(keep_claim) => {
                                    if !keep_claim {
                                        // The PMO reached a final decision — but
                                        // pmo-pending is still on the issue (the
                                        // PMO never removes it). The decision
                                        // was already acted on inside
                                        // process_action_required_issue. Release
                                        // the claim so the normal flow continues.
                                        let _ =
                                            claim::release_claim(gitlab, held_iid, &state.agent_id);
                                        *claimed_issue_iid = None;
                                        state.clear_state();
                                    }
                                    // If keep_claim, the PMO refined the plan and
                                    // is still waiting. Keep the claim and wait
                                    // for the next cycle.
                                }
                                Err(e) => {
                                    warn!(
                                        "{}: Failed to re-triage pmo-pending issue #{}: {}",
                                        &state.agent_id, held_iid, e
                                    );
                                }
                            }
                        } else {
                            debug!(
                                "{}: Issue #{} still pmo-pending, no new comments, waiting",
                                &state.agent_id, held_iid
                            );
                        }
                    }
                    Err(e) => warn!(
                        "{}: Could not list issues to refresh context (pmo-pending #{}): {}",
                        &state.agent_id, held_iid, e
                    ),
                }

                return Ok(());
            } else {
                // No longer pending (human removed the label) — release claim so it
                // can be re-processed as a fresh action-required issue.
                info!(
                    "{}: Releasing claim on issue #{} from previous run",
                    &state.agent_id, held_iid
                );

                let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                *claimed_issue_iid = None;

                state.clear_state();
            }
        } else {
            // Can't fetch issue — release to be safe
            let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
            *claimed_issue_iid = None;

            state.clear_state();
        }
    }

    let pending_file = state.task_path();
    if let Some(pending_split) = load_pending_split(&pending_file)? {
        info!(
            "{}: Resuming pending split for issue #{} ({} sub-issues remaining)",
            &state.agent_id,
            pending_split.parent_issue_iid,
            pending_split.sub_issues.len()
        );

        match (
            gitlab.get_issue(pending_split.parent_issue_iid),
            gitlab.list_issues(),
        ) {
            (Ok(parent_issue), Ok(issues)) => {
                if let Err(e) =
                    refresh_pmo_issue_context_file(state, gitlab, &parent_issue, &issues)
                {
                    warn!(
                        "{}: Could not refresh PMO context file before pending split on #{}: {}",
                        &state.agent_id, pending_split.parent_issue_iid, e
                    );
                }
            }
            (Err(e), _) => warn!(
                "{}: Could not fetch parent issue #{} for context refresh: {}",
                &state.agent_id, pending_split.parent_issue_iid, e
            ),
            (_, Err(e)) => warn!(
                "{}: Could not list issues for context refresh (pending split): {}",
                &state.agent_id, e
            ),
        }

        match resume_split(&pending_file, gitlab, &pending_split, scope_label) {
            Ok(_) => {
                info!(
                    "{}: Successfully completed pending split for issue #{}",
                    &state.agent_id, pending_split.parent_issue_iid
                );

                delete_pending_split(&pending_file)?;
                // Release the claim from the split
                if let Some(held_iid) = *claimed_issue_iid {
                    let _ = claim::release_claim(gitlab, held_iid, &state.agent_id);
                    *claimed_issue_iid = None;

                    state.clear_state();
                }
            }
            Err(e) => {
                error!(
                    "{}: Failed to complete pending split: {}",
                    &state.agent_id, e
                );
            }
        }

        return Ok(());
    }

    let issues = gitlab.list_issues()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    for issue in &issues {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !should_process_issue(issue, scope_label) {
            continue;
        }

        if claim::is_claimed(&issue.labels) {
            debug!(
                "{}: Issue #{} already claimed, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if !claim::try_claim_issue(gitlab, issue.iid, &state.agent_id, shutdown.as_ref())? {
            info!(
                "{}: Failed to claim issue #{}, skipping",
                &state.agent_id, issue.iid
            );
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            let _ = claim::release_claim(gitlab, issue.iid, &state.agent_id);
            return Ok(());
        }

        *claimed_issue_iid = Some(issue.iid);

        state.save_state(issue.iid);

        info!(
            "{}: Processing issue #{}: {}",
            &state.agent_id, issue.iid, issue.title
        );

        match process_action_required_issue(
            state,
            gitlab,
            model,
            issue,
            &issues,
            scope_label,
            Arc::clone(&shutdown),
            pmo_config,
        ) {
            Ok(keep_claim) => {
                info!(
                    "{}: Successfully processed issue #{}",
                    &state.agent_id, issue.iid
                );
                if keep_claim {
                    info!(
                        "{}: Keeping claim on issue #{} (pmo-pending)",
                        &state.agent_id, issue.iid
                    );
                } else {
                    claim::release_claim(gitlab, issue.iid, &state.agent_id)?;
                    *claimed_issue_iid = None;

                    state.clear_state();
                }
            }
            Err(e) => {
                error!(
                    "{}: Failed to process issue #{}: {}",
                    &state.agent_id, issue.iid, e
                );

                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }

                claim::release_claim(gitlab, issue.iid, &state.agent_id)?;
                *claimed_issue_iid = None;

                state.clear_state();
            }
        }

        break;
    }

    // Assign default priority to open issues that lack a priority label.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    assign_default_priority(&state.agent_id, &issues, gitlab, scope_label);

    // Close stale pmo-processed issues that have not been picked up for over 1 hour.
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    close_stale_processed_issues(&state.agent_id, &issues, gitlab, scope_label);

    Ok(())
}

fn assign_default_priority(
    agent_id: &str,
    issues: &[Issue],
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) {
    for issue in issues {
        if issue.state != "opened" {
            continue;
        }
        if !issue_in_scope(issue, scope_label) {
            continue;
        }
        let has_priority = issue
            .labels
            .iter()
            .any(|l| l.starts_with(gitlab::PRIORITY_LABEL_PREFIX));
        if !has_priority {
            let label = gitlab::priority_label(gitlab::DEFAULT_PRIORITY);
            debug!(
                "{}: Assigning default {} to issue #{}",
                agent_id, label, issue.iid
            );
            let _ = gitlab.add_issue_label(issue.iid, &label);
        }
    }
}

const STALE_THRESHOLD_SECS: u64 = 3600; // 1 hour

fn close_stale_processed_issues(
    agent_id: &str,
    issues: &[Issue],
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    for issue in issues {
        if issue.state != "opened" {
            continue;
        }
        if !issue_in_scope(issue, scope_label) {
            continue;
        }
        if !issue.labels.contains(&PMO_PROCESSED_LABEL.to_string()) {
            continue;
        }

        let Some(ref updated_str) = issue.updated_at else {
            continue;
        };

        let Some(updated_epoch) = parse_iso8601_to_epoch(updated_str) else {
            debug!(
                "{}: Could not parse updated_at for issue #{}: {}",
                agent_id, issue.iid, updated_str
            );
            continue;
        };

        if now.saturating_sub(updated_epoch) >= STALE_THRESHOLD_SECS {
            info!(
                "{}: Issue #{} has been pmo-processed for over 1 hour with no activity, closing",
                agent_id, issue.iid
            );
            let _ = gitlab.add_issue_comment(
                issue.iid,
                "Closing this issue — it has been marked as `pmo-processed` for over 1 hour with no further activity.",
            );
            let _ = gitlab.close_issue(issue.iid);
        }
    }
}

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

/// Returns `Ok(true)` if the PMO should keep its claim (pmo-pending / needs clarification).
/// Returns `Ok(false)` if the claim can be released.
#[allow(clippy::too_many_arguments)]
fn process_action_required_issue(
    state: &AgentState,
    gitlab: &GitLabClient,
    model: &AgentModel,
    issue: &Issue,
    all_issues: &[Issue],
    scope_label: Option<&str>,
    shutdown: Arc<AtomicBool>,
    pmo_config: &PmoConfig,
) -> Result<bool> {
    let parent_priority = issue.priority();
    let context_path = refresh_pmo_issue_context_file(state, gitlab, issue, all_issues)?;
    let prompt = build_split_prompt(state, issue, &context_path, parent_priority)?;

    let provider: Option<
        std::sync::Arc<dyn crate::core::model::acp::capabilities::CapabilityProvider>,
    > = if pmo_config.ask_via_gitlab {
        let timeout = if pmo_config.ask_gitlab_timeout_secs > 0 {
            Some(std::time::Duration::from_secs(
                pmo_config.ask_gitlab_timeout_secs,
            ))
        } else {
            None
        };
        Some(std::sync::Arc::new(GitLabIssueAskHandler::new(
            issue.iid,
            gitlab.clone(),
            Arc::clone(&shutdown),
            timeout,
        )))
    } else {
        None
    };

    model.set_capability_provider(provider);

    info!(
        "{}: PMO agent triaging issue #{}",
        &state.agent_id, issue.iid
    );
    let completion = model.complete_typed::<PmoOutput>(
        &prompt,
        &InvokeOptions {
            activity_label: Some(format!("{} triaging issue #{}", &state.agent_id, issue.iid)),
            ..InvokeOptions::default()
        },
    )?;
    model.set_capability_provider(None);
    info!(
        "{}: PMO agent finished triaging issue #{}",
        &state.agent_id, issue.iid
    );

    let sub_issues = match completion.output {
        // --- NEEDS_CLARIFICATION: PMO needs human input, refine plan ---
        PmoOutput::NeedsClarification {
            question,
            plan_text,
        } => {
            let question = strip_internal_markers(&clarification_question_or_default(question));
            info!(
                "PMO: Issue #{} needs clarification, marking pmo-pending and updating plan",
                issue.iid
            );

            // Update the issue description with the PMO's current plan draft, so
            // humans can see and refine the proposed approach directly in the
            // GitLab issue body.
            if let Some(plan_text) = plan_text.filter(|t| !t.trim().is_empty()) {
                let cleaned = strip_internal_markers(plan_text.trim());
                if !cleaned.is_empty()
                    && let Err(e) = gitlab.update_issue_description(issue.iid, &cleaned)
                {
                    warn!(
                        "PMO: Failed to update issue #{} description with plan: {}",
                        issue.iid, e
                    );
                }
            }

            gitlab.add_issue_comment(
                issue.iid,
                &format!(
                    "**PMO needs clarification before proceeding:**\n\n{}\n\n\
                     Please reply to this comment with the requested information. \
                     The PMO will refine the plan based on your feedback. \
                     Remove the `pmo-pending` label when you are satisfied with the plan to let the PMO proceed.",
                    question
                ),
            )?;

            gitlab.add_issue_label(issue.iid, labels::PMO_PENDING)?;
            return Ok(true); // keep claim
        }

        // --- ALREADY_DONE: work is already implemented, close the issue ---
        PmoOutput::AlreadyDone { reason } => {
            let reason = strip_internal_markers(&already_done_reason_or_default(reason));
            info!("PMO: Issue #{} is already implemented, closing", issue.iid);
            gitlab.add_issue_comment(
                issue.iid,
                &format!(
                    "**PMO: Closing — this work is already implemented.**\n\n{}",
                    reason
                ),
            )?;
            let _ = gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL);
            let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
            gitlab.close_issue(issue.iid)?;
            return Ok(false);
        }

        // --- WAIT_FOR_DEPENDENCY: issue depends on another open issue, park it ---
        PmoOutput::WaitForDependency {
            dependency_issue_iid,
        } => {
            let Some(dep_iid) = dependency_issue_iid else {
                warn!(
                    "PMO: wait_for_dependency decision for issue #{} but no dependency_issue_iid provided, releasing claim for retry",
                    issue.iid
                );
                anyhow::bail!(
                    "PMO wait_for_dependency for issue #{} missing dependency_issue_iid; retrying later",
                    issue.iid
                );
            };
            info!(
                "PMO: Issue #{} depends on open issue #{}, parking",
                issue.iid, dep_iid
            );
            let label = format!("waiting-on-issue:#{dep_iid}");
            let _ = gitlab.add_issue_label(issue.iid, &label);
            let _ = gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL);
            let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
            gitlab.add_issue_comment(
                issue.iid,
                &format!(
                    "**PMO: Parking — this issue depends on issue #{dep_iid} which is still open.**\n\n\
                     The worker will skip this issue until #{} is closed, then resume automatically.",
                    dep_iid
                ),
            )?;
            return Ok(false);
        }

        // --- GUIDE_WORKER: single focused retry instruction ---
        PmoOutput::GuideWorker { instructions } => {
            let guidance = strip_internal_markers(&guidance_or_empty(instructions));
            if guidance.trim().is_empty() {
                warn!(
                    "PMO: GUIDE_WORKER output for issue #{} had no usable guidance, releasing claim for retry",
                    issue.iid
                );
                anyhow::bail!(
                    "PMO GUIDE_WORKER output for issue #{} had no usable guidance; retrying later",
                    issue.iid
                );
            }
            info!("PMO: Issue #{} needs guidance, not splitting", issue.iid);
            gitlab.add_issue_comment(issue.iid, &format_pmo_guidance_comment(&guidance))?;
            gitlab.remove_issue_label(issue.iid, ACTION_REQUIRED_LABEL)?;
            let _ = gitlab.remove_issue_label(issue.iid, PMO_PROCESSED_LABEL);
            return Ok(false);
        }

        // --- SPLIT: create sub-issues, close the parent as a task container ---
        PmoOutput::Split { sub_issues } => normalize_sub_issues(sub_issues),
    };

    if sub_issues.is_empty() {
        warn!("PMO: No sub-issues from plan tool for issue #{}", issue.iid);

        anyhow::bail!(
            "PMO split output for issue #{} was not machine-readable; retrying later",
            issue.iid
        );
    }

    let pending_file = state.task_path();

    let pending_split = PendingSplit {
        parent_issue_iid: issue.iid,
        parent_issue_title: issue.title.clone(),
        parent_priority: issue.priority(),
        sub_issues: sub_issues.clone(),
        created_issue_ids: Vec::new(),
    };

    save_pending_split(&pending_file, &pending_split)?;

    match resume_split(&pending_file, gitlab, &pending_split, scope_label) {
        Ok(_) => {
            info!(
                "{}: Successfully split issue #{} into sub-issues",
                &state.agent_id, issue.iid
            );
            delete_pending_split(&pending_file)?;
        }
        Err(e) => {
            error!(
                "{}: Failed to create all sub-issues: {}",
                &state.agent_id, e
            );
            return Err(e);
        }
    }

    // Close the parent issue — it served as a task container, the real work
    // is now tracked in the sub-issues.
    info!(
        "{}: Closing parent issue #{} (task container)",
        &state.agent_id, issue.iid
    );

    let _ = gitlab.add_issue_comment(
        issue.iid,
        "Closing this issue — it has been split into sub-issues above. \
         The sub-issues now track the actual work.",
    );

    let _ = gitlab.close_issue(issue.iid);

    Ok(false)
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
) -> Result<String> {
    let prompt = format!(
        r#"You are a Project Management Office (PMO) agent responsible for triaging issues that an automated worker agent could not implement.

PROJECT: {project}

ISSUE #{iid}: {title}  _(summary only — not sufficient by itself)_

TASK CONTEXT FILE (you MUST open and read this path on disk — it has the full picture):
{context_path}

CONTEXT:
An automated worker agent attempted to implement this issue but was unable to complete it.
Potlatch wrote the path above as a markdown file: **full issue description**, **every GitLab issue comment** (including worker rejection / PMO notes), **closed merge request context** (MR comments, reviewer feedback, and diff from the worker's closed MR, when one exists), and **the list of other open issues**. That file is the authoritative written context for this triage.
- Use your **file-reading** capability on the absolute path and read it **end-to-end** before you decide the situation is unclear.
- The single line `ISSUE #…: title` in this prompt is **not** a substitute for the file; do not claim "no context" or choose NEEDS_CLARIFICATION only because you did not read the task context file.
- The **Closed merge request context** section is especially important when the worker closed an MR after failing to resolve reviewer feedback — the MR comments and diff show what the reviewer asked for and what the worker tried.
Your job is to analyze the failure reason (from the file + repo when needed) and take the appropriate action.

CRITICAL REQUIREMENTS:
- This is a NON-INTERACTIVE automated system
- You do NOT write new production code — you inspect the existing project state, then write comments and create issue descriptions as needed
- The worker agent has FULL ACCESS to shell commands (rm, mv, git, etc.) and all build/test tools
- If the worker claimed it "cannot run commands" or "cannot delete files", that is WRONG — it CAN. Instruct it clearly.
- Before deciding GUIDE_WORKER or SPLIT, you MUST verify whether the issue is already implemented in the current project state when that is plausible from the issue, comments, or worker output. If the behavior/tests/code already exist, choose ALREADY_DONE so Potlatch will close the issue and add a comment.

## Output — call the `plan` tool with your decision

Call the `plan` tool with your decision as its arguments. Potlatch reads the tool's arguments directly — your streamed text is ignored for decision parsing. This is the ONLY output channel; you MUST call `plan` with your decision. The tool's parameters are:

- `decision` (required): one of `guide_worker`, `split`, `already_done`, `needs_clarification`, `wait_for_dependency`
- `instructions` (for guide_worker): 3-5 sentences, one clear action for the worker — posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable.
- `sub_issues` (for split): array of `{{"title": "...", "description": "...", "priority": 1-3, "depends_on": N}}` where `depends_on` is the 1-based index of another sub-issue this one depends on (omit or 0 if none). Example: if sub-issue 2 builds on sub-issue 1's work, set `"depends_on": 1`. The system automatically labels dependent issues so the worker won't start them until their dependency is closed.
- `reason` (for already_done): why the codebase already satisfies the issue
- `question` (for needs_clarification): specific questions for a human. Posted as a GitLab comment.
- `plan_text` (for needs_clarification): your current best plan for this issue. The system updates the issue description with this text so humans can see and refine your proposed approach. Write a structured plan including scope, proposed approach, and any open questions. Each refinement cycle overwrites the description with an improved version.
- `dependency_issue_iid` (for wait_for_dependency): the IID of the existing open issue this issue depends on and must wait for

Only fill the parameter(s) relevant to your decision. Everything you put in `instructions`, `reason`, or `question` is human-facing and posted to GitLab verbatim — do not include any internal markers, harness instructions, or meta-commentary.

DECISION — choose EXACTLY ONE of the following:

1. GUIDE_WORKER — Use ONLY when ALL of these are true:
   a) The issue describes a SINGLE, focused task (not a list of modules/files/components)
   b) The worker failed due to a specific misunderstanding, wrong command, or simple technical obstacle
   c) The fix is ONE clear action (e.g. "use flag X instead of Y", "the config file is at path Z")
   If your guidance would enumerate 2+ independent modules, files, or components, you MUST choose SPLIT instead.

2. SPLIT — Use when ANY of these are true:
   - The issue is a "task container" describing a broad goal (e.g. "add tests for module X", "refactor all Y", "check full code for Z") — these ALWAYS need splitting into concrete sub-tasks
   - The issue involves work on 2+ independent modules, files, or components
   - The issue is too large (estimated >500 lines of non-test code, or >1500 lines total including tests; auto-generated code does not count)
   - The issue description or worker rejection lists multiple distinct things to do
   - Your guidance would need to enumerate 2+ independent items
   Do NOT force a speculative split. If you cannot clearly define the sub-issues with concrete titles and descriptions, choose NEEDS_CLARIFICATION instead and draft your best understanding of the decomposition in `plan_text`.
   When splitting, the PARENT ISSUE will be CLOSED automatically as a task container. The sub-issues become the real tracked work.
   Each sub-issue should target ~500 lines of non-test code, ~1500 total including tests; auto-generated code does not count.

   PRIORITY LEVELS:
   - 1 = Critical: blocking other work, security fix, core dependency that other sub-issues depend on
   - 2 = High: important feature, depended on by lower-priority sub-issues
   - 3 = Normal: independent work, enhancements, nice-to-haves
   The parent issue has priority {parent_priority}. Sub-issues that are dependencies for others should get higher priority (lower number). Independent leaf tasks can inherit the parent priority or be lower.

   DEPENDENCIES:
   When a sub-issue cannot be started until another sub-issue is done, set `"depends_on"` to the 1-based index of the dependency. Example: if sub-issue 2 builds on the primitives from sub-issue 1, set `"depends_on": 1` on sub-issue 2. The system parks sub-issue 2 (labels it `waiting-on-issue:#N`) so the worker skips it until sub-issue 1 is closed. Use this for real build-order dependencies — do not set `depends_on` for issues that are merely related or could run in parallel.

3. ALREADY_DONE — Use when the work described in the issue is ALREADY fully implemented in the codebase:
   - The worker's output or your analysis shows the feature/tests/code already exists
   - There is nothing left to implement — the issue is simply outdated or redundant
   - Prefer ALREADY_DONE over GUIDE_WORKER or SPLIT when the required behavior is already present in the repository as it exists now

4. NEEDS_CLARIFICATION — Use when you need more information from a human, OR when you are unsure how to decompose the issue:
   - After reading the **task context file** and (if needed) the repo, the issue is still too vague to determine scope or intent
   - The worker's rejection and the issue (as given in that file) still don't give enough to guide or split
   - You need specific information from a human (e.g. which modules to cover, what the acceptance criteria are)
   - You are unsure how to split the issue into well-defined sub-issues
   Do **not** use this option because you skipped reading the task context file.
   When you choose NEEDS_CLARIFICATION, write your current best plan in `plan_text` — the system will update the issue description so the human can see your proposed approach. Post your specific questions in `question` — they'll appear as a comment. You may be re-triaged multiple times as the human replies; each time, refine `plan_text` with your updated understanding. The `pmo-pending` label stays until a human removes it — when you reach a confident decision during refinement, still use `needs_clarification` and describe your recommendation in `question`. The human will remove the label to trigger final processing.

5. WAIT_FOR_DEPENDENCY — Use when this issue cannot be implemented until another EXISTING OPEN issue is closed:
   - The dependency is a real build-order blocker (the code from the other issue is a prerequisite)
   - Set `dependency_issue_iid` to the IID of the blocking issue (it must appear in the EXISTING OPEN ISSUES list)
   - The system parks this issue (labels it `waiting-on-issue:#N`) so the worker skips it until the dependency closes, then resumes automatically
   - Do NOT use this for issues that are merely related or could run in parallel — only for real prerequisites
   - Do NOT use this as a substitute for SPLIT's `depends_on` (that's for sub-issues you're creating now); use WAIT_FOR_DEPENDENCY only when the dependency is an already-existing separate issue

DUPLICATE / OVERLAP RULES (STRICT):
- Review the EXISTING OPEN ISSUES list above before creating any sub-issue.
- Do NOT create a sub-issue that duplicates or substantially overlaps with an existing open issue.
- If an existing OPEN (not IN-PROGRESS) issue covers part of the work, reference it (e.g. "See existing #42") instead of creating a new sub-issue for that part.
- If an existing IN-PROGRESS issue already covers it, simply skip that part entirely — do not create a sub-issue or reference.
- If ALL sub-issues would duplicate existing issues, choose GUIDE_WORKER instead and tell the worker which existing issues already cover the work.

INSTRUCTIONS:
1. Open and read the **entire** TASK CONTEXT FILE at the absolute path above (description, GitLab comments, closed MR context, existing issues). Do this first.
2. From that file, read the issue description and **all** comments — especially the worker's rejection reason. If there is a **Closed merge request context** section, read the MR comments and diff to understand what the reviewer asked for and what the worker tried.
3. Review the EXISTING OPEN ISSUES section in that same file to see what is already tracked.
4. TASK CONTAINER TEST: Does the issue describe a broad goal that involves multiple independent pieces of work (e.g. "add tests for all modules", "refactor X across the codebase", "check code for Y")? If YES → SPLIT. The parent issue is just a container; the real work is in the sub-issues.
5. GUIDANCE TEST: Is there ONE specific thing the worker misunderstood or did wrong? If YES → GUIDE_WORKER.
6. ENUMERATION TEST: If your guidance would list 2+ independent modules, files, or components → SPLIT, not GUIDE_WORKER.
7. CLARITY TEST: Only if the task context file plus (if needed) repo inspection still leaves intent unclear → NEEDS_CLARIFICATION.
8. COMPLETION TEST: Does the worker's output, the comments in the file, or your direct inspection of the current project state indicate the work is already fully implemented in the codebase? If YES → ALREADY_DONE.
9. DEPENDENCY TEST: Does this issue require code from another EXISTING OPEN issue to be implemented first? If YES → WAIT_FOR_DEPENDENCY (set `dependency_issue_iid`).
10. Choose EXACTLY ONE of GUIDE_WORKER, SPLIT, ALREADY_DONE, NEEDS_CLARIFICATION, or WAIT_FOR_DEPENDENCY — never combine them.
11. When in doubt between GUIDE_WORKER and SPLIT, prefer SPLIT — it's better to create focused sub-issues than to give the worker a laundry list.
12. Call the `plan` tool with the JSON shape above. This is the last step — your turn is not complete until you call `plan`.

Proceed with analyzing the issue autonomously.
"#,
        project = &state.project_name,
        iid = issue.iid,
        title = issue.title,
        context_path = context_path,
        parent_priority = parent_priority,
    );

    Ok(prompt)
}

#[derive(Debug, Serialize, Deserialize)]
struct PendingSplit {
    parent_issue_iid: u64,
    parent_issue_title: String,
    #[serde(default)]
    parent_priority: u8,
    sub_issues: Vec<PmoSubIssue>,
    created_issue_ids: Vec<u64>,
}

// --- Field extractors (typed `PmoOutput` fields only) ---

fn already_done_reason_or_default(reason: Option<String>) -> String {
    reason
        .map(|r| r.trim().to_string())
        .filter(|r| !r.is_empty())
        .unwrap_or_else(|| {
            "The work described in this issue is already fully implemented in the codebase."
                .to_string()
        })
}

fn clarification_question_or_default(question: Option<String>) -> String {
    question
        .map(|q| q.trim().to_string())
        .filter(|q| !q.is_empty())
        .unwrap_or_else(|| {
            "The PMO agent could not determine how to proceed with this issue. Please provide more details about the expected scope and acceptance criteria.".to_string()
        })
}

fn guidance_or_empty(instructions: Option<String>) -> String {
    instructions
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .map(|t| cap_guidance_length(&t))
        .unwrap_or_default()
}

/// Check whether there are new human comments on the issue since the last
/// PMO comment. Used by the reconcile phase to decide whether to re-triage
/// a `pmo-pending` issue. Comments are in chronological order from the
/// GitLab discussions API. Returns `true` if any comment after the last
/// PMO-authored comment was written by a different author.
fn has_new_comments_since_last_pmo_comment(
    gitlab: &GitLabClient,
    issue: &Issue,
    pmo_agent_id: &str,
) -> bool {
    let Ok(comments) = gitlab.get_issue_comments(issue.iid) else {
        return false;
    };

    // Find the index of the last PMO-authored comment.
    let last_pmo_idx = comments
        .iter()
        .rposition(|c| c.author == pmo_agent_id || c.body.contains("**PMO needs clarification"));

    match last_pmo_idx {
        Some(idx) => {
            // Any non-system comment after the last PMO comment is a "new" human comment.
            comments[idx + 1..]
                .iter()
                .any(|c| !c.author.is_empty() && c.author != pmo_agent_id)
        }
        None => {
            // No PMO comment found — if there are any human comments, they're all "new".
            comments
                .iter()
                .any(|c| !c.author.is_empty() && c.author != pmo_agent_id)
        }
    }
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
    crate::core::state::StateStore::new(path)
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
    crate::core::state::StateStore::new(path)
        .load()
        .context("Failed to load pending split file")
}

fn delete_pending_split(path: &str) -> Result<()> {
    let store: crate::core::state::StateStore<PendingSplit> =
        crate::core::state::StateStore::new(path);
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
) -> Option<u64> {
    // Invalid persisted PMO state has historically been ignored.
    let persisted: PersistedPmoState = crate::core::state::StateStore::new(state.state_path())
        .load()
        .ok()
        .flatten()?;
    let issue_iid = persisted.claimed_issue_iid;

    let claim_label = state.claim_label();
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
            if !issue.labels.contains(&claim_label) {
                info!(
                    "{}: Claim label missing from issue #{}, discarding state",
                    &state.agent_id, issue_iid
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

            info!("{}: Resumed claim on issue #{}", &state.agent_id, issue_iid);

            Some(issue_iid)
        }
        Err(e) => {
            warn!(
                "{}: Failed to verify issue #{}: {}, discarding state",
                &state.agent_id, issue_iid, e
            );

            state.clear_state();

            None
        }
    }
}

fn resume_split(
    pending_file: &str,
    gitlab: &GitLabClient,
    pending: &PendingSplit,
    scope_label: Option<&str>,
) -> Result<()> {
    let mut created_issue_ids = pending.created_issue_ids.clone();
    let total_sub_issues = pending.sub_issues.len();
    let already_created = created_issue_ids.len();

    // Create remaining sub-issues
    for (index, sub_issue) in pending.sub_issues.iter().enumerate() {
        // Skip already created sub-issues
        if index < already_created {
            debug!(
                "PMO: Skipping already created sub-issue {}/{}",
                index + 1,
                total_sub_issues
            );
            continue;
        }

        let sub_issue_title = if sub_issue.title.is_empty() {
            &pending.parent_issue_title
        } else {
            &sub_issue.title
        };

        let resolved_description = sub_issue.description.clone();

        match gitlab.create_issue(sub_issue_title, &resolved_description) {
            Ok(sub_issue_iid) => {
                info!(
                    "PMO: Created sub-issue #{}: {}",
                    sub_issue_iid, sub_issue_title
                );

                let p = sub_issue.priority.unwrap_or(pending.parent_priority);
                if let Err(e) = gitlab.add_issue_label(sub_issue_iid, &gitlab::priority_label(p)) {
                    warn!(
                        "PMO: Failed to set priority label on #{}: {}",
                        sub_issue_iid, e
                    );
                }

                if let Some(lbl) = scope_label
                    && let Err(e) = gitlab.add_issue_label(sub_issue_iid, lbl)
                {
                    warn!(
                        "PMO: Failed to add scope label {:?} on #{}: {}",
                        lbl, sub_issue_iid, e
                    );
                }

                // If this sub-issue declares a dependency, add a temporary
                // `do-not-implement` label so the worker skips it immediately.
                // The label is swapped for the real `waiting-on-issue:#N` label
                // in the post-loop pass once all IIDs are known. This closes
                // the race: the issue is blocked from the moment it's created.
                if sub_issue.depends_on > 0
                    && let Err(e) = gitlab.add_issue_label(sub_issue_iid, labels::DO_NOT_IMPLEMENT)
                {
                    warn!(
                        "PMO: Failed to add temporary do-not-implement label to sub-issue #{sub_issue_iid}: {e}"
                    );
                }

                created_issue_ids.push(sub_issue_iid);

                let updated_pending = PendingSplit {
                    parent_issue_iid: pending.parent_issue_iid,
                    parent_issue_title: pending.parent_issue_title.clone(),
                    parent_priority: pending.parent_priority,
                    sub_issues: pending.sub_issues.clone(),
                    created_issue_ids: created_issue_ids.clone(),
                };
                save_pending_split(pending_file, &updated_pending)?;
            }
            Err(e) => {
                error!(
                    "PMO: Failed to create sub-issue {}/{}: {}",
                    index + 1,
                    total_sub_issues,
                    e
                );
                return Err(e);
            }
        }
    }

    // Post-loop pass: replace the temporary `do-not-implement` label with the
    // real `waiting-on-issue:#N` label now that all sub-issue IIDs are known.
    // Both backward and forward references are handled — the temporary label
    // blocked the worker throughout the creation loop, so there's no race.
    for (index, sub_issue) in pending.sub_issues.iter().enumerate() {
        if sub_issue.depends_on == 0 {
            continue;
        }
        let dep_idx = sub_issue.depends_on - 1;
        let Some(&dep_iid) = created_issue_ids.get(dep_idx) else {
            warn!(
                "PMO: sub-issue {} depends on sub-issue {} but the dependency was not created, leaving do-not-implement label",
                index + 1,
                sub_issue.depends_on
            );
            continue;
        };
        let Some(&dependent_iid) = created_issue_ids.get(index) else {
            continue;
        };
        // Swap: remove the temporary hold, add the real dependency label.
        let _ = gitlab.remove_issue_label(dependent_iid, labels::DO_NOT_IMPLEMENT);
        let label = format!("waiting-on-issue:#{dep_iid}");
        if let Err(e) = gitlab.add_issue_label(dependent_iid, &label) {
            warn!("PMO: Failed to add dependency label {label} to sub-issue #{dependent_iid}: {e}");
        } else {
            info!("PMO: Sub-issue #{dependent_iid} depends on #{dep_iid}, labeled {label}");
        }
    }

    // All sub-issues created successfully, add comment to parent issue
    if !created_issue_ids.is_empty() {
        let sub_issue_links = created_issue_ids
            .iter()
            .map(|iid| format!("- #{}", iid))
            .collect::<Vec<_>>()
            .join("\n");

        gitlab.add_issue_comment(
            pending.parent_issue_iid,
            &format!(
                "This issue has been split into {} smaller sub-issues by the PMO Agent:\n\n{}\n\n\
                 Each sub-issue is designed to stay around ~500 lines of non-test code (~1500 total including tests). Auto-generated code is excluded from these limits.",
                created_issue_ids.len(),
                sub_issue_links
            ),
        )?;

        // Remove action-required and mark as processed
        gitlab.remove_issue_label(pending.parent_issue_iid, ACTION_REQUIRED_LABEL)?;
        gitlab.add_issue_label(pending.parent_issue_iid, PMO_PROCESSED_LABEL)?;

        info!(
            "PMO: Successfully completed split of issue #{} into {} sub-issues",
            pending.parent_issue_iid,
            created_issue_ids.len()
        );
    }

    Ok(())
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
    fn build_split_prompt_documents_plan_tool_only() {
        let state = AgentState {
            sessions_dir: "/tmp".into(),
            agent_id: "pmo-test".into(),
            project_name: "test-proj".into(),
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
        let prompt = build_split_prompt(&state, &issue, "/abs/pmo-issue-42.md", 2).unwrap();
        // The plan tool is the ONLY output channel.
        assert!(prompt.contains("call the `plan` tool"));
        assert!(prompt.contains("`decision`"));
        assert!(prompt.contains("guide_worker"));
        assert!(prompt.contains("split"));
        assert!(prompt.contains("already_done"));
        assert!(prompt.contains("needs_clarification"));
        // No text-marker fallback.
        assert!(!prompt.contains("SUB_ISSUE_N:"));
        assert!(!prompt.contains("GUIDE_WORKER\n"));
        assert!(!prompt.contains("ALREADY_DONE\n"));
        assert!(!prompt.contains("NEEDS_CLARIFICATION\n"));
        assert!(!prompt.contains("PUBLIC_COMMENT_BEGIN"));
        assert!(!prompt.contains("text-marker"));
        // Reasoning guidance is intact.
        assert!(prompt.contains("TASK CONTAINER TEST"));
        assert!(prompt.contains("DUPLICATE / OVERLAP RULES"));
    }

    #[test]
    fn pmo_output_deserializes_split_with_sub_issues() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "split",
            "sub_issues": [
                {"title": "First", "description": "Do the first thing", "priority": 1},
                {"title": "Second", "description": "Do the second thing", "priority": 2, "depends_on": 1}
            ]
        }))
        .unwrap();
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
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "guide_worker",
            "instructions": "Use flag --foo instead of --bar."
        }))
        .unwrap();
        match output {
            PmoOutput::GuideWorker { instructions } => {
                assert_eq!(
                    instructions.as_deref(),
                    Some("Use flag --foo instead of --bar.")
                );
            }
            other => panic!("expected GuideWorker, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_decision_is_case_insensitive() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "GUIDE_WORKER",
            "instructions": "Proceed."
        }))
        .unwrap();
        assert!(matches!(output, PmoOutput::GuideWorker { .. }));
    }

    #[test]
    fn pmo_output_deserializes_already_done_reason() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "already_done",
            "reason": "The feature exists in src/lib.rs."
        }))
        .unwrap();
        match output {
            PmoOutput::AlreadyDone { reason } => {
                assert!(reason.unwrap().contains("src/lib.rs"));
            }
            other => panic!("expected AlreadyDone, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_deserializes_needs_clarification_question() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "needs_clarification",
            "question": "Which modules should be covered?"
        }))
        .unwrap();
        match output {
            PmoOutput::NeedsClarification { question, .. } => {
                assert!(question.unwrap().contains("modules"));
            }
            other => panic!("expected NeedsClarification, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_deserializes_wait_for_dependency_iid() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "wait_for_dependency",
            "dependency_issue_iid": 47
        }))
        .unwrap();
        match output {
            PmoOutput::WaitForDependency {
                dependency_issue_iid,
            } => {
                assert_eq!(dependency_issue_iid, Some(47));
            }
            other => panic!("expected WaitForDependency, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_ignores_zero_dependency_issue_iid() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "wait_for_dependency",
            "dependency_issue_iid": 0
        }))
        .unwrap();
        match output {
            PmoOutput::WaitForDependency {
                dependency_issue_iid,
            } => {
                assert_eq!(dependency_issue_iid, None);
            }
            other => panic!("expected WaitForDependency, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_accepts_string_dependency_issue_iid() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "wait_for_dependency",
            "dependency_issue_iid": "#727"
        }))
        .unwrap();
        match output {
            PmoOutput::WaitForDependency {
                dependency_issue_iid,
            } => {
                assert_eq!(dependency_issue_iid, Some(727));
            }
            other => panic!("expected WaitForDependency, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_accepts_alternative_dependency_field_names() {
        for key in [
            "dependency_iid",
            "depends_on_issue",
            "blocked_by",
            "dependency",
        ] {
            let output: PmoOutput = serde_json::from_value(serde_json::json!({
                "decision": "wait_for_dependency",
                key: 727
            }))
            .unwrap();
            match output {
                PmoOutput::WaitForDependency {
                    dependency_issue_iid,
                } => {
                    assert_eq!(
                        dependency_issue_iid,
                        Some(727),
                        "failed for alternative field name `{key}`"
                    );
                }
                other => panic!("expected WaitForDependency, got {other:?}"),
            }
        }
    }

    #[test]
    fn pmo_output_rejects_unknown_decision() {
        let err = serde_json::from_value::<PmoOutput>(serde_json::json!({"decision": "bogus"}))
            .unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
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
            guidance_or_empty(Some("Use the existing config loader.".into())),
            "Use the existing config loader."
        );
    }

    #[test]
    fn guidance_or_empty_returns_empty_when_no_instructions() {
        assert!(guidance_or_empty(None).is_empty());
    }

    #[test]
    fn guidance_or_empty_truncates_long_instructions() {
        let long = "Do this. ".repeat(80);
        let extracted = guidance_or_empty(Some(long));
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
        assert!(
            already_done_reason_or_default(Some("Implemented in module X.".into()))
                .contains("module X")
        );
    }

    #[test]
    fn already_done_reason_or_default_falls_back_when_absent() {
        assert!(!already_done_reason_or_default(None).is_empty());
    }

    #[test]
    fn clarification_question_or_default_uses_field() {
        assert_eq!(
            clarification_question_or_default(Some("Which modules?".into())),
            "Which modules?"
        );
    }

    #[test]
    fn clarification_question_or_default_falls_back_when_absent() {
        assert!(!clarification_question_or_default(None).is_empty());
    }

    #[test]
    fn pmo_output_deserializes_needs_clarification_plan_text() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "needs_clarification",
            "plan_text": "## Plan Draft\n\n1. Implement X\n2. Test X\n\nOpen question: which config?"
        }))
        .unwrap();
        match output {
            PmoOutput::NeedsClarification { plan_text, .. } => {
                let plan = plan_text.unwrap();
                assert!(plan.contains("## Plan Draft"));
                assert!(plan.contains("Implement X"));
            }
            other => panic!("expected NeedsClarification, got {other:?}"),
        }
    }

    #[test]
    fn pmo_output_needs_clarification_plan_text_absent_when_not_provided() {
        let output: PmoOutput = serde_json::from_value(serde_json::json!({
            "decision": "needs_clarification"
        }))
        .unwrap();
        match output {
            PmoOutput::NeedsClarification { plan_text, .. } => {
                assert!(plan_text.is_none());
            }
            other => panic!("expected NeedsClarification, got {other:?}"),
        }
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
