use anyhow::{Context, Result};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{claim, extract_public_comment_block, mr_in_scope, write_task_context_file};
use crate::agents::git::GitRepo;
use crate::agents::gitlab::{self, GitLabClient, Issue, MergeRequest};
use crate::agents::settings;
use crate::agents::workspace::{
    ensure_agent_repo, extract_project_name, require_gitlab_repo, sessions_dir, work_dir,
};
use crate::core::agent::{AgentHandoff, InvokeOptions};
use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::model::acp::ACP_SESSION_MODE_ASK;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};

const REVIEWER_APPROVED_LABEL: &str = "reviewer-approved";
const NEED_AI_WORKER_LABEL: &str = "need-ai-worker";

#[derive(Debug, Clone)]
struct ReviewerConfig {
    poll_interval_secs: u64,
    merge_when_approved: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ReviewerAgentSettings {
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

impl ReviewerAgentSettings {
    fn from_raw(raw: &toml::Value) -> Result<Self> {
        raw.clone()
            .try_into()
            .context("reviewer agent settings from config")
    }
}

pub(crate) struct ReviewerAgent {
    agent_id: String,
    project_name: String,
    reviewer_dir: String,
    sessions_dir: String,
    config: ReviewerConfig,
    git_repo: GitRepo,
    gitlab: GitLabClient,
    model: AgentModel,
    scope_label: String,
    merged_mrs: HashSet<u64>,
    claimed_mr_iid: Option<u64>,
}

impl CoreAgent for ReviewerAgent {
    type SpawnContext = crate::core::workflow::AgentSpawnContext;

    fn name() -> &'static str {
        "reviewer"
    }

    fn model(&self) -> &AgentModel {
        &self.model
    }

    fn banner(_config: &Config, banner: &mut Banner) {
        if let Some(repo) = settings::settings().gitlab_repo() {
            banner.set_once("repo", repo);
        }
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: "gitlab_poll",
            interval: Duration::from_secs(self.config.poll_interval_secs),
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 5000,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        match task_id {
            "gitlab_poll" => {
                let scope = crate::agents::scope_label_filter(&self.scope_label);
                let model = &self.model;
                let shutdown = Arc::clone(model.shutdown());
                if let Err(e) = reviewer_cycle(
                    &self.agent_id,
                    &self.project_name,
                    &self.reviewer_dir,
                    &self.sessions_dir,
                    &self.config,
                    &self.git_repo,
                    &self.gitlab,
                    model,
                    &mut self.merged_mrs,
                    &mut self.claimed_mr_iid,
                    &shutdown,
                    scope,
                ) && !shutdown.load(Ordering::SeqCst)
                {
                    error!("{}: Cycle error: {}", self.agent_id, e);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn from_spawn(ctx: crate::core::workflow::AgentSpawnContext) -> Result<Self> {
        let gitlab_repo = require_gitlab_repo()?;
        let section = ctx
            .workflow
            .config
            .agent("reviewer")
            .context("[agent.reviewer] section required")?;
        let settings = ReviewerAgentSettings::from_raw(&section.raw)?;
        let project_name = extract_project_name(&gitlab_repo)?;
        let agent_id = format!("reviewer-{}", ctx.instance_id);
        ensure_agent_repo(
            &ctx.workflow.base_dir,
            &gitlab_repo,
            &project_name,
            &agent_id,
        )?;
        let reviewer_dir = work_dir(&ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&ctx.workflow.base_dir, &project_name);
        let config = ReviewerConfig {
            poll_interval_secs: settings.poll_interval_secs,
            merge_when_approved: settings.merge_when_approved,
        };
        let git_repo = GitRepo::new(reviewer_dir.clone());
        let gitlab = GitLabClient::new(reviewer_dir.clone(), &gitlab_repo)?;
        let model = AgentModel::connect(
            &ctx,
            "reviewer",
            reviewer_dir.clone(),
            ModelPreferences {
                preferred_session_mode: Some(ACP_SESSION_MODE_ASK),
                structured_output_tools: None,
            },
        )?;
        let agent_settings = settings::settings();
        let scope = agent_settings.scope_label_filter();
        let claimed_mr_iid = find_claimed_mr(&agent_id, &gitlab, scope);
        Ok(Self {
            agent_id,
            project_name,
            reviewer_dir,
            sessions_dir,
            config,
            git_repo,
            gitlab,
            model,
            scope_label: agent_settings.scope_label.clone(),
            merged_mrs: HashSet::new(),
            claimed_mr_iid,
        })
    }

    fn on_shutdown(&mut self) {
        info!("{}: Shutting down, cleaning up...", self.agent_id);
        if let Some(mr_iid) = self.claimed_mr_iid {
            info!(
                "{}: Preserving claim on MR !{} for restart",
                self.agent_id, mr_iid
            );
        }
        let _ = self.git_repo.reset_hard();
        info!("{}: Stopped", self.agent_id);
    }
}

enum ReviewOutcome {
    Merged,
    ApprovedWithoutMerge,
    NeedsChanges,
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
    claimed_mr_iid: &mut Option<u64>,
    shutdown: &AtomicBool,
    scope_label: Option<&str>,
) -> Result<()> {
    let default_branch = git_repo.get_default_branch()?;
    git_repo.fetch()?;
    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }
    let _ = git_repo.reset_hard();
    git_repo.checkout_remote_branch(&default_branch)?;

    // Recover orphaned claims from GitLab when in-memory state was lost (e.g. release failed).
    if claimed_mr_iid.is_none() {
        *claimed_mr_iid = find_claimed_mr(agent_id, gitlab, scope_label);
    }

    // If we still hold a claim from a previous cycle, release it now.
    // Do not proceed to claim a new MR if the release fails.
    if let Some(held_iid) = *claimed_mr_iid {
        info!(
            "{}: Releasing held claim on MR !{} from previous cycle",
            agent_id, held_iid
        );
        match claim::release_mr_claim(gitlab, held_iid, agent_id) {
            Ok(_) => {
                *claimed_mr_iid = None;
            }
            Err(e) => {
                warn!(
                    "{}: Failed to release claim on MR !{}: {}, will retry next cycle",
                    agent_id, held_iid, e
                );
                return Ok(());
            }
        }
    }

    let mut mrs = gitlab.list_merge_requests()?;

    if shutdown.load(Ordering::SeqCst) {
        return Ok(());
    }

    // Sort MRs by the priority of their linked issue (lowest number first),
    // then by MR IID ascending (oldest MR first) for tie-breaking.
    let issues = gitlab.list_issues().unwrap_or_default();
    let priority_map: std::collections::HashMap<u64, u8> =
        issues.iter().map(|i| (i.iid, i.priority())).collect();
    mrs.sort_by(|a, b| {
        let aa = mr_has_label(a, NEED_AI_WORKER_LABEL);
        let bb = mr_has_label(b, NEED_AI_WORKER_LABEL);
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

    for mr in mrs {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        if mr.state != "opened" {
            continue;
        }

        if !mr_in_scope(&mr, scope_label) {
            continue;
        }

        if mr_has_label(&mr, REVIEWER_APPROVED_LABEL) {
            debug!(
                "{}: MR !{} already marked {}, skipping",
                agent_id, mr.iid, REVIEWER_APPROVED_LABEL
            );
            continue;
        }

        if claim::has_our_mr_claim(&mr.labels, agent_id) {
            warn!(
                "{}: MR !{} still has our claim label, releasing before retry",
                agent_id, mr.iid
            );
            release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
        } else if claim::is_mr_claimed(&mr.labels) {
            debug!("{}: MR !{} already claimed, skipping", agent_id, mr.iid);
            continue;
        }

        match has_unresolved_comments(gitlab, mr.iid) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(e) => {
                warn!(
                    "{}: Could not check discussions for MR !{}: {}, skipping this cycle",
                    agent_id, mr.iid, e
                );
                continue;
            }
        }

        if !claim::try_claim_mr(gitlab, mr.iid, agent_id, shutdown)? {
            info!("{}: Failed to claim MR !{}, skipping", agent_id, mr.iid);
            continue;
        }

        if shutdown.load(Ordering::SeqCst) {
            release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
            return Ok(());
        }

        *claimed_mr_iid = Some(mr.iid);

        info!("{}: Reviewing MR !{}: {}", agent_id, mr.iid, mr.title);

        match review_merge_request(
            project_name,
            sessions_dir,
            config,
            git_repo,
            gitlab,
            model,
            &mr,
        ) {
            Ok(ReviewOutcome::Merged) => {
                info!("{}: MR !{} approved and merged", agent_id, mr.iid);
                merged_mrs.insert(mr.iid);
                release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
            }
            Ok(ReviewOutcome::ApprovedWithoutMerge) => {
                info!(
                    "{}: MR !{} approved without merge, releasing claim",
                    agent_id, mr.iid
                );
                release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
            }
            Ok(ReviewOutcome::NeedsChanges) => {
                info!(
                    "{}: MR !{} reviewed with feedback, releasing claim",
                    agent_id, mr.iid
                );
                release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
            }
            Err(e) => {
                error!("{}: Failed to review MR !{}: {}", agent_id, mr.iid, e);
                release_mr_claim_or_warn(gitlab, mr.iid, agent_id);
                *claimed_mr_iid = None;
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
            }
        }

        break;
    }

    Ok(())
}

/// Check if the MR has any resolvable discussion threads that are still unresolved,
/// using the GitLab discussions API `resolved` / `resolvable` fields.
fn has_unresolved_comments(gitlab: &GitLabClient, mr_iid: u64) -> Result<bool> {
    match gitlab.get_unresolved_discussion_count(mr_iid) {
        Ok((unresolved, total)) => {
            if unresolved > 0 {
                return Ok(true);
            }
            if total > 0 {
                info!(
                    "MR !{} all {} discussion(s) resolved, ready for re-review",
                    mr_iid, total
                );
            }
            Ok(false)
        }
        Err(e) => Err(e.context(format!("Failed to fetch discussions for MR !{mr_iid}"))),
    }
}

fn release_mr_claim_or_warn(gitlab: &GitLabClient, mr_iid: u64, agent_id: &str) {
    if let Err(e) = claim::release_mr_claim(gitlab, mr_iid, agent_id) {
        warn!(
            "{}: Failed to release claim on MR !{}: {}",
            agent_id, mr_iid, e
        );
    }
}

fn review_merge_request(
    project_name: &str,
    sessions_dir: &str,
    config: &ReviewerConfig,
    git_repo: &GitRepo,
    gitlab: &GitLabClient,
    model: &AgentModel,
    mr: &MergeRequest,
) -> Result<ReviewOutcome> {
    let is_need_ai_worker_mr = mr_has_label(mr, NEED_AI_WORKER_LABEL);
    // Check if the MR links to an issue via description first, then branch name
    let issue_iid = {
        let re = regex::Regex::new(r"(?i)closes?\s+#(\d+)").ok();
        re.and_then(|r| {
            r.captures(&mr.description)
                .and_then(|c| c.get(1)?.as_str().parse().ok())
        })
        .or_else(|| gitlab::issue_iid_from_branch(&mr.source_branch))
    };

    if issue_iid.is_none() && !is_need_ai_worker_mr {
        warn!(
            "MR !{} does not reference any issue, requesting fix",
            mr.iid
        );
        gitlab.add_mr_discussion(
            mr.iid,
            "This MR does not reference an issue. Please link it to the relevant issue by using a branch name like `issue-N` or adding `Closes #N` in the MR description.",
        )?;
        return Ok(ReviewOutcome::NeedsChanges);
    }

    if has_bad_title_or_description(mr) {
        warn!(
            "MR !{} has a generic or missing title/description, requesting fix",
            mr.iid
        );
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
        gitlab.add_mr_discussion(mr.iid, &issues.join("\n\n"))?;
        return Ok(ReviewOutcome::NeedsChanges);
    }

    let _ = git_repo.reset_hard();
    git_repo.checkout_remote_branch(&mr.source_branch)?;

    let source_sha = git_repo.rev_parse("HEAD")?;
    let target_sha = git_repo.rev_parse(&format!("origin/{}", mr.target_branch))?;
    info!(
        "MR !{} diff: {} ({}) -> {} ({})",
        mr.iid, mr.source_branch, source_sha, mr.target_branch, target_sha
    );

    if !git_repo.try_merge(&mr.target_branch)? {
        warn!(
            "MR !{} has merge conflicts with {}",
            mr.iid, mr.target_branch
        );
        gitlab.add_mr_discussion(
            mr.iid,
            &format!(
                "This MR has merge conflicts with `{}`. Please rebase or resolve conflicts before review can proceed.",
                mr.target_branch
            ),
        )?;
        git_repo.checkout_remote_branch(&mr.target_branch)?;
        return Ok(ReviewOutcome::NeedsChanges);
    }

    let diff_stat = git_repo.diff_stat_against(&mr.target_branch)?;
    let changed_files = git_repo.changed_files_against(&mr.target_branch)?;

    let prompt = build_review_prompt(ReviewPromptInput {
        project_name,
        gitlab,
        mr,
        diff_stat: &diff_stat,
        changed_files: &changed_files,
        issue_iid,
        is_need_ai_worker_mr,
        sessions_dir,
    })?;

    let agent_output = model.complete(
        &prompt,
        &InvokeOptions {
            activity_label: Some(format!("{} reviewing MR !{}", model.agent_id(), mr.iid)),
            ..InvokeOptions::default()
        },
    )?;
    info!(
        "{}: Reviewer agent finished MR !{}",
        model.agent_id(),
        mr.iid
    );

    git_repo.checkout_remote_branch(&mr.target_branch)?;

    if reviewer_approves(&agent_output) {
        info!("MR !{} approved by reviewer", mr.iid);

        // Re-check for unresolved discussions before merging — another reviewer
        // or the worker may have left new comments during the review.
        match has_unresolved_comments(gitlab, mr.iid) {
            Ok(true) => {
                warn!(
                    "MR !{} approved but has unresolved discussions, skipping merge",
                    mr.iid
                );
                return Ok(ReviewOutcome::NeedsChanges);
            }
            Ok(false) => {}
            Err(e) => {
                return Err(e.context(format!(
                    "Could not verify discussions are resolved for MR !{} before merge",
                    mr.iid
                )));
            }
        }

        if config.merge_when_approved {
            match gitlab.merge_mr(mr.iid) {
                Ok(_) => {
                    info!("Successfully merged MR !{}", mr.iid);
                    return Ok(ReviewOutcome::Merged);
                }
                Err(e) => {
                    warn!("Failed to merge MR !{}: {}", mr.iid, e);
                    gitlab.add_mr_discussion(
                        mr.iid,
                        &format!(
                            "Code is approved, but automatic merge failed (`{}`). \
                             This is likely due to merge conflicts with the target branch. \
                             Please rebase or resolve conflicts and push again.",
                            e
                        ),
                    )?;
                }
            }
        } else {
            let approval_message = extract_approval_message(&agent_output);
            gitlab.add_resolved_mr_discussion(mr.iid, &approval_message)?;
            gitlab.add_mr_label_with_transient_retries(mr.iid, REVIEWER_APPROVED_LABEL)?;
            return Ok(ReviewOutcome::ApprovedWithoutMerge);
        }
    } else if reviewer_requests_changes(&agent_output) {
        info!("MR !{} needs changes", mr.iid);

        let feedback = extract_review_feedback(&agent_output);
        gitlab.add_mr_discussion(mr.iid, &feedback)?;
    } else if let Some(feedback) = extract_fallback_review_feedback(&agent_output) {
        warn!(
            "MR !{} reviewer output missed decision marker, posting fallback feedback",
            mr.iid
        );
        gitlab.add_mr_discussion(mr.iid, &feedback)?;
    } else {
        warn!(
            "MR !{} reviewer output missed decision marker; not posting unstructured output",
            mr.iid
        );
    }

    Ok(ReviewOutcome::NeedsChanges)
}

struct ReviewPromptInput<'a> {
    project_name: &'a str,
    gitlab: &'a GitLabClient,
    mr: &'a MergeRequest,
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
        "9. COMPLETENESS CHECK (STRICT): For `need-ai-worker` MRs, evaluate completeness against the current MR title, MR description, diff, and comment history (do not require linked issue context). Treat later comments as updates to the requested work. If the current scope implied by those sources is missing or partial, list missing items and REQUEST_CHANGES.".to_string()
    } else {
        "9. COMPLETENESS CHECK (STRICT): Compare the actual local diff and changed files against the CURRENT linked issue requirements: issue title, issue description, issue comments, MR description, and MR comment history. Later comments may clarify, narrow, expand, or supersede earlier issue text. Every current requirement MUST be addressed in the implementation, but do not request changes for an older constraint that later comments removed, changed, or accepted as intentionally out of scope. If any current requirement is missing or only partially implemented, list missing items and REQUEST_CHANGES. This check is critical to avoid shipping incomplete features.".to_string()
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

GITLAB COMMENT STYLE (STRICT — for REQUEST_CHANGES and any posted feedback):
- Do NOT start with a long paragraph of hollow praise or thanks that only restates the diff or issue number (e.g. listing routes, files, or "aligns with #N" without adding a review decision). That adds no value and wastes the reader's time.
- Lead with what matters: **what must change before merge**, or **why you approve**. Use a direct lead-in such as `Request before merge:` or `Blocking:` when the MR must not merge until the item is addressed.
- Only request MR description updates after you have read the full `## MR description` section in the task context file (including everything after any `Closes #N` line). Do **not** treat an opening `Closes #N` as “description is only the closing line” when the rest of that section documents the work. If it already states goal, implementation approach, and verification, do not ask to expand the description.
- Public GitLab comments must use reader-facing wording only. Do NOT mention internal response fields or protocol tokens such as `MR_DESCRIPTION`, `MR_TITLE`, `FEEDBACK`, `PUBLIC_COMMENT_BEGIN`, or `PUBLIC_COMMENT_END`. For example, say "Please update the MR description to include the actual verification and testing performed", not "Update MR_DESCRIPTION with the actual verification/testing performed."
- Keep the public comment focused: one short optional line of genuine substance is OK, but **never** pad with a multi-sentence "thanks for the thorough coverage" preface that duplicates the diff.

INSTRUCTIONS:
1. Read `AGENTS.md` from the repository root before starting the review. Treat it as authoritative project policy.
2. Before inspecting or judging code, perform any repository setup or pre-review steps required by `AGENTS.md` (for example, updating submodules when the project policy says to do so). If a required setup command fails, REQUEST_CHANGES and include the failure as blocking review feedback.
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

After your review, provide your decision:

If the MR is good to merge:
APPROVE
LGTM: <brief summary of what was reviewed>

If changes are needed:
REQUEST_CHANGES
FEEDBACK:
- <specific issue 1>
- <specific issue 2>
- <etc>

For any human-facing GitLab comment text, include a stable block (use the same style as above: no hollow opening paragraph; put the request first):
PUBLIC_COMMENT_BEGIN
<only final public comment text; no progress/status/tool logs>
PUBLIC_COMMENT_END

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

fn has_bad_title_or_description(mr: &MergeRequest) -> bool {
    is_generic_title(&mr.title) || is_generic_description(&mr.description)
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

fn reviewer_approves(agent_output: &AgentHandoff) -> bool {
    agent_output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("approve"))
        || (agent_output.response.contains("APPROVE") && agent_output.response.contains("LGTM"))
}

fn reviewer_requests_changes(agent_output: &AgentHandoff) -> bool {
    agent_output
        .decision
        .as_deref()
        .is_some_and(|d| d.eq_ignore_ascii_case("request_changes"))
        || agent_output.response.contains("REQUEST_CHANGES")
}

/// Resolved approval thread on GitLab: keep the body minimal (no long LGTM narrative).
fn extract_approval_message(_agent_output: &AgentHandoff) -> String {
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

fn extract_review_feedback(agent_output: &AgentHandoff) -> String {
    if let Some(block) = extract_public_comment_block(&agent_output.response) {
        let out = normalize_review_comment_body(block.trim());
        if !out.is_empty() {
            return out;
        }
    }
    if let Some(feedback) = &agent_output.feedback {
        if let Some(block) = extract_public_comment_block(feedback) {
            let out = normalize_review_comment_body(block.trim());
            if !out.is_empty() {
                return out;
            }
        }
        let trimmed = feedback.trim();
        if !trimmed.is_empty() {
            let out = normalize_review_comment_body(trimmed);
            if !out.is_empty() {
                return out;
            }
        }
    }
    if let Some(pos) = agent_output.response.find("FEEDBACK:") {
        let feedback = &agent_output.response[pos + 9..];
        let out = normalize_review_comment_body(feedback.trim());
        if !out.is_empty() {
            return out;
        }
    }

    if let Some(pos) = find_request_changes_ignore_case(&agent_output.response) {
        let tail = agent_output.response[pos..].trim_start();
        let out = normalize_review_comment_body(tail);
        if !out.is_empty() {
            return out;
        }
    }

    "Please review the changes and address any issues.".to_string()
}

fn find_request_changes_ignore_case(haystack: &str) -> Option<usize> {
    const NEEDLE: &[u8] = b"REQUEST_CHANGES";
    let h = haystack.as_bytes();
    let n = NEEDLE.len();
    if h.len() < n {
        return None;
    }
    for i in 0..=h.len() - n {
        if h[i..i + n].eq_ignore_ascii_case(NEEDLE) {
            return Some(i);
        }
    }
    None
}

fn extract_fallback_review_feedback(agent_output: &AgentHandoff) -> Option<String> {
    if let Some(block) = extract_public_comment_block(&agent_output.response) {
        let trimmed = block.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(feedback) = &agent_output.feedback {
        if let Some(block) = extract_public_comment_block(feedback) {
            let trimmed = block.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        let trimmed = feedback.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn mr_has_label(mr: &MergeRequest, label: &str) -> bool {
    mr.labels
        .as_ref()
        .is_some_and(|labels| labels.iter().any(|existing| existing == label))
}

// ---------------------------------------------------------------------------
// Reviewer claim recovery
// ---------------------------------------------------------------------------

/// Scan all open MRs for this reviewer's claim label.
/// The GitLab label is the single source of truth — no local state file needed.
fn find_claimed_mr(
    agent_id: &str,
    gitlab: &GitLabClient,
    scope_label: Option<&str>,
) -> Option<u64> {
    let claim_label = claim::claim_label(agent_id);

    match gitlab.list_merge_requests() {
        Ok(mrs) => {
            for mr in &mrs {
                if mr.state != "opened" {
                    continue;
                }
                let has_claim = mr.labels.as_ref().is_some_and(|l| l.contains(&claim_label));
                if has_claim && mr_in_scope(mr, scope_label) {
                    info!(
                        "{}: Found existing claim on MR !{}, will release next cycle",
                        agent_id, mr.iid
                    );
                    return Some(mr.iid);
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

    #[test]
    fn fallback_review_feedback_ignores_unstructured_reviewer_output() {
        let output = AgentHandoff {
            response: r#"I’ll review the MR from the local merged branch.
A key prior blocker is still present.
Error: T: Connection stalled"#
                .to_string(),
            ..Default::default()
        };

        assert!(extract_fallback_review_feedback(&output).is_none());
    }

    #[test]
    fn extract_approval_message_is_always_short_lgtm() {
        let output = AgentHandoff::default();

        assert_eq!(extract_approval_message(&output), "LGTM");
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
        let output = AgentHandoff {
            response: "REQUEST_CHANGES — please fix the following before review:\n\nThe MR title `x` is too generic."
                .to_string(),
            ..Default::default()
        };
        assert_eq!(
            extract_review_feedback(&output),
            "The MR title `x` is too generic."
        );
    }

    #[test]
    fn extract_review_feedback_prefers_public_comment_block() {
        let output = AgentHandoff {
            response:
                "REQUEST_CHANGES\nPUBLIC_COMMENT_BEGIN\nPlease add one integration test.\nPUBLIC_COMMENT_END"
                    .to_string(),
            ..Default::default()
        };
        assert_eq!(
            extract_review_feedback(&output),
            "Please add one integration test."
        );
    }

    #[test]
    fn find_request_changes_ignore_case_finds_mixed_case() {
        let s = "Prefix request_changes: more text";
        assert_eq!(find_request_changes_ignore_case(s), Some(7));
    }
}
