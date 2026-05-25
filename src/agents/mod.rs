use anyhow::{Context, Result};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;
use tracing::info;

use crate::core::config::Config;
use crate::core::registry::AgentRegistry;
use crate::core::workflow::{Workflow, WorkflowContext};

pub mod claim;
pub mod git;
pub mod gitlab;
pub mod labels;
pub mod pmo;
pub mod pmo_cursor_ask;
pub mod register;
pub mod reviewer;
pub mod settings;
pub mod worker;
pub mod workspace;

use crate::agents::gitlab::{Issue, MergeRequest};

/// Empty or whitespace-only `scope_label` means handle all items.
pub fn scope_label_filter(scope_label: &str) -> Option<&str> {
    let t = scope_label.trim();
    if t.is_empty() { None } else { Some(t) }
}

fn prepare_codepair_workflow(ctx: &WorkflowContext) -> Result<()> {
    flag::register(SIGINT, ctx.shutdown.clone()).context("Failed to register SIGINT handler")?;
    flag::register(SIGTERM, ctx.shutdown.clone()).context("Failed to register SIGTERM handler")?;
    flag::register_conditional_shutdown(SIGINT, 1, ctx.shutdown.clone())
        .context("Failed to register conditional shutdown")?;
    flag::register_conditional_shutdown(SIGTERM, 1, ctx.shutdown.clone())
        .context("Failed to register conditional shutdown")?;
    Ok(())
}

/// Stable machine-readable block for public GitLab comments embedded in agent text output.
pub(crate) fn extract_public_comment_block(text: &str) -> Option<String> {
    const BEGIN: &str = "PUBLIC_COMMENT_BEGIN";
    const END: &str = "PUBLIC_COMMENT_END";
    let start = text.find(BEGIN)?;
    let body_start = start + BEGIN.len();
    let rest = &text[body_start..];
    let end_rel = rest.find(END)?;
    let body = rest[..end_rel].trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_string())
    }
}

pub(crate) fn strip_public_comment_blocks(text: &str) -> String {
    const BEGIN: &str = "PUBLIC_COMMENT_BEGIN";
    const END: &str = "PUBLIC_COMMENT_END";
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(BEGIN) {
        out.push_str(&rest[..start]);
        let after_begin = &rest[start + BEGIN.len()..];
        if let Some(end_rel) = after_begin.find(END) {
            rest = &after_begin[end_rel + END.len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

pub(crate) fn write_task_context_file(
    work_dir: &str,
    file_name: &str,
    content: &str,
) -> Result<String> {
    use anyhow::Context;
    use std::path::Path;

    anyhow::ensure!(
        !file_name.contains('/') && !file_name.contains('\\') && !file_name.is_empty(),
        "task context file_name must be a simple file name, got {:?}",
        file_name
    );
    let base = Path::new(work_dir);
    std::fs::create_dir_all(base).context("Failed to create task context directory")?;
    let path = base.join(file_name);
    std::fs::write(&path, content).context("Failed to write task context file")?;
    let abs = std::fs::canonicalize(&path).unwrap_or(path);
    Ok(abs.to_string_lossy().into_owned())
}

pub(crate) fn issue_in_scope(issue: &Issue, scope_label: Option<&str>) -> bool {
    match scope_label {
        None => true,
        Some(l) => issue.labels.iter().any(|x| x == l),
    }
}

pub(crate) fn mr_in_scope(mr: &MergeRequest, scope_label: Option<&str>) -> bool {
    if mr
        .labels
        .as_ref()
        .is_some_and(|labels| labels.iter().any(|x| x == labels::NEED_AI_WORKER))
    {
        return true;
    }
    match scope_label {
        None => true,
        Some(l) => mr
            .labels
            .as_ref()
            .is_some_and(|labels| labels.iter().any(|x| x == l)),
    }
}

pub fn run(config: Config, agent_settings: settings::AgentSettings) -> Result<()> {
    settings::init(agent_settings);
    let agent_names: Vec<_> = config.agent_names().collect();
    info!(
        "Starting configured agents: {}",
        if agent_names.is_empty() {
            "(none)".to_string()
        } else {
            agent_names.join(", ")
        }
    );

    let mut registry = AgentRegistry::new();
    register::register_codepair_agents(&mut registry);
    Workflow::run(config, &registry, prepare_codepair_workflow)
}

#[cfg(test)]
mod scope_tests {
    use super::{
        extract_public_comment_block, issue_in_scope, mr_in_scope, strip_public_comment_blocks,
    };
    use crate::agents::gitlab::{Issue, MergeRequest};

    fn sample_issue(labels: Vec<&str>) -> Issue {
        Issue {
            iid: 1,
            title: "t".to_string(),
            description: "".to_string(),
            labels: labels.into_iter().map(String::from).collect(),
            state: "opened".to_string(),
            created_at: None,
            updated_at: None,
        }
    }

    fn sample_mr(labels: Option<Vec<&str>>) -> MergeRequest {
        MergeRequest {
            iid: 1,
            title: "t".to_string(),
            description: "".to_string(),
            source_branch: "issue-1".to_string(),
            target_branch: "main".to_string(),
            state: "opened".to_string(),
            sha: None,
            labels: labels.map(|v| v.into_iter().map(String::from).collect()),
            has_conflicts: false,
        }
    }

    #[test]
    fn issue_in_scope_respects_label() {
        let issue = sample_issue(vec!["codepair", "bug"]);
        assert!(issue_in_scope(&issue, None));
        assert!(issue_in_scope(&issue, Some("codepair")));
        assert!(!issue_in_scope(&issue, Some("other")));
    }

    #[test]
    fn mr_in_scope_respects_label() {
        let mr = sample_mr(Some(vec!["codepair"]));
        assert!(mr_in_scope(&mr, None));
        assert!(mr_in_scope(&mr, Some("codepair")));
        assert!(!mr_in_scope(&mr, Some("other")));

        let no_labels = sample_mr(None);
        assert!(!mr_in_scope(&no_labels, Some("codepair")));

        let ai_worker = sample_mr(Some(vec![super::labels::NEED_AI_WORKER]));
        assert!(mr_in_scope(&ai_worker, Some("other-scope")));
    }

    #[test]
    fn extract_public_comment_block_reads_stable_markers() {
        let text = "noise\nPUBLIC_COMMENT_BEGIN\nFinal public comment.\nPUBLIC_COMMENT_END\nmore";
        assert_eq!(
            extract_public_comment_block(text).as_deref(),
            Some("Final public comment.")
        );
    }

    #[test]
    fn strip_public_comment_blocks_removes_one_block() {
        let text = "Goal line\nPUBLIC_COMMENT_BEGIN\nThanks.\nPUBLIC_COMMENT_END\n## Testing\nx";
        assert_eq!(
            strip_public_comment_blocks(text),
            "Goal line\n\n## Testing\nx"
        );
    }

    #[test]
    fn strip_public_comment_blocks_removes_multiple() {
        let t = "A\nPUBLIC_COMMENT_BEGIN\n1\nPUBLIC_COMMENT_END\nB\nPUBLIC_COMMENT_BEGIN\n2\nPUBLIC_COMMENT_END\nC";
        assert_eq!(strip_public_comment_blocks(t), "A\n\nB\n\nC");
    }

    #[test]
    fn strip_public_comment_blocks_truncates_unclosed_begin() {
        let t = "Keep\nPUBLIC_COMMENT_BEGIN\ndangling";
        assert_eq!(strip_public_comment_blocks(t), "Keep");
    }
}
