use anyhow::Result;

use crate::core::workflow::Workflow;

pub mod claim;
pub mod git;
pub mod gitlab;
pub mod labels;
pub mod ops;
pub mod pmo;
pub mod pmo_cursor_ask;
pub(crate) mod retry;
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

/// Stable machine-readable block markers the PMO emits around worker-facing
/// guidance in GitLab comments. The worker scans issue comments for this block
/// so it can pick up PMO guidance reliably, regardless of any surrounding
/// prose the PMO (or a human) added to the same comment.
pub(crate) const PMO_GUIDANCE_BEGIN: &str = "PMO_GUIDANCE_BEGIN";
pub(crate) const PMO_GUIDANCE_END: &str = "PMO_GUIDANCE_END";

/// Wrap `body` in `PMO_GUIDANCE_BEGIN` / `PMO_GUIDANCE_END` markers. Returns
/// `None` when `body` is empty after trimming so callers can fall back to a
/// plain prose comment.
pub(crate) fn wrap_pmo_guidance_block(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!(
        "{begin}\n{body}\n{end}",
        begin = PMO_GUIDANCE_BEGIN,
        body = trimmed,
        end = PMO_GUIDANCE_END,
    ))
}

/// Extract the first `PMO_GUIDANCE_BEGIN` … `PMO_GUIDANCE_END` block from
/// `text`. Returns `None` when no closed block is present or the body is empty.
pub(crate) fn extract_pmo_guidance_block(text: &str) -> Option<String> {
    extract_marker_block(text, PMO_GUIDANCE_BEGIN, PMO_GUIDANCE_END)
}

/// Stable machine-readable block for public GitLab comments embedded in agent text output.
pub(crate) fn extract_public_comment_block(text: &str) -> Option<String> {
    extract_marker_block(text, "PUBLIC_COMMENT_BEGIN", "PUBLIC_COMMENT_END")
}

/// Generic single-block extractor: returns the trimmed body between the first
/// `begin` marker and the next `end` marker. `None` when either marker is
/// missing or the body is empty.
fn extract_marker_block(text: &str, begin: &str, end: &str) -> Option<String> {
    let start = text.find(begin)?;
    let body_start = start + begin.len();
    let rest = &text[body_start..];
    let end_rel = rest.find(end)?;
    let body = rest[..end_rel].trim();
    if body.is_empty() {
        None
    } else {
        Some(body.to_string())
    }
}

pub(crate) fn strip_public_comment_blocks(text: &str) -> String {
    strip_marker_blocks(text, "PUBLIC_COMMENT_BEGIN", "PUBLIC_COMMENT_END")
}

/// Remove every `<begin>…<end>` block from `text`. If a `begin` marker has no
/// matching `end`, the remainder is dropped (an unclosed block would
/// otherwise leak machine text into a human-facing surface).
fn strip_marker_blocks(text: &str, begin: &str, end: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(begin) {
        out.push_str(&rest[..start]);
        let after_begin = &rest[start + begin.len()..];
        if let Some(end_rel) = after_begin.find(end) {
            rest = &after_begin[end_rel + end.len()..];
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

pub fn register(workflow: &mut Workflow) {
    workflow.register_agent::<worker::WorkerAgent>();
    workflow.register_agent::<reviewer::ReviewerAgent>();
    workflow.register_agent::<pmo::PmoAgent>();
    workflow.register_agent::<ops::OpsAgent>();
}

#[cfg(test)]
mod scope_tests {
    use super::{
        extract_pmo_guidance_block, extract_public_comment_block, issue_in_scope, mr_in_scope,
        register, strip_public_comment_blocks, wrap_pmo_guidance_block,
    };
    use crate::agents::gitlab::{Issue, MergeRequest};
    use crate::core::config::Config;
    use crate::core::workflow::Workflow;

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
    fn register_adds_potlatch_agents_to_workflow() {
        let config = Config::from_toml_str("").unwrap();
        let mut workflow = Workflow::new(config, "potlatch.toml");
        register(&mut workflow);
        let mut names: Vec<_> = workflow.registered_agent_names().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["ops", "pmo", "reviewer", "worker"]);
    }

    #[test]
    fn issue_in_scope_respects_label() {
        let issue = sample_issue(vec!["potlatch", "bug"]);
        assert!(issue_in_scope(&issue, None));
        assert!(issue_in_scope(&issue, Some("potlatch")));
        assert!(!issue_in_scope(&issue, Some("other")));
    }

    #[test]
    fn mr_in_scope_respects_label() {
        let mr = sample_mr(Some(vec!["potlatch"]));
        assert!(mr_in_scope(&mr, None));
        assert!(mr_in_scope(&mr, Some("potlatch")));
        assert!(!mr_in_scope(&mr, Some("other")));

        let no_labels = sample_mr(None);
        assert!(!mr_in_scope(&no_labels, Some("potlatch")));

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
    fn extract_pmo_guidance_block_reads_stable_markers() {
        let text = "noise\nPMO_GUIDANCE_BEGIN\nUse --foo not --bar.\nPMO_GUIDANCE_END\nmore";
        assert_eq!(
            extract_pmo_guidance_block(text).as_deref(),
            Some("Use --foo not --bar.")
        );
    }

    #[test]
    fn extract_pmo_guidance_block_returns_none_for_empty_body() {
        let text = "PMO_GUIDANCE_BEGIN\n\nPMO_GUIDANCE_END";
        assert!(extract_pmo_guidance_block(text).is_none());
    }

    #[test]
    fn extract_pmo_guidance_block_returns_none_when_missing() {
        assert!(extract_pmo_guidance_block("no markers here").is_none());
        assert!(extract_pmo_guidance_block("PMO_GUIDANCE_BEGIN no end").is_none());
    }

    #[test]
    fn wrap_pmo_guidance_block_round_trips() {
        let wrapped = wrap_pmo_guidance_block("Do X.\nThen Y.").unwrap();
        assert!(wrapped.starts_with("PMO_GUIDANCE_BEGIN\n"));
        assert!(wrapped.ends_with("\nPMO_GUIDANCE_END"));
        assert_eq!(
            extract_pmo_guidance_block(&wrapped).as_deref(),
            Some("Do X.\nThen Y.")
        );
    }

    #[test]
    fn wrap_pmo_guidance_block_none_for_empty() {
        assert!(wrap_pmo_guidance_block("").is_none());
        assert!(wrap_pmo_guidance_block("   \n  ").is_none());
    }

    #[test]
    fn extract_marker_block_takes_first_pair() {
        // When two blocks are present, the first complete pair wins.
        let text = "PMO_GUIDANCE_BEGIN\nfirst\nPMO_GUIDANCE_END\nPMO_GUIDANCE_BEGIN\nsecond\nPMO_GUIDANCE_END";
        assert_eq!(extract_pmo_guidance_block(text).as_deref(), Some("first"));
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
