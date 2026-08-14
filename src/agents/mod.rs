use anyhow::Result;

use crate::core::workflow::Workflow;

pub mod claim;
pub mod git;
pub mod gitlab;
pub mod labels;
pub mod ops;
pub mod pmo;
pub(crate) mod qa;
pub(crate) mod retry;
pub mod reviewer;
pub mod settings;
pub(crate) mod ssh_util;
pub mod worker;
pub mod workspace;

use crate::agents::gitlab::{Issue, MergeRequest};

/// Empty or whitespace-only `scope_label` means handle all items.
pub fn scope_label_filter(scope_label: &str) -> Option<&str> {
    let t = scope_label.trim();
    if t.is_empty() { None } else { Some(t) }
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

/// Strip internal potlatch markers from `text` before posting it to GitLab.
/// Removes `PUBLIC_COMMENT_BEGIN/END` blocks and any stray marker lines so
/// internal harness markers don't leak into human-facing GitLab comments when
/// the model includes them in fields like `question` or `reason`.
pub(crate) fn strip_internal_markers(text: &str) -> String {
    let out = strip_marker_blocks(text, "PUBLIC_COMMENT_BEGIN", "PUBLIC_COMMENT_END");
    // Also strip bare marker lines the model might emit without a matching end.
    out.lines()
        .filter(|line| {
            let t = line.trim();
            t != "PUBLIC_COMMENT_BEGIN" && t != "PUBLIC_COMMENT_END"
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
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
    workflow.register_agent::<qa::QaAgent>();
}

#[cfg(test)]
mod scope_tests {
    use super::{
        extract_public_comment_block, issue_in_scope, mr_in_scope, register,
        strip_internal_markers, strip_public_comment_blocks,
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
        assert_eq!(names, vec!["ops", "pmo", "qa", "reviewer", "worker"]);
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
    fn strip_internal_markers_removes_public_comment_blocks() {
        let text = "PUBLIC_COMMENT_BEGIN\nhidden\nPUBLIC_COMMENT_END\nVisible.";
        let clean = strip_internal_markers(text);
        assert!(!clean.contains("PUBLIC_COMMENT_BEGIN"));
        assert!(!clean.contains("hidden"));
        assert!(clean.contains("Visible."));
    }

    #[test]
    fn strip_internal_markers_removes_bare_marker_lines() {
        let text = "PUBLIC_COMMENT_BEGIN\nPUBLIC_COMMENT_END\nClean text.";
        let clean = strip_internal_markers(text);
        assert!(!clean.contains("PUBLIC_COMMENT_BEGIN"));
        assert!(!clean.contains("PUBLIC_COMMENT_END"));
        assert!(clean.contains("Clean text."));
    }

    #[test]
    fn strip_internal_markers_preserves_clean_text() {
        let text = "This is a normal comment for humans.";
        assert_eq!(strip_internal_markers(text), text);
    }

    #[test]
    fn strip_internal_markers_handles_empty_result() {
        let text = "PUBLIC_COMMENT_BEGIN\nPUBLIC_COMMENT_END";
        let clean = strip_internal_markers(text);
        assert!(clean.is_empty());
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
