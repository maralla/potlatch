use anyhow::Result;

use crate::core::workflow::Workflow;

pub(crate) mod artifact;
pub(crate) mod claim;
pub(crate) mod clerk;
pub mod forge;
pub mod git;
pub mod labels;
pub mod ops;
pub mod pmo;
pub(crate) mod qa;
pub mod reviewer;
pub mod settings;
pub(crate) mod ssh_util;
pub(crate) mod state;
pub(crate) mod web;
pub mod worker;
pub mod workspace;

use crate::agents::forge::{Issue, MergeRequest};

/// Empty or whitespace-only `scope_label` means handle all items.
pub fn scope_label_filter(scope_label: &str) -> Option<&str> {
    let t = scope_label.trim();
    if t.is_empty() { None } else { Some(t) }
}

const PMO_SPLIT_PARENT_PREFIX: &str = "This issue was split from parent issue #";

/// Append stable, human-readable split provenance to a PMO-created child issue.
pub(crate) fn with_split_parent(description: &str, parent_iid: u64) -> String {
    format!(
        "{}\n\n---\n\n{PMO_SPLIT_PARENT_PREFIX}{parent_iid}.",
        description.trim_end()
    )
}

/// Read the parent issue IID from PMO split provenance, when present.
pub(crate) fn split_parent_iid(description: &str) -> Option<u64> {
    description.lines().find_map(|line| {
        line.trim()
            .strip_prefix(PMO_SPLIT_PARENT_PREFIX)?
            .strip_suffix('.')?
            .parse()
            .ok()
    })
}

pub(crate) fn write_task_context_file(
    work_dir: &str,
    file_name: &str,
    content: &str,
) -> Result<String> {
    let store = crate::agents::artifact::ArtifactStore::new(work_dir);
    Ok(store
        .write(file_name, content)?
        .to_string_lossy()
        .into_owned())
}

pub(crate) fn issue_in_scope(issue: &Issue, scope_label: Option<&str>) -> bool {
    issue_labels_in_scope(&issue.labels, scope_label)
}

/// Scope test for an issue known only by its labels, so a role that
/// observes an issue through its own snapshot type still applies exactly
/// the same rule as [`issue_in_scope`].
pub(crate) fn issue_labels_in_scope(labels: &[String], scope_label: Option<&str>) -> bool {
    match scope_label {
        None => true,
        Some(l) => labels.iter().any(|x| x == l),
    }
}

pub(crate) fn mr_in_scope(mr: &MergeRequest, scope_label: Option<&str>) -> bool {
    mr_labels_in_scope(mr.labels.as_deref(), scope_label)
}

/// Scope test for a merge request known only by its labels, so a role that
/// observes an MR through its own snapshot type still applies exactly the
/// same rule as [`mr_in_scope`].
pub(crate) fn mr_labels_in_scope(labels: Option<&[String]>, scope_label: Option<&str>) -> bool {
    if labels.is_some_and(|labels| labels.iter().any(|x| x == labels::NEED_AI_WORKER)) {
        return true;
    }
    match scope_label {
        None => true,
        Some(l) => labels.is_some_and(|labels| labels.iter().any(|x| x == l)),
    }
}

pub fn register(workflow: &mut Workflow) {
    workflow.register_agent::<worker::WorkerAgent>();
    workflow.register_agent::<reviewer::ReviewerAgent>();
    workflow.register_agent::<pmo::PmoAgent>();
    workflow.register_agent::<ops::OpsAgent>();
    workflow.register_agent::<qa::QaAgent>();
    workflow.register_agent::<web::WebAgent>();
    workflow.register_agent::<clerk::ClerkAgent>();
}

#[cfg(test)]
mod scope_tests {
    use super::{issue_in_scope, mr_in_scope, register, split_parent_iid, with_split_parent};
    use crate::agents::forge::{Issue, MergeRequest};
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
        assert_eq!(
            names,
            vec!["clerk", "ops", "pmo", "qa", "reviewer", "web", "worker"]
        );
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
    fn split_parent_provenance_round_trips_without_changing_the_child_scope() {
        let description = with_split_parent("Implement the parser.\n", 42);

        assert!(description.starts_with("Implement the parser.\n\n---"));
        assert_eq!(split_parent_iid(&description), Some(42));
        assert_eq!(split_parent_iid("Implement an unrelated issue."), None);
    }
}
