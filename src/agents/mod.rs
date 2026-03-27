pub mod claim;
pub mod labels;
pub mod pmo;
pub mod pmo_cursor_ask;
pub mod reviewer;
pub mod worker;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;
use tracing::{error, info, warn};

use crate::config::Config;
use crate::git::GitRepo;
use crate::gitlab::{Issue, MergeRequest};

pub fn extract_project_name(repo_url: &str) -> Result<String> {
    let parts: Vec<&str> = repo_url.trim_end_matches('/').split('/').collect();
    let name = parts
        .last()
        .context("Invalid repository URL")?
        .trim_end_matches(".git");
    Ok(name.to_string())
}

/// Writes `{work_dir}/.codepair-context/{file_name}`. `file_name` must be a single path segment
/// (e.g. `pmo-issue-1.md`), not a nested path.
pub(crate) fn write_task_context_file(
    work_dir: &str,
    file_name: &str,
    content: &str,
) -> Result<String> {
    anyhow::ensure!(
        !file_name.contains('/') && !file_name.contains('\\') && !file_name.is_empty(),
        "task context file_name must be a simple file name, got {:?}",
        file_name
    );
    let base = Path::new(work_dir);
    std::fs::create_dir_all(&base).context("Failed to create task context directory")?;
    let path = base.join(file_name);
    std::fs::write(&path, content).context("Failed to write task context file")?;
    let abs = std::fs::canonicalize(&path).unwrap_or(path);
    Ok(abs.to_string_lossy().into_owned())
}

fn agent_dir(base_dir: &str, project_name: &str, agent_id: &str) -> String {
    std::path::Path::new(base_dir)
        .join(format!("{}-{}", project_name, agent_id))
        .to_string_lossy()
        .into_owned()
}

/// The actual git working directory inside the agent container.
/// Layout: `<base>/<project>-<agent_id>/<project>/`
/// This ensures the directory name matches the real project name,
/// which helps the AI agent understand the project context better.
fn work_dir(base_dir: &str, project_name: &str, agent_id: &str) -> String {
    std::path::Path::new(base_dir)
        .join(format!("{}-{}", project_name, agent_id))
        .join(project_name)
        .to_string_lossy()
        .into_owned()
}

/// Shared directory for session state files. All workers read/write here
/// so that any worker can pick up orphaned sessions on restart.
/// When `scope_label` is `Some`, the issue must include that label (exact match against GitLab).
pub(crate) fn issue_in_scope(issue: &Issue, scope_label: Option<&str>) -> bool {
    match scope_label {
        None => true,
        Some(l) => issue.labels.iter().any(|x| x == l),
    }
}

/// When `scope_label` is `Some`, the merge request must include that label (exact match).
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

fn sessions_dir(base_dir: &str, project_name: &str) -> String {
    let dir = std::path::Path::new(base_dir)
        .join(format!("{}-sessions", project_name))
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn prepare_repos(base_dir: &str, repo_url: &str, config: &Config) -> Result<()> {
    let project_name = extract_project_name(repo_url)?;

    let mut agent_ids = Vec::new();

    for i in 0..config.worker.instances {
        agent_ids.push(format!("worker-{}", i));
    }
    for i in 0..config.reviewer.instances {
        agent_ids.push(format!("reviewer-{}", i));
    }
    for i in 0..config.pmo.instances {
        agent_ids.push(format!("pmo-{}", i));
    }

    for id in agent_ids {
        let container = agent_dir(base_dir, &project_name, &id);
        let repo_path = work_dir(base_dir, &project_name, &id);
        let git_repo = GitRepo::new(repo_path.clone());
        if !git_repo.exists() {
            // Remove leftover partial repo directory from a previously failed clone
            let repo_dir = Path::new(&repo_path);
            if repo_dir.exists() {
                info!("Removing partial directory {}...", repo_path);
                std::fs::remove_dir_all(repo_dir)
                    .context("Failed to remove partial clone directory")?;
            }
            // Ensure the container directory exists
            std::fs::create_dir_all(&container)
                .context("Failed to create agent container directory")?;
            info!("Cloning repository into {}...", repo_path);
            git_repo.clone(repo_url)?;
        }
    }

    Ok(())
}

pub fn run(git_repo_address: String, config: Config) -> Result<()> {
    info!(
        "Starting {} worker(s), {} reviewer(s), {} PMO(s)",
        config.worker.instances, config.reviewer.instances, config.pmo.instances
    );

    let base_dir = std::env::current_dir()
        .context("Failed to get current directory")?
        .to_string_lossy()
        .into_owned();

    let project_name = extract_project_name(&git_repo_address)?;

    prepare_repos(&base_dir, &git_repo_address, &config)?;

    let shutdown = Arc::new(AtomicBool::new(false));

    let mcp_coordinator = if config.mcp.enabled {
        let (coord, mcp_port) = crate::mcp_http::spawn_http_mcp_server(Arc::clone(&shutdown))
            .context("Failed to start MCP HTTP coordinator")?;
        let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_port);
        info!("MCP coordinator at {}", mcp_url);
        crate::cursor_mcp_config::ensure_for_all_agent_workspaces(
            &base_dir,
            &project_name,
            &config,
            &mcp_url,
        )
        .context("Failed to write .cursor/mcp.json in agent workspaces")?;
        Some(coord)
    } else {
        info!(
            "MCP HTTP coordinator disabled (default). Tasks use ACP only. \
             Enable MCP with [mcp] enabled = true in codepair.toml."
        );
        None
    };

    flag::register(SIGINT, Arc::clone(&shutdown)).context("Failed to register SIGINT handler")?;
    flag::register(SIGTERM, Arc::clone(&shutdown)).context("Failed to register SIGTERM handler")?;

    // Also handle double Ctrl+C: if shutdown flag is already set, force exit
    flag::register_conditional_shutdown(SIGINT, 1, Arc::clone(&shutdown))
        .context("Failed to register conditional shutdown")?;
    flag::register_conditional_shutdown(SIGTERM, 1, Arc::clone(&shutdown))
        .context("Failed to register conditional shutdown")?;

    let mut handles = Vec::new();
    let scope_label = config.scope_label.clone();

    for i in 0..config.worker.instances {
        let repo = git_repo_address.clone();
        let worker_config = config.worker.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        let coord = mcp_coordinator.clone();
        let scope_label = scope_label.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) = worker::run(repo, worker_config, i, shutdown, base, coord, scope_label)
            {
                error!("Worker-{} error: {}", i, e);
            }
        }));
    }

    for i in 0..config.reviewer.instances {
        let repo = git_repo_address.clone();
        let reviewer_config = config.reviewer.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        let coord = mcp_coordinator.clone();
        let scope_label = scope_label.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) =
                reviewer::run(repo, reviewer_config, i, shutdown, base, coord, scope_label)
            {
                error!("Reviewer-{} error: {}", i, e);
            }
        }));
    }

    for i in 0..config.pmo.instances {
        let repo = git_repo_address.clone();
        let pmo_config = config.pmo.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        let coord = mcp_coordinator.clone();
        let scope_label = scope_label.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) = pmo::run(repo, pmo_config, i, shutdown, base, coord, scope_label) {
                error!("PMO-{} error: {}", i, e);
            }
        }));
    }

    // Wait for shutdown signal, then give threads a grace period to exit.
    // This handles the case where `kill <pid>` sends SIGINT/SIGTERM only to
    // the parent — child processes (git, glab) may still be running and
    // blocking threads in .output() calls.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }

    info!("Shutdown signal received, waiting for agents to stop...");

    let deadline = std::time::Instant::now() + Duration::from_secs(5);

    loop {
        handles.retain(|h| !h.is_finished());
        if handles.is_empty() {
            break;
        }
        if std::time::Instant::now() >= deadline {
            warn!(
                "{} agent thread(s) did not exit in time, forcing exit",
                handles.len()
            );
            std::process::exit(1);
        }
        thread::sleep(Duration::from_millis(200));
    }

    info!("All agents stopped");

    Ok(())
}

#[cfg(test)]
mod scope_tests {
    use super::{issue_in_scope, mr_in_scope};
    use crate::gitlab::{Issue, MergeRequest};

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
}
