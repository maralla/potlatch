pub mod claim;
pub mod pmo;
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

pub fn extract_project_name(repo_url: &str) -> Result<String> {
    let parts: Vec<&str> = repo_url.trim_end_matches('/').split('/').collect();
    let name = parts
        .last()
        .context("Invalid repository URL")?
        .trim_end_matches(".git");
    Ok(name.to_string())
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

    prepare_repos(&base_dir, &git_repo_address, &config)?;

    let shutdown = Arc::new(AtomicBool::new(false));

    flag::register(SIGINT, Arc::clone(&shutdown)).context("Failed to register SIGINT handler")?;
    flag::register(SIGTERM, Arc::clone(&shutdown)).context("Failed to register SIGTERM handler")?;

    // Also handle double Ctrl+C: if shutdown flag is already set, force exit
    flag::register_conditional_shutdown(SIGINT, 1, Arc::clone(&shutdown))
        .context("Failed to register conditional shutdown")?;
    flag::register_conditional_shutdown(SIGTERM, 1, Arc::clone(&shutdown))
        .context("Failed to register conditional shutdown")?;

    let mut handles = Vec::new();

    for i in 0..config.worker.instances {
        let repo = git_repo_address.clone();
        let worker_config = config.worker.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) = worker::run(repo, worker_config, i, shutdown, base) {
                error!("Worker-{} error: {}", i, e);
            }
        }));
    }

    for i in 0..config.reviewer.instances {
        let repo = git_repo_address.clone();
        let reviewer_config = config.reviewer.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) = reviewer::run(repo, reviewer_config, i, shutdown, base) {
                error!("Reviewer-{} error: {}", i, e);
            }
        }));
    }

    for i in 0..config.pmo.instances {
        let repo = git_repo_address.clone();
        let pmo_config = config.pmo.clone();
        let shutdown = shutdown.clone();
        let base = base_dir.clone();
        handles.push(thread::spawn(move || {
            if let Err(e) = pmo::run(repo, pmo_config, i, shutdown, base) {
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
