//! Agent workspace paths and repository bootstrap on disk.

use std::path::Path;

use anyhow::{Context, Result};

use super::git::GitRepo;
use super::settings;

pub fn require_gitlab_repo() -> Result<String> {
    settings::settings()
        .require_gitlab_repo()
        .map(str::to_string)
}

pub fn agent_dir(base_dir: &str, project_name: &str, agent_id: &str) -> String {
    Path::new(base_dir)
        .join(format!("{project_name}-{agent_id}"))
        .to_string_lossy()
        .into_owned()
}

pub fn work_dir(base_dir: &str, project_name: &str, agent_id: &str) -> String {
    Path::new(base_dir)
        .join(format!("{project_name}-{agent_id}"))
        .join(project_name)
        .to_string_lossy()
        .into_owned()
}

pub fn sessions_dir(base_dir: &str, project_name: &str) -> String {
    let dir = Path::new(base_dir)
        .join(format!("{project_name}-sessions"))
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::create_dir_all(&dir);
    dir
}

pub fn ensure_agent_repo(
    base_dir: &str,
    gitlab_repo: &str,
    project_name: &str,
    agent_id: &str,
) -> Result<String> {
    let container = agent_dir(base_dir, project_name, agent_id);
    let repo_path = work_dir(base_dir, project_name, agent_id);
    let git_repo = GitRepo::new(repo_path.clone());
    if !git_repo.exists() {
        let repo_dir = Path::new(&repo_path);
        if repo_dir.exists() {
            tracing::info!("Removing partial directory {}...", repo_path);
            std::fs::remove_dir_all(repo_dir)
                .context("Failed to remove partial clone directory")?;
        }
        std::fs::create_dir_all(&container)
            .context("Failed to create agent container directory")?;
        tracing::info!("Cloning repository into {}...", repo_path);
        git_repo.clone(gitlab_repo)?;
    }
    Ok(repo_path)
}

pub fn extract_project_name(repo_url: &str) -> Result<String> {
    let parts: Vec<&str> = repo_url.trim_end_matches('/').split('/').collect();
    let name = parts
        .last()
        .context("Invalid repository URL")?
        .trim_end_matches(".git");
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_project_name_parses_gitlab_urls() {
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project.git").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project/").unwrap(),
            "project"
        );
    }
}
