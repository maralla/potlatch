//! Agent workspace paths and repository bootstrap on disk.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use super::git::GitRepo;
use super::hosting::{self, CodeHostingClient};
use super::settings::AgentSettings;
use crate::core::agent::{AgentModel, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::runtime::AgentRuntime as CoreAgentRuntime;
use crate::core::workflow::AgentSpawnContext;

/// Shared runtime resources for one code-hosting-backed agent instance.
pub struct AgentWorkspace {
    pub agent_id: String,
    pub project_name: String,
    pub working_dir: String,
    pub sessions_dir: String,
    pub git_repo: GitRepo,
    pub hosting: Arc<dyn CodeHostingClient>,
    pub scope_label: String,
    pub model: AgentModel,
    pub core: CoreAgentRuntime,
}

pub struct AgentBootstrap<'a> {
    ctx: &'a AgentSpawnContext,
    model_preferences: ModelPreferences,
}

impl<'a> AgentBootstrap<'a> {
    pub fn new(ctx: &'a AgentSpawnContext, model_preferences: ModelPreferences) -> Self {
        Self {
            ctx,
            model_preferences,
        }
    }

    pub fn build(self) -> Result<AgentWorkspace> {
        self.ctx
            .workflow
            .config
            .agent(self.ctx.agent_name)
            .with_context(|| format!("[agent.{}] section required", self.ctx.agent_name))?;
        let settings = AgentSettings::from_config(&self.ctx.workflow.config)?;
        let repo_url = settings.require_repo_url()?.to_string();
        let project_name = extract_project_name(&repo_url)?;
        let agent_id = self.ctx.runtime.agent_id().to_string();
        ensure_agent_repo(
            &self.ctx.workflow.base_dir,
            &repo_url,
            &project_name,
            &agent_id,
            Arc::clone(&self.ctx.workflow.shutdown),
            Arc::clone(&self.ctx.workflow.activity),
        )?;
        let working_dir = work_dir(&self.ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&self.ctx.workflow.base_dir, &project_name);
        let git_repo = GitRepo::new(working_dir.clone(), Arc::clone(&self.ctx.workflow.shutdown));
        let hosting = hosting::create_client(
            &working_dir,
            &repo_url,
            Arc::clone(&self.ctx.workflow.shutdown),
        )?;
        let model = AgentModel::connect(self.ctx, working_dir.clone(), self.model_preferences)?;

        Ok(AgentWorkspace {
            agent_id,
            project_name,
            working_dir,
            sessions_dir,
            git_repo,
            hosting,
            scope_label: settings.scope_label,
            model,
            core: self.ctx.runtime.clone(),
        })
    }
}

pub fn repo_banner(config: &Config, banner: &mut Banner) {
    if let Ok(settings) = AgentSettings::from_config(config)
        && let Some(repo) = settings.repo_url()
    {
        banner.set_once("repo", repo);
    }
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
    repo_url: &str,
    project_name: &str,
    agent_id: &str,
    shutdown: Arc<AtomicBool>,
    activity: crate::core::activity::SharedActivityReporter,
) -> Result<String> {
    let container = agent_dir(base_dir, project_name, agent_id);
    let repo_path = work_dir(base_dir, project_name, agent_id);
    let git_repo = GitRepo::new(repo_path.clone(), shutdown);
    if git_repo.exists() {
        let existing_url = git_repo.remote_url().ok().flatten();
        if existing_url.as_deref() == Some(repo_url) {
            return Ok(repo_path);
        }
        tracing::info!(
            "Repo at {} has remote {:?} but config says {:?}; removing and re-cloning",
            repo_path,
            existing_url,
            repo_url,
        );
        std::fs::remove_dir_all(&repo_path).context("Failed to remove stale repo directory")?;
    } else {
        let repo_dir = Path::new(&repo_path);
        if repo_dir.exists() {
            tracing::info!("Removing partial directory {}...", repo_path);
            std::fs::remove_dir_all(repo_dir)
                .context("Failed to remove partial clone directory")?;
        }
    }
    std::fs::create_dir_all(&container).context("Failed to create agent container directory")?;
    let _activity = activity.start(format!("cloning {agent_id} into {repo_path}"));
    let clone_result = git_repo.clone(repo_url);
    drop(_activity);
    clone_result?;
    tracing::info!("Clone done {}", repo_path);
    Ok(repo_path)
}

pub fn extract_project_name(repo_url: &str) -> Result<String> {
    let trimmed = repo_url.trim_end_matches('/');
    anyhow::ensure!(!trimmed.is_empty(), "Invalid repository URL");
    let parts: Vec<&str> = trimmed.split('/').collect();
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
    fn extract_project_name_parses_repo_urls() {
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("https://gitlab.com/user/project.git").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("https://github.com/user/project").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("git@gitlab.com:user/project.git").unwrap(),
            "project"
        );
        assert_eq!(
            extract_project_name("git@github.com:user/project.git").unwrap(),
            "project"
        );
    }

    #[test]
    fn extract_project_name_rejects_empty_url() {
        assert!(extract_project_name("").is_err());
    }
}
