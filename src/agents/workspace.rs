//! Agent workspace paths and repository bootstrap on disk.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use super::git::GitRepo;
use super::gitlab::GitLabClient;
use super::settings::AgentSettings;
use crate::core::agent::{AgentModel, ModelPreferences};
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentSpawnContext;

/// Shared runtime resources for one GitLab-backed agent instance.
///
/// Each role (`worker`, `reviewer`, `pmo`, `qa`, `ops`) embeds exactly one
/// instance of this type instead of destructuring it into separate,
/// duplicated struct fields. Role-specific state/config stays on the role's
/// own struct; this type only ever grows fields that are genuinely shared
/// bootstrap resources.
pub struct GitLabAgentRuntime {
    pub agent_id: String,
    pub project_name: String,
    pub working_dir: String,
    pub sessions_dir: String,
    pub git_repo: GitRepo,
    pub gitlab: GitLabClient,
    pub scope_label: String,
    pub model: AgentModel,
    /// The GENERAL core runtime for this instance: identity, shared
    /// shutdown, and observable health. Supplied by the supervisor.
    /// [`crate::core::agent::CoreAgent::runtime`] delegates to this field.
    pub core: AgentRuntime,
}

/// Builds the common runtime resources for a configured GitLab agent role.
pub struct GitLabAgentBootstrap<'a> {
    ctx: &'a AgentSpawnContext,
    model_preferences: ModelPreferences,
}

impl<'a> GitLabAgentBootstrap<'a> {
    pub fn new(ctx: &'a AgentSpawnContext, model_preferences: ModelPreferences) -> Self {
        Self {
            ctx,
            model_preferences,
        }
    }

    pub fn build(self) -> Result<GitLabAgentRuntime> {
        self.ctx
            .workflow
            .config
            .agent(self.ctx.agent_name)
            .with_context(|| format!("[agent.{}] section required", self.ctx.agent_name))?;
        let settings = AgentSettings::from_config(&self.ctx.workflow.config)?;
        let gitlab_repo = settings.require_gitlab_repo()?.to_string();
        let project_name = extract_project_name(&gitlab_repo)?;
        let agent_id = self.ctx.runtime.agent_id().to_string();
        ensure_agent_repo(
            &self.ctx.workflow.base_dir,
            &gitlab_repo,
            &project_name,
            &agent_id,
            Arc::clone(&self.ctx.workflow.shutdown),
            Arc::clone(&self.ctx.workflow.activity),
        )?;
        let working_dir = work_dir(&self.ctx.workflow.base_dir, &project_name, &agent_id);
        let sessions_dir = sessions_dir(&self.ctx.workflow.base_dir, &project_name);
        let git_repo = GitRepo::new(working_dir.clone(), Arc::clone(&self.ctx.workflow.shutdown));
        let gitlab = GitLabClient::new(
            working_dir.clone(),
            &gitlab_repo,
            Arc::clone(&self.ctx.workflow.shutdown),
        )?;
        let model = AgentModel::connect(self.ctx, working_dir.clone(), self.model_preferences)?;

        Ok(GitLabAgentRuntime {
            agent_id,
            project_name,
            working_dir,
            sessions_dir,
            git_repo,
            gitlab,
            scope_label: settings.scope_label,
            model,
            core: self.ctx.runtime.clone(),
        })
    }
}

pub fn gitlab_banner(config: &Config, banner: &mut Banner) {
    if let Ok(settings) = AgentSettings::from_config(config)
        && let Some(repo) = settings.gitlab_repo()
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
    gitlab_repo: &str,
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
        if existing_url.as_deref() == Some(gitlab_repo) {
            return Ok(repo_path);
        }
        tracing::info!(
            "Repo at {} has remote {:?} but config says {:?}; \
             removing and re-cloning",
            repo_path,
            existing_url,
            gitlab_repo,
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
    let clone_result = git_repo.clone(gitlab_repo);
    drop(_activity);
    clone_result?;
    tracing::info!("Clone done {}", repo_path);
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

    #[test]
    fn resolves_workspace_paths_from_core_identity() {
        let agent_id = "worker-2";
        assert_eq!(
            agent_dir("/srv/potlatch", "project", agent_id),
            "/srv/potlatch/project-worker-2"
        );
        assert_eq!(
            work_dir("/srv/potlatch", "project", agent_id),
            "/srv/potlatch/project-worker-2/project"
        );
    }

    /// Create a bare git repo at `remote_path` with one commit, suitable as a
    /// clone source. Returns the file:// URL to clone from.
    fn make_bare_remote(remote_path: &Path) -> String {
        std::fs::create_dir_all(remote_path).unwrap();
        run_git(remote_path, &["init", "--bare", "-b", "main"]);
        // Seed with a commit via a temporary working clone.
        let work = remote_path.parent().unwrap().join(format!(
            "{}-seed",
            remote_path.file_name().unwrap().to_string_lossy()
        ));
        let _ = std::fs::remove_dir_all(&work);
        run_git(
            remote_path.parent().unwrap(),
            &[
                "clone",
                &remote_path.to_string_lossy(),
                &work.to_string_lossy(),
            ],
        );
        run_git(&work, &["config", "user.email", "test@example.com"]);
        run_git(&work, &["config", "user.name", "test"]);
        std::fs::write(work.join("README.md"), "seed\n").unwrap();
        run_git(&work, &["add", "README.md"]);
        run_git(&work, &["commit", "-m", "seed"]);
        run_git(&work, &["push", "origin", "main"]);
        let _ = std::fs::remove_dir_all(&work);
        format!("file://{}", remote_path.to_string_lossy())
    }

    fn run_git(repo: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .expect("git command");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn ensure_agent_repo_reclones_when_remote_url_differs() {
        let tmp =
            std::env::temp_dir().join(format!("potlatch-workspace-reclone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let remote_a = tmp.join("remote-a.git");
        let remote_b = tmp.join("remote-b.git");
        let url_a = make_bare_remote(&remote_a);
        let url_b = make_bare_remote(&remote_b);

        let base_dir = tmp.join("base").to_string_lossy().into_owned();
        let project_name = "project";
        let agent_id = "worker-0";
        let shutdown = Arc::new(AtomicBool::new(false));
        let activity: crate::core::activity::SharedActivityReporter =
            Arc::new(crate::core::activity::NoopActivityReporter);

        // Initial clone from remote A.
        ensure_agent_repo(
            &base_dir,
            &url_a,
            project_name,
            agent_id,
            Arc::clone(&shutdown),
            Arc::clone(&activity),
        )
        .unwrap();

        let repo_path = work_dir(&base_dir, project_name, agent_id);
        let git_repo = GitRepo::new(repo_path.clone(), Arc::clone(&shutdown));
        assert!(git_repo.exists());
        assert_eq!(git_repo.remote_url().unwrap(), Some(url_a.clone()));

        // Simulate a config change: now point at remote B. The project name is
        // the same, so the workspace path is unchanged.
        ensure_agent_repo(
            &base_dir,
            &url_b,
            project_name,
            agent_id,
            Arc::clone(&shutdown),
            Arc::clone(&activity),
        )
        .unwrap();

        // The repo should now point at remote B.
        assert!(git_repo.exists());
        assert_eq!(git_repo.remote_url().unwrap(), Some(url_b));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ensure_agent_repo_reuses_when_remote_url_matches() {
        let tmp =
            std::env::temp_dir().join(format!("potlatch-workspace-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let remote = tmp.join("remote.git");
        let url = make_bare_remote(&remote);

        let base_dir = tmp.join("base").to_string_lossy().into_owned();
        let project_name = "project";
        let agent_id = "worker-0";
        let shutdown = Arc::new(AtomicBool::new(false));
        let activity: crate::core::activity::SharedActivityReporter =
            Arc::new(crate::core::activity::NoopActivityReporter);

        // Initial clone.
        ensure_agent_repo(
            &base_dir,
            &url,
            project_name,
            agent_id,
            Arc::clone(&shutdown),
            Arc::clone(&activity),
        )
        .unwrap();

        let repo_path = work_dir(&base_dir, project_name, agent_id);

        // Drop a marker file inside the repo to detect a re-clone (a fresh
        // clone would not contain it).
        let marker = std::path::Path::new(&repo_path).join(".potlatch-reuse-marker");
        std::fs::write(&marker, "present").unwrap();

        // Second call with the same URL should reuse, not re-clone.
        ensure_agent_repo(
            &base_dir,
            &url,
            project_name,
            agent_id,
            Arc::clone(&shutdown),
            Arc::clone(&activity),
        )
        .unwrap();

        assert!(
            marker.exists(),
            "marker survived — repo was reused, not re-cloned"
        );

        let git_repo = GitRepo::new(repo_path, Arc::clone(&shutdown));
        assert_eq!(git_repo.remote_url().unwrap(), Some(url));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
