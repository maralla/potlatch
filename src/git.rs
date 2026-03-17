use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use tracing::{debug, info};

pub struct GitRepo {
    pub path: String,
}

impl GitRepo {
    pub fn new(path: String) -> Self {
        Self { path }
    }

    pub fn exists(&self) -> bool {
        Path::new(&self.path).exists()
    }

    pub fn clone(&self, repo_url: &str) -> Result<()> {
        info!("Cloning repository {} to {}", repo_url, self.path);

        let output = Command::new("git")
            .args(["clone", repo_url, &self.path])
            .output()
            .context("Failed to execute git clone")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git clone failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn fetch(&self) -> Result<()> {
        debug!("Fetching latest changes in {}", self.path);

        let output = Command::new("git")
            .args(["fetch", "origin"])
            .current_dir(&self.path)
            .output()
            .context("Failed to execute git fetch")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git fetch failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn get_default_branch(&self) -> Result<String> {
        let output = Command::new("git")
            .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
            .current_dir(&self.path)
            .output()
            .context("Failed to get default branch")?;

        if !output.status.success() {
            return Ok("main".to_string());
        }

        let branch = String::from_utf8_lossy(&output.stdout)
            .trim()
            .strip_prefix("refs/remotes/origin/")
            .unwrap_or("main")
            .to_string();

        Ok(branch)
    }

    pub fn branch_exists(&self, branch_name: &str) -> Result<bool> {
        debug!("Checking if branch {} exists", branch_name);

        let output = Command::new("git")
            .args(["rev-parse", "--verify", branch_name])
            .current_dir(&self.path)
            .output()
            .context("Failed to check branch existence")?;

        Ok(output.status.success())
    }

    pub fn checkout_branch(&self, branch_name: &str) -> Result<()> {
        info!("Checking out existing branch {}", branch_name);

        let output = Command::new("git")
            .args(["checkout", branch_name])
            .current_dir(&self.path)
            .output()
            .context("Failed to checkout branch")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git checkout failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Checkout a branch and force-reset it to match the remote version.
    /// Uses `git checkout -B <branch> origin/<branch>` so the local branch
    /// always reflects the latest remote state.
    pub fn checkout_remote_branch(&self, branch_name: &str) -> Result<()> {
        debug!("Checking out branch {} from origin", branch_name);

        let output = Command::new("git")
            .args([
                "checkout",
                "-B",
                branch_name,
                &format!("origin/{}", branch_name),
            ])
            .current_dir(&self.path)
            .output()
            .context("Failed to checkout remote branch")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git checkout remote branch failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn create_branch_from(&self, branch_name: &str, base: &str) -> Result<()> {
        info!("Creating branch {} from origin/{}", branch_name, base);

        let output = Command::new("git")
            .args(["checkout", "-b", branch_name, &format!("origin/{}", base)])
            .current_dir(&self.path)
            .output()
            .context("Failed to create branch from base")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git checkout -b from base failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    /// Try merging a remote branch into the current branch.
    /// Returns Ok(true) on success, Ok(false) on conflict (aborts the merge).
    pub fn try_merge(&self, branch: &str) -> Result<bool> {
        info!("Merging origin/{} into current branch", branch);

        let output = Command::new("git")
            .args(["merge", &format!("origin/{}", branch), "--no-edit"])
            .current_dir(&self.path)
            .output()
            .context("Failed to execute git merge")?;

        if output.status.success() {
            return Ok(true);
        }

        // Merge conflict — abort and report
        let _ = Command::new("git")
            .args(["merge", "--abort"])
            .current_dir(&self.path)
            .output();

        Ok(false)
    }

    pub fn rev_parse(&self, rev: &str) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--short", rev])
            .current_dir(&self.path)
            .output()
            .context("Failed to rev-parse")?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub fn diff_against(&self, base_branch: &str) -> Result<String> {
        let output = Command::new("git")
            .args(["diff", &format!("origin/{}...HEAD", base_branch)])
            .current_dir(&self.path)
            .output()
            .context("Failed to compute diff")?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn has_staged_changes(&self) -> Result<bool> {
        let output = Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(&self.path)
            .output()
            .context("Failed to check for staged changes")?;
        // exit 0 = no diff, exit 1 = has diff
        Ok(!output.status.success())
    }

    pub fn add_all(&self) -> Result<()> {
        let output = Command::new("git")
            .args(["add", "."])
            .current_dir(&self.path)
            .output()
            .context("Failed to git add")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git add failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn commit(&self, message: &str) -> Result<()> {
        let output = Command::new("git")
            .args(["commit", "-m", message])
            .current_dir(&self.path)
            .output()
            .context("Failed to git commit")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git commit failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn push(&self, branch: &str) -> Result<()> {
        info!("Pushing branch {}", branch);

        let output = Command::new("git")
            .args(["push", "-u", "origin", branch])
            .current_dir(&self.path)
            .output()
            .context("Failed to git push")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git push failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }
}
