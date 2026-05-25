use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use tracing::debug;

pub struct GitRepo {
    pub path: String,
}

impl GitRepo {
    pub fn new(path: String) -> Self {
        Self { path }
    }

    pub fn exists(&self) -> bool {
        Path::new(&self.path).join(".git").exists()
    }

    pub fn clone(&self, repo_url: &str) -> Result<()> {
        debug!("Cloning repository {} to {}", repo_url, self.path);

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

    pub fn remote_branch_exists(&self, branch_name: &str) -> Result<bool> {
        debug!("Checking if remote branch origin/{} exists", branch_name);

        let output = Command::new("git")
            .args(["rev-parse", "--verify", &format!("origin/{}", branch_name)])
            .current_dir(&self.path)
            .output()
            .context("Failed to check remote branch existence")?;

        Ok(output.status.success())
    }

    /// Checkout a branch and force-reset it to match the remote version.
    /// Uses `git checkout -B <branch> origin/<branch>` so the local branch
    /// always reflects the latest remote state.
    pub fn checkout_remote_branch(&self, branch_name: &str) -> Result<()> {
        debug!("Checking out branch {} from origin", branch_name);

        let output = Command::new("git")
            .args([
                "checkout",
                "--force",
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
        debug!("Creating branch {} from origin/{}", branch_name, base);

        let output = Command::new("git")
            .args(["checkout", "-B", branch_name, &format!("origin/{}", base)])
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
    /// Attempt to merge origin/<branch> into the current branch.
    /// On conflict, aborts the merge and returns `Ok(false)`.
    pub fn try_merge(&self, branch: &str) -> Result<bool> {
        debug!("Merging origin/{} into current branch", branch);

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

    /// Merge origin/<branch> into the current branch, leaving conflict markers
    /// in the working tree if there are conflicts (does NOT abort).
    /// Returns `true` if merge succeeded cleanly, `false` if there are conflicts.
    pub fn merge_no_abort(&self, branch: &str) -> Result<bool> {
        debug!("Merging origin/{} into current branch (no abort)", branch);

        let output = Command::new("git")
            .args(["merge", &format!("origin/{}", branch), "--no-edit"])
            .current_dir(&self.path)
            .output()
            .context("Failed to execute git merge")?;

        Ok(output.status.success())
    }

    pub fn rev_parse(&self, rev: &str) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--short", rev])
            .current_dir(&self.path)
            .output()
            .context("Failed to rev-parse")?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub fn diff_stat_against(&self, base_branch: &str) -> Result<String> {
        let output = Command::new("git")
            .args(["diff", "--stat", &format!("origin/{}...HEAD", base_branch)])
            .current_dir(&self.path)
            .output()
            .context("Failed to compute diff stat")?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn changed_files_against(&self, base_branch: &str) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args([
                "diff",
                "--name-only",
                &format!("origin/{}...HEAD", base_branch),
            ])
            .current_dir(&self.path)
            .output()
            .context("Failed to compute changed files")?;
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }

    pub fn diff_patch_against(&self, base_branch: &str) -> Result<String> {
        let output = Command::new("git")
            .args([
                "diff",
                "--no-color",
                &format!("origin/{}...HEAD", base_branch),
            ])
            .current_dir(&self.path)
            .output()
            .context("Failed to compute diff patch against base")?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn has_diff_against(&self, base_branch: &str) -> Result<bool> {
        let output = Command::new("git")
            .args(["diff", "--quiet", &format!("origin/{}...HEAD", base_branch)])
            .current_dir(&self.path)
            .output()
            .context("Failed to check diff against base")?;
        Ok(!output.status.success())
    }

    /// Returns true if the current working tree (committed + staged +
    /// unstaged) differs from the given commit ref in any way.
    pub fn has_changes_since(&self, base_ref: &str) -> Result<bool> {
        // Check for committed changes beyond base_ref
        let rev_output = Command::new("git")
            .args(["rev-list", "--count", &format!("{}..HEAD", base_ref)])
            .current_dir(&self.path)
            .output()
            .context("Failed to check for commits since base")?;
        let commit_count: u64 = String::from_utf8_lossy(&rev_output.stdout)
            .trim()
            .parse()
            .unwrap_or(0);
        if commit_count > 0 {
            return Ok(true);
        }

        // Check for staged changes
        let staged = Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .current_dir(&self.path)
            .output()
            .context("Failed to check staged changes")?;
        if !staged.status.success() {
            return Ok(true);
        }

        // Check for unstaged changes (working tree)
        let unstaged = Command::new("git")
            .args(["diff", "--quiet"])
            .current_dir(&self.path)
            .output()
            .context("Failed to check unstaged changes")?;
        if !unstaged.status.success() {
            return Ok(true);
        }

        // Check for untracked files
        let untracked = Command::new("git")
            .args(["ls-files", "--others", "--exclude-standard"])
            .current_dir(&self.path)
            .output()
            .context("Failed to check untracked files")?;
        let has_untracked = !String::from_utf8_lossy(&untracked.stdout).trim().is_empty();

        Ok(has_untracked)
    }

    pub fn diff_shortstat_since(&self, base_ref: &str) -> Result<String> {
        let output = Command::new("git")
            .args(["diff", "--shortstat", &format!("{}..HEAD", base_ref)])
            .current_dir(&self.path)
            .output()
            .context("Failed to compute short diff stat since base")?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub fn changed_files_since(&self, base_ref: &str) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args(["diff", "--name-only", &format!("{}..HEAD", base_ref)])
            .current_dir(&self.path)
            .output()
            .context("Failed to list changed files since base")?;
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }

    pub fn delete_remote_branch(&self, branch_name: &str) -> Result<()> {
        debug!("Deleting remote branch origin/{}", branch_name);
        let output = Command::new("git")
            .args(["push", "origin", "--delete", branch_name])
            .current_dir(&self.path)
            .output()
            .context("Failed to delete remote branch")?;
        if !output.status.success() {
            anyhow::bail!(
                "Git push --delete failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
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
        // Exclude Potlatch-generated task context (PMO/worker/reviewer prompts) from commits.
        let output = Command::new("git")
            .args(["add", "--", ".", ":(exclude).potlatch-context"])
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
        debug!("Pushing branch {}", branch);

        let output = Command::new("git")
            .args(["push", "--force", "-u", "origin", branch])
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

    pub fn reset_hard(&self) -> Result<()> {
        debug!("Resetting working directory in {}", self.path);

        let output = Command::new("git")
            .args(["reset", "--hard"])
            .current_dir(&self.path)
            .output()
            .context("Failed to git reset --hard")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git reset --hard failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let output = Command::new("git")
            .args(["clean", "-fd"])
            .current_dir(&self.path)
            .output()
            .context("Failed to git clean -fd")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git clean failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    pub fn delete_local_branch(&self, branch_name: &str) -> Result<()> {
        debug!("Deleting local branch {}", branch_name);

        let output = Command::new("git")
            .args(["branch", "-D", branch_name])
            .current_dir(&self.path)
            .output()
            .context("Failed to delete branch")?;

        if !output.status.success() {
            anyhow::bail!(
                "Git branch -D failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }
}
