use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tracing::{debug, warn};

use crate::core::retry::with_backoff_retries;

pub struct GitRepo {
    pub path: String,
    shutdown: Arc<AtomicBool>,
}

impl GitRepo {
    pub fn new(path: String, shutdown: Arc<AtomicBool>) -> Self {
        Self { path, shutdown }
    }

    pub fn exists(&self) -> bool {
        Path::new(&self.path).join(".git").exists()
    }

    /// Returns the `remote.origin.url` of this repo, or `None` if there is no
    /// origin remote (e.g. not a git repo, or no origin configured).
    pub fn remote_url(&self) -> Result<Option<String>> {
        let output = Command::new("git")
            .args(["remote", "get-url", "origin"])
            .current_dir(&self.path)
            .output()
            .context("Failed to execute git remote get-url")?;
        if !output.status.success() {
            return Ok(None);
        }
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if url.is_empty() {
            Ok(None)
        } else {
            Ok(Some(url))
        }
    }

    fn command_error(output: &std::process::Output) -> String {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stderr.trim().is_empty() {
            stdout.to_string()
        } else if stdout.trim().is_empty() {
            stderr.to_string()
        } else {
            format!("{stderr}{stdout}")
        }
    }

    pub fn clone(&self, repo_url: &str) -> Result<()> {
        with_backoff_retries(
            &self.shutdown,
            &format!("git clone into {}", self.path),
            || {
                debug!("Cloning repository {} to {}", repo_url, self.path);

                let output = Command::new("git")
                    .args(["clone", repo_url, &self.path])
                    .output()
                    .context("Failed to execute git clone")?;

                if !output.status.success() {
                    anyhow::bail!("Git clone failed: {}", Self::command_error(&output));
                }

                Ok(())
            },
        )
    }

    pub fn fetch(&self) -> Result<()> {
        with_backoff_retries(
            &self.shutdown,
            &format!("git fetch in {}", self.path),
            || {
                debug!("Fetching latest changes in {}", self.path);

                let output = Command::new("git")
                    .args(["fetch", "origin"])
                    .current_dir(&self.path)
                    .output()
                    .context("Failed to execute git fetch")?;

                if !output.status.success() {
                    let err = Self::command_error(&output);
                    // Ref namespace conflict: a branch like `hotfix` conflicts
                    // with `hotfix/branch` because Git can't have both a file
                    // and a directory at the same ref path. Prune stale refs and
                    // retry once. If it still fails, the conflict is on the
                    // remote (both branches exist) — fetch only the default
                    // branch ref to avoid the conflicting ref path.
                    if err.contains("cannot lock ref") || err.contains("could not be updated") {
                        debug!("git fetch ref conflict, pruning and retrying: {err}");
                        let _ = Command::new("git")
                            .args(["remote", "prune", "origin"])
                            .current_dir(&self.path)
                            .output();
                        let retry = Command::new("git")
                            .args(["fetch", "origin"])
                            .current_dir(&self.path)
                            .output()
                            .context("Failed to execute git fetch (retry)")?;
                        if retry.status.success() {
                            return Ok(());
                        }
                        let retry_err = Self::command_error(&retry);
                        if retry_err.contains("cannot lock ref")
                            || retry_err.contains("could not be updated")
                        {
                            debug!(
                                "git fetch still has ref conflict after prune, fetching HEAD only"
                            );
                            let head = Command::new("git")
                                .args(["fetch", "origin", "HEAD"])
                                .current_dir(&self.path)
                                .output()
                                .context("Failed to execute git fetch HEAD")?;
                            if head.status.success() {
                                return Ok(());
                            }
                            anyhow::bail!("Git fetch failed: {}", Self::command_error(&head));
                        }
                        anyhow::bail!("Git fetch failed: {retry_err}");
                    }
                    anyhow::bail!("Git fetch failed: {err}");
                }

                Ok(())
            },
        )
    }

    /// Fetch the latest tips for specific remote branches (e.g. MR source and target).
    pub fn fetch_branches(&self, branches: &[&str]) -> Result<()> {
        if branches.is_empty() {
            return self.fetch();
        }

        let spec: Vec<String> = branches
            .iter()
            .map(|branch| format!("{branch}:refs/remotes/origin/{branch}"))
            .collect();
        let mut args = vec!["fetch", "origin"];
        args.extend(spec.iter().map(String::as_str));

        with_backoff_retries(
            &self.shutdown,
            &format!("git fetch branches {:?} in {}", branches, self.path),
            || {
                debug!("Fetching branches {:?} in {}", branches, self.path);

                let output = Command::new("git")
                    .args(&args)
                    .current_dir(&self.path)
                    .output()
                    .context("Failed to execute git fetch for branches")?;

                if !output.status.success() {
                    anyhow::bail!("Git fetch failed: {}", Self::command_error(&output));
                }

                Ok(())
            },
        )
    }

    pub fn remote_short_sha(&self, branch: &str) -> Result<String> {
        self.rev_parse(&format!("origin/{branch}"))
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

    /// Paths with unmerged index entries (merge/rebase in progress).
    pub fn list_unmerged_paths(&self) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args(["diff", "--name-only", "--diff-filter=U"])
            .current_dir(&self.path)
            .output()
            .context("Failed to list unmerged paths")?;

        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }

    /// Tracked or untracked files that still contain Git conflict markers.
    pub fn list_conflict_marker_files(&self) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args(["grep", "-l", "^<<<<<<<"])
            .current_dir(&self.path)
            .output()
            .context("Failed to scan for conflict markers")?;

        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect());
        }

        // git grep exits 1 when there are no matches.
        if output.status.code() == Some(1) {
            return Ok(Vec::new());
        }

        anyhow::bail!(
            "git grep for conflict markers failed: {}",
            Self::command_error(&output)
        );
    }

    /// True when the working tree still has an in-progress merge or conflict markers.
    pub fn merge_conflicts_present(&self) -> Result<bool> {
        Ok(!self.list_unmerged_paths()?.is_empty()
            || !self.list_conflict_marker_files()?.is_empty())
    }

    /// True when `git merge` left a merge in progress (`.git/MERGE_HEAD` exists).
    pub fn is_merge_in_progress(&self) -> Result<bool> {
        let output = Command::new("git")
            .args(["rev-parse", "-q", "--verify", "MERGE_HEAD"])
            .current_dir(&self.path)
            .output()
            .context("Failed to check merge in progress")?;
        Ok(output.status.success())
    }

    /// Returns true when `origin/<branch>` is an ancestor of `descendant_rev`.
    pub fn remote_branch_is_ancestor_of(&self, branch: &str, descendant_rev: &str) -> Result<bool> {
        let output = Command::new("git")
            .args([
                "merge-base",
                "--is-ancestor",
                &format!("origin/{branch}"),
                descendant_rev,
            ])
            .current_dir(&self.path)
            .output()
            .context("Failed to check branch ancestry")?;
        Ok(output.status.success())
    }

    /// Ensure `HEAD` contains all commits from `origin/<target>`, merging if needed.
    /// Returns `Ok(true)` when the branch is mergeable/up-to-date, `Ok(false)` when conflicts remain.
    pub fn verify_up_to_date_with_target(&self, target: &str) -> Result<bool> {
        if self.merge_conflicts_present()? || self.is_merge_in_progress()? {
            return Ok(false);
        }
        if self.remote_branch_is_ancestor_of(target, "HEAD")? {
            return Ok(true);
        }
        self.try_merge(target)
    }

    /// When a merge is in progress and conflict markers are gone, stage unmerged paths
    /// so Git treats them as resolved (common when the agent edits files but does not run `git add`).
    pub fn stage_resolved_unmerged_paths(&self) -> Result<bool> {
        if !self.is_merge_in_progress()? {
            return Ok(false);
        }
        if !self.list_conflict_marker_files()?.is_empty() {
            return Ok(false);
        }
        let unmerged = self.list_unmerged_paths()?;
        if unmerged.is_empty() {
            return Ok(false);
        }
        for path in &unmerged {
            let output = Command::new("git")
                .args(["add", "--", path])
                .current_dir(&self.path)
                .output()
                .with_context(|| format!("Failed to git add resolved merge path {path}"))?;
            if !output.status.success() {
                anyhow::bail!(
                    "Git add failed for {path}: {}",
                    Self::command_error(&output)
                );
            }
        }
        Ok(self.list_unmerged_paths()?.is_empty())
    }

    /// Conclude an in-progress merge when conflict markers are gone and all paths are staged.
    pub fn complete_merge_if_ready(&self, message: &str) -> Result<bool> {
        if !self.is_merge_in_progress()? {
            return Ok(false);
        }
        let _ = self.stage_resolved_unmerged_paths()?;
        self.add_all()?;
        if !self.list_conflict_marker_files()?.is_empty() {
            return Ok(false);
        }
        if !self.list_unmerged_paths()?.is_empty() {
            return Ok(false);
        }
        if !self.has_staged_changes()? {
            return Ok(false);
        }
        self.commit(message)?;
        Ok(true)
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
        with_backoff_retries(
            &self.shutdown,
            &format!("git push --delete origin {branch_name}"),
            || self.delete_remote_branch_once(branch_name),
        )
    }

    /// Best-effort remote branch delete for cleanup paths that must not block the agent loop.
    pub fn delete_remote_branch_best_effort(&self, branch_name: &str) {
        match self.delete_remote_branch_once(branch_name) {
            Ok(()) => {}
            Err(e) => {
                warn!(
                    "Best-effort delete of remote branch {} failed: {}",
                    branch_name, e
                );
            }
        }
    }

    fn delete_remote_branch_once(&self, branch_name: &str) -> Result<()> {
        debug!("Deleting remote branch origin/{}", branch_name);
        let output = Command::new("git")
            .args(["push", "origin", "--delete", branch_name])
            .current_dir(&self.path)
            .output()
            .context("Failed to delete remote branch")?;
        if !output.status.success() {
            anyhow::bail!("Git push --delete failed: {}", Self::command_error(&output));
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
            .args([
                "add",
                "--",
                ".",
                ":(exclude).potlatch-context",
                ":(exclude).potlatch",
            ])
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
        with_backoff_retries(&self.shutdown, &format!("git push origin {branch}"), || {
            debug!("Pushing branch {}", branch);

            let output = Command::new("git")
                .args(["push", "--force", "-u", "origin", branch])
                .current_dir(&self.path)
                .output()
                .context("Failed to git push")?;

            if !output.status.success() {
                anyhow::bail!("Git push failed: {}", Self::command_error(&output));
            }

            Ok(())
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::AtomicBool;

    fn run_git(repo: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .expect("git command");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            GitRepo::command_error(&output)
        );
    }

    #[test]
    fn list_conflict_marker_files_detects_leftover_markers() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-git-conflict-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        run_git(&dir, &["init"]);
        run_git(&dir, &["config", "user.email", "test@example.com"]);
        run_git(&dir, &["config", "user.name", "test"]);
        fs::write(dir.join("conflicted.rs"), "fn main() {\n<<<<<<< HEAD\n}\n").unwrap();
        run_git(&dir, &["add", "conflicted.rs"]);
        run_git(&dir, &["commit", "-m", "add conflict markers"]);

        let repo = GitRepo::new(
            dir.to_string_lossy().into_owned(),
            Arc::new(AtomicBool::new(false)),
        );
        let files = repo.list_conflict_marker_files().unwrap();
        assert_eq!(files, vec!["conflicted.rs".to_string()]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_resolved_unmerged_paths_stages_clean_unmerged_files() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-git-stage-unmerged-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        run_git(&dir, &["init", "-b", "main"]);
        run_git(&dir, &["config", "user.email", "test@example.com"]);
        run_git(&dir, &["config", "user.name", "test"]);
        fs::write(dir.join("file.txt"), "base\n").unwrap();
        run_git(&dir, &["add", "file.txt"]);
        run_git(&dir, &["commit", "-m", "base"]);

        run_git(&dir, &["checkout", "-b", "feature"]);
        fs::write(dir.join("file.txt"), "feature\n").unwrap();
        run_git(&dir, &["commit", "-am", "feature"]);

        run_git(&dir, &["checkout", "main"]);
        fs::write(dir.join("file.txt"), "main\n").unwrap();
        run_git(&dir, &["commit", "-am", "main"]);

        let merge = Command::new("git")
            .args(["merge", "feature", "--no-edit"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(!merge.status.success());

        fs::write(dir.join("file.txt"), "resolved\n").unwrap();

        let repo = GitRepo::new(
            dir.to_string_lossy().into_owned(),
            Arc::new(AtomicBool::new(false)),
        );
        assert!(repo.is_merge_in_progress().unwrap());
        assert!(!repo.list_unmerged_paths().unwrap().is_empty());
        assert!(repo.stage_resolved_unmerged_paths().unwrap());
        assert!(repo.list_unmerged_paths().unwrap().is_empty());
        assert!(repo.complete_merge_if_ready("Merge feature").unwrap());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remote_url_returns_origin_url() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-git-remote-url-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        run_git(&dir, &["init", "-b", "main"]);
        run_git(&dir, &["config", "user.email", "test@example.com"]);
        run_git(&dir, &["config", "user.name", "test"]);
        run_git(
            &dir,
            &[
                "remote",
                "add",
                "origin",
                "https://gitlab.example.com/group/project",
            ],
        );

        let repo = GitRepo::new(
            dir.to_string_lossy().into_owned(),
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(
            repo.remote_url().unwrap(),
            Some("https://gitlab.example.com/group/project".to_string())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remote_url_returns_none_without_origin() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-git-remote-none-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        run_git(&dir, &["init", "-b", "main"]);
        run_git(&dir, &["config", "user.email", "test@example.com"]);
        run_git(&dir, &["config", "user.name", "test"]);

        let repo = GitRepo::new(
            dir.to_string_lossy().into_owned(),
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(repo.remote_url().unwrap(), None);

        let _ = fs::remove_dir_all(&dir);
    }
}
