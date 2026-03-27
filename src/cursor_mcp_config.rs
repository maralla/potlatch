//! Write or merge `.cursor/mcp.json` under each **agent container** (`<project>-<agent_id>/`) so it
//! survives `git clean -fd` in the inner clone. A **mirror** is copied into the repo worktree
//! (`<container>/<project>/.cursor/mcp.json`) for Cursor when `agent` uses the clone as cwd.
//!
//! **Git safety:** MCP config is applied only with `std::fs` (`write` / `copy`). We never use
//! `git add`, `git checkout`, or any command to materialize `mcp.json` from a revision, so it is
//! not tied to a commit and is not meant to be tracked. After `reset --hard` / `clean` or
//! `checkout --force`, [`crate::git::GitRepo`] re-copies from the canonical parent file so the
//! workspace stays in sync without reading tree state.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use crate::config::Config;
use tracing::warn;

/// Key under `mcpServers` for the codepair coordinator.
pub const CODEPAIR_MCP_SERVER_ID: &str = "codepair";

/// `<base>/<project>-<agent_id>/` — parent of the git worktree, safe from `git clean` in the clone.
fn agent_container(base_dir: &str, project_name: &str, agent_id: &str) -> PathBuf {
    Path::new(base_dir).join(format!("{}-{}", project_name, agent_id))
}

fn collect_agent_ids(config: &Config) -> Vec<String> {
    let mut ids = Vec::new();
    for i in 0..config.worker.instances {
        ids.push(format!("worker-{}", i));
    }
    for i in 0..config.reviewer.instances {
        ids.push(format!("reviewer-{}", i));
    }
    for i in 0..config.pmo.instances {
        ids.push(format!("pmo-{}", i));
    }
    ids
}

/// Copy canonical `mcp.json` from the agent container into the inner repo (for Cursor cwd).
pub fn sync_mcp_into_repo_from_container(repo_workdir: &Path) -> Result<()> {
    let container = repo_workdir
        .parent()
        .context("repo workdir has no parent (unexpected layout)")?;
    let src = container.join(".cursor/mcp.json");
    if !src.exists() {
        return Ok(());
    }
    let dst_dir = repo_workdir.join(".cursor");
    std::fs::create_dir_all(&dst_dir)
        .with_context(|| format!("failed to create {}", dst_dir.display()))?;
    let dst = dst_dir.join("mcp.json");
    std::fs::copy(&src, &dst)
        .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

/// Copy canonical `mcp.json` from the agent container into the repo (log only on failure).
/// Use at **worker / reviewer / PMO** thread start and, for roles that may skip `GitRepo::reset_hard`
/// for long stretches, at the beginning of each poll cycle.
pub fn refresh_mcp_mirror_best_effort(repo_workdir: &str) {
    if let Err(e) = sync_mcp_into_repo_from_container(Path::new(repo_workdir)) {
        warn!(
            "MCP: could not mirror .cursor/mcp.json into {} from agent container: {}",
            repo_workdir, e
        );
    }
}

/// Ensure each role has canonical MCP config under its container and a mirror in the clone (if present).
pub fn ensure_for_all_agent_workspaces(
    base_dir: &str,
    project_name: &str,
    config: &Config,
    mcp_url: &str,
) -> Result<()> {
    for agent_id in collect_agent_ids(config) {
        let container = agent_container(base_dir, project_name, &agent_id);
        ensure_one(&container, mcp_url, &agent_id, project_name)
            .with_context(|| format!("MCP config for container {}", container.display()))?;
    }
    Ok(())
}

fn ensure_one(
    container_root: &Path,
    mcp_url: &str,
    agent_id: &str,
    project_name: &str,
) -> Result<()> {
    let cursor_dir = container_root.join(".cursor");
    std::fs::create_dir_all(&cursor_dir)
        .with_context(|| format!("failed to create {}", cursor_dir.display()))?;

    let path = cursor_dir.join("mcp.json");
    let mut doc: Value = if path.exists() {
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parse {} as JSON", path.display()))?
    } else {
        json!({ "mcpServers": {} })
    };

    let obj = doc
        .as_object_mut()
        .with_context(|| format!("{}: root must be a JSON object", path.display()))?;

    if !obj.get("mcpServers").is_some_and(|v| v.is_object()) {
        obj.insert("mcpServers".to_string(), json!({}));
    }

    let servers = obj
        .get_mut("mcpServers")
        .and_then(|v| v.as_object_mut())
        .context("mcpServers must be a JSON object")?;

    let agent_url = format!("{}/{}", mcp_url, agent_id);
    let entry = json!({
        "url": agent_url
    });
    servers.insert(CODEPAIR_MCP_SERVER_ID.to_string(), entry);

    let out = serde_json::to_string_pretty(&doc).context("serialize mcp.json")?;
    std::fs::write(&path, &out).with_context(|| format!("write {}", path.display()))?;

    tracing::info!(
        "Cursor MCP config (canonical) at {} (server {:?}, agent_id {:?})",
        path.display(),
        CODEPAIR_MCP_SERVER_ID,
        agent_id
    );

    let repo_root = container_root.join(project_name);
    if repo_root.is_dir() {
        sync_mcp_into_repo_from_container(&repo_root)
            .with_context(|| format!("mirror MCP config into repo {}", repo_root.display()))?;
        tracing::info!(
            "Cursor MCP config mirrored to {}",
            repo_root.join(".cursor/mcp.json").display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn merges_without_dropping_other_servers() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("codepair-mcp-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("repo"))?;
        let path_canonical = dir.join(".cursor/mcp.json");
        let path_mirror = dir.join("repo").join(".cursor/mcp.json");

        ensure_one(&dir, "http://127.0.0.1:9/mcp", "worker-0", "repo")?;

        assert!(path_canonical.exists(), "canonical under container");
        assert!(path_mirror.exists(), "mirror in repo");

        let v: Value = serde_json::from_str(&fs::read_to_string(&path_canonical)?)?;
        assert!(v["mcpServers"]["codepair"].is_object());
        assert_eq!(
            v["mcpServers"]["codepair"]["url"],
            "http://127.0.0.1:9/mcp/worker-0"
        );

        fs::write(
            &path_canonical,
            r#"{"mcpServers":{"other":{"command":"x","args":[]},"codepair":{"url":"old","headers":{}}}}"#,
        )?;
        ensure_one(&dir, "http://127.0.0.1:9/mcp", "worker-0", "repo")?;
        let v2: Value = serde_json::from_str(&fs::read_to_string(&path_canonical)?)?;
        assert!(v2["mcpServers"]["other"].is_object());

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn sync_from_container_after_clean_scenario() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("codepair-mcp-sync-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(".cursor"))?;
        fs::create_dir_all(dir.join("repo"))?;
        fs::write(
            dir.join(".cursor/mcp.json"),
            r#"{"mcpServers":{"codepair":{"url":"http://x/mcp","headers":{"X-Codepair-Agent-ID":"r0"}}}}"#,
        )?;
        let repo = dir.join("repo");
        let _ = fs::remove_file(repo.join(".cursor/mcp.json"));
        sync_mcp_into_repo_from_container(&repo)?;
        assert!(repo.join(".cursor/mcp.json").exists());
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
