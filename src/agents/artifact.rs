//! Agent-layer artifact writing: plain (non-JSON, unversioned) file content
//! written atomically beneath one directory, with a path-safety check on
//! the artifact name.
//!
//! Core must not own or assume any filesystem artifact/session persistence
//! — this lives in the agent layer instead, alongside `crate::agents::state`
//! (which it shares the atomic write primitive with).

use std::path::{Component, Path, PathBuf};

use anyhow::{Result, ensure};

use crate::agents::state::atomic_write;

/// Writes named artifacts atomically beneath one directory.
pub(crate) struct ArtifactStore {
    directory: PathBuf,
}

impl ArtifactStore {
    /// Create an artifact store rooted at `directory`.
    pub(crate) fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Resolve a safe simple artifact name beneath the store directory.
    pub(crate) fn path(&self, name: &str) -> Result<PathBuf> {
        ensure!(
            is_simple_name(name),
            "artifact name must be a safe simple name"
        );
        Ok(self.directory.join(name))
    }

    /// Atomically write an artifact and return its canonical path when available.
    pub(crate) fn write(&self, name: &str, content: impl AsRef<[u8]>) -> Result<PathBuf> {
        let path = self.path(name)?;
        atomic_write(&path, content.as_ref())?;
        Ok(std::fs::canonicalize(&path).unwrap_or(path))
    }
}

fn is_simple_name(name: &str) -> bool {
    if name.contains('\\') {
        return false;
    }
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(component)), None) if !component.is_empty()
    )
}

/// Write a task-context artifact (agent prompt context, archival/debugging
/// copies) beneath `work_dir` and return its path.
pub(crate) fn write_task_context_file(
    work_dir: &str,
    file_name: &str,
    content: &str,
) -> Result<String> {
    let store = ArtifactStore::new(work_dir);
    Ok(store
        .write(file_name, content)?
        .to_string_lossy()
        .into_owned())
}

/// Task context embedded inline in a task prompt is capped: the embedded copy
/// becomes part of the never-evicted user prompt, so an oversized brief would
/// sit in every turn for the whole task. Above the cap the prompt falls back
/// to referencing the on-disk file (the pre-embedding behavior).
const MAX_EMBEDDED_TASK_CONTEXT_CHARS: usize = 40_000;

/// Build the TASK CONTEXT section of a task prompt. The context is always
/// written to disk (reference, debugging, recovery); when it fits under the
/// embed cap it is also carried verbatim in the prompt, so the model works
/// from the brief directly instead of spending turns re-reading the file and
/// risking the brief being compacted away mid-task. `content` is `None` when
/// the caller could not read the file back — the section then references the
/// path.
pub(crate) fn task_context_section(path: &str, content: Option<&str>) -> String {
    match content {
        Some(content) if content.len() <= MAX_EMBEDDED_TASK_CONTEXT_CHARS => {
            format!("TASK CONTEXT (embedded in full below; also saved at {path}):\n\n{content}")
        }
        _ => {
            format!("TASK CONTEXT FILE (open and read it first — it has the full picture):\n{path}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        std::env::temp_dir().join(format!("potlatch-agents-artifacts-{}", std::process::id()))
    }

    #[test]
    fn writes_artifact_and_returns_its_path() {
        let dir = test_dir();
        let _ = std::fs::remove_dir_all(&dir);
        let store = ArtifactStore::new(&dir);

        let path = store.write("context.txt", "contents").unwrap();

        assert_eq!(std::fs::read_to_string(path).unwrap(), "contents");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_non_simple_artifact_names() {
        let store = ArtifactStore::new(test_dir());

        for name in [
            "",
            ".",
            "..",
            "../secret",
            "nested/file",
            "/absolute",
            r"nested\file",
        ] {
            assert!(store.path(name).is_err(), "{name:?} should be rejected");
        }
    }

    #[test]
    fn small_task_context_is_embedded_with_the_path() {
        let section = task_context_section("/tmp/task.md", Some("## Brief\nbody"));
        assert!(section.contains("/tmp/task.md"));
        assert!(section.contains("## Brief\nbody"));
    }

    #[test]
    fn oversized_task_context_falls_back_to_the_path() {
        let big = "x".repeat(MAX_EMBEDDED_TASK_CONTEXT_CHARS + 1);
        let section = task_context_section("/tmp/task.md", Some(&big));
        assert!(section.contains("/tmp/task.md"));
        // The oversized body is not embedded.
        assert!(!section.contains("xxxxx"));
    }

    #[test]
    fn unreadable_task_context_falls_back_to_the_path() {
        let section = task_context_section("/tmp/task.md", None);
        assert!(section.contains("/tmp/task.md"));
    }
}
