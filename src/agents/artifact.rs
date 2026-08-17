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
}
