use std::fs::{self, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A typed JSON file persisted through an atomic temporary-file rename.
pub struct StateStore<T> {
    path: PathBuf,
    marker: PhantomData<fn() -> T>,
}

impl<T> StateStore<T> {
    /// Create a store backed by `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            marker: PhantomData,
        }
    }

    /// Return the backing file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remove the state file, succeeding when it does not exist.
    pub fn remove(&self) -> Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("Failed to remove state at {}", self.path.display())),
        }
    }
}

impl<T: DeserializeOwned> StateStore<T> {
    /// Load typed state, returning `None` when the file does not exist.
    pub fn load(&self) -> Result<Option<T>> {
        let content = match fs::read(&self.path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("Failed to read state at {}", self.path.display()));
            }
        };
        serde_json::from_slice(&content)
            .with_context(|| format!("Failed to parse state JSON at {}", self.path.display()))
            .map(Some)
    }
}

impl<T: Serialize> StateStore<T> {
    /// Serialize and atomically replace the state file.
    pub fn save(&self, value: &T) -> Result<()> {
        let json = serde_json::to_vec_pretty(value).context("Failed to serialize state JSON")?;
        atomic_write(&self.path, &json)
            .with_context(|| format!("Failed to write state at {}", self.path.display()))
    }
}

pub(crate) fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("state path must end in a UTF-8 file name")?;

    loop {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.tmp",
            std::process::id(),
            sequence
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(mut file) => {
                let result = (|| -> Result<()> {
                    file.write_all(content)?;
                    file.sync_all()?;
                    fs::rename(&temporary, path)?;
                    fs::File::open(parent)?.sync_all()?;
                    Ok(())
                })();
                if result.is_err() {
                    let _ = fs::remove_file(&temporary);
                }
                return result;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestState {
        value: String,
    }

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "potlatch-state-{name}-{}-{}",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn typed_state_roundtrips_and_removes() {
        let dir = test_dir("roundtrip");
        let store = StateStore::new(dir.join("state.json"));
        let expected = TestState {
            value: "saved".into(),
        };

        assert_eq!(store.load().unwrap(), None);
        store.save(&expected).unwrap();
        assert_eq!(store.load().unwrap(), Some(expected));
        store.remove().unwrap();
        assert_eq!(store.load().unwrap(), None);
        store.remove().unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_state_is_an_error() {
        let dir = test_dir("corrupt");
        let store: StateStore<TestState> = StateStore::new(dir.join("state.json"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(store.path(), b"not json").unwrap();

        assert!(
            store
                .load()
                .unwrap_err()
                .to_string()
                .contains("parse state JSON")
        );

        let _ = fs::remove_dir_all(dir);
    }
}
