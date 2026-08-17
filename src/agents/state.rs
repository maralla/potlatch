//! Agent-layer, versioned JSON state persistence.
//!
//! Core must not own or assume any filesystem artifact/session persistence
//! — this lives in the agent layer instead. Every role (`worker`, `pmo`,
//! `qa`, `ops`) persists its own typed state through [`StateStore`]; how a
//! role reacts to a load failure (tolerant default vs. a hard error) is a
//! per-role policy decided by the caller, not by this module.
//!
//! ## Versioned envelope
//!
//! State is persisted as `{"version":1,"state":<payload>}`. Loading stays
//! backward compatible with files written before versioning existed (a bare
//! `<payload>` with no envelope): those load transparently and the file is
//! then atomically rewritten to the current envelope, so the legacy path is
//! only ever taken once per file. A file carrying an envelope with an
//! unsupported version, or content that fails to parse at all, is
//! quarantined — renamed aside, byte-for-byte, to a collision-safe sibling
//! path, never overwriting anything — and loading reports an error; the
//! caller decides whether that error is fatal for its role.
//!
//! Artifact writing (`crate::agents::artifact`) shares the same atomic
//! rename-based file write but is not JSON/versioned — it writes arbitrary
//! byte content (e.g. task context files) as-is.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// The only state envelope version this build understands.
const CURRENT_VERSION: u64 = 1;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A typed JSON file persisted through an atomic temporary-file rename,
/// wrapped in a `{"version":1,"state":...}` envelope.
pub(crate) struct StateStore<T> {
    path: PathBuf,
    marker: PhantomData<fn() -> T>,
}

impl<T> StateStore<T> {
    /// Create a store backed by `path`.
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            marker: PhantomData,
        }
    }

    /// Return the backing file path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Remove the state file, succeeding when it does not exist.
    pub(crate) fn remove(&self) -> Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("Failed to remove state at {}", self.path.display())),
        }
    }
}

impl<T: DeserializeOwned + Serialize> StateStore<T> {
    /// Load typed state, returning `None` when the file does not exist.
    ///
    /// A legacy (pre-envelope) file loads transparently and is atomically
    /// rewritten to the current envelope. A file with an unsupported
    /// envelope version, or content that cannot be parsed at all, is
    /// quarantined beside the original — original bytes preserved, nothing
    /// overwritten — and this returns `Err` describing why and where it was
    /// quarantined. Callers decide whether that error is fatal for their
    /// role (strict) or should be logged and treated as absent (tolerant).
    pub(crate) fn load(&self) -> Result<Option<T>> {
        let content = match fs::read(&self.path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("Failed to read state at {}", self.path.display()));
            }
        };

        match decode::<T>(&content) {
            Ok(Decoded::Current(value)) => Ok(Some(value)),
            Ok(Decoded::Legacy(value)) => {
                // Best-effort: a failed migration write must not turn a
                // successful legacy read into a load failure.
                if let Err(error) = self.save(&value) {
                    tracing::warn!(
                        "Failed to migrate legacy state at {} to the current envelope: {error:#}",
                        self.path.display()
                    );
                }
                Ok(Some(value))
            }
            Err(reason) => {
                let quarantined = quarantine(&self.path)?;
                bail!(
                    "{reason} at {} (quarantined to {})",
                    self.path.display(),
                    quarantined.display()
                );
            }
        }
    }
}

impl<T: Serialize> StateStore<T> {
    /// Serialize and atomically replace the state file with the current
    /// versioned envelope.
    pub(crate) fn save(&self, value: &T) -> Result<()> {
        #[derive(Serialize)]
        struct Envelope<'a, T> {
            version: u64,
            state: &'a T,
        }

        let envelope = Envelope {
            version: CURRENT_VERSION,
            state: value,
        };
        let json =
            serde_json::to_vec_pretty(&envelope).context("Failed to serialize state JSON")?;
        atomic_write(&self.path, &json)
            .with_context(|| format!("Failed to write state at {}", self.path.display()))
    }
}

enum Decoded<T> {
    /// Parsed from a `{"version":1,"state":...}` envelope.
    Current(T),
    /// Parsed from a bare, pre-envelope payload — needs migrating.
    Legacy(T),
}

/// Decode `content` into either a current-envelope or legacy value, or a
/// human-readable reason it could not be decoded at all.
fn decode<T: DeserializeOwned>(content: &[u8]) -> std::result::Result<Decoded<T>, String> {
    let value: Value =
        serde_json::from_slice(content).map_err(|e| format!("Malformed state JSON ({e})"))?;

    if let Value::Object(map) = &value
        && let Some(version) = map.get("version")
    {
        let version = version
            .as_u64()
            .ok_or_else(|| "state envelope 'version' is not an integer".to_string())?;
        if version != CURRENT_VERSION {
            return Err(format!("Unsupported state envelope version {version}"));
        }
        let state = map
            .get("state")
            .ok_or_else(|| "state envelope missing 'state' field".to_string())?;
        let parsed: T = serde_json::from_value(state.clone())
            .map_err(|e| format!("Invalid v{version} state payload ({e})"))?;
        return Ok(Decoded::Current(parsed));
    }

    let parsed: T = serde_json::from_value(value)
        .map_err(|e| format!("Unrecognized legacy state payload ({e})"))?;
    Ok(Decoded::Legacy(parsed))
}

/// Move a bad state file aside to a collision-safe sibling path, preserving
/// its bytes exactly (a rename, never a rewrite), so nothing is ever lost.
fn quarantine(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("state path must end in a UTF-8 file name")?;

    loop {
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{file_name}.quarantined.{}.{sequence}",
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => {
                fs::rename(path, &candidate).with_context(|| {
                    format!(
                        "Failed to quarantine {} to {}",
                        path.display(),
                        candidate.display()
                    )
                })?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to reserve a quarantine path beside {}",
                        path.display()
                    )
                });
            }
        }
    }
}

/// Write `content` to `path` through a temp-file-then-rename so readers
/// never observe a partially written file.
pub(crate) fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("state path must end in a UTF-8 file name")?;

    loop {
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
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
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestState {
        value: String,
    }

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "potlatch-agents-state-{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
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
    fn save_writes_a_v1_envelope_on_disk() {
        let dir = test_dir("v1-envelope-shape");
        let store = StateStore::new(dir.join("state.json"));
        store
            .save(&TestState {
                value: "saved".into(),
            })
            .unwrap();

        let on_disk: Value = serde_json::from_slice(&fs::read(store.path()).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["value"], "saved");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn legacy_unversioned_file_loads_transparently_and_is_migrated_to_v1() {
        let dir = test_dir("migration");
        fs::create_dir_all(&dir).unwrap();
        let store: StateStore<TestState> = StateStore::new(dir.join("state.json"));

        // A file written before the envelope existed: the bare payload,
        // with no "version"/"state" wrapper.
        fs::write(
            store.path(),
            serde_json::to_vec(&TestState {
                value: "legacy".into(),
            })
            .unwrap(),
        )
        .unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(
            loaded,
            Some(TestState {
                value: "legacy".into()
            })
        );

        // The legacy file is atomically rewritten to the v1 envelope, so a
        // second load takes the current-version path.
        let on_disk: Value = serde_json::from_slice(&fs::read(store.path()).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 1);
        assert_eq!(on_disk["state"]["value"], "legacy");
        assert_eq!(
            store.load().unwrap(),
            Some(TestState {
                value: "legacy".into()
            })
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unsupported_version_is_quarantined_and_reported_as_an_error() {
        let dir = test_dir("unsupported-version");
        fs::create_dir_all(&dir).unwrap();
        let store: StateStore<TestState> = StateStore::new(dir.join("state.json"));
        let original_bytes = br#"{"version":2,"state":{"value":"future"}}"#;
        fs::write(store.path(), original_bytes).unwrap();

        let error = store.load().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Unsupported state envelope version 2")
        );

        // The original path no longer holds the bad data — it was moved,
        // not deleted — and the quarantined sibling has the exact bytes.
        assert!(!store.path().exists());
        let quarantined = only_sibling_matching(&dir, "state.json.quarantined");
        assert_eq!(fs::read(quarantined).unwrap(), original_bytes);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn malformed_json_is_quarantined_and_reported_as_an_error() {
        let dir = test_dir("malformed");
        fs::create_dir_all(&dir).unwrap();
        let store: StateStore<TestState> = StateStore::new(dir.join("state.json"));
        fs::write(store.path(), b"not json").unwrap();

        let error = store.load().unwrap_err();
        assert!(error.to_string().contains("Malformed state JSON"));

        assert!(!store.path().exists());
        let quarantined = only_sibling_matching(&dir, "state.json.quarantined");
        assert_eq!(fs::read(quarantined).unwrap(), b"not json");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn envelope_missing_state_field_is_quarantined_and_reported_as_an_error() {
        let dir = test_dir("missing-state-field");
        fs::create_dir_all(&dir).unwrap();
        let store: StateStore<TestState> = StateStore::new(dir.join("state.json"));
        fs::write(store.path(), br#"{"version":1}"#).unwrap();

        let error = store.load().unwrap_err();
        assert!(error.to_string().contains("missing 'state' field"));
        assert!(!store.path().exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn quarantine_is_collision_safe_and_never_overwrites_a_prior_quarantine() {
        let dir = test_dir("quarantine-collision");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");

        fs::write(&path, b"first-bad-payload").unwrap();
        let first = quarantine(&path).unwrap();

        fs::write(&path, b"second-bad-payload").unwrap();
        let second = quarantine(&path).unwrap();

        assert_ne!(first, second);
        assert_eq!(fs::read(&first).unwrap(), b"first-bad-payload");
        assert_eq!(fs::read(&second).unwrap(), b"second-bad-payload");

        let _ = fs::remove_dir_all(dir);
    }

    fn only_sibling_matching(dir: &Path, needle: &str) -> PathBuf {
        let matches: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().contains(needle))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one match for {needle:?} in {matches:?}"
        );
        matches.into_iter().next().unwrap()
    }
}
