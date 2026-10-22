//! The file scope of one harness session.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

/// The files and directories one harness session may touch: the working
/// directory plus the extra roots its agent declared. Every part is
/// provided by the agent — the cwd when it opens the session
/// (`session/new`'s `cwd`, the agent's own working dir), the roots with
/// its model preferences (`session/new`'s `write_roots`, typically the
/// project's session directory where its session state lives) — and this
/// one structure is everything downstream of that declaration:
///
/// - the harness's file tools (`read`, `grep`, `glob`, `write`, `edit`)
///   resolve every path against it and reject anything outside;
/// - the cross-agent bus carries it as request metadata on every
///   bus-served tool call ([`Self::write_metadata`]);
/// - the sandbox agent (sandal) parses it back out of that metadata
///   ([`Self::from_metadata`]) and shares exactly these directories into
///   the session's VM — so a path that resolves inside the sandbox
///   resolves identically through the file tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionScope {
    cwd: String,
    roots: Vec<PathBuf>,
}

impl SessionScope {
    /// Build the scope from the agent-provided session parameters: the
    /// working directory and the declared roots (an agent that declared
    /// none gets a cwd-only scope).
    pub fn new(
        cwd: impl Into<String>,
        roots: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        Self {
            cwd: cwd.into(),
            roots: roots.into_iter().map(Into::into).collect(),
        }
    }

    /// The session's working directory (the agent's own working dir).
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// The extra roots the agent declared, beyond the working directory.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Whether `path` (already resolved to an absolute path) falls inside
    /// the scope: the working directory or one of the declared roots.
    /// Both sides are canonicalized before comparing so a symlinked
    /// working directory or root still matches paths resolved through it.
    pub fn permits(&self, path: &Path) -> bool {
        let cwd = Path::new(&self.cwd)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(&self.cwd));
        if path.starts_with(&cwd) {
            return true;
        }
        self.roots.iter().any(|root| {
            let root = root.canonicalize().unwrap_or_else(|_| root.clone());
            path.starts_with(&root)
        })
    }

    /// Inject the scope into a bus tool call's payload as request
    /// metadata (`_session_cwd`, `_session_write_roots`): the wire format
    /// every bus-served handler that needs the scope reads — every other
    /// handler ignores the keys.
    pub fn write_metadata(&self, payload: &mut Value) {
        if let Some(map) = payload.as_object_mut() {
            map.insert("_session_cwd".into(), Value::String(self.cwd.clone()));
            map.insert(
                "_session_write_roots".into(),
                Value::Array(
                    self.roots
                        .iter()
                        .map(|root| Value::String(root.display().to_string()))
                        .collect(),
                ),
            );
        }
    }

    /// Parse the scope back out of a bus tool call's payload (the format
    /// [`Self::write_metadata`] writes). The cwd is required — without it
    /// there is nothing to pin per-session state on; the roots are
    /// optional and default to empty.
    pub fn from_metadata(payload: &Value) -> Result<Self> {
        let cwd = payload
            .get("_session_cwd")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .context("the call carries no session working directory")?;
        let roots = payload
            .get("_session_write_roots")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(Self::new(cwd, roots))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn metadata_round_trips_cwd_and_roots() {
        let scope = SessionScope::new("/tmp/workdir", ["/tmp/project-sessions"]);
        let mut payload = json!({"command": "echo hi"});
        scope.write_metadata(&mut payload);
        let parsed = SessionScope::from_metadata(&payload).unwrap();
        assert_eq!(parsed, scope);
        assert_eq!(parsed.cwd(), "/tmp/workdir");
        assert_eq!(parsed.roots(), [PathBuf::from("/tmp/project-sessions")]);
    }

    #[test]
    fn metadata_round_trips_an_empty_root_list() {
        // An agent that declared no roots sends an empty list — the
        // metadata is always present, never silently absent.
        let scope = SessionScope::new("/tmp/workdir", [] as [PathBuf; 0]);
        let mut payload = json!({});
        scope.write_metadata(&mut payload);
        let parsed = SessionScope::from_metadata(&payload).unwrap();
        assert_eq!(parsed, scope);
        assert!(parsed.roots().is_empty());
    }

    #[test]
    fn metadata_requires_a_cwd() {
        let error = SessionScope::from_metadata(&json!({"_session_write_roots": []})).unwrap_err();
        assert!(error.to_string().contains("no session working directory"));
    }

    #[test]
    fn permits_paths_under_the_cwd_or_a_root_only() {
        let scope = SessionScope::new("/tmp/workdir", ["/tmp/project-sessions"]);
        assert!(scope.permits(Path::new("/tmp/workdir/src/main.rs")));
        assert!(scope.permits(Path::new("/tmp/workdir")));
        assert!(scope.permits(Path::new("/tmp/project-sessions/notes.md")));
        assert!(!scope.permits(Path::new("/tmp/elsewhere/file.rs")));
        assert!(!scope.permits(Path::new("/etc/passwd")));
        // A sibling directory sharing the cwd's name as a prefix is not
        // inside the scope ("/tmp/workdir-2" does not start with
        // "/tmp/workdir" as a path prefix).
        assert!(!scope.permits(Path::new("/tmp/workdir-2/file.rs")));
    }
}
