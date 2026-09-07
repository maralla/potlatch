//! Filesystem layout for potlatch session data under `~/.potlatch`.
//!
//! Layout:
//! - `~/.potlatch/sessions/<session-id>/run.log` — the harness run log for
//!   one session (switched per `session/new`).
//! - `~/.potlatch/sessions/<session-id>/context` — the session's full
//!   current context, rewritten on every context update so it always holds
//!   the latest state.
//! - `~/.potlatch/agents/<agent-id>/current` — the session the agent is
//!   currently running. Emptied when the task finishes, so a recovered
//!   process resumes only interrupted work, never finished work.
//!
//! The marker is cross-process coordination: the harness child writes it at
//! `session/new` and empties it at `session/close`; the orchestrator empties
//! it when it abandons a session without closing it (external cancel).

use std::fs;
use std::path::{Path, PathBuf};

/// The two roots of the on-disk layout. Injectable so tests run against a
/// temp directory instead of the real `~/.potlatch`.
#[derive(Debug, Clone)]
pub(crate) struct SessionRoots {
    pub sessions: PathBuf,
    pub agents: PathBuf,
}

impl SessionRoots {
    /// The real `~/.potlatch` layout.
    pub fn real() -> Self {
        Self {
            sessions: super::logging_dir(),
            agents: super::home_dir().join(".potlatch").join("agents"),
        }
    }
}

/// `run.log` inside one session's directory.
pub(crate) fn session_run_log(root: &Path, session_id: &str) -> PathBuf {
    root.join(session_id).join("run.log")
}

/// The persisted-context file inside one session's directory.
pub(crate) fn session_context_file(root: &Path, session_id: &str) -> PathBuf {
    root.join(session_id).join("context")
}

/// The `current` session marker for one agent.
pub(crate) fn agent_current_marker(root: &Path, agent_id: &str) -> PathBuf {
    root.join(agent_id).join("current")
}

/// The session the marker names, if any. An empty or missing marker means
/// the agent has no session in flight.
pub(crate) fn read_current_session(marker: &Path) -> Option<String> {
    fs::read_to_string(marker)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|sid| !sid.is_empty())
}

pub(crate) fn write_current_session(marker: &Path, session_id: &str) {
    if let Some(parent) = marker.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(marker, session_id);
}

/// Empty the marker when it still names this session. A marker naming a
/// different session is left alone — that session, not this one, is the
/// one in flight.
pub(crate) fn clear_current_session(marker: &Path, session_id: &str) {
    if read_current_session(marker).as_deref() == Some(session_id) {
        let _ = fs::write(marker, "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_nest_under_their_roots() {
        let root = Path::new("/tmp/sessions");
        assert_eq!(
            session_run_log(root, "abc"),
            PathBuf::from("/tmp/sessions/abc/run.log")
        );
        assert_eq!(
            session_context_file(root, "abc"),
            PathBuf::from("/tmp/sessions/abc/context")
        );

        let agents = Path::new("/tmp/agents");
        assert_eq!(
            agent_current_marker(agents, "worker-7"),
            PathBuf::from("/tmp/agents/worker-7/current")
        );
    }

    #[test]
    fn marker_roundtrip_and_clear() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch_marker_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("worker-7").join("current");

        assert_eq!(read_current_session(&marker), None);

        write_current_session(&marker, "session-1");
        assert_eq!(read_current_session(&marker).as_deref(), Some("session-1"));

        // Clearing a marker that names a different session is a no-op.
        clear_current_session(&marker, "session-2");
        assert_eq!(read_current_session(&marker).as_deref(), Some("session-1"));

        clear_current_session(&marker, "session-1");
        assert_eq!(read_current_session(&marker), None);
        // The file exists but is empty, not deleted.
        assert!(marker.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn whitespace_only_marker_reads_as_none() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch_marker_ws_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("current");
        fs::write(&marker, "  \n").unwrap();
        assert_eq!(read_current_session(&marker), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
