//! Shared SSH command helpers used by agents that execute commands on remote
//! servers (ops, qa).

use anyhow::{Result, ensure};

/// Single-quote a string for safe interpolation into a remote shell command.
/// Every embedded `'` is replaced with `'\''` so the value is treated as a
/// literal by the remote shell.
pub fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Validate a remote filesystem path. Rejects shell metacharacters that could
/// allow command injection when the path is interpolated into an SSH command.
pub fn validate_remote_path(path: &str) -> Result<()> {
    ensure!(!path.is_empty(), "remote path must not be empty");
    for ch in path.chars() {
        ensure!(
            !matches!(ch, '\0' | '\n' | '\r' | ';' | '|' | '&' | '$' | '`'),
            "remote path contains unsupported character: {ch:?}"
        );
    }
    Ok(())
}

/// Validate an SSH identity (username or hostname). Rejects empty strings and
/// any whitespace, which would break the `user@host` target format.
pub fn validate_ssh_identity(value: &str, field: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{field} must not be empty");
    ensure!(
        !value.chars().any(|c| c.is_whitespace()),
        "{field} must not contain whitespace"
    );
    Ok(())
}
