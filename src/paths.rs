//! The per-user on-disk layout of the application.
//!
//! One module owns the `~/.<app>` tree so no other file constructs it:
//! every directory under it is a named function here, and a rename of the
//! tree (or a future `XDG_STATE_HOME` move) is a one-file change.
//!
//! Layout:
//! - `~/.potlatch/sessions/<session-id>/` — per-session `run.log` + `context`
//! - `~/.potlatch/agents/<agent-id>/current` — the agent's current session
//! - `~/.potlatch/auth-provider/` — cross-process auth-provider locks
//! - `~/.potlatch/memory/<hash>/` — durable project memory
//! - `~/.potlatch/web-chrome-profile/` — headless browser profile

use std::path::{Path, PathBuf};

/// The application name as it appears on the wire and in UI chrome.
/// Defined once, by the package manifest: renaming the crate renames
/// everything — the CLI name, the `~/.<name>` data root, config discovery.
pub(crate) const APP_NAME: &str = env!("CARGO_PKG_NAME");

/// The name as a proper noun in prose ("Potlatch"): first letter uppercased.
/// Prompt text and agent-facing context refer to the system by name; this
/// keeps those references in step with a crate rename.
pub(crate) fn display_name() -> String {
    let mut chars = APP_NAME.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

/// The config file name: `<name>.toml`, searched in the working directory
/// (and `config/`) when no path is given.
pub(crate) const CONFIG_FILE_NAME: &str = concat!(env!("CARGO_PKG_NAME"), ".toml");

/// The user's home directory, or `.` when `HOME` is unset (Windows service
/// contexts); a relative path keeps log/data writes local instead of failing.
pub(crate) fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The application's per-user root: `~/.potlatch`. Every other directory
/// in this module is derived from it.
pub(crate) fn app_home() -> PathBuf {
    home_dir().join(format!(".{APP_NAME}"))
}

/// Per-session data: `~/.potlatch/sessions/<session-id>/` (run.log, context).
pub(crate) fn sessions_dir() -> PathBuf {
    app_home().join("sessions")
}

/// Per-agent current-session markers: `~/.potlatch/agents/<agent-id>/`.
pub(crate) fn agents_dir() -> PathBuf {
    app_home().join("agents")
}

/// Cross-process auth-provider coordination: `~/.potlatch/auth-provider/`.
pub(crate) fn auth_provider_state_dir() -> PathBuf {
    app_home().join("auth-provider")
}

/// Durable project memory: `~/.potlatch/memory/<repo-hash>/`.
pub(crate) fn memory_dir() -> PathBuf {
    app_home().join("memory")
}

/// The browser profile under an explicit home directory: `~/.potlatch/web-chrome-profile/`.
/// Callers that read `HOME` themselves (rather than the ambient environment) —
/// e.g. to fail fast with an actionable message when it is unset — resolve
/// through this.
pub(crate) fn web_profile_dir_with_home(home: &Path) -> PathBuf {
    home.join(format!(".{APP_NAME}")).join("web-chrome-profile")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_directory_derives_from_the_app_home() {
        let home = app_home();
        assert!(home.starts_with(home_dir()));
        assert!(home.ends_with(format!(".{APP_NAME}")));

        assert!(sessions_dir().starts_with(&home));
        assert!(agents_dir().starts_with(&home));
        assert!(auth_provider_state_dir().starts_with(&home));
        assert!(memory_dir().starts_with(&home));
        assert_eq!(
            web_profile_dir_with_home(&home_dir()),
            home.join("web-chrome-profile")
        );
    }

    #[test]
    fn explicit_home_profile_matches_the_default_layout() {
        assert_eq!(
            web_profile_dir_with_home(&home_dir()),
            home_dir()
                .join(format!(".{APP_NAME}"))
                .join("web-chrome-profile")
        );
    }

    #[test]
    fn names_come_from_the_package_manifest() {
        assert_eq!(APP_NAME, env!("CARGO_PKG_NAME"));
        assert_eq!(CONFIG_FILE_NAME, format!("{APP_NAME}.toml"));
    }

    #[test]
    fn display_name_capitalizes_the_first_letter() {
        let display = display_name();
        assert_eq!(
            display,
            APP_NAME[..1].to_uppercase() + &APP_NAME[1..],
            "display name must track the manifest name"
        );
        assert!(!display.is_empty());
    }
}
