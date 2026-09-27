//! Shared sensitive-path guard for the built-in harness tools.
//!
//! Every tool that touches the filesystem funnels its target through this
//! module so the workspace-confinement and sensitive-path policy lives in one
//! place instead of being re-implemented per tool.
//!
//! The built-in tools do not call into it yet — each tool is wired up by its
//! own follow-up change — so the compiler reports the module as unused until
//! then.

use std::path::{Component, Path, PathBuf};

use super::WriteRoots;

/// File and directory names that must never be exposed to an agent, matched
/// against every component of a resolved path (case-insensitively).
///
/// A leading `*` is a suffix match (`*.pem` covers `server.pem`) and a trailing
/// `*` is a prefix match (`.env*` covers `.env.local`); everything else must
/// match a component exactly. This is a deny-list, not a security boundary on
/// its own: it complements the workspace confinement that
/// [`ensure_workspace_read`] enforces.
pub const SENSITIVE_NAMES: &[&str] = &[
    // Version control and tooling state.
    ".git",
    ".hg",
    ".svn",
    // Credentials, keys, and env files.
    ".ssh",
    ".aws",
    ".gnupg",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".dockercfg",
    "credentials",
    "credentials.json",
    "secret",
    "secrets",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    // A nested path that cannot be named by a single component.
    ".docker/config.json",
];

/// Whether `path` is sensitive: a dot-prefixed component, a known credential
/// name, or a component matching one of the deny-list patterns.
///
/// The check is lexical over the path as given, so callers should pass a
/// resolved (preferably canonicalized) path — see [`ensure_workspace_read`].
pub fn is_sensitive_path(path: &Path) -> bool {
    path.components().any(|c| {
        let Component::Normal(name) = c else {
            return false;
        };
        let Some(name) = name.to_str() else {
            // A non-UTF-8 component cannot be proven safe: fail closed.
            return true;
        };
        name.starts_with('.') || SENSITIVE_NAMES.iter().any(|pat| name_matches(name, pat))
    })
}

/// Match one path component against a deny-list pattern. A leading `*` matches
/// any prefix and a trailing `*` any suffix; a `*` elsewhere is literal.
fn name_matches(name: &str, pattern: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let pattern = pattern.to_ascii_lowercase();
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix) && name.len() > suffix.len();
    }
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => name == pattern,
    }
}

/// Resolve `path` for read-only access, enforcing the shared policy:
///
/// 1. Canonicalize the target, failing closed when that is impossible (a read
///    of a non-existent path is an error, never a best-effort path).
/// 2. Reject the read when the resolved path is sensitive (see
///    [`is_sensitive_path`]).
/// 3. Reject the read when the resolved path is outside `cwd` and outside
///    every configured root.
///
/// `cwd` is canonicalized too, so a workspace reached through a symlink still
/// matches the canonical resolved path. On success the canonicalized path is
/// returned.
pub fn ensure_workspace_read(
    path: &Path,
    cwd: &Path,
    roots: &WriteRoots,
) -> Result<PathBuf, String> {
    let resolved = path.canonicalize().map_err(|e| {
        format!(
            "path '{}' cannot be resolved for reading: {e}",
            path.display()
        )
    })?;

    if is_sensitive_path(&resolved) {
        return Err(format!(
            "path '{}' is sensitive and cannot be read.",
            resolved.display()
        ));
    }

    let cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    if !resolved.starts_with(&cwd) && !roots.contains(&resolved) {
        return Err(format!(
            "path '{}' is outside the working directory.",
            resolved.display()
        ));
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::super::test_util::unique_test_dir;
    use super::*;

    fn write_file(dir: &Path, rel: &str) {
        let full = dir.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, "x").unwrap();
    }

    #[test]
    fn detects_sensitive_absolute_paths() {
        assert!(is_sensitive_path(Path::new("/home/user/.ssh/id_rsa")));
        assert!(is_sensitive_path(Path::new("/home/user/.aws/credentials")));
        assert!(is_sensitive_path(Path::new(
            "/home/user/.gnupg/secring.gpg"
        )));
        assert!(is_sensitive_path(Path::new("/repo/.git/config")));
        assert!(is_sensitive_path(Path::new("/repo/deploy/service.pem")));
        assert!(is_sensitive_path(Path::new("/repo/deploy/service.key")));
        assert!(is_sensitive_path(Path::new("/repo/certs/identity.p12")));
        assert!(is_sensitive_path(Path::new("/repo/certs/identity.pfx")));
        assert!(is_sensitive_path(Path::new("/repo/.env")));
        assert!(is_sensitive_path(Path::new("/repo/.env.local")));
        assert!(is_sensitive_path(Path::new("/home/user/.netrc")));
        assert!(is_sensitive_path(Path::new("/home/user/.npmrc")));
        assert!(is_sensitive_path(Path::new(
            "/home/user/.docker/config.json"
        )));
        assert!(is_sensitive_path(Path::new("/repo/secrets")));
    }

    #[test]
    fn allows_ordinary_paths() {
        assert!(!is_sensitive_path(Path::new("/repo/src/main.rs")));
        assert!(!is_sensitive_path(Path::new("/repo/src/harness/read.rs")));
        assert!(!is_sensitive_path(Path::new("/repo/docs/environment.md")));
    }

    #[test]
    fn allows_non_sensitive_workspace_path() {
        let dir = unique_test_dir();
        write_file(dir.path(), "src/main.rs");
        let target = dir.path().join("src/main.rs");
        let resolved =
            ensure_workspace_read(&target, dir.path(), &WriteRoots::default()).expect("readable");
        assert!(resolved.ends_with("src/main.rs"));
    }

    #[test]
    fn rejects_sensitive_path_inside_workspace() {
        let dir = unique_test_dir();
        write_file(dir.path(), ".ssh/id_rsa");
        let target = dir.path().join(".ssh/id_rsa");
        let err = ensure_workspace_read(&target, dir.path(), &WriteRoots::default()).unwrap_err();
        assert!(err.contains("sensitive"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_sensitive_path_inside_configured_root() {
        let dir = unique_test_dir();
        let root = dir.path().join("scratch");
        write_file(&root, ".aws/credentials");
        let roots = WriteRoots::from_paths([root.clone()]);
        let err =
            ensure_workspace_read(&root.join(".aws/credentials"), dir.path(), &roots).unwrap_err();
        assert!(err.contains("sensitive"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_non_sensitive_path_outside_cwd() {
        let dir = unique_test_dir();
        let inside = dir.path().join("workspace");
        let outside = dir.path().join("outside");
        write_file(&outside, "notes.md");
        let err = ensure_workspace_read(&outside.join("notes.md"), &inside, &WriteRoots::default())
            .unwrap_err();
        assert!(err.contains("outside"), "unexpected error: {err}");
    }

    #[test]
    fn allows_non_sensitive_path_inside_configured_root() {
        let dir = unique_test_dir();
        let inside = dir.path().join("workspace");
        fs::create_dir_all(&inside).unwrap();
        let root = dir.path().join("scratch");
        write_file(&root, "notes.md");
        let roots = WriteRoots::from_paths([root.clone()]);
        let resolved = ensure_workspace_read(&root.join("notes.md"), &inside, &roots)
            .expect("configured root is readable");
        assert!(resolved.ends_with("notes.md"));
    }

    #[test]
    fn rejects_traversal_to_sensitive_path() {
        let dir = unique_test_dir();
        let workspace = dir.path().join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        write_file(dir.path(), ".ssh/id_rsa");
        let traversal = workspace.join("../.ssh/id_rsa");
        let err =
            ensure_workspace_read(&traversal, &workspace, &WriteRoots::default()).unwrap_err();
        assert!(err.contains("sensitive"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_nonexistent_path_by_failing_closed() {
        let dir = unique_test_dir();
        let missing = dir.path().join("does/not/exist.rs");
        let err = ensure_workspace_read(&missing, dir.path(), &WriteRoots::default()).unwrap_err();
        assert!(
            err.contains("cannot be resolved"),
            "unexpected error: {err}"
        );
    }
}
