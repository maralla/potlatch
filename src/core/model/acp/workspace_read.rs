//! Read text files under the agent workspace for ACP `fs/read_text_file`.

use std::fs;
use std::path::Path;

/// Maximum bytes read in one `fs/read_text_file` response (avoid huge allocations).
const MAX_READ_BYTES: u64 = 2 * 1024 * 1024;

/// Read a UTF-8 text file that resolves inside `workspace` (after canonicalization).
///
/// `raw_path` may be absolute or relative to `workspace`. Path traversal outside
/// `workspace` is rejected.
pub fn read_text_file_under_workspace(workspace: &Path, raw_path: &str) -> Result<String, String> {
    let workspace = workspace
        .canonicalize()
        .map_err(|e| format!("workspace canonicalize: {e}"))?;
    let path = Path::new(raw_path);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };
    let resolved = candidate
        .canonicalize()
        .map_err(|e| format!("path not found: {e}"))?;
    if !resolved.starts_with(&workspace) {
        return Err("path escapes workspace".into());
    }
    let meta = fs::metadata(&resolved).map_err(|e| e.to_string())?;
    if !meta.is_file() {
        return Err("not a regular file".into());
    }
    if meta.len() > MAX_READ_BYTES {
        return Err(format!(
            "file larger than {} MiB",
            MAX_READ_BYTES / 1024 / 1024
        ));
    }
    fs::read_to_string(&resolved).map_err(|e| e.to_string())
}

/// Optional 1-based start line and max line count (ACP `fs/read_text_file` params).
pub fn slice_by_line_range(content: &str, line: Option<u64>, limit: Option<u64>) -> String {
    let start_line = line.unwrap_or(1).max(1) as usize;
    let lines: Vec<&str> = content.lines().collect();
    let start_idx = start_line.saturating_sub(1);
    if start_idx >= lines.len() {
        return String::new();
    }
    let rest = &lines[start_idx..];
    let out = if let Some(lim) = limit.filter(|&l| l > 0) {
        &rest[..rest.len().min(lim as usize)]
    } else {
        rest
    };
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("potlatch-acp-{name}-{}", std::process::id()))
    }

    #[test]
    fn rejects_escape_attempt() {
        let base = scratch_dir("escape");
        let _ = fs::remove_dir_all(&base);
        let ws = base.join("ws");
        fs::create_dir_all(&ws).unwrap();
        fs::create_dir_all(base.join("other")).unwrap();
        fs::write(base.join("other").join("x.txt"), "no").unwrap();
        let err = read_text_file_under_workspace(&ws, "../other/x.txt").unwrap_err();
        assert!(err.contains("escape"), "expected escape error, got: {err}");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn reads_file_under_workspace() {
        let base = scratch_dir("read");
        let _ = fs::remove_dir_all(&base);
        let ws = base.join("repo");
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("a.txt"), "hello\nworld").unwrap();
        let got = read_text_file_under_workspace(&ws, "a.txt").unwrap();
        assert_eq!(got, "hello\nworld");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn line_limit_slices() {
        let s = "a\nb\nc\nd";
        assert_eq!(slice_by_line_range(s, Some(2), Some(2)), "b\nc");
        assert_eq!(slice_by_line_range(s, Some(1), None), s);
    }
}
