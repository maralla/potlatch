//! Project-level persistent memory: fundamental facts that survive across
//! sessions, keyed by the repository's git remote URL.
//!
//! Facts are stored as a markdown file at `~/.potlatch/memory/<hash>.md` where
//! `hash` is a deterministic hash of the remote URL. This keeps memory stable
//! across different clone locations of the same repository.
//!
//! During context compaction, the LLM receives the existing facts alongside
//! the tool outputs being summarized, and produces a fresh, complete set of
//! facts that **overwrites** the file. This keeps the memory concise and
//! up-to-date — stale facts are replaced, not accumulated.
//!
//! The conciseness is enforced by the LLM prompt, not a hard line limit — the
//! model is instructed to keep only a few essential bullet points.

use std::fs;
use std::path::PathBuf;

/// Load project facts from the persistent memory file for the given cwd.
///
/// Reads the git remote URL from the repository at `cwd`, hashes it, and loads
/// `~/.potlatch/memory/<hash>.md`. Returns `None` if the repo has no remote or
/// the memory file doesn't exist yet.
pub fn load_facts(cwd: &str) -> Option<String> {
    let memory_path = memory_path_for_cwd(cwd)?;
    let content = fs::read_to_string(&memory_path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Overwrite the memory file with a fresh, complete set of facts.
/// Called during compaction — the LLM produces a merged set of existing + new
/// facts, and this replaces the file entirely. Empty facts are filtered.
pub fn save_facts(cwd: &str, facts: &[String]) {
    let Some(memory_path) = memory_path_for_cwd(cwd) else {
        return;
    };

    let lines: Vec<String> = facts
        .iter()
        .map(|f| f.trim())
        .filter(|f| !f.is_empty())
        .map(|f| {
            // Ensure each fact is a bullet point
            if f.starts_with("- ") || f.starts_with("* ") {
                f.to_string()
            } else {
                format!("- {f}")
            }
        })
        .collect();

    if lines.is_empty() {
        return;
    }

    if let Some(parent) = memory_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let content = lines.join("\n") + "\n";
    let _ = fs::write(&memory_path, content);
}

/// Build the memory file path for the repo at `cwd`.
/// Returns `None` if the repo has no remote.origin.url.
fn memory_path_for_cwd(cwd: &str) -> Option<PathBuf> {
    let remote_url = git_remote_url(cwd)?;
    let hash = hash_str(&remote_url);
    Some(
        home_dir()
            .join(".potlatch")
            .join("memory")
            .join(format!("{hash}.md")),
    )
}

/// Get the git remote URL for the repository at `cwd`.
fn git_remote_url(cwd: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if url.is_empty() { None } else { Some(url) }
}

/// Simple deterministic hash (FNV-1a) for the remote URL.
fn hash_str(s: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn home_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home);
    }
    PathBuf::from(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_str("https://gitlab.com/group/project");
        let h2 = hash_str("https://gitlab.com/group/project");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_differs_for_different_urls() {
        let h1 = hash_str("https://gitlab.com/group/project-a");
        let h2 = hash_str("https://gitlab.com/group/project-b");
        assert_ne!(h1, h2);
    }

    fn test_path() -> std::path::PathBuf {
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("potlatch_mem_test_{id}.md"));
        std::fs::remove_file(&path).ok();
        path
    }

    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    #[test]
    fn save_facts_overwrites_not_appends() {
        let path = test_path();

        save_facts_to_path(&path, &["fact A".into(), "fact B".into()]);
        let facts = load_facts_from_path(&path).unwrap();
        assert!(facts.contains("fact A"));
        assert!(facts.contains("fact B"));

        save_facts_to_path(&path, &["fact C".into(), "fact D".into()]);
        let facts2 = load_facts_from_path(&path).unwrap();
        assert!(
            !facts2.contains("fact A"),
            "old fact should be gone after overwrite"
        );
        assert!(
            !facts2.contains("fact B"),
            "old fact should be gone after overwrite"
        );
        assert!(facts2.contains("fact C"));
        assert!(facts2.contains("fact D"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_facts_filters_empty_lines() {
        let path = test_path();

        save_facts_to_path(&path, &["".into(), "   ".into(), "real fact".into()]);
        let facts = load_facts_from_path(&path).unwrap();
        assert_eq!(facts.lines().filter(|l| !l.is_empty()).count(), 1);
        assert!(facts.contains("real fact"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_facts_empty_list_is_noop() {
        let path = test_path();

        save_facts_to_path(&path, &["fact A".into()]);
        assert!(load_facts_from_path(&path).is_some());

        save_facts_to_path(&path, &[]);
        let facts = load_facts_from_path(&path).unwrap();
        assert!(facts.contains("fact A"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_facts_returns_none_when_no_file() {
        let path = test_path();
        assert!(load_facts_from_path(&path).is_none());
    }

    fn save_facts_to_path(path: &std::path::Path, facts: &[String]) {
        let lines: Vec<String> = facts
            .iter()
            .map(|f| f.trim())
            .filter(|f| !f.is_empty())
            .map(|f| {
                if f.starts_with("- ") || f.starts_with("* ") {
                    f.to_string()
                } else {
                    format!("- {f}")
                }
            })
            .collect();

        if lines.is_empty() {
            return;
        }

        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let content = lines.join("\n") + "\n";
        let _ = fs::write(path, content);
    }

    fn load_facts_from_path(path: &std::path::Path) -> Option<String> {
        let content = fs::read_to_string(path).ok()?;
        let trimmed = content.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}
