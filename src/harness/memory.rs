//! Grounded project-level memory that survives across harness sessions.
//!
//! Durable facts are keyed by the repository's git remote URL and stored as
//! structured JSON. Each fact has a category and repository evidence. Legacy
//! markdown notes are intentionally not loaded: they had no provenance or
//! quality boundary and mixed task notes with long-term project knowledge.

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tracing::warn;

const MEMORY_VERSION: u32 = 1;
const MAX_FACTS: usize = 32;
const MIN_FACT_CHARS: usize = 20;
const MAX_FACT_CHARS: usize = 240;
const MAX_EVIDENCE: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MemoryCategory {
    Architecture,
    Invariant,
    Convention,
    Integration,
    Domain,
}

impl MemoryCategory {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Architecture => "architecture",
            Self::Invariant => "invariant",
            Self::Convention => "convention",
            Self::Integration => "integration",
            Self::Domain => "domain",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MemoryFactInput {
    pub(crate) category: MemoryCategory,
    pub(crate) fact: String,
    pub(crate) evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DurableFact {
    id: String,
    category: MemoryCategory,
    fact: String,
    evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredMemory {
    version: u32,
    facts: Vec<DurableFact>,
}

impl Default for StoredMemory {
    fn default() -> Self {
        Self {
            version: MEMORY_VERSION,
            facts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryUpdateOutcome {
    pub(crate) remembered: usize,
    pub(crate) forgotten: usize,
    pub(crate) total: usize,
}

/// Load durable project memory and render it for model context.
pub fn load_facts(cwd: &str) -> Option<String> {
    load_facts_in(&default_memory_dir(), cwd)
}

pub(crate) fn load_facts_in(memory_dir: &Path, cwd: &str) -> Option<String> {
    let path = memory_path_for_cwd_in(memory_dir, cwd)?;
    match load_stored_memory(&path) {
        Ok(memory) if !memory.facts.is_empty() => Some(render_memory(&memory)),
        Ok(_) => None,
        Err(error) => {
            warn!("Failed to load durable memory {}: {error}", path.display());
            None
        }
    }
}

/// Merge grounded durable facts and forget selected fact IDs under a file lock.
pub(crate) fn update_facts_in(
    memory_dir: &Path,
    cwd: &str,
    remember: &[MemoryFactInput],
    forget: &[String],
) -> Result<MemoryUpdateOutcome> {
    ensure!(
        !remember.is_empty() || !forget.is_empty(),
        "memory update must remember or forget at least one fact"
    );
    let remote_url = git_remote_url(cwd).context("repository has no remote.origin.url")?;
    let hash = hash_str(&remote_url);
    fs::create_dir_all(memory_dir)
        .with_context(|| format!("create memory directory {}", memory_dir.display()))?;
    let path = memory_dir.join(format!("{hash}.json"));
    let lock_path = memory_dir.join(format!("{hash}.lock"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open memory lock {}", lock_path.display()))?;
    lock.lock_exclusive()
        .with_context(|| format!("lock memory {}", lock_path.display()))?;

    let result = update_locked(&path, cwd, remember, forget);
    if let Err(error) = FileExt::unlock(&lock) {
        warn!(
            "Failed to unlock durable memory {}: {error}",
            lock_path.display()
        );
    }
    result
}

fn update_locked(
    path: &Path,
    cwd: &str,
    remember: &[MemoryFactInput],
    forget: &[String],
) -> Result<MemoryUpdateOutcome> {
    let mut memory = load_stored_memory(path)?;
    let forget: HashSet<&str> = forget.iter().map(|id| id.trim()).collect();
    for id in &forget {
        ensure!(is_fact_id(id), "invalid memory fact ID {id:?}");
    }
    let before_forget = memory.facts.len();
    memory
        .facts
        .retain(|fact| !forget.contains(fact.id.as_str()));
    let forgotten = before_forget - memory.facts.len();

    let mut remembered = 0;
    for input in remember {
        let fact = validate_fact(cwd, input)?;
        if let Some(existing) = memory.facts.iter_mut().find(|item| item.id == fact.id) {
            *existing = fact;
        } else {
            memory.facts.push(fact);
        }
        remembered += 1;
    }

    memory.facts.sort_by(|left, right| {
        left.category
            .as_str()
            .cmp(right.category.as_str())
            .then_with(|| left.fact.cmp(&right.fact))
    });
    ensure!(
        memory.facts.len() <= MAX_FACTS,
        "durable memory may contain at most {MAX_FACTS} facts; forget stale facts first"
    );
    save_stored_memory(path, &memory)?;
    Ok(MemoryUpdateOutcome {
        remembered,
        forgotten,
        total: memory.facts.len(),
    })
}

fn validate_fact(cwd: &str, input: &MemoryFactInput) -> Result<DurableFact> {
    ensure!(
        !input.fact.contains(['\r', '\n']),
        "durable fact must be one line"
    );
    let fact = normalize_sentence(&input.fact);
    let char_count = fact.chars().count();
    ensure!(
        (MIN_FACT_CHARS..=MAX_FACT_CHARS).contains(&char_count),
        "durable fact must be {MIN_FACT_CHARS}..={MAX_FACT_CHARS} characters"
    );
    ensure!(
        (1..=MAX_EVIDENCE).contains(&input.evidence.len()),
        "durable fact requires 1..={MAX_EVIDENCE} evidence paths"
    );

    let cwd = Path::new(cwd)
        .canonicalize()
        .with_context(|| format!("canonicalize repository cwd {cwd}"))?;
    let mut evidence = BTreeSet::new();
    for raw in &input.evidence {
        let trimmed = raw.trim();
        ensure!(
            !trimmed.is_empty(),
            "memory evidence path must not be empty"
        );
        let relative = Path::new(trimmed);
        ensure!(
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "memory evidence must be a normalized repository-relative path: {trimmed:?}"
        );
        let resolved = cwd
            .join(relative)
            .canonicalize()
            .with_context(|| format!("memory evidence does not exist: {trimmed}"))?;
        ensure!(
            resolved.starts_with(&cwd),
            "memory evidence escapes the repository: {trimmed}"
        );
        evidence.insert(trimmed.to_string());
    }

    let id_source = format!("{}:{fact}", input.category.as_str());
    Ok(DurableFact {
        id: hash_str(&id_source),
        category: input.category,
        fact,
        evidence: evidence.into_iter().collect(),
    })
}

fn normalize_sentence(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_fact_id(value: &str) -> bool {
    value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn load_stored_memory(path: &Path) -> Result<StoredMemory> {
    if !path.exists() {
        return Ok(StoredMemory::default());
    }
    let content =
        fs::read(path).with_context(|| format!("read durable memory {}", path.display()))?;
    let memory: StoredMemory = serde_json::from_slice(&content)
        .with_context(|| format!("parse durable memory {}", path.display()))?;
    ensure!(
        memory.version == MEMORY_VERSION,
        "unsupported durable memory version {}",
        memory.version
    );
    Ok(memory)
}

fn save_stored_memory(path: &Path, memory: &StoredMemory) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(memory).context("serialize durable memory")?;
    let temp_path = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut file = File::create(&temp_path)
        .with_context(|| format!("create temporary memory {}", temp_path.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("write temporary memory {}", temp_path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("finish temporary memory {}", temp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync temporary memory {}", temp_path.display()))?;
    fs::rename(&temp_path, path)
        .with_context(|| format!("replace durable memory {}", path.display()))?;
    Ok(())
}

fn render_memory(memory: &StoredMemory) -> String {
    memory
        .facts
        .iter()
        .map(|fact| {
            format!(
                "- [{}][{}] {}\n  Evidence: {}",
                fact.id,
                fact.category.as_str(),
                fact.fact,
                fact.evidence.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn memory_path_for_cwd_in(memory_dir: &Path, cwd: &str) -> Option<PathBuf> {
    let remote_url = git_remote_url(cwd)?;
    Some(memory_dir.join(format!("{}.json", hash_str(&remote_url))))
}

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

fn hash_str(value: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub(crate) fn default_memory_dir() -> PathBuf {
    home_dir().join(".potlatch").join("memory")
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
    use crate::harness::tools::test_util;
    use std::sync::Arc;

    fn repo() -> test_util::TestDir {
        let dir = test_util::unique_test_dir();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "remote",
                "add",
                "origin",
                "https://example.com/group/project.git",
            ])
            .current_dir(dir.path())
            .output()
            .unwrap();
        fs::write(dir.path().join("architecture.txt"), "boundaries").unwrap();
        fs::write(dir.path().join("conventions.txt"), "rules").unwrap();
        dir
    }

    fn fact(category: MemoryCategory, text: &str, evidence: &str) -> MemoryFactInput {
        MemoryFactInput {
            category,
            fact: text.to_string(),
            evidence: vec![evidence.to_string()],
        }
    }

    #[test]
    fn hash_is_deterministic_and_remote_specific() {
        assert_eq!(
            hash_str("https://example.com/a"),
            hash_str("https://example.com/a")
        );
        assert_ne!(
            hash_str("https://example.com/a"),
            hash_str("https://example.com/b")
        );
    }

    #[test]
    fn remember_merges_grounded_facts_without_replacing_existing_memory() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        update_facts_in(
            memory_dir.path(),
            repo.as_str(),
            &[fact(
                MemoryCategory::Architecture,
                "Core orchestration depends only on role-local ports.",
                "architecture.txt",
            )],
            &[],
        )
        .unwrap();
        let outcome = update_facts_in(
            memory_dir.path(),
            repo.as_str(),
            &[fact(
                MemoryCategory::Convention,
                "Public APIs require documentation and focused tests.",
                "conventions.txt",
            )],
            &[],
        )
        .unwrap();

        assert_eq!(outcome.total, 2);
        let rendered = load_facts_in(memory_dir.path(), repo.as_str()).unwrap();
        assert!(rendered.contains("[architecture]"));
        assert!(rendered.contains("[convention]"));
        assert!(rendered.contains("Evidence: architecture.txt"));
    }

    #[test]
    fn remember_updates_a_fact_with_the_same_category_and_text() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let input = fact(
            MemoryCategory::Architecture,
            "Core orchestration depends only on role-local ports.",
            "architecture.txt",
        );
        update_facts_in(
            memory_dir.path(),
            repo.as_str(),
            std::slice::from_ref(&input),
            &[],
        )
        .unwrap();
        let mut updated = input;
        updated.evidence.push("conventions.txt".to_string());
        let outcome = update_facts_in(memory_dir.path(), repo.as_str(), &[updated], &[]).unwrap();
        assert_eq!(outcome.total, 1);
        let rendered = load_facts_in(memory_dir.path(), repo.as_str()).unwrap();
        assert!(rendered.contains("architecture.txt, conventions.txt"));
    }

    #[test]
    fn forget_removes_a_fact_by_rendered_id() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let input = fact(
            MemoryCategory::Invariant,
            "Every workflow preserves explicit shutdown boundaries.",
            "architecture.txt",
        );
        update_facts_in(memory_dir.path(), repo.as_str(), &[input], &[]).unwrap();
        let rendered = load_facts_in(memory_dir.path(), repo.as_str()).unwrap();
        let id = rendered
            .split_once('[')
            .and_then(|(_, rest)| rest.split_once(']'))
            .map(|(id, _)| id)
            .unwrap();
        let outcome =
            update_facts_in(memory_dir.path(), repo.as_str(), &[], &[id.to_string()]).unwrap();
        assert_eq!(outcome.forgotten, 1);
        assert_eq!(outcome.total, 0);
        assert!(load_facts_in(memory_dir.path(), repo.as_str()).is_none());
    }

    #[test]
    fn rejects_unstructured_or_ungrounded_notes() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let short = fact(MemoryCategory::Domain, "small note", "architecture.txt");
        assert!(
            update_facts_in(memory_dir.path(), repo.as_str(), &[short], &[])
                .unwrap_err()
                .to_string()
                .contains("characters")
        );
        let missing = fact(
            MemoryCategory::Domain,
            "This domain fact has enough text but no valid evidence.",
            "missing.txt",
        );
        assert!(
            update_facts_in(memory_dir.path(), repo.as_str(), &[missing], &[])
                .unwrap_err()
                .to_string()
                .contains("does not exist")
        );
    }

    #[test]
    fn ignores_legacy_markdown_notes() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let remote = git_remote_url(repo.as_str()).unwrap();
        fs::write(
            memory_dir.path().join(format!("{}.md", hash_str(&remote))),
            "- task-specific legacy note\n",
        )
        .unwrap();
        assert!(load_facts_in(memory_dir.path(), repo.as_str()).is_none());
    }

    #[test]
    fn concurrent_updates_merge_under_the_project_lock() {
        let repo = Arc::new(repo());
        let memory_dir = Arc::new(test_util::unique_test_dir());
        let mut threads = Vec::new();
        for (category, text, evidence) in [
            (
                MemoryCategory::Architecture,
                "Core orchestration depends only on role-local ports.",
                "architecture.txt",
            ),
            (
                MemoryCategory::Convention,
                "Public APIs require documentation and focused tests.",
                "conventions.txt",
            ),
        ] {
            let repo = Arc::clone(&repo);
            let memory_dir = Arc::clone(&memory_dir);
            threads.push(std::thread::spawn(move || {
                update_facts_in(
                    memory_dir.path(),
                    repo.as_str(),
                    &[fact(category, text, evidence)],
                    &[],
                )
                .unwrap();
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        let rendered = load_facts_in(memory_dir.path(), repo.as_str()).unwrap();
        assert!(rendered.contains("[architecture]"));
        assert!(rendered.contains("[convention]"));
    }
}
