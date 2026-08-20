//! Tool for maintaining grounded, durable project memory.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use tracing::info;

use super::Tool;
use crate::harness::memory::MemoryFactInput;

pub struct MemoryTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryUpdate {
    #[serde(default)]
    remember: Vec<MemoryFactInput>,
    #[serde(default)]
    forget: Vec<String>,
}

impl MemoryTool {
    fn execute_in(&self, args: &Value, cwd: &str, memory_dir: Option<&Path>) -> Result<String> {
        let update: MemoryUpdate =
            serde_json::from_value(args.clone()).context("invalid memory update")?;
        let memory_dir = memory_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(crate::harness::memory::default_memory_dir);
        let outcome = crate::harness::memory::update_facts_in(
            &memory_dir,
            cwd,
            &update.remember,
            &update.forget,
        )?;

        info!(
            "harness: durable memory remembered={} forgotten={} total={}",
            outcome.remembered, outcome.forgotten, outcome.total
        );
        Ok(format!(
            "Durable memory updated: remembered {}, forgot {}, total {}.",
            outcome.remembered, outcome.forgotten, outcome.total
        ))
    }
}

impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Maintain durable project memory across sessions. Store ONLY stable, project-wide architecture, invariants, conventions, integrations, or domain rules that are grounded in repository files and useful across unrelated future tasks. Do not store task progress, implementation notes, explored symbols, issue/MR details, commands, temporary state, or facts already clearly documented in project guidance. `remember` merges facts instead of replacing existing memory. Use `forget` with IDs shown in Durable Project Memory to remove stale facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remember": {
                        "type": "array",
                        "description": "Stable facts to merge into durable memory.",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["architecture", "invariant", "convention", "integration", "domain"]
                                },
                                "fact": {
                                    "type": "string",
                                    "description": "One stable, self-contained project fact in 20-240 characters."
                                },
                                "evidence": {
                                    "type": "array",
                                    "description": "One to five existing repository-relative file paths that ground this fact.",
                                    "items": { "type": "string" },
                                    "minItems": 1,
                                    "maxItems": 5
                                }
                            },
                            "required": ["category", "fact", "evidence"],
                            "additionalProperties": false
                        }
                    },
                    "forget": {
                        "type": "array",
                        "description": "Durable fact IDs to remove because they are stale or incorrect.",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "pattern": "^[0-9a-fA-F]{16}$"
                        }
                    }
                },
                "anyOf": [
                    { "required": ["remember"] },
                    { "required": ["forget"] }
                ],
                "additionalProperties": false
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        self.execute_in(args, cwd, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::tools::test_util;
    use std::fs;

    fn repo() -> test_util::TestDir {
        let dir = test_util::unique_test_dir();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        let remote_url = format!("https://example.com/test-memory-{}.git", std::process::id());
        std::process::Command::new("git")
            .args(["remote", "add", "origin", &remote_url])
            .current_dir(dir.path())
            .output()
            .unwrap();
        fs::write(dir.path().join("architecture.txt"), "ports").unwrap();
        dir
    }

    #[test]
    fn remembers_grounded_durable_facts_in_an_isolated_directory() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let tool = MemoryTool;
        let args = json!({
            "remember": [{
                "category": "architecture",
                "fact": "Core orchestration depends only on role-local ports.",
                "evidence": ["architecture.txt"]
            }]
        });
        let result = tool
            .execute_in(&args, repo.as_str(), Some(memory_dir.path()))
            .unwrap();
        assert!(result.contains("remembered 1"));

        let loaded =
            crate::harness::memory::load_facts_in(memory_dir.path(), repo.as_str()).unwrap();
        assert!(loaded.contains("[architecture]"));
        assert!(loaded.contains("Evidence: architecture.txt"));
    }

    #[test]
    fn rejects_empty_updates_and_note_shaped_facts() {
        let repo = repo();
        let memory_dir = test_util::unique_test_dir();
        let tool = MemoryTool;
        assert!(
            tool.execute_in(&json!({}), repo.as_str(), Some(memory_dir.path()))
                .unwrap_err()
                .to_string()
                .contains("remember or forget")
        );
        let note = json!({
            "remember": [{
                "category": "domain",
                "fact": "small note",
                "evidence": ["architecture.txt"]
            }]
        });
        assert!(
            tool.execute_in(&note, repo.as_str(), Some(memory_dir.path()))
                .unwrap_err()
                .to_string()
                .contains("characters")
        );
    }

    #[test]
    fn schema_requires_categories_and_repository_evidence() {
        let schema = MemoryTool.schema();
        let serialized = schema.to_string();
        assert!(serialized.contains("\"architecture\""));
        assert!(serialized.contains("\"evidence\""));
        assert!(serialized.contains("\"forget\""));
        assert!(serialized.contains("task progress"));
    }
}
