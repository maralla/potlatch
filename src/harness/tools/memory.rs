//! Memory tool: lets the model explicitly save fundamental project facts
//! to persistent memory that survives across sessions.

use anyhow::Result;
use serde_json::{Value, json};
use tracing::info;

use super::Tool;

pub struct MemoryTool;

impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Save fundamental project facts to persistent memory that survives across sessions and context compaction. Use ONLY for facts that are permanently true for the entire project and broadly useful for any future task: critical architecture rules, key conventions, and structural knowledge discovered through exploration. Do NOT save build/test/lint commands — those are easy to discover. Do NOT save anything already written in AGENTS.md, README, or other on-disk project config files — those are re-read each session, so storing them here is redundant duplication. Do NOT use for task-specific details or implementation notes. Think: 'is this fact written down anywhere in the repo, and if not, would it help a fresh session?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fundamental project facts to save. Each should be a single concise line (under 80 chars). The new facts replace all previous facts — include the full set you want to keep."
                    }
                },
                "required": ["facts"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let facts = args["facts"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'facts' array argument"))?;

        let facts: Vec<String> = facts
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if facts.is_empty() {
            return Ok("Error: 'facts' array must not be empty".into());
        }

        info!("harness: memory save {} facts", facts.len());
        for f in &facts {
            info!("harness: memory fact: {f}");
        }

        crate::harness::memory::save_facts(cwd, &facts);

        Ok(format!(
            "Saved {} fact(s) to persistent memory.",
            facts.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::tools::test_util;

    #[test]
    fn saves_facts_to_memory() {
        let dir = test_util::unique_test_dir();
        // Initialize a git repo so memory can find the remote
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .ok();
        let remote_url = format!("https://example.com/test-memory-{}.git", std::process::id());
        std::process::Command::new("git")
            .args(["remote", "add", "origin", &remote_url])
            .current_dir(dir.path())
            .output()
            .ok();

        let tool = MemoryTool;
        let args = json!({"facts": ["Go project", "Build: go build", "Test: go test ./..."]});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("3 fact(s)"));

        let loaded = crate::harness::memory::load_facts(dir.as_str()).unwrap();
        assert!(loaded.contains("Go project"));
        assert!(loaded.contains("Build: go build"));
    }

    #[test]
    fn rejects_empty_facts() {
        let dir = test_util::unique_test_dir();
        let tool = MemoryTool;
        let args = json!({"facts": []});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Error"));
    }
}
