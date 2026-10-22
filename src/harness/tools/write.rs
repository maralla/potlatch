//! Write tool: create or overwrite files.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;
use crate::core::scope::SessionScope;

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Create a new file or overwrite an existing file with the given content. Creates parent directories if needed. Use this for new files; prefer edit for modifying existing files. Paths must be within the working directory or an agent-managed session directory named by the task instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write. Relative to the working directory, or an absolute path within the working directory or an agent-managed session directory named by the task."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file"
                    },
                },
                "required": ["path", "content"]
            }
        })
    }

    fn execute(&self, args: &Value, scope: &SessionScope) -> Result<String> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'path' argument"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'content' argument"))?;

        let full_path = match super::resolve_write_path(path, scope) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };

        // Create parent directories if needed
        if let Some(parent) = full_path.parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("failed to create directories: {e}"))?;
        }

        let existed = full_path.exists();
        fs::write(&full_path, content)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", full_path.display()))?;

        let line_count = content.lines().count();
        let action = if existed { "Overwrote" } else { "Created" };
        Ok(format!(
            "{action} {path} ({line_count} lines, {} bytes)",
            content.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::*;

    fn tool() -> WriteTool {
        WriteTool
    }

    #[test]
    fn creates_new_file() {
        let dir = test_util::unique_test_dir();

        let args = json!({
            "path": "new.txt",
            "content": "line one\nline two\n"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(result.contains("Created"));
        assert!(result.contains("2 lines"));

        let content = std::fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("over.txt"), "old content").unwrap();

        let args = json!({
            "path": "over.txt",
            "content": "new content"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(result.contains("Overwrote"));

        let content = std::fs::read_to_string(dir.path().join("over.txt")).unwrap();
        assert_eq!(content, "new content");
    }

    #[test]
    fn creates_parent_directories() {
        let dir = test_util::unique_test_dir();

        let args = json!({
            "path": "nested/deep/file.txt",
            "content": "nested"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(result.contains("Created"));

        assert!(dir.path().join("nested/deep/file.txt").exists());
    }

    #[test]
    fn rejects_a_path_outside_the_scope() {
        // Absolute path outside the scope.
        let dir = test_util::unique_test_dir();
        let args = json!({
            "path": "/tmp/evil.txt",
            "content": "bad"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
        assert!(!dir.path().join("evil.txt").exists());
    }

    #[test]
    fn rejects_parent_traversal() {
        let dir = test_util::unique_test_dir();
        let args = json!({
            "path": "../escape.txt",
            "content": "bad"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
    }

    #[test]
    fn write_inside_a_declared_root_is_allowed() {
        let dir = test_util::unique_test_dir();
        let root = test_util::unique_test_dir();
        let target = root.path().join("outside.txt");
        let scope = super::super::SessionScope::new(dir.as_str(), [root.path().to_path_buf()]);

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "from outside"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(result.contains("Created"), "{result}");

        let content = std::fs::read_to_string(&target).unwrap();
        assert_eq!(content, "from outside");
    }

    #[test]
    fn absolute_path_inside_the_cwd_is_allowed() {
        // An absolute path that lands inside the working directory is fine —
        // the cwd is always in scope.
        let dir = test_util::unique_test_dir();
        let target = dir.path().join("in-cwd.txt");

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "in cwd"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "in cwd");
    }

    #[test]
    fn without_roots_the_cwd_is_the_whole_scope() {
        // An agent that declared no roots has the cwd as its whole scope:
        // nothing outside it is writable.
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("outside.txt");

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "bad"
        });
        let result = tool().execute(&args, &dir.scope()).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
        assert!(!target.exists());
    }

    #[test]
    fn write_to_a_declared_root_creates_parent_dirs() {
        let dir = test_util::unique_test_dir();
        let root = test_util::unique_test_dir();
        let target = root.path().join("nested/deep/script.py");
        let scope = super::super::SessionScope::new(dir.as_str(), [root.path().to_path_buf()]);

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "print('hi')"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn write_to_a_declared_root_overwrites_existing_file() {
        let dir = test_util::unique_test_dir();
        let root = test_util::unique_test_dir();
        let target = root.path().join("existing.txt");
        std::fs::write(&target, "old").unwrap();
        let scope = super::super::SessionScope::new(dir.as_str(), [root.path().to_path_buf()]);

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "new"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(result.contains("Overwrote"), "{result}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn write_inside_a_configured_root_is_allowed() {
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("knowledge/notes.md");
        let scope = super::super::SessionScope::new(dir.as_str(), [outside.path().to_path_buf()]);

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "notes"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn write_inside_the_cwd_is_allowed_with_roots() {
        let dir = test_util::unique_test_dir();
        let other = test_util::unique_test_dir();
        let target = dir.path().join("in-workspace.txt");

        // Roots are configured, but the write targets the cwd itself.
        let scope = super::super::SessionScope::new(dir.as_str(), [other.path().to_path_buf()]);
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "ws"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn write_outside_every_root_is_rejected() {
        let dir = test_util::unique_test_dir();
        let allowed = test_util::unique_test_dir();
        let forbidden = test_util::unique_test_dir();
        let target = forbidden.path().join("evil.txt");
        let scope = super::super::SessionScope::new(dir.as_str(), [allowed.path().to_path_buf()]);

        let args = json!({
            "path": target.to_string_lossy(),
            "content": "bad"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
        assert!(!target.exists());
    }

    #[test]
    fn write_to_an_arbitrary_system_path_is_rejected_with_roots() {
        let dir = test_util::unique_test_dir();
        let allowed = test_util::unique_test_dir();
        let scope = super::super::SessionScope::new(dir.as_str(), [allowed.path().to_path_buf()]);

        let args = json!({
            "path": "/etc/passwd",
            "content": "bad"
        });
        let result = tool().execute(&args, &scope).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
    }
}
