//! Write tool: create or overwrite files.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

pub struct WriteTool {
    roots: super::WriteRoots,
}

impl WriteTool {
    pub fn new(roots: super::WriteRoots) -> Self {
        Self { roots }
    }
}

impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Create a new file or overwrite an existing file with the given content. Creates parent directories if needed. Use this for new files; prefer edit for modifying existing files. By default paths must be relative to the working directory; set outside_cwd: true to write to an absolute path outside the workspace (only for agent-managed scratch files explicitly permitted by the task instructions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file"
                    },
                    "outside_cwd": {
                        "type": "boolean",
                        "description": "When true, allow absolute paths outside the working directory. Default false. Only set this when the task instructions explicitly direct you to write to a specific absolute path (e.g. an agent session directory).",
                        "default": false
                    }
                },
                "required": ["path", "content"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'path' argument"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'content' argument"))?;
        let outside_cwd = args["outside_cwd"].as_bool().unwrap_or(false);

        let full_path = match super::resolve_write_path(path, cwd, outside_cwd, &self.roots) {
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

    fn unrestricted_tool() -> WriteTool {
        WriteTool::new(super::super::WriteRoots::default())
    }

    #[test]
    fn creates_new_file() {
        let dir = test_util::unique_test_dir();

        let tool = unrestricted_tool();
        let args = json!({
            "path": "new.txt",
            "content": "line one\nline two\n"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"));
        assert!(result.contains("2 lines"));

        let content = std::fs::read_to_string(dir.path().join("new.txt")).unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("over.txt"), "old content").unwrap();

        let tool = unrestricted_tool();
        let args = json!({
            "path": "over.txt",
            "content": "new content"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Overwrote"));

        let content = std::fs::read_to_string(dir.path().join("over.txt")).unwrap();
        assert_eq!(content, "new content");
    }

    #[test]
    fn creates_parent_directories() {
        let dir = test_util::unique_test_dir();

        let tool = unrestricted_tool();
        let args = json!({
            "path": "nested/deep/file.txt",
            "content": "nested"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"));

        assert!(dir.path().join("nested/deep/file.txt").exists());
    }

    #[test]
    fn rejects_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = unrestricted_tool();
        let args = json!({
            "path": "/tmp/evil.txt",
            "content": "bad"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("absolute paths are not allowed"));
    }

    #[test]
    fn rejects_parent_traversal() {
        let dir = test_util::unique_test_dir();
        let tool = unrestricted_tool();
        let args = json!({
            "path": "../escape.txt",
            "content": "bad"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("escapes the workspace"));
    }

    #[test]
    fn outside_cwd_allows_absolute_path() {
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("outside.txt");

        let tool = unrestricted_tool();
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "from outside",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"), "{result}");

        let content = std::fs::read_to_string(&target).unwrap();
        assert_eq!(content, "from outside");
    }

    #[test]
    fn outside_cwd_creates_parent_dirs() {
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("nested/deep/script.py");

        let tool = unrestricted_tool();
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "print('hi')",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn outside_cwd_overwrites_existing_file() {
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("existing.txt");
        std::fs::write(&target, "old").unwrap();

        let tool = unrestricted_tool();
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "new",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Overwrote"), "{result}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn outside_cwd_defaults_to_false_when_absent() {
        let dir = test_util::unique_test_dir();
        let tool = unrestricted_tool();
        // No outside_cwd field — must still reject absolute paths.
        let args = json!({
            "path": "/tmp/evil.txt",
            "content": "bad"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("absolute paths are not allowed"));
    }

    fn rooted_tool(roots: Vec<std::path::PathBuf>) -> WriteTool {
        WriteTool::new(super::super::WriteRoots::from_paths(roots))
    }

    #[test]
    fn outside_cwd_write_inside_a_configured_root_is_allowed() {
        let dir = test_util::unique_test_dir();
        let outside = test_util::unique_test_dir();
        let target = outside.path().join("knowledge/notes.md");

        let tool = rooted_tool(vec![outside.path().to_path_buf()]);
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "notes",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn outside_cwd_write_inside_the_cwd_is_allowed_with_roots() {
        let dir = test_util::unique_test_dir();
        let other = test_util::unique_test_dir();
        let target = dir.path().join("in-workspace.txt");

        // Roots are configured, but the write targets the cwd itself.
        let tool = rooted_tool(vec![other.path().to_path_buf()]);
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "ws",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Created"), "{result}");
        assert!(target.exists());
    }

    #[test]
    fn outside_cwd_write_outside_every_root_is_rejected() {
        let dir = test_util::unique_test_dir();
        let allowed = test_util::unique_test_dir();
        let forbidden = test_util::unique_test_dir();
        let target = forbidden.path().join("evil.txt");

        let tool = rooted_tool(vec![allowed.path().to_path_buf()]);
        let args = json!({
            "path": target.to_string_lossy(),
            "content": "bad",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
        assert!(!target.exists());
    }

    #[test]
    fn outside_cwd_write_to_an_arbitrary_system_path_is_rejected_with_roots() {
        let dir = test_util::unique_test_dir();
        let allowed = test_util::unique_test_dir();

        let tool = rooted_tool(vec![allowed.path().to_path_buf()]);
        let args = json!({
            "path": "/etc/passwd",
            "content": "bad",
            "outside_cwd": true
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(
            result.contains("outside the session's writable directories"),
            "{result}"
        );
    }
}
