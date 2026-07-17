//! File write tool: create or overwrite files.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

pub struct FileWriteTool;

impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Create a new file or overwrite an existing file with the given content. Creates parent directories if needed. Use this for new files; prefer file_edit for modifying existing files.",
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

        let full_path = match super::resolve_workspace_path(path, cwd) {
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

    #[test]
    fn creates_new_file() {
        let dir = test_util::unique_test_dir();

        let tool = FileWriteTool;
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

        let tool = FileWriteTool;
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

        let tool = FileWriteTool;
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
        let tool = FileWriteTool;
        let args = json!({
            "path": "/tmp/evil.txt",
            "content": "bad"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("absolute paths are not allowed"));
    }
}
