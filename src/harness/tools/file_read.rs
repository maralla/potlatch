//! File read tool with line range support.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

pub struct FileReadTool;

impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Read the contents of a file. Supports optional line range (start_line and end_line, 1-indexed). Very large files are truncated with a marker. Always prefer grep to find relevant code before reading entire files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read. May be relative to the working directory, or an absolute path to a file explicitly referenced in the task context."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed). Optional."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-indexed, inclusive). Optional."
                    }
                },
                "required": ["path"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'path' argument"))?;

        let full_path = match super::resolve_read_path(path, cwd) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };
        let content = fs::read_to_string(&full_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", full_path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = args["start_line"].as_u64().unwrap_or(1).max(1) as usize;
        let end = args["end_line"]
            .as_u64()
            .map(|e| e as usize)
            .unwrap_or(total_lines)
            .min(total_lines);

        if start > total_lines {
            return Ok(format!(
                "File has {total_lines} lines. Requested start_line {start} is out of range."
            ));
        }

        let selected: Vec<&str> = lines[start - 1..end].to_vec();
        let mut result = String::new();

        if start > 1 || end < total_lines {
            result.push_str(&format!(
                "File: {} (lines {}-{} of {})\n\n",
                path, start, end, total_lines
            ));
        } else {
            result.push_str(&format!("File: {} ({} lines)\n\n", path, total_lines));
        }

        for (i, line) in selected.iter().enumerate() {
            result.push_str(&format!("{:>6}| {}\n", start + i, line));
        }

        const MAX_OUTPUT: usize = 50_000;
        if result.len() > MAX_OUTPUT {
            let truncated = truncate_at_char_boundary(&result, MAX_OUTPUT);
            result = format!(
                "{truncated}\n[...file truncated at {} chars, use start_line/end_line to read specific sections...]",
                MAX_OUTPUT
            );
        }

        Ok(result)
    }
}

fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_file_with_line_numbers() {
        let dir = test_util::unique_test_dir();
        let path = dir.path().join("test.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        writeln!(f, "line three").unwrap();

        let tool = FileReadTool;
        let args = json!({"path": "test.txt"});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
        assert!(result.contains("line three"));
        assert!(result.contains("3 lines"));
    }

    #[test]
    fn reads_line_range() {
        let dir = test_util::unique_test_dir();
        let path = dir.path().join("range.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = FileReadTool;
        let args = json!({"path": "range.txt", "start_line": 3, "end_line": 5});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("line 3"));
        assert!(result.contains("line 4"));
        assert!(result.contains("line 5"));
        assert!(!result.contains("line 1"));
        assert!(!result.contains("line 10"));
        assert!(result.contains("lines 3-5 of 10"));
    }

    #[test]
    fn errors_on_missing_file() {
        let dir = test_util::unique_test_dir();
        let tool = FileReadTool;
        let args = json!({"path": "nonexistent.txt"});
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
    }

    #[test]
    fn reads_absolute_path_outside_workspace() {
        // An absolute path to a file outside the cwd must be readable when it is
        // explicitly provided (e.g. referenced in the task context).
        let external = test_util::unique_test_dir();
        std::fs::write(external.path().join("external.txt"), "external content\n").unwrap();

        let cwd = test_util::unique_test_dir();
        let abs = external.path().join("external.txt");
        let args = json!({"path": abs.to_str().unwrap()});
        let result = FileReadTool.execute(&args, cwd.as_str()).unwrap();
        assert!(result.contains("external content"));
    }

    #[test]
    fn errors_on_missing_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = FileReadTool;
        let args = json!({"path": "/nonexistent/path/file.txt"});
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
    }
}
