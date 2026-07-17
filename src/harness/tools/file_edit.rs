//! File edit tool: str_replace with line-number anchored errors and read-after-edit verification.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

pub struct FileEditTool;

impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "file_edit"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Edit a file by replacing an exact string (old_string) with a new string (new_string). The old_string must match exactly and uniquely. On failure, the error includes line numbers of near-matches so you can retry with a more specific match. After editing, the edited region is read back and included in the result so you can verify the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find in the file. Must match uniquely."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'path' argument"))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'old_string' argument"))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'new_string' argument"))?;

        let full_path = match super::resolve_workspace_path(path, cwd) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };
        let content = fs::read_to_string(&full_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", full_path.display()))?;

        // Count matches
        let match_count = content.matches(old_string).count();
        if match_count == 0 {
            // Find near-matches to help the model correct itself
            let near_matches = find_near_matches(&content, old_string);
            let mut error = format!("old_string not found in {path}.\n");
            if !near_matches.is_empty() {
                error.push_str("\nNear matches (similar text found at these locations):\n");
                for (line_num, snippet) in near_matches.iter().take(5) {
                    error.push_str(&format!("  line {line_num}: {snippet}\n"));
                }
            }
            error.push_str("\nRe-read the file to get the exact current content before retrying.");
            return Ok(error);
        }
        if match_count > 1 {
            let locations = find_all_match_locations(&content, old_string);
            let mut error = format!(
                "old_string matches {match_count} times in {path}. It must match uniquely.\n"
            );
            error.push_str("\nMatch locations:\n");
            for (line_num, snippet) in locations.iter().take(10) {
                error.push_str(&format!("  line {line_num}: {snippet}\n"));
            }
            error.push_str("\nInclude more surrounding context in old_string to make it unique.");
            return Ok(error);
        }

        // Apply the edit to a buffer
        let new_content = content.replacen(old_string, new_string, 1);

        // Write atomically
        fs::write(&full_path, &new_content)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", full_path.display()))?;

        // Read back the edited region for verification
        let verification = verify_edit(&new_content, new_string);

        Ok(format!(
            "Successfully edited {path}.\n\nEdited region (verified after write):\n{verification}"
        ))
    }
}

/// Find the line number and snippet of each occurrence of `needle`.
fn find_all_match_locations(content: &str, needle: &str) -> Vec<(usize, String)> {
    let mut results = Vec::new();
    let mut search_start = 0;
    while let Some(pos) = content[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let line_num = content[..abs_pos].lines().count() + 1;
        let line_start = content[..abs_pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = content[abs_pos..]
            .find('\n')
            .map(|i| abs_pos + i)
            .unwrap_or(content.len());
        let snippet = &content[line_start..line_end.min(line_start + 100)];
        results.push((line_num, snippet.to_string()));
        search_start = abs_pos + needle.len();
    }
    results
}

/// Find lines that share words with the needle (simple near-match heuristic).
fn find_near_matches(content: &str, needle: &str) -> Vec<(usize, String)> {
    let needle_words: Vec<&str> = needle.split_whitespace().filter(|w| w.len() > 3).collect();
    if needle_words.is_empty() {
        return Vec::new();
    }

    content
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let line_lower = line.to_lowercase();
            needle_words
                .iter()
                .filter(|w| line_lower.contains(*w))
                .count()
                >= (needle_words.len() / 2).max(1)
        })
        .map(|(i, line)| (i + 1, line.chars().take(100).collect()))
        .take(5)
        .collect()
}

/// Extract the edited region from the new content for verification.
fn verify_edit(new_content: &str, new_string: &str) -> String {
    // Find the new_string in the content and show surrounding context
    if let Some(pos) = new_content.find(new_string) {
        let line_num = new_content[..pos].lines().count() + 1;
        let context_before = new_content[..pos]
            .lines()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        let context_after_end = new_content[pos + new_string.len()..]
            .find('\n')
            .map(|i| pos + new_string.len() + i + 1)
            .unwrap_or(new_content.len());
        let context_after = new_content[pos + new_string.len()..context_after_end]
            .lines()
            .take(2)
            .collect::<Vec<_>>()
            .join("\n");

        let mut result = String::new();
        if !context_before.is_empty() {
            result.push_str(&context_before);
            result.push('\n');
        }
        for (i, line) in new_string.lines().enumerate() {
            result.push_str(&format!("{:>6}| {}", line_num + i, line));
            if i < new_string.lines().count() - 1 {
                result.push('\n');
            }
        }
        if !context_after.is_empty() {
            result.push('\n');
            result.push_str(&context_after);
        }
        result
    } else {
        // new_string not found (e.g. it was a deletion) — show what's around the edit point
        new_content.lines().take(10).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::*;
    use std::io::Write;

    fn make_test_file(dir: &test_util::TestDir, name: &str, content: &str) -> String {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "{content}").unwrap();
        name.to_string()
    }

    #[test]
    fn edits_file_successfully() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "hello world\nfoo bar\n");
        let tool = FileEditTool;
        let args = json!({
            "path": name,
            "old_string": "hello world",
            "new_string": "hello universe"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Successfully edited"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert!(content.contains("hello universe"));
        assert!(!content.contains("hello world"));
    }

    #[test]
    fn errors_on_no_match() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "hello world\n");
        let tool = FileEditTool;
        let args = json!({
            "path": name,
            "old_string": "nonexistent text",
            "new_string": "replacement"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("not found"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert!(content.contains("hello world"));
    }

    #[test]
    fn errors_on_multiple_matches() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "dup\ndup\ndup\n");
        let tool = FileEditTool;
        let args = json!({
            "path": name,
            "old_string": "dup",
            "new_string": "unique"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("matches 3 times"));
    }

    #[test]
    fn read_after_edit_verification_included() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "fn old_name() {}\n");
        let tool = FileEditTool;
        let args = json!({
            "path": name,
            "old_string": "fn old_name() {}",
            "new_string": "fn new_name() {\n    // renamed\n}"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("verified after write"));
        assert!(result.contains("new_name"));
    }

    #[test]
    fn rejects_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = FileEditTool;
        let args = json!({
            "path": "/etc/passwd",
            "old_string": "x",
            "new_string": "y"
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("absolute paths are not allowed"));
    }
}
