//! Read tool: read one or more files with line numbers.
//!
//! [`ReadTool`] takes a `files` array of `{path, start_line?, end_line?}`
//! objects. Reads run concurrently; per-file errors are reported inline and do
//! not block the other reads. Very large files are truncated with a marker.
//!
//! Reads are not sandboxed: absolute paths and paths outside the working
//! directory are allowed when a file is explicitly referenced in the task
//! context.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

pub struct ReadTool;

impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Read file contents with line numbers. Pass a 'files' array of {path, start_line?, end_line?} objects; reads run concurrently. Per-file errors are reported inline and do not block the other reads. Very large files are truncated with a marker. Always prefer grep to find relevant code before reading entire files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "description": "List of files to read.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the file. May be relative to the working directory, or an absolute path to a file explicitly referenced in the task context."
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
                        },
                        "minItems": 1
                    }
                },
                "required": ["files"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let files = args["files"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing or invalid 'files' array argument"))?;

        if files.is_empty() {
            anyhow::bail!("'files' array must contain at least one entry");
        }

        // Resolve and validate each entry up front (cheap, no file I/O), so the
        // concurrent phase just needs the parsed args.
        let entries: Vec<FileEntry> = files
            .iter()
            .map(|entry| {
                let path = entry["path"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("each file entry requires a 'path' string"))?;
                let start_line = entry["start_line"].as_u64();
                let end_line = entry["end_line"].as_u64();
                Ok(FileEntry {
                    path: path.to_string(),
                    start_line,
                    end_line,
                })
            })
            .collect::<Result<_>>()?;

        // Read concurrently. `thread::scope` borrows `cwd` and the entries by ref,
        // and joins all spawned threads before returning.
        let results: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = entries
                .iter()
                .map(|e| s.spawn(move || read_single_file(&e.path, e.start_line, e.end_line, cwd)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| "Error: reader thread panicked".to_string())
                })
                .collect()
        });

        let mut output = format!("Read {} file(s):\n\n", results.len());
        for (i, r) in results.iter().enumerate() {
            if i > 0 {
                output.push_str("\n---\n\n");
            }
            output.push_str(r);
            output.push('\n');
        }
        Ok(output)
    }
}

struct FileEntry {
    path: String,
    start_line: Option<u64>,
    end_line: Option<u64>,
}

/// Read a single file and format it with line numbers. Returns a user-facing
/// string; errors are formatted as `Error: ...` strings (not `Err`) so a batch
/// read can report per-file failures inline without aborting the whole call.
fn read_single_file(
    path: &str,
    start_line: Option<u64>,
    end_line: Option<u64>,
    cwd: &str,
) -> String {
    let full_path: PathBuf = match super::resolve_read_path(path, cwd) {
        Ok(p) => p,
        Err(msg) => return format!("Error: {msg}"),
    };
    let content = match fs::read_to_string(&full_path) {
        Ok(c) => c,
        Err(e) => return format!("Error: failed to read {}: {e}", full_path.display()),
    };

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let start = start_line.unwrap_or(1).max(1) as usize;
    let end = end_line
        .map(|e| e as usize)
        .unwrap_or(total_lines)
        .min(total_lines);

    if start > total_lines {
        return format!(
            "File: {path}\nFile has {total_lines} lines. Requested start_line {start} is out of range."
        );
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

    result
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

        let tool = ReadTool;
        let args = json!({"files": [{"path": "test.txt"}]});
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

        let tool = ReadTool;
        let args = json!({
            "files": [{"path": "range.txt", "start_line": 3, "end_line": 5}]
        });
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
        let tool = ReadTool;
        let args = json!({"files": [{"path": "nonexistent.txt"}]});
        // The tool itself returns Ok with an inline "Error:" — a missing file
        // is a per-file failure, not a whole-call failure.
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Error"));
    }

    #[test]
    fn reads_absolute_path_outside_workspace() {
        let external = test_util::unique_test_dir();
        std::fs::write(external.path().join("external.txt"), "external content\n").unwrap();

        let cwd = test_util::unique_test_dir();
        let abs = external.path().join("external.txt");
        let args = json!({"files": [{"path": abs.to_str().unwrap()}]});
        let result = ReadTool.execute(&args, cwd.as_str()).unwrap();
        assert!(result.contains("external content"));
    }

    #[test]
    fn errors_on_missing_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = ReadTool;
        let args = json!({"files": [{"path": "/nonexistent/path/file.txt"}]});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Error"));
    }

    #[test]
    fn reads_multiple_files_concurrently() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("a.txt"), "alpha\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta\nbeta2\n").unwrap();
        std::fs::write(dir.path().join("c.rs"), "fn main() {}\n").unwrap();

        let tool = ReadTool;
        let args = json!({
            "files": [
                {"path": "a.txt"},
                {"path": "b.txt"},
                {"path": "c.rs"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Read 3 file(s)"));
        assert!(result.contains("alpha"));
        assert!(result.contains("beta"));
        assert!(result.contains("beta2"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn reports_per_file_errors_without_aborting() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("ok.txt"), "good\n").unwrap();

        let tool = ReadTool;
        let args = json!({
            "files": [
                {"path": "ok.txt"},
                {"path": "missing.txt"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        // The existing file is still read despite the missing one.
        assert!(result.contains("good"));
        assert!(result.contains("Error"));
    }

    #[test]
    fn supports_per_file_line_ranges() {
        let dir = test_util::unique_test_dir();
        let p = dir.path().join("nums.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadTool;
        let args = json!({
            "files": [
                {"path": "nums.txt", "start_line": 1, "end_line": 2},
                {"path": "nums.txt", "start_line": 9, "end_line": 10}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("lines 1-2 of 10"));
        assert!(result.contains("lines 9-10 of 10"));
        assert!(result.contains("line 1"));
        assert!(result.contains("line 10"));
        assert!(!result.contains("line 5"));
    }

    #[test]
    fn rejects_empty_files_array() {
        let dir = test_util::unique_test_dir();
        let tool = ReadTool;
        let args = json!({"files": []});
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_missing_files_array() {
        let dir = test_util::unique_test_dir();
        let tool = ReadTool;
        let args = json!({});
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
    }
}
