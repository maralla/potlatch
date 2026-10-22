//! Read tool: read one or more files with line numbers.
//!
//! [`ReadTool`] takes a `files` array of `{path, start_line?, end_line?}`
//! objects. Reads run concurrently; per-file errors are reported inline and do
//! not block the other reads. Very large files are truncated with a marker.
//!
//! Reads are scoped to the session's file scope ([`SessionScope`]): the
//! working directory plus the roots the calling agent declared — the same
//! directory set the sandbox shares into the session's VM. Absolute paths
//! and `..` traversals that land outside that scope are rejected.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;
use crate::core::scope::SessionScope;

pub struct ReadTool;
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Read file contents with line numbers. Pass a 'files' array of {path, start_line?, end_line?} objects; reads run concurrently, and repeated entries for the same path read several ranges in one call. Reads up to ~250 lines come back complete; larger outputs are cut on a line boundary with the resume line marked — request the remainder in the same call instead of paging in small windows. Per-file errors are reported inline and do not block the other reads. Grep to locate code before reading.",
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
                                    "description": "Path to the file. Relative to the working directory, or an absolute path within the working directory or an agent-managed session directory named by the task."
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

    fn execute(&self, args: &Value, scope: &SessionScope) -> Result<String> {
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

        // Read concurrently. `thread::scope` borrows `scope` and the entries
        // by ref, and joins all spawned threads before returning.
        let results: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = entries
                .iter()
                .map(|e| {
                    s.spawn(move || read_single_file(&e.path, e.start_line, e.end_line, scope))
                })
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
    scope: &SessionScope,
) -> String {
    let full_path: PathBuf = match super::resolve_read_path(path, scope) {
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

    // Pack whole lines up to the output cap. At least one line always ships,
    // so a pathologically long line is still returned (the generic tool-result
    // truncator in the agent loop is the last-resort backstop for that case).
    const MAX_OUTPUT: usize = 16_000;
    let mut body = String::new();
    let mut shown_end = start;
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let piece = format!("{:>6}| {}\n", start + i, line);
        if !body.is_empty() && body.len() + piece.len() > MAX_OUTPUT {
            break;
        }
        body.push_str(&piece);
        shown_end = start + i;
    }

    let mut result = String::new();
    if start > 1 || shown_end < total_lines {
        result.push_str(&format!(
            "File: {} (lines {}-{} of {})\n\n",
            path, start, shown_end, total_lines
        ));
    } else {
        result.push_str(&format!("File: {} ({} lines)\n\n", path, total_lines));
    }
    result.push_str(&body);

    // The requested range did not fit: tell the model exactly where the shown
    // range ends and how to resume, so the next read starts at the right line
    // instead of re-guessing an overlapping window.
    if shown_end < end {
        result.push_str(&format!(
            "\n[...read truncated at line {shown_end} of {total_lines}: \
             re-read with start_line={} to continue...]\n",
            shown_end + 1
        ));
    }

    result
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
        assert!(result.contains("Error"));
    }

    #[test]
    fn reads_absolute_path_inside_a_declared_root() {
        let root = test_util::unique_test_dir();
        std::fs::write(root.path().join("session-notes.md"), "session content\n").unwrap();

        let cwd = test_util::unique_test_dir();
        let scope = super::super::SessionScope::new(cwd.as_str(), [root.path().to_path_buf()]);
        let abs = root.path().join("session-notes.md");
        let args = json!({"files": [{"path": abs.to_str().unwrap()}]});
        let result = ReadTool.execute(&args, &scope).unwrap();
        assert!(result.contains("session content"));
    }

    #[test]
    fn rejects_absolute_path_outside_the_scope() {
        let external = test_util::unique_test_dir();
        std::fs::write(external.path().join("external.txt"), "external content\n").unwrap();

        let cwd = test_util::unique_test_dir();
        let abs = external.path().join("external.txt");
        let args = json!({"files": [{"path": abs.to_str().unwrap()}]});
        let result = ReadTool.execute(&args, &cwd.scope()).unwrap();
        // A per-file scope rejection is reported inline like any other
        // per-file failure.
        assert!(result.contains("Error"));
        assert!(result.contains("outside"));
    }

    #[test]
    fn errors_on_missing_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = ReadTool;
        let args = json!({"files": [{"path": "/nonexistent/path/file.txt"}]});
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope()).unwrap();
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
        let result = tool.execute(&args, &dir.scope());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_missing_files_array() {
        let dir = test_util::unique_test_dir();
        let tool = ReadTool;
        let args = json!({});
        let result = tool.execute(&args, &dir.scope());
        assert!(result.is_err());
    }

    #[test]
    fn oversized_range_is_cut_on_a_line_boundary_with_a_resume_marker() {
        let dir = test_util::unique_test_dir();
        let p = dir.path().join("big.txt");
        let mut f = std::fs::File::create(&p).unwrap();
        // 600 lines x ~101 bytes > the 16k output cap.
        for i in 1..=600 {
            writeln!(f, "{:06}{}|", i, "x".repeat(93)).unwrap();
        }
        drop(f);

        let tool = ReadTool;
        let args = json!({"files": [{"path": "big.txt", "start_line": 1}]});
        let result = tool.execute(&args, &dir.scope()).unwrap();

        // Header reports the range actually shown, not the requested one.
        assert!(result.contains("lines 1-"));
        assert!(!result.contains("lines 1-600 of 600"));
        // The cut left a complete final line (every shown line is intact).
        let body_lines: Vec<&str> = result.lines().filter(|l| l.contains("xxxxx")).collect();
        assert!(!body_lines.is_empty());
        for l in &body_lines {
            assert!(l.trim_end().ends_with('|'), "partial line shown: {l:?}");
        }
        // The resume marker names the next line to read.
        let last_shown: usize = body_lines
            .last()
            .and_then(|l| l.trim().split('|').next())
            .and_then(|n| n.parse().ok())
            .expect("numbered line");
        assert!(result.contains(&format!("start_line={}", last_shown + 1)));
        // Whole output stays under the cap (header + marker included).
        assert!(result.len() < 17_000);
    }

    #[test]
    fn in_range_read_is_not_marked_truncated() {
        let dir = test_util::unique_test_dir();
        let p = dir.path().join("small.txt");
        std::fs::write(&p, "a\nb\nc\n").unwrap();

        let tool = ReadTool;
        let args = json!({"files": [{"path": "small.txt"}]});
        let result = tool.execute(&args, &dir.scope()).unwrap();
        assert!(!result.contains("truncated"));
        assert!(result.contains("3 lines"));
    }
}
