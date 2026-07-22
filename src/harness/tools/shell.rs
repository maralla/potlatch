//! Shell command execution tool with timeout and process group control.

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_OUTPUT: usize = 50_000;

pub struct ShellTool {
    timeout_secs: u64,
}

impl ShellTool {
    pub fn new() -> Self {
        Self {
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Run a shell command in the working directory (cwd). You are already in the working directory — no need to `cd` into it. Returns stdout, stderr, and exit code. Commands have a timeout (default 120s). Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `file_write` or `file_edit` instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default: 120."
                    }
                },
                "required": ["command"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'command' argument"))?;
        let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(self.timeout_secs);

        // Detect file-writing patterns and log a warning. We don't block the
        // command (some legitimate uses exist, e.g. `git commit` writes files),
        // but we surface it so the model gets feedback in the next turn's logs.
        if looks_like_file_write(command) {
            tracing::warn!(
                "harness: shell command appears to write files directly — use file_write/file_edit instead: {}",
                command.chars().take(200).collect::<String>()
            );
        }

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.args(["/C", command]);
            c
        } else {
            let mut c = Command::new("bash");
            c.args(["-c", command]);
            c
        };

        cmd.current_dir(cwd);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::null());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Ok(format!("Error: failed to spawn: {e}")),
        };

        let pid = child.id();
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);

        // Read stdout and stderr in separate threads to avoid deadlock
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();

        let stdout_handle = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });

        let stderr_handle = thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        // Poll for completion with timeout
        let timed_out = loop {
            match child.try_wait() {
                Ok(Some(_)) => break false,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        break true;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break false,
            }
        };

        if timed_out {
            #[cfg(unix)]
            {
                unsafe {
                    libc::killpg(pid as i32, libc::SIGKILL);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = child.kill();
            }
            let _ = child.wait();
        }

        let stdout_buf = stdout_handle.join().unwrap_or_default();
        let stderr_buf = stderr_handle.join().unwrap_or_default();
        let exit_status = child.wait().ok();

        let stdout_str = String::from_utf8_lossy(&stdout_buf);
        let stderr_str = String::from_utf8_lossy(&stderr_buf);
        let code = exit_status
            .and_then(|s| s.code())
            .unwrap_or(if timed_out { -1 } else { 1 });

        let mut result = String::new();
        if timed_out {
            result.push_str(&format!(
                "Command timed out after {timeout_secs}s and was killed.\n"
            ));
        }
        if !stdout_str.is_empty() {
            result.push_str("stdout:\n");
            result.push_str(&stdout_str);
        }
        if !stderr_str.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("stderr:\n");
            result.push_str(&stderr_str);
        }
        result.push_str(&format!("\nexit code: {code}"));

        if result.len() > MAX_OUTPUT {
            let truncated = truncate_at_char_boundary(&result, MAX_OUTPUT);
            return Ok(format!(
                "{truncated}\n\n[...output truncated, {} total chars...]",
                result.len()
            ));
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

/// Detect shell patterns that create or modify files directly, bypassing the
/// sandboxed `file_write`/`file_edit` tools. Returns true for patterns like
/// `cat > file`, `echo > file`, `sed -i`, `tee file`, `cp`, `mv` into the
/// workspace. Does NOT match `git`, `go build`, `mkdir`, or read-only commands.
fn looks_like_file_write(command: &str) -> bool {
    // Redirection to a file: `> file` or `>> file` (but not `2>` stderr-only)
    if command.contains(">>") || command.contains("> ") {
        // Exclude stderr redirection `2>` and process substitution
        if !command.contains("2>") || command.contains(">>") {
            return true;
        }
    }
    // Heredocs writing to files: `cat > file << 'EOF'`
    if command.contains("<<") && command.contains("cat") {
        return true;
    }
    // In-place file editing
    if command.contains("sed -i") || command.contains("sed --in-place") {
        return true;
    }
    // tee writing to files
    if command.contains("tee ") && !command.contains("tee /dev/null") {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_echo_command() {
        let tool = ShellTool::new();
        let args = json!({"command": "echo hello"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("hello"));
        assert!(result.contains("exit code: 0"));
    }

    #[test]
    fn captures_stderr() {
        let tool = ShellTool::new();
        let args = json!({"command": "echo err >&2"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("err"));
    }

    #[test]
    fn captures_nonzero_exit() {
        let tool = ShellTool::new();
        let args = json!({"command": "exit 42"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("exit code: 42"));
    }

    #[test]
    fn kills_on_timeout() {
        let tool = ShellTool::new();
        let args = json!({"command": "sleep 60", "timeout_secs": 2});
        let start = Instant::now();
        let result = tool.execute(&args, "/tmp").unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "should kill within ~2s, took {elapsed:?}"
        );
        assert!(result.contains("timed out"));
    }

    #[test]
    fn looks_like_file_write_detects_cat_redirect() {
        assert!(looks_like_file_write(
            "cat > inventory/excel/checksum.go << 'GOEOF'\npackage main\nGOEOF"
        ));
        assert!(looks_like_file_write("echo hello > file.txt"));
        assert!(looks_like_file_write("echo >> log.txt"));
    }

    #[test]
    fn looks_like_file_write_detects_sed_inplace() {
        assert!(looks_like_file_write("sed -i 's/old/new/g' file.go"));
        assert!(looks_like_file_write("sed --in-place 's/a/b/' file.go"));
    }

    #[test]
    fn looks_like_file_write_detects_tee() {
        assert!(looks_like_file_write("echo hello | tee file.txt"));
        assert!(!looks_like_file_write("echo hello | tee /dev/null"));
    }

    #[test]
    fn looks_like_file_write_ignores_readonly_commands() {
        assert!(!looks_like_file_write("go build ./..."));
        assert!(!looks_like_file_write("git status"));
        assert!(!looks_like_file_write("ls -la"));
        assert!(!looks_like_file_write("grep -rn 'pattern' ."));
        assert!(!looks_like_file_write("go test ./..."));
        assert!(!looks_like_file_write("echo hello"));
    }

    #[test]
    fn looks_like_file_write_ignores_stderr_redirect() {
        // `2>` is stderr redirect, not file creation
        assert!(!looks_like_file_write("go build ./... 2> /dev/null"));
    }
}
