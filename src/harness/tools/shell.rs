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
            "description": "Run a shell command in the repo directory. Returns stdout, stderr, and exit code. Use for build, test, git, and any system command. Commands have a timeout (default 120s).",
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
}
