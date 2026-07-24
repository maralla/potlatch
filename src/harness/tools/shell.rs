//! Shell command execution tool with timeout and process group control.

use std::collections::HashMap;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_OUTPUT: usize = 50_000;

/// A running background shell job. The child process is kept alive across
/// tool calls; stdout/stderr are drained into shared buffers by reader threads
/// so the pipe buffer never fills and blocks the process.
struct Job {
    child: Child,
    pid: u32,
    command: String,
    started_at: Instant,
    stdout_buf: Arc<Mutex<Vec<u8>>>,
    stderr_buf: Arc<Mutex<Vec<u8>>>,
    /// Set once the process exits (observed via `try_wait`).
    exit_code: Option<i32>,
    /// Set when the job was killed via `kill`/`kill_all` rather than exiting.
    killed: bool,
}

/// Shared table of background jobs, keyed by job id. Held as `Arc<JobTable>`
/// by both `ShellTool` (for spawn/poll/kill) and `Session` (for cleanup on
/// close). All methods take `&self` and lock internally, so the `Tool::execute`
/// `&self` borrow is sufficient.
pub struct JobTable {
    jobs: Mutex<HashMap<String, Job>>,
}

impl JobTable {
    pub fn new() -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
        }
    }

    /// Spawn a command in the background. Returns the assigned job id.
    pub fn spawn(&self, command: &str, cwd: &str) -> Result<String> {
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

        let mut child = cmd.spawn()?;
        let pid = child.id();

        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let stdout_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

        let stdout_buf_clone = Arc::clone(&stdout_buf);
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            *stdout_buf_clone.lock().unwrap() = buf;
        });
        let stderr_buf_clone = Arc::clone(&stderr_buf);
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            *stderr_buf_clone.lock().unwrap() = buf;
        });

        let id = uuid::Uuid::new_v4().to_string();
        let job = Job {
            child,
            pid,
            command: command.to_string(),
            started_at: Instant::now(),
            stdout_buf,
            stderr_buf,
            exit_code: None,
            killed: false,
        };
        self.jobs.lock().unwrap().insert(id.clone(), job);
        Ok(id)
    }

    /// Poll a background job: update its exit status and return accumulated
    /// stdout/stderr. The buffers grow until the process exits; we return the
    /// full contents each poll (truncated to `MAX_OUTPUT`).
    pub fn poll(&self, job_id: &str) -> Result<String> {
        let mut jobs = self.jobs.lock().unwrap();
        let job = jobs
            .get_mut(job_id)
            .ok_or_else(|| anyhow::anyhow!("unknown job id: {job_id}"))?;

        // If we haven't observed exit yet, try to reap without blocking.
        if job.exit_code.is_none()
            && let Ok(Some(status)) = job.child.try_wait()
        {
            job.exit_code = status.code();
        }

        let running = job.exit_code.is_none() && !job.killed;
        let status = if job.killed {
            "killed"
        } else if running {
            "running"
        } else {
            "exited"
        };

        let stdout_str = {
            let buf = job.stdout_buf.lock().unwrap();
            String::from_utf8_lossy(&buf).into_owned()
        };
        let stderr_str = {
            let buf = job.stderr_buf.lock().unwrap();
            String::from_utf8_lossy(&buf).into_owned()
        };
        let elapsed = job.started_at.elapsed();
        let exit_line = match job.exit_code {
            Some(code) => format!("exit code: {code}"),
            None if job.killed => "exit code: -1 (killed)".to_string(),
            None => "exit code: (still running)".to_string(),
        };

        let mut result = format!(
            "job: {job_id}\ncommand: {}\nstatus: {status}\nelapsed: {:.1}s\n{exit_line}",
            job.command,
            elapsed.as_secs_f64()
        );
        if !stdout_str.is_empty() {
            result.push_str("\nstdout:\n");
            result.push_str(&stdout_str);
        }
        if !stderr_str.is_empty() {
            result.push_str("\nstderr:\n");
            result.push_str(&stderr_str);
        }

        if result.len() > MAX_OUTPUT {
            let truncated = truncate_at_char_boundary(&result, MAX_OUTPUT);
            result = format!(
                "{truncated}\n\n[...output truncated, {} total chars...]",
                result.len()
            );
        }
        Ok(result)
    }

    /// Kill a background job. Sends SIGKILL to the process group (Unix) or
    /// `child.kill()` (Windows), marks it killed, and reaps the child.
    pub fn kill(&self, job_id: &str) -> Result<String> {
        let mut jobs = self.jobs.lock().unwrap();
        let job = jobs
            .get_mut(job_id)
            .ok_or_else(|| anyhow::anyhow!("unknown job id: {job_id}"))?;

        if job.exit_code.is_some() {
            // Already exited — report final status without killing.
            return self.poll(job_id);
        }

        #[cfg(unix)]
        {
            unsafe {
                libc::killpg(job.pid as i32, libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = job.child.kill();
        }
        job.killed = true;
        if let Ok(Some(status)) = job.child.try_wait() {
            job.exit_code = status.code();
        }
        drop(jobs);
        self.poll(job_id)
    }

    /// Kill all still-running jobs. Called on session close to avoid leaking
    /// background processes.
    pub fn kill_all(&self) {
        let mut jobs = self.jobs.lock().unwrap();
        for job in jobs.values_mut() {
            if job.exit_code.is_some() {
                continue;
            }
            #[cfg(unix)]
            {
                unsafe {
                    libc::killpg(job.pid as i32, libc::SIGKILL);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = job.child.kill();
            }
            job.killed = true;
            if let Ok(Some(status)) = job.child.try_wait() {
                job.exit_code = status.code();
            }
        }
    }
}

impl Default for JobTable {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ShellTool {
    timeout_secs: u64,
    jobs: Arc<JobTable>,
}

impl ShellTool {
    pub fn new() -> Self {
        Self {
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            jobs: Arc::new(JobTable::new()),
        }
    }

    /// Construct with an externally-owned job table, so background jobs are
    /// shared across tool instances (e.g. across multiple prompts in a
    /// session).
    pub fn with_job_table(jobs: Arc<JobTable>) -> Self {
        Self {
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            jobs,
        }
    }
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Run a shell command in the working directory (cwd). You are already in the working directory — no need to `cd` into it. Returns stdout, stderr, and exit code. Commands have a timeout (default 120s). Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `file_write` or `file_edit` instead. To run a long-running command in the background, set `background: true`; you get a job id back and can poll its output later with `job_id`, or terminate it with `job_id` + `kill: true`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Required unless polling or killing a background job (`job_id`)."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds for foreground commands. Default: 120. Ignored for background jobs (they run until exit or kill)."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "If true, spawn the command in the background and return a job id immediately instead of waiting. Poll with `job_id`."
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Poll or kill a background job. Returns accumulated stdout/stderr and current status. Ignored unless polling or killing."
                    },
                    "kill": {
                        "type": "boolean",
                        "description": "When true with `job_id`, terminate the background job."
                    }
                },
                "required": []
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let job_id = args["job_id"].as_str();
        let kill = args["kill"].as_bool().unwrap_or(false);

        // Polling or killing a background job takes precedence — `command`
        // is ignored in these modes.
        if let Some(id) = job_id {
            if kill {
                return self.jobs.kill(id);
            }
            return self.jobs.poll(id);
        }

        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'command' argument"))?;
        let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(self.timeout_secs);
        let background = args["background"].as_bool().unwrap_or(false);

        // Detect wrong-directory `cd` prefixes and warn. We don't strip or
        // modify the command — if the model cd's to the wrong path, the
        // command fails naturally, which is clearer feedback than silently
        // fixing it. The warning is prepended to the tool result so the model
        // sees it.
        let cd_warning = detect_wrong_cd(command, cwd);

        // Detect file-writing patterns and log a warning. We don't block the
        // command (some legitimate uses exist, e.g. `git commit` writes files),
        // but we surface it so the model gets feedback in the next turn's logs.
        if looks_like_file_write(command) {
            tracing::warn!(
                "harness: shell command appears to write files directly — use file_write/file_edit instead: {}",
                command.chars().take(200).collect::<String>()
            );
        }

        if background {
            let id = self.jobs.spawn(command, cwd)?;
            let mut result = format!("Background job started: {id}\ncommand: {command}");
            if let Some(ref warning) = cd_warning {
                result.push_str("\n\n");
                result.push_str(warning);
            }
            return Ok(result);
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
        if let Some(ref warning) = cd_warning {
            result.push_str(warning);
            result.push_str("\n\n");
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

/// Detect a leading `cd <path> &&` (or `cd <path>;`) prefix that targets a
/// directory other than the working directory. The shell tool already runs in
/// the working directory, so such a `cd` sends the command to the wrong place.
///
/// We do NOT strip or modify the command — if the model cd's to the wrong
/// path, the command fails naturally, which is clearer feedback than silently
/// fixing it. Instead we return a warning string that is prepended to the tool
/// result so the model sees it. Returns `None` when there is no `cd` prefix,
/// the `cd` targets the working directory, or the `cd` is a standalone
/// command with no separator.
fn detect_wrong_cd(command: &str, cwd: &str) -> Option<String> {
    let trimmed = command.trim_start();

    // Match `cd <path> && <rest>` or `cd <path>;<rest>`.
    let after_cd = trimmed.strip_prefix("cd ")?;

    // Find the separator: `&&` or `;`
    let sep_pos = after_cd.find("&&").or_else(|| after_cd.find(';'))?;
    let rest = after_cd[sep_pos..]
        .trim_start_matches("&&")
        .trim_start_matches(';')
        .trim_start();

    if rest.is_empty() {
        return None;
    }

    let cd_path = after_cd[..sep_pos]
        .trim()
        .trim_matches('\'')
        .trim_matches('"');

    // Canonicalize both paths to compare. If we can't canonicalize (path
    // doesn't exist), treat the cd as wrong.
    let cwd_canonical = std::path::Path::new(cwd)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| cwd.to_string());

    let cd_canonical = std::path::Path::new(cd_path)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| cd_path.to_string());

    if cd_canonical == cwd_canonical {
        // Redundant cd to the same directory — harmless, no warning.
        None
    } else {
        // Wrong directory — warn. The warning is visible in the tool result
        // so the model learns to stop guessing paths. The command runs as
        // given so the failure (if any) is honest feedback.
        let warning = format!(
            "WARNING: the command begins with `cd {cd_path}`, which is not the working directory. \
             Commands already run in the working directory. Do not use `cd` with absolute paths; \
             use relative paths instead. The command ran as given."
        );
        tracing::warn!("harness: wrong-directory cd '{cd_path}' (cwd is '{cwd_canonical}')");
        Some(warning)
    }
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

    #[test]
    fn detect_wrong_cd_silent_for_same_directory() {
        assert!(
            detect_wrong_cd("cd /tmp && echo hi", "/tmp").is_none(),
            "no warning for redundant cd to cwd"
        );
    }

    #[test]
    fn detect_wrong_cd_warns_for_wrong_directory() {
        let warning = detect_wrong_cd("cd /home/user/wrong-project && go test ./...", "/tmp")
            .expect("should warn for wrong-directory cd");
        assert!(warning.contains("not the working directory"));
        assert!(!warning.contains("stripped"));
    }

    #[test]
    fn detect_wrong_cd_handles_semicolon_separator() {
        assert!(detect_wrong_cd("cd /tmp; echo hi", "/tmp").is_none());
    }

    #[test]
    fn detect_wrong_cd_ignores_commands_without_cd() {
        assert!(detect_wrong_cd("go test ./...", "/tmp").is_none());
    }

    #[test]
    fn detect_wrong_cd_ignores_cd_without_separator() {
        // `cd /tmp` alone (no && or ;) is a valid standalone command — don't warn.
        assert!(detect_wrong_cd("cd /tmp", "/tmp").is_none());
    }

    #[test]
    fn detect_wrong_cd_handles_quoted_path() {
        assert!(detect_wrong_cd("cd '/tmp' && echo hi", "/tmp").is_none());
    }

    #[test]
    fn detect_wrong_cd_treats_trailing_slash_as_same() {
        // /tmp and /tmp/ are the same directory.
        assert!(detect_wrong_cd("cd /tmp/ && echo hi", "/tmp").is_none());
    }

    #[test]
    fn spawn_background_returns_job_id() {
        let tool = ShellTool::new();
        let args = json!({"command": "echo hi", "background": true});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.starts_with("Background job started:"));
        // The job id follows the label.
        let id = result
            .strip_prefix("Background job started: ")
            .unwrap()
            .lines()
            .next()
            .unwrap();
        assert!(!id.is_empty());

        // The job should exit quickly; poll until we see the output.
        let poll_args = json!({"job_id": id});
        let mut found = false;
        for _ in 0..20 {
            let poll = tool.execute(&poll_args, "/tmp").unwrap();
            if poll.contains("hi") && poll.contains("exited") {
                found = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(found, "expected to poll output 'hi' and exited status");
    }

    #[test]
    fn poll_running_job_returns_running_status() {
        let tool = ShellTool::new();
        let args = json!({"command": "sleep 2", "background": true});
        let result = tool.execute(&args, "/tmp").unwrap();
        let id = result
            .strip_prefix("Background job started: ")
            .unwrap()
            .lines()
            .next()
            .unwrap();

        let poll_args = json!({"job_id": id});
        let poll = tool.execute(&poll_args, "/tmp").unwrap();
        assert!(poll.contains("status: running"));
        assert!(poll.contains("sleep 2"));

        // Clean up.
        let kill_args = json!({"job_id": id, "kill": true});
        tool.execute(&kill_args, "/tmp").unwrap();
    }

    #[test]
    fn kill_terminates_background_job() {
        let tool = ShellTool::new();
        let args = json!({"command": "sleep 60", "background": true});
        let result = tool.execute(&args, "/tmp").unwrap();
        let id = result
            .strip_prefix("Background job started: ")
            .unwrap()
            .lines()
            .next()
            .unwrap();

        let kill_args = json!({"job_id": id, "kill": true});
        let kill_result = tool.execute(&kill_args, "/tmp").unwrap();
        assert!(
            kill_result.contains("status: killed"),
            "expected killed status, got: {kill_result}"
        );

        // Polling again should still report killed.
        let poll_args = json!({"job_id": id});
        let poll = tool.execute(&poll_args, "/tmp").unwrap();
        assert!(poll.contains("status: killed"));
    }

    #[test]
    fn kill_all_kills_all_running_jobs() {
        let jobs = Arc::new(JobTable::new());
        let tool = ShellTool::with_job_table(Arc::clone(&jobs));

        let a = jobs.spawn("sleep 60", "/tmp").unwrap();
        let b = jobs.spawn("sleep 60", "/tmp").unwrap();

        jobs.kill_all();

        let poll_a = tool.execute(&json!({"job_id": a}), "/tmp").unwrap();
        let poll_b = tool.execute(&json!({"job_id": b}), "/tmp").unwrap();
        assert!(
            poll_a.contains("status: killed"),
            "job a not killed: {poll_a}"
        );
        assert!(
            poll_b.contains("status: killed"),
            "job b not killed: {poll_b}"
        );
    }

    #[test]
    fn poll_unknown_job_id_returns_error() {
        let tool = ShellTool::new();
        let args = json!({"job_id": "nonexistent-id"});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown job id"));
    }

    #[test]
    fn foreground_execution_unchanged() {
        // `background` defaults to false — the command runs synchronously and
        // returns the full output, not a job id.
        let tool = ShellTool::new();
        let args = json!({"command": "echo foreground"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("foreground"));
        assert!(result.contains("exit code: 0"));
        assert!(!result.contains("Background job started"));
    }
}
