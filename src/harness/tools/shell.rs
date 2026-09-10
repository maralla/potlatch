//! Shell command execution tool with timeout and process group control.

use std::collections::HashMap;
use std::io::Read;
#[cfg(unix)]
use std::io::{Error, ErrorKind};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_OUTPUT: usize = 50_000;
/// How long the foreground path waits for its output readers to see EOF after
/// the child is gone. Normally EOF arrives the instant the child (and its
/// process group) dies; the wait only binds when a grandchild outlived the
/// child while holding the output pipes (a daemon, a tunnel), in which case
/// collection is abandoned rather than blocking the agent loop forever.
const ORPHAN_PIPE_GRACE: Duration = Duration::from_secs(5);
/// Read chunk size for the capped background-job drain: large enough that a
/// single `read` call moves a meaningful amount of pipe data, small enough
/// that the cap is enforced without over-reading past it by much.
const DRAIN_CHUNK: usize = 8_192;
/// Longest a drain reader ever blocks without re-checking its `stop` flag.
/// The reader polls the stream with this timeout instead of blocking in
/// `read` until EOF, so setting `stop` always ends the thread within one
/// slice — abandoning a reader could not do that: a thread blocked inside
/// `read` never re-checks the flag and stays parked (holding its stack, its
/// pipe fds, and the output buffers) until the pipe's write end closes,
/// which for a setsid'd daemon or tunnel may be never.
const DRAIN_POLL_SLICE_MS: i32 = 100;
/// Finished background jobs retained per session table, under two caps:
/// entry count and total retained output bytes. Eviction is LRU — the least
/// recently spawned or polled finished job goes first — and only ever touches
/// finished jobs; running jobs are never dropped. The table lives from
/// session/new to session/close, so the caps are what keep a long session
/// that spawns many jobs bounded (each entry holds its output buffers and
/// command text).
const MAX_RETAINED_FINISHED_JOBS: usize = 64;
/// Total retained output bytes across finished jobs. A single job can hold
/// up to `2 * MAX_OUTPUT` (both streams), so this bounds the table's buffers
/// independently of how the model splits its work across jobs.
const MAX_RETAINED_FINISHED_BYTES: usize = 2 * 1024 * 1024;

/// Whether both reader handles finished within [`ORPHAN_PIPE_GRACE`]. After a
/// successful kill the child's own pipe ends are closed, so EOF normally
/// arrives immediately; the wait only elapses when some other process still
/// holds the pipes (a grandchild the process-group kill could not reach —
/// e.g. a daemon that re-parented to init, or an ssh tunnel that setsid'd
/// away), in which case the caller must abandon the readers instead of
/// joining them or the agent loop hangs on the tool call forever.
fn wait_for_pipes_or_orphan(
    stdout_handle: &thread::JoinHandle<()>,
    stderr_handle: &thread::JoinHandle<()>,
) -> bool {
    let deadline = Instant::now() + ORPHAN_PIPE_GRACE;
    while Instant::now() < deadline {
        if stdout_handle.is_finished() && stderr_handle.is_finished() {
            return true;
        }
        thread::sleep(Duration::from_millis(20));
    }
    stdout_handle.is_finished() && stderr_handle.is_finished()
}

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
    /// Set when the job was killed via `kill` rather than exiting.
    killed: bool,
    /// Last spawn/poll touch — the LRU clock for eviction of finished jobs.
    /// A job the model re-polls stays evictable-last even if it is old.
    last_used: Instant,
}

/// Read `stream` to EOF into `buf`, storing at most `MAX_OUTPUT` bytes (the
/// head) and discarding the rest. The stream is still drained to EOF — a
/// stopped reader would block the child once the pipe filled — but nothing
/// past the cap is ever stored, so a background job's buffer is bounded by
/// `MAX_OUTPUT` per stream for its entire lifetime, polled or not.
/// Runs on a reader thread; publishes under the buffer's mutex per chunk so
/// a concurrent `poll` sees progress without waiting for EOF.
/// Returns when EOF is seen or `stop` is set; on stop, whatever has arrived
/// so far stays in `buf`.
///
/// Waits in [`DRAIN_POLL_SLICE_MS`] poll slices rather than blocking in
/// `read` so `stop` always ends the thread promptly, even when a descendant
/// of the command holds the pipe open and EOF never comes.
#[cfg(unix)]
fn drain_capped(stream: &mut (impl Read + AsRawFd), buf: &Arc<Mutex<Vec<u8>>>, stop: &AtomicBool) {
    let fd = stream.as_raw_fd();
    let mut chunk = [0u8; DRAIN_CHUNK];
    loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        let mut fds = [libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        }];
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), 1, DRAIN_POLL_SLICE_MS) };
        if ready < 0 {
            let err = Error::last_os_error();
            if err.kind() == ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        if ready == 0 {
            // Slice elapsed with no data — loop back to the stop check.
            continue;
        }
        if fds[0].revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0
            && fds[0].revents & libc::POLLIN == 0
        {
            return;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let mut buf = buf.lock().unwrap();
                let room = MAX_OUTPUT.saturating_sub(buf.len());
                buf.extend_from_slice(&chunk[..n.min(room)]);
            }
        }
    }
}

/// Non-unix fallback: identical buffer semantics, but a reader blocked in
/// `read` can only observe `stop` between chunks, so callers on this platform
/// still abandon (and leak) the thread when no EOF arrives.
#[cfg(not(unix))]
fn drain_capped(stream: &mut impl Read, buf: &Arc<Mutex<Vec<u8>>>, stop: &AtomicBool) {
    let mut chunk = [0u8; DRAIN_CHUNK];
    loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let mut buf = buf.lock().unwrap();
                let room = MAX_OUTPUT.saturating_sub(buf.len());
                buf.extend_from_slice(&chunk[..n.min(room)]);
            }
        }
    }
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

        // Drain each stream into its shared buffer, capped at MAX_OUTPUT
        // bytes (head kept). The cap is on what the buffer STORES, not what
        // `poll` returns — an uncapped `read_to_end` on a chatty background
        // job (a full test suite, a watch loop) grew without limit for the
        // job's lifetime, which for a never-polled or never-killed job is
        // the harness process's lifetime; observed as one process ballooning
        // to ~93 GB of anonymous RSS until the OOM killer took it and its
        // tmux pane with it. Draining continues past the cap and discards
        // the overflow, so the child never blocks on a full pipe; the tail
        // past MAX_OUTPUT is output `poll` already reports as truncated, so
        // nothing observable is lost.
        let stdout_buf_clone = Arc::clone(&stdout_buf);
        let stdout_stop = Arc::new(AtomicBool::new(false));
        let stdout_stop_clone = Arc::clone(&stdout_stop);
        thread::spawn(move || {
            drain_capped(&mut stdout, &stdout_buf_clone, &stdout_stop_clone);
        });
        let stderr_buf_clone = Arc::clone(&stderr_buf);
        let stderr_stop = Arc::new(AtomicBool::new(false));
        let stderr_stop_clone = Arc::clone(&stderr_stop);
        thread::spawn(move || {
            drain_capped(&mut stderr, &stderr_buf_clone, &stderr_stop_clone);
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
            last_used: Instant::now(),
        };
        let mut jobs = self.jobs.lock().unwrap();
        jobs.insert(id.clone(), job);
        prune_finished_jobs(&mut jobs);
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
        // A poll is a use: recently polled jobs are evicted last.
        job.last_used = Instant::now();

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
        let command = job.command.clone();

        let mut result = format!(
            "job: {job_id}\ncommand: {}\nstatus: {status}\nelapsed: {:.1}s\n{exit_line}",
            command,
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
        // This poll may have just observed a job's exit: shed oldest finished
        // entries so the table (and its retained output buffers) stays bounded
        // even in a session that spawns background jobs liberally.
        prune_finished_jobs(&mut jobs);
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

    /// Kill every running job and drop all entries, finished or not.
    ///
    /// Called on session close: a session's background jobs live from
    /// session/new to session/close, and nothing they started may outlive
    /// the session. A daemon or tunnel left behind would otherwise keep its
    /// output pipes open, pinning this process's reader threads and buffers
    /// until it dies on its own.
    pub fn clear(&self) {
        let mut jobs = self.jobs.lock().unwrap();
        kill_running_jobs(&mut jobs);
        jobs.clear();
    }
}

/// SIGKILL every job whose process has not been observed exiting, marking it
/// killed and reaping the child. Entries are retained; callers decide whether
/// to drop them afterwards.
fn kill_running_jobs(jobs: &mut HashMap<String, Job>) {
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
        // A SIGKILLed process dies at once, so the blocking wait returns in
        // milliseconds and the child is reaped here — a try_wait probe could
        // lose the race and leave a permanent zombie once the entry (holding
        // the `Child`) is dropped.
        if let Ok(status) = job.child.wait() {
            job.exit_code = status.code();
        }
    }
}

impl Default for JobTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Evict finished jobs past the retention caps, least recently used first
/// (LRU by last spawn/poll touch). Running jobs are never evicted — only
/// entries whose process has exited or been killed, which hold nothing the
/// model needs beyond their output. Caps: [`MAX_RETAINED_FINISHED_JOBS`]
/// entries and [`MAX_RETAINED_FINISHED_BYTES`] of retained output, so a
/// session that spawns many jobs (or a few very chatty ones) stays bounded
/// while recently-used output stays readable.
fn prune_finished_jobs(jobs: &mut HashMap<String, Job>) {
    let mut finished: Vec<(String, Instant, usize)> = jobs
        .iter()
        .filter(|(_, job)| job.exit_code.is_some() || job.killed)
        .map(|(id, job)| {
            let bytes = job.stdout_buf.lock().unwrap().len()
                + job.stderr_buf.lock().unwrap().len()
                + job.command.len();
            (id.clone(), job.last_used, bytes)
        })
        .collect();
    let total_bytes: usize = finished.iter().map(|(_, _, bytes)| bytes).sum();
    if finished.len() <= MAX_RETAINED_FINISHED_JOBS && total_bytes <= MAX_RETAINED_FINISHED_BYTES {
        return;
    }
    finished.sort_by_key(|(_, last_used, _)| *last_used);
    let mut excess = finished.len().saturating_sub(MAX_RETAINED_FINISHED_JOBS);
    let mut retained_bytes = total_bytes;
    for (id, _, bytes) in &finished {
        if excess == 0 && retained_bytes <= MAX_RETAINED_FINISHED_BYTES {
            break;
        }
        jobs.remove(id);
        excess = excess.saturating_sub(1);
        retained_bytes = retained_bytes.saturating_sub(*bytes);
    }
}

impl super::SessionState for JobTable {
    fn shutdown(&self) {
        // Session close: kill running jobs and drop every entry (and its
        // output buffers) immediately, not just when the session struct
        // happens to be dropped.
        self.clear();
    }
}

pub struct ShellTool {
    timeout_secs: u64,
    jobs: Arc<JobTable>,
}

impl ShellTool {
    /// Construct with session-scoped state. Creates a `JobTable` in
    /// `SessionStates` if not already present (first prompt), then retrieves
    /// it so background jobs survive across prompts and are killed on
    /// session close.
    pub fn new(states: &mut super::SessionStates, _cwd: &str) -> Self {
        if states.get::<JobTable>().is_none() {
            states.insert(Arc::new(JobTable::new()));
        }
        Self::with_job_table(states.get::<JobTable>().unwrap())
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
            "description": "Run a shell command in the working directory (cwd). You are already in the working directory — no need to `cd` into it. Returns stdout, stderr, and exit code. Commands have a timeout (default 120s). Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `write` or `edit` instead. To run a long-running command in the background, set `background: true`; you get a job id back and can poll its output later with `job_id`, or terminate it with `job_id` + `kill: true`. Set `outside_cwd: true` when you legitimately need to `cd` into or operate on a path outside the working directory (e.g. an agent session directory explicitly permitted by the task instructions).",
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
                        "description": "If true, spawn the command in the background and return a job id immediately instead of waiting. Poll with `job_id` at any time later in the session; jobs live until session close. Or terminate a job with `job_id` + `kill: true`."
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Poll or kill a background job. Returns accumulated stdout/stderr and current status. Ignored unless polling or killing."
                    },
                    "kill": {
                        "type": "boolean",
                        "description": "When true with `job_id`, terminate the background job."
                    },
                    "outside_cwd": {
                        "type": "boolean",
                        "description": "When true, suppress the warning about `cd`-ing to a path outside the working directory. Use this when the task instructions explicitly direct you to operate on a path outside the workspace (e.g. running a script from an agent session directory). Default false.",
                        "default": false
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
        let outside_cwd = args["outside_cwd"].as_bool().unwrap_or(false);

        // Detect wrong-directory `cd` prefixes and warn. We don't strip or
        // modify the command — if the model cd's to the wrong path, the
        // command fails naturally, which is clearer feedback than silently
        // fixing it. The warning is prepended to the tool result so the model
        // sees it. Suppressed when the caller explicitly opted into operating
        // outside the working directory (e.g. running a script from an agent
        // session directory).
        let cd_warning = if outside_cwd {
            None
        } else {
            detect_wrong_cd(command, cwd)
        };

        // Detect file-writing patterns and log a warning. We don't block the
        // command (some legitimate uses exist, e.g. `git commit` writes files),
        // but we surface it so the model gets feedback in the next turn's logs.
        if looks_like_file_write(command) {
            tracing::warn!(
                "harness: shell command appears to write files directly — use write/edit instead: {}",
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

        // Read stdout and stderr on separate threads so a chatty child can't
        // deadlock on a full pipe while we poll for exit. Reads are bounded:
        // when the command times out (or its pipes are otherwise still open
        // with no writer making progress), the readers are stopped instead of
        // joined, so a grandchild that inherited the pipes and outlived the
        // command (a tunnel, a daemon) can never hang the agent loop.
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));
        let stdout_stop = Arc::new(AtomicBool::new(false));
        let stderr_stop = Arc::new(AtomicBool::new(false));

        let stdout_buf_clone = Arc::clone(&stdout_buf);
        let stdout_stop_clone = Arc::clone(&stdout_stop);
        let stdout_handle = thread::spawn(move || {
            drain_capped(&mut stdout, &stdout_buf_clone, &stdout_stop_clone);
        });

        let stderr_buf_clone = Arc::clone(&stderr_buf);
        let stderr_stop_clone = Arc::clone(&stderr_stop);
        let stderr_handle = thread::spawn(move || {
            drain_capped(&mut stderr, &stderr_buf_clone, &stderr_stop_clone);
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

        // The child is gone. EOF on the output pipes follows the moment every
        // process holding them is dead. Give the readers that moment; if a
        // grandchild still holds a pipe, stop the readers (keeping whatever
        // output arrived) rather than blocking forever. The drain waits in
        // poll slices, so both threads observe `stop` and exit within one
        // slice and join returns — abandoning the handles here would leak the
        // threads, their stacks, their pipe fds, and the output buffers for
        // the rest of the process lifetime.
        let collected = wait_for_pipes_or_orphan(&stdout_handle, &stderr_handle);
        if !collected {
            stdout_stop.store(true, Ordering::SeqCst);
            stderr_stop.store(true, Ordering::SeqCst);
        }
        let _ = stdout_handle.join();
        let _ = stderr_handle.join();
        let exit_status = child.wait().ok();

        let stdout_guard = stdout_buf.lock().unwrap();
        let stderr_guard = stderr_buf.lock().unwrap();
        let stdout_str = String::from_utf8_lossy(&stdout_guard);
        let stderr_str = String::from_utf8_lossy(&stderr_guard);
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
    // doesn't exist), treat the cd as wrong — but only for absolute paths.
    // Relative paths (e.g. `cd subdir && go build`) are legitimate: the model
    // is cd-ing into a subdirectory to run a command there, not guessing the
    // wrong project root.
    let cwd_canonical = Path::new(cwd)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| cwd.to_string());

    let cd_path_absolute = Path::new(cd_path).is_absolute();

    if !cd_path_absolute {
        // Relative cd paths are always legitimate — no warning.
        return None;
    }

    let cd_canonical = Path::new(cd_path)
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
/// sandboxed `write`/`edit` tools. Returns true for patterns like
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
    use crate::harness::tools::test_util;
    use std::fs;
    use std::io;
    use std::process::ChildStdout;

    #[test]
    fn drain_capped_stores_at_most_max_output_bytes() {
        // A chatty background job must not grow its buffer without limit:
        // this is the ~93 GB OOM failure mode. The stream is drained to EOF
        // (a stopped reader would block the child on a full pipe), but only
        // the first MAX_OUTPUT bytes are ever stored.
        let buf = Arc::new(Mutex::new(Vec::new()));
        let stop = AtomicBool::new(false);
        let command = format!("head -c {} /dev/zero | tr '\\0' x", MAX_OUTPUT * 4);
        let mut holder = PipeHolder::new(&command).expect("spawn output source");
        let mut read = holder.take_stdout();

        drain_capped(&mut read, &buf, &stop);
        let _ = holder.child.kill();
        let _ = holder.child.wait();

        let stored = buf.lock().unwrap().clone();
        assert_eq!(stored.len(), MAX_OUTPUT);
        assert!(stored.iter().all(|&b| b == b'x'));
    }

    #[test]
    fn drain_capped_keeps_short_streams_whole() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let stop = AtomicBool::new(false);
        let mut holder = PipeHolder::new("printf short-output").expect("spawn output source");
        let mut read = holder.take_stdout();

        drain_capped(&mut read, &buf, &stop);
        let _ = holder.child.wait();

        assert_eq!(buf.lock().unwrap().as_slice(), b"short-output");
    }

    #[test]
    fn drain_capped_stops_when_the_stop_flag_is_set() {
        // The orphan-pipe abandonment path relies on the stop flag ending a
        // reader that will never see EOF (a grandchild holding the pipe).
        // The pipe read is exercised through a child process holding the
        // write end (a bare `File` pair here is subject to whatever fd
        // hygiene the test harness applies between tests — the first
        // `read` came back EBADF when run in the full suite).
        let buf = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let mut holder = PipeHolder::new("sleep 30").expect("spawn pipe holder");
        let mut read = holder.take_stdout();

        let handle = {
            let buf = Arc::clone(&buf);
            let stop = Arc::clone(&stop);
            thread::spawn(move || drain_capped(&mut read, &buf, &stop))
        };
        thread::sleep(Duration::from_millis(100));
        assert!(
            !handle.is_finished(),
            "reader must block without stop (finished={:?})",
            handle.join()
        );
        stop.store(true, Ordering::SeqCst);
        handle.join().unwrap();
        assert!(buf.lock().unwrap().is_empty());
        let _ = holder.child.kill();
        let _ = holder.child.wait();
    }

    /// A live child whose stdout pipe never reaches EOF while it runs.
    struct PipeHolder {
        child: Child,
    }

    impl PipeHolder {
        fn new(command: &str) -> io::Result<Self> {
            let child = Command::new("bash")
                .args(["-c", command])
                .stdout(Stdio::piped())
                .spawn()?;
            Ok(Self { child })
        }

        fn take_stdout(&mut self) -> ChildStdout {
            self.child.stdout.take().unwrap()
        }
    }

    #[test]
    fn background_job_buffer_is_bounded_by_max_output() {
        // End to end: spawn a background job that emits far more than
        // MAX_OUTPUT, poll it after it exits, and assert the stored buffer
        // stayed at the cap rather than growing with the stream.
        let jobs = Arc::new(JobTable::new());
        let id = jobs
            .spawn("yes hello | head -c $(( 50 * 1024 * 1024 ))", "/tmp")
            .unwrap();
        // Wait for the job to finish so the readers have drained to EOF.
        loop {
            let done = {
                let mut table = jobs.jobs.lock().unwrap();
                let job = table.get_mut(&id).unwrap();
                match job.child.try_wait() {
                    Ok(Some(status)) => {
                        job.exit_code = status.code();
                        true
                    }
                    _ => false,
                }
            };
            if done {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        // Give the reader threads a moment to finish draining to EOF.
        thread::sleep(Duration::from_millis(200));
        let table = jobs.jobs.lock().unwrap();
        let job = table.get(&id).unwrap();
        assert!(
            job.stdout_buf.lock().unwrap().len() <= MAX_OUTPUT,
            "stdout buffer must be capped: {}",
            job.stdout_buf.lock().unwrap().len()
        );
        assert!(job.stderr_buf.lock().unwrap().len() <= MAX_OUTPUT);
    }

    #[test]
    fn runs_echo_command() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let args = json!({"command": "echo hello"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("hello"));
        assert!(result.contains("exit code: 0"));
    }

    #[test]
    fn captures_stderr() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let args = json!({"command": "echo err >&2"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("err"));
    }

    #[test]
    fn captures_nonzero_exit() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let args = json!({"command": "exit 42"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("exit code: 42"));
    }

    #[test]
    fn kills_on_timeout() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
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
    fn timeout_returns_even_when_a_grandchild_holds_the_pipes() {
        // A command that spawns a setsid'd descendant (an ssh tunnel, a
        // daemon) and exits. killpg cannot reach the detached grandchild, so
        // the output pipes stay open after the kill. The tool must return
        // anyway — hanging here froze the whole QA agent loop (run.log ended
        // mid-command at 11:18:26 with the harness threads parked in
        // `read_to_end` forever).
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        // The grandchild inherits the command's stdout and outlives the kill:
        // `sleep 301` holds the pipe write end open for the grace window. The
        // sleep durations are unique to this test so the pattern-based pkill
        // cleanup below can never kill another test's `sleep` job.
        let args = json!({
            "command": "setsid sh -c 'sleep 301' & echo staged; sleep 302",
            "timeout_secs": 2,
        });
        let start = Instant::now();
        let result = tool.execute(&args, "/tmp").unwrap();
        let elapsed = start.elapsed();
        assert!(result.contains("timed out"));
        assert!(
            elapsed < Duration::from_secs(15),
            "orphaned pipes must not block return, took {elapsed:?}"
        );
        // Clean up the leftover sleeps so the test run is self-contained.
        let _ = Command::new("pkill").args(["-f", "sleep 301"]).status();
        let _ = Command::new("pkill").args(["-f", "sleep 302"]).status();
    }

    #[cfg(unix)]
    #[test]
    fn orphaned_pipes_do_not_accumulate_reader_threads() {
        // Every command whose pipes outlive it used to abandon its reader
        // threads (mem::forget): blocked in `read`, they never observed
        // the stop flag and leaked their stacks, pipe fds, and output buffers
        // for the process lifetime. The poll-slice drain must let each
        // command's readers exit after the grace window, keeping the process
        // thread count flat across repeated orphaned commands.
        let thread_count = || -> usize {
            fs::read_to_string("/proc/self/status")
                .expect("read /proc/self/status")
                .lines()
                .find_map(|line| line.strip_prefix("Threads:"))
                .expect("Threads: line")
                .trim()
                .parse()
                .expect("thread count")
        };

        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let args = json!({
            "command": "setsid sh -c 'sleep 303' & echo staged; sleep 304",
            "timeout_secs": 2,
        });

        // Warm up: one command settles any lazily-created runtime threads.
        tool.execute(&args, "/tmp").unwrap();
        let baseline = thread_count();

        for _ in 0..3 {
            tool.execute(&args, "/tmp").unwrap();
        }
        assert!(
            thread_count() <= baseline,
            "orphaned commands must not leak reader threads: baseline {baseline}, now {}",
            thread_count()
        );

        // Clean up the leftover sleeps so the test run is self-contained.
        let _ = Command::new("pkill").args(["-f", "sleep 303"]).status();
        let _ = Command::new("pkill").args(["-f", "sleep 304"]).status();
    }

    #[test]
    fn finished_jobs_evict_least_recently_used_first() {
        // LRU: eviction touches the least recently spawned/polled finished
        // job first, so a job the model re-polls survives even when old, and
        // running jobs are never evicted.
        let jobs = JobTable::new();
        let running = jobs.spawn("sleep 60", "/tmp").expect("spawn running job");

        let spawn_finished = |jobs: &JobTable, i: usize| {
            let id = jobs
                .spawn(&format!("echo job-{i}"), "/tmp")
                .expect("spawn job");
            loop {
                if jobs.poll(&id).unwrap().contains("exited") {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            id
        };

        // 64 finished jobs, polled sequentially so their last_use order
        // matches spawn order (job-0 oldest). A running job in the middle is
        // never touched again.
        let first = spawn_finished(&jobs, 0);
        let middle: Vec<String> = (1..MAX_RETAINED_FINISHED_JOBS)
            .map(|i| spawn_finished(&jobs, i))
            .collect();

        // Re-poll the oldest: it becomes the most recently used.
        jobs.poll(&first).unwrap();

        // One more finished job pushes the table over the cap.
        let newest = spawn_finished(&jobs, MAX_RETAINED_FINISHED_JOBS);

        let table = jobs.jobs.lock().unwrap();
        assert!(
            !table.contains_key(&middle[0]),
            "least recently used finished job must be evicted"
        );
        assert!(
            table.contains_key(&first),
            "recently re-polled job must survive eviction"
        );
        assert!(
            table.contains_key(&newest),
            "the newest job must survive eviction"
        );
        assert!(
            table.contains_key(&running),
            "running jobs are never evicted"
        );
        let finished_count = table
            .values()
            .filter(|job| job.exit_code.is_some() || job.killed)
            .count();
        assert_eq!(
            finished_count, MAX_RETAINED_FINISHED_JOBS,
            "finished jobs are held at the cap"
        );
    }

    #[test]
    fn finished_jobs_are_evicted_when_retained_bytes_exceed_the_cap() {
        // The byte cap bounds the table's retained output regardless of job
        // count: chatty jobs evict LRU-style until the total fits.
        let jobs = JobTable::new();
        let command = format!("head -c {} /dev/zero | tr '\\0' x", MAX_OUTPUT);
        let mut ids = Vec::new();
        for _ in 0..50 {
            let id = jobs.spawn(&command, "/tmp").expect("spawn chatty job");
            loop {
                if jobs.poll(&id).unwrap().contains("exited") {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            ids.push(id);
        }

        let table = jobs.jobs.lock().unwrap();
        let retained: usize = table
            .values()
            .map(|job| job.stdout_buf.lock().unwrap().len() + job.stderr_buf.lock().unwrap().len())
            .sum();
        assert!(
            retained <= MAX_RETAINED_FINISHED_BYTES,
            "retained output must fit the byte cap: {retained}"
        );
        assert!(
            table.len() < 50,
            "byte cap must have forced evictions: {} entries",
            table.len()
        );
        assert!(
            table.contains_key(ids.last().unwrap()),
            "the most recently used job must survive"
        );
    }

    #[cfg(unix)]
    #[test]
    fn clear_kills_running_jobs_and_drops_all_entries() {
        let jobs = JobTable::new();
        let running = jobs.spawn("sleep 60", "/tmp").unwrap();
        let finished = jobs.spawn("echo done", "/tmp").unwrap();
        loop {
            if jobs.poll(&finished).unwrap().contains("exited") {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        // Spawned jobs are their own process-group leader (process_group(0)).
        let pgid = jobs.jobs.lock().unwrap().get(&running).unwrap().pid as i32;

        jobs.clear();

        assert!(
            jobs.jobs.lock().unwrap().is_empty(),
            "clear must drop every entry, running or finished"
        );
        assert!(jobs.poll(&running).is_err());
        assert!(jobs.poll(&finished).is_err());

        // The running job must actually be dead, not just forgotten: poll
        // until the process group is gone (SIGKILL delivery is asynchronous).
        let mut gone = false;
        for _ in 0..100 {
            let rc = unsafe { libc::kill(pgid, 0) };
            if rc == -1 && Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                gone = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(gone, "clear must kill the running job's process group");
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
    fn detect_wrong_cd_silent_for_relative_dot() {
        // `cd .` is the same directory — no warning (relative path).
        assert!(detect_wrong_cd("cd . && go build ./...", "/tmp").is_none());
    }

    #[test]
    fn detect_wrong_cd_silent_for_relative_subdirectory() {
        // Relative cd paths are always legitimate — no warning, even if the
        // subdirectory doesn't exist (the command will fail naturally).
        assert!(detect_wrong_cd("cd subdir && go build ./...", "/tmp").is_none());
        assert!(detect_wrong_cd("cd inventory/pdf && go build ./...", "/tmp").is_none());
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
    fn execute_suppresses_cd_warning_when_outside_cwd_true() {
        // The QA agent legitimately `cd`s into its session test-scripts dir to
        // run scripts. With outside_cwd: true, the wrong-directory cd warning
        // must not appear in the tool result.
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let temp = test_util::unique_test_dir();
        let outside = temp.path().to_path_buf();
        let args = json!({
            "command": format!("cd {} && echo ran", outside.to_string_lossy()),
            "outside_cwd": true
        });
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(
            !result.contains("not the working directory"),
            "no cd warning expected with outside_cwd, got: {result}"
        );
        assert!(result.contains("ran"));
    }

    #[test]
    fn execute_warns_on_cd_outside_when_outside_cwd_absent() {
        // Default (outside_cwd false/absent): a cd to a path that isn't cwd
        // still produces the warning.
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let temp = test_util::unique_test_dir();
        let outside = temp.path().to_path_buf();
        let args = json!({
            "command": format!("cd {} && echo ran", outside.to_string_lossy())
        });
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(
            result.contains("not the working directory"),
            "expected cd warning without outside_cwd, got: {result}"
        );
    }

    #[test]
    fn spawn_background_returns_job_id() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
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
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
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
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
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
    fn poll_unknown_job_id_returns_error() {
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
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
        let tool = ShellTool::with_job_table(Arc::new(JobTable::new()));
        let args = json!({"command": "echo foreground"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("foreground"));
        assert!(result.contains("exit code: 0"));
        assert!(!result.contains("Background job started"));
    }
}
