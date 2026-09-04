//! Auth-provider command: dynamic per-request headers for the LLM endpoint.
//!
//! An endpoint may declare `auth_provider = "uv run python auth-helper.py"` in its
//! `[acp.*]` config. The command is arbitrary — it only has to follow the
//! output contract: print progress for the user (OAuth callback URLs, "open
//! this link" notices, retry hints) to stderr as it goes, and print exactly
//! one JSON document as the final stdout line:
//!
//! ```json
//! {"expiration": 1735689600, "headers": {"Authorization": "Bearer ..."}}
//! ```
//!
//! `expiration` is a Unix timestamp (seconds; fractional seconds and
//! millisecond values are accepted) after which the headers must be
//! refreshed by re-running the command. `headers` maps header names to
//! values and is applied verbatim to every request to the endpoint — the
//! command owns the full header set, whether or not it is about auth.
//!
//! Anything the command writes to stdout before the final document, or to
//! stderr at any point (e.g. a URL the user must visit to complete an OAuth
//! flow), is progress output: it is surfaced to the user via `warn!` tracing
//! events, which the TUI renders as ⚠ lines — never mistaken for the final
//! output. Once the command exits with a parseable document, the headers are
//! cached and reused until shortly before the expiration, so the interactive
//! part of an OAuth dance only happens when the credentials actually need
//! refreshing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Progress lines longer than this are truncated before surfacing.
const MAX_PROGRESS_LINE_LEN: usize = 2048;
/// Guard against a misbehaving provider flooding stdout with non-JSON lines.
const MAX_PROGRESS_LINES: usize = 100;
/// Upper bound on how long a single auth-provider invocation may run. OAuth
/// flows wait for the user to complete the dance in a browser, so this must
/// be generous.
const PROVIDER_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a follower process waits for the leader's provider run to finish
/// before giving up. Larger than [`PROVIDER_TIMEOUT`] plus [`POST_EXIT_GRACE`]
/// so the leader can complete a full interactive flow.
const FOLLOWER_WAIT: Duration = Duration::from_secs(700);
/// After the provider's stderr pipe closes, buffered stdout gets this long to
/// deliver. A provider that backgrounds a daemon holding the stdout pipe must
/// not hang us.
const POST_EXIT_GRACE: Duration = Duration::from_secs(2);
/// Re-run the provider this long before the cached headers actually expire,
/// so requests never go out with stale credentials.
const EXPIRY_REFRESH_MARGIN: Duration = Duration::from_secs(60);
/// How long a published failure stays authoritative: within this window a
/// follower that takes over the lock surfaces the leader's error instead of
/// starting its own (possibly interactive) run. After it passes, the next
/// caller may retry — e.g. once the user has completed the OAuth dance.
const ERROR_REPUBLISH_TTL: Duration = Duration::from_secs(30);

/// Headers for the endpoint, valid until `expires_at`.
#[derive(Debug, Clone)]
pub struct CachedHeaders {
    pub headers: HashMap<String, String>,
    pub expires_at: SystemTime,
}

impl CachedHeaders {
    /// True when the headers are expired or within the refresh margin.
    fn needs_refresh(&self, now: SystemTime) -> bool {
        now >= self
            .expires_at
            .checked_sub(EXPIRY_REFRESH_MARGIN)
            .unwrap_or(now)
    }
}

/// Runs the auth-provider command on demand and caches the returned headers
/// until their expiration approaches.
///
/// Multiple harness processes (one per agent instance) may hit the same
/// endpoint at the same time. A cross-process single-flight guard ensures
/// only ONE of them runs the provider command at any moment — the first to
/// arrive executes it while the others hold on an exclusive file lock and
/// then reuse the leader's cached result. That matters for interactive
/// providers: without the guard, ten workers hitting an expired token would
/// open ten OAuth browser flows.
pub struct AuthProvider {
    argv: Vec<String>,
    cache: Mutex<Option<CachedHeaders>>,
    /// Cross-process coordination: an exclusive flock serializes provider
    /// runs, and the leader publishes its result (or error) to a sidecar
    /// file so followers skip re-running.
    state_dir: PathBuf,
    /// Working directory for the provider command — the potlatch config
    /// directory, so `./auth-tool.py` resolves relative to the config.
    /// `None` inherits the harness's cwd.
    working_dir: Option<PathBuf>,
}

impl AuthProvider {
    pub fn new(argv: Vec<String>, working_dir: Option<PathBuf>) -> Self {
        Self::with_state_dir(argv, working_dir, default_state_dir())
    }

    /// `state_dir` is injectable for tests; production uses the shared
    /// per-user directory so all harness processes coordinate through it.
    pub(crate) fn with_state_dir(
        argv: Vec<String>,
        working_dir: Option<PathBuf>,
        state_dir: PathBuf,
    ) -> Self {
        Self {
            argv,
            cache: Mutex::new(None),
            state_dir,
            working_dir,
        }
    }

    /// The provider command's program name, for log lines.
    pub fn program(&self) -> &str {
        self.argv.first().map(String::as_str).unwrap_or("(none)")
    }

    /// Headers to use right now, running (or re-running) the provider
    /// command when nothing is cached or the cache is at/near expiry.
    ///
    /// Cross-process single-flight: only one harness process runs the
    /// provider at a time; followers hold on the file lock and then reuse
    /// the leader's published result.
    pub fn headers(&self) -> Result<HashMap<String, String>> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let now = SystemTime::now();
        if let Some(cached) = cache.as_ref()
            && !cached.needs_refresh(now)
        {
            return Ok(cached.headers.clone());
        }

        let fresh = self.run_provider_single_flight()?;
        *cache = Some(fresh);
        Ok(cache.as_ref().expect("just stored").headers.clone())
    }

    /// Run the provider under the cross-process guard. The first process to
    /// take the lock executes the command; the others hold on the lock, then
    /// load the leader's published result. If the leader failed, followers
    /// surface that failure (until [`ERROR_REPUBLISH_TTL`] passes) instead of
    /// stampeding into their own interactive runs.
    fn run_provider_single_flight(&self) -> Result<CachedHeaders> {
        std::fs::create_dir_all(&self.state_dir).with_context(|| {
            format!(
                "create auth-provider state dir {}",
                self.state_dir.display()
            )
        })?;

        let key = state_key(&self.argv);
        let lock_path = self.state_dir.join(format!("run-{key}.lock"));
        let result_path = self.state_dir.join(format!("run-{key}.json"));

        let lock_file = File::create(&lock_path)
            .with_context(|| format!("open lock file {}", lock_path.display()))?;
        let mut logged_hold = false;
        let started = Instant::now();
        loop {
            match try_lock_exclusive(&lock_file) {
                Ok(()) => break,
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    if started.elapsed() >= FOLLOWER_WAIT {
                        bail!(
                            "another auth-provider run holds the lock (waited {FOLLOWER_WAIT:?})"
                        );
                    }
                    if !logged_hold {
                        logged_hold = true;
                        tracing::warn!(
                            target: "potlatch::auth_provider",
                            "another harness process is running the auth provider; holding"
                        );
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(err) => {
                    return Err(err).with_context(|| format!("lock {}", lock_path.display()));
                }
            }
        }

        // We hold the lock. If the previous holder published a fresh result,
        // reuse it instead of re-running.
        match load_published(&result_path) {
            PublishedState::Fresh(cached) if !cached.needs_refresh(SystemTime::now()) => {
                return Ok(cached);
            }
            PublishedState::Failed(message) => {
                bail!("auth provider recently failed: {message}");
            }
            PublishedState::Fresh(_) | PublishedState::None => {}
        }

        match run_provider(&self.argv, self.working_dir.as_deref()) {
            Ok(cached) => {
                publish(&result_path, &cached);
                Ok(cached)
            }
            Err(e) => {
                publish_error(&result_path, &format!("{e:#}"));
                Err(e)
            }
        }
    }
}

/// Per-user directory shared by all harness processes for auth-provider
/// coordination files.
fn default_state_dir() -> PathBuf {
    crate::harness::home_dir()
        .join(".potlatch")
        .join("auth-provider")
}

/// A stable, filesystem-safe key for an argv so different provider commands
/// coordinate on different lock files. Non-security: FNV-1a suffices.
fn state_key(argv: &[String]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in argv.join("\u{0}").as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Non-blocking exclusive flock across processes.
fn try_lock_exclusive(file: &File) -> std::io::Result<()> {
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Resolve the program path for spawning. Relative paths (`./auth-tool.py`,
/// `bin/tool`) are made absolute against the working dir — the executable is
/// resolved against the parent's cwd before the child's `current_dir` takes
/// effect, so this is required for them to work at all. A bare command name
/// (`auth-tool.py`, `sh`) is tried under the working dir first — so a script
/// sitting alongside `potlatch.toml` runs with plain `auth-tool.py` — and
/// falls back to `PATH` when not found there.
fn resolve_program(program: &str, working_dir: Option<&Path>) -> String {
    let Some(dir) = working_dir else {
        return program.to_string();
    };
    if program.starts_with("./") || program.starts_with("../") {
        return dir.join(program).display().to_string();
    }
    if program.contains('/') {
        if Path::new(program).is_absolute() {
            return program.to_string();
        }
        return dir.join(program).display().to_string();
    }
    let candidate = dir.join(program);
    if candidate.is_file() {
        candidate.display().to_string()
    } else {
        program.to_string()
    }
}

/// What the previous lock holder left behind.
#[derive(Debug)]
enum PublishedState {
    Fresh(CachedHeaders),
    Failed(String),
    None,
}

/// Serialized leader result: headers plus expiration, or an error string.
#[derive(Debug, Serialize, Deserialize)]
struct PublishedRun {
    #[serde(with = "system_time_millis")]
    expires_at: SystemTime,
    headers: Option<HashMap<String, String>>,
    error: Option<String>,
}

mod system_time_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error> {
        let millis = time
            .duration_since(UNIX_EPOCH)
            .map_err(serde::ser::Error::custom)?
            .as_millis() as u64;
        millis.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<SystemTime, D::Error> {
        let millis = u64::deserialize(deserializer)?;
        UNIX_EPOCH
            .checked_add(std::time::Duration::from_millis(millis))
            .ok_or_else(|| serde::de::Error::custom("expiration out of range"))
    }
}

fn publish(result_path: &Path, cached: &CachedHeaders) {
    let published = PublishedRun {
        expires_at: cached.expires_at,
        headers: Some(cached.headers.clone()),
        error: None,
    };
    let _ = std::fs::write(
        result_path,
        serde_json::to_vec(&published).unwrap_or_default(),
    );
}

fn publish_error(result_path: &Path, message: &str) {
    let published = PublishedRun {
        expires_at: SystemTime::now() + ERROR_REPUBLISH_TTL,
        headers: None,
        error: Some(message.to_string()),
    };
    let _ = std::fs::write(
        result_path,
        serde_json::to_vec(&published).unwrap_or_default(),
    );
}

fn load_published(result_path: &Path) -> PublishedState {
    let Ok(bytes) = std::fs::read(result_path) else {
        return PublishedState::None;
    };
    let Ok(published) = serde_json::from_slice::<PublishedRun>(&bytes) else {
        return PublishedState::None;
    };
    if let Some(message) = published.error {
        // A recently failed run: honor it so followers don't stampede, but
        // let a later caller retry once the TTL passes.
        if SystemTime::now() < published.expires_at {
            return PublishedState::Failed(message);
        }
        return PublishedState::None;
    }
    published
        .headers
        .map(|headers| {
            PublishedState::Fresh(CachedHeaders {
                headers,
                expires_at: published.expires_at,
            })
        })
        .unwrap_or(PublishedState::None)
}

/// Run the provider command once. Stderr surfaces line-by-line as it arrives
/// (an OAuth provider may print a URL and then block for minutes while the
/// user completes the dance). Non-final stdout lines are progress; the last
/// stdout line must be the JSON document. The whole invocation is bounded by
/// [`PROVIDER_TIMEOUT`].
///
/// `working_dir` (the potlatch config directory, when known) becomes the
/// child's cwd, and a *relative* program path is resolved against it first —
/// so `./auth-tool.py` or `auth-tool.py` sitting alongside `potlatch.toml`
/// just work. Absolute paths and bare command names found on `PATH` are
/// unaffected.
fn run_provider(argv: &[String], working_dir: Option<&Path>) -> Result<CachedHeaders> {
    let program = argv
        .first()
        .ok_or_else(|| anyhow::anyhow!("auth_provider command is empty"))?;

    let resolved = resolve_program(program, working_dir);
    let mut cmd = Command::new(&resolved);
    cmd.args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to run auth_provider `{resolved}`"))?;

    let stdout = child
        .stdout
        .take()
        .context("auth_provider stdout not piped")?;
    let stderr = child
        .stderr
        .take()
        .context("auth_provider stderr not piped")?;

    // stderr surfaces as it arrives; the channel closes when the pipe does.
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let stderr_thread = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });

    // stdout accumulates in parallel; the sender drops when the pipe closes,
    // which delivers the result to doc_rx.
    let (doc_tx, doc_rx) = mpsc::channel::<Result<(String, Vec<String>)>>();
    let stdout_thread = std::thread::spawn(move || {
        let _ = doc_tx.send(parse_stdout_lines(BufReader::new(stdout)));
    });

    let started = Instant::now();
    let deadline = started + PROVIDER_TIMEOUT;

    // Drain stderr until its pipe closes (process exiting) or we time out.
    loop {
        let now = Instant::now();
        if now >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("auth_provider `{program}` timed out after {PROVIDER_TIMEOUT:?}");
        }
        match line_rx.recv_timeout(deadline - now) {
            Ok(line) => surface_progress(&line),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("auth_provider `{program}` timed out after {PROVIDER_TIMEOUT:?}");
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // The stderr pipe closed, so the process is exiting. The stdout pipe may
    // still hold buffered data; give the reader a short grace period.
    let stdout_result = match doc_rx.recv_timeout(POST_EXIT_GRACE) {
        Ok(result) => Some(result),
        Err(mpsc::RecvTimeoutError::Timeout) | Err(mpsc::RecvTimeoutError::Disconnected) => None,
    };

    let _ = child.kill();
    let status = child.wait().unwrap_or_default();
    let _ = stderr_thread.join();
    let _ = stdout_thread.join();

    if !status.success() {
        // Progress lines (including the failure reason) were already
        // surfaced live while the command ran.
        bail!("auth_provider `{program}` exited with {status}");
    }

    let Some(Ok((document, progress))) = stdout_result else {
        bail!("auth_provider `{program}` produced no final JSON document (exit: {status})");
    };

    for line in progress {
        surface_progress(&line);
    }

    parse_auth_document(&document, program)
}

/// Read stdout line by line. Every line except the last non-empty one is
/// progress output; the last non-empty line must be the JSON document.
fn parse_stdout_lines(stdout: impl BufRead) -> Result<(String, Vec<String>)> {
    let mut last: Option<String> = None;
    let mut progress: Vec<String> = Vec::new();

    for line in stdout.lines() {
        let line = line.context("read auth_provider stdout")?;
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        if let Some(prev) = last.take() {
            push_progress(&mut progress, &prev);
        }
        last = Some(trimmed.to_string());
    }

    match last {
        Some(doc) => Ok((doc, progress)),
        None => bail!("auth_provider produced no output (expected the headers JSON document)"),
    }
}

fn push_progress(progress: &mut Vec<String>, line: &str) {
    if progress.len() >= MAX_PROGRESS_LINES {
        return;
    }
    progress.push(compact(line));
}

/// Emit a progress line as a warning so the TUI shows it to the user (⚠
/// line) instead of it being silently swallowed or treated as the output.
fn surface_progress(line: &str) {
    let line = compact(line);
    let is_url = line
        .split_whitespace()
        .any(|word| word.starts_with("http://") || word.starts_with("https://"));
    if is_url {
        tracing::warn!(target: "potlatch::auth_provider", "auth provider: open this URL to continue: {line}");
    } else {
        tracing::warn!(target: "potlatch::auth_provider", "auth provider: {line}");
    }
}

fn compact(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.len() > MAX_PROGRESS_LINE_LEN {
        format!(
            "{}…",
            joined
                .chars()
                .take(MAX_PROGRESS_LINE_LEN)
                .collect::<String>()
        )
    } else {
        joined
    }
}

/// Parse the final stdout document into cached headers.
fn parse_auth_document(document: &str, program: &str) -> Result<CachedHeaders> {
    let value: Value = serde_json::from_str(document).with_context(|| {
        format!(
            "auth_provider `{program}` final output is not a JSON document: {}",
            preview(document)
        )
    })?;

    let expiration = value
        .get("expiration")
        .ok_or_else(|| anyhow::anyhow!("auth_provider document missing `expiration`"))?;
    let expires_at =
        parse_expiration(expiration).context("auth_provider `expiration` is not a timestamp")?;

    let headers_value = value
        .get("headers")
        .ok_or_else(|| anyhow::anyhow!("auth_provider document missing `headers`"))?;
    let headers_obj = headers_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("auth_provider `headers` must be a JSON object"))?;

    let mut headers = HashMap::new();
    for (name, value) in headers_obj {
        let Some(value) = value.as_str() else {
            bail!("auth_provider header `{name}` must be a string");
        };
        if name.is_empty() {
            bail!("auth_provider header name must not be empty");
        }
        headers.insert(name.clone(), value.to_string());
    }

    Ok(CachedHeaders {
        headers,
        expires_at,
    })
}

/// Accept seconds (integer or fractional), milliseconds, or an RFC 3339
/// timestamp. Milliseconds are detected by magnitude (> 10^11 is implausible
/// as seconds and matches ms timestamps through the year 5138).
fn parse_expiration(value: &Value) -> Result<SystemTime> {
    if let Some(text) = value.as_str() {
        let text = text.trim();
        if text.len() >= 20
            && (text.ends_with('Z') || text.ends_with('z') || text.contains('+'))
            && let Ok(dt) = chrono::DateTime::parse_from_rfc3339(text)
        {
            return SystemTime::UNIX_EPOCH
                .checked_add(Duration::from_secs(dt.timestamp().max(0) as u64))
                .context("expiration out of range");
        }
        let n: f64 = text.parse().map_err(|_| {
            anyhow::anyhow!("expiration must be a Unix timestamp or RFC 3339 string")
        })?;
        return parse_numeric_expiration(n);
    }

    let number = value
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("expiration must be a number or RFC 3339 string"))?;
    parse_numeric_expiration(number)
}

fn parse_numeric_expiration(seconds: f64) -> Result<SystemTime> {
    if !seconds.is_finite() || seconds < 0.0 {
        bail!("expiration must be a non-negative number");
    }
    let (secs, nanos) = if seconds > 1.0e11 {
        let secs = (seconds / 1000.0).trunc() as u64;
        let nanos = ((seconds / 1000.0).fract() * 1.0e9) as u32;
        (secs, nanos)
    } else {
        let secs = seconds.trunc() as u64;
        let nanos = (seconds.fract() * 1.0e9) as u32;
        (secs, nanos)
    };
    UNIX_EPOCH
        .checked_add(Duration::new(secs, nanos))
        .context("expiration out of range")
}

fn preview(text: &str) -> String {
    let text = text.trim();
    if text.len() <= 200 {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(200).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn epoch_plus(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("potlatch-auth-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An AuthProvider coordinating through an isolated state dir, running
    /// commands with `dir` as the working directory (standing in for the
    /// potlatch config directory).
    fn provider_in(dir: &std::path::Path, argv: Vec<String>) -> AuthProvider {
        AuthProvider::with_state_dir(argv, Some(dir.to_path_buf()), dir.join("state"))
    }

    fn provider(dir: &std::path::Path, script_path: &std::path::Path) -> AuthProvider {
        provider_in(dir, vec!["sh".into(), script_path.display().to_string()])
    }

    // --- document parsing ---

    #[test]
    fn parses_document_with_numeric_expiration() {
        let doc = serde_json::to_string(&json!({
            "expiration": 1_800_000_000,
            "headers": {"Authorization": "Bearer abc"}
        }))
        .unwrap();
        let cached = parse_auth_document(&doc, "test").unwrap();
        assert_eq!(
            cached.headers.get("Authorization").map(String::as_str),
            Some("Bearer abc")
        );
        assert_eq!(cached.expires_at, epoch_plus(1_800_000_000));
    }

    #[test]
    fn parses_expiration_as_milliseconds() {
        let doc = serde_json::to_string(&json!({
            "expiration": 1_800_000_000_000u64,
            "headers": {}
        }))
        .unwrap();
        let cached = parse_auth_document(&doc, "test").unwrap();
        assert_eq!(cached.expires_at, epoch_plus(1_800_000_000));
    }

    #[test]
    fn parses_expiration_as_rfc3339_string() {
        let doc = serde_json::to_string(&json!({
            "expiration": "2038-01-19T03:14:07Z",
            "headers": {}
        }))
        .unwrap();
        let cached = parse_auth_document(&doc, "test").unwrap();
        assert_eq!(cached.expires_at, epoch_plus(2_147_483_647));
    }

    #[test]
    fn parses_fractional_and_string_numeric_expiration() {
        let doc = serde_json::to_string(&json!({"expiration": 100.5, "headers": {}})).unwrap();
        let cached = parse_auth_document(&doc, "t").unwrap();
        assert_eq!(
            cached.expires_at.duration_since(UNIX_EPOCH).unwrap(),
            Duration::from_millis(100_500)
        );

        let doc =
            serde_json::to_string(&json!({"expiration": "1800000000", "headers": {}})).unwrap();
        let cached = parse_auth_document(&doc, "t").unwrap();
        assert_eq!(cached.expires_at, epoch_plus(1_800_000_000));
    }

    #[test]
    fn rejects_documents_missing_fields_or_malformed() {
        let missing_headers = serde_json::to_string(&json!({"expiration": 100})).unwrap();
        assert!(parse_auth_document(&missing_headers, "t").is_err());
        let missing_expiration = serde_json::to_string(&json!({"headers": {"A": "B"}})).unwrap();
        assert!(parse_auth_document(&missing_expiration, "t").is_err());
        let not_json = "open https://example.invalid/auth to continue";
        assert!(parse_auth_document(not_json, "t").is_err());
        let negative = serde_json::to_string(&json!({"expiration": -5, "headers": {}})).unwrap();
        assert!(parse_auth_document(&negative, "t").is_err());
        let non_string_header =
            serde_json::to_string(&json!({"expiration": 100, "headers": {"A": 1}})).unwrap();
        assert!(parse_auth_document(&non_string_header, "t").is_err());
    }

    // --- stdout line classification ---

    #[test]
    fn parse_stdout_lines_picks_last_nonempty_line_as_document() {
        let input =
            "visit https://auth.example.invalid/device\n\n{\"expiration\": 5, \"headers\": {}}\n";
        let (doc, progress) = parse_stdout_lines(std::io::Cursor::new(input)).unwrap();
        assert_eq!(doc, "{\"expiration\": 5, \"headers\": {}}");
        assert_eq!(
            progress,
            vec!["visit https://auth.example.invalid/device".to_string()]
        );
    }

    #[test]
    fn parse_stdout_lines_rejects_empty_output() {
        let err = parse_stdout_lines(std::io::Cursor::new("")).unwrap_err();
        assert!(err.to_string().contains("no output"));
    }

    // --- refresh policy ---

    #[test]
    fn headers_within_refresh_margin_need_refresh() {
        let cached = CachedHeaders {
            headers: HashMap::new(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
        };
        assert!(!cached.needs_refresh(SystemTime::now()));
        let soon = CachedHeaders {
            headers: HashMap::new(),
            expires_at: SystemTime::now() + Duration::from_secs(30),
        };
        assert!(soon.needs_refresh(SystemTime::now()));
    }

    // --- end-to-end subprocess behavior ---

    #[test]
    fn provider_cache_hit_avoids_second_run() {
        let dir = temp_dir("cache");
        let counter = dir.join("counter");
        std::fs::write(&counter, b"0").unwrap();

        // Each run increments the counter then emits a far-future expiration,
        // so the second headers() call must come from cache.
        let script_path = dir.join("provider.sh");
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nc=$(cat {})\nc=$((c+1))\necho $c > {}\nprintf '%s' \"{{\\\"expiration\\\": 99999999999, \\\"headers\\\": {{\\\"Authorization\\\": \\\"Bearer token-$c\\\"}}}}\"\n",
                counter.display(),
                counter.display()
            ),
        )
        .unwrap();

        let provider = provider(&dir, &script_path);
        let h1 = provider.headers().unwrap();
        assert_eq!(
            h1.get("Authorization").map(String::as_str),
            Some("Bearer token-1")
        );
        let h2 = provider.headers().unwrap();
        assert_eq!(
            h2.get("Authorization").map(String::as_str),
            Some("Bearer token-1")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provider_reruns_after_expiry() {
        let dir = temp_dir("rerun");
        let counter = dir.join("counter");
        std::fs::write(&counter, b"0").unwrap();
        let script_path = dir.join("provider.sh");
        // Expiration one second in the past → always refresh.
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nc=$(cat {})\nc=$((c+1))\necho $c > {}\nexpiry=$(date -u -d '-1 second' +%s)\nprintf '%s' \"{{\\\"expiration\\\": $expiry, \\\"headers\\\": {{\\\"Authorization\\\": \\\"Bearer token-$c\\\"}}}}\"\n",
                counter.display(),
                counter.display()
            ),
        )
        .unwrap();

        let provider = provider(&dir, &script_path);
        let h1 = provider.headers().unwrap();
        let h2 = provider.headers().unwrap();
        assert_ne!(
            h1.get("Authorization"),
            h2.get("Authorization"),
            "expired headers must trigger a re-run"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provider_failure_reports_exit_status() {
        let dir = temp_dir("fail");
        let script_path = dir.join("provider.sh");
        std::fs::write(
            &script_path,
            "#!/bin/sh\necho 'token expired, re-run login' >&2\nexit 3\n",
        )
        .unwrap();

        let provider = provider(&dir, &script_path);
        let err = provider.headers().unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("exited with"), "{msg}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provider_stdout_progress_does_not_break_parsing() {
        // Progress on stdout before the final JSON line must not be mistaken
        // for the document, and parsing must still succeed.
        let dir = temp_dir("progress");
        let script_path = dir.join("provider.sh");
        std::fs::write(
            &script_path,
            "#!/bin/sh\necho 'visit https://auth.example.invalid/device to login'\nprintf '%s' '{\"expiration\": 99999999999, \"headers\": {\"Authorization\": \"Bearer t\"}}'\n",
        )
        .unwrap();

        let provider = provider(&dir, &script_path);
        let headers = provider.headers().unwrap();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer t")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provider_missing_binary_errors() {
        let dir = temp_dir("missing");
        let provider = provider_in(&dir, vec!["potlatch-nonexistent-binary-xyz".into()]);
        let err = provider.headers().unwrap_err();
        assert!(format!("{err:#}").contains("failed to run auth_provider"));
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- provider working directory ---

    fn emit_document_script() -> String {
        "printf '%s' '{\"expiration\": 99999999999, \"headers\": {\"Authorization\": \"Bearer cwd-ok\"}}'".to_string()
    }

    /// Write an executable script into `dir` (the fake config dir).
    fn write_exec(dir: &std::path::Path, name: &str, body: &str) {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn bare_command_resolves_relative_to_working_dir() {
        // Auth-script-style config: the script sits alongside the config; a
        // bare command name must be found there without `./` or a path.
        let dir = temp_dir("bare-cmd");
        write_exec(&dir, "auth-tool.py", &emit_document_script());

        let provider = provider_in(&dir, vec!["auth-tool.py".into()]);
        let headers = provider.headers().unwrap();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer cwd-ok")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dot_slash_command_resolves_relative_to_working_dir() {
        let dir = temp_dir("dot-slash");
        write_exec(&dir, "auth-tool.py", &emit_document_script());

        let provider = provider_in(&dir, vec!["./auth-tool.py".into()]);
        let headers = provider.headers().unwrap();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer cwd-ok")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provider_runs_with_config_dir_as_cwd() {
        // The child must observe the working dir: it writes a marker file
        // with a relative path, which must land in the config dir.
        let dir = temp_dir("child-cwd");
        write_exec(
            &dir,
            "child.sh",
            "touch marker-from-child\nprintf '%s' '{\"expiration\": 99999999999, \"headers\": {}}'",
        );

        let provider = provider_in(&dir, vec!["./child.sh".into()]);
        provider.headers().unwrap();
        assert!(
            dir.join("marker-from-child").is_file(),
            "relative writes from the provider must land in the working dir"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn path_tools_still_resolve_with_working_dir_set() {
        // `sh` lives on PATH, not next to the config; the bare-name lookup
        // must fall back to PATH instead of failing.
        let dir = temp_dir("path-tool");
        let script_path = dir.join("provider.sh");
        std::fs::write(
            &script_path,
            format!("#!/bin/sh\n{}\n", emit_document_script()),
        )
        .unwrap();

        let provider = provider_in(&dir, vec!["sh".into(), script_path.display().to_string()]);
        let headers = provider.headers().unwrap();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer cwd-ok")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- cross-process single-flight ---

    #[test]
    fn concurrent_processes_run_provider_once() {
        // Spawn N real processes that each request headers through the same
        // shared state dir at the same time. The provider increments a
        // counter file per run; total runs must be 1 (single-flight), and
        // every process must end up with the same headers.
        let dir = temp_dir("singleflight");
        let counter = dir.join("counter");
        std::fs::write(&counter, b"0").unwrap();
        let script_path = dir.join("provider.sh");
        std::fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nc=$(cat {})\nc=$((c+1))\necho $c > {}\nprintf '%s' \"{{\\\"expiration\\\": 99999999999, \\\"headers\\\": {{\\\"Authorization\\\": \\\"Bearer shared-$c\\\"}}}}\"\n",
                counter.display(),
                counter.display()
            ),
        )
        .unwrap();

        // The runner binary: this test binary itself, re-invoked with an
        // env marker so it acts as a one-shot client.
        let exe = std::env::current_exe().unwrap();
        let state_dir = dir.join("state");
        let argv = format!("{}\n{}", script_path.display(), state_dir.display());

        let mut children = Vec::new();
        for _ in 0..8 {
            let exe = exe.clone();
            let argv = argv.clone();
            children.push(std::thread::spawn(move || {
                let output = std::process::Command::new(&exe)
                    .env("POTLATCH_AUTH_SINGLEFLIGHT_ARGV", argv)
                    .arg("--exact")
                    .arg("harness::auth_provider::tests::single_flight_client_main")
                    .arg("--nocapture")
                    .arg("--test-threads=1")
                    .output()
                    .expect("spawn client process");
                let combined = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                combined
            }));
        }

        let mut tokens = Vec::new();
        for child in children {
            let out = child.join().unwrap();
            let doc_line = out
                .lines()
                .rev()
                .find(|l| l.contains("TOKEN:"))
                .unwrap_or_else(|| panic!("client printed no token: {out}"));
            let token = doc_line
                .rsplit("TOKEN:")
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            tokens.push(token);
        }

        let runs: u32 = std::fs::read_to_string(&counter)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(runs, 1, "provider must run exactly once across processes");
        assert!(
            tokens.iter().all(|t| t == &tokens[0]),
            "all processes must observe the same headers: {tokens:?}"
        );
        assert_eq!(tokens[0], "Bearer shared-1");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_flight_client_main() {
        // Acts as the one-shot client process for
        // `concurrent_processes_run_provider_once`. Not a real test body.
        if let Ok(argv_raw) = std::env::var("POTLATCH_AUTH_SINGLEFLIGHT_ARGV") {
            let mut lines = argv_raw.splitn(2, '\n');
            let script = lines.next().unwrap();
            let state_dir = lines.next().unwrap();
            let provider = AuthProvider::with_state_dir(
                vec!["sh".into(), script.to_string()],
                None,
                PathBuf::from(state_dir),
            );
            let headers = provider.headers().unwrap();
            println!(
                "TOKEN:{}",
                headers.get("Authorization").cloned().unwrap_or_default()
            );
        }
    }
}
