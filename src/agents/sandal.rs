//! Sandal agent: maintains the platform's sandbox VMs.
//!
//! Bus-served: exposes a `shell` tool that OVERRIDES the built-in shell
//! tool while `[agent.sandal]` is configured — every harness session's
//! `shell` calls land here, and without the section the built-in host
//! shell stays. The override is the point: the shell tool is the one
//! tool that must never silently fall back to the host when a sandbox
//! deployment is configured.
//!
//! One VM per calling session, reused across that session's calls (see
//! [`VmPool`]): the first call boots the VM with the configured layers,
//! the session's working directory, and the configured persistent
//! shares, and every later call lands in the same VM. The platform pays
//! one boot per session, not one per call, and a session's installed
//! packages and written files survive to its next call. A VM nobody has
//! called on for the idle timeout is closed with its whole overlay —
//! nothing a session wrote survives it, and a fleet of sessions cannot
//! keep a fleet of VMs standing.
//!
//! Configured paths are relative to the config file's directory — the
//! directory that holds `potlatch.toml` — because the VMs run from the
//! agent worktree, a different working directory that must never
//! influence how a configured relative path resolves.

use std::collections::HashMap;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::bus::{AgentInbox, AgentToolDefinition};
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;
use crate::util::dies_with_parent;

const AGENT_NAME: &str = "sandal";
const SHELL_TOOL: &str = "shell";
const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
/// Default per-command timeout (seconds): the same default the built-in
/// shell tool carries.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Output kept per command (the head); the rest is counted, not stored.
/// The same cap the built-in shell tool carries.
const MAX_OUTPUT: usize = 50_000;
/// Poll slice for the output reader: how long a read may block before the
/// deadline is re-checked. A command that produces nothing blocks here,
/// never past the deadline.
const READ_SLICE_MS: i32 = 200;

/// The marker the session shell prints once it is up and reading: the
/// agent waits for it once per VM, because input written before the
/// guest finished booting is dropped, not buffered — a protocol line
/// written at spawn would be lost silently.
const READY_MARKER: &str = "__POTLATCH_VM_SESSION_READY__";
/// How long a booting VM may take to reach the ready marker (the guest
/// boots in well under a second; a layer-carrying VM takes a few).
const READY_DEADLINE: Duration = Duration::from_secs(30);
/// The session shell wrapper: print the ready marker, then eval one
/// protocol line per input line for the VM's whole life. Spawned as `sh
/// -c` (never `sh -i`): no interactive prompt ever lands in the stream,
/// and the wrapper is the only thing the guest shell ever runs directly.
const SESSION_WRAPPER: &str = "echo __POTLATCH_VM_SESSION_READY__; \
while IFS= read -r line; do eval \"$line\"; done";

/// Default VM memory cap (MB) when layers are configured without
/// `memory`. The cap no longer sizes the footprint: the sandbox boots a
/// small RAM floor and hot-adds the rest on demand, so idle cost is flat
/// and a workload only brings online what it actually touches. The cap
/// bounds runaway allocations — 2048 covers the measured go toolchain
/// (peaks at ~650MB) with 3x headroom; heavier toolchains (rustc, CGO,
/// large links) should set `memory` explicitly.
const DEFAULT_MEMORY_MB: u64 = 2048;

/// Default writable overlay disk (MB) when layers are configured without
/// `disk`: build caches and package downloads land on the overlay, whose
/// size without this is the layer data plus 16 MB — real builds fill
/// that within minutes.
const DEFAULT_DISK_MB: u64 = 16384;

/// Default idle timeout (seconds) for a session VM nobody has called the
/// shell tool on. A long-idle VM is closed (killed with its whole
/// overlay — nothing a session wrote survives it) so a fleet of agent
/// sessions cannot keep a fleet of VMs standing.
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 30 * 60;

/// The `shell` tool this agent publishes — the override for the built-in
/// shell tool while `[agent.sandal]` is configured.
const SHELL_DESCRIPTION: &str = "Run a shell command inside the session's sandbox VM. You are already in the working directory (cwd) — no need to `cd` into it. The VM is reused across this session's calls: packages installed by an earlier call are still installed, and files written by an earlier call are still there. Returns the command's output and exit code. Foreground commands have a timeout (default 120s); on timeout the whole VM is closed and the session's next call boots a fresh one — anything longer than a minute or two belongs in the background: set background: true, get a job id back immediately, and poll it with job_id (or kill it with job_id + kill: true). Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `write` or `edit` instead.";

/// `[agent.sandal]` settings as configured: paths still relative to the
/// config file, share specs unresolved, sizing unset where absent.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandalAgentSettings {
    /// Toolchain images the session VMs boot with, as configured
    /// (relative paths resolve against the config directory).
    #[serde(default)]
    pub layers: Vec<String>,
    /// `host:guest` share specs — persistent host directories mapped
    /// into the VM (the go module/build caches: each VM boots fresh, so
    /// without a host-backed cache every build starts cold).
    #[serde(default)]
    pub shared: Vec<String>,
    pub memory: Option<u64>,
    pub disk: Option<u64>,
    /// Idle timeout (seconds) for a session VM whose last shell call is
    /// older than this; see [`DEFAULT_IDLE_TIMEOUT_SECS`].
    pub idle_timeout_secs: Option<u64>,
}

/// The resolved settings the agent runs VMs with: absolute layer paths,
/// resolved share specs, VM sizing, and the idle timeout.
#[derive(Debug, Clone)]
pub struct ResolvedSettings {
    pub layers: Vec<PathBuf>,
    /// `host:guest` share specs, resolved on both sides (see
    /// [`SandalAgentSettings::shared`]).
    pub shares: Vec<String>,
    pub memory_mb: Option<u64>,
    pub disk_mb: Option<u64>,
    pub idle_timeout: Duration,
}

impl SandalAgentSettings {
    /// Resolve the settings into what the agent runs VMs with. Layer
    /// paths resolve against the config file's directory (see the module
    /// docs); a configured layer that does not exist is an error, not a
    /// warning — a silently dropped layer leaves every command that needs
    /// the toolchain failing inside the VM with no hint why. Shares
    /// resolve per side (see [`resolve_share`]) without an existence
    /// check — a share whose host side does not exist yet (a cache
    /// directory go has not created) is dropped with a warning at spawn
    /// time, and appears once the directory does.
    ///
    /// Sizing applies only when layers are configured: a bare VM (no
    /// `layers`) runs trivial commands fine at the sandbox's own
    /// defaults, and a toolchain VM gets `memory`/`disk` defaults sized
    /// for real builds (see [`DEFAULT_MEMORY_MB`]/[`DEFAULT_DISK_MB`]).
    pub fn resolve(&self, config_dir: Option<&Path>) -> Result<ResolvedSettings> {
        let config_dir = config_dir
            .context("cannot resolve [agent.sandal] paths: no config directory (synthetic config)")?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for configured in &self.layers {
            let path = config_dir.join(configured);
            let path = path
                .canonicalize()
                .with_context(|| format!("[agent.sandal] layer not found: {configured}"))?;
            ensure!(path.is_file(), "[agent.sandal] layer is not a file: {configured}");
            layers.push(path);
        }
        let mut shares = Vec::with_capacity(self.shared.len());
        for spec in &self.shared {
            shares.push(resolve_share(spec, config_dir)?);
        }
        let sized = !layers.is_empty();
        Ok(ResolvedSettings {
            layers,
            shares,
            memory_mb: sized.then(|| self.memory.unwrap_or(DEFAULT_MEMORY_MB)),
            disk_mb: sized.then(|| self.disk.unwrap_or(DEFAULT_DISK_MB)),
            idle_timeout: Duration::from_secs(
                self.idle_timeout_secs.unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
            ),
        })
    }
}

/// Resolve one `host:guest` share spec. `~` expands per side — the host
/// home on the host side, `/root` on the guest side (the sandbox guest
/// always runs as root) — so `~/go/pkg:~/go/pkg` maps the host's go
/// module cache to the guest's default cache path. Relative host paths
/// resolve against the config dir (like layers); the guest side stays
/// literal.
fn resolve_share(spec: &str, config_dir: &Path) -> Result<String> {
    let (host, guest) = spec
        .split_once(':')
        .with_context(|| format!("invalid [agent.sandal] share {spec:?} (expected host:guest)"))?;
    let host = expand_home(host, &crate::paths::home_dir());
    let host = if host.is_absolute() {
        host
    } else {
        config_dir.join(host)
    };
    // Resolve to the real path when it exists (so the spawn filters the
    // directory that is actually there); keep the expanded path otherwise.
    let host = host.canonicalize().unwrap_or(host);
    let guest = expand_home(guest, Path::new("/root"));
    Ok(format!("{}:{}", host.display(), guest.display()))
}

/// Expand a leading `~` to `home`; everything else stays as written.
fn expand_home(path: &str, home: &Path) -> PathBuf {
    if path == "~" {
        home.to_path_buf()
    } else if let Some(rest) = path.strip_prefix("~/") {
        home.join(rest)
    } else {
        PathBuf::from(path)
    }
}

pub struct SandalAgent {
    runtime: AgentRuntime,
    inbox: AgentInbox,
    pool: Arc<VmPool>,
}

impl CoreAgent for SandalAgent {
    type Settings = SandalAgentSettings;
    const FIXED_INSTANCES: Option<usize> = Some(1);

    fn name() -> &'static str {
        AGENT_NAME
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: REQUEST_TASK,
            interval: Duration::ZERO,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 0,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        ensure!(task_id == REQUEST_TASK, "unknown sandal task {task_id:?}");
        // Idle VMs first: a long-idle session VM is dead weight nobody
        // will ever call again — close it with its whole overlay.
        self.pool.reap_idle();
        if let Some(mut request) = self.inbox.recv_timeout(INBOX_WAIT)? {
            // The harness proxy always sets the calling session's id;
            // without it there is nothing to key the VM on.
            let Some(session_id) = request.session_id.as_deref() else {
                request.respond(Err(anyhow::anyhow!(
                    "sandal shell call carries no session id"
                )));
                return Ok(());
            };
            // The VM boots HERE, on this long-lived thread, never on a
            // per-call thread: PR_SET_PDEATHSIG kills a child when its
            // parent *thread* exits, and a per-call thread exits after
            // every command — a VM spawned there would die with each
            // delivered command. The boot blocks this loop for well under
            // a second (measured); the command itself may run for its
            // whole timeout, so it runs on its own thread below —
            // same-session calls serialize on the VM's own state lock,
            // cross-session calls run concurrently.
            let cwd = request
                .payload
                .get("_session_cwd")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Err(error) = self.pool.ensure(session_id, &cwd) {
                request
                    .respond(Err(error.context("the session's sandbox VM cannot boot")));
                return Ok(());
            }
            let pool = Arc::clone(&self.pool);
            let session_id = session_id.to_string();
            std::thread::spawn(move || {
                let result = pool
                    .execute(&session_id, &request.payload)
                    .map(Value::String);
                request.respond(result);
            });
        }
        Ok(())
    }

    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
        let bus = ctx
            .workflow
            .bus
            .as_ref()
            .context("sandal agent requires the cross-agent bus")?;
        let settings = ctx
            .settings
            .resolve(ctx.workflow.config.config_dir())
            .context("[agent.sandal] settings are not usable")?;
        let bin = resolve_sandal_bin()
            .context("[agent.sandal] requires the sandal sandbox binary")?;
        let pool = Arc::new(VmPool::new(settings, bin));
        let inbox = bus.register(
            AGENT_NAME,
            vec![AgentToolDefinition {
                name: SHELL_TOOL.to_string(),
                description: SHELL_DESCRIPTION.to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute inside the session's sandbox VM. Required unless polling or killing a background job (job_id)."
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Timeout in seconds for foreground commands. Default: 120. On timeout the whole VM is closed and the session's next call boots a fresh one. Ignored for background jobs (they run until exit or kill)."
                        },
                        "background": {
                            "type": "boolean",
                            "description": "If true, start the command detached inside the session's VM and return a job id immediately instead of waiting — a call may not outlast the transport anyway. Poll with job_id at any time later in the session; the job's output and exit code are kept inside the VM and die with it. Or terminate it with job_id + kill: true."
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Poll or kill a background job. Returns the job's newest output and its status when it has one. Ignored unless polling or killing."
                        },
                        "kill": {
                            "type": "boolean",
                            "description": "When true with job_id, terminate the job best-effort: the job's own shell is killed; processes it spawned are orphans."
                        }
                    },
                    "required": []
                }),
                operation: SHELL_TOOL.to_string(),
            }]
        )?;
        info!(
            "sandal agent ready ({} layers, {} persistent shares)",
            pool.settings.layers.len(),
            pool.settings.shares.len()
        );
        Ok(Self {
            runtime: ctx.runtime.clone(),
            inbox,
            pool,
        })
    }
}

impl Drop for SandalAgent {
    fn drop(&mut self) {
        // The VMs die with the agent: nothing outlives the platform, and
        // PR_SET_PDEATHSIG in each child is the belt this is the
        // suspenders for.
        self.pool.close_all();
    }
}

/// Resolve the sandbox binary: `POTLATCH_SANDBOX_BIN` overrides, else a
/// `sandal` on PATH. Resolved once at build so a missing binary is a
/// startup error, not a per-call one.
fn resolve_sandal_bin() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("POTLATCH_SANDBOX_BIN") {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "POTLATCH_SANDBOX_BIN is not a file: {}",
            path.display()
        );
        return Ok(path);
    }
    let path = std::env::var_os("PATH").context("no PATH to search for the sandal binary")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("sandal");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "no `sandal` on PATH — install sandal (github.com/maralla/sandal) \
         or set POTLATCH_SANDBOX_BIN"
    )
}

/// One live session VM. `cwd` is pinned at spawn and readable without
/// the state lock; everything mutable lives behind one state lock that a
/// command holds for its whole run.
struct VmEntry {
    /// The working directory this VM was booted with — pinned on the
    /// session's first call and never renegotiated: every later call
    /// lands in the same VM, so a session that somehow presents a
    /// different directory gets an error, not a second VM.
    cwd: String,
    /// The whole mutable state behind one lock: held for an entire
    /// command, so same-session commands serialize while cross-session
    /// commands run concurrently; only ever `try_lock`ed by the idle
    /// reaper and close-all, where a held lock means a running command
    /// and they skip it.
    state: Mutex<VmState>,
}

/// The mutable state of one live session VM.
struct VmState {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    /// Set once the session shell printed the ready marker. Input written
    /// before the guest finished booting is dropped, so the first call
    /// waits for the marker; later calls skip the wait.
    ready: bool,
    /// Last shell call on this VM — updated at each command's start AND
    /// end: a long-running command is neither reaped mid-run (the reaper
    /// skips a held lock anyway) nor left stale the moment it finishes.
    last_used: Instant,
}

/// The session VM pool: at most one VM per calling session, reused across
/// that session's calls. `execute` boots a session's VM on its first
/// call and reuses it after; `reap_idle` closes VMs whose last call is
/// older than the configured idle timeout; `close_all` closes everything
/// (agent shutdown).
struct VmPool {
    settings: ResolvedSettings,
    bin: PathBuf,
    entries: Mutex<HashMap<String, Arc<VmEntry>>>,
}

impl VmPool {
    fn new(settings: ResolvedSettings, bin: PathBuf) -> Self {
        Self {
            settings,
            bin,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Boot a session's VM: the configured layers and sizing, the session's
    /// working directory, and the configured persistent shares whose host
    /// side exists (a share for a cache directory that has not been
    /// created yet is dropped with a warning, and appears once it has).
    fn spawn(&self, cwd: &str) -> Result<VmEntry> {
        let mut cmd = Command::new(&self.bin);
        cmd.arg("run");
        if let Some(memory_mb) = self.settings.memory_mb {
            cmd.arg("-m").arg(memory_mb.to_string());
        }
        if let Some(disk_mb) = self.settings.disk_mb {
            cmd.arg("--disk-size").arg(disk_mb.to_string());
        }
        for layer in &self.settings.layers {
            cmd.arg("--layer").arg(layer);
        }
        // The working directory rides the same path on both sides so
        // guest paths resolve exactly as the session's paths do.
        cmd.arg("--share").arg(format!("{cwd}:{cwd}"));
        for spec in &self.settings.shares {
            let (host, _) = spec
                .split_once(':')
                .context("resolved share without a separator")?;
            if Path::new(host).is_dir() {
                cmd.arg("--share").arg(spec);
            } else {
                warn!(
                    target: "potlatch::sandal",
                    "dropping share {spec:?}: {} is not a directory yet",
                    host
                );
            }
        }
        cmd.arg("--").arg("sh").arg("-c").arg(SESSION_WRAPPER);
        // stderr is inherited, never piped: the boot log lands in this
        // process's own log (where it is greppable), and an unread pipe
        // could never fill and stall the VM mid-boot.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        dies_with_parent(&mut cmd);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", self.bin.display()))?;
        // Safety: take() leaves the pipes Some-less in `child`; the entry
        // owns them for the VM's life and closes them by dropping.
        let stdin = child.stdin.take().context("sandbox stdin not piped")?;
        let stdout = child.stdout.take().context("sandbox stdout not piped")?;
        Ok(VmEntry {
            cwd: cwd.to_string(),
            state: Mutex::new(VmState {
                child,
                stdin,
                stdout,
                ready: false,
                last_used: Instant::now(),
            }),
        })
    }

    /// Run one shell command in the session's VM. Boots the VM on the
    /// session's first call and reuses it after; a VM that died (or was
    /// closed by a timeout) is removed so the session's next call boots a
    /// fresh one.
    /// Boot a session's VM if it has none. Run-loop only: the spawn must
    /// happen on a long-lived thread (see the run loop), never on a
    /// per-call thread.
    fn ensure(&self, session_id: &str, cwd: &str) -> Result<()> {
        let mut entries = lock(&self.entries);
        match entries.get(session_id) {
            Some(_) => Ok(()),
            None => {
                let entry = Arc::new(self.spawn(cwd)?);
                entries.insert(session_id.to_string(), entry);
                Ok(())
            }
        }
    }

    fn execute(&self, session_id: &str, payload: &Value) -> Result<String> {
        // Polling or killing a background job takes precedence — `command`
        // is ignored in these modes.
        let job_id = payload
            .get("job_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string);
        if let Some(job) = job_id {
            return self.execute_job_control(session_id, payload, &job);
        }
        let command = payload
            .get("command")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .context("missing 'command' argument")?;
        // The harness proxy pins the session's working directory as call
        // metadata (like the session id itself): the first call's is what
        // the VM boots with, and a later call presenting a different one
        // is an error, not a second VM.
        let cwd = payload
            .get("_session_cwd")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .context("the shell call carries no session working directory")?;
        let timeout_secs = payload
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let timeout = Duration::from_secs(timeout_secs.max(1));
        // At most one VM per session: reuse the live one, or boot the
        // first. The map lock covers only the lookup and the insert —
        // never a command — so one session's boot never blocks another
        // session's lookup.
        let entry = {
            let mut entries = lock(&self.entries);
            match entries.get(session_id) {
                Some(entry) => {
                    if entry.cwd != cwd {
                        return Err(anyhow::anyhow!(
                            "the session's sandbox VM is bound to its first \
                             working directory"
                        ));
                    }
                    Arc::clone(entry)
                }
                // The run loop boots session VMs; a missing one here was
                // reaped (or died) between the run loop's check and this
                // call. The caller retries, and the run loop boots it.
                None => {
                    return Err(anyhow::anyhow!(
                        "the session's sandbox VM is not running; retry"
                    ));
                }
            }
        };
        // This session's commands serialize on the VM's own state lock;
        // the Arc keeps the entry alive for the command even if the map
        // drops it concurrently (a reap or a close-all).
        let background = payload.get("background").and_then(Value::as_bool).unwrap_or(false);
        let job = background.then(job_tag);
        let mut state = lock_state(&entry.state);
        state.last_used = Instant::now();
        let (sentinel, line) = match &job {
            Some(job) => job_start_line(&entry.cwd, command, job),
            None => foreground_line(&entry.cwd, command),
        };
        let outcome = run_line(&mut state, line.as_str(), sentinel.as_str(), timeout);
        // The command's end is the last use: without this a finished
        // long-running command leaves the VM stale, and the next reaping
        // pass closes it the moment it finishes.
        state.last_used = Instant::now();
        drop(state);
        // A dead or closed VM never serves another call: remove it so the
        // next call boots a fresh one instead of hitting a corpse.
        if outcome.vm_dead() {
            lock(&self.entries).remove(session_id);
        }
        outcome.into_result().map(|raw| {
            // A background start reports its job id (the start line's own
            // status is 0); anything else is a refusal, formatted like a
            // foreground run.
            match (&job, raw.code) {
                (Some(job), 0) => format!(
                    "Background job started: {job}\ncommand: {command}\nPoll with job_id \
                     later in this session; the job's output and exit code are kept inside \
                     the VM and die with it."
                ),
                _ => format_result(&raw.out, raw.code, raw.dropped),
            }
        })
    }

    /// Poll or kill one background job. The job's state is the VM's (its
    /// output file and its own exit code); this only reads what it left.
    fn execute_job_control(
        &self,
        session_id: &str,
        payload: &Value,
        job: &str,
    ) -> Result<String> {
        // The job tag is ours (hex from run_job_start); never let a caller
        // inject one into a shell line.
        ensure!(
            job.len() <= 64 && job.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "invalid job id {job:?}"
        );
        let kill = payload.get("kill").and_then(Value::as_bool).unwrap_or(false);
        let cwd = payload
            .get("_session_cwd")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .context("the shell call carries no session working directory")?;
        let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS.max(1));
        let entry = {
            let mut entries = lock(&self.entries);
            match entries.get(session_id) {
                Some(entry) => {
                    if entry.cwd != cwd {
                        return Err(anyhow::anyhow!(
                            "the session's sandbox VM is bound to its first working directory"
                        ));
                    }
                    Arc::clone(entry)
                }
                None => {
                return Err(anyhow::anyhow!(
                    "the session's sandbox VM is not running; retry"
                ));
            }
            }
        };
        let mut state = lock_state(&entry.state);
        state.last_used = Instant::now();
        let (sentinel, line) = if kill {
            job_kill_line(job)
        } else {
            job_poll_line(job)
        };
        let outcome = run_line(&mut state, line.as_str(), sentinel.as_str(), timeout);
        state.last_used = Instant::now();
        drop(state);
        if outcome.vm_dead() {
            lock(&self.entries).remove(session_id);
        }
    let raw = outcome.into_result()?;
    if kill {
        return Ok(format!("Job {job} killed (best-effort: its own shell only)"));
    }
    // The job's own exit code rides its output file.
    let output = &raw.out[..];
    let marker = b"JOB-EXIT:";
    let exit = find_subslice(output, marker, 0).and_then(|at| {
        let rest = &output[at + marker.len()..];
        let end = rest.iter().position(|b| !b.is_ascii_digit()).unwrap_or(rest.len());
        std::str::from_utf8(&rest[..end])
            .ok()
            .and_then(|digits| digits.parse::<i32>().ok())
    });
    let mut result = match exit {
        Some(code) => format!("exit code: {code}\n\n"),
        None => "exit code: (still running)\n\n".to_string(),
    };
    // Strip the JOB-EXIT line: the status is the header, not output.
    let head_end = find_subslice(output, b"\nJOB-EXIT:", 0).unwrap_or(output.len());
    let output_part = String::from_utf8_lossy(&output[..head_end]);
    if output_part.trim().is_empty() {
        result.push_str("(no output yet)");
    } else {
        result.push_str(&output_part);
    }
    Ok(result)
    }


    /// Close VMs whose last shell call is older than the configured idle
    /// timeout. Each close kills the VM with its whole overlay — nothing
    /// a session wrote survives it.
    fn reap_idle(&self) {
        let mut entries = lock(&self.entries);
        let idle = self.settings.idle_timeout;
        entries.retain(|session_id, entry| {
            // A held state lock is a running command: not idle, whatever
            // the clock says — never killed mid-command.
            let Ok(mut state) = entry.state.try_lock() else {
                return true;
            };
            if state.last_used.elapsed() < idle {
                return true;
            }
            info!(
                target: "potlatch::sandal",
                "closing idle session VM (session {session_id}, idle {}s)",
                state.last_used.elapsed().as_secs()
            );
            close_state(&mut state);
            false
        });
    }

    /// Close every VM (agent shutdown).
    fn close_all(&self) {
        let mut entries = lock(&self.entries);
        for (session_id, entry) in entries.iter_mut() {
            // A running command closes its own VM by its own deadline
            // (and PR_SET_PDEATHSIG closes the rest when this process
            // dies); shutdown never blocks on one.
            if let Ok(mut state) = entry.state.try_lock() {
                info!(target: "potlatch::sandal", "closing session VM (session {session_id})");
                close_state(&mut state);
            } else {
                info!(
                    target: "potlatch::sandal",
                    "session VM busy with a running command (session {session_id}); \
                     it closes itself by its own deadline"
                );
            }
        }
        entries.clear();
    }
}

/// A fresh job tag per start: it rides the job's command line (so a kill
/// can find it) and names its output file.
fn job_tag() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("potlatch-job-{nanos:x}")
}

/// Close one VM: kill the child and drop its pipes. The overlay dies with
/// the process; nothing is persisted.
fn close_state(state: &mut VmState) {
    let _ = state.child.kill();
    let _ = state.child.wait();
}

/// Lock the pool map, poisoning and all: a panicked command thread must
/// not brick the pool.
fn lock(
    map: &Mutex<HashMap<String, Arc<VmEntry>>>,
) -> std::sync::MutexGuard<'_, HashMap<String, Arc<VmEntry>>> {
    map.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lock one VM's state, poisoning and all: a panicked earlier command
/// leaves the VM in whatever state it died in, and the next command's
/// read EOFs into the dead-VM path — a fresh boot, not a bricked
/// session.
fn lock_state(state: &Mutex<VmState>) -> std::sync::MutexGuard<'_, VmState> {
    state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One command's outcome inside a VM: the formatted result, or the
/// failure that closed the VM.
enum Outcome {
    /// The raw run: the kept output, the sentinel's exit code, the dropped
    /// count. The caller formats (foreground) or parses (job control).
    Done(Result<RawRun>),
    /// The VM is dead or was closed; the entry must be removed. The
    /// payload is the error, never a run.
    Closed(Result<String>),
}

impl Outcome {
    fn vm_dead(&self) -> bool {
        matches!(self, Outcome::Closed(_))
    }

    fn into_result(self) -> Result<RawRun> {
        match self {
            Outcome::Done(result) => result,
            Outcome::Closed(result) => result.map(|_| RawRun {
                out: Vec::new(),
                code: -1,
                dropped: 0,
            }),
        }
    }
}

/// Run one command in a live VM: write the protocol line, read output
/// until the per-call sentinel, and report the exit code it carries.
/// EOF before the sentinel means the VM died; a deadline passing means
/// the command hung — the VM is closed either way.


/// The per-call sentinel: a fresh marker each call, so output echoed by an
/// earlier command can never terminate a later one.
fn fresh_sentinel() -> String {
    static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("__POTLATCH_VM_EXIT_{call:x}__")
}

/// The foreground protocol line: the `cd` guard (the VM's working
/// directory or exit 127), the base64 command (multi-line commands
/// survive verbatim) in a subshell (an `exit` inside it exits the
/// subshell, not the session shell), and the sentinel with the command's
/// exit code.
fn foreground_line(cwd: &str, command: &str) -> (String, String) {
    let sentinel = fresh_sentinel();
    let encoded = base64::engine::general_purpose::STANDARD.encode(command);
    let cwd = cwd.replace('\'', "'\\''");
    let line = format!(
        "cd '{cwd}' 2>/dev/null && {{ ( eval \"$(printf %s {encoded} | base64 -d)\" ); \
         printf '\\n{sentinel}:%d\\n' \"$?\"; }} || printf '\\n{sentinel}:127\\n'\n"
    );
    (sentinel, line)
}

/// The background-start protocol line: the job runs detached in the VM
/// with its whole output (and its own `JOB-EXIT:` code) going to a
/// job-tagged file, and the sentinel carries only the backgrounding's
/// own status. The job tag rides the job's command line, so a kill can
/// find it with pkill.
fn job_start_line(cwd: &str, command: &str, job: &str) -> (String, String) {
    let sentinel = fresh_sentinel();
    let encoded = base64::engine::general_purpose::STANDARD.encode(command);
    let cwd = cwd.replace('\'', "'\\''");
    let out = job_output_path(job);
    let line = format!(
        "cd '{cwd}' 2>/dev/null && {{ ( eval \": {job}; $(printf %s {encoded} | base64 -d)\"; \
         printf 'JOB-EXIT:%d\\n' \"$?\" >>{out} ) >{out} 2>&1 & \
         printf '\\n{sentinel}:0\\n'; }} || printf '\\n{sentinel}:127\\n'\n"
    );
    (sentinel, line)
}

/// The job-poll protocol line: the job's own output tail (the newest
/// bytes are what a poll must surface) plus its own exit code when it
/// has one.
fn job_poll_line(job: &str) -> (String, String) {
    let sentinel = fresh_sentinel();
    let out = job_output_path(job);
    let line = format!(
        "tail -c 50000 {out} 2>/dev/null; printf '\\n{sentinel}:0\\n'\n"
    );
    (sentinel, line)
}

/// The job-kill protocol line: best-effort — the job tag rides the job's
/// command line, so pkill finds the job's shell; the children it spawned
/// are orphans pkill cannot reach.
fn job_kill_line(job: &str) -> (String, String) {
    let sentinel = fresh_sentinel();
    let out = job_output_path(job);
    let line = format!(
        "pkill -9 -f '{job}' 2>/dev/null; rm -f {out}; \
         printf '\\n{sentinel}:0\\n'\n"
    );
    (sentinel, line)
}

/// Where a job's output (and its own exit code) lives: inside the VM,
/// never on a share — the job state is the VM's, and it dies with the VM.
fn job_output_path(job: &str) -> String {
    format!("/tmp/{job}.out")
}

/// Run one protocol line in a live VM: write it, read output until the
/// per-call sentinel, and report the exit code it carries. EOF before the
/// sentinel means the VM died; a deadline passing means the line hung —
/// the VM is closed either way.
fn run_line(state: &mut VmState, line: &str, sentinel: &str, timeout: Duration) -> Outcome {
    if !state.ready {
        match wait_ready(state) {
            Ok(()) => state.ready = true,
            Err(error) => return Outcome::Closed(Err(error)),
        }
    }
    let deadline = Instant::now() + timeout;
    // The write retries refusals — but a refusal that persists is almost
    // always a dead VM (the memory cap OOM-killed the guest, the wrapper
    // died with it, and the write hits a closed pipe), not a busy
    // transport. Detect the death fast: the entry is removed and the
    // session's next call boots a fresh VM in about a second, instead of
    // burning the whole timeout against a corpse.
    match write_line_retrying(&mut state.stdin, &mut state.child, &line, deadline) {
        Ok(()) => {}
        Err(WriteError::Dead(status)) => {
            close_state(state);
            let message = format!(
                "the session's sandbox VM died before the command ran (status {:?}); \
                 the session's next call boots a fresh one",
                status
            );
            return Outcome::Closed(Err(anyhow::anyhow!(message)));
        }
        Err(WriteError::Refused(error)) => {
            return Outcome::Closed(Err(anyhow::anyhow!(
                "the session's sandbox VM refused the command for the whole timeout: {error}"
            )));
        }
    }
    let mut out: Vec<u8> = Vec::new();
    let mut total = 0usize;
    // The carry keeps the bytes that could straddle a read boundary of the
    // sentinel search.
    let mut carry: Vec<u8> = Vec::new();
    let fd = state.stdout.as_raw_fd();
    let mut chunk = [0u8; 8192];
    loop {
        if Instant::now() >= deadline {
            close_state(state);
            return Outcome::Closed(Err(anyhow::anyhow!(
                "timed out after {}s; the whole VM was closed — the session's next \
                 call boots a fresh one",
                timeout.as_secs()
            )));
        }
        let mut fds = [libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        }];
        // Safety: poll(2) on the pipe's own fd; the array is local.
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), 1, READ_SLICE_MS) };
        if ready < 0 {
            close_state(state);
            return Outcome::Closed(Err(anyhow::anyhow!("failed to poll the sandbox VM's output")));
        }
        if ready == 0 {
            continue;
        }
        // Safety: read(2) into a local buffer; the length is the buffer's.
        let n = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n < 0 {
            close_state(state);
            return Outcome::Closed(Err(anyhow::anyhow!("failed to read the sandbox VM's output")));
        }
        if n == 0 {
            // EOF: the VM died mid-command. Report the status; the why is
            // in this process's own log (the child's stderr is inherited).
            let status = state
                .child
                .wait()
                .ok()
                .and_then(|status| status.code())
                .unwrap_or(-1);
            return Outcome::Closed(Err(anyhow::anyhow!(
                "the session's sandbox VM exited during the command (status {status}); \
                 see the platform log for the sandbox's own output"
            )));
        }
        let data = &chunk[..n as usize];
        // Search for the sentinel before the cap decides anything: the
        // sentinel is the last bytes in the stream, but it must be found
        // wherever it lands.
        carry.extend_from_slice(data);
        if let Some((at, code)) = find_sentinel(&carry, sentinel) {
            // The command finished before this read: its whole output and
            // the sentinel arrived together. The output is everything
            // before the marker (the marker's own leading newline
            // stripped), capped like any other read.
            let head_end = if at > 0 && carry[at - 1] == b'\n' {
                at - 1
            } else {
                at
            };
            let head = &carry[..head_end];
            let room = MAX_OUTPUT.saturating_sub(out.len());
            out.extend_from_slice(&head[..room.min(head.len())]);
            total += head.len();
            let dropped = total.saturating_sub(out.len());
            return Outcome::Done(Ok(RawRun {
                out: std::mem::take(&mut out),
                code,
                dropped,
            }));
        }
        // Keep only the tail that could straddle the next boundary.
        let keep = carry.len().saturating_sub(sentinel.len() + 16);
        carry.drain(..keep);
        // The cap: keep the head, count the rest.
        let room = MAX_OUTPUT.saturating_sub(out.len());
        if room > 0 {
            let take = room.min(data.len());
            out.extend_from_slice(&data[..take]);
        }
        total += data.len();
    }
}

/// Wait for the session shell's ready marker. Everything before it is
/// boot noise — read past, never kept. EOF means the VM died before its
/// session shell came up; a deadline passing means it never did.
fn wait_ready(state: &mut VmState) -> Result<()> {
    let deadline = Instant::now() + READY_DEADLINE;
    // The rolling tail: only the bytes that could hold the marker are
    // kept, never the noise before them.
    let mut tail: Vec<u8> = Vec::new();
    let fd = state.stdout.as_raw_fd();
    let mut chunk = [0u8; 8192];
    loop {
        if Instant::now() >= deadline {
            bail!("the session's sandbox VM never became ready; see the platform log for the sandbox's own output");
        }
        let mut fds = [libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        }];
        // Safety: poll(2) on the pipe's own fd; the array is local.
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), 1, READ_SLICE_MS) };
        if ready < 0 {
            bail!("failed to poll the sandbox VM's output");
        }
        if ready == 0 {
            continue;
        }
        // Safety: read(2) into a local buffer; the length is the buffer's.
        let n = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n < 0 {
            bail!("failed to read the sandbox VM's output");
        }
        if n == 0 {
            let status = state
                .child
                .wait()
                .ok()
                .and_then(|status| status.code())
                .unwrap_or(-1);
            bail!(
                "the session's sandbox VM exited before its session shell was ready \
                 (status {status}); see the platform log for the sandbox's own output"
            );
        }
        let data = &chunk[..n as usize];
        tail.extend_from_slice(data);
        if find_subslice(&tail, READY_MARKER.as_bytes(), 0).is_some() {
            return Ok(());
        }
        let keep = tail.len().saturating_sub(READY_MARKER.len() + 16);
        tail.drain(..keep);
    }
}

/// One finished protocol line's raw result: the kept output head, the
/// exit code the sentinel carried, and how much the cap dropped. The
/// foreground path formats this; the job paths parse it.
struct RawRun {
    out: Vec<u8>,
    code: i32,
    dropped: usize,
}

/// Why a protocol write did not land.
enum WriteError {
    /// The VM's process exited — the write hit a closed pipe. The entry
    /// must be closed and removed so the next call boots fresh.
    Dead(Option<i32>),
    /// The write was still being refused at the deadline.
    Refused(std::io::Error),
}

/// Write one protocol line to the VM, retrying refusals until the
/// deadline. A refused write is usually a dead VM (see the caller): the
/// child is checked on every refusal, so a death costs one refusal
/// round instead of the whole timeout. A live refusal — the transport
/// dropping input the busy guest refused, asking for a re-send — is
/// ridden out with one warning per episode, never one per round.
fn write_line_retrying(
    stdin: &mut ChildStdin,
    child: &mut Child,
    line: &str,
    deadline: Instant,
) -> Result<(), WriteError> {
    use std::io::Write;
    let mut warned = false;
    loop {
        match stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()) {
            Ok(()) => return Ok(()),
            Err(error) => {
                // A dead child explains a closed pipe; check it before
                // anything else so a death is never retried.
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(WriteError::Dead(status.code()));
                }
                if Instant::now() >= deadline {
                    return Err(WriteError::Refused(error));
                }
                if !warned {
                    warned = true;
                    warn!(
                        target: "potlatch::sandal",
                        "the sandbox refused a protocol write ({}); retrying until the \
                         command deadline — a persistent refusal is a dead VM",
                        error
                    );
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    }
}

/// Find `<sentinel>:<code>` in `data` and return where the marker
/// starts and the code it carries.
fn find_sentinel(data: &[u8], sentinel: &str) -> Option<(usize, i32)> {
    let marker = format!("{sentinel}:");
    let mut from = 0;
    while let Some(at) = find_subslice(data, marker.as_bytes(), from) {
        let rest = &data[at + marker.len()..];
        let end = rest
            .iter()
            .position(|b| !b.is_ascii_digit())
            .unwrap_or(rest.len());
        if end > 0 {
            return std::str::from_utf8(&rest[..end])
                .ok()
                .and_then(|digits| digits.parse::<i32>().ok())
                .map(|code| (at, code));
        }
        from = at + 1;
    }
    None
}

/// Find `needle` in `hay` at or after `from`.
fn find_subslice(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|at| &hay[*at..*at + needle.len()] == needle)
}

/// Format one command's result: the exit code, the kept output head, and
/// the note about what the cap dropped.
fn format_result(out: &[u8], code: i32, dropped: usize) -> String {
    let mut text = format!("exit code: {code}\n\n");
    if out.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str(&String::from_utf8_lossy(out));
    }
    if dropped > 0 {
        text.push_str(&format!(
            "\n\n[...output truncated, {dropped} total bytes dropped...]"
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::tools::test_util::unique_test_dir;
    use std::fs;

    /// A fake sandbox binary: a shell script that evals the protocol lines
    /// verbatim, so the whole protocol round-trips without a VM. The
    /// commands run host-side in one long-lived process — exactly what the
    /// reuse tests need to observe.
    fn fake_sandal(dir: &Path) -> PathBuf {
        let path = dir.join("fake-sandal");
        fs::write(
            &path,
            format!("#!/bin/sh\necho {READY_MARKER}\nwhile IFS= read -r line; do eval \"$line\"; done\n"),
        )
        .unwrap();
        fs::set_permissions(
            &path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();
        path
    }

    /// A pool with a fake binary, no layers (no sizing flags), and the
    /// given idle timeout.
    fn pool(dir: &Path, idle_timeout_secs: u64) -> Arc<VmPool> {
        Arc::new(VmPool::new(
            ResolvedSettings {
                layers: vec![],
                shares: vec![],
                memory_mb: None,
                disk_mb: None,
                idle_timeout: Duration::from_secs(idle_timeout_secs),
            },
            fake_sandal(dir),
        ))
    }

    fn call(pool: &VmPool, session: &str, command: &str) -> Result<String> {
        let cwd = std::env::current_dir().unwrap();
        // The run loop's half of the protocol: boot the session's VM.
        pool.ensure(session, &cwd.to_string_lossy())?;
        pool.execute(
            session,
            &json!({"command": command, "_session_cwd": cwd}),
        )
    }

    // ── Settings resolution ────────────────────────────────────────────

    /// Parse like the agent layer does: the `[agent.sandal]` table's
    /// value, not the whole config file.
    fn settings(content: &str) -> SandalAgentSettings {
        let root: toml::Value = toml::from_str(content).unwrap();
        root["agent"]["sandal"].clone().try_into().unwrap()
    }

    #[test]
    fn settings_parse_all_fields() {
        let raw = settings(
            "[agent.sandal]\nlayers = ['go.layer']\nshared = ['~/go/pkg:~/go/pkg']\nmemory = 4096\ndisk = 8192\nidle_timeout_secs = 600\n",
        );
        assert_eq!(raw.layers, ["go.layer"]);
        assert_eq!(raw.shared, ["~/go/pkg:~/go/pkg"]);
        assert_eq!(raw.memory, Some(4096));
        assert_eq!(raw.disk, Some(8192));
        assert_eq!(raw.idle_timeout_secs, Some(600));
    }

    #[test]
    fn unknown_settings_fields_are_rejected() {
        // A typo'd field is an error, not a silently ignored one.
        let error = toml::from_str::<SandalAgentSettings>("[agent.sandal]\nlayer = ['go.layer']\n")
            .unwrap_err();
        assert!(error.to_string().contains("layer"));
    }

    #[test]
    fn empty_layers_keep_default_sizing() {
        // No layers -> no sizing flags at all, whatever memory/disk say:
        // sizing exists for toolchain VMs, not bare ones — and shares
        // alone (a cache mapping without a toolchain) do not turn sizing
        // on either.
        let dir = unique_test_dir();
        let resolved = settings(
            "[agent.sandal]\nlayers = []\nshared = ['~/go/pkg:~/go/pkg']\nmemory = 65536\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();
        assert!(resolved.layers.is_empty());
        assert!(resolved.memory_mb.is_none());
        assert!(resolved.disk_mb.is_none());
    }

    #[test]
    fn shared_resolves_each_side() {
        // `~` expands per side: the host home on the host side, /root (the
        // sandbox guest always runs as root) on the guest side — so the
        // canonical cache mapping puts the guest's default cache path on
        // the host's persistent cache.
        let home = crate::paths::home_dir();
        let dir = unique_test_dir();
        let resolved = settings(
            "[agent.sandal]\nlayers = []\nshared = ['~/go/pkg:~/go/pkg', '~/.cache/go-build:~/.cache/go-build']\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();
        assert_eq!(
            resolved.shares,
            [
                format!("{}:/root/go/pkg", home.join("go/pkg").display()),
                format!(
                    "{}:/root/.cache/go-build",
                    home.join(".cache/go-build").display()
                ),
            ]
        );
    }

    #[test]
    fn shared_resolves_relative_host_paths_against_the_config_dir() {
        let dir = unique_test_dir();
        std::fs::create_dir_all(dir.path().join("cache")).unwrap();
        let resolved = settings(
            "[agent.sandal]\nlayers = []\nshared = ['cache:/root/cache']\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();
        assert_eq!(
            resolved.shares,
            [format!(
                "{}:/root/cache",
                dir.path().join("cache").display()
            )]
        );
    }

    #[test]
    fn shared_without_a_separator_is_an_error() {
        let dir = unique_test_dir();
        let error = settings("[agent.sandal]\nlayers = []\nshared = ['oops']\n")
            .resolve(Some(dir.path()))
            .unwrap_err();
        assert!(error.to_string().contains("expected host:guest"));
    }

    #[test]
    fn layers_resolve_against_the_config_dir() {
        let dir = unique_test_dir();
        let layer = dir.path().join("go.layer");
        std::fs::write(&layer, "layer").unwrap();

        let resolved = settings("[agent.sandal]\nlayers = ['go.layer']\n")
            .resolve(Some(dir.path()))
            .unwrap();

        assert_eq!(resolved.layers, vec![layer]);
        // Sizing defaults apply: a toolchain VM must not boot at the
        // sandbox's own 256 MB / layer+16 MB defaults.
        assert_eq!(resolved.memory_mb, Some(DEFAULT_MEMORY_MB));
        assert_eq!(resolved.disk_mb, Some(DEFAULT_DISK_MB));
    }

    #[test]
    fn explicit_sizing_wins_over_the_defaults() {
        let dir = unique_test_dir();
        std::fs::write(dir.path().join("go.layer"), "layer").unwrap();

        let resolved = settings(
            "[agent.sandal]\nlayers = ['go.layer']\nmemory = 2048\ndisk = 4096\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();

        assert_eq!(resolved.memory_mb, Some(2048));
        assert_eq!(resolved.disk_mb, Some(4096));
    }

    #[test]
    fn missing_layer_is_an_error() {
        let dir = unique_test_dir();
        let error = settings("[agent.sandal]\nlayers = ['go.layer']\n")
            .resolve(Some(dir.path()))
            .unwrap_err();
        assert!(error.to_string().contains("layer not found"));
    }

    #[test]
    fn nested_relative_paths_resolve() {
        // A configured nested path resolves through subdirectories of the
        // config dir — the check is existence, not containment: an operator
        // pointing outside the config dir is configuring their own deployment.
        let dir = unique_test_dir();
        std::fs::create_dir_all(dir.path().join("images")).unwrap();
        std::fs::write(dir.path().join("images/go.layer"), "layer").unwrap();
        let resolved = settings("[agent.sandal]\nlayers = ['images/go.layer']\n")
            .resolve(Some(dir.path()))
            .unwrap();
        assert_eq!(
            resolved.layers,
            vec![dir.path().join("images/go.layer")]
        );
    }

    #[test]
    fn synthetic_config_cannot_resolve_layers() {
        let error = settings("[agent.sandal]\nlayers = ['go.layer']\n")
            .resolve(None)
            .unwrap_err();
        assert!(error.to_string().contains("no config directory"));
    }

    #[test]
    fn idle_timeout_defaults_and_overrides() {
        let dir = unique_test_dir();
        std::fs::write(dir.path().join("go.layer"), "layer").unwrap();
        let default = settings("[agent.sandal]\nlayers = ['go.layer']\n")
            .resolve(Some(dir.path()))
            .unwrap();
        assert_eq!(
            default.idle_timeout,
            Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS)
        );
        let configured = settings(
            "[agent.sandal]\nlayers = ['go.layer']\nidle_timeout_secs = 60\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();
        assert_eq!(configured.idle_timeout, Duration::from_secs(60));
    }

    // ── VM pool behavior ───────────────────────────────────────────────

    #[test]
    fn one_vm_per_session_is_reused_across_calls() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // The first call boots the VM; the second call must land in the
        // same VM. A per-command VM would boot fresh each call.
        let first = call(&pool, "session-1", "echo vm-booted").unwrap();
        assert!(first.contains("vm-booted"));
        assert!(first.contains("exit code: 0"));
        let second = call(&pool, "session-1", "echo still-alive").unwrap();
        assert!(second.contains("still-alive"));
    }

    #[test]
    fn sessions_are_isolated_from_each_other() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // Two sessions, two VMs: distinct processes, both alive. The
        // host-side fake shares the host filesystem, so file-level
        // isolation is the real-sandbox e2e's proof (each VM's overlay
        // is its own); what the pool guarantees here is that no session
        // ever lands in another session's process.
        call(&pool, "session-a", "echo a").unwrap();
        call(&pool, "session-b", "echo b").unwrap();
        let entries = lock(&pool.entries);
        let a = entries.get("session-a").unwrap();
        let b = entries.get("session-b").unwrap();
        let pid_a = lock_state(&a.state).child.id();
        let pid_b = lock_state(&b.state).child.id();
        assert_ne!(pid_a, pid_b, "two sessions share one process");
        drop(entries);
        // And both keep serving their own sessions.
        assert!(call(&pool, "session-a", "echo a2").unwrap().contains("a2"));
        assert!(call(&pool, "session-b", "echo b2").unwrap().contains("b2"));
    }

    #[test]
    fn dead_vm_is_replaced_on_the_next_call() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // A command that kills the VM's own shell: EOF before the
        // sentinel, the entry is removed, and the next call boots fresh.
        let dead = call(&pool, "session-1", "kill -9 $$").unwrap_err();
        assert!(
            dead.to_string().contains("exited during the command"),
            "{dead}"
        );
        let fresh = call(&pool, "session-1", "echo fresh-vm").unwrap();
        assert!(fresh.contains("fresh-vm"));
    }

    #[test]
    fn timeout_closes_the_vm() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let timed_out = {
            let cwd = std::env::current_dir().unwrap();
            pool.ensure("session-1", &cwd.to_string_lossy()).unwrap();
            pool
                .execute(
                    "session-1",
                    &json!({
                        "command": "sleep 30",
                        "timeout_secs": 1,
                        "_session_cwd": cwd,
                    }),
                )
                .unwrap_err()
        };
        assert!(
            timed_out.to_string().contains("timed out"),
            "{timed_out}"
        );
        // The VM was closed: the next call boots a fresh one, and the
        // sleep from the timed-out call is gone with it.
        let fresh = call(&pool, "session-1", "echo fresh-vm").unwrap();
        assert!(fresh.contains("fresh-vm"));
    }

    #[test]
    fn idle_vms_are_reaped() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 0);
        call(&pool, "session-1", "echo booted").unwrap();
        // Idle timeout 0: the very next reaping pass closes the VM.
        pool.reap_idle();
        let entries = lock(&pool.entries);
        assert!(entries.is_empty(), "idle VM was not reaped");
        drop(entries);
        // And the next call boots a fresh one (not a corpse).
        let fresh = call(&pool, "session-1", "echo fresh-vm").unwrap();
        assert!(fresh.contains("fresh-vm"));
    }

    #[test]
    fn close_all_shuts_every_vm_down() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        call(&pool, "session-1", "echo one").unwrap();
        call(&pool, "session-2", "echo two").unwrap();
        pool.close_all();
        let entries = lock(&pool.entries);
        assert!(entries.is_empty(), "close_all left VMs standing");
    }

    #[test]
    fn missing_command_is_an_error() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let error = pool
            .execute(
                "session-1",
                &json!({"command": "   ", "_session_cwd": "/tmp"}),
            )
            .unwrap_err();
        assert!(error.to_string().contains("missing 'command'"));
    }

    #[test]
    fn missing_working_directory_is_an_error() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let error = pool
            .execute("session-1", &json!({"command": "echo hi"}))
            .unwrap_err();
        assert!(error.to_string().contains("no session working directory"));
    }

    #[test]
    fn output_is_capped_and_counted() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "seq 1 20000").unwrap();
        // The head is kept (bounded by MAX_OUTPUT plus the framing), the
        // rest is counted, the exit code rides the sentinel.
        assert!(result.starts_with("exit code: 0"));
        assert!(result.len() < MAX_OUTPUT + 2_000, "{}", result.len());
        assert!(result.contains("output truncated"));
    }

    #[test]
    fn nonzero_exit_codes_ride_the_sentinel() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "exit 42").unwrap();
        assert!(result.contains("exit code: 42"), "{result}");
    }

    /// The real-sandbox end-to-end: one VM per session, reused across the
    /// session's calls, and isolated from every other session's VM. Skips
    /// when the real sandbox binary is not installed — the pool behavior
    /// above is already covered by the fake.
    #[test]
    fn real_sandbox_reuse_and_isolation() {
        let Ok(bin) = resolve_sandal_bin() else {
            return;
        };
        let dir = unique_test_dir();
        let pool = VmPool::new(
            ResolvedSettings {
                layers: vec![],
                shares: vec![],
                memory_mb: None,
                disk_mb: None,
                idle_timeout: Duration::from_secs(3600),
            },
            bin,
        );
        let marker = "__reuse_marker__";
        // Reuse: a file the session wrote is still there on its next call.
        // A per-command VM would lose it.
        let wrote = call(&pool, "session-1", &format!("echo {marker} > /tmp/reuse.txt"))
            .unwrap();
        assert!(wrote.starts_with("exit code: 0"), "{wrote}");
        let read_back = call(&pool, "session-1", "cat /tmp/reuse.txt").unwrap();
        assert!(read_back.contains(marker), "the session VM did not survive: {read_back}");
        // Isolation: another session's VM boots fresh — the file is
        // invisible to it.
        let other = call(&pool, "session-2", "cat /tmp/reuse.txt 2>/dev/null").unwrap();
        assert!(
            other.starts_with("exit code: 1"),
            "another session saw the first session's file: {other}"
        );
    }

    /// The real toolchain e2e: a layer-carrying VM builds and runs a go
    /// program, and the session VM persists what the build left. Skips
    /// without the real binary or the real deployment's layer — the pool
    /// and protocol behavior above is already covered without them.
    #[test]
    fn real_sandbox_toolchain_build() {
        let Ok(bin) = resolve_sandal_bin() else {
            return;
        };
        let layer = crate::paths::home_dir().join("projects/kmcrawler-agent/go.layer");
        if !layer.is_file() {
            return;
        }
        let dir = unique_test_dir();
        // A dependency-free go project: no module fetches, so nothing
        // touches a host cache — the build cache stays inside the VM.
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("go.mod"), "module probe\n\ngo 1.21\n").unwrap();
        fs::write(
            project.join("main.go"),
            "package main\n\nfunc main() { println(\"__TOOLCHAIN_OK__\") }\n",
        )
        .unwrap();
        let pool = VmPool::new(
            ResolvedSettings {
                layers: vec![layer],
                shares: vec![],
                // The toolchain sizing defaults — a toolchain VM must not
                // boot at the sandbox's own 256 MB default.
                memory_mb: Some(DEFAULT_MEMORY_MB),
                disk_mb: Some(DEFAULT_DISK_MB),
                idle_timeout: Duration::from_secs(3600),
            },
            bin,
        );
        // The session VM binds to this test directory (the project's
        // parent), so the project is visible inside it — the same pinning
        // every session's working directory gets.
        let session_cwd = dir.path().to_string_lossy().to_string();
        pool.ensure("session-1", &session_cwd).unwrap();
        let run = |command: &str| {
            pool.execute(
                "session-1",
                &json!({"command": command, "_session_cwd": session_cwd}),
            )
            .unwrap()
        };
        let guest_project = "/tmp/project";
        let copy = format!(
            "mkdir -p {guest_project} && cp {}/* {guest_project}/",
            project.display()
        );
        let built = run(&copy);
        assert!(built.starts_with("exit code: 0"), "{built}");
        let built = run(&format!(
            "cd {guest_project} && go build -o probe . && ./probe"
        ));
        assert!(built.contains("__TOOLCHAIN_OK__"), "{built}");
        // The session VM persisted the build: a later call still sees it.
        let persisted = run(&format!(
            "test -x {guest_project}/probe && echo persisted"
        ));
        assert!(persisted.contains("persisted"), "{persisted}");
    }

    /// The guarantee the dispatch exists for: one session's long-running
    /// command must not stall another session's shell call. A
    /// single-threaded pool serializes them and this test measures it.
    #[test]
    fn cross_session_commands_run_concurrently() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // Session A blocks for 3s; session B's fast call must complete
        // while A's is still running, not after it.
        let pool_a = Arc::clone(&pool);
        let a = std::thread::spawn(move || {
            let cwd = std::env::current_dir().unwrap();
            pool_a
                .ensure("session-a", &cwd.to_string_lossy())
                .unwrap();
            pool_a
                .execute(
                    "session-a",
                    &json!({
                        "command": "sleep 3; echo a-done",
                        "timeout_secs": 30,
                        "_session_cwd": cwd,
                    }),
                )
                .unwrap()
        });
        std::thread::sleep(Duration::from_millis(500));
        let started = Instant::now();
        let b = call(&pool, "session-b", "echo b-fast").unwrap();
        let elapsed = started.elapsed();
        assert!(b.contains("b-fast"), "{b}");
        assert!(
            elapsed < Duration::from_secs(2),
            "session B waited {elapsed:?} behind session A's command"
        );
        assert!(a.join().unwrap().contains("a-done"));
    }

    /// The reaper never kills a running command's VM, whatever the idle
    /// timeout says — a held state lock is a running command.
    #[test]
    fn the_reaper_skips_running_commands() {
        let dir = unique_test_dir();
        // Idle timeout 0: everything idle closes on the next pass — the
        // running command's VM must survive even that.
        let pool = pool(dir.path(), 0);
        let pool_a = Arc::clone(&pool);
        let a = std::thread::spawn(move || {
            let cwd = std::env::current_dir().unwrap();
            pool_a
                .ensure("session-1", &cwd.to_string_lossy())
                .unwrap();
            pool_a
                .execute(
                    "session-1",
                    &json!({
                        "command": "sleep 2; echo survived",
                        "timeout_secs": 30,
                        "_session_cwd": cwd,
                    }),
                )
                .unwrap()
        });
        std::thread::sleep(Duration::from_millis(500));
        pool.reap_idle();
        pool.reap_idle();
        let result = a.join().unwrap();
        assert!(result.contains("survived"), "{result}");
        // And once the command finished, the stale VM IS closed.
        pool.reap_idle();
        assert!(
            lock(&pool.entries).is_empty(),
            "the finished command's VM was not reaped"
        );
    }

    /// A VM that dies between calls (no EOF ever observed — the previous
    /// command completed fine) is detected at the next write in one
    /// refusal round, not retried against for the whole timeout: the
    /// entry is removed and the session's next call boots fresh.
    #[test]
    fn a_vm_that_dies_between_calls_is_detected_fast() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // A fine first command — the VM looks healthy.
        assert!(call(&pool, "session-1", "echo healthy").unwrap().contains("healthy"));
        // The VM dies out-of-band (the memory cap OOM-killed the guest).
        {
            let entries = lock(&pool.entries);
            let entry = entries.get("session-1").unwrap();
            close_state(&mut lock_state(&entry.state));
        }
        // The next call's write hits the corpse: detected in one round.
        let started = Instant::now();
        let error = call(&pool, "session-1", "echo next").unwrap_err();
        assert!(
            error.to_string().contains("died before the command ran"),
            "{error}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the dead-VM detection took {:?}",
            started.elapsed()
        );
        // The entry is gone: the next call boots a fresh VM.
        assert!(lock(&pool.entries).get("session-1").is_none());
        assert!(call(&pool, "session-1", "echo fresh-vm").unwrap().contains("fresh-vm"));
    }

    /// The background-job path: a job that outlives any transport starts
    /// detached inside the VM (here: the fake's host-side process), is
    /// polled by id, and reports its own exit code when it finishes.
    #[test]
    fn background_jobs_start_poll_and_finish() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
        pool.ensure("session-1", &cwd).unwrap();
        let start = pool
            .execute(
                "session-1",
                &json!({
                    "command": "sleep 1; echo job-done",
                    "background": true,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap();
        assert!(start.starts_with("Background job started: potlatch-job-"), "{start}");
        let job = start
            .lines()
            .find_map(|l| l.strip_prefix("Background job started: "))
            .unwrap()
            .to_string();
        // Poll while it runs: no exit code, no output.
        let running = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(running.contains("(still running)"), "{running}");
        // Wait for the job, then poll: its own exit code and output.
        std::thread::sleep(Duration::from_secs(2));
        let finished = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(finished.contains("exit code: 0"), "{finished}");
        assert!(finished.contains("job-done"), "{finished}");
    }

    /// A killed job is gone: best-effort kill (its own shell), then the
    /// poll finds nothing — no output, no exit code, ever.
    #[test]
    fn background_jobs_kill_best_effort() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
        pool.ensure("session-1", &cwd).unwrap();
        let start = pool
            .execute(
                "session-1",
                &json!({
                    "command": "sleep 30",
                    "background": true,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap();
        let job = start
            .lines()
            .find_map(|l| l.strip_prefix("Background job started: "))
            .unwrap()
            .to_string();
        let killed = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "kill": true, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(killed.contains("killed"), "{killed}");
        let after = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(after.contains("(no output yet)"), "{after}");
    }

    /// The regression the agent's worker pool exists for: PR_SET_PDEATHSIG
    /// kills a child when its parent *thread* exits, so a VM spawned from
    /// a per-call thread dies with every delivered command. The
    /// production shape — a long-lived worker serving a session's whole
    /// run of calls — must keep the VM alive across them; the worker's
    /// own eventual exit is the PDEATHSIG belt, not a bug.
    #[test]
    fn a_long_lived_worker_serves_a_whole_run_of_calls() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let pool = std::sync::Arc::new(pool);
        let (tx, rx) = std::sync::mpsc::channel::<&'static str>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<String>();
        let worker = std::thread::spawn(move || {
            let cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
            pool.ensure("session-1", &cwd).unwrap();
            while let Ok(command) = rx.recv() {
                let result = pool
                    .execute(
                        "session-1",
                        &json!({"command": command, "_session_cwd": cwd}),
                    )
                    .unwrap_or_else(|error| format!("ERROR: {error}"));
                let _ = done_tx.send(result);
            }
        });
        // The worker serves a whole run of calls; the VM survives each.
        let _ = tx.send("echo first-call");
        assert!(
            done_rx.recv().unwrap().contains("first-call"),
            "the first call did not run"
        );
        let _ = tx.send("echo second-call");
        assert!(
            done_rx.recv().unwrap().contains("second-call"),
            "the VM did not survive the first call's completion — \
             PR_SET_PDEATHSIG killed it with a per-call thread"
        );
        // The worker's exit is the belt: it kills the VM, by design.
        drop(tx);
        worker.join().unwrap();
    }

    /// A foreground command sent while a background job is running must
    /// execute immediately: the start returns when the backgrounding
    /// lands (never when the job finishes), the job's output is
    /// redirected to its own file (never the shared transport), and the
    /// wrapper keeps reading lines — so nothing about the job can stall
    /// or pollute the next command.
    #[test]
    fn a_foreground_command_runs_while_a_background_job_is_running() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
        pool.ensure("session-1", &cwd).unwrap();
        // Start a long-running background job.
        let start = pool
            .execute(
                "session-1",
                &json!({
                    "command": "sleep 10; echo job-done",
                    "background": true,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap();
        assert!(start.starts_with("Background job started: "), "{start}");
        let job = start
            .lines()
            .find_map(|l| l.strip_prefix("Background job started: "))
            .unwrap()
            .to_string();
        // A foreground command while the job runs: immediate, not behind
        // the job.
        let started = Instant::now();
        let interleaved = call(&pool, "session-1", "echo interleaved-ok").unwrap();
        assert!(interleaved.contains("interleaved-ok"), "{interleaved}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the foreground command waited {:?} behind the background job",
            started.elapsed()
        );
        // The job is still running, untouched by the foreground command.
        let running = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(running.contains("(still running)"), "{running}");
        // Kill the job; the poll finds nothing after.
        let killed = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "kill": true, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(killed.contains("killed"), "{killed}");
        let after = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(after.contains("(no output yet)"), "{after}");
    }

    /// Multiple background jobs run at the same time: the starts serialize
    /// on the VM's state lock (each well under a second), and the jobs
    /// themselves run concurrently — one per command, none waiting behind
    /// another.
    #[test]
    fn multiple_background_jobs_run_concurrently() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
        pool.ensure("session-1", &cwd).unwrap();
        let started = Instant::now();
        let starts: Vec<String> = (0..3)
            .map(|i| {
                pool.execute(
                    "session-1",
                    &json!({
                        "command": format!("sleep 8; echo job-{i}-done"),
                        "background": true,
                        "_session_cwd": cwd,
                    }),
                )
                .unwrap()
            })
            .collect();
        // All three starts returned — the serialized part — in seconds,
        // not behind any job.
        assert!(started.elapsed() < Duration::from_secs(15));
        let jobs: Vec<String> = starts
            .iter()
            .map(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("Background job started: "))
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(jobs.len(), 3);
        // And all three jobs are running at the same time.
        for job in &jobs {
            let running = pool
                .execute(
                    "session-1",
                    &json!({"job_id": job, "_session_cwd": cwd}),
                )
                .unwrap();
            assert!(running.contains("(still running)"), "{running}");
        }
        // None finished behind another: all three report their own exit
        // within the same window after the sleep.
        std::thread::sleep(Duration::from_secs(9));
        for (i, job) in jobs.iter().enumerate() {
            let finished = pool
                .execute(
                    "session-1",
                    &json!({"job_id": job, "_session_cwd": cwd}),
                )
                .unwrap();
            assert!(
                finished.contains(&format!("job-{i}-done")),
                "job {i}: {finished}"
            );
            assert!(finished.contains("exit code: 0"), "job {i}: {finished}");
        }
    }

    #[test]
    fn sentinel_locates_split_across_reads() {
        // The sentinel search must find a marker wherever it lands,
        // including straddling a read boundary.
        let sentinel = "__POTLATCH_VM_EXIT_1__";
        let mut data = b"junkjunkjunk".to_vec();
        data.extend_from_slice(format!("{sentinel}:7\n").as_bytes());
        let (at, code) = find_sentinel(&data, sentinel).unwrap();
        assert_eq!(code, 7);
        assert_eq!(&data[at..at + sentinel.len()], sentinel.as_bytes());
        // And not find what is not there.
        assert_eq!(find_sentinel(b"junk", sentinel), None);
        assert_eq!(find_sentinel(&data, "__OTHER__"), None);
    }
}
