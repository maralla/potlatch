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
//! Commands ride the sandbox's own command line (the vsock line): the
//! VM boots with `--port`, and every call opens ONE connection to that
//! listener, hands it the command (a length-prefixed line the guest's
//! /bin/vsockd runs under /bin/sh with the connection as its stdio),
//! and reads the command's own bytes back until EOF — the 17-byte
//! status trailer at the stream's end carries the exit code. No sshd,
//! no keys, no session shell: the connection per command IS the
//! concurrency model. Commands of one session run in parallel inside
//! its VM (the guest forks per connection), the pool's state lock
//! guards only the VM's lifecycle — never a run — and a command's
//! timeout kills the command (the connection drops, and the guest
//! reaps what the RST reaches) instead of the whole VM.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::bus::{AgentInbox, AgentToolDefinition};
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::core::scope::SessionScope;
use crate::core::ssh_util::shell_single_quote;
use crate::core::workflow::AgentBuildContext;
use crate::util::dies_with_parent;

const AGENT_NAME: &str = "sandal";
const SHELL_TOOL: &str = "shell";
const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
/// Default per-command timeout (seconds): the same default the built-in
/// shell tool carries.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Maximum per-command timeout (seconds): the request's own deadline
/// (30 minutes — the bus deadline that carries the response). A request
/// past it is capped, not honored.
const MAX_TIMEOUT_SECS: u64 = 30 * 60;
/// Output kept per command (the head); the rest is counted, not stored.
/// The same cap the built-in shell tool carries.
const MAX_OUTPUT: usize = 50_000;
/// The socket read slice while a command's output drains: how long a
/// read may block before the deadline is re-checked. A command that
/// produces nothing blocks here, never past the deadline.
const READ_SLICE: Duration = Duration::from_millis(200);
/// Bound on the TCP connect itself: the VM's listener is local, so a
/// connect that cannot complete in this window is a dead VM, not a slow
/// one (the same bound the harness `ssh` tool puts on its connect).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a booting VM may take to answer the readiness probe (the
/// guest boots in well under a second; a layerchain VM takes a few).
const BOOT_DEADLINE: Duration = Duration::from_secs(60);

/// The command line's wire protocol, as the guest's /bin/vsockd reads
/// it: one connection header — {mode: u8, rows: u16, cols: u16,
/// cmd_len: u32}, little-endian — then exactly cmd_len command bytes.
/// Mode 0x01 (the oneshot) is the only mode a shell call needs: the
/// command runs with the connection as its stdio, the output comes back
/// verbatim, and EOF ends the line.
const MODE_ONESHOT: u8 = 0x01;
/// The header's size on the wire.
const HEADER_LEN: usize = 9;
/// The command's size limit on the guest side (vsockd's own CMD_MAX): a
/// longer payload is rejected HERE, with the reason named, instead of
/// in the guest with a silent exit 124.
const MAX_COMMAND_BYTES: usize = 65535;
/// The 17-byte status trailer that ends every run: 16 magic bytes, then
/// the status byte (the shell convention: a signal death rides as
/// 128+signal). EOF without a whole valid trailer is a dead line, not a
/// finished run.
const TRAILER_MAGIC: &[u8; 16] = b"VSOCKD:EXIT:01\n\0";
const TRAILER_LEN: usize = 17;

/// The console command: a keepalive that holds the console session open
/// for the VM's life — sandal tears the VM down when the console session
/// dies, so the keepalive IS the VM's life from the console side. It
/// produces nothing, so the held pipes never fill; commands no longer
/// ride the console at all (they ride the vsock line).
const CONSOLE_KEEPALIVE: &str = "tail -f /dev/null";

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
const SHELL_DESCRIPTION: &str = "Run a shell command inside the session's sandbox VM. You are already in the working directory (cwd) — no need to `cd` into it. The VM is reused across this session's calls: packages installed by an earlier call are still installed, and files written by an earlier call are still there. Calls run concurrently: a long-running command does not block the next call. Returns the command's output and exit code. Foreground commands have a timeout (default 120s, maximum 30 minutes); on timeout the command is killed and the VM stays up. Anything longer than a minute or two belongs in the background: set background: true, get a job id back immediately, and poll it with job_id (or kill it with job_id + kill: true). Poll a background job a couple of times, not forever: if a job has produced no new output after a few polls over several minutes, kill it (job_id + kill: true) and run a narrower command instead of sleeping and re-polling in a loop. Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `write` or `edit` instead.";

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
    /// The session VM's memory cap (MB).
    pub memory_mb: u64,
    /// The session VM's writable overlay disk (MB).
    pub disk_mb: u64,
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
    pub fn resolve(&self, config_dir: Option<&Path>) -> Result<ResolvedSettings> {
        let config_dir = config_dir.context(
            "cannot resolve [agent.sandal] paths: no config directory (synthetic config)",
        )?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for configured in &self.layers {
            let path = config_dir.join(configured);
            let path = path
                .canonicalize()
                .with_context(|| format!("[agent.sodal] layer not found: {configured}"))?;
            ensure!(
                path.is_file(),
                "[agent.sandal] layer is not a file: {configured}"
            );
            layers.push(path);
        }
        let mut shares = Vec::with_capacity(self.shared.len());
        for spec in &self.shared {
            shares.push(resolve_share(spec, config_dir)?);
        }
        Ok(ResolvedSettings {
            layers,
            shares,
            memory_mb: self.memory.unwrap_or(DEFAULT_MEMORY_MB),
            disk_mb: self.disk.unwrap_or(DEFAULT_DISK_MB),
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
        if let Some(request) = self.inbox.recv_timeout(INBOX_WAIT)? {
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
            // commands run concurrently: one session's long command no
            // longer blocks another session's (or its own next) call.
            let boot = SessionScope::from_metadata(&request.payload)
                .and_then(|scope| self.pool.ensure(session_id, &scope));
            if let Err(error) = boot {
                request.respond(Err(error.context("the session's sandbox VM cannot boot")));
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
        let bin =
            resolve_sandal_bin().context("[agent.sandal] requires the sandal sandbox binary")?;
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
                            "description": "Timeout in seconds for foreground commands. Default: 120, maximum: 1800 (30 minutes) — a request past the maximum is capped to it. On timeout the command is killed (the VM stays up). Ignored for background jobs (they run until exit or kill)."
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
                for_agent: None,
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

/// Allocate one command line port: bind 127.0.0.1:0, take the port,
/// drop the listener. The window between the drop and the sandbox's own
/// bind is microseconds; a collision in it fails the sandbox's bind
/// loudly (the process exits with the error), never silently.
fn allocate_vsock_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .context("failed to allocate a sandbox command line port")?;
    let port = listener
        .local_addr()
        .context("failed to read the allocated sandbox command line port")?
        .port();
    drop(listener);
    Ok(port)
}

/// One live session VM. The session's file scope is pinned at spawn and
/// readable without the state lock; everything mutable lives behind one
/// state lock that a command takes only at its edges (never for the
/// run), and the running count tells the reaper a command is in flight.
struct VmEntry {
    /// The session's file scope this VM was booted with — the calling
    /// agent's cwd and declared roots, pinned on the session's first call
    /// and never renegotiated: every later call lands in the same VM, so
    /// a session that somehow presents a different scope gets an error,
    /// not a second VM or a reshare.
    scope: SessionScope,
    /// The command line's host port: the sandbox's `--port` listener,
    /// fixed for the VM's life. Every command opens its own connection
    /// to it — the concurrency model, no shared transport to serialize
    /// on.
    port: u16,
    /// Commands in flight on this VM. The reaper and close-all skip an
    /// entry with a nonzero count — a running command is neither idle
    /// whatever the clock says, nor closed under itself.
    running: AtomicUsize,
    /// The whole mutable state behind one lock. A command takes it only
    /// to stamp `last_used` at its start and end and to close the VM —
    /// never for the run itself, so same-session commands run in
    /// parallel.
    state: Mutex<VmState>,
}

/// The mutable state of one live session VM: the VM process and the
/// console session that keeps it alive.
struct VmState {
    /// The sandbox process — the VM itself. Lives for the entry's whole
    /// life; killed in [`close_state`].
    vm: Child,
    /// The console pipes, held for the VM's life: dropping them closes
    /// the console session (sandal tears the VM down with it), and the
    /// keepalive holds it open. Taken and dropped in [`close_state`] —
    /// the console session dies first, sandal's own teardown runs, and
    /// the kill below is the belt. Commands never touch these: they
    /// ride the vsock line.
    console: Option<(ChildStdin, ChildStdout)>,
    /// Last shell call on this VM — stamped at each command's start AND
    /// end, under the state lock.
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
    fn spawn(&self, scope: &SessionScope) -> Result<VmEntry> {
        // (a) The sandbox process: --port (the command line's listener —
        // every command connection lands here and the VMM bridges it
        // through vsock to the guest's /bin/vsockd), the layers, the
        // shares, the sizing, and the console keepalive. Take the console
        // pipes and hold them for the VM's life.
        let port = allocate_vsock_port()?;
        let mut cmd = Command::new(&self.bin);
        cmd.arg("run")
            .arg("--port")
            .arg(format!("127.0.0.1:{port}"))
            .arg("-m")
            .arg(self.settings.memory_mb.to_string())
            .arg("--disk-size")
            .arg(self.settings.disk_mb.to_string());
        for layer in &self.settings.layers {
            cmd.arg("--layer").arg(layer);
        }
        // The working directory rides the same path on both sides so
        // guest paths resolve exactly as the session's paths do.
        let cwd = scope.cwd();
        cmd.arg("--share").arg(format!("{cwd}:{cwd}"));
        // The declared write roots ride the same path on both sides for the
        // same reason: the session's out-of-cwd work — its test scripts,
        // its scratch files — must resolve inside the sandbox exactly as it
        // does outside. A root that is not a directory fails the boot, not a
        // silently dropped share: the declaration is the calling agent's
        // own, and a dropped root would surface later as a confusing path
        // error inside the sandbox.
        for root in scope.roots() {
            let root = root.display().to_string();
            ensure!(
                Path::new(&root).is_dir(),
                "[agent.sandal] the session's declared write root is not \
                 a directory: {root}"
            );
            cmd.arg("--share").arg(format!("{root}:{root}"));
        }
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
        // The console: the keepalive only. The VM's life is the console
        // session's life; commands ride the vsock line, never the
        // console.
        let keepalive = format!("exec {CONSOLE_KEEPALIVE}");
        cmd.arg("--").arg("sh").arg("-c").arg(&keepalive);
        // stderr is inherited, never piped: the boot log lands in this
        // process's own log (where it is greppable), and an unread pipe
        // could never fill and stall the VM mid-boot.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        dies_with_parent(&mut cmd);
        Self::spawn_vm(cmd, self.bin.clone(), scope, port)
    }

    /// The rest of one session VM's spawn: the VM process and its console
    /// pipes, then the readiness probe — one real command through the
    /// line, so a broken deployment (a sandbox that dies at boot, a
    /// guest whose vsockd never comes up) fails the boot loudly here,
    /// never as a confusing timeout on the session's first command.
    fn spawn_vm(
        mut cmd: Command,
        bin: PathBuf,
        scope: &SessionScope,
        port: u16,
    ) -> Result<VmEntry> {
        let mut vm = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", bin.display()))?;
        let console = Some((
            vm.stdin.take().context("sandbox stdin not piped")?,
            vm.stdout.take().context("sandbox stdout not piped")?,
        ));
        if let Err(error) = probe_ready(&mut vm, port) {
            let _ = vm.kill();
            let _ = vm.wait();
            return Err(error);
        }
        Ok(VmEntry {
            scope: scope.clone(),
            port,
            running: AtomicUsize::new(0),
            state: Mutex::new(VmState {
                vm,
                console,
                last_used: Instant::now(),
            }),
        })
    }

    /// Boot a session's VM if it has none. Run-loop only: the spawn must
    /// happen on a long-lived thread (see the run loop), never on a
    /// per-call thread.
    fn ensure(&self, session_id: &str, scope: &SessionScope) -> Result<()> {
        let mut entries = lock(&self.entries);
        match entries.get(session_id) {
            Some(_) => Ok(()),
            None => {
                let entry = Arc::new(self.spawn(scope)?);
                entries.insert(session_id.to_string(), entry);
                Ok(())
            }
        }
    }

    /// The live entry for a session's already-booted VM, checked against
    /// this call's context: the cwd and the declared write roots are pinned
    /// at boot, so a call presenting a different pair is an error, not a
    /// renegotiation and not a second VM. A missing entry was reaped (or
    /// died) between the run loop's boot and this call: the caller retries,
    /// and the run loop boots it.
    fn entry_for(&self, session_id: &str, payload: &Value) -> Result<Arc<VmEntry>> {
        let scope = SessionScope::from_metadata(payload)?;
        let entries = lock(&self.entries);
        match entries.get(session_id) {
            Some(entry) => {
                if entry.scope != scope {
                    return Err(anyhow::anyhow!(
                        "the session's sandbox VM is bound to its first call's \
                         working directory and write roots"
                    ));
                }
                Ok(Arc::clone(entry))
            }
            None => Err(anyhow::anyhow!(
                "the session's sandbox VM is not running; retry"
            )),
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
        let timeout_secs = payload
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);
        let timeout = Duration::from_secs(timeout_secs.max(1));
        // At most one VM per session: reuse the live one, or boot the
        // first. The map lock covers only the lookup and the insert —
        // never a command — so one session's boot never blocks another
        // session's lookup.
        let entry = self.entry_for(session_id, payload)?;
        let background = payload
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let outcome = if background {
            let job = job_tag();
            let line = job_start_line(entry.scope.cwd(), command, &job);
            match self.run_command(session_id, &entry, line, timeout) {
                Outcome::Done(Ok(raw)) if raw.code == 0 => {
                    return Ok(format!(
                        "Background job started: {job}\ncommand: {command}\nPoll with job_id \
                         later in this session; the job's output and exit code are kept inside \
                         the VM and die with it."
                    ));
                }
                other => other,
            }
        } else {
            let line = foreground_payload(entry.scope.cwd(), command);
            self.run_command(session_id, &entry, line, timeout)
        };
        outcome
            .into_result()
            .map(|raw| format_result(&raw.out, raw.code, raw.dropped))
    }

    /// Poll or kill one background job. The job's state is the VM's (its
    /// output file and its own exit code); this only reads what it left.
    fn execute_job_control(&self, session_id: &str, payload: &Value, job: &str) -> Result<String> {
        // The job tag is ours (hex from job_tag); never let a caller
        // inject one into a shell line.
        ensure!(
            job.len() <= 64 && job.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "invalid job id {job:?}"
        );
        let kill = payload
            .get("kill")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS.max(1));
        let entry = self.entry_for(session_id, payload)?;
        let line = if kill {
            job_kill_line(job)
        } else {
            job_poll_line(job)
        };
        let outcome = self.run_command(session_id, &entry, line, timeout);
        let raw = outcome.into_result()?;
        if kill {
            return Ok(format!(
                "Job {job} killed (best-effort: its own shell only)"
            ));
        }
        // The job's own exit code and start stamp ride its output file.
        let output = &raw.out[..];
        let exit = digits_after_marker(output, b"JOB-EXIT:");
        let started_epoch = digits_after_marker(output, b"JOB-START:");
        let mut result = match exit {
            Some(code) => format!("exit code: {code}\n\n"),
            // The job's age is what tells a waiting model whether to keep
            // waiting or to kill the job and change course: without it,
            // "(still running)" reads the same at 5s as at 5h.
            None => match job_age_seconds(started_epoch) {
                Some(age) => {
                    format!("exit code: (still running, {})\n\n", format_job_age(age))
                }
                None => "exit code: (still running)\n\n".to_string(),
            },
        };
        // Strip the bookkeeping markers: the status is the header, not
        // output. JOB-EXIT rides the file's tail; JOB-START its head.
        let head_end = find_subslice(output, b"\nJOB-EXIT:", 0).unwrap_or(output.len());
        let output_part = String::from_utf8_lossy(strip_job_start_stamp(&output[..head_end]));
        if output_part.trim().is_empty() {
            result.push_str("(no output yet)");
        } else {
            result.push_str(&output_part);
        }
        Ok(result)
    }

    /// Run one command line in the session's VM over its own connection.
    /// The state lock is taken only at the edges (the last-use stamps and
    /// a dead-VM close) — never for the run, so commands of one session
    /// run in parallel inside its VM, the concurrency the vsock line's
    /// fork-per-connection guest is built for.
    fn run_command(
        &self,
        session_id: &str,
        entry: &VmEntry,
        payload: String,
        timeout: Duration,
    ) -> Outcome {
        // The command's size limit is the guest's own; a longer payload
        // is refused HERE with the reason named, never sent to die
        // silently in the guest's protocol check.
        if payload.len() > MAX_COMMAND_BYTES {
            return Outcome::Done(Err(anyhow::anyhow!(
                "command is {} bytes; the sandbox line carries at most {MAX_COMMAND_BYTES}",
                payload.len()
            )));
        }
        if payload.contains('\0') {
            // The payload is a C string on the guest side (execve's
            // argv): a NUL would truncate the command mid-flight.
            return Outcome::Done(Err(anyhow::anyhow!("command contains a NUL byte")));
        }
        let _busy = entry.busy();
        {
            let mut state = lock_state(&entry.state);
            state.last_used = Instant::now();
        }
        let run = vsockd_run(
            entry.port,
            payload.as_bytes(),
            Instant::now() + timeout,
            false,
        );
        {
            let mut state = lock_state(&entry.state);
            state.last_used = Instant::now();
        }
        match run {
            Run::Finished(raw) => Outcome::Done(Ok(raw)),
            Run::TimedOut => Outcome::Done(Err(anyhow::anyhow!(
                "timed out after {}s; the command was killed — the session's VM stays up",
                timeout.as_secs()
            ))),
            Run::Lost(reason) => {
                // The line broke before the trailer: the VM died, or the
                // command line itself failed mid-run. Either way the VM
                // is closed and the entry removed — the session's next
                // call boots a fresh one instead of hitting a corpse.
                let mut state = lock_state(&entry.state);
                let exited = state.vm.try_wait().ok().flatten();
                close_state(&mut state);
                drop(state);
                lock(&self.entries).remove(session_id);
                let why = match exited {
                    Some(status) => format!(
                        "the session's sandbox VM exited during the command (status {}); \
                         see the platform log for the sandbox's own output; the session's \
                         next call boots a fresh one",
                        status.code().unwrap_or(-1)
                    ),
                    None => format!(
                        "the sandbox command line broke before the command's status arrived \
                         ({reason}); the session's next call boots a fresh one"
                    ),
                };
                Outcome::Closed(anyhow::anyhow!(why))
            }
        }
    }

    /// Close VMs whose last shell call is older than the configured idle
    /// timeout. Each close kills the VM with its whole overlay — nothing
    /// a session wrote survives it.
    fn reap_idle(&self) {
        let mut entries = lock(&self.entries);
        let idle = self.settings.idle_timeout;
        entries.retain(|session_id, entry| {
            // A command in flight keeps its VM whatever the clock says —
            // the running count is the reaper's busy signal (the state
            // lock is not held for a run).
            if entry.running.load(Ordering::SeqCst) > 0 {
                return true;
            }
            let mut state = lock_state(&entry.state);
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
            // A running command finishes against its own connection (and
            // PR_SET_PDEATHSIG closes the rest when this process dies);
            // shutdown never closes a VM under a running command.
            if entry.running.load(Ordering::SeqCst) > 0 {
                info!(
                    target: "potlatch::sandal",
                    "session VM busy with a running command (session {session_id}); \
                     it closes itself when the command ends"
                );
                continue;
            }
            info!(target: "potlatch::sandal", "closing session VM (session {session_id})");
            close_state(&mut lock_state(&entry.state));
        }
        entries.clear();
    }
}

/// A command in flight: the entry's running count is bumped on construction
/// and dropped back on scope exit — every exit path of a command (a run, a
/// timeout, a dead VM, a panic) passes through here.
struct Busy<'a>(&'a AtomicUsize);

impl VmEntry {
    fn busy(&self) -> Busy<'_> {
        self.running.fetch_add(1, Ordering::SeqCst);
        Busy(&self.running)
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A fresh job tag per start: it rides the job's command line (so a kill
/// can find it) and names its output file.
fn job_tag() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("potlatch-job-{nanos:x}")
}

/// Close one VM: drop the console pipes (the console session dies, and
/// sandal tears the VM down from the inside — its own teardown, the
/// guest cleaning up after itself) and kill the process. The kills are
/// the belt. The overlay dies with the process; nothing is persisted.
fn close_state(state: &mut VmState) {
    drop(state.console.take());
    let _ = state.vm.kill();
    let _ = state.vm.wait();
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
/// dead-VM path boots a fresh one, not a bricked session.
fn lock_state(state: &Mutex<VmState>) -> std::sync::MutexGuard<'_, VmState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ── The command line ─────────────────────────────────────────────────

/// One command's outcome inside a VM: the formatted result, or the
/// failure that closed the VM (the entry is already removed; the next
/// call boots fresh).
enum Outcome {
    /// The run finished (a real exit code off the trailer) — or failed
    /// without killing the VM (a timeout, a rejected command): the VM
    /// keeps serving.
    Done(Result<RawRun>),
    /// The VM is gone; the entry was removed. The payload is the error,
    /// never a run.
    Closed(anyhow::Error),
}

impl Outcome {
    fn into_result(self) -> Result<RawRun> {
        match self {
            Outcome::Done(result) => result,
            Outcome::Closed(error) => Err(error),
        }
    }
}

/// One finished line's raw result: the kept output head, the exit code
/// the trailer carried, and how much the cap dropped. The foreground
/// path formats this; the job paths parse it.
struct RawRun {
    out: Vec<u8>,
    code: i32,
    dropped: usize,
}

/// How one command's connection ended.
enum Run {
    /// EOF with a whole valid trailer: the run's output and exit code.
    Finished(RawRun),
    /// The deadline passed first: the connection was dropped, which is
    /// the kill signal the guest's own reaper acts on. The VM keeps
    /// serving.
    TimedOut,
    /// The line broke before the trailer: EOF without one, a connect
    /// refusal, a write or read error. Carries what the transport said.
    Lost(String),
}

/// The foreground command's payload: the `cd` guard (the VM's working
/// directory or exit 127) and the command itself, verbatim — the line
/// carries the command as length-prefixed bytes, so unlike the shell
/// era nothing needs quoting or encoding; the guest runs the payload
/// under /bin/sh and the trailer carries the exit code.
fn foreground_payload(cwd: &str, command: &str) -> String {
    format!(
        "cd {} 2>/dev/null || exit 127\n{command}",
        shell_single_quote(cwd)
    )
}

/// The background-start line: the job runs detached in the VM with its
/// whole output going to a job-tagged file — a `JOB-START:` epoch first
/// line (a poll reports the job's age from it), the job's own output,
/// and its own `JOB-EXIT:` code when it finishes. The job tag rides the
/// job's command line, so a kill can find it with pkill. The line's own
/// exit code (the trailer's) is the backgrounding's: 0 when the job
/// started, a refusal code when the cd guard failed.
fn job_start_line(cwd: &str, command: &str, job: &str) -> String {
    let encoded = STANDARD.encode(command);
    let out = job_output_path(job);
    // The stamp is written synchronously, before the backgrounding: a job
    // that has not run a single instruction yet still reports its age to
    // the first poll (the job writing its own stamp would race it). The
    // job's own output and its JOB-EXIT code append after the stamp. The
    // command rides base64: it is embedded in a shell expression here,
    // where verbatim bytes (quotes, newlines, expansions) would not
    // survive.
    format!(
        "cd {} 2>/dev/null || exit 127; printf 'JOB-START:%s\\n' \"$(date +%s)\" >{out}; \
         ( eval \": {job}; $(printf %s {encoded} | base64 -d)\"; \
         printf 'JOB-EXIT:%d\\n' \"$?\" >>{out} ) >>{out} 2>&1 &",
        shell_single_quote(cwd),
    )
}

/// The job-poll line: the job's own output tail (the newest bytes are
/// what a poll must surface) plus its own exit code when it has one.
fn job_poll_line(job: &str) -> String {
    let out = job_output_path(job);
    format!("tail -c 50000 {out} 2>/dev/null")
}

/// The job-kill line: best-effort — the job tag rides the job's command
/// line, so pkill finds the job's shell; the children it spawned are
/// orphans pkill cannot reach.
fn job_kill_line(job: &str) -> String {
    let out = job_output_path(job);
    format!("pkill -9 -f '{job}' 2>/dev/null; rm -f {out}")
}

/// Where a job's output (and its own exit code) lives: inside the VM,
/// never on a share — the job state is the VM's, and it dies with the VM.
fn job_output_path(job: &str) -> String {
    format!("/tmp/{job}.out")
}

/// Run one command through the line and collect its run: connect (one
/// attempt unless `retry_connect` — the readiness probe retries, the
/// boot window's listener bind is microseconds behind the spawn), write
/// the connection header and the payload, then drain the output to EOF
/// and read the trailer off the stream's end.
///
/// The deadline bounds the whole run. The output's head is kept up to
/// [`MAX_OUTPUT`], the rest is counted; the last [`TRAILER_LEN`] bytes
/// are always carried aside so the trailer is validated wherever the
/// cap falls.
fn vsockd_run(port: u16, payload: &[u8], deadline: Instant, retry_connect: bool) -> Run {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = loop {
        if Instant::now() >= deadline {
            return Run::Lost("the VM's command line never answered".to_string());
        }
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => break stream,
            Err(error) if retry_connect => {
                // The VM is still coming up (the sandbox binds its
                // listener microseconds after the spawn): retry until the
                // deadline.
                std::thread::sleep(Duration::from_millis(100));
                let _ = error;
            }
            Err(error) => {
                return Run::Lost(format!(
                    "could not connect to the session's sandbox VM: {error}"
                ));
            }
        }
    };
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(READ_SLICE));
    let mut header = [0u8; HEADER_LEN];
    header[0] = MODE_ONESHOT;
    header[5..9].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    if let Err(error) = stream
        .write_all(&header)
        .and_then(|()| stream.write_all(payload))
        .and_then(|()| stream.flush())
    {
        return Run::Lost(format!("the command never reached the VM: {error}"));
    }
    let mut head: Vec<u8> = Vec::new();
    let mut tail: Vec<u8> = Vec::new();
    let mut total = 0usize;
    let mut chunk = [0u8; 8192];
    loop {
        // The deadline check is unconditional — at the loop top, not just
        // on an idle read: a command streaming output continuously would
        // otherwise never let the deadline arrive.
        if Instant::now() >= deadline {
            // Drop the connection: the RST is the kill signal the guest's
            // reaper acts on (a command's next write or read against the
            // dead line ends it). The VM keeps serving.
            return Run::TimedOut;
        }
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                let room = MAX_OUTPUT.saturating_sub(head.len());
                head.extend_from_slice(&chunk[..room.min(n)]);
                // The rolling tail: the last bytes of the stream, wherever
                // the cap falls — the trailer is validated at EOF against
                // exactly this. Only the n bytes actually read go in: the
                // buffer past them is stale zeros, never stream bytes.
                tail.extend_from_slice(&chunk[..n]);
                let keep_from = tail.len().saturating_sub(TRAILER_LEN);
                tail.drain(..keep_from);
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                return Run::Lost(format!("the VM's output stream failed: {error}"));
            }
        }
    }
    // EOF: the whole trailer must sit at the stream's end — the run is
    // not finished without it.
    let Some(status) = trailer_status(&tail) else {
        return Run::Lost(format!(
            "the command's output ended without a status trailer ({} bytes)",
            total
        ));
    };
    // The output is everything before the trailer; the cap keeps its
    // head and counts the rest.
    let output_len = total.saturating_sub(TRAILER_LEN);
    let kept = output_len.min(MAX_OUTPUT);
    head.truncate(kept);
    let dropped = output_len - kept;
    Run::Finished(RawRun {
        out: head,
        code: status as i32,
        dropped,
    })
}

/// The status byte of a whole trailer sitting at `tail`'s end, or None —
/// a prefix of the magic, a short read, or trailing junk is not a
/// finished run.
fn trailer_status(tail: &[u8]) -> Option<u8> {
    if tail.len() != TRAILER_LEN {
        return None;
    }
    let (status, magic) = tail.split_last()?;
    if magic == TRAILER_MAGIC.as_slice() {
        Some(*status)
    } else {
        None
    }
}

/// The readiness probe: one real command through the line. The VM is up
/// when the guest's vsockd has answered it — a broken deployment (the
/// sandbox dying at boot, a guest that never comes up) fails HERE,
/// loudly, instead of as a confusing timeout on the session's first
/// command. The probe retries connects (the sandbox binds its listener
/// microseconds after the spawn) and holds the line open until the
/// guest answers: the VMM parks connections that arrive before the
/// guest's vsockd signals ready, so a parked connect needs no special
/// handling — it simply completes when the guest is up.
fn probe_ready(vm: &mut Child, port: u16) -> Result<()> {
    let deadline = Instant::now() + BOOT_DEADLINE;
    loop {
        if let Ok(Some(status)) = vm.try_wait() {
            bail!(
                "the session VM exited at boot (status {}); see the platform log \
                 for the sandbox's own output",
                status.code().unwrap_or(-1)
            );
        }
        if Instant::now() >= deadline {
            bail!(
                "the session VM's command line never became ready; see the platform \
                 log for the sandbox's own output"
            );
        }
        match vsockd_run(port, b":", deadline, true) {
            Run::Finished(_) => return Ok(()),
            Run::TimedOut => bail!(
                "the session VM's command line never became ready; see the platform \
                 log for the sandbox's own output"
            ),
            // The connect was refused before the sandbox's listener was
            // bound: retry until the deadline.
            Run::Lost(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    }
}

/// The non-negative integer following a marker's first occurrence in the
/// job's output file: `JOB-EXIT:` and `JOB-START:` both ride that file
/// and are read the same way — first marker, digits to the first
/// non-digit.
fn digits_after_marker(output: &[u8], marker: &[u8]) -> Option<u64> {
    let at = find_subslice(output, marker, 0)?;
    let rest = &output[at + marker.len()..];
    let end = rest
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(rest.len());
    std::str::from_utf8(&rest[..end])
        .ok()
        .and_then(|digits| digits.parse().ok())
}

/// Strip the `JOB-START:<epoch>` stamp line job_start_line writes as the
/// output file's first line: harness bookkeeping, not the job's output.
/// Only a whole leading line is a stamp — a tail window that cuts the
/// file mid-stream leaves fragments that belong to the job's output.
fn strip_job_start_stamp(output: &[u8]) -> &[u8] {
    let Some(rest) = output.strip_prefix(b"JOB-START:") else {
        return output;
    };
    let digits = rest
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(rest.len());
    match rest.get(digits) {
        Some(b'\n') => &rest[digits + 1..],
        _ => output,
    }
}

/// The job's age from its start stamp (the epoch job_start_line writes
/// into its output file), or None without one. Guest and host clocks are
/// both UTC epoch seconds; skew only skews the age, and a negative one
/// clamps to zero rather than underflowing.
fn job_age_seconds(started_epoch: Option<u64>) -> Option<u64> {
    let start = started_epoch?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(now.saturating_sub(start))
}

/// A job's age, compact: `42s`, `5m03s`, `1h02m03s` — readable at a
/// glance mid-poll, minutes being the unit a wait-or-give-up decision
/// turns on.
fn format_job_age(secs: u64) -> String {
    if secs < 60 {
        return format!("{secs}s");
    }
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else {
        format!("{minutes}m{seconds:02}s")
    }
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

    /// A fake sandbox binary: the host half of the command line. It
    /// dumps its own argv (the spawn args — the layers, the shares, the
    /// sizing — are the only fake-level observation of what the pool
    /// asked the sandbox for), binds the --port listener, and serves the
    /// vsockd wire protocol: per connection the 9-byte header, the
    /// length-prefixed command, the command run host-side under /bin/sh
    /// with stderr merged, and the 17-byte status trailer at the end. A
    /// thread per connection — the concurrency the real line's
    /// fork-per-connection guest delivers.
    fn fake_sandal(dir: &Path) -> PathBuf {
        let path = dir.join("fake-sandal");
        let script = format!(
            "#!/usr/bin/python3\n\
             import socket, struct, subprocess, sys, threading\n\
             DUMP = '{dump}'\n\
             MAGIC = b'VSOCKD:EXIT:01\\n\\0'\n\
             argv = sys.argv[1:]\n\
             port = None\n\
             for i, arg in enumerate(argv):\n\
             \x20   if arg == '--port':\n\
             \x20       port = int(argv[i + 1].rsplit(':', 1)[-1])\n\
             if DUMP:\n\
             \x20   with open(DUMP, 'w') as f:\n\
             \x20       f.write('\\n'.join(argv) + '\\n')\n\
             def handle(conn):\n\
             \x20   try:\n\
             \x20       header = b''\n\
             \x20       while len(header) < 9:\n\
             \x20           chunk = conn.recv(9 - len(header))\n\
             \x20           if not chunk:\n\
             \x20               return\n\
             \x20           header += chunk\n\
             \x20       _mode, _rows, _cols, cmd_len = struct.unpack('<BHHI', header)\n\
             \x20       payload = b''\n\
             \x20       while len(payload) < cmd_len:\n\
             \x20           chunk = conn.recv(cmd_len - len(payload))\n\
             \x20           if not chunk:\n\
             \x20               return\n\
             \x20           payload += chunk\n\
             \x20       proc = subprocess.Popen(['/bin/sh', '-c', payload.decode()],\n\
             \x20           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n\
             \x20       output = proc.stdout.read()\n\
             \x20       code = proc.wait()\n\
             \x20       status = (128 - code) & 0xFF if code < 0 else code & 0xFF\n\
             \x20       conn.sendall(output)\n\
             \x20       conn.sendall(MAGIC + bytes([status]))\n\
             \x20   except OSError:\n\
             \x20       pass\n\
             \x20   finally:\n\
             \x20       conn.close()\n\
             server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n\
             server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n\
             server.bind(('127.0.0.1', port))\n\
             server.listen(128)\n\
             while True:\n\
             \x20   conn, _ = server.accept()\n\
             \x20   threading.Thread(target=handle, args=(conn,), daemon=True).start()\n",
            dump = dir.join("sandal.args").display(),
        );
        fs::write(&path, script).unwrap();
        fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        path
    }

    /// A pool with the fake sandbox and the given idle timeout. The
    /// settings resolve through the real path, so the resolve path is
    /// under test too. The ssh era needed a key and a server layer
    /// wiring; the vsock line needs nothing but the sandbox binary.
    fn pool(dir: &Path, idle_timeout_secs: u64) -> Arc<VmPool> {
        let raw = settings(&format!(
            "[agent.sandal]\nidle_timeout_secs = {idle_timeout_secs}\n"
        ));
        let resolved = raw.resolve(Some(dir)).unwrap();
        Arc::new(VmPool::new(resolved, fake_sandal(dir)))
    }

    fn call(pool: &VmPool, session: &str, command: &str) -> Result<String> {
        call_with_roots(pool, session, command, &[])
    }

    /// A session scope for pool tests: the given cwd, no declared roots
    /// unless the test passes them.
    fn test_scope(cwd: &Path, roots: &[String]) -> SessionScope {
        SessionScope::new(
            cwd.to_string_lossy().into_owned(),
            roots.iter().map(PathBuf::from),
        )
    }

    /// The run loop's half of the protocol: boot the session's VM with the
    /// calling agent's declared write roots (empty for an agent that
    /// declared none — no root shares, and the same empty list on every
    /// later call), then the command.
    fn call_with_roots(
        pool: &VmPool,
        session: &str,
        command: &str,
        roots: &[String],
    ) -> Result<String> {
        let cwd = std::env::current_dir().unwrap();
        pool.ensure(session, &test_scope(&cwd, roots))?;
        pool.execute(
            session,
            &json!({"command": command, "_session_cwd": cwd, "_session_write_roots": roots}),
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

    /// The ssh era's settings (the session key, the sshd layer) are gone
    /// with the ssh channel: a config that still carries them fails the
    /// parse loudly — never a silently ignored key over a VM that boots
    /// without the security the operator asked for.
    #[test]
    fn ssh_era_settings_are_rejected() {
        let root: toml::Value = toml::from_str(
            "[agent.sandal]\nssh_key = '/keys/vm'\nssh_server_layer = 'ssh-server.layer'\n",
        )
        .unwrap();
        let error = root["agent"]["sandal"]
            .clone()
            .try_into::<SandalAgentSettings>()
            .unwrap_err();
        assert!(error.to_string().contains("ssh_key"), "{error}");
    }

    #[test]
    fn empty_layers_keep_default_sizing() {
        // No toolchain layers: sizing is the defaults anyway — a bare VM
        // runs trivial commands fine at them, and a toolchain VM gets the
        // same floor unless it asks for more.
        let dir = unique_test_dir();
        let resolved = settings(
            "[agent.sandal]\nlayers = []\nshared = ['~/go/pkg:~/go/pkg']\nmemory = 65536\n",
        )
        .resolve(Some(dir.path()))
        .unwrap();
        assert!(resolved.layers.is_empty());
        assert_eq!(resolved.memory_mb, 65536);
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
        let resolved = settings("[agent.sandal]\nlayers = []\nshared = ['cache:/root/cache']\n")
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
        assert_eq!(resolved.memory_mb, DEFAULT_MEMORY_MB);
        assert_eq!(resolved.disk_mb, DEFAULT_DISK_MB);
    }

    #[test]
    fn explicit_sizing_wins_over_the_defaults() {
        let dir = unique_test_dir();
        std::fs::write(dir.path().join("go.layer"), "layer").unwrap();

        let resolved =
            settings("[agent.sandal]\nlayers = ['go.layer']\nmemory = 2048\ndisk = 4096\n")
                .resolve(Some(dir.path()))
                .unwrap();

        assert_eq!(resolved.memory_mb, 2048);
        assert_eq!(resolved.disk_mb, 4096);
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
        assert_eq!(resolved.layers, vec![dir.path().join("images/go.layer")]);
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
        let configured =
            settings("[agent.sandal]\nlayers = ['go.layer']\nidle_timeout_secs = 60\n")
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
        let pid_a = lock_state(&a.state).vm.id();
        let pid_b = lock_state(&b.state).vm.id();
        assert_ne!(pid_a, pid_b, "two sessions share one process");
        drop(entries);
        // And both keep serving their own sessions.
        assert!(call(&pool, "session-a", "echo a2").unwrap().contains("a2"));
        assert!(call(&pool, "session-b", "echo b2").unwrap().contains("b2"));
    }

    /// A command's signal death rides the trailer (the shell convention:
    /// 128+signal) — and it kills the COMMAND, not the session: the VM
    /// keeps serving the next call.
    #[test]
    fn a_signal_death_rides_the_trailer_and_the_vm_survives() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "kill -9 $$").unwrap();
        assert!(result.contains("exit code: 137"), "{result}");
        // The VM survived its command's signal death.
        let fresh = call(&pool, "session-1", "echo still-here").unwrap();
        assert!(fresh.contains("still-here"), "{fresh}");
    }

    /// A foreground timeout kills the COMMAND (the connection drops, the
    /// guest's own reaper acts on the RST) — never the whole VM. The old
    /// ssh-channel behavior closed the VM and made the next call boot
    /// fresh; the vsock line's per-command connections retire that.
    #[test]
    fn a_timeout_kills_the_command_but_not_the_vm() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap();
        pool.ensure("session-1", &test_scope(&cwd, &[])).unwrap();
        let timed_out = pool
            .execute(
                "session-1",
                &json!({
                    "command": "sleep 30",
                    "timeout_secs": 1,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap_err();
        assert!(timed_out.to_string().contains("timed out"), "{timed_out}");
        // The same VM keeps serving: not a fresh boot, the same process.
        let pid_before = lock_state(&lock(&pool.entries).get("session-1").unwrap().state)
            .vm
            .id();
        let fresh = call(&pool, "session-1", "echo survived-timeout").unwrap();
        assert!(fresh.contains("survived-timeout"), "{fresh}");
        let pid_after = lock_state(&lock(&pool.entries).get("session-1").unwrap().state)
            .vm
            .id();
        assert_eq!(pid_before, pid_after, "the timeout closed the VM");
    }

    /// A command that streams output continuously (never idle) still
    /// hits the deadline: the check is unconditional, not idle-only — a
    /// silent deadline would never arrive on a chatty stream.
    #[test]
    fn a_streaming_command_still_hits_the_deadline() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap();
        pool.ensure("session-1", &test_scope(&cwd, &[])).unwrap();
        let started = Instant::now();
        let error = pool
            .execute(
                "session-1",
                &json!({
                    "command": "yes",
                    "timeout_secs": 1,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the streaming command outlived its deadline: {:?}",
            started.elapsed()
        );
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

    /// The per-agent session-directory share: the roots an agent declared
    /// for its session ride the call (the same list its write tools
    /// permit) and into the session's own VM — every root, same path on
    /// both sides, exactly like the working directory.
    #[test]
    fn declared_write_roots_are_shared_into_the_session_vm() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let root_a = dir.path().join("session-root-a");
        let root_b = dir.path().join("session-root-b");
        fs::create_dir_all(&root_a).unwrap();
        fs::create_dir_all(&root_b).unwrap();
        let roots = [root_a.display().to_string(), root_b.display().to_string()];
        // A fine command first: the VM boots with the declared roots.
        let ok = call_with_roots(&pool, "session-1", "echo booted", &roots).unwrap();
        assert!(ok.contains("booted"));
        // The spawn asked for every declared root, same path on both sides.
        let args = fs::read_to_string(dir.path().join("sandal.args")).unwrap();
        for root in &roots {
            assert!(
                args.contains(&format!("--share\n{root}:{root}")),
                "the spawn never asked to share declared root {root}: {args}"
            );
        }
    }

    /// The roots are pinned at boot like the working directory: a later
    /// call presenting a different list is an error, not a reshare — and
    /// the VM keeps serving the list it booted with.
    #[test]
    fn a_changed_write_root_list_is_an_error_not_a_reshare() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let root = dir.path().join("session-root");
        fs::create_dir_all(&root).unwrap();
        let roots = [root.display().to_string()];
        let first = call_with_roots(&pool, "session-1", "echo first", &roots).unwrap();
        assert!(first.contains("first"));
        // A different root, the same root dropped, a root added — every
        // shape is a mismatch, never a renegotiation.
        let other = dir.path().join("other-root");
        fs::create_dir_all(&other).unwrap();
        for changed in [
            vec![other.display().to_string()],
            vec![],
            vec![roots[0].clone(), other.display().to_string()],
        ] {
            let error = call_with_roots(&pool, "session-1", "echo changed", &changed).unwrap_err();
            assert!(
                error.to_string().contains("bound to its first call's"),
                "a changed root list was not an error: {error}"
            );
        }
        // And the reverse: a session that declared none cannot add one.
        let none = call(&pool, "session-2", "echo none").unwrap();
        assert!(none.contains("none"));
        let error = call_with_roots(&pool, "session-2", "echo added", &roots).unwrap_err();
        assert!(
            error.to_string().contains("bound to its first call's"),
            "adding a root to a rootless session was not an error: {error}"
        );
        // The pinned VM still serves its own list.
        let still = call_with_roots(&pool, "session-1", "echo still", &roots).unwrap();
        assert!(still.contains("still"));
    }

    /// A declared root that is not a directory fails the boot, loudly:
    /// the declaration is the calling agent's own, and a silently dropped
    /// root would surface later as a confusing path error inside the VM.
    #[test]
    fn a_missing_declared_root_fails_the_boot() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let missing = dir.path().join("no-such-root").display().to_string();
        let error = call_with_roots(
            &pool,
            "session-1",
            "echo boot",
            std::slice::from_ref(&missing),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("not a directory") && error.to_string().contains(&missing),
            "a missing declared root did not fail the boot: {error}"
        );
        // Nothing was booted: the session has no VM.
        assert!(lock(&pool.entries).get("session-1").is_none());
    }

    #[test]
    fn output_is_capped_and_counted() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "seq 1 20000").unwrap();
        // The head is kept (bounded by MAX_OUTPUT), the rest is counted,
        // the exit code rides the trailer.
        assert!(result.starts_with("exit code: 0"));
        assert!(result.len() < MAX_OUTPUT + 2_000, "{}", result.len());
        assert!(result.contains("output truncated"));
    }

    #[test]
    fn nonzero_exit_codes_ride_the_trailer() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "exit 42").unwrap();
        assert!(result.contains("exit code: 42"), "{result}");
    }

    /// A command's stderr rides the line, merged into its own output:
    /// the host shell tool's contract (stdout, stderr, exit code) — the
    /// command's stdio IS the connection, so its diagnostics land with
    /// its output instead of streaming to the platform log.
    #[test]
    fn a_commands_stderr_rides_the_protocol() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let result = call(&pool, "session-1", "echo err-text 1>&2").unwrap();
        assert!(
            result.contains("err-text"),
            "the command's stderr did not ride the protocol: {result}"
        );
    }

    /// A command bigger than the line's own limit is refused HERE, with
    /// the reason named — never sent to die silently in the guest's
    /// protocol check.
    #[test]
    fn a_command_larger_than_the_line_limit_is_rejected() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap();
        pool.ensure("session-1", &test_scope(&cwd, &[])).unwrap();
        let big = "x".repeat(MAX_COMMAND_BYTES);
        let error = pool
            .execute(
                "session-1",
                &json!({"command": format!("echo {big}"), "_session_cwd": cwd}),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("at most"),
            "an oversized command was not refused with the limit named: {error}"
        );
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
            pool_a.ensure("session-a", &test_scope(&cwd, &[])).unwrap();
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

    /// The concurrency the vsock line exists for, inside ONE session:
    /// the old ssh channel was one serialized session shell — a
    /// session's own commands queued behind each other on the state
    /// lock. A connection per command retires that: a slow command in a
    /// session must not delay the session's next call.
    #[test]
    fn same_session_commands_run_concurrently() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir().unwrap();
        pool.ensure("session-1", &test_scope(&cwd, &[])).unwrap();
        let pool_slow = Arc::clone(&pool);
        let slow = std::thread::spawn(move || {
            pool_slow
                .execute(
                    "session-1",
                    &json!({
                        "command": "sleep 3; echo slow-done",
                        "timeout_secs": 30,
                        "_session_cwd": cwd,
                    }),
                )
                .unwrap()
        });
        std::thread::sleep(Duration::from_millis(500));
        let started = Instant::now();
        let fast = call(&pool, "session-1", "echo fast").unwrap();
        let elapsed = started.elapsed();
        assert!(fast.contains("fast"), "{fast}");
        assert!(
            elapsed < Duration::from_secs(2),
            "the session's own next call waited {elapsed:?} behind its slow command"
        );
        assert!(slow.join().unwrap().contains("slow-done"));
    }

    /// The reaper never kills a running command's VM, whatever the idle
    /// timeout says — the running count is the busy signal (the state
    /// lock is not held for a run).
    #[test]
    fn the_reaper_skips_running_commands() {
        let dir = unique_test_dir();
        // Idle timeout 0: everything idle closes on the next pass — the
        // running command's VM must survive even that.
        let pool = pool(dir.path(), 0);
        let pool_a = Arc::clone(&pool);
        let a = std::thread::spawn(move || {
            let cwd = std::env::current_dir().unwrap();
            pool_a.ensure("session-1", &test_scope(&cwd, &[])).unwrap();
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
        // Wait until the command is actually in flight (the entry's
        // running count is the reaper's own busy signal), so the reap
        // below tests the guarantee, not a boot race under load.
        let started = Instant::now();
        loop {
            let running = lock(&pool.entries)
                .get("session-1")
                .map(|entry| entry.running.load(Ordering::SeqCst))
                .unwrap_or(0);
            if running > 0 {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the command never started"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
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
    /// command completed fine) is detected at the next call's connect in
    /// one round, not retried against for the whole timeout: the entry
    /// is removed and the session's next call boots fresh.
    #[test]
    fn a_vm_that_dies_between_calls_is_detected_fast() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        // A fine first command — the VM looks healthy.
        assert!(
            call(&pool, "session-1", "echo healthy")
                .unwrap()
                .contains("healthy")
        );
        // The VM dies out-of-band (the memory cap OOM-killed the guest).
        {
            let entries = lock(&pool.entries);
            let entry = entries.get("session-1").unwrap();
            close_state(&mut lock_state(&entry.state));
        }
        // The next call's connect hits the corpse: detected in one round.
        let started = Instant::now();
        let error = call(&pool, "session-1", "echo next").unwrap_err();
        assert!(
            error.to_string().contains("exited during the command"),
            "{error}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the dead-VM detection took {:?}",
            started.elapsed()
        );
        // The entry is gone: the next call boots a fresh VM.
        assert!(lock(&pool.entries).get("session-1").is_none());
        assert!(
            call(&pool, "session-1", "echo fresh-vm")
                .unwrap()
                .contains("fresh-vm")
        );
    }

    /// The background-job path: a job that outlives any transport starts
    /// detached inside the VM (here: the fake's host-side process), is
    /// polled by id, and reports its own exit code when it finishes.
    #[test]
    fn background_jobs_start_poll_and_finish() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
            .unwrap();
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
        assert!(
            start.starts_with("Background job started: potlatch-job-"),
            "{start}"
        );
        let job = start
            .lines()
            .find_map(|l| l.strip_prefix("Background job started: "))
            .unwrap()
            .to_string();
        // Poll while it runs: no exit code, no output, and the job's age
        // from its start stamp.
        let running = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        assert!(running.contains("(still running, "), "{running}");
        std::thread::sleep(Duration::from_secs(2));
        let finished = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        assert!(finished.contains("exit code: 0"), "{finished}");
        assert!(finished.contains("job-done"), "{finished}");
    }

    /// A running job's poll carries its age: the stamp job_start_line
    /// writes into the job's output file is what turns a poll from a
    /// numberless wait into one a model can give up on. The regression:
    /// a silent job polled for half an hour read the same at every poll
    /// — "(still running)", "(no output yet)" — so the model slept in a
    /// loop forever.
    #[test]
    fn a_running_jobs_poll_reports_its_age() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
            .unwrap();
        let start = pool
            .execute(
                "session-1",
                &json!({"command": "sleep 5", "background": true, "_session_cwd": cwd}),
            )
            .unwrap();
        let job = start
            .lines()
            .find_map(|l| l.strip_prefix("Background job started: "))
            .unwrap()
            .to_string();
        std::thread::sleep(Duration::from_secs(2));
        let running = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        let age = running
            .strip_prefix("exit code: (still running, ")
            .and_then(|rest| rest.strip_suffix(")\n\n(no output yet)"))
            .unwrap_or_else(|| panic!("running poll must carry the job's age: {running}"));
        // Two seconds waited, one second of format granularity: the age
        // reflects elapsed time, it is not a constant zero.
        let secs = age.trim_end_matches('s').parse::<u64>().unwrap_or(0);
        assert!(secs >= 1, "age must reflect the wait: {running}");
    }

    /// A killed job is gone: best-effort kill (its own shell), then the
    /// poll finds nothing — no output, no exit code, ever.
    #[test]
    fn background_jobs_kill_best_effort() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
            .unwrap();
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
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
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
            let cwd = std::env::current_dir()
                .unwrap()
                .to_string_lossy()
                .to_string();
            pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
                .unwrap();
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
    /// next command rides its own connection — so nothing about the job
    /// can stall or pollute it.
    #[test]
    fn a_foreground_command_runs_while_a_background_job_is_running() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
            .unwrap();
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
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        assert!(running.contains("(still running, "), "{running}");
        // Kill the job; the poll finds nothing after.
        let killed = pool
            .execute(
                "session-1",
                &json!({"job_id": job, "kill": true, "_session_cwd": cwd}),
            )
            .unwrap();
        assert!(killed.contains("killed"), "{killed}");
        let after = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        assert!(after.contains("(no output yet)"), "{after}");
    }

    /// Multiple background jobs run at the same time: the starts serialize
    /// on nothing (each its own connection), and the jobs themselves run
    /// concurrently — one per command, none waiting behind another.
    #[test]
    fn multiple_background_jobs_run_concurrently() {
        let dir = unique_test_dir();
        let pool = pool(dir.path(), 3600);
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &test_scope(Path::new(&cwd), &[]))
            .unwrap();
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
        // All three starts returned — in seconds, not behind any job.
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
                .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
                .unwrap();
            assert!(running.contains("(still running, "), "{running}");
        }
        // None finished behind another: all three report their own exit
        // within the same window after the sleep.
        std::thread::sleep(Duration::from_secs(9));
        for (i, job) in jobs.iter().enumerate() {
            let finished = pool
                .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
                .unwrap();
            assert!(
                finished.contains(&format!("job-{i}-done")),
                "job {i}: {finished}"
            );
            assert!(finished.contains("exit code: 0"), "job {i}: {finished}");
        }
    }

    #[test]
    fn job_start_stamp_is_stripped_as_a_whole_leading_line() {
        assert_eq!(
            strip_job_start_stamp(b"JOB-START:1729584000\njob output\n"),
            b"job output\n"
        );
        // Not a stamp: a different prefix, a non-numeric stamp line, or a
        // tail window that starts mid-line — the job's own output is kept.
        assert_eq!(
            strip_job_start_stamp(b"JOB-STARTX:1729584000\nout\n"),
            b"JOB-STARTX:1729584000\nout\n"
        );
        assert_eq!(
            strip_job_start_stamp(b"JOB-START:not-a-number\nout\n"),
            b"JOB-START:not-a-number\nout\n"
        );
        assert_eq!(strip_job_start_stamp(b"9584000\nout\n"), b"9584000\nout\n");
    }

    #[test]
    fn job_age_formats_in_units_a_wait_turns_on() {
        assert_eq!(format_job_age(0), "0s");
        assert_eq!(format_job_age(42), "42s");
        assert_eq!(format_job_age(65), "1m05s");
        assert_eq!(format_job_age(27 * 60), "27m00s");
        assert_eq!(format_job_age(3600 + 125), "1h02m05s");
    }

    // ── The line itself ────────────────────────────────────────────────

    /// The trailer is validated WHOLE at the stream's end: a prefix of
    /// the magic, a short tail, or a corrupted magic byte is not a
    /// finished run — the run is lost, not misread. (The trailer is the
    /// stream's own last 17 bytes: magic, then the status byte — any
    /// byte after the magic IS a status, never junk to reject.)
    #[test]
    fn the_trailer_is_validated_whole() {
        assert_eq!(trailer_status(&[0u8; 17]), None);
        assert_eq!(trailer_status(b"VSOCKD:EXIT:01\n\0"), None);
        let mut corrupt = TRAILER_MAGIC.to_vec();
        corrupt.insert(3, b'X');
        corrupt.push(7);
        assert_eq!(trailer_status(&corrupt), None);
        let mut whole = TRAILER_MAGIC.to_vec();
        whole.push(42);
        assert_eq!(trailer_status(&whole), Some(42));
    }

    /// The foreground payload: the command rides verbatim behind the cd
    /// guard (the line is length-prefixed — nothing needs quoting), and
    /// a cwd with a shell metacharacter still guards correctly.
    #[test]
    fn the_foreground_payload_rides_verbatim_behind_the_cd_guard() {
        let payload = foreground_payload("/tmp/it's", "echo multi\nline\nexit 3\n");
        assert!(
            payload.starts_with("cd '/tmp/it'\\''s' 2>/dev/null || exit 127\n"),
            "the cd guard did not quote the cwd: {payload}"
        );
        assert!(
            payload.ends_with("echo multi\nline\nexit 3\n"),
            "the command did not ride verbatim: {payload}"
        );
    }
}
