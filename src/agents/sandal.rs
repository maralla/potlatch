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
/// Maximum per-command timeout (seconds): the request's own deadline
/// (30 minutes — the bus deadline that carries the response). A request
/// past it is capped, not honored: on the cap the whole VM closes (the
/// timeout path) and the session's next call boots a fresh one, so the
/// session recovers — an uncapped hour-long command would block every
/// retry behind it while its own response was already dropped at the
/// deadline.
const MAX_TIMEOUT_SECS: u64 = 30 * 60;
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
/// boots in well under a second; a layerchain VM takes a few).
const READY_DEADLINE: Duration = Duration::from_secs(30);
/// The session shell wrapper: print the ready marker, then eval one
/// protocol line per input line for the VM's whole life. Spawned as `sh
/// -c` (never `sh -i`): no interactive prompt ever lands in the stream,
/// and the wrapper is the only thing the guest shell ever runs directly.
/// Each eval merges its command's stderr into the stream (`2>&1`): the
/// model sees its own command's stderr (the host shell tool's contract
/// — stdout, stderr, exit code), and nothing streams to the inherited
/// ssh stderr (the platform log) but ssh's own connection errors.
/// Runs inside the VM over the SSH channel (see [`VmState::ssh`]): the
/// console wrapper below only bootstraps sshd and keeps the VM's console
/// session alive; every protocol line rides SSH.
const SESSION_WRAPPER: &str = "echo __POTLATCH_VM_SESSION_READY__; \
while IFS= read -r line; do eval \"$line\" 2>&1; done";
/// The marker the console's bootstrap line prints once the guest's sshd
/// is up and the session key is in place: the agent waits for it once
/// per VM, then never touches the console again (the console pipes are
/// held for the VM's life — dropping them closes the console session,
/// and sandal tears the VM down with it).
const BOOTSTRAP_MARKER: &str = "__POTLATCH_VM_BOOTSTRAP_READY__";
/// How long a booting VM may take to reach the bootstrap marker (the
/// guest boots in well under a second; a layerchain VM takes a few).
const BOOTSTRAP_DEADLINE: Duration = Duration::from_secs(60);
/// The console command: one sshd-setup line (the session key in, sshd
/// started on the guest's own loopback — the vsock channel the VMM's
/// --ssh-port listener bridges to), the marker, then a keepalive. The
/// keepalive holds the console session open for the VM's life — sandal
/// tears the VM down when the console session dies, so the keepalive IS
/// the VM's life from the console side; it produces nothing, so the held
/// pipes never fill. Everything after the marker rides the SSH channel.
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
const SHELL_DESCRIPTION: &str = "Run a shell command inside the session's sandbox VM. You are already in the working directory (cwd) — no need to `cd` into it. The VM is reused across this session's calls: packages installed by an earlier call are still installed, and files written by an earlier call are still there. Returns the command's output and exit code. Foreground commands have a timeout (default 120s, maximum 30 minutes); on timeout the whole VM is closed and the session's next call boots a fresh one — anything longer than a minute or two belongs in the background: set background: true, get a job id back immediately, and poll it with job_id (or kill it with job_id + kill: true). Do NOT use this tool to create or edit files (no `cat >`, `echo >`, `sed -i`, `tee`) — use `write` or `edit` instead.";

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
    /// Path to the SSH private key the session shells authenticate with.
    /// The public key is read from `<ssh_key>.pub` (or derived with
    /// `ssh-keygen -y`) and injected into the guest's sshd at boot —
    /// the session's shell rides the SSH channel, never the console.
    /// Paths resolve like layers: `~` expands, relative paths resolve
    /// against the config directory.
    pub ssh_key: Option<String>,
    /// Path to the SSH server layer (sshd + sshd-setup + one host key)
    /// every session VM boots with — required: the session's shell rides
    /// the SSH channel this layer's sshd opens. Resolves like the
    /// toolchain layers: relative paths resolve against the config
    /// directory.
    pub ssh_server_layer: Option<String>,
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
    /// The SSH private key the session shells authenticate with, and the
    /// public key material injected into the guest's sshd at boot.
    pub ssh_key: PathBuf,
    pub ssh_public: String,
    /// The SSH server layer (sshd + sshd-setup + one host key):
    /// configured like the toolchain layers (required) and always
    /// carried — every session VM boots with it, and the session's
    /// shell rides the SSH channel its sshd opens.
    pub ssh_server_layer: PathBuf,
    /// The session VM's memory cap (MB). Always resolved: every session
    /// VM carries the SSH server layer (sized for it alone), so the
    /// `layers`-gated sizing of the console era is gone.
    pub memory_mb: u64,
    /// The session VM's writable overlay disk (MB). Always resolved.
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
    ///
    /// Sizing applies only when layers are configured: a bare VM (no
    /// `layers`) runs trivial commands fine at the sandbox's own
    /// defaults, and a toolchain VM gets `memory`/`disk` defaults sized
    /// for real builds (see [`DEFAULT_MEMORY_MB`]/[`DEFAULT_DISK_MB`]).
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
        // The SSH channel: the session's shell rides it (never the
        // console), so the key and the server layer are resolved here,
        // once, the same way the sandbox binary is resolved once.
        let ssh_key = resolve_ssh_key(self.ssh_key.as_deref(), config_dir)?;
        let ssh_public = derive_public_key(&ssh_key)?;
        let ssh_server_layer =
            resolve_ssh_server_layer(self.ssh_server_layer.as_deref(), config_dir)?;
        // Sizing is always on: every session VM carries the SSH server
        // layer, so the `layers`-gated sizing of the console era is
        // gone — the defaults exist for it alone, and a toolchain VM
        // gets the same floor.
        Ok(ResolvedSettings {
            layers,
            shares,
            ssh_key,
            ssh_public,
            ssh_server_layer,
            memory_mb: self.memory.unwrap_or(DEFAULT_MEMORY_MB),
            disk_mb: self.disk.unwrap_or(DEFAULT_DISK_MB),
            idle_timeout: Duration::from_secs(
                self.idle_timeout_secs.unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
            ),
        })
    }
}

/// Resolve the configured SSH key path: `~` expands, relative paths
/// resolve against the config directory (like layers), and a missing
/// key is an error, not a warning — without it no session VM can boot.
fn resolve_ssh_key(configured: Option<&str>, config_dir: &Path) -> Result<PathBuf> {
    let configured = configured.context(
        "[agent.sandal] ssh_key is required: the session's shell rides the \
         SSH channel, and the session key is what sshd authenticates",
    )?;
    let home = crate::paths::home_dir();
    let path = expand_home(configured, &home);
    let path = if path.is_absolute() {
        path
    } else {
        config_dir.join(path)
    };
    let path = path
        .canonicalize()
        .with_context(|| format!("[agent.sandal] ssh_key not found: {configured}"))?;
    ensure!(
        path.is_file(),
        "[agent.sandal] ssh_key is not a file: {configured}"
    );
    Ok(path)
}

/// Derive the public key material from the private key: read
/// `<key>.pub` when it exists (the standard pairing), else derive it
/// with `ssh-keygen -y` (a passphrase-less key). One line, exactly what
/// sshd-setup injects.
fn derive_public_key(key: &Path) -> Result<String> {
    // <key>.pub: the sibling file (id_ed25519 -> id_ed25519.pub), the
    // standard pairing.
    let mut s = key.as_os_str().to_os_string();
    s.push(".pub");
    let pub_path = PathBuf::from(s);
    if pub_path.is_file() {
        let public = std::fs::read_to_string(&pub_path)
            .with_context(|| format!("failed to read {}", pub_path.display()))?;
        let public = public.trim().to_string();
        ensure!(
            !public.is_empty(),
            "[agent.sandal] ssh_key's .pub file is empty"
        );
        return Ok(public);
    }
    let output = Command::new("ssh-keygen")
        .arg("-y")
        .arg("-f")
        .arg(key)
        .output()
        .context("failed to run ssh-keygen (is it installed?)")?;
    ensure!(
        output.status.success(),
        "ssh-keygen -y failed (passphrase-protected key?): {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let public = String::from_utf8_lossy(&output.stdout).trim().to_string();
    ensure!(!public.is_empty(), "ssh-keygen -y produced no public key");
    Ok(public)
}

/// Resolve the configured SSH server layer path: relative paths resolve
/// against the config directory — it is a layer (the SSH server one),
/// resolved exactly like the toolchain layers — and a missing or
/// non-file layer is an error, not a warning: without it no session VM
/// can boot, and the session's shell rides the SSH channel its sshd
/// opens.
fn resolve_ssh_server_layer(configured: Option<&str>, config_dir: &Path) -> Result<PathBuf> {
    let configured = configured.context(
        "[agent.sandal] ssh_server_layer is required: every session VM \
         boots with the SSH server (sshd + sshd-setup), and the session's \
         shell rides the SSH channel it opens",
    )?;
    let path = config_dir.join(configured);
    let path = path
        .canonicalize()
        .with_context(|| format!("[agent.sandal] ssh_server_layer not found: {configured}"))?;
    ensure!(
        path.is_file(),
        "[agent.sandal] ssh_server_layer is not a file: {configured}"
    );
    Ok(path)
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
            let boot = session_context(&request.payload)
                .and_then(|(cwd, roots)| self.pool.ensure(session_id, &cwd, &roots));
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
        let ssh_bin = resolve_ssh_bin().context("[agent.sandal] requires the ssh binary")?;
        let pool = Arc::new(VmPool::new(settings, bin, ssh_bin));
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
                            "description": "Timeout in seconds for foreground commands. Default: 120, maximum: 1800 (30 minutes) — a request past the maximum is capped to it. On timeout the whole VM is closed and the session's next call boots a fresh one. Ignored for background jobs (they run until exit or kill)."
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

/// Resolve the ssh binary: `POTLATCH_SSH_BIN` overrides, else `ssh`
/// on PATH. Resolved once at build so a missing binary is a startup
/// error, not a per-call one.
fn resolve_ssh_bin() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("POTLATCH_SSH_BIN") {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "POTLATCH_SSH_BIN is not a file: {}",
            path.display()
        );
        return Ok(path);
    }
    Ok(PathBuf::from("ssh"))
}

/// Allocate one SSH channel port: bind 127.0.0.1:0, take the port,
/// drop the listener. The window between the drop and the sandbox's own
/// bind is microseconds; a collision in it fails the sandbox's bind
/// loudly (the process exits with the error), never silently.
fn allocate_ssh_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .context("failed to allocate an SSH channel port")?;
    let port = listener
        .local_addr()
        .context("failed to read the allocated SSH channel port")?
        .port();
    drop(listener);
    Ok(port)
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

/// The session's shell-call context: the working directory and the
/// declared write roots (the same per-agent list the harness's write/edit
/// tools permit outside the cwd — e.g. the QA agent's session directory,
/// where its test scripts live). Both are pinned on the session's first
/// call: a later call presenting a different pair is an error, not a
/// renegotiation and not a second VM.
fn session_context(payload: &Value) -> Result<(String, Vec<String>)> {
    let cwd = payload
        .get("_session_cwd")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .context("the shell call carries no session working directory")?;
    let roots = payload
        .get("_session_write_roots")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok((cwd, roots))
}

/// One live session VM. `cwd`/`roots` are pinned at spawn and readable
/// without the state lock; everything mutable lives behind one state lock
/// that a command holds for its whole run.
struct VmEntry {
    /// The working directory this VM was booted with — pinned on the
    /// session's first call and never renegotiated: every later call
    /// lands in the same VM, so a session that somehow presents a
    /// different directory gets an error, not a second VM.
    cwd: String,
    /// The declared write roots this VM was booted with — pinned like
    /// [`Self::cwd`]: the session's out-of-cwd work must resolve the
    /// same way on every call, so a later call presenting a different
    /// list gets the same error, not a reshare.
    roots: Vec<String>,
    /// The whole mutable state behind one lock: held for an entire
    /// command, so same-session commands serialize while cross-session
    /// commands run concurrently; only ever `try_lock`ed by the idle
    /// reaper and close-all, where a held lock means a running command
    /// and they skip it.
    state: Mutex<VmState>,
}

/// The mutable state of one live session VM: the VM process, the
/// console (held after the bootstrap, never used again), and the SSH
/// session that carries every protocol line.
struct VmState {
    /// The sandbox process — the VM itself. Lives for the entry's whole
    /// life; killed in [`close_state`].
    vm: Child,
    /// The console pipes, held for the VM's life: dropping them closes
    /// the console session (sandal tears the VM down with it), and the
    /// bootstrap's keepalive holds it open. Taken and dropped in
    /// [`close_state`] — the console session dies first, sandal's own
    /// teardown runs, and the kills below are the belt. Everything
    /// between rides the SSH channel.
    console: Option<(ChildStdin, ChildStdout)>,
    /// The SSH session process: the session shell (the wrapper inside
    /// it) and its pipes. Its death is the session's death — the entry
    /// is removed and the VM is closed with it.
    ssh: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
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
    /// The ssh binary (the session channel's client): resolved once at
    /// build (see [`resolve_ssh_bin`]), held here so the spawn never
    /// resolves and the tests inject a fake.
    ssh_bin: PathBuf,
    entries: Mutex<HashMap<String, Arc<VmEntry>>>,
}

impl VmPool {
    fn new(settings: ResolvedSettings, bin: PathBuf, ssh_bin: PathBuf) -> Self {
        Self {
            settings,
            bin,
            ssh_bin,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Boot a session's VM: the configured layers and sizing, the session's
    /// working directory, and the configured persistent shares whose host
    /// side exists (a share for a cache directory that has not been
    /// created yet is dropped with a warning, and appears once it has).
    fn spawn(&self, cwd: &str, roots: &[String]) -> Result<VmEntry> {
        // (a) The sandbox process: --ssh-port (the listener the session's
        // ssh connects to, bridged through vsock to the guest's own
        // loopback where the bootstrap's sshd listens), the layers (the
        // configured toolchains plus the SSH server layer), the shares,
        // the sizing, and the console bootstrap command. Take the console
        // pipes: the bootstrap waits on them, then holds them for the
        // VM's life.
        let port = allocate_ssh_port()?;
        let mut cmd = Command::new(&self.bin);
        cmd.arg("run")
            .arg("--ssh-port")
            .arg(port.to_string())
            .arg("-m")
            .arg(self.settings.memory_mb.to_string())
            .arg("--disk-size")
            .arg(self.settings.disk_mb.to_string());
        for layer in &self.settings.layers {
            cmd.arg("--layer").arg(layer);
        }
        // The SSH server layer: configured like the toolchain layers
        // (required) and appended after them — sshd and sshd-setup ride
        // with every session VM, and the session's shell rides the SSH
        // channel it opens.
        cmd.arg("--layer").arg(&self.settings.ssh_server_layer);
        // The working directory rides the same path on both sides so
        // guest paths resolve exactly as the session's paths do.
        cmd.arg("--share").arg(format!("{cwd}:{cwd}"));
        // The declared write roots ride the same path on both sides for the
        // same reason: the session's out-of-cwd work — its test scripts,
        // its scratch files — must resolve inside the sandbox exactly as it
        // does outside. A root that is not a directory fails the boot, not a
        // silently dropped share: the declaration is the calling agent's
        // own, and a dropped root would surface later as a confusing path
        // error inside the sandbox.
        for root in roots {
            ensure!(
                Path::new(root).is_dir(),
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
        // The console command: the session key into sshd, sshd started,
        // the marker, then the keepalive that holds the console session
        // (the VM's life from the console side) for the entry's whole
        // life.
        let public = self.settings.ssh_public.replace('\'', "'\\''");
        let console = format!(
            "/usr/sbin/sshd-setup '{public}' && echo {BOOTSTRAP_MARKER}; \
             exec {CONSOLE_KEEPALIVE}"
        );
        cmd.arg("--").arg("sh").arg("-c").arg(&console);
        // stderr is inherited, never piped: the boot log lands in this
        // process's own log (where it is greppable), and an unread pipe
        // could never fill and stall the VM mid-boot.
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        dies_with_parent(&mut cmd);
        let entry = Self::spawn_vm(
            cmd,
            self.bin.clone(),
            self.ssh_bin.clone(),
            cwd,
            roots,
            port,
            &self.settings,
        )?;
        Ok(entry)
    }

    /// The rest of one session VM's spawn: the VM process and its console
    /// pipes, the console bootstrap wait, the ssh session spawn and its
    /// ready wait. One place so a failure anywhere closes everything —
    /// a half-spawned session VM (a VM whose sshd never came up, an ssh
    /// process that never connected) must never be left standing.
    fn spawn_vm(
        mut cmd: Command,
        bin: PathBuf,
        ssh_bin: PathBuf,
        cwd: &str,
        roots: &[String],
        port: u16,
        settings: &ResolvedSettings,
    ) -> Result<VmEntry> {
        let mut vm = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", bin.display()))?;
        let (console_in, mut console_out) = (
            vm.stdin.take().context("sandbox stdin not piped")?,
            vm.stdout.take().context("sandbox stdout not piped")?,
        );
        // (b) The console bootstrap wait: the BOOTSTRAP_MARKER is the
        // last thing the console ever prints (the keepalive after it
        // prints nothing), so this read drains the boot noise and ends
        // at the marker — nothing after it can ever fill the held pipe.
        if let Err(error) = wait_marker(
            &mut console_out,
            &mut vm,
            BOOTSTRAP_MARKER,
            BOOTSTRAP_DEADLINE,
            "the session VM's sshd never came up (no bootstrap marker); \
             see the platform log for the sandbox's own output",
        ) {
            let _ = vm.kill();
            let _ = vm.wait();
            return Err(error);
        }
        // (c) The SSH session: the session key against the --ssh-port
        // listener, the session wrapper inside. stderr is inherited: the
        // session's own errors land in the platform log, greppable like
        // the boot log.
        let mut ssh_cmd = Command::new(&ssh_bin);
        ssh_cmd
            .arg("-p")
            .arg(port.to_string())
            .arg("-i")
            .arg(&settings.ssh_key)
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("IdentitiesOnly=yes")
            .arg("-o")
            .arg("LogLevel=ERROR")
            .arg("root@127.0.0.1")
            // The wrapper as ONE remote command: ssh flattens its command
            // argv into a single line the guest's login shell re-parses —
            // `sh -c <wrapper>` as separate argv splits into `sh -c echo`
            // (an empty line) with the read loop in the login shell, and
            // the ready marker never prints. One argument, one command.
            .arg(SESSION_WRAPPER);
        ssh_cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        dies_with_parent(&mut ssh_cmd);
        let mut ssh = ssh_cmd
            .spawn()
            .context("failed to spawn the session's ssh channel")?;
        let (ssh_in, mut ssh_out) = (
            ssh.stdin
                .take()
                .context("the ssh session's stdin not piped")?,
            ssh.stdout
                .take()
                .context("the ssh session's stdout not piped")?,
        );
        // (d) The ssh session's ready wait: the session shell is up and
        // reading. A failure here closes everything — a session VM whose
        // shell never came up must never be left standing.
        if let Err(error) = wait_marker(
            &mut ssh_out,
            &mut ssh,
            READY_MARKER,
            READY_DEADLINE,
            "the session VM's ssh channel never became ready; \
             see the platform log for the session's own output",
        ) {
            let _ = ssh.kill();
            let _ = ssh.wait();
            let _ = vm.kill();
            let _ = vm.wait();
            return Err(error);
        }
        Ok(VmEntry {
            cwd: cwd.to_string(),
            roots: roots.to_vec(),
            state: Mutex::new(VmState {
                vm,
                console: Some((console_in, console_out)),
                ssh,
                stdin: ssh_in,
                stdout: ssh_out,
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
    fn ensure(&self, session_id: &str, cwd: &str, roots: &[String]) -> Result<()> {
        let mut entries = lock(&self.entries);
        match entries.get(session_id) {
            Some(_) => Ok(()),
            None => {
                let entry = Arc::new(self.spawn(cwd, roots)?);
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
        let (cwd, roots) = session_context(payload)?;
        let entries = lock(&self.entries);
        match entries.get(session_id) {
            Some(entry) => {
                if entry.cwd != cwd || entry.roots != roots {
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
        // This session's commands serialize on the VM's own state lock;
        // the Arc keeps the entry alive for the command even if the map
        // drops it concurrently (a reap or a close-all).
        let background = payload
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
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
    fn execute_job_control(&self, session_id: &str, payload: &Value, job: &str) -> Result<String> {
        // The job tag is ours (hex from run_job_start); never let a caller
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
            return Ok(format!(
                "Job {job} killed (best-effort: its own shell only)"
            ));
        }
        // The job's own exit code rides its output file.
        let output = &raw.out[..];
        let marker = b"JOB-EXIT:";
        let exit = find_subslice(output, marker, 0).and_then(|at| {
            let rest = &output[at + marker.len()..];
            let end = rest
                .iter()
                .position(|b| !b.is_ascii_digit())
                .unwrap_or(rest.len());
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
    // The console pipes first: dropping them closes the console session,
    // and sandal tears the VM down from the inside — its own teardown,
    // the guest cleaning up after itself. The kills below are the belt.
    drop(state.console.take());
    let _ = state.ssh.kill();
    let _ = state.ssh.wait();
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
/// read EOFs into the dead-VM path — a fresh boot, not a bricked
/// session.
fn lock_state(state: &Mutex<VmState>) -> std::sync::MutexGuard<'_, VmState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    let line = format!("tail -c 50000 {out} 2>/dev/null; printf '\\n{sentinel}:0\\n'\n");
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
    // The session shell is up (the spawn completed the whole bootstrap —
    // the console's sshd and the ssh channel's ready marker): the
    // protocol starts here, first line to last.
    let deadline = Instant::now() + timeout;
    // The write retries refusals — but a refusal that persists is almost
    // always a dead VM (the memory cap OOM-killed the guest, the wrapper
    // died with it, and the write hits a closed pipe), not a busy
    // transport. Detect the death fast: the entry is removed and the
    // session's next call boots a fresh VM in about a second, instead of
    // burning the whole timeout against a corpse.
    match write_line_retrying(&mut state.stdin, &mut state.ssh, &line, deadline) {
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
            return Outcome::Closed(Err(anyhow::anyhow!(
                "failed to poll the sandbox VM's output"
            )));
        }
        if ready == 0 {
            continue;
        }
        // Safety: read(2) into a local buffer; the length is the buffer's.
        let n = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n < 0 {
            close_state(state);
            return Outcome::Closed(Err(anyhow::anyhow!(
                "failed to read the sandbox VM's output"
            )));
        }
        if n == 0 {
            // EOF: the VM died mid-command. Report the status; the why is
            // in this process's own log (the child's stderr is inherited).
            let status = state
                .ssh
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

/// Wait for one process's marker: the console's bootstrap marker (sshd
/// is up) or the ssh session's ready marker (the session shell is up).
/// Everything before the marker is boot noise — read past, never kept.
/// EOF means the process died before the marker; a deadline passing
/// means it never printed one. Either way the caller closes everything.
fn wait_marker(
    stdout: &mut ChildStdout,
    child: &mut Child,
    marker: &str,
    deadline: Duration,
    never: &str,
) -> Result<()> {
    let deadline = Instant::now() + deadline;
    // The rolling tail: only the bytes that could hold the marker are
    // kept, never the noise before them.
    let mut tail: Vec<u8> = Vec::new();
    let fd = stdout.as_raw_fd();
    let mut chunk = [0u8; 8192];
    loop {
        if Instant::now() >= deadline {
            bail!("{never}");
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
            let status = child
                .wait()
                .ok()
                .and_then(|status| status.code())
                .unwrap_or(-1);
            bail!(
                "the session's sandbox VM exited before its channel was ready \
                 (status {status}); see the platform log for the sandbox's own output"
            );
        }
        let data = &chunk[..n as usize];
        tail.extend_from_slice(data);
        if find_subslice(&tail, marker.as_bytes(), 0).is_some() {
            return Ok(());
        }
        let keep = tail.len().saturating_sub(marker.len() + 16);
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

    /// A fake sandbox binary: prints the bootstrap marker (sshd "came
    /// up"), then the keepalive — the console half of the SSH split. It
    /// also dumps its own argv: the spawn args (the layers, the shares,
    /// the sizing) are the only fake-level observation of what the pool
    /// asked the sandbox for.
    fn fake_sandal(dir: &Path) -> PathBuf {
        let path = dir.join("fake-sandal");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{dump}'\necho {BOOTSTRAP_MARKER}\nexec {CONSOLE_KEEPALIVE}\n",
                dump = dir.join("sandal.args").display()
            ),
        )
        .unwrap();
        fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        path
    }

    /// A fake ssh binary: prints the ready marker, then evals the
    /// protocol lines verbatim — the session half of the SSH split. The
    /// commands run host-side in one long-lived process, exactly what the
    /// reuse tests need to observe.
    fn fake_ssh(dir: &Path) -> PathBuf {
        let path = dir.join("fake-ssh");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\necho {READY_MARKER}\nwhile IFS= read -r line; do eval \"$line\" 2>&1; done\n"
            ),
        )
        .unwrap();
        fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        path
    }

    /// A fake session key pair (a private key file and its .pub), the
    /// fake SSH server layer, and the wiring the resolve path needs: the
    /// key path is configured, the .pub sits next to it, the layer sits
    /// in the config directory. Nothing real — the fakes never
    /// authenticate anything.
    fn fake_ssh_wiring(dir: &Path) -> (String, String, String) {
        let key = dir.join("session-key");
        fs::write(&key, "FAKE PRIVATE KEY").unwrap();
        let pub_path = dir.join("session-key.pub");
        let public = "FAKE PUBLIC KEY";
        fs::write(&pub_path, public).unwrap();
        let layer = dir.join("ssh-server.layer");
        fs::write(&layer, "FAKE LAYER").unwrap();
        (
            key.display().to_string(),
            public.to_string(),
            layer.display().to_string(),
        )
    }

    /// A pool with both fakes (the sandbox and its ssh), the fake
    /// wiring, and the given idle timeout. The settings resolve through
    /// the real path (the fake key/layer in the test directory), so the
    /// resolve path is under test too.
    fn pool(dir: &Path, idle_timeout_secs: u64) -> Arc<VmPool> {
        let (key_path, public, layer_path) = fake_ssh_wiring(dir);
        let raw = settings(&format!(
            "[agent.sandal]\nssh_key = '{key_path}'\nssh_server_layer = '{layer_path}'\nidle_timeout_secs = {idle_timeout_secs}\n"
        ));
        let resolved = raw.resolve(Some(dir)).unwrap();
        assert_eq!(resolved.ssh_public, public);
        Arc::new(VmPool::new(resolved, fake_sandal(dir), fake_ssh(dir)))
    }

    fn call(pool: &VmPool, session: &str, command: &str) -> Result<String> {
        call_with_roots(pool, session, command, &[])
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
        pool.ensure(session, &cwd.to_string_lossy(), roots)?;
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

    /// Parse and resolve with the fake session key spliced in: every
    /// resolve-path test needs the key now (it is required, not
    /// optional), and this is the one place that wires the fake.
    fn settings_resolved(dir: &Path, content: &str) -> Result<ResolvedSettings> {
        let (key_path, _, layer_path) = fake_ssh_wiring(dir);
        let content = content.replace(
            "[agent.sandal]\n",
            &format!("[agent.sandal]\nssh_key = '{key_path}'\nssh_server_layer = '{layer_path}'\n"),
        );
        settings(&content).resolve(Some(dir))
    }

    #[test]
    fn settings_parse_all_fields() {
        let raw = settings(
            "[agent.sandal]\nlayers = ['go.layer']\nssh_server_layer = 'ssh-server.layer'\nshared = ['~/go/pkg:~/go/pkg']\nmemory = 4096\ndisk = 8192\nidle_timeout_secs = 600\n",
        );
        assert_eq!(raw.layers, ["go.layer"]);
        assert_eq!(raw.ssh_server_layer, Some("ssh-server.layer".to_string()));
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
        // No toolchain layers: sizing is the defaults anyway — every
        // session VM carries the SSH server layer, so the `layers`
        // gate of the console era is gone (a toolchain VM gets the
        // same floor).
        let dir = unique_test_dir();
        let (key_path, _, layer_path) = fake_ssh_wiring(dir.path());
        let resolved = settings(&format!(
            "[agent.sandal]\nlayers = []\nshared = ['~/go/pkg:~/go/pkg']\nssh_key = '{key_path}'\nssh_server_layer = '{layer_path}'\nmemory = 65536\n",
        ))
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
        let resolved = settings_resolved(
            dir.path(),
            "[agent.sandal]\nlayers = []\nshared = ['~/go/pkg:~/go/pkg', '~/.cache/go-build:~/.cache/go-build']\n",
        )
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
        let resolved = settings_resolved(
            dir.path(),
            "[agent.sandal]\nlayers = []\nshared = ['cache:/root/cache']\n",
        )
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

        let resolved =
            settings_resolved(dir.path(), "[agent.sandal]\nlayers = ['go.layer']\n").unwrap();

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

        let resolved = settings_resolved(
            dir.path(),
            "[agent.sandal]\nlayers = ['go.layer']\nmemory = 2048\ndisk = 4096\n",
        )
        .unwrap();

        assert_eq!(resolved.memory_mb, 2048);
        assert_eq!(resolved.disk_mb, 4096);
    }

    #[test]
    fn missing_layer_is_an_error() {
        let dir = unique_test_dir();
        let error =
            settings_resolved(dir.path(), "[agent.sandal]\nlayers = ['go.layer']\n").unwrap_err();
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
        let resolved =
            settings_resolved(dir.path(), "[agent.sandal]\nlayers = ['images/go.layer']\n")
                .unwrap();
        assert_eq!(resolved.layers, vec![dir.path().join("images/go.layer")]);
    }

    #[test]
    fn synthetic_config_cannot_resolve_layers() {
        let error = settings("[agent.sandal]\nssh_key = '/nonexistent'\nlayers = ['go.layer']\n")
            .resolve(None)
            .unwrap_err();
        assert!(error.to_string().contains("no config directory"));
    }

    #[test]
    fn idle_timeout_defaults_and_overrides() {
        let dir = unique_test_dir();
        std::fs::write(dir.path().join("go.layer"), "layer").unwrap();
        let default =
            settings_resolved(dir.path(), "[agent.sandal]\nlayers = ['go.layer']\n").unwrap();
        assert_eq!(
            default.idle_timeout,
            Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS)
        );
        let configured = settings_resolved(
            dir.path(),
            "[agent.sandal]\nlayers = ['go.layer']\nidle_timeout_secs = 60\n",
        )
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
        let pid_a = lock_state(&a.state).ssh.id();
        let pid_b = lock_state(&b.state).ssh.id();
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
            pool.ensure("session-1", &cwd.to_string_lossy(), &[])
                .unwrap();
            pool.execute(
                "session-1",
                &json!({
                    "command": "sleep 30",
                    "timeout_secs": 1,
                    "_session_cwd": cwd,
                }),
            )
            .unwrap_err()
        };
        assert!(timed_out.to_string().contains("timed out"), "{timed_out}");
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

    /// A command's stderr rides the protocol, merged into its own output:
    /// the host shell tool's contract (stdout, stderr, exit code) — the
    /// model sees its own command's diagnostics instead of them streaming
    /// to the inherited ssh stderr (the platform log, the operator's UI).
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
                .ensure("session-a", &cwd.to_string_lossy(), &[])
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
                .ensure("session-1", &cwd.to_string_lossy(), &[])
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
        pool.ensure("session-1", &cwd, &[]).unwrap();
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
        // Poll while it runs: no exit code, no output.
        let running = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
            .unwrap();
        assert!(running.contains("(still running)"), "{running}");
        // Wait for the job, then poll: its own exit code and output.
        std::thread::sleep(Duration::from_secs(2));
        let finished = pool
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
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
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &cwd, &[]).unwrap();
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
            pool.ensure("session-1", &cwd, &[]).unwrap();
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
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &cwd, &[]).unwrap();
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
            .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
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
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        pool.ensure("session-1", &cwd, &[]).unwrap();
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
                .execute("session-1", &json!({"job_id": job, "_session_cwd": cwd}))
                .unwrap();
            assert!(running.contains("(still running)"), "{running}");
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
