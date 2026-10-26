//! The `ssh` tool: run a shell command on one of the adopting agent's
//! configured remote hosts.
//!
//! Core defines the tool (schema, config format, execution) but never
//! registers it. An agent whose block configures `ssh = [{hostname, ...}]`
//! adopts it: [`configured_hosts`] checks its own section for the `ssh`
//! table and validates the format, [`tool_definition`] builds the bus
//! definition with every configured hostname injected into the `hostname`
//! field description, and [`execute`] serves one call — the hostname
//! allow-list being the tool's whole authorization model, so a `hostname`
//! outside the list is rejected before any process is spawned.

use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use crate::core::bus::{AgentBus, AgentInbox, AgentToolDefinition};
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::ssh_util::validate_ssh_identity;
use crate::util::dies_with_parent;

/// Wall-clock envelope for one ssh command, matching the `shell` tool's
/// default. The tool exposes no timeout argument: a remote command expected
/// to outlive this belongs in a remote background job (`nohup ... &`), not in
/// a blocked tool call.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Cap on the collected output, matching the `shell` tool.
const MAX_OUTPUT: usize = 50_000;
/// Bound on the connection phase only (TCP + auth); a command that connects
/// fine may run as long as it wants up to [`DEFAULT_TIMEOUT_SECS`].
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// The tool's name on the bus and as the model sees it.
pub const TOOL_NAME: &str = "ssh";
/// The bus operation one execution request carries.
pub const OPERATION: &str = "ssh";

/// The periodic task id a serving agent schedules the inbox drain under, and
/// how long one drain blocks waiting for a proxied execution.
pub const REQUESTS_TASK: &str = "ssh_requests";
pub const INBOX_WAIT: Duration = Duration::from_millis(200);

/// One configured SSH destination. Only `hostname` is required: without
/// `user` ssh connects as its default, and without `identity_file` ssh's own
/// configuration (agent, default keys) authenticates. Every present field is
/// validated — a value starting with `-` would be parsed as an ssh option,
/// not a destination.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SshHost {
    pub hostname: String,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub identity_file: Option<String>,
}

/// Check an agent block's config for the `ssh` table: `Ok(None)` when the
/// block configures none, `Ok(Some(hosts))` when it does, `Err` when the
/// table does not match the required format (a malformed entry must not
/// silently drop a configured destination).
pub fn configured_hosts(section: &Value) -> Result<Option<Vec<SshHost>>> {
    let Some(ssh) = section.get("ssh") else {
        return Ok(None);
    };
    let hosts: Vec<SshHost> = serde_json::from_value(ssh.clone()).with_context(
        || "invalid `ssh` config: expected a table of {hostname, user?, identity_file?}",
    )?;
    for host in &hosts {
        validate_host(host)?;
    }
    Ok(Some(hosts))
}

fn validate_host(host: &SshHost) -> Result<()> {
    validate_ssh_identity(&host.hostname, "hostname")?;
    ensure!(
        !host.hostname.starts_with('-'),
        "hostname must not start with '-'"
    );
    if let Some(user) = &host.user {
        validate_ssh_identity(user, "user")?;
        ensure!(!user.starts_with('-'), "user must not start with '-'");
    }
    if let Some(identity_file) = &host.identity_file {
        ensure!(
            !identity_file.trim().is_empty(),
            "identity_file must not be empty when set"
        );
    }
    Ok(())
}

/// The bus definition an adopting agent registers: the schema exposes
/// `hostname` and `command`, and every configured hostname is injected into
/// the `hostname` field description so the model picks from the actual
/// allow-list. `for_agent` scopes the tool to the adopting agent's own
/// sessions — execution is proxied back into that agent's process.
pub fn tool_definition(hosts: &[SshHost], agent_id: &str) -> AgentToolDefinition {
    AgentToolDefinition {
        name: TOOL_NAME.to_string(),
        description:
            "Run a shell command on a configured remote host over SSH. The command runs under the remote host's default shell as the configured user; returns stdout, stderr, and exit code. Commands have a timeout (120s). Use it only for hosts you were told to operate on."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "hostname": {
                    "type": "string",
                    "description": hostname_description(hosts),
                },
                "command": {
                    "type": "string",
                    "description": "The shell command to execute on the remote host."
                },
            },
            "required": ["hostname", "command"],
            "additionalProperties": false
        }),
        operation: OPERATION.to_string(),
        for_agent: Some(agent_id.to_string()),
    }
}

/// The adopted tool's serving state, held by an agent that registered it:
/// the validated host list plus the bus inbox receiving proxied executions.
/// Any fleet agent (worker, reviewer, pmo, qa, ...) adopts one in its
/// `build` when its own block configures hosts, schedules [`Self::task`],
/// and answers calls through [`Self::serve_once`] — the config is consumed
/// and the command executed here, in the adopting agent's own process.
pub struct SshServer {
    hosts: Vec<SshHost>,
    inbox: AgentInbox,
}

impl SshServer {
    /// Adopt the tool when the agent block configures hosts: check the
    /// section for the `ssh` table and its format, build the definition with
    /// every hostname injected, and register it on the bus scoped to
    /// `agent_id` — the tool is then offered only to that agent's own
    /// sessions. `Ok(None)` when the block configures none.
    pub fn adopt<S: serde::Serialize>(
        bus: Option<&AgentBus>,
        agent_id: &str,
        section: &S,
    ) -> Result<Option<Self>> {
        let section =
            serde_json::to_value(section).context("agent section is not JSON-representable")?;
        let Some(hosts) = configured_hosts(&section)? else {
            return Ok(None);
        };
        let bus = bus.context("the `ssh` tool requires the cross-agent bus")?;
        let definition = tool_definition(&hosts, agent_id);
        let inbox = bus.register(agent_id, vec![definition])?;
        info!(
            "{agent_id}: adopted the `ssh` tool for {} configured host(s)",
            hosts.len()
        );
        Ok(Some(Self { hosts, inbox }))
    }

    /// The periodic task a serving agent adds to its scheduler: a fast-spin
    /// cycle whose handler blocks on the inbox wait, not on the interval.
    pub fn task() -> PeriodicTaskSpec {
        PeriodicTaskSpec {
            id: REQUESTS_TASK,
            interval: Duration::ZERO,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 0,
            autostart: true,
        }
    }

    /// Drain and answer one proxied execution from the inbox, if any.
    pub fn serve_once(&mut self) -> Result<()> {
        if let Some(request) = self.inbox.recv_timeout(INBOX_WAIT)? {
            let result = execute(&self.hosts, &request.payload);
            request.respond(result.map(Value::String));
        }
        Ok(())
    }
}
/// The `hostname` field description: every configured host, so the model
/// picks from the actual allow-list instead of guessing.
fn hostname_description(hosts: &[SshHost]) -> String {
    let mut description = "The remote host to run the command on. Configured hosts:".to_string();
    for host in hosts {
        description.push_str(&format!("\n- {}", host.hostname));
    }
    description
}

/// Serve one tool call: validate the target against the adopting agent's
/// allow-list, then run ssh. The remote command is passed as a single argv
/// element (never routed through a local shell), and the ssh process gets
/// its own process group so a timeout kills it cleanly instead of leaving a
/// stuck connection.
pub fn execute(hosts: &[SshHost], args: &Value) -> Result<String> {
    let hostname = args["hostname"]
        .as_str()
        .map(str::trim)
        .filter(|hostname| !hostname.is_empty())
        .context("missing 'hostname' argument")?;
    let command = args["command"]
        .as_str()
        .filter(|command| !command.trim().is_empty())
        .context("missing 'command' argument")?;

    let host = hosts
        .iter()
        .find(|host| host.hostname == hostname)
        .with_context(|| {
            format!(
                "host {hostname:?} is not configured for ssh; configured hosts: {}",
                hosts
                    .iter()
                    .map(|host| host.hostname.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    run_ssh_command(&ssh_argv(host, command))
}

/// The ssh argv for one configured host and remote command. `BatchMode` plus
/// a connect timeout keep an unreachable host from hanging the session: both
/// failure modes surface as a normal non-zero exit the model can read.
fn ssh_argv(host: &SshHost, command: &str) -> Vec<String> {
    let mut argv = vec![
        "ssh".to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        format!("ConnectTimeout={CONNECT_TIMEOUT_SECS}"),
    ];
    if let Some(identity_file) = &host.identity_file {
        argv.push("-i".to_string());
        argv.push(identity_file.clone());
    }
    argv.push(match &host.user {
        Some(user) => format!("{user}@{}", host.hostname),
        None => host.hostname.clone(),
    });
    argv.push(command.to_string());
    argv
}

/// Run one prepared ssh argv and collect its output. The ssh process gets
/// its own process group so a timeout kills it cleanly instead of leaving a
/// stuck connection.
fn run_ssh_command(argv: &[String]) -> Result<String> {
    let target = argv
        .iter()
        .find(|arg| arg.contains('@'))
        .cloned()
        .unwrap_or_default();
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    dies_with_parent(&mut cmd);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to execute ssh for {target}"))?;
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");

    // Drain both streams to EOF on reader threads while polling for exit, so
    // a chatty command never fills its pipe and blocks. After the local ssh
    // process (the whole process group) is gone, its pipe ends close and EOF
    // arrives — ssh runs remote commands remotely, so no local descendant
    // can inherit and hold these pipes.
    let stdout_buf = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf = Arc::new(Mutex::new(Vec::new()));
    let stdout_handle = {
        let buf = Arc::clone(&stdout_buf);
        std::thread::spawn(move || {
            let read = read_capped(&mut stdout);
            *buf.lock().unwrap() = read;
        })
    };
    let stderr_handle = {
        let buf = Arc::clone(&stderr_buf);
        std::thread::spawn(move || {
            let read = read_capped(&mut stderr);
            *buf.lock().unwrap() = read;
        })
    };

    let deadline = Instant::now() + Duration::from_secs(DEFAULT_TIMEOUT_SECS);
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(_)) => break false,
            Ok(None) => {
                if Instant::now() >= deadline {
                    break true;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => break false,
        }
    };

    if timed_out {
        #[cfg(unix)]
        unsafe {
            libc::killpg(child.id() as i32, libc::SIGKILL);
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }
        let _ = child.wait();
    }
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();

    let stdout_output = String::from_utf8_lossy(&stdout_buf.lock().unwrap()).into_owned();
    let stderr_output = String::from_utf8_lossy(&stderr_buf.lock().unwrap()).into_owned();
    let code = child
        .wait()
        .ok()
        .and_then(|status| status.code())
        .unwrap_or(if timed_out { -1 } else { 1 });

    let mut result = String::new();
    if timed_out {
        result.push_str(&format!(
            "Command timed out after {DEFAULT_TIMEOUT_SECS}s and was killed.\n"
        ));
    }
    if !stdout_output.is_empty() {
        result.push_str("stdout:\n");
        result.push_str(&stdout_output);
    }
    if !stderr_output.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("stderr:\n");
        result.push_str(&stderr_output);
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

/// Read a stream to EOF, keeping at most [`MAX_OUTPUT`] bytes (the head);
/// the stream is always drained so a chatty command never blocks on a full
/// pipe.
fn read_capped(stream: &mut impl std::io::Read) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return buf,
            Ok(read) => {
                let keep = read.min(MAX_OUTPUT.saturating_sub(buf.len()));
                buf.extend_from_slice(&chunk[..keep]);
            }
            Err(_) => return buf,
        }
    }
}

fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(hostname: &str) -> SshHost {
        SshHost {
            hostname: hostname.to_string(),
            user: Some("ci".to_string()),
            identity_file: None,
        }
    }

    #[test]
    fn configured_hosts_absent_when_the_block_has_no_ssh_table() {
        let section = json!({"poll_interval": "1m"});
        assert_eq!(configured_hosts(&section).unwrap(), None);
    }

    #[test]
    fn configured_hosts_parses_hostname_only_entries() {
        let section = json!({
            "ssh": [
                {"hostname": "build.example", "user": "ci", "identity_file": "/keys/ci"},
                {"hostname": "db.example"},
            ]
        });
        assert_eq!(
            configured_hosts(&section).unwrap(),
            Some(vec![
                SshHost {
                    hostname: "build.example".into(),
                    user: Some("ci".into()),
                    identity_file: Some("/keys/ci".into()),
                },
                SshHost {
                    hostname: "db.example".into(),
                    user: None,
                    identity_file: None,
                },
            ])
        );
    }

    #[test]
    fn configured_hosts_rejects_malformed_tables() {
        // Unknown fields are a config/binary disagreement.
        let error = configured_hosts(&json!({"ssh": [{"hostname": "b", "port": 22}]}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid `ssh` config"), "{error}");

        // Option-like targets would be parsed by ssh as options.
        let error = configured_hosts(&json!({"ssh": [{"hostname": "-J", "user": "ci"}]}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("must not start with '-'"), "{error}");

        // An empty identity file is never a meaningful value.
        let error = configured_hosts(&json!({"ssh": [{"hostname": "b", "identity_file": " "}]}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("identity_file"), "{error}");
    }

    #[test]
    fn definition_injects_every_hostname_and_scopes_to_the_agent() {
        let definition = tool_definition(&[host("build.example"), host("db.example")], "worker-0");
        assert_eq!(definition.name, "ssh");
        assert_eq!(definition.operation, "ssh");
        assert_eq!(definition.for_agent.as_deref(), Some("worker-0"));

        let description = definition.parameters["properties"]["hostname"]["description"]
            .as_str()
            .unwrap();
        assert!(description.contains("build.example"), "{description}");
        assert!(description.contains("db.example"), "{description}");

        // Only hostname and command are exposed; both required.
        let properties = definition.parameters["properties"].as_object().unwrap();
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("hostname"));
        assert!(properties.contains_key("command"));
        assert_eq!(
            definition.parameters["required"],
            json!(["hostname", "command"])
        );
    }

    #[test]
    fn unknown_hostnames_are_rejected_before_any_process_is_spawned() {
        let error = execute(
            &[host("build.example")],
            &json!({"hostname": "evil.example", "command": "echo hi"}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("evil.example"), "{error}");
        assert!(error.contains("build.example"), "{error}");
    }

    #[test]
    fn missing_or_empty_arguments_are_rejected() {
        let error = execute(&[host("build.example")], &json!({"command": "echo hi"}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("hostname"));

        let error = execute(
            &[host("build.example")],
            &json!({"hostname": "build.example"}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("command"));

        let error = execute(
            &[host("build.example")],
            &json!({"hostname": "build.example", "command": "   "}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("command"));
    }

    #[test]
    fn adoption_registers_a_scoped_definition_with_every_hostname() {
        let bus = AgentBus::new();

        // A block without an `ssh` table adopts nothing.
        assert!(
            SshServer::adopt(Some(&bus), "worker-0", &json!({}))
                .unwrap()
                .is_none()
        );

        let server = SshServer::adopt(
            Some(&bus),
            "worker-0",
            &json!({
                "ssh": [
                    {"hostname": "build.example"},
                    {"hostname": "db.example", "user": "admin"}
                ]
            }),
        )
        .unwrap()
        .expect("adoption for a configured block");
        assert_eq!(server.hosts.len(), 2);

        let registered = bus.registered_tools(Duration::ZERO).unwrap();
        let definition = registered
            .iter()
            .find(|tool| tool.name == TOOL_NAME)
            .expect("the adopted tool is registered");
        assert_eq!(definition.target, "worker-0");
        assert_eq!(definition.for_agent.as_deref(), Some("worker-0"));
        let description = definition.parameters["properties"]["hostname"]["description"]
            .as_str()
            .unwrap();
        assert!(description.contains("build.example"), "{description}");
        assert!(description.contains("db.example"), "{description}");

        // A malformed table refuses to adopt rather than silently dropping
        // configured destinations.
        assert!(
            SshServer::adopt(Some(&bus), "worker-1", &json!({"ssh": [{"user": "ci"}]})).is_err()
        );
    }

    #[test]
    fn argv_targets_the_configured_user_host_and_identity() {
        let with_identity = SshHost {
            hostname: "build.example".into(),
            user: Some("ci".into()),
            identity_file: Some("/keys/ci_ed25519".into()),
        };
        assert_eq!(
            ssh_argv(&with_identity, "uptime"),
            vec![
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-i",
                "/keys/ci_ed25519",
                "ci@build.example",
                "uptime",
            ]
        );

        // Without `user`, ssh's own target default applies; without an
        // identity file, ssh's own configuration authenticates.
        let bare = SshHost {
            hostname: "db.example".into(),
            user: None,
            identity_file: None,
        };
        let argv = ssh_argv(&bare, "df -h");
        assert!(!argv.contains(&"-i".to_string()));
        assert_eq!(
            argv.get(argv.len() - 2).map(String::as_str),
            Some("db.example")
        );
        assert_eq!(argv.last().map(String::as_str), Some("df -h"));
    }
}
