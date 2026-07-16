//! Configurable ACP server command (default: `agent acp`) runs as **one long-lived subprocess**
//! per Potlatch agent role. Each task
//! calls **`session/close`** (best effort) then **`session/new`** on the same stdio connection so
//! the model does not keep prior in-agent transcript; workflow continuity stays in Potlatch’s
//! state files and token use stays lower than reusing one session for every task.
//!
//! [ACP slash commands](https://agentclientprotocol.com/protocol/slash-commands): the agent may
//! send `available_commands_update`; [`StreamTextHooks`] records command names for future
//! role-specific prompt logic (not wired into prompts yet).
//!
//! [Session modes](https://agentclientprotocol.com/protocol/session-modes): callers may request
//! **`ask`** or **`plan`**. Modes apply only when the agent advertises them (config-option `mode`
//! value or non-empty legacy `availableModes`). Otherwise the agent default is kept;
//! `current_mode_update` keeps hooks in sync.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use tracing::{debug, info, warn};

use super::client::{AcpClient, CursorAskQuestionHandler};
use super::orchestrator_hooks::StreamTextHooks;
use super::types::{
    ClientCapabilities, ClientFsCapabilities, DEFAULT_PROTOCOL_VERSION, ImplementationInfo,
    InitializeParams, InitializeResult, NewSessionParams, NewSessionResult, PromptResult,
    mode_id_is_available, model_selector_for_session, select_option_allows_value,
    session_mode_config_option,
};
use crate::core::agent::AgentHandoff;
use crate::core::config::build_acp_spawn_command;

const TASK_CONTEXT_RESET_GUIDANCE: &str = r#"IMPORTANT CONTEXT HANDLING:
Treat this assignment as a fresh task. Do not rely on prior chat history or assumptions from earlier assignments unless this prompt explicitly refers to them. Use only the repository state, issue/MR context, and instructions present in this task.

"#;

/// ACP session mode for read-oriented work, permission before edits.
pub const ACP_SESSION_MODE_ASK: &str = "ask";
/// ACP session mode for planning and decomposition work.
pub const ACP_SESSION_MODE_PLAN: &str = "plan";

struct AcpSession {
    client: Arc<AcpClient>,
    child: Child,
    session_id: String,
    hooks: Arc<StreamTextHooks>,
}

fn dispose_acp_session(s: AcpSession) {
    let _ = s.client.session_cancel(&s.session_id);
    let mut s = s;
    let _ = s.child.kill();
    let _ = s.child.wait();
    drop(s.client);
}

fn close_acp_session_best_effort(client: &AcpClient, session_id: &str) {
    match client.session_close(session_id) {
        Ok(()) => {}
        Err(e) => debug!(
            target: "potlatch::acp",
            "session/close failed for session {} (continuing with session/new): {}",
            session_id,
            e
        ),
    }
}

pub(crate) struct AcpRuntime {
    repo_path: String,
    model_uri: Option<String>,
    endpoint_model: Option<String>,
    spawn_model: Option<String>,
    acp_command: Vec<String>,
    acp_env: std::collections::HashMap<String, String>,
    /// When set, applied after `session/new` via ACP mode APIs.
    preferred_session_mode: Option<&'static str>,
    shutdown: Arc<AtomicBool>,
    agent_id: String,
    acp: Mutex<Option<AcpSession>>,
    unexpected_quits_count: Arc<AtomicU64>,
}

impl AcpRuntime {
    pub fn new(
        repo_path: String,
        model_uri: Option<String>,
        endpoint_model: Option<String>,
        spawn_model: Option<String>,
        acp_command: Vec<String>,
        acp_env: std::collections::HashMap<String, String>,
        preferred_session_mode: Option<&'static str>,
        shutdown: Arc<AtomicBool>,
        agent_id: String,
    ) -> Self {
        Self {
            repo_path,
            model_uri,
            endpoint_model,
            spawn_model,
            acp_command,
            acp_env,
            preferred_session_mode,
            shutdown,
            agent_id,
            acp: Mutex::new(None),
            unexpected_quits_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Run one task: ensure a fresh ACP session (`session/close` then `session/new` on the same
    /// child when the process is already running), send `session/prompt`. Retries after transport
    /// failures keep the same session while the child stays up; if the child exits, a new process
    /// and session are created.
    pub fn run_with_cancel(
        &self,
        prompt: &str,
        cancel_check: Option<&dyn Fn() -> bool>,
        cursor_ask_question_handler: Option<Arc<dyn CursorAskQuestionHandler>>,
    ) -> Result<AgentHandoff> {
        let prompt = prepare_task_prompt(prompt);
        const MAX_UNFINISHED_TASK_RETRIES: u32 = 5;
        if let Some(model_uri) = &self.model_uri {
            info!(
                "Running ACP agent {} model={:?} prompt_len={} (new session per task, same process)",
                self.agent_id(),
                model_uri,
                prompt.len()
            );
        } else {
            info!(
                "Running ACP agent {} prompt_len={} (new session per task, same process)",
                self.agent_id(),
                prompt.len()
            );
        }
        debug!("Agent task prompt: {}", prompt);

        self.rotate_acp_session_for_new_task()?;

        let mut unfinished_task_retries: u32 = 0;
        let mut polls_since_cancel_check: u32 = 0;
        const CANCEL_CHECK_INTERVAL: u32 = 25;

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                self.kill_child();
                anyhow::bail!("Agent interrupted by shutdown");
            }

            let (client, session_id, hooks) = {
                let g = self.acp.lock().unwrap();
                let s = g.as_ref().context("ACP session missing after ensure")?;
                (
                    Arc::clone(&s.client),
                    s.session_id.clone(),
                    Arc::clone(&s.hooks),
                )
            };

            hooks.clear();
            hooks.set_cursor_ask_question_handler(cursor_ask_question_handler.clone());
            let prompt_owned = prompt.clone();
            let (tx, rx) = std::sync::mpsc::channel::<Result<PromptResult, String>>();
            thread::spawn(move || {
                let out = client
                    .session_prompt(&session_id, &prompt_owned)
                    .map_err(|e| e.to_string());
                let _ = tx.send(out);
            });

            let handoff_result = loop {
                if self.shutdown.load(Ordering::SeqCst) {
                    self.kill_child();
                    break Err(anyhow::anyhow!("Agent interrupted by shutdown"));
                }
                polls_since_cancel_check += 1;
                if polls_since_cancel_check >= CANCEL_CHECK_INTERVAL {
                    polls_since_cancel_check = 0;
                    if let Some(check) = cancel_check
                        && check()
                    {
                        warn!(
                            "Cancel check triggered, killing ACP agent {}",
                            self.agent_id()
                        );
                        self.kill_child();
                        break Err(anyhow::anyhow!("Agent cancelled by external condition"));
                    }
                }

                if let Some(exit) = self.take_child_exit_status()? {
                    self.kill_child();
                    break Err(anyhow::anyhow!(
                        "agent {} exited during prompt ({})",
                        self.agent_id(),
                        exit
                    ));
                }

                match rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(Ok(pr)) => {
                        self.wait_for_plan_followup_updates(&hooks, cancel_check)?;
                        break Ok(handoff_from_prompt_hooks(&hooks, pr));
                    }
                    Ok(Err(e)) => break Err(anyhow::anyhow!("ACP session/prompt: {}", e)),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        break Err(anyhow::anyhow!(
                            "ACP prompt thread died for {}",
                            self.agent_id()
                        ));
                    }
                }
            };

            hooks.set_cursor_ask_question_handler(None);

            match handoff_result {
                Ok(h) => return Ok(h),
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("exited during prompt") || msg.contains("ACP session/prompt") {
                        unfinished_task_retries += 1;
                        if unfinished_task_retries > MAX_UNFINISHED_TASK_RETRIES {
                            return Err(e);
                        }
                        warn!(
                            "ACP task failed for {} ({msg}). Retry {unfinished_task_retries}/{MAX_UNFINISHED_TASK_RETRIES}",
                            self.agent_id(),
                        );
                        self.unexpected_quits_count.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(retry_backoff_for_unfinished_task(unfinished_task_retries));
                        self.ensure_acp_session_for_retry()?;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
    }

    fn wait_for_plan_followup_updates(
        &self,
        hooks: &StreamTextHooks,
        cancel_check: Option<&dyn Fn() -> bool>,
    ) -> Result<()> {
        if self.preferred_session_mode != Some(ACP_SESSION_MODE_PLAN) {
            return Ok(());
        }

        const MAX_WAIT: Duration = Duration::from_secs(3);
        const QUIET_WINDOW: Duration = Duration::from_millis(500);
        const POLL: Duration = Duration::from_millis(100);

        let start = Instant::now();
        let mut last_change_at = Instant::now();
        let mut last_seq = hooks.notification_seq();

        loop {
            if hooks.has_cursor_plan_paths() {
                return Ok(());
            }
            if start.elapsed() >= MAX_WAIT || last_change_at.elapsed() >= QUIET_WINDOW {
                return Ok(());
            }
            if self.shutdown.load(Ordering::SeqCst) {
                self.kill_child();
                anyhow::bail!("Agent interrupted by shutdown");
            }
            if let Some(check) = cancel_check
                && check()
            {
                self.kill_child();
                anyhow::bail!("Agent cancelled by external condition");
            }

            let seq = hooks.notification_seq();
            if seq != last_seq {
                last_seq = seq;
                last_change_at = Instant::now();
            }
            thread::sleep(POLL);
        }
    }

    fn kill_child(&self) {
        let mut g = self.acp.lock().unwrap();
        if let Some(s) = g.take() {
            dispose_acp_session(s);
        }
    }

    /// Applies [`Self::preferred_session_mode`] only when the agent advertises that mode:
    ///
    /// - Config option with `id`/`category` `mode` and the value in `options`, or
    /// - Legacy `session/new` `modes.availableModes` non-empty and containing the id.
    ///
    /// Otherwise leaves the agent default and logs at `debug` (`potlatch::acp_modes`).
    fn try_apply_preferred_session_mode(
        &self,
        client: &AcpClient,
        session: &NewSessionResult,
        hooks: &StreamTextHooks,
    ) {
        let Some(mode_id) = self.preferred_session_mode else {
            return;
        };

        let legacy_advertises = session
            .modes
            .as_ref()
            .is_some_and(|m| !m.available_modes.is_empty() && mode_id_is_available(m, mode_id));

        if let Some(cfg) = session.config_options.as_deref()
            && let Some(opt) = session_mode_config_option(cfg)
            && select_option_allows_value(opt, mode_id)
        {
            match client.session_set_config_option(&session.session_id, &opt.id, mode_id) {
                Ok(_) => {
                    info!(
                        "ACP session mode {:?} via session/set_config_option for {}",
                        mode_id, self.agent_id
                    );
                    hooks.sync_tracked_current_mode(mode_id);
                    return;
                }
                Err(e) => debug!(
                    target: "potlatch::acp_modes",
                    agent_id = %self.agent_id,
                    err = %e,
                    "session/set_config_option for mode failed",
                ),
            }
            if legacy_advertises {
                match client.session_set_mode(&session.session_id, mode_id) {
                    Ok(_) => {
                        info!(
                            "ACP session mode {:?} via session/set_mode (fallback) for {}",
                            mode_id, self.agent_id
                        );
                        hooks.sync_tracked_current_mode(mode_id);
                    }
                    Err(e) => debug!(
                        target: "potlatch::acp_modes",
                        agent_id = %self.agent_id,
                        err = %e,
                        "session/set_mode fallback after set_config_option failure also failed",
                    ),
                }
            } else {
                debug!(
                    target: "potlatch::acp_modes",
                    agent_id = %self.agent_id,
                    preferred = mode_id,
                    "set_config_option for mode failed and agent does not advertise this mode in legacy availableModes; leaving default",
                );
            }
            return;
        }

        if legacy_advertises {
            match client.session_set_mode(&session.session_id, mode_id) {
                Ok(_) => {
                    info!(
                        "ACP session mode {:?} via session/set_mode for {}",
                        mode_id, self.agent_id
                    );
                    hooks.sync_tracked_current_mode(mode_id);
                }
                Err(e) => debug!(
                    target: "potlatch::acp_modes",
                    agent_id = %self.agent_id,
                    err = %e,
                    "session/set_mode failed",
                ),
            }
            return;
        }

        debug!(
            target: "potlatch::acp_modes",
            agent_id = %self.agent_id,
            preferred = mode_id,
            "preferred session mode not advertised (no mode config option value match and no legacy availableModes entry); leaving agent default mode",
        );
    }

    /// True when [`InitializeResult::auth_methods`] includes Cursor's `cursor_login` method.
    ///
    /// Cursor ACP requires [`AcpClient::authenticate`] with `cursor_login` after `initialize` and
    /// before `session/new` ([Cursor ACP docs](https://cursor.com/docs/cli/acp)).
    fn acp_init_advertises_cursor_login(init: &InitializeResult) -> bool {
        init.auth_methods.iter().any(|m| {
            m.get("id")
                .or_else(|| m.get("methodId"))
                .and_then(|v| v.as_str())
                .is_some_and(|id| id == "cursor_login")
        })
    }

    /// Spawns the configured ACP server command, attaches stdio, `initialize`, and Cursor
    /// `authenticate` when advertised — no `session/new` yet.
    fn spawn_acp_connection(&self) -> Result<(Arc<AcpClient>, Child, Arc<StreamTextHooks>)> {
        let program = self
            .acp_command
            .first()
            .map(String::as_str)
            .unwrap_or("agent");
        let mut cmd = build_acp_spawn_command(
            &self.acp_command,
            self.spawn_model.as_deref(),
            &self.acp_env,
        )
        .with_context(|| format!("build ACP spawn command for {program}"))?;

        let mut child = cmd
            .current_dir(&self.repo_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn `{program}` ACP process"))?;

        let stderr = child.stderr.take();
        if let Some(mut err) = stderr {
            let aid = self.agent_id().to_string();
            thread::spawn(move || {
                let mut buf = Vec::new();
                let n = err.read_to_end(&mut buf).unwrap_or(0);
                if n > 0 {
                    let preview = String::from_utf8_lossy(&buf[..n.min(2048)]);
                    debug!(target: "potlatch::agent_stderr", agent_id = %aid, "stderr: {}", preview);
                    if preview.contains("Cannot use this model") {
                        warn!(target: "potlatch::agent_stderr", agent_id = %aid, "ACP server stderr: {}", preview);
                    }
                }
            });
        }

        let hooks = Arc::new(StreamTextHooks::with_workspace(PathBuf::from(
            self.repo_path.clone(),
        )));
        let (client, child) = AcpClient::from_child_stdio(child, hooks.clone())
            .context("attach ACP client to agent stdio")?;
        let client = Arc::new(client);

        let init_result = client
            .initialize(&InitializeParams {
                protocol_version: DEFAULT_PROTOCOL_VERSION,
                client_capabilities: ClientCapabilities {
                    fs: ClientFsCapabilities {
                        read_text_file: true,
                        write_text_file: false,
                    },
                    terminal: false,
                },
                client_info: ImplementationInfo {
                    name: "potlatch".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
            })
            .context("ACP initialize")?;

        if Self::acp_init_advertises_cursor_login(&init_result) {
            debug!(
                target: "potlatch::acp",
                agent_id = %self.agent_id,
                auth_methods = init_result.auth_methods.len(),
                "initialize result includes authMethods; calling authenticate(cursor_login)"
            );
        } else {
            warn!(
                target: "potlatch::acp",
                agent_id = %self.agent_id,
                "initialize result has no recognizable cursor_login authMethods; still calling authenticate(cursor_login) (required by Cursor ACP before session/new)"
            );
        }

        client.authenticate_cursor_login().context(
            "ACP authenticate (cursor_login). Run `agent login` or set CURSOR_API_KEY / CURSOR_AUTH_TOKEN; see https://cursor.com/docs/cli/acp",
        )?;

        Ok((client, child, hooks))
    }

    fn start_acp_session_on_connection(
        &self,
        client: &Arc<AcpClient>,
        hooks: &Arc<StreamTextHooks>,
    ) -> Result<String> {
        let cwd: PathBuf = std::fs::canonicalize(&self.repo_path)
            .unwrap_or_else(|_| PathBuf::from(&self.repo_path));
        let session = client
            .session_new(&NewSessionParams {
                cwd: cwd.to_string_lossy().into_owned(),
                mcp_servers: vec![],
            })
            .context("ACP session/new")?;

        if let Some(ref modes) = session.modes {
            hooks.seed_session_modes(modes);
            if !mode_id_is_available(modes, &modes.current_mode_id) {
                debug!(
                    target: "potlatch::acp_modes",
                    agent_id = %self.agent_id,
                    current = %modes.current_mode_id,
                    "ACP session/new currentModeId not listed in availableModes",
                );
            }
            debug!(
                target: "potlatch::acp_modes",
                agent_id = %self.agent_id,
                current = %modes.current_mode_id,
                available = ?modes
                    .available_modes
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>(),
                "ACP session modes from session/new",
            );
        }

        self.try_apply_preferred_session_mode(client, &session, hooks);

        if let Some(model) = &self.endpoint_model {
            let mut applied = false;
            if let Some(cfg) = session.config_options.as_deref()
                && let Some(opt) = model_selector_for_session(cfg)
                && select_option_allows_value(opt, model)
            {
                match client.session_set_config_option(&session.session_id, &opt.id, model) {
                    Ok(_) => {
                        info!(
                            "ACP model set via session/set_config_option for agent {}",
                            self.agent_id
                        );
                        applied = true;
                    }
                    Err(e) => debug!(
                        "ACP session/set_config_option failed for {}, trying session/set_model: {}",
                        self.agent_id, e
                    ),
                }
            }
            if !applied {
                match client.session_set_model(&session.session_id, model) {
                    Ok(_) => info!(
                        "ACP model set via session/set_model for agent {}",
                        self.agent_id
                    ),
                    Err(e) => debug!(
                        "ACP session/set_model model={model:?} skipped for {}: {} (CLI --model still in effect)",
                        self.agent_id, e
                    ),
                }
            }
        }

        info!(
            "ACP session {} ready for agent {}",
            session.session_id, self.agent_id
        );

        Ok(session.session_id)
    }

    /// At the start of each task: new `session/new` on the existing child, or spawn if needed.
    fn rotate_acp_session_for_new_task(&self) -> Result<()> {
        let mut g = self.acp.lock().unwrap();
        let need_fresh_spawn = match g.as_mut() {
            None => true,
            Some(s) => match s.child.try_wait() {
                Ok(Some(_)) => {
                    if let Some(s) = g.take() {
                        dispose_acp_session(s);
                    }
                    true
                }
                Ok(None) => false,
                Err(_) => {
                    if let Some(s) = g.take() {
                        dispose_acp_session(s);
                    }
                    true
                }
            },
        };

        if need_fresh_spawn {
            let (client, child, hooks) = self.spawn_acp_connection()?;
            let session_id = self.start_acp_session_on_connection(&client, &hooks)?;
            *g = Some(AcpSession {
                client,
                child,
                session_id,
                hooks,
            });
            return Ok(());
        }

        let s = g
            .as_mut()
            .context("ACP session missing after child check")?;
        close_acp_session_best_effort(&s.client, &s.session_id);
        s.session_id = self.start_acp_session_on_connection(&s.client, &s.hooks)?;
        Ok(())
    }

    /// After a failed prompt while the child may have exited: respawn + `session/new` if needed.
    /// If the child is still running, the current session is left in place for the next attempt.
    fn ensure_acp_session_for_retry(&self) -> Result<()> {
        let mut g = self.acp.lock().unwrap();
        let need_spawn = match g.as_mut() {
            None => true,
            Some(s) => match s.child.try_wait() {
                Ok(Some(_)) => {
                    if let Some(s) = g.take() {
                        dispose_acp_session(s);
                    }
                    true
                }
                Ok(None) => false,
                Err(_) => {
                    if let Some(s) = g.take() {
                        dispose_acp_session(s);
                    }
                    true
                }
            },
        };

        if need_spawn {
            let (client, child, hooks) = self.spawn_acp_connection()?;
            let session_id = self.start_acp_session_on_connection(&client, &hooks)?;
            *g = Some(AcpSession {
                client,
                child,
                session_id,
                hooks,
            });
        }

        Ok(())
    }

    fn take_child_exit_status(&self) -> Result<Option<String>> {
        let mut g = self.acp.lock().unwrap();
        let Some(s) = g.as_mut() else {
            return Ok(Some("ACP child missing".to_string()));
        };
        match s.child.try_wait() {
            Ok(Some(status)) => Ok(Some(status.to_string())),
            Ok(None) => Ok(None),
            Err(err) => Ok(Some(format!("unable to query child status: {}", err))),
        }
    }
}

fn collect_text_fragments_from_value(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_text_fragments_from_value(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                let t = text.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
            }
            for key in [
                "message", "output", "response", "content", "contents", "messages", "parts",
                "blocks", "items",
            ] {
                if let Some(child) = map.get(key) {
                    collect_text_fragments_from_value(child, out);
                }
            }
        }
        _ => {}
    }
}

fn final_text_from_prompt_extra(extra: &Map<String, Value>) -> Option<String> {
    for key in ["message", "output", "response", "text"] {
        if let Some(v) = extra.get(key)
            && let Some(s) = v.as_str()
        {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }

    let mut pieces: Vec<String> = Vec::new();
    for key in ["message", "output", "response", "content", "messages"] {
        if let Some(v) = extra.get(key) {
            collect_text_fragments_from_value(v, &mut pieces);
        }
    }
    pieces.dedup();
    let joined = pieces.join("\n\n").trim().to_string();
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn handoff_from_prompt_hooks(hooks: &StreamTextHooks, pr: PromptResult) -> AgentHandoff {
    let stream = hooks.take_text();
    let final_text = final_text_from_prompt_extra(&pr.extra);
    let create_plan_text = hooks.take_cursor_create_plan_text();
    let has_create_plan_text = !create_plan_text.trim().is_empty();
    let has_final_result_text = final_text.is_some() || has_create_plan_text;

    // Prefer final prompt result text (`message` / `output`) over streamed chunks.
    // Streamed chunks can contain intermediate progress narration, while `extra` carries
    // the end-of-turn canonical answer that downstream parsers should consume.
    let mut response = if let Some(s) = final_text {
        s.to_string()
    } else {
        stream
    };

    if has_create_plan_text {
        if response.trim().is_empty() {
            response = create_plan_text;
        } else {
            response.push_str("\n\n");
            response.push_str(&create_plan_text);
        }
    }

    let cursor_plan_paths = hooks.take_cursor_plan_paths();

    AgentHandoff {
        response,
        has_final_result_text,
        cursor_plan_paths,
        ..Default::default()
    }
}

fn prepare_task_prompt(prompt: &str) -> String {
    format!("{}{}", TASK_CONTEXT_RESET_GUIDANCE, prompt)
}

fn retry_backoff_for_unfinished_task(attempt: u32) -> Duration {
    let capped = attempt.min(5) as u64;
    Duration::from_millis(capped * 200)
}

impl Drop for AcpRuntime {
    fn drop(&mut self) {
        let mut g = self.acp.lock().unwrap();
        if let Some(s) = g.take() {
            dispose_acp_session(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::AcpHooks;
    use super::super::types::InitializeResult;
    use super::*;
    use serde_json::json;

    #[test]
    fn acp_init_detects_cursor_login_in_auth_methods() {
        let with_login: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "agentCapabilities": {},
            "authMethods": [{"id": "cursor_login", "name": "Cursor Login"}]
        }))
        .unwrap();
        assert!(AcpRuntime::acp_init_advertises_cursor_login(&with_login));

        let empty: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": 1,
            "agentCapabilities": {},
            "authMethods": []
        }))
        .unwrap();
        assert!(!AcpRuntime::acp_init_advertises_cursor_login(&empty));
    }

    #[test]
    fn prepends_fresh_context_guidance_to_each_task() {
        let prompt = prepare_task_prompt("Implement issue #42.");
        assert!(prompt.starts_with(TASK_CONTEXT_RESET_GUIDANCE));
        assert!(prompt.ends_with("Implement issue #42."));
    }

    #[test]
    fn unfinished_task_retry_backoff_is_capped() {
        assert_eq!(
            retry_backoff_for_unfinished_task(1),
            Duration::from_millis(200)
        );
        assert_eq!(
            retry_backoff_for_unfinished_task(3),
            Duration::from_millis(600)
        );
        assert_eq!(
            retry_backoff_for_unfinished_task(99),
            Duration::from_millis(1000)
        );
    }

    #[test]
    fn handoff_merges_cursor_create_plan_text() {
        let hooks = StreamTextHooks::new();
        let params = json!({
            "plan": "SUB_ISSUE_1:\nTITLE: Add tests\nPRIORITY: 2\nDESCRIPTION:\nDo it."
        });
        hooks.handle_agent_request("cursor/create_plan", &params, &json!(1));
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn"
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr);
        assert!(h.has_final_result_text);
        assert!(h.response.contains("SUB_ISSUE_1:"));
    }

    #[test]
    fn handoff_prefers_prompt_extra_over_stream_buffer() {
        let hooks = StreamTextHooks::new();
        let params = serde_json::json!({
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": { "text": "from stream" }
            }
        });
        hooks.on_agent_notification("session/update", &params);
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "from result"
        }))
        .unwrap();
        let h = handoff_from_prompt_hooks(&hooks, pr);
        assert_eq!(h.response, "from result");
        assert!(h.has_final_result_text);

        let hooks_m = StreamTextHooks::new();
        hooks_m.on_agent_notification("session/update", &params);
        let pr_m: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "SUB_ISSUE_1:\nTITLE: T\nDESCRIPTION:\nD"
        }))
        .unwrap();
        let hm = handoff_from_prompt_hooks(&hooks_m, pr_m);
        assert_eq!(hm.response, "SUB_ISSUE_1:\nTITLE: T\nDESCRIPTION:\nD");
        assert!(hm.has_final_result_text);

        let hooks2 = StreamTextHooks::new();
        let pr2: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "message": "only result"
        }))
        .unwrap();
        let h2 = handoff_from_prompt_hooks(&hooks2, pr2);
        assert_eq!(h2.response, "only result");
        assert!(h2.has_final_result_text);

        let hooks3 = StreamTextHooks::new();
        hooks3.on_agent_notification("session/update", &params);
        let pr3: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn"
        }))
        .unwrap();
        let h3 = handoff_from_prompt_hooks(&hooks3, pr3);
        assert_eq!(h3.response, "from stream");
        assert!(!h3.has_final_result_text);
    }

    #[test]
    fn handoff_extracts_final_text_from_structured_content_blocks() {
        let hooks = StreamTextHooks::new();
        let pr: PromptResult = serde_json::from_value(json!({
            "stopReason": "end_turn",
            "content": [
                {"type": "text", "text": "SUB_ISSUE_1:"},
                {"type": "text", "text": "TITLE: Refactor queue"},
                {"type": "text", "text": "DESCRIPTION:\nDo the refactor."}
            ]
        }))
        .unwrap();
        let out = handoff_from_prompt_hooks(&hooks, pr);
        assert!(out.has_final_result_text);
        assert!(out.response.contains("SUB_ISSUE_1:"));
        assert!(out.response.contains("TITLE: Refactor queue"));
    }
}
