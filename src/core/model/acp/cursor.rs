//! Cursor ACP extension support.
//!
//! Encapsulates all [Cursor ACP extension](https://cursor.com/docs/cli/acp)
//! methods (`cursor/ask_question`, `cursor/create_plan`, `cursor/update_todos`,
//! etc.) so the core ACP runtime and hooks stay generic. The harness (potlatch's
//! own ACP server) never sends `cursor/` requests — these are only triggered
//! when the backend is the Cursor CLI.
//!
//! Implements [`AcpVendorExtension`] and [`AcpVendorState`] from
//! [`super::vendor`]. The runtime creates a [`CursorExtension`] when the model
//! URI vendor is `cursor`; all vendor-specific logic stays here.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use super::capabilities::{AskAnswer, AskChoice, AskQuestion, CapabilityProvider};
use super::client::AcpClient;
use super::types::{
    InitializeResult, NewSessionResult, SessionModeStateBrief, mode_id_is_available,
    select_option_allows_value, session_mode_config_option,
};
use super::vendor::{AcpVendorExtension, AcpVendorState};

// ---------------------------------------------------------------------------
// Headless replies for cursor/* extension requests
// ---------------------------------------------------------------------------

/// Markdown body from a `cursor/create_plan` request.
fn extract_create_plan_text(params: &Value) -> Option<String> {
    let plan = params.get("plan").and_then(Value::as_str)?.trim();
    if plan.is_empty() {
        return None;
    }

    let mut out = String::new();
    if let Some(name) = params.get("name").and_then(Value::as_str) {
        let name = name.trim();
        if !name.is_empty() {
            out.push_str("# ");
            out.push_str(name);
            out.push_str("\n\n");
        }
    }
    if let Some(overview) = params.get("overview").and_then(Value::as_str) {
        let overview = overview.trim();
        if !overview.is_empty() {
            out.push_str(overview);
            out.push_str("\n\n");
        }
    }
    out.push_str(plan);
    Some(out)
}

fn headless_create_plan_reply() -> Value {
    json!({ "outcome": { "outcome": "accepted" } })
}

fn headless_ask_question_reply(params: &Value) -> Value {
    if let Some(opts) = params.get("options").and_then(|v| v.as_array())
        && let Some(first) = opts.first()
        && let Some(id) = first
            .get("id")
            .or_else(|| first.get("optionId"))
            .or_else(|| first.get("option_id"))
            .and_then(|v| v.as_str())
    {
        return json!({
            "outcome": {
                "outcome": "answered",
                "answers": [{
                    "questionId": params
                        .get("questions")
                        .and_then(|v| v.as_array())
                        .and_then(|q| q.first())
                        .and_then(|q| q.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("q1"),
                    "selectedOptionIds": [id]
                }]
            }
        });
    }
    if let Some(questions) = params.get("questions").and_then(|v| v.as_array())
        && let Some(first_q) = questions.first()
        && let Some(qid) = first_q.get("id").and_then(|v| v.as_str())
        && let Some(opts) = first_q.get("options").and_then(|v| v.as_array())
        && let Some(first_opt) = opts.first()
        && let Some(oid) = first_opt.get("id").and_then(|v| v.as_str())
    {
        return json!({
            "outcome": {
                "outcome": "answered",
                "answers": [{
                    "questionId": qid,
                    "selectedOptionIds": [oid]
                }]
            }
        });
    }
    json!({
        "outcome": {
            "outcome": "answered",
            "answers": [{ "questionId": "q1", "selectedOptionIds": ["0"] }]
        }
    })
}

fn headless_extension_reply(method: &str, params: &Value) -> Value {
    match method {
        "cursor/create_plan" => headless_create_plan_reply(),
        "cursor/ask_question" => headless_ask_question_reply(params),
        "cursor/update_todos" => json!({
            "outcome": { "outcome": "accepted", "todos": params.get("todos").cloned().unwrap_or_else(|| json!([])) }
        }),
        "cursor/task" => json!({ "outcome": { "outcome": "completed" } }),
        "cursor/generate_image" => json!({
            "outcome": { "outcome": "generated", "filePath": params.get("filePath").and_then(Value::as_str).unwrap_or("") }
        }),
        other => {
            warn!(target: "potlatch::acp_cursor", "auto-accepting unhandled extension request `{other}`");
            json!({ "outcome": { "outcome": "accepted" } })
        }
    }
}

// ---------------------------------------------------------------------------
// Ask question parsing / reply formatting (cursor/ask_question ↔ neutral model)
// ---------------------------------------------------------------------------

fn option_entry_id(opt: &Value) -> Option<&str> {
    opt.get("id")
        .or_else(|| opt.get("optionId"))
        .or_else(|| opt.get("option_id"))
        .and_then(|v| v.as_str())
}

fn option_entry_label(opt: &Value) -> String {
    for k in ["label", "title", "name", "text"] {
        if let Some(s) = opt.get(k).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return s.to_string();
        }
    }
    option_entry_id(opt).unwrap_or("(option)").to_string()
}

/// Option array from a `cursor/ask_question` request: top-level `options`, or
/// the first entry's `options` under `questions`.
fn question_options(params: &Value) -> Option<&Vec<Value>> {
    params
        .get("options")
        .and_then(|v| v.as_array())
        .or_else(|| {
            params
                .get("questions")
                .and_then(|v| v.as_array())
                .and_then(|q| q.first())
                .and_then(|q| q.get("options"))
                .and_then(|v| v.as_array())
        })
}

/// Question id from a `cursor/ask_question` request (first `questions[].id`,
/// else `"q1"`), used when formatting the reply.
fn question_id_from_params(params: &Value) -> &str {
    params
        .get("questions")
        .and_then(|v| v.as_array())
        .and_then(|q| q.first())
        .and_then(|q| q.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("q1")
}

/// Extract question prompt text from a `cursor/ask_question` request.
fn extract_question_text(params: &Value) -> String {
    for k in ["question", "message", "text", "prompt", "title", "body"] {
        if let Some(s) = params.get(k).and_then(|v| v.as_str()) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    if let Some(question) = params
        .get("questions")
        .and_then(|v| v.as_array())
        .and_then(|q| q.first())
    {
        for k in ["prompt", "question", "text", "title"] {
            if let Some(s) = question.get(k).and_then(|v| v.as_str()) {
                let t = s.trim();
                if !t.is_empty() {
                    return t.to_string();
                }
            }
        }
    }
    params
        .as_object()
        .map(|o| {
            let preview: serde_json::Map<String, Value> = o
                .iter()
                .filter(|(key, _)| *key != "sessionId" && *key != "session_id")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            serde_json::to_string_pretty(&preview).unwrap_or_else(|_| "{}".to_string())
        })
        .filter(|s| s != "{}")
        .unwrap_or_else(|| "_(no question text in params)_".to_string())
}

/// Parse a `cursor/ask_question` request into the neutral [`AskQuestion`].
fn parse_ask_question(params: &Value) -> AskQuestion {
    let text = extract_question_text(params);
    let choices = question_options(params)
        .map(|opts| {
            opts.iter()
                .map(|o| AskChoice {
                    id: option_entry_id(o).unwrap_or("").to_string(),
                    label: option_entry_label(o),
                })
                .collect()
        })
        .unwrap_or_default();
    AskQuestion { text, choices }
}

/// Format a neutral [`AskAnswer`] as a `cursor/ask_question` JSON-RPC `result`.
fn format_ask_reply(params: &Value, answer: &AskAnswer) -> Value {
    let question_id = question_id_from_params(params);
    match answer {
        AskAnswer::Choice(id) | AskAnswer::FreeText(id) => json!({
            "outcome": {
                "outcome": "answered",
                "answers": [{
                    "questionId": question_id,
                    "selectedOptionIds": [id]
                }]
            }
        }),
        AskAnswer::Auto => headless_ask_question_reply(params),
    }
}

// ---------------------------------------------------------------------------
// CursorPlanState — per-session vendor state (AcpVendorState)
// ---------------------------------------------------------------------------

/// `current_mode_update` notification → mode id.
fn extract_current_mode_update(params: &Value) -> Option<String> {
    let update = params.get("update")?;
    let kind = update
        .get("sessionUpdate")
        .or_else(|| update.get("session_update"))
        .and_then(|v| v.as_str())?;
    if kind != "current_mode_update" && kind != "currentModeUpdate" {
        return None;
    }
    update
        .get("modeId")
        .or_else(|| update.get("mode_id"))
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)
}

/// Cursor-specific per-session state. Stored on `StreamTextHooks` via the
/// `AcpVendorState` trait. Plan-text and mode tracking are internal — not
/// exposed to generic code.
pub struct CursorPlanState {
    provider: Option<Arc<dyn CapabilityProvider>>,
    create_plan_text: Mutex<String>,
    session_modes: Mutex<Option<SessionModeStateBrief>>,
    notification_seq: AtomicU64,
}

impl CursorPlanState {
    pub fn new(provider: Option<Arc<dyn CapabilityProvider>>) -> Self {
        Self {
            provider,
            create_plan_text: Mutex::new(String::new()),
            session_modes: Mutex::new(None),
            notification_seq: AtomicU64::new(0),
        }
    }

    fn apply_current_mode_update(&self, mode_id: String) {
        let mut g = self.session_modes.lock().unwrap();
        match g.as_mut() {
            Some(s) => s.current_mode_id = mode_id,
            None => {
                *g = Some(SessionModeStateBrief {
                    current_mode_id: mode_id,
                    available_modes: Vec::new(),
                });
            }
        }
    }

    fn handle_create_plan(&self, params: &Value) -> Value {
        if let Some(plan) = extract_create_plan_text(params) {
            info!(target: "potlatch::acp_cursor", plan_len = plan.len(), "Captured create_plan markdown");
            *self.create_plan_text.lock().unwrap() = plan;
        }
        headless_create_plan_reply()
    }
}

impl AcpVendorState for CursorPlanState {
    fn handle_agent_request(&self, method: &str, params: &Value, _id: &Value) -> Option<Value> {
        if method == "cursor/ask_question" {
            let question = parse_ask_question(params);
            let answer = match &self.provider {
                Some(p) => p.ask(&question),
                None => AskAnswer::Auto,
            };
            return Some(format_ask_reply(params, &answer));
        }
        if method == "cursor/create_plan" {
            return Some(self.handle_create_plan(params));
        }
        if method.starts_with("cursor/") {
            debug!(target: "potlatch::acp_cursor", %method, "handling extension request");
            return Some(headless_extension_reply(method, params));
        }
        None
    }

    fn on_session_update(&self, params: &Value) -> bool {
        self.notification_seq.fetch_add(1, Ordering::SeqCst);

        if let Some(mode_id) = extract_current_mode_update(params) {
            debug!(target: "potlatch::acp_modes", "current_mode_update: {}", mode_id);
            self.apply_current_mode_update(mode_id);
            return true;
        }
        false
    }

    fn clear(&self) {
        self.create_plan_text.lock().unwrap().clear();
        self.notification_seq.store(0, Ordering::SeqCst);
    }

    fn seed_session_modes(&self, modes: &SessionModeStateBrief) {
        *self.session_modes.lock().unwrap() = Some(modes.clone());
    }

    fn sync_current_mode(&self, mode_id: &str) {
        self.apply_current_mode_update(mode_id.to_string());
    }

    fn notification_seq(&self) -> u64 {
        self.notification_seq.load(Ordering::SeqCst)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// CursorExtension — vendor extension (AcpVendorExtension)
// ---------------------------------------------------------------------------

/// Cursor ACP vendor extension. Auto-detected from the model URI vendor.
/// Owns preferred session mode and plan-followup logic.
pub struct CursorExtension {
    preferred_session_mode: Option<&'static str>,
}

impl CursorExtension {
    /// Create a Cursor extension if the model URI vendor is `cursor`.
    pub fn new(model_uri: Option<&str>) -> Option<Self> {
        let is_cursor = model_uri
            .and_then(|uri| crate::core::config::uri::ModelUri::parse(uri).ok())
            .is_some_and(|u| u.vendor == "cursor");

        if !is_cursor {
            return None;
        }

        Some(Self {
            preferred_session_mode: None,
        })
    }

    pub fn set_preferred_session_mode(&mut self, mode: Option<&'static str>) {
        self.preferred_session_mode = mode;
    }

    fn is_plan_mode(&self) -> bool {
        self.preferred_session_mode
            .is_some_and(|m| m.eq_ignore_ascii_case("plan"))
    }
}

impl AcpVendorExtension for CursorExtension {
    fn create_state(
        &self,
        provider: Option<Arc<dyn CapabilityProvider>>,
    ) -> Arc<dyn AcpVendorState> {
        Arc::new(CursorPlanState::new(provider))
    }

    fn authenticate(&self, client: &AcpClient, init: &InitializeResult) -> Result<()> {
        let has_cursor_login = init.auth_methods.iter().any(|m| {
            m.get("id")
                .or_else(|| m.get("methodId"))
                .and_then(|v| v.as_str())
                .is_some_and(|id| id == "cursor_login")
        });

        if has_cursor_login {
            debug!(target: "potlatch::acp", "authMethods includes cursor_login; calling authenticate");
            client.authenticate_cursor_login().context(
                "ACP authenticate (cursor_login). Run `agent login` or set CURSOR_API_KEY / CURSOR_AUTH_TOKEN",
            )?;
        }
        Ok(())
    }

    /// Applies [`Self::preferred_session_mode`] only when the agent advertises that mode:
    ///
    /// - Config option with `id`/`category` `mode` and the value in `options`, or
    /// - Legacy `session/new` `modes.availableModes` non-empty and containing the id.
    ///
    /// Otherwise leaves the agent default and logs at `debug` (`potlatch::acp_modes`).
    fn try_apply_preferred_mode(
        &self,
        client: &AcpClient,
        session: &NewSessionResult,
        state: &dyn AcpVendorState,
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
                        target: "potlatch::acp_modes",
                        mode = mode_id,
                        "session mode via session/set_config_option"
                    );
                    state.sync_current_mode(mode_id);
                    return;
                }
                Err(e) => debug!(
                    target: "potlatch::acp_modes",
                    err = %e,
                    "session/set_config_option for mode failed",
                ),
            }
            if legacy_advertises {
                match client.session_set_mode(&session.session_id, mode_id) {
                    Ok(_) => {
                        info!(
                            target: "potlatch::acp_modes",
                            mode = mode_id,
                            "session mode via session/set_mode (fallback)"
                        );
                        state.sync_current_mode(mode_id);
                    }
                    Err(e) => debug!(
                        target: "potlatch::acp_modes",
                        err = %e,
                        "session/set_mode fallback after set_config_option failure also failed",
                    ),
                }
            } else {
                debug!(
                    target: "potlatch::acp_modes",
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
                        target: "potlatch::acp_modes",
                        mode = mode_id,
                        "session mode via session/set_mode"
                    );
                    state.sync_current_mode(mode_id);
                }
                Err(e) => debug!(
                    target: "potlatch::acp_modes",
                    err = %e,
                    "session/set_mode failed",
                ),
            }
            return;
        }

        debug!(
            target: "potlatch::acp_modes",
            preferred = mode_id,
            "preferred session mode not advertised (no mode config option value match and no legacy availableModes entry); leaving agent default mode",
        );
    }

    fn wait_for_followup(
        &self,
        state: &dyn AcpVendorState,
        cancel_check: Option<&dyn Fn() -> bool>,
        shutdown: &AtomicBool,
    ) -> Result<()> {
        if !self.is_plan_mode() {
            return Ok(());
        }

        const MAX_WAIT: Duration = Duration::from_secs(3);
        const QUIET_WINDOW: Duration = Duration::from_millis(500);
        const POLL: Duration = Duration::from_millis(100);

        let start = Instant::now();
        let mut last_change_at = Instant::now();
        let mut last_seq = state.notification_seq();

        loop {
            if start.elapsed() >= MAX_WAIT || last_change_at.elapsed() >= QUIET_WINDOW {
                return Ok(());
            }
            if shutdown.load(Ordering::SeqCst) {
                anyhow::bail!("Agent interrupted by shutdown");
            }
            if let Some(check) = cancel_check
                && check()
            {
                anyhow::bail!("Agent cancelled by external condition");
            }

            let seq = state.notification_seq();
            if seq != last_seq {
                last_seq = seq;
                last_change_at = Instant::now();
            }
            std::thread::sleep(POLL);
        }
    }

    fn process_response(&self, state: &dyn AcpVendorState, response: &mut String) {
        let any = state.as_any();
        let Some(cs) = any.downcast_ref::<CursorPlanState>() else {
            return;
        };

        // Merge create_plan text into the response.
        let plan_text = std::mem::take(&mut *cs.create_plan_text.lock().unwrap());
        if !plan_text.trim().is_empty() {
            if response.trim().is_empty() {
                *response = plan_text;
            } else {
                response.push_str("\n\n");
                response.push_str(&plan_text);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mode_params() -> Value {
        json!({
            "questions": [{
                "id": "mode",
                "prompt": "Choose a mode",
                "options": [
                    { "id": "guide", "label": "Guide worker" },
                    { "id": "split", "label": "Split issue" }
                ]
            }]
        })
    }

    #[test]
    fn parse_ask_question_extracts_text_and_choices() {
        let q = parse_ask_question(&mode_params());
        assert_eq!(q.text, "Choose a mode");
        assert_eq!(q.choices.len(), 2);
        assert_eq!(q.choices[0].id, "guide");
        assert_eq!(q.choices[0].label, "Guide worker");
        assert_eq!(q.choices[1].id, "split");
        assert_eq!(q.choices[1].label, "Split issue");
    }

    #[test]
    fn format_ask_reply_choice_uses_question_id_and_selected_option() {
        let params = mode_params();
        let r = format_ask_reply(&params, &AskAnswer::Choice("split".into()));
        assert_eq!(r["outcome"]["outcome"], "answered");
        assert_eq!(r["outcome"]["answers"][0]["questionId"], "mode");
        assert_eq!(r["outcome"]["answers"][0]["selectedOptionIds"][0], "split");
    }

    #[test]
    fn format_ask_reply_free_text_echoes_as_selected_option_id() {
        let params = mode_params();
        let r = format_ask_reply(&params, &AskAnswer::FreeText("something".into()));
        assert_eq!(r["outcome"]["outcome"], "answered");
        assert_eq!(
            r["outcome"]["answers"][0]["selectedOptionIds"][0],
            "something"
        );
    }

    #[test]
    fn format_ask_reply_auto_picks_first_option() {
        let params = mode_params();
        let r = format_ask_reply(&params, &AskAnswer::Auto);
        assert_eq!(r["outcome"]["outcome"], "answered");
        assert_eq!(r["outcome"]["answers"][0]["selectedOptionIds"][0], "guide");
    }

    #[test]
    fn format_ask_reply_auto_with_no_options_falls_back_to_default_id() {
        let params = json!({ "sessionId": "s" });
        let r = format_ask_reply(&params, &AskAnswer::Auto);
        assert_eq!(r["outcome"]["outcome"], "answered");
        assert_eq!(r["outcome"]["answers"][0]["selectedOptionIds"][0], "0");
    }

    #[test]
    fn handle_agent_request_dispatches_ask_to_provider() {
        struct Echo;
        impl CapabilityProvider for Echo {
            fn ask(&self, question: &AskQuestion) -> AskAnswer {
                AskAnswer::Choice(question.choices[1].id.clone())
            }
        }

        let state = CursorPlanState::new(Some(Arc::new(Echo)));
        let result = state.handle_agent_request("cursor/ask_question", &mode_params(), &json!(1));
        let r = result.expect("ask_question handled");
        assert_eq!(r["outcome"]["answers"][0]["selectedOptionIds"][0], "split");
    }

    #[test]
    fn handle_agent_request_ask_without_provider_uses_auto() {
        let state = CursorPlanState::new(None);
        let result = state.handle_agent_request("cursor/ask_question", &mode_params(), &json!(1));
        let r = result.expect("ask_question handled");
        assert_eq!(r["outcome"]["answers"][0]["selectedOptionIds"][0], "guide");
    }
}
