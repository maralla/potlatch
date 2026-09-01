//! Typed ACP request/response fragments used by [`super::AcpClient`].

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Protocol version sent on `initialize` (numeric per Cursor minimal client example).
pub const DEFAULT_PROTOCOL_VERSION: u64 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientFsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: ClientFsCapabilities,
    #[serde(default)]
    pub terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: u64,
    pub client_capabilities: ClientCapabilities,
    pub client_info: ImplementationInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    #[serde(default)]
    pub protocol_version: Value,
    #[serde(default)]
    pub agent_capabilities: Value,
    #[serde(default)]
    pub auth_methods: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthenticateParams {
    pub method_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<Value>,
    /// Caller-defined structured-output tool definitions. Each entry has
    /// `name`, `description`, and `parameters` (JSON schema). The harness
    /// creates a generic `StructuredOutputTool` per definition and returns
    /// captured output in the `session/prompt` response. Potlatch extension —
    /// not part of the ACP spec; ignored by non-potlatch backends.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "structured_output_tools"
    )]
    pub structured_output_tools: Option<Vec<Value>>,
    /// Tools registered by in-process agents. Potlatch extension; ignored by
    /// ACP servers that do not implement remote agent tools.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "agent_tools"
    )]
    pub agent_tools: Option<Vec<Value>>,
    /// Context channels registered by in-process agents on the bus. Each
    /// entry has a `name` and `content` string; the harness injects each as a
    /// system message at session init. Potlatch extension; ignored by non-potlatch
    /// backends.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "context_channels"
    )]
    pub context_channels: Option<Vec<Value>>,
    /// Directories the harness permits `write`/`edit` to touch outside the
    /// session cwd (via `outside_cwd: true`). Potlatch extension; ignored by
    /// ACP servers that do not implement it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "write_roots"
    )]
    pub write_roots: Option<Vec<String>>,
}

/// One allowed value in a `select` session config option ([`SessionConfigOptionBrief`]).
///
/// See [Session Config Options](https://agentclientprotocol.com/protocol/session-config-options).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionConfigSelectEntry {
    pub value: String,
}

/// Subset of a session config option from `session/new` / `session/set_config_option` results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionConfigOptionBrief {
    pub id: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(rename = "type", default)]
    pub option_type: Option<String>,
    #[serde(default)]
    pub options: Option<Vec<SessionConfigSelectEntry>>,
}

/// One mode from [`SessionModeStateBrief::available_modes`] ([session modes](https://agentclientprotocol.com/protocol/session-modes)).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeEntry {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// `modes` object on `session/new` / legacy mode API ([session modes](https://agentclientprotocol.com/protocol/session-modes)).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeStateBrief {
    pub current_mode_id: String,
    #[serde(default)]
    pub available_modes: Vec<SessionModeEntry>,
}

/// True if `mode_id` is listed in `available_modes` (for future `session/set_mode` validation).
pub fn mode_id_is_available(state: &SessionModeStateBrief, mode_id: &str) -> bool {
    state.available_modes.iter().any(|m| m.id == mode_id)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionResult {
    pub session_id: String,
    #[serde(default)]
    pub config_options: Option<Vec<SessionConfigOptionBrief>>,
    /// Legacy session modes; agents may also expose mode via [`SessionConfigOptionBrief`] (`category: "mode"`).
    #[serde(default)]
    pub modes: Option<SessionModeStateBrief>,
}

/// Mode selector from [`NewSessionResult::config_options`] (`id == "mode"` or `category == "mode"`).
pub fn session_mode_config_option(
    config: &[SessionConfigOptionBrief],
) -> Option<&SessionConfigOptionBrief> {
    config.iter().find(|o| o.id == "mode").or_else(|| {
        config
            .iter()
            .find(|o| o.category.as_deref() == Some("mode"))
    })
}

/// Picks the model selector from [`NewSessionResult::config_options`]: `id == "model"`, else first
/// with `category == "model"` ([ordering](https://agentclientprotocol.com/protocol/session-config-options#option-ordering)).
pub fn model_selector_for_session(
    config: &[SessionConfigOptionBrief],
) -> Option<&SessionConfigOptionBrief> {
    config.iter().find(|o| o.id == "model").or_else(|| {
        config
            .iter()
            .find(|o| o.category.as_deref() == Some("model"))
    })
}

/// True if `value` is listed in the option's `options` array (required by the spec before calling
/// `session/set_config_option`).
pub fn select_option_allows_value(opt: &SessionConfigOptionBrief, value: &str) -> bool {
    if matches!(opt.option_type.as_deref(), Some(t) if t != "select") {
        return false;
    }
    opt.options
        .as_ref()
        .is_some_and(|entries| entries.iter().any(|e| e.value == value))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    #[serde(default)]
    pub stop_reason: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl PromptResult {
    pub fn from_value(v: &Value) -> serde_json::Result<Self> {
        serde_json::from_value(v.clone())
    }
}

/// Build `params` for `session/prompt` with only text blocks.
pub fn text_prompt(session_id: impl Into<String>, text: impl Into<String>) -> Value {
    json!({
        "sessionId": session_id.into(),
        "prompt": [{ "type": "text", "text": text.into() }]
    })
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PromptContentBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[cfg(test)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<PromptContentBlock>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_deserializes_config_options() {
        let v = json!({
            "sessionId": "s1",
            "configOptions": [
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [
                        { "value": "model-1", "name": "M1" },
                        { "value": "model-2", "name": "M2" }
                    ]
                }
            ]
        });
        let r: NewSessionResult = serde_json::from_value(v).unwrap();
        assert_eq!(r.session_id, "s1");
        let opts = r.config_options.as_ref().unwrap();
        let sel = model_selector_for_session(opts).unwrap();
        assert_eq!(sel.id, "model");
        assert!(select_option_allows_value(sel, "model-2"));
        assert!(!select_option_allows_value(sel, "unknown"));
    }

    #[test]
    fn new_session_deserializes_session_modes() {
        let v = json!({
            "sessionId": "s1",
            "modes": {
                "currentModeId": "ask",
                "availableModes": [
                    { "id": "ask", "name": "Ask" },
                    { "id": "code", "name": "Code", "description": "Full access" }
                ]
            }
        });
        let r: NewSessionResult = serde_json::from_value(v).unwrap();
        let m = r.modes.as_ref().unwrap();
        assert_eq!(m.current_mode_id, "ask");
        assert_eq!(m.available_modes.len(), 2);
        assert!(mode_id_is_available(m, "code"));
        assert!(!mode_id_is_available(m, "plan"));
    }

    #[test]
    fn model_selector_prefers_id_model_over_category() {
        let opts = vec![
            SessionConfigOptionBrief {
                id: "other".into(),
                category: Some("model".into()),
                option_type: Some("select".into()),
                options: Some(vec![SessionConfigSelectEntry { value: "a".into() }]),
            },
            SessionConfigOptionBrief {
                id: "model".into(),
                category: None,
                option_type: Some("select".into()),
                options: Some(vec![SessionConfigSelectEntry { value: "b".into() }]),
            },
        ];
        assert_eq!(model_selector_for_session(&opts).unwrap().id, "model");
    }

    #[test]
    fn session_mode_config_option_prefers_id_mode() {
        let opts = vec![
            SessionConfigOptionBrief {
                id: "other".into(),
                category: Some("mode".into()),
                option_type: Some("select".into()),
                options: Some(vec![SessionConfigSelectEntry {
                    value: "ask".into(),
                }]),
            },
            SessionConfigOptionBrief {
                id: "mode".into(),
                category: None,
                option_type: Some("select".into()),
                options: Some(vec![SessionConfigSelectEntry {
                    value: "code".into(),
                }]),
            },
        ];
        assert_eq!(session_mode_config_option(&opts).unwrap().id, "mode");
    }

    #[test]
    fn text_prompt_shape_matches_cursor_docs() {
        let p = text_prompt("s1", "hello");
        assert_eq!(p["sessionId"], "s1");
        assert_eq!(p["prompt"][0]["type"], "text");
        assert_eq!(p["prompt"][0]["text"], "hello");
    }

    #[test]
    fn prompt_params_serializes_like_session_prompt() {
        let p = super::PromptParams {
            session_id: "sid".into(),
            prompt: vec![
                super::PromptContentBlock::Text { text: "hi".into() },
                super::PromptContentBlock::Other,
            ],
        };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["sessionId"], "sid");
        assert_eq!(v["prompt"][0]["type"], "text");
    }
}
