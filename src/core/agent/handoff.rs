use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffSubIssue {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentHandoff {
    #[serde(default)]
    pub response: String,
    /// True when ACP `session/prompt` returned non-empty final text (`message`/`output`).
    /// Callers use this to avoid parsing stream-only intermediate chunks.
    #[serde(default)]
    pub has_final_result_text: bool,
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub mr_title: Option<String>,
    #[serde(default)]
    pub mr_description: Option<String>,
    #[serde(default)]
    pub changes_summary: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub needs_split: Option<String>,
    #[serde(default)]
    pub needs_clarification: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub lgtm: Option<String>,
    #[serde(default)]
    pub sub_issues: Vec<HandoffSubIssue>,
    /// Absolute paths from Cursor plan-mode `tool_call_update` ("Plan saved to file://…").
    #[serde(default)]
    pub cursor_plan_paths: Vec<String>,
}
