use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffSubIssue {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub priority: Option<u8>,
    /// 1-based index of another sub-issue in the same split that this one
    /// depends on (i.e. this sub-issue cannot start until the referenced
    /// sub-issue is done). The PMO applies a `waiting-on-issue:#N` label
    /// after both issues are created, and the worker skips the issue until
    /// the dependency is closed. 0/absent means no dependency.
    #[serde(default)]
    pub depends_on: usize,
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
    /// IID of an existing open issue that this issue depends on, declared by
    /// the PMO via the `wait_for_dependency` decision. The PMO applies a
    /// `waiting-on-issue:#N` label so the worker parks the issue until the
    /// dependency closes. `None` when not declared.
    #[serde(default)]
    pub depends_on_issue: Option<u64>,
    /// Absolute paths from Cursor plan-mode `tool_call_update` ("Plan saved to file://…").
    #[serde(default)]
    pub cursor_plan_paths: Vec<String>,
    /// Structured JSON plan emitted by the model via the `plan` tool
    /// (potlatch harness ACP backend, plan mode only). `None` when the model
    /// didn't call the tool, the session wasn't in plan mode, or when using
    /// a non-potlatch ACP backend (e.g. Cursor).
    #[serde(default)]
    pub plan_output: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_output_defaults_to_none() {
        let h = AgentHandoff::default();
        assert!(h.plan_output.is_none());
    }

    #[test]
    fn plan_output_serializes_and_deserializes() {
        let h = AgentHandoff {
            response: "text".into(),
            plan_output: Some(serde_json::json!({"decision": "split"})),
            ..Default::default()
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: AgentHandoff = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.plan_output,
            Some(serde_json::json!({"decision": "split"}))
        );
    }

    #[test]
    fn plan_output_absent_in_json_deserializes_to_none() {
        // JSON without the plan_output field should deserialize to None.
        let json = r#"{"response":"text"}"#;
        let h: AgentHandoff = serde_json::from_str(json).unwrap();
        assert_eq!(h.response, "text");
        assert!(h.plan_output.is_none());
    }

    #[test]
    fn plan_output_null_in_json_deserializes_to_none() {
        let json = r#"{"response":"text","plan_output":null}"#;
        let h: AgentHandoff = serde_json::from_str(json).unwrap();
        assert_eq!(h.response, "text");
        assert!(h.plan_output.is_none());
    }
}
