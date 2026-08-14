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
    pub reason: Option<String>,
    #[serde(default)]
    pub needs_clarification: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub sub_issues: Vec<HandoffSubIssue>,
    /// IID of an existing open issue that this issue depends on, declared by
    /// the PMO via the `wait_for_dependency` decision. The PMO applies a
    /// `waiting-on-issue:#N` label so the worker parks the issue until the
    /// dependency closes. `None` when not declared.
    #[serde(default)]
    pub depends_on_issue: Option<u64>,
    /// Captured JSON from caller-defined structured-output tools (e.g. the
    /// worker's `handoff` tool). A JSON object mapping tool name to the args
    /// the model passed. `None` when no structured-output tools were
    /// registered or the model didn't call them.
    #[serde(default)]
    pub structured_outputs: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_outputs_defaults_to_none() {
        let h = AgentHandoff::default();
        assert!(h.structured_outputs.is_none());
    }

    #[test]
    fn structured_outputs_serializes_and_deserializes() {
        let h = AgentHandoff {
            response: "text".into(),
            structured_outputs: Some(serde_json::json!({"plan": {"decision": "split"}})),
            ..Default::default()
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: AgentHandoff = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.structured_outputs,
            Some(serde_json::json!({"plan": {"decision": "split"}}))
        );
    }

    #[test]
    fn structured_outputs_absent_in_json_deserializes_to_none() {
        let json = r#"{"response":"text"}"#;
        let h: AgentHandoff = serde_json::from_str(json).unwrap();
        assert_eq!(h.response, "text");
        assert!(h.structured_outputs.is_none());
    }
}
