use serde::{Deserialize, Serialize};

/// Backend-neutral model completion transport: response text plus any
/// structured-output tool calls the backend captured. Role-specific typed
/// decoding happens per agent via [`super::schema::StructuredOutput`] and
/// [`super::AgentModel::complete_typed`] — this type carries no role-specific
/// fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentHandoff {
    #[serde(default)]
    pub response: String,
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
