//! Generic proxy tools registered by in-process agents.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::Tool;
use crate::core::bus::RemoteAgentToolDefinition;

pub trait AgentToolCaller: Send + Sync {
    fn call(&self, target: &str, operation: &str, arguments: Value) -> Result<Value>;
}

pub struct RemoteAgentTool {
    caller: Arc<dyn AgentToolCaller>,
    definition: RemoteAgentToolDefinition,
}

impl RemoteAgentTool {
    pub(crate) fn new(
        caller: Arc<dyn AgentToolCaller>,
        definition: RemoteAgentToolDefinition,
    ) -> Self {
        Self { caller, definition }
    }
}

impl Tool for RemoteAgentTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "description": self.definition.description,
            "parameters": self.definition.parameters,
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let result = self.caller.call(
            &self.definition.target,
            &self.definition.operation,
            args.clone(),
        )?;
        serde_json::to_string_pretty(&result).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoCaller;

    impl AgentToolCaller for EchoCaller {
        fn call(&self, _target: &str, _operation: &str, arguments: Value) -> Result<Value> {
            Ok(arguments)
        }
    }

    fn tool() -> RemoteAgentTool {
        RemoteAgentTool::new(
            Arc::new(EchoCaller),
            RemoteAgentToolDefinition {
                name: "example".to_string(),
                description: "Agent-provided description.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"]
                }),
                target: "provider".to_string(),
                operation: "run".to_string(),
            },
        )
    }

    #[test]
    fn proxy_schema_is_owned_by_the_registering_agent() {
        let tool = tool();
        assert_eq!(tool.name(), "example");
        assert_eq!(tool.schema()["description"], "Agent-provided description.");
        assert_eq!(
            tool.schema()["parameters"]["required"],
            serde_json::json!(["value"])
        );
    }

    #[test]
    fn proxy_returns_the_parent_call_result() {
        let result = tool()
            .execute(&serde_json::json!({"value": "hello"}), "/tmp")
            .unwrap();
        assert!(result.contains("hello"));
    }
}
