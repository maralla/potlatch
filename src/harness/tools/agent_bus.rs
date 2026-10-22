//! Generic proxy tools registered by in-process agents.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::Tool;
use crate::core::bus::RemoteAgentToolDefinition;
use crate::core::scope::SessionScope;

pub trait AgentToolCaller: Send + Sync {
    /// Forward a tool call to the owning agent. `session_id` is the
    /// calling harness session — transport metadata for the target agent,
    /// never part of the tool arguments.
    fn call(
        &self,
        target: &str,
        operation: &str,
        arguments: Value,
        session_id: &str,
    ) -> Result<Value>;
}

pub struct RemoteAgentTool {
    caller: Arc<dyn AgentToolCaller>,
    definition: RemoteAgentToolDefinition,
    /// The harness session this proxy is registered for. Forwarded with
    /// every call so bus-served agents can scope per-caller state (e.g.
    /// the subagent agent owns each subagent to its caller's task session).
    session_id: String,
}

impl RemoteAgentTool {
    pub(crate) fn new(
        caller: Arc<dyn AgentToolCaller>,
        definition: RemoteAgentToolDefinition,
        session_id: &str,
    ) -> Self {
        Self {
            caller,
            definition,
            session_id: session_id.to_string(),
        }
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

    fn execute(&self, args: &Value, scope: &SessionScope) -> Result<String> {
        // The session's file scope rides the call as request metadata (like
        // the session id): bus-served tools that need it (the sandal agent's
        // shell tool boots the session's VM with the scope's cwd and shares
        // its roots into it) parse it back out; every other handler ignores
        // the keys.
        let mut arguments = args.clone();
        scope.write_metadata(&mut arguments);
        let result = self.caller.call(
            &self.definition.target,
            &self.definition.operation,
            arguments,
            &self.session_id,
        )?;
        match result {
            Value::String(text) => Ok(text),
            other => serde_json::to_string_pretty(&other).map_err(Into::into),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoCaller;
    struct TextCaller;

    impl AgentToolCaller for EchoCaller {
        fn call(
            &self,
            _target: &str,
            _operation: &str,
            arguments: Value,
            _session_id: &str,
        ) -> Result<Value> {
            Ok(arguments)
        }
    }

    impl AgentToolCaller for TextCaller {
        fn call(
            &self,
            _target: &str,
            _operation: &str,
            _arguments: Value,
            _session_id: &str,
        ) -> Result<Value> {
            Ok(Value::String("# Rendered\n\nContent.".to_string()))
        }
    }

    fn tool_with_caller(caller: Arc<dyn AgentToolCaller>) -> RemoteAgentTool {
        RemoteAgentTool::new(
            caller,
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
            "caller-session-1",
        )
    }

    fn scope() -> SessionScope {
        SessionScope::new("/tmp", ["/tmp/project-sessions"])
    }

    #[test]
    fn proxy_passes_the_calling_session_as_metadata() {
        // Bus-served agents scope per-caller state (e.g. subagent ownership)
        // by the calling harness session; the proxy forwards it with every
        // call — as transport metadata, never inside the tool arguments.
        let received = Arc::new(std::sync::Mutex::new(None::<String>));
        let capture = Arc::clone(&received);
        struct CapturingCaller(Arc<std::sync::Mutex<Option<String>>>);
        impl AgentToolCaller for CapturingCaller {
            fn call(
                &self,
                _target: &str,
                _operation: &str,
                arguments: Value,
                session_id: &str,
            ) -> Result<Value> {
                *self.0.lock().unwrap() = Some(session_id.to_string());
                Ok(arguments)
            }
        }
        let tool = tool_with_caller(Arc::new(CapturingCaller(capture)));
        let result = tool
            .execute(&serde_json::json!({"value": "hello"}), &scope())
            .unwrap();
        assert_eq!(
            received.lock().unwrap().as_deref(),
            Some("caller-session-1")
        );
        // The session's working directory and declared write roots ride the
        // call as request metadata too (like the session id): bus-served
        // tools that need them (the sandal agent's shell tool boots the
        // session's VM with the cwd and shares the roots into it) read them
        // from `_session_cwd`/`_session_write_roots`; every other handler
        // ignores them. An agent that declared no roots sends an empty list —
        // the metadata is always present, never silently absent. Key order
        // is the pretty-printer's (alphabetical).
        assert_eq!(
            result,
            "{\n  \"_session_cwd\": \"/tmp\",\n  \"_session_write_roots\": [\n    \"/tmp/project-sessions\"\n  ],\n  \"value\": \"hello\"\n}"
        );
    }

    fn tool() -> RemoteAgentTool {
        tool_with_caller(Arc::new(EchoCaller))
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
            .execute(&serde_json::json!({"value": "hello"}), &scope())
            .unwrap();
        assert!(result.contains("hello"));
    }

    #[test]
    fn proxy_returns_string_results_without_json_encoding() {
        let result = tool_with_caller(Arc::new(TextCaller))
            .execute(&serde_json::json!({}), &scope())
            .unwrap();
        assert_eq!(result, "# Rendered\n\nContent.");
    }
}
