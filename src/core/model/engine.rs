use std::sync::Arc;

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::core::agent::InvokeOptions;
use crate::core::agent::schema::{ObjectSchema, SchemaField, StructuredOutputTool};
use crate::core::config::{AcpSpawnConfig, Config};
use crate::core::model::acp::AcpRuntime;

/// Model backend boundary: converts the neutral [`StructuredOutputTool`]
/// contract into the ACP/harness `structured_output_tools` wire JSON
/// (`{"name", "description", "parameters"}` with a JSON-schema `parameters`
/// object). The harness (ACP runtime, `NewSessionParams`) only ever sees
/// this raw JSON and stays fully generic — it has no notion of
/// [`SchemaField`] or [`ObjectSchema`].
pub(crate) fn structured_output_tools_wire_json(tools: &[StructuredOutputTool]) -> Vec<Value> {
    tools.iter().map(structured_output_tool_wire_json).collect()
}

fn structured_output_tool_wire_json(tool: &StructuredOutputTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": object_schema_wire_json(&tool.parameters),
    })
}

fn object_schema_wire_json(schema: &ObjectSchema) -> Value {
    let mut properties = Map::new();
    for (name, field) in &schema.properties {
        properties.insert(name.clone(), schema_field_wire_json(field));
    }
    let mut obj = json!({
        "type": "object",
        "properties": Value::Object(properties),
    });
    if !schema.required.is_empty() {
        obj["required"] = Value::Array(schema.required.iter().cloned().map(Value::from).collect());
    }
    obj
}

fn schema_field_wire_json(field: &SchemaField) -> Value {
    match field {
        SchemaField::String {
            description,
            enum_values,
        } => {
            let mut obj = json!({"type": "string", "description": description});
            if !enum_values.is_empty() {
                obj["enum"] = Value::Array(enum_values.iter().cloned().map(Value::from).collect());
            }
            obj
        }
        SchemaField::Integer {
            description,
            enum_values,
        } => {
            let mut obj = json!({"type": "integer", "description": description});
            if !enum_values.is_empty() {
                obj["enum"] = Value::Array(enum_values.iter().cloned().map(Value::from).collect());
            }
            obj
        }
        SchemaField::Boolean { description } => {
            json!({"type": "boolean", "description": description})
        }
        SchemaField::Array { description, items } => {
            json!({
                "type": "array",
                "description": description,
                "items": schema_field_wire_json(items),
            })
        }
        SchemaField::Object(schema) => object_schema_wire_json(schema),
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelSessionOptions {
    pub preferred_session_mode: Option<&'static str>,
    pub structured_output_tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct AcpBuildOptions {
    pub preferred_session_mode: Option<&'static str>,
    pub structured_output_tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct ModelRuntimeContext {
    pub repo_path: String,
    pub shutdown: Arc<std::sync::atomic::AtomicBool>,
    pub agent_id: String,
}

pub(crate) struct ModelEngine {
    inner: AcpRuntime,
}

/// Build a [`ModelEngine`] from config, an agent section, and runtime context.
pub(crate) fn spawn_model_engine(
    config: &Config,
    section: &crate::core::config::AgentSection,
    repo_path: impl Into<String>,
    agent_id: impl Into<String>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    session: ModelSessionOptions,
) -> Result<ModelEngine> {
    ModelEngine::from_agent_section(
        config,
        section,
        ModelRuntimeContext {
            repo_path: repo_path.into(),
            shutdown,
            agent_id: agent_id.into(),
        },
        session,
    )
}

impl ModelEngine {
    pub fn from_agent_section(
        config: &Config,
        section: &crate::core::config::AgentSection,
        runtime: ModelRuntimeContext,
        session: ModelSessionOptions,
    ) -> Result<Self> {
        let acp_spawn = config.resolve_acp_spawn(section)?;
        let acp_opts = AcpBuildOptions {
            preferred_session_mode: session.preferred_session_mode,
            structured_output_tools: session.structured_output_tools,
        };
        Ok(Self::build_from_acp_spawn(acp_spawn, acp_opts, runtime))
    }

    fn build_from_acp_spawn(
        acp_spawn: AcpSpawnConfig,
        opts: AcpBuildOptions,
        runtime: ModelRuntimeContext,
    ) -> Self {
        Self::wrap_runtime(AcpRuntime::new(
            runtime.repo_path,
            acp_spawn.model_uri,
            acp_spawn.endpoint_model,
            acp_spawn.command,
            acp_spawn.env,
            opts.preferred_session_mode,
            opts.structured_output_tools,
            runtime.shutdown,
            runtime.agent_id,
        ))
    }

    fn wrap_runtime(inner: AcpRuntime) -> Self {
        Self { inner }
    }

    pub fn invoke(
        &self,
        prompt: &str,
        options: &InvokeOptions,
    ) -> Result<crate::core::agent::ModelResponse> {
        let cancel_check = options
            .cancel_check
            .as_ref()
            .map(|check| check.as_ref() as &dyn Fn() -> bool);
        let follow_up_poll = options
            .follow_up_poll
            .as_ref()
            .map(|f| f.as_ref() as &dyn Fn() -> Vec<String>);
        let handoff = self
            .inner
            .run_with_cancel(prompt, cancel_check, follow_up_poll)?;
        Ok(crate::core::agent::ModelResponse { handoff })
    }

    /// Set the capability provider (called by the agent at construction).
    pub fn set_capability_provider(
        &self,
        provider: Option<Arc<dyn crate::core::model::acp::capabilities::CapabilityProvider>>,
    ) {
        self.inner.set_capability_provider(provider);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;
    use std::sync::atomic::AtomicBool;

    fn sample_config(model: Option<&str>) -> Config {
        let toml = match model {
            Some(m) => format!("[agent.alpha]\nmodel = \"{m}\"\ninstances = 1"),
            None => "[agent.alpha]\ninstances = 1".to_string(),
        };
        Config::from_toml_str(&toml).unwrap()
    }

    fn sample_section(model: Option<&str>) -> crate::core::config::AgentSection {
        sample_config(model).agent("alpha").unwrap().clone()
    }

    fn runtime(agent_id: &str) -> ModelRuntimeContext {
        ModelRuntimeContext {
            repo_path: "/tmp/repo".into(),
            shutdown: Arc::new(AtomicBool::new(false)),
            agent_id: agent_id.into(),
        }
    }

    #[test]
    fn from_agent_section_without_model_uri() {
        let config = sample_config(None);
        ModelEngine::from_agent_section(
            &config,
            &sample_section(None),
            runtime("alpha-0"),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn from_agent_section_with_acp_model_uri() {
        let config = sample_config(Some("acp://cursor/composer-2"));
        ModelEngine::from_agent_section(
            &config,
            &sample_section(Some("acp://cursor/composer-2")),
            runtime("alpha-1"),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn spawn_model_engine_wraps_runtime_context() {
        let config = sample_config(Some("acp://cursor/composer-2"));
        let section = sample_section(Some("acp://cursor/composer-2"));
        spawn_model_engine(
            &config,
            &section,
            "/tmp/repo",
            "alpha-2",
            Arc::new(AtomicBool::new(false)),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn from_agent_section_uses_configured_acp_client() {
        let config = Config::from_toml_str(
            r#"
            [acp.cursor-local]
            base_url = "http://prod-model1.example/v1"
            api_key = "EMPTY"
            acp_command = ["agent-local", "--print", "--trust", "--force", "--approve-mcps", "acp"]
            env = [
                "CURSOR_LOCAL_AGENT_BASE_URL={base_url}",
                "CURSOR_LOCAL_AGENT_API_KEY={api_key}",
            ]

            [agent.alpha]
            model = "model1-fp8"
            acp_client = "cursor-local"
            instances = 1
            "#,
        )
        .unwrap();
        let section = config.agent("alpha").unwrap().clone();
        let engine = ModelEngine::from_agent_section(
            &config,
            &section,
            runtime("alpha-local"),
            ModelSessionOptions::default(),
        )
        .unwrap();
        let _ = engine;
    }

    #[test]
    fn structured_output_tools_wire_json_converts_flat_object_schema() {
        let tool = StructuredOutputTool {
            name: "review".into(),
            description: "Emit your review decision".into(),
            parameters: ObjectSchema::new()
                .property(
                    "decision",
                    SchemaField::string_enum("Your decision.", &["approve", "request_changes"]),
                )
                .property("feedback", SchemaField::string("Feedback text."))
                .required("decision"),
        };

        let wire = structured_output_tools_wire_json(std::slice::from_ref(&tool));
        assert_eq!(wire.len(), 1);
        assert_eq!(
            wire[0],
            serde_json::json!({
                "name": "review",
                "description": "Emit your review decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "description": "Your decision.",
                            "enum": ["approve", "request_changes"]
                        },
                        "feedback": {
                            "type": "string",
                            "description": "Feedback text."
                        }
                    },
                    "required": ["decision"]
                }
            })
        );
    }

    #[test]
    fn structured_output_tools_wire_json_converts_nested_array_of_objects() {
        let tool = StructuredOutputTool {
            name: "plan".into(),
            description: "Emit a plan".into(),
            parameters: ObjectSchema::new().property(
                "sub_issues",
                SchemaField::array(
                    "The sub-issues to create.",
                    SchemaField::object(
                        ObjectSchema::new()
                            .property("title", SchemaField::string("Title."))
                            .property(
                                "priority",
                                SchemaField::integer_enum("Priority.", &[1, 2, 3]),
                            )
                            .required("title"),
                    ),
                ),
            ),
        };

        let wire = structured_output_tool_wire_json(&tool);
        assert_eq!(
            wire["parameters"]["properties"]["sub_issues"],
            serde_json::json!({
                "type": "array",
                "description": "The sub-issues to create.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title."},
                        "priority": {"type": "integer", "description": "Priority.", "enum": [1, 2, 3]}
                    },
                    "required": ["title"]
                }
            })
        );
    }

    #[test]
    fn structured_output_tools_wire_json_omits_required_when_empty() {
        let tool = StructuredOutputTool {
            name: "t".into(),
            description: "d".into(),
            parameters: ObjectSchema::new().property("x", SchemaField::boolean("x flag")),
        };
        let wire = structured_output_tool_wire_json(&tool);
        assert!(wire["parameters"].get("required").is_none());
    }
}
