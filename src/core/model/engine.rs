use std::sync::Arc;

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::core::agent::InvokeOptions;
use crate::core::agent::ModelResponse;
use crate::core::agent::schema::{ObjectSchema, OneOfSchema, Schema, StructuredOutputTool};
use crate::core::bus::AgentBus;
use crate::core::config::{AcpSpawnConfig, AgentSection, Config};
use crate::core::model::acp::AcpRuntime;
use crate::core::model::acp::capabilities::CapabilityProvider;

/// Model backend boundary: converts the neutral [`StructuredOutputTool`]
/// contract into generic JSON (`{"name", "description", "parameters"}` with
/// JSON-schema parameters). The selected ACP backend then either exposes that
/// JSON as Potlatch harness tools or renders it into a portable marker prompt.
///
/// Objects become closed (`additionalProperties: false`) and tagged unions
/// become `oneOf` branches whose discriminator property carries a `const`,
/// so the backend enforces the same shape core validates.
pub(crate) fn structured_output_contracts_json(tools: &[StructuredOutputTool]) -> Vec<Value> {
    tools.iter().map(structured_output_contract_json).collect()
}

fn structured_output_contract_json(tool: &StructuredOutputTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": schema_wire_json(&tool.parameters),
    })
}

fn schema_wire_json(schema: &Schema) -> Value {
    let mut obj = match schema {
        Schema::String { enum_values, .. } => {
            let mut obj = json!({"type": "string"});
            if !enum_values.is_empty() {
                obj["enum"] = Value::Array(enum_values.iter().cloned().map(Value::from).collect());
            }
            obj
        }
        Schema::Integer { enum_values, .. } => {
            let mut obj = json!({"type": "integer"});
            if !enum_values.is_empty() {
                obj["enum"] = Value::Array(enum_values.iter().cloned().map(Value::from).collect());
            }
            obj
        }
        Schema::Boolean { .. } => json!({"type": "boolean"}),
        Schema::Array { items, .. } => {
            json!({"type": "array", "items": schema_wire_json(items)})
        }
        Schema::Object(object) => object_wire_json(object, None),
        Schema::OneOf(one_of) => one_of_wire_json(one_of),
    };
    let description = schema.description();
    if !description.is_empty() {
        obj["description"] = Value::from(description);
    }
    obj
}

fn one_of_wire_json(one_of: &OneOfSchema) -> Value {
    let branches: Vec<Value> = one_of
        .variants
        .iter()
        .map(|variant| {
            object_wire_json(
                &variant.fields,
                Some(DiscriminatorConst {
                    name: &one_of.discriminator,
                    value: &variant.tag,
                    description: &variant.description,
                }),
            )
        })
        .collect();
    json!({"type": "object", "oneOf": branches})
}

/// The discriminator property a `oneOf` branch pins with `const`.
struct DiscriminatorConst<'a> {
    name: &'a str,
    value: &'a str,
    description: &'a str,
}

fn object_wire_json(schema: &ObjectSchema, discriminator: Option<DiscriminatorConst<'_>>) -> Value {
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();
    if let Some(ref tag) = discriminator {
        properties.insert(
            tag.name.to_string(),
            json!({
                "type": "string",
                "const": tag.value,
                "enum": [tag.value],
                "description": tag.description,
            }),
        );
        required.push(Value::from(tag.name));
    }
    for (name, field) in &schema.properties {
        properties.insert(name.clone(), schema_wire_json(field));
    }
    required.extend(schema.required.iter().cloned().map(Value::from));

    let mut obj = json!({
        "type": "object",
        "properties": Value::Object(properties),
        "additionalProperties": false,
    });
    let description = if schema.description.is_empty() {
        discriminator.map(|tag| tag.description)
    } else {
        Some(schema.description.as_str())
    };
    if let Some(description) = description.filter(|text| !text.is_empty()) {
        obj["description"] = Value::from(description);
    }
    if !required.is_empty() {
        obj["required"] = Value::Array(required);
    }
    obj
}

/// The caller's liveness callbacks as the runtime takes them: whether the task
/// should be cancelled, and any new messages to inject mid-turn.
type LiveCallbacks<'a> = (
    Option<&'a dyn Fn() -> bool>,
    Option<&'a dyn Fn() -> Vec<String>>,
);

/// Borrow the caller's cancellation check and follow-up poll as plain
/// function references for the runtime. Shared by both entry points so a
/// repair turn always runs under the same liveness callbacks as the task turn.
fn live_callbacks(options: &InvokeOptions) -> LiveCallbacks<'_> {
    (
        options
            .cancel_check
            .as_ref()
            .map(|check| check.as_ref() as &dyn Fn() -> bool),
        options
            .follow_up_poll
            .as_ref()
            .map(|poll| poll.as_ref() as &dyn Fn() -> Vec<String>),
    )
}

#[derive(Debug, Clone, Default)]
pub struct ModelSessionOptions {
    pub preferred_session_mode: Option<&'static str>,
    pub(crate) agent_bus: Option<AgentBus>,
}

#[derive(Debug, Clone)]
pub struct ModelRuntimeContext {
    pub working_dir: String,
    pub shutdown: Arc<std::sync::atomic::AtomicBool>,
    pub agent_id: String,
}

pub(crate) struct ModelEngine {
    inner: AcpRuntime,
}

/// Build a [`ModelEngine`] from config, an agent section, and runtime context.
pub(crate) fn spawn_model_engine(
    config: &Config,
    section: &AgentSection,
    working_dir: impl Into<String>,
    agent_id: impl Into<String>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    session: ModelSessionOptions,
) -> Result<ModelEngine> {
    ModelEngine::from_agent_section(
        config,
        section,
        ModelRuntimeContext {
            working_dir: working_dir.into(),
            shutdown,
            agent_id: agent_id.into(),
        },
        session,
    )
}

impl ModelEngine {
    pub fn from_agent_section(
        config: &Config,
        section: &AgentSection,
        runtime: ModelRuntimeContext,
        session: ModelSessionOptions,
    ) -> Result<Self> {
        let acp_spawn = config.resolve_acp_spawn(section)?;
        Ok(Self::build_from_acp_spawn(acp_spawn, session, runtime))
    }

    fn build_from_acp_spawn(
        acp_spawn: AcpSpawnConfig,
        session: ModelSessionOptions,
        runtime: ModelRuntimeContext,
    ) -> Self {
        Self::wrap_runtime(AcpRuntime::new(
            runtime.working_dir,
            acp_spawn.model_uri,
            acp_spawn.endpoint_model,
            acp_spawn.command,
            acp_spawn.env,
            session.preferred_session_mode,
            session.agent_bus,
            runtime.shutdown,
            runtime.agent_id,
        ))
    }

    fn wrap_runtime(inner: AcpRuntime) -> Self {
        Self { inner }
    }

    /// Start a new task: the structured-output contract for *this* task is
    /// converted to generic JSON and handed to the selected ACP backend.
    pub fn invoke(
        &self,
        prompt: &str,
        options: &InvokeOptions,
        tools: &[StructuredOutputTool],
    ) -> Result<ModelResponse> {
        let contracts = structured_output_contracts_json(tools);
        let (cancel_check, follow_up_poll) = live_callbacks(options);
        let handoff = self
            .inner
            .run_task(prompt, contracts, cancel_check, follow_up_poll)?;
        Ok(ModelResponse { handoff })
    }

    /// Continue the *current* task in the same session (structured-output
    /// repair). No session rotation, no re-registration, no task restatement.
    /// The caller's cancellation check and follow-up polling come along
    /// unchanged, so a repair turn is as interruptible and as reachable by new
    /// comments as the task turn it is fixing.
    pub fn invoke_in_session(
        &self,
        prompt: &str,
        options: &InvokeOptions,
    ) -> Result<ModelResponse> {
        let (cancel_check, follow_up_poll) = live_callbacks(options);
        let handoff = self
            .inner
            .run_in_current_session(prompt, cancel_check, follow_up_poll)?;
        Ok(ModelResponse { handoff })
    }

    /// Set the capability provider (called by the agent at construction).
    pub fn set_capability_provider(&self, provider: Option<Arc<dyn CapabilityProvider>>) {
        self.inner.set_capability_provider(provider);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::OneOfSchema;
    use crate::core::config::{AgentSection, Config};
    use std::sync::atomic::AtomicBool;

    fn sample_config(model: Option<&str>) -> Config {
        let toml = match model {
            Some(m) => format!(
                r#"[agent.alpha]
model = "{m}"
instances = 1

[acp.cursor]
acp_command = ["agent", "acp"]"#
            ),
            None => r#"[agent.alpha]
instances = 1

[acp.cursor]
acp_command = ["agent", "acp"]"#
                .to_string(),
        };
        Config::from_toml_str(&toml).unwrap()
    }

    fn sample_section(model: Option<&str>) -> AgentSection {
        sample_config(model).agent("alpha").unwrap().clone()
    }

    fn runtime(agent_id: &str) -> ModelRuntimeContext {
        ModelRuntimeContext {
            working_dir: "/tmp/repo".into(),
            shutdown: Arc::new(AtomicBool::new(false)),
            agent_id: agent_id.into(),
        }
    }

    #[test]
    fn from_agent_section_without_model_uri_errors() {
        let config = sample_config(None);
        let result = ModelEngine::from_agent_section(
            &config,
            &sample_section(None),
            runtime("alpha-0"),
            ModelSessionOptions::default(),
        );
        assert!(result.is_err());
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
            model = "acp://cursor-local/model1-fp8"
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
    fn live_callbacks_carry_the_callers_cancellation_and_follow_ups() {
        let bare = InvokeOptions::default();
        let (cancel, poll) = live_callbacks(&bare);
        assert!(cancel.is_none());
        assert!(poll.is_none());

        let options = InvokeOptions {
            cancel_check: Some(Arc::new(|| true)),
            follow_up_poll: Some(Arc::new(|| vec!["new comment".to_string()])),
            activity_label: None,
        };
        let (cancel, poll) = live_callbacks(&options);
        assert!(cancel.expect("cancel check")());
        assert_eq!(poll.expect("follow-up poll")(), vec!["new comment"]);
    }

    #[test]
    fn wire_json_closes_objects_and_keeps_required() {
        let tool = StructuredOutputTool {
            name: "qa_report".into(),
            description: "Emit findings".into(),
            parameters: Schema::object(
                ObjectSchema::new()
                    .describe("QA report.")
                    .required_property(
                        "findings",
                        Schema::array("Findings.", Schema::string("A finding.")),
                    )
                    .property("note", Schema::string("A note.")),
            ),
        };

        let wire = structured_output_contracts_json(std::slice::from_ref(&tool));
        assert_eq!(wire.len(), 1);
        assert_eq!(
            wire[0],
            json!({
                "name": "qa_report",
                "description": "Emit findings",
                "parameters": {
                    "type": "object",
                    "description": "QA report.",
                    "additionalProperties": false,
                    "properties": {
                        "findings": {
                            "type": "array",
                            "description": "Findings.",
                            "items": {"type": "string", "description": "A finding."}
                        },
                        "note": {"type": "string", "description": "A note."}
                    },
                    "required": ["findings"]
                }
            })
        );
    }

    #[test]
    fn wire_json_converts_nested_array_of_objects() {
        let tool = StructuredOutputTool {
            name: "plan".into(),
            description: "Emit a plan".into(),
            parameters: Schema::object(
                ObjectSchema::new().describe("Plan.").property(
                    "sub_issues",
                    Schema::array(
                        "The sub-issues to create.",
                        Schema::object(
                            ObjectSchema::new()
                                .describe("A sub-issue.")
                                .required_property("title", Schema::string("Title."))
                                .property(
                                    "priority",
                                    Schema::integer_enum("Priority.", &[1, 2, 3]),
                                ),
                        ),
                    ),
                ),
            ),
        };

        let wire = structured_output_contract_json(&tool);
        assert_eq!(
            wire["parameters"]["properties"]["sub_issues"],
            json!({
                "type": "array",
                "description": "The sub-issues to create.",
                "items": {
                    "type": "object",
                    "description": "A sub-issue.",
                    "additionalProperties": false,
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
    fn wire_json_emits_one_of_branches_with_a_const_discriminator() {
        let tool = StructuredOutputTool {
            name: "review".into(),
            description: "Emit your decision".into(),
            parameters: Schema::one_of(
                OneOfSchema::new("decision", "Your review decision.")
                    .variant(
                        "approve",
                        "The MR is good to merge.",
                        ObjectSchema::new().property("summary", Schema::string("One line.")),
                    )
                    .variant(
                        "request_changes",
                        "The MR needs work.",
                        ObjectSchema::new()
                            .required_property("feedback", Schema::string("What to fix.")),
                    ),
            ),
        };

        let wire = structured_output_contract_json(&tool);
        assert_eq!(
            wire["parameters"],
            json!({
                "type": "object",
                "description": "Your review decision.",
                "oneOf": [
                    {
                        "type": "object",
                        "description": "The MR is good to merge.",
                        "additionalProperties": false,
                        "properties": {
                            "decision": {
                                "type": "string",
                                "const": "approve",
                                "enum": ["approve"],
                                "description": "The MR is good to merge."
                            },
                            "summary": {"type": "string", "description": "One line."}
                        },
                        "required": ["decision"]
                    },
                    {
                        "type": "object",
                        "description": "The MR needs work.",
                        "additionalProperties": false,
                        "properties": {
                            "decision": {
                                "type": "string",
                                "const": "request_changes",
                                "enum": ["request_changes"],
                                "description": "The MR needs work."
                            },
                            "feedback": {"type": "string", "description": "What to fix."}
                        },
                        "required": ["decision", "feedback"]
                    }
                ]
            })
        );
    }

    #[test]
    fn wire_json_omits_required_when_a_shape_has_none() {
        let tool = StructuredOutputTool {
            name: "t".into(),
            description: "d".into(),
            parameters: Schema::object(
                ObjectSchema::new()
                    .describe("Shape.")
                    .property("x", Schema::boolean("x flag")),
            ),
        };
        let wire = structured_output_contract_json(&tool);
        assert!(wire["parameters"].get("required").is_none());
        assert_eq!(wire["parameters"]["additionalProperties"], json!(false));
    }
}
