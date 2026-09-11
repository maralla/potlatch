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
/// Objects become closed (`additionalProperties: false`). Tagged unions
/// flatten into one object — the discriminator an enum-valued property, the
/// branches' fields its siblings — never a top-level `oneOf`, a wire shape
/// providers were observed failing to serialize (see [`one_of_wire_json`]).
/// Core still validates the captured value against the full typed contract,
/// so the wire form is a transport convenience, not the enforcement point.
pub(crate) fn structured_output_contracts_json(tools: &[StructuredOutputTool]) -> Vec<Value> {
    tools.iter().map(structured_output_contract_json).collect()
}

fn structured_output_contract_json(tool: &StructuredOutputTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": schema_wire_json(&tool.parameters),
        "terminal": tool.terminal,
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
        Schema::Object(object) => object_wire_json(object),
        Schema::OneOf(one_of) => one_of_wire_json(one_of),
    };
    let description = schema.description();
    if !description.is_empty() {
        obj["description"] = Value::from(description);
    }
    obj
}

/// The wire form of a tagged union: one flat `object` whose discriminator is
/// an ordinary enum-valued property and whose branches' fields are sibling
/// properties of that same object.
///
/// Not a top-level `oneOf`: providers observed dropping the arguments of
/// structured-output tool calls entirely (or dropping every parameter but the
/// longest one) do so specifically for tools whose parameters are
/// `{"type":"object","oneOf":[...]}` — a shape no other tool on the wire
/// has. Every tool that carries plain `object` parameters transmitted fine
/// in the same sessions. Flattening makes a structured-output tool's
/// parameters indistinguishable on the wire from every working tool.
///
/// Per-branch wire strictness is given up deliberately: a field from an
/// unselected branch is no longer rejected by the schema alone. The typed
/// contract still rejects it when the captured value is decoded
/// (see [`crate::core::agent::model::capture_structured_output`]), so
/// nothing invalid can be recorded — only reported later, after a repair
/// turn, instead of at emission.
fn one_of_wire_json(one_of: &OneOfSchema) -> Value {
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();

    // The discriminator: one enum-valued property listing every branch tag,
    // described by the union's own description.
    let tags: Vec<Value> = one_of
        .variants
        .iter()
        .map(|variant| Value::from(variant.tag.as_str()))
        .collect();
    properties.insert(
        one_of.discriminator.clone(),
        json!({
            "type": "string",
            "enum": tags,
            "description": one_of.description,
        }),
    );
    required.push(Value::from(one_of.discriminator.as_str()));

    // Each branch's fields become sibling properties. A field name shared by
    // branches (same name, same type in each) serializes once; required-ness
    // is dropped, since only the selected branch's required fields are
    // actually required and the wire form cannot express that per branch.
    for variant in &one_of.variants {
        for (name, field) in &variant.fields.properties {
            properties
                .entry(name.clone())
                .or_insert_with(|| schema_wire_json(field));
        }
    }

    json!({
        "type": "object",
        "properties": Value::Object(properties),
        "required": Value::Array(required),
        "additionalProperties": false,
    })
}

/// The wire form of a plain object schema: properties, required names, and
/// `additionalProperties: false` so providers treat it as closed.
fn object_wire_json(schema: &ObjectSchema) -> Value {
    let properties: Map<String, Value> = schema
        .properties
        .iter()
        .map(|(name, field)| (name.clone(), schema_wire_json(field)))
        .collect();
    let mut obj = json!({
        "type": "object",
        "properties": Value::Object(properties),
        "additionalProperties": false,
    });
    if !schema.description.is_empty() {
        obj["description"] = Value::from(schema.description.as_str());
    }
    if !schema.required.is_empty() {
        obj["required"] = Value::Array(schema.required.iter().cloned().map(Value::from).collect());
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
    /// Directories the harness's write/edit tools may touch outside the
    /// session cwd (via `outside_cwd: true`). Forwarded to the harness in
    /// `session/new` as the `write_roots` extension.
    pub write_roots: Vec<String>,
    /// Root for the harness's agent current-session markers (see
    /// [`ModelPreferences::agents_dir`]).
    pub agents_dir: Option<String>,
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
        let mut env = acp_spawn.env;
        if let Some(argv) = &acp_spawn.auth_command {
            // JSON-encoded argv so the harness can execute it without a shell.
            let serialized =
                serde_json::to_string(argv).expect("serializing a Vec<String> cannot fail");
            env.insert(crate::core::config::AUTH_COMMAND_ENV.into(), serialized);
            if let Some(config_dir) = &acp_spawn.config_dir {
                env.insert(
                    crate::core::config::AUTH_COMMAND_DIR_ENV.into(),
                    config_dir.display().to_string(),
                );
            }
        }
        Self::wrap_runtime(AcpRuntime::new(
            runtime.working_dir,
            acp_spawn.model_uri,
            acp_spawn.endpoint_model,
            acp_spawn.command,
            env,
            session.preferred_session_mode,
            session.agent_bus,
            session.write_roots,
            session.agents_dir,
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
            base_url = "http://endpoint1.example/v1"
            api_key = "EMPTY"
            acp_command = ["agent-local", "--print", "--trust", "--force", "--approve-mcps", "acp"]
            env = [
                "CURSOR_LOCAL_AGENT_BASE_URL={base_url}",
                "CURSOR_LOCAL_AGENT_API_KEY={api_key}",
            ]

            [agent.alpha]
            model = "acp://cursor-local/model1"
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
            terminal: false,
        };

        let wire = structured_output_contracts_json(std::slice::from_ref(&tool));
        assert_eq!(wire.len(), 1);
        assert_eq!(
            wire[0],
            json!({
                "name": "qa_report",
                "description": "Emit findings",
                "terminal": false,
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
            terminal: false,
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
    fn wire_json_flattens_one_of_into_a_single_object() {
        // The wire form must NOT be a top-level oneOf: providers observed
        // dropping the arguments of structured-output tool calls do so for
        // exactly this shape. The flattened form — discriminator as an enum
        // property, branch fields as siblings — is indistinguishable from
        // every plain-object tool that transmitted fine.
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
            terminal: true,
        };

        let wire = structured_output_contract_json(&tool);
        assert_eq!(wire["terminal"], json!(true));
        assert_eq!(
            wire["parameters"],
            json!({
                "type": "object",
                "description": "Your review decision.",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["approve", "request_changes"],
                        "description": "Your review decision."
                    },
                    "summary": {"type": "string", "description": "One line."},
                    "feedback": {"type": "string", "description": "What to fix."}
                },
                "required": ["decision"],
                "additionalProperties": false
            })
        );
        // The one shape providers fail to serialize must not appear.
        assert!(
            wire["parameters"].get("oneOf").is_none(),
            "wire form must not be a top-level oneOf: {}",
            wire["parameters"]
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
            terminal: false,
        };
        let wire = structured_output_contract_json(&tool);
        assert!(wire["parameters"].get("required").is_none());
        assert_eq!(wire["parameters"]["additionalProperties"], json!(false));
    }
}
