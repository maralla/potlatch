use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use tracing::debug;

use super::AgentHandoff;
use super::schema::{StructuredOutput, StructuredOutputTool};
use crate::core::activity::SharedActivityReporter;
use crate::core::model::engine::{
    ModelEngine, ModelSessionOptions, spawn_model_engine, structured_output_tools_wire_json,
};
use crate::core::workflow::AgentSpawnContext;

use super::{InvokeOptions, ModelResponse};

/// Model preferences selected by callers without exposing engine construction.
#[derive(Debug, Clone, Default)]
pub struct ModelPreferences {
    pub preferred_session_mode: Option<&'static str>,
    /// Caller-defined structured-output tool contracts, backend-agnostic.
    /// Converted to the model backend's wire format (e.g. ACP/harness
    /// `structured_output_tools` JSON) at connect time — see
    /// [`crate::core::model::engine::structured_output_tools_wire_json`].
    pub structured_output_tools: Option<Vec<StructuredOutputTool>>,
}

/// Result of a typed structured-output completion: the backend-neutral
/// response text plus the deserialized role output.
#[derive(Debug, Clone)]
pub struct TypedCompletion<T> {
    pub response: String,
    pub output: T,
}

/// Agent-facing model API. Engine construction is internal to core.
pub struct AgentModel {
    agent_id: String,
    shutdown: Arc<AtomicBool>,
    activity: SharedActivityReporter,
    engine: ModelEngine,
}

impl AgentModel {
    /// Connect a model for a configured agent role at spawn time.
    pub fn connect(
        ctx: &AgentSpawnContext,
        role: &str,
        repo_path: impl Into<String>,
        prefs: ModelPreferences,
    ) -> Result<Self> {
        let section = ctx
            .workflow
            .config
            .agent(role)
            .with_context(|| format!("[agent.{role}] section required"))?;
        let agent_id = format!("{role}-{}", ctx.instance_id);
        let shutdown = Arc::clone(&ctx.workflow.shutdown);
        let activity = Arc::clone(&ctx.workflow.activity);
        let engine = spawn_model_engine(
            &ctx.workflow.config,
            section,
            repo_path,
            agent_id.clone(),
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                structured_output_tools: prefs
                    .structured_output_tools
                    .as_deref()
                    .map(structured_output_tools_wire_json),
            },
        )?;
        Ok(Self {
            agent_id,
            shutdown,
            activity,
            engine,
        })
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn shutdown(&self) -> &Arc<AtomicBool> {
        &self.shutdown
    }

    pub fn invoke(&self, prompt: &str, options: &InvokeOptions) -> Result<ModelResponse> {
        let activity_label = options
            .activity_label
            .clone()
            .unwrap_or_else(|| self.agent_id.clone());
        let _activity = self.activity.start(activity_label);
        self.engine.invoke(prompt, options)
    }

    /// Run a prompt and return the model handoff (primary API for agent cycle logic).
    pub fn complete(&self, prompt: &str, options: &InvokeOptions) -> Result<AgentHandoff> {
        self.invoke(prompt, options).map(|r| r.handoff)
    }

    /// Run a prompt and deserialize the role's required structured-output
    /// tool call into `T`. This is the primary typed completion API — role
    /// cycle logic should prefer this over reading `structured_outputs` JSON
    /// by hand.
    ///
    /// Returns a clear error when: the model never called the `T::tool_name()`
    /// tool (missing tool output), the captured JSON has the wrong shape or
    /// types, an enum field has an unrecognized value, or a required field is
    /// missing. Retry behavior for a model that fails to call the tool lives
    /// at the ACP runtime boundary (nudge-and-retry within the same session)
    /// and is unaffected by this method.
    pub fn complete_typed<T: StructuredOutput>(
        &self,
        prompt: &str,
        options: &InvokeOptions,
    ) -> Result<TypedCompletion<T>> {
        let handoff = self.complete(prompt, options)?;
        let completion = decode_structured_completion::<T>(handoff, &self.agent_id)?;
        let narration = completion.response.trim();
        if !narration.is_empty() {
            debug!(
                "{}: model narration alongside `{}` tool call: {narration}",
                self.agent_id,
                T::tool_name()
            );
        }
        Ok(completion)
    }

    /// Set the capability provider (called by the agent at construction).
    /// The vendor extension reads from it to wire handlers to its protocol.
    pub fn set_capability_provider(
        &self,
        provider: Option<Arc<dyn crate::core::model::acp::capabilities::CapabilityProvider>>,
    ) {
        self.engine.set_capability_provider(provider);
    }

    #[cfg(test)]
    pub(crate) fn from_section_for_test(
        config: &crate::core::config::Config,
        section: &crate::core::config::AgentSection,
        agent_id: &str,
        repo_path: &str,
        prefs: ModelPreferences,
    ) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let activity: SharedActivityReporter =
            Arc::new(crate::core::activity::NoopActivityReporter);
        let engine = spawn_model_engine(
            config,
            section,
            repo_path,
            agent_id,
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                structured_output_tools: prefs
                    .structured_output_tools
                    .as_deref()
                    .map(structured_output_tools_wire_json),
            },
        )?;
        Ok(Self {
            agent_id: agent_id.to_string(),
            shutdown,
            activity,
            engine,
        })
    }
}

/// Decode one role's structured-output tool call out of a backend-neutral
/// [`AgentHandoff`]. Shared by [`AgentModel::complete_typed`] and unit tests
/// so the error messages stay consistent without going through a live model.
fn decode_structured_completion<T: StructuredOutput>(
    handoff: AgentHandoff,
    agent_id: &str,
) -> Result<TypedCompletion<T>> {
    let tool_name = T::tool_name();
    let response_preview: String = handoff.response.chars().take(800).collect();
    let value = handoff
        .structured_outputs
        .as_ref()
        .and_then(|outputs| outputs.get(tool_name))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{agent_id}: model did not call the required `{tool_name}` structured-output tool; \
                 response preview: {response_preview:?}"
            )
        })?;
    let output: T = serde_json::from_value(value.clone()).with_context(|| {
        format!("{agent_id}: `{tool_name}` structured-output tool call had invalid output: {value}")
    })?;
    Ok(TypedCompletion {
        response: handoff.response,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;

    fn sample_section(model: Option<&str>) -> crate::core::config::AgentSection {
        let toml = match model {
            Some(m) => format!("[agent.alpha]\nmodel = \"{m}\"\ninstances = 1"),
            None => "[agent.alpha]\ninstances = 1".to_string(),
        };
        Config::from_toml_str(&toml)
            .unwrap()
            .agent("alpha")
            .unwrap()
            .clone()
    }

    fn sample_config(model: Option<&str>) -> Config {
        let toml = match model {
            Some(m) => format!("[agent.alpha]\nmodel = \"{m}\"\ninstances = 1"),
            None => "[agent.alpha]\ninstances = 1".to_string(),
        };
        Config::from_toml_str(&toml).unwrap()
    }

    #[test]
    fn model_preferences_default_has_no_session_mode() {
        assert_eq!(ModelPreferences::default().preferred_session_mode, None);
    }

    #[test]
    fn from_section_for_test_connects_engine() {
        let config = sample_config(Some("acp://cursor/composer-2"));
        let section = sample_section(Some("acp://cursor/composer-2"));
        let model = AgentModel::from_section_for_test(
            &config,
            &section,
            "alpha-test",
            "/tmp/repo",
            ModelPreferences::default(),
        )
        .unwrap();
        assert_eq!(model.agent_id(), "alpha-test");
        assert!(!model.shutdown().load(std::sync::atomic::Ordering::SeqCst));
    }

    #[derive(Debug, serde::Deserialize, PartialEq)]
    #[serde(tag = "decision", rename_all = "snake_case")]
    enum SampleOutput {
        Approve,
        RequestChanges { feedback: String },
    }

    impl StructuredOutput for SampleOutput {
        fn tool_name() -> &'static str {
            "sample_tool"
        }

        fn tool_description() -> &'static str {
            "sample"
        }

        fn schema() -> crate::core::agent::schema::ObjectSchema {
            crate::core::agent::schema::ObjectSchema::new()
        }
    }

    fn handoff_with_tool(tool: &str, value: serde_json::Value) -> AgentHandoff {
        AgentHandoff {
            response: "narration".into(),
            structured_outputs: Some(serde_json::json!({ tool: value })),
        }
    }

    #[test]
    fn decode_structured_completion_returns_response_and_typed_output() {
        let handoff = handoff_with_tool("sample_tool", serde_json::json!({"decision": "approve"}));
        let completion = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap();
        assert_eq!(completion.response, "narration");
        assert_eq!(completion.output, SampleOutput::Approve);
    }

    #[test]
    fn decode_structured_completion_errors_when_tool_not_called() {
        let handoff = AgentHandoff {
            response: "no tool call".into(),
            structured_outputs: None,
        };
        let err = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap_err();
        assert!(err.to_string().contains("did not call"));
        assert!(err.to_string().contains("sample_tool"));
        assert!(err.to_string().contains("no tool call"));
    }

    #[test]
    fn decode_structured_completion_errors_when_other_tool_called() {
        let handoff = handoff_with_tool("other_tool", serde_json::json!({"decision": "approve"}));
        let err = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap_err();
        assert!(err.to_string().contains("did not call"));
    }

    #[test]
    fn decode_structured_completion_errors_on_unknown_enum_value() {
        let handoff = handoff_with_tool("sample_tool", serde_json::json!({"decision": "bogus"}));
        let err = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap_err();
        assert!(err.to_string().contains("invalid output"));
    }

    #[test]
    fn decode_structured_completion_errors_on_missing_required_field() {
        let handoff = handoff_with_tool(
            "sample_tool",
            serde_json::json!({"decision": "request_changes"}),
        );
        let err = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap_err();
        assert!(err.to_string().contains("invalid output"));
    }

    #[test]
    fn decode_structured_completion_errors_on_wrong_type() {
        let handoff = handoff_with_tool(
            "sample_tool",
            serde_json::json!({"decision": "request_changes", "feedback": 123}),
        );
        let err = decode_structured_completion::<SampleOutput>(handoff, "agent-0").unwrap_err();
        assert!(err.to_string().contains("invalid output"));
    }
}
