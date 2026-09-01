use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::{error, fmt};

use anyhow::{Context, Result};
use tracing::{debug, warn};

use super::AgentHandoff;
use super::schema::{OutputError, StructuredOutput};
#[cfg(test)]
use crate::core::activity::NoopActivityReporter;
use crate::core::activity::SharedActivityReporter;
#[cfg(test)]
use crate::core::config::{AgentSection, Config};
use crate::core::model::acp::capabilities::CapabilityProvider;
use crate::core::model::engine::{ModelEngine, ModelSessionOptions, spawn_model_engine};
use crate::core::workflow::AgentSpawnContext;

use super::InvokeOptions;

/// How many times a role's structured output may be repaired inside the task's
/// own session before the task fails. Each repair is one extra prompt asking
/// the model to fix its tool call — the task itself is never restated and the
/// session is never rotated.
const MAX_STRUCTURED_OUTPUT_REPAIRS: u32 = 10;

/// Model preferences selected by callers without exposing engine construction.
/// Structured-output contracts are *not* here: they belong to a single
/// completion, not to the connection, and are passed to
/// [`AgentModel::complete_typed`].
#[derive(Debug, Clone, Default)]
pub struct ModelPreferences {
    pub preferred_session_mode: Option<&'static str>,
    /// Directories the harness's write/edit tools may touch outside the
    /// session cwd (via `outside_cwd: true`). Forwarded to the harness in
    /// `session/new` as the `write_roots` extension; empty keeps the
    /// harness's historical permissive behavior.
    pub write_roots: Vec<String>,
}

/// Result of a typed structured-output completion: the backend-neutral
/// response text plus the deserialized role output.
#[derive(Debug, Clone)]
pub struct TypedCompletion<T> {
    pub response: String,
    pub output: T,
}

#[derive(Debug)]
struct StructuredOutputRetriesExhausted {
    message: String,
}

impl fmt::Display for StructuredOutputRetriesExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl error::Error for StructuredOutputRetriesExhausted {}

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
        working_dir: impl Into<String>,
        prefs: ModelPreferences,
    ) -> Result<Self> {
        let section = ctx
            .workflow
            .config
            .agent(ctx.agent_name)
            .with_context(|| format!("[agent.{}] section required", ctx.agent_name))?;
        let agent_id = ctx.runtime.agent_id().to_string();
        let shutdown = Arc::clone(&ctx.workflow.shutdown);
        let activity = Arc::clone(&ctx.workflow.activity);
        let engine = spawn_model_engine(
            &ctx.workflow.config,
            section,
            working_dir,
            agent_id.clone(),
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                agent_bus: ctx.workflow.bus.clone(),
                write_roots: prefs.write_roots.clone(),
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

    /// Run a prompt and deserialize the role's required structured-output
    /// tool call into `T`. This is the only typed completion API — role cycle
    /// logic never reads `structured_outputs` JSON by hand.
    ///
    /// The contract for `T` is registered with the ACP session created for
    /// *this* task, so a role can ask for a different shape on every call.
    /// When the model's answer does not satisfy that contract — it never
    /// called the tool, called a different one, broke the schema, or produced
    /// something the Rust type rejects — the model is asked to correct itself
    /// inside the same session, up to [`MAX_STRUCTURED_OUTPUT_REPAIRS`] times,
    /// and the task is never restated. Transport-level retries (the ACP child
    /// dying mid-prompt) are handled below this layer and are not counted as
    /// repairs.
    pub fn complete_typed<T: StructuredOutput>(
        &self,
        prompt: &str,
        options: &InvokeOptions,
    ) -> Result<TypedCompletion<T>> {
        let activity_label = options
            .activity_label
            .clone()
            .unwrap_or_else(|| self.agent_id.clone());
        let _activity = self.activity.start(activity_label);

        let tools = [T::tool_definition()];
        let initial = self.engine.invoke(prompt, options, &tools)?.handoff;
        let completion = self.resolve_typed_in_session::<T>(initial, options)?;

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

    /// Continue the current task session and decode another structured result
    /// using the contract already registered by [`Self::complete_typed`].
    /// Worker uses this to nudge an implementation that returned no usable
    /// result or made no code changes without discarding its session context.
    pub fn continue_typed<T: StructuredOutput>(
        &self,
        prompt: &str,
        options: &InvokeOptions,
    ) -> Result<TypedCompletion<T>> {
        let activity_label = options
            .activity_label
            .clone()
            .unwrap_or_else(|| self.agent_id.clone());
        let _activity = self.activity.start(activity_label);
        let initial = self.engine.invoke_in_session(prompt, options)?.handoff;
        self.resolve_typed_in_session::<T>(initial, options)
    }

    /// Whether a typed completion exhausted its in-session correction turns.
    /// Transport, cancellation, and task errors intentionally return false.
    pub fn structured_output_retries_exhausted(error: &anyhow::Error) -> bool {
        error.is::<StructuredOutputRetriesExhausted>()
    }

    fn resolve_typed_in_session<T: StructuredOutput>(
        &self,
        initial: AgentHandoff,
        options: &InvokeOptions,
    ) -> Result<TypedCompletion<T>> {
        resolve_structured_output::<T>(&self.agent_id, initial, |repair_prompt| {
            self.engine
                .invoke_in_session(repair_prompt, options)
                .map(|response| response.handoff)
        })
    }

    /// Set the capability provider (called by the agent at construction).
    /// The vendor extension reads from it to wire handlers to its protocol.
    pub fn set_capability_provider(&self, provider: Option<Arc<dyn CapabilityProvider>>) {
        self.engine.set_capability_provider(provider);
    }

    #[cfg(test)]
    pub(crate) fn from_section_for_test(
        config: &Config,
        section: &AgentSection,
        agent_id: &str,
        working_dir: &str,
        prefs: ModelPreferences,
    ) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let activity: SharedActivityReporter = Arc::new(NoopActivityReporter);
        let engine = spawn_model_engine(
            config,
            section,
            working_dir,
            agent_id,
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                agent_bus: None,
                write_roots: prefs.write_roots.clone(),
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

/// Why one model turn did not produce the role's structured output.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CaptureError {
    /// The expected tool was not called. `called` lists the structured-output
    /// tools that *were* called, if any.
    MissingToolCall { called: Vec<String> },
    /// The tool was called with arguments the contract rejects.
    Invalid(OutputError),
}

impl CaptureError {
    /// The concise, model-facing statement of what is wrong. This is what a
    /// repair prompt carries, so it names the exact path and expectation
    /// rather than dumping the whole captured value back at the model.
    fn correction(&self, tool: &str) -> String {
        match self {
            CaptureError::MissingToolCall { called } if called.is_empty() => {
                format!("you ended your turn without calling the required `{tool}` tool")
            }
            CaptureError::MissingToolCall { called } => format!(
                "you called {} instead of the required `{tool}` tool",
                called
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            CaptureError::Invalid(error) => {
                format!("your `{tool}` arguments were rejected — {error}")
            }
        }
    }
}

/// Pull the role's tool call out of one handoff and decode it.
fn capture_structured_output<T: StructuredOutput>(
    handoff: &AgentHandoff,
) -> std::result::Result<T, CaptureError> {
    let tool_name = T::tool_name();
    let outputs = handoff
        .structured_outputs
        .as_ref()
        .and_then(|v| v.as_object());
    let Some(value) = outputs.and_then(|outputs| outputs.get(tool_name)) else {
        return Err(CaptureError::MissingToolCall {
            called: outputs
                .map(|outputs| outputs.keys().cloned().collect())
                .unwrap_or_default(),
        });
    };
    T::decode(value.clone()).map_err(CaptureError::Invalid)
}

/// The concise correction prompt sent back into the task's own session.
fn structured_output_repair_prompt(
    tool: &str,
    error: &CaptureError,
    attempt: u32,
    max_attempts: u32,
) -> String {
    let reinforcement = if attempt == max_attempts {
        "FINAL CORRECTION: every requirement below is mandatory. Another invalid response will fail the task."
    } else if attempt == 1 {
        "REQUIRED CORRECTION: you ended the turn without producing the required structured output."
    } else {
        "REINFORCEMENT: your previous correction was still invalid. Follow every requirement below exactly."
    };
    format!(
        "{reinforcement}\n\n\
         Exact failure: {correction}.\n\n\
         This session still contains all prior context — continue the task from where you left off, \
         and call the `{tool}` tool with the correct result when the task is complete. \
         Send arrays and objects as native structured arguments, never as quoted JSON strings. \
         (Correction attempt {attempt} of {max_attempts}.)",
        correction = error.correction(tool),
    )
}

/// Decode the role's output from `initial`, asking the model to correct itself
/// through `send_repair` when the contract is not met. `send_repair` continues
/// the same session, so every attempt keeps the task's context — and, at the
/// call site, its cancellation check and follow-up polling.
///
/// Split out from [`AgentModel::complete_typed`] so the repair sequence can be
/// driven without a live model.
fn resolve_structured_output<T: StructuredOutput>(
    agent_id: &str,
    initial: AgentHandoff,
    mut send_repair: impl FnMut(&str) -> Result<AgentHandoff>,
) -> Result<TypedCompletion<T>> {
    let tool_name = T::tool_name();
    let mut handoff = initial;
    let mut attempt: u32 = 0;
    loop {
        let error = match capture_structured_output::<T>(&handoff) {
            Ok(output) => {
                return Ok(TypedCompletion {
                    response: handoff.response,
                    output,
                });
            }
            Err(error) => error,
        };

        if attempt >= MAX_STRUCTURED_OUTPUT_REPAIRS {
            let preview: String = handoff.response.chars().take(800).collect();
            return Err(anyhow::Error::new(StructuredOutputRetriesExhausted {
                message: format!(
                    "{agent_id}: `{tool_name}` structured output still invalid after \
                     {MAX_STRUCTURED_OUTPUT_REPAIRS} correction attempt(s) — {correction}; \
                     last response preview: {preview:?}",
                    correction = error.correction(tool_name),
                ),
            }));
        }

        attempt += 1;
        warn!(
            "{agent_id}: {correction}. Repair {attempt}/{MAX_STRUCTURED_OUTPUT_REPAIRS} in the same session",
            correction = error.correction(tool_name),
        );
        handoff = send_repair(&structured_output_repair_prompt(
            tool_name,
            &error,
            attempt,
            MAX_STRUCTURED_OUTPUT_REPAIRS,
        ))?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::schema::{ObjectSchema, OneOfSchema, Schema, compat};
    use crate::core::config::{AgentSection, Config};
    use serde_json::{Value, json};
    use std::cell::RefCell;

    fn sample_section(model: Option<&str>) -> AgentSection {
        sample_config(model).agent("alpha").unwrap().clone()
    }

    fn sample_config(model: Option<&str>) -> Config {
        let toml = match model {
            Some(m) => format!(
                r#"[agent.alpha]
model = "{m}"
instances = 1

[acp.cursor]
acp_command = ["agent", "acp"]"#
            ),
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
    #[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
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

        fn schema() -> Schema {
            Schema::one_of(
                OneOfSchema::new("decision", "The decision.")
                    .variant("approve", "Approve it.", ObjectSchema::new())
                    .variant(
                        "request_changes",
                        "Ask for changes.",
                        ObjectSchema::new()
                            .required_property("feedback", Schema::string("What to fix.")),
                    ),
            )
        }

        fn normalize(value: &mut Value) {
            compat::normalize_tag(value, "decision");
        }
    }

    fn handoff_with_tool(tool: &str, value: Value) -> AgentHandoff {
        AgentHandoff {
            response: "narration".into(),
            structured_outputs: Some(json!({ tool: value })),
        }
    }

    /// Drives [`resolve_structured_output`] with a scripted sequence of model
    /// answers, recording every repair prompt that was sent.
    fn resolve_with_script(
        initial: AgentHandoff,
        script: Vec<AgentHandoff>,
    ) -> (Result<TypedCompletion<SampleOutput>>, Vec<String>) {
        let prompts = RefCell::new(Vec::new());
        let remaining = RefCell::new(script.into_iter());
        let result = resolve_structured_output::<SampleOutput>("agent-0", initial, |prompt| {
            prompts.borrow_mut().push(prompt.to_string());
            remaining
                .borrow_mut()
                .next()
                .context("test script ran out of scripted model answers")
        });
        (result, prompts.into_inner())
    }

    #[test]
    fn valid_first_answer_needs_no_repair() {
        let (result, prompts) = resolve_with_script(
            handoff_with_tool("sample_tool", json!({"decision": "approve"})),
            vec![],
        );
        let completion = result.unwrap();
        assert_eq!(completion.response, "narration");
        assert_eq!(completion.output, SampleOutput::Approve);
        assert!(prompts.is_empty());
    }

    #[test]
    fn missing_tool_call_is_repaired_in_the_same_session() {
        let (result, prompts) = resolve_with_script(
            AgentHandoff {
                response: "I'm done!".into(),
                structured_outputs: None,
            },
            vec![handoff_with_tool(
                "sample_tool",
                json!({"decision": "approve"}),
            )],
        );
        assert_eq!(result.unwrap().output, SampleOutput::Approve);
        assert_eq!(prompts.len(), 1);
    }

    #[test]
    fn calling_the_wrong_tool_names_it_in_the_correction() {
        let (result, prompts) = resolve_with_script(
            handoff_with_tool("plan", json!({"decision": "approve"})),
            vec![handoff_with_tool(
                "sample_tool",
                json!({"decision": "approve"}),
            )],
        );
        assert!(result.is_ok());
        assert_eq!(prompts.len(), 1);
    }

    #[test]
    fn schema_violations_are_repaired_with_the_offending_path() {
        let (result, prompts) = resolve_with_script(
            handoff_with_tool("sample_tool", json!({"decision": "maybe"})),
            vec![handoff_with_tool(
                "sample_tool",
                json!({"decision": "request_changes", "feedback": "fix the test"}),
            )],
        );
        assert_eq!(
            result.unwrap().output,
            SampleOutput::RequestChanges {
                feedback: "fix the test".into()
            }
        );
        assert_eq!(prompts.len(), 1);
    }

    #[test]
    fn missing_branch_field_is_repaired_with_the_offending_path() {
        let (result, prompts) = resolve_with_script(
            handoff_with_tool("sample_tool", json!({"decision": "request_changes"})),
            vec![handoff_with_tool(
                "sample_tool",
                json!({"decision": "request_changes", "feedback": "fix it"}),
            )],
        );
        assert!(result.is_ok());
        assert_eq!(prompts.len(), 1);
    }

    #[test]
    fn every_repair_attempt_is_actually_sent_before_giving_up() {
        let bad = || handoff_with_tool("sample_tool", json!({"decision": "maybe"}));
        let script = std::iter::repeat_with(bad)
            .take(MAX_STRUCTURED_OUTPUT_REPAIRS as usize)
            .collect();
        let (result, prompts) = resolve_with_script(bad(), script);
        let error = result.unwrap_err();
        assert!(AgentModel::structured_output_retries_exhausted(&error));
        assert_eq!(prompts.len(), MAX_STRUCTURED_OUTPUT_REPAIRS as usize);
    }

    #[test]
    fn exhaustion_reports_the_last_failure_not_the_first() {
        let mut script: Vec<_> = (0..MAX_STRUCTURED_OUTPUT_REPAIRS - 1)
            .map(|_| handoff_with_tool("sample_tool", json!({"decision": "maybe"})))
            .collect();
        script.push(handoff_with_tool(
            "sample_tool",
            json!({"decision": "approve", "extra": 1}),
        ));
        let (result, _prompts) = resolve_with_script(
            AgentHandoff {
                response: "nothing".into(),
                structured_outputs: None,
            },
            script,
        );
        assert!(result.is_err());
    }

    #[test]
    fn a_transport_failure_during_repair_is_returned_unchanged() {
        let prompts = RefCell::new(0u32);
        let result = resolve_structured_output::<SampleOutput>(
            "agent-0",
            AgentHandoff {
                response: String::new(),
                structured_outputs: None,
            },
            |_| {
                *prompts.borrow_mut() += 1;
                Err(anyhow::anyhow!("ACP session/prompt: broken pipe"))
            },
        );
        assert_eq!(prompts.into_inner(), 1);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("ACP session/prompt: broken pipe")
        );
    }

    #[test]
    fn compatibility_normalization_runs_before_a_repair_is_considered() {
        let (result, prompts) = resolve_with_script(
            handoff_with_tool("sample_tool", json!({"decision": "Approve"})),
            vec![],
        );
        assert_eq!(result.unwrap().output, SampleOutput::Approve);
        assert!(prompts.is_empty());
    }
}
