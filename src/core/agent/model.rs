use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use super::AgentHandoff;
use crate::core::model::engine::{ModelEngine, ModelSessionOptions, spawn_model_engine};
use crate::core::workflow::AgentSpawnContext;

use super::{InvokeOptions, ModelResponse};

/// Model preferences selected by callers without exposing engine construction.
#[derive(Debug, Clone, Default)]
pub struct ModelPreferences {
    pub preferred_session_mode: Option<&'static str>,
    /// Caller-defined structured-output tool definitions. Each entry is a JSON
    /// object with `name`, `description`, and `parameters` (JSON schema).
    /// Passed to the harness via `session/new` params. The harness creates a
    /// generic `StructuredOutputTool` per definition and returns captured
    /// output in the `session/prompt` response.
    pub structured_output_tools: Option<Vec<serde_json::Value>>,
}

/// Agent-facing model API. Engine construction is internal to core.
pub struct AgentModel {
    agent_id: String,
    shutdown: Arc<AtomicBool>,
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
        let engine = spawn_model_engine(
            &ctx.workflow.config,
            section,
            repo_path,
            agent_id.clone(),
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                structured_output_tools: prefs.structured_output_tools,
            },
        )?;
        Ok(Self {
            agent_id,
            shutdown,
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
        let _activity = crate::ui::activity(activity_label);
        self.engine.invoke(prompt, options)
    }

    /// Run a prompt and return the model handoff (primary API for agent cycle logic).
    pub fn complete(&self, prompt: &str, options: &InvokeOptions) -> Result<AgentHandoff> {
        self.invoke(prompt, options).map(|r| r.handoff)
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
        let engine = spawn_model_engine(
            config,
            section,
            repo_path,
            agent_id,
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
                structured_output_tools: prefs.structured_output_tools,
            },
        )?;
        Ok(Self {
            agent_id: agent_id.to_string(),
            shutdown,
            engine,
        })
    }
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
}
