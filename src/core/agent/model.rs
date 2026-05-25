use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use super::AgentHandoff;
use crate::core::model::acp::{PREFERRED_SESSION_MODE_PMO, PREFERRED_SESSION_MODE_REVIEWER};
use crate::core::model::engine::{ModelEngine, ModelSessionOptions, spawn_model_engine};
use crate::core::workflow::AgentSpawnContext;

use super::{InvokeOptions, ModelResponse};

/// Agent-specific model preferences (session mode, etc.). Does not expose engine construction.
#[derive(Debug, Clone, Default)]
pub struct ModelPreferences {
    pub preferred_session_mode: Option<&'static str>,
}

impl ModelPreferences {
    pub fn worker() -> Self {
        Self::default()
    }

    pub fn reviewer() -> Self {
        Self {
            preferred_session_mode: Some(PREFERRED_SESSION_MODE_REVIEWER),
        }
    }

    pub fn pmo() -> Self {
        Self {
            preferred_session_mode: Some(PREFERRED_SESSION_MODE_PMO),
        }
    }
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
            section,
            repo_path,
            agent_id.clone(),
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
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

    pub fn runtime_meta(&self) -> String {
        self.engine.runtime_meta()
    }

    pub fn invoke(&self, prompt: &str, options: &InvokeOptions) -> Result<ModelResponse> {
        self.engine.invoke(prompt, options)
    }

    /// Run a prompt and return the model handoff (primary API for agent cycle logic).
    pub fn complete(&self, prompt: &str, options: &InvokeOptions) -> Result<AgentHandoff> {
        self.invoke(prompt, options).map(|r| r.handoff)
    }

    pub fn complete_prompt(&self, prompt: &str) -> Result<AgentHandoff> {
        self.complete(prompt, &InvokeOptions::default())
    }

    pub fn complete_with_cancel(
        &self,
        prompt: &str,
        cancel_check: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Result<AgentHandoff> {
        self.complete(
            prompt,
            &InvokeOptions {
                cancel_check: Some(cancel_check),
                ..InvokeOptions::default()
            },
        )
    }

    pub fn complete_with_ask_handler(
        &self,
        prompt: &str,
        ask_handler: Option<Arc<dyn crate::core::model::acp::client::CursorAskQuestionHandler>>,
    ) -> Result<AgentHandoff> {
        self.complete(
            prompt,
            &InvokeOptions {
                cursor_ask_question_handler: ask_handler,
                ..InvokeOptions::default()
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn from_section_for_test(
        section: &crate::core::config::AgentSection,
        agent_id: &str,
        repo_path: &str,
        prefs: ModelPreferences,
    ) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let engine = spawn_model_engine(
            section,
            repo_path,
            agent_id,
            Arc::clone(&shutdown),
            ModelSessionOptions {
                preferred_session_mode: prefs.preferred_session_mode,
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
            Some(m) => format!("[agent.worker]\nmodel = \"{m}\"\ninstances = 1"),
            None => "[agent.worker]\ninstances = 1".to_string(),
        };
        Config::from_toml_str(&toml)
            .unwrap()
            .agent("worker")
            .unwrap()
            .clone()
    }

    #[test]
    fn model_preferences_set_session_modes() {
        assert_eq!(ModelPreferences::worker().preferred_session_mode, None);
        assert_eq!(
            ModelPreferences::reviewer().preferred_session_mode,
            Some(PREFERRED_SESSION_MODE_REVIEWER)
        );
        assert_eq!(
            ModelPreferences::pmo().preferred_session_mode,
            Some(PREFERRED_SESSION_MODE_PMO)
        );
    }

    #[test]
    fn from_section_for_test_connects_engine() {
        let model = AgentModel::from_section_for_test(
            &sample_section(Some("acp://cursor/composer-2")),
            "worker-test",
            "/tmp/repo",
            ModelPreferences::worker(),
        )
        .unwrap();
        assert_eq!(model.agent_id(), "worker-test");
        assert!(!model.shutdown().load(std::sync::atomic::Ordering::SeqCst));
    }
}
