use std::sync::Arc;

use anyhow::Result;

use crate::core::agent::InvokeOptions;
use crate::core::config::{AcpSpawnConfig, Config};
use crate::core::model::acp::AcpRuntime;

#[derive(Debug, Clone, Default)]
pub struct ModelSessionOptions {
    pub preferred_session_mode: Option<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub struct AcpBuildOptions {
    pub preferred_session_mode: Option<&'static str>,
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
            acp_spawn.spawn_model,
            acp_spawn.command,
            acp_spawn.env,
            opts.preferred_session_mode,
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
        let handoff = self.inner.run_with_cancel(
            prompt,
            cancel_check,
            options.cursor_ask_question_handler.clone(),
        )?;
        Ok(crate::core::agent::ModelResponse { handoff })
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
}
