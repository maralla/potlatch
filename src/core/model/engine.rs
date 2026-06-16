use std::sync::Arc;

use anyhow::Result;

use crate::core::agent::InvokeOptions;
use crate::core::config::AgentSection;
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
pub enum BackendOptions {
    Acp(AcpBuildOptions),
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

/// Build a [`ModelEngine`] from an agent config section and runtime context.
pub(crate) fn spawn_model_engine(
    section: &AgentSection,
    repo_path: impl Into<String>,
    agent_id: impl Into<String>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    session: ModelSessionOptions,
) -> Result<ModelEngine> {
    ModelEngine::from_agent_section(
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
        section: &AgentSection,
        runtime: ModelRuntimeContext,
        session: ModelSessionOptions,
    ) -> Result<Self> {
        let acp_opts = AcpBuildOptions {
            preferred_session_mode: session.preferred_session_mode,
        };
        match section.core.model.as_ref() {
            Some(model_uri) => Self::build(model_uri, BackendOptions::Acp(acp_opts), runtime),
            None => Ok(Self::build_bare(runtime, acp_opts)),
        }
    }

    pub fn build(
        model_uri: &crate::core::config::ModelUri,
        backend_options: BackendOptions,
        runtime: ModelRuntimeContext,
    ) -> Result<Self> {
        if model_uri.scheme != "acp" {
            anyhow::bail!("unsupported model scheme: {}", model_uri.scheme);
        }
        let BackendOptions::Acp(opts) = backend_options;
        Ok(Self::wrap_runtime(AcpRuntime::new(
            runtime.repo_path,
            Some(model_uri.bare_model().to_string()),
            opts.preferred_session_mode,
            runtime.shutdown,
            runtime.agent_id,
        )))
    }

    fn build_bare(runtime: ModelRuntimeContext, opts: AcpBuildOptions) -> Self {
        Self::wrap_runtime(AcpRuntime::new(
            runtime.repo_path,
            None,
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

    fn sample_section(model: Option<&str>) -> AgentSection {
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

    fn runtime(agent_id: &str) -> ModelRuntimeContext {
        ModelRuntimeContext {
            repo_path: "/tmp/repo".into(),
            shutdown: Arc::new(AtomicBool::new(false)),
            agent_id: agent_id.into(),
        }
    }

    #[test]
    fn from_agent_section_without_model_uri() {
        ModelEngine::from_agent_section(
            &sample_section(None),
            runtime("alpha-0"),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn from_agent_section_with_acp_model_uri() {
        ModelEngine::from_agent_section(
            &sample_section(Some("acp://cursor/composer-2")),
            runtime("alpha-1"),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn spawn_model_engine_wraps_runtime_context() {
        let section = sample_section(Some("acp://cursor/composer-2"));
        spawn_model_engine(
            &section,
            "/tmp/repo",
            "alpha-2",
            Arc::new(AtomicBool::new(false)),
            ModelSessionOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn build_rejects_unsupported_scheme() {
        let Err(error) = ModelEngine::build(
            &crate::core::config::ModelUri::parse("http://example/m").unwrap(),
            BackendOptions::Acp(AcpBuildOptions::default()),
            runtime("alpha-3"),
        ) else {
            panic!("expected unsupported scheme error");
        };
        assert!(error.to_string().contains("unsupported model scheme"));
    }
}
