use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;

use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::registry::AgentRegistry;

pub struct WorkflowContext {
    pub config: Arc<Config>,
    pub base_dir: String,
    pub shutdown: Arc<AtomicBool>,
}

/// Spawn inputs passed from workflow into a [`CoreAgent`] implementation.
pub struct AgentSpawnContext {
    pub workflow: WorkflowContext,
    pub instance_id: usize,
}

impl WorkflowContext {
    pub fn clone_for_spawn(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            base_dir: self.base_dir.clone(),
            shutdown: Arc::clone(&self.shutdown),
        }
    }
}

fn prepare_shutdown_handlers(ctx: &WorkflowContext) -> Result<()> {
    flag::register(SIGINT, ctx.shutdown.clone()).context("Failed to register SIGINT handler")?;
    flag::register(SIGTERM, ctx.shutdown.clone()).context("Failed to register SIGTERM handler")?;
    flag::register_conditional_shutdown(SIGINT, 1, ctx.shutdown.clone())
        .context("Failed to register conditional shutdown")?;
    flag::register_conditional_shutdown(SIGTERM, 1, ctx.shutdown.clone())
        .context("Failed to register conditional shutdown")?;
    Ok(())
}

/// Start and run a registered [`CoreAgent`] until workflow shutdown.
pub(crate) fn spawn_core_agent<A>(workflow: WorkflowContext, instance_id: usize) -> Result<()>
where
    A: CoreAgent<SpawnContext = AgentSpawnContext>,
{
    A::run_from(AgentSpawnContext {
        workflow,
        instance_id,
    })
}

pub fn build_startup_banner(
    config: &Config,
    registry: &AgentRegistry,
    config_path: &str,
) -> Banner {
    let mut banner = Banner::default();
    let config_display = Path::new(config_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(config_path);
    banner.set("config", config_display);

    let mut agent_names: Vec<String> = config.agent_names().map(str::to_string).collect();
    agent_names.sort();

    for agent_name in &agent_names {
        if let Some(registration) = registry.find(agent_name) {
            (registration.banner)(config, &mut banner);
        }
    }

    let agents_line = if agent_names.is_empty() {
        "(none configured)".to_string()
    } else {
        agent_names.join(", ")
    };
    banner.set("agents", agents_line);
    banner
}

pub struct Workflow {
    config: Config,
    config_path: String,
    registry: AgentRegistry,
}

impl Workflow {
    pub fn new(config: Config, config_path: impl Into<String>) -> Self {
        Self {
            config,
            config_path: config_path.into(),
            registry: AgentRegistry::new(),
        }
    }

    pub fn register_agent<A>(&mut self)
    where
        A: CoreAgent<SpawnContext = AgentSpawnContext> + 'static,
    {
        self.registry.register_agent::<A>();
    }

    #[cfg(test)]
    pub fn registered_agent_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.registry.agent_names()
    }

    pub fn run(self) -> Result<()> {
        let base_dir = std::env::current_dir()
            .context("Failed to get current directory")?
            .to_string_lossy()
            .into_owned();

        let shutdown = Arc::new(AtomicBool::new(false));
        let ctx = WorkflowContext {
            config: Arc::new(self.config),
            base_dir,
            shutdown: Arc::clone(&shutdown),
        };

        prepare_shutdown_handlers(&ctx)?;

        let banner = build_startup_banner(&ctx.config, &self.registry, &self.config_path);
        crate::ui::print_banner(&banner);

        let mut agent_names: Vec<String> = ctx.config.agent_names().map(str::to_string).collect();
        agent_names.sort();

        let mut handles = Vec::new();

        for agent_name in agent_names {
            let section = ctx
                .config
                .agent(&agent_name)
                .with_context(|| format!("missing agent section [agent.{agent_name}]"))?;
            if section.core.instances == 0 {
                continue;
            }
            let registration = self.registry.find(&agent_name).with_context(|| {
                format!("no registration for configured agent [agent.{agent_name}]")
            })?;
            (registration.validate_config)(section)
                .with_context(|| format!("invalid config for [agent.{agent_name}]"))?;

            for instance_id in 0..section.core.instances {
                let spawn_ctx = ctx.clone_for_spawn();
                let spawn = registration.spawn;
                let agent_name_log = agent_name.clone();
                handles.push(std::thread::spawn(move || {
                    if let Err(e) = spawn(spawn_ctx, instance_id) {
                        tracing::error!(
                            "Agent {}-{} failed to start: {}",
                            agent_name_log,
                            instance_id,
                            e
                        );
                    }
                }));
            }
        }

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        tracing::info!("Shutdown signal received, waiting for agents to stop...");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            handles.retain(|h| !h.is_finished());
            if handles.is_empty() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    "{} agent thread(s) did not exit in time, forcing exit",
                    handles.len()
                );
                std::process::exit(1);
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        tracing::info!("All agents stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::AgentModel;
    use anyhow::Result;

    struct AlphaAgent;
    struct BetaAgent;

    impl CoreAgent for AlphaAgent {
        type SpawnContext = AgentSpawnContext;

        fn name() -> &'static str {
            "alpha"
        }

        fn model(&self) -> &AgentModel {
            unreachable!("test agent is never run")
        }

        fn banner(_config: &Config, banner: &mut Banner) {
            banner.set_once("repo", "first");
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn from_spawn(_ctx: Self::SpawnContext) -> Result<Self> {
            Ok(Self)
        }

        fn on_shutdown(&mut self) {}
    }

    impl CoreAgent for BetaAgent {
        type SpawnContext = AgentSpawnContext;

        fn name() -> &'static str {
            "beta"
        }

        fn model(&self) -> &AgentModel {
            unreachable!("test agent is never run")
        }

        fn banner(_config: &Config, banner: &mut Banner) {
            banner.set_once("repo", "second");
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn from_spawn(_ctx: Self::SpawnContext) -> Result<Self> {
            Ok(Self)
        }

        fn on_shutdown(&mut self) {}
    }

    #[test]
    fn startup_banner_uses_system_fields_and_agent_fields_first_writer_wins() {
        let config = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            [agent.beta]
            instances = 1
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<AlphaAgent>();
        registry.register_agent::<BetaAgent>();

        let banner = build_startup_banner(&config, &registry, "/tmp/potlatch.toml");
        let fields: Vec<_> = banner
            .fields()
            .iter()
            .map(|field| (field.key.as_str(), field.value.as_str()))
            .collect();
        assert_eq!(
            fields,
            vec![
                ("config", "potlatch.toml"),
                ("repo", "first"),
                ("agents", "alpha, beta"),
            ]
        );
    }
}
