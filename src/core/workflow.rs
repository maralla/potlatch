use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use crate::core::agent::CoreAgent;
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

/// Start and run a registered [`CoreAgent`] until workflow shutdown.
pub fn spawn_core_agent<A>(workflow: WorkflowContext, instance_id: usize) -> Result<()>
where
    A: CoreAgent<SpawnContext = AgentSpawnContext>,
{
    A::run_from(AgentSpawnContext {
        workflow,
        instance_id,
    })
}

pub struct Workflow;

impl Workflow {
    pub fn run(
        config: Config,
        registry: &AgentRegistry,
        prepare: impl FnOnce(&WorkflowContext) -> Result<()>,
    ) -> Result<()> {
        let base_dir = std::env::current_dir()
            .context("Failed to get current directory")?
            .to_string_lossy()
            .into_owned();

        let shutdown = Arc::new(AtomicBool::new(false));
        let ctx = WorkflowContext {
            config: Arc::new(config),
            base_dir,
            shutdown: Arc::clone(&shutdown),
        };

        prepare(&ctx)?;

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
            let registration = registry.find(&agent_name).with_context(|| {
                format!("no registration for configured agent [agent.{agent_name}]")
            })?;

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
