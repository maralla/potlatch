use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;

use crate::core::activity::SharedActivityReporter;
use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::registry::AgentRegistry;
use crate::core::runtime::AgentRuntime;

pub struct WorkflowContext {
    pub config: Arc<Config>,
    pub base_dir: String,
    pub shutdown: Arc<AtomicBool>,
    pub activity: SharedActivityReporter,
}

/// Spawn inputs passed from workflow into a [`CoreAgent`] implementation.
pub struct AgentSpawnContext {
    pub workflow: WorkflowContext,
    /// Registered role name selected by core for this spawn.
    pub agent_name: &'static str,
    pub instance_id: usize,
    /// The shared runtime for this planned instance, constructed once by
    /// the [`crate::core::supervisor`] and handed to every construction
    /// attempt across restarts.
    pub runtime: AgentRuntime,
}

/// Spawn inputs after core has resolved and parsed the registered agent's
/// settings. Agent implementations only compose their role-specific
/// dependencies from this context; section lookup, parsing, identity, and
/// instance validation stay in core.
pub struct AgentBuildContext<S> {
    spawn: AgentSpawnContext,
    pub settings: S,
}

impl<S> AgentBuildContext<S> {
    pub(crate) fn new(spawn: AgentSpawnContext, settings: S) -> Self {
        Self { spawn, settings }
    }
}

impl<S> std::ops::Deref for AgentBuildContext<S> {
    type Target = AgentSpawnContext;

    fn deref(&self) -> &Self::Target {
        &self.spawn
    }
}

impl WorkflowContext {
    pub fn clone_for_spawn(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            base_dir: self.base_dir.clone(),
            shutdown: Arc::clone(&self.shutdown),
            activity: Arc::clone(&self.activity),
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

/// Supervise a registered [`CoreAgent`] for one planned instance until
/// global shutdown. See [`crate::core::supervisor`] for the restart,
/// backoff, and panic-recovery behavior.
pub(crate) fn spawn_core_agent<A>(workflow: WorkflowContext, instance_id: usize) -> Result<()>
where
    A: CoreAgent,
{
    crate::core::supervisor::supervise::<A>(workflow, instance_id)
}

#[derive(Clone)]
struct AgentSpawnPlan {
    agent_name: String,
    instance_id: usize,
    spawn: fn(WorkflowContext, instance_id: usize) -> Result<()>,
}

fn plan_agent_spawns(config: &Config, registry: &AgentRegistry) -> Result<Vec<AgentSpawnPlan>> {
    let mut agent_names: Vec<_> = config.agent_names().collect();
    agent_names.sort_unstable();
    let mut plan = Vec::new();

    for agent_name in agent_names {
        let section = config
            .agent(agent_name)
            .with_context(|| format!("missing agent section [agent.{agent_name}]"))?;
        if section.core.instances == 0 {
            continue;
        }

        let registration = registry.find(agent_name).with_context(|| {
            format!("no registration for configured agent [agent.{agent_name}]")
        })?;
        (registration.validate_config)(config, section)
            .with_context(|| format!("invalid config for [agent.{agent_name}]"))?;
        config
            .resolve_acp_spawn(section)
            .with_context(|| format!("invalid core config for [agent.{agent_name}]"))?;

        for instance_id in 0..section.core.instances {
            plan.push(AgentSpawnPlan {
                agent_name: agent_name.to_string(),
                instance_id,
                spawn: registration.spawn,
            });
        }
    }

    Ok(plan)
}

/// Spawn one supervisor thread per planned instance. Called only after
/// [`plan_agent_spawns`] has validated every configured agent, so the plan
/// passed in is already fully valid and deterministic.
fn execute_spawn_plan(
    ctx: &WorkflowContext,
    plan: Vec<AgentSpawnPlan>,
) -> Vec<std::thread::JoinHandle<()>> {
    plan.into_iter()
        .map(|planned| {
            let spawn_ctx = ctx.clone_for_spawn();
            std::thread::spawn(move || {
                if let Err(e) = (planned.spawn)(spawn_ctx, planned.instance_id) {
                    tracing::error!(
                        "Agent {}-{} supervisor exited with an error: {}",
                        planned.agent_name,
                        planned.instance_id,
                        e
                    );
                }
            })
        })
        .collect()
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
    activity: SharedActivityReporter,
}

impl Workflow {
    pub fn new(config: Config, config_path: impl Into<String>) -> Self {
        Self {
            config,
            config_path: config_path.into(),
            registry: AgentRegistry::new(),
            activity: Arc::new(crate::core::activity::NoopActivityReporter),
        }
    }

    /// Install the activity reporter propagated to every spawned agent.
    pub fn with_activity_reporter(mut self, activity: SharedActivityReporter) -> Self {
        self.activity = activity;
        self
    }

    pub fn register_agent<A>(&mut self)
    where
        A: CoreAgent + 'static,
    {
        self.registry.register_agent::<A>();
    }

    #[cfg(test)]
    pub fn registered_agent_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.registry.agent_names()
    }

    pub fn run(self) -> Result<()> {
        let spawn_plan = plan_agent_spawns(&self.config, &self.registry)?;
        let base_dir = std::env::current_dir()
            .context("Failed to get current directory")?
            .to_string_lossy()
            .into_owned();

        let shutdown = Arc::new(AtomicBool::new(false));
        let ctx = WorkflowContext {
            config: Arc::new(self.config),
            base_dir,
            shutdown: Arc::clone(&shutdown),
            activity: Arc::clone(&self.activity),
        };

        prepare_shutdown_handlers(&ctx)?;

        let banner = build_startup_banner(&ctx.config, &self.registry, &self.config_path);
        crate::ui::print_banner(&banner);

        let mut handles = execute_spawn_plan(&ctx, spawn_plan);

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
    use anyhow::Result;

    struct AlphaAgent;
    struct BetaAgent;
    struct InvalidAgent;

    impl CoreAgent for AlphaAgent {
        type Settings = toml::Value;

        fn name() -> &'static str {
            "alpha"
        }

        fn runtime(&self) -> &AgentRuntime {
            unreachable!("test agent is never run")
        }

        fn banner(_config: &Config, banner: &mut Banner) {
            banner.set_once("repo", "first");
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn build(_ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
            Ok(Self)
        }

        fn on_shutdown(&mut self) {}
    }

    impl CoreAgent for BetaAgent {
        type Settings = toml::Value;

        fn name() -> &'static str {
            "beta"
        }

        fn runtime(&self) -> &AgentRuntime {
            unreachable!("test agent is never run")
        }

        fn banner(_config: &Config, banner: &mut Banner) {
            banner.set_once("repo", "second");
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn build(_ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
            Ok(Self)
        }

        fn on_shutdown(&mut self) {}
    }

    impl CoreAgent for InvalidAgent {
        type Settings = toml::Value;

        fn name() -> &'static str {
            "invalid"
        }

        fn runtime(&self) -> &AgentRuntime {
            unreachable!("test agent is never run")
        }

        fn validate_settings(
            _config: &Config,
            _section: &crate::core::config::AgentSection,
            _settings: &Self::Settings,
        ) -> Result<()> {
            anyhow::bail!("deliberately invalid")
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn build(_ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
            unreachable!("invalid config must prevent construction")
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

    #[test]
    fn spawn_plan_is_sorted_and_expands_instances_deterministically() {
        let config = Config::from_toml_str(
            r#"
            [agent.beta]
            instances = 2
            [agent.alpha]
            instances = 1
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<AlphaAgent>();
        registry.register_agent::<BetaAgent>();

        let plan = plan_agent_spawns(&config, &registry).unwrap();
        let entries: Vec<_> = plan
            .iter()
            .map(|planned| (planned.agent_name.as_str(), planned.instance_id))
            .collect();

        assert_eq!(entries, vec![("alpha", 0), ("beta", 0), ("beta", 1)]);
    }

    #[test]
    fn spawn_plan_skips_disabled_agents_without_requiring_registration() {
        let config = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            [agent.unregistered]
            instances = 0
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<AlphaAgent>();

        let plan = plan_agent_spawns(&config, &registry).unwrap();

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].agent_name, "alpha");
    }

    #[test]
    fn spawn_plan_rejects_all_registration_errors_before_execution() {
        let config = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            [agent.unregistered]
            instances = 1
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<AlphaAgent>();

        let error = plan_agent_spawns(&config, &registry).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("no registration for configured agent [agent.unregistered]")
        );
    }

    #[test]
    fn spawn_plan_runs_agent_validation_before_execution() {
        let config = Config::from_toml_str(
            r#"
            [agent.invalid]
            instances = 1
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<InvalidAgent>();

        let error = plan_agent_spawns(&config, &registry).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("invalid config for [agent.invalid]")
        );
        assert!(
            format!("{error:#}").contains("deliberately invalid"),
            "validation cause should be preserved"
        );
    }

    #[test]
    fn spawn_plan_validates_shared_core_config_before_execution() {
        let config = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            acp_client = "missing"
            "#,
        )
        .unwrap();
        let mut registry = AgentRegistry::new();
        registry.register_agent::<AlphaAgent>();

        let error = plan_agent_spawns(&config, &registry).err().unwrap();

        assert!(
            error
                .to_string()
                .contains("invalid core config for [agent.alpha]")
        );
        assert!(
            format!("{error:#}").contains("unknown acp_client `missing`"),
            "core validation cause should be preserved"
        );
    }
}
