mod handoff;
mod model;
pub mod schema;

pub use handoff::AgentHandoff;
pub use model::{AgentModel, ModelPreferences};
pub(crate) use schema::structured_output;
pub use schema::{ObjectSchema, OneOfSchema, Schema, StructuredOutput, compat};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;

use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::{PeriodicTaskSpec, run_periodic_scheduler};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::{AgentBuildContext, AgentSpawnContext};

#[derive(Clone, Default)]
pub struct InvokeOptions {
    pub cancel_check: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub follow_up_poll: Option<Arc<dyn Fn() -> Vec<String> + Send + Sync>>,
    pub activity_label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub handoff: AgentHandoff,
}

pub trait CoreAgent: Sized {
    /// Role-specific settings parsed from this agent's config section.
    type Settings: DeserializeOwned;

    /// Fixed process count for roles whose topology is intrinsic. When set,
    /// the configured `instances` value is ignored.
    const FIXED_INSTANCES: Option<usize> = None;

    /// Optional generic instance limit enforced by core during startup
    /// validation and every supervised construction attempt.
    const MAX_INSTANCES: Option<usize> = None;

    fn name() -> &'static str;

    /// Parse role settings from the already-resolved agent section.
    fn parse_settings(_config: &Config, section: &AgentSection) -> Result<Self::Settings> {
        section
            .raw
            .clone()
            .try_into()
            .context("deserialize agent settings")
    }

    /// Validate parsed settings against shared configuration. Domain-specific
    /// checks remain with the agent layer; core owns invocation and context.
    fn validate_settings(
        _config: &Config,
        _section: &AgentSection,
        _settings: &Self::Settings,
    ) -> Result<()> {
        Ok(())
    }

    /// The shared runtime backing this instance: identity, the process-wide
    /// shutdown flag, and observable health. Supplied by the supervisor at
    /// spawn time through [`AgentBuildContext`] and stored by the
    /// implementation. See [`crate::core::runtime::AgentRuntime`].
    fn runtime(&self) -> &AgentRuntime;

    fn agent_id(&self) -> &str {
        self.runtime().agent_id()
    }

    fn shutdown(&self) -> &Arc<AtomicBool> {
        self.runtime().shutdown()
    }

    fn periodic_tasks(&self) -> Vec<crate::core::periodic::PeriodicTaskSpec> {
        vec![]
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()>;

    /// Construct role resources from typed settings and the authoritative
    /// core spawn context.
    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self>;
    fn on_start(&mut self) -> Result<()> {
        Ok(())
    }
    fn on_shutdown(&mut self) {}

    fn run(self) -> Result<()> {
        let _badge = crate::ui::AgentBadgeGuard::new(self.agent_id());
        self.run_with_scheduler(run_periodic_scheduler::<Self>)
    }

    fn run_with_scheduler<F>(mut self, scheduler: F) -> Result<()>
    where
        F: FnOnce(&mut Self, &[PeriodicTaskSpec], &AtomicBool) -> Result<()>,
    {
        // Guarantees `on_shutdown` runs exactly once for this constructed
        // agent, even if `on_start` or the scheduler panics: the guard's
        // `Drop` fires during unwind, before a supervisor's `catch_unwind`
        // boundary is reached.
        let guard = ShutdownGuard::new(&mut self);
        let result = match guard.agent.on_start() {
            Ok(()) => {
                guard.agent.runtime().mark_idle();
                let autostart: Vec<_> = guard
                    .agent
                    .periodic_tasks()
                    .into_iter()
                    .filter(|t| t.autostart)
                    .collect();
                let shutdown = Arc::clone(guard.agent.shutdown());
                scheduler(guard.agent, &autostart, &shutdown)
            }
            Err(error) => Err(error),
        };
        guard.finish();
        result
    }

    fn run_from(ctx: AgentSpawnContext) -> Result<()> {
        let section = ctx
            .workflow
            .config
            .agent(ctx.agent_name)
            .with_context(|| format!("[agent.{}] section required", ctx.agent_name))?;
        let instances = Self::FIXED_INSTANCES.unwrap_or(section.core.instances);
        anyhow::ensure!(
            ctx.instance_id < instances,
            "[agent.{}] invalid instance id {} (effective instances: {})",
            Self::name(),
            ctx.instance_id,
            instances,
        );
        let settings = prepare_agent_settings::<Self>(&ctx.workflow.config, section)?;
        let agent_name = ctx.agent_name;
        Self::build(AgentBuildContext::new(ctx, settings))
            .with_context(|| format!("build [agent.{agent_name}]"))?
            .run()
    }
}

/// Validate one registered agent section through core's fixed pipeline.
pub(crate) fn validate_agent_config<A: CoreAgent>(
    config: &Config,
    section: &AgentSection,
) -> Result<()> {
    prepare_agent_settings::<A>(config, section).map(drop)
}

fn prepare_agent_settings<A: CoreAgent>(
    config: &Config,
    section: &AgentSection,
) -> Result<A::Settings> {
    if let Some(max) = A::MAX_INSTANCES {
        anyhow::ensure!(
            section.core.instances <= max,
            "[agent.{}] supports at most {} instance(s) (got {})",
            A::name(),
            max,
            section.core.instances,
        );
    }
    let settings = A::parse_settings(config, section)
        .with_context(|| format!("parse settings for [agent.{}]", A::name()))?;
    A::validate_settings(config, section, &settings)
        .with_context(|| format!("validate settings for [agent.{}]", A::name()))?;
    Ok(settings)
}

/// See [`CoreAgent::run_with_scheduler`] for the guarantee this provides.
struct ShutdownGuard<'a, A: CoreAgent> {
    agent: &'a mut A,
    done: bool,
}

impl<'a, A: CoreAgent> ShutdownGuard<'a, A> {
    fn new(agent: &'a mut A) -> Self {
        Self { agent, done: false }
    }

    /// Consume the guard, running the once-only shutdown sequence now
    /// (rather than waiting for `Drop`) so normal control flow stays
    /// visible at the call site.
    fn finish(mut self) {
        self.run_once();
    }

    fn run_once(&mut self) {
        if !self.done {
            self.done = true;
            self.agent.runtime().mark_stopping();
            self.agent.on_shutdown();
            self.agent.runtime().mark_stopped();
        }
    }
}

impl<'a, A: CoreAgent> Drop for ShutdownGuard<'a, A> {
    fn drop(&mut self) {
        self.run_once();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use anyhow::{Result, anyhow};

    use super::*;
    use crate::core::runtime::HealthState;

    struct LifecycleAgent {
        runtime: AgentRuntime,
        shutdown_calls: Arc<AtomicUsize>,
        start_error: bool,
    }

    impl CoreAgent for LifecycleAgent {
        type Settings = toml::Value;

        fn name() -> &'static str {
            "lifecycle-test"
        }

        fn runtime(&self) -> &AgentRuntime {
            &self.runtime
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn build(_ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
            unreachable!("lifecycle tests construct the agent directly")
        }

        fn on_start(&mut self) -> Result<()> {
            if self.start_error {
                Err(anyhow!("start failed"))
            } else {
                Ok(())
            }
        }

        fn on_shutdown(&mut self) {
            self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn agent(start_error: bool) -> (LifecycleAgent, Arc<AtomicUsize>) {
        let shutdown_calls = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        (
            LifecycleAgent {
                runtime: AgentRuntime::new("lifecycle-test-0", shutdown),
                shutdown_calls: Arc::clone(&shutdown_calls),
                start_error,
            },
            shutdown_calls,
        )
    }

    #[test]
    fn shutdown_runs_when_start_fails() {
        let (agent, shutdown_calls) = agent(true);

        let result = agent.run_with_scheduler(|_, _, _| Ok(()));

        assert!(result.unwrap_err().to_string().contains("start failed"));
        assert_eq!(shutdown_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn shutdown_runs_when_scheduler_fails() {
        let (agent, shutdown_calls) = agent(false);

        let result = agent.run_with_scheduler(|_, _, _| Err(anyhow!("scheduler failed")));

        assert!(result.unwrap_err().to_string().contains("scheduler failed"));
        assert_eq!(shutdown_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn shutdown_runs_exactly_once_when_scheduler_panics() {
        let (agent, shutdown_calls) = agent(false);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent.run_with_scheduler(|_, _, _| panic!("scheduler exploded"))
        }));

        assert!(result.is_err());
        assert_eq!(shutdown_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn shutdown_runs_exactly_once_when_on_start_panics() {
        struct PanicsOnStart {
            runtime: AgentRuntime,
            shutdown_calls: Arc<AtomicUsize>,
        }

        impl CoreAgent for PanicsOnStart {
            type Settings = toml::Value;

            fn name() -> &'static str {
                "panics-on-start-test"
            }

            fn runtime(&self) -> &AgentRuntime {
                &self.runtime
            }

            fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
                Ok(())
            }

            fn build(_ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
                unreachable!()
            }

            fn on_start(&mut self) -> Result<()> {
                panic!("start exploded")
            }

            fn on_shutdown(&mut self) {
                self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
            }
        }

        let shutdown_calls = Arc::new(AtomicUsize::new(0));
        let agent = PanicsOnStart {
            runtime: AgentRuntime::new("panics-on-start-test-0", Arc::new(AtomicBool::new(false))),
            shutdown_calls: Arc::clone(&shutdown_calls),
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent.run_with_scheduler(|_, _, _| Ok(()))
        }));

        assert!(result.is_err());
        assert_eq!(shutdown_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn health_is_idle_once_started_and_stopped_once_shutdown_completes() {
        let (agent, _shutdown_calls) = agent(false);
        let runtime = agent.runtime().clone();
        let mut seen_idle_during_scheduler = false;

        agent
            .run_with_scheduler(|a, _, _| {
                seen_idle_during_scheduler = a.runtime().health().state == HealthState::Idle;
                Ok(())
            })
            .unwrap();

        assert!(seen_idle_during_scheduler);
        assert_eq!(runtime.health().state, HealthState::Stopped);
    }
}
