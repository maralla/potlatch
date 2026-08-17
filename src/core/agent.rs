mod handoff;
mod model;
pub mod schema;

pub use handoff::AgentHandoff;
pub use model::{AgentModel, ModelPreferences};
pub use schema::{ObjectSchema, SchemaField, StructuredOutput};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::{PeriodicTaskSpec, run_periodic_scheduler};

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
    type SpawnContext;

    fn name() -> &'static str;

    fn agent_id(&self) -> &str;

    fn shutdown(&self) -> &Arc<AtomicBool>;

    fn periodic_tasks(&self) -> Vec<crate::core::periodic::PeriodicTaskSpec> {
        vec![]
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn validate_config(_config: &Config, _section: &AgentSection) -> Result<()> {
        Ok(())
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()>;

    fn from_spawn(ctx: Self::SpawnContext) -> Result<Self>;
    fn on_start(&mut self) -> Result<()> {
        Ok(())
    }
    fn on_shutdown(&mut self);

    fn run(self) -> Result<()> {
        let _badge = crate::ui::AgentBadgeGuard::new(self.agent_id());
        self.run_with_scheduler(run_periodic_scheduler::<Self>)
    }

    fn run_with_scheduler<F>(mut self, scheduler: F) -> Result<()>
    where
        F: FnOnce(&mut Self, &[PeriodicTaskSpec], &AtomicBool) -> Result<()>,
    {
        let result = match self.on_start() {
            Ok(()) => {
                let autostart: Vec<_> = self
                    .periodic_tasks()
                    .into_iter()
                    .filter(|t| t.autostart)
                    .collect();
                let shutdown = Arc::clone(self.shutdown());
                scheduler(&mut self, &autostart, &shutdown)
            }
            Err(error) => Err(error),
        };
        self.on_shutdown();
        result
    }

    fn run_from(ctx: Self::SpawnContext) -> Result<()> {
        Self::from_spawn(ctx)?.run()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use anyhow::{Result, anyhow};

    use super::*;

    struct LifecycleAgent {
        shutdown: Arc<AtomicBool>,
        shutdown_calls: Arc<AtomicUsize>,
        start_error: bool,
    }

    impl CoreAgent for LifecycleAgent {
        type SpawnContext = ();

        fn name() -> &'static str {
            "lifecycle-test"
        }

        fn agent_id(&self) -> &str {
            "lifecycle-test-0"
        }

        fn shutdown(&self) -> &Arc<AtomicBool> {
            &self.shutdown
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn from_spawn(_ctx: Self::SpawnContext) -> Result<Self> {
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
        (
            LifecycleAgent {
                shutdown: Arc::new(AtomicBool::new(false)),
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
}
