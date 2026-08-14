mod handoff;
mod model;

pub use handoff::{AgentHandoff, HandoffSubIssue};
pub use model::{AgentModel, ModelPreferences};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::run_periodic_scheduler;

#[derive(Clone, Default)]
pub struct InvokeOptions {
    pub cancel_check: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub activity_label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub handoff: AgentHandoff,
}

pub trait CoreAgent: Sized {
    type SpawnContext;

    fn name() -> &'static str;

    fn model(&self) -> &AgentModel;

    fn agent_id(&self) -> &str {
        self.model().agent_id()
    }

    fn shutdown(&self) -> &Arc<AtomicBool> {
        self.model().shutdown()
    }

    fn periodic_tasks(&self) -> Vec<crate::core::periodic::PeriodicTaskSpec> {
        vec![]
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn validate_config(_section: &AgentSection) -> Result<()> {
        Ok(())
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()>;

    fn from_spawn(ctx: Self::SpawnContext) -> Result<Self>;
    fn on_start(&mut self) -> Result<()> {
        Ok(())
    }
    fn on_shutdown(&mut self);

    fn run(mut self) -> Result<()> {
        let _badge = crate::ui::AgentBadgeGuard::new(self.agent_id());
        self.on_start()?;
        let autostart: Vec<_> = self
            .periodic_tasks()
            .into_iter()
            .filter(|t| t.autostart)
            .collect();
        let shutdown = Arc::clone(self.shutdown());
        run_periodic_scheduler(&mut self, &autostart, &shutdown)?;
        self.on_shutdown();
        Ok(())
    }

    fn run_from(ctx: Self::SpawnContext) -> Result<()> {
        Self::from_spawn(ctx)?.run()
    }
}
