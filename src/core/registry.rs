use anyhow::Result;

use crate::core::agent::CoreAgent;
use crate::core::banner::Banner;
use crate::core::config::Config;
use crate::core::workflow::{AgentSpawnContext, WorkflowContext, spawn_core_agent};

pub(crate) struct AgentRegistration {
    pub name: &'static str,
    pub spawn: fn(WorkflowContext, instance_id: usize) -> Result<()>,
    pub banner: fn(&Config, &mut Banner),
}

pub struct AgentRegistry {
    registrations: Vec<AgentRegistration>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
        }
    }

    pub fn register_agent<A>(&mut self)
    where
        A: CoreAgent<SpawnContext = AgentSpawnContext> + 'static,
    {
        self.registrations.push(AgentRegistration {
            name: A::name(),
            spawn: spawn_core_agent::<A>,
            banner: A::banner,
        });
    }

    pub(crate) fn find(&self, name: &str) -> Option<&AgentRegistration> {
        self.registrations.iter().find(|r| r.name == name)
    }

    #[cfg(test)]
    pub fn agent_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.registrations
            .iter()
            .map(|registration| registration.name)
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::{AgentModel, CoreAgent};
    use crate::core::config::Config;

    struct WorkerAgentForTest;

    impl CoreAgent for WorkerAgentForTest {
        type SpawnContext = AgentSpawnContext;

        fn name() -> &'static str {
            "worker"
        }

        fn model(&self) -> &AgentModel {
            unreachable!("test agent is never run")
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
    fn find_by_name() {
        let mut reg = AgentRegistry::new();
        reg.register_agent::<WorkerAgentForTest>();
        assert!(reg.find("worker").is_some());
        assert!(reg.find("reviewer").is_none());
    }

    #[test]
    fn workflow_spawns_only_configured_agents() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.worker]
            instances = 1
            [agent.reviewer]
            instances = 0
            "#,
        )
        .unwrap();
        let mut names: Vec<_> = cfg.agent_names().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["reviewer", "worker"]);
        let worker = cfg.agent("worker").unwrap();
        assert_eq!(worker.core.instances, 1);
        let reviewer = cfg.agent("reviewer").unwrap();
        assert_eq!(reviewer.core.instances, 0);
    }
}
