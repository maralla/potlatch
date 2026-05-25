use anyhow::Result;

use crate::core::workflow::WorkflowContext;

pub struct AgentRegistration {
    pub name: &'static str,
    pub spawn: fn(WorkflowContext, instance_id: usize) -> Result<()>,
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

    pub fn register(&mut self, registration: AgentRegistration) {
        self.registrations.push(registration);
    }

    pub fn find(&self, name: &str) -> Option<&AgentRegistration> {
        self.registrations.iter().find(|r| r.name == name)
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
    use crate::core::config::Config;

    fn noop_spawn(_ctx: WorkflowContext, _id: usize) -> Result<()> {
        Ok(())
    }

    #[test]
    fn find_by_name() {
        let mut reg = AgentRegistry::new();
        reg.register(AgentRegistration {
            name: "worker",
            spawn: noop_spawn,
        });
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
