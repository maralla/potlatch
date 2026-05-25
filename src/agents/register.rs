use crate::core::registry::{AgentRegistration, AgentRegistry};
use crate::core::workflow::{WorkflowContext, spawn_core_agent};

use super::pmo::PmoAgent;
use super::reviewer::ReviewerAgent;
use super::worker::WorkerAgent;

fn spawn_worker(ctx: WorkflowContext, instance_id: usize) -> anyhow::Result<()> {
    spawn_core_agent::<WorkerAgent>(ctx, instance_id)
}

fn spawn_reviewer(ctx: WorkflowContext, instance_id: usize) -> anyhow::Result<()> {
    spawn_core_agent::<ReviewerAgent>(ctx, instance_id)
}

fn spawn_pmo(ctx: WorkflowContext, instance_id: usize) -> anyhow::Result<()> {
    spawn_core_agent::<PmoAgent>(ctx, instance_id)
}

pub fn register_potlatch_agents(registry: &mut AgentRegistry) {
    registry.register(AgentRegistration {
        name: "worker",
        spawn: spawn_worker,
    });
    registry.register(AgentRegistration {
        name: "reviewer",
        spawn: spawn_reviewer,
    });
    registry.register(AgentRegistration {
        name: "pmo",
        spawn: spawn_pmo,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;

    #[test]
    fn registry_has_all_roles() {
        let mut reg = AgentRegistry::new();
        register_potlatch_agents(&mut reg);
        for name in ["worker", "reviewer", "pmo"] {
            assert!(reg.find(name).is_some(), "missing {name}");
        }
    }

    #[test]
    fn noop_spawn_context_clone() {
        let cfg = Config::from_toml_str("[agent.worker]\ninstances = 1").unwrap();
        let ctx = WorkflowContext {
            config: std::sync::Arc::new(cfg),
            base_dir: "/tmp".to_string(),
            shutdown: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let _cloned = ctx.clone_for_spawn();
    }
}
