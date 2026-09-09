use crate::core::workflow::Workflow;

mod artifact;
mod claim;
mod clerk;
mod forge;
mod git;
mod labels;
mod ops;
mod pmo;
mod qa;
mod reviewer;
mod settings;
mod ssh_util;
mod state;
mod subagent;
mod web;
mod worker;
mod workspace;

pub(crate) fn register(workflow: &mut Workflow) {
    workflow.register_agent::<worker::WorkerAgent>();
    workflow.register_agent::<reviewer::ReviewerAgent>();
    workflow.register_agent::<pmo::PmoAgent>();
    workflow.register_agent::<ops::OpsAgent>();
    workflow.register_agent::<qa::QaAgent>();
    workflow.register_agent::<web::WebAgent>();
    workflow.register_agent::<clerk::ClerkAgent>();
    workflow.register_agent::<subagent::SubagentAgent>();
}

#[cfg(test)]
mod scope_tests {
    use super::register;
    use crate::core::config::Config;
    use crate::core::workflow::Workflow;

    #[test]
    fn register_adds_potlatch_agents_to_workflow() {
        let config = Config::from_toml_str("").unwrap();
        let mut workflow = Workflow::new(config, "potlatch.toml");
        register(&mut workflow);
        let mut names: Vec<_> = workflow.registered_agent_names().collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "clerk", "ops", "pmo", "qa", "reviewer", "subagent", "web", "worker"
            ]
        );
    }
}
