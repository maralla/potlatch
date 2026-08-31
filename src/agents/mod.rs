use crate::core::workflow::Workflow;

pub(crate) mod artifact;
pub(crate) mod claim;
pub(crate) mod clerk;
pub mod forge;
pub mod git;
pub mod labels;
pub mod ops;
pub mod pmo;
pub(crate) mod qa;
pub mod reviewer;
pub mod settings;
pub(crate) mod ssh_util;
pub(crate) mod state;
pub(crate) mod web;
pub mod worker;
pub mod workspace;

pub fn register(workflow: &mut Workflow) {
    workflow.register_agent::<worker::WorkerAgent>();
    workflow.register_agent::<reviewer::ReviewerAgent>();
    workflow.register_agent::<pmo::PmoAgent>();
    workflow.register_agent::<ops::OpsAgent>();
    workflow.register_agent::<qa::QaAgent>();
    workflow.register_agent::<web::WebAgent>();
    workflow.register_agent::<clerk::ClerkAgent>();
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
            vec!["clerk", "ops", "pmo", "qa", "reviewer", "web", "worker"]
        );
    }
}
