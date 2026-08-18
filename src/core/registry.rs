use anyhow::Result;

use crate::core::agent::{CoreAgent, validate_agent_config};
use crate::core::banner::Banner;
use crate::core::config::{AgentSection, Config};
use crate::core::workflow::{WorkflowContext, spawn_core_agent};

pub(crate) struct AgentRegistration {
    pub name: &'static str,
    pub spawn: fn(WorkflowContext, instance_id: usize) -> Result<()>,
    pub banner: fn(&Config, &mut Banner),
    pub validate_config: fn(&Config, &AgentSection) -> Result<()>,
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
        A: CoreAgent + 'static,
    {
        self.registrations.push(AgentRegistration {
            name: A::name(),
            spawn: spawn_core_agent::<A>,
            banner: A::banner,
            validate_config: validate_agent_config::<A>,
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
    use crate::core::agent::CoreAgent;
    use crate::core::config::Config;

    struct AlphaAgentForTest;

    #[derive(serde::Deserialize)]
    struct AlphaSettings {
        #[serde(default)]
        _enabled: bool,
    }

    impl CoreAgent for AlphaAgentForTest {
        type Settings = AlphaSettings;
        const MAX_INSTANCES: Option<usize> = Some(1);

        fn name() -> &'static str {
            "alpha"
        }

        fn runtime(&self) -> &crate::core::runtime::AgentRuntime {
            unreachable!("test agent is never run")
        }

        fn run_periodic_task(&mut self, _task_id: &str) -> Result<()> {
            Ok(())
        }

        fn build(_ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
            Ok(Self)
        }

        fn on_shutdown(&mut self) {}
    }

    #[test]
    fn find_by_name() {
        let mut reg = AgentRegistry::new();
        reg.register_agent::<AlphaAgentForTest>();
        assert!(reg.find("alpha").is_some());
        assert!(reg.find("beta").is_none());
    }

    #[test]
    fn workflow_spawns_only_configured_agents() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            [agent.beta]
            instances = 0
            "#,
        )
        .unwrap();
        let mut names: Vec<_> = cfg.agent_names().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
        let alpha = cfg.agent("alpha").unwrap();
        assert_eq!(alpha.core.instances, 1);
        let beta = cfg.agent("beta").unwrap();
        assert_eq!(beta.core.instances, 0);
    }

    #[test]
    fn registration_dispatches_agent_config_validation() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 2
            "#,
        )
        .unwrap();
        let mut reg = AgentRegistry::new();
        reg.register_agent::<AlphaAgentForTest>();
        let registration = reg.find("alpha").unwrap();

        let err = (registration.validate_config)(&cfg, cfg.agent("alpha").unwrap()).unwrap_err();
        assert!(err.to_string().contains("at most 1 instance"));
    }

    #[test]
    fn registration_dispatches_typed_settings_deserialization() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            _enabled = "not-a-bool"
            "#,
        )
        .unwrap();
        let mut reg = AgentRegistry::new();
        reg.register_agent::<AlphaAgentForTest>();
        let registration = reg.find("alpha").unwrap();

        let err = (registration.validate_config)(&cfg, cfg.agent("alpha").unwrap()).unwrap_err();
        assert!(format!("{err:#}").contains("invalid type"));
    }
}
