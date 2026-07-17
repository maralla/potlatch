mod acp;
mod agent;
mod uri;

pub use acp::{
    AcpClientProfile, AcpSpawnConfig, build_acp_spawn_command, build_profile_command,
    default_acp_command, parse_acp_profiles, resolve_profile_env,
};
pub use agent::{AgentSection, parse_agent_sections};

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use toml::Value;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct Config {
    agents: HashMap<String, AgentSection>,
    acp_clients: HashMap<String, AcpClientProfile>,
}

impl Config {
    pub fn load_with_content(path: Option<&str>) -> Result<(Self, String)> {
        let config_path = if let Some(p) = path {
            PathBuf::from(p)
        } else {
            Self::find_config_file()?
        };

        if !config_path.exists() {
            info!(
                "No config file found at {}, using empty config",
                config_path.display()
            );
            return Ok((
                Config {
                    agents: HashMap::new(),
                    acp_clients: HashMap::new(),
                },
                String::new(),
            ));
        }

        let content = fs::read_to_string(&config_path).context("Failed to read config file")?;
        Ok((Self::from_toml_str(&content)?, content))
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let root: Value = toml::from_str(content).context("Failed to parse config file")?;
        let agents = parse_agent_sections(&root)?;
        let acp_clients = parse_acp_profiles(&root)?;

        debug!(
            "Config loaded: {} agent section(s), {} acp client profile(s)",
            agents.len(),
            acp_clients.len()
        );

        Ok(Config {
            agents,
            acp_clients,
        })
    }

    fn find_config_file() -> Result<PathBuf> {
        for candidate in ["potlatch.toml", ".potlatch.toml", "config/potlatch.toml"] {
            let path = Path::new(candidate);
            if path.exists() {
                return Ok(path.to_path_buf());
            }
        }
        Ok(PathBuf::from("potlatch.toml"))
    }

    pub fn save_example(path: &str) -> Result<()> {
        let example = include_str!("../../../potlatch.toml.example");
        fs::write(path, example).context("Failed to write config file")?;
        info!("Example config saved to: {}", path);
        Ok(())
    }

    pub fn agent_names(&self) -> impl Iterator<Item = &str> {
        self.agents.keys().map(String::as_str)
    }

    pub fn agent(&self, name: &str) -> Option<&AgentSection> {
        self.agents.get(name)
    }

    /// Resolve the ACP executable/args, subprocess env, and model for an agent role.
    pub fn resolve_acp_spawn(&self, section: &AgentSection) -> Result<AcpSpawnConfig> {
        let model_uri = section
            .core
            .model
            .as_ref()
            .map(|uri| uri.as_configured().to_string());
        let endpoint_model = section
            .core
            .model
            .as_ref()
            .map(|uri| uri.endpoint_model_name().to_string());

        let Some(client_name) = section.core.acp_client.as_deref() else {
            return Ok(AcpSpawnConfig {
                command: default_acp_command(),
                model_uri,
                endpoint_model,
                env: HashMap::new(),
            });
        };

        let profile = self
            .acp_clients
            .get(client_name)
            .with_context(|| format!("unknown acp_client `{client_name}`"))?;

        Ok(AcpSpawnConfig {
            command: build_profile_command(profile),
            model_uri,
            endpoint_model,
            env: resolve_profile_env(profile)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_agent_sections_only_when_present() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.alpha]
            model = "composer-2"
            instances = 1
            poll_interval_secs = 60
            "#,
        )
        .unwrap();
        let names: Vec<_> = cfg.agent_names().collect();
        assert_eq!(names, vec!["alpha"]);
        assert!(cfg.agent("beta").is_none());
    }

    #[test]
    fn agent_without_acp_client_uses_default_command() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.worker]
            model = "acp://cursor/composer-2"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        assert_eq!(spawn.command[0], "agent");
        assert_eq!(spawn.model_uri.as_deref(), Some("acp://cursor/composer-2"));
        assert_eq!(spawn.endpoint_model.as_deref(), Some("composer-2"));
        assert!(spawn.env.is_empty());
    }

    #[test]
    fn agent_with_acp_client_uses_profile() {
        let cfg = Config::from_toml_str(
            r#"
            [acp.cursor-local]
            base_url = "http://prod-model1.example/v1"
            api_key = "EMPTY"
            acp_command = ["agent-local", "--print", "--trust", "--force", "--approve-mcps", "acp"]
            env = [
                "CURSOR_LOCAL_AGENT_BASE_URL={base_url}",
                "CURSOR_LOCAL_AGENT_API_KEY={api_key}",
            ]

            [agent.worker]
            model = "acp://cursor/model1-fp8"
            acp_client = "cursor-local"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        assert_eq!(spawn.command[0], "agent-local");
        assert_eq!(spawn.model_uri.as_deref(), Some("acp://cursor/model1-fp8"));
        assert_eq!(spawn.endpoint_model.as_deref(), Some("model1-fp8"));
        // Command is used verbatim from config — no injection
        assert_eq!(spawn.command[1], "--print");
        assert_eq!(spawn.command[2], "--trust");
        assert_eq!(spawn.command[3], "--force");
        assert_eq!(spawn.command[4], "--approve-mcps");
        assert_eq!(spawn.command[5], "acp");
        assert_eq!(
            spawn
                .env
                .get("CURSOR_LOCAL_AGENT_BASE_URL")
                .map(String::as_str),
            Some("http://prod-model1.example/v1")
        );
        assert_eq!(
            spawn
                .env
                .get("CURSOR_LOCAL_AGENT_API_KEY")
                .map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn unknown_acp_client_errors() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.worker]
            acp_client = "missing"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let err = cfg.resolve_acp_spawn(section).unwrap_err();
        assert!(err.to_string().contains("unknown acp_client"));
    }
}
