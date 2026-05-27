mod agent;
mod uri;

pub use agent::{AgentSection, parse_agent_sections};
pub use uri::ModelUri;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use toml::Value;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct Config {
    agents: HashMap<String, AgentSection>,
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
        debug!("Config loaded: {} agent section(s)", agents.len());

        Ok(Config { agents })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_agent_sections_only_when_present() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.worker]
            model = "composer-2"
            instances = 1
            poll_interval_secs = 60
            "#,
        )
        .unwrap();
        let names: Vec<_> = cfg.agent_names().collect();
        assert_eq!(names, vec!["worker"]);
        assert!(cfg.agent("reviewer").is_none());
    }
}
