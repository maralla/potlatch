use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub worker: WorkerConfig,
    #[serde(default)]
    pub reviewer: ReviewerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_reviewer_poll_interval")]
    pub poll_interval_secs: u64,
}

fn default_poll_interval() -> u64 {
    60
}

fn default_reviewer_poll_interval() -> u64 {
    120
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model: None,
            poll_interval_secs: default_poll_interval(),
        }
    }
}

impl Default for ReviewerConfig {
    fn default() -> Self {
        Self {
            model: None,
            poll_interval_secs: default_reviewer_poll_interval(),
        }
    }
}

impl Config {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = if let Some(p) = path {
            p.to_string()
        } else {
            Self::find_config_file()?
        };

        if !Path::new(&config_path).exists() {
            info!("No config file found, using defaults");
            return Ok(Config::default());
        }

        info!("Loading config from: {}", config_path);
        let content = fs::read_to_string(&config_path).context("Failed to read config file")?;

        let config: Config = toml::from_str(&content).context("Failed to parse config file")?;

        debug!("Config loaded: {:?}", config);
        Ok(config)
    }

    fn find_config_file() -> Result<String> {
        let candidates = vec!["codepair.toml", ".codepair.toml", "config/codepair.toml"];

        for candidate in candidates {
            if Path::new(candidate).exists() {
                return Ok(candidate.to_string());
            }
        }

        Ok("codepair.toml".to_string())
    }

    pub fn save_example(path: &str) -> Result<()> {
        let example = Config::default();
        let content = toml::to_string_pretty(&example).context("Failed to serialize config")?;

        fs::write(path, content).context("Failed to write config file")?;

        info!("Example config saved to: {}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.worker.poll_interval_secs, 60);
        assert_eq!(config.reviewer.poll_interval_secs, 120);
        assert!(config.worker.model.is_none());
        assert!(config.reviewer.model.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config {
            worker: WorkerConfig {
                model: Some("claude-3-5-sonnet".to_string()),
                poll_interval_secs: 30,
            },
            reviewer: ReviewerConfig {
                model: Some("claude-3-opus".to_string()),
                poll_interval_secs: 60,
            },
        };

        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.worker.model, Some("claude-3-5-sonnet".to_string()));
        assert_eq!(parsed.reviewer.model, Some("claude-3-opus".to_string()));
    }
}
