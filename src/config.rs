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
    #[serde(default)]
    pub pmo: PmoConfig,
    /// When true, starts the Codepair MCP HTTP server and writes `.cursor/mcp.json` for each agent workspace.
    #[serde(default)]
    pub mcp: McpConfig,
    /// When non-empty, worker, reviewer, and PMO only consider issues (and merge requests, for review) that carry this label (exact match).
    /// New MRs (worker) and sub-issues (PMO) receive this label automatically. Empty string disables scoping.
    #[serde(default)]
    pub scope_label: String,
}

/// Active scope filter: empty or whitespace-only `scope_label` means handle all items (returns `None`).
pub fn scope_label_filter(scope_label: &str) -> Option<&str> {
    let t = scope_label.trim();
    if t.is_empty() { None } else { Some(t) }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    #[serde(default = "default_instances")]
    pub instances: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_reviewer_poll_interval")]
    pub poll_interval_secs: u64,
    #[serde(default = "default_instances")]
    pub instances: usize,
    #[serde(default = "default_merge_when_approved")]
    pub merge_when_approved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmoConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_pmo_poll_interval")]
    pub poll_interval_secs: u64,
    #[serde(default = "default_instances")]
    pub instances: usize,
    /// When true, answer Cursor ACP `cursor/ask_question` by posting on the GitLab issue and
    /// blocking until a `PMO_ACP_ANSWER:` comment (can stall the agent with no streamed output).
    /// Default false: immediate headless reply so the PMO run always progresses.
    #[serde(default)]
    pub cursor_ask_via_gitlab: bool,
    /// Used only when `cursor_ask_via_gitlab` is true. After this many seconds without an answer,
    /// fall back to a headless choice and post a timeout note on the issue. `0` means wait indefinitely.
    #[serde(default = "default_cursor_ask_gitlab_timeout_secs")]
    pub cursor_ask_gitlab_timeout_secs: u64,
}

fn default_poll_interval() -> u64 {
    60
}

fn default_reviewer_poll_interval() -> u64 {
    120
}

fn default_pmo_poll_interval() -> u64 {
    180
}

fn default_cursor_ask_gitlab_timeout_secs() -> u64 {
    600
}

fn default_merge_when_approved() -> bool {
    true
}

fn default_instances() -> usize {
    1
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model: None,
            poll_interval_secs: default_poll_interval(),
            instances: default_instances(),
        }
    }
}

impl Default for ReviewerConfig {
    fn default() -> Self {
        Self {
            model: None,
            poll_interval_secs: default_reviewer_poll_interval(),
            instances: default_instances(),
            merge_when_approved: default_merge_when_approved(),
        }
    }
}

impl Default for PmoConfig {
    fn default() -> Self {
        Self {
            model: None,
            poll_interval_secs: default_pmo_poll_interval(),
            instances: default_instances(),
            cursor_ask_via_gitlab: false,
            cursor_ask_gitlab_timeout_secs: default_cursor_ask_gitlab_timeout_secs(),
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

        let mut config: Config = toml::from_str(&content).context("Failed to parse config file")?;
        // Whitespace-only is treated as "no scope"
        if config.scope_label.trim().is_empty() {
            config.scope_label.clear();
        }

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
        assert!(!config.mcp.enabled);
        assert_eq!(config.worker.poll_interval_secs, 60);
        assert_eq!(config.reviewer.poll_interval_secs, 120);
        assert_eq!(config.pmo.poll_interval_secs, 180);
        assert!(config.worker.model.is_none());
        assert!(config.reviewer.model.is_none());
        assert!(config.pmo.model.is_none());
        assert!(!config.pmo.cursor_ask_via_gitlab);
        assert!(config.reviewer.merge_when_approved);
        assert!(config.scope_label.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config {
            worker: WorkerConfig {
                model: Some("claude-3-5-sonnet".to_string()),
                poll_interval_secs: 30,
                instances: 3,
            },
            reviewer: ReviewerConfig {
                model: Some("claude-3-opus".to_string()),
                poll_interval_secs: 60,
                instances: 2,
                merge_when_approved: false,
            },
            pmo: PmoConfig {
                model: Some("claude-3-5-sonnet".to_string()),
                poll_interval_secs: 90,
                instances: 1,
                ..Default::default()
            },
            mcp: McpConfig { enabled: true },
            scope_label: "codepair".to_string(),
        };

        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.worker.model, Some("claude-3-5-sonnet".to_string()));
        assert_eq!(parsed.reviewer.model, Some("claude-3-opus".to_string()));
        assert_eq!(parsed.pmo.model, Some("claude-3-5-sonnet".to_string()));
        assert!(!parsed.reviewer.merge_when_approved);
        assert!(parsed.mcp.enabled);
        assert_eq!(parsed.scope_label, "codepair");
    }

    #[test]
    fn scope_label_filter_empty_means_all() {
        assert_eq!(scope_label_filter(""), None);
        assert_eq!(scope_label_filter("  "), None);
        assert_eq!(scope_label_filter("codepair"), Some("codepair"));
        assert_eq!(scope_label_filter(" codepair "), Some("codepair"));
    }
}
