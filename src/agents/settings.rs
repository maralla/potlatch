use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use serde::Deserialize;
use toml::Value;

#[derive(Debug, Clone, Default)]
pub struct AgentSettings {
    pub gitlab_repo: Option<String>,
    pub scope_label: String,
}

#[derive(Debug, Default, Deserialize)]
struct SettingsTopLevel {
    gitlab_repo: Option<String>,
    #[serde(default)]
    scope_label: String,
}

static SETTINGS: OnceLock<Arc<AgentSettings>> = OnceLock::new();

impl AgentSettings {
    pub fn from_toml_str(content: &str) -> Result<Self> {
        let root: Value = toml::from_str(content).context("Failed to parse config file")?;
        let top: SettingsTopLevel = root
            .try_into()
            .context("Failed to parse agent settings from config")?;
        let gitlab_repo = top
            .gitlab_repo
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let mut scope_label = top.scope_label;
        if scope_label.trim().is_empty() {
            scope_label.clear();
        }
        Ok(Self {
            gitlab_repo,
            scope_label,
        })
    }

    pub fn gitlab_repo(&self) -> Option<&str> {
        self.gitlab_repo.as_deref()
    }

    pub fn require_gitlab_repo(&self) -> Result<&str> {
        self.gitlab_repo()
            .context("gitlab_repo is required in config for this agent")
    }

    pub fn scope_label_filter(&self) -> Option<&str> {
        super::scope_label_filter(&self.scope_label)
    }
}

pub fn init(settings: AgentSettings) {
    let _ = SETTINGS.set(Arc::new(settings));
}

pub fn settings() -> Arc<AgentSettings> {
    Arc::clone(
        SETTINGS
            .get()
            .expect("agents::run must call init before accessing settings"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gitlab_repo_from_toml() {
        let s = AgentSettings::from_toml_str(
            r#"
            gitlab_repo = "https://gitlab.com/group/project"
            scope_label = "potlatch"

            [agent.worker]
            instances = 1
            "#,
        )
        .unwrap();
        assert_eq!(s.gitlab_repo(), Some("https://gitlab.com/group/project"));
        assert_eq!(s.scope_label_filter(), Some("potlatch"));
    }

    #[test]
    fn scope_label_filter_empty_means_all() {
        let s = AgentSettings::default();
        assert_eq!(s.scope_label_filter(), None);
        assert_eq!(super::super::scope_label_filter("  "), None);
        assert_eq!(super::super::scope_label_filter("potlatch"), Some("potlatch"));
    }

    #[test]
    fn require_gitlab_repo_errors_when_missing() {
        let s = AgentSettings::default();
        assert!(s.require_gitlab_repo().is_err());
    }
}
