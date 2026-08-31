use anyhow::{Context, Result};
use serde::Deserialize;

use crate::core::config::Config;

#[derive(Debug, Clone, Default)]
pub struct AgentSettings {
    pub repo_url: Option<String>,
    pub scope_label: String,
}

#[derive(Debug, Default, Deserialize)]
struct SettingsTopLevel {
    repo_url: Option<String>,
    gitlab_repo: Option<String>,
    #[serde(default)]
    scope_label: String,
}

impl AgentSettings {
    pub fn from_config(config: &Config) -> Result<Self> {
        let top: SettingsTopLevel = config
            .deserialize()
            .context("Failed to parse agent settings from config")?;
        let repo_url = top
            .repo_url
            .or(top.gitlab_repo)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let mut scope_label = top.scope_label;
        if scope_label.trim().is_empty() {
            scope_label.clear();
        }
        Ok(Self {
            repo_url,
            scope_label,
        })
    }

    pub fn repo_url(&self) -> Option<&str> {
        self.repo_url.as_deref()
    }

    pub fn require_repo_url(&self) -> Result<&str> {
        self.repo_url()
            .context("repo_url (or gitlab_repo) is required in config for this agent")
    }

    pub fn require_gitlab_repo(&self) -> Result<&str> {
        self.require_repo_url()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repo_url_from_config() {
        let config = Config::from_toml_str(
            r#"
            repo_url = "https://github.com/group/project"
            scope_label = "potlatch"

            [agent.worker]
            instances = 1
            "#,
        )
        .unwrap();
        let s = AgentSettings::from_config(&config).unwrap();
        assert_eq!(s.repo_url(), Some("https://github.com/group/project"));
        assert_eq!(
            super::super::forge::scope_label_filter(&s.scope_label),
            Some("potlatch")
        );
    }

    #[test]
    fn parses_legacy_gitlab_repo_from_config() {
        let config = Config::from_toml_str(
            r#"
            gitlab_repo = "https://gitlab.com/group/project"
            "#,
        )
        .unwrap();
        let s = AgentSettings::from_config(&config).unwrap();
        assert_eq!(s.repo_url(), Some("https://gitlab.com/group/project"));
    }

    #[test]
    fn repo_url_takes_priority_over_gitlab_repo() {
        let config = Config::from_toml_str(
            r#"
            repo_url = "https://github.com/a/b"
            gitlab_repo = "https://gitlab.com/c/d"
            "#,
        )
        .unwrap();
        let s = AgentSettings::from_config(&config).unwrap();
        assert_eq!(s.repo_url(), Some("https://github.com/a/b"));
    }

    #[test]
    fn normalizes_empty_top_level_values() {
        let config = Config::from_toml_str(
            r#"
            repo_url = "  "
            scope_label = "  "
            "#,
        )
        .unwrap();

        let settings = AgentSettings::from_config(&config).unwrap();
        assert_eq!(settings.repo_url(), None);
        assert_eq!(
            super::super::forge::scope_label_filter(&settings.scope_label),
            None
        );
    }

    #[test]
    fn scope_label_filter_empty_means_all() {
        let s = AgentSettings::default();
        assert_eq!(
            super::super::forge::scope_label_filter(&s.scope_label),
            None
        );
        assert_eq!(super::super::forge::scope_label_filter("  "), None);
        assert_eq!(
            super::super::forge::scope_label_filter("potlatch"),
            Some("potlatch")
        );
    }

    #[test]
    fn require_repo_url_errors_when_missing() {
        let s = AgentSettings::default();
        assert!(s.require_repo_url().is_err());
    }
}
