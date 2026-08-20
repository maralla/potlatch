use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;
use toml::Value;

use super::uri::ModelUri;

#[derive(Debug, Clone)]
pub struct AgentCoreFields {
    pub instances: usize,
    pub model: Option<ModelUri>,
    /// Name of an `[acp.<name>]` profile to use when spawning the ACP server.
    pub acp_client: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AgentSection {
    pub core: AgentCoreFields,
    pub raw: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AgentCoreFieldsRaw {
    #[serde(default = "default_instances")]
    instances: usize,
    model: Option<String>,
    acp_client: Option<String>,
}

fn default_instances() -> usize {
    1
}

pub fn parse_agent_sections(root: &Value) -> Result<HashMap<String, AgentSection>> {
    let mut agents = HashMap::new();
    let Some(agent_root) = root.get("agent").and_then(Value::as_table) else {
        return Ok(agents);
    };

    for (name, section_value) in agent_root {
        let section_table = section_value
            .as_table()
            .with_context(|| format!("[agent.{name}] must be a table"))?;

        let core_raw: AgentCoreFieldsRaw = section_value
            .clone()
            .try_into()
            .with_context(|| format!("failed to parse core fields for [agent.{name}]"))?;

        let model = core_raw
            .model
            .as_deref()
            .map(ModelUri::parse)
            .transpose()
            .with_context(|| format!("invalid model URI in [agent.{name}]"))?;

        let acp_client = core_raw
            .acp_client
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let mut raw_table = section_table.clone();
        raw_table.remove("instances");
        raw_table.remove("model");
        raw_table.remove("acp_client");

        agents.insert(
            name.clone(),
            AgentSection {
                core: AgentCoreFields {
                    instances: core_raw.instances,
                    model,
                    acp_client,
                },
                raw: Value::Table(raw_table),
            },
        );
    }

    Ok(agents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_core_and_raw() {
        let doc: Value = toml::from_str(
            r#"
            [agent.alpha]
            model = "acp://cursor/gpt-5.3-codex"
            instances = 2
            poll_interval = "2m"
            merge_when_approved = true
            "#,
        )
        .unwrap();
        let agents = parse_agent_sections(&doc).unwrap();
        let section = agents.get("alpha").unwrap();
        assert_eq!(section.core.instances, 2);
        assert_eq!(
            section.core.model.as_ref().unwrap().model_name,
            "gpt-5.3-codex"
        );
        assert_eq!(
            section.raw.get("poll_interval").unwrap().as_str(),
            Some("2m")
        );
        assert!(section.raw.get("instances").is_none());
        assert!(section.raw.get("model").is_none());
        assert!(section.raw.get("acp_client").is_none());
    }

    #[test]
    fn parses_acp_client() {
        let doc: Value = toml::from_str(
            r#"
            [agent.alpha]
            acp_client = "cursor-local"
            instances = 1
            "#,
        )
        .unwrap();
        let agents = parse_agent_sections(&doc).unwrap();
        let section = agents.get("alpha").unwrap();
        assert_eq!(section.core.acp_client.as_deref(), Some("cursor-local"));
        assert!(section.raw.get("acp_client").is_none());
    }
}
