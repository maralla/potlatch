pub(crate) mod acp;
mod agent;
pub(crate) mod duration;
pub mod uri;

use acp::apply_api_flavor;
pub use acp::{
    AcpClientProfile, AcpSpawnConfig, POTLATCH_ACP_PROFILE, build_acp_spawn_command,
    build_profile_command, parse_acp_profiles, resolve_profile_env,
};
pub use agent::{AgentSection, parse_agent_sections};

use crate::core::config::uri::ModelUri;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::de::DeserializeOwned;
use toml::Value;
use tracing::{debug, info};

use crate::paths::CONFIG_FILE_NAME;

/// Env var through which an endpoint's `auth_provider` command argv (a JSON
/// array) is passed from the resolved `[acp.*]` profile to the harness
/// subprocess.
pub const AUTH_COMMAND_ENV: &str = "POTLATCH_AUTH_COMMAND";

/// Env var through which the directory containing the potlatch config file
/// is passed to the harness subprocess. The harness runs the auth-provider
/// command with this as its working directory, so `./auth-tool.py` resolves
/// relative to the config, not the agent's repo checkout.
pub const AUTH_COMMAND_DIR_ENV: &str = "POTLATCH_AUTH_DIR";

#[derive(Debug, Clone)]
pub struct Config {
    raw: Value,
    agents: HashMap<String, AgentSection>,
    acp_clients: HashMap<String, AcpClientProfile>,
    /// Directory containing the loaded config file. `None` for synthetic
    /// configs (`from_toml_str`) — their spawn configs carry no config dir.
    config_dir: Option<PathBuf>,
}

impl Config {
    pub fn load(path: Option<&str>) -> Result<Self> {
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
            return Self::from_toml_str("");
        }

        let content = fs::read_to_string(&config_path).context("Failed to read config file")?;
        // Canonicalize so the dir is absolute (the harness child resolves it
        // against its own cwd, which differs from ours) and so `potlatch run`
        // from the config's own directory ("potlatch.toml" → parent "") still
        // yields the real directory instead of an empty path.
        let config_dir = config_path
            .canonicalize()
            .ok()
            .and_then(|abs| abs.parent().map(Path::to_path_buf));
        Self::from_toml_str_with_dir(&content, config_dir)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        Self::from_toml_str_with_dir(content, None)
    }

    fn from_toml_str_with_dir(content: &str, config_dir: Option<PathBuf>) -> Result<Self> {
        let root: Value = toml::from_str(content).context("Failed to parse config file")?;
        let agents = parse_agent_sections(&root)?;
        let acp_clients = parse_acp_profiles(&root)?;

        for (name, section) in &agents {
            if let Some(model) = &section.core.model {
                let vendor = &model.vendor;
                ensure!(
                    acp_clients.contains_key(vendor),
                    "[agent.{name}] model vendor `{vendor}` has no matching [acp.{vendor}] section"
                );
            }
        }

        debug!(
            "Config loaded: {} agent section(s), {} acp client profile(s)",
            agents.len(),
            acp_clients.len()
        );

        Ok(Config {
            raw: root,
            agents,
            acp_clients,
            config_dir,
        })
    }

    fn find_config_file() -> Result<PathBuf> {
        let dotted = format!(".{CONFIG_FILE_NAME}");
        let nested = format!("config/{CONFIG_FILE_NAME}");
        for candidate in [CONFIG_FILE_NAME, &dotted, &nested] {
            let path = Path::new(candidate);
            if path.exists() {
                return Ok(path.to_path_buf());
            }
        }
        Ok(PathBuf::from(CONFIG_FILE_NAME))
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

    /// Deserialize configuration from the already-parsed top-level TOML value.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        self.raw
            .clone()
            .try_into()
            .context("Failed to deserialize top-level config")
    }

    /// Resolve the ACP executable/args, subprocess env, and model for an agent role.
    pub fn resolve_acp_spawn(&self, section: &AgentSection) -> Result<AcpSpawnConfig> {
        let model = section
            .core
            .model
            .as_ref()
            .context("no model configured for this agent")?;

        let model_uri = model.as_configured().to_string();
        let endpoint_model_name = model.endpoint_model_name().to_string();
        let bare_model = model.bare_model_name().to_string();

        let profile = self
            .acp_clients
            .get(&model.vendor)
            .with_context(|| format!("unknown acp client `{}`", model.vendor))?;

        let endpoint = acp::select_endpoint(profile, &bare_model)?;
        // An aliased request is rewritten to the entry's real model name for
        // the backend; any query suffix (e.g. `?effort=high`) is preserved.
        let endpoint_model = match endpoint {
            Some(entry) => {
                let wire_bare = entry.wire_model_name(&bare_model);
                match endpoint_model_name.strip_prefix(&bare_model) {
                    Some(query) => format!("{wire_bare}{query}"),
                    None => wire_bare.to_string(),
                }
            }
            None => endpoint_model_name,
        };

        let mut env = resolve_profile_env(profile, Some(&bare_model))?;
        apply_api_flavor(
            &mut env,
            endpoint
                .and_then(|e| e.api.as_deref())
                .or_else(|| profile.fields.get("api").map(String::as_str)),
        );

        Ok(AcpSpawnConfig {
            command: build_profile_command(profile),
            model_uri: Some(model_uri),
            endpoint_model: Some(endpoint_model),
            env,
            auth_command: endpoint.and_then(|e| e.auth_command.clone()),
            config_dir: self.config_dir.clone(),
            model_aliases: profile.model_aliases(),
        })
    }

    /// Resolve the spawn config for the platform's own harness — the shared
    /// child the subagent agent multiplexes subagent sessions onto. There is
    /// no per-agent model URI: the command is the running executable itself
    /// (`potlatch harness`), the well-known `potlatch` ACP profile supplies
    /// the endpoint env/auth, and every provided model must resolve to that
    /// profile's same endpoint — one harness child serves exactly one
    /// endpoint.
    /// Resolve the spawn config for one subagent ACP client: the child
    /// process serving the subagent sessions of one vendor. `provided_models`
    /// are the full model URIs of that vendor; the `[acp.<vendor>]` profile
    /// supplies the child's env/auth, and every model must resolve to the
    /// profile's same endpoint — one child serves exactly one endpoint. The
    /// platform's own harness vendor spawns the running executable
    /// (`potlatch harness`); every other vendor uses its profile's
    /// `acp_command`.
    pub fn resolve_client_spawn(
        &self,
        vendor: &str,
        provided_models: &[String],
    ) -> Result<AcpSpawnConfig> {
        let profile = self.acp_clients.get(vendor).with_context(|| {
            format!(
                "the `[acp.{vendor}]` profile is required for the \
                subagent clients of vendor '{vendor}'"
            )
        })?;

        // One ACP child serves one endpoint (its base_url/auth are baked
        // into the child's env): every provided model of this vendor must
        // select the same one. Profiles without an `endpoints` table are
        // single-endpoint by construction — nothing to select. Entries are
        // compared by their addressable name (alias or bare model), which is
        // unique per profile — two aliases of one model on different
        // endpoints are different endpoints.
        let addressable_name =
            |e: &acp::EndpointEntry| e.alias.clone().unwrap_or_else(|| e.model.clone());
        let mut endpoint: Option<&acp::EndpointEntry> = None;
        for model in provided_models {
            let uri = ModelUri::parse(model).with_context(|| format!("invalid model '{model}'"))?;
            match acp::select_endpoint(profile, uri.bare_model_name())? {
                None => {}
                Some(entry) => match endpoint {
                    Some(selected) if addressable_name(selected) == addressable_name(entry) => {}
                    Some(selected) => anyhow::bail!(
                        "provided_models of vendor '{vendor}' must all belong to one ACP \
                         endpoint (the shared child serves exactly one): '{}' and '{}' differ",
                        addressable_name(selected),
                        addressable_name(entry)
                    ),
                    None => endpoint = Some(entry),
                },
            }
        }

        let command = if vendor == acp::POTLATCH_ACP_PROFILE {
            // The platform's own harness: run THIS binary — no PATH lookup.
            vec![
                std::env::current_exe()
                    .context("locate the running potlatch executable for the harness")?
                    .display()
                    .to_string(),
                "harness".to_string(),
            ]
        } else {
            build_profile_command(profile)
        };

        let mut env = resolve_profile_env(
            profile,
            endpoint.map(|e| e.alias.as_deref().unwrap_or(&e.model)),
        )?;
        apply_api_flavor(
            &mut env,
            endpoint
                .and_then(|e| e.api.as_deref())
                .or_else(|| profile.fields.get("api").map(String::as_str)),
        );
        Ok(AcpSpawnConfig {
            command,
            model_uri: None,
            endpoint_model: None,
            env,
            auth_command: endpoint.and_then(|e| e.auth_command.clone()),
            config_dir: self.config_dir.clone(),
            model_aliases: profile.model_aliases(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_config_dir_resolves_absolute_config_dir() {
        // Regression: running `potlatch run` inside the config's directory
        // resolves the config as the bare relative filename
        // "potlatch.toml", whose `parent()` is "" — the config dir was
        // dropped, the harness got no POTLATCH_AUTH_DIR, and the auth
        // provider failed to find its script. The loaded config must carry
        // the config's *absolute* directory.
        let dir = std::env::temp_dir().join(format!("potlatch-cfgdir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("potlatch.toml"),
            r#"
[acp.potlatch]
acp_command = ["potlatch", "harness"]
endpoints = [
  { model = "m", endpoint = "http://prod.example", auth_provider = "auth-tool.py" },
]

[agent.worker]
model = "acp://potlatch/m"
"#,
        )
        .unwrap();

        // Load exactly the way `potlatch run` does from inside the dir.
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let loaded = Config::load(Some("potlatch.toml"));
        std::env::set_current_dir(previous).unwrap();
        let cfg = loaded.unwrap();

        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        let expected = dir.canonicalize().unwrap();
        assert_eq!(spawn.config_dir.as_deref(), Some(expected.as_path()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_toml_str_has_no_config_dir() {
        let cfg = Config::from_toml_str("[agent.alpha]\ninstances = 1\n").unwrap();
        let section = cfg.agent("alpha").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap_err();
        assert!(spawn.to_string().contains("no model configured"));
    }

    #[test]
    fn loads_agent_sections_only_when_present() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.alpha]
            instances = 1
            poll_interval = "1m"
            "#,
        )
        .unwrap();
        let names: Vec<_> = cfg.agent_names().collect();
        assert_eq!(names, vec!["alpha"]);
        assert!(cfg.agent("beta").is_none());
    }

    #[derive(Debug, serde::Deserialize, PartialEq, Eq)]
    struct TopLevelFixture {
        repository: String,
        #[serde(default)]
        scope: String,
    }

    #[test]
    fn injects_top_level_settings_from_single_parsed_value() {
        let cfg = Config::from_toml_str(
            r#"
            repository = "group/project"
            scope = "potlatch"

            [agent.worker]
            instances = 1
            "#,
        )
        .unwrap();

        assert_eq!(
            cfg.deserialize::<TopLevelFixture>().unwrap(),
            TopLevelFixture {
                repository: "group/project".to_string(),
                scope: "potlatch".to_string(),
            }
        );
    }

    #[test]
    fn agent_without_model_errors_on_resolve() {
        let cfg = Config::from_toml_str(
            r#"
            [agent.worker]
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let result = cfg.resolve_acp_spawn(section);
        assert!(result.is_err());
    }

    #[test]
    fn agent_model_vendor_must_match_acp_section() {
        let result = Config::from_toml_str(
            r#"
            [agent.worker]
            model = "acp://nonexistent/composer-2"
            instances = 1
            "#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn agent_vendor_resolves_acp_profile() {
        let cfg = Config::from_toml_str(
            r#"
            [acp.cursor-local]
            base_url = "http://endpoint1.example/v1"
            api_key = "EMPTY"
            acp_command = ["agent-local", "--print", "--trust", "--force", "--approve-mcps", "acp"]
            env = [
                "CURSOR_LOCAL_AGENT_BASE_URL={base_url}",
                "CURSOR_LOCAL_AGENT_API_KEY={api_key}",
            ]

            [agent.worker]
            model = "acp://cursor-local/model1"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        assert_eq!(spawn.command[0], "agent-local");
        assert_eq!(
            spawn.model_uri.as_deref(),
            Some("acp://cursor-local/model1")
        );
        assert_eq!(spawn.endpoint_model.as_deref(), Some("model1"));
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
            Some("http://endpoint1.example/v1")
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
    fn unknown_vendor_errors() {
        let result = Config::from_toml_str(
            r#"
            [agent.worker]
            model = "acp://nonexistent/model"
            instances = 1
            "#,
        );
        assert!(result.is_err());
    }
}
