use std::collections::HashMap;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use toml::Value;

/// Default Cursor Agent ACP invocation (matches pre-config behavior).
pub fn default_acp_command() -> Vec<String> {
    vec![
        "agent".into(),
        "--print".into(),
        "--trust".into(),
        "--force".into(),
        "--approve-mcps".into(),
        "acp".into(),
    ]
}

#[derive(Debug, Clone)]
pub struct AcpClientProfile {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub acp_command: Vec<String>,
    pub env: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct AcpSpawnConfig {
    pub command: Vec<String>,
    /// Configured model URI (e.g. `acp://cursor/model1-fp8`).
    pub model_uri: Option<String>,
    /// Parsed `<model-name>` for `session/set_model` after `session/new`.
    pub endpoint_model: Option<String>,
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AcpClientProfileRaw {
    base_url: Option<String>,
    api_key: Option<String>,
    acp_command: Vec<String>,
    #[serde(default)]
    env: Vec<String>,
}

pub fn parse_acp_profiles(root: &Value) -> Result<HashMap<String, AcpClientProfile>> {
    let Some(acp_root) = root.get("acp").and_then(Value::as_table) else {
        return Ok(HashMap::new());
    };

    let mut profiles = HashMap::new();
    for (name, value) in acp_root {
        let raw: AcpClientProfileRaw = value
            .clone()
            .try_into()
            .with_context(|| format!("failed to parse [acp.{name}]"))?;

        validate_acp_command(&raw.acp_command)
            .with_context(|| format!("invalid acp_command in [acp.{name}]"))?;

        let profile = AcpClientProfile {
            base_url: normalize_optional_string(raw.base_url),
            api_key: normalize_optional_string(raw.api_key),
            acp_command: raw.acp_command,
            env: raw.env,
        };

        resolve_profile_env(&profile).with_context(|| format!("invalid env in [acp.{name}]"))?;

        profiles.insert(name.clone(), profile);
    }

    Ok(profiles)
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn validate_acp_command(command: &[String]) -> Result<()> {
    if command.is_empty() {
        bail!("acp_command must include at least the executable name");
    }
    if command[0].trim().is_empty() {
        bail!("acp_command executable must be non-empty");
    }
    Ok(())
}

/// Resolve `env` entries (`KEY=value`) with optional `{base_url}` / `{api_key}` placeholders.
pub fn resolve_profile_env(profile: &AcpClientProfile) -> Result<HashMap<String, String>> {
    let mut env = HashMap::new();
    for entry in &profile.env {
        let (key, value) = parse_env_entry(entry)?;
        let value = substitute_profile_placeholders(&value, profile)?;
        if env.insert(key.clone(), value).is_some() {
            bail!("duplicate env key `{key}` in ACP profile");
        }
    }
    Ok(env)
}

fn parse_env_entry(entry: &str) -> Result<(String, String)> {
    let entry = entry.trim();
    if entry.is_empty() {
        bail!("env entry must not be empty");
    }
    let Some((key, value)) = entry.split_once('=') else {
        bail!("env entry must be `KEY=value`, got `{entry}`");
    };
    let key = key.trim();
    if key.is_empty() {
        bail!("env entry key must not be empty");
    }
    Ok((key.to_string(), value.to_string()))
}

fn substitute_profile_placeholders(value: &str, profile: &AcpClientProfile) -> Result<String> {
    let mut out = value.to_string();
    if out.contains("{base_url}") {
        let base_url = profile
            .base_url
            .as_deref()
            .with_context(|| "env references {base_url} but profile has no base_url")?;
        out = out.replace("{base_url}", base_url);
    }
    if out.contains("{api_key}") {
        let api_key = profile
            .api_key
            .as_deref()
            .with_context(|| "env references {api_key} but profile has no api_key")?;
        out = out.replace("{api_key}", api_key);
    }
    Ok(out)
}

/// Return the argv for an ACP profile as configured. No injection — the config's
/// `acp_command` is used verbatim. Any CLI flags the agent needs (e.g. `--base-url`)
/// must be explicitly specified in the config file.
pub fn build_profile_command(profile: &AcpClientProfile) -> Vec<String> {
    profile.acp_command.clone()
}

/// Build the subprocess `Command` for an ACP server.
///
/// No `--model` injection — the model is passed via the ACP `session/set_model` call.
/// Any CLI flags the agent needs must be explicitly specified in the config's `acp_command`.
pub fn build_acp_spawn_command(
    command: &[String],
    env: &HashMap<String, String>,
) -> Result<Command> {
    validate_acp_command(command)?;
    let mut cmd = Command::new(&command[0]);
    for arg in &command[1..] {
        cmd.arg(arg);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_command_includes_acp_subcommand() {
        let cmd = default_acp_command();
        validate_acp_command(&cmd).unwrap();
        assert_eq!(cmd[0], "agent");
        assert_eq!(cmd.last().map(String::as_str), Some("acp"));
    }

    #[test]
    fn build_uses_command_verbatim_no_model_injection() {
        // No --model injection — the model is passed via ACP session/set_model.
        let argv = vec![
            "agent-local".into(),
            "--print".into(),
            "--trust".into(),
            "acp".into(),
        ];
        let cmd = build_acp_spawn_command(&argv, &HashMap::new()).unwrap();
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, vec!["--print", "--trust", "acp"]);
    }

    #[test]
    fn build_applies_profile_env() {
        let mut env = HashMap::new();
        env.insert(
            "EXAMPLE_BASE_URL".to_string(),
            "http://example/v1".to_string(),
        );
        env.insert("EXAMPLE_API_KEY".to_string(), "secret".to_string());
        let cmd = build_acp_spawn_command(&default_acp_command(), &env).unwrap();
        assert_eq!(
            cmd.get_envs()
                .find(|(k, _)| *k == "EXAMPLE_BASE_URL")
                .and_then(|(_, v)| v)
                .map(|v| v.to_string_lossy().into_owned()),
            Some("http://example/v1".to_string())
        );
        assert_eq!(
            cmd.get_envs()
                .find(|(k, _)| *k == "EXAMPLE_API_KEY")
                .and_then(|(_, v)| v)
                .map(|v| v.to_string_lossy().into_owned()),
            Some("secret".to_string())
        );
    }

    #[test]
    fn accepts_command_without_acp_subcommand() {
        // After relaxing validation, commands without the "acp" literal are accepted
        // (e.g. ["potlatch", "harness"]).
        validate_acp_command(&["potlatch".into(), "harness".into()]).unwrap();
    }

    #[test]
    fn rejects_empty_executable() {
        let err = validate_acp_command(&["".into(), "acp".into()]).unwrap_err();
        assert!(err.to_string().contains("non-empty"));
    }

    #[test]
    fn parses_acp_client_profiles() {
        let root: Value = toml::from_str(
            r#"
            [acp.cursor-local]
            base_url = "http://prod-model1.example/v1"
            api_key = "EMPTY"
            acp_command = ["agent-local", "--print", "--trust", "--force", "--approve-mcps", "acp"]
            env = [
                "CURSOR_LOCAL_AGENT_BASE_URL={base_url}",
                "CURSOR_LOCAL_AGENT_API_KEY={api_key}",
            ]
            "#,
        )
        .unwrap();
        let profiles = parse_acp_profiles(&root).unwrap();
        let profile = profiles.get("cursor-local").unwrap();
        assert_eq!(
            profile.base_url.as_deref(),
            Some("http://prod-model1.example/v1")
        );
        assert_eq!(profile.api_key.as_deref(), Some("EMPTY"));
        assert_eq!(profile.acp_command[0], "agent-local");
        let env = resolve_profile_env(profile).unwrap();
        assert_eq!(
            env.get("CURSOR_LOCAL_AGENT_BASE_URL").map(String::as_str),
            Some("http://prod-model1.example/v1")
        );
        assert_eq!(
            env.get("CURSOR_LOCAL_AGENT_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn build_profile_command_returns_verbatim() {
        // No injection — the config's acp_command is used as-is.
        // Any CLI flags must be explicitly specified in the config.
        let profile = AcpClientProfile {
            base_url: Some("http://prod-model1.example/v1".into()),
            api_key: Some("EMPTY".into()),
            acp_command: vec![
                "agent-local".into(),
                "--base-url".into(),
                "http://prod-model1.example/v1".into(),
                "--local-agent-api-key".into(),
                "EMPTY".into(),
                "--print".into(),
                "--trust".into(),
                "acp".into(),
            ],
            env: vec![],
        };
        let cmd = build_profile_command(&profile);
        assert_eq!(cmd, profile.acp_command);
    }

    #[test]
    fn env_placeholder_requires_field() {
        let profile = AcpClientProfile {
            base_url: None,
            api_key: None,
            acp_command: default_acp_command(),
            env: vec!["SOME_KEY={base_url}".into()],
        };
        let err = resolve_profile_env(&profile).unwrap_err();
        assert!(err.to_string().contains("base_url"));
    }
}
