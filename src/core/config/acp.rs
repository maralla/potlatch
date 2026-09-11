use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use toml::Value;

const RESERVED_KEYS: &[&str] = &["acp_command", "env", "endpoints"];

/// The ACP profile name of the platform's own harness: the app name itself
/// (`[acp.<app-name>]`). Subagent children for this vendor spawn the running
/// executable and take their endpoint env/auth from this profile.
pub const POTLATCH_ACP_PROFILE: &str = crate::paths::APP_NAME;

/// The `endpoints`-entry key holding the auth-provider command.
const AUTH_PROVIDER_KEY: &str = "auth_provider";

/// A parsed `[acp.<name>]` profile. Structural fields (`acp_command`, `env`,
/// `endpoints`) are stored separately; every other key in the section is a
/// user-defined field available as a `{field}` reference in `env` values.
#[derive(Debug, Clone)]
pub struct AcpClientProfile {
    pub acp_command: Vec<String>,
    pub env: Vec<String>,
    pub endpoints: Vec<EndpointEntry>,
    /// User-defined fields from the `[acp.*]` section top level (e.g.
    /// `base_url`, `api_key`, or anything else), available as `{field}`
    /// references when no endpoint is selected.
    pub fields: HashMap<String, String>,
}

/// A single `endpoints` entry. The `model` field selects which entry is
/// used; every field (including `model`) is available as a `{field}`
/// reference in `env` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointEntry {
    pub model: String,
    /// Optional alternative name for addressing this entry from a model URI
    /// (`acp://<vendor>/<alias>`). Lets one model be exposed under several
    /// names — possibly through different endpoints: two entries may share
    /// the same `model` as long as each carries a distinct alias. The
    /// backend still receives the real `model` name; the alias is a routing
    /// key only. An entry with an alias is addressed **only** by the alias —
    /// its `model` name stops being an address. Must be unique within the
    /// profile.
    pub alias: Option<String>,
    pub fields: HashMap<String, String>,
    /// `auth_provider` command argv. When set, the harness runs this command
    /// before talking to the endpoint and applies the returned headers
    /// (`{"expiration": <unix-seconds>, "headers": {..}}`) to its requests,
    /// cached until the expiration.
    pub auth_command: Option<Vec<String>>,
}

impl EndpointEntry {
    /// The model name to send to the backend for a request addressed by
    /// `requested` (the URI's bare model segment): an alias resolves to the
    /// entry's real `model` name; a direct model match passes through.
    pub fn wire_model_name<'a>(&'a self, requested: &'a str) -> &'a str {
        match &self.alias {
            Some(alias) if alias == requested => &self.model,
            _ => requested,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AcpSpawnConfig {
    pub command: Vec<String>,
    /// Configured model URI (e.g. `acp://cursor/model1`).
    pub model_uri: Option<String>,
    /// Parsed `<model-name>` for `session/set_model` after `session/new`.
    pub endpoint_model: Option<String>,
    pub env: HashMap<String, String>,
    /// `auth_provider` argv for the selected endpoint, forwarded to the
    /// harness via the parent's env. `None` when the endpoint has no
    /// auth provider.
    pub auth_command: Option<Vec<String>>,
    /// Directory containing the potlatch config file. The harness runs the
    /// auth-provider command from here, so `./auth-tool.py` or `auth-tool.py` resolve
    /// relative to the config, not the agent's repo checkout.
    pub config_dir: Option<PathBuf>,
    /// Alias → real model name for the vendor profile's endpoint entries.
    /// Consumers that forward model names to the backend (the subagent hub's
    /// `session/set_model`) rewrite an aliased request to the real name.
    pub model_aliases: HashMap<String, String>,
}

pub fn parse_acp_profiles(root: &Value) -> Result<HashMap<String, AcpClientProfile>> {
    let Some(acp_root) = root.get("acp").and_then(Value::as_table) else {
        return Ok(HashMap::new());
    };

    let mut profiles = HashMap::new();
    for (name, value) in acp_root {
        let section = value
            .as_table()
            .with_context(|| format!("[acp.{name}] must be a table"))?;

        let acp_command: Vec<String> = section
            .get("acp_command")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .with_context(|| format!("[acp.{name}] missing `acp_command`"))?;
        validate_acp_command(&acp_command)
            .with_context(|| format!("invalid acp_command in [acp.{name}]"))?;

        let env: Vec<String> = section
            .get("env")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut endpoint_list = Vec::new();
        if let Some(arr) = section.get("endpoints").and_then(Value::as_array) {
            for entry_val in arr {
                let entry_table = entry_val
                    .as_table()
                    .with_context(|| "endpoints entry must be a table")?;
                let auth_command = match entry_table.get(AUTH_PROVIDER_KEY) {
                    Some(Value::String(raw)) => Some(split_command(raw).with_context(
                        || "invalid `auth_provider` in endpoints entry (model `{model}`)",
                    )?),
                    Some(_) => bail!("`auth_provider` in an endpoints entry must be a string"),
                    None => None,
                };
                let mut fields: HashMap<String, String> = entry_table
                    .iter()
                    .filter(|(k, _)| k.as_str() != AUTH_PROVIDER_KEY)
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect();
                let model = fields
                    .remove("model")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .with_context(|| "endpoints entry missing required `model` field")?;
                // `model` stays accessible as a {model} reference.
                fields.insert("model".into(), model.clone());
                let alias = match fields.remove("alias") {
                    Some(raw) => {
                        let trimmed = raw.trim().to_string();
                        if trimmed.is_empty() {
                            bail!("endpoints entry `{model}` has an empty `alias`");
                        }
                        Some(trimmed)
                    }
                    None => None,
                };
                // `alias` stays accessible as an {alias} reference.
                if let Some(alias) = &alias {
                    fields.insert("alias".into(), alias.clone());
                }
                endpoint_list.push(EndpointEntry {
                    model,
                    alias,
                    fields,
                    auth_command,
                });
            }

            // Addressable names must be unique: an entry with an alias is
            // addressed by the alias (its model name is no longer an
            // address — that is what lets one model be exposed several times
            // through different endpoints); an unaliased entry is addressed
            // by its model name.
            let mut addressable: HashMap<String, String> = HashMap::new();
            for entry in &endpoint_list {
                let name = entry.alias.as_ref().unwrap_or(&entry.model);
                if let Some(previous) = addressable.insert(name.clone(), entry.model.clone()) {
                    bail!(
                        "duplicate endpoint name `{name}` in [acp.{name}] endpoints (two entries \
                         resolve to `{previous}` and `{}`) — aliases and model names must be unique",
                        entry.model
                    );
                }
            }
        }

        // Every non-structural key becomes a user-defined field.
        let fields: HashMap<String, String> = section
            .iter()
            .filter(|(k, _)| !RESERVED_KEYS.contains(&k.as_str()))
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();

        let profile = AcpClientProfile {
            acp_command,
            env,
            endpoints: endpoint_list,
            fields,
        };

        resolve_profile_env(&profile, None)
            .with_context(|| format!("invalid env in [acp.{name}]"))?;

        profiles.insert(name.clone(), profile);
    }

    Ok(profiles)
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

/// Split a shell-like command string into argv following POSIX shell word
/// rules ([`shell_words::split`]): whitespace separation with single/double
/// quoting and backslash escapes, no variable or command expansion. The
/// result must be an executable plus arguments.
pub fn split_command(raw: &str) -> Result<Vec<String>> {
    let argv: Vec<String> =
        shell_words::split(raw).map_err(|e| anyhow::anyhow!("invalid command `{raw}`: {e}"))?;
    if argv.is_empty() {
        bail!("command must not be empty");
    }
    if argv[0].trim().is_empty() {
        bail!("command executable must be non-empty");
    }
    Ok(argv)
}

/// Resolve `env` entries (`KEY=value`) with `{field}` references.
///
/// Each `{name}` in an env value is a field reference. When a model name is
/// provided, the matching `endpoints` entry supplies the field values. When
/// no model name is given (e.g. config-load validation), placeholders are
/// looked up against the profile-level fields only; endpoint-entry fields
/// are not available until an endpoint is selected.
pub fn resolve_profile_env(
    profile: &AcpClientProfile,
    model_name: Option<&str>,
) -> Result<HashMap<String, String>> {
    let selected = match model_name {
        Some(name) => select_endpoint(profile, name)?,
        None => None,
    };

    let mut env = HashMap::new();
    for entry in &profile.env {
        let (key, value) = parse_env_entry(entry)?;
        let value = substitute_placeholders(&value, &profile.fields, selected)?;
        if env.insert(key.clone(), value).is_some() {
            bail!("duplicate env key `{key}` in ACP profile");
        }
    }
    Ok(env)
}

/// Pick the `EndpointEntry` addressed by `model_name`: an entry with an
/// alias is addressed only by that alias (its real model name is not an
/// address — that is what lets one model be exposed several times through
/// different endpoints); an unaliased entry by its model name. Returns
/// `Ok(None)` when the profile has no `endpoints` table. Returns an error
/// when the profile has endpoints but none match.
pub(crate) fn select_endpoint<'a>(
    profile: &'a AcpClientProfile,
    model_name: &str,
) -> Result<Option<&'a EndpointEntry>> {
    if profile.endpoints.is_empty() {
        return Ok(None);
    }
    profile
        .endpoints
        .iter()
        .find(|e| e.alias.as_ref().unwrap_or(&e.model).as_str() == model_name)
        .map(Some)
        .with_context(|| {
            format!(
                "no endpoint matching model `{model_name}` in endpoints table (available: {})",
                profile
                    .endpoints
                    .iter()
                    .map(|e| e.alias.as_deref().unwrap_or(e.model.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
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

/// Substitute every `{name}` placeholder in `value`. Field values are looked
/// up by name: the selected endpoint entry (if any) is checked first, then
/// the profile-level fields. When an endpoint is selected but the field
/// doesn't exist on it, that's an error — no fallback to profile-level
/// fields. When no endpoint is selected, unresolved placeholders are left
/// as-is (deferred to spawn time).
fn substitute_placeholders(
    value: &str,
    profile_fields: &HashMap<String, String>,
    selected: Option<&EndpointEntry>,
) -> Result<String> {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;

    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        rest = &rest[open..];
        let Some(close) = rest.find('}') else {
            out.push_str(rest);
            return Ok(out);
        };
        let name = &rest[1..close];
        rest = &rest[close + 1..];

        if let Some(entry) = selected {
            if let Some(v) = entry.fields.get(name) {
                out.push_str(v);
                continue;
            }
            bail!(
                "placeholder {{{name}}} not found on endpoint `{}`",
                entry.model
            );
        }

        if let Some(v) = profile_fields.get(name) {
            out.push_str(v);
            continue;
        }

        out.push('{');
        out.push_str(name);
        out.push('}');
    }

    out.push_str(rest);
    Ok(out)
}

/// Return the argv for an ACP profile as configured. No injection — the config's
/// `acp_command` is used verbatim. Any CLI flags the agent needs (e.g. `--base-url`)
/// must be explicitly specified in the config file.
pub fn build_profile_command(profile: &AcpClientProfile) -> Vec<String> {
    profile.acp_command.clone()
}

impl AcpClientProfile {
    /// Alias → real model name for every endpoint entry that declares one.
    /// Consumers that forward model names to a backend (e.g. the subagent
    /// hub's `session/set_model`) use this to rewrite an aliased request to
    /// the entry's real model name.
    pub fn model_aliases(&self) -> HashMap<String, String> {
        self.endpoints
            .iter()
            .filter_map(|e| e.alias.as_ref().map(|a| (a.clone(), e.model.clone())))
            .collect()
    }
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
    use crate::core::config::Config;

    fn test_command() -> Vec<String> {
        vec!["agent".into(), "acp".into()]
    }

    fn profile(
        env: Vec<&str>,
        fields: Vec<(&str, &str)>,
        endpoints: Vec<EndpointEntry>,
    ) -> AcpClientProfile {
        AcpClientProfile {
            acp_command: test_command(),
            env: env.into_iter().map(String::from).collect(),
            fields: fields
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            endpoints,
        }
    }

    #[test]
    fn build_uses_command_verbatim_no_model_injection() {
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
        let cmd = build_acp_spawn_command(&test_command(), &env).unwrap();
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
            base_url = "http://endpoint1.example/v1"
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
        let p = profiles.get("cursor-local").unwrap();
        assert_eq!(
            p.fields.get("base_url").map(String::as_str),
            Some("http://endpoint1.example/v1")
        );
        assert_eq!(p.fields.get("api_key").map(String::as_str), Some("EMPTY"));
        assert_eq!(p.acp_command[0], "agent-local");
        let env = resolve_profile_env(p, None).unwrap();
        assert_eq!(
            env.get("CURSOR_LOCAL_AGENT_BASE_URL").map(String::as_str),
            Some("http://endpoint1.example/v1")
        );
        assert_eq!(
            env.get("CURSOR_LOCAL_AGENT_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn build_profile_command_returns_verbatim() {
        let p = profile(vec![], vec![], vec![]);
        let cmd = build_profile_command(&p);
        assert_eq!(cmd, p.acp_command);
    }

    #[test]
    fn profile_fields_resolve_without_endpoints() {
        let p = profile(
            vec!["URL={base_url}", "KEY={api_key}"],
            vec![("base_url", "http://prod.example"), ("api_key", "SECRET")],
            vec![],
        );
        let env = resolve_profile_env(&p, None).unwrap();
        assert_eq!(
            env.get("URL").map(String::as_str),
            Some("http://prod.example")
        );
        assert_eq!(env.get("KEY").map(String::as_str), Some("SECRET"));
    }

    fn endpoint_entry(model: &str, endpoint: &str) -> EndpointEntry {
        EndpointEntry {
            model: model.into(),
            fields: HashMap::from([
                ("endpoint".into(), endpoint.into()),
                ("key".into(), "EMPTY".into()),
                ("model".into(), model.into()),
            ]),
            alias: None,
            auth_command: None,
        }
    }

    fn endpoint_entry_with_alias(model: &str, alias: &str, endpoint: &str) -> EndpointEntry {
        EndpointEntry {
            alias: Some(alias.into()),
            ..endpoint_entry(model, endpoint)
        }
    }

    #[test]
    fn parse_endpoints_entry_with_alias() {
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", alias = "model-x", endpoint = "http://endpoint1.example" },
            ]
            "#,
        )
        .unwrap();
        let p = &parse_acp_profiles(&root).unwrap()["potlatch"];
        assert_eq!(p.endpoints[0].alias.as_deref(), Some("model-x"));
    }

    #[test]
    fn parse_endpoints_rejects_duplicate_names_and_aliases() {
        // Two entries sharing one alias collide.
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", alias = "shared", endpoint = "http://a.example" },
              { model = "model2", alias = "shared", endpoint = "http://b.example" },
            ]
            "#,
        )
        .unwrap();
        let err = parse_acp_profiles(&root).unwrap_err();
        assert!(
            err.to_string().contains("duplicate endpoint name `shared`"),
            "{err:#}"
        );

        // An alias colliding with an unaliased entry's model name is the
        // same ambiguity.
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", endpoint = "http://a.example" },
              { model = "model2", alias = "model1", endpoint = "http://b.example" },
            ]
            "#,
        )
        .unwrap();
        let err = parse_acp_profiles(&root).unwrap_err();
        assert!(
            err.to_string().contains("duplicate endpoint name `model1`"),
            "{err:#}"
        );
    }

    #[test]
    fn select_endpoint_matches_by_alias() {
        let p = profile(
            vec![],
            vec![],
            vec![
                endpoint_entry("model1", "http://endpoint1.example"),
                endpoint_entry_with_alias("model1", "model-x", "http://endpoint2.example"),
            ],
        );
        assert_eq!(
            select_endpoint(&p, "model-x").unwrap().unwrap().fields["endpoint"],
            "http://endpoint2.example"
        );
        // Plain model names still match directly.
        assert_eq!(
            select_endpoint(&p, "model1").unwrap().unwrap().fields["endpoint"],
            "http://endpoint1.example"
        );
        // The error lists the addressable names.
        let err = select_endpoint(&p, "nope").unwrap_err();
        assert!(err.to_string().contains("model1, model-x"), "{err:#}");
    }

    #[test]
    fn wire_model_name_rewrites_alias_to_real_model() {
        let entry = endpoint_entry_with_alias("model1", "model-x", "http://x.example");
        assert_eq!(entry.wire_model_name("model-x"), "model1");
        assert_eq!(entry.wire_model_name("model1"), "model1");
        assert_eq!(entry.wire_model_name("other"), "other");
    }

    fn endpoint_profile() -> AcpClientProfile {
        profile(
            vec!["POTLATCH_BASE_URL={endpoint}", "POTLATCH_API_KEY={key}"],
            vec![],
            vec![
                endpoint_entry("model1", "http://endpoint1.example"),
                endpoint_entry("model2", "http://endpoint2.example"),
            ],
        )
    }

    #[test]
    fn resolve_profile_env_selects_endpoint_by_model_name() {
        let p = endpoint_profile();
        let env = resolve_profile_env(&p, Some("model2")).unwrap();
        assert_eq!(
            env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint2.example")
        );
        assert_eq!(
            env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn resolve_profile_env_selects_first_endpoint() {
        let p = endpoint_profile();
        let env = resolve_profile_env(&p, Some("model1")).unwrap();
        assert_eq!(
            env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint1.example")
        );
    }

    #[test]
    fn resolve_profile_env_errors_on_unmatched_model() {
        let p = endpoint_profile();
        let err = resolve_profile_env(&p, Some("nonexistent-model")).unwrap_err();
        assert!(err.to_string().contains("nonexistent-model"));
        assert!(err.to_string().contains("model1"));
        assert!(err.to_string().contains("model2"));
    }

    #[test]
    fn resolve_profile_env_leaves_placeholders_without_model() {
        let p = endpoint_profile();
        let env = resolve_profile_env(&p, None).unwrap();
        assert_eq!(
            env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("{endpoint}")
        );
        assert_eq!(
            env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("{key}")
        );
    }

    #[test]
    fn resolve_profile_env_resolves_profile_fields_without_endpoints() {
        let p = profile(
            vec!["ONLY_BASE={base_url}"],
            vec![("base_url", "http://fallback.example")],
            vec![],
        );
        let env = resolve_profile_env(&p, None).unwrap();
        assert_eq!(
            env.get("ONLY_BASE").map(String::as_str),
            Some("http://fallback.example")
        );
    }

    #[test]
    fn parse_acp_profile_with_endpoints() {
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model2", endpoint = "http://endpoint2.example", key = "EMPTY" },
              { model = "deepseek-v4-flash", endpoint = "http://deepseek-flash.example" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]
            "#,
        )
        .unwrap();
        let profiles = parse_acp_profiles(&root).unwrap();
        let p = profiles.get("potlatch").unwrap();
        assert_eq!(p.endpoints.len(), 2);
        assert_eq!(p.endpoints[0].model, "model2");
        assert_eq!(
            p.endpoints[0].fields.get("endpoint").map(String::as_str),
            Some("http://endpoint2.example")
        );
        assert_eq!(
            p.endpoints[0].fields.get("key").map(String::as_str),
            Some("EMPTY")
        );
        // No `key` field when omitted — it's a user-defined field, not a default.
        assert!(!p.endpoints[1].fields.contains_key("key"));
    }

    fn harness_config() -> Config {
        Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", endpoint = "http://endpoint1.example", key = "EMPTY" },
              { model = "model2", endpoint = "http://endpoint2.example", key = "EMPTY" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]
            "#,
        )
        .unwrap()
    }

    #[test]
    fn resolve_client_spawn_selects_the_endpoint_of_the_provided_models() {
        // The provided models pick the endpoint whose env/auth the shared
        // ACP child is spawned with. Query suffixes are stripped before
        // matching.
        let spawn = harness_config()
            .resolve_client_spawn(
                "potlatch",
                &["acp://potlatch/model2?effort=high".to_string()],
            )
            .unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint2.example")
        );
        assert_eq!(
            spawn.env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
        // The platform's own harness vendor runs THIS executable.
        let exe = std::env::current_exe().unwrap();
        assert_eq!(spawn.command[0], exe.display().to_string());
        assert_eq!(spawn.command[1], "harness");
        assert_eq!(spawn.model_uri, None);
    }

    #[test]
    fn resolve_client_spawn_accepts_models_sharing_one_endpoint() {
        // Two provided models whose bare names resolve to the same endpoint
        // are fine.
        let spawn = harness_config()
            .resolve_client_spawn(
                "potlatch",
                &[
                    "acp://potlatch/model1".to_string(),
                    "acp://potlatch/model1?effort=high".to_string(),
                ],
            )
            .unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint1.example")
        );
    }

    #[test]
    fn resolve_client_spawn_rejects_models_spanning_endpoints() {
        // One ACP child serves one endpoint: models from different endpoints
        // of the same vendor cannot share it.
        let error = harness_config()
            .resolve_client_spawn(
                "potlatch",
                &[
                    "acp://potlatch/model1".to_string(),
                    "acp://potlatch/model2".to_string(),
                ],
            )
            .unwrap_err();
        assert!(error.to_string().contains("one ACP endpoint"), "{error}");
    }

    #[test]
    fn resolve_client_spawn_rejects_unknown_provided_models() {
        let error = harness_config()
            .resolve_client_spawn("potlatch", &["acp://potlatch/no-such-model".to_string()])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no endpoint matching model `no-such-model`"),
            "{error}"
        );
    }

    #[test]
    fn resolve_client_spawn_requires_the_vendor_profile() {
        let cfg = Config::from_toml_str("").unwrap();
        let error = cfg
            .resolve_client_spawn("potlatch", &["model1".to_string()])
            .unwrap_err();
        assert!(error.to_string().contains("[acp.potlatch]"), "{error}");
    }

    #[test]
    fn resolve_client_spawn_uses_the_profile_command_for_other_vendors() {
        // Non-harness vendors run their profile's acp_command (e.g. a CLI).
        let cfg = Config::from_toml_str(
            r#"
            [acp.cursor]
            acp_command = ["agent", "acp"]
            "#,
        )
        .unwrap();
        let spawn = cfg
            .resolve_client_spawn("cursor", &["composer-2".to_string()])
            .unwrap();
        assert_eq!(spawn.command, vec!["agent", "acp"]);
    }

    #[test]
    fn resolve_client_spawn_handles_full_uris_against_a_multi_endpoint_profile() {
        // Regression: provided_models are full model URIs — the endpoint
        // lookup must use the bare model name, not the whole URI string.
        // Several endpoints, one provided model with a query suffix.
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model-a", endpoint = "http://model-a.example", key = "EMPTY" },
              { model = "model-b", endpoint = "http://model-b.example", key = "EMPTY" },
              { model = "model-c", endpoint = "http://model-c.example", key = "EMPTY" },
              { model = "model-d", endpoint = "http://model-d.example", key = "EMPTY" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]
            "#,
        )
        .unwrap();
        let spawn = cfg
            .resolve_client_spawn(
                "potlatch",
                &["acp://potlatch/model-b?effort=high".to_string()],
            )
            .unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://model-b.example")
        );
        assert_eq!(
            spawn.env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn resolve_client_spawn_resolves_profile_fields_without_endpoints() {
        // Single-endpoint style: top-level fields, no `endpoints` table —
        // nothing to select, placeholders resolve from profile fields.
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            base_url = "http://endpoint1.example"
            api_key = "EMPTY"
            env = [
              "POTLATCH_BASE_URL={base_url}",
              "POTLATCH_API_KEY={api_key}",
            ]
            "#,
        )
        .unwrap();
        let spawn = cfg
            .resolve_client_spawn(
                "potlatch",
                &["acp://potlatch/model1?effort=high".to_string()],
            )
            .unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint1.example")
        );
    }

    #[test]
    fn resolve_acp_spawn_picks_endpoint_by_bare_model_name() {
        use super::super::Config;
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model2", endpoint = "http://endpoint2.example", key = "EMPTY" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]

            [agent.worker]
            model = "acp://potlatch/model2?effort=high"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint2.example")
        );
        assert_eq!(
            spawn.env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
        assert_eq!(spawn.endpoint_model.as_deref(), Some("model2?effort=high"));
    }

    #[test]
    fn resolve_acp_spawn_rewrites_aliased_model_to_real_name() {
        use super::super::Config;
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", alias = "model-x", endpoint = "http://endpoint-x.example", key = "EMPTY" },
              { model = "model1", alias = "model-y", endpoint = "http://endpoint-y.example", key = "EMPTY" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]

            [agent.x]
            model = "acp://potlatch/model-x?effort=high"
            instances = 1

            [agent.y]
            model = "acp://potlatch/model-y"
            instances = 1
            "#,
        )
        .unwrap();

        let x = cfg.resolve_acp_spawn(cfg.agent("x").unwrap()).unwrap();
        // The alias selects its endpoint, and the backend receives the real
        // model name with the caller's query preserved.
        assert_eq!(x.endpoint_model.as_deref(), Some("model1?effort=high"));
        assert_eq!(
            x.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint-x.example")
        );

        let y = cfg.resolve_acp_spawn(cfg.agent("y").unwrap()).unwrap();
        assert_eq!(y.endpoint_model.as_deref(), Some("model1"));
        assert_eq!(
            y.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://endpoint-y.example")
        );

        // The alias map is published for consumers that forward model names
        // (the subagent hub's session/set_model).
        assert_eq!(
            x.model_aliases.get("model-x").map(String::as_str),
            Some("model1")
        );

        // An aliased entry is addressed only by its alias: the model name
        // stops being an address (its whole point is disambiguating several
        // entries that share one model).
        let cfg2 = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", alias = "model-x", endpoint = "http://endpoint-x.example", key = "EMPTY" },
              { model = "model2", endpoint = "http://endpoint2.example", key = "EMPTY" },
            ]
            env = ["POTLATCH_BASE_URL={endpoint}"]

            [agent.via_alias]
            model = "acp://potlatch/model-x"
            instances = 1

            [agent.direct]
            model = "acp://potlatch/model2"
            instances = 1
            "#,
        )
        .unwrap();
        let via_alias = cfg2
            .resolve_acp_spawn(cfg2.agent("via_alias").unwrap())
            .unwrap();
        assert_eq!(via_alias.endpoint_model.as_deref(), Some("model1"));
        let direct = cfg2
            .resolve_acp_spawn(cfg2.agent("direct").unwrap())
            .unwrap();
        assert_eq!(direct.endpoint_model.as_deref(), Some("model2"));
        // Addressing the aliased entry by its model name is refused — the
        // name is no longer an address; the error says what is.
        let cfg3 = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", alias = "model-x", endpoint = "http://endpoint-x.example", key = "EMPTY" },
            ]
            env = ["POTLATCH_BASE_URL={endpoint}"]

            [agent.by_model]
            model = "acp://potlatch/model1"
            instances = 1
            "#,
        )
        .unwrap();
        let err = cfg3
            .resolve_acp_spawn(cfg3.agent("by_model").unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("available: model-x"), "{err:#}");
    }

    #[test]
    fn resolve_acp_spawn_forwards_auth_provider() {
        use super::super::Config;
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", endpoint = "http://endpoint1.example", auth_provider = "your-auth-command --login" },
              { model = "model2", endpoint = "http://endpoint2.example" },
            ]

            [agent.worker]
            model = "acp://potlatch/model1"
            [agent.flash]
            model = "acp://potlatch/model2"
            "#,
        )
        .unwrap();
        let worker = cfg.resolve_acp_spawn(cfg.agent("worker").unwrap()).unwrap();
        assert_eq!(
            worker.auth_command.as_deref(),
            Some(&["your-auth-command".to_string(), "--login".to_string()][..])
        );
        let flash = cfg.resolve_acp_spawn(cfg.agent("flash").unwrap()).unwrap();
        assert!(flash.auth_command.is_none());
    }

    #[test]
    fn substitute_resolves_arbitrary_field_names() {
        let entry = EndpointEntry {
            model: "test".into(),
            fields: HashMap::from([
                ("ep".into(), "http://custom:9999".into()),
                ("secret".into(), "abc123".into()),
                ("model".into(), "test".into()),
            ]),
            alias: None,
            auth_command: None,
        };
        let p = profile(
            vec!["URL={ep}", "TOKEN={secret}", "MODEL={model}"],
            vec![],
            vec![entry],
        );
        let env = resolve_profile_env(&p, Some("test")).unwrap();
        assert_eq!(
            env.get("URL").map(String::as_str),
            Some("http://custom:9999")
        );
        assert_eq!(env.get("TOKEN").map(String::as_str), Some("abc123"));
        assert_eq!(env.get("MODEL").map(String::as_str), Some("test"));
    }

    #[test]
    fn substitute_errors_on_missing_field_when_endpoint_selected() {
        let entry = EndpointEntry {
            model: "test".into(),
            fields: HashMap::from([("endpoint".into(), "http://x".into())]),
            alias: None,
            auth_command: None,
        };
        let p = profile(vec!["X={nonexistent}"], vec![], vec![entry]);
        let err = resolve_profile_env(&p, Some("test")).unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn substitute_handles_multiple_placeholders_in_one_value() {
        let entry = EndpointEntry {
            model: "test".into(),
            fields: HashMap::from([
                ("endpoint".into(), "http://x".into()),
                ("key".into(), "K".into()),
            ]),
            alias: None,
            auth_command: None,
        };
        let p = profile(vec!["URL={endpoint}?token={key}"], vec![], vec![entry]);
        let env = resolve_profile_env(&p, Some("test")).unwrap();
        assert_eq!(env.get("URL").map(String::as_str), Some("http://x?token=K"));
    }

    #[test]
    fn substitute_leaves_unmatched_braces_intact() {
        let p = profile(vec!["JSON={\"a\": 1}"], vec![], vec![]);
        let env = resolve_profile_env(&p, None).unwrap();
        assert_eq!(env.get("JSON").map(String::as_str), Some("{\"a\": 1}"));
    }

    #[test]
    fn custom_profile_fields_are_referenceable() {
        // Any non-structural top-level key is a field, not just base_url/api_key.
        let root: Value = toml::from_str(
            r#"
            [acp.my-provider]
            acp_command = ["my-agent"]
            my_custom_field = "hello"
            another = "world"
            env = [
              "GREETING={my_custom_field}",
              "PLACE={another}",
            ]
            "#,
        )
        .unwrap();
        let profiles = parse_acp_profiles(&root).unwrap();
        let p = profiles.get("my-provider").unwrap();
        let env = resolve_profile_env(p, None).unwrap();
        assert_eq!(env.get("GREETING").map(String::as_str), Some("hello"));
        assert_eq!(env.get("PLACE").map(String::as_str), Some("world"));
    }

    #[test]
    fn missing_field_on_endpoint_errors_when_selected() {
        // `key` is not special — if the endpoint doesn't define it, {key} errors.
        let entry = EndpointEntry {
            model: "test".into(),
            fields: HashMap::from([("endpoint".into(), "http://x".into())]),
            alias: None,
            auth_command: None,
        };
        let p = profile(vec!["KEY={key}"], vec![], vec![entry]);
        let err = resolve_profile_env(&p, Some("test")).unwrap_err();
        assert!(err.to_string().contains("key"));
    }

    #[test]
    fn parse_endpoint_with_auth_provider() {
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", endpoint = "http://endpoint1.example", auth_provider = "auth helper --login --profile work" },
              { model = "model2", endpoint = "http://endpoint2.example" },
            ]
            "#,
        )
        .unwrap();
        let profiles = parse_acp_profiles(&root).unwrap();
        let p = profiles.get("potlatch").unwrap();
        assert_eq!(
            p.endpoints[0].auth_command.as_deref(),
            Some(
                &[
                    "auth".to_string(),
                    "helper".to_string(),
                    "--login".to_string(),
                    "--profile".to_string(),
                    "work".to_string()
                ][..]
            )
        );
        assert!(p.endpoints[1].auth_command.is_none());
    }

    #[test]
    fn auth_provider_not_exposed_as_field() {
        // `auth_provider` is structural: it must not leak into the {field}
        // namespace used for env interpolation.
        let root: Value = toml::from_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model1", endpoint = "http://endpoint1.example", auth_provider = "auth helper" },
            ]
            env = ["X={auth_provider}"]
            "#,
        )
        .unwrap();
        let profiles = parse_acp_profiles(&root).unwrap();
        let p = profiles.get("potlatch").unwrap();
        assert!(!p.endpoints[0].fields.contains_key("auth_provider"));
        // Referencing it as a placeholder is an error when the endpoint is
        // selected (it's not a field).
        assert!(resolve_profile_env(p, Some("model1")).is_err());
    }

    #[test]
    fn split_command_handles_quoting() {
        let argv = split_command("auth 'run helper' --model \"model 1\"").unwrap();
        assert_eq!(
            argv,
            vec![
                "auth".to_string(),
                "run helper".to_string(),
                "--model".to_string(),
                "model 1".to_string(),
            ]
        );
        // POSIX backslash escapes (a difference from the previous hand-rolled
        // splitter, which treated backslashes literally). Inside double
        // quotes only certain escapes are special, so a literal apostrophe
        // needs no escape at all.
        let argv = split_command("auth helper --greeting \"a'b\"").unwrap();
        assert_eq!(argv[2], "--greeting");
        assert_eq!(argv[3], "a'b");
        // Empty command is rejected.
        assert!(split_command("   ").is_err());
        // Unterminated quote is rejected.
        assert!(split_command("auth 'helper").is_err());
    }
}
