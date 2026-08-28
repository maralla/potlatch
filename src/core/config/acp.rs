use std::collections::HashMap;
use std::process::Command;

use anyhow::{Context, Result, bail};
use toml::Value;

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
    pub fields: HashMap<String, String>,
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

/// Structural keys that are not user-defined fields.
const RESERVED_KEYS: &[&str] = &["acp_command", "env", "endpoints"];

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
                let mut fields: HashMap<String, String> = entry_table
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect();
                let model = fields
                    .remove("model")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .with_context(|| "endpoints entry missing required `model` field")?;
                // `model` stays accessible as a {model} reference.
                fields.insert("model".into(), model.clone());
                endpoint_list.push(EndpointEntry { model, fields });
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

/// Pick the `EndpointEntry` whose `model` matches `model_name`.
/// Returns `Ok(None)` when the profile has no `endpoints` table. Returns
/// an error when the profile has endpoints but none match.
fn select_endpoint<'a>(
    profile: &'a AcpClientProfile,
    model_name: &str,
) -> Result<Option<&'a EndpointEntry>> {
    if profile.endpoints.is_empty() {
        return Ok(None);
    }
    profile
        .endpoints
        .iter()
        .find(|e| e.model == model_name)
        .map(Some)
        .with_context(|| {
            format!(
                "no endpoint matching model `{model_name}` in endpoints table (available: {})",
                profile
                    .endpoints
                    .iter()
                    .map(|e| e.model.as_str())
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
        let p = profiles.get("cursor-local").unwrap();
        assert_eq!(
            p.fields.get("base_url").map(String::as_str),
            Some("http://prod-model1.example/v1")
        );
        assert_eq!(p.fields.get("api_key").map(String::as_str), Some("EMPTY"));
        assert_eq!(p.acp_command[0], "agent-local");
        let env = resolve_profile_env(p, None).unwrap();
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
        }
    }

    fn endpoint_profile() -> AcpClientProfile {
        profile(
            vec!["POTLATCH_BASE_URL={endpoint}", "POTLATCH_API_KEY={key}"],
            vec![],
            vec![
                endpoint_entry("model1-fp8", "http://prod-model1.example"),
                endpoint_entry("model2-flash", "http://model2.example"),
            ],
        )
    }

    #[test]
    fn resolve_profile_env_selects_endpoint_by_model_name() {
        let p = endpoint_profile();
        let env = resolve_profile_env(&p, Some("model2-flash")).unwrap();
        assert_eq!(
            env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://model2.example")
        );
        assert_eq!(
            env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
    }

    #[test]
    fn resolve_profile_env_selects_first_endpoint() {
        let p = endpoint_profile();
        let env = resolve_profile_env(&p, Some("model1-fp8")).unwrap();
        assert_eq!(
            env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://prod-model1.example")
        );
    }

    #[test]
    fn resolve_profile_env_errors_on_unmatched_model() {
        let p = endpoint_profile();
        let err = resolve_profile_env(&p, Some("nonexistent-model")).unwrap_err();
        assert!(err.to_string().contains("nonexistent-model"));
        assert!(err.to_string().contains("model1-fp8"));
        assert!(err.to_string().contains("model2-flash"));
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
              { model = "model2-flash", endpoint = "http://model2.example", key = "EMPTY" },
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
        assert_eq!(p.endpoints[0].model, "model2-flash");
        assert_eq!(
            p.endpoints[0].fields.get("endpoint").map(String::as_str),
            Some("http://model2.example")
        );
        assert_eq!(
            p.endpoints[0].fields.get("key").map(String::as_str),
            Some("EMPTY")
        );
        // No `key` field when omitted — it's a user-defined field, not a default.
        assert!(p.endpoints[1].fields.get("key").is_none());
    }

    #[test]
    fn resolve_acp_spawn_picks_endpoint_by_bare_model_name() {
        use super::super::Config;
        let cfg = Config::from_toml_str(
            r#"
            [acp.potlatch]
            acp_command = ["potlatch", "harness"]
            endpoints = [
              { model = "model2-flash", endpoint = "http://model2.example", key = "EMPTY" },
            ]
            env = [
              "POTLATCH_BASE_URL={endpoint}",
              "POTLATCH_API_KEY={key}",
            ]

            [agent.worker]
            model = "acp://potlatch/model2-flash?thinking=true"
            instances = 1
            "#,
        )
        .unwrap();
        let section = cfg.agent("worker").unwrap();
        let spawn = cfg.resolve_acp_spawn(section).unwrap();
        assert_eq!(
            spawn.env.get("POTLATCH_BASE_URL").map(String::as_str),
            Some("http://model2.example")
        );
        assert_eq!(
            spawn.env.get("POTLATCH_API_KEY").map(String::as_str),
            Some("EMPTY")
        );
        assert_eq!(
            spawn.endpoint_model.as_deref(),
            Some("model2-flash?thinking=true")
        );
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
        };
        let p = profile(vec!["KEY={key}"], vec![], vec![entry]);
        let err = resolve_profile_env(&p, Some("test")).unwrap_err();
        assert!(err.to_string().contains("key"));
    }
}
