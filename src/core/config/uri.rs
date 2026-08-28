use anyhow::{Context, Result};

/// Parsed model URI: `<protocol>://<vendor-label>/<model-name>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelUri {
    /// Value as written in config (full URI or legacy bare `<model-name>`).
    pub original: String,
    pub protocol: String,
    pub vendor: String,
    pub model_name: String,
}

impl ModelUri {
    /// Parse `<protocol>://<vendor-label>/<model-name>`.
    ///
    /// Legacy bare ids (no `://`) are treated as `acp://cursor/<model-name>`.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            anyhow::bail!("model URI must not be empty");
        }
        if let Some(rest) = s.strip_prefix("acp://") {
            return Self::parse_vendor_model(s, "acp", rest);
        }
        if s.contains("://") {
            let (protocol, rest) = s.split_once("://").context("invalid model URI")?;
            return Self::parse_vendor_model(s, protocol, rest);
        }
        Ok(Self {
            original: s.to_string(),
            protocol: "acp".to_string(),
            vendor: "cursor".to_string(),
            model_name: s.to_string(),
        })
    }

    fn parse_vendor_model(original: &str, protocol: &str, rest: &str) -> Result<Self> {
        let (vendor, model_name) = rest.split_once('/').with_context(|| {
            format!(
                "model URI must be <protocol>://<vendor-label>/<model-name>, got `{protocol}://{rest}`"
            )
        })?;
        if vendor.is_empty() || model_name.is_empty() {
            anyhow::bail!("model URI vendor-label and model-name must be non-empty");
        }
        Ok(Self {
            original: original.to_string(),
            protocol: protocol.to_string(),
            vendor: vendor.to_string(),
            model_name: model_name.to_string(),
        })
    }

    /// Model URI exactly as configured.
    pub fn as_configured(&self) -> &str {
        &self.original
    }

    /// `<model-name>` segment passed to the ACP CLI and inference endpoint.
    pub fn endpoint_model_name(&self) -> &str {
        &self.model_name
    }

    /// `<model-name>` with any `?query` suffix stripped, for matching against
    /// the `endpoints` table of an ACP profile (e.g. `model1-fp8` from
    /// `acp://potlatch/model1-fp8?thinking=true`).
    pub fn bare_model_name(&self) -> &str {
        self.model_name
            .split('?')
            .next()
            .unwrap_or(&self.model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_uri() {
        let u = ModelUri::parse("acp://cursor/composer-2").unwrap();
        assert_eq!(u.as_configured(), "acp://cursor/composer-2");
        assert_eq!(u.protocol, "acp");
        assert_eq!(u.vendor, "cursor");
        assert_eq!(u.model_name, "composer-2");
        assert_eq!(u.endpoint_model_name(), "composer-2");
    }

    #[test]
    fn endpoint_model_name_from_uri() {
        let u = ModelUri::parse("acp://cursor/model1-fp8").unwrap();
        assert_eq!(u.as_configured(), "acp://cursor/model1-fp8");
        assert_eq!(u.endpoint_model_name(), "model1-fp8");
    }

    #[test]
    fn parse_legacy_bare_model_name() {
        let u = ModelUri::parse("gpt-5.3-codex").unwrap();
        assert_eq!(u.as_configured(), "gpt-5.3-codex");
        assert_eq!(u.protocol, "acp");
        assert_eq!(u.vendor, "cursor");
        assert_eq!(u.model_name, "gpt-5.3-codex");
        assert_eq!(u.endpoint_model_name(), "gpt-5.3-codex");
    }

    #[test]
    fn bare_model_name_strips_query_suffix() {
        let u = ModelUri::parse("acp://potlatch/model1-fp8?thinking=true").unwrap();
        assert_eq!(u.model_name, "model1-fp8?thinking=true");
        assert_eq!(u.endpoint_model_name(), "model1-fp8?thinking=true");
        assert_eq!(u.bare_model_name(), "model1-fp8");
    }

    #[test]
    fn bare_model_name_without_query() {
        let u = ModelUri::parse("acp://cursor/composer-2").unwrap();
        assert_eq!(u.bare_model_name(), "composer-2");
    }
}
