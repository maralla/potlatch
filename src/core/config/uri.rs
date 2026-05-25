use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelUri {
    pub scheme: String,
    pub name: String,
    pub model: String,
}

impl ModelUri {
    /// Parse `scheme://name/model` or legacy bare model id (treated as `acp://cursor/<id>`).
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            anyhow::bail!("model URI must not be empty");
        }
        if let Some(rest) = s.strip_prefix("acp://") {
            return Self::parse_slash_pair("acp", rest);
        }
        if s.contains("://") {
            let (scheme, rest) = s.split_once("://").context("invalid model URI")?;
            return Self::parse_slash_pair(scheme, rest);
        }
        Ok(Self {
            scheme: "acp".to_string(),
            name: "cursor".to_string(),
            model: s.to_string(),
        })
    }

    fn parse_slash_pair(scheme: &str, rest: &str) -> Result<Self> {
        let (name, model) = rest.split_once('/').with_context(|| {
            format!("model URI missing provider/model segment: {scheme}://{rest}")
        })?;
        if name.is_empty() || model.is_empty() {
            anyhow::bail!("model URI provider and model must be non-empty");
        }
        Ok(Self {
            scheme: scheme.to_string(),
            name: name.to_string(),
            model: model.to_string(),
        })
    }

    pub fn bare_model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_uri() {
        let u = ModelUri::parse("acp://cursor/composer-2").unwrap();
        assert_eq!(u.scheme, "acp");
        assert_eq!(u.name, "cursor");
        assert_eq!(u.model, "composer-2");
    }

    #[test]
    fn parse_legacy_bare_model() {
        let u = ModelUri::parse("gpt-5.3-codex").unwrap();
        assert_eq!(u.scheme, "acp");
        assert_eq!(u.name, "cursor");
        assert_eq!(u.model, "gpt-5.3-codex");
    }
}
