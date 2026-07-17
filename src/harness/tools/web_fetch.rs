//! Web fetch tool: HTTP GET to retrieve web page content, with proper HTML-to-text conversion.

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

const MAX_OUTPUT: usize = 50_000;

pub struct WebFetchTool {
    client: reqwest::blocking::Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { client }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Fetch the content of a web page by URL. Returns the text content (HTML converted to readable text). Follows up to 5 redirects. Use for looking up documentation, API references, or other web resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'url' argument"))?;

        let resp = self
            .client
            .get(url)
            .header("User-Agent", "potlatch-harness/0.1")
            .send()
            .map_err(|e| anyhow::anyhow!("fetch failed: {e}"))?;

        let status = resp.status();
        let final_url = resp.url().to_string();
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if !status.is_success() {
            return Ok(format!("HTTP {status} from {url}"));
        }

        let body = resp.text().map_err(|e| anyhow::anyhow!("read body: {e}"))?;

        let text = if content_type.contains("html") {
            let config = html2text::config::plain().max_wrap_width(120);
            let dom = config.parse_html(body.as_bytes())?;
            let render_tree = config.dom_to_render_tree(&dom)?;
            config.render_to_string(render_tree, 120)?
        } else {
            body
        };

        let header = if final_url != url {
            format!("[redirected to {final_url}]\n\n")
        } else {
            String::new()
        };

        let result = if text.len() + header.len() > MAX_OUTPUT {
            let truncated = truncate_at_char_boundary(&text, MAX_OUTPUT - header.len());
            format!(
                "{header}{truncated}\n[...content truncated at {} chars...]",
                MAX_OUTPUT
            )
        } else {
            format!("{header}{text}")
        };

        Ok(result)
    }
}

fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_char_boundary() {
        // Multi-byte UTF-8: "é" is 2 bytes
        let s = "abcéfg";
        let truncated = truncate_at_char_boundary(s, 4);
        assert!(truncated.is_char_boundary(truncated.len()));
    }
}
