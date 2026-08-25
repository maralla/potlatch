//! HTTP client tool: issues arbitrary HTTP requests (GET, POST, PUT, PATCH,
//! DELETE) with optional headers and body, and returns the response as
//! readable text (HTML → text, plain otherwise). Use this instead of the
//! browser-backed `web_fetch` when you only need a static or non-HTML response
//! and do not need JavaScript rendering.

use std::collections::HashMap;

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use super::Tool;

const MAX_OUTPUT: usize = 50_000;
const DEFAULT_USER_AGENT: &str = "http-tool/0.1";

pub struct HttpTool {
    client: reqwest::blocking::Client,
}

impl HttpTool {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { client }
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "HTTP client: issue an HTTP request with a chosen method, optional headers, and optional body, returning the response as readable text (static HTML is converted to text). Use for API calls, JSON endpoints, and static contents. Follows up to 5 redirects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute URL to request."
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method. Defaults to GET.",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional request headers as a JSON object of header name to value. A `User-Agent` header here overrides the default `http-tool/0.1`.",
                        "additionalProperties": { "type": "string" }
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (e.g. JSON, form, or raw text). Ignored for GET/HEAD."
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
        let method = args["method"]
            .as_str()
            .map(str::to_ascii_uppercase)
            .unwrap_or_else(|| "GET".to_string());
        let headers: HashMap<String, String> = args
            .get("headers")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let body = args.get("body").and_then(Value::as_str);

        let method = reqwest::Method::from_bytes(method.as_bytes())
            .with_context(|| format!("unsupported HTTP method {method:?}"))?;
        // Default headers; a `User-Agent` in the caller's `headers` overrides
        // the default because `HeaderMap::insert` replaces existing values.
        let mut header_map = HeaderMap::new();
        header_map.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static(DEFAULT_USER_AGENT),
        );
        for (name, value) in &headers {
            let header_name = HeaderName::try_from(name.as_str())
                .with_context(|| format!("invalid header name {name:?}"))?;
            let header_value = HeaderValue::try_from(value.as_str())
                .with_context(|| format!("invalid header value for {name:?}: {value:?}"))?;
            header_map.insert(header_name, header_value);
        }
        let mut request = self.client.request(method, url).headers(header_map);
        if let Some(body) = body {
            request = request.body(body.to_string());
        }

        let resp = request
            .send()
            .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;

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

        let response_body = resp
            .text()
            .map_err(|e| anyhow::anyhow!("read response body: {e}"))?;

        let text = if content_type.contains("html") {
            let config = html2text::config::plain().max_wrap_width(120);
            let dom = config.parse_html(response_body.as_bytes())?;
            let render_tree = config.dom_to_render_tree(&dom)?;
            config.render_to_string(render_tree, 120)?
        } else {
            response_body
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

    #[test]
    fn schema_supports_method_headers_and_body() {
        let schema = HttpTool::new().schema();
        let props = schema["parameters"]["properties"].as_object().unwrap();
        assert!(props.contains_key("method"));
        assert!(props.contains_key("headers"));
        assert!(props.contains_key("body"));
        let methods = schema["parameters"]["properties"]["method"]["enum"]
            .as_array()
            .unwrap();
        let method_strings: Vec<&str> = methods.iter().filter_map(Value::as_str).collect();
        assert!(method_strings.contains(&"GET"));
        assert!(method_strings.contains(&"POST"));
        assert!(method_strings.contains(&"PUT"));
        assert!(method_strings.contains(&"DELETE"));
    }
}
