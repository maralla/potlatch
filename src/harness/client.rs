//! LLM client: trait + OpenAI-compatible implementation with streaming.

use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Value, json};

/// Callback for streaming chunks.
pub type StreamCallback = dyn Fn(&str) + Send + Sync;

/// Abstraction over LLM chat completion backends. Enables test doubles.
pub trait ChatClient: Send + Sync {
    /// Send a chat completion request with tools. Calls `on_chunk` for each streamed delta.
    /// Returns the full accumulated response.
    fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        on_chunk: Option<&StreamCallback>,
    ) -> Result<ChatResponse>;

    /// List available models from the backend. Returns empty if unsupported.
    fn list_models(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// A complete chat response (accumulated from streaming or received at once).
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Assistant text content.
    pub content: String,
    /// Reasoning content (if the model produces it, e.g. Model/o1-style reasoning).
    pub reasoning: Option<String>,
    /// Tool calls (OpenAI format): `[{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]`
    pub tool_calls: Vec<Value>,
    /// `"stop"`, `"tool_calls"`, or other finish reasons.
    pub finish_reason: String,
}

/// OpenAI-compatible LLM client using reqwest blocking.
///
/// Works with any backend that implements the OpenAI `/v1/chat/completions`
/// and `/v1/models` API (vLLM, llama.cpp, Ollama, OpenRouter, etc.).
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self {
            base_url,
            api_key,
            client,
        }
    }

    /// Build the full URL for an API path. Handles base_url that may or may not
    /// already end with `/v1`.
    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{base}{path}")
        } else {
            format!("{base}/v1{path}")
        }
    }
}

impl ChatClient for OpenAiClient {
    fn list_models(&self) -> Result<Vec<String>> {
        let url = self.url("/models");
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .context("GET /v1/models")?;
        let body: Value = resp.json().context("parse /v1/models response")?;
        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        on_chunk: Option<&StreamCallback>,
    ) -> Result<ChatResponse> {
        let url = self.url("/chat/completions");

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .context("POST /v1/chat/completions")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("LLM request failed ({}): {}", status, text);
        }

        // Parse SSE stream line-by-line (true streaming, not buffering the whole response)
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut finish_reason = String::new();

        let reader = BufReader::new(resp);
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            let chunk: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let delta = &chunk["choices"][0]["delta"];

            // Text content
            if let Some(text) = delta["content"].as_str() {
                content.push_str(text);
                if let Some(cb) = on_chunk {
                    cb(text);
                }
            }

            // Reasoning content (model-specific, e.g. Model/o1-style)
            if let Some(r) = delta["reasoning"].as_str() {
                reasoning.push_str(r);
            }

            // Tool calls (streaming accumulation)
            if let Some(tc_array) = delta["tool_calls"].as_array() {
                for tc in tc_array {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    while tool_calls.len() <= idx {
                        tool_calls.push(json!({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        }));
                    }
                    if let Some(id) = tc["id"].as_str()
                        && !id.is_empty()
                    {
                        tool_calls[idx]["id"] = json!(id);
                    }
                    if let Some(name) = tc["function"]["name"].as_str()
                        && !name.is_empty()
                    {
                        tool_calls[idx]["function"]["name"] = json!(name);
                    }
                    if let Some(args) = tc["function"]["arguments"].as_str() {
                        let current = tool_calls[idx]["function"]["arguments"]
                            .as_str()
                            .unwrap_or("");
                        tool_calls[idx]["function"]["arguments"] =
                            json!(format!("{current}{args}"));
                    }
                }
            }

            // Finish reason
            if let Some(fr) = chunk["choices"][0]["finish_reason"].as_str()
                && !fr.is_empty()
            {
                finish_reason = fr.to_string();
            }
        }

        // Fallback: if no SSE data was received, the endpoint may not support streaming
        if content.is_empty() && tool_calls.is_empty() && finish_reason.is_empty() {
            // The response was already consumed by the BufReader; we can't re-read it.
            // This fallback only triggers when the endpoint returns no SSE lines at all,
            // which indicates a misconfigured endpoint. Log and return empty.
            tracing::warn!(
                "harness: LLM endpoint returned no SSE data; check if streaming is supported"
            );
        }

        Ok(ChatResponse {
            content,
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
            tool_calls,
            finish_reason,
        })
    }
}

/// Fake LLM client for testing — returns scripted responses in sequence.
#[cfg(test)]
pub struct FakeChatClient {
    responses: std::sync::Mutex<Vec<ChatResponse>>,
}

#[cfg(test)]
impl FakeChatClient {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[cfg(test)]
impl ChatClient for FakeChatClient {
    fn chat(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: &[Value],
        _on_chunk: Option<&StreamCallback>,
    ) -> Result<ChatResponse> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Ok(ChatResponse {
                content: "No more scripted responses".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
            });
        }
        Ok(responses.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_client_returns_scripted_responses() {
        let client = FakeChatClient::new(vec![
            ChatResponse {
                content: String::new(),
                reasoning: None,
                tool_calls: vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\":\"echo hi\"}"}
                })],
                finish_reason: "tool_calls".into(),
            },
            ChatResponse {
                content: "Done!".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]);

        let resp1 = client.chat("m", &[], &[], None).unwrap();
        assert_eq!(resp1.finish_reason, "tool_calls");
        assert_eq!(resp1.tool_calls.len(), 1);

        let resp2 = client.chat("m", &[], &[], None).unwrap();
        assert_eq!(resp2.finish_reason, "stop");
        assert_eq!(resp2.content, "Done!");
    }
}
