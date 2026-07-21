//! LLM client: trait + OpenAI-compatible implementation with streaming.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Value, json};

/// Callback for streaming chunks.
pub type StreamCallback = dyn Fn(&str) + Send + Sync;

/// Callback invoked when a single tool call's arguments appear complete during
/// streaming (before `finish_reason`). Fires once per tool call, as soon as
/// the accumulated `arguments` string parses as valid JSON and the next SSE
/// chunk does not extend that tool call's arguments. Lets the caller start
/// read-only tools speculatively, overlapping their I/O with the model's
/// remaining generation (reasoning tail, finish_reason, usage chunk).
///
/// Receives `(index, tool_call)` and returns `Some(result)` if the caller
/// executed the tool (caching the result for later collection), or `None` if
/// the caller chose not to execute speculatively (e.g. mutating tool). The
/// caller is responsible for caching results by `index` and returning them
/// from the final [`ToolExecCallback`] invocation.
pub type EarlyToolExecCallback<'a> = dyn Fn(usize, &Value) -> Option<String> + Send + Sync + 'a;

/// Callback invoked when the LLM response is fully received but before
/// `chat()` returns. Allows the caller to start executing tool calls while
/// the client finishes parsing trailing SSE data (usage, [DONE]).
/// Receives `(tool_calls, finish_reason)` and returns a vector of result
/// strings (one per tool call, in order).
pub type ToolExecCallback<'a> = dyn Fn(&[Value], &str) -> Vec<String> + Send + Sync + 'a;

/// Abstraction over LLM chat completion backends. Enables test doubles.
pub trait ChatClient: Send + Sync {
    /// Send a chat completion request with tools. Calls `on_chunk` for each streamed delta.
    /// If `on_tool_calls` is provided, it is invoked as soon as the complete tool-call list
    /// is known (when `finish_reason` arrives), overlapping tool execution with the tail of
    /// the SSE stream. If `on_early_tool_call` is provided, it is invoked per tool call as
    /// soon as that call's arguments parse as valid JSON during streaming — enabling
    /// speculative execution of read-only tools before `finish_reason` arrives.
    fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse>;

    /// List available models from the backend. Returns empty if unsupported.
    fn list_models(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Token usage reported by the model API for a single chat completion.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Input (prompt) tokens.
    pub input_tokens: u64,
    /// Output (completion) tokens.
    pub output_tokens: u64,
    /// Input tokens served from the prefix cache (a subset of `input_tokens`).
    pub cached_tokens: u64,
}

impl Usage {
    fn from_json(usage: &Value) -> Self {
        Self {
            input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        }
    }
}

/// A complete chat response (accumulated from streaming or received at once).
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Assistant text content.
    pub content: String,
    /// Tool calls (OpenAI format): `[{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]`
    pub tool_calls: Vec<Value>,
    /// `"stop"`, `"tool_calls"`, or other finish reasons.
    pub finish_reason: String,
    /// Token usage for this completion, if reported by the backend.
    pub usage: Usage,
    /// Pre-computed tool results, populated when `on_tool_calls` is used.
    /// Indexes align with `tool_calls`. Empty if no overlap execution was used.
    pub tool_results: Vec<String>,
    /// Wall-clock time for the API call, in milliseconds.
    pub elapsed_ms: u128,
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
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        let url = self.url("/chat/completions");

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            // Ask the backend to include token usage in the final SSE chunk.
            "stream_options": {"include_usage": true},
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        let started = Instant::now();
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
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut finish_reason = String::new();
        let mut usage = Usage::default();
        // Tool results computed via overlap execution (populated when finish_reason arrives).
        let mut tool_results: Vec<String> = Vec::new();
        // Indices that have been speculatively executed via on_early_tool_call.
        // Tracked so we don't fire twice for the same tool call.
        let mut speculatively_executed: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        // Helper: try to fire on_early_tool_call for a given index if its
        // arguments parse as valid JSON and it hasn't been executed yet.
        // Returns true if the callback was invoked.
        let try_early_exec = |idx: usize,
                              tool_calls: &Vec<Value>,
                              executed: &mut std::collections::HashSet<usize>,
                              cb: Option<&EarlyToolExecCallback<'_>>|
         -> bool {
            if executed.contains(&idx) {
                return false;
            }
            let Some(cb) = cb else {
                return false;
            };
            let Some(tc) = tool_calls.get(idx) else {
                return false;
            };
            // Only fire when the tool call has a name and parseable arguments.
            let name = tc["function"]["name"].as_str().unwrap_or("");
            if name.is_empty() {
                return false;
            }
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("");
            if args_str.is_empty() {
                return false;
            }
            // Arguments must parse as valid JSON — partial streaming chunks
            // won't parse, so this gates speculative execution until the
            // arguments are likely complete.
            if serde_json::from_str::<Value>(args_str).is_err() {
                return false;
            }
            // Mark as executed BEFORE calling the callback so the caller's
            // closure can safely mutate shared state without re-entry.
            executed.insert(idx);
            // Fire and discard the result — the caller caches it internally
            // and returns it from the final on_tool_calls callback.
            let _ = cb(idx, tc);
            true
        };

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

            // Token usage (arrives in the final chunk when include_usage is set)
            if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
                usage = Usage::from_json(u);
            }

            let delta = &chunk["choices"][0]["delta"];

            // Text content
            if let Some(text) = delta["content"].as_str() {
                content.push_str(text);
                if let Some(cb) = on_chunk {
                    cb(text);
                }
            }

            // Track the highest tool-call index seen in this chunk so we can
            // fire speculative execution for earlier indices whose arguments
            // are now complete (a new index appearing means the previous one
            // is done streaming arguments).
            let mut new_max_index: Option<usize> = None;

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
                    new_max_index = Some(idx.max(new_max_index.unwrap_or(0)));
                }
            }

            // Speculative execution: when a new tool-call index appears, all
            // earlier indices whose arguments parse as valid JSON are complete.
            // Fire on_early_tool_call for them so read-only tools start
            // executing while the model continues generating the later calls.
            if let Some(max_idx) = new_max_index {
                for earlier in 0..max_idx {
                    try_early_exec(
                        earlier,
                        &tool_calls,
                        &mut speculatively_executed,
                        on_early_tool_call,
                    );
                }
            }

            // Finish reason — when this arrives, all tool calls are complete.
            // Kick off tool execution immediately to overlap with the stream tail
            // (usage chunk, [DONE]).
            if let Some(fr) = chunk["choices"][0]["finish_reason"].as_str()
                && !fr.is_empty()
            {
                finish_reason = fr.to_string();
                if let Some(exec) = on_tool_calls
                    && !tool_calls.is_empty()
                {
                    tool_results = exec(&tool_calls, &finish_reason);
                }
            }
        }

        let elapsed = started.elapsed();

        // Fallback: if no SSE data was received, the endpoint may not support streaming
        if content.is_empty() && tool_calls.is_empty() && finish_reason.is_empty() {
            tracing::warn!(
                "harness: LLM endpoint returned no SSE data; check if streaming is supported"
            );
        }

        Ok(ChatResponse {
            content,
            tool_calls,
            finish_reason,
            usage,
            tool_results,
            elapsed_ms: elapsed.as_millis(),
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
        _on_tool_calls: Option<&ToolExecCallback<'_>>,
        _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Ok(ChatResponse {
                content: "No more scripted responses".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
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
                tool_calls: vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\":\"echo hi\"}"}
                })],
                finish_reason: "tool_calls".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
            ChatResponse {
                content: "Done!".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
        ]);

        let resp1 = client.chat("m", &[], &[], None, None, None).unwrap();
        assert_eq!(resp1.finish_reason, "tool_calls");
        assert_eq!(resp1.tool_calls.len(), 1);

        let resp2 = client.chat("m", &[], &[], None, None, None).unwrap();
        assert_eq!(resp2.finish_reason, "stop");
        assert_eq!(resp2.content, "Done!");
    }

    #[test]
    fn usage_from_json_parses_standard_fields() {
        let v = json!({
            "prompt_tokens": 1200,
            "completion_tokens": 350,
            "prompt_tokens_details": {"cached_tokens": 800}
        });
        let u = Usage::from_json(&v);
        assert_eq!(u.input_tokens, 1200);
        assert_eq!(u.output_tokens, 350);
        assert_eq!(u.cached_tokens, 800);
    }

    #[test]
    fn usage_from_json_defaults_missing_fields() {
        let v = json!({"prompt_tokens": 100});
        let u = Usage::from_json(&v);
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.cached_tokens, 0);
    }

    #[test]
    fn usage_from_json_handles_empty() {
        let u = Usage::from_json(&json!({}));
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.cached_tokens, 0);
    }
}
