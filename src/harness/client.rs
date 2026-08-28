//! LLM client: trait + OpenAI-compatible implementation with streaming.

use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Value, json};

/// Callback for streaming text deltas during the LLM response.
pub type StreamCallback = dyn Fn(&str) + Send + Sync;

/// Callback invoked after each complete LLM response turn, providing the full
/// `ChatResponse` (content, reasoning, tool calls, etc.) for transcript
/// logging and progress tracking.
pub type TurnCallback = dyn Fn(&ChatResponse) + Send + Sync;

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
    ///
    /// The `model` string may be a plain model name (e.g. `"model1-fp8"`) or an
    /// `acp://` URL (e.g. `"acp://zhipu/model1-fp8?thinking=false"`). Implementations
    /// that support the URL form parse it via [`ModelSpec::parse`] to extract the
    /// real model name and options like `thinking`.
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

/// Parsed model specification from an `acp://` URL.
///
/// Format: `acp://<vendor>/<model>?thinking=false&reasoning_effort=high`
///
/// When the model string is a plain name (no `acp://` prefix), it's treated as
/// the model name with default options. The `thinking` query param controls
/// whether reasoning is enabled. The `reasoning_effort` query param controls
/// the effort level (e.g. `low`, `high`, `max`) for models that support it
/// (currently DeepSeek).
///
/// Examples:
/// - `"model1-fp8"` → `ModelSpec { model: "model1-fp8", thinking: false, .. }`
/// - `"acp://zhipu/model1-fp8?thinking=true"` → `ModelSpec { model: "model1-fp8", thinking: true, .. }`
/// - `"acp://deepseek/deepseek-v4-flash?thinking=true&reasoning_effort=high"`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// The actual model name to send to the API (e.g. `"model1-fp8"`).
    pub model: String,
    /// Whether to enable thinking/reasoning tokens. Defaults to `false`.
    pub thinking: bool,
    /// Reasoning effort level (e.g. `"low"`, `"high"`, `"max"`). Only
    /// meaningful when `thinking` is true and the model supports effort
    /// control (currently DeepSeek). `None` leaves the backend default.
    pub reasoning_effort: Option<String>,
}

impl ModelSpec {
    /// Parse a model string that may be a plain name, an `acp://` URL, or a
    /// plain name carrying a `?thinking=` query. The latter arrives via the ACP
    /// `session/set_model` command: the orchestrator forwards the config URI's
    /// model segment verbatim (e.g. `model1-fp8?thinking=true`), so the harness
    /// must honor the query even without the `acp://` prefix.
    pub fn parse(s: &str) -> Self {
        let trimmed = s.trim();
        let rest = trimmed.strip_prefix("acp://").unwrap_or(trimmed);
        // Split path and query: `vendor/model?thinking=false`
        let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
        // Extract model name: take the last path segment after `/`.
        let model = path.rsplit('/').next().unwrap_or(path).to_string();
        let thinking = parse_query_bool(query, "thinking").unwrap_or(false);
        let reasoning_effort = parse_query_str(query, "reasoning_effort");
        ModelSpec {
            model,
            thinking,
            reasoning_effort,
        }
    }

    /// Whether this model uses the DeepSeek thinking-mode API (the model name
    /// starts with `deepseek`).
    fn is_deepseek(&self) -> bool {
        self.model.to_lowercase().starts_with("deepseek")
    }
}

/// Parse a boolean query parameter from a query string like `thinking=false`.
/// Returns `None` when the parameter is absent.
fn parse_query_bool(query: &str, key: &str) -> Option<bool> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=')
            && k == key
        {
            return match v.to_lowercase().as_str() {
                "false" | "0" | "no" | "off" => Some(false),
                "true" | "1" | "yes" | "on" => Some(true),
                _ => None,
            };
        }
    }
    None
}

/// Parse a string query parameter from a query string like `reasoning_effort=high`.
/// Returns `None` when the parameter is absent or empty.
fn parse_query_str(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=')
            && k == key
        {
            let v = v.trim();
            return (!v.is_empty()).then(|| v.to_string());
        }
    }
    None
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
    /// Reasoning/thinking content streamed by reasoning models (Model1,
    /// DeepSeek R1, etc.) via `delta.reasoning_content`. Re-injected into the
    /// context on the next turn so the model can see its prior reasoning and
    /// build on it instead of re-deriving the same conclusions. Empty when
    /// thinking is disabled or the backend doesn't emit reasoning.
    pub reasoning: String,
}

/// OpenAI-compatible LLM client using reqwest blocking.
///
/// Works with any backend that implements the OpenAI `/v1/chat/completions`
/// and `/v1/models` API (vLLM, llama.cpp, Ollama, OpenRouter, etc.).
///
/// The underlying `reqwest::blocking::Client` (and thus its HTTP connection
/// pool) is held behind a `Mutex` so it can be swapped for a fresh client on a
/// transient failure. A pooled connection that produced an error may be in a
/// bad state; reusing it for a retry would just fail again. On each transient
/// failure we drop the old client and build a new one, so the retry opens a
/// brand-new connection from a fresh pool.
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    client: Mutex<reqwest::blocking::Client>,
}

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        let client = Self::build_client();
        Self {
            base_url,
            api_key,
            client: Mutex::new(client),
        }
    }

    /// Build a fresh `reqwest::blocking::Client` with the harness's standard
    /// timeout. Each call produces an independent connection pool, used to
    /// discard a poisoned pool after a transient failure.
    fn build_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new())
    }

    /// Replace the pooled client with a fresh one, dropping the old connection
    /// pool. Called after a transient failure so the retry doesn't reuse a
    /// connection that may be in a bad state.
    fn reset_client(&self) {
        let new_client = Self::build_client();
        if let Ok(mut guard) = self.client.lock() {
            *guard = new_client;
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
        let resp = {
            let client = self
                .client
                .lock()
                .map_err(|_| anyhow::anyhow!("client lock poisoned"))?;
            client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .context("GET /v1/models")?
        };
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
        let body = build_chat_body(model, messages, tools);

        let started = Instant::now();

        // Send the request with transient-failure retry. The retry wraps only
        // the request-send + status-check phase — once a 200 response arrives
        // and we start reading the SSE stream, the streaming callbacks have
        // fired and we can't transparently retry. On each transient failure we
        // drop the pooled client and build a fresh one so the retry opens a
        // brand-new connection (a pooled connection that produced the error
        // may be in a bad state and reusing it would just fail again).
        let resp = self.send_with_retry(&url, &body)?;

        // Parse SSE stream line-by-line (true streaming, not buffering the whole response)
        let mut content = String::new();
        let mut reasoning = String::new();
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

            // Text content. Reasoning models (Model1, DeepSeek R1, etc.)
            // stream reasoning in `delta.reasoning_content` — we capture it
            // for re-injection into the next turn's context so the model can
            // build on its prior reasoning instead of re-deriving it.
            if let Some(text) = delta["content"].as_str() {
                content.push_str(text);
                if let Some(cb) = on_chunk {
                    cb(text);
                }
            }
            if let Some(r) = delta["reasoning_content"].as_str() {
                reasoning.push_str(r);
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
            reasoning,
        })
    }
}

impl OpenAiClient {
    /// Send a single chat-completion request. Locks the pooled client, sends
    /// the request, and checks the HTTP status. Returns the streaming
    /// `Response` on success. The lock is released as soon as `send()` returns
    /// — reading the SSE body later does not hold the lock.
    fn send_request(&self, url: &str, body: &Value) -> Result<reqwest::blocking::Response> {
        let client = self
            .client
            .lock()
            .map_err(|_| anyhow::anyhow!("client lock poisoned"))?;
        let resp = client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .context("POST /v1/chat/completions")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("LLM request failed ({}): {}", status, text);
        }

        Ok(resp)
    }

    /// Send a chat-completion request with transient-failure retry. On each
    /// transient failure (network error, 429, 5xx), the pooled client is
    /// replaced with a fresh one so the retry opens a new connection — the
    /// failed connection may be in a bad state and reusing it would just fail
    /// again. Permanent errors (400, 401, 403, 404) propagate immediately.
    fn send_with_retry(&self, url: &str, body: &Value) -> Result<reqwest::blocking::Response> {
        const MAX_ATTEMPTS: u32 = 5;
        const INITIAL_DELAY: Duration = Duration::from_secs(1);
        const MAX_DELAY: Duration = Duration::from_secs(30);

        let mut attempt = 0u32;
        let mut delay = INITIAL_DELAY;
        loop {
            attempt += 1;
            match self.send_request(url, body) {
                Ok(resp) => {
                    if attempt > 1 {
                        tracing::info!(
                            "harness: LLM request succeeded after {} attempt(s)",
                            attempt
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    if attempt >= MAX_ATTEMPTS || !is_transient_llm_error(&e) {
                        return Err(e);
                    }
                    tracing::warn!(
                        "harness: LLM request failed (attempt {attempt}/{MAX_ATTEMPTS}), \
                         resetting connection and retrying in {delay:?}: {e:#}"
                    );
                    self.reset_client();
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(MAX_DELAY);
                }
            }
        }
    }
}

/// Build the JSON request body for a chat-completion request.
fn build_chat_body(model: &str, messages: &[Value], tools: &[Value]) -> Value {
    let spec = ModelSpec::parse(model);

    let mut body = json!({
        "model": spec.model,
        "messages": messages,
        "stream": true,
        // Ask the backend to include token usage in the final SSE chunk.
        "stream_options": {"include_usage": true},
    });

    if spec.is_deepseek() {
        // DeepSeek uses `{"thinking": {"type": "enabled/disabled"}}` in the
        // request body (OpenAI extra_body format). When thinking is enabled,
        // send `reasoning_effort` — defaulting to "max" when unspecified.
        if !spec.thinking {
            body["thinking"] = json!({"type": "disabled"});
        } else {
            let effort = spec.reasoning_effort.as_deref().unwrap_or("max");
            body["reasoning_effort"] = json!(effort);
        }
    } else if !spec.thinking {
        // Non-DeepSeek reasoning models (Model1, etc.) served via
        // sglang/vLLM honor this chat-template kwarg to skip the reasoning
        // phase entirely. Harmless on backends that don't recognize it.
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
    }

    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    body
}

/// Whether an LLM API error is worth retrying. Transient errors include
/// network-level failures (timeouts, connection resets, TLS blips) and
/// transient HTTP status codes (429 rate-limit, 500/502/503/504 server errors).
/// Permanent client/validation errors (400, 401, 403, 404) propagate
/// immediately without retry.
fn is_transient_llm_error(err: &anyhow::Error) -> bool {
    // Search the full error chain — reqwest wraps the underlying network error
    // as a source, and our own `.context()` adds another layer.
    let full = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    let m = full.to_lowercase();

    // Permanent client/validation errors — repeating the request won't help.
    // Note: 400 is handled specially by the agent loop (malformed tool-call
    // sanitization), so it must propagate immediately rather than being
    // retried at the client level.
    if m.contains("400 bad request")
        || m.contains("401 unauthorized")
        || m.contains("403 forbidden")
        || m.contains("404 not found")
        || m.contains("405 method not allowed")
        || m.contains("422 unprocessable entity")
    {
        return false;
    }

    // Transient HTTP status codes: rate-limit and server-side errors.
    if m.contains("429")
        || m.contains("500 internal server error")
        || m.contains("502 bad gateway")
        || m.contains("503 service unavailable")
        || m.contains("504 gateway timeout")
    {
        return true;
    }

    // Network-level failures (the request never completed — no HTTP status).
    // These are always potentially transient.
    if m.contains("timed out")
        || m.contains("timeout")
        || m.contains("connection refused")
        || m.contains("connection reset")
        || m.contains("connection closed")
        || m.contains("broken pipe")
        || m.contains("tls")
        || m.contains("dns")
        || m.contains("lookup")
        || m.contains("connect error")
        || m.contains("network")
        || m.contains("eof")
    {
        return true;
    }

    // Default: treat unknown errors as transient. A spurious retry is cheaper
    // than failing a long-running agent task on a one-off blip, and the retry
    // count is bounded.
    true
}

/// Fake LLM client for testing — returns scripted responses in sequence.
#[cfg(test)]
type ChatCallback = Box<dyn Fn(&[Value]) + Send + Sync>;

#[cfg(test)]
pub struct FakeChatClient {
    responses: std::sync::Mutex<Vec<ChatResponse>>,
    on_chat: Option<ChatCallback>,
}

#[cfg(test)]
impl FakeChatClient {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            on_chat: None,
        }
    }

    /// Create a fake client that invokes `on_chat` with the messages array
    /// before returning each scripted response. Useful for verifying that
    /// injected messages appear in the LLM call.
    pub fn with_callback(
        responses: Vec<ChatResponse>,
        on_chat: impl Fn(&[Value]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            on_chat: Some(Box::new(on_chat)),
        }
    }
}

#[cfg(test)]
impl ChatClient for FakeChatClient {
    fn chat(
        &self,
        _model: &str,
        messages: &[Value],
        _tools: &[Value],
        _on_chunk: Option<&StreamCallback>,
        _on_tool_calls: Option<&ToolExecCallback<'_>>,
        _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        if let Some(ref cb) = self.on_chat {
            cb(messages);
        }
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Ok(ChatResponse {
                content: "No more scripted responses".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
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
                reasoning: String::new(),
            },
            ChatResponse {
                content: "Done!".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
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

    #[test]
    fn model_spec_parses_plain_model_name() {
        let spec = ModelSpec::parse("model1-fp8");
        assert_eq!(spec.model, "model1-fp8");
        assert!(
            !spec.thinking,
            "plain model name defaults to thinking=false"
        );
    }

    #[test]
    fn model_spec_parses_bare_name_with_thinking_query() {
        // The orchestrator forwards the config URI's model segment verbatim via
        // ACP session/set_model, so the harness receives a bare name carrying
        // the ?thinking= query (e.g. `model1-fp8?thinking=true`). The query
        // must be honored and stripped from the model name sent to the API.
        let spec = ModelSpec::parse("model1-fp8?thinking=true");
        assert_eq!(spec.model, "model1-fp8");
        assert!(spec.thinking);

        let spec = ModelSpec::parse("model1-fp8?thinking=false");
        assert_eq!(spec.model, "model1-fp8");
        assert!(!spec.thinking);
    }

    #[test]
    fn model_spec_parses_acp_url_with_thinking_false() {
        let spec = ModelSpec::parse("acp://zhipu/model1-fp8?thinking=false");
        assert_eq!(spec.model, "model1-fp8");
        assert!(!spec.thinking);
    }

    #[test]
    fn model_spec_parses_acp_url_with_thinking_true() {
        let spec = ModelSpec::parse("acp://zhipu/model1-fp8?thinking=true");
        assert_eq!(spec.model, "model1-fp8");
        assert!(spec.thinking);
    }

    #[test]
    fn model_spec_parses_acp_url_without_query() {
        let spec = ModelSpec::parse("acp://openai/gpt-4o");
        assert_eq!(spec.model, "gpt-4o");
        assert!(!spec.thinking, "missing thinking param defaults to false");
    }

    #[test]
    fn model_spec_parses_acp_url_with_vendor_prefix() {
        // The vendor segment is stripped; only the model name matters.
        let spec = ModelSpec::parse("acp://zhipu/model1-fp8?thinking=false");
        assert_eq!(spec.model, "model1-fp8");
    }

    #[test]
    fn model_spec_parses_nested_model_path() {
        // Some vendors use org/model format.
        let spec = ModelSpec::parse("acp://hf/Qwen/QwQ-32B?thinking=true");
        assert_eq!(spec.model, "QwQ-32B");
        assert!(spec.thinking);
    }

    #[test]
    fn model_spec_parses_thinking_param_variants() {
        assert!(!ModelSpec::parse("acp://v/m?thinking=false").thinking);
        assert!(!ModelSpec::parse("acp://v/m?thinking=0").thinking);
        assert!(!ModelSpec::parse("acp://v/m?thinking=no").thinking);
        assert!(!ModelSpec::parse("acp://v/m?thinking=off").thinking);
        assert!(ModelSpec::parse("acp://v/m?thinking=true").thinking);
        assert!(ModelSpec::parse("acp://v/m?thinking=1").thinking);
        assert!(ModelSpec::parse("acp://v/m?thinking=yes").thinking);
        assert!(ModelSpec::parse("acp://v/m?thinking=on").thinking);
    }

    #[test]
    fn model_spec_ignores_unknown_query_params() {
        let spec = ModelSpec::parse("acp://v/m?foo=bar&thinking=false&baz=1");
        assert_eq!(spec.model, "m");
        assert!(!spec.thinking);
    }

    #[test]
    fn model_spec_handles_empty_string() {
        let spec = ModelSpec::parse("");
        assert_eq!(spec.model, "");
        assert!(!spec.thinking);
    }

    // --- reasoning_effort query param tests ---

    #[test]
    fn model_spec_parses_reasoning_effort() {
        let spec = ModelSpec::parse(
            "acp://deepseek/deepseek-v4-flash?thinking=true&reasoning_effort=high",
        );
        assert_eq!(spec.model, "deepseek-v4-flash");
        assert!(spec.thinking);
        assert_eq!(spec.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn model_spec_reasoning_effort_defaults_to_none() {
        let spec = ModelSpec::parse("acp://deepseek/deepseek-v4-flash?thinking=true");
        assert!(spec.thinking);
        assert!(spec.reasoning_effort.is_none());
    }

    #[test]
    fn model_spec_detects_deepseek_model() {
        assert!(ModelSpec::parse("deepseek-v4-flash").is_deepseek());
        assert!(ModelSpec::parse("acp://deepseek/deepseek-v4-pro").is_deepseek());
        assert!(!ModelSpec::parse("model1-fp8").is_deepseek());
        assert!(!ModelSpec::parse("acp://cursor/gpt-4o").is_deepseek());
    }

    // --- build_chat_body thinking format tests ---

    #[test]
    fn build_chat_body_deepseek_disabled_thinking() {
        let body = build_chat_body(
            "acp://deepseek/deepseek-v4-flash?thinking=false",
            &[json!({"role": "user", "content": "hi"})],
            &[],
        );
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "deepseek should not use chat_template_kwargs"
        );
    }

    #[test]
    fn build_chat_body_deepseek_enabled_thinking_with_effort() {
        let body = build_chat_body(
            "acp://deepseek/deepseek-v4-flash?thinking=true&reasoning_effort=high",
            &[json!({"role": "user", "content": "hi"})],
            &[],
        );
        assert_eq!(body["reasoning_effort"], json!("high"));
        assert!(
            body.get("thinking").is_none(),
            "no explicit thinking toggle when enabled (default)"
        );
        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn build_chat_body_deepseek_enabled_thinking_without_effort() {
        let body = build_chat_body(
            "acp://deepseek/deepseek-v4-flash?thinking=true",
            &[json!({"role": "user", "content": "hi"})],
            &[],
        );
        assert_eq!(body["reasoning_effort"], json!("max"));
        assert!(body.get("thinking").is_none());
        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn build_chat_body_non_deepseek_disabled_thinking() {
        let body = build_chat_body(
            "acp://zhipu/model1-fp8?thinking=false",
            &[json!({"role": "user", "content": "hi"})],
            &[],
        );
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_chat_body_non_deepseek_enabled_thinking() {
        let body = build_chat_body(
            "acp://zhipu/model1-fp8?thinking=true",
            &[json!({"role": "user", "content": "hi"})],
            &[],
        );
        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("thinking").is_none());
    }

    // --- is_transient_llm_error classifier tests ---

    #[test]
    fn transient_error_retries_429_rate_limit() {
        let err = anyhow::anyhow!("LLM request failed (429 Too Many Requests): rate limited");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_500_server_error() {
        let err = anyhow::anyhow!("LLM request failed (500 Internal Server Error): upstream boom");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_502_bad_gateway() {
        let err = anyhow::anyhow!("LLM request failed (502 Bad Gateway): bad gateway");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_503_service_unavailable() {
        let err =
            anyhow::anyhow!("LLM request failed (503 Service Unavailable): temporarily overloaded");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_504_gateway_timeout() {
        let err = anyhow::anyhow!("LLM request failed (504 Gateway Timeout): upstream timed out");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_network_timeout() {
        let err = anyhow::anyhow!("POST /v1/chat/completions: operation timed out");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_connection_reset() {
        let err = anyhow::anyhow!("POST /v1/chat/completions: connection reset by peer");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_tls_handshake_failure() {
        let err = anyhow::anyhow!(
            "POST /v1/chat/completions: error sending request: error trying to connect: \
             error:0A000418:SSL routines:tls_construct_server_key_exchange:tlsv1 alert unknown ca"
        );
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_dns_lookup_failure() {
        let err = anyhow::anyhow!(
            "POST /v1/chat/completions: error sending request: dns error: failed to lookup address"
        );
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_retries_connection_refused() {
        let err =
            anyhow::anyhow!("POST /v1/chat/completions: error sending request: connection refused");
        assert!(is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_does_not_retry_400_bad_request() {
        // 400 is permanent — handled by the agent loop's malformed-tool-call
        // sanitization, not retried at the client level.
        let err = anyhow::anyhow!(
            "LLM request failed (400 Bad Request): function.arguments must be valid JSON"
        );
        assert!(!is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_does_not_retry_401_unauthorized() {
        let err = anyhow::anyhow!("LLM request failed (401 Unauthorized): invalid api key");
        assert!(!is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_does_not_retry_403_forbidden() {
        let err = anyhow::anyhow!("LLM request failed (403 Forbidden): no access");
        assert!(!is_transient_llm_error(&err));
    }

    #[test]
    fn transient_error_does_not_retry_404_not_found() {
        let err = anyhow::anyhow!("LLM request failed (404 Not Found): model not found");
        assert!(!is_transient_llm_error(&err));
    }
}
