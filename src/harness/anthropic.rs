//! Anthropic Messages API client (`POST /v1/messages`).
//!
//! Implements [`ChatClient`] by translating the harness's OpenAI-shaped
//! internal representation into the Anthropic wire format:
//!
//! - system messages move to the top-level `system` parameter,
//! - assistant `tool_calls` become `tool_use` content blocks (arguments
//!   parsed from the OpenAI JSON string into an object),
//! - `role: "tool"` results become a user message carrying `tool_result`
//!   blocks (consecutive same-role messages are merged — the API requires
//!   strictly alternating roles),
//! - tool schemas become `{name, description, input_schema}`,
//! - `?effort=<v>` enables extended thinking with a budget derived from the
//!   model-defined value (a numeric value is the literal budget).
//!
//! Extended thinking + tool use has a protocol constraint: assistant turns
//! carrying `tool_use` must replay their signed `thinking` block. The
//! signature is captured during streaming and stored per tool-call id
//! ([`AnthropicClient::signatures`]); the next request looks it up and
//! replays `{"type": "thinking", "thinking", "signature"}` ahead of the
//! turn's content blocks.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::Method;
use reqwest::blocking::{Client, Response};

use anyhow::{Context as _, Error, Result, anyhow, bail};
use serde_json::error::Category;
use serde_json::{Value, from_str, json};

use super::auth_provider::AuthProvider;
use super::client::{
    ChatClient, ChatResponse, EarlyToolExecCallback, ModelSpec, StreamCallback, ToolExecCallback,
    Usage, is_transient_llm_error,
};

/// Signature entries kept per client. Each assistant turn adds one entry
/// (~2 KB); the cap bounds a long session's memory without affecting
/// replay — only recent turns need signatures under normal budget pressure.
const MAX_SIGNATURE_ENTRIES: usize = 128;
/// Output budget requested from the model. The Anthropic API requires
/// `max_tokens`; with extended thinking enabled it must exceed the thinking
/// budget, so the budget is added on top.
const BASE_MAX_TOKENS: u64 = 16_384;
/// Smallest legal `budget_tokens` for extended thinking.
const MIN_THINKING_BUDGET: u64 = 1024;

pub struct AnthropicClient {
    base_url: String,
    api_key: String,
    auth_provider: Option<Arc<AuthProvider>>,
    client: Mutex<Client>,
    /// tool_use id → thinking signature, captured while streaming. Insertion
    /// order kept so the oldest entries can be dropped at the cap.
    signatures: Mutex<Vec<(String, String)>>,
}

impl AnthropicClient {
    pub fn new(
        base_url: String,
        api_key: String,
        auth_provider: Option<Arc<AuthProvider>>,
    ) -> Self {
        let client = Self::build_client();
        Self {
            base_url,
            api_key,
            auth_provider,
            client: Mutex::new(client),
            signatures: Mutex::new(Vec::new()),
        }
    }

    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .unwrap_or_else(|_| Client::new())
    }

    fn reset_client(&self) {
        let new_client = Self::build_client();
        if let Ok(mut guard) = self.client.lock() {
            *guard = new_client;
        }
    }

    /// Parse a complete (non-streaming) Messages API response into the
    /// normalized [`ChatResponse`]. Content blocks are folded in order:
    /// `text` blocks concatenate into the response text, `thinking` blocks
    /// into the reasoning (the last block's signature is remembered per
    /// tool-call id for the next request's replay), and `tool_use` blocks
    /// become OpenAI-format tool calls (the `input` object re-serialized as
    /// the arguments string).
    fn parse_message_response(
        &self,
        message: &Value,
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        started: Instant,
    ) -> Result<ChatResponse> {
        if message["type"].as_str() == Some("error") {
            let err_message = message["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            bail!("Anthropic API error: {err_message}");
        }

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut first_signature: Option<String> = None;
        let mut tool_calls: Vec<Value> = Vec::new();

        for block in message["content"].as_array().unwrap_or(&Vec::new()) {
            match block["type"].as_str().unwrap_or("") {
                "text" => {
                    if let Some(text) = block["text"].as_str() {
                        content.push_str(text);
                    }
                }
                "thinking" => {
                    if let Some(thinking) = block["thinking"].as_str() {
                        reasoning.push_str(thinking);
                    }
                    if first_signature.is_none()
                        && let Some(signature) = block["signature"].as_str()
                    {
                        first_signature = Some(signature.to_string());
                    }
                }
                "tool_use" => {
                    tool_calls.push(json!({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": block["input"].to_string(),
                        }
                    }));
                }
                _ => {}
            }
        }

        // Bind the turn's thinking signature to every tool-call id so the
        // next request can replay the signed thinking block.
        if let Some(signature) = first_signature {
            for call in &tool_calls {
                if let Some(id) = call["id"].as_str() {
                    self.remember_signature(id, &signature);
                }
            }
        }

        if let Some(text) = content.strip_prefix("")
            && !text.is_empty()
            && let Some(cb) = on_chunk
        {
            cb(&content);
        }

        let finish_reason = match message["stop_reason"].as_str().unwrap_or("") {
            "" => "stop".to_string(),
            reason => map_stop_reason(reason),
        };

        let usage_value = &message["usage"];
        let usage = Usage {
            input_tokens: usage_value["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage_value["output_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage_value["cache_read_input_tokens"].as_u64().unwrap_or(0),
        };

        let mut tool_results: Vec<String> = Vec::new();
        if let Some(exec) = on_tool_calls
            && !tool_calls.is_empty()
        {
            tool_results = exec(&tool_calls, &finish_reason);
        }

        Ok(ChatResponse {
            content,
            tool_calls,
            finish_reason,
            usage,
            tool_results,
            elapsed_ms: started.elapsed().as_millis(),
            reasoning,
        })
    }

    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        // An endpoint that already carries the full messages path is posted
        // verbatim — the operator specified the exact URL; nothing is
        // appended or rewritten.
        if path == "/messages" && base.ends_with("/messages") {
            return base.to_string();
        }
        // Otherwise the endpoint is the API root: the version segment is
        // part of Anthropic's fixed path (`/v1/messages`), so it is joined
        // here rather than expected in the base.
        let root = base.strip_suffix("/messages").unwrap_or(base);
        format!("{root}/v1{path}")
    }

    /// Remember the thinking signature for a tool-call id (the message the
    /// thinking block belongs to).
    fn remember_signature(&self, tool_use_id: &str, signature: &str) {
        if signature.is_empty() || tool_use_id.is_empty() {
            return;
        }
        let mut signatures = self.signatures.lock().unwrap();
        signatures.push((tool_use_id.to_string(), signature.to_string()));
        if signatures.len() > MAX_SIGNATURE_ENTRIES {
            let excess = signatures.len() - MAX_SIGNATURE_ENTRIES;
            signatures.drain(..excess);
        }
    }

    /// Apply the endpoint auth provider's headers when configured, then the
    /// Anthropic-native headers (`x-api-key` + version) unless the provider
    /// already supplied an `x-api-key` or `Authorization`.
    fn send_request_with_provider(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<&Value>,
        context: &str,
    ) -> Result<Response> {
        let provider_headers = match &self.auth_provider {
            Some(provider) => Some(
                provider
                    .headers()
                    .context("resolve endpoint headers via auth provider")?,
            ),
            None => None,
        };

        let client = self
            .client
            .lock()
            .map_err(|_| anyhow!("client lock poisoned"))?;
        let mut req = client.request(method, url);
        if let Some(headers) = &provider_headers {
            for (name, value) in headers {
                req = req.header(name.as_str(), value.as_str());
            }
        }
        let has_native_auth = provider_headers
            .as_ref()
            .is_some_and(|h| h.contains_key("x-api-key") || h.contains_key("Authorization"));
        if !has_native_auth {
            req = req
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01");
        }
        let req = match body {
            Some(body) => req.header("Content-Type", "application/json").json(body),
            None => req,
        };
        let resp = req.send().with_context(|| context.to_string())?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("LLM request failed ({}): {}", status, text);
        }

        Ok(resp)
    }

    /// Send with transient-failure retry, mirroring the OpenAI client: on
    /// each transient failure the pooled client is replaced so the retry
    /// opens a fresh connection. The response is read and parsed HERE so an
    /// empty body (gateways return bare 200s when an upstream times out) is
    /// treated as transient and retried like any other failure.
    fn send_and_parse_with_retry(
        &self,
        url: &str,
        body: &Value,
        started: Instant,
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        const MAX_ATTEMPTS: u32 = 5;
        const INITIAL_DELAY: Duration = Duration::from_secs(1);
        const MAX_DELAY: Duration = Duration::from_secs(30);

        let mut attempt = 0u32;
        let mut delay = INITIAL_DELAY;
        loop {
            attempt += 1;
            let outcome = self
                .send_request_with_provider(Method::POST, url, Some(body), "/v1/messages")
                .and_then(|resp| {
                    let text = resp.text().context("read Anthropic message response")?;
                    // The gateway sometimes streams even for non-stream
                    // requests (a buffering-timeout fallback on slow
                    // prefills): route by body shape, not by what we asked
                    // for.
                    let trimmed = text.trim_start();
                    if trimmed.starts_with("event:") || trimmed.starts_with("data:") {
                        self.parse_sse_response(
                            &text,
                            on_chunk,
                            on_tool_calls,
                            started,
                            on_early_tool_call,
                        )
                        .map(|response| (response, text))
                    } else {
                        let parse_result = from_str::<Value>(&text)
                            .map_err(|e| (e, text.chars().take(300).collect::<String>()));
                        match parse_result {
                            Ok(message) => self
                                .parse_message_response(&message, on_chunk, on_tool_calls, started)
                                .map(|response| (response, text)),
                            Err((e, excerpt)) => Err(Error::new(e)).with_context(|| {
                                format!("parse Anthropic message response (body: {excerpt:?})")
                            }),
                        }
                    }
                });

            match outcome {
                Ok((response, _)) => {
                    if attempt > 1 {
                        tracing::info!(
                            "harness: LLM request succeeded after {} attempt(s)",
                            attempt
                        );
                    }
                    return Ok(response);
                }
                Err(e) => {
                    let transient = is_transient_llm_error(&e) || is_empty_body_error(&e);
                    if attempt >= MAX_ATTEMPTS || !transient {
                        return Err(e);
                    }
                    tracing::warn!(
                        "harness: LLM request failed (attempt {attempt}/{MAX_ATTEMPTS}), \
                         resetting connection and retrying in {delay:?}: {e:#}"
                    );
                    self.reset_client();
                    thread::sleep(delay);
                    delay = (delay * 2).min(MAX_DELAY);
                }
            }
        }
    }

    /// Parse a complete SSE body (the gateway streams for some requests even
    /// when `stream` is absent) into a [`ChatResponse`].
    fn parse_sse_response(
        &self,
        text: &str,
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        started: Instant,
        on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        let mut state = StreamState::default();
        let mut speculatively_executed: HashSet<usize> = HashSet::new();
        let mut signature_sink: Vec<(String, String)> = Vec::new();

        for line in text.lines() {
            let Some(data) = parse_sse_line(line) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let event: Value = match serde_json::from_str(data) {
                Ok(event) => event,
                Err(_) => continue,
            };
            let done = apply_event(
                &mut state,
                &event,
                on_chunk,
                on_tool_calls,
                on_early_tool_call,
                &mut speculatively_executed,
                &mut signature_sink,
            )?;
            if done {
                break;
            }
        }

        for (id, signature) in signature_sink {
            self.remember_signature(&id, &signature);
        }
        if state.finish_reason.is_empty() {
            state.finish_reason = "stop".to_string();
        }

        Ok(ChatResponse {
            content: state.content,
            tool_calls: state.tool_calls,
            finish_reason: state.finish_reason,
            usage: state.usage,
            tool_results: state.tool_results,
            elapsed_ms: started.elapsed().as_millis(),
            reasoning: state.reasoning,
        })
    }

    /// Build the Messages API request body from the harness's OpenAI-shaped
    /// message history.
    fn build_body(
        &self,
        spec: &ModelSpec,
        messages: &[Value],
        tools: &[Value],
        signatures: &Mutex<Vec<(String, String)>>,
    ) -> Result<Value> {
        let effort = spec.effort.as_deref();
        let thinking_budget = effort.map(thinking_budget);
        let max_tokens = BASE_MAX_TOKENS + thinking_budget.unwrap_or(0);

        let mut body = json!({
            "model": spec.model,
            "max_tokens": max_tokens,
            "stream": true,
        });

        let converted = convert_messages(messages, effort.is_some(), signatures)?;
        let system: Vec<String> = converted
            .system
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();

        if !tools.is_empty() {
            let mut translated = translate_tools(tools);
            // Prompt caching on the Anthropic protocol is opt-in via
            // `cache_control` breakpoints (max 4 per request). Mark the
            // stable prefix: the tool definitions, the system prompt, and
            // the last message (a moving breakpoint that covers the whole
            // conversation as it grows — unlike OpenAI, nothing is cached
            // unless a breakpoint requests it).
            if let Some(last_tool) = translated.last_mut()
                && let Some(schema) = last_tool.as_object_mut()
            {
                schema.insert("cache_control".into(), json!({"type": "ephemeral"}));
            }
            body["tools"] = json!(translated);
        }
        if !system.is_empty() {
            body["system"] = json!([{
                "type": "text",
                "text": system.join("\n\n"),
                "cache_control": {"type": "ephemeral"},
            }]);
        }

        let mut messages = converted.messages;
        if let Some(last_message) = messages.last_mut()
            && let Some(blocks) = last_message["content"].as_array_mut()
            && let Some(last_block) = blocks.last_mut()
            && let Some(block) = last_block.as_object_mut()
        {
            block.insert("cache_control".into(), json!({"type": "ephemeral"}));
        }
        body["messages"] = json!(messages);

        if let Some(budget) = thinking_budget {
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
        }
        Ok(body)
    }
}

/// The extended-thinking budget for a model-defined effort value: the common
/// keywords map to graduated budgets, a numeric value is the literal budget,
/// and anything else falls back to the high tier. The API requires the
/// budget to exceed [`MIN_THINKING_BUDGET`].
fn thinking_budget(effort: &str) -> u64 {
    let budget = match effort.to_lowercase().as_str() {
        "low" => 4096,
        "medium" => 8192,
        "high" => 16384,
        "max" => 32768,
        other => other.parse::<u64>().unwrap_or(16384),
    };
    budget.max(MIN_THINKING_BUDGET)
}

/// Translate OpenAI tool schemas to Anthropic's `{name, description,
/// input_schema}` shape.
fn translate_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            json!({
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"].clone(),
            })
        })
        .collect()
}

/// The converted message list plus the collected system prompt.
struct ConvertedMessages {
    system: Vec<String>,
    messages: Vec<Value>,
}

/// Convert the OpenAI-shaped history to Anthropic messages: system messages
/// are lifted to the `system` parameter, assistant tool calls become
/// `tool_use` blocks, `role: "tool"` results become user-carried
/// `tool_result` blocks, and consecutive same-role messages are merged (the
/// API requires strictly alternating roles).
fn convert_messages(
    messages: &[Value],
    replay_thinking: bool,
    signatures: &Mutex<Vec<(String, String)>>,
) -> Result<ConvertedMessages> {
    let mut system = Vec::new();
    let mut converted: Vec<Value> = Vec::new();

    // Append a block list to the last message if it has the same role,
    // otherwise start a new message.
    let push = |role: &str, mut blocks: Vec<Value>, converted: &mut Vec<Value>| {
        if blocks.is_empty() {
            return;
        }
        if let Some(last) = converted.last_mut()
            && last["role"] == json!(role)
        {
            if let Some(last_blocks) = last["content"].as_array_mut() {
                last_blocks.append(&mut blocks);
            }
            return;
        }
        converted.push(json!({ "role": role, "content": blocks }));
    };

    let last_index = messages.len().saturating_sub(1);
    for (index, message) in messages.iter().enumerate() {
        let role = message["role"].as_str().unwrap_or("user");
        match role {
            // A system message in the FINAL position is volatile per-turn
            // content (the agent loop appends the todo checklist there every
            // turn). Lifting it into `system` would change the top-level
            // prefix every turn and defeat prompt caching entirely — keep it
            // at the end, as user content, exactly where OpenAI-style
            // requests place it for the same reason.
            "system" if index == last_index => {
                let text = message["content"].as_str().unwrap_or("");
                push(
                    "user",
                    vec![json!({ "type": "text", "text": text })],
                    &mut converted,
                );
            }
            "system" => {
                system.push(message["content"].as_str().unwrap_or("").to_string());
            }
            "user" => {
                let text = message["content"].as_str().unwrap_or("");
                push(
                    "user",
                    vec![json!({ "type": "text", "text": text })],
                    &mut converted,
                );
            }
            "assistant" => {
                let tool_calls = message["tool_calls"].as_array();
                let mut blocks: Vec<Value> = Vec::new();

                // Extended thinking + tool use: the signed thinking block
                // must lead the assistant turn.
                if replay_thinking
                    && tool_calls.is_some()
                    && let Some(first) = tool_calls.unwrap().first()
                    && let Some(id) = first["id"].as_str()
                    && let Some((_, signature)) = signatures
                        .lock()
                        .unwrap()
                        .iter()
                        .rev()
                        .find(|(known, _)| known == id)
                    && let Some(thinking) = message["reasoning_content"].as_str()
                    && !thinking.is_empty()
                {
                    blocks.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": signature,
                    }));
                }

                if let Some(text) = message["content"].as_str()
                    && !text.is_empty()
                {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
                if let Some(tool_calls) = tool_calls {
                    for call in tool_calls {
                        let arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
                        let input: Value =
                            serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": input,
                        }));
                    }
                }
                push("assistant", blocks, &mut converted);
            }
            "tool" => {
                let text = message["content"].as_str().unwrap_or("");
                push(
                    "user",
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": text,
                    })],
                    &mut converted,
                );
            }
            other => bail!("unsupported message role in history: {other}"),
        }
    }

    Ok(ConvertedMessages {
        system,
        messages: converted,
    })
}

/// Map an Anthropic `stop_reason` to the harness's finish-reason vocabulary.
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "tool_use" => "tool_calls",
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        other => other,
    }
    .to_string()
}

/// Recognize the empty-body failure mode: some gateways return a bare 200
/// with no body when their upstream times out. Retrying is meaningful;
/// parsing the empty text is not.
fn is_empty_body_error(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|e| e.downcast_ref::<serde_json::Error>())
        .any(|serde_err| matches!(serde_err.classify(), Category::Eof | Category::Syntax))
}

/// Accumulates one streamed response. Tool calls are stored compactly in
/// emission order — content-block indices count ALL blocks (text, thinking,
/// tool_use…), so a block-index → tool-call-position map routes deltas.
#[derive(Default)]
struct StreamState {
    content: String,
    reasoning: String,
    tool_calls: Vec<Value>,
    finish_reason: String,
    usage: Usage,
    pending_signature: String,
    signature_bound: bool,
    block_types: HashMap<u64, String>,
    block_tool_positions: HashMap<u64, usize>,
    tool_results: Vec<String>,
    tool_results_taken: bool,
}

/// Apply one SSE `data:` payload to the stream state. Returns true when the
/// stream is complete (`message_stop`).
fn apply_event(
    state: &mut StreamState,
    data: &Value,
    on_chunk: Option<&StreamCallback>,
    on_tool_calls: Option<&ToolExecCallback<'_>>,
    on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    speculatively_executed: &mut HashSet<usize>,
    signature_sink: &mut Vec<(String, String)>,
) -> Result<bool> {
    let event_type = data["type"].as_str().unwrap_or("");
    match event_type {
        "message_start" => {
            let usage = &data["message"]["usage"];
            state.usage.input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
            state.usage.cached_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
        }
        "content_block_start" => {
            let index = data["index"].as_u64().unwrap_or(0);
            let block = &data["content_block"];
            let block_type = block["type"].as_str().unwrap_or("");
            state.block_types.insert(index, block_type.to_string());
            if block_type == "tool_use" {
                let position = state.tool_calls.len();
                state.tool_calls.push(json!({
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": ""}
                }));
                state.block_tool_positions.insert(index, position);
                if !state.signature_bound && !state.pending_signature.is_empty() {
                    if let Some(id) = block["id"].as_str() {
                        signature_sink.push((id.to_string(), state.pending_signature.clone()));
                    }
                    state.signature_bound = true;
                }
            }
        }
        "content_block_delta" => {
            let index = data["index"].as_u64().unwrap_or(0);
            let delta = &data["delta"];
            match delta["type"].as_str().unwrap_or("") {
                "text_delta" => {
                    if let Some(text) = delta["text"].as_str() {
                        state.content.push_str(text);
                        if let Some(cb) = on_chunk {
                            cb(text);
                        }
                    }
                }
                "thinking_delta" => {
                    if let Some(thinking) = delta["thinking"].as_str() {
                        state.reasoning.push_str(thinking);
                    }
                }
                "signature_delta" => {
                    if let Some(signature) = delta["signature"].as_str() {
                        state.pending_signature.push_str(signature);
                    }
                }
                "input_json_delta" => {
                    if let Some(partial) = delta["partial_json"].as_str()
                        && let Some(position) = state.block_tool_positions.get(&index)
                        && let Some(call) = state.tool_calls.get_mut(*position)
                    {
                        let current = call["function"]["arguments"].as_str().unwrap_or("");
                        call["function"]["arguments"] = json!(format!("{current}{partial}"));
                    }
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            let index = data["index"].as_u64().unwrap_or(0);
            if state.block_types.get(&index).map(String::as_str) == Some("tool_use")
                && let Some(position) = state.block_tool_positions.get(&index)
            {
                try_early_exec(*position, state, speculatively_executed, on_early_tool_call);
            }
        }
        "message_delta" => {
            if let Some(reason) = data["delta"]["stop_reason"].as_str()
                && !reason.is_empty()
            {
                state.finish_reason = map_stop_reason(reason);
                if let Some(exec) = on_tool_calls
                    && !state.tool_calls.is_empty()
                    && !state.tool_results_taken
                {
                    state.tool_results = exec(&state.tool_calls.clone(), &state.finish_reason);
                    state.tool_results_taken = true;
                }
            }
            if let Some(output) = data["usage"]["output_tokens"].as_u64() {
                state.usage.output_tokens = output;
            }
        }
        "message_stop" => return Ok(true),
        "error" => {
            let message = data["error"]["message"].as_str().unwrap_or("unknown error");
            bail!("Anthropic stream error: {message}");
        }
        _ => {}
    }
    Ok(false)
}

/// Fire speculative execution for a tool call whose arguments are complete.
fn try_early_exec(
    index: usize,
    state: &StreamState,
    executed: &mut HashSet<usize>,
    callback: Option<&EarlyToolExecCallback<'_>>,
) {
    if executed.contains(&index) {
        return;
    }
    let Some(callback) = callback else {
        return;
    };
    let Some(call) = state.tool_calls.get(index) else {
        return;
    };
    let name = call["function"]["name"].as_str().unwrap_or("");
    if name.is_empty() {
        return;
    }
    let arguments = call["function"]["arguments"].as_str().unwrap_or("");
    if arguments.is_empty() || from_str::<Value>(arguments).is_err() {
        return;
    }
    executed.insert(index);
    let _ = callback(index, call);
}

fn parse_sse_line(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed.strip_prefix("data:").map(str::trim_start)
}

impl ChatClient for AnthropicClient {
    fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        let spec = ModelSpec::parse(model);
        let body = self.build_body(&spec, messages, tools, &self.signatures)?;
        let url = self.url("/messages");
        let started = Instant::now();

        // The gateway may answer a non-stream request with an SSE stream
        // (buffering fallback on slow prefills) — send_and_parse routes by
        // body shape, and the SSE path uses speculative execution.
        self.send_and_parse_with_retry(
            &url,
            &body,
            started,
            on_chunk,
            on_tool_calls,
            on_early_tool_call,
        )
    }

    fn list_models(&self) -> Result<Vec<String>> {
        let url = self.url("/models");
        let resp = self.send_request_with_provider(Method::GET, &url, None, "GET /v1/models")?;
        let body: Value = resp.json().context("parse /v1/models response")?;
        Ok(body["data"]
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn preserves_reasoning(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    fn system(text: &str) -> Value {
        json!({"role": "system", "content": text})
    }

    fn assistant_tool_call(id: &str, name: &str, arguments: &str) -> Value {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }]
        })
    }

    fn tool_result(id: &str, text: &str) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": text})
    }

    fn signatures_of(pairs: &[(&str, &str)]) -> Mutex<Vec<(String, String)>> {
        Mutex::new(
            pairs
                .iter()
                .map(|(id, sig)| (id.to_string(), sig.to_string()))
                .collect(),
        )
    }

    #[test]
    fn build_body_lifts_system_and_translates_tools() {
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("claude-sonnet-4");
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let messages = vec![system("be brief"), user("hello")];
        let body = client
            .build_body(&spec, &messages, &tools, &Mutex::new(Vec::new()))
            .unwrap();

        assert_eq!(body["model"], json!("claude-sonnet-4"));
        assert_eq!(
            body["system"],
            json!([{
                "type": "text",
                "text": "be brief",
                "cache_control": {"type": "ephemeral"}
            }])
        );
        assert_eq!(body["max_tokens"], json!(BASE_MAX_TOKENS));
        assert!(body.get("thinking").is_none(), "no effort = no thinking");
        assert_eq!(
            body["tools"],
            json!([{
                "name": "shell",
                "description": "Run a command.",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"}
            }])
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{
                "type": "text",
                "text": "hello",
                "cache_control": {"type": "ephemeral"}
            }]}]),
            "the last message carries the moving cache breakpoint"
        );
        assert_eq!(
            body["tools"][0]["cache_control"],
            json!({"type": "ephemeral"}),
            "the last tool carries a cache breakpoint"
        );
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral"}),
            "the system prompt carries a cache breakpoint"
        );
    }

    #[test]
    fn build_body_converts_tool_calls_and_results() {
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("claude-sonnet-4");
        let messages = vec![
            user("list files"),
            assistant_tool_call("call_1", "shell", "{\"command\":\"ls\"}"),
            tool_result("call_1", "file.rs\nmain.rs"),
        ];
        let body = client
            .build_body(&spec, &messages, &[], &Mutex::new(Vec::new()))
            .unwrap();

        let messages = body["messages"].as_array().unwrap();
        // user, assistant (tool_use), user (tool_result) — strictly
        // alternating, three messages.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            json!([{
                "type": "tool_use",
                "id": "call_1",
                "name": "shell",
                "input": {"command": "ls"}
            }])
        );
        assert_eq!(
            messages[2]["role"], "user",
            "tool results ride in a user message"
        );
        assert_eq!(
            messages[2]["content"],
            json!([{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "file.rs\nmain.rs",
                "cache_control": {"type": "ephemeral"}
            }])
        );
    }

    #[test]
    fn build_body_merges_consecutive_tool_results() {
        // Parallel tool calls produce several tool messages in a row; the
        // API requires them in ONE user message.
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("claude-sonnet-4");
        let messages = vec![
            assistant_tool_call("c1", "read", "{}"),
            tool_result("c1", "a"),
            tool_result("c1", "b"),
            tool_result("c1", "c"),
        ];
        let body = client
            .build_body(&spec, &messages, &[], &Mutex::new(Vec::new()))
            .unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "consecutive tool results merge");
        let blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[2]["content"], json!("c"));
    }

    #[test]
    fn build_body_keeps_the_trailing_todo_system_message_out_of_system() {
        // The agent loop appends the todo checklist as a system message at
        // the END of every turn, and its content changes as items complete.
        // Lifting it into the top-level `system` would rewrite the cached
        // prefix every turn — a full cache miss per turn (observed as
        // cached=0 on Anthropic endpoints). It must stay at the end as user
        // content.
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let signatures = Mutex::new(Vec::new());
        let spec = ModelSpec::parse("claude-sonnet-4");
        let messages = vec![
            system("stable system prompt"),
            user("do the task"),
            assistant_tool_call("call_1", "shell", "{}"),
            tool_result("call_1", "done"),
            system("## Todo (current state, set by your todo tool)\n\n0. [x] step one"),
        ];
        let body = client
            .build_body(&spec, &messages, &[], &signatures)
            .unwrap();

        // `system` carries ONLY the stable prompt.
        assert_eq!(
            body["system"],
            json!([{
                "type": "text",
                "text": "stable system prompt",
                "cache_control": {"type": "ephemeral"}
            }])
        );

        // The todo rides at the end of the final user message (merged with
        // the tool result), where it cannot invalidate the cached prefix.
        let last = body["messages"].as_array().unwrap().last().unwrap();
        assert_eq!(last["role"], "user");
        let blocks = last["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[1]["type"], "text");
        assert!(
            blocks[1]["text"]
                .as_str()
                .unwrap()
                .starts_with("## Todo (current state"),
            "the volatile todo stays in the trailing user message: {blocks:?}"
        );
    }

    #[test]
    fn build_body_replays_signed_thinking_for_tool_turns() {
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let signatures = signatures_of(&[("call_1", "sig==abc")]);
        let spec = ModelSpec::parse("claude-sonnet-4?effort=high");
        let assistant = json!({
            "role": "assistant",
            "content": "On it.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }],
            "reasoning_content": "I should look at the files."
        });
        let messages = vec![user("go"), assistant, tool_result("call_1", "done")];
        let body = client
            .build_body(&spec, &messages, &[], &signatures)
            .unwrap();

        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 16384}),
            "effort=high maps to the high thinking budget"
        );
        assert_eq!(
            body["max_tokens"],
            json!(BASE_MAX_TOKENS + 16384),
            "max_tokens must exceed the thinking budget"
        );
        let assistant_message = &body["messages"][1];
        let blocks = assistant_message["content"].as_array().unwrap();
        assert_eq!(
            blocks[0],
            json!({"type": "thinking", "thinking": "I should look at the files.", "signature": "sig==abc"}),
            "the signed thinking block leads the tool-use turn"
        );
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], json!("On it."));
        assert_eq!(blocks[2]["type"], "tool_use");
    }

    #[test]
    fn build_body_omits_thinking_without_signature_or_effort() {
        let client = AnthropicClient::new("http://endpoint.example".into(), "key".into(), None);
        let signatures = signatures_of(&[]);
        let spec = ModelSpec::parse("claude-sonnet-4?effort=high");
        let assistant = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{}"}
            }],
            "reasoning_content": "unsigned thinking"
        });
        let messages = vec![user("go"), assistant, tool_result("call_1", "done")];
        let body = client
            .build_body(&spec, &messages, &[], &signatures)
            .unwrap();
        let blocks = body["messages"][1]["content"].as_array().unwrap();
        assert!(
            blocks.iter().all(|b| b["type"] != "thinking"),
            "no signature, no replay: {blocks:?}"
        );

        // Without effort, thinking is disabled entirely — even a stored
        // signature is not replayed.
        let signatures = signatures_of(&[("call_1", "sig==abc")]);
        let spec = ModelSpec::parse("claude-sonnet-4");
        let body = client
            .build_body(&spec, &messages, &[], &signatures)
            .unwrap();
        assert!(body.get("thinking").is_none());
        let blocks = body["messages"][1]["content"].as_array().unwrap();
        assert!(blocks.iter().all(|b| b["type"] != "thinking"));
    }

    #[test]
    fn thinking_budget_maps_keywords_and_numbers() {
        assert_eq!(thinking_budget("low"), 4096);
        assert_eq!(thinking_budget("medium"), 8192);
        assert_eq!(thinking_budget("high"), 16384);
        assert_eq!(thinking_budget("max"), 32768);
        assert_eq!(thinking_budget("8192"), 8192, "numeric = literal budget");
        assert_eq!(
            thinking_budget("whatever"),
            16384,
            "unknown values fall back to the high tier"
        );
        assert_eq!(
            thinking_budget("1"),
            MIN_THINKING_BUDGET,
            "the API floor applies"
        );
    }

    #[test]
    fn parse_sse_response_parses_a_streamed_body() {
        // The gateway sometimes streams even for non-stream requests: the
        // body-shape routing must fall back to the SSE parser and produce
        // the same normalized response the JSON path would.
        let client = AnthropicClient::new("http://endpoint.example".into(), "k".into(), None);
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":40}}}\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Working.\"}}\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"shell\"}}\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
        );

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_for_cb = Arc::clone(&chunks);
        let on_chunk: &StreamCallback = &move |text: &str| {
            chunks_for_cb.lock().unwrap().push(text.to_string());
        };

        let started = Instant::now();
        let response = client
            .parse_sse_response(sse, Some(on_chunk), None, started, None)
            .unwrap();

        assert_eq!(response.content, "Working.");
        assert_eq!(response.finish_reason, "tool_calls");
        assert_eq!(response.usage.input_tokens, 50);
        assert_eq!(response.usage.cached_tokens, 40);
        assert_eq!(response.usage.output_tokens, 9);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0]["id"], json!("t1"));
        assert_eq!(
            response.tool_calls[0]["function"]["arguments"],
            json!("{\"command\":\"ls\"}")
        );
        assert_eq!(*chunks.lock().unwrap(), vec!["Working.".to_string()]);
    }

    #[test]
    fn parse_sse_response_binds_thinking_signatures_for_replay() {
        let client = AnthropicClient::new("http://endpoint.example".into(), "k".into(), None);
        let sse = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"thought\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig==\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_9\",\"name\":\"shell\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        client
            .parse_sse_response(sse, None, None, Instant::now(), None)
            .unwrap();

        // The signature is bound for replay in the next request.
        let signatures = client.signatures.lock().unwrap();
        assert_eq!(
            signatures.as_slice(),
            vec![("tu_9".to_string(), "sig==".to_string())]
        );
    }

    #[test]
    fn url_joins_anthropic_paths_across_base_forms() {
        // Bare host: /v1 inserted.
        let bare = AnthropicClient::new("https://api.example".into(), "k".into(), None);
        assert_eq!(bare.url("/messages"), "https://api.example/v1/messages");
        assert_eq!(bare.url("/models"), "https://api.example/v1/models");

        // Gateway prefix base: the path is kept, the /v1 subpath appended.
        let prefix =
            AnthropicClient::new("https://gateway.example/prefix".into(), "k".into(), None);
        assert_eq!(
            prefix.url("/messages"),
            "https://gateway.example/prefix/v1/messages"
        );

        // A full messages URL as the endpoint is posted VERBATIM — the
        // operator specified the exact URL; nothing is appended.
        let full = AnthropicClient::new(
            "https://gateway.example/prefix/v1/messages".into(),
            "k".into(),
            None,
        );
        assert_eq!(
            full.url("/messages"),
            "https://gateway.example/prefix/v1/messages"
        );

        // Same for a /messages path that isn't the /v1 form.
        let plain = AnthropicClient::new(
            "https://proxy.example/prefix/messages".into(),
            "k".into(),
            None,
        );
        assert_eq!(
            plain.url("/messages"),
            "https://proxy.example/prefix/messages"
        );

        // Trailing slash trimmed before joining.
        let slash =
            AnthropicClient::new("https://gateway.example/prefix/".into(), "k".into(), None);
        assert_eq!(
            slash.url("/messages"),
            "https://gateway.example/prefix/v1/messages"
        );
    }

    #[test]
    fn map_stop_reason_translates_the_vocabulary() {
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("pause_turn"), "pause_turn");
    }
}
