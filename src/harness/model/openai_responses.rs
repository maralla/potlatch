//! OpenAI Responses API client (`POST /v1/responses`, stateless mode).
//!
//! Implements [`ChatClient`] by translating the harness's OpenAI-shaped
//! internal representation into Responses items:
//!
//! - system messages (except a trailing volatile one — the todo checklist)
//!   move to the top-level `instructions` parameter,
//! - assistant `tool_calls` become `function_call` items, `role: "tool"`
//!   results become `function_call_output` items keyed by `call_id`,
//! - tool schemas are flattened: chat's nested `function` object becomes a
//!   top-level `{type: "function", name, description, parameters}`,
//! - `?effort=<v>` maps to `reasoning: {"effort": <v>}`,
//! - `store: false` — the harness owns the conversation context (it is
//!   resent in full every request), so client-side compaction stays
//!   authoritative. Crash recovery rides on the persisted context file,
//!   not on provider-side thread storage.
//!
//! Requests are non-streaming: the complete message is parsed from one JSON
//! response (the same choice as the Anthropic client — measured against the
//! real gateway, streaming bypassed prompt caching entirely).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Error, Result, anyhow, bail};
use serde_json::error::Category;
use serde_json::{Value, from_str, json};

use crate::harness::auth_provider::AuthProvider;
use crate::harness::client::{
    ChatClient, ChatResponse, EarlyToolExecCallback, ModelSpec, StreamCallback, ToolExecCallback,
    Usage, is_transient_llm_error, join_api_path,
};
use reqwest::Method;
use reqwest::blocking::{Client, Response};

pub struct OpenAiResponsesClient {
    base_url: String,
    api_key: String,
    auth_provider: Option<Arc<AuthProvider>>,
    client: Mutex<Client>,
}

impl OpenAiResponsesClient {
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

    /// Join the API base onto a Responses path. A complete operation URL as
    /// the endpoint is posted verbatim; otherwise the path is appended (with
    /// `/v1` inserted for bare hosts, per the shared joining convention).
    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        if path == "/responses" && base.ends_with("/responses") {
            return base.to_string();
        }
        let root = base.strip_suffix("/responses").unwrap_or(base);
        join_api_path(root, path)
    }

    /// Apply the endpoint auth provider's headers when configured; the
    /// default `Authorization: Bearer {key}` is only added when the provider
    /// did not supply an `Authorization` header.
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
        if provider_headers
            .as_ref()
            .is_none_or(|h| !h.contains_key("Authorization"))
        {
            req = req.header("Authorization", format!("Bearer {}", self.api_key));
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

    /// Send and parse with transient-failure retry, mirroring the other
    /// clients. An empty body (gateways return bare 200s when an upstream
    /// times out) is classified as transient and retried.
    fn send_and_parse_with_retry(
        &self,
        url: &str,
        body: &Value,
        started: Instant,
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        const MAX_ATTEMPTS: u32 = 5;
        const INITIAL_DELAY: Duration = Duration::from_secs(1);
        const MAX_DELAY: Duration = Duration::from_secs(30);

        let mut attempt = 0u32;
        let mut delay = INITIAL_DELAY;
        loop {
            attempt += 1;
            let outcome = self
                .send_request_with_provider(Method::POST, url, Some(body), "/v1/responses")
                .and_then(|resp| {
                    let text = resp.text().context("read Responses API response")?;
                    let parse_result = from_str::<Value>(&text)
                        .map_err(|e| (e, text.chars().take(300).collect::<String>()));
                    match parse_result {
                        Ok(message) => self
                            .parse_response(&message, on_chunk, on_tool_calls, started)
                            .map(|response| (response, text)),
                        Err((e, excerpt)) => Err(Error::new(e)).with_context(|| {
                            format!("parse Responses API response (body: {excerpt:?})")
                        }),
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
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(MAX_DELAY);
                }
            }
        }
    }

    /// Build the Responses request body from the harness's OpenAI-shaped
    /// message history. Stateless (`store: false`): the full history is
    /// resent every request, so client-side compaction stays authoritative.
    fn build_body(&self, spec: &ModelSpec, messages: &[Value], tools: &[Value]) -> Result<Value> {
        let mut body = json!({
            "model": spec.model,
            "store": false,
        });

        let converted = convert_messages(messages)?;
        let instructions: Vec<String> = converted
            .instructions
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        if !instructions.is_empty() {
            body["instructions"] = json!(instructions.join("\n\n"));
        }
        body["input"] = json!(converted.input);

        if !tools.is_empty() {
            body["tools"] = json!(translate_tools(tools));
        }
        // Model-defined effort maps to the Responses reasoning parameter.
        // `summary: "auto"` requests the reasoning summary items — without
        // it the model's thinking never appears in the response output
        // (raw reasoning content is not returned on this surface).
        if let Some(effort) = spec.effort.as_deref() {
            body["reasoning"] = json!({"effort": effort, "summary": "auto"});
        }
        Ok(body)
    }

    /// Parse a complete Responses object into the normalized
    /// [`ChatResponse`]. Output items fold in order: `reasoning` items into
    /// the reasoning string, `message` items into the response text, and
    /// `function_call` items into OpenAI-format tool calls.
    fn parse_response(
        &self,
        message: &Value,
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        started: Instant,
    ) -> Result<ChatResponse> {
        if message["status"].as_str() == Some("failed") {
            let err_message = message["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            bail!("Responses API error: {err_message}");
        }

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();

        for item in message["output"].as_array().unwrap_or(&Vec::new()) {
            match item["type"].as_str().unwrap_or("") {
                "message" => {
                    for block in item["content"].as_array().unwrap_or(&Vec::new()) {
                        if let Some(text) = block["text"].as_str() {
                            content.push_str(text);
                        }
                    }
                }
                "reasoning" => {
                    for block in item["content"].as_array().unwrap_or(&Vec::new()) {
                        if let Some(text) = block["text"].as_str() {
                            reasoning.push_str(text);
                        }
                    }
                    for block in item["summary"].as_array().unwrap_or(&Vec::new()) {
                        if let Some(text) = block["text"].as_str() {
                            reasoning.push_str(text);
                        }
                    }
                }
                "function_call" => {
                    tool_calls.push(json!({
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        }
                    }));
                }
                _ => {}
            }
        }
        if let Some(cb) = on_chunk
            && !content.is_empty()
        {
            cb(&content);
        }

        // Tool calls present → the model ended its turn requesting execution.
        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls".to_string()
        } else if message["incomplete_details"]["reason"].as_str() == Some("max_output_tokens") {
            "length".to_string()
        } else {
            "stop".to_string()
        };

        let usage_value = &message["usage"];
        let usage = Usage {
            input_tokens: usage_value["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage_value["output_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage_value["input_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
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
}

/// Translate OpenAI chat tool schemas to the flattened Responses shape.
fn translate_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            json!({
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"].clone(),
            })
        })
        .collect()
}

/// The converted request: lifted `instructions` plus the typed input items.
struct ConvertedMessages {
    instructions: Vec<String>,
    input: Vec<Value>,
}

/// Convert the OpenAI-shaped history to Responses input items. System
/// messages that are NOT in the final position are lifted to
/// `instructions`; a system message in the final position is volatile
/// per-turn content (the todo checklist) and stays at the end as a user
/// message — same prefix-stability reasoning as the Anthropic converter.
fn convert_messages(messages: &[Value]) -> Result<ConvertedMessages> {
    let mut instructions = Vec::new();
    let mut converted: Vec<Value> = Vec::new();
    let last_index = messages.len().saturating_sub(1);

    let push = |role: &str, text: &str, converted: &mut Vec<Value>| {
        if text.is_empty() {
            return;
        }
        if let Some(last) = converted.last_mut()
            && last["type"] == json!("message")
            && last["role"] == json!(role)
        {
            // Merge consecutive same-role messages (text concat).
            let existing = last["content"].as_str().unwrap_or("").to_string();
            last["content"] = json!(format!("{existing}\n{text}"));
            return;
        }
        converted.push(json!({ "type": "message", "role": role, "content": text }));
    };

    for (index, message) in messages.iter().enumerate() {
        let role = message["role"].as_str().unwrap_or("user");
        match role {
            "system" if index == last_index => {
                let text = message["content"].as_str().unwrap_or("");
                push("user", text, &mut converted);
            }
            "system" => {
                instructions.push(message["content"].as_str().unwrap_or("").to_string());
            }
            "user" => {
                push(
                    "user",
                    message["content"].as_str().unwrap_or(""),
                    &mut converted,
                );
            }
            "assistant" => {
                if let Some(text) = message["content"].as_str()
                    && !text.is_empty()
                {
                    push("assistant", text, &mut converted);
                }
                for call in message["tool_calls"].as_array().unwrap_or(&Vec::new()) {
                    converted.push(json!({
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }));
                }
            }
            "tool" => {
                converted.push(json!({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }));
            }
            other => bail!("unsupported message role in history: {other}"),
        }
    }

    Ok(ConvertedMessages {
        instructions,
        input: converted,
    })
}

/// Recognize the empty-body failure mode: gateways return a bare 200 with
/// no body when their upstream times out. Retrying is meaningful.
fn is_empty_body_error(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|e| e.downcast_ref::<serde_json::Error>())
        .any(|serde_err| matches!(serde_err.classify(), Category::Eof | Category::Syntax))
}

impl ChatClient for OpenAiResponsesClient {
    fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        on_chunk: Option<&StreamCallback>,
        on_tool_calls: Option<&ToolExecCallback<'_>>,
        // Speculative mid-stream execution does not apply: responses arrive
        // complete.
        _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
    ) -> Result<ChatResponse> {
        let spec = ModelSpec::parse(model);
        let body = self.build_body(&spec, messages, tools)?;
        let url = self.url("/responses");
        let started = Instant::now();

        self.send_and_parse_with_retry(&url, &body, started, on_chunk, on_tool_calls)
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

    #[test]
    fn build_body_lifts_instructions_and_flattens_tools() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("gpt-5");
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let messages = vec![system("be brief"), user("hello")];
        let body = client.build_body(&spec, &messages, &tools).unwrap();

        assert_eq!(body["model"], json!("gpt-5"));
        assert_eq!(body["instructions"], json!("be brief"));
        assert_eq!(
            body["store"],
            json!(false),
            "stateless: the harness owns context"
        );
        assert!(body.get("stream").is_none(), "non-streaming requests");
        assert!(
            body.get("max_output_tokens").is_none(),
            "backend default applies"
        );
        assert!(
            body.get("reasoning").is_none(),
            "no effort = no reasoning param"
        );
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "shell",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {}}
            }])
        );
        assert_eq!(
            body["input"],
            json!([{"type": "message", "role": "user", "content": "hello"}])
        );
    }

    #[test]
    fn build_body_converts_tool_calls_and_results() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("gpt-5");
        let messages = vec![
            user("list files"),
            assistant_tool_call("call_1", "shell", "{\"command\":\"ls\"}"),
            tool_result("call_1", "file.rs\nmain.rs"),
        ];
        let body = client.build_body(&spec, &messages, &[]).unwrap();

        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(
            input[1],
            json!({"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"command\":\"ls\"}"})
        );
        assert_eq!(
            input[2],
            json!({"type": "function_call_output", "call_id": "call_1", "output": "file.rs\nmain.rs"})
        );
    }

    #[test]
    fn build_body_keeps_the_trailing_todo_out_of_instructions() {
        // The per-turn todo checklist is volatile: it must ride at the end
        // of the input, never in `instructions` (which would rewrite the
        // cached prefix every turn).
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("gpt-5");
        let messages = vec![
            system("stable system prompt"),
            user("do the task"),
            assistant_tool_call("call_1", "shell", "{}"),
            tool_result("call_1", "done"),
            system("## Todo (current state, set by your todo tool)\n\n0. [x] step one"),
        ];
        let body = client.build_body(&spec, &messages, &[]).unwrap();

        assert_eq!(body["instructions"], json!("stable system prompt"));
        let last = body["input"].as_array().unwrap().last().unwrap();
        assert_eq!(last["role"], "user");
        assert!(
            last["content"]
                .as_str()
                .unwrap()
                .starts_with("## Todo (current state")
        );
    }

    #[test]
    fn build_body_maps_effort_to_reasoning() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("gpt-5?effort=high");
        let body = client.build_body(&spec, &[user("hello")], &[]).unwrap();
        assert_eq!(
            body["reasoning"],
            json!({"effort": "high", "summary": "auto"}),
            "summaries requested so reasoning reaches the log"
        );
    }

    #[test]
    fn build_body_merges_consecutive_user_messages() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let spec = ModelSpec::parse("gpt-5");
        let messages = vec![user("part one"), user("part two")];
        let body = client.build_body(&spec, &messages, &[]).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1, "consecutive same-role messages merge");
        assert_eq!(input[0]["content"], json!("part one\npart two"));
    }

    #[test]
    fn parse_response_folds_output_items() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let message = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}],
                 "content": [{"type": "reasoning_text", "text": "because"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Working."}]},
                {"type": "function_call", "call_id": "call_1", "name": "shell",
                 "arguments": "{\"command\":\"ls\"}"}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 9,
                      "input_tokens_details": {"cached_tokens": 90}}
        });

        let response = client
            .parse_response(&message, None, None, Instant::now())
            .unwrap();
        assert_eq!(response.content, "Working.");
        assert_eq!(response.reasoning, "becausewhy");
        assert_eq!(response.finish_reason, "tool_calls");
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.cached_tokens, 90);
        assert_eq!(response.usage.output_tokens, 9);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0]["id"], json!("call_1"));
        assert_eq!(
            response.tool_calls[0]["function"]["arguments"],
            json!("{\"command\":\"ls\"}"),
            "the arguments object is re-serialized as the OpenAI-format string"
        );
    }

    #[test]
    fn parse_response_maps_length_and_surfaces_failures() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let message = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });
        let response = client
            .parse_response(&message, None, None, Instant::now())
            .unwrap();
        assert_eq!(response.finish_reason, "length");
        assert_eq!(response.content, "partial");

        let failed = json!({
            "status": "failed",
            "error": {"code": "server_error", "message": "upstream boom"},
            "output": []
        });
        let error = client
            .parse_response(&failed, None, None, Instant::now())
            .unwrap_err();
        assert!(error.to_string().contains("upstream boom"));
    }

    #[test]
    fn parse_response_maps_status_completed_to_stop() {
        let client =
            OpenAiResponsesClient::new("http://endpoint.example".into(), "key".into(), None);
        let message = json!({
            "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });
        let response = client
            .parse_response(&message, None, None, Instant::now())
            .unwrap();
        assert_eq!(response.finish_reason, "stop");
    }

    #[test]
    fn url_post_verbatim_when_the_endpoint_carries_the_full_path() {
        let client = OpenAiResponsesClient::new(
            "https://gateway.example/openai/responses".into(),
            "key".into(),
            None,
        );
        assert_eq!(
            client.url("/responses"),
            "https://gateway.example/openai/responses",
            "a complete operation URL is used verbatim, never doubled"
        );
        // Other paths still join onto the base minus the known suffix.
        assert_eq!(
            client.url("/models"),
            "https://gateway.example/openai/models"
        );
    }

    #[test]
    fn url_inserts_v1_for_bare_hosts() {
        let client =
            OpenAiResponsesClient::new("http://localhost:11434".into(), "key".into(), None);
        assert_eq!(
            client.url("/responses"),
            "http://localhost:11434/v1/responses"
        );
    }

    #[test]
    fn is_empty_body_error_recognizes_json_failures() {
        let parse_err: anyhow::Error = anyhow::Error::from(
            serde_json::from_str::<Value>("").expect_err("empty input is a parse error"),
        );
        assert!(is_empty_body_error(&parse_err));

        let other: anyhow::Error = anyhow::anyhow!("connection reset");
        assert!(!is_empty_body_error(&other));
    }
}
