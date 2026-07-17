//! Agentic tool-calling loop: reasoning extraction, stuck detection, streaming,
//! error recovery, and cancel handling.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use super::client::{ChatClient, ChatResponse, StreamCallback};
use super::context::{Context, ContextKind, Role};
use super::prompt;
use super::tools::ToolRegistry;

/// A signal to break the agent loop.
enum LoopControl {
    Continue,
    Stop,
    Stuck,
}

/// The agentic loop that runs a task: sends the prompt to the LLM, executes tool
/// calls, and loops until the model stops or is cancelled.
pub struct AgentLoop {
    llm: Arc<dyn ChatClient>,
    tools: ToolRegistry,
    context: Context,
    model: String,
    cancel: Arc<AtomicBool>,
    /// Sliding window of recent tool calls for stuck detection.
    recent_calls: VecDeque<(String, String)>,
    /// Number of stuck signals encountered.
    stuck_count: u32,
}

impl AgentLoop {
    pub fn new(
        llm: Arc<dyn ChatClient>,
        tools: ToolRegistry,
        model: String,
        token_budget: usize,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            llm,
            tools,
            context: Context::new(token_budget),
            model,
            cancel,
            recent_calls: VecDeque::with_capacity(8),
            stuck_count: 0,
        }
    }

    /// Run the agentic loop for a single prompt. Calls `on_chunk` for streamed text deltas.
    /// Returns the final assistant response text.
    pub fn run(
        &mut self,
        prompt: &str,
        cwd: &str,
        on_chunk: Option<&StreamCallback>,
    ) -> Result<String> {
        // Initialize context with the system prompt and user prompt.
        let system_prompt = prompt::system_prompt();
        self.context
            .push(Role::System, ContextKind::System, system_prompt);
        self.context
            .push(Role::User, ContextKind::UserPrompt, prompt);

        let tool_schemas = self.tools.tools_schema();

        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Ok("[cancelled]".into());
            }

            // Build messages and enforce context budget
            self.context.enforce_budget();
            let messages = self.context.to_messages();

            debug!(
                "harness loop: {} messages, {} tokens",
                messages.len(),
                self.context.total_tokens()
            );

            // Wire up streaming: the LLM client calls on_chunk for each text delta,
            // which we forward to the ACP server for session/update notifications
            let response = self
                .llm
                .chat(&self.model, &messages, &tool_schemas, on_chunk)?;

            // Also emit the full response text (for non-streaming fallback or completeness)
            if let Some(cb) = on_chunk
                && !response.content.is_empty()
            {
                cb(&response.content);
            }

            // Check finish reason
            match self.handle_response(&response, cwd)? {
                LoopControl::Stop => {
                    return Ok(response.content);
                }
                LoopControl::Stuck => {
                    warn!(
                        "harness: stuck detected (count={}), breaking loop",
                        self.stuck_count
                    );
                    return Ok(format!(
                        "{}\n\n[agent loop stopped: stuck after {} attempts]",
                        response.content, self.stuck_count
                    ));
                }
                LoopControl::Continue => continue,
            }
        }
    }

    fn handle_response(&mut self, response: &ChatResponse, cwd: &str) -> Result<LoopControl> {
        // Store reasoning if present
        if let Some(reasoning) = &response.reasoning
            && !reasoning.trim().is_empty()
        {
            self.context
                .push(Role::System, ContextKind::Reasoning, reasoning);
        }

        if response.finish_reason == "stop" || response.tool_calls.is_empty() {
            // Model is done — store the final text
            if !response.content.is_empty() {
                self.context.push_assistant_text(&response.content);
            }
            info!("harness: model finished with stop reason");
            return Ok(LoopControl::Stop);
        }

        // Model requested tool calls — store the assistant message with tool_calls
        self.context.push_assistant_with_tools(
            if response.content.is_empty() {
                None
            } else {
                Some(&response.content)
            },
            &response.tool_calls,
            response.reasoning.as_deref(),
        );

        // Check for stuck patterns
        if self.detect_stuck(&response.tool_calls) {
            self.stuck_count += 1;
            if self.stuck_count >= 3 {
                return Ok(LoopControl::Stuck);
            }
            // Inject a nudge
            self.context.push(
                Role::System,
                ContextKind::System,
                "You appear to be repeating the same tool call. Reconsider your approach. Try a different strategy or explain what's blocking you.",
            );
        } else {
            self.stuck_count = 0;
        }

        // Execute each tool call
        for tc in &response.tool_calls {
            let tc_id = tc["id"].as_str().unwrap_or("unknown").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");

            let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

            info!("harness: executing tool {name} (call_id={tc_id}) args={args_str}");

            // Track for stuck detection
            self.recent_calls
                .push_back((name.clone(), args_str.to_string()));
            if self.recent_calls.len() > 5 {
                self.recent_calls.pop_front();
            }

            let result = match self.tools.execute(&name, &args, cwd) {
                Ok(result) => {
                    let preview: String = result.chars().take(200).collect();
                    info!("harness: tool {name} result: {preview}");
                    result
                }
                Err(e) => {
                    let err_msg = format!("Tool '{name}' error: {e}");
                    warn!("harness: {err_msg}");
                    err_msg
                }
            };

            // Classify the result kind for context management
            let kind = classify_tool_result(&name, &result);
            self.context.push_tool_result(kind, result, &tc_id);
        }

        Ok(LoopControl::Continue)
    }

    /// Detect if the agent is repeating the same tool call.
    fn detect_stuck(&self, tool_calls: &[Value]) -> bool {
        if tool_calls.is_empty() || self.recent_calls.is_empty() {
            return false;
        }

        for tc in tool_calls {
            let name = tc["function"]["name"].as_str().unwrap_or("");
            let args = tc["function"]["arguments"].as_str().unwrap_or("");
            let current = (name.to_string(), args.to_string());

            // Check if this exact call was in the recent window
            if self.recent_calls.iter().any(|prev| prev == &current) {
                return true;
            }
        }

        false
    }
}

fn classify_tool_result(tool_name: &str, _result: &str) -> ContextKind {
    match tool_name {
        "shell" => ContextKind::ShellOutput,
        "file_read" => ContextKind::FileRead,
        "file_edit" => ContextKind::EditResult,
        "file_write" => ContextKind::EditResult,
        "grep" | "glob" => ContextKind::Exploration,
        "web_fetch" => ContextKind::WebFetch,
        _ => ContextKind::ToolResult,
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::FakeChatClient;
    use super::*;

    #[test]
    fn loop_runs_single_tool_call_then_stops() {
        let llm = Arc::new(FakeChatClient::new(vec![
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
                content: "Done, the command ran.".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(llm, tools, "test-model".into(), 100_000, cancel);

        let result = agent.run("run echo hi", "/tmp", None).unwrap();
        assert!(result.contains("Done"));
    }

    #[test]
    fn loop_handles_tool_error_and_continues() {
        let llm = Arc::new(FakeChatClient::new(vec![
            ChatResponse {
                content: String::new(),
                reasoning: None,
                tool_calls: vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nonexistent_tool", "arguments": "{}"}
                })],
                finish_reason: "tool_calls".into(),
            },
            ChatResponse {
                content: "Recovered from error.".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(llm, tools, "test-model".into(), 100_000, cancel);

        let result = agent.run("test error recovery", "/tmp", None).unwrap();
        assert!(result.contains("Recovered"));
    }

    #[test]
    fn loop_respects_cancel() {
        let llm = Arc::new(FakeChatClient::new(vec![
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
                content: "should not reach".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools();
        let cancel = Arc::new(AtomicBool::new(true)); // Pre-cancelled
        let mut agent = AgentLoop::new(llm, tools, "test-model".into(), 100_000, cancel);

        let result = agent.run("test cancel", "/tmp", None).unwrap();
        assert!(result.contains("cancelled"));
    }
}
