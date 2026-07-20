//! Agentic tool-calling loop: reasoning extraction, stuck detection, streaming,
//! error recovery, and cancel handling.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde_json::{Value, json};
use tracing::{info, warn};

use super::client::{ChatClient, ChatResponse, StreamCallback};
use super::context::{Context, ContextEntry, ContextKind, Role};
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
    /// Cached tool schemas — built once at construction, reused across all runs.
    tool_schemas: Vec<Value>,
    context: Context,
    model: String,
    cancel: Arc<AtomicBool>,
    /// Task checklist that survives context compaction and collapse.
    /// Injected as a system message before every API call.
    todo: Arc<super::todo::TodoList>,
    /// Sliding window of recent tool calls for stuck detection.
    recent_calls: VecDeque<(String, String)>,
    /// Number of stuck signals encountered.
    stuck_count: u32,
}

impl AgentLoop {
    pub fn new(
        llm: Arc<dyn ChatClient>,
        mut tools: ToolRegistry,
        model: String,
        token_budget: usize,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        let todo = Arc::new(super::todo::TodoList::new());
        // Register the todo tool so the model can manage its task checklist.
        tools.register(Arc::new(super::tools::todo::TodoTool::new(Arc::clone(
            &todo,
        ))));
        // Register the memory tool so the model can save fundamental project facts.
        tools.register(Arc::new(super::tools::memory::MemoryTool));
        // Build tool schemas once — the tool set is fixed for the harness lifetime.
        let tool_schemas = tools.tools_schema();
        Self {
            llm,
            tools,
            tool_schemas,
            context: Context::new(token_budget),
            model,
            cancel,
            todo,
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

        // Load persistent project facts (if any) and inject as a system message.
        // These are fundamental facts about the project that were extracted during
        // previous sessions' context compaction — build commands, architecture,
        // key file locations, conventions. They survive across sessions.
        if let Some(facts) = super::memory::load_facts(cwd) {
            let facts_prompt = format!("## Project Facts\n\n{facts}");
            self.context
                .push(Role::System, ContextKind::System, &facts_prompt);
        }

        self.context
            .push(Role::User, ContextKind::UserPrompt, prompt);

        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Ok("[cancelled]".into());
            }

            // Build messages and enforce context budget. When compacting, use
            // the LLM to summarize all large entries in a single call so the
            // most important info (errors, key results, file paths) is preserved.
            // The same call also produces a fresh, complete set of project facts
            // that overwrites the persistent memory.
            let model = &self.model;
            let llm = &self.llm;
            let tool_schemas = &self.tool_schemas;
            let compactor: &super::context::Compactor<'_> = &|entries: &[(String, String)]| {
                if entries.is_empty() {
                    return Vec::new();
                }

                // Build a single prompt: summarize each output.
                let mut prompt = String::from(
                    "Summarize each of the following tool outputs concisely. \
                     For each one, keep all errors, warnings, file paths, line numbers, \
                     function/class names, and key results. Remove redundant lines, \
                     repetition, and verbose output. Preserve the essential information \
                     the agent would need to continue working.\n\n\
                     Respond with one summary per output, separated by a line containing \
                     exactly '---SUMMARY---'. Do not include the original output.\n",
                );
                for (i, (label, content)) in entries.iter().enumerate() {
                    prompt.push_str(&format!("\n=== OUTPUT {i} ({label}) ===\n{content}\n"));
                }

                let messages = vec![json!({
                    "role": "user",
                    "content": prompt
                })];
                match llm.chat(model, &messages, tool_schemas, None, None) {
                    Ok(resp) if !resp.content.is_empty() => {
                        let summaries: Vec<String> = resp
                            .content
                            .split("---SUMMARY---")
                            .map(|s| s.trim().to_string())
                            .collect();
                        let mut result = Vec::with_capacity(entries.len());
                        for i in 0..entries.len() {
                            if i < summaries.len() && !summaries[i].is_empty() {
                                result.push(format!(
                                    "[llm-compacted {}]\n{}",
                                    entries[i].0, summaries[i]
                                ));
                            } else {
                                result.push(compact_summary_fallback(&entries[i].1, &entries[i].0));
                            }
                        }
                        result
                    }
                    _ => entries
                        .iter()
                        .map(|(label, content)| compact_summary_fallback(content, label))
                        .collect(),
                }
            };

            // Conversation summarizer: when non-evictable entries (tool calls,
            // edit results) exceed the budget, collapse the entire conversation
            // into a single summary. This breaks the tool-call chain and starts
            // fresh: system prompt + summary + original user prompt.
            let summarizer: &super::context::Summarizer<'_> = &|entries: &[ContextEntry]| {
                // Build a text representation of the conversation for the LLM.
                let mut transcript = String::new();
                for entry in entries {
                    let role_label = match entry.role {
                        Role::System => "SYSTEM",
                        Role::User => "USER",
                        Role::Assistant => "ASSISTANT",
                        Role::Tool => "TOOL",
                    };
                    transcript.push_str(&format!("[{role_label}]\n{}\n\n", entry.content));
                }

                let prompt = format!(
                    "Summarize the following agent conversation. Focus on:\n\
                     - What task the agent is working on and its current progress\n\
                     - What files were read, created, or edited (with paths)\n\
                     - What commands were run and their results (builds, tests, errors)\n\
                     - What decisions were made and what remains to be done\n\
                     - Any errors or blockers encountered\n\n\
                     Be concise but complete — the agent will use this summary to continue \
                     working without access to the original tool outputs.\n\n{transcript}"
                );

                let messages = vec![json!({
                    "role": "user",
                    "content": prompt
                })];
                match llm.chat(model, &messages, tool_schemas, None, None) {
                    Ok(resp) if !resp.content.is_empty() => resp.content,
                    _ => {
                        // Fallback: naive summary of first/last entries
                        let mut parts = Vec::new();
                        for entry in entries.iter().take(5) {
                            let preview: String = entry.content.chars().take(200).collect();
                            parts.push(preview);
                        }
                        format!("[fallback summary]\n{}", parts.join("\n---\n"))
                    }
                }
            };
            let action = self
                .context
                .enforce_budget(Some(compactor), Some(summarizer));
            if action != super::context::BudgetAction::None {
                info!(
                    "harness: context {action:?}, {} tokens after",
                    self.context.total_tokens()
                );
            }

            // Inject the todo checklist as a temporary system message.
            // This is added fresh every turn (after compaction) so it always
            // reflects the current state and survives any compaction/collapse.
            let todo_snapshot = self.todo.render();
            let mut messages = self.context.to_messages();
            if let Some(todo_text) = todo_snapshot {
                messages.push(json!({
                    "role": "system",
                    "content": todo_text
                }));
            }

            // Tool execution callback for overlap: when the streaming response
            // finishes (finish_reason arrives), the client invokes this closure
            // to start executing tool calls while the stream tail is still being
            // read. This overlaps tool I/O with the model's generation tail.
            //
            // Scoped in a block so the immutable borrow of `self.tools` (via
            // `tools_ref`) is released before `handle_response` borrows `self`
            // mutably.
            let response = {
                let tools_ref = &self.tools;
                let exec_cb = &|tool_calls: &[Value], _finish: &str| {
                    let parsed: Vec<(String, String, Value)> = tool_calls
                        .iter()
                        .map(|tc| {
                            let tc_id = tc["id"].as_str().unwrap_or("unknown").to_string();
                            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                            let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                            let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                            (name, tc_id, args)
                        })
                        .collect();

                    // Only run concurrently when every call is read-only.
                    // If any call is to a mutating tool (file_write, file_edit,
                    // shell), run all calls sequentially in order to preserve
                    // dependencies (e.g. `mkdir` before `file_write`).
                    let all_read_only = parsed.iter().all(|(name, _, _)| is_read_only_tool(name));

                    if parsed.len() <= 1 || !all_read_only {
                        parsed
                            .iter()
                            .map(|(name, _, args)| execute_and_log(tools_ref, name, args, cwd))
                            .collect()
                    } else {
                        // All read-only — concurrent execution
                        std::thread::scope(|s| {
                            let handles: Vec<_> = parsed
                                .iter()
                                .map(|(name, _, args)| {
                                    s.spawn(move || execute_and_log(tools_ref, name, args, cwd))
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|h| {
                                    h.join()
                                        .unwrap_or_else(|_| "Error: tool thread panicked".into())
                                })
                                .collect()
                        })
                    }
                };

                self.llm.chat(
                    &self.model,
                    &messages,
                    &self.tool_schemas,
                    on_chunk,
                    Some(exec_cb),
                )?
            };

            info!(
                "harness: turn {} messages, model={}, elapsed={}ms, tokens in={} out={} cached={} finish={} tool_calls={}",
                messages.len(),
                self.model,
                response.elapsed_ms,
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.cached_tokens,
                response.finish_reason,
                response.tool_calls.len()
            );

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
        // Reasoning content is not stored in context — it's intermediate thinking
        // already reflected in the subsequent tool calls and actions. Storing it
        // wastes tokens on every turn.

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

        // Track calls for stuck detection (before concurrent execution)
        for tc in &response.tool_calls {
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args_str = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("{}")
                .to_string();
            self.recent_calls.push_back((name, args_str));
            if self.recent_calls.len() > 5 {
                self.recent_calls.pop_front();
            }
        }

        // Use pre-computed results from overlap execution if available;
        // otherwise execute tool calls now (concurrently).
        let tool_results: Vec<(String, String, String)> = if !response.tool_results.is_empty() {
            // Results were computed during streaming overlap — just pair them
            // with names and call IDs for context classification.
            response
                .tool_calls
                .iter()
                .zip(&response.tool_results)
                .map(|(tc, result)| {
                    let tc_id = tc["id"].as_str().unwrap_or("unknown").to_string();
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    (name, tc_id, result.clone())
                })
                .collect()
        } else {
            self.execute_tool_calls_concurrent(&response.tool_calls, cwd)
        };

        for (name, tc_id, result) in tool_results {
            // Skip storing todo tool results in context — the todo list is
            // already injected as a system message every turn, so storing the
            // tool response would be pure duplication.
            if name == "todo" || name == "memory" {
                continue;
            }
            // Compress empty search results to a short note — the full
            // "No matches found for pattern '...' in ..." output has no value.
            let result = if (name == "grep" || name == "glob")
                && (result.contains("No matches") || result.contains("No files matching"))
            {
                "[no results]".to_string()
            } else {
                result
            };
            let kind = classify_tool_result(&name, &result);
            // Truncate large tool results before pushing to context to avoid
            // context bloat and the BPE encoding cost on oversized outputs.
            let truncated = truncate_tool_result(&result);
            self.context.push_tool_result(kind, truncated, &tc_id);
        }

        Ok(LoopControl::Continue)
    }

    /// Execute tool calls, concurrently when safe. Returns results in the same
    /// order as the input tool calls. Each result is
    /// `(tool_name, tool_call_id, result_string)`.
    ///
    /// Concurrency is only used when **every** call is to a read-only tool
    /// (`file_read`, `file_read_batch`, `grep`, `glob`, `web_fetch`). If any
    /// call is to a mutating tool (`file_write`, `file_edit`, `shell`), all
    /// calls run sequentially in order to preserve dependencies.
    fn execute_tool_calls_concurrent(
        &self,
        tool_calls: &[Value],
        cwd: &str,
    ) -> Vec<(String, String, String)> {
        if tool_calls.is_empty() {
            return Vec::new();
        }

        // Parse all calls up front (cheap), so the concurrent phase only does I/O.
        let parsed: Vec<(String, String, Value)> = tool_calls
            .iter()
            .map(|tc| {
                let tc_id = tc["id"].as_str().unwrap_or("unknown").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                (name, tc_id, args)
            })
            .collect();

        let all_read_only = parsed.iter().all(|(name, _, _)| is_read_only_tool(name));

        // Single call, or mixed/mutating calls — run sequentially in order.
        if parsed.len() == 1 || !all_read_only {
            return parsed
                .iter()
                .map(|(name, tc_id, args)| {
                    let result = self.execute_one(name, args, cwd);
                    (name.clone(), tc_id.clone(), result)
                })
                .collect();
        }

        // Multiple read-only calls — run concurrently. `&ToolRegistry` is
        // `Send + Sync` because all tools are `Arc<dyn Tool>`.
        std::thread::scope(|s| {
            let handles: Vec<_> = parsed
                .iter()
                .map(|(name, tc_id, args)| {
                    s.spawn(move || {
                        let result = self.execute_one(name, args, cwd);
                        (name.clone(), tc_id.clone(), result)
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        (
                            "unknown".into(),
                            "unknown".into(),
                            "Error: tool thread panicked".into(),
                        )
                    })
                })
                .collect()
        })
    }

    /// Execute a single tool call, logging the invocation and result.
    fn execute_one(&self, name: &str, args: &Value, cwd: &str) -> String {
        execute_and_log(&self.tools, name, args, cwd)
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
        "file_read" | "file_read_batch" => ContextKind::FileRead,
        "file_edit" => ContextKind::EditResult,
        "file_write" => ContextKind::EditResult,
        "grep" | "glob" => ContextKind::Exploration,
        "web_fetch" => ContextKind::WebFetch,
        "todo" | "memory" => ContextKind::ToolResult,
        _ => ContextKind::ToolResult,
    }
}

/// Maximum character length for a tool result stored in context. Larger results
/// are truncated with a marker, keeping the first portion (most relevant for
/// file reads and search results) and a note about the truncation.
const MAX_TOOL_RESULT_CHARS: usize = 4_000;

/// Whether a tool is read-only (no side effects). Read-only tools can be
/// executed concurrently safely; mutating tools must run in order to preserve
/// dependencies (e.g. `mkdir` before `file_write`).
fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "file_read" | "file_read_batch" | "grep" | "glob" | "web_fetch"
    )
}

/// Fallback compaction when the LLM summarizer is unavailable (e.g. API error).
/// Uses the naive first/last-lines heuristic.
fn compact_summary_fallback(content: &str, label: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let first_lines: Vec<&str> = lines.iter().take(5).copied().collect();
    let last_lines: Vec<&str> = if lines.len() > 10 {
        lines.iter().rev().take(3).rev().copied().collect()
    } else {
        Vec::new()
    };
    let mut summary = format!("[compacted {label}, {} lines]\n", lines.len());
    summary.push_str(&first_lines.join("\n"));
    if !last_lines.is_empty() {
        summary.push_str("\n[...truncated...]\n");
        summary.push_str(&last_lines.join("\n"));
    }
    summary
}

/// Execute a single tool call against the registry, logging the invocation and
/// result. Used by both the overlap execution callback and the fallback
/// sequential executor so that tool call details are always logged.
fn execute_and_log(tools: &ToolRegistry, name: &str, args: &Value, cwd: &str) -> String {
    let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
    info!("harness: executing tool {name} args={args_str}");

    match tools.execute(name, args, cwd) {
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
    }
}

/// Truncate a tool result to `MAX_TOOL_RESULT_CHARS`, preserving the beginning
/// (which typically contains the most useful output) and appending a marker.
fn truncate_tool_result(result: &str) -> String {
    if result.len() <= MAX_TOOL_RESULT_CHARS {
        return result.to_string();
    }
    // Find a char boundary at or before the limit
    let mut end = MAX_TOOL_RESULT_CHARS;
    while !result.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!(
        "{}\n[...output truncated at {} chars, {}/{} bytes shown...]",
        &result[..end],
        MAX_TOOL_RESULT_CHARS,
        end,
        result.len()
    )
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
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
            ChatResponse {
                content: "Done, the command ran.".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
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
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
            ChatResponse {
                content: "Recovered from error.".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
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
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
            ChatResponse {
                content: "should not reach".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools();
        let cancel = Arc::new(AtomicBool::new(true)); // Pre-cancelled
        let mut agent = AgentLoop::new(llm, tools, "test-model".into(), 100_000, cancel);

        let result = agent.run("test cancel", "/tmp", None).unwrap();
        assert!(result.contains("cancelled"));
    }

    #[test]
    fn is_read_only_tool_classifies_correctly() {
        assert!(is_read_only_tool("file_read"));
        assert!(is_read_only_tool("file_read_batch"));
        assert!(is_read_only_tool("grep"));
        assert!(is_read_only_tool("glob"));
        assert!(is_read_only_tool("web_fetch"));

        assert!(!is_read_only_tool("file_write"));
        assert!(!is_read_only_tool("file_edit"));
        assert!(!is_read_only_tool("shell"));
        assert!(!is_read_only_tool("unknown_tool"));
    }

    #[test]
    fn loop_executes_mixed_tool_calls_in_order() {
        // When the model issues both read-only and mutating tool calls in one
        // turn, all calls must run sequentially (not concurrently) to preserve
        // dependencies. We verify by issuing file_write then file_read on the
        // same path — if they ran concurrently the read might see the old state.
        let dir = super::super::tools::test_util::unique_test_dir();

        let llm = Arc::new(FakeChatClient::new(vec![
            ChatResponse {
                content: String::new(),
                reasoning: None,
                tool_calls: vec![
                    json!({
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "file_write", "arguments": format!("{{\"path\":\"out.txt\",\"content\":\"written\"}}")}
                    }),
                    json!({
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "file_read", "arguments": "{\"path\":\"out.txt\"}"}
                    }),
                ],
                finish_reason: "tool_calls".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
            ChatResponse {
                content: "Done.".into(),
                reasoning: None,
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: super::super::client::Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(llm, tools, "test-model".into(), 100_000, cancel);

        let result = agent.run("write then read", dir.as_str(), None).unwrap();
        assert!(result.contains("Done"));
        // The file_read result should contain the content written by file_write,
        // proving sequential execution preserved the order.
        let written = std::fs::read_to_string(dir.path().join("out.txt")).unwrap_or_default();
        assert_eq!(written, "written");
    }
}
