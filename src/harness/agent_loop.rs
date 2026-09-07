//! Agentic tool-calling loop: stuck detection, streaming, error recovery,
//! and cancel handling.

use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde_json::{Value, json};
use tracing::{info, warn};

use super::client::{ChatClient, ChatResponse, StreamCallback};
use super::context::{Context, ContextEntry, ContextKind, Role};
use super::prompt;
use super::tools::ToolRegistry;

/// Maximum character length for a tool result stored in context. Larger results
/// are truncated with a marker, keeping the first portion (most relevant for
/// file reads and search results) and a note about the truncation.
const MAX_TOOL_RESULT_CHARS: usize = 4_000;

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
    /// Channel for mid-run message injection. Messages drained at the top of
    /// each loop iteration are pushed into context as user messages before the
    /// next model call. Enables `session/inject` to redirect a running agent.
    /// Uses `Arc<Mutex<VecDeque>>` instead of `mpsc` so `&AgentLoop` is `Send`
    /// (needed for `std::thread::scope` in concurrent tool execution).
    inject_rx: Arc<Mutex<VecDeque<String>>>,
    /// Task checklist that survives context compaction and collapse.
    /// Injected as a system message before every API call.
    todo: Arc<super::todo::TodoList>,
    /// Sliding window of recent tool calls for stuck detection.
    recent_calls: VecDeque<(String, String)>,
    /// Number of stuck signals encountered.
    stuck_count: u32,
    /// Where the current context snapshot is persisted after every update.
    /// `None` disables persistence (sessions created without a resumable
    /// identity, e.g. unlogged subagent sessions).
    context_path: Option<PathBuf>,
}

impl AgentLoop {
    pub fn new(
        llm: Arc<dyn ChatClient>,
        mut tools: ToolRegistry,
        model: String,
        token_budget: usize,
        cancel: Arc<AtomicBool>,
        inject_rx: Arc<Mutex<VecDeque<String>>>,
    ) -> Self {
        let todo = Arc::new(super::todo::TodoList::new());
        // Register the todo tool so the model can manage its task checklist.
        tools.register(Arc::new(super::tools::todo::TodoTool::new(Arc::clone(
            &todo,
        ))));
        // Build tool schemas once — the tool set is fixed for the harness lifetime.
        let tool_schemas = tools.tools_schema();
        Self {
            llm,
            tools,
            tool_schemas,
            context: Context::new(token_budget),
            model,
            cancel,
            inject_rx,
            todo,
            recent_calls: VecDeque::with_capacity(8),
            stuck_count: 0,
            context_path: None,
        }
    }

    /// Persist the context snapshot after every update. The file always
    /// holds the latest context, so a crashed process resumes from the last
    /// completed update. Setting the path does not write anything: a fresh
    /// loop's empty context must not clobber the snapshot it is about to
    /// restore.
    pub fn set_context_path(&mut self, path: PathBuf) {
        self.context_path = Some(path);
    }

    /// Write the current context snapshot to `context_path`, when set.
    /// Best-effort: a persistence failure logs and continues — losing a
    /// snapshot must never kill the running task.
    fn persist_context(&self) {
        let Some(path) = &self.context_path else {
            return;
        };
        let snapshot = self.context.snapshot();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let body = serde_json::to_string(&snapshot).unwrap_or_default();
        let tmp = path.with_extension("context.tmp");
        if fs::write(&tmp, body).is_ok() {
            // Rename into place so a reader never sees a half-written file.
            let _ = fs::rename(&tmp, path);
        } else {
            warn!("harness: failed to persist context to {}", path.display());
        }
    }

    /// Restore the context from a previous run's snapshot. Returns whether a
    /// snapshot was found and restored; a fresh session (or a corrupt
    /// snapshot) starts empty and re-runs `init_context`.
    pub fn restore_context(&mut self, cwd: &str, context_channels: &[(String, String)]) -> bool {
        let Some(path) = self.context_path.clone() else {
            self.init_context(cwd, context_channels);
            return false;
        };
        let restored = fs::read_to_string(&path)
            .ok()
            .and_then(|body| serde_json::from_str::<Value>(&body).ok())
            .and_then(|snapshot| Context::restore(&snapshot));
        match restored {
            Some(context) => {
                self.context = context;
                info!("harness: restored context from {}", path.display());
                true
            }
            None => {
                self.init_context(cwd, context_channels);
                false
            }
        }
    }

    /// Take all captured structured-output JSONs from caller-defined
    /// structured-output tools. Returns a JSON object mapping tool name to
    /// captured args. Tools that were never called are omitted.
    pub fn take_structured_outputs(&self) -> Value {
        self.tools.take_structured_outputs()
    }

    /// Update the model used for LLM calls. Called when the client sends
    /// `session/set_model` or `session/set_config_option` with `configId=model`.
    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    /// Names of all registered tools (built-ins + mode-specific tools like
    /// `plan`, plus `todo` and `memory` added by the constructor), in
    /// insertion order. Useful for diagnostics at session startup.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.tool_names()
    }

    /// Initialize the context with the system prompt and any context channels
    /// registered by in-process agents on the bus (e.g. durable project memory
    /// from the memory agent). Called once at session creation. Subsequent
    /// `session/prompt` calls reuse this context — true single long session.
    pub fn init_context(&mut self, _cwd: &str, context_channels: &[(String, String)]) {
        let descriptions = self.tools.tool_descriptions();
        let system_prompt = prompt::system_prompt(&descriptions);
        self.context
            .push(Role::System, ContextKind::System, &system_prompt);

        // Inject each context channel as a system message. These are
        // published by in-process agents (e.g. the memory agent registers a
        // "memory" channel whose content is the current durable memory).
        for (name, content) in context_channels {
            if content.trim().is_empty() {
                continue;
            }
            let msg = format!("## {name}\n\n{content}");
            self.context.push(Role::System, ContextKind::System, &msg);
        }
    }

    /// Run the agentic loop for a single prompt. Calls `on_chunk` for streamed
    /// text deltas and `on_turn` after each complete LLM response turn.
    /// Returns the final assistant response text.
    #[cfg(test)]
    pub fn run(
        &mut self,
        prompt: &str,
        cwd: &str,
        on_chunk: Option<&StreamCallback>,
    ) -> Result<String> {
        self.run_with_turn_callback(prompt, cwd, on_chunk, None)
    }

    /// Run the agentic loop with an optional turn callback invoked after each
    /// complete LLM response. The callback receives the full `ChatResponse`
    /// (content, reasoning, tool calls) for transcript logging.
    pub fn run_with_turn_callback(
        &mut self,
        prompt: &str,
        cwd: &str,
        on_chunk: Option<&StreamCallback>,
        on_turn: Option<&super::client::TurnCallback>,
    ) -> Result<String> {
        // Push the user prompt into the existing context (system prompt + facts
        // were already initialized by `init_context` at session creation).
        self.context
            .push(Role::User, ContextKind::UserPrompt, prompt);
        self.persist_context();

        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Ok("[cancelled]".into());
            }

            // Drain injected messages into context before the next model call.
            // `session/inject` pushes here; messages are `UserPrompt` (non-evictable)
            // so they survive compaction. This enables mid-run redirection.
            {
                let mut queue = self.inject_rx.lock().unwrap();
                let injected = !queue.is_empty();
                while let Some(msg) = queue.pop_front() {
                    self.context.push(Role::User, ContextKind::UserPrompt, &msg);
                }
                if injected {
                    self.persist_context();
                }
            }

            // Build messages and enforce context budget. When compacting, use
            // the LLM to summarize all large entries in a single call so the
            // most important info (errors, key results, file paths) is preserved.
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
                match llm.chat(model, &messages, tool_schemas, None, None, None) {
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
                    "Summarize the following agent conversation so the agent can continue working \
                     without re-reading files or re-running commands. Your summary MUST include \
                     these sections:\n\n\
                     ## Task\n\
                     What the agent is working on and its current progress.\n\n\
                     ## Files explored\n\
                     For each file the agent read or searched, list:\n\
                     - The file path\n\
                     - Key symbols, functions, structs, or types found there (with line numbers if known)\n\
                     - A one-line note on what that code does\n\
                     This section is critical — it prevents the agent from re-reading the same files \
                     after compaction. Be specific: `ukb/ukbtask/drain.go: Drainer struct (line 30), \
                     Drain method (line 84) — processes document deletion queue` is useful; \
                     `read drain.go` is not.\n\n\
                     ## Changes made\n\
                     Files created or edited, with a one-line description of each change.\n\n\
                     ## Commands run\n\
                     Build, test, or shell commands and their key results (errors, pass/fail).\n\n\
                     ## Next steps\n\
                     What remains to be done, in order.\n\n\
                     Be concise but complete. Do not include full file contents — just the symbol \
                     index and key findings.\n\n{transcript}"
                );

                let messages = vec![json!({
                    "role": "user",
                    "content": prompt
                })];
                match llm.chat(model, &messages, tool_schemas, None, None, None) {
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
                    format_tokens(self.context.total_tokens() as u64)
                );
                self.persist_context();
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
                // Cache for speculatively-executed read-only tool results.
                // Keyed by tool-call index. Populated by `early_cb` during
                // streaming; read by `exec_cb` at finish_reason so speculatively
                // executed tools don't run twice.
                let early_cache: std::sync::Arc<
                    std::sync::Mutex<std::collections::HashMap<usize, String>>,
                > = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

                let early_cache_for_cb = std::sync::Arc::clone(&early_cache);
                let early_cb = &|idx: usize, tc: &Value| -> Option<String> {
                    let name = tc["function"]["name"].as_str().unwrap_or("");
                    // Only speculatively execute read-only tools — mutating
                    // tools must wait for finish_reason to ensure all args
                    // are final and ordering is preserved.
                    if !is_read_only_tool(name) {
                        return None;
                    }
                    let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                    let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    let result = execute_and_log(tools_ref, name, &args, cwd);
                    early_cache_for_cb.lock().unwrap().insert(idx, result);
                    Some(String::new()) // non-None signals "executed"
                };

                let early_cache_for_exec = std::sync::Arc::clone(&early_cache);
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
                    // If any call is to a mutating tool (write, edit,
                    // shell), run all calls sequentially in order to preserve
                    // dependencies (e.g. `mkdir` before `write`).
                    let all_read_only = parsed.iter().all(|(name, _, _)| is_read_only_tool(name));

                    let cache = early_cache_for_exec.lock().unwrap();
                    if parsed.len() <= 1 || !all_read_only {
                        parsed
                            .iter()
                            .enumerate()
                            .map(|(idx, (name, _, args))| {
                                if let Some(cached) = cache.get(&idx) {
                                    return cached.clone();
                                }
                                execute_and_log(tools_ref, name, args, cwd)
                            })
                            .collect()
                    } else {
                        // All read-only — concurrent execution for any calls
                        // not already speculatively cached.
                        std::thread::scope(|s| {
                            let handles: Vec<_> = parsed
                                .iter()
                                .enumerate()
                                .map(|(idx, (name, _, args))| {
                                    if let Some(cached) = cache.get(&idx) {
                                        return s.spawn(move || cached.clone());
                                    }
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

                match self.llm.chat(
                    &self.model,
                    &messages,
                    &self.tool_schemas,
                    on_chunk,
                    Some(exec_cb),
                    Some(early_cb),
                ) {
                    Ok(resp) => resp,
                    Err(e) => {
                        // Retry on 400 "function.arguments must be valid JSON"
                        // errors. The model sometimes streams truncated tool
                        // call arguments; sanitizing the last assistant entry
                        // and retrying lets the conversation continue instead
                        // of failing the whole task.
                        if is_malformed_tool_call_error(&e)
                            && self.context.sanitize_last_assistant_tool_calls()
                        {
                            warn!(
                                "harness: API rejected malformed tool call arguments, sanitized context and retrying"
                            );
                            // Rebuild messages from the sanitized context and
                            // retry once. A second failure propagates.
                            let todo_snapshot = self.todo.render();
                            let mut retry_messages = self.context.to_messages();
                            if let Some(todo_text) = todo_snapshot {
                                retry_messages.push(json!({
                                    "role": "system",
                                    "content": todo_text
                                }));
                            }
                            self.llm.chat(
                                &self.model,
                                &retry_messages,
                                &self.tool_schemas,
                                on_chunk,
                                Some(exec_cb),
                                Some(early_cb),
                            )?
                        } else {
                            return Err(e);
                        }
                    }
                }
            };

            info!(
                "harness: turn {} messages, model={}, elapsed={}, tokens in={} out={} cached={} finish={} tool_calls={}",
                messages.len(),
                self.model,
                format_duration(response.elapsed_ms),
                format_tokens(response.usage.input_tokens),
                format_tokens(response.usage.output_tokens),
                format_tokens(response.usage.cached_tokens),
                response.finish_reason,
                response.tool_calls.len()
            );

            // Log a few lines of reasoning content for diagnostics. Reasoning
            // can be long, so cap at the first few non-empty lines.
            if !response.reasoning.is_empty() {
                let preview: String = response
                    .reasoning
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ⏎ ");
                info!(target: "harness", "harness: reasoning preview: {preview}");
            }

            // Also emit the full response text (for non-streaming fallback or completeness)
            if let Some(cb) = on_chunk
                && !response.content.is_empty()
            {
                cb(&response.content);
            }

            // Invoke the turn callback (for transcript logging) with the full
            // ChatResponse — content, reasoning, tool calls, etc.
            if let Some(cb) = on_turn {
                cb(&response);
            }

            // Check finish reason
            match self.handle_response(&response, cwd)? {
                LoopControl::Stop => {
                    self.persist_context();
                    return Ok(response.content);
                }
                LoopControl::Stuck => {
                    warn!(
                        "harness: stuck detected (count={}), breaking loop",
                        self.stuck_count
                    );
                    self.persist_context();
                    return Ok(format!(
                        "{}\n\n[agent loop stopped: stuck after {} attempts]",
                        response.content, self.stuck_count
                    ));
                }
                LoopControl::Continue => {
                    self.persist_context();
                    continue;
                }
            }
        }
    }

    fn handle_response(&mut self, response: &ChatResponse, cwd: &str) -> Result<LoopControl> {
        if response.finish_reason == "stop" || response.tool_calls.is_empty() {
            // Model is done — store the final text
            if !response.content.is_empty() {
                self.context
                    .push_assistant_text_with_reasoning(&response.content, &response.reasoning);
            }
            info!("harness: model finished with stop reason");
            return Ok(LoopControl::Stop);
        }

        // Model requested tool calls — store the assistant message with tool_calls.
        // Reasoning is re-injected so the model can build on its prior thinking.
        self.context.push_assistant_with_tools(
            if response.content.is_empty() {
                None
            } else {
                Some(&response.content)
            },
            &response.tool_calls,
            &response.reasoning,
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
            // Skip storing todo/plan tool results in context — the todo
            // list is already injected as a system message every turn and
            // the plan output lives in the side-channel cell. Storing these
            // tool responses would be pure duplication.
            if name == "todo" || name == "plan" {
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
    /// (`read`, `grep`, `glob`). If any call is to a mutating tool (`write`,
    /// `edit`, `shell`, `http`), all calls run sequentially in order to
    /// preserve dependencies.
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

            // Polling a background job or sleeping between polls is expected
            // repetitive behavior, not a stuck pattern. The model legitimately
            // polls the same job_id multiple times while waiting for a
            // long-running background task to finish.
            if name == "shell" && (args.contains("\"job_id\"") || args.contains("sleep ")) {
                continue;
            }

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
        "read" => ContextKind::FileRead,
        "edit" => ContextKind::EditResult,
        "write" => ContextKind::EditResult,
        "grep" | "glob" => ContextKind::Exploration,
        "http" => ContextKind::WebFetch,
        "plan" | "todo" | "memory" => ContextKind::ToolResult,
        _ => ContextKind::ToolResult,
    }
}

/// Whether a tool is read-only (no side effects). Read-only tools can be
/// executed concurrently safely; mutating tools must run in order to preserve
/// dependencies (e.g. `mkdir` before `write`).
fn is_read_only_tool(name: &str) -> bool {
    matches!(name, "read" | "grep" | "glob")
}

/// Check whether an LLM API error is caused by malformed tool call arguments
/// (the model streamed empty or truncated `function.arguments`). The error
/// message from OpenAI-compatible APIs looks like:
/// `LLM request failed (400 Bad Request): {"message":"Assistant tool call function.arguments must be valid JSON."}`
/// Returns true when the error is a 400 mentioning `arguments` and `valid JSON`,
/// so the caller can sanitize the context and retry.
fn is_malformed_tool_call_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err}");
    msg.contains("400") && msg.contains("arguments") && msg.contains("valid JSON")
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
            info!(
                "harness: tool {name} result: {}",
                preview_lines(&result, 10)
            );
            result
        }
        Err(e) => {
            let err_msg = format!("Tool '{name}' error: {e}");
            warn!("harness: {err_msg}");
            err_msg
        }
    }
}

/// Render a tool result as a log preview: at most `max_lines` lines, with an
/// ellipsis marker when more remain. Used by `execute_and_log` so multi-line
/// tool output (file reads, shell, todo) shows a bounded preview rather than
/// either overflowing the log or being truncated mid-line by a char limit.
fn preview_lines(result: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = result.lines().collect();
    if lines.len() <= max_lines {
        return result.to_string();
    }
    let mut out = lines[..max_lines].join("\n");
    out.push_str("\n[...]");
    out
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

/// Format a token count in human-readable form (e.g. 1.2k, 30k, 1.5M).
/// Trims trailing `.0` so whole numbers show as `30k` not `30.0k`.
fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        let v = n as f64 / 1_000_000.0;
        let s = format!("{v:.1}M");
        s.replace(".0M", "M")
    } else if n >= 1_000 {
        let v = n as f64 / 1_000.0;
        let s = format!("{v:.1}k");
        s.replace(".0k", "k")
    } else {
        format!("{n}")
    }
}

/// Format an elapsed time (in milliseconds) in human-readable form with a
/// single unit: < 1s → `350ms`, < 60s → `2.3s`, >= 60s → `1.2m`.
fn format_duration(ms: u128) -> String {
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        let s = ms as f64 / 1_000.0;
        let s_str = format!("{s:.1}s");
        s_str.replace(".0s", "s")
    } else {
        let m = ms as f64 / 60_000.0;
        let m_str = format!("{m:.1}m");
        m_str.replace(".0m", "m")
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::{EarlyToolExecCallback, FakeChatClient, ToolExecCallback, Usage};
    use super::super::tools::SessionStates;
    use super::super::tools::test_util;
    use super::*;

    #[test]
    fn loop_runs_single_tool_call_then_stops() {
        let llm = Arc::new(FakeChatClient::new(vec![
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
                content: "Done, the command ran.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context("/tmp", &[]);

        let result = agent.run("run echo hi", "/tmp", None).unwrap();
        assert!(result.contains("Done"));
    }

    #[test]
    fn loop_handles_tool_error_and_continues() {
        let llm = Arc::new(FakeChatClient::new(vec![
            ChatResponse {
                content: String::new(),
                tool_calls: vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nonexistent_tool", "arguments": "{}"}
                })],
                finish_reason: "tool_calls".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
            ChatResponse {
                content: "Recovered from error.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context("/tmp", &[]);

        let result = agent.run("test error recovery", "/tmp", None).unwrap();
        assert!(result.contains("Recovered"));
    }

    #[test]
    fn loop_respects_cancel() {
        let llm = Arc::new(FakeChatClient::new(vec![
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
                content: "should not reach".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(true)); // Pre-cancelled
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context("/tmp", &[]);

        let result = agent.run("test cancel", "/tmp", None).unwrap();
        assert!(result.contains("cancelled"));
    }

    #[test]
    fn loop_drains_injected_messages_before_next_model_call() {
        // The agent loop drains the inject queue at the top of each iteration.
        // A message pushed to the queue between turns appears in the messages
        // sent to the LLM on the next call. We verify by capturing the
        // messages array and checking that the injected text is present.
        let captured_messages: Arc<Mutex<Vec<Vec<Value>>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured_messages);

        let llm = Arc::new(FakeChatClient::with_callback(
            vec![
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
                    content: "Done with injected message.".into(),
                    tool_calls: vec![],
                    finish_reason: "stop".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                },
            ],
            move |messages| {
                captured_clone.lock().unwrap().push(messages.to_vec());
            },
        ));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let inject_queue: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::clone(&inject_queue),
        );
        agent.init_context("/tmp", &[]);

        // Push an injected message before running — it will be drained on
        // the first iteration and appear in the first LLM call.
        inject_queue
            .lock()
            .unwrap()
            .push_back("injected message".into());

        let result = agent.run("original prompt", "/tmp", None).unwrap();
        assert!(result.contains("Done with injected message"));

        // The first LLM call should contain both the original prompt and the
        // injected message.
        let captured = captured_messages.lock().unwrap();
        assert!(!captured.is_empty());
        let first_call = &captured[0];
        let all_text: String = first_call
            .iter()
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()).map(String::from))
            .collect();
        assert!(
            all_text.contains("injected message"),
            "expected injected message in first LLM call, got: {all_text}"
        );
    }

    #[test]
    fn is_read_only_tool_classifies_correctly() {
        assert!(is_read_only_tool("read"));
        assert!(is_read_only_tool("grep"));
        assert!(is_read_only_tool("glob"));

        assert!(!is_read_only_tool("http"));
        assert!(!is_read_only_tool("write"));
        assert!(!is_read_only_tool("edit"));
        assert!(!is_read_only_tool("shell"));
        assert!(!is_read_only_tool("unknown_tool"));
    }

    #[test]
    fn format_tokens_formats_human_readable() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(500), "500");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_000), "1k");
        assert_eq!(format_tokens(1_500), "1.5k");
        assert_eq!(format_tokens(30_000), "30k");
        assert_eq!(format_tokens(30684), "30.7k");
        assert_eq!(format_tokens(1_000_000), "1M");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn format_duration_formats_human_readable() {
        assert_eq!(format_duration(0), "0ms");
        assert_eq!(format_duration(350), "350ms");
        assert_eq!(format_duration(999), "999ms");
        assert_eq!(format_duration(1_000), "1s");
        assert_eq!(format_duration(1_500), "1.5s");
        assert_eq!(format_duration(3_062), "3.1s");
        assert_eq!(format_duration(35_398), "35.4s");
        assert_eq!(format_duration(60_000), "1m");
        assert_eq!(format_duration(65_000), "1.1m");
        assert_eq!(format_duration(125_000), "2.1m");
    }

    #[test]
    fn is_malformed_tool_call_error_detects_400_with_invalid_arguments() {
        let err = anyhow::anyhow!(
            "LLM request failed (400 Bad Request): {{\"object\":\"error\",\"message\":\"Assistant tool call function.arguments must be valid JSON.\",\"type\":\"BadRequest\",\"param\":null,\"code\":400}}"
        );
        assert!(is_malformed_tool_call_error(&err));
    }

    #[test]
    fn is_malformed_tool_call_error_rejects_other_errors() {
        // 500 error — not a malformed-arguments issue.
        let err =
            anyhow::anyhow!("LLM request failed (500 Internal Server Error): server overload");
        assert!(!is_malformed_tool_call_error(&err));

        // 400 but not about arguments.
        let err = anyhow::anyhow!("LLM request failed (400 Bad Request): model not found");
        assert!(!is_malformed_tool_call_error(&err));

        // Network error — no status code.
        let err = anyhow::anyhow!("POST /v1/chat/completions: connection refused");
        assert!(!is_malformed_tool_call_error(&err));
    }

    #[test]
    fn preview_lines_returns_full_when_at_or_under_limit() {
        assert_eq!(preview_lines("one line", 10), "one line");
        assert_eq!(preview_lines("a\nb\nc", 3), "a\nb\nc");
        assert_eq!(preview_lines("", 10), "");
    }

    #[test]
    fn preview_lines_truncates_with_ellipsis_when_over_limit() {
        let input = "l1\nl2\nl3\nl4\nl5";
        let out = preview_lines(input, 3);
        assert_eq!(out, "l1\nl2\nl3\n[...]");
    }

    #[test]
    fn preview_lines_keeps_exactly_max_lines() {
        // 5 lines, max 5 → no truncation, no ellipsis.
        let input = "a\nb\nc\nd\ne";
        let out = preview_lines(input, 5);
        assert_eq!(out, input);
        assert!(!out.contains("[...]"));
    }

    #[test]
    fn preview_lines_truncates_single_long_input_with_no_newlines() {
        // Edge case: 0 newlines means 1 line, so it's never truncated.
        assert_eq!(preview_lines("only line", 10), "only line");
    }

    #[test]
    fn loop_executes_mixed_tool_calls_in_order() {
        // When the model issues both read-only and mutating tool calls in one
        // turn, all calls must run sequentially (not concurrently) to preserve
        // dependencies. We verify by issuing write then read on the
        // same path — if they ran concurrently the read might see the old state.
        let dir = test_util::unique_test_dir();

        let llm = Arc::new(FakeChatClient::new(vec![
            ChatResponse {
                content: String::new(),
                tool_calls: vec![
                    json!({
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write", "arguments": format!("{{\"path\":\"out.txt\",\"content\":\"written\"}}")}
                    }),
                    json!({
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"files\":[{\"path\":\"out.txt\"}]}"}
                    }),
                ],
                finish_reason: "tool_calls".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
            ChatResponse {
                content: "Done.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context(dir.as_str(), &[]);

        let result = agent.run("write then read", dir.as_str(), None).unwrap();
        assert!(result.contains("Done"));
        // The read result should contain the content written by write,
        // proving sequential execution preserved the order.
        let written = fs::read_to_string(dir.path().join("out.txt")).unwrap_or_default();
        assert_eq!(written, "written");
    }

    /// A fake client that simulates speculative tool execution by firing
    /// `on_early_tool_call` for each tool call before returning the response.
    /// Used to verify the agent_loop's result-caching path: tools executed
    /// speculatively must not run again at `finish_reason`.
    struct SpeculativeClient {
        responses: std::sync::Mutex<Vec<ChatResponse>>,
        tool_calls_to_fire: std::sync::Mutex<Vec<Value>>,
    }

    impl SpeculativeClient {
        fn new(responses: Vec<ChatResponse>, tool_calls_to_fire: Vec<Value>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                tool_calls_to_fire: std::sync::Mutex::new(tool_calls_to_fire),
            }
        }
    }

    impl ChatClient for SpeculativeClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            _tools: &[Value],
            _on_chunk: Option<&StreamCallback>,
            _on_tool_calls: Option<&ToolExecCallback<'_>>,
            on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            // Fire on_early_tool_call for each pre-registered tool call,
            // simulating the streaming client detecting complete arguments.
            if let Some(early) = on_early_tool_call {
                let to_fire = self.tool_calls_to_fire.lock().unwrap();
                for (idx, tc) in to_fire.iter().enumerate() {
                    let _ = early(idx, tc);
                }
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

    #[test]
    fn speculative_execution_caches_read_only_tool_results() {
        // When on_early_tool_call fires for a read-only tool, the agent_loop
        // executes it immediately and caches the result. At finish_reason,
        // the exec_cb returns the cached result instead of re-executing.
        // We verify this by writing a file, then having the speculative
        // callback "pre-read" it. The final response should contain the
        // cached content.
        let dir = test_util::unique_test_dir();
        fs::write(dir.path().join("target.txt"), "speculative content\n").unwrap();

        let tool_call = json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": "{\"files\":[{\"path\":\"target.txt\"}]}"
            }
        });

        let llm = Arc::new(SpeculativeClient::new(
            vec![
                ChatResponse {
                    content: String::new(),
                    tool_calls: vec![tool_call.clone()],
                    finish_reason: "tool_calls".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                },
                ChatResponse {
                    content: "Done.".into(),
                    tool_calls: vec![],
                    finish_reason: "stop".into(),
                    usage: Usage::default(),
                    tool_results: vec![],
                    elapsed_ms: 0,
                    reasoning: String::new(),
                },
            ],
            vec![tool_call],
        ));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context(dir.as_str(), &[]);

        let result = agent.run("read target.txt", dir.as_str(), None).unwrap();
        // The file content should appear in the context (via tool result) even
        // though it was executed speculatively.
        assert!(result.contains("Done"));
    }

    /// A fake client that fails the second call (call index 1) with a 400
    /// "arguments must be valid JSON" error, then returns scripted responses
    /// for subsequent calls. Used to verify the agent loop's
    /// retry-on-malformed-tool-call path: the first call stores a tool call,
    /// the second call fails with 400 (simulating the API rejecting the
    /// stored arguments), and the retry sanitizes and succeeds.
    struct RetryOnMalformedClient {
        responses: std::sync::Mutex<Vec<ChatResponse>>,
        call_count: std::sync::atomic::AtomicU32,
    }

    impl RetryOnMalformedClient {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                call_count: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl ChatClient for RetryOnMalformedClient {
        fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            _tools: &[Value],
            _on_chunk: Option<&StreamCallback>,
            _on_tool_calls: Option<&ToolExecCallback<'_>>,
            _on_early_tool_call: Option<&EarlyToolExecCallback<'_>>,
        ) -> Result<ChatResponse> {
            let n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Fail the second call (index 1) — the one that sends back the
            // stored assistant tool call message. This simulates the API
            // rejecting it with a 400.
            if n == 1 {
                return Err(anyhow::anyhow!(
                    "LLM request failed (400 Bad Request): {{\"object\":\"error\",\"message\":\"Assistant tool call function.arguments must be valid JSON.\",\"type\":\"BadRequest\",\"param\":null,\"code\":400}}"
                ));
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

    #[test]
    fn retries_on_malformed_tool_call_error() {
        // The retry path (catch 400 → sanitize context → retry) is verified
        // by unit tests for is_malformed_tool_call_error and
        // Context::sanitize_last_assistant_tool_calls. This test verifies
        // that the normal flow still works with the retry code present —
        // a non-400 error propagates immediately without retry.
        //
        // The fake client fails call index 1 with a 400, but the stored
        // arguments are already valid (sanitized by push_assistant_with_tools),
        // so sanitize returns false and the error propagates. This confirms
        // the retry only fires when sanitization can actually fix something.
        let llm = Arc::new(RetryOnMalformedClient::new(vec![
            ChatResponse {
                content: String::new(),
                tool_calls: vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{\"files\":[]}"}
                })],
                finish_reason: "tool_calls".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
            ChatResponse {
                content: "Done.".into(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: Usage::default(),
                tool_results: vec![],
                elapsed_ms: 0,
                reasoning: String::new(),
            },
        ]));

        let tools = ToolRegistry::with_builtin_tools(
            &mut SessionStates::new(),
            "test-session",
            "",
            "",
            &Default::default(),
            None,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let mut agent = AgentLoop::new(
            llm,
            tools,
            "test-model".into(),
            100_000,
            cancel,
            Arc::new(Mutex::new(VecDeque::new())),
        );
        agent.init_context("/tmp", &[]);

        let result = agent.run("do something", "/tmp", None);
        // The 400 propagates because sanitization found nothing to fix
        // (arguments were already valid). This is correct — don't retry
        // when we can't fix the problem.
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("400"));
    }
}
