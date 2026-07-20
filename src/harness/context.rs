//! Structured conversation context with tiered retention.
//!
//! The context is NOT a flat `Vec<Value>` of messages. It uses structured entries
//! with provenance, token accounting, and a tiered eviction policy that preserves
//! the agent's decisions and edits while shedding exploration artifacts.

use std::sync::OnceLock;

use serde_json::{Value, json};

/// Cached BPE encoder for token estimation. Initialized once, reused across all calls.
static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();

/// Categories of context entries. Drives the tiered retention policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextKind {
    /// System prompt — never evicted.
    System,
    /// Original user prompt — never evicted.
    UserPrompt,
    /// Assistant text response — last N are never evicted.
    AssistantText,
    /// Reasoning content — evictable (intermediate thinking).
    Reasoning,
    /// A tool call the assistant issued — never evicted (the agent must remember its actions).
    ToolCall,
    /// Result of a file edit — never evicted (the agent must remember every change).
    EditResult,
    /// Result of a file read — compactable then evictable.
    FileRead,
    /// Result of a shell command — compactable then evictable.
    ShellOutput,
    /// Result of grep/glob/search — truncatable then evictable.
    Exploration,
    /// Result of web fetch — compactable then evictable.
    WebFetch,
    /// Generic tool result — evictable.
    ToolResult,
}

impl ContextKind {
    /// Whether this entry can ever be evicted or compacted.
    fn is_evictable(&self) -> bool {
        matches!(
            self,
            ContextKind::Reasoning
                | ContextKind::FileRead
                | ContextKind::ShellOutput
                | ContextKind::Exploration
                | ContextKind::WebFetch
                | ContextKind::ToolResult
                | ContextKind::AssistantText
        )
    }

    /// Short human-readable label for this kind, used in compaction prompts.
    fn label(&self) -> &'static str {
        match self {
            ContextKind::ShellOutput => "shell command output",
            ContextKind::FileRead => "file read output",
            ContextKind::WebFetch => "web fetch output",
            ContextKind::Exploration => "search results",
            ContextKind::Reasoning => "reasoning",
            ContextKind::ToolResult => "tool result",
            _ => "tool result",
        }
    }
}

/// Chat role for OpenAI message format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One entry in the conversation context.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub role: Role,
    pub kind: ContextKind,
    pub content: String,
    pub tokens: usize,
    /// OpenAI tool_call_id for tool role messages.
    pub tool_call_id: Option<String>,
}

/// Structured conversation context with token budget management.
pub struct Context {
    entries: Vec<ContextEntry>,
    total_tokens: usize,
    token_budget: usize,
}

/// Number of recent assistant text entries to protect from eviction. Older
/// assistant text is evictable since the model's intermediate narration
/// ("Let me check...", "I'll now edit...") is low-value once the action is done.
const KEEP_LAST_ASSISTANT_TEXT: usize = 3;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetAction {
    /// Context was within budget — no action taken.
    None,
    /// Large evictable entries were summarized via the LLM compactor.
    Compacted,
    /// Remaining evictable entries were truncated to a few lines.
    Truncated,
    /// Oldest evictable entries were removed entirely.
    Evicted,
    /// The entire conversation was collapsed into a summary (non-evictable
    /// entries exceeded the threshold).
    Collapsed,
}
/// Receives a list of `(label, content)` pairs and returns a list of summaries
/// (same length, same order). When `None`, a naive first/last-lines heuristic
/// is used instead.
pub type Compactor<'a> = dyn Fn(&[(String, String)]) -> Vec<String> + Send + Sync + 'a;

/// Callback that summarizes the entire conversation into a single text block.
/// Used when the non-evictable entries (tool calls, edit results) exceed the
/// budget and per-entry compaction can't bring it down. The conversation is
/// collapsed into a summary, breaking the tool-call chain and starting fresh.
pub type Summarizer<'a> = dyn Fn(&[ContextEntry]) -> String + Send + Sync + 'a;

impl Context {
    pub fn new(token_budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            total_tokens: 0,
            token_budget,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Whether the entry at `index` is protected from eviction/compaction.
    /// Non-evictable entries are always protected. `AssistantText` entries are
    /// protected if they're among the last `KEEP_LAST_ASSISTANT_TEXT` ones.
    fn is_protected(&self, index: usize) -> bool {
        let entry = &self.entries[index];
        if !entry.kind.is_evictable() {
            return true;
        }
        if entry.kind == ContextKind::AssistantText {
            // Count how many AssistantText entries come after this one.
            let after = self.entries[index + 1..]
                .iter()
                .filter(|e| e.kind == ContextKind::AssistantText)
                .count();
            after < KEEP_LAST_ASSISTANT_TEXT
        } else {
            false
        }
    }

    #[cfg(test)]
    pub fn entries(&self) -> &[ContextEntry] {
        &self.entries
    }

    fn estimate_tokens(text: &str) -> usize {
        let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok());
        bpe.as_ref()
            .map(|bpe| bpe.encode_with_special_tokens(text).len())
            .unwrap_or_else(|| text.len().max(1) / 4)
    }

    /// Append a new entry. Does NOT trigger eviction — call [`Self::enforce_budget`] after.
    pub fn push(&mut self, role: Role, kind: ContextKind, content: impl Into<String>) {
        let content = content.into();
        let tokens = Self::estimate_tokens(&content) + 4;
        self.total_tokens += tokens;
        self.entries.push(ContextEntry {
            role,
            kind,
            content,
            tokens,
            tool_call_id: None,
        });
    }

    /// Append a tool-result message with a tool_call_id.
    pub fn push_tool_result(
        &mut self,
        kind: ContextKind,
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) {
        let content = content.into();
        let tokens = Self::estimate_tokens(&content) + 4;
        self.total_tokens += tokens;
        self.entries.push(ContextEntry {
            role: Role::Tool,
            kind,
            content,
            tokens,
            tool_call_id: Some(tool_call_id.into()),
        });
    }

    /// Append an assistant message with tool_calls (the OpenAI format).
    pub fn push_assistant_with_tools(
        &mut self,
        text: Option<&str>,
        tool_calls: &[Value],
        reasoning: Option<&str>,
    ) {
        if let Some(r) = reasoning
            && !r.trim().is_empty()
        {
            self.push(Role::System, ContextKind::Reasoning, r);
        }
        let content_json = if let Some(t) = text {
            json!({
                "role": "assistant",
                "content": t,
                "tool_calls": tool_calls,
            })
        } else {
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": tool_calls,
            })
        };
        let content_str = content_json.to_string();
        let tokens = Self::estimate_tokens(&content_str) + 4;
        self.total_tokens += tokens;
        self.entries.push(ContextEntry {
            role: Role::Assistant,
            kind: ContextKind::ToolCall,
            content: content_str,
            tokens,
            tool_call_id: None,
        });
    }

    /// Append a plain assistant text response.
    pub fn push_assistant_text(&mut self, text: &str) {
        self.push(Role::Assistant, ContextKind::AssistantText, text);
    }

    /// Enforce the token budget using tiered retention.
    ///
    /// Phases (in order of increasing information loss):
    /// 0. **Collapse** — if non-evictable entries alone exceed the threshold,
    ///    the entire conversation is summarized into a single system message via
    ///    the `summarizer` callback. This breaks the tool-call chain (which
    ///    can't be evicted piecemeal) and starts fresh: system prompt + summary
    ///    + original user prompt.
    /// 1. **Compact** — large evictable entries are summarized in a single LLM call.
    /// 2. **Truncate** — remaining evictable entries are cut to a few lines.
    /// 3. **Evict** — oldest evictable entries are removed entirely.
    ///
    /// Non-evictable entries (system prompt, user prompt, assistant text, tool
    /// calls, edit results) are never touched by phases 1–3. Phase 0 is the
    /// escape hatch when they accumulate beyond the budget.
    pub fn enforce_budget(
        &mut self,
        compactor: Option<&Compactor<'_>>,
        summarizer: Option<&Summarizer<'_>>,
    ) -> BudgetAction {
        // Trigger compaction when context exceeds 60% of budget.
        let trigger = (self.token_budget as f64 * 0.6) as usize;
        // After compaction, reduce to 30% of budget — not just barely under the
        // trigger. This keeps the context genuinely small after compaction so
        // subsequent turns are fast, instead of hovering right at the trigger.
        let target = (self.token_budget as f64 * 0.3) as usize;

        if self.total_tokens <= trigger {
            return BudgetAction::None;
        }

        // Check if protected entries alone exceed the target.
        // If so, per-entry compaction won't help — we must collapse the conversation.
        let non_evictable_tokens: usize = self
            .entries
            .iter()
            .enumerate()
            .filter(|(i, _)| self.is_protected(*i))
            .map(|(_, e)| e.tokens)
            .sum();

        if non_evictable_tokens >= target
            && let Some(summ) = summarizer
        {
            self.collapse(summ);
            return BudgetAction::Collapsed;
        }

        // Phase 1: Compact large evictable entries into summaries.
        // Collect all entries that need compaction, then make a single LLM call.
        let to_compact: Vec<(usize, String, String)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(i, e)| !self.is_protected(*i) && e.tokens >= 200)
            .map(|(i, e)| (i, e.kind.label().to_string(), e.content.clone()))
            .collect();

        if !to_compact.is_empty() {
            let inputs: Vec<(String, String)> = to_compact
                .iter()
                .map(|(_, label, content)| (label.clone(), content.clone()))
                .collect();

            let summaries: Vec<String> = match compactor {
                Some(c) => c(&inputs),
                None => inputs
                    .iter()
                    .map(|(label, content)| compact_summary(content, label))
                    .collect(),
            };

            // Apply summaries back to entries
            for (idx, (entry_idx, _, _)) in to_compact.iter().enumerate() {
                if idx >= summaries.len() {
                    break;
                }
                let entry = &mut self.entries[*entry_idx];
                let summary = &summaries[idx];
                let new_tokens = Self::estimate_tokens(summary) + 4;
                if new_tokens < entry.tokens {
                    self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
                    entry.content = summary.clone();
                    entry.tokens = new_tokens;
                }
            }

            if self.total_tokens <= target {
                return BudgetAction::Compacted;
            }
        }

        // Phase 2: Truncate remaining evictable entries to a few lines.
        // Compute protected indices first to avoid borrow conflict.
        let protected: Vec<bool> = (0..self.entries.len())
            .map(|i| self.is_protected(i))
            .collect();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if protected[i] || entry.tokens < 100 {
                continue;
            }
            let truncated = truncate_lines(&entry.content, 10);
            let new_tokens = Self::estimate_tokens(&truncated) + 4;
            if new_tokens < entry.tokens {
                self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
                entry.content = truncated;
                entry.tokens = new_tokens;
            }
            if self.total_tokens <= target {
                return BudgetAction::Truncated;
            }
        }

        // Phase 3: Evict oldest evictable entries entirely.
        let mut i = 0;
        while self.total_tokens > target && i < self.entries.len() {
            if !self.is_protected(i) {
                let removed = self.entries.remove(i);
                self.total_tokens -= removed.tokens;
            } else {
                i += 1;
            }
        }

        BudgetAction::Evicted
    }

    /// Collapse the entire conversation into a summary, then restart with
    /// system prompt + summary + original user prompt. This breaks the tool-call
    /// chain (which can't be evicted piecemeal) and gives the model a fresh start.
    fn collapse(&mut self, summarizer: &Summarizer<'_>) {
        // Summarize the entire conversation.
        let summary = summarizer(&self.entries);

        // Preserve the system prompt entries (they're at the front).
        let system_entries: Vec<ContextEntry> = self
            .entries
            .iter()
            .filter(|e| e.role == Role::System && e.kind == ContextKind::System)
            .cloned()
            .collect();

        // Preserve the original user prompt.
        let user_prompt_entry = self
            .entries
            .iter()
            .find(|e| e.kind == ContextKind::UserPrompt)
            .cloned();

        // Rebuild: system prompt + summary + original user prompt
        let mut new_entries = Vec::new();
        let mut new_total = 0usize;

        for entry in &system_entries {
            new_total += entry.tokens;
            new_entries.push(entry.clone());
        }

        // Add the conversation summary as a system message.
        let summary_text = format!(
            "## Conversation Summary\n\n\
             The following is a summary of everything that happened so far in this task. \
             Use it as context to continue working. The detailed tool call history has been \
             compacted — refer to this summary for what has been done, what files were changed, \
             and what remains.\n\n{summary}"
        );
        let summary_tokens = Self::estimate_tokens(&summary_text) + 4;
        new_total += summary_tokens;
        new_entries.push(ContextEntry {
            role: Role::System,
            kind: ContextKind::System,
            content: summary_text,
            tokens: summary_tokens,
            tool_call_id: None,
        });

        // Re-add the original user prompt.
        if let Some(user_entry) = user_prompt_entry {
            new_total += user_entry.tokens;
            new_entries.push(user_entry);
        }

        self.entries = new_entries;
        self.total_tokens = new_total;
    }

    /// Serialize to OpenAI chat messages format.
    pub fn to_messages(&self) -> Vec<Value> {
        let mut messages = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            match entry.role {
                Role::System => {
                    if matches!(entry.kind, ContextKind::Reasoning) {
                        // Reasoning is stored as system but sent as a system note
                        messages.push(json!({
                            "role": "system",
                            "content": format!("[internal reasoning]\n{}", entry.content),
                        }));
                    } else {
                        messages.push(json!({
                            "role": "system",
                            "content": entry.content,
                        }));
                    }
                }
                Role::User => {
                    messages.push(json!({
                        "role": "user",
                        "content": entry.content,
                    }));
                }
                Role::Assistant => {
                    // Assistant entries with tool_calls are stored as raw JSON
                    if let Ok(parsed) = serde_json::from_str::<Value>(&entry.content) {
                        messages.push(parsed);
                    } else {
                        messages.push(json!({
                            "role": "assistant",
                            "content": entry.content,
                        }));
                    }
                }
                Role::Tool => {
                    messages.push(json!({
                        "role": "tool",
                        "content": entry.content,
                        "tool_call_id": entry.tool_call_id.as_deref().unwrap_or(""),
                    }));
                }
            }
        }
        messages
    }
}

/// Summarize a large tool output into a compact form: a header showing the
/// original size, the first few lines (most relevant), a marker, and the last
/// few lines (often contains errors or final results).
fn compact_summary(content: &str, label: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let char_count = content.len();
    let first_lines: Vec<&str> = lines.iter().take(5).copied().collect();
    let last_lines: Vec<&str> = if lines.len() > 10 {
        lines.iter().rev().take(3).rev().copied().collect()
    } else {
        Vec::new()
    };

    let mut summary = format!(
        "[compacted {label}, {char_count} chars, {} lines]\n",
        lines.len()
    );
    summary.push_str(&first_lines.join("\n"));
    if !last_lines.is_empty() {
        summary.push_str("\n[...truncated...]\n");
        summary.push_str(&last_lines.join("\n"));
    }
    summary
}

/// Truncate content to the first `keep` and last `keep` lines, with a marker
/// showing how many lines were dropped.
fn truncate_lines(content: &str, keep: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= keep * 2 {
        return content.to_string();
    }
    let first: Vec<&str> = lines.iter().take(keep).copied().collect();
    let last: Vec<&str> = lines.iter().rev().take(keep).rev().copied().collect();
    let dropped = lines.len() - keep * 2;
    format!(
        "{}\n[...truncated {} lines...]\n{}",
        first.join("\n"),
        dropped,
        last.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_tracks_tokens() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "hello world");
        assert!(ctx.total_tokens() > 0);
    }

    #[test]
    fn eviction_preserves_system_and_edits() {
        let mut ctx = Context::new(100);
        ctx.push(Role::System, ContextKind::System, "system prompt");
        ctx.push(Role::User, ContextKind::UserPrompt, "do a task");
        ctx.push_assistant_text("I'll help.");
        // Add large evictable entries
        for _ in 0..50 {
            ctx.push(
                Role::Tool,
                ContextKind::ShellOutput,
                "line of output\n".repeat(50),
            );
        }
        // Add an edit result (should never be evicted)
        ctx.push(Role::Tool, ContextKind::EditResult, "edited src/main.rs");

        ctx.enforce_budget(None, None);

        // System prompt, user prompt, and edit result must survive
        let kinds: Vec<&ContextKind> = ctx.entries().iter().map(|e| &e.kind).collect();
        assert!(kinds.contains(&&ContextKind::System));
        assert!(kinds.contains(&&ContextKind::UserPrompt));
        assert!(kinds.contains(&&ContextKind::EditResult));
    }

    #[test]
    fn eviction_reduces_tokens() {
        let mut ctx = Context::new(500);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(
            Role::Tool,
            ContextKind::ShellOutput,
            "output line\n".repeat(200),
        );
        let before = ctx.total_tokens();
        ctx.enforce_budget(None, None);
        let after = ctx.total_tokens();
        assert!(after < before, "eviction should reduce tokens");
    }

    #[test]
    fn compaction_preserves_first_and_last_lines() {
        let mut ctx = Context::new(500);
        ctx.push(Role::System, ContextKind::System, "sys");
        let mut content = String::new();
        for i in 0..100 {
            content.push_str(&format!("line {i}\n"));
        }
        content.push_str("final error: something failed\n");
        ctx.push(Role::Tool, ContextKind::ShellOutput, content);

        ctx.enforce_budget(None, None);

        // The compacted entry should still contain the first and last lines.
        let entries = ctx.entries();
        let shell_entry = entries
            .iter()
            .find(|e| e.kind == ContextKind::ShellOutput)
            .expect("shell output entry should survive (compacted, not evicted)");
        assert!(shell_entry.content.contains("line 0"));
        assert!(shell_entry.content.contains("final error"));
        assert!(shell_entry.content.contains("[compacted"));
    }

    #[test]
    fn compaction_then_eviction_as_pressure_increases() {
        let mut ctx = Context::new(200);
        ctx.push(
            Role::System,
            ContextKind::System,
            "system prompt that is long enough to matter",
        );
        // Add multiple large evictable entries
        for i in 0..5 {
            ctx.push(
                Role::Tool,
                ContextKind::ShellOutput,
                format!("entry {i}\n").repeat(100),
            );
        }
        let before = ctx.total_tokens();
        ctx.enforce_budget(None, None);
        let after = ctx.total_tokens();
        assert!(
            after < before,
            "should reduce tokens via compaction+eviction"
        );
        assert!(after <= 150, "should be under 75% threshold of 200");
    }

    #[test]
    fn llm_compactor_is_used_when_provided() {
        let mut ctx = Context::new(500);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(
            Role::Tool,
            ContextKind::ShellOutput,
            "line 1\nline 2\nline 3\n".repeat(100),
        );

        // Custom compactor that receives all entries at once and returns summaries.
        let compactor: &Compactor<'_> = &|entries: &[(String, String)]| {
            entries
                .iter()
                .map(|(label, content)| {
                    format!(
                        "[llm summary of {label}, {} chars]\nKey result: all good",
                        content.len()
                    )
                })
                .collect()
        };

        ctx.enforce_budget(Some(compactor), None);

        let entries = ctx.entries();
        let shell_entry = entries
            .iter()
            .find(|e| e.kind == ContextKind::ShellOutput)
            .expect("shell output should survive (compacted)");
        assert!(shell_entry.content.contains("[llm summary of"));
        assert!(shell_entry.content.contains("Key result: all good"));
    }

    #[test]
    fn llm_compactor_receives_all_entries_in_single_call() {
        let mut ctx = Context::new(200);
        ctx.push(Role::System, ContextKind::System, "sys");
        // Three large evictable entries
        for i in 0..3 {
            ctx.push(
                Role::Tool,
                ContextKind::ShellOutput,
                format!("entry {i}\n").repeat(100),
            );
        }

        // Track how many times the compactor is called — should be exactly once.
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = std::sync::Arc::clone(&call_count);
        let compactor: &Compactor<'_> = &|entries: &[(String, String)]| {
            call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Verify we received all 3 entries
            assert_eq!(
                entries.len(),
                3,
                "compactor should receive all entries at once"
            );
            entries
                .iter()
                .map(|(label, _)| format!("[summary for {label}]"))
                .collect()
        };

        ctx.enforce_budget(Some(compactor), None);
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "compactor should be called exactly once"
        );
    }

    #[test]
    fn collapse_triggers_when_non_evictable_exceeds_budget() {
        let mut ctx = Context::new(200);
        ctx.push(Role::System, ContextKind::System, "system prompt");
        ctx.push(Role::User, ContextKind::UserPrompt, "do the task");

        // Fill with non-evictable entries (assistant tool calls + edit results)
        // that alone exceed the 150-token threshold.
        for i in 0..10 {
            ctx.push_assistant_with_tools(
                Some(&format!("editing file {i}")),
                &[json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": "file_edit", "arguments": "{}"}
                })],
                None,
            );
            ctx.push_tool_result(
                ContextKind::EditResult,
                format!("edited file {i} successfully"),
                format!("call_{i}"),
            );
        }

        let before_tokens = ctx.total_tokens();
        let before_len = ctx.entries().len();
        assert!(before_tokens > 150, "should exceed threshold");

        // Summarizer that returns a short summary.
        let summarizer: &Summarizer<'_> =
            &|entries: &[ContextEntry]| format!("Summary: {} entries processed", entries.len());

        ctx.enforce_budget(None, Some(summarizer));

        let after_tokens = ctx.total_tokens();
        let after_len = ctx.entries().len();

        // Should have collapsed to: system prompt + summary + user prompt = 3 entries
        assert!(
            after_len < before_len,
            "should have fewer entries after collapse"
        );
        assert!(
            after_tokens < before_tokens,
            "should have fewer tokens after collapse"
        );
        assert!(after_tokens <= 150, "should be under threshold");

        // Verify structure: system, system (summary), user
        let kinds: Vec<&ContextKind> = ctx.entries().iter().map(|e| &e.kind).collect();
        assert!(
            kinds
                .iter()
                .all(|k| *k == &ContextKind::System || *k == &ContextKind::UserPrompt)
        );
        assert!(
            ctx.entries()
                .iter()
                .any(|e| e.content.contains("Conversation Summary"))
        );
        assert!(ctx.entries().iter().any(|e| e.content.contains("Summary:")));
    }

    #[test]
    fn collapse_preserves_system_and_user_prompt() {
        let mut ctx = Context::new(100);
        ctx.push(Role::System, ContextKind::System, "you are an agent");
        ctx.push(Role::User, ContextKind::UserPrompt, "implement feature X");
        // Non-evictable entries that exceed budget
        for i in 0..5 {
            ctx.push_assistant_with_tools(
                Some(&format!("step {i}")),
                &[json!({"id": format!("c{i}"), "type": "function", "function": {"name": "shell", "arguments": "{}"}})],
                None,
            );
            ctx.push_tool_result(
                ContextKind::EditResult,
                format!("done {i}"),
                format!("c{i}"),
            );
        }

        let summarizer: &Summarizer<'_> = &|_| "Task in progress".to_string();
        ctx.enforce_budget(None, Some(summarizer));

        // System prompt and user prompt must survive
        let contents: Vec<&str> = ctx.entries().iter().map(|e| e.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("you are an agent")));
        assert!(contents.iter().any(|c| c.contains("implement feature X")));
    }

    #[test]
    fn to_messages_produces_valid_openai_format() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "you are a coder");
        ctx.push(Role::User, ContextKind::UserPrompt, "write hello world");
        ctx.push_assistant_with_tools(
            Some("I'll create the file"),
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "file_write", "arguments": "{\"path\":\"hi.py\",\"content\":\"print('hi')\"}"}
            })],
            None,
        );
        ctx.push_tool_result(ContextKind::EditResult, "wrote hi.py", "call_1");

        let messages = ctx.to_messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert!(messages[2]["tool_calls"].is_array());
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_1");
    }
}
