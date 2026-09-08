//! Structured conversation context with tiered retention.
//!
//! The context is NOT a flat `Vec<Value>` of messages. It uses structured entries
//! with provenance, token accounting, and a tiered eviction policy that preserves
//! the agent's decisions and edits while shedding exploration artifacts.

use std::sync::OnceLock;

use serde_json::{Value, json};

/// Cached BPE encoder for token estimation. Initialized once, reused across all calls.
static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();

/// Version tag written into persisted context snapshots. Bump when the
/// snapshot shape changes so an old snapshot never silently misrestores.
const SNAPSHOT_VERSION: u32 = 1;

/// Number of recent assistant text entries to protect from eviction. Older
/// assistant text is evictable since the model's intermediate narration
/// ("Let me check...", "I'll now edit...") is low-value once the action is done.
const KEEP_LAST_ASSISTANT_TEXT: usize = 3;

/// Number of recent tool-call entries whose `reasoning_content` is preserved.
/// Older tool-call entries have their reasoning stripped during budget
/// pressure (the `tool_calls` JSON itself is always kept — only the thinking
/// trace is dropped, since the tool result already reflects the outcome).
const KEEP_LAST_REASONING: usize = 2;

/// Categories of context entries. Drives the tiered retention policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextKind {
    /// System prompt — never evicted.
    System,
    /// Original user prompt — never evicted.
    UserPrompt,
    /// Assistant text response — last N are never evicted.
    AssistantText,
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
            ContextKind::FileRead
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
    /// `reasoning_content` was stripped from old tool-call entries to reclaim
    /// tokens without breaking the tool-call chain. The `tool_calls` JSON and
    /// recent reasoning (last [`KEEP_LAST_REASONING`] turns) are preserved.
    ReasoningStripped,
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

    /// Strip `reasoning_content` from old tool-call entries, preserving the
    /// `tool_calls` JSON and the reasoning on the most recent
    /// [`KEEP_LAST_REASONING`] tool-call entries. Returns true if any entry
    /// was modified. This reclaims tokens from stale reasoning without breaking
    /// the tool-call chain (which the OpenAI API requires to stay intact).
    fn strip_old_reasoning(&mut self) -> bool {
        // Count tool-call entries from the end to find the cutoff: entries with
        // fewer than KEEP_LAST_REASONING tool-call entries after them keep
        // their reasoning.
        let mut tool_call_after: Vec<usize> = vec![0; self.entries.len()];
        let mut running = 0usize;
        for i in (0..self.entries.len()).rev() {
            tool_call_after[i] = running;
            if self.entries[i].kind == ContextKind::ToolCall {
                running += 1;
            }
        }

        let mut changed = false;
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if entry.kind != ContextKind::ToolCall || tool_call_after[i] < KEEP_LAST_REASONING {
                continue;
            }
            let Ok(mut msg) = serde_json::from_str::<Value>(&entry.content) else {
                continue;
            };
            if msg.get("reasoning_content").is_none() {
                continue;
            }
            msg.as_object_mut()
                .expect("assistant tool-call message is a JSON object")
                .remove("reasoning_content");
            let new_content = msg.to_string();
            let new_tokens = Self::estimate_tokens(&new_content) + 4;
            self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
            entry.content = new_content;
            entry.tokens = new_tokens;
            changed = true;
        }
        changed
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
    ///
    /// Reasoning content is re-injected into the stored message so the model
    /// can see its prior reasoning on the next turn and build on it instead of
    /// re-deriving the same conclusions. The `reasoning_content` field is the
    /// convention used by Model1, DeepSeek R1, and other reasoning models for
    /// their thinking trace; backends that don't recognize it simply ignore it.
    ///
    /// Tool call arguments are sanitized: if the streamed `arguments` string
    /// is empty or not valid JSON, it is replaced with `"{}"`. This prevents
    /// the next API call from rejecting the conversation with a 400
    /// "function.arguments must be valid JSON" error when the model emits
    /// a malformed or truncated tool call.
    pub fn push_assistant_with_tools(
        &mut self,
        text: Option<&str>,
        tool_calls: &[Value],
        reasoning: &str,
    ) {
        let sanitized = sanitize_tool_calls(tool_calls);
        let mut msg = if let Some(t) = text {
            json!({
                "role": "assistant",
                "content": t,
                "tool_calls": sanitized,
            })
        } else {
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": sanitized,
            })
        };
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        let content_str = msg.to_string();
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

    /// Append a plain assistant text response, optionally with reasoning.
    pub fn push_assistant_text(&mut self, text: &str) {
        self.push(Role::Assistant, ContextKind::AssistantText, text);
    }

    /// Append an assistant text response with reasoning content re-injected,
    /// so the model can see its prior reasoning on the next turn. Used for
    /// `stop` turns where the model produced a final text answer after thinking.
    pub fn push_assistant_text_with_reasoning(&mut self, text: &str, reasoning: &str) {
        if reasoning.is_empty() {
            self.push_assistant_text(text);
            return;
        }
        let msg = json!({
            "role": "assistant",
            "content": text,
            "reasoning_content": reasoning,
        });
        let content_str = msg.to_string();
        let tokens = Self::estimate_tokens(&content_str) + 4;
        self.total_tokens += tokens;
        self.entries.push(ContextEntry {
            role: Role::Assistant,
            kind: ContextKind::AssistantText,
            content: content_str,
            tokens,
            tool_call_id: None,
        });
    }

    /// Sanitize tool-call arguments in the last assistant entry. Called as a
    /// recovery path when the API rejects the conversation with a 400
    /// "function.arguments must be valid JSON" error: rewrites the stored
    /// JSON in place so every tool call's `arguments` is valid JSON. Returns
    /// true if any entry was modified.
    pub fn sanitize_last_assistant_tool_calls(&mut self) -> bool {
        let Some(entry) = self.entries.last_mut() else {
            return false;
        };
        if entry.role != Role::Assistant || entry.kind != ContextKind::ToolCall {
            return false;
        }
        let Ok(mut parsed) = serde_json::from_str::<Value>(&entry.content) else {
            return false;
        };
        let Some(tool_calls) = parsed["tool_calls"].as_array_mut() else {
            return false;
        };
        let mut changed = false;
        for tc in tool_calls.iter_mut() {
            let args = tc["function"]["arguments"].as_str().unwrap_or("");
            if args.is_empty() || serde_json::from_str::<Value>(args).is_err() {
                tc["function"]["arguments"] = json!("{}");
                changed = true;
            }
        }
        if changed {
            let new_content = parsed.to_string();
            let new_tokens = Self::estimate_tokens(&new_content) + 4;
            self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
            entry.content = new_content;
            entry.tokens = new_tokens;
        }
        changed
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

        // Phase 0: Strip reasoning_content from old tool-call entries.
        // Reasoning models (Model1, DeepSeek R1) emit a thinking trace that is
        // re-injected so the model can build on recent reasoning. But once a
        // tool call is several turns old, its reasoning is dead weight — the
        // tool result already reflects the outcome. Stripping `reasoning_content`
        // reclaims 100-500 tokens per entry without breaking the tool-call
        // chain (the `tool_calls` JSON stays intact). Recent reasoning (last
        // KEEP_LAST_REASONING turns) is preserved for chain-of-thought continuity.
        if self.strip_old_reasoning() && self.total_tokens <= target {
            return BudgetAction::ReasoningStripped;
        }
        // If still over target after stripping, continue to compaction/eviction.
        // The reclaimed headroom may avoid a collapse.

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
                    messages.push(json!({
                        "role": "system",
                        "content": entry.content,
                    }));
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

    /// Render the full context as a plain markdown transcript for
    /// persistence. Every entry appears verbatim — role, content,
    /// tool_call_id — as a readable section, so the file reads as the
    /// conversation it records. Entry content is never escaped or
    /// reflowed; it runs from its section heading to the next heading,
    /// so multi-line content restores as written. Kinds and token counts
    /// are not persisted: kinds are re-inferred from role + content on
    /// restore, and tokens are recomputed with the same estimator every
    /// context mutation already uses.
    pub fn snapshot_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Context\n\n");
        out.push_str(&format!(
            "- version: {SNAPSHOT_VERSION}\n- token budget: {}\n- entries: {}\n",
            self.token_budget,
            self.entries.len()
        ));
        for entry in &self.entries {
            out.push_str("\n## ");
            out.push_str(role_name(entry.role));
            if let Some(id) = &entry.tool_call_id {
                out.push_str(&format!(" {id}"));
            }
            out.push_str("\n\n");
            out.push_str(entry.content.as_str());
            out.push('\n');
        }
        out
    }

    /// Restore a context from a [`Self::snapshot_markdown`] transcript.
    /// Returns `None` when the text does not parse — wrong version, missing
    /// header, unknown role — leaving the caller to start a fresh context
    /// rather than resume from a corrupt one.
    ///
    /// Only `## <known role>` lines open a section; every other line —
    /// including `## ` lines with other names, like the headings inside a
    /// system prompt — is content. A content line that spells a role
    /// heading exactly would split its entry; the entry-count check
    /// catches that (the count mismatches and the file is refused),
    /// trading a rare false refusal for never misrestoring.
    pub fn restore_markdown(text: &str) -> Option<Self> {
        let mut version = None;
        let mut token_budget = None;
        let mut count = None;
        for line in text.lines() {
            if line.starts_with("## ") {
                break;
            }
            if let Some(v) = line.strip_prefix("- version: ") {
                version = v.trim().parse::<u32>().ok();
            } else if let Some(v) = line.strip_prefix("- token budget: ") {
                token_budget = v.trim().parse::<usize>().ok();
            } else if let Some(v) = line.strip_prefix("- entries: ") {
                count = v.trim().parse::<usize>().ok();
            }
        }
        if version != Some(SNAPSHOT_VERSION) {
            return None;
        }
        let token_budget = token_budget?;

        // Section boundaries: indices of lines that open a section. Only a
        // `## ` line naming a known role opens one; every other line —
        // including `## ` lines with other names, like the headings inside
        // a system prompt — is content.
        let lines: Vec<&str> = text.lines().collect();
        let mut starts = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line
                .strip_prefix("## ")
                .and_then(split_role_heading)
                .is_some()
            {
                starts.push(i);
            }
        }
        let mut entries = Vec::with_capacity(starts.len());
        for (n, &start) in starts.iter().enumerate() {
            let heading = lines[start].strip_prefix("## ").expect("checked above");
            let (role, tool_call_id) = split_role_heading(heading)?;
            let end = starts.get(n + 1).copied().unwrap_or(lines.len());
            // Content runs from the line after the heading's separator
            // blank to the line before the next heading's separator blank.
            // Interior blank lines are content, verbatim.
            let mut content_start = start + 1;
            if lines.get(content_start) == Some(&"") {
                content_start += 1;
            }
            let mut content_end = end;
            if content_end > content_start && lines.get(content_end - 1) == Some(&"") {
                content_end -= 1;
            }
            let content = lines[content_start..content_end].join("\n");
            entries.push(ContextEntry {
                role,
                kind: ContextKind::ToolResult,
                content,
                tokens: 0,
                tool_call_id,
            });
        }
        if count.is_some_and(|n| n != entries.len()) {
            return None;
        }
        for entry in &mut entries {
            finish_entry_in_place(entry);
        }
        let mut restored = Self {
            entries,
            total_tokens: 0,
            token_budget,
        };
        restored.recount_tokens();
        Some(restored)
    }

    /// Recompute per-entry and total token counts with the standard
    /// estimator. Every in-memory mutation already maintains exactly this
    /// arithmetic, so a restored context's budget enforcement behaves the
    /// same as one that never left memory.
    fn recount_tokens(&mut self) {
        let mut total = 0;
        for entry in &mut self.entries {
            entry.tokens = Self::estimate_tokens(&entry.content) + 4;
            total += entry.tokens;
        }
        self.total_tokens = total;
    }
}

/// Summarize a large tool output into a compact form: a header showing the
/// Return a copy of `tool_calls` where every call's `function.arguments` is
/// valid JSON. The model sometimes streams empty or truncated arguments (e.g.
/// `""` or a partial `{"path":`), which the API rejects with a 400 on the next
/// turn. Replacing invalid arguments with `"{}"` lets the conversation
/// continue — the tool returns an error, and the model retries with proper
/// arguments.
fn sanitize_tool_calls(tool_calls: &[Value]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|tc| {
            let args = tc["function"]["arguments"].as_str().unwrap_or("");
            if !args.is_empty() && serde_json::from_str::<Value>(args).is_ok() {
                return tc.clone();
            }
            // Arguments are empty or invalid — replace with "{}".
            let mut fixed = tc.clone();
            fixed["function"]["arguments"] = json!("{}");
            fixed
        })
        .collect()
}

/// Split a `## <role>` or `## <role> <tool_call_id>` section heading.
/// Returns `None` when the role is unknown.
fn split_role_heading(heading: &str) -> Option<(Role, Option<String>)> {
    let heading = heading.trim();
    let (name, tool_call_id) = match heading.split_once(' ') {
        Some((name, id)) => (name, Some(id.to_string())),
        None => (heading, None),
    };
    Some((role_from_name(name)?, tool_call_id))
}

/// Infer the entry kind a restored section carries, from its role and
/// content, in place. Assistant content that parses as JSON carrying
/// `tool_calls` is a tool call; everything else maps by role. Kinds are
/// not persisted — the inferred kind preserves the eviction-relevant
/// distinction (what may be compacted vs. never dropped).
fn finish_entry_in_place(entry: &mut ContextEntry) {
    entry.kind = match entry.role {
        Role::System => ContextKind::System,
        Role::User => ContextKind::UserPrompt,
        Role::Tool => ContextKind::ToolResult,
        Role::Assistant => {
            let is_tool_call = serde_json::from_str::<Value>(&entry.content)
                .ok()
                .and_then(|v| v.get("tool_calls").cloned())
                .is_some();
            if is_tool_call {
                ContextKind::ToolCall
            } else {
                ContextKind::AssistantText
            }
        }
    };
}

/// Stable wire names for [`Role`] in context snapshots.
fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn role_from_name(name: &str) -> Option<Role> {
    match name {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

/// Naive compaction: preserve the first few and last few lines of a tool
/// output, with a marker showing the original size, the first few lines (most
/// relevant), a marker, and the last few lines (often contains errors or final
/// results).
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
    fn snapshot_roundtrip_preserves_conversation_and_tokens() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "you are a coder");
        ctx.push(Role::User, ContextKind::UserPrompt, "write hello world");
        ctx.push_assistant_with_tools(
            Some("I'll create the file"),
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "write", "arguments": "{\"path\":\"hi.py\",\"content\":\"print('hi')\"}"}
            })],
            "think about the path first",
        );
        ctx.push_tool_result(ContextKind::EditResult, "wrote hi.py", "call_1");
        ctx.push_assistant_text("done");
        let before_tokens = ctx.total_tokens();

        let transcript = ctx.snapshot_markdown();
        let restored = Context::restore_markdown(&transcript).expect("transcript restores");
        // Tokens are recomputed with the same estimator, so the restored
        // context resumes with identical budget math.
        assert_eq!(restored.total_tokens(), before_tokens);
        assert_eq!(ctx.entries().len(), restored.entries().len());

        // The restored conversation serializes to the same messages.
        let original_msgs = ctx.to_messages();
        let restored_msgs = restored.to_messages();
        assert_eq!(original_msgs.len(), restored_msgs.len());
        assert_eq!(restored_msgs[0]["content"], "you are a coder");
        assert_eq!(restored_msgs[1]["content"], "write hello world");
        assert!(restored_msgs[2]["tool_calls"].is_array());
        assert_eq!(restored_msgs[3]["tool_call_id"], "call_1");
        assert_eq!(restored_msgs[4]["content"], "done");
        // Reasoning was persisted inside the assistant tool-call entry's
        // JSON content and survives the roundtrip.
        assert_eq!(
            restored_msgs[2]["reasoning_content"],
            "think about the path first"
        );
    }

    #[test]
    fn restore_rejects_foreign_versions_and_garbage() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "body text");
        let transcript = ctx.snapshot_markdown();

        // A future version is rejected rather than misrestored.
        let future = transcript.replace("- version: 1\n", "- version: 4294967295\n");
        assert!(Context::restore_markdown(&future).is_none());

        // Missing header and unknown role are rejected: a corrupt
        // transcript starts a fresh context instead of misrestoring.
        assert!(Context::restore_markdown("").is_none());
        assert!(Context::restore_markdown("## wizard\n\nx\n").is_none());

        // A content line that spells a role heading splits its entry and
        // changes the section count; the count check refuses the file.
        let split = transcript.replace(
            "body text",
            "body text\n\n## user\n\nsneaky heading inside content",
        );
        assert!(
            Context::restore_markdown(&split).is_none(),
            "a content line that looks like a heading changes the entry count and must be refused"
        );

        // A `## ` heading that is not a role name — like the headings
        // inside a system prompt — is content, not a section boundary.
        let with_inner_heading =
            transcript.replace("body text", "body text\n\n## Workspace\n\nthe repo root");
        assert_eq!(
            Context::restore_markdown(&with_inner_heading)
                .expect("non-role headings are content")
                .entries()
                .len(),
            ctx.entries().len()
        );
    }

    #[test]
    fn snapshot_markdown_preserves_multi_line_content_verbatim() {
        // Tool outputs and file reads are multi-line; the transcript must
        // carry them unescaped and restore them byte-for-byte.
        let mut ctx = Context::new(100_000);
        ctx.push(
            Role::Tool,
            ContextKind::ShellOutput,
            "line one\nline two\n\nindented:\n    keep me\ntrailing spaces   ",
        );
        let transcript = ctx.snapshot_markdown();
        let restored = Context::restore_markdown(&transcript).unwrap();
        assert_eq!(restored.entries()[0].content, ctx.entries()[0].content);
    }

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
                    "function": {"name": "edit", "arguments": "{}"}
                })],
                "",
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
                "",
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
                "function": {"name": "write", "arguments": "{\"path\":\"hi.py\",\"content\":\"print('hi')\"}"}
            })],
            "",
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

    #[test]
    fn sanitize_tool_calls_replaces_empty_arguments() {
        let tool_calls = vec![
            json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": ""}
            }),
            json!({
                "id": "call_2",
                "type": "function",
                "function": {"name": "grep", "arguments": "{\"pattern\":\"foo\"}"}
            }),
        ];
        let sanitized = sanitize_tool_calls(&tool_calls);
        // Empty arguments replaced with "{}".
        assert_eq!(sanitized[0]["function"]["arguments"], "{}");
        // Valid arguments left unchanged.
        assert_eq!(
            sanitized[1]["function"]["arguments"],
            "{\"pattern\":\"foo\"}"
        );
    }

    #[test]
    fn sanitize_tool_calls_replaces_invalid_json_arguments() {
        // The model streamed a partial JSON object — not valid JSON.
        let tool_calls = vec![json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": "{\"path\":"}
        })];
        let sanitized = sanitize_tool_calls(&tool_calls);
        assert_eq!(sanitized[0]["function"]["arguments"], "{}");
    }

    #[test]
    fn sanitize_tool_calls_preserves_valid_arguments() {
        let tool_calls = vec![json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": "{\"files\":[{\"path\":\"a.rs\"}]}"}
        })];
        let sanitized = sanitize_tool_calls(&tool_calls);
        assert_eq!(
            sanitized[0]["function"]["arguments"],
            "{\"files\":[{\"path\":\"a.rs\"}]}"
        );
    }

    #[test]
    fn push_assistant_with_tools_sanitizes_empty_arguments() {
        // When the model streams empty arguments, the stored context entry
        // must have "{}" so the next API call doesn't reject with a 400.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_with_tools(
            None,
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": ""}
            })],
            "",
        );
        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(
            assistant_msg["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn sanitize_last_assistant_tool_calls_fixes_invalid_arguments() {
        // Simulate a stored assistant entry with invalid arguments (as if
        // the model streamed a truncated tool call that bypassed sanitization
        // — e.g. from an older session or a bug). The method should fix it.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        // Manually push an assistant entry with invalid arguments by
        // constructing the raw JSON.
        let raw = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": ""}
            }]
        })
        .to_string();
        ctx.push(Role::Assistant, ContextKind::ToolCall, raw);

        let changed = ctx.sanitize_last_assistant_tool_calls();
        assert!(changed);

        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert_eq!(
            assistant_msg["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn sanitize_last_assistant_tool_calls_returns_false_when_nothing_to_fix() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_with_tools(
            None,
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"files\":[]}"}
            })],
            "",
        );
        // Arguments are already valid — no change.
        assert!(!ctx.sanitize_last_assistant_tool_calls());
    }

    #[test]
    fn sanitize_last_assistant_tool_calls_returns_false_for_text_entry() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_text("just text, no tool calls");
        assert!(!ctx.sanitize_last_assistant_tool_calls());
    }

    #[test]
    fn push_assistant_with_tools_stores_reasoning_when_present() {
        // Reasoning captured from the stream must be re-injected into the
        // stored assistant message so the model can build on its prior thinking.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_with_tools(
            Some("running edit"),
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "edit", "arguments": "{}"}
            })],
            "I should edit the file then verify.",
        );
        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(
            assistant_msg["reasoning_content"],
            "I should edit the file then verify."
        );
        assert_eq!(assistant_msg["tool_calls"][0]["function"]["name"], "edit");
    }

    #[test]
    fn push_assistant_with_tools_omits_reasoning_when_empty() {
        // Empty reasoning must not produce an empty reasoning_content field,
        // which could confuse backends or waste tokens.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_with_tools(
            None,
            &[json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"}
            })],
            "",
        );
        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert!(
            assistant_msg.get("reasoning_content").is_none(),
            "reasoning_content should be absent when reasoning is empty"
        );
    }

    #[test]
    fn push_assistant_text_with_reasoning_stores_reasoning() {
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_text_with_reasoning(
            "Done, the file is edited.",
            "Considered two approaches; chose the simpler one.",
        );
        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"], "Done, the file is edited.");
        assert_eq!(
            assistant_msg["reasoning_content"],
            "Considered two approaches; chose the simpler one."
        );
    }

    #[test]
    fn push_assistant_text_with_reasoning_falls_back_when_empty() {
        // Empty reasoning should produce a plain assistant text entry with no
        // reasoning_content field, matching push_assistant_text behavior.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        ctx.push_assistant_text_with_reasoning("just an answer", "");
        let messages = ctx.to_messages();
        let assistant_msg = &messages[2];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"], "just an answer");
        assert!(
            assistant_msg.get("reasoning_content").is_none(),
            "reasoning_content should be absent when reasoning is empty"
        );
    }

    #[test]
    fn strip_old_reasoning_removes_stale_reasoning_keeps_recent() {
        // With KEEP_LAST_REASONING=2, the last 2 tool-call entries keep their
        // reasoning; older ones have it stripped. The tool_calls JSON itself
        // must survive so the OpenAI conversation stays valid.
        // Use a small budget so the total exceeds the 60% trigger and forces
        // enforce_budget to actually run its reduction phases.
        let mut ctx = Context::new(200);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        // Four tool-call turns, each with reasoning.
        for i in 0..4 {
            ctx.push_assistant_with_tools(
                Some(&format!("step {i}")),
                &[json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"}
                })],
                &format!("reasoning for step {i} that is long enough to matter"),
            );
            ctx.push_tool_result(
                ContextKind::ShellOutput,
                format!("result {i}"),
                format!("call_{i}"),
            );
        }
        // Sanity: total must exceed the 60% trigger (120) for stripping to run.
        assert!(
            ctx.total_tokens() > 120,
            "test setup should exceed trigger, got {}",
            ctx.total_tokens()
        );

        let before_tokens = ctx.total_tokens();
        let action = ctx.enforce_budget(None, None);
        // Stripping should have run. It may bring us under target (60) or not;
        // either way reasoning on old entries is gone.
        assert!(
            action == BudgetAction::ReasoningStripped
                || action == BudgetAction::Truncated
                || action == BudgetAction::Evicted,
            "expected a reduction action, got {action:?}"
        );

        let messages = ctx.to_messages();
        // Find the assistant tool-call messages (entries with tool_calls).
        let tool_call_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("tool_calls").is_some())
            .collect();
        assert_eq!(
            tool_call_msgs.len(),
            4,
            "tool-call entries must not be evicted"
        );

        // The last 2 keep reasoning; the first 2 have it stripped.
        assert!(
            tool_call_msgs[0].get("reasoning_content").is_none(),
            "oldest tool-call reasoning should be stripped"
        );
        assert!(
            tool_call_msgs[1].get("reasoning_content").is_none(),
            "second-oldest tool-call reasoning should be stripped"
        );
        assert_eq!(
            tool_call_msgs[2]["reasoning_content"],
            "reasoning for step 2 that is long enough to matter"
        );
        assert_eq!(
            tool_call_msgs[3]["reasoning_content"],
            "reasoning for step 3 that is long enough to matter"
        );
        // tool_calls JSON is intact on all entries.
        for (i, msg) in tool_call_msgs.iter().enumerate() {
            assert_eq!(msg["tool_calls"][0]["function"]["name"], "shell");
            assert_eq!(msg["tool_calls"][0]["id"], format!("call_{i}"));
        }
        // Tokens should not have increased.
        assert!(
            ctx.total_tokens() <= before_tokens,
            "stripping reasoning should not increase tokens"
        );
    }

    #[test]
    fn strip_old_reasoning_preserves_entries_without_reasoning() {
        // Tool-call entries that never had reasoning_content should be
        // untouched (no spurious modification, no token change).
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        for i in 0..3 {
            ctx.push_assistant_with_tools(
                Some(&format!("step {i}")),
                &[json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"}
                })],
                "", // no reasoning
            );
            ctx.push_tool_result(
                ContextKind::ShellOutput,
                format!("result {i}"),
                format!("call_{i}"),
            );
        }
        let before = ctx.total_tokens();
        let action = ctx.enforce_budget(None, None);
        assert_eq!(action, BudgetAction::None, "nothing to strip");
        assert_eq!(ctx.total_tokens(), before, "tokens unchanged");
    }

    #[test]
    fn strip_old_reasoning_keeps_recent_two_when_fewer_entries() {
        // With only 2 tool-call entries (== KEEP_LAST_REASONING), nothing is
        // stripped even if reasoning is present.
        let mut ctx = Context::new(100_000);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(Role::User, ContextKind::UserPrompt, "do task");
        for i in 0..2 {
            ctx.push_assistant_with_tools(
                Some(&format!("step {i}")),
                &[json!({
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"}
                })],
                &format!("reasoning {i}"),
            );
            ctx.push_tool_result(
                ContextKind::ShellOutput,
                format!("result {i}"),
                format!("call_{i}"),
            );
        }
        let before = ctx.total_tokens();
        let action = ctx.enforce_budget(None, None);
        assert_eq!(action, BudgetAction::None);
        assert_eq!(ctx.total_tokens(), before);
        // Both still have reasoning.
        let messages = ctx.to_messages();
        let tool_call_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("tool_calls").is_some())
            .collect();
        assert_eq!(tool_call_msgs[0]["reasoning_content"], "reasoning 0");
        assert_eq!(tool_call_msgs[1]["reasoning_content"], "reasoning 1");
    }
}
