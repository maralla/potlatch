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
        )
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

    /// Enforce the token budget using tiered retention:
    /// 1. Compact large tool outputs into summaries
    /// 2. Truncate exploration artifacts
    /// 3. Evict oldest evictable entries
    pub fn enforce_budget(&mut self) {
        let threshold = (self.token_budget as f64 * 0.75) as usize;
        if self.total_tokens <= threshold {
            return;
        }

        // Phase 1: Compact large ShellOutput / FileRead / WebFetch entries
        for entry in &mut self.entries {
            if !entry.kind.is_evictable() {
                continue;
            }
            if entry.tokens < 500 {
                continue;
            }
            if matches!(
                entry.kind,
                ContextKind::ShellOutput | ContextKind::FileRead | ContextKind::WebFetch
            ) {
                let summary = compact_summary(&entry.content, &entry.kind);
                let new_tokens = Self::estimate_tokens(&summary) + 4;
                self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
                entry.content = summary;
                entry.tokens = new_tokens;
            }
            if self.total_tokens <= threshold {
                return;
            }
        }

        // Phase 2: Truncate Exploration entries
        for entry in &mut self.entries {
            if entry.kind != ContextKind::Exploration || entry.tokens < 200 {
                continue;
            }
            let truncated = truncate_lines(&entry.content, 20);
            let new_tokens = Self::estimate_tokens(&truncated) + 4;
            self.total_tokens = self.total_tokens - entry.tokens + new_tokens;
            entry.content = truncated;
            entry.tokens = new_tokens;
            if self.total_tokens <= threshold {
                return;
            }
        }

        // Phase 3: Evict oldest evictable entries
        let mut i = 0;
        while self.total_tokens > threshold && i < self.entries.len() {
            if self.entries[i].kind.is_evictable() {
                let removed = self.entries.remove(i);
                self.total_tokens -= removed.tokens;
            } else {
                i += 1;
            }
        }
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

fn compact_summary(content: &str, kind: &ContextKind) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let char_count = content.len();
    let label = match kind {
        ContextKind::ShellOutput => "shell output",
        ContextKind::FileRead => "file read",
        ContextKind::WebFetch => "web fetch",
        _ => "tool result",
    };
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

        ctx.enforce_budget();

        // System prompt, user prompt, and edit result must survive
        let kinds: Vec<&ContextKind> = ctx.entries().iter().map(|e| &e.kind).collect();
        assert!(kinds.contains(&&ContextKind::System));
        assert!(kinds.contains(&&ContextKind::UserPrompt));
        assert!(kinds.contains(&&ContextKind::EditResult));
    }

    #[test]
    fn compaction_reduces_large_entries() {
        let mut ctx = Context::new(500);
        ctx.push(Role::System, ContextKind::System, "sys");
        ctx.push(
            Role::Tool,
            ContextKind::ShellOutput,
            "output line\n".repeat(200),
        );
        let before = ctx.total_tokens();
        ctx.enforce_budget();
        let after = ctx.total_tokens();
        assert!(after < before, "compaction should reduce tokens");
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
