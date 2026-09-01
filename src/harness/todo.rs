//! Task-tracking todo list that survives context compaction and collapse.
//!
//! The todo list lives outside the context window in an `Arc<Mutex>`. The agent
//! loop injects it as a system message before every API call (after compaction),
//! so it's always visible to the model regardless of how much context was
//! evicted. The model manages it via the `todo` tool.
//!
//! The wire format mirrors the mainstream agent todo format (Cursor TodoWrite,
//! ACP, etc.): a list of `{description, status}` items where `status` is one of
//! `pending`, `in_progress`, `completed`. The model sends the full list on
//! every call — this is a replace-all API, which keeps item indices stable
//! across updates and avoids the per-item `start`/`complete` round-trips that
//! drift out of sync when items are inserted or reordered.

use std::sync::{Arc, Mutex};

/// Status of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    fn label(&self) -> &'static str {
        match self {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Completed => "[x]",
        }
    }

    /// Parse a status string from the tool's wire format. Returns None for
    /// unrecognized values so the tool can surface a clear error.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            _ => None,
        }
    }
}

/// A single todo item.
#[derive(Debug, Clone)]
pub struct TodoItem {
    pub description: String,
    pub status: TodoStatus,
}

/// Shared todo list, safe to access from the tool and the agent loop.
#[derive(Debug, Clone)]
pub struct TodoList {
    items: Arc<Mutex<Vec<TodoItem>>>,
}

impl TodoList {
    pub fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Replace the entire list with `items`. This is the only mutating API:
    /// the model sends the full desired state on every call, which keeps
    /// indices stable across updates (insertions, reordering, status changes)
    /// and avoids per-item round-trips.
    pub fn replace_all(&self, items: Vec<TodoItem>) {
        *self.items.lock().unwrap() = items;
    }

    /// Render the todo list as a formatted string for injection into the system
    /// prompt. Returns None if the list is empty.
    ///
    /// The header states provenance explicitly — this list was set by the
    /// model's own `todo` calls and is re-injected by the harness each turn —
    /// and each item carries a stable harness-assigned ID. This is a guard
    /// against a specific failure seen with models whose training data
    /// contains a similar-looking planner checklist: when the injected list
    /// resembles their own planning template, they continue that template in
    /// their reasoning, inventing items the harness never sent, and then
    /// "synchronize" the tool to those phantom items. Naming the source and
    /// using IDs makes the real list unambiguous and any phantom item
    /// visibly foreign.
    pub fn render(&self) -> Option<String> {
        let items = self.items.lock().unwrap();
        if items.is_empty() {
            return None;
        }
        let mut out = String::from(
            "## Harness todo list (this exact list was set by your own `todo` tool calls and \
             is re-injected here by the harness every turn — it contains nothing else, and any \
             item or note about it that appears in your own reasoning but not in this list is \
             not part of this list)\n\n",
        );
        for (i, item) in items.iter().enumerate() {
            out.push_str(&format!(
                "T{i} {} {}\n",
                item.status.label(),
                item.description
            ));
        }
        Some(out)
    }
}

impl Default for TodoList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(desc: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            description: desc.into(),
            status,
        }
    }

    #[test]
    fn replace_all_creates_list() {
        let todo = TodoList::new();
        todo.replace_all(vec![
            item("read files", TodoStatus::Pending),
            item("edit code", TodoStatus::Pending),
        ]);
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Harness todo list (this exact list was set by your own `todo` tool calls and is re-injected here by the harness every turn — it contains nothing else, and any item or note about it that appears in your own reasoning but not in this list is not part of this list)\n\nT0 [ ] read files\nT1 [ ] edit code\n"
        );
    }

    #[test]
    fn replace_all_preserves_status_from_items() {
        let todo = TodoList::new();
        todo.replace_all(vec![
            item("task A", TodoStatus::InProgress),
            item("task B", TodoStatus::Completed),
            item("task C", TodoStatus::Pending),
        ]);
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Harness todo list (this exact list was set by your own `todo` tool calls and is re-injected here by the harness every turn — it contains nothing else, and any item or note about it that appears in your own reasoning but not in this list is not part of this list)\n\nT0 [~] task A\nT1 [x] task B\nT2 [ ] task C\n"
        );
    }

    #[test]
    fn replace_all_overwrites_previous_list() {
        let todo = TodoList::new();
        todo.replace_all(vec![item("old task", TodoStatus::InProgress)]);
        todo.replace_all(vec![
            item("new task 1", TodoStatus::Pending),
            item("new task 2", TodoStatus::Pending),
        ]);
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Harness todo list (this exact list was set by your own `todo` tool calls and is re-injected here by the harness every turn — it contains nothing else, and any item or note about it that appears in your own reasoning but not in this list is not part of this list)\n\nT0 [ ] new task 1\nT1 [ ] new task 2\n"
        );
    }

    #[test]
    fn render_empty_returns_none() {
        let todo = TodoList::new();
        assert!(todo.render().is_none());
    }

    #[test]
    fn status_parse_recognizes_known_values() {
        assert_eq!(TodoStatus::parse("pending"), Some(TodoStatus::Pending));
        assert_eq!(
            TodoStatus::parse("in_progress"),
            Some(TodoStatus::InProgress)
        );
        assert_eq!(TodoStatus::parse("completed"), Some(TodoStatus::Completed));
    }

    #[test]
    fn status_parse_rejects_unknown_values() {
        assert!(TodoStatus::parse("done").is_none());
        assert!(TodoStatus::parse("in-progress").is_none());
        assert!(TodoStatus::parse("").is_none());
    }
}
