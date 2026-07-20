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
    pub fn render(&self) -> Option<String> {
        let items = self.items.lock().unwrap();
        if items.is_empty() {
            return None;
        }
        let mut out = String::from("## Task Checklist\n\n");
        for (i, item) in items.iter().enumerate() {
            out.push_str(&format!(
                "{}. {} {}\n",
                i,
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
        assert!(rendered.contains("[ ] read files"));
        assert!(rendered.contains("[ ] edit code"));
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
        assert!(rendered.contains("[~] task A"));
        assert!(rendered.contains("[x] task B"));
        assert!(rendered.contains("[ ] task C"));
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
        assert!(!rendered.contains("old task"));
        assert!(rendered.contains("new task 1"));
        assert!(rendered.contains("[ ] new task 1"));
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
