//! Task-tracking todo list that survives context compaction and collapse.
//!
//! The todo list lives outside the context window in an `Arc<Mutex>`. The agent
//! loop injects it as a system message before every API call (after compaction),
//! so it's always visible to the model regardless of how much context was
//! evicted. The model manages it via the `todo_update` tool.

use std::sync::{Arc, Mutex};

/// Status of a todo item.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

/// A single todo item.
#[derive(Debug, Clone)]
pub struct TodoItem {
    pub text: String,
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

    /// Set the full list of items (replaces all). All items start as Pending.
    pub fn set_items(&self, texts: Vec<String>) {
        let mut items = self.items.lock().unwrap();
        items.clear();
        for text in texts {
            items.push(TodoItem {
                text,
                status: TodoStatus::Pending,
            });
        }
    }

    /// Mark an item as in-progress by index (0-based).
    pub fn start(&self, index: usize) -> Result<(), String> {
        let mut items = self.items.lock().unwrap();
        if index >= items.len() {
            return Err(format!(
                "index {index} out of range ({} items)",
                items.len()
            ));
        }
        items[index].status = TodoStatus::InProgress;
        Ok(())
    }

    /// Mark an item as completed by index (0-based).
    pub fn complete(&self, index: usize) -> Result<(), String> {
        let mut items = self.items.lock().unwrap();
        if index >= items.len() {
            return Err(format!(
                "index {index} out of range ({} items)",
                items.len()
            ));
        }
        items[index].status = TodoStatus::Completed;
        Ok(())
    }

    /// Render the todo list as a formatted string for injection into the system prompt.
    /// Returns None if the list is empty.
    pub fn render(&self) -> Option<String> {
        let items = self.items.lock().unwrap();
        if items.is_empty() {
            return None;
        }
        let mut out = String::from("## Task Checklist\n\n");
        for (i, item) in items.iter().enumerate() {
            out.push_str(&format!("{}. {} {}\n", i, item.status.label(), item.text));
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

    #[test]
    fn set_items_creates_pending() {
        let todo = TodoList::new();
        todo.set_items(vec!["read files".into(), "edit code".into()]);
        let rendered = todo.render().unwrap();
        assert!(rendered.contains("[ ] read files"));
        assert!(rendered.contains("[ ] edit code"));
    }

    #[test]
    fn start_and_complete_update_status() {
        let todo = TodoList::new();
        todo.set_items(vec!["task A".into(), "task B".into()]);

        todo.start(0).unwrap();
        let rendered = todo.render().unwrap();
        assert!(rendered.contains("[~] task A"));
        assert!(rendered.contains("[ ] task B"));

        todo.complete(0).unwrap();
        let rendered = todo.render().unwrap();
        assert!(rendered.contains("[x] task A"));
        assert!(rendered.contains("[ ] task B"));
    }

    #[test]
    fn out_of_range_returns_error() {
        let todo = TodoList::new();
        todo.set_items(vec!["only task".into()]);
        assert!(todo.start(5).is_err());
        assert!(todo.complete(99).is_err());
    }

    #[test]
    fn render_empty_returns_none() {
        let todo = TodoList::new();
        assert!(todo.render().is_none());
    }

    #[test]
    fn set_items_replaces_previous() {
        let todo = TodoList::new();
        todo.set_items(vec!["old task".into()]);
        todo.start(0).unwrap();
        todo.set_items(vec!["new task 1".into(), "new task 2".into()]);
        let rendered = todo.render().unwrap();
        assert!(!rendered.contains("old task"));
        assert!(rendered.contains("new task 1"));
        assert!(rendered.contains("[ ] new task 1"));
    }
}
