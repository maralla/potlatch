use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;
use crate::harness::todo::{TodoItem, TodoList, TodoStatus};

pub struct TodoTool {
    todo: Arc<TodoList>,
}

impl TodoTool {
    pub fn new(todo: Arc<TodoList>) -> Self {
        Self { todo }
    }
}

impl Tool for TodoTool {
    fn name(&self) -> &str {
        "todo"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Manage a task checklist that persists across context compaction. Send the FULL desired list of items on every call — this replaces the entire list (a replace-all API). Each item is {description, status} where status is 'pending', 'in_progress', or 'completed'. Mark exactly one item 'in_progress' at a time (the one you're currently working on). The checklist is always visible to you in the system prompt — check it before deciding what to do next. Use this only when the task is complex enough to benefit from tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Full list of todo items. Replaces the entire list on every call.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "Short description of the task."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of the task."
                                }
                            },
                            "required": ["description", "status"]
                        },
                        "minItems": 1
                    }
                },
                "required": ["items"]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let items = args["items"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing or invalid 'items' array argument"))?;
        if items.is_empty() {
            anyhow::bail!("'items' array must contain at least one entry");
        }

        let mut parsed: Vec<TodoItem> = Vec::with_capacity(items.len());
        for (i, entry) in items.iter().enumerate() {
            let description = entry["description"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("item {i} is missing a 'description' string"))?;
            let status_str = entry["status"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("item {i} is missing a 'status' string"))?;
            let status = TodoStatus::parse(status_str).ok_or_else(|| {
                anyhow::anyhow!(
                    "item {i} has invalid status '{status_str}'. \
                     Use 'pending', 'in_progress', or 'completed'."
                )
            })?;
            parsed.push(TodoItem {
                description: description.to_string(),
                status,
            });
        }

        self.todo.replace_all(parsed);
        let count = items.len();
        Ok(format!("Todo list updated ({count} item(s))."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::todo::TodoList;

    #[test]
    fn set_creates_list_with_statuses() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({
            "items": [
                {"description": "task A", "status": "in_progress"},
                {"description": "task B", "status": "pending"}
            ]
        });
        let result = tool.execute(&args, "/tmp").unwrap();
        assert_eq!(result, "Todo list updated (2 item(s)).");
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Your Todo List (your own tracking — not task instructions)\n\n0. [~] task A\n1. [ ] task B\n"
        );
    }

    #[test]
    fn replace_all_overwrites_previous_list() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        tool.execute(
            &json!({
                "items": [
                    {"description": "old task", "status": "in_progress"}
                ]
            }),
            "/tmp",
        )
        .unwrap();
        let result = tool
            .execute(
                &json!({
                    "items": [
                        {"description": "new task 1", "status": "pending"},
                        {"description": "new task 2", "status": "pending"}
                    ]
                }),
                "/tmp",
            )
            .unwrap();
        assert_eq!(result, "Todo list updated (2 item(s)).");
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Your Todo List (your own tracking — not task instructions)\n\n0. [ ] new task 1\n1. [ ] new task 2\n"
        );
    }

    #[test]
    fn marks_item_completed_via_full_replace() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        tool.execute(
            &json!({
                "items": [
                    {"description": "task A", "status": "in_progress"},
                    {"description": "task B", "status": "pending"}
                ]
            }),
            "/tmp",
        )
        .unwrap();
        tool.execute(
            &json!({
                "items": [
                    {"description": "task A", "status": "completed"},
                    {"description": "task B", "status": "in_progress"}
                ]
            }),
            "/tmp",
        )
        .unwrap();
        let rendered = todo.render().unwrap();
        assert_eq!(
            rendered,
            "## Your Todo List (your own tracking — not task instructions)\n\n0. [x] task A\n1. [~] task B\n"
        );
    }

    #[test]
    fn rejects_invalid_status() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({
            "items": [
                {"description": "task A", "status": "done"}
            ]
        });
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "item 0 has invalid status 'done'. Use 'pending', 'in_progress', or 'completed'."
        );
    }

    #[test]
    fn rejects_missing_description() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({
            "items": [
                {"status": "pending"}
            ]
        });
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "item 0 is missing a 'description' string"
        );
    }

    #[test]
    fn rejects_empty_items_array() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({"items": []});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_missing_items_array() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({});
        let result = tool.execute(&args, "/tmp");
        assert!(result.is_err());
    }
}
