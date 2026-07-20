//! Todo list management tool. Lets the model track task progress with a list
//! that survives context compaction.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};
use tracing::info;

use super::Tool;
use crate::harness::todo::TodoList;

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
            "description": "Manage a task checklist that persists across context compaction. Use 'set' to create the full list at the start, 'start' to mark an item in-progress, and 'complete' to mark it done. The checklist is always visible to you in the system prompt — check it before deciding what to do next.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "start", "complete"],
                        "description": "Action: 'set' replaces the entire list (all items start pending), 'start' marks an item in-progress, 'complete' marks an item done."
                    },
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task descriptions. Required for 'set' action. Ignored for 'start'/'complete'."
                    },
                    "index": {
                        "type": "integer",
                        "description": "0-based index of the item to start/complete. Required for 'start'/'complete' actions."
                    }
                },
                "required": ["action"]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let action = args["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'action' argument"))?;

        match action {
            "set" => {
                let items = args["items"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("'set' action requires 'items' array"))?;
                let texts: Vec<String> = items
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if texts.is_empty() {
                    return Ok("Error: 'items' array must not be empty".into());
                }
                self.todo.set_items(texts);
                let rendered = self.todo.render().unwrap_or_default();
                info!("harness: todo set\n{rendered}");
                let count = rendered
                    .lines()
                    .filter(|l| l.starts_with(|c: char| c.is_numeric()))
                    .count();
                Ok(format!("Todo list set with {count} item(s).\n{rendered}"))
            }
            "start" => {
                let index = args["index"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("'start' action requires 'index'"))?
                    as usize;
                match self.todo.start(index) {
                    Ok(()) => {
                        let rendered = self.todo.render().unwrap_or_default();
                        info!("harness: todo start item {index}\n{rendered}");
                        Ok(format!("Started item {index}.\n{rendered}"))
                    }
                    Err(e) => Ok(format!("Error: {e}")),
                }
            }
            "complete" => {
                let index = args["index"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("'complete' action requires 'index'"))?
                    as usize;
                match self.todo.complete(index) {
                    Ok(()) => {
                        let rendered = self.todo.render().unwrap_or_default();
                        info!("harness: todo complete item {index}\n{rendered}");
                        Ok(format!("Completed item {index}.\n{rendered}"))
                    }
                    Err(e) => Ok(format!("Error: {e}")),
                }
            }
            other => Ok(format!(
                "Error: unknown action '{other}'. Use 'set', 'start', or 'complete'."
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::todo::TodoList;

    #[test]
    fn set_creates_list() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        let args = json!({"action": "set", "items": ["task A", "task B"]});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert!(result.contains("2 item(s)"));
        let rendered = todo.render().unwrap();
        assert!(rendered.contains("[ ] task A"));
        assert!(rendered.contains("[ ] task B"));
    }

    #[test]
    fn start_marks_in_progress() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        tool.execute(
            &json!({"action": "set", "items": ["task A", "task B"]}),
            "/tmp",
        )
        .unwrap();
        let result = tool
            .execute(&json!({"action": "start", "index": 0}), "/tmp")
            .unwrap();
        assert!(result.contains("[~] task A"));
    }

    #[test]
    fn complete_marks_done() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        tool.execute(&json!({"action": "set", "items": ["task A"]}), "/tmp")
            .unwrap();
        let result = tool
            .execute(&json!({"action": "complete", "index": 0}), "/tmp")
            .unwrap();
        assert!(result.contains("[x] task A"));
    }

    #[test]
    fn out_of_range_error() {
        let todo = Arc::new(TodoList::new());
        let tool = TodoTool::new(Arc::clone(&todo));
        tool.execute(&json!({"action": "set", "items": ["only"]}), "/tmp")
            .unwrap();
        let result = tool
            .execute(&json!({"action": "complete", "index": 5}), "/tmp")
            .unwrap();
        assert!(result.contains("Error"));
    }
}
