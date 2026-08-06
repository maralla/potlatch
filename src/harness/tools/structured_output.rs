//! Generic structured-output tool: a side-channel cell tool whose name,
//! description, and parameter schema are defined by the caller (the potlatch
//! orchestrator) and passed via `session/new` params. The harness creates one
//! `StructuredOutputTool` per definition, captures the model's call in a
//! per-tool cell, and returns the captured JSON in the `session/prompt`
//! response. This is the same pattern as the `plan` tool, but generalized —
//! any agent can define structured-output tools without harness changes.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

/// Shared side-channel cell holding the JSON the model emitted via a
/// structured-output tool call.
pub type StructuredOutputCell = Arc<Mutex<Option<Value>>>;

pub struct StructuredOutputTool {
    name: String,
    schema: Value,
    cell: StructuredOutputCell,
}

impl StructuredOutputTool {
    /// Create a new structured-output tool and return both the tool and its
    /// cell. The caller stores the cell to read the captured JSON after the
    /// agent loop completes.
    pub fn new(name: String, description: &str, parameters: Value) -> (Self, StructuredOutputCell) {
        let cell: StructuredOutputCell = Arc::new(Mutex::new(None));
        let schema = json!({
            "description": description,
            "parameters": parameters,
        });
        (
            Self {
                name,
                schema,
                cell: Arc::clone(&cell),
            },
            cell,
        )
    }
}

impl Tool for StructuredOutputTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        if !args.is_object() {
            anyhow::bail!("{} arguments must be a JSON object", self.name);
        }
        // Store in the side-channel cell (last call wins).
        *self.cell.lock().unwrap() = Some(args.clone());
        Ok(format!("{} recorded.", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_args_in_cell() {
        let (tool, cell) = StructuredOutputTool::new(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object", "properties": {"mr_title": {"type": "string"}}}),
        );
        let args = json!({"mr_title": "Add tests"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert_eq!(result, "handoff recorded.");
        let captured = cell.lock().unwrap().clone();
        assert_eq!(captured, Some(json!({"mr_title": "Add tests"})));
    }

    #[test]
    fn second_call_overwrites_first() {
        let (tool, cell) = StructuredOutputTool::new(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
        );
        tool.execute(&json!({"mr_title": "A"}), "/tmp").unwrap();
        tool.execute(&json!({"mr_title": "B"}), "/tmp").unwrap();
        let captured = cell.lock().unwrap().clone();
        assert_eq!(captured, Some(json!({"mr_title": "B"})));
    }

    #[test]
    fn rejects_non_object_args() {
        let (tool, _cell) = StructuredOutputTool::new(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
        );
        let result = tool.execute(&json!("not an object"), "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn cell_starts_empty() {
        let (_tool, cell) = StructuredOutputTool::new(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
        );
        assert!(cell.lock().unwrap().is_none());
    }

    #[test]
    fn schema_carries_name_and_description() {
        let (tool, _cell) = StructuredOutputTool::new(
            "handoff".into(),
            "Emit your output as structured JSON.",
            json!({
                "type": "object",
                "properties": {
                    "mr_title": {"type": "string"}
                }
            }),
        );
        assert_eq!(tool.name(), "handoff");
        assert_eq!(
            tool.schema()["description"].as_str(),
            Some("Emit your output as structured JSON.")
        );
        assert!(tool.schema()["parameters"]["properties"]["mr_title"].is_object());
    }
}
