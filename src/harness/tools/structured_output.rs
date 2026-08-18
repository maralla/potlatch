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
        let mut args = args.clone();
        let decoded = decode_stringified_containers(&self.schema["parameters"], &mut args);
        if decoded > 0 {
            tracing::debug!(
                tool = %self.name,
                decoded,
                "decoded JSON-string container arguments using the registered schema"
            );
        }
        if !args.is_object() {
            anyhow::bail!("{} arguments must be a JSON object", self.name);
        }
        // Store in the side-channel cell (last call wins).
        *self.cell.lock().unwrap() = Some(args);
        Ok(format!("{} recorded.", self.name))
    }
}

/// Some tool-call dialects serialize nested arrays and objects as JSON strings
/// even when the registered schema declares native containers. Decode only
/// valid JSON whose resulting container matches the expected schema type.
/// Ordinary string fields and malformed/wrong-shaped JSON remain untouched
/// for the orchestrator's normal schema validation to reject.
fn decode_stringified_containers(schema: &Value, value: &mut Value) -> usize {
    let mut decoded = decode_container(schema, value) as usize;

    if let Some(branches) = schema.get("oneOf").and_then(Value::as_array) {
        let matching: Vec<_> = branches
            .iter()
            .filter(|branch| branch_matches(branch, value))
            .collect();
        for branch in matching {
            decoded += decode_stringified_containers(branch, value);
        }
    }

    match schema.get("type").and_then(Value::as_str) {
        Some("object") => {
            let Some(object) = value.as_object_mut() else {
                return decoded;
            };
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
                return decoded;
            };
            for (name, field_schema) in properties {
                if let Some(field) = object.get_mut(name) {
                    decoded += decode_stringified_containers(field_schema, field);
                }
            }
        }
        Some("array") => {
            let Some(items_schema) = schema.get("items") else {
                return decoded;
            };
            let Some(items) = value.as_array_mut() else {
                return decoded;
            };
            for item in items {
                decoded += decode_stringified_containers(items_schema, item);
            }
        }
        _ => {}
    }
    decoded
}

fn decode_container(schema: &Value, value: &mut Value) -> bool {
    let Some(expected) = schema.get("type").and_then(Value::as_str) else {
        return false;
    };
    if !matches!(expected, "array" | "object") {
        return false;
    }
    let Some(raw) = value.as_str() else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return false;
    };
    let matches = matches!(
        (expected, &parsed),
        ("array", Value::Array(_)) | ("object", Value::Object(_))
    );
    if matches {
        *value = parsed;
    }
    matches
}

/// Select a tagged-union branch by its `const` properties. Branches without
/// constants remain applicable, matching ordinary JSON Schema semantics.
fn branch_matches(branch: &Value, value: &Value) -> bool {
    let Some(properties) = branch.get("properties").and_then(Value::as_object) else {
        return true;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    properties.iter().all(|(name, property)| {
        property
            .get("const")
            .is_none_or(|expected| object.get(name) == Some(expected))
    })
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
    fn decodes_stringified_array_for_the_selected_union_branch() {
        let parameters = json!({
            "type": "object",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "const": "split"},
                        "sub_issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "const": "guide_worker"},
                        "instructions": {"type": "string"}
                    }
                }
            ]
        });
        let (tool, cell) = StructuredOutputTool::new("plan".into(), "Plan the issue.", parameters);

        tool.execute(
            &json!({
                "decision": "split",
                "sub_issues": "[{\"title\":\"First\"},{\"title\":\"Second\"}]"
            }),
            "/tmp",
        )
        .unwrap();

        assert_eq!(
            cell.lock().unwrap().clone(),
            Some(json!({
                "decision": "split",
                "sub_issues": [{"title": "First"}, {"title": "Second"}]
            }))
        );
    }

    #[test]
    fn recursively_decodes_nested_stringified_containers() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        });
        let (tool, cell) =
            StructuredOutputTool::new("configure".into(), "Configure it.", parameters);

        tool.execute(
            &json!({"config": "{\"labels\":\"[\\\"one\\\",\\\"two\\\"]\"}"}),
            "/tmp",
        )
        .unwrap();

        assert_eq!(
            cell.lock().unwrap().clone(),
            Some(json!({"config": {"labels": ["one", "two"]}}))
        );
    }

    #[test]
    fn leaves_malformed_and_wrong_container_strings_for_validation() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "metadata": {"type": "object"}
            }
        });
        let (tool, cell) = StructuredOutputTool::new("report".into(), "Report it.", parameters);

        tool.execute(&json!({"items": "not json", "metadata": "[1,2]"}), "/tmp")
            .unwrap();

        assert_eq!(
            cell.lock().unwrap().clone(),
            Some(json!({"items": "not json", "metadata": "[1,2]"}))
        );
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
