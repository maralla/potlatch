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

/// Appended to a terminal structured output's result. A terminal call's
/// record IS the run's deliverable — the neutral ack gave a reviewer no
/// completion signal and it re-emitted the identical review three times.
pub(crate) const COMPLETION_NOTICE: &str = " This submission is the task's final deliverable — do not call any more tools; end your turn now.";

pub struct StructuredOutputTool {
    name: String,
    schema: Value,
    cell: StructuredOutputCell,
    terminal: bool,
}

impl StructuredOutputTool {
    /// Create a new structured-output tool and return both the tool and its
    /// cell. The caller stores the cell to read the captured JSON after the
    /// agent loop completes. `terminal` marks a contract whose successful
    /// call is the run's final deliverable — the result then tells the model
    /// the task is complete so it ends its turn.
    pub fn with_terminal(
        name: String,
        description: &str,
        parameters: Value,
        terminal: bool,
    ) -> (Self, StructuredOutputCell) {
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
                terminal,
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
        // An argumentless call is a transport failure, not a schema violation:
        // the call reached the tool but its arguments were never transmitted.
        // Seen in the wild as a model emitting a named tool call with empty
        // `function.arguments` nine times in a row — every repair retry also
        // arrived empty, and the orchestrator's schema validation then
        // reported a missing discriminator, sending the model chasing JSON
        // shape problems that were never the issue. Rejecting here surfaces
        // the real failure in the tool result channel, where the model can
        // see it on the very next turn.
        if args.is_null() {
            anyhow::bail!(
                "{} tool call arrived with NO arguments (null) — the call reached the tool \
                 but no arguments were transmitted. This is a tool-call transport problem, \
                 not a JSON shape problem: re-emit the call with the full arguments object.",
                self.name
            );
        }
        if !args.is_object() {
            anyhow::bail!(
                "{} arguments must be a JSON object; received: {}",
                self.name,
                truncate_for_error(&args)
            );
        }
        if args.as_object().is_some_and(|object| object.is_empty()) {
            anyhow::bail!(
                "{} tool call arrived with EMPTY arguments ({{}}) — the call reached the \
                 tool but no arguments were transmitted. This is a tool-call transport \
                 problem, not a JSON shape problem: re-emit the call with the full \
                 arguments object.",
                self.name
            );
        }
        // Store in the side-channel cell (last call wins).
        *self.cell.lock().unwrap() = Some(args);
        if self.terminal {
            Ok(format!("{} recorded.{COMPLETION_NOTICE}", self.name))
        } else {
            Ok(format!("{} recorded.", self.name))
        }
    }
}

/// Render a received value for an error message, bounded so a huge or
/// deeply-nested argument can't flood the tool result.
fn truncate_for_error(value: &Value) -> String {
    let text = value.to_string();
    const MAX: usize = 400;
    if text.len() <= MAX {
        return text;
    }
    format!("{}… ({} bytes total)", &text[..MAX], text.len())
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
    fn terminal_result_tells_the_model_to_stop() {
        let (tool, _cell) = StructuredOutputTool::with_terminal(
            "review".into(),
            "The review decision.",
            json!({"type": "object", "properties": {}}),
            true,
        );
        let result = tool
            .execute(&json!({"decision": "approve"}), "/tmp")
            .unwrap();
        // The completion notice IS the deliverable under test: without it
        // the model re-emits the decision (observed: three identical
        // reviews in a row).
        assert!(result.starts_with("review recorded."));
        assert!(result.contains(COMPLETION_NOTICE));
        assert!(result.contains("end your turn"));
    }

    #[test]
    fn stores_args_in_cell() {
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object", "properties": {"mr_title": {"type": "string"}}}),
            false,
        );
        let args = json!({"mr_title": "Add tests"});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert_eq!(result, "handoff recorded.");
        let captured = cell.lock().unwrap().clone();
        assert_eq!(captured, Some(json!({"mr_title": "Add tests"})));
    }

    #[test]
    fn rejects_null_arguments_as_a_transport_failure() {
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "review".into(),
            "Emit the review decision.",
            json!({"type": "object"}),
            false,
        );
        let result = tool.execute(&Value::Null, "/tmp");
        let error = result.unwrap_err().to_string();
        assert!(error.contains("arrived with NO arguments"), "{error}");
        assert!(error.contains("transport"), "{error}");
        // Nothing was recorded: a null capture must not shadow a prior value.
        assert!(cell.lock().unwrap().is_none());
    }

    #[test]
    fn rejects_empty_object_arguments_as_a_transport_failure() {
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "review".into(),
            "Emit the review decision.",
            json!({"type": "object"}),
            false,
        );
        let result = tool.execute(&json!({}), "/tmp");
        let error = result.unwrap_err().to_string();
        assert!(error.contains("EMPTY arguments"), "{error}");
        assert!(error.contains("transport"), "{error}");
        // Nothing was recorded: the empty capture is rejected, not stored.
        assert!(cell.lock().unwrap().is_none());
    }

    #[test]
    fn rejects_non_object_arguments_with_the_received_value() {
        let (tool, _cell) = StructuredOutputTool::with_terminal(
            "review".into(),
            "Emit the review decision.",
            json!({"type": "object"}),
            false,
        );
        let result = tool.execute(&json!([1, 2, 3]), "/tmp");
        let error = result.unwrap_err().to_string();
        assert!(error.contains("must be a JSON object"), "{error}");
        assert!(error.contains("received"), "{error}");
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
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "plan".into(),
            "Plan the issue.",
            parameters,
            false,
        );

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
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "configure".into(),
            "Configure it.",
            parameters,
            false,
        );

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
        let (tool, cell) =
            StructuredOutputTool::with_terminal("report".into(), "Report it.", parameters, false);

        tool.execute(&json!({"items": "not json", "metadata": "[1,2]"}), "/tmp")
            .unwrap();

        assert_eq!(
            cell.lock().unwrap().clone(),
            Some(json!({"items": "not json", "metadata": "[1,2]"}))
        );
    }

    #[test]
    fn second_call_overwrites_first() {
        let (tool, cell) = StructuredOutputTool::with_terminal(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
            false,
        );
        tool.execute(&json!({"mr_title": "A"}), "/tmp").unwrap();
        tool.execute(&json!({"mr_title": "B"}), "/tmp").unwrap();
        let captured = cell.lock().unwrap().clone();
        assert_eq!(captured, Some(json!({"mr_title": "B"})));
    }

    #[test]
    fn rejects_non_object_args() {
        let (tool, _cell) = StructuredOutputTool::with_terminal(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
            false,
        );
        let result = tool.execute(&json!("not an object"), "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn cell_starts_empty() {
        let (_tool, cell) = StructuredOutputTool::with_terminal(
            "handoff".into(),
            "Emit structured output.",
            json!({"type": "object"}),
            false,
        );
        assert!(cell.lock().unwrap().is_none());
    }

    #[test]
    fn schema_carries_name_and_description() {
        let (tool, _cell) = StructuredOutputTool::with_terminal(
            "handoff".into(),
            "Emit your output as structured JSON.",
            json!({
                "type": "object",
                "properties": {
                    "mr_title": {"type": "string"}
                }
            }),
            false,
        );
        assert_eq!(tool.name(), "handoff");
        assert_eq!(
            tool.schema()["description"].as_str(),
            Some("Emit your output as structured JSON.")
        );
        assert!(tool.schema()["parameters"]["properties"]["mr_title"].is_object());
    }
}
