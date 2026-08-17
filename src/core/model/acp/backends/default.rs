//! Portable ACP backend for vendors without structured-output tools.
//!
//! The neutral contract is rendered into the task prompt as a JSON marker
//! envelope and parsed back out of the final response.

use serde_json::{Map, Value, json};

use super::StructuredOutputBackend;

const STRUCTURED_OUTPUT_BEGIN: &str = "BREEZE_STRUCTURED_OUTPUT_BEGIN";
const STRUCTURED_OUTPUT_END: &str = "BREEZE_STRUCTURED_OUTPUT_END";

pub(super) struct DefaultBackend;

impl StructuredOutputBackend for DefaultBackend {
    fn prepare_prompt(&self, prompt: &str, tools: &[Value]) -> String {
        if tools.is_empty() {
            return prompt.to_string();
        }

        let examples = Value::Object(
            tools
                .iter()
                .filter_map(|tool| {
                    let name = tool.get("name")?.as_str()?.to_string();
                    let arguments = tool
                        .get("parameters")
                        .map(schema_example)
                        .unwrap_or_else(|| json!({}));
                    Some((name, arguments))
                })
                .collect(),
        );
        let contracts = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());
        let example = serde_json::to_string_pretty(&examples).unwrap_or_else(|_| "{}".to_string());
        format!(
            "{prompt}\n\n\
             ## Required structured result\n\n\
             Return exactly one JSON object between these marker lines. The object maps each \
             requested output name to its arguments. Do not put prose inside the markers.\n\n\
             {STRUCTURED_OUTPUT_BEGIN}\n{example}\n{STRUCTURED_OUTPUT_END}\n\n\
             The arguments must satisfy these contracts:\n{contracts}"
        )
    }

    fn session_tools(&self, _tools: &[Value]) -> Option<Vec<Value>> {
        None
    }

    fn extract_outputs(&self, response: &str) -> Option<Value> {
        let start = response.rfind(STRUCTURED_OUTPUT_BEGIN)? + STRUCTURED_OUTPUT_BEGIN.len();
        let tail = &response[start..];
        let end = tail.find(STRUCTURED_OUTPUT_END)?;
        let value: Value = serde_json::from_str(tail[..end].trim()).ok()?;
        value
            .as_object()
            .is_some_and(|outputs| !outputs.is_empty())
            .then_some(value)
    }
}

fn schema_example(schema: &Value) -> Value {
    if let Some(branch) = schema
        .get("oneOf")
        .and_then(Value::as_array)
        .and_then(|branches| branches.first())
    {
        return schema_example(branch);
    }

    match schema.get("type").and_then(Value::as_str) {
        Some("object") => {
            let required = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str);
            let properties = schema.get("properties").and_then(Value::as_object);
            let mut example = Map::new();
            for name in required {
                let value = properties
                    .and_then(|properties| properties.get(name))
                    .map(schema_example)
                    .unwrap_or(Value::Null);
                example.insert(name.to_string(), value);
            }
            Value::Object(example)
        }
        Some("array") => Value::Array(Vec::new()),
        Some("integer") => schema
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .cloned()
            .unwrap_or_else(|| json!(0)),
        Some("boolean") => Value::Bool(false),
        _ => schema
            .get("const")
            .cloned()
            .or_else(|| {
                schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .and_then(|values| values.first())
                    .cloned()
            })
            .unwrap_or_else(|| Value::String(String::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> Value {
        json!({
            "name": "plan",
            "description": "Choose a plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["split", "done"]},
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["decision", "items"],
                "additionalProperties": false
            }
        })
    }

    #[test]
    fn renders_marker_prompt_from_tool_contract() {
        let prompt = DefaultBackend.prepare_prompt("Triage.", &[tool()]);

        assert!(prompt.starts_with("Triage."));
        assert!(prompt.contains(STRUCTURED_OUTPUT_BEGIN));
        assert!(prompt.contains(r#""decision": "split""#));
        assert!(prompt.contains(r#""additionalProperties": false"#));
    }

    #[test]
    fn extracts_last_complete_marker_object() {
        let response = format!(
            "progress\n{STRUCTURED_OUTPUT_BEGIN}\n{{\"plan\":{{\"decision\":\"split\",\"items\":[]}}}}\n\
             {STRUCTURED_OUTPUT_END}"
        );

        assert_eq!(
            DefaultBackend.extract_outputs(&response),
            Some(json!({"plan": {"decision": "split", "items": []}}))
        );
    }

    #[test]
    fn rejects_missing_or_malformed_marker_output() {
        assert_eq!(DefaultBackend.extract_outputs("plain text"), None);
        assert_eq!(
            DefaultBackend.extract_outputs(&format!(
                "{STRUCTURED_OUTPUT_BEGIN}\nnot json\n{STRUCTURED_OUTPUT_END}"
            )),
            None
        );
    }
}
