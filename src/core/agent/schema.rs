//! Backend-agnostic structured-output contract for agent/model completions.
//!
//! Agents declare what they want back from the model as a typed Rust type
//! implementing [`StructuredOutput`], built from the declarative
//! [`SchemaField`]/[`ObjectSchema`] model below. Core never lets an agent
//! construct raw backend JSON tool definitions — [`StructuredOutputTool`] is
//! the neutral contract; a model backend adapter (see
//! `core::model::engine`) converts it to whatever wire format that backend
//! expects (e.g. ACP/harness `structured_output_tools`).

use serde::de::DeserializeOwned;

/// One property in a declarative JSON-schema-shaped object description.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaField {
    String {
        description: String,
        enum_values: Vec<String>,
    },
    Integer {
        description: String,
        enum_values: Vec<i64>,
    },
    Boolean {
        description: String,
    },
    Array {
        description: String,
        items: Box<SchemaField>,
    },
    Object(ObjectSchema),
}

impl SchemaField {
    pub fn string(description: impl Into<String>) -> Self {
        SchemaField::String {
            description: description.into(),
            enum_values: Vec::new(),
        }
    }

    pub fn string_enum(description: impl Into<String>, values: &[&str]) -> Self {
        SchemaField::String {
            description: description.into(),
            enum_values: values.iter().map(|v| v.to_string()).collect(),
        }
    }

    pub fn integer(description: impl Into<String>) -> Self {
        SchemaField::Integer {
            description: description.into(),
            enum_values: Vec::new(),
        }
    }

    pub fn integer_enum(description: impl Into<String>, values: &[i64]) -> Self {
        SchemaField::Integer {
            description: description.into(),
            enum_values: values.to_vec(),
        }
    }

    pub fn boolean(description: impl Into<String>) -> Self {
        SchemaField::Boolean {
            description: description.into(),
        }
    }

    pub fn array(description: impl Into<String>, items: SchemaField) -> Self {
        SchemaField::Array {
            description: description.into(),
            items: Box::new(items),
        }
    }

    pub fn object(schema: ObjectSchema) -> Self {
        SchemaField::Object(schema)
    }
}

/// A declarative `{"type": "object", "properties": {...}, "required": [...]}` shape.
/// Property order is preserved for readability; backends are free to reorder
/// when converting to wire JSON.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectSchema {
    pub properties: Vec<(String, SchemaField)>,
    pub required: Vec<String>,
}

impl ObjectSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn property(mut self, name: impl Into<String>, field: SchemaField) -> Self {
        self.properties.push((name.into(), field));
        self
    }

    pub fn required(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }
}

/// Neutral, backend-agnostic structured-output tool contract. A model
/// backend adapter converts this to its own wire format (e.g. ACP's
/// `structured_output_tools` JSON) — core and agent code never build that
/// wire JSON directly.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutputTool {
    pub name: String,
    pub description: String,
    pub parameters: ObjectSchema,
}

/// Implemented by each role's typed completion output. Ties together the
/// tool identity/schema the model is asked to call with the Rust type its
/// JSON arguments deserialize into.
pub trait StructuredOutput: DeserializeOwned {
    /// Name of the structured-output tool the model must call.
    fn tool_name() -> &'static str;

    /// Human-readable description passed to the model backend.
    fn tool_description() -> &'static str;

    /// Declarative schema for the tool's arguments.
    fn schema() -> ObjectSchema;

    /// Build the neutral tool contract from `tool_name`/`tool_description`/`schema`.
    fn tool_definition() -> StructuredOutputTool {
        StructuredOutputTool {
            name: Self::tool_name().to_string(),
            description: Self::tool_description().to_string(),
            parameters: Self::schema(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_schema_builder_preserves_property_order_and_required() {
        let schema = ObjectSchema::new()
            .property(
                "decision",
                SchemaField::string_enum("pick one", &["a", "b"]),
            )
            .property("count", SchemaField::integer("how many"))
            .required("decision");

        assert_eq!(schema.properties.len(), 2);
        assert_eq!(schema.properties[0].0, "decision");
        assert_eq!(schema.properties[1].0, "count");
        assert_eq!(schema.required, vec!["decision".to_string()]);
    }

    #[test]
    fn schema_field_constructors_build_expected_variants() {
        assert_eq!(
            SchemaField::string("d"),
            SchemaField::String {
                description: "d".into(),
                enum_values: vec![]
            }
        );
        assert_eq!(
            SchemaField::string_enum("d", &["x", "y"]),
            SchemaField::String {
                description: "d".into(),
                enum_values: vec!["x".into(), "y".into()]
            }
        );
        assert_eq!(
            SchemaField::integer("d"),
            SchemaField::Integer {
                description: "d".into(),
                enum_values: vec![]
            }
        );
        assert_eq!(
            SchemaField::integer_enum("d", &[1, 2, 3]),
            SchemaField::Integer {
                description: "d".into(),
                enum_values: vec![1, 2, 3]
            }
        );
        assert_eq!(
            SchemaField::boolean("d"),
            SchemaField::Boolean {
                description: "d".into()
            }
        );
        match SchemaField::array("d", SchemaField::string("item")) {
            SchemaField::Array { description, items } => {
                assert_eq!(description, "d");
                assert_eq!(*items, SchemaField::string("item"));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[derive(Debug, serde::Deserialize)]
    struct SampleOutput {
        decision: String,
    }

    impl StructuredOutput for SampleOutput {
        fn tool_name() -> &'static str {
            "sample"
        }

        fn tool_description() -> &'static str {
            "sample tool"
        }

        fn schema() -> ObjectSchema {
            ObjectSchema::new()
                .property("decision", SchemaField::string_enum("d", &["a", "b"]))
                .required("decision")
        }
    }

    #[test]
    fn tool_definition_combines_name_description_and_schema() {
        let tool = SampleOutput::tool_definition();
        let output: SampleOutput = serde_json::from_value(serde_json::json!({
            "decision": "a"
        }))
        .unwrap();
        assert_eq!(tool.name, "sample");
        assert_eq!(tool.description, "sample tool");
        assert_eq!(tool.parameters, SampleOutput::schema());
        assert_eq!(output.decision, "a");
    }
}
