//! Backend-agnostic structured-output contract for agent/model completions.
//!
//! Agents declare what they want back from the model as a typed Rust type
//! implementing [`StructuredOutput`], described by the recursive, declarative
//! [`Schema`] below. Core never lets an agent construct raw backend JSON tool
//! definitions — [`StructuredOutputTool`] is the neutral contract; a model
//! backend adapter (see `core::model::engine`) converts it to whatever wire
//! format that backend expects (e.g. ACP/harness `structured_output_tools`).
//!
//! The schema is also executable: [`validate`] checks a captured JSON value
//! against it deterministically and reports the first violation with a JSON
//! path (`$.sub_issues[0].title`), which is what the structured-output repair
//! loop feeds back to the model.

use std::fmt;

use serde::de::DeserializeOwned;
use serde_json::Value;

/// A recursive, backend-neutral description of one JSON value.
///
/// Objects are **closed**: a property that is not declared is a violation
/// (and the adapter emits `additionalProperties: false`). This is what keeps
/// a role's schema and its Serde type from drifting apart.
#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
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
        items: Box<Schema>,
    },
    Object(ObjectSchema),
    /// A tagged union: exactly one variant applies, selected by the value of
    /// the discriminator property.
    OneOf(OneOfSchema),
}

impl Schema {
    pub fn string(description: impl Into<String>) -> Self {
        Schema::String {
            description: description.into(),
            enum_values: Vec::new(),
        }
    }

    pub fn string_enum(description: impl Into<String>, values: &[&str]) -> Self {
        Schema::String {
            description: description.into(),
            enum_values: values.iter().map(|v| v.to_string()).collect(),
        }
    }

    pub fn integer(description: impl Into<String>) -> Self {
        Schema::Integer {
            description: description.into(),
            enum_values: Vec::new(),
        }
    }

    pub fn integer_enum(description: impl Into<String>, values: &[i64]) -> Self {
        Schema::Integer {
            description: description.into(),
            enum_values: values.to_vec(),
        }
    }

    pub fn boolean(description: impl Into<String>) -> Self {
        Schema::Boolean {
            description: description.into(),
        }
    }

    pub fn array(description: impl Into<String>, items: Schema) -> Self {
        Schema::Array {
            description: description.into(),
            items: Box::new(items),
        }
    }

    pub fn object(schema: ObjectSchema) -> Self {
        Schema::Object(schema)
    }

    pub fn one_of(schema: OneOfSchema) -> Self {
        Schema::OneOf(schema)
    }

    pub fn description(&self) -> &str {
        match self {
            Schema::String { description, .. }
            | Schema::Integer { description, .. }
            | Schema::Boolean { description }
            | Schema::Array { description, .. } => description,
            Schema::Object(object) => &object.description,
            Schema::OneOf(one_of) => &one_of.description,
        }
    }
}

/// A closed `{"type": "object"}` shape. Property order is preserved for
/// readability and for deterministic validation order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectSchema {
    pub description: String,
    pub properties: Vec<(String, Schema)>,
    pub required: Vec<String>,
}

impl ObjectSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Declare an optional property.
    pub fn property(mut self, name: impl Into<String>, field: Schema) -> Self {
        self.properties.push((name.into(), field));
        self
    }

    /// Declare a property that must be present. Required-ness is expressed
    /// together with the property itself so a schema can never require a
    /// property it does not declare.
    pub fn required_property(mut self, name: impl Into<String>, field: Schema) -> Self {
        let name = name.into();
        self.required.push(name.clone());
        self.properties.push((name, field));
        self
    }

    pub fn property_names(&self) -> Vec<&str> {
        self.properties.iter().map(|(n, _)| n.as_str()).collect()
    }
}

/// A tagged union of object shapes, selected by a string discriminator
/// property that every variant carries.
#[derive(Debug, Clone, PartialEq)]
pub struct OneOfSchema {
    pub description: String,
    pub discriminator: String,
    pub variants: Vec<SchemaVariant>,
}

impl OneOfSchema {
    pub fn new(discriminator: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            discriminator: discriminator.into(),
            variants: Vec::new(),
        }
    }

    pub fn variant(
        mut self,
        tag: impl Into<String>,
        description: impl Into<String>,
        fields: ObjectSchema,
    ) -> Self {
        self.variants.push(SchemaVariant {
            tag: tag.into(),
            description: description.into(),
            fields,
        });
        self
    }

    pub fn tags(&self) -> Vec<&str> {
        self.variants.iter().map(|v| v.tag.as_str()).collect()
    }

    pub fn variant_for(&self, tag: &str) -> Option<&SchemaVariant> {
        self.variants.iter().find(|v| v.tag == tag)
    }
}

/// One branch of a [`OneOfSchema`]: the discriminator value that selects it
/// plus the properties that branch allows.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaVariant {
    pub tag: String,
    pub description: String,
    pub fields: ObjectSchema,
}

/// Neutral, backend-agnostic structured-output tool contract. A model
/// backend adapter converts this to its own wire format (e.g. ACP's
/// `structured_output_tools` JSON) — core and agent code never build that
/// wire JSON directly.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutputTool {
    pub name: String,
    pub description: String,
    pub parameters: Schema,
}

/// Declare a typed structured-output contract without repeating the schema
/// builder plumbing. Requiredness stays explicit and the generated value is
/// the same backend-neutral [`Schema`] used by validation and adapters.
///
/// `fields(expression)` is an escape hatch for shared [`ObjectSchema`]
/// builders. It keeps reusable field groups composable without teaching the
/// macro about role-specific concepts.
macro_rules! structured_output {
    (
        impl $output:ty {
            tool_name: $tool_name:expr;
            tool_description: $tool_description:expr;
            schema: $schema_kind:ident $schema_args:tt;
            $(
                $(#[$normalize_meta:meta])*
                normalize($value:ident) $normalize:block
            )?
        }
    ) => {
        impl $crate::core::agent::StructuredOutput for $output {
            fn tool_name() -> &'static str {
                $tool_name
            }

            fn tool_description() -> &'static str {
                $tool_description
            }

            fn schema() -> $crate::core::agent::Schema {
                structured_output!(@schema $schema_kind $schema_args)
            }

            $(
                $(#[$normalize_meta])*
                fn normalize($value: &mut serde_json::Value) $normalize
            )?
        }
    };

    (@schema string($description:expr)) => {
        $crate::core::agent::Schema::string($description)
    };
    (@schema string_enum($description:expr, $values:expr)) => {
        $crate::core::agent::Schema::string_enum($description, $values)
    };
    (@schema integer($description:expr)) => {
        $crate::core::agent::Schema::integer($description)
    };
    (@schema integer_enum($description:expr, $values:expr)) => {
        $crate::core::agent::Schema::integer_enum($description, $values)
    };
    (@schema boolean($description:expr)) => {
        $crate::core::agent::Schema::boolean($description)
    };
    (@schema array($description:expr, $item_kind:ident $item_args:tt)) => {
        $crate::core::agent::Schema::array(
            $description,
            structured_output!(@schema $item_kind $item_args),
        )
    };
    (@schema object({ $($fields:tt)* })) => {
        $crate::core::agent::Schema::object(
            structured_output!(@object object({ $($fields)* }))
        )
    };
    (@schema object($description:expr, { $($fields:tt)* })) => {
        $crate::core::agent::Schema::object(
            structured_output!(@object object($description, { $($fields)* }))
        )
    };
    (
        @schema one_of(
            $discriminator:expr,
            $description:expr,
            {
                $(
                    $tag:literal => (
                        $variant_description:expr,
                        $object_kind:ident $object_args:tt
                    )
                ),* $(,)?
            }
        )
    ) => {{
        let schema = $crate::core::agent::OneOfSchema::new(
            $discriminator,
            $description,
        );
        $(
            let schema = schema.variant(
                $tag,
                $variant_description,
                structured_output!(@object $object_kind $object_args),
            );
        )*
        $crate::core::agent::Schema::one_of(schema)
    }};

    (@object object({ $($fields:tt)* })) => {{
        let schema = $crate::core::agent::ObjectSchema::new();
        structured_output!(@fields schema; $($fields)*)
    }};
    (@object object($description:expr, { $($fields:tt)* })) => {{
        let schema = $crate::core::agent::ObjectSchema::new().describe($description);
        structured_output!(@fields schema; $($fields)*)
    }};
    (@object fields($fields:expr)) => {
        $fields
    };

    (@fields $schema:ident;) => {
        $schema
    };
    (
        @fields $schema:ident;
        required $name:ident: $field_kind:ident $field_args:tt,
        $($remaining:tt)*
    ) => {{
        let $schema = $schema.required_property(
            stringify!($name),
            structured_output!(@schema $field_kind $field_args),
        );
        structured_output!(@fields $schema; $($remaining)*)
    }};
    (
        @fields $schema:ident;
        optional $name:ident: $field_kind:ident $field_args:tt,
        $($remaining:tt)*
    ) => {{
        let $schema = $schema.property(
            stringify!($name),
            structured_output!(@schema $field_kind $field_args),
        );
        structured_output!(@fields $schema; $($remaining)*)
    }};
}

pub(crate) use structured_output;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// One schema violation, located by a JSON path rooted at `$`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    pub path: String,
    pub message: String,
}

impl SchemaError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Why a captured structured-output value could not become a role's typed
/// output. Both variants are actionable by the model, so both are worth
/// sending back in a repair prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputError {
    /// The value violated the declared [`Schema`].
    Schema(SchemaError),
    /// The value matched the schema but not the Rust type behind it — a
    /// schema/Serde drift bug rather than a model mistake.
    Deserialize(String),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputError::Schema(error) => write!(f, "{error}"),
            OutputError::Deserialize(error) => {
                write!(
                    f,
                    "$: value does not match the expected output type ({error})"
                )
            }
        }
    }
}

impl std::error::Error for OutputError {}

/// Check `value` against `schema`, returning the first violation in a
/// deterministic order: type of the value itself, then unexpected
/// properties, then missing required properties, then property values in
/// declaration order.
pub fn validate(schema: &Schema, value: &Value) -> Result<(), SchemaError> {
    validate_at(schema, value, "$")
}

fn validate_at(schema: &Schema, value: &Value, path: &str) -> Result<(), SchemaError> {
    match schema {
        Schema::String { enum_values, .. } => {
            let Some(text) = value.as_str() else {
                return Err(SchemaError::new(
                    path,
                    format!("expected a string, got {}", json_type_name(value)),
                ));
            };
            if !enum_values.is_empty() && !enum_values.iter().any(|allowed| allowed == text) {
                return Err(SchemaError::new(
                    path,
                    format!(
                        "expected one of [{}], got {text:?}",
                        quoted_list(enum_values.iter().map(String::as_str))
                    ),
                ));
            }
            Ok(())
        }
        Schema::Integer { enum_values, .. } => {
            let number = value.as_i64().filter(|_| !value.is_boolean());
            let Some(number) = number else {
                return Err(SchemaError::new(
                    path,
                    format!("expected an integer, got {}", json_type_name(value)),
                ));
            };
            if !enum_values.is_empty() && !enum_values.contains(&number) {
                return Err(SchemaError::new(
                    path,
                    format!(
                        "expected one of [{}], got {number}",
                        enum_values
                            .iter()
                            .map(i64::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ));
            }
            Ok(())
        }
        Schema::Boolean { .. } => {
            if value.is_boolean() {
                Ok(())
            } else {
                Err(SchemaError::new(
                    path,
                    format!("expected a boolean, got {}", json_type_name(value)),
                ))
            }
        }
        Schema::Array { items, .. } => {
            let Some(entries) = value.as_array() else {
                return Err(SchemaError::new(
                    path,
                    format!("expected an array, got {}", json_type_name(value)),
                ));
            };
            for (index, entry) in entries.iter().enumerate() {
                validate_at(items, entry, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        Schema::Object(object) => validate_object(object, value, path, None),
        Schema::OneOf(one_of) => validate_one_of(one_of, value, path),
    }
}

fn validate_one_of(one_of: &OneOfSchema, value: &Value, path: &str) -> Result<(), SchemaError> {
    if !value.is_object() {
        return Err(SchemaError::new(
            path,
            format!("expected an object, got {}", json_type_name(value)),
        ));
    }
    let tag_path = child_path(path, &one_of.discriminator);
    let tags = quoted_list(one_of.tags().into_iter());
    let Some(tag) = value.get(&one_of.discriminator).filter(|v| !v.is_null()) else {
        return Err(SchemaError::new(
            &tag_path,
            format!(
                "required discriminator is missing; expected one of [{tags}]. \
                 The arguments received carried only these top-level keys: [{}] — \
                 the `{}` field itself was not transmitted. Re-emit the call with \
                 `{}` set alongside the keys you already sent.",
                received_keys(value),
                one_of.discriminator,
                one_of.discriminator
            ),
        ));
    };
    let Some(tag) = tag.as_str() else {
        return Err(SchemaError::new(
            &tag_path,
            format!("expected a string, got {}", json_type_name(tag)),
        ));
    };
    let Some(variant) = one_of.variant_for(tag) else {
        return Err(SchemaError::new(
            &tag_path,
            format!("expected one of [{tags}], got {tag:?}"),
        ));
    };
    validate_object(
        &variant.fields,
        value,
        path,
        Some(one_of.discriminator.as_str()),
    )
}

/// The top-level keys an arguments object actually carried, quoted and
/// comma-joined for an error message. Sorted so the message is stable across
/// calls (and testable); empty for a non-object, which the caller has
/// already rejected.
fn received_keys(value: &Value) -> String {
    let mut keys: Vec<&str> = value
        .as_object()
        .map(|object| object.keys().map(String::as_str).collect())
        .unwrap_or_default();
    keys.sort_unstable();
    quoted_list(keys.into_iter())
}

fn validate_object(
    object: &ObjectSchema,
    value: &Value,
    path: &str,
    discriminator: Option<&str>,
) -> Result<(), SchemaError> {
    let Some(map) = value.as_object() else {
        return Err(SchemaError::new(
            path,
            format!("expected an object, got {}", json_type_name(value)),
        ));
    };

    let mut allowed: Vec<&str> = Vec::new();
    if let Some(tag) = discriminator {
        allowed.push(tag);
    }
    allowed.extend(object.property_names());

    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(SchemaError::new(
                child_path(path, key),
                format!(
                    "unexpected property; allowed properties here are [{}]",
                    quoted_list(allowed.iter().copied())
                ),
            ));
        }
    }

    for name in &object.required {
        if map.get(name).filter(|v| !v.is_null()).is_none() {
            return Err(SchemaError::new(
                child_path(path, name),
                "required property is missing",
            ));
        }
    }

    for (name, field) in &object.properties {
        let Some(entry) = map.get(name).filter(|v| !v.is_null()) else {
            continue;
        };
        validate_at(field, entry, &child_path(path, name))?;
    }

    Ok(())
}

fn child_path(path: &str, name: &str) -> String {
    format!("{path}.{name}")
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn quoted_list<'a>(values: impl Iterator<Item = &'a str>) -> String {
    values
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// The role contract
// ---------------------------------------------------------------------------

/// Implemented by each role's typed completion output. Ties together the
/// tool identity/schema the model is asked to call with the Rust type its
/// JSON arguments deserialize into.
pub trait StructuredOutput: DeserializeOwned + Sized {
    /// Name of the structured-output tool the model must call.
    fn tool_name() -> &'static str;

    /// Human-readable description passed to the model backend.
    fn tool_description() -> &'static str;

    /// Declarative schema for the tool's arguments.
    fn schema() -> Schema;

    /// Compatibility fixups applied to the raw captured JSON *before*
    /// validation and deserialization. This is the one place where a role
    /// documents the shapes it forgives (a decision spelled with different
    /// case, an issue IID sent as `"#727"`, a boolean sent as `"true"`, a
    /// severity the model invented). Everything a role does not normalize
    /// here is a contract violation and gets repaired by asking the model
    /// again. Defaults to no fixups.
    fn normalize(_value: &mut Value) {}

    /// Build the neutral tool contract from `tool_name`/`tool_description`/`schema`.
    fn tool_definition() -> StructuredOutputTool {
        StructuredOutputTool {
            name: Self::tool_name().to_string(),
            description: Self::tool_description().to_string(),
            parameters: Self::schema(),
        }
    }

    /// The full decode pipeline for one captured tool call: normalize, then
    /// validate against [`Self::schema`], then deserialize.
    fn decode(mut value: Value) -> Result<Self, OutputError> {
        Self::normalize(&mut value);
        validate(&Self::schema(), &value).map_err(OutputError::Schema)?;
        serde_json::from_value(value).map_err(|error| OutputError::Deserialize(error.to_string()))
    }
}

/// Deserialization support for the tagged-union outputs described by
/// [`OneOfSchema`]. Splitting the discriminator off first lets each branch
/// deserialize into its own `deny_unknown_fields` wire type, so the Rust side
/// is as closed as the schema and the wire JSON are.
pub mod tagged {
    use serde::de::{self, Deserialize, DeserializeOwned, Deserializer};
    use serde_json::{Map, Value};

    /// Split a tagged-union object into its discriminator value and the
    /// remaining properties.
    pub fn parts<'de, D>(deserializer: D, discriminator: &str) -> Result<(String, Value), D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut map = Map::deserialize(deserializer)?;
        let tag = map
            .remove(discriminator)
            .as_ref()
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| de::Error::custom(format!("missing `{discriminator}`")))?;
        Ok((tag, Value::Object(map)))
    }

    /// Deserialize one branch's properties into its wire type.
    pub fn branch<T, E>(fields: Value) -> Result<T, E>
    where
        T: DeserializeOwned,
        E: de::Error,
    {
        serde_json::from_value(fields).map_err(de::Error::custom)
    }
}

/// Compatibility fixups shared by role [`StructuredOutput::normalize`]
/// implementations. Each one is a documented tolerance, not a general
/// coercion: anything these do not repair stays a contract violation.
pub mod compat {
    use serde_json::{Map, Value};

    fn object_mut(value: &mut Value) -> Option<&mut Map<String, Value>> {
        value.as_object_mut()
    }

    /// Move `from` to `to` when the model used a legacy property name and
    /// did not also send the canonical one.
    pub fn rename_property(value: &mut Value, from: &str, to: &str) {
        let Some(map) = object_mut(value) else { return };
        if map.contains_key(to) {
            map.remove(from);
            return;
        }
        if let Some(moved) = map.remove(from) {
            map.insert(to.to_string(), moved);
        }
    }

    /// Lower-case and trim a discriminator the model spelled differently
    /// (`"Approve"`, `" approve "`).
    pub fn normalize_tag(value: &mut Value, key: &str) {
        let Some(map) = object_mut(value) else { return };
        let Some(tag) = map.get(key).and_then(Value::as_str) else {
            return;
        };
        let normalized = tag.trim().to_ascii_lowercase();
        map.insert(key.to_string(), Value::String(normalized));
    }

    /// Coerce an issue/MR IID sent as `"727"`, `"#727"`, `"!727"` or `727.0`
    /// into an integer. Values that are absent, zero, or not parseable as a
    /// leading number are removed so the schema's required/optional rules
    /// decide what happens next.
    pub fn normalize_iid(value: &mut Value, key: &str) {
        let Some(map) = object_mut(value) else { return };
        let Some(raw) = map.get(key) else { return };
        let parsed = raw
            .as_u64()
            .or_else(|| raw.as_f64().filter(|n| n.fract() == 0.0).map(|n| n as u64))
            .or_else(|| {
                let text = raw.as_str()?.trim().trim_start_matches(['#', '!']).trim();
                let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
                digits.parse::<u64>().ok()
            })
            .filter(|iid| *iid > 0);
        match parsed {
            Some(iid) => {
                map.insert(key.to_string(), Value::from(iid));
            }
            None => {
                map.remove(key);
            }
        }
    }

    /// Coerce a boolean sent as `"true"`/`"False"`. Anything else is removed.
    pub fn normalize_bool(value: &mut Value, key: &str) {
        let Some(map) = object_mut(value) else { return };
        let Some(raw) = map.get(key) else { return };
        let parsed = raw.as_bool().or_else(|| {
            raw.as_str()
                .and_then(|text| text.trim().to_ascii_lowercase().parse::<bool>().ok())
        });
        match parsed {
            Some(flag) => {
                map.insert(key.to_string(), Value::Bool(flag));
            }
            None => {
                map.remove(key);
            }
        }
    }

    /// Fold a free-form enum value into the allowed set: trim/lower-case it,
    /// and replace anything unrecognized with `fallback`.
    pub fn normalize_enum(value: &mut Value, key: &str, allowed: &[&str], fallback: &str) {
        let Some(map) = object_mut(value) else { return };
        let Some(raw) = map.get(key) else { return };
        let normalized = raw
            .as_str()
            .map(|text| text.trim().to_ascii_lowercase())
            .filter(|text| allowed.contains(&text.as_str()))
            .unwrap_or_else(|| fallback.to_string());
        map.insert(key.to_string(), Value::String(normalized));
    }

    /// Drop an optional integer property whose value is outside the allowed
    /// set, so an out-of-range grade (a priority of `9`) degrades to "not
    /// stated" instead of failing the whole contract.
    pub fn drop_integer_outside(value: &mut Value, key: &str, allowed: &[i64]) {
        let Some(map) = object_mut(value) else { return };
        let Some(raw) = map.get(key) else { return };
        let in_range = raw
            .as_i64()
            .filter(|_| !raw.is_boolean())
            .is_some_and(|number| allowed.contains(&number));
        if !in_range {
            map.remove(key);
        }
    }

    /// Apply `fixup` to every element of an array property, so nested item
    /// contracts get the same tolerances as the top level.
    pub fn each_in_array(value: &mut Value, key: &str, mut fixup: impl FnMut(&mut Value)) {
        let Some(map) = object_mut(value) else { return };
        let Some(entries) = map.get_mut(key).and_then(Value::as_array_mut) else {
            return;
        };
        for entry in entries {
            fixup(entry);
        }
    }
}

/// Reusable conformance checks every role's structured-output contract must
/// pass. [`assert_contract`] is the mandatory one: it walks the declared
/// schema, synthesizes the minimal and maximal value each shape allows, and
/// asserts that both survive the real decode pipeline. That is what catches
/// schema/Serde drift — a property added to the schema but not to the Rust
/// type (or vice versa) fails immediately.
#[cfg(test)]
pub mod conformance {
    use super::*;
    use serde_json::{Map, json};

    /// The mandatory per-role contract suite. Call this from each role's
    /// tests with the role's output type.
    pub fn assert_contract<T: StructuredOutput + fmt::Debug>() {
        let tool = T::tool_definition();
        assert!(!tool.name.is_empty(), "tool name must not be empty");
        assert!(
            !tool.description.is_empty(),
            "{}: tool description must not be empty",
            tool.name
        );

        assert_schema_is_sound(&tool.parameters, &tool.name, "$");

        for (label, sample) in samples(&tool.parameters) {
            let minimal = T::decode(sample.minimal.clone());
            assert!(
                minimal.is_ok(),
                "{}: minimal {label} value {} was rejected: {}",
                tool.name,
                sample.minimal,
                minimal.unwrap_err()
            );
            let maximal = T::decode(sample.maximal.clone());
            assert!(
                maximal.is_ok(),
                "{}: maximal {label} value {} was rejected: {} \
                 (schema and Rust output type have drifted apart)",
                tool.name,
                sample.maximal,
                maximal.unwrap_err()
            );

            let mut polluted = sample.maximal.clone();
            polluted
                .as_object_mut()
                .expect("structured output values are objects")
                .insert("potlatch_unexpected_property".into(), json!(true));
            let error = T::decode(polluted).expect_err(&format!(
                "{}: {label} accepted an undeclared property; the contract must be closed",
                tool.name
            ));
            assert!(
                error.to_string().contains("unexpected property"),
                "{}: {label} rejected an undeclared property with the wrong error: {error}",
                tool.name
            );
        }
    }

    /// Assert a value decodes, returning the typed output.
    pub fn assert_accepts<T: StructuredOutput + fmt::Debug>(value: Value) -> T {
        match T::decode(value.clone()) {
            Ok(output) => output,
            Err(error) => panic!("expected {value} to decode, got: {error}"),
        }
    }

    /// Assert a value is rejected, returning the rendered error so a test can
    /// pin the message the repair prompt will carry.
    pub fn assert_rejects<T: StructuredOutput + fmt::Debug>(value: Value) -> String {
        match T::decode(value.clone()) {
            Ok(output) => panic!("expected {value} to be rejected, got: {output:?}"),
            Err(error) => error.to_string(),
        }
    }

    fn assert_schema_is_sound(schema: &Schema, tool: &str, path: &str) {
        assert!(
            !schema.description().is_empty(),
            "{tool}: {path} has no description",
        );
        match schema {
            Schema::Array { items, .. } => {
                assert_schema_is_sound(items, tool, &format!("{path}[]"))
            }
            Schema::Object(object) => assert_object_is_sound(object, tool, path, None),
            Schema::OneOf(one_of) => {
                assert!(
                    !one_of.discriminator.is_empty(),
                    "{tool}: {path} has an empty discriminator",
                );
                assert!(
                    one_of.variants.len() > 1,
                    "{tool}: {path} is a union with fewer than two variants",
                );
                let mut seen: Vec<&str> = Vec::new();
                for variant in &one_of.variants {
                    assert!(
                        !seen.contains(&variant.tag.as_str()),
                        "{tool}: {path} repeats the variant tag {:?}",
                        variant.tag
                    );
                    seen.push(&variant.tag);
                    assert!(
                        !variant.description.is_empty(),
                        "{tool}: {path} variant {:?} has no description",
                        variant.tag
                    );
                    assert_object_is_sound(
                        &variant.fields,
                        tool,
                        &format!("{path}({})", variant.tag),
                        Some(&one_of.discriminator),
                    );
                }
            }
            _ => {}
        }
    }

    fn assert_object_is_sound(
        object: &ObjectSchema,
        tool: &str,
        path: &str,
        discriminator: Option<&str>,
    ) {
        let mut seen: Vec<&str> = Vec::new();
        for (name, field) in &object.properties {
            assert!(
                !seen.contains(&name.as_str()),
                "{tool}: {path} declares the property {name:?} twice",
            );
            seen.push(name);
            assert!(
                Some(name.as_str()) != discriminator,
                "{tool}: {path} redeclares the discriminator {name:?} as a property",
            );
            assert_schema_is_sound(field, tool, &format!("{path}.{name}"));
        }
        for name in &object.required {
            assert!(
                seen.contains(&name.as_str()),
                "{tool}: {path} requires the undeclared property {name:?}",
            );
        }
    }

    struct Sample {
        minimal: Value,
        maximal: Value,
    }

    /// One sample pair per shape the schema allows: a single pair for a plain
    /// object contract, one per variant for a tagged union.
    fn samples(schema: &Schema) -> Vec<(String, Sample)> {
        match schema {
            Schema::Object(object) => vec![(
                "object".to_string(),
                Sample {
                    minimal: object_sample(object, false),
                    maximal: object_sample(object, true),
                },
            )],
            Schema::OneOf(one_of) => one_of
                .variants
                .iter()
                .map(|variant| {
                    let mut minimal = object_sample(&variant.fields, false);
                    let mut maximal = object_sample(&variant.fields, true);
                    for value in [&mut minimal, &mut maximal] {
                        value
                            .as_object_mut()
                            .expect("object sample")
                            .insert(one_of.discriminator.clone(), json!(variant.tag));
                    }
                    (
                        format!("variant {:?}", variant.tag),
                        Sample { minimal, maximal },
                    )
                })
                .collect(),
            other => panic!(
                "a structured-output contract must be an object or a tagged union, got {other:?}"
            ),
        }
    }

    fn object_sample(object: &ObjectSchema, maximal: bool) -> Value {
        let mut map = Map::new();
        for (name, field) in &object.properties {
            if maximal || object.required.contains(name) {
                map.insert(name.clone(), value_sample(field, maximal));
            }
        }
        Value::Object(map)
    }

    fn value_sample(schema: &Schema, maximal: bool) -> Value {
        match schema {
            Schema::String { enum_values, .. } => match enum_values.first() {
                Some(first) => json!(first),
                None => json!("sample"),
            },
            Schema::Integer { enum_values, .. } => match enum_values.first() {
                Some(first) => json!(first),
                None => json!(1),
            },
            Schema::Boolean { .. } => json!(true),
            Schema::Array { items, .. } => {
                if maximal {
                    json!([value_sample(items, maximal)])
                } else {
                    json!([])
                }
            }
            Schema::Object(object) => object_sample(object, maximal),
            Schema::OneOf(one_of) => {
                let variant = one_of.variants.first().expect("union has variants");
                let mut sample = object_sample(&variant.fields, maximal);
                sample
                    .as_object_mut()
                    .expect("object sample")
                    .insert(one_of.discriminator.clone(), json!(variant.tag));
                sample
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize, PartialEq)]
    struct MacroOutput {
        title: String,
        priority: Option<i64>,
    }

    structured_output! {
        impl MacroOutput {
            tool_name: "macro_output";
            tool_description: "Macro test output.";
            schema: object("A macro-generated object.", {
                required title: string("Title."),
                optional priority: integer_enum("Priority.", &[1, 2, 3]),
            });
            normalize(value) {
                compat::rename_property(value, "name", "title");
            }
        }
    }

    fn sample_object() -> ObjectSchema {
        ObjectSchema::new()
            .describe("A sample object.")
            .required_property("title", Schema::string("Title."))
            .property("priority", Schema::integer_enum("Priority.", &[1, 2, 3]))
            .property("draft", Schema::boolean("Draft flag."))
            .property("tags", Schema::array("Tags.", Schema::string("A tag.")))
    }

    fn sample_union() -> Schema {
        Schema::one_of(
            OneOfSchema::new("decision", "The decision.")
                .variant(
                    "approve",
                    "Approve it.",
                    ObjectSchema::new().property("summary", Schema::string("Summary.")),
                )
                .variant(
                    "request_changes",
                    "Ask for changes.",
                    ObjectSchema::new().required_property("feedback", Schema::string("Feedback.")),
                ),
        )
    }

    #[test]
    fn structured_output_macro_generates_metadata_schema_and_normalization() {
        let tool = MacroOutput::tool_definition();
        assert_eq!(tool.name, "macro_output");
        assert_eq!(tool.description, "Macro test output.");
        assert_eq!(
            tool.parameters,
            Schema::object(
                ObjectSchema::new()
                    .describe("A macro-generated object.")
                    .required_property("title", Schema::string("Title."))
                    .property("priority", Schema::integer_enum("Priority.", &[1, 2, 3])),
            )
        );

        let decoded = MacroOutput::decode(json!({"name": "ready", "priority": 2})).unwrap();
        assert_eq!(
            decoded,
            MacroOutput {
                title: "ready".to_string(),
                priority: Some(2),
            }
        );
    }

    #[test]
    fn object_builder_preserves_property_order_and_required() {
        let schema = sample_object();
        assert_eq!(
            schema.property_names(),
            vec!["title", "priority", "draft", "tags"]
        );
        assert_eq!(schema.required, vec!["title".to_string()]);
    }

    #[test]
    fn scalar_constructors_build_expected_variants() {
        assert_eq!(
            Schema::string("d"),
            Schema::String {
                description: "d".into(),
                enum_values: vec![]
            }
        );
        assert_eq!(
            Schema::string_enum("d", &["x", "y"]),
            Schema::String {
                description: "d".into(),
                enum_values: vec!["x".into(), "y".into()]
            }
        );
        assert_eq!(
            Schema::integer_enum("d", &[1, 2]),
            Schema::Integer {
                description: "d".into(),
                enum_values: vec![1, 2]
            }
        );
        assert_eq!(
            Schema::boolean("d"),
            Schema::Boolean {
                description: "d".into()
            }
        );
        assert_eq!(Schema::array("d", Schema::string("i")).description(), "d");
    }

    #[test]
    fn validate_accepts_a_conforming_object() {
        let schema = Schema::object(sample_object());
        validate(
            &schema,
            &json!({"title": "T", "priority": 2, "draft": false, "tags": ["a", "b"]}),
        )
        .unwrap();
    }

    #[test]
    fn validate_reports_missing_required_property_with_path() {
        let error =
            validate(&Schema::object(sample_object()), &json!({"priority": 1})).unwrap_err();
        assert_eq!(error.path, "$.title");
        assert!(error.message.contains("required property is missing"));
    }

    #[test]
    fn validate_treats_explicit_null_as_missing() {
        let error =
            validate(&Schema::object(sample_object()), &json!({"title": null})).unwrap_err();
        assert_eq!(error.path, "$.title");
        let schema = Schema::object(sample_object());
        validate(&schema, &json!({"title": "T", "priority": null})).unwrap();
    }

    #[test]
    fn validate_rejects_undeclared_properties() {
        let error = validate(
            &Schema::object(sample_object()),
            &json!({"title": "T", "surprise": 1}),
        )
        .unwrap_err();
        assert_eq!(error.path, "$.surprise");
        assert!(error.message.contains("unexpected property"));
        assert!(error.message.contains("\"title\""));
    }

    #[test]
    fn validate_reports_wrong_scalar_types() {
        let schema = Schema::object(sample_object());
        let error = validate(&schema, &json!({"title": 5})).unwrap_err();
        assert_eq!(
            error.to_string(),
            "$.title: expected a string, got a number"
        );
        let error = validate(&schema, &json!({"title": "T", "draft": "yes"})).unwrap_err();
        assert_eq!(
            error.to_string(),
            "$.draft: expected a boolean, got a string"
        );
        let error = validate(&schema, &json!({"title": "T", "priority": true})).unwrap_err();
        assert_eq!(
            error.to_string(),
            "$.priority: expected an integer, got a boolean"
        );
    }

    #[test]
    fn validate_reports_enum_violations() {
        let error = validate(
            &Schema::object(sample_object()),
            &json!({"title": "T", "priority": 9}),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "$.priority: expected one of [1, 2, 3], got 9"
        );
    }

    #[test]
    fn validate_walks_into_arrays_with_indexed_paths() {
        let schema = Schema::object(
            ObjectSchema::new().describe("d").required_property(
                "items",
                Schema::array(
                    "Items.",
                    Schema::object(
                        ObjectSchema::new()
                            .describe("An item.")
                            .required_property("name", Schema::string("Name.")),
                    ),
                ),
            ),
        );
        let error = validate(&schema, &json!({"items": [{"name": "a"}, {}]})).unwrap_err();
        assert_eq!(error.path, "$.items[1].name");
    }

    #[test]
    fn validate_selects_the_union_branch_by_discriminator() {
        let schema = sample_union();
        validate(&schema, &json!({"decision": "approve"})).unwrap();
        validate(&schema, &json!({"decision": "approve", "summary": "ok"})).unwrap();
        validate(
            &schema,
            &json!({"decision": "request_changes", "feedback": "fix it"}),
        )
        .unwrap();
    }

    #[test]
    fn validate_reports_missing_or_unknown_discriminator() {
        let schema = sample_union();
        let error = validate(&schema, &json!({"summary": "ok"})).unwrap_err();
        assert_eq!(error.path, "$.decision");
        assert!(error.message.contains("required discriminator is missing"));

        let error = validate(&schema, &json!({"decision": "maybe"})).unwrap_err();
        assert_eq!(
            error.to_string(),
            "$.decision: expected one of [\"approve\", \"request_changes\"], got \"maybe\""
        );
    }

    #[test]
    fn missing_discriminator_names_the_keys_that_were_received() {
        // A provider that drops all but one parameter in a tool call leaves
        // the discriminator absent while a branch's real fields arrive. The
        // message must say which keys made it through, so one repair turn
        // reveals "only one parameter is being transmitted" — seen in the
        // wild as 55 review calls where every emission carried exactly one
        // key and the model never learned that from "discriminator missing".
        let schema = sample_union();
        let error = validate(&schema, &json!({"feedback": "fix it"})).unwrap_err();
        assert_eq!(error.path, "$.decision");
        let message = error.message;
        assert!(
            message.contains("carried only these top-level keys: [\"feedback\"]"),
            "{message}"
        );
        assert!(
            message.contains("the `decision` field itself was not transmitted"),
            "{message}"
        );

        // Multiple received keys are listed sorted and joined.
        let error = validate(
            &schema,
            &json!({"public_comment": "note", "feedback": "fix it"}),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("carried only these top-level keys: [\"feedback\", \"public_comment\"]"),
            "{}",
            error.message
        );

        // An empty object reports an empty key list.
        let error = validate(&schema, &json!({})).unwrap_err();
        assert!(
            error
                .message
                .contains("carried only these top-level keys: []"),
            "{}",
            error.message
        );
    }

    #[test]
    fn validate_confines_properties_to_the_selected_branch() {
        let schema = sample_union();
        let error = validate(
            &schema,
            &json!({"decision": "approve", "feedback": "fix it"}),
        )
        .unwrap_err();
        assert_eq!(error.path, "$.feedback");
        assert!(error.message.contains("unexpected property"));

        let error = validate(&schema, &json!({"decision": "request_changes"})).unwrap_err();
        assert_eq!(error.path, "$.feedback");
        assert!(error.message.contains("required property is missing"));
    }

    #[derive(Debug, serde::Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct SampleOutput {
        decision: String,
        #[serde(default)]
        summary: Option<String>,
    }

    impl StructuredOutput for SampleOutput {
        fn tool_name() -> &'static str {
            "sample"
        }

        fn tool_description() -> &'static str {
            "sample tool"
        }

        fn schema() -> Schema {
            Schema::object(
                ObjectSchema::new()
                    .describe("Sample.")
                    .required_property("decision", Schema::string_enum("d", &["a", "b"]))
                    .property("summary", Schema::string("s")),
            )
        }

        fn normalize(value: &mut Value) {
            compat::normalize_tag(value, "decision");
        }
    }

    #[test]
    fn tool_definition_combines_name_description_and_schema() {
        let tool = SampleOutput::tool_definition();
        assert_eq!(tool.name, "sample");
        assert_eq!(tool.description, "sample tool");
        assert_eq!(tool.parameters, SampleOutput::schema());
    }

    #[test]
    fn decode_normalizes_then_validates_then_deserializes() {
        let output = SampleOutput::decode(json!({"decision": " A "})).unwrap();
        assert_eq!(
            output,
            SampleOutput {
                decision: "a".into(),
                summary: None
            }
        );
    }

    #[test]
    fn decode_reports_schema_violations_before_serde() {
        let error = SampleOutput::decode(json!({"decision": "z"})).unwrap_err();
        assert!(matches!(error, OutputError::Schema(_)));
        assert!(error.to_string().starts_with("$.decision: expected one of"));
    }

    #[test]
    fn decode_reports_deserialize_failures_with_a_path_prefix() {
        let error = OutputError::Deserialize("invalid type".into());
        assert!(error.to_string().starts_with("$: value does not match"));
    }

    #[test]
    fn sample_output_passes_the_shared_conformance_suite() {
        conformance::assert_contract::<SampleOutput>();
    }

    // --- compatibility normalization ---

    #[test]
    fn compat_normalize_tag_trims_and_lowercases() {
        let mut value = json!({"decision": "  Request_Changes "});
        compat::normalize_tag(&mut value, "decision");
        assert_eq!(value, json!({"decision": "request_changes"}));
    }

    #[test]
    fn compat_normalize_iid_accepts_prefixed_and_stringified_numbers() {
        for raw in [json!(727), json!("727"), json!("#727"), json!("!727 ")] {
            let mut value = json!({ "iid": raw });
            compat::normalize_iid(&mut value, "iid");
            assert_eq!(value, json!({"iid": 727}), "raw value did not normalize");
        }
    }

    #[test]
    fn compat_normalize_iid_drops_zero_and_unparseable_values() {
        for raw in [json!(0), json!("none"), json!(true), json!("#0")] {
            let mut value = json!({ "iid": raw });
            compat::normalize_iid(&mut value, "iid");
            assert_eq!(value, json!({}), "raw value should have been dropped");
        }
    }

    #[test]
    fn compat_normalize_bool_accepts_stringified_booleans() {
        let mut value = json!({"flag": "True"});
        compat::normalize_bool(&mut value, "flag");
        assert_eq!(value, json!({"flag": true}));

        let mut value = json!({"flag": "maybe"});
        compat::normalize_bool(&mut value, "flag");
        assert_eq!(value, json!({}));
    }

    #[test]
    fn compat_normalize_enum_falls_back_for_unknown_values() {
        let mut value = json!({"severity": "BLOCKER"});
        compat::normalize_enum(&mut value, "severity", &["critical", "low"], "low");
        assert_eq!(value, json!({"severity": "low"}));

        let mut value = json!({"severity": " Critical "});
        compat::normalize_enum(&mut value, "severity", &["critical", "low"], "low");
        assert_eq!(value, json!({"severity": "critical"}));
    }

    #[test]
    fn compat_rename_property_prefers_the_canonical_name() {
        let mut value = json!({"sample_line": "x"});
        compat::rename_property(&mut value, "sample_line", "log_line");
        assert_eq!(value, json!({"log_line": "x"}));

        let mut value = json!({"sample_line": "x", "log_line": "y"});
        compat::rename_property(&mut value, "sample_line", "log_line");
        assert_eq!(value, json!({"log_line": "y"}));
    }

    #[test]
    fn compat_each_in_array_visits_every_element() {
        let mut value = json!({"items": [{"n": "1"}, {"n": "#2"}]});
        compat::each_in_array(&mut value, "items", |item| {
            compat::normalize_iid(item, "n");
        });
        assert_eq!(value, json!({"items": [{"n": 1}, {"n": 2}]}));
    }
}
