//! Plan tool: lets the model emit a structured JSON plan that the harness
//! forwards to the caller (e.g. the PMO agent) via a side-channel cell.
//!
//! Only registered when the session is in plan mode — the harness ACP server
//! gates registration on `session.mode == "plan"`. The caller reads the
//! captured JSON via [`PlanTool::take`] after the agent loop completes.
//!
//! The tool's result is also stored in the conversation context as
//! `ContextKind::ToolResult` (evictable), but the canonical copy lives in the
//! side-channel cell, so context compaction can't lose it.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

/// Shared side-channel cell that holds the JSON the model emitted via the
/// `plan` tool. The `AgentLoop` (or its caller) reads this after `run()`.
pub type PlanCell = Arc<Mutex<Option<Value>>>;

pub struct PlanTool {
    cell: PlanCell,
}

impl PlanTool {
    pub fn new(cell: PlanCell) -> Self {
        Self { cell }
    }

    /// Take the captured plan JSON, clearing the cell. Returns `None` if the
    /// model never called the tool.
    pub fn take(cell: &PlanCell) -> Option<Value> {
        cell.lock().unwrap().take()
    }
}

impl Tool for PlanTool {
    fn name(&self) -> &str {
        "plan"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Emit your triage decision as a structured plan. This is the primary output channel — Potlatch reads the tool's JSON, not your streamed text. Call this exactly once with your decision and the fields relevant to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "description": "Your triage decision. Must be exactly one of: \"guide_worker\", \"split\", \"already_done\", \"needs_clarification\", \"wait_for_dependency\".",
                        "type": "string",
                        "enum": ["guide_worker", "split", "already_done", "needs_clarification", "wait_for_dependency"]
                    },
                    "instructions": {
                        "description": "For guide_worker: 3-5 sentences with one clear action for the worker. Posted to GitLab as a plain issue comment that the worker reads from the comment stream. Keep it worker-facing and actionable.",
                        "type": "string"
                    },
                    "sub_issues": {
                        "description": "For split: the sub-issues to create. Each must have a title and description.",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "description": "Concise sub-issue title.",
                                    "type": "string"
                                },
                                "description": {
                                    "description": "Scope and acceptance criteria. If this sub-issue depends on another, reference it by title.",
                                    "type": "string"
                                },
                                "priority": {
                                    "description": "Priority: 1 (critical/blocking), 2 (high/depended-on), 3 (normal/independent).",
                                    "type": "integer",
                                    "enum": [1, 2, 3]
                                }
                            },
                            "required": ["title", "description"]
                        }
                    },
                    "reason": {
                        "description": "For already_done: why the codebase already satisfies the issue.",
                        "type": "string"
                    },
                    "question": {
                        "description": "For needs_clarification: specific questions for a human.",
                        "type": "string"
                    },
                    "dependency_issue_iid": {
                        "description": "For wait_for_dependency: the IID (number) of the existing open issue this issue depends on and must wait for. Must be a positive integer.",
                        "type": "integer"
                    }
                },
                "required": ["decision"]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        // The model passes the fields directly as top-level arguments (no
        // wrapping "plan" key). Store the entire args object as the plan.
        if !args.is_object() {
            anyhow::bail!("plan arguments must be a JSON object");
        }
        if args.get("decision").and_then(Value::as_str).is_none() {
            anyhow::bail!("missing or invalid 'decision' field");
        }
        // Store in the side-channel cell (last call wins).
        *self.cell.lock().unwrap() = Some(args.clone());
        Ok("Plan recorded.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> PlanCell {
        Arc::new(Mutex::new(None))
    }

    #[test]
    fn stores_plan_json() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let args = json!({"decision": "split", "sub_issues": [{"title": "A"}]});
        let result = tool.execute(&args, "/tmp").unwrap();
        assert_eq!(result, "Plan recorded.");
        let captured = c.lock().unwrap().clone();
        assert_eq!(
            captured,
            Some(json!({"decision": "split", "sub_issues": [{"title": "A"}]}))
        );
    }

    #[test]
    fn second_call_overwrites_first() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        tool.execute(&json!({"decision": "a"}), "/tmp").unwrap();
        tool.execute(&json!({"decision": "b"}), "/tmp").unwrap();
        let captured = PlanTool::take(&c);
        assert_eq!(captured, Some(json!({"decision": "b"})));
    }

    #[test]
    fn rejects_missing_decision_field() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let result = tool.execute(&json!({}), "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_object_args() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let result = tool.execute(&json!("not an object"), "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn take_returns_none_when_never_called() {
        let c = cell();
        assert_eq!(PlanTool::take(&c), None);
    }

    #[test]
    fn take_clears_cell() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        tool.execute(&json!({"decision": "split"}), "/tmp").unwrap();
        assert!(PlanTool::take(&c).is_some());
        // Second take returns None — cell was cleared.
        assert_eq!(PlanTool::take(&c), None);
    }

    #[test]
    fn schema_lists_explicit_properties() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let schema = tool.schema();
        let props = schema["parameters"]["properties"].as_object().unwrap();
        // The decision field is required and has an enum.
        assert!(props.contains_key("decision"));
        assert_eq!(
            schema["parameters"]["required"][0].as_str(),
            Some("decision")
        );
        let decision = &props["decision"];
        assert_eq!(decision["type"].as_str(), Some("string"));
        let enums: Vec<&str> = decision["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            enums,
            vec![
                "guide_worker",
                "split",
                "already_done",
                "needs_clarification",
                "wait_for_dependency"
            ]
        );
        // Sub-issues have explicit title/description/priority.
        let sub_issue_props = props["sub_issues"]["items"]["properties"]
            .as_object()
            .unwrap();
        assert!(sub_issue_props.contains_key("title"));
        assert!(sub_issue_props.contains_key("description"));
        assert!(sub_issue_props.contains_key("priority"));
        // The dependency_issue_iid field is advertised for wait_for_dependency.
        assert!(props.contains_key("dependency_issue_iid"));
        assert_eq!(
            props["dependency_issue_iid"]["type"].as_str(),
            Some("integer")
        );
        // No wrapping "plan" key — fields are top-level.
        assert!(!props.contains_key("plan"));
    }
}
