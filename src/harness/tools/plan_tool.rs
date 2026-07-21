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
            "description": "Emit a structured JSON plan that Potlatch reads as the canonical handoff. Call this with your decision and supporting details. The JSON shape depends on your role — for the PMO agent, use {decision, instructions?, sub_issues?, reason?, question?}. This is the primary output channel when available; streamed text is secondary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "description": "The structured plan as a JSON object. Shape depends on role.",
                        "type": "object"
                    }
                },
                "required": ["plan"]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let plan = args
            .get("plan")
            .ok_or_else(|| anyhow::anyhow!("missing 'plan' argument"))?;
        if !plan.is_object() {
            anyhow::bail!("'plan' must be a JSON object");
        }
        // Store in the side-channel cell (last call wins).
        *self.cell.lock().unwrap() = Some(plan.clone());
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
        let args = json!({"plan": {"decision": "split", "sub_issues": [{"title": "A"}]}});
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
        tool.execute(&json!({"plan": {"decision": "a"}}), "/tmp")
            .unwrap();
        tool.execute(&json!({"plan": {"decision": "b"}}), "/tmp")
            .unwrap();
        let captured = PlanTool::take(&c);
        assert_eq!(captured, Some(json!({"decision": "b"})));
    }

    #[test]
    fn rejects_missing_plan_field() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let result = tool.execute(&json!({}), "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_non_object_plan() {
        let c = cell();
        let tool = PlanTool::new(Arc::clone(&c));
        let result = tool.execute(&json!({"plan": "not an object"}), "/tmp");
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
        tool.execute(&json!({"plan": {"x": 1}}), "/tmp").unwrap();
        assert!(PlanTool::take(&c).is_some());
        // Second take returns None — cell was cleared.
        assert_eq!(PlanTool::take(&c), None);
    }
}
