//! Issue labels shared across agent roles.

/// Issue label: PMO needs human clarification before it can plan. While this
/// label is present the PMO never auto-grabs the issue — a human must reply and
/// remove the label to unblock planning. See [`PMO_PLANNING`] for the active
/// planning state.
pub const PMO_PENDING: &str = "pmo-pending";
/// Issue label: PMO is actively planning this issue. While present the PMO
/// keeps the issue claimed and continues refining the plan whenever new human
/// comments arrive. Distinct from [`PMO_PENDING`], which means the PMO is
/// blocked waiting for a human answer and must not re-grab the issue.
pub const PMO_PLANNING: &str = "pmo-planning";
/// MR label: highest-priority AI worker/reviewer flow, independent of scope label.
pub const NEED_AI_WORKER: &str = "need-ai-worker";
/// Issue label: worker skips this issue entirely. Used by QA agent for
/// clarification issues that are questions for humans, not implementation tasks.
pub const DO_NOT_IMPLEMENT: &str = "do-not-implement";
/// Issue label: created by the QA agent (findings and clarification questions).
pub const QA: &str = "qa";
