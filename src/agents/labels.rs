//! Issue labels shared across agent roles.

pub const PMO_PENDING: &str = "pmo-pending";
/// MR label: highest-priority AI worker/reviewer flow, independent of scope label.
pub const NEED_AI_WORKER: &str = "need-ai-worker";
/// Issue label: worker skips this issue entirely. Used by QA agent for
/// clarification issues that are questions for humans, not implementation tasks.
pub const DO_NOT_IMPLEMENT: &str = "do-not-implement";
/// Issue label: created by the QA agent (findings and clarification questions).
pub const QA: &str = "qa";
