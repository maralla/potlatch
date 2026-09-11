//! Clerk agent: maintains durable project memory as a plain markdown file.
//!
//! Bus-served: exposes a `memory` tool that other agents call to replace the
//! full memory content. Also registers a context channel on the bus so other
//! agents' harness sessions receive the current memory at session init as a
//! system message.
//!
//! Model-backed: after each write, the agent asks the model to reorganize,
//! deduplicate, and filter the memory down to stable, project-wide facts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use crate::core::agent::{AgentModel, CoreAgent, ModelPreferences};
use crate::core::agent::{InvokeOptions, structured_output};
use crate::core::banner::Banner;
use crate::core::bus::{
    AgentInbox, AgentRequest, AgentToolDefinition, ContextChannelGuard, ContextProvider,
};
use crate::core::config::Config;
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::paths::memory_dir;

const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
const MEMORY_OPERATION: &str = "memory_write";
const MAX_MEMORY_BYTES: usize = 16_384;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ClerkAgentSettings {}

pub struct ClerkAgent {
    runtime: AgentRuntime,
    inbox: AgentInbox,
    model: AgentModel,
    memory_path: PathBuf,
    _context_guard: ContextChannelGuard,
}

impl CoreAgent for ClerkAgent {
    type Settings = ClerkAgentSettings;
    const FIXED_INSTANCES: Option<usize> = Some(1);

    fn name() -> &'static str {
        "clerk"
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    fn banner(_config: &Config, _banner: &mut Banner) {}

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: REQUEST_TASK,
            interval: Duration::ZERO,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 0,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        ensure!(task_id == REQUEST_TASK, "unknown clerk task {task_id:?}");
        if let Some(request) = self.inbox.recv_timeout(INBOX_WAIT)? {
            self.handle_request(request);
        }
        Ok(())
    }

    fn build(ctx: crate::core::workflow::AgentBuildContext<Self::Settings>) -> Result<Self> {
        let bus = ctx
            .workflow
            .bus
            .as_ref()
            .context("clerk agent requires the cross-agent bus")?;
        let inbox = bus.register(Self::name(), vec![memory_tool_definition()])?;
        let working_dir = ctx.workflow.base_dir.clone();
        // Clerk's agent working dir IS the project working dir, but pin the
        // marker root explicitly like every other agent — no reliance on
        // that coincidence.
        let model = AgentModel::connect(
            &ctx,
            &working_dir,
            ModelPreferences {
                agents_dir: Some(
                    crate::paths::agents_dir_in_project_working_dir(Path::new(&working_dir))
                        .display()
                        .to_string(),
                ),
                ..ModelPreferences::default()
            },
        )?;
        let memory_path = memory_file_path(&working_dir);
        let memory_path_for_provider = memory_path.clone();
        let context_guard = bus.register_context(
            "memory",
            Arc::new(move || fs::read_to_string(&memory_path_for_provider).unwrap_or_default())
                as ContextProvider,
        );
        Ok(Self {
            runtime: ctx.runtime.clone(),
            inbox,
            model,
            memory_path,
            _context_guard: context_guard,
        })
    }

    fn on_start(&mut self) -> Result<()> {
        info!("Clerk agent ready");
        Ok(())
    }
}

impl ClerkAgent {
    fn handle_request(&mut self, request: AgentRequest) {
        let payload = request.payload.clone();
        let write_result = handle_memory_request(&self.memory_path, &request.operation, payload);
        // Respond immediately after the append so the caller is not blocked by
        // the model-driven reorganization. The reorg runs after the response
        // is sent; new requests arriving during reorg are queued in the inbox
        // and processed on the next periodic tick.
        let success = write_result.is_ok();
        let response: Result<Value> = match write_result {
            Ok(()) => fs::read_to_string(&self.memory_path)
                .map(Value::String)
                .context("read memory after write"),
            Err(error) => Err(error),
        };
        request.respond(response);
        // After responding, reorganize the full memory with the model to merge
        // the new content, deduplicate, and keep only stable project-wide facts.
        if success && let Err(error) = self.reorganize() {
            warn!("memory reorganization after write failed: {error}");
        }
    }

    fn reorganize(&mut self) -> Result<()> {
        let current = fs::read_to_string(&self.memory_path).unwrap_or_default();
        if current.trim().is_empty() {
            return Ok(());
        }
        let prompt = build_reorganize_prompt(&current);
        let result = self.model.complete_typed::<MemoryReorgOutput>(
            &prompt,
            &InvokeOptions {
                activity_label: Some(format!("{} reorganizing memory", self.runtime.agent_id())),
                ..InvokeOptions::default()
            },
        );
        match result {
            Ok(completion) => {
                let reorganized = completion.output.content;
                if !reorganized.trim().is_empty()
                    && reorganized.len() <= MAX_MEMORY_BYTES
                    && reorganized != current
                {
                    save_memory(&self.memory_path, &reorganized)?;
                    info!(
                        "memory reorganized ({} -> {} bytes)",
                        current.len(),
                        reorganized.len()
                    );
                }
            }
            Err(error) => {
                warn!("memory reorganization failed: {error}");
            }
        }
        Ok(())
    }
}

fn handle_memory_request(memory_path: &Path, operation: &str, payload: Value) -> Result<()> {
    ensure!(
        operation == MEMORY_OPERATION,
        "unsupported memory operation {operation:?}"
    );
    let content = payload
        .get("content")
        .and_then(Value::as_str)
        .context("memory write requires a 'content' string")?;
    ensure!(
        !content.trim().is_empty(),
        "memory content must not be empty"
    );
    ensure!(
        content.len() <= MAX_MEMORY_BYTES,
        "memory content exceeds {MAX_MEMORY_BYTES} bytes"
    );
    // Append the new content to the existing memory. The model-driven
    // reorganization step merges, deduplicates, and filters the combined
    // content into the final memory.
    let existing = fs::read_to_string(memory_path).unwrap_or_default();
    let combined = if existing.trim().is_empty() {
        content.to_string()
    } else {
        format!("{existing}\n\n{content}")
    };
    save_memory(memory_path, &combined)?;
    Ok(())
}

fn save_memory(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create memory directory {}", parent.display()))?;
    }
    let temp_path = path.with_extension("md.tmp");
    fs::write(&temp_path, content)
        .with_context(|| format!("write temporary memory {}", temp_path.display()))?;
    fs::rename(&temp_path, path).with_context(|| format!("replace memory {}", path.display()))?;
    Ok(())
}

fn build_reorganize_prompt(current: &str) -> String {
    format!(
        r#"You are the memory curator for THIS software project. The durable project memory is not a log of what happened — it is the project's wisdom: fundamental knowledge that guides future action across unrelated tasks, like the judgment a seasoned engineer carries from one project to the next.

Do NOT inspect the filesystem. The working directory may contain stale agent worktrees and checkout directories that do not reflect the current project state. Work ONLY from the memory text provided below — your job is to curate it, not to verify it against the codebase.

Think of memory as wisdom, not notes. Wisdom is knowledge that, once learned, changes how you approach ALL future work on the project — not just the task that surfaced it. Test each entry against this question: "If I forgot this, would I make a wrong decision on an unrelated future task?" If the answer is no, it is not wisdom.

Wisdom is:
- Principles and invariants that constrain design choices ("the data model forbids X", "module Y is the sole authority for Z — all writes must go through it").
- Conventions that prevent recurring mistakes ("never commit generated code", "all public APIs require integration tests").
- Cross-cutting constraints that surprise newcomers ("the build system caches at P; a stale cache causes silent test failures — always clean after switching branches").

Wisdom is NOT:
- Implementation details of specific functions, types, or files. "Function F calls G" or "struct S has field X" describes the current code, not a durable principle. These are notes, not wisdom — they become stale on the next refactor and mislead.
- Debugging findings or issue-specific observations. "While working on issue #42, the breaker tripped because threshold X was too low" is a task recording, not wisdom. The wisdom would be "breaker thresholds below N cause false trips on transient 500s" — IF that generalizes.
- Code structure descriptions. "Package A imports B but not C" or "handler H resolves its sink via the registry" describes the current tree; it is not a guiding principle.
- Subsystem-specific operational procedures. Describing how one subsystem currently behaves ("X retries indefinitely", "Y is in-memory by design") is an operational note about that subsystem, not a project-wide principle. Unless the lesson generalizes to ALL similar work, it is not wisdom.
- Triage procedures for specific issue patterns. A step-by-step procedure for one kind of situation is not a guiding principle. Wisdom shapes how you think; procedures dictate what you do in one scenario.
- Anything that names specific identifiers (function names, struct fields, variable names, error codes) as the subject of the fact. Wisdom is phrased as principles, not as code readings.
- Facts about other codebases. Wisdom is about THIS project only.
- Restatements of what is already documented in project guidance files (README, AGENTS.md, CONTRIBUTING).

Be ruthless. The memory above is almost certainly full of task recordings and subsystem notes disguised as principles. Drop anything that describes how a specific subsystem currently behaves or what to do in a specific situation — those are operational notes, not wisdom. If a genuine principle is buried inside a detailed note, extract ONLY the principle and discard the specific situation. Prefer a short memory of real wisdom over a long memory of accurate-but-narrow operational notes. If nothing in the memory rises to the level of guiding future action across unrelated tasks, return empty content — an empty memory is better than a clutter of situation-specific procedures.

Recency: recently added facts (at the end of the memory) are more likely to still be relevant. When the memory is too large, drop the oldest, most detailed, most implementation-specific entries first. Never drop a recent genuine principle to keep an old code reading.

## Current memory

{current}
"#
    )
}

/// The structured output of a memory reorganization: the full reorganized
/// markdown content.
#[derive(Debug, Clone, Deserialize)]
struct MemoryReorgOutput {
    content: String,
}

structured_output! {
    impl MemoryReorgOutput {
        tool_name: "memory_reorganize";
        tool_description: "Return the reorganized durable project memory as markdown.";
        schema: object("The reorganized memory content.", {
            required content: string("The full reorganized memory content as markdown."),
        });
    }
}

fn memory_tool_definition() -> AgentToolDefinition {
    AgentToolDefinition {
        name: "memory".to_string(),
        description: "Store durable project wisdom — knowledge that guides future action across unrelated tasks. Store only principles, invariants, and conventions that, if forgotten, would cause a wrong decision on an unrelated future task. Do NOT store implementation details (what a function does, how a struct is shaped), debugging findings, code structure descriptions, or task-specific observations. The current memory is shown as '## memory' in your context.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The new wisdom to add to memory, as markdown. State the principle, not the code reading."
                }
            },
            "required": ["content"],
            "additionalProperties": false
        }),
        operation: MEMORY_OPERATION.to_string(),
    }
}

/// Derive the memory file path from the working directory.
/// `~/.potlatch/memory/<hash>/memory.md` where `<hash>` is a stable hash of the
/// working directory path.
fn memory_file_path(working_dir: &str) -> PathBuf {
    let hash = hash_str(working_dir);
    memory_dir().join(&hash).join("memory.md")
}

fn hash_str(value: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_path_specific() {
        assert_eq!(
            hash_str("/home/user/project-a"),
            hash_str("/home/user/project-a")
        );
        assert_ne!(
            hash_str("/home/user/project-a"),
            hash_str("/home/user/project-b")
        );
    }

    #[test]
    fn memory_file_path_is_under_potlatch_memory_with_hash() {
        let path = memory_file_path("/home/user/project");
        assert!(path.starts_with(memory_dir()));
        assert!(path.ends_with("memory.md"));
    }

    #[test]
    fn save_and_read_memory_round_trips() {
        let dir = std::env::temp_dir().join(format!("potlatch-memory-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        save_memory(&path, "# Project Memory\n\nFact one.").unwrap();
        let read = fs::read_to_string(&path).unwrap();
        assert_eq!(read, "# Project Memory\n\nFact one.");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_memory_creates_parent_directories() {
        let dir = std::env::temp_dir().join(format!(
            "potlatch-memory-test-nested-{}",
            std::process::id()
        ));
        let path = dir.join("sub").join("memory.md");
        save_memory(&path, "content").unwrap();
        assert!(path.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn memory_tool_definition_has_content_parameter() {
        let def = memory_tool_definition();
        assert_eq!(def.name, "memory");
        let params = &def.parameters;
        assert_eq!(params["required"], serde_json::json!(["content"]));
        assert_eq!(params["properties"]["content"]["type"], "string");
    }

    #[test]
    fn handle_memory_request_appends_content() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-memory-test-req-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        // First write.
        handle_memory_request(
            &path,
            MEMORY_OPERATION,
            serde_json::json!({"content": "fact one"}),
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "fact one");
        // Second write appends.
        handle_memory_request(
            &path,
            MEMORY_OPERATION,
            serde_json::json!({"content": "fact two"}),
        )
        .unwrap();
        let combined = fs::read_to_string(&path).unwrap();
        assert!(combined.contains("fact one"));
        assert!(combined.contains("fact two"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn handle_memory_request_rejects_empty_content() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-memory-test-empty-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        let result = handle_memory_request(
            &path,
            MEMORY_OPERATION,
            serde_json::json!({"content": "   "}),
        );
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn handle_memory_request_rejects_oversized_content() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-memory-test-big-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        let big = "x".repeat(MAX_MEMORY_BYTES + 1);
        let result =
            handle_memory_request(&path, MEMORY_OPERATION, serde_json::json!({"content": big}));
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn handle_memory_request_rejects_unknown_operation() {
        let dir =
            std::env::temp_dir().join(format!("potlatch-memory-test-op-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        let result = handle_memory_request(&path, "unknown", serde_json::json!({}));
        assert!(result.is_err());
        fs::remove_dir_all(&dir).ok();
    }
}
