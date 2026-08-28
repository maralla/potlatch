//! Tool trait and registry.

pub mod agent_bus;
pub mod edit;
pub mod http;
pub mod lsp;
pub mod read;
pub mod search;
pub mod shell;
pub mod structured_output;
pub mod subagent;
pub mod todo;
pub mod write;

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

/// Resolve a path relative to the workspace (cwd), enforcing sandboxing.
///
/// Rejects:
/// - Absolute paths (e.g. `/etc/passwd`)
/// - Paths that escape the workspace via `..` traversal (e.g. `../../secret`)
///
/// Returns the resolved absolute path on success, or an error message string
/// explaining why the path was rejected.
pub fn resolve_workspace_path(path: &str, cwd: &str) -> Result<PathBuf, String> {
    if Path::new(path).is_absolute() {
        return Err(format!(
            "absolute paths are not allowed. Use a relative path within the workspace. For example, use 'src/main.rs' instead of '{path}'."
        ));
    }

    let cwd_path = Path::new(cwd);
    let resolved = cwd_path.join(path);

    // Canonicalize both the cwd and the resolved path to detect `..` escapes.
    // If canonicalization fails (file doesn't exist yet for write), use lexical normalization.
    let canonical_cwd = cwd_path
        .canonicalize()
        .unwrap_or_else(|_| cwd_path.to_path_buf());
    let canonical_resolved = resolved.canonicalize().or_else(|_| {
        // For non-existent paths (new files), canonicalize the parent and join
        if let Some(parent) = resolved.parent() {
            parent
                .canonicalize()
                .map(|p| p.join(resolved.file_name().unwrap_or_default()))
        } else {
            Ok(resolved.clone())
        }
    });

    match canonical_resolved {
        Ok(cr) => {
            if !cr.starts_with(&canonical_cwd) {
                return Err(format!(
                    "path '{path}' escapes the workspace. All file operations must be within the working directory."
                ));
            }
            Ok(cr)
        }
        Err(_) => {
            // If we can't canonicalize at all, fall back to lexical check
            let normalized = resolve_lexical(&resolved, cwd_path);
            if !normalized.starts_with(&canonical_cwd) {
                return Err(format!(
                    "path '{path}' escapes the workspace. All file operations must be within the working directory."
                ));
            }
            Ok(normalized)
        }
    }
}

/// Resolve a path for read-only access. Unlike `resolve_workspace_path`, this is
/// permissive: absolute paths and paths outside the working directory are allowed,
/// so the agent can read files explicitly referenced in the task context even when
/// they live elsewhere on the filesystem. Relative paths are resolved against `cwd`.
///
/// Returns the resolved absolute path (canonicalized when possible).
pub fn resolve_read_path(path: &str, cwd: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    let resolved = if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(cwd).join(path)
    };
    // Canonicalize if the target exists; otherwise return the joined path as-is
    // (the caller will surface a clear "not found" error on read).
    Ok(resolved.canonicalize().unwrap_or(resolved))
}

/// Resolve a write path, optionally allowing locations outside the workspace.
///
/// When `outside_cwd` is false (the default for all callers), this is exactly
/// [`resolve_workspace_path`]: absolute paths and `..` escapes are rejected, so
/// writes stay inside the working directory. When `outside_cwd` is true, the
/// workspace check is bypassed and absolute paths are accepted — used by agents
/// (e.g. QA) that manage their own scratch files in a dedicated session
/// directory outside the checked-out repo. This is no weaker than the `shell`
/// tool, which can already write anywhere via `bash -c`; the in-workspace guard
/// in `write` is a footgun guardrail, not a security boundary.
pub fn resolve_write_path(path: &str, cwd: &str, outside_cwd: bool) -> Result<PathBuf, String> {
    if outside_cwd {
        let p = Path::new(path);
        let resolved = if p.is_absolute() {
            p.to_path_buf()
        } else {
            Path::new(cwd).join(path)
        };
        Ok(resolved.canonicalize().unwrap_or(resolved))
    } else {
        resolve_workspace_path(path, cwd)
    }
}

/// Format `path` for display in tool output. When `path` is inside `cwd`,
/// returns the path relative to `cwd` (e.g. `taskapp/tasks/handler.go` instead of
/// `/home/user/project/taskapp/tasks/handler.go`). When `path` is outside `cwd`
/// (or `cwd` can't be resolved), returns the path as-is. This keeps tool output
/// short and workspace-relative for in-repo files while still showing full
/// paths for files the agent reads from outside the workspace.
pub fn display_path(path: &Path, cwd: &str) -> String {
    let cwd_path = Path::new(cwd);
    let cwd_canonical = cwd_path
        .canonicalize()
        .unwrap_or_else(|_| cwd_path.to_path_buf());
    if let Ok(rel) = path.strip_prefix(&cwd_canonical) {
        // Don't return an empty string for the cwd itself.
        let s = rel.to_string_lossy().to_string();
        if s.is_empty() { ".".to_string() } else { s }
    } else {
        path.to_string_lossy().to_string()
    }
}

/// Lexically normalize a path (resolve `.` and `..` without filesystem access).
fn resolve_lexical(path: &Path, base: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if result.components().next().is_some() {
                    result.pop();
                }
            }
            Component::RootDir => {
                result = PathBuf::from("/");
            }
            Component::Normal(c) => {
                result.push(c);
            }
            Component::Prefix(_) => {}
        }
    }
    if result.is_absolute() {
        result
    } else {
        base.join(result)
    }
}

/// A tool the agent can call. Implementations live in sibling modules.
pub trait Tool: Send + Sync {
    /// Tool name (matches the function name in the OpenAI tool schema).
    fn name(&self) -> &str;

    /// JSON schema for the tool's parameters (OpenAI function-calling format).
    /// The `description` field is used both for the API tool schema and for
    /// the system prompt's Tool Usage section.
    fn schema(&self) -> Value;

    /// Execute the tool with parsed arguments. Returns a string result for the model.
    fn execute(&self, args: &Value, cwd: &str) -> Result<String>;
}

/// Session-level state that needs cleanup on `session/close`. Tools that hold
/// long-lived resources (background processes, language servers, etc.) implement
/// this trait so the `Session` can shut them all down generically without
/// knowing their concrete types.
pub trait SessionState: Send + Sync {
    /// Release all resources (kill processes, close connections, etc.).
    /// Called once when the session is closed.
    fn shutdown(&self);
}

/// A type-keyed map of session state objects. Stores `Arc<T>` values keyed by
/// `TypeId`, so tools can retrieve their state by concrete type without the
/// session knowing about specific tools. All stored state is shut down in
/// reverse insertion order on [`Self::shutdown`].
pub struct SessionStates {
    /// `Arc<T>` boxed as `dyn Any` for typed retrieval via [`Self::get`].
    typed: HashMap<std::any::TypeId, Box<dyn Any + Send + Sync>>,
    /// `Arc<dyn SessionState>` for generic shutdown (insertion order).
    for_shutdown: Vec<Arc<dyn SessionState>>,
}

impl SessionStates {
    pub fn new() -> Self {
        Self {
            typed: HashMap::new(),
            for_shutdown: Vec::new(),
        }
    }

    /// Insert a typed state object. Stores the `Arc<T>` for retrieval and an
    /// `Arc<dyn SessionState>` for shutdown.
    pub fn insert<T: SessionState + 'static>(&mut self, state: Arc<T>) {
        let tid = std::any::TypeId::of::<T>();
        self.typed.insert(tid, Box::new(Arc::clone(&state)));
        self.for_shutdown.push(state);
    }

    /// Retrieve a typed `Arc<T>` by concrete type. Returns `None` if no state
    /// of type `T` was inserted.
    pub fn get<T: SessionState + 'static>(&self) -> Option<Arc<T>> {
        let tid = std::any::TypeId::of::<T>();
        self.typed
            .get(&tid)
            .and_then(|b| b.downcast_ref::<Arc<T>>())
            .cloned()
    }

    /// Shut down all stored state in reverse insertion order.
    pub fn shutdown(&self) {
        for state in self.for_shutdown.iter().rev() {
            state.shutdown();
        }
    }
}

impl Default for SessionStates {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry of available tools. Builds the OpenAI `tools` array and dispatches calls.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
    /// Side-channel cells for structured-output tools, keyed by tool name.
    /// Populated by [`Self::register_structured_output`]; read by
    /// [`Self::take_structured_outputs`].
    structured_outputs: HashMap<String, structured_output::StructuredOutputCell>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            order: Vec::new(),
            structured_outputs: HashMap::new(),
        }
    }

    /// Build a registry with all built-in tools. Session-scoped tools (shell,
    /// lsp, subagent) receive `&mut SessionStates` so they can create/retrieve
    /// their long-lived state. Stateless tools don't take it.
    ///
    /// `model` is the session model, forwarded to the subagent tool so spawned
    /// subagents default to the parent's model.
    /// `session_id` groups subagent transcripts under the parent session.
    ///
    /// `allowed_tools` filters which tools are registered: `None` registers
    /// all; `Some(names)` registers only tools whose name is in the list.
    pub fn with_builtin_tools(
        states: &mut SessionStates,
        session_id: &str,
        cwd: &str,
        model: &str,
        allowed_tools: Option<&[String]>,
    ) -> Self {
        let mut reg = Self::new();
        let allowed = |name: &str| -> bool {
            allowed_tools
                .map(|names| names.iter().any(|n| n == name))
                .unwrap_or(true)
        };

        if allowed("shell") {
            reg.register(Arc::new(shell::ShellTool::new(states, cwd)));
        }
        if allowed("read") {
            reg.register(Arc::new(read::ReadTool));
        }
        if allowed("edit") {
            reg.register(Arc::new(edit::EditTool));
        }
        if allowed("write") {
            reg.register(Arc::new(write::WriteTool));
        }
        if allowed("grep") {
            reg.register(Arc::new(search::GrepTool));
        }
        if allowed("glob") {
            reg.register(Arc::new(search::GlobTool));
        }
        if allowed("http") {
            reg.register(Arc::new(http::HttpTool::new()));
        }
        if allowed("lsp") {
            reg.register(Arc::new(lsp::LspTool::new(states, cwd)));
        }
        if allowed("subagent") {
            reg.register(Arc::new(subagent::SubagentTool::new(
                states, session_id, cwd, model,
            )));
        }
        reg
    }

    pub fn register_agent_tools(
        &mut self,
        caller: Arc<dyn agent_bus::AgentToolCaller>,
        definitions: Vec<crate::core::bus::RemoteAgentToolDefinition>,
        allowed_tools: Option<&[String]>,
    ) {
        let allowed = |name: &str| {
            allowed_tools
                .map(|names| names.iter().any(|allowed| allowed == name))
                .unwrap_or(true)
        };
        for definition in definitions {
            if !allowed(&definition.name) || self.tools.contains_key(&definition.name) {
                continue;
            }
            self.register(Arc::new(agent_bus::RemoteAgentTool::new(
                Arc::clone(&caller),
                definition,
            )));
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    /// Register a structured-output tool: a side-channel cell tool whose name,
    /// description, and parameter schema are caller-defined. The harness
    /// captures the model's call and returns it in the `session/prompt`
    /// response via [`Self::take_structured_outputs`].
    pub fn register_structured_output(&mut self, name: &str, description: &str, parameters: Value) {
        let (tool, cell) =
            structured_output::StructuredOutputTool::new(name.to_string(), description, parameters);
        self.structured_outputs.insert(name.to_string(), cell);
        self.register(Arc::new(tool));
    }

    /// Take all captured structured-output JSONs, clearing the cells. Returns
    /// a JSON object mapping tool name to captured args. Tools that were never
    /// called are omitted from the object.
    pub fn take_structured_outputs(&self) -> Value {
        let mut map = serde_json::Map::new();
        for (name, cell) in &self.structured_outputs {
            if let Some(v) = cell.lock().unwrap().take() {
                map.insert(name.clone(), v);
            }
        }
        Value::Object(map)
    }

    /// Registered tool names in insertion order.
    pub fn tool_names(&self) -> Vec<&str> {
        self.order.iter().map(|n| n.as_str()).collect()
    }

    /// OpenAI `tools` array for the chat completion request.
    pub fn tools_schema(&self) -> Vec<Value> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.schema()["description"],
                        "parameters": tool.schema()["parameters"],
                    }
                })
            })
            .collect()
    }

    /// Execute a tool call by name. Returns the result string or an error message.
    /// Common aliases (`bash` → `shell`) are resolved to the registered tool.
    pub fn execute(&self, name: &str, args: &Value, cwd: &str) -> Result<String> {
        let resolved = match name {
            "bash" => "shell",
            other => other,
        };
        match self.tools.get(resolved) {
            Some(tool) => tool.execute(args, cwd),
            None => anyhow::bail!("unknown tool: {name}"),
        }
    }

    /// Collect `(name, description)` pairs for the system prompt, in
    /// registration order. Uses each tool's schema description.
    pub fn tool_descriptions(&self) -> Vec<(String, String)> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .filter_map(|tool| {
                let schema = tool.schema();
                let desc = schema["description"].as_str().unwrap_or("");
                if desc.is_empty() {
                    None
                } else {
                    Some((tool.name().to_string(), desc.to_string()))
                }
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create a unique temp directory for a test, so parallel tests don't collide.
    /// The directory is removed at the end via the returned `TestDir` guard's `Drop`.
    pub fn unique_test_dir() -> TestDir {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("potlatch_test_{pid}_{id}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    /// Owning guard for a test directory. Derefs to the `PathBuf`; cleans up on drop.
    pub struct TestDir(PathBuf);

    impl TestDir {
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }

        pub fn as_str(&self) -> &str {
            self.0.to_str().expect("temp dir is valid utf-8")
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_absolute_path() {
        let dir = test_util::unique_test_dir();
        let result = resolve_workspace_path("/etc/passwd", dir.as_str());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("absolute"));
    }

    #[test]
    fn rejects_dotdot_escape() {
        let dir = test_util::unique_test_dir();
        let result = resolve_workspace_path("../../etc/passwd", dir.as_str());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("escapes"));
    }

    #[test]
    fn allows_relative_path() {
        let dir = test_util::unique_test_dir();
        let result = resolve_workspace_path("src/main.rs", dir.as_str());
        assert!(result.is_ok());
    }

    #[test]
    fn allows_nested_relative_path() {
        let dir = test_util::unique_test_dir();
        let result = resolve_workspace_path("src/deep/nested/file.rs", dir.as_str());
        assert!(result.is_ok());
    }

    #[test]
    fn allows_dot_path() {
        let dir = test_util::unique_test_dir();
        let result = resolve_workspace_path(".", dir.as_str());
        assert!(result.is_ok());
    }

    #[test]
    fn rejects_dotdot_within_subdirectory_is_ok() {
        // a/b/../c should resolve to a/c, which is still inside the workspace
        let dir = test_util::unique_test_dir();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::create_dir_all(dir.path().join("a/c")).unwrap();
        let result = resolve_workspace_path("a/b/../c/file.rs", dir.as_str());
        assert!(result.is_ok());
    }

    #[test]
    fn read_path_allows_absolute_path() {
        let dir = test_util::unique_test_dir();
        // /tmp itself exists and is outside the workspace; reads must allow it.
        let result = resolve_read_path("/tmp", dir.as_str());
        assert!(result.is_ok());
        assert!(result.unwrap().is_absolute());
    }

    #[test]
    fn read_path_allows_escape_outside_workspace() {
        let dir = test_util::unique_test_dir();
        // A `..` traversal that lands outside the workspace is allowed for reads.
        let result = resolve_read_path("../../", dir.as_str());
        assert!(result.is_ok());
    }

    #[test]
    fn read_path_resolves_relative_against_cwd() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("inside.txt"), "x").unwrap();
        let result = resolve_read_path("inside.txt", dir.as_str());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dir.path().join("inside.txt"));
    }

    #[test]
    fn display_path_strips_cwd_prefix() {
        let dir = test_util::unique_test_dir();
        std::fs::create_dir_all(dir.path().join("taskapp/tasks")).unwrap();
        let file = dir.path().join("taskapp/tasks/handler.go");
        std::fs::write(&file, "x").unwrap();
        assert_eq!(display_path(&file, dir.as_str()), "taskapp/tasks/handler.go");
    }

    #[test]
    fn display_path_returns_dot_for_cwd_itself() {
        let dir = test_util::unique_test_dir();
        assert_eq!(display_path(dir.path(), dir.as_str()), ".");
    }

    #[test]
    fn display_path_keeps_absolute_for_external_paths() {
        let dir = test_util::unique_test_dir();
        let external = test_util::unique_test_dir();
        std::fs::write(external.path().join("outside.txt"), "x").unwrap();
        let file = external.path().join("outside.txt");
        // Path outside cwd is returned as-is (absolute).
        assert_eq!(display_path(&file, dir.as_str()), file.to_string_lossy());
    }
}
