//! Tool trait and registry.

pub mod agent_bus;
pub mod edit;
pub mod http;
pub mod lsp;
pub mod read;
pub mod search;
pub mod shell;
pub mod structured_output;
pub mod todo;
pub mod write;

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

use crate::core::bus::RemoteAgentToolDefinition;
use crate::core::scope::SessionScope;

/// Resolve a path for read-only access (the `read`, `grep`, and `glob`
/// tools) within the session's file scope ([`SessionScope`]: the working
/// directory plus the roots the calling agent declared). Relative paths
/// resolve against the scope's cwd; absolute paths are accepted when they
/// land inside the scope, and anything else is rejected — there is no
/// separate working-directory rule to satisfy first.
///
/// Returns the resolved absolute path (canonicalized when the target — or
/// its parent — exists, so a path inside the scope that does not exist yet
/// still resolves and the caller surfaces a clear "not found" error on read).
pub fn resolve_read_path(path: &str, scope: &SessionScope) -> Result<PathBuf, String> {
    let resolved = resolve_in_scope(path, scope);
    if !scope.permits(&resolved) {
        return Err(format!(
            "path '{path}' is outside the session's readable directories. \\
             Read and search paths must be within the working directory or an \\
             agent-managed session directory named by the task."
        ));
    }
    Ok(resolved)
}

/// Resolve a write path (the `write` and `edit` tools) within the session's
/// file scope ([`SessionScope`]): the working directory plus the roots the
/// calling agent declared — used by agents (e.g. QA) that manage their own
/// scratch files in a dedicated session directory outside the checked-out
/// repo. Anything else is rejected. This is no weaker than the `shell` tool,
/// which can already write anywhere via `bash -c`; the scope guard in
/// `write` is a footgun guardrail, not a security boundary.
pub fn resolve_write_path(path: &str, scope: &SessionScope) -> Result<PathBuf, String> {
    let resolved = resolve_in_scope(path, scope);
    if !scope.permits(&resolved) {
        return Err(format!(
            "path '{path}' is outside the session's writable directories. \\
             Writes must target the working directory or an agent-managed \\
             session directory named by the task."
        ));
    }
    Ok(resolved)
}

/// Resolve `path` for a file-tool operation against the session's cwd:
/// relative paths join the cwd, absolute paths are kept as-is, and the
/// result is canonicalized when the target — or its parent — exists (so a
/// `..` traversal through an existing parent is visible to the scope check,
/// and a not-yet-existing file inside the scope still resolves).
fn resolve_in_scope(path: &str, scope: &SessionScope) -> PathBuf {
    let p = Path::new(path);
    let resolved = if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(scope.cwd()).join(path)
    };
    canonicalize_flexible(&resolved)
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

/// Canonicalize a path for scope checks: canonicalize the path itself when
/// it exists, else canonicalize its parent and rejoin the final component
/// (a file that does not exist yet), else return the path as-is. Keeps a
/// `..` traversal through an existing parent visible to the `starts_with`
/// scope check instead of leaving it lexical.
fn canonicalize_flexible(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    match path.file_name() {
        Some(name) => path
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .map(|parent| parent.join(name))
            .unwrap_or_else(|| path.to_path_buf()),
        None => path.to_path_buf(),
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

    /// Execute the tool with parsed arguments, against the session's file
    /// scope (the agent-provided cwd + declared roots). Returns a string
    /// result for the model.
    fn execute(&self, args: &Value, scope: &SessionScope) -> Result<String>;
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
    /// lsp) receive `&mut SessionStates` so they can create/retrieve
    /// their long-lived state. Stateless tools don't take it.
    ///
    /// `scope` is the session's file scope (the agent-provided cwd plus its
    /// declared roots): every file tool (`read`, `grep`, `glob`, `write`,
    /// `edit`) confines paths to it — the same directory set the sandbox
    /// agent shares into the session's VM.
    ///
    /// `allowed_tools` filters which tools are registered: `None` registers
    /// all; `Some(names)` registers only tools whose name is in the list.
    /// Cross-agent tools (like `subagent`) are not built-in — they arrive as
    /// bus-registered proxy tools via [`Self::register_agent_tools`].
    pub fn with_builtin_tools(
        states: &mut SessionStates,
        scope: &SessionScope,
        allowed_tools: Option<&[String]>,
    ) -> Self {
        let mut reg = Self::new();
        let allowed = |name: &str| -> bool {
            allowed_tools
                .map(|names| names.iter().any(|n| n == name))
                .unwrap_or(true)
        };

        if allowed("shell") {
            reg.register(Arc::new(shell::ShellTool::new(states)));
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
            reg.register(Arc::new(lsp::LspTool::new(states, scope.cwd())));
        }
        reg
    }

    pub fn register_agent_tools(
        &mut self,
        caller: Arc<dyn agent_bus::AgentToolCaller>,
        definitions: Vec<RemoteAgentToolDefinition>,
        allowed_tools: Option<&[String]>,
        session_id: &str,
    ) {
        let allowed = |name: &str| {
            allowed_tools
                .map(|names| names.iter().any(|allowed| allowed == name))
                .unwrap_or(true)
        };
        for definition in definitions {
            if !allowed(&definition.name) {
                continue;
            }
            // Bus tools OVERRIDE same-named built-ins: the sandal agent's
            // `shell` replaces the built-in host shell while `[agent.sandal]`
            // is configured, and nothing else in the fleet publishes a
            // built-in name. A later definition of the same name wins
            // (last registration wins); nothing registers twice today.
            self.register(Arc::new(agent_bus::RemoteAgentTool::new(
                Arc::clone(&caller),
                definition,
                session_id,
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
    /// response via [`Self::take_structured_outputs`]. A terminal contract's
    /// successful record is the run's deliverable: the result tells the
    /// model the task is complete so it ends its turn.
    pub fn register_structured_output(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        terminal: bool,
    ) {
        let (tool, cell) = structured_output::StructuredOutputTool::with_terminal(
            name.to_string(),
            description,
            parameters,
            terminal,
        );
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
    pub fn execute(&self, name: &str, args: &Value, scope: &SessionScope) -> Result<String> {
        let resolved = match name {
            "bash" => "shell",
            other => other,
        };
        match self.tools.get(resolved) {
            Some(tool) => tool.execute(args, scope),
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

        /// A session scope rooted at this directory, with no extra roots.
        pub fn scope(&self) -> super::SessionScope {
            super::SessionScope::new(self.as_str(), std::iter::empty::<PathBuf>())
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
    fn read_path_allows_absolute_path_inside_the_cwd() {
        let dir = test_util::unique_test_dir();
        let result = resolve_read_path(dir.as_str(), &dir.scope());
        assert!(result.is_ok());
    }

    #[test]
    fn read_path_rejects_absolute_path_outside_the_scope() {
        let dir = test_util::unique_test_dir();
        // /tmp itself exists and is outside the scope (no roots declared).
        let result = resolve_read_path("/tmp", &dir.scope());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside"));
    }

    #[test]
    fn read_path_allows_absolute_path_inside_a_declared_root() {
        let dir = test_util::unique_test_dir();
        let root = test_util::unique_test_dir();
        let scope = SessionScope::new(dir.as_str(), [root.path().to_path_buf()]);
        let target = root.path().join("session-state.json");
        let result = resolve_read_path(target.to_str().unwrap(), &scope);
        assert!(result.is_ok());
    }

    #[test]
    fn read_path_rejects_escape_outside_workspace() {
        let dir = test_util::unique_test_dir();
        // A `..` traversal that lands outside the workspace is rejected.
        let result = resolve_read_path("../../", &dir.scope());
        assert!(result.is_err());
    }

    #[test]
    fn read_path_resolves_relative_against_cwd() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("inside.txt"), "x").unwrap();
        let result = resolve_read_path("inside.txt", &dir.scope());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dir.path().join("inside.txt"));
    }

    #[test]
    fn write_path_rejects_parent_traversal_through_an_existing_parent() {
        // `a/../..` resolves through the existing directory `a` and lands
        // outside the scope — rejected, not lexically smuggled through.
        let dir = test_util::unique_test_dir();
        std::fs::create_dir_all(dir.path().join("a")).unwrap();
        let result = resolve_write_path("a/../../escape.txt", &dir.scope());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside"));
    }

    #[test]
    fn write_path_allows_a_new_file_in_a_new_directory_inside_the_scope() {
        // The write tool creates parent directories: a path whose own
        // directory does not exist yet has nothing to canonicalize through,
        // and the scope check sees a plain in-scope path.
        let dir = test_util::unique_test_dir();
        let result = resolve_write_path("newdir/deep/file.txt", &dir.scope());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dir.path().join("newdir/deep/file.txt"));
    }

    #[test]
    fn display_path_strips_cwd_prefix() {
        let dir = test_util::unique_test_dir();
        std::fs::create_dir_all(dir.path().join("taskapp/tasks")).unwrap();
        let file = dir.path().join("taskapp/tasks/handler.go");
        std::fs::write(&file, "x").unwrap();
        assert_eq!(
            display_path(&file, dir.as_str()),
            "taskapp/tasks/handler.go"
        );
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
