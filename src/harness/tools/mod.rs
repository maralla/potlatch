//! Tool trait and registry.

pub mod file_edit;
pub mod file_read;
pub mod file_write;
pub mod memory;
pub mod plan_tool;
pub mod search;
pub mod shell;
pub mod todo;
pub mod web_fetch;

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
    // If canonicalization fails (file doesn't exist yet for file_write), use lexical normalization.
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
    fn schema(&self) -> Value;

    /// Execute the tool with parsed arguments. Returns a string result for the model.
    fn execute(&self, args: &Value, cwd: &str) -> Result<String>;
}

/// Registry of available tools. Builds the OpenAI `tools` array and dispatches calls.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Build a registry with all built-in tools.
    pub fn with_builtin_tools() -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(shell::ShellTool::new()));
        reg.register(Arc::new(file_read::FileReadTool));
        reg.register(Arc::new(file_edit::FileEditTool));
        reg.register(Arc::new(file_write::FileWriteTool));
        reg.register(Arc::new(search::GrepTool));
        reg.register(Arc::new(search::GlobTool));
        reg.register(Arc::new(web_fetch::WebFetchTool::new()));
        reg
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    /// Unregister a tool by name. No-op when the tool isn't registered.
    /// Used by mode-specific gating (e.g. plan mode drops `file_edit` so the
    /// model can't mutate files while triaging).
    pub fn unregister(&mut self, name: &str) {
        if self.tools.remove(name).is_some() {
            self.order.retain(|n| n != name);
        }
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
    pub fn execute(&self, name: &str, args: &Value, cwd: &str) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(args, cwd),
            None => anyhow::bail!("unknown tool: {name}"),
        }
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

    #[test]
    fn unregister_removes_tool_from_registry() {
        let mut reg = ToolRegistry::with_builtin_tools();
        assert!(reg.tool_names().contains(&"file_edit"));
        reg.unregister("file_edit");
        assert!(!reg.tool_names().contains(&"file_edit"));
        // Schemas reflect the removal too.
        let schemas = reg.tools_schema();
        let names: Vec<String> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap_or("").to_string())
            .collect();
        assert!(!names.contains(&"file_edit".to_string()));
        // Execution now fails with "unknown tool".
        let res = reg.execute("file_edit", &json!({}), "/tmp");
        assert!(res.is_err());
    }

    #[test]
    fn unregister_is_noop_for_unknown_tool() {
        let mut reg = ToolRegistry::with_builtin_tools();
        let before: Vec<String> = reg.tool_names().iter().map(|s| s.to_string()).collect();
        reg.unregister("nonexistent");
        let after: Vec<String> = reg.tool_names().iter().map(|s| s.to_string()).collect();
        assert_eq!(after, before);
    }

    #[test]
    fn unregister_preserves_order_of_remaining_tools() {
        let mut reg = ToolRegistry::with_builtin_tools();
        reg.unregister("file_edit");
        let names = reg.tool_names();
        // file_edit was in the middle; the rest keep their relative order.
        let pos_shell = names.iter().position(|n| *n == "shell").unwrap();
        let pos_read = names.iter().position(|n| *n == "file_read").unwrap();
        let pos_write = names.iter().position(|n| *n == "file_write").unwrap();
        let pos_grep = names.iter().position(|n| *n == "grep").unwrap();
        assert!(pos_shell < pos_read);
        assert!(pos_read < pos_write);
        assert!(pos_write < pos_grep);
    }
}
