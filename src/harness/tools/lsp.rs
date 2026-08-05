//! Generic LSP tool — model-supplied language server queries over stdio.
//!
//! A single `lsp` tool is always registered. The model discovers which
//! language server to use (e.g. `which gopls`, `which rust-analyzer`) and
//! passes the `server` command and `language` ID as parameters. The harness
//! spawns the server lazily on first use and caches it for the session via
//! `LspClientCache` (stored on the `Session` struct, mirroring `JobTable`).
//!
//! Only read-only LSP methods are exposed (definition, references, hover,
//! symbols, diagnostics). Mutating operations are intentionally absent — the
//! agent uses `edit`/`write` for code changes.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use super::Tool;

// ---------------------------------------------------------------------------
// LspClient — raw JSON-RPC over stdio to a language server subprocess
// ---------------------------------------------------------------------------

struct LspClient {
    stdin: ChildStdin,
    _child: Child,
    next_id: AtomicI64,
    /// Pending request senders keyed by JSON-RPC `id`.
    pending: Arc<Mutex<HashMap<i64, SyncSender<Value>>>>,
    /// Cached `textDocument/publishDiagnostics` keyed by file URI.
    diagnostics: Arc<Mutex<HashMap<String, Vec<Value>>>>,
}

impl LspClient {
    /// Spawn a language server subprocess and start the reader thread.
    fn spawn(program: &str, args: &[&str], cwd: &str, display_name: &str) -> Result<Self> {
        let mut cmd = Command::new(program);
        for arg in args {
            cmd.arg(arg);
        }
        cmd.current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn `{program}` for {display_name}"))?;

        let stdin = child
            .stdin
            .take()
            .with_context(|| format!("{display_name} stdin not captured"))?;
        let stdout = child
            .stdout
            .take()
            .with_context(|| format!("{display_name} stdout not captured"))?;

        let pending: Arc<Mutex<HashMap<i64, SyncSender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let diagnostics: Arc<Mutex<HashMap<String, Vec<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Reader thread: drains server stdout, dispatches responses to pending
        // request senders, and caches publishDiagnostics notifications.
        {
            let pending = Arc::clone(&pending);
            let diagnostics = Arc::clone(&diagnostics);
            std::thread::spawn(move || {
                reader_loop(stdout, pending, diagnostics);
            });
        }

        Ok(Self {
            stdin,
            _child: child,
            next_id: AtomicI64::new(1),
            pending,
            diagnostics,
        })
    }

    /// Send a JSON-RPC request and wait for the response (blocking).
    fn request(&self, method: &str, params: Value, display_name: &str) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let (tx, rx) = mpsc::sync_channel::<Value>(1);
        self.pending.lock().unwrap().insert(id, tx);
        self.send_raw(&msg)?;
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(resp) => {
                if let Some(err) = resp.get("error")
                    && !err.is_null()
                {
                    bail!("{display_name} error ({method}): {err}");
                }
                Ok(resp.get("result").cloned().unwrap_or(Value::Null))
            }
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                bail!("{display_name} request '{method}' timed out (60s)");
            }
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    fn notify(&self, method: &str, params: Value) -> Result<()> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_raw(&msg)
    }

    fn send_raw(&self, msg: &Value) -> Result<()> {
        let body = serde_json::to_vec(msg).context("serialize JSON-RPC message")?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let mut stdin = &self.stdin;
        stdin
            .write_all(header.as_bytes())
            .context("write LSP header")?;
        stdin.write_all(&body).context("write LSP body")?;
        stdin.flush().context("flush language server stdin")?;
        Ok(())
    }

    /// LSP initialize + initialized handshake.
    fn initialize(&self, root_uri: &str, display_name: &str) -> Result<()> {
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "definition": {},
                    "references": {},
                    "hover": {},
                    "documentSymbol": {},
                    "signatureHelp": {},
                    "publishDiagnostics": {}
                },
                "workspace": {
                    "symbol": {}
                }
            }
        });
        self.request("initialize", params, display_name)
            .with_context(|| format!("{display_name} initialize failed"))?;
        self.notify("initialized", json!({}))?;
        Ok(())
    }

    /// didOpen a file with current disk content.
    fn did_open(&self, abs_path: &Path, language_id: &str) -> Result<()> {
        let uri = path_to_uri(abs_path);
        let text = std::fs::read_to_string(abs_path)
            .with_context(|| format!("read file for didOpen: {}", abs_path.display()))?;
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            }),
        )
    }

    /// didClose a file.
    fn did_close(&self, abs_path: &Path) -> Result<()> {
        let uri = path_to_uri(abs_path);
        self.notify(
            "textDocument/didClose",
            json!({
                "textDocument": { "uri": uri }
            }),
        )
    }

    /// Get cached diagnostics for a file URI (empty if none).
    fn get_diagnostics(&self, uri: &str) -> Vec<Value> {
        self.diagnostics
            .lock()
            .unwrap()
            .get(uri)
            .cloned()
            .unwrap_or_default()
    }

    /// Snapshot the full diagnostics cache (for the "all diagnostics" query).
    fn diagnostics_snapshot(&self) -> Vec<(String, Vec<Value>)> {
        self.diagnostics
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.request("shutdown", Value::Null, "LSP server");
        let _ = self.notify("exit", Value::Null);
        let _ = self._child.kill();
    }
}

/// Reader thread: parses LSP Content-Length framing from server stdout,
/// dispatches responses to pending request senders, and caches diagnostics.
fn reader_loop(
    stdout: ChildStdout,
    pending: Arc<Mutex<HashMap<i64, SyncSender<Value>>>>,
    diagnostics: Arc<Mutex<HashMap<String, Vec<Value>>>>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        // Read headers until blank line.
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => return, // EOF — server exited
                Ok(_) => {}
                Err(_) => return,
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(val) = trimmed.strip_prefix("Content-Length:") {
                content_length = val.trim().parse::<usize>().ok();
            }
        }
        let Some(len) = content_length else {
            continue;
        };
        // Read exactly `len` bytes of body.
        let mut buf = vec![0u8; len];
        if reader.read_exact(&mut buf).is_err() {
            return;
        }
        let Ok(msg): std::result::Result<Value, _> = serde_json::from_slice(&buf) else {
            continue;
        };

        // Response to a request (has `id`).
        if let Some(id_val) = msg.get("id")
            && let Some(id) = id_val.as_i64()
            && let Some(tx) = pending.lock().unwrap().remove(&id)
        {
            let _ = tx.send(msg);
            continue;
        }

        // Notification (has `method`, no `id`).
        if let Some(method) = msg.get("method").and_then(Value::as_str)
            && method == "textDocument/publishDiagnostics"
            && let Some(uri) = msg["params"]["uri"].as_str()
        {
            let diags = msg["params"]["diagnostics"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            diagnostics.lock().unwrap().insert(uri.to_string(), diags);
        }
    }
}

// ---------------------------------------------------------------------------
// LspClientCache — session-scoped cache of spawned language servers
// ---------------------------------------------------------------------------

/// Session-scoped cache of LSP clients, keyed by the raw server command
/// string (e.g. `"gopls -stdio"`). Stored on the `Session` struct so servers
/// survive across prompts and are properly shut down on `session/close`.
pub struct LspClientCache {
    clients: Mutex<HashMap<String, Arc<Mutex<LspClient>>>>,
    cwd: String,
}

impl LspClientCache {
    pub fn new(cwd: &str) -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
            cwd: cwd.to_string(),
        }
    }

    /// Get an existing client or spawn a new one for the given server command.
    /// The server string is split on whitespace into program + args.
    fn get_or_spawn(&self, server: &str) -> Result<Arc<Mutex<LspClient>>> {
        let mut clients = self.clients.lock().unwrap();
        if let Some(client) = clients.get(server) {
            return Ok(Arc::clone(client));
        }

        let parts: Vec<&str> = server.split_whitespace().collect();
        let program = parts
            .first()
            .copied()
            .ok_or_else(|| anyhow!("empty server command"))?;
        let args = &parts[1..];
        let display_name = program;

        let client = LspClient::spawn(program, args, &self.cwd, display_name)?;

        let root_uri = Path::new(&self.cwd)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(&self.cwd))
            .to_string_lossy()
            .to_string();
        let root_uri = format!("file://{root_uri}");
        client.initialize(&root_uri, display_name)?;

        let client = Arc::new(Mutex::new(client));
        clients.insert(server.to_string(), Arc::clone(&client));
        Ok(client)
    }

    /// Shut down all cached language servers. Called on `session/close`.
    /// Dropping each `LspClient` sends `shutdown` + `exit` + kills the process.
    pub fn shutdown_all(&self) {
        self.clients.lock().unwrap().clear();
    }
}

impl super::SessionState for LspClientCache {
    fn shutdown(&self) {
        self.shutdown_all();
    }
}

// ---------------------------------------------------------------------------
// LspTool — always-registered tool with operation + server + language params
// ---------------------------------------------------------------------------

pub struct LspTool {
    cache: Arc<LspClientCache>,
}

impl LspTool {
    /// Construct with session-scoped state. Creates an `LspClientCache` in
    /// `SessionStates` if not already present (first prompt), then retrieves
    /// it so language servers survive across prompts and shut down on close.
    pub fn new(states: &mut super::SessionStates, cwd: &str) -> Self {
        if states.get::<LspClientCache>().is_none() {
            states.insert(Arc::new(LspClientCache::new(cwd)));
        }
        Self {
            cache: states.get::<LspClientCache>().unwrap(),
        }
    }
}

impl Tool for LspTool {
    fn name(&self) -> &str {
        "lsp"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Language server protocol queries — go-to-definition, find-references, hover, workspace symbols, and diagnostics. More precise and faster than grep for navigating code: use it to find where a symbol is defined, who calls it, its type signature, or compile errors. First, discover the language server for your project: run `which gopls` (Go), `which rust-analyzer` (Rust), `which clangd` (C/C++), `which typescript-language-server` (TypeScript) etc. Then pass the server command and language. The server is spawned once and reused for the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["definition", "references", "hover", "symbols", "diagnostics"],
                        "description": "LSP query to run"
                    },
                    "server": {
                        "type": "string",
                        "description": "Command to spawn the language server over stdio, e.g. 'gopls -stdio', 'rust-analyzer', 'clangd'. Discover it with `which <binary>` or `shell` before first use."
                    },
                    "language": {
                        "type": "string",
                        "description": "LSP languageId for the file, e.g. 'go', 'rust', 'c', 'cpp', 'typescript', 'python'."
                    },
                    "path": {
                        "type": "string",
                        "description": "File path (relative to working directory). Required for definition/references/hover/diagnostics."
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number. Required for definition/references/hover."
                    },
                    "character": {
                        "type": "integer",
                        "description": "0-based character offset on the line. Required for definition/references/hover."
                    },
                    "query": {
                        "type": "string",
                        "description": "Symbol search query. Required for symbols."
                    }
                },
                "required": ["operation", "server", "language"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let operation = args["operation"]
            .as_str()
            .ok_or_else(|| anyhow!("missing 'operation' parameter"))?;
        let server = args["server"]
            .as_str()
            .ok_or_else(|| anyhow!("missing 'server' parameter (e.g. 'gopls -stdio')"))?;
        let language = args["language"]
            .as_str()
            .ok_or_else(|| anyhow!("missing 'language' parameter (e.g. 'go')"))?;

        let client = self.cache.get_or_spawn(server)?;
        let client = client.lock().unwrap();
        let display_name = server;

        match operation {
            "definition" => exec_position_query(
                &client,
                cwd,
                args,
                language,
                "textDocument/definition",
                display_name,
                |locs| format_locations(locs, cwd),
            ),
            "references" => exec_position_query(
                &client,
                cwd,
                args,
                language,
                "textDocument/references",
                display_name,
                |locs| format_locations(locs, cwd),
            ),
            "hover" => exec_position_query(
                &client,
                cwd,
                args,
                language,
                "textDocument/hover",
                display_name,
                format_hover,
            ),
            "symbols" => exec_symbols(&client, cwd, args, display_name),
            "diagnostics" => exec_diagnostics(&client, cwd, args),
            other => bail!("unknown operation: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Operation implementations
// ---------------------------------------------------------------------------

/// Execute a position-based LSP query (definition, references, hover).
fn exec_position_query(
    client: &LspClient,
    cwd: &str,
    args: &Value,
    language: &str,
    method: &str,
    display_name: &str,
    formatter: impl Fn(&Value) -> String,
) -> Result<String> {
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow!("'{method}' requires a 'path'"))?;
    let line = args["line"]
        .as_u64()
        .ok_or_else(|| anyhow!("'{method}' requires a 'line' (1-based)"))?;
    let character = args["character"]
        .as_u64()
        .ok_or_else(|| anyhow!("'{method}' requires a 'character' (0-based)"))?;

    let abs_path = resolve_path(path, cwd)?;
    client.did_open(&abs_path, language)?;

    let uri = path_to_uri(&abs_path);
    let params = json!({
        "textDocument": { "uri": uri },
        "position": {
            "line": line - 1, // LSP uses 0-based lines
            "character": character,
        }
    });

    // references needs context.includeDeclaration
    let params = if method == "textDocument/references" {
        let mut p = params;
        p["context"] = json!({ "includeDeclaration": true });
        p
    } else {
        params
    };

    let result = client.request(method, params, display_name);
    client.did_close(&abs_path)?;

    let result = result?;
    let output = formatter(&result);
    if output.is_empty() {
        Ok("No results.".to_string())
    } else {
        Ok(output)
    }
}

/// Execute workspace/symbol query.
fn exec_symbols(client: &LspClient, cwd: &str, args: &Value, display_name: &str) -> Result<String> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow!("'symbols' requires a 'query'"))?;

    let result = client.request("workspace/symbol", json!({ "query": query }), display_name)?;

    let symbols = result.as_array().cloned().unwrap_or_default();
    if symbols.is_empty() {
        return Ok("No symbols found.".to_string());
    }

    let mut out = String::new();
    for sym in symbols.iter().take(50) {
        let name = sym["name"].as_str().unwrap_or("");
        let kind = symbol_kind_label(sym["kind"].as_u64().unwrap_or(0));
        let container = sym["containerName"].as_str().unwrap_or("");
        let loc = &sym["location"];
        let uri = loc["uri"].as_str().unwrap_or("");
        let line = loc["range"]["start"]["line"].as_u64().unwrap_or(0);
        let path = uri_to_path(uri, cwd);
        if container.is_empty() {
            out.push_str(&format!("{kind} {name} — {path}:{}\n", line + 1));
        } else {
            out.push_str(&format!(
                "{kind} {container}.{name} — {path}:{}\n",
                line + 1
            ));
        }
    }
    Ok(out)
}

/// Execute diagnostics query (reads cached publishDiagnostics).
fn exec_diagnostics(client: &LspClient, cwd: &str, args: &Value) -> Result<String> {
    if let Some(path) = args["path"].as_str() {
        let abs_path = resolve_path(path, cwd)?;
        let uri = path_to_uri(&abs_path);
        let diags = client.get_diagnostics(&uri);
        if diags.is_empty() {
            return Ok("No diagnostics for this file.".to_string());
        }
        Ok(format_diagnostics(&diags))
    } else {
        // All files — dump the full diagnostics cache.
        let all = client.diagnostics_snapshot();
        let mut out = String::new();
        let mut total = 0usize;
        for (uri, diags) in &all {
            if diags.is_empty() {
                continue;
            }
            let path = uri_to_path(uri, cwd);
            out.push_str(&format!("--- {path} ---\n"));
            out.push_str(&format_diagnostics(diags));
            total += diags.len();
        }
        if out.is_empty() {
            Ok("No diagnostics.".to_string())
        } else {
            Ok(format!("{total} diagnostic(s):\n{out}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format LSP Location/Location[] result as `path:line` lines.
fn format_locations(result: &Value, cwd: &str) -> String {
    let locations = match result {
        Value::Null => return String::new(),
        Value::Array(arr) => arr.clone(),
        other => vec![other.clone()],
    };
    let mut out = String::new();
    for loc in &locations {
        // LocationLink has `targetUri`; plain Location has `uri`.
        let uri = loc["uri"]
            .as_str()
            .or_else(|| loc["targetUri"].as_str())
            .unwrap_or("");
        let line = loc["range"]["start"]["line"]
            .as_u64()
            .or_else(|| loc["targetSelectionRange"]["start"]["line"].as_u64())
            .unwrap_or(0);
        let path = uri_to_path(uri, cwd);
        out.push_str(&format!("{}:{}\n", path, line + 1));
    }
    out
}

/// Format hover result as markdown text.
fn format_hover(result: &Value) -> String {
    if result.is_null() {
        return String::new();
    }
    let content = &result["contents"];
    // MarkupContent: { kind, value }
    if let Some(value) = content["value"].as_str() {
        return value.to_string();
    }
    // MarkedString: string or array of {language, value}
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for item in arr {
            if let Some(value) = item["value"].as_str() {
                out.push_str(value);
                out.push('\n');
            } else if let Some(s) = item.as_str() {
                out.push_str(s);
                out.push('\n');
            }
        }
        return out;
    }
    "No hover content.".to_string()
}

/// Format diagnostics as readable lines.
fn format_diagnostics(diags: &[Value]) -> String {
    let mut out = String::new();
    for diag in diags {
        let severity = severity_label(diag["severity"].as_u64().unwrap_or(0));
        let line = diag["range"]["start"]["line"].as_u64().unwrap_or(0);
        let message = diag["message"].as_str().unwrap_or("");
        out.push_str(&format!("  {} line {}: {}\n", severity, line + 1, message));
    }
    out
}

/// Map LSP symbol kind integer to a short label.
fn symbol_kind_label(kind: u64) -> &'static str {
    match kind {
        1 => "File",
        2 => "Module",
        3 => "Namespace",
        4 => "Package",
        5 => "Class",
        6 => "Method",
        7 => "Property",
        8 => "Field",
        9 => "Constructor",
        10 => "Enum",
        11 => "Interface",
        12 => "Function",
        13 => "Variable",
        14 => "Constant",
        15 => "String",
        16 => "Number",
        17 => "Boolean",
        18 => "Array",
        23 => "Struct",
        24 => "Event",
        25 => "Operator",
        26 => "TypeParameter",
        _ => "Symbol",
    }
}

/// Map LSP diagnostic severity to a label.
fn severity_label(severity: u64) -> &'static str {
    match severity {
        1 => "ERROR",
        2 => "WARN",
        3 => "INFO",
        4 => "HINT",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// Path/URI helpers
// ---------------------------------------------------------------------------

/// Convert an absolute path to a `file://` URI.
fn path_to_uri(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.to_string_lossy())
}

/// Convert a `file://` URI to a display path relative to cwd.
fn uri_to_path(uri: &str, cwd: &str) -> String {
    let path_str = uri.strip_prefix("file://").unwrap_or(uri);
    let path = Path::new(path_str);
    super::display_path(path, cwd)
}

/// Resolve a (possibly relative) path against cwd.
fn resolve_path(path: &str, cwd: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    Ok(Path::new(cwd).join(path))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsp_tool_name_is_lsp() {
        let mut states = super::super::SessionStates::new();
        let tool = LspTool::new(&mut states, "/tmp");
        assert_eq!(tool.name(), "lsp");
    }

    #[test]
    fn lsp_tool_schema_has_server_and_language_params() {
        let mut states = super::super::SessionStates::new();
        let tool = LspTool::new(&mut states, "/tmp");
        let schema = tool.schema();
        let props = &schema["parameters"]["properties"];
        assert!(props.get("server").is_some());
        assert!(props.get("language").is_some());
        assert!(props.get("operation").is_some());
        let required: Vec<&str> = schema["parameters"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"operation"));
        assert!(required.contains(&"server"));
        assert!(required.contains(&"language"));
    }

    #[test]
    fn lsp_tool_schema_description_mentions_discovery() {
        let mut states = super::super::SessionStates::new();
        let tool = LspTool::new(&mut states, "/tmp");
        let schema = tool.schema();
        let desc = schema["description"].as_str().unwrap();
        assert!(desc.contains("which gopls"));
        assert!(desc.contains("rust-analyzer"));
    }

    #[test]
    fn symbol_kind_label_covers_common_kinds() {
        assert_eq!(symbol_kind_label(12), "Function");
        assert_eq!(symbol_kind_label(6), "Method");
        assert_eq!(symbol_kind_label(8), "Field");
        assert_eq!(symbol_kind_label(23), "Struct");
        assert_eq!(symbol_kind_label(11), "Interface");
        assert_eq!(symbol_kind_label(999), "Symbol");
    }

    #[test]
    fn severity_label_maps_correctly() {
        assert_eq!(severity_label(1), "ERROR");
        assert_eq!(severity_label(2), "WARN");
        assert_eq!(severity_label(3), "INFO");
        assert_eq!(severity_label(4), "HINT");
        assert_eq!(severity_label(0), "?");
    }

    #[test]
    fn format_locations_handles_single_and_array() {
        let cwd = "/tmp";
        let single = json!({
            "uri": "file:///tmp/main.go",
            "range": { "start": { "line": 9 } }
        });
        let out = format_locations(&single, cwd);
        assert!(out.contains("main.go:10"));

        let arr = json!([
            { "uri": "file:///tmp/a.go", "range": { "start": { "line": 4 } } },
            { "uri": "file:///tmp/b.go", "range": { "start": { "line": 19 } } }
        ]);
        let out = format_locations(&arr, cwd);
        assert!(out.contains("a.go:5"));
        assert!(out.contains("b.go:20"));
    }

    #[test]
    fn format_locations_handles_null() {
        assert_eq!(format_locations(&Value::Null, "/tmp"), "");
    }

    #[test]
    fn format_hover_extracts_markup_content() {
        let hover = json!({
            "contents": { "kind": "markdown", "value": "func foo() bar" }
        });
        assert_eq!(format_hover(&hover), "func foo() bar");
    }

    #[test]
    fn format_hover_extracts_plain_string() {
        let hover = json!({ "contents": "plain text" });
        assert_eq!(format_hover(&hover), "plain text");
    }

    #[test]
    fn path_to_uri_produces_file_scheme() {
        let dir = super::super::test_util::unique_test_dir();
        let path = dir.path().join("main.go");
        std::fs::write(&path, "package main\n").unwrap();
        let uri = path_to_uri(&path);
        assert!(uri.starts_with("file://"));
        assert!(uri.ends_with("main.go"));
    }
}
