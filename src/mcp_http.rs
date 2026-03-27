//! Streamable HTTP MCP front door: `POST /mcp` with JSON-RPC body. Binds loopback only.

use anyhow::{Context, Result};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Response, Server};
use tracing::{error, info};

use crate::mcp_coord::{CoordinatorHandle, McpDispatchOutcome, dispatch_mcp_json_rpc};

const HEADER_AGENT_ID: &str = "X-Codepair-Agent-ID";

fn header_value<'a>(req: &'a tiny_http::Request, name: &'static str) -> Option<&'a str> {
    req.headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .and_then(|h| std::str::from_utf8(h.value.as_bytes()).ok())
}

/// Extract agent_id from URL path `/mcp/<agent_id>`.
fn agent_id_from_url(url: &str) -> Option<&str> {
    let path = url.split('?').next().unwrap_or(url);
    path.strip_prefix("/mcp/").filter(|s| !s.is_empty())
}

/// Connect to the MCP HTTP port and send `OPTIONS /mcp`; expect `204` (CORS preflight handler).
/// Retries TCP connect for a few seconds so the background thread can start accepting.
pub fn verify_mcp_http_server_ready(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("127.0.0.1:{}", port)
        .parse()
        .context("MCP verify: invalid 127.0.0.1 address")?;

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut stream = loop {
        match TcpStream::connect_timeout(&addr, Duration::from_millis(250)) {
            Ok(s) => break s,
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(e)
                        .context(format!("MCP verify: cannot connect to {} within 3s", addr));
                }
                thread::sleep(Duration::from_millis(20));
            }
        }
    };

    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));

    let req = format!(
        "OPTIONS /mcp HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
        port
    );
    stream
        .write_all(req.as_bytes())
        .context("MCP verify: write OPTIONS")?;
    stream.flush().context("MCP verify: flush")?;

    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).context("MCP verify: read response")?;
    let head = std::str::from_utf8(&buf[..n]).context("MCP verify: response not UTF-8")?;

    let first = head.lines().next().unwrap_or("");
    if !first.starts_with("HTTP/1.") {
        anyhow::bail!(
            "MCP verify: expected HTTP/1.x status line from 127.0.0.1:{}, got {:?}",
            port,
            first.chars().take(120).collect::<String>()
        );
    }
    if !first.contains("204") {
        anyhow::bail!(
            "MCP verify: expected 204 from OPTIONS /mcp on 127.0.0.1:{}, got {:?}",
            port,
            first
        );
    }

    info!("MCP HTTP readiness check passed on 127.0.0.1:{}/mcp", port);
    Ok(())
}

fn json_response(status: u16, body: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut r = Response::from_string(body).with_status_code(status);
    if status != 204 {
        r = r.with_header(json_content_type());
    }
    r.with_header(cors_star())
}

fn handle_request(
    mut req: tiny_http::Request,
    coord: CoordinatorHandle,
    shutdown: Arc<AtomicBool>,
) {
    let url = req.url().to_string();
    let method = req.method().clone();
    let origin_ok = validate_origin(&req);

    if method == Method::Options {
        let _ = req.respond(cors_preflight_response());
        return;
    }

    // Accept `/mcp` (bare) or `/mcp/<agent_id>`.
    let is_mcp_path = url == "/mcp" || url.starts_with("/mcp/") || url.starts_with("/mcp?");
    if !is_mcp_path {
        let _ = req.respond(json_response(404, "{\"error\":\"not found\"}"));
        return;
    }

    if method != Method::Post {
        let _ = req.respond(
            Response::from_string("method not allowed")
                .with_status_code(405)
                .with_header(Header::from_bytes(&b"Allow"[..], &b"POST, OPTIONS"[..]).unwrap())
                .with_header(cors_star()),
        );
        return;
    }

    let mut body = String::new();
    if let Err(e) = std::io::Read::read_to_string(&mut req.as_reader(), &mut body) {
        error!("read body: {}", e);
        let _ = req.respond(json_response(
            400,
            &format!(
                "{{\"error\":\"read body: {}\"}}",
                escape_json_str(&e.to_string())
            ),
        ));
        return;
    }

    if !origin_ok {
        let _ = req.respond(json_response(403, "{\"error\":\"forbidden origin\"}"));
        return;
    }

    let msg: Value = match serde_json::from_str(body.trim()) {
        Ok(v) => v,
        Err(e) => {
            let err_json = serde_json::json!({
                "jsonrpc": "2.0",
                "id": serde_json::Value::Null,
                "error": { "code": -32700, "message": e.to_string() }
            });
            let _ = req.respond(json_response(
                400,
                &serde_json::to_string(&err_json).unwrap_or_else(|_| "{}".to_string()),
            ));
            return;
        }
    };

    // Resolve agent_id: URL path takes priority, then header fallback.
    let url_agent_id = agent_id_from_url(&url).map(|s| s.to_string());
    let header_agent_id = header_value(&req, HEADER_AGENT_ID).map(|s| s.to_string());
    let effective_agent_id = url_agent_id.as_deref().or(header_agent_id.as_deref());

    let rpc_method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("?");
    info!("MCP {} agent_id={:?}", rpc_method, effective_agent_id,);

    let outcome = dispatch_mcp_json_rpc(&msg, &coord, effective_agent_id, &shutdown);

    match outcome {
        McpDispatchOutcome::NoContent => {
            if let Err(e) = req.respond(json_response(204, "")) {
                error!("MCP HTTP respond: {}", e);
            }
        }
        McpDispatchOutcome::Json(v) => {
            let s = serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string());
            if let Err(e) = req.respond(json_response(200, &s)) {
                error!("MCP HTTP respond: {}", e);
            }
        }
        McpDispatchOutcome::JsonWithHeldTask {
            json,
            agent_id,
            task,
        } => {
            let s = serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string());
            if let Err(e) = req.respond(json_response(200, &s)) {
                error!("MCP HTTP respond: {}", e);
                coord.restore_pending_task(&agent_id, task);
            }
        }
    }
}

/// Spawn a background thread serving MCP at `http://127.0.0.1:<port>/mcp`.
pub fn spawn_http_mcp_server(shutdown: Arc<AtomicBool>) -> Result<(CoordinatorHandle, u16)> {
    let server =
        Server::http("127.0.0.1:0").map_err(|e| anyhow::anyhow!("bind MCP HTTP: {}", e))?;
    let port = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .context("MCP server address must be TCP (127.0.0.1)")?;

    let coord = CoordinatorHandle::new_pair(Arc::clone(&shutdown));
    let coord_thr = coord.clone();

    thread::Builder::new()
        .name("mcp-http".into())
        .spawn(move || {
            info!("MCP HTTP listening on http://127.0.0.1:{}/mcp", port);
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let req = match server.recv_timeout(Duration::from_millis(100)) {
                    Ok(Some(r)) => r,
                    Ok(None) => continue,
                    Err(e) => {
                        error!("MCP HTTP recv error: {}", e);
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                };

                let coord_req = coord_thr.clone();
                let shutdown_req = Arc::clone(&shutdown);
                thread::spawn(move || handle_request(req, coord_req, shutdown_req));
            }
            info!("MCP HTTP server stopped");
        })
        .context("spawn mcp-http thread")?;

    verify_mcp_http_server_ready(port).context(
        "MCP HTTP server did not respond to readiness probe (OPTIONS /mcp → 204). \
         The listener may have failed to start.",
    )?;

    Ok((coord, port))
}

fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_content_type() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

fn cors_star() -> Header {
    Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap()
}

fn cors_preflight_response() -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string("")
        .with_status_code(204)
        .with_header(cors_star())
        .with_header(
            Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"POST, OPTIONS"[..])
                .unwrap(),
        )
        .with_header(
            Header::from_bytes(
                &b"Access-Control-Allow-Headers"[..],
                &b"Content-Type, X-Codepair-Agent-ID, Mcp-Session-Id"[..],
            )
            .unwrap(),
        )
}

/// Reject browser requests from unexpected origins (non-loopback). Local agents typically omit Origin.
fn validate_origin(req: &tiny_http::Request) -> bool {
    let Some(origin) = header_value(req, "Origin") else {
        return true;
    };
    let o = origin.to_ascii_lowercase();
    o.starts_with("http://127.0.0.1")
        || o.starts_with("http://localhost")
        || o.starts_with("http://[::1]")
        || o == "null"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;

    fn http_post_json(port: u16, agent_id: &str, body: &str) -> Result<(u16, String)> {
        let req = format!(
            "POST /mcp/{agent_id} HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {body}",
            body.len()
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port))?;
        stream.write_all(req.as_bytes())?;
        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line)?;
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line == "\n" {
                break;
            }
            let l = line.trim_end_matches(['\r', '\n']);
            let lower = l.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("content-length:") {
                content_length = rest.trim().parse().ok();
            }
        }
        let mut body_out = String::new();
        if let Some(n) = content_length {
            let mut buf = vec![0u8; n];
            reader.read_exact(&mut buf)?;
            body_out = String::from_utf8_lossy(&buf).into_owned();
        } else {
            reader.read_to_string(&mut body_out)?;
        }
        Ok((status_code, body_out))
    }

    #[test]
    fn mcp_server_starts_and_readiness_probe_succeeds() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (_coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        // spawn_http_mcp_server already ran verify_mcp_http_server_ready; call again to exercise API
        verify_mcp_http_server_ready(port)?;
        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(200));
        Ok(())
    }

    #[test]
    fn rejects_non_loopback_origin() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (_coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        thread::sleep(Duration::from_millis(50));

        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t"}}}"#;
        let req = format!(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: https://evil.example\r\nContent-Type: application/json\r\nX-Codepair-Agent-ID: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port))?;
        std::io::Write::write_all(&mut stream, req.as_bytes())?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf)?;
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("403"), "{}", text);

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(200));
        Ok(())
    }

    #[test]
    fn serves_second_request_while_first_long_poll_waits() -> Result<()> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (coord, port) = spawn_http_mcp_server(Arc::clone(&shutdown))?;
        let _bridge = coord.register_agent("worker-0")?;
        coord.register_agent("reviewer-0")?;

        let port_wait = port;
        let waiter = thread::spawn(move || -> Result<(u16, String)> {
            let wait = serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "codepair/wait_for_next_task",
                    "arguments": { "response": "" }
                }
            }))?;
            http_post_json(port_wait, "worker-0", &wait)
        });

        thread::sleep(Duration::from_millis(150));

        let start = Instant::now();
        let list = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }))?;
        let (status, body) = http_post_json(port, "reviewer-0", &list)?;
        assert_eq!(status, 200);
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "second request was blocked by first long poll: {:?}",
            start.elapsed()
        );
        assert!(
            body.contains("\"tools\":[]") || body.contains("\"tools\": []"),
            "{}",
            body
        );

        coord.submit_mcp_task("worker-0", "wake up".to_string())?;
        let (_status_wait, body_wait) = waiter.join().expect("waiter thread")?;
        assert!(body_wait.contains("wake up"), "{}", body_wait);

        shutdown.store(true, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(200));
        Ok(())
    }
}
