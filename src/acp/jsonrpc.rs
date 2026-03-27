//! JSON-RPC 2.0 helpers for ACP (newline-delimited bodies).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

#[derive(Debug, Clone)]
pub enum Outbound {
    Request {
        id: u64,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    Response {
        id: Value,
        result: Value,
    },
}

impl Outbound {
    pub fn to_json_line(&self) -> serde_json::Result<String> {
        let v = match self {
            Outbound::Request { id, method, params } => json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            }),
            Outbound::Notification { method, params } => json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }),
            Outbound::Response { id, result } => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result,
            }),
        };
        let mut s = serde_json::to_string(&v)?;
        s.push('\n');
        Ok(s)
    }
}

/// JSON-RPC `id` from a response: we only use numeric client ids; agents may echo as number or string.
pub fn jsonrpc_response_id_as_u64(id: &Value) -> Option<u64> {
    match id {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

/// True if this is a JSON-RPC response object (result or error) tied to a client-issued request id.
pub fn is_response_to_request(msg: &Value, our_numeric_ids: bool) -> bool {
    if msg.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
        return false;
    }
    if msg.get("method").is_some() {
        return false;
    }
    let has_result = msg.get("result").is_some();
    let has_error = msg.get("error").is_some();
    if !has_result && !has_error {
        return false;
    }
    let id = msg.get("id");
    if id.is_none() || id == Some(&Value::Null) {
        return false;
    }
    if our_numeric_ids {
        id.and_then(jsonrpc_response_id_as_u64).is_some()
    } else {
        true
    }
}

/// True if the agent is calling into the client (request expecting a result).
pub fn is_incoming_request(msg: &Value) -> bool {
    if msg.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
        return false;
    }
    msg.get("method").and_then(|m| m.as_str()).is_some()
        && msg.get("id").is_some()
        && msg.get("id") != Some(&Value::Null)
        && msg.get("result").is_none()
        && msg.get("error").is_none()
}

/// True if this is a JSON-RPC notification from the agent (no response).
pub fn is_incoming_notification(msg: &Value) -> bool {
    if msg.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
        return false;
    }
    msg.get("method").and_then(|m| m.as_str()).is_some()
        && (msg.get("id").is_none() || msg.get("id") == Some(&Value::Null))
}

pub fn parse_response_result(msg: &Value) -> Result<Value, JsonRpcError> {
    if let Some(err) = msg.get("error") {
        let e: JsonRpcError = serde_json::from_value(err.clone()).unwrap_or(JsonRpcError {
            code: -1,
            message: err.to_string(),
            data: None,
        });
        return Err(e);
    }
    msg.get("result").cloned().ok_or_else(|| JsonRpcError {
        code: -1,
        message: "missing result".to_string(),
        data: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outbound_request_line_is_valid_json_with_newline() {
        let line = Outbound::Request {
            id: 1,
            method: "initialize".to_string(),
            params: json!({"a": 1}),
        }
        .to_json_line()
        .unwrap();
        assert!(line.ends_with('\n'));
        let v: Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["method"], "initialize");
    }

    #[test]
    fn classifies_response_request_notification() {
        let res = json!({"jsonrpc":"2.0","id": 3, "result": {}});
        assert!(is_response_to_request(&res, true));
        assert!(!is_incoming_request(&res));
        assert!(!is_incoming_notification(&res));

        let res_str_id = json!({"jsonrpc":"2.0","id": "4", "result": {}});
        assert!(is_response_to_request(&res_str_id, true));
        assert_eq!(jsonrpc_response_id_as_u64(&res_str_id["id"]), Some(4));

        let req =
            json!({"jsonrpc":"2.0","id": 9, "method": "session/request_permission", "params": {}});
        assert!(!is_response_to_request(&req, true));
        assert!(is_incoming_request(&req));
        assert!(!is_incoming_notification(&req));

        let notif = json!({"jsonrpc":"2.0","method": "session/update", "params": {}});
        assert!(!is_response_to_request(&notif, true));
        assert!(!is_incoming_request(&notif));
        assert!(is_incoming_notification(&notif));
    }
}
