//! [Agent Client Protocol](https://agentclientprotocol.com/) (ACP) client for Cursor `agent acp`.
//!
//! Transport is JSON-RPC 2.0 with one message per line (newline-delimited), matching
//! [Cursor’s ACP documentation](https://cursor.com/docs/cli/acp).
//!
//! Primary entry points: [`client::AcpClient`]; in-memory transport is test-only ([`transport`]).
pub mod client;
pub mod jsonrpc;
pub mod orchestrator_hooks;
#[cfg(test)]
pub mod transport;
pub mod types;
pub mod workspace_read;
