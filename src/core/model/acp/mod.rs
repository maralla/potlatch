//! [Agent Client Protocol](https://agentclientprotocol.com/) (ACP) client and model runtime.
//!
//! Transport is JSON-RPC 2.0 with one message per line (newline-delimited), matching
//! [Cursor's ACP documentation](https://cursor.com/docs/cli/acp).
//!
//! Primary entry points: [`client::AcpClient`]; subprocess orchestration in [`runtime`];
//! in-memory transport is test-only ([`transport`]).

pub mod client;
pub mod jsonrpc;
pub mod orchestrator_hooks;
pub(crate) mod runtime;
#[cfg(test)]
pub mod transport;
pub mod types;
pub mod workspace_read;

pub use runtime::ACP_SESSION_MODE_ASK;
pub(crate) use runtime::AcpRuntime;
