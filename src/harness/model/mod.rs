//! Model-specific API protocol clients. Each implements [`ChatClient`]
//! (see `super::client`) over one wire protocol; the flavor is selected
//! per endpoint via the `api` field (see `crate::core::config::acp`).

pub mod anthropic;
pub mod openai_responses;

pub use anthropic::AnthropicClient;
pub use openai_responses::OpenAiResponsesClient;
