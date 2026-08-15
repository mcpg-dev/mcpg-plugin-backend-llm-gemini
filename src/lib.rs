//! # mcpg-plugin-backend-llm-gemini
//!
//! Google Gemini (AI Studio) chat-completion binding plugin for
//! MCPG. Ships [`GeminiChatPlugin`] (`kind: "gemini.chat"`).

mod adapter;
/// cdylib sync bridge + `declare_plugin!` export (backend-plugin-migration).
/// Additive: the gateway keeps using the static `new()` + `set_host_handle`
/// path. The `mcpg_plugin_register` FFI symbol is gated behind the
/// `cdylib-export` feature inside the macro expansion. Public so the
/// wrapper types + macro-generated entity modules are part of the
/// crate's public surface (mirrors the nats / kafka pilots, which keep
/// their bridges at crate root) — this also keeps the wrappers from
/// tripping `dead_code` on the default rlib build where neither
/// `cdylib-export` nor `static-firstparty` references them.
pub mod cdylib;
mod config;
mod embedding_adapter;
mod embedding_plugin;
mod host_handle_obs;
mod image_adapter;
mod image_plugin;
mod plugin;

pub use adapter::GeminiAdapter;
pub use config::{GeminiChatSpec, GeminiEmbeddingSpec, GeminiImageSpec};
pub use embedding_adapter::{GEMINI_MAX_INPUTS, GeminiEmbeddingAdapter};
pub use embedding_plugin::GeminiEmbeddingPlugin;
pub use image_adapter::GeminiImageAdapter;
pub use image_plugin::GeminiImagePlugin;
pub use plugin::GeminiChatPlugin;
