//! `BackendPlugin` impl for Google Gemini (AI Studio) embeddings.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use mcpg_backend_llm_shared::{
    EmbeddingEngine, EmbeddingProviderAdapter, ProviderError, resolve_api_key,
};
use mcpg_plugin_protocol::{
    BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest, BackendResponse,
    PluginManifest, async_trait, firstparty_manifest,
};
use serde_json::Value;

use crate::config::GeminiEmbeddingSpec;
use crate::embedding_adapter::GeminiEmbeddingAdapter;

pub struct GeminiEmbeddingPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<EmbeddingEngine>>>>,
}

impl std::fmt::Debug for GeminiEmbeddingPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiEmbeddingPlugin").finish()
    }
}

impl Default for GeminiEmbeddingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiEmbeddingPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.gemini.embedding",
                name: "Google Gemini Embeddings",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }
}

#[async_trait]
impl BackendPlugin for GeminiEmbeddingPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "gemini.embedding"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: GeminiEmbeddingSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("gemini_embedding spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.embedding.connect_timeout();

        let adapter = GeminiEmbeddingAdapter::new(base_url, api_key, connect_timeout).map_err(
            |e: ProviderError| BackendError::InvalidSpec {
                message: format!("build gemini embedding adapter: {e}"),
            },
        )?;
        let adapter: Arc<dyn EmbeddingProviderAdapter> = Arc::new(adapter);

        let engine = EmbeddingEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.embedding,
            host: host.clone(),
        };

        self.engines
            .write()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .insert(backend_name.to_owned(), Arc::new(engine));
        Ok(())
    }

    async fn execute(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendResponse, BackendError> {
        let engine = self
            .engines
            .read()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .get(backend_name)
            .cloned()
            .ok_or_else(|| BackendError::ProfileNotFound {
                backend_name: backend_name.to_owned(),
            })?;
        let args: Value = if request.payload.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&request.payload).map_err(|e| BackendError::InvalidSpec {
                message: format!("execute payload was not JSON: {e}"),
            })?
        };
        let result = engine.execute(&args).await;
        metrics::counter!(
            "mcpg_embedding_calls_total",
            "binding" => backend_name.to_owned(),
            "provider" => engine.adapter.label().to_string(),
            "model" => engine.spec.model.clone(),
            "status" => if result.is_ok() { "ok" } else { "error" },
        )
        .increment(1);
        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize embedding response: {e}"),
        })?;
        Ok(BackendResponse {
            payload,
            truncated: false,
        })
    }

    async fn execute_streaming(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendChunkStream, BackendError> {
        let resp = self.execute(backend_name, request).await?;
        Ok(Box::pin(futures::stream::once(async move {
            Ok(mcpg_plugin_protocol::BackendChunk::Done(resp))
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_plugin_protocol::noop_backend_host;

    #[test]
    fn plugin_kind_and_manifest() {
        let p = GeminiEmbeddingPlugin::new();
        assert_eq!(p.kind(), "gemini.embedding");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.gemini.embedding");
    }

    #[tokio::test]
    async fn register_minimal_spec() {
        let plugin = GeminiEmbeddingPlugin::new();
        plugin
            .register_profile(
                "embed",
                &serde_json::json!({
                    "model": "text-embedding-004",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }
}
