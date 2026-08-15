//! Google Gemini (AI Studio) embedding adapter.
//!
//! Endpoint: `POST {base}/models/{model}:batchEmbedContents` (always
//! batch — single inputs are sent as a length-1 batch). Auth via the
//! `x-goog-api-key` header.
//!
//! Wire format:
//!
//! ```json
//! // Request
//! {
//!   "requests": [
//!     {"model": "models/text-embedding-004", "content": {"parts": [{"text": "..."}]}},
//!     ...
//!   ]
//! }
//!
//! // Response
//! {
//!   "embeddings": [
//!     {"values": [0.1, 0.2, ...]},
//!     ...
//!   ]
//! }
//! ```
//!
//! Gemini does not surface token counts on embedding calls in the
//! AI Studio surface; `usage` is always `None`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mcpg_backend_llm_shared::embedding::{
    EmbeddingProviderAdapter, NormalizedEmbeddingRequest, NormalizedEmbeddingResponse,
};
use mcpg_backend_llm_shared::error::ProviderError;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

/// Per-call cap: Gemini accepts up to 100 inputs per
/// `batchEmbedContents` call.
pub const GEMINI_MAX_INPUTS: usize = 100;

pub struct GeminiEmbeddingAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
}

impl std::fmt::Debug for GeminiEmbeddingAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiEmbeddingAdapter")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl GeminiEmbeddingAdapter {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        connect_timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("mcpg-plugin-backend-llm-gemini/1.0")
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|e| ProviderError::Network {
                message: format!("build http client: {e}"),
            })?;
        let base_url = base_url.into();
        if base_url.is_empty() {
            return Err(ProviderError::BadRequest {
                message: "base_url is empty".into(),
            });
        }
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: Arc::from(api_key.into()),
        })
    }

    fn endpoint_url(&self, model: &str) -> String {
        format!("{}/models/{}:batchEmbedContents", self.base_url, model)
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let key = self.api_key.as_ref();
        if key.is_empty() {
            return Err(ProviderError::AuthFailed {
                message: "api_key is empty".into(),
            });
        }
        let v = HeaderValue::from_str(key).map_err(|_| ProviderError::BadRequest {
            message: "api_key contains characters not allowed in HTTP headers".into(),
        })?;
        h.insert(HeaderName::from_static("x-goog-api-key"), v);
        Ok(h)
    }
}

#[async_trait]
impl EmbeddingProviderAdapter for GeminiEmbeddingAdapter {
    fn label(&self) -> &'static str {
        "gemini"
    }

    fn max_batch_size(&self) -> usize {
        GEMINI_MAX_INPUTS
    }

    async fn embed(
        &self,
        request: &NormalizedEmbeddingRequest,
        timeout: Duration,
    ) -> Result<NormalizedEmbeddingResponse, ProviderError> {
        let body = encode_request(request);
        let headers = self.build_headers()?;
        let url = self.endpoint_url(&request.model);

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Network {
                message: format!("send: {e}"),
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Malformed {
                message: format!("parse response json: {e}"),
            })?;
        decode_response(&value)
    }
}

fn encode_request(request: &NormalizedEmbeddingRequest) -> Value {
    let qualified_model = format!("models/{}", request.model);
    let requests: Vec<Value> = request
        .inputs
        .iter()
        .map(|text| {
            let mut req = json!({
                "model": qualified_model,
                "content": {"parts": [{"text": text}]},
            });
            if let Some(d) = request.dimensions {
                // `outputDimensionality` is the camelCase per the
                // AI Studio surface; see
                // https://ai.google.dev/api/embeddings.
                req["outputDimensionality"] = json!(d);
            }
            req
        })
        .collect();
    json!({"requests": requests})
}

fn decode_response(value: &Value) -> Result<NormalizedEmbeddingResponse, ProviderError> {
    let embeddings_arr = value
        .get("embeddings")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response missing `embeddings`".into(),
        })?;

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(embeddings_arr.len());
    let mut dimensions: u32 = 0;
    for entry in embeddings_arr {
        let arr = entry
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ProviderError::Malformed {
                message: "embedding entry missing `values`".into(),
            })?;
        let mut v = Vec::with_capacity(arr.len());
        for e in arr {
            let f = e.as_f64().ok_or_else(|| ProviderError::Malformed {
                message: "embedding contains non-number".into(),
            })?;
            v.push(f as f32);
        }
        if dimensions == 0 {
            dimensions = v.len() as u32;
        } else if v.len() as u32 != dimensions {
            return Err(ProviderError::Malformed {
                message: "embeddings have inconsistent dimensions".into(),
            });
        }
        vectors.push(v);
    }

    Ok(NormalizedEmbeddingResponse {
        embeddings: vectors,
        dimensions,
        usage: None,
    })
}

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let body_str = String::from_utf8_lossy(body).to_string();
    match status.as_u16() {
        401 | 403 => ProviderError::AuthFailed { message: body_str },
        429 => ProviderError::RateLimited { message: body_str },
        400 if body_str.to_lowercase().contains("token") => {
            ProviderError::ContextLimit { message: body_str }
        }
        400..=499 => ProviderError::BadRequest { message: body_str },
        500..=599 => ProviderError::Server { message: body_str },
        _ => ProviderError::Network {
            message: format!("unexpected status {status}: {body_str}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_request_emits_qualified_model_per_entry() {
        let r = NormalizedEmbeddingRequest {
            model: "text-embedding-004".into(),
            inputs: vec!["a".into(), "b".into()],
            dimensions: None,
        };
        let body = encode_request(&r);
        let arr = body["requests"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["model"], "models/text-embedding-004");
        assert_eq!(arr[0]["content"]["parts"][0]["text"], "a");
        assert!(arr[0].get("outputDimensionality").is_none());
    }

    #[test]
    fn encode_request_includes_output_dimensionality_when_set() {
        let r = NormalizedEmbeddingRequest {
            model: "gemini-embedding-001".into(),
            inputs: vec!["a".into()],
            dimensions: Some(768),
        };
        let body = encode_request(&r);
        assert_eq!(body["requests"][0]["outputDimensionality"], 768);
    }

    #[test]
    fn decode_response_parses_well_formed_data() {
        let raw = json!({
            "embeddings": [
                {"values": [0.1, 0.2, 0.3, 0.4]},
                {"values": [0.5, 0.6, 0.7, 0.8]}
            ]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.dimensions, 4);
        assert_eq!(r.embeddings.len(), 2);
        assert!(r.usage.is_none());
    }

    #[test]
    fn decode_response_rejects_inconsistent_dimensions() {
        let raw = json!({
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4, 0.5]}
            ]
        });
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn decode_response_rejects_missing_embeddings_field() {
        let raw = json!({});
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn endpoint_url_appends_model_path() {
        let a = GeminiEmbeddingAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url("text-embedding-004"),
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents"
        );
    }

    #[test]
    fn build_headers_rejects_empty_api_key() {
        let a = GeminiEmbeddingAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "",
            Duration::from_secs(1),
        )
        .unwrap();
        let err = a.build_headers().unwrap_err();
        assert!(matches!(err, ProviderError::AuthFailed { .. }));
    }

    #[test]
    fn map_status_429_rate_limited() {
        let e = map_status_error(reqwest::StatusCode::from_u16(429).unwrap(), b"slow down");
        assert!(matches!(e, ProviderError::RateLimited { .. }));
    }
}
