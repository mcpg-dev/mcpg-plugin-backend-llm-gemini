//! Google Gemini (AI Studio) image-generation adapter — Imagen.
//!
//! Endpoint: `POST {base}/models/{model}:predict`. Auth via the
//! `x-goog-api-key` header.
//!
//! Wire format:
//!
//! ```json
//! // Request
//! {
//!   "instances": [{"prompt": "a cat sitting on a chair"}],
//!   "parameters": {
//!     "sampleCount": 1,
//!     "sampleImageSize": "1024x1024",
//!     "aspectRatio": "1:1"
//!   }
//! }
//!
//! // Response
//! {
//!   "predictions": [
//!     { "bytesBase64Encoded": "iVBORw0…", "mimeType": "image/png" }
//!   ]
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use mcpg_backend_llm_shared::error::ProviderError;
use mcpg_backend_llm_shared::image::{
    GeneratedImage, ImageProviderAdapter, NormalizedImageRequest, NormalizedImageResponse,
};
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

pub struct GeminiImageAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
}

impl std::fmt::Debug for GeminiImageAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiImageAdapter")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl GeminiImageAdapter {
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
        format!("{}/models/{}:predict", self.base_url, model)
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
impl ImageProviderAdapter for GeminiImageAdapter {
    fn label(&self) -> &'static str {
        "gemini"
    }

    async fn generate(
        &self,
        request: &NormalizedImageRequest,
        timeout: Duration,
    ) -> Result<NormalizedImageResponse, ProviderError> {
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

fn encode_request(request: &NormalizedImageRequest) -> Value {
    let mut params = json!({"sampleCount": request.n});
    if let Some(s) = request.size.as_deref() {
        params["sampleImageSize"] = json!(s);
        // Imagen prefers `aspectRatio` separately too — derive from
        // size when it parses cleanly. Operators wanting a specific
        // aspect ratio supply `size` in WxH form.
        if let Some(ratio) = aspect_ratio_for_size(s) {
            params["aspectRatio"] = json!(ratio);
        }
    }
    if let Some(seed) = request.seed {
        params["seed"] = json!(seed);
    }
    // Imagen 3 supports `negativePrompt` on the parameters block.
    // Models that don't surface 400; we don't filter here.
    if let Some(neg) = request.negative_prompt.as_deref() {
        params["negativePrompt"] = json!(neg);
    }
    json!({
        "instances": [{"prompt": request.prompt}],
        "parameters": params,
    })
}

/// Map a `WxH` size string to one of Imagen's supported aspect-ratio
/// labels. Returns `None` for sizes that don't fit a known
/// label — Imagen will infer from `sampleImageSize` alone.
fn aspect_ratio_for_size(size: &str) -> Option<&'static str> {
    let (w, h) = size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))?;
    if w == h {
        return Some("1:1");
    }
    if w * 9 == h * 16 {
        return Some("16:9");
    }
    if w * 16 == h * 9 {
        return Some("9:16");
    }
    if w * 3 == h * 4 {
        return Some("4:3");
    }
    if w * 4 == h * 3 {
        return Some("3:4");
    }
    None
}

fn decode_response(value: &Value) -> Result<NormalizedImageResponse, ProviderError> {
    let preds = value
        .get("predictions")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response missing `predictions`".into(),
        })?;

    let mut images: Vec<GeneratedImage> = Vec::with_capacity(preds.len());
    for entry in preds {
        let b64 = entry
            .get("bytesBase64Encoded")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProviderError::Malformed {
                message: "prediction missing `bytesBase64Encoded`".into(),
            })?;
        let raw = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| ProviderError::Malformed {
                message: format!("decode bytesBase64Encoded: {e}"),
            })?;
        let mime_type = entry
            .get("mimeType")
            .and_then(|v| v.as_str())
            .unwrap_or("image/png")
            .to_owned();
        images.push(GeneratedImage {
            bytes: bytes::Bytes::from(raw),
            mime_type,
            // Imagen doesn't surface a revised prompt.
            revised_prompt: None,
        });
    }
    Ok(NormalizedImageResponse { images })
}

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let body_str = String::from_utf8_lossy(body).to_string();
    match status.as_u16() {
        401 | 403 => ProviderError::AuthFailed { message: body_str },
        429 => ProviderError::RateLimited { message: body_str },
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
    fn encode_request_includes_aspect_ratio_for_known_size() {
        let r = NormalizedImageRequest {
            model: "imagen-3.0-generate-001".into(),
            prompt: "a cat".into(),
            n: 2,
            size: Some("1024x1024".into()),
            quality: None,
            style: None,
            seed: None,
            negative_prompt: None,
            output_format: None,
        };
        let body = encode_request(&r);
        assert_eq!(body["instances"][0]["prompt"], "a cat");
        assert_eq!(body["parameters"]["sampleCount"], 2);
        assert_eq!(body["parameters"]["sampleImageSize"], "1024x1024");
        assert_eq!(body["parameters"]["aspectRatio"], "1:1");
    }

    #[test]
    fn encode_request_includes_seed_when_set() {
        let r = NormalizedImageRequest {
            model: "imagen".into(),
            prompt: "x".into(),
            n: 1,
            size: None,
            quality: None,
            style: None,
            seed: Some(42),
            negative_prompt: None,
            output_format: None,
        };
        let body = encode_request(&r);
        assert_eq!(body["parameters"]["seed"], 42);
    }

    #[test]
    fn encode_request_passes_negative_prompt_through() {
        let r = NormalizedImageRequest {
            model: "imagen-3.0-generate-001".into(),
            prompt: "a cat".into(),
            n: 1,
            size: None,
            quality: None,
            style: None,
            seed: None,
            negative_prompt: Some("blurry, extra limbs".into()),
            output_format: None,
        };
        let body = encode_request(&r);
        assert_eq!(body["parameters"]["negativePrompt"], "blurry, extra limbs");
    }

    #[test]
    fn aspect_ratio_for_size_handles_common_ratios() {
        assert_eq!(aspect_ratio_for_size("1024x1024"), Some("1:1"));
        assert_eq!(aspect_ratio_for_size("1920x1080"), Some("16:9"));
        assert_eq!(aspect_ratio_for_size("1080x1920"), Some("9:16"));
        assert_eq!(aspect_ratio_for_size("1024x768"), Some("4:3"));
        assert_eq!(aspect_ratio_for_size("768x1024"), Some("3:4"));
        assert_eq!(aspect_ratio_for_size("1234x567"), None);
        assert_eq!(aspect_ratio_for_size("not a size"), None);
    }

    #[test]
    fn decode_response_parses_predictions() {
        let payload = b"image bytes here";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let raw = json!({
            "predictions": [
                { "bytesBase64Encoded": b64, "mimeType": "image/png" }
            ]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].bytes.as_ref(), payload);
        assert_eq!(r.images[0].mime_type, "image/png");
        assert!(r.images[0].revised_prompt.is_none());
    }

    #[test]
    fn decode_response_rejects_missing_predictions() {
        let err = decode_response(&json!({})).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn decode_response_rejects_missing_bytes() {
        let raw = json!({
            "predictions": [ { } ]
        });
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn endpoint_url_appends_predict_path() {
        let a = GeminiImageAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url("imagen-3.0-generate-001"),
            "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict"
        );
    }

    #[test]
    fn build_headers_rejects_empty_api_key() {
        let a = GeminiImageAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "",
            Duration::from_secs(1),
        )
        .unwrap();
        let err = a.build_headers().unwrap_err();
        assert!(matches!(err, ProviderError::AuthFailed { .. }));
    }
}
