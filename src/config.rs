//! Operator-facing config for the Gemini chat binding.

use mcpg_backend_llm_shared::{
    ApiKeyRef, ChatExecutionSpec, ConfigError, EmbeddingExecutionSpec, ImageExecutionSpec,
};
use serde::{Deserialize, Serialize};

/// Spec for `binding_type: gemini_chat`. Default base URL is
/// `https://generativelanguage.googleapis.com/v1beta` (Google AI
/// Studio). Vertex AI uses a different URL + auth model — operators
/// on Vertex use `compat_chat` against Vertex's OpenAI-compat
/// endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiChatSpec {
    #[serde(default)]
    pub base_url: Option<String>,

    pub api_key: ApiKeyRef,

    #[serde(flatten)]
    pub chat: ChatExecutionSpec,
}

impl GeminiChatSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://generativelanguage.googleapis.com/v1beta";

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.chat.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

/// Spec for `binding_type: gemini_embedding`. Default base URL
/// matches the chat spec (Google AI Studio). Same flatten-passthrough
/// pattern: operator-side `base_url` + `api_key`, plus an embedded
/// [`EmbeddingExecutionSpec`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiEmbeddingSpec {
    #[serde(default)]
    pub base_url: Option<String>,

    pub api_key: ApiKeyRef,

    #[serde(flatten)]
    pub embedding: EmbeddingExecutionSpec,
}

impl GeminiEmbeddingSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.embedding.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or(GeminiChatSpec::DEFAULT_BASE_URL)
    }
}

/// Spec for `binding_type: gemini_image` (Imagen).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiImageSpec {
    #[serde(default)]
    pub base_url: Option<String>,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub image: ImageExecutionSpec,
}

impl GeminiImageSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.image.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or(GeminiChatSpec::DEFAULT_BASE_URL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::PromptSpec;
    use serde_json::json;

    fn minimal() -> ChatExecutionSpec {
        ChatExecutionSpec {
            model: "gemini-1.5-pro".into(),
            timeout_ms: 30_000,
            connect_timeout_ms: 5_000,
            prompt: PromptSpec {
                system: "you are helpful".into(),
                user: "{{ input.text }}".into(),
                ..Default::default()
            },
            sampling: Default::default(),
            response_format: Default::default(),
            tools: Default::default(),
            streaming: Default::default(),
            retry: Default::default(),
            guardrails: Default::default(),
            cache: Default::default(),
            budget: Default::default(),
        }
    }

    #[test]
    fn default_base_url() {
        let s = GeminiChatSpec {
            base_url: None,
            api_key: ApiKeyRef::new("k"),
            chat: minimal(),
        };
        assert_eq!(
            s.resolved_base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        s.validate().unwrap();
    }

    #[test]
    fn json_round_trip() {
        let json = json!({
            "model": "gemini-1.5-pro",
            "api_key": "k",
            "prompt": { "system": "x", "user": "y" }
        });
        let s: GeminiChatSpec = serde_json::from_value(json).unwrap();
        s.validate().unwrap();
    }

    // ----- Embedding spec -----

    #[test]
    fn embedding_default_base_url() {
        let s = GeminiEmbeddingSpec {
            base_url: None,
            api_key: ApiKeyRef::new("k"),
            embedding: EmbeddingExecutionSpec {
                model: "text-embedding-004".into(),
                ..Default::default()
            },
        };
        assert_eq!(
            s.resolved_base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        s.validate().unwrap();
    }

    #[test]
    fn embedding_json_round_trip() {
        let json = json!({
            "model": "gemini-embedding-001",
            "api_key": "k",
            "dimensions": 768,
            "max_batch_size": 50
        });
        let s: GeminiEmbeddingSpec = serde_json::from_value(json).unwrap();
        s.validate().unwrap();
        assert_eq!(s.embedding.dimensions, Some(768));
        assert_eq!(s.embedding.max_batch_size, Some(50));
    }
}
