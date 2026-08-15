//! Google Gemini (AI Studio) adapter.
//!
//! Translates the engine's canonical (OpenAI-shaped)
//! [`NormalizedChatRequest`] / [`NormalizedChatResponse`] to/from
//! Gemini's `generateContent` REST surface at
//! `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`.
//!
//! ## Wire-format differences from OpenAI
//!
//! - **Model in URL**: not in the body. Per-call URL is built from
//!   `request.model`.
//! - **Roles**: `user` / `model` (no `system`, no `assistant`, no
//!   `tool`). System prompt rides on the top-level
//!   `systemInstruction.parts`. Assistant turns become `model` turns.
//!   Tool-result messages become a `user` turn whose `parts` are
//!   `functionResponse` blocks.
//! - **Messages**: called `contents`, each with a `parts` array of
//!   typed blocks (`text`, `functionCall`, `functionResponse`).
//! - **Structured output**: native — `generationConfig.responseSchema`
//!   plus `responseMimeType: "application/json"`. No forced-tool
//!   gymnastics, unlike Anthropic.
//! - **Tool choice**: `toolConfig.functionCallingConfig.mode`
//!   uppercase enum `AUTO` / `ANY` / `NONE`.
//! - **Auth**: `x-goog-api-key` header (also accepted via `?key=`
//!   query string, but the header is cleaner).
//!
//! ## Vertex AI
//!
//! This adapter targets **Google AI Studio** (`generativelanguage…`).
//! Vertex AI uses a different URL pattern and OAuth-based auth — out
//! of scope here. Operators on Vertex use `provider: openai-compatible`
//! against Vertex's OpenAI-compatible endpoint.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use mcpg_backend_llm_shared::normalized::{
    AudioSource, ContentPart, FileSource, FinishReason, ImageSource, Message, MessageContent,
    NormalizedChatRequest, NormalizedChatResponse, Role, TokenUsage, ToolCall, ToolChoiceWire,
    ToolDef,
};
use mcpg_backend_llm_shared::{
    ChatProviderAdapter, NormalizedStreamEvent, ProviderError, StreamEventReceiver,
};

pub struct GeminiAdapter {
    client: Client,
    /// Base URL up to and including the API version. The model name
    /// is appended per-request (`/models/{model}:generateContent`).
    /// Default: `https://generativelanguage.googleapis.com/v1beta`.
    base_url: String,
    api_key: Arc<str>,
}

impl GeminiAdapter {
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
                message: "gemini base_url is empty".into(),
            });
        }
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: Arc::from(api_key.into()),
        })
    }

    fn endpoint_url(&self, model: &str) -> String {
        format!("{}/models/{}:generateContent", self.base_url, model)
    }

    /// Gemini's streaming endpoint is a separate path with the `alt=sse`
    /// query string. Sends `data: {...}` events containing partial
    /// candidate chunks.
    fn streaming_endpoint_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        )
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let key_value = HeaderValue::from_str(self.api_key.as_ref()).map_err(|_| {
            ProviderError::BadRequest {
                message: "api_key contains characters not allowed in HTTP headers".into(),
            }
        })?;
        h.insert(HeaderName::from_static("x-goog-api-key"), key_value);
        Ok(h)
    }
}

#[async_trait]
impl ChatProviderAdapter for GeminiAdapter {
    fn label(&self) -> &'static str {
        "gemini"
    }

    async fn chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<NormalizedChatResponse, ProviderError> {
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
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read response body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Malformed {
                message: format!("response is not JSON: {e}"),
            })?;

        decode_response(&value)
    }

    async fn stream_chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<StreamEventReceiver, ProviderError> {
        let body = encode_request(request);
        let headers = self.build_headers()?;
        let url = self.streaming_endpoint_url(&request.model);

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            let bytes = resp.bytes().await.unwrap_or_default();
            return Err(map_status_error(status, &bytes));
        }

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<NormalizedStreamEvent, ProviderError>>(32);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            // Gemini emits one `data: {...}` per chunk, where each
            // `{...}` is a *partial* response object — one or more
            // candidates with partial parts, optionally a final
            // `usageMetadata`. Function calls arrive as a `parts`
            // entry whose `functionCall.args` is the FULL final
            // value (Gemini doesn't fragment function args the way
            // OpenAI does), but the call may still arrive in a chunk
            // separate from any text deltas. We forward each chunk's
            // text as TextDelta and emit ToolCallReady when a
            // functionCall part appears.
            let mut buffer: Vec<u8> = Vec::new();
            let mut function_call_counter: u32 = 0;
            let mut final_finish: Option<FinishReason> = None;
            let mut input_tokens: u32 = 0;
            let mut output_tokens: u32 = 0;
            let mut cached_tokens: u32 = 0;
            let mut stream_error: Option<ProviderError> = None;
            let mut emitted_tool_call = false;

            'outer: while let Some(chunk_res) = byte_stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(e) => {
                        stream_error = Some(ProviderError::Network {
                            message: format!("read sse chunk: {e}"),
                        });
                        break 'outer;
                    }
                };
                buffer.extend_from_slice(&chunk);

                while let Some(boundary) = find_event_boundary(&buffer) {
                    let event_bytes = buffer.drain(..boundary).collect::<Vec<u8>>();
                    let _ = strip_boundary_prefix(&mut buffer);

                    let event_text = match std::str::from_utf8(&event_bytes) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let data = match event_text
                        .lines()
                        .find_map(|line| line.strip_prefix("data:"))
                        .map(|s| s.trim_start_matches(' '))
                    {
                        Some(d) => d,
                        None => continue,
                    };
                    if data.trim() == "[DONE]" {
                        break 'outer;
                    }
                    let event: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    if let Some(usage) = event.get("usageMetadata") {
                        let u = decode_usage(Some(usage));
                        input_tokens = u.input_tokens;
                        output_tokens = u.output_tokens;
                        cached_tokens = u.cached_input_tokens;
                    }

                    let Some(candidate) = event
                        .get("candidates")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                    else {
                        continue;
                    };

                    if let Some(parts) = candidate
                        .get("content")
                        .and_then(|c| c.get("parts"))
                        .and_then(|p| p.as_array())
                    {
                        for part in parts {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str())
                                && !t.is_empty()
                            {
                                if tx
                                    .send(Ok(NormalizedStreamEvent::TextDelta(t.to_owned())))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            } else if let Some(fc) = part.get("functionCall") {
                                let name = fc
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_owned();
                                let args = fc.get("args").cloned().unwrap_or(Value::Null);
                                function_call_counter += 1;
                                let id = format!("gemini_call_{function_call_counter}");
                                emitted_tool_call = true;
                                if tx
                                    .send(Ok(NormalizedStreamEvent::ToolCallReady(ToolCall {
                                        id,
                                        name,
                                        arguments: args,
                                    })))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                        }
                    }

                    if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str()) {
                        final_finish = Some(match reason {
                            "STOP" if emitted_tool_call => FinishReason::ToolCalls,
                            "STOP" => FinishReason::Stop,
                            "MAX_TOKENS" => FinishReason::Length,
                            "SAFETY" => FinishReason::ContentFilter,
                            _ => FinishReason::Other,
                        });
                    }
                }
            }

            if let Some(err) = stream_error {
                let _ = tx.send(Err(err)).await;
                return;
            }

            let _ = tx
                .send(Ok(NormalizedStreamEvent::Finish {
                    reason: final_finish.unwrap_or(FinishReason::Other),
                    usage: TokenUsage {
                        input_tokens,
                        output_tokens,
                        cached_input_tokens: cached_tokens,
                    },
                }))
                .await;
        });

        Ok(rx)
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos);
    }
    buf.windows(2).position(|w| w == b"\n\n")
}
fn strip_boundary_prefix(buf: &mut Vec<u8>) -> usize {
    if buf.starts_with(b"\r\n\r\n") {
        buf.drain(..4);
        4
    } else if buf.starts_with(b"\n\n") {
        buf.drain(..2);
        2
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Request encoding
// ---------------------------------------------------------------------------

fn encode_request(req: &NormalizedChatRequest) -> Value {
    let (system, contents) = encode_messages(&req.messages);

    let mut body = serde_json::Map::new();

    if let Some(s) = system {
        body.insert("systemInstruction".into(), json!({"parts": [{"text": s}]}));
    }
    body.insert("contents".into(), Value::Array(contents));

    if !req.tools.is_empty() {
        let function_decls: Vec<Value> = req.tools.iter().map(encode_tool_def).collect();
        body.insert(
            "tools".into(),
            json!([{"functionDeclarations": function_decls}]),
        );
        let mode = match req.tool_choice {
            ToolChoiceWire::Auto => "AUTO",
            ToolChoiceWire::Required => "ANY",
            ToolChoiceWire::None => "NONE",
        };
        body.insert(
            "toolConfig".into(),
            json!({"functionCallingConfig": {"mode": mode}}),
        );
    }

    let mut gen_config = serde_json::Map::new();
    if let Some(t) = req.temperature {
        gen_config.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        gen_config.insert("topP".into(), json!(p));
    }
    if let Some(n) = req.max_completion_tokens {
        gen_config.insert("maxOutputTokens".into(), json!(n));
    }
    if let Some(s) = req.seed {
        // Gemini accepts `seed` on most models; harmless to pass.
        gen_config.insert("seed".into(), json!(s));
    }
    if let Some(schema) = &req.response_schema {
        gen_config.insert(
            "responseMimeType".into(),
            Value::String("application/json".into()),
        );
        gen_config.insert("responseSchema".into(), schema.clone());
    }
    if !gen_config.is_empty() {
        body.insert("generationConfig".into(), Value::Object(gen_config));
    }

    Value::Object(body)
}

fn encode_tool_def(t: &ToolDef) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
    })
}

/// Returns (system_text, contents). System messages are pulled out
/// of the flat `messages` list because Gemini carries them in
/// `systemInstruction`. Multiple system messages join with `\n\n`.
fn encode_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut out: Vec<Value> = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                system_parts.push(m.content.as_text());
            }
            Role::User => match &m.content {
                MessageContent::Text(s) => {
                    out.push(json!({
                        "role": "user",
                        "parts": [{"text": s}]
                    }));
                }
                MessageContent::Parts(parts) => {
                    out.push(json!({
                        "role": "user",
                        "parts": encode_user_parts(parts),
                    }));
                }
            },
            Role::Assistant => {
                let mut parts: Vec<Value> = Vec::new();
                let text = m.content.as_text();
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
                for tc in &m.tool_calls {
                    parts.push(json!({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }));
                }
                if parts.is_empty() {
                    parts.push(json!({"text": ""}));
                }
                out.push(json!({"role": "model", "parts": parts}));
            }
            Role::Tool => {
                // Gemini wraps tool results in a `user` turn whose
                // parts are `functionResponse` blocks. The `name`
                // identifies which function this is a response to;
                // we read it from the engine's tool_call_id (which
                // we keep in normalized.rs). To make this work, the
                // tool-result message's content is forwarded as the
                // `response` payload (parsed if JSON, otherwise
                // wrapped as `{ "result": "<raw>" }`).
                let raw = m.content.as_text();
                let response_payload: Value =
                    serde_json::from_str(&raw).unwrap_or_else(|_| json!({"result": raw}));
                let function_name = m
                    .tool_call_id
                    .as_deref()
                    // Gemini doesn't carry an id; the convention is
                    // to round-trip the tool name. The engine stores
                    // the original call id which (for Gemini) won't
                    // exist — fall back to a synthetic name.
                    .unwrap_or("tool_response");
                out.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": function_name,
                            "response": response_payload,
                        }
                    }]
                }));
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, out)
}

/// Encode a [`MessageContent::Parts`] user message as Gemini's `parts`
/// array. Gemini accepts:
///
/// - `{"text": "..."}` for prose.
/// - `{"inline_data": {"mime_type": "...", "data": "<base64>"}}`
///   for everything else (images, audio, files). Inline blobs are
///   capped at ~9.5 MB; larger uploads need the Files API (out of
///   scope for the binding; operators chain through a separate
///   `gemini.files` binding when they hit the limit).
/// - `{"file_data": {"mime_type": "...", "file_uri": "..."}}` for
///   the Files API path. Used only for `FileSource::Url` since that
///   matches Gemini's preferred URL handoff.
///
/// `mcpg-resource://` should be pre-resolved upstream; if one slips
/// through we fall back to text so the model gets a notice rather
/// than a malformed request.
fn encode_user_parts(parts: &[ContentPart]) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            ContentPart::Text(s) => {
                out.push(json!({"text": s}));
            }
            ContentPart::Image(img) => match &img.source {
                ImageSource::Url(u) => {
                    // Gemini doesn't accept HTTP URLs in
                    // `inline_data`; the engine should have
                    // pre-fetched. Forward through `file_data` —
                    // operators using Vertex / Files API can have it
                    // resolve, others get an error from upstream that
                    // surfaces the unsupported URL.
                    out.push(json!({
                        "file_data": {
                            "mime_type": "image/*",
                            "file_uri": u,
                        }
                    }));
                }
                ImageSource::Base64 { mime_type, data } => {
                    out.push(json!({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    }));
                }
                ImageSource::McpResource(uri) => {
                    out.push(json!({
                        "text": format!("[unresolved image resource: {uri}]"),
                    }));
                }
            },
            ContentPart::Audio(au) => match &au.source {
                AudioSource::Url(u) => {
                    out.push(json!({
                        "file_data": {
                            "mime_type": au.format.mime_type(),
                            "file_uri": u,
                        }
                    }));
                }
                AudioSource::Base64 { data } => {
                    out.push(json!({
                        "inline_data": {
                            "mime_type": au.format.mime_type(),
                            "data": data,
                        }
                    }));
                }
                AudioSource::McpResource(uri) => {
                    out.push(json!({
                        "text": format!("[unresolved audio resource: {uri}]"),
                    }));
                }
            },
            ContentPart::File(f) => match &f.source {
                FileSource::Url(u) => {
                    out.push(json!({
                        "file_data": {
                            "mime_type": f.mime_type,
                            "file_uri": u,
                        }
                    }));
                }
                FileSource::Base64 { data } => {
                    out.push(json!({
                        "inline_data": {
                            "mime_type": f.mime_type,
                            "data": data,
                        }
                    }));
                }
                FileSource::McpResource(uri) => {
                    out.push(json!({
                        "text": format!("[unresolved file resource: {uri}]"),
                    }));
                }
            },
        }
    }
    Value::Array(out)
}

// ---------------------------------------------------------------------------
// Response decoding
// ---------------------------------------------------------------------------

fn decode_response(value: &Value) -> Result<NormalizedChatResponse, ProviderError> {
    let candidate = value
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response has no candidates[0]".into(),
        })?;

    let content = candidate
        .get("content")
        .ok_or_else(|| ProviderError::Malformed {
            message: "candidates[0].content missing".into(),
        })?;

    let parts = content
        .get("parts")
        .and_then(|p| p.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "candidates[0].content.parts missing".into(),
        })?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut function_call_counter: u32 = 0;

    for part in parts {
        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
            text_parts.push(t.to_owned());
        } else if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
            let args = fc.get("args").cloned().unwrap_or(Value::Null);
            // Gemini doesn't return a stable call_id; synthesize a
            // monotonic one so the engine can round-trip its tool
            // result back. Round-tripping keeps the engine's
            // provider-agnostic loop happy.
            function_call_counter += 1;
            let id = format!("gemini_call_{function_call_counter}");
            tool_calls.push(ToolCall {
                id,
                name,
                arguments: args,
            });
        }
        // Other part shapes (inlineData for vision, executableCode,
        // codeExecutionResult) are not currently supported; silently
        // skipped.
    }

    let finish_reason = match candidate.get("finishReason").and_then(|v| v.as_str()) {
        // Gemini docs: STOP / MAX_TOKENS / SAFETY / RECITATION /
        // OTHER. We map SAFETY → ContentFilter so callers can
        // distinguish from generic finishes.
        Some("STOP") if !tool_calls.is_empty() => FinishReason::ToolCalls,
        Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("SAFETY") => FinishReason::ContentFilter,
        Some("TOOL_USE") => FinishReason::ToolCalls,
        _ => FinishReason::Other,
    };

    let usage = decode_usage(value.get("usageMetadata"));

    Ok(NormalizedChatResponse {
        content: text_parts.join(""),
        tool_calls,
        finish_reason,
        usage,
    })
}

fn decode_usage(value: Option<&Value>) -> TokenUsage {
    let Some(u) = value else {
        return TokenUsage::default();
    };
    let input = u
        .get("promptTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output = u
        .get("candidatesTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached = u
        .get("cachedContentTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: cached,
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let message = body_excerpt(body);
    let code = status.as_u16();
    if code == 429 {
        return ProviderError::RateLimited { message };
    }
    if code == 401 || code == 403 {
        return ProviderError::AuthFailed { message };
    }
    if code == 400 {
        // Gemini returns 400 + "exceeds the maximum number of tokens"
        // for context overflow. Other 400 causes are config errors.
        if message.contains("maximum number of tokens")
            || message.contains("exceeds")
            || message.contains("context length")
        {
            return ProviderError::ContextLimit { message };
        }
        return ProviderError::BadRequest { message };
    }
    if (500..600).contains(&code) {
        return ProviderError::Server { message };
    }
    ProviderError::Server { message }
}

fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        return ProviderError::Network {
            message: format!("timeout: {err}"),
        };
    }
    if err.is_connect() {
        return ProviderError::Network {
            message: format!("connect failed: {err}"),
        };
    }
    if err.is_request() || err.is_body() || err.is_decode() {
        return ProviderError::Network {
            message: format!("transport: {err}"),
        };
    }
    ProviderError::Network {
        message: err.to_string(),
    }
}

fn body_excerpt(body: &[u8]) -> String {
    const MAX: usize = 512;
    let s = String::from_utf8_lossy(body);
    if s.len() <= MAX {
        s.into_owned()
    } else {
        format!("{}…[truncated]", &s[..MAX])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::normalized::{Message, ToolDef};

    fn baseline(messages: Vec<Message>) -> NormalizedChatRequest {
        NormalizedChatRequest {
            model: "gemini-1.5-pro".into(),
            messages,
            response_schema: None,
            strict_response: false,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        }
    }

    #[test]
    fn endpoint_url_includes_model_path_segment() {
        let a = GeminiAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url("gemini-1.5-pro"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        );
    }

    #[test]
    fn endpoint_url_strips_trailing_slash_in_base() {
        let a = GeminiAdapter::new(
            "https://generativelanguage.googleapis.com/v1beta/",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert!(!a.endpoint_url("m").contains("//models"), "no double slash");
    }

    #[test]
    fn encode_minimal_text_pulls_system_to_system_instruction() {
        let r = baseline(vec![Message::system("be brief"), Message::user("hi")]);
        let body = encode_request(&r);
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            json!("be brief")
        );
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], json!("user"));
        assert_eq!(contents[0]["parts"][0]["text"], json!("hi"));
        // Sanity: no tools / no generationConfig (sampling unset).
        assert!(body.get("tools").is_none());
        assert!(body.get("generationConfig").is_none());
    }

    #[test]
    fn encode_translates_assistant_role_to_model() {
        let r = baseline(vec![
            Message::user("first"),
            Message::assistant_text_and_tool_calls("answering", vec![]),
        ]);
        let body = encode_request(&r);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[1]["role"], json!("model"));
        assert_eq!(contents[1]["parts"][0]["text"], json!("answering"));
    }

    #[test]
    fn encode_tool_call_round_trip() {
        let r = baseline(vec![
            Message::user("look it up"),
            Message::assistant_text_and_tool_calls(
                "checking",
                vec![ToolCall {
                    id: "fetch".into(),
                    name: "fetch".into(),
                    arguments: json!({"q": "x"}),
                }],
            ),
            Message::tool_result("fetch", "{\"data\":42}"),
        ]);
        let body = encode_request(&r);
        let contents = body["contents"].as_array().unwrap();
        // [user, model_with_text_and_functionCall, user_with_functionResponse]
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], json!("model"));
        assert_eq!(contents[1]["parts"][0]["text"], json!("checking"));
        assert_eq!(
            contents[1]["parts"][1]["functionCall"]["name"],
            json!("fetch")
        );
        assert_eq!(
            contents[1]["parts"][1]["functionCall"]["args"],
            json!({"q": "x"})
        );
        assert_eq!(contents[2]["role"], json!("user"));
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["name"],
            json!("fetch")
        );
        // Tool result was JSON, parsed and inlined.
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"],
            json!({"data": 42})
        );
    }

    #[test]
    fn encode_tool_result_non_json_is_wrapped() {
        let r = baseline(vec![
            Message::user("hi"),
            Message::tool_result("t1", "plain text result"),
        ]);
        let body = encode_request(&r);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["response"],
            json!({"result": "plain text result"})
        );
    }

    #[test]
    fn encode_with_tools_emits_function_declarations() {
        let mut r = baseline(vec![Message::user("hi")]);
        r.tools = vec![ToolDef {
            name: "linear.fetch".into(),
            description: "Fetch a Linear issue".into(),
            parameters: json!({"type": "object", "properties": {"id": {"type": "string"}}}),
        }];
        let body = encode_request(&r);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let decls = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], json!("linear.fetch"));
        assert_eq!(decls[0]["description"], json!("Fetch a Linear issue"));
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            json!("AUTO")
        );
    }

    #[test]
    fn encode_required_choice_maps_to_any_uppercase() {
        let mut r = baseline(vec![Message::user("hi")]);
        r.tools = vec![ToolDef {
            name: "t".into(),
            description: "t".into(),
            parameters: json!({"type": "object"}),
        }];
        r.tool_choice = ToolChoiceWire::Required;
        let body = encode_request(&r);
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            json!("ANY")
        );
    }

    #[test]
    fn encode_response_schema_uses_response_mime_type_and_response_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        });
        let mut r = baseline(vec![Message::user("hi")]);
        r.response_schema = Some(schema.clone());
        let body = encode_request(&r);
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            json!("application/json")
        );
        assert_eq!(body["generationConfig"]["responseSchema"], schema);
    }

    #[test]
    fn encode_sampling_pass_through() {
        let mut r = baseline(vec![Message::user("hi")]);
        r.temperature = Some(0.5);
        r.top_p = Some(0.5);
        r.max_completion_tokens = Some(2048);
        r.seed = Some(7);
        let body = encode_request(&r);
        let gc = &body["generationConfig"];
        assert_eq!(gc["temperature"].as_f64().unwrap(), 0.5);
        assert_eq!(gc["topP"].as_f64().unwrap(), 0.5);
        assert_eq!(gc["maxOutputTokens"], json!(2048));
        assert_eq!(gc["seed"], json!(7));
    }

    #[test]
    fn decode_text_only() {
        let raw = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "hi back"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3
            }
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.content, "hi back");
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.finish_reason, FinishReason::Stop);
        assert_eq!(r.usage.input_tokens, 10);
        assert_eq!(r.usage.output_tokens, 3);
    }

    #[test]
    fn decode_function_call_round_trip() {
        let raw = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "let me check"},
                        {"functionCall": {"name": "fetch", "args": {"q": "hello"}}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.content, "let me check");
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "fetch");
        assert_eq!(r.tool_calls[0].arguments, json!({"q": "hello"}));
        // Synthetic id since Gemini doesn't return one.
        assert!(r.tool_calls[0].id.starts_with("gemini_call_"));
        // STOP + tool_calls present → ToolCalls finish.
        assert_eq!(r.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn decode_safety_finish_maps_to_content_filter() {
        let raw = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": ""}]
                },
                "finishReason": "SAFETY"
            }]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.finish_reason, FinishReason::ContentFilter);
    }

    #[test]
    fn decode_max_tokens_finish_maps_to_length() {
        let raw = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "trunc..."}]},
                "finishReason": "MAX_TOKENS"
            }]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.finish_reason, FinishReason::Length);
    }

    #[test]
    fn decode_rejects_response_without_candidates() {
        let raw = json!({"usageMetadata": {}});
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn decode_unknown_part_shapes_skipped() {
        let raw = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "hi"},
                        {"inlineData": {"mimeType": "image/png", "data": "..."}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.content, "hi");
    }

    #[test]
    fn map_status_400_with_token_overflow_is_context_limit() {
        let e = map_status_error(
            reqwest::StatusCode::from_u16(400).unwrap(),
            b"{\"error\":{\"message\":\"input exceeds the maximum number of tokens\"}}",
        );
        assert!(matches!(e, ProviderError::ContextLimit { .. }));
    }

    #[test]
    fn map_status_401_is_auth_failed() {
        let e = map_status_error(reqwest::StatusCode::from_u16(401).unwrap(), b"bad key");
        assert!(matches!(e, ProviderError::AuthFailed { .. }));
    }

    #[test]
    fn map_status_429_is_rate_limited() {
        let e = map_status_error(reqwest::StatusCode::from_u16(429).unwrap(), b"slow down");
        assert!(matches!(e, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn map_status_503_is_server_retryable() {
        let e = map_status_error(reqwest::StatusCode::from_u16(503).unwrap(), b"unavailable");
        assert!(matches!(e, ProviderError::Server { .. }));
        assert!(e.is_retryable());
    }

    #[test]
    fn encode_no_tools_omits_tool_config() {
        let body = encode_request(&baseline(vec![Message::user("hi")]));
        assert!(body.get("toolConfig").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn decode_usage_handles_missing_fields() {
        let raw = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP"
            }]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.usage.input_tokens, 0);
        assert_eq!(r.usage.output_tokens, 0);
    }

    // ----- Multimodal user-parts encoding -----

    #[test]
    fn encode_user_image_base64_emits_inline_data() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, ImageContent, ImageSource};
        let parts = vec![
            ContentPart::Text("what's here".into()),
            ContentPart::Image(ImageContent {
                source: ImageSource::Base64 {
                    mime_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                },
                detail: None,
            }),
        ];
        let (system, contents) = encode_messages(&[Message::user_parts(parts)]);
        assert!(system.is_none());
        let arr = contents[0]["parts"].as_array().unwrap();
        assert_eq!(arr[0]["text"], "what's here");
        assert_eq!(arr[1]["inline_data"]["mime_type"], "image/png");
        assert_eq!(arr[1]["inline_data"]["data"], "aGVsbG8=");
    }

    #[test]
    fn encode_user_audio_emits_inline_data_with_format_mime() {
        use mcpg_backend_llm_shared::normalized::{
            AudioContent, AudioFormat, AudioSource, ContentPart,
        };
        let parts = vec![ContentPart::Audio(AudioContent {
            source: AudioSource::Base64 {
                data: "QUJDRA==".into(),
            },
            format: AudioFormat::Wav,
        })];
        let (_, contents) = encode_messages(&[Message::user_parts(parts)]);
        let inline = &contents[0]["parts"][0]["inline_data"];
        assert_eq!(inline["mime_type"], "audio/wav");
        assert_eq!(inline["data"], "QUJDRA==");
    }

    #[test]
    fn encode_user_file_url_emits_file_data() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, FileContent, FileSource};
        let parts = vec![ContentPart::File(FileContent {
            source: FileSource::Url("https://ex.com/x.pdf".into()),
            mime_type: "application/pdf".into(),
            filename: None,
        })];
        let (_, contents) = encode_messages(&[Message::user_parts(parts)]);
        let fd = &contents[0]["parts"][0]["file_data"];
        assert_eq!(fd["mime_type"], "application/pdf");
        assert_eq!(fd["file_uri"], "https://ex.com/x.pdf");
    }
}
