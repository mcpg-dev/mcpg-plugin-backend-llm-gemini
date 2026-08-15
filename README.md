# Google Gemini Backends — `dev.mcpg.backend.llm.gemini`

> class `backend` · `native` · package `mcpg-plugin-backend-llm-gemini` · artifact `libmcpg_plugin_backend_llm_gemini.so` · Apache-2.0

Exposes Google's AI Studio surface as MCP capabilities: Gemini chat, Gemini
embeddings and Imagen image generation, three backend entities in one artifact.
A binding pins one model and one execution policy; the plugin turns a tool call
into a `generativelanguage.googleapis.com` request and hands back a validated
result. Reach for it when Gemini should be reachable as a governed, budgeted,
audited MCP tool rather than an API key distributed to every client. Vertex AI
users go through `libs/plugins/backend/llms/compat` against Vertex's
OpenAI-compatible endpoint instead — Vertex has a different URL shape and
OAuth-based auth, deliberately out of scope here.

## What it does
- Registers three backend entities under one cdylib. Each self-describes its
  `BackendPlugin::kind()` at load time, so the gateway dispatches every binding
  to the right one.
- Translates the gateway's canonical chat shape into Gemini's `generateContent`
  wire format: model in the URL, `user`/`model` roles, system prompt on
  `systemInstruction`, tool results as `functionResponse` parts.
- Uses Gemini's native structured output — `generationConfig.responseSchema`
  with `responseMimeType: application/json` — then re-validates the reply
  binding-side.
- Renders `prompt.system` and `prompt.user` as MiniJinja templates over
  `input.*` (the caller's tool arguments) and `meta.*` (`backend_name`,
  `request_id`, `session_id`, `timestamp_iso8601`).
- Runs a bounded agentic loop over child MCP tools named in `tools.allowed`,
  refusing any call the model invents outside that list before it leaves the
  plugin; `tool_choice` maps to `toolConfig.functionCallingConfig.mode`.
- Accepts image, audio and file parts in the user turn, resolving
  `mcpg-resource://` URIs, `data:` URLs, plain URLs and raw base64.
- Pushes generated Imagen bytes into the gateway's content store and returns
  `mcpg-resource://<id>` URIs, so tool results stay small.
- Batches every embedding call through `batchEmbedContents`, capped at 100
  inputs per request, splitting larger batches across parallel calls.
- Retries rate-limit, 5xx and network failures with exponential backoff, and
  enforces per-binding token and daily-USD budget caps before spending.
- Declares the `network_outbound` capability — required in every mode, since
  every call is an outbound HTTPS request to Google.

| `backend.kind` | Registry kind | Entity id | Surface |
|---|---|---|---|
| `gemini_chat` | `gemini.chat` | `dev.mcpg.backend.gemini.chat` | chat completions |
| `gemini_embedding` | `gemini.embedding` | `dev.mcpg.backend.gemini.embedding` | embeddings |
| `gemini_image` | `gemini.image` | `dev.mcpg.backend.gemini.image` | Imagen image generation |

## Configuration

Load the artifact once from the flat top-level `plugins:` list — all three
entities come with it — then declare one binding per capability under
`mcp.capabilities.tools[]` (or `.prompts[]` / `.resources[]`), selecting the
entity with `backend.kind`. Everything else inside the `backend:` block is the
plugin's own spec, forwarded verbatim and validated by the plugin at boot, so an
invalid value fails gateway startup rather than the first call.

```yaml
plugins:
  - id: dev.mcpg.backend.llm.gemini
    class: backend
    source:
      oci: ghcr.io/mcpg-dev/source-code/plugins/backend-llm-gemini:protocol-1

mcp:
  capabilities:
    tools:
      - name: page.extract
        description: Extract structured facts from a screenshot.
        input_schema:
          type: object
          properties:
            shot: { type: string, description: "image URL or mcpg-resource:// URI" }
          required: [shot]
        backend:
          kind: gemini_chat
          api_key: "${env.GEMINI_API_KEY}"
          model: gemini-2.0-flash
          prompt:
            system: You read screenshots and answer only as JSON.
            user: Extract the visible order details.
            image_inputs: [shot]
          sampling:
            temperature: 0
          response_format:
            mode: json_schema
          # Read by the plugin when `response_format.mode: json_schema`.
          output_schema:
            type: object
            properties:
              order_id: { type: string }
              total:    { type: string }
            required: [order_id]
```

### Provider fields (every kind)

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | *(required)* | Sent as the `x-goog-api-key` header. Supply `${env.NAME}` or a `scheme://` URI bound to a `secret_provider` plugin (for example `vault://secret/gemini#key`); the gateway substitutes the literal value at config load. An empty resolved value is rejected. |
| `base_url` | string | `https://generativelanguage.googleapis.com/v1beta` | Override only for a forwarding proxy or a test fixture. The adapter appends `/models/{model}:generateContent`, `:batchEmbedContents` or `:predict`. |

### Chat execution fields (`gemini_chat`)

Shared with every other MCPG chat binding, so switching providers means changing
`kind` and `model` — not relearning the schema.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Gemini model id. |
| `prompt.system` | string | *(required)* | System-prompt template; carried on `systemInstruction`. Must be non-empty after trimming. |
| `prompt.user` | string | *(required)* | User-prompt template. Must be non-empty after trimming. |
| `prompt.image_inputs` | string[] | `[]` | Argument names carrying image content (URL, `data:` URL, raw base64, `mcpg-resource://` URI, or an explicit object). An array value fans out to several parts. |
| `prompt.audio_inputs` | string[] | `[]` | Argument names carrying audio. |
| `prompt.file_inputs` | string[] | `[]` | Argument names carrying documents; object values may set `mime_type` and `filename`. |
| `timeout_ms` | integer | `60000` | Per-iteration wall-clock budget upstream, retries included. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout, kept separate so a slow-but-connected upstream is not killed early. |
| `sampling.temperature` | number | *(unset)* | Passed through when set. |
| `sampling.top_p` | number | *(unset)* | Passed through when set. |
| `sampling.max_completion_tokens` | integer | *(unset)* | Per-iteration output cap. |
| `sampling.seed` | integer | *(unset)* | Passed through on `generationConfig.seed` when set. |
| `response_format.mode` | `json_schema` \| `text` | `json_schema` | `text` wraps the reply as `{"text": "…"}` and skips validation. |
| `response_format.strict` | boolean | `true` | Requests provider-side strictness where available; binding-side validation runs either way. |
| `response_format.on_mismatch` | `error` \| `retry_once` \| `return_raw` | `error` | `return_raw` is legal only with `mode: text`. |
| `tools.allowed` | string[] | `[]` | Names of other bindings in this gateway the model may call. Empty means single-shot. |
| `tools.max_iterations` | integer | `1` when `allowed` is empty, else `5` | Maximum model round-trips. Values above `50` are refused at boot. |
| `tools.tool_choice` | `auto` \| `required` \| `none` | `auto` | Maps to Gemini's `AUTO` / `ANY` / `NONE` function-calling mode. |
| `tools.tool_result_max_bytes` | integer | `16384` | Each child result is truncated to this before re-entering the conversation. |
| `tools.on_iteration_exhausted` | `error` \| `return_partial` | `error` | What happens when the loop runs out of iterations. |
| `retry.max_attempts` | integer | `3` | Attempts per upstream call. |
| `retry.initial_backoff_ms` | integer | `500` | First backoff; must not exceed `max_backoff_ms`. |
| `retry.max_backoff_ms` | integer | `8000` | Backoff ceiling. |
| `retry.retry_on` | list of `rate_limited` \| `server` \| `network` | all three | Failure classes worth retrying. |
| `guardrails.max_output_tokens_per_iteration` | integer | *(unset)* | Hard cap that overrides `sampling.max_completion_tokens`. |
| `cache.enabled` | boolean | `false` | Opt-in response cache. Refused at boot together with a non-empty `tools.allowed`. |
| `cache.ttl_seconds` | integer | `3600000` | Per-entry TTL, in seconds. |
| `budget.tokens_per_call_cap` | integer | `0` (uncapped) | Total input + output tokens across all loop iterations of one call. Checked between iterations, never on the first. |
| `budget.usd_daily_cap` | number | `0` (uncapped) | Aggregate spend for this binding per UTC day, checked before each call. |
| `output_schema` | object | *(unset)* | JSON Schema the reply must satisfy under `mode: json_schema`. Read out of this `backend:` block, not the binding-level field. |

### Embedding fields (`gemini_embedding`)

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Embedding model id. |
| `dimensions` | integer | *(unset)* | Requests reduced vectors where the model supports it. |
| `max_batch_size` | integer | `100` (provider cap) | Per-call batch size, clamped to Gemini's 100-input ceiling. Larger inputs split into parallel calls. |
| `timeout_ms` | integer | `10000` | Per-call timeout. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout. |
| `retry.max_attempts` | integer | `3` | Attempts per upstream call. |
| `retry.initial_backoff_ms` | integer | `200` | First backoff. |
| `retry.max_backoff_ms` | integer | `2000` | Backoff ceiling. |
| `cache.enabled` | boolean | `false` | Opt-in; `text → vector` is deterministic, so caching is sound. |
| `cache.ttl_seconds` | integer | `86400` | Per-entry TTL, in seconds. |

### Image fields (`gemini_image`)

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Imagen model id; it becomes the `:predict` URL segment. |
| `timeout_ms` | integer | `60000` | Per-call timeout. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout. |
| `defaults.size` | string | *(unset)* | Default framing, overridable per call. Sent as `sampleImageSize`; a `WxH` value that reduces to `1:1`, `16:9`, `9:16`, `4:3` or `3:4` also sets `aspectRatio`. |
| `defaults.n` | integer | *(unset)* | Default image count, sent as `sampleCount`; the engine falls back to `1` when neither the binding nor the call sets it. Must be at least `1` when set. |
| `defaults.negative_prompt` | string | *(unset)* | Sent as `negativePrompt`; Imagen 3 accepts it, older models reject the request. |
| `retry.max_attempts` / `retry.initial_backoff_ms` / `retry.max_backoff_ms` | integer | `3` / `200` / `2000` | Same retry shape as embeddings. |

The shared image spec also carries `defaults.quality`, `defaults.style` and
`defaults.output_format` for parity with the OpenAI and Stability image
bindings; the Imagen request carries none of them, so setting them here has no
effect.

## Operations

Each non-chat kind takes its own per-call argument shape and returns its own
envelope.

| Kind | Arguments | Result |
|---|---|---|
| `gemini_embedding` | `input` — a string or an array of strings | `{embeddings, dimensions, usage}`; `embeddings` always carries one entry per input |
| `gemini_image` | `prompt` (required), plus optional `size`, `n`, `seed`, `negative_prompt` | `{images: [{image_uri, mime_type, revised_prompt?}]}`, always an array |

Imagen bytes never travel inline: the engine pushes them into the gateway's
content store and returns an `mcpg-resource://<id>` URI that clients fetch with
an MCP `resources/read`. AI Studio does not report token counts on embedding
calls, so an embedding result's `usage` is absent.

```yaml
      - name: docs.embed
        description: Embed one or more passages.
        backend:
          kind: gemini_embedding
          api_key: "${env.GEMINI_API_KEY}"
          model: text-embedding-004
          cache: { enabled: true }

      - name: art.generate
        description: Generate an illustration with Imagen.
        backend:
          kind: gemini_image
          api_key: "${env.GEMINI_API_KEY}"
          model: imagen-3.0-generate-002
          defaults: { size: "1024x1024" }
```

## Response envelope

Chat bindings under `response_format.mode: json_schema` return the validated
object as-is; a reply that is not valid JSON or does not satisfy the schema
either fails the call or earns one corrective round-trip, per
`response_format.on_mismatch`. Under `mode: text` they return `{"text": "…"}` and
skip validation entirely.

## Security

- The API key is held in a redacting wrapper — `Debug` renders `***`, so it
  cannot leak through logs or error strings. A key that resolves to an empty
  value is rejected at boot rather than producing unauthenticated calls. The key
  travels in the `x-goog-api-key` header, never in a query string, so it does
  not end up in proxy access logs.
- Prompt templates can reference only `input.*` and `meta.*`. There is no
  filesystem loader, no env-var lookup, and the `debug` filter is removed, so a
  template cannot dump gateway state or exfiltrate the context. Undefined
  variables fail loudly instead of rendering empty.
- `tools.allowed` is an explicit allowlist enforced inside the plugin: a tool
  call the model invents that is not on the list never leaves the plugin. The
  gateway refuses a child call that targets the initiating binding itself and
  caps child-invocation depth at 8, on top of `tools.max_iterations`.
- Child tool calls carry no caller identity, and `cred://` credential threading
  is unsupported on that path. They are ungated unless you turn on
  `governance.child_invoke.enforce_gates`, which makes each child call run the
  same policy chain, trust floor, CEL `allow_if` gate and tool-gate chain a
  direct `tools/call` runs.
- Budget caps fail closed: exceeding `budget.usd_daily_cap` refuses the call
  before any upstream request is made. Models absent from the bundled rate card
  cannot accumulate cost, so a USD cap is inert for them.

## Observability

Every chat call opens a span (`llm_gemini.execute`, or
`llm_gemini.execute_streaming`) and emits a latency histogram
(`mcpg_llm_gemini_latency_seconds`) plus a call counter
(`mcpg_llm_gemini_calls_total`), both labelled with a bounded `outcome`
(`ok`, `rate_limited`, `auth_failed`, `model_not_found`, `server_error`,
`client_error`, `timeout`, `transport`) and `model`. When token usage is known
— the streaming path — it also emits `mcpg_llm_gemini_input_tokens_total`,
`mcpg_llm_gemini_output_tokens_total` and
`mcpg_llm_gemini_cost_usd_micros_total`.

One audit event lands per chat call at `dev.mcpg.llm.gemini.completion` or
`dev.mcpg.llm.gemini.failure`, carrying binding, model, outcome, duration and —
when known — token counts and cost in micro-USD. The embedding and image
engines emit their own counters and histograms (`mcpg_embedding_*`,
`mcpg_image_*`).

## MCP surfaces & composition

### As a child tool

Any binding backed by this plugin can appear in another chat binding's
`tools.allowed`, which is how you let one model reach for Gemini's long-context
or vision strengths mid-turn without any gateway-side orchestration code.

```yaml
        backend:
          kind: openai_chat
          api_key: "${env.OPENAI_API_KEY}"
          model: gpt-4o-mini
          prompt:
            system: Send screenshots to `page.extract` before answering.
            user: "{{ input.question }}"
          tools:
            allowed: [page.extract]   # a binding backed by gemini_chat
```

### Schemas & annotations

The binding-level `input_schema` is what clients see in `tools/list` and what
the gateway validates arguments against. The `output_schema` *inside* the
`backend:` block is what a chat binding enforces on the model's reply; declare
the binding-level `output_schema` too when you want clients to see the
contract. Mark bindings that only read as side-effect-free:

```yaml
        annotations: { read_only: true, open_world: true }
```

## Build
Default feature set is OFF (avoids `mcpg_plugin_register` linker
collisions in the workspace build); opt in to the cdylib:

```bash
cargo build -p mcpg-plugin-backend-llm-gemini --features cdylib-export --release   # → target/release/libmcpg_plugin_backend_llm_gemini.so
```

## Sign & load (production)
Sign the artifact, pin/verify via the entry's `signature:` block, and honour
revocations. See <https://mcpg.dev/docs/security/plugin-security>.

## See also
- Backend binding reference: <https://mcpg.dev/docs/reference/backends>
- Full gateway config schema: <https://mcpg.dev/docs/reference/configuration>
- Vertex AI and any other OpenAI-ABI endpoint: `libs/plugins/backend/llms/compat`
- Provider-agnostic engines and shared config types: `libs/plugins/backend/llms/shared`
