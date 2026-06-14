# ADR 0002: Consolidate LLM access on an OpenAI-compatible gateway

**Status:** accepted (2026-06)

**Amends:** [ADR 0001](0001-llm-provider-abstraction.md)

## Context

ADR 0001 gave each provider its own client module (`openai_client`, `yandex_client`,
`gigachat_client`, `ollama_client`) selected by `llm.factory.get_client` based on
`settings.llm_model`. In practice the deployment grew a LiteLLM gateway that already
proxies every upstream (OpenRouter, YandexGPT, GigaChat, Ollama), and the running
`.env` routed all traffic through it (`LLM_PROVIDER=openai_compatible`).

That left provider selection driven by **three independent axes** colliding in one
factory: the model name, the `LLM_PROVIDER` switch, and the mere presence of
`LOCAL_API_BASE` (`if model == "gpt-oss:latest" or settings.local_api_base` silently
forced Ollama). The native Yandex / GigaChat / Ollama clients had become dead code,
and `.env` carried five provider blocks with two parallel credential pairs — making
it impossible to tell at a glance which model actually served a request.

## Decision

Make the OpenAI-compatible gateway the only provider path.

- `get_client` always returns one `OpenAIChatClient(llm_model, llm_api_key, llm_base_url)`;
  the gateway resolves `llm_model` to its real upstream.
- Settings collapse to three fields: `llm_model`, `llm_base_url`, `llm_api_key`.
  `.env` / `.env.example` carry the same three LLM lines.
- The native `yandex_client`, `gigachat_client` and `ollama_client` modules are removed.
- The auxiliary judge / formality / image-decision client reuses the primary client
  (`local_llm = primary_llm`), so those agents are always available.

## Consequences

- Which model serves a request is unambiguous: whatever `LLM_MODEL` names, sent to
  `LLM_BASE_URL`. No hidden model-name or endpoint-presence routing.
- Adding an upstream is now a gateway-config change, not a code change — the inverse
  of ADR 0001's "new provider = one file + one factory line".
- Direct, gateway-bypassing access to Yandex / GigaChat / Ollama is no longer
  supported from the bot; restoring it means re-adding a client module.
- `LLMClient` (ADR 0001) stays the seam `ChatService` depends on, so the service and
  its tests are untouched.
