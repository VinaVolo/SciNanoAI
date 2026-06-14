# ADR 0001: LLM provider abstraction

**Status:** accepted (2026-05)

## Context

The previous chatbot embedded four provider branches (OpenAI, YandexGPT, GigaChat,
Ollama) inside `ChatBot.generate_response` as an `if/elif/else` chain. The same
raw `requests.post(...)` call to the Ollama endpoint was duplicated in
`judge_answer`, `handle_incomplete_answer`, `summarize_messages` and the main
response generator — four copies of the same HTTP pattern that already existed
in `local_agents.LocalLLMAgent.complete`.

The double-call to `judge_answer` (it was called twice in the same boolean
expression) doubled token spend for every reply.

## Decision

Introduce `scinanoai.chatbot.llm.base.LLMClient`, a runtime-checkable `Protocol`
with a single method:

```python
def complete(messages, *, temperature, max_tokens) -> str: ...
```

Each provider is its own module (`openai_client`, `yandex_client`,
`gigachat_client`, `ollama_client`). `llm.factory.get_client(settings)` returns
a configured client based on `settings.llm_model`.

The `ChatService` depends on `LLMClient`, not on a concrete provider. The
formality / image-decision / summarise / judge helpers all consume the same
abstraction (specifically the local client, kept on `ChatService._local`).

## Consequences

- A new provider is a single file + one line in the factory.
- Unit tests stub `LLMClient` with a plain dataclass — no HTTP mocking needed.
- The judge call now runs exactly once per turn (saved 50% on local LLM
  invocations for replies that pass on the first try).
- The duplicated `requests.post` blocks (4× in the legacy file) are gone.
- Retries and timeouts live in `OllamaChatClient` (tenacity) rather than at
  five different call sites.
