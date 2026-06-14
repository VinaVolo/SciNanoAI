# SciNanoAI — architecture

## Top-level view

```
                        ┌────────────────────────────┐
                        │       Gradio UI (8517)     │
                        │   scinanoai.chatbot.ui     │
                        └──────────────┬─────────────┘
                                       │  HTTP JSON
                                       ▼
                        ┌────────────────────────────┐
                        │   Chatbot API (8001)       │
                        │   scinanoai.chatbot.api    │
                        └──┬───────────┬──────────┬──┘
                           │           │          │
                  ┌────────▼──┐  ┌─────▼─────┐ ┌──▼──────────┐
                  │ LLM       │  │ Vector    │ │ Image       │
                  │ Providers │  │ Service   │ │ Analyzer    │
                  │ (OpenAI / │  │ (8000)    │ │ Cellpose +  │
                  │ Yandex /  │  │ FAISS     │ │ stub        │
                  │ Sber /    │  │ MMR       │ │             │
                  │ Ollama)   │  │           │ │             │
                  └───────────┘  └───────────┘ └─────────────┘
```

Three Docker services orchestrated via `docker-compose`:

| Service        | Port | Image                | Entrypoint                                       |
|----------------|------|----------------------|--------------------------------------------------|
| `vector_service` | 8000 | `docker/vector.Dockerfile` | `scinanoai.vector_service.api:app`     |
| `chatbot_api`    | 8001 | `docker/chatbot.Dockerfile` | `scinanoai.chatbot.api:app`            |
| `chatbot_ui`     | 8517 | `docker/chatbot.Dockerfile` | `python -m scinanoai.chatbot.ui`       |

All services run as non-root, have a `HEALTHCHECK`, and read their config from `.env`.

## Package layout

```
src/scinanoai/
├── chatbot/
│   ├── api.py              # FastAPI app — POST /chat, /clear_history, GET /health
│   ├── ui.py               # Gradio frontend (uses CHAT_API_URL)
│   ├── service.py          # ChatService — orchestrates LLM, retrieval, images
│   ├── factory.py          # build_chat_service() — DI wiring from settings
│   ├── settings.py         # ChatbotSettings (pydantic-settings)
│   ├── schemas.py          # API DTOs
│   ├── core/
│   │   ├── history.py      # ConversationHistory + HistoryStore (multi-session)
│   │   ├── prompts.py      # All prompt templates
│   │   └── router.py       # Routing decision (DB vs image vs plain)
│   ├── llm/
│   │   ├── base.py         # LLMClient Protocol
│   │   ├── factory.py      # get_client(settings)
│   │   └── openai_client.py  # OpenAI-compatible gateway client
│   └── services/
│       ├── vector_client.py     # httpx + tenacity retry
│       ├── decomposer.py        # zero-shot classifier (mDeBERTa)
│       ├── formality.py         # FormalityAgent / ImageDecisionAgent
│       ├── image_analysis.py    # Cellpose pipeline (decoupled)
│       └── references.py        # bracket-citation enrichment
│
├── vector_service/
│   ├── api.py              # FastAPI — POST /v1/query, GET /v1/stats, /health
│   ├── repository.py       # FAISS loader with manifest check
│   ├── schemas.py          # request/response DTOs (Pydantic Field bounds)
│   ├── settings.py         # VectorSettings
│   └── ingest/
│       ├── excel.py        # Excel -> FAISS (idempotent additive)
│       └── pdf.py          # parsed-PDF JSONs -> FAISS (full build)
│
├── training/
│   ├── cellpose/
│   │   ├── trainer.py      # CellposeTrainer (legacy — to be split further)
│   │   ├── data.py         # MaskDatasetBuilder
│   │   ├── metrics.py      # InstanceMetrics (pure functions)
│   │   ├── config.py       # CellposeTrainingConfig dataclass
│   │   └── cli.py          # YAML-driven CLI (replaces 3 entrypoints)
│   └── yolo/train.py       # YOLO training (side-effect-free)
│
├── data/                   # PDF parsing, mask -> COCO conversions
├── storage/
│   ├── s3.py               # S3Client (idempotent download/upload, retries)
│   └── cli.py              # `scinanoai-s3 {download|upload} ...`
└── utils/
    ├── logging.py          # setup_logging() — rotating, env-driven level
    └── paths.py            # get_project_root() / get_project_path()
```

## Key flows

### Chat request

1. `POST /chat` arrives at `chatbot.api`.
2. `ChatService.handle()`:
   - Appends message to the session-keyed `ConversationHistory`.
   - Summarises if over the token budget.
   - `Router.route()` consults Decomposer + (optionally) formality/image-decision agents.
   - Based on the route:
     - `image_analysis` — calls `ImageAnalyzer` (Cellpose if available; brightness stub fallback) and feeds metrics into the local LLM.
     - `literature` / `no_images` with `use_database=True` — queries `VectorServiceClient` and builds a RAG prompt.
     - plain answer — builds a direct prompt.
   - LLM client is selected by `llm.factory.get_client(settings)`.
   - If the **local** LLM is configured, the answer is passed through `_is_insufficient` (single LLM judge call — the previous code triggered two calls per turn) and may be reformulated once.
   - `ReferenceResolver` rewrites `[filename.pdf]` citations into URLs.

### Vector query

`POST /v1/query` validates bounds via Pydantic, server-side clamps them via `VectorSettings.max_k`/`max_fetch_k`, and runs MMR search through the lazily-loaded `VectorRepository`. The repository checks `db/<index>/manifest.json` against the configured embedding model and refuses to load on mismatch.

## Configuration

Everything reads from `.env` (see [`.env.example`](../.env.example)). The two services have separate `pydantic-settings` classes (`ChatbotSettings`, `VectorSettings`) so they cannot accidentally drift.

## Security posture

- No secrets in code or git history (gitleaks runs in pre-commit and CI).
- Vector service supports `X-API-Key` auth (opt-in).
- Containers run as non-root user `appuser` (UID 10001).
- Request bodies bounded via Pydantic (`max_length=4000` query, base64 image size implicitly capped by FastAPI body limit).
- `allow_dangerous_deserialization=True` retained for FAISS load (langchain requirement) but gated by manifest validation.

## Open items (intentionally deferred)

- Split `cellpose_trainer.py` into `data/loop/trainer/plotting` — current file kept whole pending a green training run.
- Replace FAISS with pgvector/Qdrant for production-grade ops.
- Production-grade auth (OAuth/JWT) — currently only Gradio basic-auth + optional X-API-Key.
