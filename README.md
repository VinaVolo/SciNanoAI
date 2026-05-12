<h1 align="center">SciNanoAI</h1>

<p align="center">
  Scientific RAG chatbot and image-analysis pipeline for nano-structure research.
</p>

<p align="center">
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/adr">Decision Records</a> ·
  <a href=".env.example">Environment template</a> ·
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <a href="https://doi.org/10.1021/acs.jcim.5c01897"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.5c01897-B31B1B.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>

> The original RAG platform is described in **Krotkov *et&nbsp;al.*, *J. Chem. Inf. Model.* 2025**
> ([doi:10.1021/acs.jcim.5c01897](https://doi.org/10.1021/acs.jcim.5c01897)).
> The Cellpose-based image-analysis extension shipped on this branch is the subject
> of a forthcoming follow-up publication.

---

## Overview

SciNanoAI bundles three cooperating services:

| Service          | Port | Code path                                | Purpose                                          |
|------------------|------|------------------------------------------|--------------------------------------------------|
| `vector_service` | 8000 | `scinanoai.vector_service.api:app`       | FAISS retrieval (MMR) over ~300 parsed papers    |
| `chatbot_api`    | 8001 | `scinanoai.chatbot.api:app`              | RAG orchestration, LLM dispatch, Cellpose images |
| `chatbot_ui`     | 8517 | `python -m scinanoai.chatbot.ui`         | Gradio frontend with basic auth                  |

LLM providers (selectable via `LLM_MODEL`): OpenAI-compatible, YandexGPT, GigaChat (Sber), Ollama / `gpt-oss:latest`.

## Quick start (Docker)

```bash
git clone git@github.com:VinaVolo/SciNanoAI.git
cd SciNanoAI

cp .env.example .env
# Fill in: LLM_MODEL, the matching API_KEY/_BASE pair, S3 keys, GRADIO_USERNAME/PASSWORD

docker compose up --build
```

Open <http://localhost:8517/scinanoai> — log in with `GRADIO_USERNAME` / `GRADIO_PASSWORD` from `.env`.

The compose stack uses the multi-stage `docker/chatbot.Dockerfile` and `docker/vector.Dockerfile`. All
containers run as non-root and ship a `HEALTHCHECK`; `chatbot_api` waits for `vector_service` to become
healthy before starting.

## Local development

This project uses **[uv](https://docs.astral.sh/uv/)** as the package manager.
Python 3.11 is pinned via `.python-version` (the project intentionally targets `<3.13` while
torch / langchain / sentence-transformers stabilise on 3.13).

```bash
# uv installs the right Python and resolves from uv.lock
uv sync --extra chatbot --extra vector --extra dev   # services + tests
uv sync --extra ml --extra dev                       # to train Cellpose
uv sync --extra data                                 # for S3 + Excel + PDF ingest

# Pre-commit (ruff, gitleaks, mypy, etc.)
uv run pre-commit install
```

### Run the services without Docker

```bash
# Terminal 1
uv run uvicorn scinanoai.vector_service.api:app --host 0.0.0.0 --port 8000

# Terminal 2
uv run uvicorn scinanoai.chatbot.api:app --host 0.0.0.0 --port 8001

# Terminal 3
uv run python -m scinanoai.chatbot.ui
```

### Tests

```bash
uv run pytest -m unit          # fast, no external services
uv run pytest                  # all tests (some need a FAISS index + ML models)
```

## Data pipelines

### Download data and index from S3

```bash
scinanoai-s3 download --prefix data/ --dest data/      # raw datasets
scinanoai-s3 download --prefix db/   --dest db/        # FAISS index
```

Set `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` in `.env`. The CLI is idempotent — already-present
files are skipped by size.

### Build / extend the FAISS index

```bash
# Full rebuild from parsed PDF pages
scinanoai-ingest-pdf --parsed-pages data/parsed_pages --db-path db/intfloat_multilingual-e5-large

# Append rows from the image-metrics Excel file
scinanoai-ingest-excel --excel data/img_result_nn_radius_cyto_2.xlsx --db-path db/intfloat_multilingual-e5-large
```

The PDF builder writes `db/<index>/manifest.json` recording the embedding model + dimension. The vector
service refuses to start if the manifest contradicts `VECTOR_EMBEDDING_MODEL` — see
[ADR-0002](docs/adr/0002-vector-index-versioning.md).

## Training (Cellpose)

```bash
scinanoai-train-cellpose --config configs/training/cellpose_default.yaml
```

A run manifest (path, config snapshot, metric keys) is written to `<output_dir>/<model_name>/logging/run_manifest.json`.

## Repository layout

```
src/scinanoai/                  # All Python sources live here
  chatbot/                      # FastAPI + Gradio + LLM clients + RAG orchestrator
    core/                       # history, prompts, router
    llm/                        # one file per provider + factory
    services/                   # vector client, decomposer, formality, image analysis, references
  vector_service/               # FAISS retrieval microservice
    ingest/                     # excel.py + pdf.py
  training/cellpose/            # CellposeTrainer + data + metrics + CLI
  training/yolo/                # YOLO segmentation training
  data/                         # mask -> COCO, PDF parsing, renaming utilities
  storage/                      # S3 client + unified CLI
  utils/                        # logging, paths

chatbot_app/                    # Thin backward-compat shims (will be removed once
vector_service/                 # consumers migrate to scinanoai.* imports)

configs/training/               # YAML training configs
scripts/                        # CLI shims (train_cellpose.py)
docker/                         # multi-stage Dockerfiles
docs/                           # architecture.md + ADRs
tests/{unit,integration}/       # pytest suites
```

## Security

- Secrets live in `.env` (gitignored). Use `.env.example` as the template.
- `gitleaks` runs in pre-commit and CI.
- The vector service supports `X-API-Key` auth (set `VECTOR_SERVICE_API_KEY` in `.env` to enable).
- All containers run as non-root user `appuser` (UID 10001).
- Request bodies are bounded via Pydantic schemas; per-image upload size is capped by the FastAPI body
  limit configured on the chatbot service.

## Migration notes

Compared to the pre-refactor layout:

- Three `requirements.txt` files are replaced by a single `pyproject.toml` with extras
  (`chatbot`, `vector`, `ml`, `data`, `dev`). `uv.lock` is regenerated from the new schema and
  committed — install with `uv sync`, not `pip install -r`.
- `chatbot_app/chatbot.py` (718-line god class) → 17 focused modules under `src/scinanoai/chatbot/`.
- `download_data.py` / `download_db.py` / `upload_data.py` → unified `scinanoai-s3` CLI.
- Two `train_cellpose.py` entrypoints (root + `src/train/`) → single YAML-driven CLI.
- `src/evalute/` (typo) and `paths copy.py` (literal space in filename) removed.
- `chatbot_app/main.py` and `vector_service/main.py` remain as thin re-export shims so existing
  process managers (`uvicorn chatbot_app.main:app`) keep working during migration.

## Citation

If SciNanoAI helps your research, please cite the JCIM 2025 article that
introduces the platform:

> **Krotkov, N. A.; Sbytov, D. A.; Chakhoyan, A. A.; Kornienko, P. I.;
> Starikova, A. A.; Stepanov, M. G.; Piven, A. O.; Aliev, T. A.; Orlova, T.;
> Rafayelyan, M. S.; Skorb, E. V.**
> *Nanostructured Material Design via a Retrieval-Augmented Generation (RAG)
> Approach: Bridging Laboratory Practice and Scientific Literature.*
> **Journal of Chemical Information and Modeling**, 2025 — special issue
> "Machine Learning in Materials Science".
> DOI: [10.1021/acs.jcim.5c01897](https://doi.org/10.1021/acs.jcim.5c01897).

<details>
<summary>BibTeX</summary>

```bibtex
@article{krotkov2025scinanoai,
  title   = {Nanostructured Material Design via a Retrieval-Augmented
             Generation (RAG) Approach: Bridging Laboratory Practice and
             Scientific Literature},
  author  = {Krotkov, Nikita A. and Sbytov, Dmitrii A. and
             Chakhoyan, Anna A. and Kornienko, Polina I. and
             Starikova, Anna A. and Stepanov, Maxim G. and
             Piven, Anastasiia O. and Aliev, Timur A. and Orlova, Tetiana and
             Rafayelyan, Mushegh S. and Skorb, Ekaterina V.},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2025},
  note    = {Special issue ``Machine Learning in Materials Science''},
  doi     = {10.1021/acs.jcim.5c01897},
  url     = {https://doi.org/10.1021/acs.jcim.5c01897},
  publisher = {American Chemical Society}
}
```
</details>

The machine-readable [`CITATION.cff`](CITATION.cff) ships the same metadata for
GitHub's "Cite this repository" button. The Cellpose extension on this branch
will be added as a second entry once the follow-up paper is out.

## License

[MIT](LICENSE) — © the SciNanoAI authors. The published article is © American
Chemical Society and distributed under ACS publication terms.
