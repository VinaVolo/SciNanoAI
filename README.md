<h1 align="center">SciNanoAI</h1>

<p align="center">
  <b>A retrieval-augmented chatbot for nanostructured-material design.</b><br>
  Reference implementation of the <i>Journal of Chemical Information and Modeling</i> (2025) study by Krotkov <i>et&nbsp;al.</i>
</p>

<p align="center">
  <a href="https://doi.org/10.1021/acs.jcim.5c01897"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.5c01897-B31B1B.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="#citation"><img alt="Cite this work" src="https://img.shields.io/badge/cite-J.%20Chem.%20Inf.%20Model.%202025-orange.svg"></a>
</p>

---

## About

SciNanoAI is an **agent-based retrieval-augmented generation (RAG) platform** built around
large language models (LLMs), aimed at materials scientists working on
**nanostructured materials produced via two-photon polymerization (2PP)** and related
techniques. The system extracts and synthesises information from a curated corpus of
scientific literature and patent texts, and answers free-form questions in
context — bridging day-to-day laboratory practice and the published literature.

The platform demonstrated, on the test set of the accompanying article:

| Metric | Value |
|---|---|
| Semantic accuracy (cosine similarity) | **0.82** |
| Overall task precision | **0.81** |

A dynamic query-refinement mechanism reduces hallucinations and misinformation typical
of plain LLM chats, while keeping the interface friendly to non-CS researchers.

The codebase shipped in this repository is the snapshot accompanying the publication.

## Architecture

Two cooperating microservices and a Gradio frontend:

```
            ┌──────────────────────┐
            │   Gradio UI (8517)   │
            │   chatbot_app/app.py │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐         ┌──────────────────────┐
            │  Chat API (8001)     │────────▶│  Vector API (8000)   │
            │  chatbot_app/main.py │  query  │  vector_service/     │
            └──────────┬───────────┘         │  FAISS + e5-large    │
                       │                     └──────────────────────┘
                       ▼
          ┌────────────────────────────┐
          │ LLM providers (any of):    │
          │  OpenAI / YandexGPT /      │
          │  GigaChat (Sber)           │
          └────────────────────────────┘
```

- `vector_service/` — FastAPI microservice wrapping a FAISS index built from
  `intfloat/multilingual-e5-large` embeddings over **301 parsed documents**
  (scientific articles + patents on nanostructured materials, 2PP, cell
  responses to nanotopography).
- `chatbot_app/` — FastAPI chat orchestrator + `decomposer_agent` that decides
  whether a query needs the literature corpus or can be answered directly.
- `notebooks/` — research notebooks used to build the document store and
  evaluate metrics from the article.

## Installation

```bash
git clone git@github.com:VinaVolo/SciNanoAI.git
cd SciNanoAI
```

### Python environment

- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- **Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Environment variables

Copy `env-example` to `.env` and fill in the credentials of every LLM/data
provider you plan to use:

```bash
S3_ACCESS_KEY=
S3_SECRET_KEY=
OPENAI_API_KEY=
OPENAI_API_BASE=
YANDEX_API_KEY=
YANDEX_API_BASE=
SBER_API_KEY=
username=
password=
```

### Fetch data and prebuilt index

The full corpus and the prebuilt FAISS index live in an S3 bucket:

```bash
python download_data.py    # downloads data/  (parsed articles + reference table)
python download_db.py      # downloads db/    (FAISS index)
```

## Running the stack

```bash
# Terminal 1 — vector service
cd vector_service
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — chatbot API
cd chatbot_app
uvicorn main:app --host 0.0.0.0 --port 8001

# Terminal 3 — Gradio UI
cd chatbot_app
python app.py
```

Open <http://localhost:8517/scinanoai> and log in with `username` /
`password` from `.env`.

A `docker-compose.yml` is also provided for one-command startup:

```bash
docker compose up --build
```

## Repository layout

```
chatbot_app/                 RAG orchestrator + Gradio UI
  ├── chatbot.py             ChatBot — query routing, LLM dispatch, references
  ├── decomposer_agent.py    Zero-shot classifier (corpus vs. general LLM)
  ├── main.py                FastAPI app (/chat, /clear_history)
  └── app.py                 Gradio frontend
vector_service/              FAISS retrieval microservice
notebooks/                   Corpus building + evaluation (used in the article)
src/utils/                   Project-path helpers
download_data.py             Pull data/ from S3
download_db.py               Pull db/  from S3
upload_data.py               Push local data/ to S3
```

## Citation

If SciNanoAI helps your research, please cite the JCIM 2025 article:

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

The repository also ships a [`CITATION.cff`](CITATION.cff) so GitHub's
"Cite this repository" button renders the same metadata automatically.

## License

[MIT](LICENSE) — © the SciNanoAI authors. The published article is © American
Chemical Society and distributed under ACS publication terms.
