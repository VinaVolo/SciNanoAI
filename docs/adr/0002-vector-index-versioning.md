# ADR 0002: Vector index manifest & versioning

**Status:** accepted (2026-05)

## Context

The legacy `VectorDatabase` called `FAISS.load_local(..., allow_dangerous_deserialization=True)`
without any check that the embedding model declared on the service matched the
one used to build the index. If a researcher swapped `intfloat/multilingual-e5-large`
for a different model, retrieval would silently return wrong neighbours.

Separately, the production index was built out-of-band in a Jupyter notebook
(`notebooks/parse_papers.ipynb` + `create_vector_db.ipynb`) — no code in the
service explained how to rebuild it.

## Decision

Each FAISS index directory ships with a sibling `manifest.json`:

```json
{
  "embedding_model": "intfloat/multilingual-e5-large",
  "dimension": 1024
}
```

`VectorRepository.load()` parses the manifest and refuses to start if the
configured `VECTOR_EMBEDDING_MODEL` does not match. Indexes built before this
ADR have no manifest — the loader is backwards-compatible and only enforces the
check when the file exists.

The PDF ingest pipeline (`scinanoai.vector_service.ingest.pdf`) writes the
manifest on every full rebuild.

## Consequences

- Embedding-model mismatch fails fast at service startup, not silently during
  inference.
- The PDF rebuild path is now a documented, scriptable CLI:
  `scinanoai-ingest-pdf --parsed-pages data/parsed_pages --db-path db/...`.
- Future work: signed manifest with SHA256 of the index file (so we can
  detect corruption or tampering of the pickle artefact).
