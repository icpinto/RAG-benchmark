# RAG Benchmark Platform

A runnable Python MVP for benchmarking advanced RAG pipelines across ingestion and query stages.

This project is intentionally modular. Every stage is implemented behind a small interface and registered through a plugin registry, so new chunkers, extractors, embedders, retrievers, rerankers, compressors, validators, and metrics can be added without rewriting the benchmark runner.

## What v1 supports

### Ingestion pipeline

```text
Raw data
  -> Data extraction
  -> Cleaning / normalization
  -> Chunking
  -> Metadata enrichment
  -> Embedding generation
  -> Indexing into Qdrant / OpenSearch
```

### Query pipeline

```text
User question
  -> Query rewriting / expansion
  -> Retrieval
  -> Filtering
  -> Re-ranking
  -> Context assembly
  -> Prompt construction
  -> Ollama answer generation
  -> Citation / evidence validation
```

### Implemented v1 variants

| Stage | Implemented variants |
|---|---|
| Extraction | TXT, Markdown, PDF, HTML/URL |
| Cleaning | Basic cleaner |
| Chunking | Fixed-size, recursive character chunker |
| Embeddings | SentenceTransformers: MiniLM, BGE small, configurable local HF model |
| Indexing | Qdrant dense index, OpenSearch BM25 index |
| Retrieval | Dense, BM25, hybrid reciprocal-rank-fusion |
| Query rewriting | None, HyDE via Ollama |
| Reranking | None, CrossEncoder reranker |
| Context compression | None, top-N trimming |
| Generation | Ollama, default model `gpt-oss:20b` |
| Metrics | recall@k, precision@k, MRR, NDCG, latency, index size, simple citation validation, placeholder answer metrics |

## Architecture

```text
rag-benchmark-platform/
  docker-compose.yml
  .env.example
  configs/
  data/
  services/
    api/        FastAPI service
    worker/     Celery benchmark worker
    ui/         Streamlit UI
  packages/
    ragbench_core/
      components/
      engine/
      models/
      registry.py
  scripts/
  tests/
```

The core package does not depend on LangChain or LlamaIndex. Those should be added later as adapter components, not as the foundation of the system.

## Run with Docker Compose

```bash
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333/dashboard |
| OpenSearch | http://localhost:9200 |
| MinIO console | http://localhost:9001 |

Ollama is included as a service. Pull the model after containers start:

```bash
docker compose exec ollama ollama pull gpt-oss:20b
```

If your machine already runs Ollama locally, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` and remove/ignore the Ollama service.

## Run a sample benchmark

```bash
cp .env.example .env
docker compose up --build
```

Then upload files and a ground-truth JSON from the Streamlit UI, or call the API:

```bash
curl -X POST http://localhost:8000/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/default.yaml"}'
```

Check job status:

```bash
curl http://localhost:8000/benchmarks/<job_id>
```

## Ground-truth QA format

```json
[
  {
    "question": "What is the refund period?",
    "expected_answer": "Customers can request a refund within 14 days.",
    "ground_truth_doc_ids": ["sample_policy.md"],
    "ground_truth_chunk_ids": [],
    "expected_citations": ["sample_policy.md"]
  }
]
```

For v1, retrieval metrics can use `ground_truth_doc_ids`; exact `ground_truth_chunk_ids` are optional.

## Add a new component

Create a class implementing the relevant interface, then register it.

Example chunker:

```python
from ragbench_core.interfaces import BaseChunker
from ragbench_core.models.schema import Document, Chunk

class MyChunker(BaseChunker):
    name = "my_chunker"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        ...
```

Register it in `ragbench_core/registry.py`:

```python
REGISTRY.register_chunker(MyChunker())
```

Then use it in YAML:

```yaml
pipeline:
  ingestion:
    chunkers:
      - type: my_chunker
```

## Design notes

This repo deliberately avoids a blind Cartesian product explosion. The runner expands configured variants into benchmark cases, but you should keep early experiments small and controlled. Test retrieval first, then generation. If retrieval fails, answer metrics become noise.

Recommended extension roadmap:

1. Semantic chunker
2. Parent-child chunking and parent-document retriever
3. Metadata filters
4. LLM-based citation validator
5. RAGAS and DeepEval adapters
6. LangChain and LlamaIndex adapter components
7. Synthetic QA generation
8. Experiment comparison dashboard
9. Distributed worker scaling
10. Dataset versioning and artifact lineage
