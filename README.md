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
| Chunking | Fixed-size, recursive character chunker, LlamaIndex fixed-size, sentence, hierarchical/parent-child, semantic, and document-structure splitters |
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

The core package keeps its own small interfaces while exposing LlamaIndex chunking as adapter components, rather than making LlamaIndex the foundation of the system.

## Run with Docker Compose

```bash
docker compose up --build
```

Host-accessible services:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

Supporting services such as Postgres, Redis, Qdrant, OpenSearch, MinIO, and Ollama are available only on the Docker Compose network and are not published to the host by default.

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

Then upload files and a ground-truth JSON from the Streamlit UI. The UI also includes an **Edit config YAML** panel where you can load `/app/configs/default.yaml`, edit the YAML, save it back to the shared `configs/` volume, and start a benchmark using the saved config path.

You can also call the API directly:

```bash
curl -X POST http://localhost:8000/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/default.yaml"}'
```

Config files can be listed, loaded, and saved through the API:

```bash
curl http://localhost:8000/configs
curl 'http://localhost:8000/configs/content?config_path=/app/configs/default.yaml'
curl -X PUT http://localhost:8000/configs/content \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/default.yaml","content":"experiment:\n  name: edited_benchmark\n"}'
```

Check job status:

```bash
curl http://localhost:8000/benchmarks/<job_id>
```


## Component configuration parameters

The benchmark YAML is organized around top-level experiment, data, pipeline, LLM, metrics, and runtime sections. Component entries use a `type` key for registry-backed pipeline stages, except embedders and generators, which use `provider`.

### Top-level sections

| Section | Parameter | Required | Default | Restrictions / notes |
|---|---|---:|---|---|
| `experiment` | `name` | No | `benchmark` when persisted; `exp` for case IDs | String used for run labels and stable case IDs. |
| `experiment` | `description` | No | None | Free-form string. |
| `data` | `input_dir` | Yes | None | Directory containing local source files. Local extractors currently process `.txt`, `.md`, `.markdown`, and `.pdf` files. |
| `data` | `remote_urls_file` | No | None | Text file containing one URL per line. URLs are only handled by the `html_url` extractor and must start with `http://` or `https://`. |
| `data` | `ground_truth_file` | Yes | None | JSON file in the ground-truth QA format shown below. |
| `llm` | `provider` | Yes | None | Must match a registered generator. Built-in value: `ollama`. |
| `llm` | `model` | No | `gpt-oss:20b` in the Ollama generator | Ollama model name; the model must already be pulled or available to the configured Ollama server. Supports `${ENV_VAR:default}` substitution. |
| `llm` | `base_url` | No | `http://ollama:11434` | Ollama API base URL. Trailing slashes are stripped. Supports `${ENV_VAR:default}` substitution. |
| `llm` | `temperature` | No | `0` | Passed directly to Ollama under `options.temperature`; use a numeric value supported by Ollama. |
| `llm` | `timeout` | No | `240` | Request timeout in seconds; must be convertible to an integer. |
| `metrics` | `retrieval_k` | No | `[3, 5, 10]` for metrics, `[10]` for retrieval limit fallback | List of positive integers. The runner retrieves up to `max(retrieval_k)`, so this list must not be empty. |
| `metrics` | `enabled` | No | Informational in v1 | Listed metric names are documented intent; v1 computes all registered metrics regardless of this list. |
| `runtime` | `max_combinations` | No | `32` | Positive integer cap on expanded benchmark cases. The expander stops when this limit is reached. |
| `runtime` | `collection_prefix` | No | `ragbench` | Prefix for Qdrant/OpenSearch index names; final names are lowercased and hyphens are converted to underscores. |
| `runtime` | `artifact_dir` | No | `/app/artifacts` in default config/settings | Reserved for artifacts; keep it writable if future components emit files. |

### Ingestion components

| Component | Built-in variants | Parameters | Restrictions / notes |
|---|---|---|---|
| Extractors | `txt`, `markdown`, `pdf`, `html_url` | In YAML, `pipeline.ingestion.extractors` is a list of extractor names, not objects. | `txt` only accepts `.txt`; `markdown` accepts `.md` and `.markdown`; `pdf` accepts `.pdf`; `html_url` only accepts sources beginning with `http://` or `https://`. Unknown names cause registry lookup failures. |
| Cleaners | `basic` | `type: basic` | No additional parameters. The basic cleaner removes NUL characters, collapses spaces/tabs, limits repeated blank lines, and strips leading/trailing whitespace. |
| Chunkers | `fixed`, `recursive`, `llamaindex_fixed`, `llamaindex_sentence`, `llamaindex_hierarchical`, `llamaindex_semantic`, `llamaindex_structure` | `type`, `chunk_size`, `overlap`; `llamaindex_hierarchical` also accepts `parent_chunk_size`; `llamaindex_semantic` also accepts `breakpoint_percentile_threshold`, `buffer_size`, and `embed_dim` | `chunk_size` defaults to `512` and must be convertible to an integer greater than `0`. `overlap` defaults to `0`; if it is a float, it is treated as a fraction of `chunk_size`; otherwise it is treated as an absolute character count. `llamaindex_fixed` uses LlamaIndex `TokenTextSplitter` for fixed token-style chunks. `llamaindex_sentence` uses LlamaIndex `SentenceSplitter` to prefer sentence boundaries. `llamaindex_hierarchical` uses LlamaIndex `HierarchicalNodeParser` and stores child chunks with `parent_chunk_id`, `parent_chunk_index`, `parent_text`, and `chunk_role` metadata for parent-child retrieval experiments. `llamaindex_semantic` uses LlamaIndex `SemanticSplitterNodeParser` with a mock embedding model so benchmark embedders remain independently configurable. `llamaindex_structure` uses LlamaIndex `MarkdownNodeParser` to preserve document sections and heading metadata. LlamaIndex adapters fall back to local deterministic splitting when the needed `llama-index-core` parser is not installed, but Docker installs `llama-index-core` from `requirements/base.txt`. Use `0 <= overlap < 1` for fractional overlap, or an integer `0 <= overlap < chunk_size`; values that make `overlap_chars >= chunk_size` can create tiny steps or excessive duplicated context. `chunk_size`, `overlap`, and `parent_chunk_size` may be scalar values or lists; lists are expanded into combinations. |
| Metadata enrichers | `basic` | `type: basic` | No additional parameters. Adds `char_length` and `word_count` to every chunk. |
| Embeddings | `sentence_transformers` | `provider`, `model`, `batch_size` | `provider` must be `sentence_transformers`. `model` is required and must be a local or downloadable SentenceTransformers-compatible model. `batch_size` defaults to `32` and must be convertible to a positive integer. All configured embedding models should produce vectors suitable for the same Qdrant dense-index workflow. |
| Indexers | `qdrant`, `opensearch` | `pipeline.ingestion.indexers.dense.type`, `pipeline.ingestion.indexers.sparse.type` | v1 always invokes the built-in `qdrant` dense indexer and `opensearch` sparse indexer in the runner. The `type` fields document the intended backend but are not used to dispatch alternate indexers yet. Qdrant requires non-empty embeddings and uses cosine distance; OpenSearch creates a text/BM25 mapping. |

### Query components

| Component | Built-in variants | Parameters | Restrictions / notes |
|---|---|---|---|
| Query rewriters | `none`, `hyde` | `type` | `none` returns the original question. `hyde` uses the top-level `llm.base_url` and `llm.model` to call Ollama and returns the original question plus a hypothetical passage when generation succeeds. HyDE has a fixed 120-second request timeout and fixed generation temperature `0`. |
| Retrievers | `dense`, `bm25`, `hybrid` | `type`, `rrf_k` for `hybrid` | `dense` requires Qdrant and a query embedding. `bm25` requires OpenSearch. `hybrid` calls both retrievers and merges results with reciprocal-rank fusion; `rrf_k` defaults to `60` and must be convertible to a positive integer. Retrieval depth comes from `max(metrics.retrieval_k)`. |
| Filters | `none` | `type: none` | No additional parameters. Metadata filtering is not implemented in v1. |
| Rerankers | `none`, `cross_encoder` | `type`, `model` for `cross_encoder` | `cross_encoder.model` defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` and must be a local or downloadable SentenceTransformers CrossEncoder-compatible model. Reranking scores replace retriever scores and results are sorted descending. |
| Context compressors | `none`, `top_n` | `type`, `max_chunks` for `top_n` | `top_n.max_chunks` defaults to `5` and must be convertible to a non-negative integer. It keeps the first `max_chunks` ranked results after retrieval, filtering, and reranking. |
| Context assemblers | `default` | `type`, `max_context_chars` | `max_context_chars` defaults to `12000` and must be convertible to a non-negative integer. The assembler adds citation markers in the form `[doc_id::chunk_id]` and stops before adding a block that would exceed the limit. Only the first configured assembler is used. |
| Prompt builders | `default` | `type: default` | No additional parameters. The prompt requires answers to use the exact context citation markers and to say when the answer is unavailable in context. Only the first configured prompt builder is used. |
| Citation validators | `simple` | `type: simple` | No additional parameters. The simple validator counts an expected citation as found only when its string appears verbatim in the generated answer. Only the first configured citation validator is used. |
| Generation | `ollama` | Configured in top-level `llm` | The runner dispatches by `llm.provider`. Built-in generation requires Ollama's `/api/generate` endpoint and a non-streaming response with a `response` field. |

### Metrics components

| Metric component | Parameters | Restrictions / notes |
|---|---|---|
| Retrieval metrics | `metrics.retrieval_k` | Computes `recall@k`, `precision@k`, `mrr@k`, and `ndcg@k` for every `k`. It uses `ground_truth_chunk_ids` when present; otherwise it falls back to `ground_truth_doc_ids`. |
| Operational metrics | None | Always reports `answer_chars`, `context_chunks`, and `estimated_answer_tokens`. |
| Citation coverage | Ground-truth `expected_citations` | Returns `1.0` when no expected citations are provided; otherwise checks for verbatim citation strings in the answer. |
| Placeholder answer correctness | Ground-truth `expected_answer` | Lexical-overlap placeholder only; not a semantic answer judge. |
| Placeholder faithfulness | Generated answer and assembled context | Lexical support heuristic over answer words longer than four characters; not an LLM-based faithfulness metric. |

### Combination expansion behavior

Only chunkers, embeddings, rewriters, retrievers, rerankers, and compressors participate in Cartesian-product case expansion. Chunker variants are cycled before the later query-grid variants so capped runs still compare configured chunking strategies early. The default config includes both native chunkers and the LlamaIndex fixed-size, sentence, hierarchical, semantic, and document-structure chunkers. Cleaner, extractor, metadata enricher, indexer, filter, context assembler, prompt builder, citation validator, LLM, metric, and runtime settings are reused across expanded cases. For `context_assemblers`, `prompt_builders`, and `citation_validators`, v1 uses only the first item in each list.

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

1. Parent-document retriever that expands `llamaindex_hierarchical` child hits with stored parent context
2. Metadata filters
4. LLM-based citation validator
5. RAGAS and DeepEval adapters
6. Additional LangChain and LlamaIndex adapter components
7. Synthetic QA generation
8. Experiment comparison dashboard
9. Distributed worker scaling
10. Dataset versioning and artifact lineage
