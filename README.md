# RAG Benchmark Platform

A runnable Python MVP for benchmarking advanced RAG pipelines across ingestion and query stages.

This project is intentionally modular. Every stage is implemented behind a small interface and registered through a plugin registry, so new chunkers, extractors, embedders, retrievers, rerankers, compressors, validators, and metrics can be added without rewriting the benchmark runner.

## Project status

This repository is an MVP benchmark platform intended for local experimentation and comparative RAG prototyping. APIs, YAML schema fields, built-in component names, default models, metric outputs, Docker service wiring, and persisted artifacts should be treated as experimental and may change in breaking ways between releases. Pin a commit or tag before using results for longitudinal comparisons, and keep benchmark configs with any reported results.

## System requirements

Recommended local resources depend heavily on corpus size, benchmark grid size, and enabled ML components. For the checked-in sample corpus and the smallest quick-start path below, start with:

| Resource | Practical minimum | Recommended for default config | Notes |
|---|---:|---:|---|
| CPU | 4 cores | 8+ cores | Embedding, PDF extraction, OpenSearch, and reranking are CPU-intensive without GPU acceleration. |
| RAM | 8 GB | 16-32 GB | OpenSearch, Qdrant, Ollama, workers, and local models all share memory. Larger corpora and rerankers need more. |
| Disk | 15 GB free | 30+ GB free | Docker images, OpenSearch/Qdrant data, artifacts, Hugging Face caches, and Ollama model files can grow quickly. |

Local ML backends download model weights on first use. SentenceTransformers, CrossEncoder, FlashRank, Jina, BGE-M3, SPLADE, and similar optional components may populate Hugging Face or package-specific caches; Ollama models must be pulled explicitly and can require multiple gigabytes per model. If you are offline, pre-populate these caches or use already-downloaded models.

## Security notes

This project is designed for local development and trusted networks, not direct internet exposure. The Streamlit UI and FastAPI service do not provide production authentication or authorization, and supporting services may use development defaults inside Docker Compose. Review and rotate any credentials in `.env`, service configs, object-storage settings, or database settings before sharing an environment.

Do not publish the Compose stack, Docker socket, API port, UI port, OpenSearch, Qdrant, Postgres, Redis, MinIO, or Ollama directly to the internet. Put any shared deployment behind your own authenticated reverse proxy, network firewall, TLS termination, secret-management process, and resource limits.

## Minimal quick start

The fastest path is to run one benchmark case against the checked-in sample corpus and evaluation file, using the default local SentenceTransformers embedder and Ollama generator. This still requires Docker and an Ollama model pull.

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec ollama ollama pull gpt-oss:20b
python3 - <<'PY'
import json
from pathlib import Path

config = Path('configs/default.yaml').read_text()
config = config.replace('max_combinations: 32', 'max_combinations: 1')
Path('/tmp/ragbench-quickstart.yaml').write_text(config)
print(json.dumps({
    'config_path': '/app/configs/quickstart.yaml',
    'content': config,
}))
PY
```

Save the one-case config through the API, then start the benchmark:

```bash
curl -X PUT http://localhost:8000/configs/content \
  -H 'Content-Type: application/json' \
  -d @<(python3 - <<'PY'
import json
from pathlib import Path
print(json.dumps({
    'config_path': '/app/configs/quickstart.yaml',
    'content': Path('/tmp/ragbench-quickstart.yaml').read_text(),
}))
PY
)

curl -X POST http://localhost:8000/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/quickstart.yaml"}'
```

Use the returned job ID to check status:

```bash
curl http://localhost:8000/benchmarks/<job_id>
```

## Known limitations

- Answer correctness and faithfulness metrics are lightweight lexical placeholders, not semantic LLM judges; use retrieval metrics and manual review before drawing quality conclusions.
- Several embedders, rerankers, parsers, and neural sparse components depend on optional packages, API keys, or downloadable local model weights that may not be present in the base environment.
- Default local runs require Docker Compose services such as Qdrant, OpenSearch, Postgres, Redis, MinIO, and Ollama to be healthy; individual components can fail if their backing service is unavailable.
- Benchmark results are sensitive to model versions, caches, corpus contents, and expanded YAML combinations, so preserve exact configs and commits with any comparison.

## Project links

- [License](LICENSE)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Issue templates / new issue chooser](../../issues/new/choose)

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
| Indexing | Qdrant dense strategies (`dense_hnsw`, `dense_hnsw_tuned`, `dense_exact`, `dense_quantized`, `dense_on_disk`, `dense_payload_indexed`), OpenSearch BM25 strategies (`bm25_standard`, `bm25_field_weighted`, `bm25_custom_analyzer`, `bm25_phrase`) |
| Retrieval | Dense, metadata-filtered dense, BM25, hybrid reciprocal-rank-fusion, query-adaptive hybrid routing |
| Query rewriting | None, HyDE via Ollama, multi-query paraphrases, step-back conceptual queries, decomposition, local synonym expansion |
| Reranking | None, CrossEncoder reranker, FlashRank, RankLLM, ColBERT, Cohere, Jina, diversity/MMR reranker |
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
| Chunkers | `fixed`, `recursive`, `llamaindex_fixed`, `llamaindex_sentence`, `llamaindex_hierarchical`, `llamaindex_semantic`, `llamaindex_structure` | `type`, `chunk_size`, `overlap`; `llamaindex_hierarchical` also accepts `parent_chunk_size`; `llamaindex_semantic` also accepts `breakpoint_percentile_threshold`, `buffer_size`, and `embed_dim` | `chunk_size` defaults to `512`; `overlap` defaults to `0` and may be fractional (`0 <= overlap < 1`) or an absolute count (`0 <= overlap < chunk_size`). LlamaIndex variants use their matching parser when `llama-index-core` is installed and otherwise fall back to deterministic local splitting. `chunk_size`, `overlap`, and `parent_chunk_size` may be scalar values or lists for case expansion. |
| Metadata enrichers | `basic` | `type: basic` | No additional parameters. Adds `char_length` and `word_count` to every chunk. |
| Embeddings | `sentence_transformers`, `openai`, `voyage`, `cohere`, `jina`, `bge_m3`, `bge_m3_sparse`, `splade_sparse`, `bge_m3_hybrid` | `provider`, `model`, `api_key`, `base_url`, `batch_size`, `normalize` / `normalize_embeddings`, `query_prefix`, `document_prefix` / `passage_prefix`, `timeout` | `sentence_transformers` and `bge_m3` run locally with SentenceTransformers; `bge_m3` defaults to `BAAI/bge-m3`. E5 aliases `e5-base-v2` and `e5-large-v2` resolve to `intfloat/e5-base-v2` and `intfloat/e5-large-v2`; E5 models automatically receive `query: ` prefixes for queries and `passage: ` prefixes for chunks unless overridden. Hosted providers call their embedding APIs: `openai` uses `OPENAI_API_KEY` and defaults to `text-embedding-3-small`, `voyage` uses `VOYAGE_API_KEY` and defaults to `voyage-3`, `cohere` uses `COHERE_API_KEY` and defaults to `embed-v4.0`, and `jina` uses `JINA_API_KEY` and defaults to `jina-embeddings-v3`. `base_url` overrides hosted endpoints, `batch_size` defaults to `32`, and optional prefixes are applied before requests. Dense embedders produce vectors suitable for the Qdrant dense-index workflow. `bge_m3_sparse` and `splade_sparse` populate neural sparse vectors, while `bge_m3_hybrid` populates both `embedding` and `sparse_embedding` and returns both dense and sparse query vectors for hybrid dense/sparse retrieval. |
| Indexers | `qdrant`, `opensearch` | `pipeline.ingestion.indexers.dense.type`, `pipeline.ingestion.indexers.dense.strategy`, `pipeline.ingestion.indexers.dense.strategies`, `pipeline.ingestion.indexers.sparse.type`, `pipeline.ingestion.indexers.sparse.strategy`, `pipeline.ingestion.indexers.sparse.strategies`, `pipeline.ingestion.indexers.sparse.synonyms` | v1 invokes the configured dense and sparse indexers, defaulting to `qdrant` and `opensearch`. Qdrant requires non-empty embeddings and uses cosine distance. Dense Qdrant strategy names are `dense_hnsw` (current baseline), `dense_hnsw_tuned` (recall/latency trade-off), `dense_exact` (ground-truth nearest-neighbour baseline with HNSW disabled), `dense_quantized` (memory/cost trade-off), `dense_on_disk` (large corpus behaviour), and `dense_payload_indexed` (filtered retrieval benchmark with payload indexes). OpenSearch BM25 strategy names are `bm25_standard` (plain `text` baseline), `bm25_field_weighted` (boostable `title`, `heading`, and `body` fields), `bm25_custom_analyzer` (lowercase, stopword removal, English stemming, and configurable synonym expansion), and `bm25_phrase` (shingle subfields for phrase matching). Use a scalar `strategy`, a list of indexer objects, or `strategies` to expand benchmark cases across multiple Qdrant or OpenSearch strategies. |

### Query components

| Component | Built-in variants | Parameters | Restrictions / notes |
|---|---|---|---|
| Query rewriters | `none`, `hyde`, `multi_query`, `step_back`, `decompose`, `synonym_expand` | `type`; `multi_query.count`, `multi_query.temperature`; `step_back.temperature`; `decompose.max_subquestions`, `decompose.temperature`; `synonym_expand.synonyms` | All rewriters return `list[str]` so the runner can pass every rewritten query through `embedder.embed_queries(...)` and `retriever.retrieve_batch(...)`. `none` returns the original question. `hyde` uses the top-level `llm.base_url` and `llm.model` to call Ollama and returns the original question plus a hypothetical passage when generation succeeds. `multi_query` uses the LLM to generate diverse paraphrases and prepends the original question; `count` defaults to `4`. `step_back` uses the LLM to add one broader conceptual/background query. `decompose` uses the LLM to add atomic subquestions for compound or multi-hop questions; `max_subquestions` defaults to `4`. LLM-backed rewriters fall back to `[question]` on generation errors. `synonym_expand` does not call an LLM; it accepts either a mapping such as `PTO: ["paid time off", "vacation leave"]` or list groups such as `["SSO", "single sign-on"]`, and returns the original question plus a Boolean-style expanded variant when a configured term is present. |

Example query rewriter configuration:

```yaml
pipeline:
  query:
    rewriters:
      - type: "multi_query"
        count: 4
        temperature: 0.2
      - type: "step_back"
      - type: "decompose"
        max_subquestions: 4
      - type: "synonym_expand"
        synonyms:
          PTO: ["paid time off", "vacation leave"]
          SSO: ["single sign-on", "federated login"]
```
| Retrievers | `dense`, `dense_filtered`, `parent_document`, `bm25`, `hybrid`, `neural_sparse`, `weighted_hybrid`, `multi_query_fusion`, `adaptive_hybrid` | `type`, `metadata_filters` / `metadata_filter`, `payload_filter`, `doc_id`, `doc_ids`, `base_retriever`, `child_top_k`, `child_top_k_multiplier`, `rrf_k` for `hybrid`, `short_query_mode`, `long_query_mode`, `default_mode`, `short_query_max_terms`, `long_query_min_terms` for `adaptive_hybrid` | `dense` requires Qdrant and a query embedding. `dense_filtered` applies a Qdrant payload filter before vector ranking; metadata filter keys without a prefix target `metadata.<key>`, while `doc_id` and `chunk_id` target top-level payload fields. `payload_filter` can pass a raw Qdrant filter object. `parent_document` retrieves `llamaindex_hierarchical` child chunks with `dense` or `dense_filtered`, deduplicates by stored `parent_chunk_id`, and returns the stored `parent_text` as the context chunk while retaining child hit details in metadata. `bm25` requires OpenSearch. `hybrid` calls its configured `sources` and merges results with reciprocal-rank fusion; by default it fetches `dense` and `bm25`, and it can fetch `dense` plus `neural_sparse` when paired with `bge_m3_hybrid` query embeddings and a configured sparse index. `rrf_k` defaults to `60` and must be convertible to a positive integer. `adaptive_hybrid` routes each query by token count: queries at or below `short_query_max_terms` (default `4`) use `short_query_mode` (default `bm25`), queries at or above `long_query_min_terms` (default `12`) use `long_query_mode` (default `hybrid`), and the remaining queries use `default_mode` (default `dense`). Adaptive modes must be one of `dense`, `bm25`, `hybrid`, `neural_sparse`, or `weighted_hybrid`, and returned hits are labeled `adaptive_hybrid:<mode>`. Retrieval depth comes from `max(metrics.retrieval_k)`. |
| Filters | `none` | `type: none` | No additional parameters. Use the `dense_filtered` retriever when metadata constraints should be pushed into Qdrant retrieval instead of applied after retrieval. |
| Rerankers | `none`, `cross_encoder`, `flashrank`, `rankllm`, `colbert`, `cohere`, `jina`, `diversity_mmr` | `type`, `model` for model-backed rerankers; `api_key`, `top_n`, `timeout`, `url` / `base_url` for Cohere; `trust_remote_code`, `max_length`, `device` for local Jina; `lambda_mult` for `diversity_mmr` | `cross_encoder.model` defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` and must be a local or downloadable SentenceTransformers CrossEncoder-compatible model. `flashrank` uses the FlashRank `Ranker`/`RerankRequest` API; `model`, `cache_dir`, and `max_length` are passed to the ranker when provided, and `model` defaults to FlashRank's default model if omitted. `rankllm` uses the Answer.AI `rerankers` RankLLM integration; configure `model` (default `rankllm`), plus optional `api_key` and `lang` for API-backed models. `colbert` uses the Answer.AI `rerankers` ColBERT integration; configure `model` (default `colbert`) plus optional `lang`; it requires installing compatible optional dependencies outside the base Docker image because the `rerankers[transformers]` extra conflicts with the pinned FlagEmbedding dependency set. `cohere` calls Cohere's hosted Rerank API using `api_key` or `COHERE_API_KEY`, defaults to `model: rerank-v3.5`, and posts to `https://api.cohere.com/v2/rerank` unless `url` or `base_url` is set; it defaults `top_n` to the number of retrieved results and preserves any results omitted by the API at the end. `jina` runs an open-source Jina reranker locally with SentenceTransformers, defaults to `model: jinaai/jina-reranker-v2-base-multilingual`, and does not require `JINA_API_KEY`. `diversity_mmr.lambda_mult` defaults to `0.5`; `1.0` preserves relevance-only ordering, while `0.0` maximizes diversity. MMR uses chunk embeddings when present and falls back to token cosine similarity over chunk text. Rerankers replace retriever scores with reranker scores when their backend returns scores and update result ranks. |
| Context compressors | `none`, `top_n` | `type`, `max_chunks` for `top_n` | `top_n.max_chunks` defaults to `5` and must be convertible to a non-negative integer. It keeps the first `max_chunks` ranked results after retrieval, filtering, and reranking. |
| Context assemblers | `default` | `type`, `max_context_chars` | `max_context_chars` defaults to `12000` and must be convertible to a non-negative integer. The assembler adds citation markers in the form `[doc_id::chunk_id]` and stops before adding a block that would exceed the limit. Only the first configured assembler is used. |
| Prompt builders | `default` | `type: default` | No additional parameters. The prompt requires answers to use the exact context citation markers and to say when the answer is unavailable in context. Only the first configured prompt builder is used. |
| Citation validators | `simple` | `type: simple` | No additional parameters. The simple validator counts an expected citation as found only when its string appears verbatim in the generated answer. Only the first configured citation validator is used. |
| Generation | `ollama` | Configured in top-level `llm` | The runner dispatches by `llm.provider`. Built-in generation requires Ollama's `/api/generate` endpoint and a non-streaming response with a `response` field. |

### Metrics components

| Metric component | Parameters | Restrictions / notes |
|---|---|---|
| Retrieval metrics | `metrics.retrieval_k` | Must be a non-empty list of positive integers. Computes `recall@k`, `precision@k`, `mrr@k`, and `ndcg@k` for every `k`. The runner reports both `raw_retrieval_*` metrics for the initial retrieval candidates and `reranked_*` metrics for the pre-compression reranked candidates; the historical unprefixed metric names mirror the reranked pre-compression values for downstream compatibility. It uses `ground_truth_chunk_ids` when present; otherwise it falls back to `ground_truth_doc_ids`. When no retrieval ground truth is available, metric values are reported as `null`; when ground truth exists but no results are returned for a stage, metric values are `0.0` and that stage's failure flag is `true`. |
| Operational metrics | None | Always reports `answer_chars`, `context_chunks`, and `estimated_answer_tokens`. |
| Citation coverage | Ground-truth `expected_citations` | Returns `1.0` when no expected citations are provided; otherwise checks for verbatim citation strings in the answer. |
| Placeholder answer correctness | Ground-truth `expected_answer` | Lexical-overlap placeholder only; not a semantic answer judge. |
| Placeholder faithfulness | Generated answer and assembled context | Lexical support heuristic over answer words longer than four characters; not an LLM-based faithfulness metric. |

### Combination expansion behavior

Chunkers, dense indexer strategies, sparse indexer strategies, embeddings, rewriters, retrievers, rerankers, and compressors participate in Cartesian-product case expansion. Chunker, Qdrant dense strategy, and OpenSearch sparse strategy variants are cycled before the later query-grid variants so capped runs still compare configured ingestion strategies early. The default config includes both native chunkers and the LlamaIndex fixed-size, sentence, hierarchical, semantic, and document-structure chunkers; it expands Qdrant across `dense_hnsw`, `dense_hnsw_tuned`, `dense_exact`, `dense_quantized`, `dense_on_disk`, and `dense_payload_indexed`; and it expands OpenSearch across `bm25_standard`, `bm25_field_weighted`, `bm25_custom_analyzer`, and `bm25_phrase`. Cleaner, extractor, metadata enricher, filter, context assembler, prompt builder, citation validator, LLM, metric, and runtime settings are reused across expanded cases. For `context_assemblers`, `prompt_builders`, and `citation_validators`, v1 uses only the first item in each list.

## Ground-truth QA format

```json
[
  {
    "question": "Can an open-box item and downloadable software both be refunded under the current policy?",
    "expected_answer": "Open-box items qualify for the 14-day refund window when all original accessories are returned, but downloadable software is not refundable unless required by law.",
    "ground_truth_doc_ids": ["refund_windows.md", "refund_exclusions.md"],
    "ground_truth_chunk_ids": [],
    "expected_citations": ["refund_windows.md", "refund_exclusions.md"]
  }
]
```

For v1, retrieval metrics can use `ground_truth_doc_ids`; exact `ground_truth_chunk_ids` are optional. `ground_truth_doc_ids` and `expected_citations` may contain multiple documents when an answer requires evidence from more than one source file.

The checked-in sample evaluation set is split across multiple focused policy files under `data/raw/` instead of one large source document. Several questions require evidence from multiple files, while the corpus still includes archived-but-invalid policy wording, exception-heavy answers, table-only facts, negation, abbreviation disambiguation, and retention/warranty edge cases so pipeline variants can be evaluated on multi-document retrieval as well as direct fact lookup.

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

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, local run instructions, test commands, coding style, and issue/PR expectations. All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please report suspected vulnerabilities privately instead of opening a public issue. See [SECURITY.md](SECURITY.md) for supported versions and vulnerability-reporting instructions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.

## Design notes

This repo deliberately avoids a blind Cartesian product explosion. The runner expands configured variants into benchmark cases, but you should keep early experiments small and controlled. Test retrieval first, then generation. If retrieval fails, answer metrics become noise.

Recommended extension roadmap:

1. LLM-based citation validator
2. RAGAS and DeepEval adapters
3. Additional LangChain and LlamaIndex adapter components
4. Synthetic QA generation
5. Experiment comparison dashboard
6. Distributed worker scaling
7. Dataset versioning and artifact lineage
