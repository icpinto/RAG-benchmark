"""UI metadata only: no model imports, provider calls, or configuration coercion.

Defaults are hints, never a replacement for the defaults in component code.
Sweep flags follow config_expander, rather than treating every array as a grid.
"""

from copy import deepcopy


def field(key, kind="string", **options):
    return {
        "key": key,
        "label": key.replace("_", " ").capitalize(),
        "type": kind,
        **options,
    }


def component(name, fields=(), help=None):
    return {"name": name, "fields": list(fields), **({"help": help} if help else {})}


def stage(
    key, label, path, components, mode="variants", discriminator="type", help=None
):
    return {
        "key": key,
        "label": label,
        "path": path.split("."),
        "mode": mode,
        "discriminator": discriminator,
        "components": components,
        **({"help": help} if help else {}),
    }


def obj(key, fields=(), **options):
    return field(key, "object", fields=list(fields), **options)


def array(key, kind="string", **options):
    return field(key, "array", items=field("item", kind), **options)


def number(key, default=None, **options):
    return field(
        key,
        "number",
        **({"default": default} if default is not None else {}),
        **options,
    )


def flag(key, default=False, **options):
    return field(key, "boolean", default=default, **options)


def _catalog():
    chunk = [
        number("chunk_size", 512, minimum=1, sweepable=True),
        number(
            "overlap",
            0,
            minimum=0,
            sweepable=True,
            help="Fraction below 1, otherwise character count.",
        ),
    ]
    prefixes = [
        field("document_prefix", aliases=["passage_prefix"]),
        field("query_prefix"),
    ]
    embedding = [
        field("model", required=True),
        number("batch_size", 32, minimum=1),
        flag("normalize", True, aliases=["normalize_embeddings"]),
        *prefixes,
    ]
    hosted = [
        field("model"),
        field(
            "api_key",
            secret=True,
            help="An environment reference such as ${COHERE_API_KEY} is supported. Provider environment variables are used when omitted.",
        ),
        field("base_url"),
        number("timeout", 30),
        number("batch_size", 32, minimum=1),
        flag("normalize", False, aliases=["normalize_embeddings"]),
        *prefixes,
    ]
    sparse = [
        field("model", default="BAAI/bge-m3"),
        flag("use_fp16", True),
        number("batch_size", 32, minimum=1),
        number("sparse_top_k", minimum=1),
    ]
    dense = [number("hnsw_ef", aliases=["search_ef"], minimum=1)]
    filters = [
        obj(
            "metadata_filters",
            aliases=["metadata_filter"],
            help="Metadata keys map to values, lists, or range conditions such as gte/lte.",
        ),
        obj(
            "payload_filter",
            help="Qdrant filter object with must, should, and must_not conditions.",
        ),
        field("doc_id"),
        array("doc_ids"),
    ]
    neural = [
        field("backend", default="opensearch", choices=["opensearch", "qdrant"]),
        field("field", default="sparse_embedding"),
        number("query_terms", 256, minimum=1),
        field("collection_name"),
        field("vector_name", default="neural_sparse"),
    ]
    fusion = [
        number("rrf_k", 60, minimum=1),
        array("sources", default=["dense", "bm25"], aliases=["retrievers"]),
        obj("neural_sparse", neural),
    ]
    weighted = [
        number("rrf_k", 60, minimum=1),
        obj(
            "weights",
            [number("dense", 1), number("bm25", 1), number("neural_sparse", 1)],
        ),
        obj("neural_sparse", neural),
    ]
    hnsw = [
        number("m", minimum=0, sweepable=True),
        number("ef_construct", minimum=1, sweepable=True),
        number("full_scan_threshold", minimum=0, sweepable=True),
        number("max_indexing_threads", minimum=0, sweepable=True),
        flag("on_disk", sweepable=True),
        number("payload_m", minimum=0, sweepable=True),
    ]
    quantization = [
        field("type", default="int8", choices=["int8"], sweepable=True),
        number("quantile", minimum=0, maximum=1, sweepable=True),
        flag("always_ram", sweepable=True),
    ]
    qdrant_strategies = [
        "dense_hnsw",
        "dense_hnsw_tuned",
        "dense_exact",
        "dense_quantized",
        "dense_on_disk",
        "dense_payload_indexed",
        "dense_hnsw_custom",
    ]
    bm25_strategies = [
        "bm25_standard",
        "bm25_field_weighted",
        "bm25_custom_analyzer",
        "bm25_phrase",
    ]
    cache = [
        flag("enabled"),
        field("backend", default="file", choices=["file", "redis"]),
        field("path"),
        field("redis_url", secret=True),
    ]
    sections = [
        {
            "key": "experiment",
            "label": "Experiment",
            "fields": [field("name", default="benchmark"), field("description")],
        },
        {
            "key": "data",
            "label": "Dataset paths",
            "fields": [
                field("input_dir", required=True),
                field("remote_urls_file"),
                field("ground_truth_file", required=True),
            ],
        },
        {
            "key": "metrics",
            "label": "Evaluation metrics",
            "fields": [
                array(
                    "retrieval_k",
                    "integer",
                    default=[3, 5, 10],
                    help="Non-empty positive integers. Also controls retrieval depth.",
                ),
                array(
                    "enabled",
                    help="Informational in the current engine: all metrics are computed.",
                ),
            ],
        },
        {
            "key": "runtime",
            "label": "Runtime",
            "fields": [
                number("max_combinations", 32, minimum=1),
                field("collection_prefix", default="ragbench"),
                field("artifact_dir", default="/app/artifacts"),
                obj("embedding_cache", cache),
                obj("retrieval_cache", cache),
                obj(
                    "pricing",
                    [
                        number("generation_input_per_1m", minimum=0),
                        number("generation_output_per_1m", minimum=0),
                        number("embedding_per_1m", minimum=0),
                    ],
                ),
                obj(
                    "exports",
                    [
                        flag("sqlite", True),
                        flag("parquet"),
                        flag("mlflow"),
                        flag("wandb"),
                        field("wandb_project", default="ragbench"),
                    ],
                ),
            ],
        },
    ]
    stages = [
        stage(
            "extractors",
            "Document extraction",
            "pipeline.ingestion.extractors",
            [
                component("txt"),
                component("markdown"),
                component("pdf"),
                component("html_url"),
                component(
                    "docling",
                    [
                        field(
                            "export_format",
                            default="markdown",
                            choices=["markdown", "text"],
                        ),
                        flag("ignore_errors"),
                        field(
                            "image_descriptions",
                            "value",
                            default=False,
                            help="Boolean or an object with enabled. Uses descriptions already attached by Docling; no vision API call.",
                        ),
                    ],
                ),
            ],
            mode="extractors",
            help="Extractor options are stored in ingestion.extractor_configs. The current case expander does not forward that mapping to cases; changing these options does not change execution until the engine supports forwarding.",
        ),
        stage(
            "cleaners", "Cleaning", "pipeline.ingestion.cleaners", [component("basic")]
        ),
        stage(
            "chunkers",
            "Chunking",
            "pipeline.ingestion.chunkers",
            [
                component(name, chunk)
                for name in [
                    "fixed",
                    "recursive",
                    "llamaindex_fixed",
                    "llamaindex_sentence",
                    "llamaindex_structure",
                ]
            ]
            + [
                component(
                    "llamaindex_hierarchical",
                    [
                        *chunk,
                        number("child_chunk_size", minimum=1),
                        number(
                            "parent_chunk_size",
                            minimum=1,
                            help="Defaults to twice the child size. This field is not expanded as a sweep.",
                        ),
                    ],
                ),
                component(
                    "llamaindex_semantic",
                    [
                        *chunk,
                        number(
                            "breakpoint_percentile_threshold",
                            95,
                            minimum=0,
                            maximum=100,
                        ),
                        number("buffer_size", 1, minimum=1),
                        number("embed_dim", 384, minimum=1),
                    ],
                ),
            ],
        ),
        stage(
            "metadata_enrichers",
            "Metadata enrichment",
            "pipeline.ingestion.metadata_enrichers",
            [component("basic")],
        ),
        stage(
            "embedders",
            "Embedding models",
            "pipeline.ingestion.embeddings",
            [
                component("sentence_transformers", embedding),
                component(
                    "bge_m3",
                    [
                        {**embedding[0], "required": False, "default": "BAAI/bge-m3"},
                        *embedding[1:],
                    ],
                ),
                component(
                    "openai",
                    [{**hosted[0], "default": "text-embedding-3-small"}, *hosted[1:]],
                ),
                component(
                    "voyage", [{**hosted[0], "default": "voyage-3"}, *hosted[1:]]
                ),
                component(
                    "cohere", [{**hosted[0], "default": "embed-v4.0"}, *hosted[1:]]
                ),
                component(
                    "jina",
                    [
                        {**hosted[0], "default": "jina-embeddings-v3"},
                        *hosted[1:],
                        field("query_task", default="retrieval.query"),
                        field("document_task", default="retrieval.passage"),
                    ],
                ),
                component("bge_m3_sparse", sparse),
                component("bge_m3_hybrid", [*sparse, flag("normalize")]),
                component(
                    "splade_sparse",
                    [
                        field(
                            "model", default="naver/splade-cocondenser-ensembledistil"
                        ),
                        number("batch_size", 32, minimum=1),
                        number("max_length", 512, minimum=1),
                        number("sparse_top_k", minimum=1),
                    ],
                ),
            ],
            discriminator="provider",
        ),
        stage(
            "dense_indexer",
            "Dense index",
            "pipeline.ingestion.indexers.dense",
            [
                component(
                    "qdrant",
                    [
                        field(
                            "strategy", default="dense_hnsw", choices=qdrant_strategies
                        ),
                        array(
                            "strategies",
                            "value",
                            help="Strings or strategy override objects; each entry expands into cases.",
                        ),
                        number("hnsw_m", minimum=0, aliases=["m"], sweepable=True),
                        number("ef_construct", minimum=1, sweepable=True),
                        number(
                            "search_ef", minimum=1, aliases=["hnsw_ef"], sweepable=True
                        ),
                        flag("on_disk", sweepable=True),
                        flag(
                            "on_disk_payload",
                            aliases=["payload_on_disk"],
                            sweepable=True,
                        ),
                        flag("hnsw_on_disk", sweepable=True),
                        obj("hnsw_config", hnsw),
                        obj("quantization_config", quantization, sweepable=True),
                        field(
                            "payload_indexes",
                            "array",
                            items=obj(
                                "item",
                                [
                                    field("field_name"),
                                    field(
                                        "field_schema",
                                        choices=[
                                            "keyword",
                                            "integer",
                                            "float",
                                            "bool",
                                            "text",
                                            "geo",
                                            "datetime",
                                        ],
                                    ),
                                ],
                            ),
                        ),
                    ],
                )
            ],
            mode="indexer",
        ),
        stage(
            "sparse_indexer",
            "Sparse index",
            "pipeline.ingestion.indexers.sparse",
            [
                component(
                    "opensearch",
                    [
                        field(
                            "strategy", default="bm25_standard", choices=bm25_strategies
                        ),
                        array("strategies", "value"),
                        array("synonyms"),
                    ],
                ),
                component(
                    "opensearch_neural_sparse",
                    [
                        field(
                            "strategy",
                            default="neural_sparse_rank_features",
                            choices=["neural_sparse_rank_features"],
                        ),
                        array("strategies", "value"),
                        field("field", default="sparse_embedding"),
                    ],
                ),
                component(
                    "qdrant_sparse",
                    [
                        field("collection_name"),
                        field("vector_name", default="neural_sparse"),
                        flag("on_disk"),
                    ],
                ),
            ],
            mode="indexer",
            help="Omitting dense or sparse while the other is present disables that backend. An empty indexers mapping uses both engine defaults.",
        ),
        stage(
            "rewriters",
            "Query rewriting",
            "pipeline.query.rewriters",
            [
                component("none"),
                component("hyde"),
                component(
                    "multi_query",
                    [
                        number("count", 4, minimum=1, aliases=["num_queries"]),
                        number("temperature", 0.2),
                    ],
                ),
                component("step_back", [number("temperature", 0)]),
                component(
                    "decompose",
                    [
                        number("max_subquestions", 4, minimum=1),
                        number("temperature", 0),
                    ],
                ),
                component("synonym_expand", [obj("synonyms")]),
            ],
        ),
        stage(
            "retrievers",
            "Retrieval",
            "pipeline.query.retrievers",
            [
                component("dense", dense),
                component("dense_filtered", [*dense, *filters]),
                component("bm25"),
                component(
                    "parent_document",
                    [
                        field(
                            "base_retriever",
                            default="dense",
                            choices=["dense", "dense_filtered"],
                        ),
                        number("child_top_k_multiplier", 4, minimum=1),
                        number("child_top_k", minimum=1),
                        *dense,
                        *filters,
                    ],
                ),
                component("hybrid", [*dense, *fusion]),
                component("neural_sparse", neural),
                component("weighted_hybrid", [*dense, *weighted]),
                component("multi_query_fusion"),
            ],
        ),
    ]
    # Shared nested retriever options are passed intact to the selected base retriever.
    stages[-1]["components"][-1] = component(
        "multi_query_fusion",
        [
            field(
                "base_retriever",
                default="dense",
                choices=[
                    "dense",
                    "dense_filtered",
                    "bm25",
                    "hybrid",
                    "neural_sparse",
                    "weighted_hybrid",
                ],
            ),
            field("fusion", default="rrf", choices=["rrf"]),
            *dense,
            *filters,
            *fusion,
            obj(
                "weights",
                [number("dense", 1), number("bm25", 1), number("neural_sparse", 1)],
            ),
            *neural,
        ],
    )
    stages[-1]["components"].append(
        component(
            "adaptive_hybrid",
            [
                field(
                    "short_query_mode",
                    default="bm25",
                    choices=[
                        "dense",
                        "bm25",
                        "hybrid",
                        "neural_sparse",
                        "weighted_hybrid",
                    ],
                ),
                field(
                    "long_query_mode",
                    default="hybrid",
                    choices=[
                        "dense",
                        "bm25",
                        "hybrid",
                        "neural_sparse",
                        "weighted_hybrid",
                    ],
                ),
                field(
                    "default_mode",
                    default="dense",
                    choices=[
                        "dense",
                        "bm25",
                        "hybrid",
                        "neural_sparse",
                        "weighted_hybrid",
                    ],
                ),
                number("short_query_max_terms", 4, minimum=0),
                number("long_query_min_terms", 12, minimum=1),
                *dense,
                *fusion,
                obj(
                    "weights",
                    [number("dense", 1), number("bm25", 1), number("neural_sparse", 1)],
                ),
            ],
        )
    )
    stages += [
        stage(
            "filters", "Result filtering", "pipeline.query.filters", [component("none")]
        ),
        stage(
            "rerankers",
            "Reranking",
            "pipeline.query.rerankers",
            [
                component("none"),
                component(
                    "cross_encoder",
                    [field("model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")],
                ),
                component(
                    "flashrank",
                    [
                        field("model"),
                        field("cache_dir"),
                        number("max_length", minimum=1),
                    ],
                ),
                *[
                    component(
                        name,
                        [
                            field("model", default=name),
                            field("api_key", secret=True),
                            field("lang"),
                        ],
                    )
                    for name in ["rankllm", "colbert"]
                ],
                component(
                    "cohere",
                    [
                        field("model", default="rerank-v3.5"),
                        field("api_key", secret=True),
                        field("url"),
                        field("base_url"),
                        number("top_n", minimum=1),
                        number("max_chunks_per_doc", minimum=1),
                        flag("return_documents"),
                        number("timeout", 30),
                    ],
                ),
                component(
                    "jina",
                    [
                        field(
                            "model", default="jinaai/jina-reranker-v2-base-multilingual"
                        ),
                        flag("trust_remote_code", True),
                        number("max_length", minimum=1),
                        field("device"),
                    ],
                ),
                component(
                    "diversity_mmr",
                    [
                        number(
                            "lambda_mult", 0.5, aliases=["lambda"], minimum=0, maximum=1
                        )
                    ],
                ),
            ],
        ),
        stage(
            "compressors",
            "Context compression",
            "pipeline.query.compressors",
            [
                component("none"),
                component("top_n", [number("max_chunks", 5, minimum=0)]),
            ],
        ),
        stage(
            "assemblers",
            "Context assembly",
            "pipeline.query.context_assemblers",
            [component("default", [number("max_context_chars", 12000, minimum=0)])],
            help="The current engine uses only the first entry.",
        ),
        stage(
            "prompt_builders",
            "Prompt construction",
            "pipeline.query.prompt_builders",
            [component("default")],
            help="The current engine uses only the first entry.",
        ),
        stage(
            "citation_validators",
            "Citation validation",
            "pipeline.query.citation_validators",
            [component("simple")],
            help="The current engine uses only the first entry.",
        ),
        stage(
            "generators",
            "Answer generation",
            "llm",
            [
                component(
                    "ollama",
                    [
                        field("model", default="gpt-oss:20b"),
                        field("base_url", default="http://ollama:11434"),
                        number("temperature", 0),
                        number("timeout", 240, minimum=1),
                    ],
                )
            ],
            mode="single",
            discriminator="provider",
        ),
    ]
    hosted_urls = {
        "openai": "https://api.openai.com/v1",
        "voyage": "https://api.voyageai.com/v1",
        "cohere": "https://api.cohere.com/v2",
        "jina": "https://api.jina.ai/v1",
    }
    for definition in next(s for s in stages if s["key"] == "embedders")["components"]:
        if definition["name"] in hosted_urls:
            next(f for f in definition["fields"] if f["key"] == "base_url")[
                "default"
            ] = hosted_urls[definition["name"]]
    metric_components = [
        component(
            "retrieval_metrics", [array("retrieval_k", "integer", default=[3, 5, 10])]
        ),
        component(
            "operational",
            [
                obj(
                    "pricing",
                    next(
                        f["fields"]
                        for s in sections
                        if s["key"] == "runtime"
                        for f in s["fields"]
                        if f["key"] == "pricing"
                    ),
                ),
                field("generation_model"),
                field("embedding_model"),
            ],
        ),
        component("citation_coverage"),
        component("answer_correctness_placeholder"),
        component("faithfulness_placeholder"),
    ]
    next(s for s in sections if s["key"] == "metrics")["fields"].extend(
        [
            field(
                "generation_model",
                help="Optional pricing model fallback when result metadata has no generation model.",
            ),
            field(
                "embedding_model",
                help="Optional pricing model fallback when result metadata has no embedding model.",
            ),
        ]
    )
    return {
        "version": 1,
        "sections": sections,
        "stages": stages,
        "metric_components": metric_components,
    }


def get_config_schema():
    """Return a fresh catalog so callers cannot mutate shared metadata."""
    return deepcopy(_catalog())
