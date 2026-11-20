"""Bounded search spaces and executable compatibility checks, independent of LLMs."""

from __future__ import annotations

import math
from copy import deepcopy

from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.persistence import stable_hash
from ragbench_core.engine.snapshots import validate_config

SPACE_VERSION = "ragbench-search-v1"
SPACE = {
    "chunkers": [
        "fixed",
        "recursive",
        "llamaindex_sentence",
        "llamaindex_structure",
        "llamaindex_hierarchical",
    ],
    "chunk_sizes": [256, 512, 1024],
    "overlaps": [0, 0.1, 0.2],
    "retrievers": ["dense", "bm25", "hybrid", "parent_document"],
    "rewriters": ["none", "multi_query", "step_back", "hyde", "decompose"],
    "rerankers": ["none", "cross_encoder", "flashrank", "diversity_mmr"],
    "compressors": ["none", "top_n"],
    "max_chunks": [3, 5, 10],
    "max_context_chars": [6000, 12000, 24000],
    "dense_strategies": ["dense_hnsw", "dense_exact", "dense_quantized"],
    "sparse_strategies": ["bm25_standard", "bm25_phrase"],
}


def concrete_config(template, case):
    result = deepcopy(template)
    ing, qry = case.ingestion_config, case.query_config
    result["pipeline"]["ingestion"] = {
        **{
            k: deepcopy(ing[k])
            for k in (
                "extractors",
                "extractor_configs",
                "cleaners",
                "metadata_enrichers",
                "indexers",
            )
        },
        "chunkers": [deepcopy(ing["chunker"])],
        "embeddings": [deepcopy(ing["embedder"])],
    }
    result["pipeline"]["query"] = {
        plural: [deepcopy(qry[singular])]
        for plural, singular in (
            ("rewriters", "rewriter"),
            ("retrievers", "retriever"),
            ("rerankers", "reranker"),
            ("compressors", "compressor"),
            ("context_assemblers", "context_assembler"),
            ("prompt_builders", "prompt_builder"),
            ("citation_validators", "citation_validator"),
        )
    }
    result["pipeline"]["query"]["filters"] = deepcopy(qry["filters"])
    result.setdefault("runtime", {})["max_combinations"] = 1
    result.pop("_snapshot", None)
    result.pop("input_selection", None)
    return result


def candidate_hash(config):
    return stable_hash({"pipeline": config["pipeline"], "llm": config["llm"]})


def get_search_space(template, allowed=None):
    result = {
        "version": SPACE_VERSION,
        **deepcopy(SPACE),
        "embeddings": deepcopy(template["pipeline"]["ingestion"].get("embeddings", [])),
    }
    for key, names in (allowed or {}).items():
        if key not in {
            "chunkers",
            "retrievers",
            "rewriters",
            "rerankers",
            "compressors",
        }:
            raise ValueError(f"Unsupported allowlist: {key}")
        result[key] = [name for name in result[key] if name in names]
        if not result[key]:
            raise ValueError(f"No allowed {key}")
    return result


def validate_candidate(
    config, baseline, space, registry=None, *, allow_prompt_presets=False
):
    validate_config(config)
    expanded = deepcopy(config)
    expanded.setdefault("runtime", {})["max_combinations"] = 2
    cases = expand_cases(expanded)
    if len(cases) != 1:
        raise ValueError("A candidate must contain exactly one concrete pipeline")
    if config.get("llm") != baseline.get("llm"):
        raise ValueError("The answer model and generation settings are fixed")
    ing, qry = cases[0].ingestion_config, cases[0].query_config
    original = expand_cases(baseline)[0]
    for key in ("extractors", "extractor_configs", "cleaners", "metadata_enrichers"):
        if ing[key] != original.ingestion_config[key]:
            raise ValueError(f"Ingestion stage is fixed: {key}")
    for key in ("citation_validator", "filters"):
        if qry[key] != original.query_config[key]:
            raise ValueError(f"Query stage is fixed: {key}")
    if qry["prompt_builder"] != original.query_config["prompt_builder"]:
        from ragbench_core.components.context import PROMPT_PRESETS, preset_config

        prompt = qry["prompt_builder"]
        if (
            not allow_prompt_presets
            or prompt.get("preset_id") not in PROMPT_PRESETS
            or prompt != preset_config(prompt["preset_id"])
        ):
            raise ValueError(
                "Query stage is fixed except approved answer prompt presets"
            )
    for group, component in (
        ("chunkers", ing["chunker"]),
        ("retrievers", qry["retriever"]),
        ("rewriters", qry["rewriter"]),
        ("rerankers", qry["reranker"]),
        ("compressors", qry["compressor"]),
    ):
        base = (
            original.ingestion_config["chunker"]
            if group == "chunkers"
            else original.query_config[
                {
                    "retrievers": "retriever",
                    "rewriters": "rewriter",
                    "rerankers": "reranker",
                    "compressors": "compressor",
                }[group]
            ]
        )
        if component.get("type") not in space[group] and component != base:
            raise ValueError(f"Component outside approved search space: {group}")
    if ing["embedder"] not in space["embeddings"]:
        raise ValueError("Embedding model was not approved")
    # Candidate-specific transport/model loading options cannot escape the presets.
    parameter_keys = {
        "chunker": {"type", "chunk_size", "overlap", "parent_chunk_size"},
        "rewriter": {"type", "count", "temperature", "max_subquestions"},
        "retriever": {
            "type",
            "rrf_k",
            "search_ef",
            "hnsw_ef",
            "child_top_k",
            "child_top_k_multiplier",
            "base_retriever",
        },
        "reranker": {"type", "model", "lambda_mult", "top_n"},
        "compressor": {"type", "max_chunks"},
        "context_assembler": {"type", "max_context_chars"},
    }
    for name, component in [("chunker", ing["chunker"])] + [
        (k, qry[k]) for k in parameter_keys if k != "chunker"
    ]:
        base = (
            original.ingestion_config["chunker"]
            if name == "chunker"
            else original.query_config[name]
        )
        if component != base and set(component) - parameter_keys[name]:
            raise ValueError(f"Unsupported candidate parameters: {name}")
    chunker = ing["chunker"]
    if chunker != original.ingestion_config["chunker"]:
        if (
            chunker.get("chunk_size", 512) not in space["chunk_sizes"]
            or chunker.get("overlap", 0) not in space["overlaps"]
        ):
            raise ValueError("Chunk parameters are outside the bounded search space")
        if "parent_chunk_size" in chunker and chunker["parent_chunk_size"] not in {
            1024,
            2048,
            4096,
        }:
            raise ValueError("Invalid parent chunk size")
    reranker = qry["reranker"]
    if reranker != original.query_config["reranker"] and reranker.get("model") not in {
        None,
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-TinyBERT-L-2-v2",
    }:
        raise ValueError("Reranker model was not approved")
    if not 0 <= qry["reranker"].get("lambda_mult", 0.5) <= 1:
        raise ValueError("Invalid reranker diversity weight")
    for key in ("count", "max_subquestions"):
        if not 1 <= qry["rewriter"].get(key, 3) <= 5:
            raise ValueError("Query rewriting must produce at most five variants")
    for key in (
        "search_ef",
        "hnsw_ef",
        "rrf_k",
        "child_top_k",
        "child_top_k_multiplier",
    ):
        if not 1 <= qry["retriever"].get(key, 10) <= 512:
            raise ValueError("Retriever parameter is out of bounds")
    for kind, choices in (
        ("dense", space["dense_strategies"]),
        ("sparse", space["sparse_strategies"]),
    ):
        current = ing["indexers"].get(kind)
        previous = original.ingestion_config["indexers"].get(kind)
        if (
            current
            and current != previous
            and (
                set(current) - {"type", "strategy"}
                or current.get("strategy") not in choices
            )
        ):
            raise ValueError("Indexer is outside the bounded search space")
    size, overlap = (
        ing["chunker"].get("chunk_size", 512),
        ing["chunker"].get("overlap", 0),
    )
    if isinstance(size, bool) or not isinstance(size, int) or not 32 <= size <= 32768:
        raise ValueError("Invalid chunk size")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, (float, int))
        or not math.isfinite(overlap)
        or overlap < 0
        or overlap >= (1 if isinstance(overlap, float) else size)
    ):
        raise ValueError("Invalid chunk overlap")
    indexers, retriever = ing["indexers"], qry["retriever"]["type"]
    if retriever in {
        "dense",
        "dense_filtered",
        "hybrid",
        "weighted_hybrid",
        "adaptive_hybrid",
        "parent_document",
        "multi_query_fusion",
    } and not indexers.get("dense"):
        raise ValueError("Dense retrieval requires a dense index")
    if retriever in {
        "bm25",
        "hybrid",
        "weighted_hybrid",
        "adaptive_hybrid",
    } and not indexers.get("sparse"):
        raise ValueError("Lexical retrieval requires a sparse index")
    if ing["embedder"]["provider"] in {
        "bge_m3_sparse",
        "splade_sparse",
    } and indexers.get("dense"):
        raise ValueError("Sparse-only embeddings cannot populate a dense index")
    if (
        retriever == "parent_document"
        and ing["chunker"]["type"] != "llamaindex_hierarchical"
    ):
        raise ValueError("Parent-document retrieval requires hierarchical chunks")
    if not 1 <= qry["compressor"].get("max_chunks", 5) <= 100:
        raise ValueError("Invalid context chunk limit")
    if not 256 <= qry["context_assembler"].get("max_context_chars", 12000) <= 100000:
        raise ValueError("Invalid context character limit")
    if (
        qry["context_assembler"]["type"]
        != original.query_config["context_assembler"]["type"]
    ):
        raise ValueError("Context assembler implementation is fixed")
    if (
        qry["context_assembler"] != original.query_config["context_assembler"]
        and qry["context_assembler"].get("max_context_chars", 12000)
        not in space["max_context_chars"]
    ):
        raise ValueError("Context limit is outside the bounded search space")
    if registry:
        groups = [
            (registry.chunkers, ing["chunker"]["type"]),
            (registry.embedders, ing["embedder"]["provider"]),
            (registry.rewriters, qry["rewriter"]["type"]),
            (registry.retrievers, retriever),
            (registry.rerankers, qry["reranker"]["type"]),
            (registry.compressors, qry["compressor"]["type"]),
        ]
        if any(name not in group for group, name in groups):
            raise ValueError("Candidate references an unregistered component")
        if any(
            indexer.get("type") not in registry.indexers
            for indexer in indexers.values()
            if isinstance(indexer, dict)
        ):
            raise ValueError("Candidate references an unregistered indexer")
    return concrete_config(config, cases[0])


def seed_candidates(baseline, space):
    seeds = [deepcopy(baseline)]
    for name in space["retrievers"]:
        candidate = deepcopy(baseline)
        candidate["pipeline"]["query"]["retrievers"] = [{"type": name}]
        if name == "parent_document":
            candidate["pipeline"]["ingestion"]["chunkers"] = [
                {
                    "type": "llamaindex_hierarchical",
                    "chunk_size": 512,
                    "parent_chunk_size": 2048,
                }
            ]
        seeds.append(candidate)
    return list({candidate_hash(c): c for c in seeds}.values())
