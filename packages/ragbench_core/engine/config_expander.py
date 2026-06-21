from __future__ import annotations
from itertools import product
from copy import deepcopy
from ragbench_core.models.schema import BenchmarkCase
from ragbench_core.utils.ids import stable_id

def _expand_chunkers(chunkers):
    out=[]
    for c in chunkers:
        sizes=c.get("chunk_size", [512])
        if not isinstance(sizes, list): sizes=[sizes]
        overlaps=c.get("overlap", [0])
        if not isinstance(overlaps, list): overlaps=[overlaps]
        for size, overlap in product(sizes, overlaps):
            cc=dict(c); cc["chunk_size"]=size; cc["overlap"]=overlap
            out.append(cc)
    return out


_QDRANT_DENSE_TUNABLES = (
    "hnsw_m",
    "m",
    "ef_construct",
    "search_ef",
    "hnsw_ef",
    "quantization_config",
    "on_disk",
    "on_disk_payload",
    "payload_on_disk",
    "hnsw_on_disk",
)


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _expand_qdrant_dense_tunables(indexer: dict) -> list[dict]:
    if indexer.get("type", "qdrant") != "qdrant":
        return [indexer]
    tunable_keys = [key for key in _QDRANT_DENSE_TUNABLES if isinstance(indexer.get(key), list)]
    hnsw_config = indexer.get("hnsw_config")
    if isinstance(hnsw_config, dict):
        tunable_keys.extend(f"hnsw_config.{key}" for key, value in hnsw_config.items() if isinstance(value, list))
    quantization_config = indexer.get("quantization_config")
    if isinstance(quantization_config, dict):
        tunable_keys.extend(f"quantization_config.{key}" for key, value in quantization_config.items() if isinstance(value, list))
    if not tunable_keys:
        return [indexer]

    values = []
    for key in tunable_keys:
        if key.startswith("hnsw_config."):
            values.append(_as_list(hnsw_config[key.split(".", 1)[1]]))
        elif key.startswith("quantization_config."):
            values.append(_as_list(quantization_config[key.split(".", 1)[1]]))
        else:
            values.append(_as_list(indexer[key]))

    expanded = []
    for combo in product(*values):
        cc = deepcopy(indexer)
        for key, value in zip(tunable_keys, combo):
            if key.startswith("hnsw_config."):
                cc.setdefault("hnsw_config", {})[key.split(".", 1)[1]] = value
            elif key.startswith("quantization_config."):
                cc.setdefault("quantization_config", {})[key.split(".", 1)[1]] = value
            else:
                cc[key] = value
        expanded.append(cc)
    return expanded

def _expand_indexer_variants(indexer_config, key, default):
    indexer = indexer_config.get(key, default)
    if isinstance(indexer, list):
        if key != "dense":
            return indexer
        expanded = []
        for item in indexer:
            expanded.extend(_expand_qdrant_dense_tunables(item))
        return expanded
    strategies = indexer.get("strategies")
    if not strategies:
        return _expand_qdrant_dense_tunables(indexer) if key == "dense" else [indexer]
    out = []
    for strategy in strategies:
        cc = dict(indexer)
        cc.pop("strategies", None)
        if isinstance(strategy, dict):
            cc.update(strategy)
        else:
            cc["strategy"] = strategy
        if key == "dense":
            out.extend(_expand_qdrant_dense_tunables(cc))
        else:
            out.append(cc)
    return out


def _expand_dense_indexers(indexer_config):
    if indexer_config and "dense" not in indexer_config:
        return [None]
    return _expand_indexer_variants(indexer_config, "dense", {"type": "qdrant", "strategy": "dense_hnsw"})


def _expand_sparse_indexers(indexer_config):
    if indexer_config and "sparse" not in indexer_config:
        return [None]
    return _expand_indexer_variants(indexer_config, "sparse", {"type": "opensearch", "strategy": "bm25_standard"})


def expand_cases(config: dict) -> list[BenchmarkCase]:
    ing=config["pipeline"]["ingestion"]
    qry=config["pipeline"]["query"]
    chunkers=_expand_chunkers(ing.get("chunkers", []))
    embeddings=ing.get("embeddings", [])
    rewriters=qry.get("rewriters", [{"type":"none"}])
    retrievers=qry.get("retrievers", [{"type":"dense"}])
    rerankers=qry.get("rerankers", [{"type":"none"}])
    compressors=qry.get("compressors", [{"type":"none"}])
    indexer_config=ing.get("indexers",{})
    dense_indexers=_expand_dense_indexers(indexer_config)
    sparse_indexers=_expand_sparse_indexers(indexer_config)
    max_combos=int(config.get("runtime",{}).get("max_combinations", 32))
    cases=[]
    for embedder, rewriter, retriever, reranker, compressor, chunker, dense_indexer, sparse_indexer in product(embeddings, rewriters, retrievers, rerankers, compressors, chunkers, dense_indexers, sparse_indexers):
        indexers=dict(indexer_config)
        if dense_indexer is not None:
            indexers["dense"] = dense_indexer
        else:
            indexers.pop("dense", None)
        if sparse_indexer is not None:
            indexers["sparse"] = sparse_indexer
        else:
            indexers.pop("sparse", None)
        ing_conf={"chunker":chunker,"embedder":embedder,"cleaners":ing.get("cleaners",[]),"extractors":ing.get("extractors",[]),"metadata_enrichers":ing.get("metadata_enrichers",[]),"indexers":indexers}
        retriever_conf = dict(retriever)
        if dense_indexer is not None and retriever_conf.get("type") in {"dense", "dense_filtered", "hybrid", "weighted_hybrid", "adaptive_hybrid", "parent_document", "multi_query_fusion"}:
            search_ef = dense_indexer.get("search_ef", dense_indexer.get("hnsw_ef"))
            if search_ef is not None and "search_ef" not in retriever_conf and "hnsw_ef" not in retriever_conf:
                retriever_conf["search_ef"] = search_ef
        qry_conf={"rewriter":rewriter,"retriever":retriever_conf,"filters":qry.get("filters", [{"type":"none"}]),"reranker":reranker,"compressor":compressor,"context_assembler":qry.get("context_assemblers", [{"type":"default"}])[0],"prompt_builder":qry.get("prompt_builders", [{"type":"default"}])[0],"citation_validator":qry.get("citation_validators", [{"type":"simple"}])[0]}
        cid=stable_id(ing_conf,qry_conf,config.get("experiment",{}).get("name","exp"))
        cases.append(BenchmarkCase(cid, ing_conf, qry_conf, config["llm"], config["metrics"]))
        if len(cases)>=max_combos: break
    return cases
