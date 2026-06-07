from __future__ import annotations
from itertools import product
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

def _expand_indexer_variants(indexer_config, key, default):
    indexer = indexer_config.get(key, default)
    if isinstance(indexer, list):
        return indexer
    strategies = indexer.get("strategies")
    if not strategies:
        return [indexer]
    out = []
    for strategy in strategies:
        cc = dict(indexer)
        cc.pop("strategies", None)
        if isinstance(strategy, dict):
            cc.update(strategy)
        else:
            cc["strategy"] = strategy
        out.append(cc)
    return out


def _expand_dense_indexers(indexer_config):
    return _expand_indexer_variants(indexer_config, "dense", {"type": "qdrant", "strategy": "dense_hnsw"})


def _expand_sparse_indexers(indexer_config):
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
        indexers=dict(indexer_config); indexers["dense"]=dense_indexer; indexers["sparse"]=sparse_indexer
        ing_conf={"chunker":chunker,"embedder":embedder,"cleaners":ing.get("cleaners",[]),"extractors":ing.get("extractors",[]),"metadata_enrichers":ing.get("metadata_enrichers",[]),"indexers":indexers}
        qry_conf={"rewriter":rewriter,"retriever":retriever,"filters":qry.get("filters", [{"type":"none"}]),"reranker":reranker,"compressor":compressor,"context_assembler":qry.get("context_assemblers", [{"type":"default"}])[0],"prompt_builder":qry.get("prompt_builders", [{"type":"default"}])[0],"citation_validator":qry.get("citation_validators", [{"type":"simple"}])[0]}
        cid=stable_id(ing_conf,qry_conf,config.get("experiment",{}).get("name","exp"))
        cases.append(BenchmarkCase(cid, ing_conf, qry_conf, config["llm"], config["metrics"]))
        if len(cases)>=max_combos: break
    return cases
