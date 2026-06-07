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

def _expand_dense_indexers(indexer_config):
    dense = indexer_config.get("dense", {"type": "qdrant", "strategy": "dense_hnsw"})
    if isinstance(dense, list):
        return dense
    strategies = dense.get("strategies")
    if not strategies:
        return [dense]
    out = []
    for strategy in strategies:
        cc = dict(dense)
        cc.pop("strategies", None)
        if isinstance(strategy, dict):
            cc.update(strategy)
        else:
            cc["strategy"] = strategy
        out.append(cc)
    return out


def expand_cases(config: dict) -> list[BenchmarkCase]:
    ing=config["pipeline"]["ingestion"]
    qry=config["pipeline"]["query"]
    chunkers=_expand_chunkers(ing.get("chunkers", []))
    embeddings=ing.get("embeddings", [])
    rewriters=qry.get("rewriters", [{"type":"none"}])
    retrievers=qry.get("retrievers", [{"type":"dense"}])
    rerankers=qry.get("rerankers", [{"type":"none"}])
    compressors=qry.get("compressors", [{"type":"none"}])
    dense_indexers=_expand_dense_indexers(ing.get("indexers",{}))
    max_combos=int(config.get("runtime",{}).get("max_combinations", 32))
    cases=[]
    for embedder, rewriter, retriever, reranker, compressor, chunker, dense_indexer in product(embeddings, rewriters, retrievers, rerankers, compressors, chunkers, dense_indexers):
        indexers=dict(ing.get("indexers",{})); indexers["dense"]=dense_indexer
        ing_conf={"chunker":chunker,"embedder":embedder,"cleaners":ing.get("cleaners",[]),"extractors":ing.get("extractors",[]),"metadata_enrichers":ing.get("metadata_enrichers",[]),"indexers":indexers}
        qry_conf={"rewriter":rewriter,"retriever":retriever,"filters":qry.get("filters", [{"type":"none"}]),"reranker":reranker,"compressor":compressor,"context_assembler":qry.get("context_assemblers", [{"type":"default"}])[0],"prompt_builder":qry.get("prompt_builders", [{"type":"default"}])[0],"citation_validator":qry.get("citation_validators", [{"type":"simple"}])[0]}
        cid=stable_id(ing_conf,qry_conf,config.get("experiment",{}).get("name","exp"))
        cases.append(BenchmarkCase(cid, ing_conf, qry_conf, config["llm"], config["metrics"]))
        if len(cases)>=max_combos: break
    return cases
