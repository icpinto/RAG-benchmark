from copy import deepcopy

from ragbench_core.utils.config import load_yaml
from ragbench_core.engine.config_expander import expand_cases


def test_expand_cases():
    config=load_yaml("configs/default.yaml")
    cases=expand_cases(config)
    assert cases
    assert len(cases) <= config["runtime"]["max_combinations"]


def test_expand_cases_includes_qdrant_dense_strategies():
    config=load_yaml("configs/default.yaml")
    config=deepcopy(config)
    config["pipeline"]["ingestion"]["chunkers"] = [
        {"type": "fixed", "chunk_size": 256, "overlap": 0}
    ]
    config["pipeline"]["ingestion"]["embeddings"] = [
        {"provider": "sentence_transformers", "model": "stub"}
    ]
    config["pipeline"]["query"]["rewriters"] = [{"type": "none"}]
    config["pipeline"]["query"]["retrievers"] = [{"type": "dense"}]
    config["pipeline"]["query"]["rerankers"] = [{"type": "none"}]
    config["pipeline"]["query"]["compressors"] = [{"type": "none"}]
    config["pipeline"]["ingestion"]["indexers"]["sparse"] = {
        "type": "opensearch",
        "strategy": "bm25_standard",
    }
    config["runtime"]["max_combinations"] = 10

    cases=expand_cases(config)

    assert [
        case.ingestion_config["indexers"]["dense"]["strategy"] for case in cases
    ] == [
        "dense_hnsw",
        "dense_hnsw_tuned",
        "dense_exact",
        "dense_quantized",
        "dense_on_disk",
        "dense_payload_indexed",
    ]


def test_expand_cases_includes_opensearch_bm25_strategies():
    config=load_yaml("configs/default.yaml")
    config=deepcopy(config)
    config["pipeline"]["ingestion"]["chunkers"] = [
        {"type": "fixed", "chunk_size": 256, "overlap": 0}
    ]
    config["pipeline"]["ingestion"]["embeddings"] = [
        {"provider": "sentence_transformers", "model": "stub"}
    ]
    config["pipeline"]["ingestion"]["indexers"]["dense"] = {
        "type": "qdrant",
        "strategy": "dense_hnsw",
    }
    config["pipeline"]["query"]["rewriters"] = [{"type": "none"}]
    config["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    config["pipeline"]["query"]["rerankers"] = [{"type": "none"}]
    config["pipeline"]["query"]["compressors"] = [{"type": "none"}]
    config["runtime"]["max_combinations"] = 10

    cases=expand_cases(config)

    assert [
        case.ingestion_config["indexers"]["sparse"]["strategy"] for case in cases
    ] == [
        "bm25_standard",
        "bm25_field_weighted",
        "bm25_custom_analyzer",
        "bm25_phrase",
    ]
