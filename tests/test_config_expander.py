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


def test_expand_cases_preserves_sparse_only_indexer_config():
    config=load_yaml("configs/default.yaml")
    config=deepcopy(config)
    config["pipeline"]["ingestion"]["chunkers"] = [
        {"type": "fixed", "chunk_size": 256, "overlap": 0}
    ]
    config["pipeline"]["ingestion"]["embeddings"] = [
        {"provider": "splade_sparse", "model": "stub"}
    ]
    config["pipeline"]["ingestion"]["indexers"] = {
        "sparse": {"type": "qdrant_sparse", "vector_name": "neural_sparse"}
    }
    config["pipeline"]["query"]["rewriters"] = [{"type": "none"}]
    config["pipeline"]["query"]["retrievers"] = [{"type": "neural_sparse", "backend": "qdrant"}]
    config["pipeline"]["query"]["rerankers"] = [{"type": "none"}]
    config["pipeline"]["query"]["compressors"] = [{"type": "none"}]

    cases=expand_cases(config)

    assert len(cases) == 1
    assert "dense" not in cases[0].ingestion_config["indexers"]
    assert cases[0].ingestion_config["indexers"]["sparse"]["type"] == "qdrant_sparse"


def test_expand_cases_expands_qdrant_dense_hnsw_search_and_storage_tunables():
    config=load_yaml("configs/default.yaml")
    config=deepcopy(config)
    config["pipeline"]["ingestion"]["chunkers"] = [
        {"type": "fixed", "chunk_size": 256, "overlap": 0}
    ]
    config["pipeline"]["ingestion"]["embeddings"] = [
        {"provider": "sentence_transformers", "model": "stub"}
    ]
    config["pipeline"]["ingestion"]["indexers"] = {
        "dense": {
            "type": "qdrant",
            "strategy": "dense_hnsw_custom",
            "hnsw_m": [16, 32],
            "ef_construct": [100, 200],
            "search_ef": [64, 128],
            "quantization_config": {"type": "int8", "always_ram": [True, False]},
            "on_disk": [True, False],
            "on_disk_payload": [True, False],
        }
    }
    config["pipeline"]["query"]["rewriters"] = [{"type": "none"}]
    config["pipeline"]["query"]["retrievers"] = [{"type": "dense"}]
    config["pipeline"]["query"]["rerankers"] = [{"type": "none"}]
    config["pipeline"]["query"]["compressors"] = [{"type": "none"}]
    config["runtime"]["max_combinations"] = 64

    cases=expand_cases(config)

    assert len(cases) == 64
    first_dense = cases[0].ingestion_config["indexers"]["dense"]
    assert first_dense == {
        "type": "qdrant",
        "strategy": "dense_hnsw_custom",
        "hnsw_m": 16,
        "ef_construct": 100,
        "search_ef": 64,
        "quantization_config": {"type": "int8", "always_ram": True},
        "on_disk": True,
        "on_disk_payload": True,
    }
    assert cases[0].query_config["retriever"]["search_ef"] == 64
    assert cases[-1].ingestion_config["indexers"]["dense"]["hnsw_m"] == 32
    assert cases[-1].ingestion_config["indexers"]["dense"]["ef_construct"] == 200
    assert cases[-1].ingestion_config["indexers"]["dense"]["search_ef"] == 128
    assert cases[-1].ingestion_config["indexers"]["dense"]["quantization_config"]["always_ram"] is False
    assert cases[-1].ingestion_config["indexers"]["dense"]["on_disk"] is False
    assert cases[-1].ingestion_config["indexers"]["dense"]["on_disk_payload"] is False
    assert cases[-1].query_config["retriever"]["search_ef"] == 128
