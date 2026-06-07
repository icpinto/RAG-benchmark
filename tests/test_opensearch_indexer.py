from ragbench_core.components.indexers import OpenSearchIndexer
from ragbench_core.models.schema import Chunk


def test_opensearch_indexer_builds_field_weighted_document():
    indexer = OpenSearchIndexer()
    chunk = Chunk(
        "c1",
        "doc.md",
        "Refunds are available within 14 days.",
        {"title": "Policy", "section_heading": "Refunds"},
    )
    strategy = indexer._strategy_config("bm25_field_weighted", {})

    body = indexer._document_body(chunk, strategy)

    assert body["title"] == "Policy"
    assert body["heading"] == "Refunds"
    assert body["body"] == chunk.text


def test_opensearch_indexer_custom_analyzer_accepts_synonyms_override():
    indexer = OpenSearchIndexer()
    strategy = indexer._strategy_config(
        "bm25_custom_analyzer",
        {"synonyms": ["llm, large language model"]},
    )

    body = indexer._index_body(strategy)

    assert body["settings"]["analysis"]["filter"]["ragbench_synonyms"]["synonyms"] == [
        "llm, large language model"
    ]
    assert body["mappings"]["properties"]["text"]["analyzer"] == "ragbench_bm25_custom"


def test_opensearch_indexer_phrase_strategy_adds_shingle_subfields():
    indexer = OpenSearchIndexer()
    strategy = indexer._strategy_config("bm25_phrase", {})

    body = indexer._index_body(strategy)

    assert "ragbench_shingle" in body["settings"]["analysis"]["analyzer"]
    assert body["mappings"]["properties"]["text"]["fields"]["phrase"]["analyzer"] == "ragbench_shingle"
