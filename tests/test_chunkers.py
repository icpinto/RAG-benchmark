import ragbench_core.components.chunkers as chunkers_module
from ragbench_core.components.chunkers import (
    FixedSizeChunker,
    LlamaIndexDocumentStructureChunker,
    LlamaIndexFixedSizeChunker,
    LlamaIndexHierarchicalChunker,
    LlamaIndexSemanticChunker,
    LlamaIndexSentenceChunker,
)
from ragbench_core.models.schema import Document


def test_fixed_chunker():
    docs=[Document("d1","src","a"*1000,{})]
    chunks=FixedSizeChunker().chunk(docs,{"chunk_size":100,"overlap":0.2})
    assert len(chunks)>1
    assert chunks[0].doc_id=="d1"


def test_llamaindex_fixed_chunker_splits_fixed_size_chunks(monkeypatch):
    monkeypatch.setattr(chunkers_module.importlib.util, "find_spec", lambda name: None)
    docs=[Document("d1","src","abcdefghij",{})]
    chunks=LlamaIndexFixedSizeChunker().chunk(docs,{"chunk_size":4,"overlap":0})
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]
    assert chunks[0].metadata["chunker"] == "llamaindex_fixed"
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == 4


def test_llamaindex_sentence_chunker_preserves_sentence_boundaries(monkeypatch):
    monkeypatch.setattr(chunkers_module.importlib.util, "find_spec", lambda name: None)
    docs=[Document("d1","src","First sentence. Second sentence. Third sentence.",{})]
    chunks=LlamaIndexSentenceChunker().chunk(docs,{"chunk_size":30,"overlap":0})
    assert [c.text for c in chunks] == ["First sentence.", "Second sentence.", "Third sentence."]
    assert all(c.metadata["chunker"] == "llamaindex_sentence" for c in chunks)


def test_llamaindex_hierarchical_chunker_adds_parent_metadata(monkeypatch):
    monkeypatch.setattr(chunkers_module.importlib.util, "find_spec", lambda name: None)
    text = "Alpha one. Alpha two. Beta one. Beta two."
    docs = [Document("d1", "src", text, {})]

    chunks = LlamaIndexHierarchicalChunker().chunk(
        docs,
        {"chunk_size": 22, "parent_chunk_size": 45, "overlap": 0},
    )

    assert [c.metadata["chunker"] for c in chunks] == ["llamaindex_hierarchical", "llamaindex_hierarchical"]
    assert all(c.metadata["parent_chunk_id"] for c in chunks)
    assert all(c.metadata["parent_text"] for c in chunks)
    assert all(c.metadata["chunk_role"] == "child" for c in chunks)


def test_llamaindex_semantic_chunker_groups_paragraphs(monkeypatch):
    monkeypatch.setattr(chunkers_module.importlib.util, "find_spec", lambda name: None)
    text = "Cats purr softly.\n\nDogs bark loudly.\n\nBirds sing brightly."
    docs = [Document("d1", "src", text, {})]

    chunks = LlamaIndexSemanticChunker().chunk(docs, {"chunk_size": 40, "overlap": 0})

    assert [c.text for c in chunks] == ["Cats purr softly.\n\nDogs bark loudly.", "Birds sing brightly."]
    assert all(c.metadata["chunker"] == "llamaindex_semantic" for c in chunks)


def test_llamaindex_structure_chunker_preserves_markdown_sections(monkeypatch):
    monkeypatch.setattr(chunkers_module.importlib.util, "find_spec", lambda name: None)
    text = "# Intro\nWelcome text.\n\n## Details\nMore detail text."
    docs = [Document("d1", "src", text, {})]

    chunks = LlamaIndexDocumentStructureChunker().chunk(docs, {"chunk_size": 80, "overlap": 0})

    assert [c.metadata["section_heading"] for c in chunks] == ["Intro", "Details"]
    assert [c.metadata["heading_level"] for c in chunks] == [1, 2]
    assert all(c.metadata["chunker"] == "llamaindex_structure" for c in chunks)


def test_case_expansion_prioritizes_chunker_variants_under_cap():
    from ragbench_core.engine.config_expander import expand_cases

    config = {
        "experiment": {"name": "chunker_priority"},
        "pipeline": {
            "ingestion": {
                "extractors": [],
                "cleaners": [],
                "chunkers": [
                    {"type": "fixed", "chunk_size": 128, "overlap": 0},
                    {"type": "llamaindex_fixed", "chunk_size": 128, "overlap": 0},
                    {"type": "llamaindex_sentence", "chunk_size": 128, "overlap": 0},
                ],
                "metadata_enrichers": [],
                "embeddings": [{"provider": "sentence_transformers", "model": "m"}],
                "indexers": {},
            },
            "query": {
                "rewriters": [{"type": "none"}],
                "retrievers": [{"type": "dense"}],
                "rerankers": [{"type": "none"}],
                "compressors": [{"type": "none"}],
            },
        },
        "llm": {"provider": "ollama"},
        "metrics": {"retrieval_k": [3]},
        "runtime": {"max_combinations": 3},
    }

    cases = expand_cases(config)

    assert [case.ingestion_config["chunker"]["type"] for case in cases] == [
        "fixed",
        "llamaindex_fixed",
        "llamaindex_sentence",
    ]
