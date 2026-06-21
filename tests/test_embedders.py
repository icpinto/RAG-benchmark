import ragbench_core.components.embedders as embedders_module
from ragbench_core.components.embedders import SentenceTransformerEmbedder
from ragbench_core.models.schema import Chunk


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return FakeVectors([[float(i), float(i + 1)] for i, _ in enumerate(texts)])


def test_e5_alias_resolves_model_and_applies_query_and_passage_prefixes(monkeypatch):
    fake_model = FakeModel()
    loaded_models = []

    def fake_load_model(model_name):
        loaded_models.append(model_name)
        return fake_model

    monkeypatch.setattr(embedders_module, "_load_model", fake_load_model)

    embedder = SentenceTransformerEmbedder()
    chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta", {})]
    embedded_chunks = embedder.embed_chunks(chunks, {"model": "E5-large-v2"})
    query_vector = embedder.embed_query("find alpha", {"model": "E5-large-v2"})

    assert loaded_models == ["intfloat/e5-large-v2", "intfloat/e5-large-v2"]
    assert fake_model.calls[0][0] == ["passage: alpha", "passage: beta"]
    assert fake_model.calls[1][0] == ["query: find alpha"]
    assert [chunk.embedding for chunk in embedded_chunks] == [[0.0, 1.0], [1.0, 2.0]]
    assert query_vector == [0.0, 1.0]


def test_non_e5_sentence_transformer_keeps_text_unprefixed(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(embedders_module, "_load_model", lambda model_name: fake_model)

    embedder = SentenceTransformerEmbedder()
    embedder.embed_chunks([Chunk("c1", "d1", "alpha", {})], {"model": "sentence-transformers/all-MiniLM-L6-v2"})
    embedder.embed_query("find alpha", {"model": "sentence-transformers/all-MiniLM-L6-v2"})

    assert fake_model.calls[0][0] == ["alpha"]
    assert fake_model.calls[1][0] == ["find alpha"]


def test_openai_embedder_posts_batches_and_assigns_chunk_embeddings(monkeypatch):
    calls = []

    def fake_post_json(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {
            "data": [
                {"index": index, "embedding": [float(len(text)), float(index)]}
                for index, text in enumerate(payload["input"])
            ]
        }

    monkeypatch.setattr(embedders_module, "_post_json", fake_post_json)

    embedder = embedders_module.OpenAIEmbedder()
    chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta", {})]
    embedded = embedder.embed_chunks(
        chunks,
        {
            "api_key": "secret",
            "base_url": "https://example.test/v1/",
            "model": "custom-embedding",
            "batch_size": 1,
            "document_prefix": "doc: ",
        },
    )
    query_vector = embedder.embed_query(
        "find alpha",
        {"api_key": "secret", "query_prefix": "query: ", "normalize": True},
    )

    assert [chunk.embedding for chunk in embedded] == [[10.0, 0.0], [9.0, 0.0]]
    assert query_vector == [1.0, 0.0]
    assert calls[0][0] == "https://example.test/v1/embeddings"
    assert calls[0][1]["input"] == ["doc: alpha"]
    assert calls[0][1]["model"] == "custom-embedding"
    assert calls[0][2]["Authorization"] == "Bearer secret"
    assert calls[-1][1]["input"] == ["query: find alpha"]


def test_hosted_embedder_parses_voyage_cohere_and_jina_responses(monkeypatch):
    payloads = []

    def fake_post_json(url, payload, headers, timeout):
        payloads.append(payload)
        if "texts" in payload:
            return {"embeddings": {"float": [[1.0, 2.0], [3.0, 4.0]][: len(payload["texts"])]}}
        return {
            "data": [
                {"index": index, "embedding": [float(index), float(index + 1)]}
                for index, _ in enumerate(payload["input"])
            ]
        }

    monkeypatch.setattr(embedders_module, "_post_json", fake_post_json)

    chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta", {})]
    assert [c.embedding for c in embedders_module.VoyageEmbedder().embed_chunks(chunks, {"api_key": "v"})] == [
        [0.0, 1.0],
        [1.0, 2.0],
    ]
    assert embedders_module.CohereEmbedder().embed_query("alpha", {"api_key": "c"}) == [1.0, 2.0]
    assert embedders_module.JinaEmbedder().embed_query("alpha", {"api_key": "j"}) == [0.0, 1.0]
    assert payloads[0]["input_type"] == "document"
    assert payloads[1]["input_type"] == "search_query"
    assert payloads[2]["task"] == "retrieval.query"


def test_bge_m3_embedder_defaults_to_local_bge_m3_model(monkeypatch):
    fake_model = FakeModel()
    loaded_models = []

    def fake_load_model(model_name):
        loaded_models.append(model_name)
        return fake_model

    monkeypatch.setattr(embedders_module, "_load_model", fake_load_model)

    chunks = [Chunk("c1", "d1", "alpha", {})]
    embedders_module.BGEM3Embedder().embed_chunks(chunks, {})

    assert loaded_models == ["BAAI/bge-m3"]
    assert chunks[0].embedding == [0.0, 1.0]


def test_bge_m3_hybrid_embedder_populates_dense_and_sparse(monkeypatch):
    class FakeBGEM3Model:
        def encode(self, texts, **kwargs):
            assert kwargs["return_dense"] is True
            assert kwargs["return_sparse"] is True
            return {
                "dense_vecs": FakeVectors([[3.0, 4.0], [0.0, 2.0]][: len(texts)]),
                "lexical_weights": [
                    {"10": 0.5, "20": 0.0, "30": 0.25},
                    {"40": 0.75},
                ][: len(texts)],
            }

    monkeypatch.setattr(embedders_module.BGEM3HybridEmbedder, "_model", lambda self, config: FakeBGEM3Model())

    chunks = [Chunk("c1", "d1", "alpha", {}), Chunk("c2", "d1", "beta", {})]
    embedder = embedders_module.BGEM3HybridEmbedder()
    embedded = embedder.embed_chunks(chunks, {"normalize": True, "sparse_top_k": 1})
    query_embedding = embedder.embed_query("find alpha", {"normalize": True})

    assert embedded[0].embedding == [0.6, 0.8]
    assert embedded[0].sparse_embedding == {10: 0.5}
    assert embedded[1].embedding == [0.0, 1.0]
    assert embedded[1].sparse_embedding == {40: 0.75}
    assert query_embedding == {"dense": [0.6, 0.8], "sparse": {10: 0.5, 30: 0.25}}
