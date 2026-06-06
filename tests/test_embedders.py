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
