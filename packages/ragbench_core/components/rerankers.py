from functools import lru_cache
from sentence_transformers import CrossEncoder
from ragbench_core.interfaces import BaseReranker
from ragbench_core.models.schema import RetrievalResult

class NoReranker(BaseReranker):
    name = "none"
    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        return results

@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    return CrossEncoder(model_name)

class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"
    def rerank(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        if not results: return results
        model=_load_cross_encoder(config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        scores=model.predict([(question, r.chunk.text) for r in results])
        reranked=[]
        for r,s in zip(results,scores):
            r.score=float(s); reranked.append(r)
        reranked.sort(key=lambda r:r.score, reverse=True)
        for i,r in enumerate(reranked): r.rank=i+1
        return reranked
