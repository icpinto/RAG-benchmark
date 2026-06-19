from __future__ import annotations
import math, os
from ragbench_core.interfaces import BaseMetric
from ragbench_core.models.schema import QueryExample, RetrievalResult, GeneratedAnswer

class RetrievalMetrics(BaseMetric):
    name = "retrieval_metrics"

    @staticmethod
    def _validate_retrieval_k(config: dict) -> list[int]:
        ks = config.get("retrieval_k", [3, 5, 10])
        if not isinstance(ks, list) or not ks:
            raise ValueError("retrieval_k must be a non-empty list of positive integers")
        if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
            raise ValueError("retrieval_k must contain only positive integers")
        return ks

    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict) -> dict:
        ks = self._validate_retrieval_k(config)
        truth_chunks = set(example.ground_truth_chunk_ids or [])
        truth_docs = set(example.ground_truth_doc_ids or [])
        truth_ids = truth_chunks or truth_docs
        has_ground_truth = bool(truth_ids)
        retrieval_failed = has_ground_truth and not results
        metrics = {
            "retrieval_ground_truth_available": has_ground_truth,
            "retrieval_failed": retrieval_failed,
        }
        for k in ks:
            if not has_ground_truth:
                metrics[f"recall@{k}"] = None
                metrics[f"precision@{k}"] = None
                metrics[f"mrr@{k}"] = None
                metrics[f"ndcg@{k}"] = None
                continue

            top = results[:k]
            denom = len(truth_ids)
            seen_hits: set[str] = set()
            hits = 0
            dcg = 0.0
            rr = 0.0
            for idx, r in enumerate(top, start=1):
                evaluated_id = r.chunk.chunk_id if truth_chunks else r.chunk.doc_id
                ok = evaluated_id in truth_ids
                if ok and evaluated_id not in seen_hits:
                    seen_hits.add(evaluated_id)
                    hits += 1
                    if rr == 0.0:
                        rr = 1.0 / idx
                    dcg += 1.0 / math.log2(idx + 1)
            metrics[f"recall@{k}"] = min(hits / denom, 1.0)
            metrics[f"precision@{k}"] = hits / k
            metrics[f"mrr@{k}"] = rr
            ideal_hits = min(denom, k)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
            metrics[f"ndcg@{k}"] = min(dcg / idcg, 1.0) if idcg else 0.0
        return metrics

class OperationalMetric(BaseMetric):
    name = "operational"
    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict) -> dict:
        text = answer.answer if answer else ""
        return {"answer_chars":len(text),"context_chunks":len(answer.context_chunks) if answer else 0,"estimated_answer_tokens":max(1, len(text)//4) if text else 0}

class CitationCoverageMetric(BaseMetric):
    name = "citation_coverage"
    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict) -> dict:
        if not answer: return {"citation_coverage":0.0}
        expected=example.expected_citations
        if not expected: return {"citation_coverage":1.0}
        found=sum(1 for c in expected if c in answer.answer)
        return {"citation_coverage":found/len(expected)}

class PlaceholderAnswerMetric(BaseMetric):
    name = "answer_correctness_placeholder"
    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict) -> dict:
        # Replace with RAGAS/DeepEval/LLM judge later. This cheap placeholder checks lexical overlap only.
        if not answer or not example.expected_answer: return {"answer_correctness_placeholder":0.0}
        expected=set(example.expected_answer.lower().split())
        actual=set(answer.answer.lower().split())
        return {"answer_correctness_placeholder":len(expected & actual)/len(expected) if expected else 0.0}

class PlaceholderFaithfulnessMetric(BaseMetric):
    name = "faithfulness_placeholder"
    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict) -> dict:
        if not answer: return {"faithfulness_placeholder":0.0}
        context=" ".join(c.text.lower() for c in answer.context_chunks)
        answer_words=[w for w in answer.answer.lower().split() if len(w)>4]
        if not answer_words: return {"faithfulness_placeholder":0.0}
        supported=sum(1 for w in answer_words if w in context)
        return {"faithfulness_placeholder":supported/len(answer_words)}
