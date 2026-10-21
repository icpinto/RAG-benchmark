from __future__ import annotations
import math
from typing import Any
from ragbench_core.interfaces import BaseMetric
from ragbench_core.models.schema import QueryExample, RetrievalResult, GeneratedAnswer

RETRIEVAL_SELECTIONS = {"retrieval_metrics", "recall_at_k", "precision_at_k", "mrr", "ndcg"}
ANSWER_SELECTIONS = {"citation_coverage", "answer_correctness_placeholder", "faithfulness_placeholder"}
METRIC_SELECTIONS = RETRIEVAL_SELECTIONS | ANSWER_SELECTIONS | {"operational", "latency", "index_size", "answer_correctness", "claim_support", "citation_support"}


def validate_metric_selection(config: dict) -> None:
    enabled = config.get("enabled")
    if enabled is not None and (not isinstance(enabled, list) or any(not isinstance(v, str) or v not in METRIC_SELECTIONS for v in enabled)):
        raise ValueError("metrics.enabled must be a list of supported metric names")


def metric_selected(name: str, config: dict) -> bool:
    enabled = config.get("enabled")
    if enabled is None:
        return True
    if name == "retrieval_metrics":
        return bool(set(enabled) & RETRIEVAL_SELECTIONS)
    return name in enabled


def filter_metrics(metrics: dict, config: dict) -> dict:
    enabled = config.get("enabled")
    if enabled is None:
        return metrics
    selected = set(enabled)
    def keep(key):
        if key in selected:
            return True
        if "retrieval_metrics" in selected and any(term in key for term in ("recall@", "precision@", "mrr@", "ndcg@")):
            return True
        if any(alias in selected and term in key for alias, term in (("recall_at_k", "recall@"), ("precision_at_k", "precision@"), ("mrr", "mrr@"), ("ndcg", "ndcg@"))):
            return True
        if selected & RETRIEVAL_SELECTIONS and (key.endswith("failed") or key == "retrieval_ground_truth_available"):
            return True
        if "latency" in selected and key.endswith("seconds"):
            return True
        if "index_size" in selected and key == "chunk_count":
            return True
        if "operational" in selected and any(term in key for term in ("tokens", "cost", "chars", "context_chunks")):
            return True
        return False
    return {k: v for k, v in metrics.items() if keep(k)}


def count_text_tokens(text: str, model: str | None = None) -> int:
    """Count tokens with tiktoken when installed, falling back to a rough char heuristic."""
    if not text:
        return 0
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def count_texts_tokens(texts: list[str], model: str | None = None) -> int:
    return sum(count_text_tokens(text, model) for text in texts if text)


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _cost(tokens: int, price_per_1m: float) -> float:
    return (tokens / 1_000_000) * price_per_1m

class RetrievalMetrics(BaseMetric):
    name = "retrieval_metrics"
    _STAGE_RESULT_KEYS = {
        "raw_retrieval": "raw_retrieval_results",
        "reranked": "reranked_results",
    }

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

        result_sets = self._evaluation_result_sets(results, answer)
        metrics = {"retrieval_ground_truth_available": has_ground_truth}

        if set(result_sets) == {"retrieval"}:
            metrics.update(
                self._compute_result_set_metrics(
                    result_sets["retrieval"], ks, truth_chunks, truth_ids, has_ground_truth
                )
            )
            return metrics

        for stage, stage_results in result_sets.items():
            stage_metrics = self._compute_result_set_metrics(
                stage_results, ks, truth_chunks, truth_ids, has_ground_truth
            )
            metrics[f"{stage}_failed"] = stage_metrics.pop("retrieval_failed")
            for key, value in stage_metrics.items():
                metrics[f"{stage}_{key}"] = value

            if stage == "reranked":
                # Preserve the historical, unprefixed metric names for downstream
                # aggregations while ensuring they represent the pre-compression
                # reranked candidates rather than the compressed context set.
                metrics["retrieval_failed"] = metrics[f"{stage}_failed"]
                metrics.update(stage_metrics)

        return metrics

    def _evaluation_result_sets(
        self, results: list[RetrievalResult], answer: GeneratedAnswer | None
    ) -> dict[str, list[RetrievalResult]]:
        if not answer:
            return {"retrieval": results}

        result_sets = {}
        for stage, metadata_key in self._STAGE_RESULT_KEYS.items():
            stage_results = answer.metadata.get(metadata_key)
            if stage_results is not None:
                result_sets[stage] = stage_results

        return result_sets or {"retrieval": results}

    def _compute_result_set_metrics(
        self,
        results: list[RetrievalResult],
        ks: list[int],
        truth_chunks: set[str],
        truth_ids: set[str],
        has_ground_truth: bool,
    ) -> dict:
        retrieval_failed = has_ground_truth and not results
        metrics = {"retrieval_failed": retrieval_failed}
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
        if not answer:
            return {
                "answer_chars": 0,
                "context_chunks": 0,
                "estimated_answer_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "context_tokens": 0,
                "embedding_tokens": 0,
                "estimated_embedding_cost": 0.0,
                "estimated_generation_cost": 0.0,
                "total_estimated_cost": 0.0,
            }

        metadata = answer.metadata or {}
        usage = metadata.get("token_usage") or {}
        pricing = config.get("pricing") or {}
        generation_model = metadata.get("generation_model") or config.get("generation_model")
        embedding_model = metadata.get("embedding_model") or config.get("embedding_model")

        prompt_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        if prompt_tokens is None:
            prompt_tokens = count_text_tokens(answer.prompt, generation_model)

        completion_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
        if completion_tokens is None:
            completion_tokens = count_text_tokens(text, generation_model)

        context_text = metadata.get("context_text")
        if context_text is None:
            context_text = "\n\n".join(chunk.text for chunk in answer.context_chunks)
        context_tokens = count_text_tokens(str(context_text), generation_model)

        embedding_tokens = metadata.get("embedding_tokens")
        if embedding_tokens is None:
            embedding_texts = metadata.get("embedding_texts") or []
            if not embedding_texts:
                embedding_texts = [chunk.text for chunk in answer.context_chunks]
            embedding_tokens = count_texts_tokens([str(item) for item in embedding_texts], embedding_model)
        else:
            embedding_tokens = float(embedding_tokens)

        generation_input_price = float(pricing.get("generation_input_per_1m", 0.0) or 0.0)
        generation_output_price = float(pricing.get("generation_output_per_1m", 0.0) or 0.0)
        embedding_price = float(pricing.get("embedding_per_1m", 0.0) or 0.0)
        estimated_generation_cost = _cost(prompt_tokens, generation_input_price) + _cost(completion_tokens, generation_output_price)
        estimated_embedding_cost = _cost(embedding_tokens, embedding_price)

        return {
            "answer_chars": len(text),
            "context_chunks": len(answer.context_chunks),
            "estimated_answer_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_tokens": context_tokens,
            "embedding_tokens": embedding_tokens,
            "estimated_embedding_cost": estimated_embedding_cost,
            "estimated_generation_cost": estimated_generation_cost,
            "total_estimated_cost": estimated_embedding_cost + estimated_generation_cost,
        }
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
