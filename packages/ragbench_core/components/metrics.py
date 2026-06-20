from __future__ import annotations
import asyncio, inspect, math, os
from dataclasses import dataclass
from typing import Any, Callable
from ragbench_core.interfaces import BaseMetric
from ragbench_core.models.schema import QueryExample, RetrievalResult, GeneratedAnswer

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


@dataclass(frozen=True)
class JudgeMetricConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout: float


class _JudgeMetric(BaseMetric):
    """Shared implementation for optional LLM-judge based answer metrics.

    These metrics intentionally keep all third-party imports inside the scoring
    path so offline smoke tests can still import and run the benchmark without
    Ragas, DeepEval, LangChain, or Ollama client dependencies installed.
    """

    provider: str
    score_key: str
    error_key: str
    dependency_name: str
    default_timeout = 60.0
    default_model = "gpt-oss:20b"
    default_base_url = "http://ollama:11434"

    def compute(
        self,
        example: QueryExample,
        results: list[RetrievalResult],
        answer: GeneratedAnswer | None,
        config: dict,
    ) -> dict:
        if not answer:
            return self._unavailable("missing_answer")

        metric_config = self._metric_config(config)
        if not self._is_enabled(config, metric_config):
            return self._unavailable("disabled")

        try:
            score, metadata = self._score(example, results, answer, metric_config)
            return {
                self.score_key: score,
                f"{self.score_key}_metadata": {
                    "provider": self.provider,
                    "model": metric_config.model,
                    "base_url": metric_config.base_url,
                    **metadata,
                },
            }
        except Exception as exc:
            return self._unavailable(type(exc).__name__, str(exc), metric_config)

    def _metric_config(self, config: dict) -> JudgeMetricConfig:
        provider_config = config.get("providers", {}).get(self.provider, {})
        judge_config = config.get("judge", {})
        metric_config = config.get(self.name, {})
        merged = {**judge_config, **provider_config, **metric_config}
        return JudgeMetricConfig(
            provider=self.provider,
            model=merged.get("model") or os.getenv("OLLAMA_MODEL", self.default_model),
            base_url=merged.get("base_url") or os.getenv("OLLAMA_BASE_URL", self.default_base_url),
            api_key=merged.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY"),
            timeout=float(merged.get("timeout", self.default_timeout)),
        )

    def _is_enabled(self, config: dict, metric_config: JudgeMetricConfig) -> bool:
        enabled = config.get("enabled")
        if enabled is None:
            return True
        return self.name in enabled or self.score_key in enabled or self.provider in enabled

    def _unavailable(
        self,
        reason: str,
        detail: str | None = None,
        metric_config: JudgeMetricConfig | None = None,
    ) -> dict:
        metadata = {"available": False, "provider": self.provider, "reason": reason}
        if detail:
            metadata["error"] = detail
        if reason == "ModuleNotFoundError":
            metadata.update(
                {
                    "dependency": self.dependency_name,
                    "install_extra": "answer-quality",
                    "install_command": "pip install -e '.[answer-quality]'",
                }
            )
        if metric_config:
            metadata.update({"model": metric_config.model, "base_url": metric_config.base_url})
        return {self.score_key: None, f"{self.score_key}_metadata": metadata}

    def _contexts(self, answer: GeneratedAnswer) -> list[str]:
        return [chunk.text for chunk in answer.context_chunks]

    def _reference(self, example: QueryExample) -> str:
        return example.expected_answer or ""

    def _score(
        self,
        example: QueryExample,
        results: list[RetrievalResult],
        answer: GeneratedAnswer,
        metric_config: JudgeMetricConfig,
    ) -> tuple[float | None, dict[str, Any]]:
        raise NotImplementedError

    def _run_with_timeout(self, func: Callable[[], Any], timeout: float) -> Any:
        result = func()
        if inspect.isawaitable(result):
            return asyncio.run(asyncio.wait_for(result, timeout=timeout))
        return result


class _RagasMetric(_JudgeMetric):
    provider = "ragas"
    dependency_name = "ragas"

    def _build_llm(self, metric_config: JudgeMetricConfig):
        try:
            from langchain_ollama import ChatOllama
            from ragas.llms import LangchainLLMWrapper

            return LangchainLLMWrapper(
                ChatOllama(
                    model=metric_config.model,
                    base_url=metric_config.base_url,
                    temperature=0,
                    timeout=metric_config.timeout,
                )
            )
        except ModuleNotFoundError:
            # Let Ragas use its default LLM when LangChain Ollama integration is
            # not installed. If that also fails, compute() returns null metadata.
            return None

    def _score_ragas_metric(self, metric: Any, sample: Any, timeout: float) -> tuple[float | None, dict[str, Any]]:
        if hasattr(metric, "single_turn_ascore"):
            raw = self._run_with_timeout(lambda: metric.single_turn_ascore(sample), timeout)
        elif hasattr(metric, "single_turn_score"):
            raw = self._run_with_timeout(lambda: metric.single_turn_score(sample), timeout)
        else:
            raise RuntimeError("Unsupported Ragas metric API")
        return (float(raw) if raw is not None else None), {"available": True}


class RagasFaithfulnessMetric(_RagasMetric):
    name = "ragas_faithfulness"
    score_key = "ragas_faithfulness"
    error_key = "ragas_faithfulness_error"

    def _score(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer, metric_config: JudgeMetricConfig) -> tuple[float | None, dict[str, Any]]:
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness

        llm = self._build_llm(metric_config)
        metric = Faithfulness(llm=llm) if llm is not None else Faithfulness()
        sample = SingleTurnSample(user_input=example.question, response=answer.answer, retrieved_contexts=self._contexts(answer))
        return self._score_ragas_metric(metric, sample, metric_config.timeout)


class RagasAnswerRelevancyMetric(_RagasMetric):
    name = "ragas_answer_relevancy"
    score_key = "ragas_answer_relevancy"
    error_key = "ragas_answer_relevancy_error"

    def _score(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer, metric_config: JudgeMetricConfig) -> tuple[float | None, dict[str, Any]]:
        from ragas import SingleTurnSample
        from ragas.metrics import ResponseRelevancy

        llm = self._build_llm(metric_config)
        metric = ResponseRelevancy(llm=llm) if llm is not None else ResponseRelevancy()
        sample = SingleTurnSample(user_input=example.question, response=answer.answer, retrieved_contexts=self._contexts(answer))
        return self._score_ragas_metric(metric, sample, metric_config.timeout)


class _DeepEvalMetric(_JudgeMetric):
    provider = "deepeval"
    dependency_name = "deepeval"

    def _build_model(self, metric_config: JudgeMetricConfig):
        try:
            from deepeval.models import OllamaModel

            return OllamaModel(model=metric_config.model, base_url=metric_config.base_url, timeout=metric_config.timeout)
        except (ImportError, TypeError, AttributeError):
            return metric_config.model

    def _score_deepeval_metric(self, metric: Any, test_case: Any, timeout: float) -> tuple[float | None, dict[str, Any]]:
        self._run_with_timeout(lambda: metric.measure(test_case), timeout)
        return (float(metric.score) if metric.score is not None else None), {
            "available": True,
            "reason": getattr(metric, "reason", None),
            "success": getattr(metric, "success", None),
        }


class DeepEvalFaithfulnessMetric(_DeepEvalMetric):
    name = "deepeval_faithfulness"
    score_key = "deepeval_faithfulness"
    error_key = "deepeval_faithfulness_error"

    def _score(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer, metric_config: JudgeMetricConfig) -> tuple[float | None, dict[str, Any]]:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        metric = FaithfulnessMetric(model=self._build_model(metric_config), async_mode=False)
        test_case = LLMTestCase(input=example.question, actual_output=answer.answer, retrieval_context=self._contexts(answer))
        return self._score_deepeval_metric(metric, test_case, metric_config.timeout)


class DeepEvalAnswerCorrectnessMetric(_DeepEvalMetric):
    name = "deepeval_answer_correctness"
    score_key = "deepeval_answer_correctness"
    error_key = "deepeval_answer_correctness_error"

    def _score(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer, metric_config: JudgeMetricConfig) -> tuple[float | None, dict[str, Any]]:
        from deepeval.metrics import AnswerCorrectnessMetric
        from deepeval.test_case import LLMTestCase

        metric = AnswerCorrectnessMetric(model=self._build_model(metric_config), async_mode=False)
        test_case = LLMTestCase(input=example.question, actual_output=answer.answer, expected_output=self._reference(example), retrieval_context=self._contexts(answer))
        return self._score_deepeval_metric(metric, test_case, metric_config.timeout)
