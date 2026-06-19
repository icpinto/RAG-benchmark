from __future__ import annotations
import asyncio
import importlib
import importlib.util
import math
from typing import Any

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

class RagasGenerationMetric(BaseMetric):
    name = "ragas_generation"
    _DEFAULT_VALUES = {
        "ragas_faithfulness": None,
        "ragas_answer_relevancy": None,
    }

    def compute(
        self,
        example: QueryExample,
        results: list[RetrievalResult],
        answer: GeneratedAnswer | None,
        config: dict,
    ) -> dict:
        values: dict[str, Any] = dict(self._DEFAULT_VALUES)
        if not answer or not answer.answer:
            values["ragas_error"] = "missing generated answer"
            return values

        contexts = [chunk.text for chunk in answer.context_chunks if chunk.text]
        if not contexts:
            values["ragas_error"] = "missing retrieved contexts"
            return values

        missing = [
            dependency
            for dependency in self._dependencies()
            if importlib.util.find_spec(dependency) is None
        ]
        if missing:
            values["ragas_error"] = f"missing optional dependency: {', '.join(missing)}"
            return values

        try:
            HuggingFaceEmbeddings = importlib.import_module(
                "langchain_huggingface"
            ).HuggingFaceEmbeddings
            ChatOllama = importlib.import_module("langchain_ollama").ChatOllama
            LangchainEmbeddingsWrapper = importlib.import_module(
                "ragas.embeddings"
            ).LangchainEmbeddingsWrapper
            LangchainLLMWrapper = importlib.import_module(
                "ragas.llms"
            ).LangchainLLMWrapper
        except Exception as import_error:
            values["ragas_error"] = f"ragas dependency import failed: {import_error}"
            return values

        llm_config = config.get("llm", {})
        try:
            llm = self._build_llm(llm_config, ChatOllama, LangchainLLMWrapper)
            embeddings = self._build_embeddings(
                config, HuggingFaceEmbeddings, LangchainEmbeddingsWrapper
            )
        except Exception as build_error:
            values["ragas_error"] = f"ragas dependency setup failed: {build_error}"
            return values

        try:
            values.update(self._score_collections(example.question, answer.answer, contexts, llm, embeddings))
        except Exception as collections_error:
            try:
                values.update(self._score_legacy(example.question, answer.answer, contexts, llm, embeddings))
            except Exception as legacy_error:
                values["ragas_error"] = (
                    f"collections API failed: {collections_error}; "
                    f"legacy API failed: {legacy_error}"
                )
        return values

    def _dependencies(self) -> list[str]:
        return ["ragas", "langchain_ollama", "langchain_huggingface"]

    def _build_llm(self, llm_config: dict, chat_ollama_cls: Any, wrapper_cls: Any) -> Any:
        provider = llm_config.get("provider", "ollama")
        if provider != "ollama":
            raise ValueError(f"RagasGenerationMetric currently supports provider='ollama', got {provider!r}")
        model = llm_config.get("model", "gpt-oss:20b")
        base_url = llm_config.get("base_url", "http://ollama:11434")
        temperature = llm_config.get("temperature", 0)
        timeout = int(llm_config.get("timeout", 240))
        return wrapper_cls(
            chat_ollama_cls(
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
            )
        )

    def _build_embeddings(self, config: dict, embeddings_cls: Any, wrapper_cls: Any) -> Any:
        embedding_config = config.get("embedder", {})
        model = config.get("ragas_embeddings_model") or embedding_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return wrapper_cls(embeddings_cls(model_name=model))

    def _score_collections(
        self, question: str, response: str, contexts: list[str], llm: Any, embeddings: Any
    ) -> dict[str, float | None]:
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        faithfulness = Faithfulness(llm=llm).score(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
        )
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings).score(
            user_input=question,
            response=response,
        )
        return {
            "ragas_faithfulness": self._metric_value(faithfulness),
            "ragas_answer_relevancy": self._metric_value(relevancy),
        }

    def _score_legacy(
        self, question: str, response: str, contexts: list[str], llm: Any, embeddings: Any
    ) -> dict[str, float | None]:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy

        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
        )
        faithfulness = Faithfulness(llm=llm)
        relevancy = ResponseRelevancy(llm=llm, embeddings=embeddings)
        return {
            "ragas_faithfulness": self._run_async(faithfulness.single_turn_ascore(sample)),
            "ragas_answer_relevancy": self._run_async(relevancy.single_turn_ascore(sample)),
        }

    def _run_async(self, awaitable: Any) -> float | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._metric_value(asyncio.run(awaitable))
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise RuntimeError("Ragas async legacy scoring cannot run inside an active event loop") from None

    def _metric_value(self, value: Any) -> float | None:
        raw = getattr(value, "value", value)
        return float(raw) if raw is not None else None
