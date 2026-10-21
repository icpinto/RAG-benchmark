from __future__ import annotations
from dataclasses import replace
import math
import re
import traceback
from pathlib import Path
from ragbench_core.registry import register_builtin_components
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.data_loader import list_sources, load_ground_truth
from ragbench_core.engine.database import upsert_run, insert_result
from ragbench_core.engine.evaluation_trace import build_evaluation_step
from ragbench_core.models.schema import GeneratedAnswer
from ragbench_core.components.metrics import RetrievalMetrics, count_texts_tokens, metric_selected, filter_metrics, ANSWER_SELECTIONS
from ragbench_core.components.judges import judge_answer, SCORES
from ragbench_core.engine.snapshots import snapshot_inputs
from ragbench_core.utils.ids import stable_id
from ragbench_core.utils.timing import timer
from ragbench_core.utils.clients import consume_client_timings, consume_token_usage
from ragbench_core.engine.cache import (
    build_embedding_cache,
    build_retrieval_cache,
    embed_chunks_cached,
    embed_queries_cached,
    retrieve_batch_cached,
)
from ragbench_core.engine.persistence import (
    artifact_root,
    component_versions,
    dataset_hash,
    dependency_versions,
    export_run,
    model_identifiers,
    retrieval_results_payload,
    stable_hash,
    write_json_artifact,
)

class BenchmarkRunner:
    def __init__(self):
        self.registry=register_builtin_components()

    def run(self, run_id: str, config: dict) -> dict:
        run_name=config.get("experiment",{}).get("name","benchmark")
        try:
            config = snapshot_inputs(run_id, config)
        except Exception as exc:
            upsert_run(run_id, run_name, "failed", config, error=str(exc))
            raise
        artifact_dir=str(artifact_root(config, run_id))
        config_hash=stable_hash({k: v for k, v in config.items() if k != "_snapshot"})
        models=model_identifiers(config)
        versions={
            "config_hash": config_hash,
            "component_versions": component_versions(self.registry),
            "dependency_versions": dependency_versions(),
            "model_identifiers": models,
            "artifact_dir": artifact_dir,
        }
        upsert_run(run_id, run_name, "running", config, **versions)
        write_json_artifact(config, run_id, "expanded_config.json", payload=config)
        summary={"cases":0,"questions":0,"errors":0,"case_errors":[],"case_aggregates":{}}
        try:
            cases=expand_cases(config)
            write_json_artifact(config, run_id, "expanded_cases.json", payload=[case.__dict__ for case in cases])
            examples=load_ground_truth(config["_snapshot"]["ground_truth_file"])
            sources=[entry["path"] for entry in config["_snapshot"]["sources"]]
            versions["dataset_hash"]=config["_snapshot"]["dataset_hash"]
            summary["total_cases"] = len(cases)
            summary["total_questions"] = len(cases) * len(examples)
            all_metric_rows=[]
            for case in cases:
                try:
                    case_summary=self._run_case(run_id, case, sources, examples, config)
                    summary["cases"] += int(case_summary["questions"] > 0)
                    summary["questions"] += case_summary["questions"]
                    summary["errors"] += len(case_summary.get("question_errors", []))
                    summary["case_errors"].extend(case_summary.get("question_errors", []))
                    all_metric_rows.extend(case_summary["metric_rows"])
                    summary["case_aggregates"][case.case_id] = self._aggregate(case_summary["metric_rows"])
                except Exception as e:
                    summary["errors"] += 1
                    error_payload=self._case_error_payload(case.case_id, e)
                    error_payload["artifact_path"]=write_json_artifact(
                        config,
                        run_id,
                        "cases",
                        case.case_id,
                        "case_error.json",
                        payload=error_payload,
                    )
                    summary["case_errors"].append(error_payload)
                    failed_rows = [{"case_id": case.case_id, "question_id": e.question_id, "metrics": {}, "failed": True} for e in examples]
                    all_metric_rows.extend(failed_rows)
                    summary["case_aggregates"][case.case_id] = self._aggregate(failed_rows)
                    # Continue; a benchmark platform should isolate bad component configs.
            summary["aggregate"] = self._aggregate(all_metric_rows)
            exports=export_run(config, run_id, {"id":run_id,"name":run_name,"summary":summary,**versions}, all_metric_rows)
            versions["exports"]=exports
            summary["exports"]=exports
            status = "failed" if not summary["cases"] else "completed_with_errors" if summary["errors"] else "completed"
            upsert_run(run_id, run_name, status, config, summary, error="No benchmark case succeeded" if status == "failed" else None, **versions)
            return summary
        except Exception as e:
            upsert_run(run_id, run_name, "failed", config, summary, error=str(e), **versions)
            raise


    def _case_error_payload(self, case_id, error):
        formatted_traceback=traceback.format_exception(type(error), error, error.__traceback__)
        sanitized_traceback=[line.replace(str(Path.cwd()), ".") for line in formatted_traceback]
        return {
            "case_id": case_id,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": sanitized_traceback,
        }

    def _run_case(self, run_id, case, sources, examples, full_config):
        full_config = {**full_config, "_run_id": run_id}
        ingestion = self._ingest_case(case, sources, full_config)
        ingestion["per_question_embedding_tokens"] = ingestion["embedding_tokens"] / len(examples) if examples else 0
        caches = {
            "embedding": ingestion["embedding_cache"],
            "retrieval": build_retrieval_cache(full_config),
        }
        metric_config = {**case.metric_config, "pricing": full_config.get("runtime", {}).get("pricing", {}),
                         "runtime_mode": full_config.get("runtime", {}).get("mode", "end_to_end")}
        metric_rows=[]
        question_errors=[]
        for example in examples:
            try:
                answer_payload = self._answer_question(example, case, ingestion["chunks"], ingestion["index_name"], caches, metric_config)
                row = self._compute_and_persist_result(run_id, example, case, full_config, metric_config, ingestion, answer_payload)
            except Exception as exc:
                error = {**self._case_error_payload(case.case_id, exc), "question_id": example.question_id}
                path = write_json_artifact(full_config, run_id, "cases", case.case_id, f"{example.question_id}_error.json", payload=error)
                question_errors.append({**error, "artifact_path": path})
                row = {"case_id": case.case_id, "question_id": example.question_id, "question": example.question, "metrics": {}, "failed": True}
                insert_result(stable_id(run_id, case.case_id, example.question_id), run_id, case.case_id, example.question, {}, "",
                              {"question_id": example.question_id, "error": error, "outcome": "failed"},
                              artifact_paths={"error": path}, question_id=example.question_id)
                consume_client_timings()
                consume_token_usage()
            metric_rows.append(row)
        return {"questions":sum(not r.get("failed", False) for r in metric_rows),"metric_rows":metric_rows,"question_errors":question_errors}

    def _ingest_case(self, case, sources, full_config):
        ing=case.ingestion_config
        embedding_cache=build_embedding_cache(full_config)
        docs=[]
        extractor_configs = ing.get("extractor_configs", {})
        snapshot_sources = {entry["path"]: entry for entry in full_config.get("_snapshot", {}).get("sources", [])}
        for source in sources:
            entry = snapshot_sources.get(source, {})
            for ex_name in ing.get("extractors", []):
                ex_config = dict(extractor_configs.get(ex_name, {}))
                if entry.get("remote"):
                    if ex_name != "html_url":
                        continue
                    ex_config.update(_snapshot_html=source, _snapshot_encoding=entry["encoding"])
                extracted = self.registry.extractors[ex_name].extract(entry["source"] if entry.get("remote") else source, ex_config)
                for document in extracted:
                    document.source = entry.get("source", source)
                docs.extend(extracted)
        if not docs:
            raise ValueError("No documents were extracted from the selected sources")
        for c_conf in ing.get("cleaners", []):
            docs=self.registry.cleaners[c_conf["type"]].clean(docs, c_conf)
        chunks=self.registry.chunkers[ing["chunker"]["type"]].chunk(docs, ing["chunker"])
        if not chunks:
            raise ValueError("Chunking produced no chunks")
        for m_conf in ing.get("metadata_enrichers", []):
            chunks=self.registry.metadata_enrichers[m_conf["type"]].enrich(chunks, m_conf)
        embedder=self.registry.embedders[ing["embedder"]["provider"]]
        with timer() as ingest_timer:
            with timer() as ingest_embedding_timer:
                chunks=embed_chunks_cached(embedding_cache, embedder, chunks, ing["embedder"])
            prefix = re.sub(r"[^a-z0-9_]", "_", full_config.get("runtime", {}).get("collection_prefix", "ragbench").lower())[:80].strip("_") or "ragbench"
            index_name=f"{prefix}_{stable_id(full_config['_run_id'])}_{case.case_id}".lower().replace('-', '_')
            dense_indexer_conf=ing.get("indexers",{}).get("dense")
            sparse_indexer_conf=ing.get("indexers",{}).get("sparse")
            with timer() as persistence_timer:
                if dense_indexer_conf:
                    self.registry.indexers[dense_indexer_conf.get("type","qdrant")].index(chunks, index_name, dense_indexer_conf)
                if sparse_indexer_conf:
                    self.registry.indexers[sparse_indexer_conf.get("type","opensearch")].index(chunks, index_name, sparse_indexer_conf)
        ingest_client_timings=consume_client_timings()
        ingest_token_usage=consume_token_usage()
        estimated_ingest_embedding_tokens = count_texts_tokens([chunk.text for chunk in chunks], ing["embedder"].get("model"))
        ingest_embedding_tokens = ingest_token_usage.get("embedding_tokens") or estimated_ingest_embedding_tokens
        return {
            "chunks": chunks,
            "index_name": index_name,
            "embedder": embedder,
            "embedding_cache": embedding_cache,
            "client_timings": ingest_client_timings,
            "token_usage": ingest_token_usage,
            "embedding_tokens": ingest_embedding_tokens,
            "timings": {
                "ingestion_latency_seconds": ingest_timer["seconds"],
                "ingestion_embedding_seconds": ingest_embedding_timer["seconds"],
                "persistence_seconds": persistence_timer["seconds"],
            },
        }

    def _answer_question(self, example, case, chunks, index_name, caches, metric_config):
        ing=case.ingestion_config; qry=case.query_config
        retrieval_only = metric_config.get("runtime_mode") == "retrieval_only"
        max_k = max(RetrievalMetrics._validate_retrieval_k(metric_config))
        embedder=self.registry.embedders[ing["embedder"]["provider"]]
        with timer() as query_timer:
            rewritten_queries=self.registry.rewriters[qry["rewriter"]["type"]].rewrite(example.question, qry["rewriter"], case.llm_config)
            retriever=self.registry.retrievers[qry["retriever"]["type"]]
            with timer() as query_embedding_timer:
                q_embs=embed_queries_cached(caches["embedding"], embedder, rewritten_queries, ing["embedder"])
            with timer() as retrieval_timer:
                result_sets=retrieve_batch_cached(caches["retrieval"], retriever, rewritten_queries, q_embs, index_name, qry["retriever"], max_k)
            if qry["retriever"]["type"] == "multi_query_fusion":
                combined=retriever.fuse(result_sets, qry["retriever"], max_k)
            else:
                combined=self._dedupe([result for result_set in result_sets for result in result_set])[:max_k]
            raw_retrieval_results=self._snapshot_results(combined)
            for f_conf in qry.get("filters", []):
                combined=self.registry.filters[f_conf["type"]].filter(combined, f_conf)
            with timer() as reranking_timer:
                combined=self.registry.rerankers[qry["reranker"]["type"]].rerank(example.question, combined, qry["reranker"])
            reranked_results=self._snapshot_results(combined)
            context, prompt = "", ""
            if not retrieval_only:
                combined=self.registry.compressors[qry["compressor"]["type"]].compress(example.question, combined, qry["compressor"])
                context=self.registry.assemblers[qry["context_assembler"]["type"]].assemble(combined, qry["context_assembler"])
                prompt=self.registry.prompt_builders[qry["prompt_builder"]["type"]].build(example.question, context, qry["prompt_builder"])
            with timer() as generation_timer:
                generation_result="" if retrieval_only else self.registry.generators[case.llm_config["provider"]].generate(prompt, case.llm_config)
            if isinstance(generation_result, dict):
                answer_text=str(generation_result.get("answer", ""))
                generation_metadata=dict(generation_result.get("metadata") or {})
            else:
                answer_text=str(generation_result)
                generation_metadata={}
            query_embedding_token_estimate=count_texts_tokens(list(rewritten_queries), ing["embedder"].get("model"))
            gen=GeneratedAnswer(
                answer_text,
                prompt,
                [r.chunk for r in combined],
                {
                    **generation_metadata,
                    "rewritten_queries": rewritten_queries,
                    "raw_retrieval_results": raw_retrieval_results,
                    "reranked_results": reranked_results,
                    "context_text": context,
                    "embedding_model": ing["embedder"].get("model"),
                    "embedding_tokens": query_embedding_token_estimate,
                    "retrieval_only": retrieval_only,
                },
            )
            evaluation_step=build_evaluation_step(example.question, rewritten_queries, combined, context, prompt)
        query_client_timings=consume_client_timings()
        query_token_usage=consume_token_usage()
        return {
            "generated_answer": gen,
            "answer_text": answer_text,
            "selected_results": combined,
            "raw_retrieval_results": raw_retrieval_results,
            "reranked_results": reranked_results,
            "rewritten_queries": rewritten_queries,
            "context": context,
            "prompt": prompt,
            "evaluation_step": evaluation_step,
            "client_timings": query_client_timings,
            "token_usage": query_token_usage,
            "query_embedding_token_estimate": query_embedding_token_estimate,
            "timings": {
                "query_latency_seconds":query_timer["seconds"],
                "query_embedding_seconds":query_embedding_timer["seconds"],
                "retrieval_seconds":retrieval_timer["seconds"],
                "reranking_seconds":reranking_timer["seconds"],
                **({} if retrieval_only else {"generation_seconds":generation_timer["seconds"]}),
            },
        }

    def _compute_and_persist_result(self, run_id, example, case, full_config, metric_config, ingestion, answer_payload):
        ing=case.ingestion_config; qry=case.query_config
        gen=answer_payload["generated_answer"]
        retrieval_only = metric_config.get("runtime_mode") == "retrieval_only"
        query_token_usage=answer_payload["token_usage"]
        per_question_ingest_embedding_tokens = ingestion.get("per_question_embedding_tokens", 0)
        if query_token_usage.get("embedding_tokens") or ingestion["token_usage"].get("embedding_tokens"):
            gen.metadata["embedding_tokens"] = per_question_ingest_embedding_tokens + query_token_usage.get("embedding_tokens", answer_payload["query_embedding_token_estimate"])
        else:
            gen.metadata["embedding_tokens"] = per_question_ingest_embedding_tokens + answer_payload["query_embedding_token_estimate"]
        metrics={
            **ingestion["timings"],
            **answer_payload["timings"],
            "client_initialization_seconds":ingestion["client_timings"].get("client_initialization_seconds", 0.0) + answer_payload["client_timings"].get("client_initialization_seconds", 0.0),
            "chunk_count":len(ingestion["chunks"]),
        }
        for metric in self.registry.metrics.values():
            if metric_selected(metric.name, metric_config) and not (retrieval_only and metric.name in ANSWER_SELECTIONS):
                metrics.update(metric.compute(example, answer_payload["selected_results"], gen, metric_config))
        if not retrieval_only and metric_selected("citation_coverage", metric_config):
            val=self.registry.citation_validators[qry["citation_validator"]["type"]].validate(gen, example.expected_citations, qry["citation_validator"])
            metrics.update(val)
        judge = None
        judge_config = metric_config.get("judge", {})
        if not retrieval_only and judge_config.get("enabled") and any(metric_selected(name, metric_config) for name in SCORES):
            judge = judge_answer(example, gen, judge_config)
            metrics.update(judge["scores"])
        metrics = filter_metrics(metrics, metric_config)
        row={"case_id":case.case_id,"question_id":example.question_id,"question":example.question,"metrics":metrics}
        result_id=stable_id(run_id,case.case_id,example.question_id)
        insert_result(
                result_id,
                run_id,
                case.case_id,
                example.question,
                metrics,
                answer_payload["answer_text"],
                {"ingestion":ing,"query":qry,"llm":case.llm_config,"question_id":example.question_id,"evaluation_step":answer_payload["evaluation_step"],"judge":judge},
                artifact_paths=self._write_result_artifacts(full_config, run_id, case.case_id, result_id, {
                    "expanded_config":{"ingestion":ing,"query":qry,"llm":case.llm_config,"metrics":metric_config},
                    "retrieval_traces":{"raw":retrieval_results_payload(answer_payload["raw_retrieval_results"]),"reranked":retrieval_results_payload(answer_payload["reranked_results"])},
                    "prompt":{"question":example.question,"rewritten_queries":answer_payload["rewritten_queries"],"context":answer_payload["context"],"prompt":answer_payload["prompt"]},
                    "response":{"answer":answer_payload["answer_text"],"metadata":gen.metadata},
                    "metrics":metrics,
                    **({"judge": judge} if judge is not None else {}),
                }),
                model_identifiers=model_identifiers(full_config),
                question_id=example.question_id,
        )
        return row

    def _write_result_artifacts(self, config, run_id, case_id, result_id, payloads):
        paths={}
        for name,payload in payloads.items():
            paths[name]=write_json_artifact(config, run_id, "cases", case_id, f"{result_id}_{name}.json", payload=payload)
        return paths

    def _select_sources(self, sources, selected_source_paths):
        if selected_source_paths is None:
            return sources
        selected={str(Path(source).resolve()) for source in selected_source_paths}
        return [source for source in sources if str(Path(source).resolve()) in selected]

    def _select_examples(self, examples, selected_question_indices):
        if selected_question_indices is None:
            return examples
        return [example for index,example in enumerate(examples) if index in set(selected_question_indices)]

    def _snapshot_results(self, results):
        return [replace(result) for result in results]

    def _dedupe(self, results):
        seen=set(); out=[]
        for r in sorted(results, key=lambda x:x.score, reverse=True):
            if r.chunk.chunk_id in seen: continue
            seen.add(r.chunk.chunk_id); out.append(r)
        for i,r in enumerate(out): r.rank=i+1
        return out

    def _aggregate(self, rows):
        keys = {k for row in rows for k, v in row["metrics"].items() if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))}
        aggregates = {}
        for key in sorted(keys):
            values = [r["metrics"].get(key) for r in rows if not r.get("failed")]
            valid = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
            failed = sum(bool(r.get("failed")) for r in rows)
            aggregates[key] = {"mean":sum(valid)/len(valid) if valid else None, "min":min(valid) if valid else None,
                               "max":max(valid) if valid else None,"sum":sum(valid), "valid":len(valid),
                               "missing":len(rows)-failed-len(valid), "failed":failed}
        return {**aggregates,"result_counts":{"valid":sum(not r.get("failed", False) for r in rows),
                                                   "failed":sum(bool(r.get("failed")) for r in rows), "total":len(rows)}}
