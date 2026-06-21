from __future__ import annotations
from dataclasses import replace
import statistics, time
from pathlib import Path
from ragbench_core.registry import register_builtin_components
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.data_loader import list_sources, load_ground_truth
from ragbench_core.engine.database import upsert_run, insert_result
from ragbench_core.engine.evaluation_trace import build_evaluation_step
from ragbench_core.models.schema import GeneratedAnswer
from ragbench_core.components.metrics import RetrievalMetrics
from ragbench_core.utils.ids import stable_id
from ragbench_core.utils.timing import timer
from ragbench_core.utils.clients import consume_client_timings
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
        artifact_dir=str(artifact_root(config, run_id))
        config_hash=stable_hash(config)
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
        summary={"cases":0,"questions":0,"errors":0}
        try:
            cases=expand_cases(config)
            write_json_artifact(config, run_id, "expanded_cases.json", payload=[case.__dict__ for case in cases])
            examples=load_ground_truth(config["data"]["ground_truth_file"])
            sources=list_sources(config["data"]["input_dir"], config["data"].get("remote_urls_file"))
            sources=self._select_sources(sources, config.get("input_selection", {}).get("source_paths"))
            examples=self._select_examples(examples, config.get("input_selection", {}).get("question_indices"))
            versions["dataset_hash"]=dataset_hash(sources, examples)
            all_metric_rows=[]
            for case in cases:
                try:
                    case_summary=self._run_case(run_id, case, sources, examples, config)
                    summary["cases"] += 1
                    summary["questions"] += case_summary["questions"]
                    all_metric_rows.extend(case_summary["metric_rows"])
                except Exception as e:
                    summary["errors"] += 1
                    # Continue; a benchmark platform should isolate bad component configs.
            summary["aggregate"] = self._aggregate(all_metric_rows)
            exports=export_run(config, run_id, {"id":run_id,"name":run_name,"summary":summary,**versions}, all_metric_rows)
            versions["exports"]=exports
            summary["exports"]=exports
            upsert_run(run_id, run_name, "completed", config, summary, **versions)
            return summary
        except Exception as e:
            upsert_run(run_id, run_name, "failed", config, summary, error=str(e), **versions)
            raise

    def _run_case(self, run_id, case, sources, examples, full_config):
        ing=case.ingestion_config; qry=case.query_config
        embedding_cache=build_embedding_cache(full_config)
        retrieval_cache=build_retrieval_cache(full_config)
        docs=[]
        extractor_configs = ing.get("extractor_configs", {})
        for source in sources:
            for ex_name in ing.get("extractors", []):
                ex_config = extractor_configs.get(ex_name, {})
                docs.extend(self.registry.extractors[ex_name].extract(source, ex_config))
        for c_conf in ing.get("cleaners", []):
            docs=self.registry.cleaners[c_conf["type"]].clean(docs, c_conf)
        chunks=self.registry.chunkers[ing["chunker"]["type"]].chunk(docs, ing["chunker"])
        for m_conf in ing.get("metadata_enrichers", []):
            chunks=self.registry.metadata_enrichers[m_conf["type"]].enrich(chunks, m_conf)
        embedder=self.registry.embedders[ing["embedder"]["provider"]]
        with timer() as ingest_timer:
            with timer() as ingest_embedding_timer:
                chunks=embed_chunks_cached(embedding_cache, embedder, chunks, ing["embedder"])
            index_name=f"{full_config.get('runtime',{}).get('collection_prefix','ragbench')}_{case.case_id}".lower().replace('-', '_')
            dense_indexer_conf=ing.get("indexers",{}).get("dense")
            sparse_indexer_conf=ing.get("indexers",{}).get("sparse")
            with timer() as persistence_timer:
                if dense_indexer_conf:
                    self.registry.indexers[dense_indexer_conf.get("type","qdrant")].index(chunks, index_name, dense_indexer_conf)
                if sparse_indexer_conf:
                    self.registry.indexers[sparse_indexer_conf.get("type","opensearch")].index(chunks, index_name, sparse_indexer_conf)
        ingest_client_timings=consume_client_timings()
        metric_rows=[]
        max_k = max(RetrievalMetrics._validate_retrieval_k(case.metric_config))
        for example in examples:
            with timer() as query_timer:
                rewritten_queries=self.registry.rewriters[qry["rewriter"]["type"]].rewrite(example.question, qry["rewriter"], case.llm_config)
                retriever=self.registry.retrievers[qry["retriever"]["type"]]
                with timer() as query_embedding_timer:
                    q_embs=embed_queries_cached(embedding_cache, embedder, rewritten_queries, ing["embedder"])
                with timer() as retrieval_timer:
                    result_sets=retrieve_batch_cached(retrieval_cache, retriever, rewritten_queries, q_embs, index_name, qry["retriever"], max_k)
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
                combined=self.registry.compressors[qry["compressor"]["type"]].compress(example.question, combined, qry["compressor"])
                context=self.registry.assemblers[qry["context_assembler"]["type"]].assemble(combined, qry["context_assembler"])
                prompt=self.registry.prompt_builders[qry["prompt_builder"]["type"]].build(example.question, context, qry["prompt_builder"])
                with timer() as generation_timer:
                    answer_text=self.registry.generators[case.llm_config["provider"]].generate(prompt, case.llm_config)
                gen=GeneratedAnswer(
                    answer_text,
                    prompt,
                    [r.chunk for r in combined],
                    {
                        "rewritten_queries": rewritten_queries,
                        "raw_retrieval_results": raw_retrieval_results,
                        "reranked_results": reranked_results,
                    },
                )
                evaluation_step=build_evaluation_step(example.question, rewritten_queries, combined, context, prompt)
            query_client_timings=consume_client_timings()
            metrics={
                "ingestion_latency_seconds":ingest_timer["seconds"],
                "query_latency_seconds":query_timer["seconds"],
                "client_initialization_seconds":ingest_client_timings.get("client_initialization_seconds", 0.0) + query_client_timings.get("client_initialization_seconds", 0.0),
                "ingestion_embedding_seconds":ingest_embedding_timer["seconds"],
                "query_embedding_seconds":query_embedding_timer["seconds"],
                "retrieval_seconds":retrieval_timer["seconds"],
                "reranking_seconds":reranking_timer["seconds"],
                "generation_seconds":generation_timer["seconds"],
                "persistence_seconds":persistence_timer["seconds"],
                "chunk_count":len(chunks),
            }
            for metric in self.registry.metrics.values():
                metrics.update(metric.compute(example, combined, gen, case.metric_config))
            val=self.registry.citation_validators[qry["citation_validator"]["type"]].validate(gen, example.expected_citations, qry["citation_validator"])
            metrics.update(val)
            row={"case_id":case.case_id,"question":example.question,"metrics":metrics}
            metric_rows.append(row)
            result_id=stable_id(run_id,case.case_id,example.question)
            with timer() as result_persistence_timer:
                insert_result(
                    result_id,
                    run_id,
                    case.case_id,
                    example.question,
                    metrics,
                    answer_text,
                    {"ingestion":ing,"query":qry,"llm":case.llm_config,"evaluation_step":evaluation_step},
                    artifact_paths=self._write_result_artifacts(full_config, run_id, case.case_id, result_id, {
                        "expanded_config":{"ingestion":ing,"query":qry,"llm":case.llm_config,"metrics":case.metric_config},
                        "retrieval_traces":{"raw":retrieval_results_payload(raw_retrieval_results),"reranked":retrieval_results_payload(reranked_results)},
                        "prompt":{"question":example.question,"rewritten_queries":rewritten_queries,"context":context,"prompt":prompt},
                        "response":{"answer":answer_text,"metadata":gen.metadata},
                        "metrics":metrics,
                    }),
                    model_identifiers=model_identifiers(full_config),
                )
            metrics["result_persistence_seconds"]=result_persistence_timer["seconds"]
        return {"questions":len(examples),"metric_rows":metric_rows}

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
        numeric={}
        for row in rows:
            for k,v in row["metrics"].items():
                if isinstance(v,(int,float)):
                    numeric.setdefault(k,[]).append(float(v))
        return {k:{"mean":sum(vals)/len(vals),"min":min(vals),"max":max(vals)} for k,vals in numeric.items() if vals}
