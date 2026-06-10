from __future__ import annotations
import statistics, time
from pathlib import Path
from ragbench_core.registry import register_builtin_components
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.data_loader import list_sources, load_ground_truth
from ragbench_core.engine.database import upsert_run, insert_result
from ragbench_core.engine.evaluation_trace import build_evaluation_step
from ragbench_core.models.schema import GeneratedAnswer
from ragbench_core.utils.ids import stable_id
from ragbench_core.utils.timing import timer

class BenchmarkRunner:
    def __init__(self):
        self.registry=register_builtin_components()

    def run(self, run_id: str, config: dict) -> dict:
        upsert_run(run_id, config.get("experiment",{}).get("name","benchmark"), "running", config)
        summary={"cases":0,"questions":0,"errors":0}
        try:
            cases=expand_cases(config)
            examples=load_ground_truth(config["data"]["ground_truth_file"])
            sources=list_sources(config["data"]["input_dir"], config["data"].get("remote_urls_file"))
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
            upsert_run(run_id, config.get("experiment",{}).get("name","benchmark"), "completed", config, summary)
            return summary
        except Exception as e:
            upsert_run(run_id, config.get("experiment",{}).get("name","benchmark"), "failed", config, summary, error=str(e))
            raise

    def _run_case(self, run_id, case, sources, examples, full_config):
        ing=case.ingestion_config; qry=case.query_config
        docs=[]
        for source in sources:
            for ex_name in ing.get("extractors", []):
                docs.extend(self.registry.extractors[ex_name].extract(source, {}))
        for c_conf in ing.get("cleaners", []):
            docs=self.registry.cleaners[c_conf["type"]].clean(docs, c_conf)
        chunks=self.registry.chunkers[ing["chunker"]["type"]].chunk(docs, ing["chunker"])
        for m_conf in ing.get("metadata_enrichers", []):
            chunks=self.registry.metadata_enrichers[m_conf["type"]].enrich(chunks, m_conf)
        embedder=self.registry.embedders[ing["embedder"]["provider"]]
        with timer() as ingest_timer:
            chunks=embedder.embed_chunks(chunks, ing["embedder"])
            index_name=f"{full_config.get('runtime',{}).get('collection_prefix','ragbench')}_{case.case_id}".lower().replace('-', '_')
            dense_indexer_conf=ing.get("indexers",{}).get("dense",{"type":"qdrant"})
            sparse_indexer_conf=ing.get("indexers",{}).get("sparse",{"type":"opensearch"})
            self.registry.indexers[dense_indexer_conf.get("type","qdrant")].index(chunks, index_name, dense_indexer_conf)
            self.registry.indexers[sparse_indexer_conf.get("type","opensearch")].index(chunks, index_name, sparse_indexer_conf)
        metric_rows=[]
        max_k=max(case.metric_config.get("retrieval_k", [10]))
        for example in examples:
            with timer() as query_timer:
                rewritten_queries=self.registry.rewriters[qry["rewriter"]["type"]].rewrite(example.question, qry["rewriter"], case.llm_config)
                retriever=self.registry.retrievers[qry["retriever"]["type"]]
                result_sets=[]
                for q in rewritten_queries:
                    q_emb=embedder.embed_query(q, ing["embedder"])
                    result_sets.append(retriever.retrieve(q, q_emb, index_name, qry["retriever"], max_k))
                if qry["retriever"]["type"] == "multi_query_fusion":
                    combined=retriever.fuse(result_sets, qry["retriever"], max_k)
                else:
                    combined=self._dedupe([result for result_set in result_sets for result in result_set])[:max_k]
                for f_conf in qry.get("filters", []):
                    combined=self.registry.filters[f_conf["type"]].filter(combined, f_conf)
                combined=self.registry.rerankers[qry["reranker"]["type"]].rerank(example.question, combined, qry["reranker"])
                combined=self.registry.compressors[qry["compressor"]["type"]].compress(example.question, combined, qry["compressor"])
                context=self.registry.assemblers[qry["context_assembler"]["type"]].assemble(combined, qry["context_assembler"])
                prompt=self.registry.prompt_builders[qry["prompt_builder"]["type"]].build(example.question, context, qry["prompt_builder"])
                answer_text=self.registry.generators[case.llm_config["provider"]].generate(prompt, case.llm_config)
                gen=GeneratedAnswer(answer_text, prompt, [r.chunk for r in combined], {"rewritten_queries":rewritten_queries})
                evaluation_step=build_evaluation_step(example.question, rewritten_queries, combined, context, prompt)
            metrics={"ingestion_latency_seconds":ingest_timer["seconds"],"query_latency_seconds":query_timer["seconds"],"chunk_count":len(chunks)}
            for metric in self.registry.metrics.values():
                metrics.update(metric.compute(example, combined, gen, case.metric_config))
            val=self.registry.citation_validators[qry["citation_validator"]["type"]].validate(gen, example.expected_citations, qry["citation_validator"])
            metrics.update(val)
            row={"case_id":case.case_id,"question":example.question,"metrics":metrics}
            metric_rows.append(row)
            result_id=stable_id(run_id,case.case_id,example.question)
            insert_result(
                result_id,
                run_id,
                case.case_id,
                example.question,
                metrics,
                answer_text,
                {"ingestion":ing,"query":qry,"llm":case.llm_config,"evaluation_step":evaluation_step},
            )
        return {"questions":len(examples),"metric_rows":metric_rows}

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
