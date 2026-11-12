"""Optimizer policy, durable recovery, and deterministic end-to-end acceptance."""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ragbench_core.engine import database as db
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.persistence import stable_hash
from ragbench_core.engine.snapshots import (
    attach_evaluation,
    capture_corpus,
    verify_snapshot,
)
from ragbench_core.models.schema import Document, QueryExample
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.contracts import (
    OptimizationRequest,
    TrialRequest,
)
from ragbench_core.optimization.datasets import (
    freeze_split,
    legacy_question,
    validate_questions,
)
from ragbench_core.optimization.search import (
    concrete_config,
    get_search_space,
    validate_candidate,
)
from ragbench_core.optimization.selection import compare_trials
from ragbench_core.optimization.tools import OptimizationPreflightError, OptimizationTools
from ragbench_core.optimization.workflow import build_graph
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'optimization.sqlite'}")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    db.init_db()
    yield db
    engine.dispose()


@pytest.fixture
def template(tmp_path):
    return {
        "experiment": {"name": "fixture"},
        "data": {
            "input_dir": str(tmp_path / "raw"),
            "ground_truth_file": "unused.json",
        },
        "pipeline": {
            "ingestion": {
                "extractors": ["txt"],
                "cleaners": [],
                "metadata_enrichers": [],
                "chunkers": [{"type": "fixed", "chunk_size": 512, "overlap": 0}],
                "embeddings": [{"provider": "fake", "model": "fake"}],
                "indexers": {
                    "dense": {"type": "qdrant", "strategy": "dense_hnsw"},
                    "sparse": {"type": "opensearch", "strategy": "bm25_standard"},
                },
            },
            "query": {"retrievers": [{"type": "dense"}]},
        },
        "llm": {
            "provider": "ollama",
            "model": "fixed-answer",
            "max_output_tokens": 1024,
        },
        "metrics": {"retrieval_k": [10]},
        "runtime": {"artifact_dir": str(tmp_path / "artifacts"), "max_combinations": 1},
    }


@pytest.fixture
def session(database, template):
    request = OptimizationRequest(
        limits={"target_questions": 30, "retrieval_trials": 2, "answer_trials": 2}
    ).model_dump()
    with db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationSession(
                id="session", request=request, template=template, progress={}
            )
        )
    return "session"


def fake_registry():
    return SimpleNamespace(
        extractors={
            "txt": SimpleNamespace(
                extract=lambda source, cfg: [
                    Document(Path(source).name, source, Path(source).read_text())
                ]
            )
        },
        cleaners={},
        chunkers={"fixed": object(), "llamaindex_hierarchical": object()},
        embedders={"fake": object()},
        rewriters={"none": object()},
        retrievers={
            "dense": object(),
            "bm25": object(),
            "hybrid": object(),
            "parent_document": object(),
        },
        rerankers={"none": object()},
        compressors={"none": object(), "top_n": object()},
        indexers={"qdrant": object(), "opensearch": object()},
    )


def questions(count=30):
    result = []
    for index in range(count):
        token = stable_hash(index)[:20]
        result.append(
            {
                "question": f"Explain {token} requirements?",
                "expected_answer": f"Requirement {token}",
                "evidence": [
                    {"doc_id": f"{index}.txt", "quote": f"Requirement {token}"}
                ],
            }
        )
    return result


def prepare_dataset(tools):
    raw = questions()
    docs = [
        {"doc_id": q["evidence"][0]["doc_id"], "text": q["evidence"][0]["quote"]}
        for q in raw
    ]
    validated, _ = validate_questions(
        raw, docs, lambda q: {"answerable": True, "supported": True}
    )
    store.save_dataset(tools.session_id, freeze_split(validated), frozen=True)
    tools.write("documents.json", docs)
    tools.write("profile.json", {"documents": len(docs)})
    raw_dir = tools.root / "raw"
    raw_dir.mkdir(parents=True)
    paths = []
    for doc in docs:
        path = raw_dir / doc["doc_id"]
        path.write_text(doc["text"])
        paths.append(str(path))
    tools.write("corpus.json", capture_corpus(paths, tools.root / "snapshot"))


def test_corpus_capture_without_gold_and_location_independent_evaluation(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("Refunds within fourteen days")
    manifest = capture_corpus([str(source)], tmp_path / "corpus")
    source.write_text("changed after capture")
    snapshot = attach_evaluation(
        manifest,
        [QueryExample(question="When?", ground_truth_doc_ids=["policy.txt"])],
        tmp_path / "evaluation",
    )
    verify_snapshot({"_snapshot": snapshot})
    assert "fourteen" in Path(snapshot["sources"][0]["path"]).read_text()
    assert "ground_truth_file" not in manifest


def test_evidence_and_near_duplicate_rejection():
    raw = questions(2)
    docs = [
        {"doc_id": q["evidence"][0]["doc_id"], "text": q["evidence"][0]["quote"]}
        for q in raw
    ]
    bad = deepcopy(raw[1])
    bad["evidence"][0]["doc_id"] = "wrong"
    accepted, rejected = validate_questions([raw[0], deepcopy(raw[0]), bad, None], docs)
    assert len(accepted) == 1 and len(rejected) == 3
    assert not accepted[0].automated_validated
    assert not accepted[0].human_reviewed


def test_split_is_deterministic_and_keeps_overlapping_evidence_together():
    raw = questions(10)
    raw[1]["evidence"] = deepcopy(raw[0]["evidence"])
    docs = [
        {"doc_id": q["evidence"][0]["doc_id"], "text": q["evidence"][0]["quote"]}
        for q in raw
    ]
    accepted, _ = validate_questions(raw, docs)
    first = freeze_split(accepted)
    second = freeze_split(list(reversed(accepted)))
    assert {q["question_id"]: q["split"] for q in first} == {
        q["question_id"]: q["split"] for q in second
    }
    assert first[0]["split"] == first[1]["split"]
    assert set(legacy_question(first[0])) == set(QueryExample(question="x").__dict__)


def test_hosted_opt_in_and_explicit_prices():
    with pytest.raises(ValueError, match="opt-in"):
        OptimizationRequest(planner={"provider": "chat_completions"})
    with pytest.raises(ValueError, match="prices"):
        OptimizationRequest(limits={"max_cost": 1})
    assert OptimizationRequest().limits.target_questions == 100


def test_candidate_rejects_incompatible_indexes_and_hidden_sweeps(template):
    baseline = concrete_config(template, expand_cases(template)[0])
    space = get_search_space(template)
    bad = deepcopy(baseline)
    bad["pipeline"]["ingestion"]["indexers"].pop("sparse")
    bad["pipeline"]["query"]["retrievers"] = [{"type": "hybrid"}]
    with pytest.raises(ValueError, match="sparse index"):
        validate_candidate(bad, baseline, space)
    bad = deepcopy(baseline)
    bad["pipeline"]["ingestion"]["chunkers"][0]["chunk_size"] = [256, 512]
    with pytest.raises(ValueError, match="exactly one"):
        validate_candidate(bad, baseline, space)
    bad = deepcopy(baseline)
    bad["llm"]["model"] = "other"
    with pytest.raises(ValueError, match="fixed"):
        validate_candidate(bad, baseline, space)


def test_frozen_dataset_and_budget_reservations_survive_replay(session):
    store.save_dataset(session, [], frozen=True)
    with pytest.raises(ValueError, match="frozen"):
        store.save_dataset(session, [])
    assert budget.reserve(session, "op", "planner", 500, None)
    assert not budget.reserve(session, "op", "planner", 500, None)
    assert budget.totals(session)["tokens"] == 500
    assert budget.totals(session)["cost"] is None
    with store.locked(session) as (_, row):
        req = deepcopy(row.request)
        req["limits"]["max_tokens"] = 600
        row.request = req
    with pytest.raises(budget.BudgetExhausted):
        budget.reserve(session, "second", "planner", 101, None)
    store.update(session, status="paused")
    with pytest.raises(budget.OptimizationStopped):
        budget.check_control(session)


def test_session_lease_rejects_duplicate_delivery(session):
    assert store.acquire(session, "one")
    assert not store.acquire(session, "two")
    store.release(session, "two")
    assert not store.acquire(session, "three")
    store.release(session, "one")
    assert store.acquire(session, "two")


def test_ranking_requires_paired_coverage_and_reports_unknown_cost():
    ids = [str(i) for i in range(100)]
    trials = [{"id": "good"}, {"id": "missing"}]
    rows = {
        "good": [
            {
                "question_id": q,
                "metrics": {
                    "answer_correctness": 0.7,
                    "query_latency_seconds": 1,
                    "total_estimated_cost": 0,
                },
            }
            for q in ids
        ],
        "missing": [
            {"question_id": q, "metrics": {"answer_correctness": 1}} for q in ids[:94]
        ],
    }
    ranked = compare_trials(trials, rows, ids, "answer", {})
    assert len(ranked) == 1 and ranked[0]["trial_id"] == "good"
    assert ranked[0]["cost_per_query"] is None
    assert not compare_trials(trials[:1], rows, ids, "answer", {"max_query_cost": 0.1})[
        0
    ]["feasible"]


def test_trial_tool_deduplicates_dispatch_and_protects_holdout(session):
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    calls = []
    request = TrialRequest(
        session_id=session, config=tools.baseline(), phase="retrieval"
    )
    one = tools.run_trial(request, calls.append)
    two = tools.run_trial(request, calls.append)
    assert one.artifact_id == two.artifact_id and len(calls) == 1
    with pytest.raises(ValueError, match="confirmation"):
        tools.run_trial(
            TrialRequest(
                session_id=session,
                config=tools.baseline(),
                phase="answer",
                split="holdout",
            ),
            calls.append,
        )
    with pytest.raises(ValueError, match="Select a candidate"):
        tools.run_trial(
            TrialRequest(
                session_id=session,
                config=tools.baseline(),
                phase="confirm",
                split="holdout",
            ),
            calls.append,
        )


def test_graph_discovers_better_pipeline_and_exports_runnable_yaml(session):
    import yaml
    from langgraph.checkpoint.memory import InMemorySaver

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    planner_inputs = []

    def call(role, instruction, data):
        planner_inputs.append(data)
        return {"pipeline": tools.baseline()["pipeline"]}

    tools.call = call

    def dispatch(run_id):
        with db.SessionLocal() as conn:
            trial = store.payload(conn.get(store.OptimizationTrial, run_id))
        config = trial["config"]
        case = expand_cases(config)[0]
        score = 0.95 if case.query_config["retriever"]["type"] == "bm25" else 0.3
        from ragbench_core.engine.data_loader import load_ground_truth

        for q in load_ground_truth(config["_snapshot"]["ground_truth_file"]):
            db.insert_result(
                run_id + q.question_id,
                run_id,
                case.case_id,
                q.question,
                {
                    "answer_correctness": score,
                    "reranked_ndcg@10": score,
                    "reranked_recall@10": score,
                    "query_latency_seconds": 0.1,
                },
                "answer",
                {},
                question_id=q.question_id,
            )
        db.upsert_run(
            run_id,
            "fixture",
            "completed",
            config,
            dataset_hash=config["_snapshot"]["dataset_hash"],
        )
        with db.SessionLocal.begin() as conn:
            conn.get(store.OptimizationTrial, run_id).status = "completed"

    graph = build_graph(InMemorySaver(), dispatch, lambda _: tools)
    graph.invoke(
        {"session_id": session, "stage": "retrieval"},
        {"configurable": {"thread_id": session}, "recursion_limit": 100},
    )
    result = store.get(session)
    assert (
        result["status"] == "completed"
        and result["recommendation"]["confirmed_improvement"]
    )
    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    verify_snapshot(exported)
    assert expand_cases(exported)[0].query_config["retriever"]["type"] == "bm25"
    assert len(json.loads((tools.root / "qa.json").read_text())) == 30
    assert all("holdout" not in json.dumps(value) for value in planner_inputs)


def test_graph_wait_resume_does_not_dispatch_trial_twice(session):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    calls = []
    graph = build_graph(InMemorySaver(), calls.append, lambda _: tools)
    config = {"configurable": {"thread_id": session}, "recursion_limit": 100}
    graph.invoke({"session_id": session, "stage": "retrieval"}, config)
    assert len(calls) == 1 and graph.get_state(config).interrupts
    graph.invoke(Command(resume=True), config)
    assert len(calls) == 1


def test_api_draft_edit_cannot_change_frozen_dataset(session):
    from fastapi import HTTPException
    from ragbench_core.optimization.contracts import DatasetEdit

    from services.api.app import optimizations as api

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    store.update(session, status="paused")
    with pytest.raises(HTTPException) as exc:
        api.edit_dataset(session, DatasetEdit(questions=[]))
    assert exc.value.status_code == 409


def test_ingestion_reuse_key_includes_corpus_ingestion_and_versions(session, template):
    from ragbench_core.optimization.runtime import IngestionReuse

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    case = expand_cases(template)[0]
    # A minimal registry keeps this test independent of optional model packages.
    registry = SimpleNamespace(_REGISTER_METHOD_GROUPS={}, embedders={"fake": object()})
    cache = IngestionReuse(tools.root / "cache", session, registry)
    cfg = {**template, "_snapshot": tools.read("corpus.json")}
    key = cache.key(case, cfg)
    changed = deepcopy(case)
    changed.query_config["retriever"] = {"type": "bm25"}
    assert cache.key(changed, cfg) == key
    changed.ingestion_config["chunker"]["chunk_size"] = 1024
    assert cache.key(changed, cfg) != key


def test_generated_questions_are_verified_and_frozen_with_reserved_confirmation(
    session,
):
    tools = OptimizationTools(session, fake_registry())
    raw = questions()
    docs = [
        {"doc_id": q["evidence"][0]["doc_id"], "text": q["evidence"][0]["quote"]}
        for q in raw
    ]
    tools.write("documents.json", docs)
    cursor = 0

    def call(role, instruction, data):
        nonlocal cursor
        if role == "verifier":
            return {"answerable": True, "supported": True, "reason": "Supported"}
        batch = raw[cursor : cursor + 5]
        cursor += 5
        return {"questions": batch}

    tools.call = call
    result = tools.generate_evaluation_dataset()
    assert result.summary["count"] == 30
    assert all(
        q["automated_validated"] and not q["human_reviewed"]
        for q in store.dataset(session)["payload"]
    )
    tools.validate_evaluation_dataset()
    assert store.dataset(session)["frozen"]
    assert sum(q["split"] == "holdout" for q in store.dataset(session)["payload"]) == 6
    assert any(
        r["role"] == "reserved_confirmation"
        for r in store.rows(store.BudgetUsage, session)
    )
    before = budget.totals(session)["tokens"]
    tools.validate_evaluation_dataset()
    assert budget.totals(session)["tokens"] == before


def test_insufficient_independent_evidence_requires_input(session):
    tools = OptimizationTools(session, fake_registry())
    raw = questions()
    for q in raw:
        q["evidence"] = [{"doc_id": "one.txt", "quote": "Only one supporting passage"}]
    tools.write(
        "documents.json", [{"doc_id": "one.txt", "text": "Only one supporting passage"}]
    )
    store.save_dataset(session, raw)
    tools.verify_question = lambda q: {"answerable": True, "supported": True}
    result = tools.validate_evaluation_dataset()
    assert result.summary["needs_input"]
    assert not store.dataset(session)["frozen"]


def test_api_queue_failure_records_failed_session(
    database, template, tmp_path, monkeypatch
):
    import yaml
    from fastapi import HTTPException

    from services.api.app import optimizations as api

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "one.txt").write_text("A source document")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "test.yaml").write_text(yaml.safe_dump(template))
    monkeypatch.setattr(api, "CONFIG_ROOT", configs)
    monkeypatch.setattr(api, "DATA_ROOT", raw)
    monkeypatch.setattr(api, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(
        api.advance_optimization,
        "apply_async",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
    )
    with pytest.raises(HTTPException) as exc:
        api.create_optimization(OptimizationRequest(config_path="test.yaml"))
    assert exc.value.status_code == 503
    with db.SessionLocal() as conn:
        row = conn.execute(store.select(store.OptimizationSession)).scalar_one()
        assert row.status == "failed" and row.error == "Queue submission failed"


def test_trial_queue_failure_is_persisted(session):
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)

    def unavailable(run_id):
        raise RuntimeError("queue unavailable")

    with pytest.raises(ValueError, match="could not be queued"):
        tools.run_trial(
            TrialRequest(
                session_id=session, config=tools.baseline(), phase="retrieval"
            ),
            unavailable,
        )
    trial = store.rows(store.OptimizationTrial, session)[0]
    assert trial["status"] == "failed" and db.get_run(trial["id"])["status"] == "failed"


def test_ingestion_cache_restores_vectors_and_reports_reuse(session, template):
    from ragbench_core.models.schema import Chunk
    from ragbench_core.optimization.runtime import IngestionReuse

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    registry = SimpleNamespace(_REGISTER_METHOD_GROUPS={}, embedders={"fake": object()})
    cache = IngestionReuse(tools.root / "cache", session, registry)
    cfg = {**template, "_snapshot": tools.read("corpus.json")}
    case = expand_cases(template)[0]
    ingestion = {
        "chunks": [
            Chunk("c", "doc", "content", embedding=[1.0], sparse_embedding={7: 0.5})
        ],
        "index_name": cache.index_name(case, cfg),
        "timings": {"ingestion_latency_seconds": 12},
        "embedding_tokens": 50,
        "token_usage": {"embedding_tokens": 50},
        "client_timings": {},
    }
    cache.save(case, cfg, ingestion)
    restored = cache.load(case, cfg)
    assert restored["chunks"][0].sparse_embedding == {7: 0.5}
    assert restored["embedding_tokens"] == 0
    assert restored["build_timings"]["ingestion_latency_seconds"] == 12
    assert "ingestion_latency_seconds" not in restored["timings"]
    assert restored["timings"]["ingestion_reuse_seconds"] >= 0


def test_api_pause_resume_cancel_and_export(session, monkeypatch):
    from services.api.app import optimizations as api

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    calls = []
    monkeypatch.setattr(
        api.advance_optimization, "apply_async", lambda **kwargs: calls.append(kwargs)
    )
    assert api.pause(session)["status"] == "paused"
    assert api.resume(session)["status"] == "queued"
    assert calls
    assert api.cancel(session)["status"] == "cancelled"
    report = {"config": tools.baseline(), "confirmed_improvement": False}
    tools.export_recommendation(report)
    store.update(session, recommendation={"confirmed_improvement": False})
    result = api.export(session, "yaml")
    assert result.media_type == "application/yaml"
    assert b"_snapshot" in result.body


def test_worker_executes_trial_once_and_reuses_initial_reservation(
    session, monkeypatch
):
    import ragbench_core.engine.runner as runner_module
    import ragbench_core.optimization.tools as tools_module

    from services.worker.app import optimization_tasks as tasks

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    run_id = tools.run_trial(
        TrialRequest(session_id=session, config=tools.baseline(), phase="retrieval"),
        lambda run_id: None,
    ).artifact_id
    initial_tokens = budget.totals(session)["tokens"]
    executions, notifications = [], []

    class Runner:
        def __init__(self, control):
            self.control = control
            self.registry = SimpleNamespace(
                _REGISTER_METHOD_GROUPS={}, embedders={"fake": object()}
            )

        def run(self, trial_id, config):
            self.control("question")
            executions.append(trial_id)
            db.upsert_run(trial_id, "fixture", "completed", config)
            return {"questions": 1}

    monkeypatch.setattr(runner_module, "BenchmarkRunner", Runner)
    monkeypatch.setattr(tools_module, "OptimizationTools", lambda session_id: tools)
    monkeypatch.setattr(
        tasks.advance_optimization,
        "apply_async",
        lambda **kwargs: notifications.append(kwargs),
    )
    tasks.run_optimization_trial.run(run_id)
    tasks.run_optimization_trial.run(run_id)
    assert executions == [run_id]
    assert len(notifications) == 1
    assert store.rows(store.OptimizationTrial, session)[0]["status"] == "completed"
    assert budget.totals(session)["tokens"] == initial_tokens


def test_preflight_lists_all_missing_ollama_models(session, monkeypatch):
    import requests

    calls = []

    def tags(url, timeout):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"models": []})

    monkeypatch.setattr(requests, "get", tags)
    with pytest.raises(OptimizationPreflightError) as exc:
        OptimizationTools(session).preflight_models()
    assert "qwen2.5:32b (planner, synthesizer, verifier, judge)" in str(exc.value)
    assert "fixed-answer (answer)" in str(exc.value)
    assert calls == ["http://100.110.250.66:11434/api/tags"]


def test_worker_exposes_safe_preflight_error(session, monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    from ragbench_core.optimization import workflow
    from services.worker.app import optimization_tasks as tasks

    class Saver:
        def setup(self):
            pass

    class Graph:
        def get_state(self, config):
            return SimpleNamespace(values={}, interrupts=[])

        def invoke(self, data, config, durability):
            raise OptimizationPreflightError("Ollama model is not installed: example")

    monkeypatch.setattr(db, "init_db", lambda: None)
    monkeypatch.setattr(PostgresSaver, "from_conn_string", lambda uri: nullcontext(Saver()))
    monkeypatch.setattr(workflow, "build_graph", lambda saver, dispatch: Graph())
    tasks.advance_optimization.run(session)
    assert store.get(session)["error"] == "Ollama model is not installed: example"
    assert store.get(session)["status"] == "failed"


def test_review_resume_marks_human_review_only_after_dataset_is_visible(
    session, monkeypatch
):
    from services.api.app import optimizations as api

    monkeypatch.setattr(api.advance_optimization, "apply_async", lambda **kwargs: None)
    store.update(session, status="paused", stage="generate")
    api.resume(session)
    assert not store.get(session)["progress"].get("review_approved")
    raw = questions()
    store.save_dataset(session, raw)
    store.update(session, status="paused", stage="freeze")
    api.resume(session)
    assert store.get(session)["progress"]["review_approved"]
    assert all(q["human_reviewed"] for q in store.dataset(session)["payload"])
