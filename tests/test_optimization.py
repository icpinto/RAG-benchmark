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
from ragbench_core.optimization.tools import (
    OptimizationPreflightError,
    OptimizationTools,
)
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


@pytest.mark.parametrize("actual_tokens", [0, 8, 700])
def test_budget_settlement_charges_reported_tokens(session, actual_tokens):
    budget.reserve(session, "reported", "planner", 500, None)
    budget.settle("reported", actual_tokens=actual_tokens)
    budget.settle("reported", actual_tokens=actual_tokens)
    totals = budget.totals(session)
    assert totals["tokens"] == actual_tokens
    assert totals["reported_actual_tokens"] == actual_tokens
    assert totals["reservations"] == 1


def test_budget_settlement_keeps_unreported_reservations(session):
    budget.reserve(session, "unreported", "planner", 500, None)
    budget.settle("unreported", tokens=8)
    assert budget.totals(session)["tokens"] == 500
    assert budget.totals(session)["unreported_token_operations"] == 1
    budget.settle("unreported", tokens=700)
    assert budget.totals(session)["tokens"] == 700


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


def test_dspy_graph_compiles_once_and_confirms_without_training_leakage(
    session, monkeypatch
):
    import yaml
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from ragbench_core.engine.data_loader import load_ground_truth
    from ragbench_core.prompt_optimization.artifacts import CompiledPrompt
    from ragbench_core.prompt_optimization.service import get_job, run_job

    from services.worker.app import prompt_tasks

    request = store.get(session)["request"]
    request["prompt_optimization"] = {"configs": [{"preset": "mipro_zero_shot_v1"}]}
    request["limits"]["answer_trials"] = 2
    store.update(session, request=request, progress={"workflow_version": 3})
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    tools.reserve_confirmation()
    winner = tools.baseline()
    winner["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    save_stage_trial(tools, "retrieval-winner", winner, "retrieval", score=0.9)
    store.update(
        session,
        progress={
            **store.get(session)["progress"],
            "retrieval_ranking": tools.compare_trials("retrieval"),
        },
    )
    dispatches, jobs = [], []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kwargs: jobs.append(args[0]),
    )

    def dispatch(run_id):
        with db.SessionLocal() as conn:
            trial = store.payload(conn.get(store.OptimizationTrial, run_id))
        dispatches.append(trial)
        case = expand_cases(trial["config"])[0]
        compiled = case.query_config["prompt_builder"]["type"] == "compiled"
        score = 0.95 if compiled else 0.3
        for question in load_ground_truth(
            trial["config"]["_snapshot"]["ground_truth_file"]
        ):
            db.insert_result(
                run_id + question.question_id,
                run_id,
                case.case_id,
                question.question,
                {"answer_correctness": score, "query_latency_seconds": 0.1},
                "Answer",
                {},
                question_id=question.question_id,
            )
        db.upsert_run(
            run_id,
            "fixture",
            "completed",
            trial["config"],
            dataset_hash=trial["config"]["_snapshot"]["dataset_hash"],
        )
        with db.SessionLocal.begin() as conn:
            conn.get(store.OptimizationTrial, run_id).status = "completed"

    graph = build_graph(InMemorySaver(), dispatch, lambda _: tools)
    graph_config = {"configurable": {"thread_id": session}, "recursion_limit": 100}
    graph.invoke({"session_id": session, "stage": "answer"}, graph_config)
    assert graph.get_state(graph_config).interrupts and len(jobs) == 1
    job = get_job(jobs[0])
    assert job["config"]["pipeline"] == winner["pipeline"]
    saved = deepcopy(store.get(session)["progress"]["prompt_answer_plan"])
    from ragbench_core.prompt_optimization import auto

    def unexpected_replan(*args):
        pytest.fail("A saved prompt compilation plan must not be rebuilt")

    monkeypatch.setattr(auto, "plan_answers", unexpected_replan)
    graph.invoke(Command(resume=True), graph_config)
    assert jobs == [job["id"]]
    assert store.get(session)["progress"]["prompt_answer_plan"] == saved
    holdout_ids = {
        q["question_id"]
        for q in store.dataset(session)["payload"]
        if q["split"] == "holdout"
    }
    assert (
        not set(job["splits"]["training"] + job["splits"]["validation"]) & holdout_ids
    )

    def compile_recipe(recipe, prepared, splits, config, judge, account):
        return CompiledPrompt(
            instructions="Use evidence.",
            provenance={
                "training_question_ids": splits["training"],
                "validation_question_ids": splits["validation"],
            },
        ).model_dump(), {"validation_score": 0.95}

    run_job(job["id"], compiler=compile_recipe, prepare=lambda *args: [])
    graph.invoke(Command(resume=True), graph_config)
    assert store.get(session)["status"] == "completed"
    assert store.get(session)["recommendation"]["confirmed_improvement"]
    assert len(jobs) == 1
    answers = [t for t in dispatches if t["phase"] == "answer"]
    assert len(answers) == 2
    assert answers[0]["config"]["pipeline"] == winner["pipeline"]
    assert all(
        t["config"]["pipeline"]["query"]["retrievers"] == [{"type": "bm25"}]
        for t in answers
    )
    assert all(
        e["source_trial_id"] == "retrieval-winner"
        for e in store.get(session)["progress"]["answer_plan"]
    )
    assert all(
        set(t["config"]["_snapshot"]["question_ids"])
        == set(job["splits"]["validation"])
        for t in answers
    )
    assert all(
        set(t["config"]["_snapshot"]["question_ids"]) == holdout_ids
        for t in dispatches
        if t["phase"] == "confirm"
    )
    from ragbench_core.engine.snapshots import snapshot_inputs

    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    assert (
        set(snapshot_inputs("replay", exported)["_snapshot"]["question_ids"])
        == holdout_ids
    )


def test_unapproved_compiled_prompt_cannot_enter_auto_search(session):
    from ragbench_core.prompt_optimization.artifacts import (
        CompiledPrompt,
        builder_config,
    )
    store.update(session, progress={"workflow_version": 3})
    tools = OptimizationTools(session, fake_registry())
    config = tools.baseline()
    config["pipeline"]["query"]["prompt_builders"] = [builder_config(CompiledPrompt(instructions="Unapproved").model_dump())]
    with pytest.raises(ValueError, match="approved"):
        tools.validate_candidate(config)


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
        assert row.progress["workflow_version"] == 7
        assert row.progress["recommendation_policy_version"] == 2


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


def test_api_configuration_uses_saved_settings_and_redacts_credentials(
    session, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from services.api.app import optimizations as api

    saved = store.get(session)
    template = deepcopy(saved["template"])
    template["llm"].update(
        {
            "api_key": "inline-key",
            "api_key_env": "ANSWER_API_KEY",
            "base_url": "https://user:password@example.test/v1?api_key=query-key&version=2",
        }
    )
    template["pipeline"]["ingestion"]["embeddings"][0]["api_key"] = (
        "${EMBEDDING_API_KEY}"
    )
    template["headers"] = [
        {
            "Authorization": "Bearer hidden",
            "password": "hidden",
            "X-API-Key": "header-key",
        }
    ]
    template["runtime"]["database_password"] = "database-secret"
    template["_snapshot"] = {"sources": ["private"]}
    template["_optimization"] = {"budget_operation": "private"}
    store.update(session, template=template)
    monkeypatch.setattr(
        api, "load_yaml", lambda *_: pytest.fail("Must use saved template")
    )
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        response = client.get(f"/optimizations/{session}/configuration")
        assert response.status_code == 200
        result = response.json()
        assert result["request"] == saved["request"]
        assert result["template"]["llm"]["model"] == "fixed-answer"
        assert result["template"]["llm"]["api_key"] == "[REDACTED]"
        assert result["template"]["llm"]["api_key_env"] == "ANSWER_API_KEY"
        assert (
            result["template"]["pipeline"]["ingestion"]["embeddings"][0]["api_key"]
            == "${EMBEDDING_API_KEY}"
        )
        assert result["template"]["headers"] == [
            {
                "Authorization": "[REDACTED]",
                "password": "[REDACTED]",
                "X-API-Key": "[REDACTED]",
            }
        ]
        assert result["template"]["runtime"]["database_password"] == "[REDACTED]"
        assert "user" not in result["template"]["llm"]["base_url"]
        assert "password" not in result["template"]["llm"]["base_url"]
        assert "query-key" not in result["template"]["llm"]["base_url"]
        assert "version=2" in result["template"]["llm"]["base_url"]
        assert "_snapshot" not in result["template"]
        assert "_optimization" not in result["template"]
        assert "request" not in client.get(f"/optimizations/{session}").json()
        assert "template" not in client.get("/optimizations").json()[0]
        assert client.get("/optimizations/missing/configuration").status_code == 404
    assert store.get(session)["template"] == template


def test_api_trial_configuration_is_scoped_and_preserves_execution_settings(
    session, template
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from services.api.app import optimizations as api

    config = deepcopy(template)
    config["runtime"]["mode"] = "end_to_end"
    config["metrics"] = {
        "enabled": ["answer_correctness"],
        "judge": {
            "model": "saved-judge",
            "api_key_env": "JUDGE_KEY",
            "token": "hidden",
        },
    }
    config["_snapshot"] = {"sources": ["private"]}
    config["_optimization"] = {"budget_operation": "private"}
    with db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationSession(
                id="other",
                request=store.get(session)["request"],
                template=template,
                progress={},
            )
        )
        conn.add(
            store.OptimizationTrial(
                id="trial",
                session_id=session,
                operation_key="one",
                phase="answer",
                split="tuning",
                config=config,
            )
        )
        conn.add(
            store.OptimizationTrial(
                id="unavailable",
                session_id=session,
                operation_key="two",
                phase="retrieval",
                split="tuning",
                config=None,
            )
        )
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        response = client.get(f"/optimizations/{session}/trials/trial/config")
        assert response.status_code == 200
        result = response.json()["config"]
        assert result["pipeline"] == config["pipeline"]
        assert result["runtime"]["mode"] == "end_to_end"
        assert result["metrics"] == {
            "enabled": ["answer_correctness"],
            "judge": {
                "model": "saved-judge",
                "api_key_env": "JUDGE_KEY",
                "token": "[REDACTED]",
            },
        }
        assert "_snapshot" not in result and "_optimization" not in result
        assert "config" not in client.get(f"/optimizations/{session}/trials").json()[0]
        assert (
            client.get(f"/optimizations/{session}/trials/missing/config").status_code
            == 404
        )
        assert client.get("/optimizations/other/trials/trial/config").status_code == 404
        assert (
            client.get("/optimizations/missing/trials/trial/config").status_code == 404
        )
        assert client.get(
            f"/optimizations/{session}/trials/unavailable/config"
        ).json() == {"config": None}
    assert store.rows(store.OptimizationTrial, session)[0]["config"] == config


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
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"models": []}
        )

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
    monkeypatch.setattr(
        PostgresSaver, "from_conn_string", lambda uri: nullcontext(Saver())
    )
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


@pytest.fixture
def reuse_factory(database, template, tmp_path):
    """Real extraction/generation/freezing with deterministic provider responses."""
    raw = questions()
    source_dir = tmp_path / "shared-sources"
    source_dir.mkdir()
    paths = []
    for question in raw:
        evidence = question["evidence"][0]
        path = source_dir / evidence["doc_id"]
        path.write_text(evidence["quote"], encoding="utf-8")
        paths.append(str(path))

    def create(
        name, *, request_changes=None, sources=None, created_at=1, preflight=False
    ):
        request = OptimizationRequest(
            limits={"target_questions": 30, "retrieval_trials": 2, "answer_trials": 2},
            **{
                role: {"model": role}
                for role in ("planner", "synthesizer", "verifier", "judge")
            },
        ).model_dump()
        request.update(deepcopy(request_changes or {}))
        with db.SessionLocal.begin() as conn:
            conn.add(
                store.OptimizationSession(
                    id=name,
                    request=request,
                    template=deepcopy(template),
                    progress={},
                    created_at=created_at,
                )
            )
        calls = []
        cursor = 0

        def model_call(profile, instruction, data):
            nonlocal cursor
            calls.append(profile.model)
            if profile.model == "planner":
                result = {"topics": ["requirements"]}
            elif profile.model == "verifier":
                result = {"answerable": True, "supported": True, "reason": "Supported"}
            elif profile.model == "synthesizer":
                result = {"questions": deepcopy(raw[cursor : cursor + 5])}
                cursor += 5
            else:
                raise AssertionError(f"Unexpected model call: {profile.model}")
            return result, {"prompt_tokens": 1, "completion_tokens": 1}

        tools = OptimizationTools(name, fake_registry(), model_call=model_call)
        tools.write(
            "corpus.json",
            capture_corpus(
                paths if sources is None else sources, tools.root / "snapshot"
            ),
        )
        if not preflight:
            tools.preflight_models = lambda: None
        return tools, calls

    create.paths = paths
    return create


@pytest.fixture
def reusable_origin(reuse_factory):
    tools, _ = reuse_factory("origin")
    tools.profile_corpus()
    tools.generate_evaluation_dataset()
    tools.validate_evaluation_dataset()
    return tools


def test_cross_session_reuse_preserves_payload_provenance_and_skips_models(
    reusable_origin,
    reuse_factory,
):
    origin = store.dataset("origin")
    # A later trial failure does not invalidate the frozen evaluation set.
    store.update("origin", status="failed")
    tools, calls = reuse_factory("next")
    tools.profile_corpus()
    tools.generate_evaluation_dataset()
    tools.validate_evaluation_dataset()
    copied = store.dataset("next")
    assert copied["frozen"]
    assert copied["id"] != origin["id"]
    assert copied["source_dataset_id"] == origin["id"]
    assert copied["payload"] == origin["payload"]
    assert tools.read("dataset_validation.json") == reusable_origin.read(
        "dataset_validation.json"
    )
    assert calls == ["planner"]
    assert not {"synthesizer", "verifier"} & {
        usage["role"] for usage in store.rows(store.BudgetUsage, "next")
    }
    assert tools.read("dataset_manifest.json")["evaluation_dataset"] == {
        "status": "reused",
        "dataset_id": copied["id"],
        "source_dataset_id": origin["id"],
        "source_session_id": "origin",
        "question_count": 30,
    }
    assert store.get("next")["progress"]["evaluation_dataset"]["status"] == "reused"
    with pytest.raises(ValueError, match="frozen"):
        store.save_dataset("next", [])
    copied["payload"][0]["question"] = "Local mutation"
    assert store.dataset("origin")["payload"] == origin["payload"]
    tools.export_recommendation({"config": tools.baseline()})
    assert (
        tools.read("report.json")["evaluation_dataset"]["source_session_id"] == "origin"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"audience": "New audience"},
        {"task": "New task"},
        {"example_questions": ["A different example?"]},
        {
            "limits": {
                **OptimizationRequest().limits.model_dump(),
                "target_questions": 31,
            }
        },
        {"evaluation_dataset_mode": "generate_new"},
    ],
)
def test_changed_guidance_and_explicit_fresh_mode_prevent_reuse(
    reusable_origin,
    reuse_factory,
    change,
):
    tools, _ = reuse_factory("different", request_changes=change)
    tools.profile_corpus()
    assert store.dataset("different") is None
    assert not tools.reuse_evaluation_dataset()


@pytest.mark.parametrize(
    "mutation", ["bytes", "membership", "doc_id", "extracted_text", "duplicate"]
)
def test_source_changes_prevent_reuse(
    reusable_origin, reuse_factory, mutation, tmp_path
):
    paths = reuse_factory.paths.copy()
    if mutation == "bytes":
        path = Path(paths[0])
        path.write_text(path.read_text() + " additional content", encoding="utf-8")
    elif mutation == "membership":
        paths.pop()
    elif mutation == "doc_id":
        alternate = tmp_path / "renamed.txt"
        alternate.write_bytes(Path(paths[0]).read_bytes())
        paths[0] = str(alternate)
    elif mutation == "duplicate":
        duplicate = tmp_path / "z_duplicate.txt"
        duplicate.write_bytes(Path(paths[0]).read_bytes())
        paths.append(str(duplicate))
    tools, _ = reuse_factory("changed", sources=paths)
    tools.profile_corpus()
    if mutation == "extracted_text":
        # Initially compatible; a changed extractor result must independently miss.
        with db.SessionLocal.begin() as conn:
            conn.delete(conn.get(store.DatasetVersion, store.dataset("changed")["id"]))
        docs = tools.read("documents.json")
        docs[0]["text"] += " different extractor output"
        tools.write("documents.json", docs)
    assert not tools.reuse_evaluation_dataset()
    assert store.dataset("changed") is None


def test_reuse_identity_ignores_order_paths_pipeline_models_and_other_limits(
    reusable_origin,
    reuse_factory,
    tmp_path,
):
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    paths = []
    for source in reversed(reuse_factory.paths):
        target = relocated / Path(source).name
        target.write_bytes(Path(source).read_bytes())
        paths.append(str(target))
    tools, calls = reuse_factory(
        "relocated",
        sources=paths,
        request_changes={
            "synthesizer": {
                **OptimizationRequest().synthesizer.model_dump(),
                "model": "different",
            },
            "judge": {
                **OptimizationRequest().judge.model_dump(),
                "model": "another-judge",
            },
            "limits": {
                **store.get("origin")["request"]["limits"],
                "retrieval_trials": 10,
            },
        },
    )
    template = deepcopy(tools.session["template"])
    template["llm"]["model"] = "new-answer"
    template["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    template["pipeline"]["ingestion"]["chunkers"][0]["chunk_size"] = 1024
    store.update("relocated", template=template)
    tools.profile_corpus()
    assert (
        store.dataset("relocated")["source_dataset_id"] == store.dataset("origin")["id"]
    )
    assert calls == ["planner"]


def test_latest_original_selection_and_invalid_candidate_fallback(
    reusable_origin,
    reuse_factory,
):
    latest, _ = reuse_factory(
        "latest",
        created_at=2,
        request_changes={"evaluation_dataset_mode": "generate_new"},
    )
    latest.profile_corpus()
    latest.generate_evaluation_dataset()
    latest.validate_evaluation_dataset()
    first, _ = reuse_factory("first-copy", created_at=10)
    first.profile_corpus()
    assert (
        store.dataset("first-copy")["source_dataset_id"]
        == store.dataset("latest")["id"]
    )
    with db.SessionLocal.begin() as conn:
        row = conn.get(store.DatasetVersion, store.dataset("latest")["id"])
        corrupt = deepcopy(row.payload)
        corrupt[0]["split"] = "unassigned"
        row.payload = corrupt
    fallback, _ = reuse_factory("fallback", created_at=11)
    fallback.profile_corpus()
    assert (
        store.dataset("fallback")["source_dataset_id"] == store.dataset("origin")["id"]
    )
    assert any(
        e["kind"] == "dataset_reuse_rejected"
        for e in store.rows(store.OptimizationEvent, "fallback")
    )


def test_historical_indexing_and_draft_exclusion(reusable_origin, reuse_factory):
    from ragbench_core.optimization.dataset_reuse import index_historical_datasets

    validation = reusable_origin.read("dataset_validation.json")
    validation.pop("format_version", None)
    reusable_origin.write("dataset_validation.json", validation)
    with db.SessionLocal.begin() as conn:
        row = conn.get(store.DatasetVersion, store.dataset("origin")["id"])
        row.compatibility_key = None
        session_row = conn.get(store.OptimizationSession, "origin")
        legacy_request = deepcopy(session_row.request)
        legacy_request.pop("evaluation_dataset_mode")
        session_row.request = legacy_request
    draft, _ = reuse_factory("draft", created_at=20)
    draft.profile_corpus()
    # Make this newer dataset an unfrozen draft with the same key; it is ineligible.
    with db.SessionLocal.begin() as conn:
        row = conn.get(store.DatasetVersion, store.dataset("draft")["id"])
        row.frozen = 0
        row.source_dataset_id = None
    index_historical_datasets()
    assert store.dataset("origin")["compatibility_key"]
    tools, _ = reuse_factory("from-history")
    tools.profile_corpus()
    assert (
        store.dataset("from-history")["source_dataset_id"]
        == store.dataset("origin")["id"]
    )


def test_reuse_replay_repairs_artifacts_and_reserves_confirmation_once(
    reusable_origin,
    reuse_factory,
    monkeypatch,
):
    tools, calls = reuse_factory("replay")
    tools.profile_corpus()
    assert store.dataset("replay")["frozen"]
    original_write = tools.write

    def interrupted_write(name, data):
        if name == "holdout.json":
            raise OSError("simulated interruption")
        return original_write(name, data)

    monkeypatch.setattr(tools, "write", interrupted_write)
    with pytest.raises(OSError, match="interruption"):
        tools.validate_evaluation_dataset()
    monkeypatch.setattr(tools, "write", original_write)
    tools.generate_evaluation_dataset()
    tools.validate_evaluation_dataset()
    tokens = budget.totals("replay")["tokens"]
    tools.validate_evaluation_dataset()
    assert budget.totals("replay")["tokens"] == tokens
    assert len(tools.read("holdout.json")) == 6
    assert len(tools.read("tuning.json")) == 24
    assert calls == ["planner"]


def test_unused_generation_endpoints_are_not_preflighted(
    reusable_origin,
    reuse_factory,
    monkeypatch,
):
    import requests

    endpoints = []

    def tags(url, timeout):
        endpoints.append(url)
        assert "unavailable" not in url
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "models": [
                    {"name": name} for name in ("planner", "judge", "fixed-answer")
                ]
            },
        )

    monkeypatch.setattr(requests, "get", tags)
    profile = OptimizationRequest().synthesizer.model_dump()
    profile["base_url"] = "http://unavailable:11434"
    tools, calls = reuse_factory(
        "offline-generation",
        preflight=True,
        request_changes={
            "synthesizer": profile,
            "verifier": profile,
        },
    )
    tools.profile_corpus()
    assert store.dataset("offline-generation")["source_dataset_id"]
    assert len(endpoints) == 1
    assert calls == ["planner"]


def test_reused_review_is_read_only_and_provenance_is_public(
    reusable_origin,
    reuse_factory,
    monkeypatch,
):
    from langgraph.checkpoint.memory import InMemorySaver

    from services.api.app import optimizations as api

    tools, _ = reuse_factory(
        "review-copy", request_changes={"review_before_freeze": True}
    )
    tools.profile_corpus()
    graph = build_graph(InMemorySaver(), lambda _: None, lambda _: tools)
    config = {"configurable": {"thread_id": "review-copy"}}
    graph.invoke({"session_id": "review-copy", "stage": "freeze"}, config)
    assert graph.get_state(config).interrupts
    assert store.get("review-copy")["status"] == "paused"
    before = store.dataset("review-copy")["payload"]
    monkeypatch.setattr(api.advance_optimization, "apply_async", lambda **kwargs: None)
    api.resume("review-copy")
    assert store.get("review-copy")["progress"]["review_approved"]
    assert store.dataset("review-copy")["payload"] == before
    assert (
        api.get_dataset("review-copy")["evaluation_dataset"]["source_session_id"]
        == "origin"
    )
    assert (
        api.get_optimization("review-copy")["progress"]["evaluation_dataset"]["status"]
        == "reused"
    )


def test_fork_always_requests_fresh_questions(reusable_origin, monkeypatch):
    from services.api.app import optimizations as api

    submitted = []
    monkeypatch.setattr(
        api, "create_optimization", lambda req: submitted.append(req) or {"id": "fork"}
    )
    assert api.fork("origin") == {"id": "fork"}
    assert submitted[0].evaluation_dataset_mode == "generate_new"


@pytest.fixture(params=["sqlite", "postgresql"])
def migration_database(request, database, monkeypatch):
    if request.param == "sqlite":
        yield
        return
    import os
    import uuid

    from sqlalchemy import text

    uri = os.environ.get("RAGBENCH_TEST_POSTGRES_URL")
    if not uri:
        pytest.skip(
            "Set RAGBENCH_TEST_POSTGRES_URL for the isolated PostgreSQL migration test"
        )
    schema = "test_dataset_reuse_" + uuid.uuid4().hex
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, connect_args={"options": f"-csearch_path={schema}"})
    try:
        monkeypatch.setattr(db, "engine", engine)
        monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
        db.init_db()
        yield
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def test_dataset_reuse_migration_is_additive_and_repeatable(migration_database):
    from sqlalchemy import inspect, text

    # Simulate the table as deployed before this feature.
    with db.engine.begin() as conn:
        conn.execute(text("DROP TABLE optimization_datasets"))
        conn.execute(
            text(
                "CREATE TABLE optimization_datasets (id VARCHAR PRIMARY KEY, "
                "session_id VARCHAR UNIQUE, version INTEGER, frozen INTEGER, payload JSON)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO optimization_datasets (id, session_id, version, frozen, payload) "
                "VALUES ('legacy', 'old', 2, 1, '[]')"
            )
        )
    db.init_db()
    db.init_db()
    assert {"compatibility_key", "source_dataset_id"} <= {
        c["name"] for c in inspect(db.engine).get_columns("optimization_datasets")
    }
    assert "ix_optimization_datasets_compatibility_key" in {
        index["name"]
        for index in inspect(db.engine).get_indexes("optimization_datasets")
    }
    assert store.dataset("old")["version"] == 2


def test_changed_url_response_bytes_prevent_reuse(reuse_factory, monkeypatch):
    import re

    import requests

    content = b"<p>A remote policy document</p>"

    def get(url, timeout, headers):
        return SimpleNamespace(
            content=content,
            encoding="utf-8",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "get", get)

    def with_html(tools):
        tools.registry.extractors["html_url"] = SimpleNamespace(
            extract=lambda source, cfg: (
                [
                    Document(
                        "remote",
                        source,
                        re.sub(
                            "<[^>]+>",
                            "",
                            Path(cfg["_snapshot_html"]).read_text(encoding="utf-8"),
                        ),
                    )
                ]
                if "_snapshot_html" in cfg
                else []
            )
        )
        template = deepcopy(tools.session["template"])
        template["pipeline"]["ingestion"]["extractors"].append("html_url")
        store.update(tools.session_id, template=template)

    sources = reuse_factory.paths + ["https://example.test/policy"]
    original, _ = reuse_factory("url-origin", sources=sources)
    with_html(original)
    original.profile_corpus()
    original.generate_evaluation_dataset()
    original.validate_evaluation_dataset()
    same, _ = reuse_factory("same-url", sources=sources)
    with_html(same)
    same.profile_corpus()
    assert store.dataset("same-url")["source_dataset_id"]
    content = b"<article>A remote policy document</article>"
    changed, _ = reuse_factory("changed-url", sources=sources)
    with_html(changed)
    changed.profile_corpus()
    assert changed.read("documents.json") == same.read("documents.json")
    assert store.dataset("changed-url") is None


def test_reuse_preserves_original_count_and_review_flags(
    reusable_origin, reuse_factory
):
    # A valid generation can freeze fewer accepted questions than requested.
    with db.SessionLocal.begin() as conn:
        row = conn.get(store.DatasetVersion, store.dataset("origin")["id"])
        payload = deepcopy(row.payload)
        payload[0]["human_reviewed"] = True
        row.payload = payload
        row.compatibility_key = None
        session = conn.get(store.OptimizationSession, "origin")
        req = deepcopy(session.request)
        req["limits"]["target_questions"] = 100
        session.request = req
    tools, _ = reuse_factory(
        "partial-target",
        request_changes={
            "limits": {
                **store.get("origin")["request"]["limits"],
                "target_questions": 100,
            },
        },
    )
    tools.profile_corpus()
    tools.generate_evaluation_dataset()
    assert (
        store.dataset("partial-target")["payload"] == store.dataset("origin")["payload"]
    )
    assert len(store.dataset("partial-target")["payload"]) == 30
    assert store.dataset("partial-target")["payload"][0]["human_reviewed"]


def test_incomplete_history_and_unusable_candidates_generate_normally(
    reusable_origin,
    reuse_factory,
):
    # Without validation provenance, even a previously indexed record is skipped.
    (reusable_origin.root / "dataset_validation.json").unlink()
    tools, calls = reuse_factory("no-usable-history")
    tools.profile_corpus()
    assert store.dataset("no-usable-history") is None
    tools.generate_evaluation_dataset()
    tools.validate_evaluation_dataset()
    assert "synthesizer" in calls and "verifier" in calls
    assert store.dataset("no-usable-history")["source_dataset_id"] is None
    assert store.dataset("no-usable-history")["compatibility_key"]


def save_stage_trial(
    tools, trial_id, config, phase, *, score=0.8, missing=0, failed=0, pricing=True
):
    """Persist real result rows so dashboard tests exercise the SQL read path."""
    split = "holdout" if phase == "confirm" else "tuning"
    payload = [
        q for q in store.dataset(tools.session_id)["payload"] if q["split"] == split
    ]
    with db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationTrial(
                id=trial_id,
                session_id=tools.session_id,
                operation_key=trial_id,
                phase=phase,
                split=split,
                status="completed",
                config=config,
                summary={"pricing_known": pricing, "errors": failed},
            )
        )
    db.upsert_run(
        trial_id, trial_id, "completed_with_errors" if failed else "completed", config
    )
    case = expand_cases(config)[0]
    for index, question in enumerate(payload):
        metrics = {
            "raw_retrieval_ndcg@10": 0.4,
            "reranked_ndcg@10": score,
            "reranked_recall@10": 0.9,
            "query_latency_seconds": 0,
            "answer_correctness": score,
            "claim_support": 0.7,
            "citation_support": 0.6,
            "total_estimated_cost": 0,
            "estimated_generation_cost": 0,
            "chunk_count": 8,
            "context_tokens": 100,
        }
        if index < missing:
            metrics.pop("reranked_ndcg@10")
            metrics.pop("answer_correctness")
        if index < failed:
            metrics = {}
        db.insert_result(
            f"{trial_id}-{index}",
            trial_id,
            case.case_id,
            question["question"],
            metrics,
            "Answer",
            {"outcome": "failed"} if index < failed else {},
            question_id=question["question_id"],
        )


def test_stage_dashboard_keeps_independent_metrics_and_explicit_missing_values(session):
    from ragbench_core.optimization.stages import stage_dashboard

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    save_stage_trial(tools, "retrieval-zero", baseline, "retrieval", score=0)
    save_stage_trial(tools, "answer-zero", baseline, "answer", score=0, pricing=False)
    save_stage_trial(tools, "answer-errors", baseline, "answer", missing=3, failed=2)
    save_stage_trial(tools, "confirmation", baseline, "confirm", score=0.9)
    store.update(
        session,
        stage="done",
        status="completed",
        progress={"selected_config": baseline},
    )
    before = store.get(session)
    data = stage_dashboard(before)
    retrieval = data["stages"]["retrieval"]["trials"][0]
    answer, errors = data["stages"]["answer"]["trials"]
    assert retrieval["ranking"]["score"] == 0
    assert answer["ranking"]["score"] == 0
    assert answer["metrics"]["answer_correctness"]["mean"] == 0
    assert answer["p95_seconds"] == 0
    assert answer["metrics"]["total_estimated_cost"]["mean"] is None
    assert answer["metrics"]["total_estimated_cost"]["reason"] == "unknown pricing"
    assert answer["metrics"]["estimated_generation_cost"]["reason"] == "unknown pricing"
    assert errors["ranking"] is None
    assert errors["score_reason"] == "insufficient coverage"
    assert errors["metrics"]["answer_correctness"]["mean"] == pytest.approx(0.8)
    assert errors["display_status"] == "Completed with errors"
    assert errors["coverage"]["failed"] == 2
    assert data["stages"]["confirm"]["trials"][0]["metrics"]["answer_correctness"][
        "mean"
    ] == pytest.approx(0.9)
    assert data["stages"]["confirm"]["outcome"] == "candidate is baseline"
    assert store.get(session) == before


def test_stage_dashboard_reports_prompt_history_without_candidate_content(session):
    from ragbench_core.optimization.stages import stage_dashboard

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    with db.SessionLocal.begin() as conn:
        conn.add(store.PromptCompilationJob(
            id="prompt-job", status="completed", request={}, progress={"results": [{
                "index": 0, "preset": "mipro_zero_shot_v1", "status": "completed",
                "candidate_history": {"version": 1, "attempt": 2, "count": 3},
            }]},
        ))
    store.update(session, progress={
        "workflow_version": 3, "prompt_splits": {"training": [], "validation": []},
        "prompt_answer_plan": [{"compilation_job_id": "prompt-job"}],
    })
    compilation = stage_dashboard(store.get(session))["stages"]["answer"]["compilations"][0]
    assert compilation["id"] == "prompt-job"
    assert compilation["history"]["available"] is True
    assert compilation["history"]["candidate_count"] == 3
    assert compilation["history"]["configurations"][0]["attempt"] == 2
    assert "candidates" not in compilation


def test_stage_api_is_scoped_redacted_and_uses_batched_reads(session):
    from fastapi import HTTPException
    from sqlalchemy import event

    from services.api.app import optimizations as api

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    config = tools.baseline()
    config["llm"]["api_key"] = "hidden-api-key"
    for index in range(8):
        variant = deepcopy(config)
        variant["pipeline"]["query"]["rerankers"][0]["api_key"] = (
            f"private-reranker-{index}"
        )
        save_stage_trial(tools, f"trial-{index}", variant, "retrieval")
    calls = []

    def record(conn, cursor, statement, parameters, context, executemany):
        calls.append(statement)

    event.listen(db.engine, "before_cursor_execute", record)
    try:
        data = api.get_stages(session)
    finally:
        event.remove(db.engine, "before_cursor_execute", record)
    assert len(calls) == 5  # session, trials, runs, results, dataset
    assert "hidden-api-key" not in json.dumps(data)
    assert "private-reranker" not in json.dumps(data)
    assert "[REDACTED]" in json.dumps(data)
    with pytest.raises(HTTPException) as exc:
        api.get_stages("unrelated")
    assert exc.value.status_code == 404


@pytest.mark.parametrize("maximum", [1, 2, 5, 20])
def test_answer_plan_uses_only_retrieval_winner_and_respects_caps(session, maximum):
    from ragbench_core.optimization.candidates import answer_plan, retrieval_identity
    from ragbench_core.optimization.search import candidate_hash

    store.update(session, progress={"workflow_version": 2})
    registry = fake_registry()
    registry.rerankers["diversity_mmr"] = object()
    tools = OptimizationTools(session, registry)
    base = tools.baseline()
    first = deepcopy(base)
    first["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    first["pipeline"]["query"]["rerankers"] = [
        {"type": "diversity_mmr", "lambda_mult": 0.7}
    ]
    duplicate = deepcopy(first)
    duplicate["pipeline"]["query"]["rerankers"] = [{"type": "diversity_mmr"}]
    second = deepcopy(first)
    second["pipeline"]["ingestion"]["chunkers"][0]["chunk_size"] = 1024
    trials = [
        {"id": name, "phase": "retrieval", "config": cfg}
        for name, cfg in (
            ("base", base),
            ("best", first),
            ("duplicate", duplicate),
            ("second", second),
        )
    ]
    before = deepcopy(trials)
    ranking = [{"trial_id": t["id"], "feasible": True} for t in trials[1:] + trials[:1]]
    plan = answer_plan(tools, trials, ranking, maximum)
    assert [x["trial_id"] for x in plan["retrieval_shortlist"]] == ["best"]
    assert candidate_hash(plan["answer_candidates"][0]) == candidate_hash(first)
    assert plan["answer_candidates"][0] == first
    assert len(plan["answer_candidates"]) <= maximum
    assert len({candidate_hash(c) for c in plan["answer_candidates"]}) == len(
        plan["answer_candidates"]
    )
    assert plan == answer_plan(tools, trials, ranking, maximum)
    assert trials == before
    assert all(e["source_trial_id"] == "best" for e in plan["answer_plan"])
    for candidate, entry in zip(plan["answer_candidates"], plan["answer_plan"]):
        parent = next(
            t["config"] for t in trials if t["id"] == entry["source_trial_id"]
        )
        assert retrieval_identity(candidate) == retrieval_identity(parent)
        assert candidate["llm"] == first["llm"]
        if not entry["variant"].startswith("reranker:"):
            assert (
                candidate["pipeline"]["query"]["rerankers"]
                == first["pipeline"]["query"]["rerankers"]
            )
    if maximum == 5:
        assert [e["variant"] for e in plan["answer_plan"][:3]] == [
            "inherited",
            "concise_v1",
            "evidence_first_v1",
        ]
        assert any(not entry["scheduled"] for entry in plan["answer_plan"])
    if maximum == 20:
        assert any(
            entry["variant"].startswith("reranker:") and entry["scheduled"]
            for entry in plan["answer_plan"]
        )


@pytest.mark.parametrize(
    "winner_kind", ["reranker_only", "baseline", "none", "infeasible"]
)
def test_answer_plan_selects_ranked_winner_or_records_baseline_fallback(
    session, winner_kind
):
    from ragbench_core.optimization.candidates import answer_plan, retrieval_identity

    store.update(session, progress={"workflow_version": 2})
    registry = fake_registry()
    registry.rerankers["diversity_mmr"] = object()
    tools = OptimizationTools(session, registry)
    baseline = tools.baseline()
    changed = deepcopy(baseline)
    changed["pipeline"]["query"]["rerankers"] = [
        {"type": "diversity_mmr", "lambda_mult": 0.7}
    ]
    assert retrieval_identity(baseline) == retrieval_identity(changed)
    trials = [
        {"id": "baseline", "phase": "retrieval", "config": baseline},
        {"id": "changed", "phase": "retrieval", "config": changed},
        {"id": "answer", "phase": "answer", "config": changed},
    ]
    ranking = (
        []
        if winner_kind == "none"
        else [
            {"trial_id": "answer", "feasible": True},
            {"trial_id": "missing", "feasible": True},
            {"trial_id": "changed", "feasible": winner_kind == "reranker_only"},
            {"trial_id": "baseline", "feasible": winner_kind == "baseline"},
        ]
    )
    plan = answer_plan(tools, trials, ranking, 1)
    expected = changed if winner_kind == "reranker_only" else baseline
    source = "changed" if winner_kind == "reranker_only" else "baseline"
    assert plan["answer_candidates"] == [expected]
    assert all(e["source_trial_id"] == source for e in plan["answer_plan"])
    if winner_kind in {"none", "infeasible"}:
        assert plan["retrieval_shortlist"] == []
        assert "Baseline fallback" in plan["answer_plan"][0]["reason"]
    else:
        assert [e["trial_id"] for e in plan["retrieval_shortlist"]] == [source]


@pytest.mark.parametrize("maximum", [1, 2, 5, 20])
def test_dspy_answer_plan_compiles_from_retrieval_winner(session, maximum):
    from ragbench_core.optimization.candidates import retrieval_identity
    from ragbench_core.prompt_optimization.auto import plan_answers

    request = store.get(session)["request"]
    request["prompt_optimization"] = {"configs": [{"preset": "mipro_zero_shot_v1"}]}
    store.update(
        session,
        request=request,
        progress={
            "workflow_version": 3,
            "retrieval_ranking": [{"trial_id": "best", "feasible": True}],
        },
    )
    registry = fake_registry()
    registry.rerankers["diversity_mmr"] = object()
    tools = OptimizationTools(session, registry)
    winner = tools.baseline()
    winner["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    winner["pipeline"]["query"]["rerankers"] = [
        {"type": "diversity_mmr", "lambda_mult": 0.7}
    ]
    with db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationTrial(
                id="best",
                session_id=session,
                operation_key="best",
                phase="retrieval",
                split="tuning",
                status="completed",
                config=winner,
            )
        )
    plan = plan_answers(tools, maximum)
    queue = plan["prompt_answer_plan"]
    assert [e["trial_id"] for e in plan["retrieval_shortlist"]] == ["best"]
    assert queue[0]["variant"] == "inherited"
    assert queue[0]["config"] == winner
    assert queue[0]["scheduled"]
    assert queue[1]["variant"] == "dspy:mipro_zero_shot_v1"
    assert queue[1]["config"] == winner
    assert queue[1]["scheduled"] == (maximum > 1)
    assert all(e["source_trial_id"] == "best" for e in queue)
    assert plan["answer_plan"] == [
        {k: v for k, v in entry.items() if k != "config"} for entry in queue
    ]
    assert sum(e["scheduled"] for e in queue) <= maximum
    for entry in queue:
        if "config" in entry:
            assert retrieval_identity(entry["config"]) == retrieval_identity(winner)
    assert plan_answers(tools, maximum) == plan


def test_prompt_presets_are_distinct_versioned_and_phase_restricted(session):
    from ragbench_core.components.context import (
        BASELINE_PROMPT_TEMPLATE,
        DefaultPromptBuilder,
        PresetPromptBuilder,
        preset_config,
    )
    from ragbench_core.optimization.search import candidate_hash

    store.update(session, progress={"workflow_version": 2})
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    assert DefaultPromptBuilder().build(
        "Question?", "Evidence", {}
    ) == BASELINE_PROMPT_TEMPLATE.format(question="Question?", context="Evidence")
    concise = deepcopy(baseline)
    concise["pipeline"]["query"]["prompt_builders"] = [preset_config("concise_v1")]
    assert candidate_hash(tools.validate_candidate(concise)) != candidate_hash(baseline)
    assert PresetPromptBuilder().build(
        "Question?", "Evidence", preset_config("concise_v1")
    ) != DefaultPromptBuilder().build("Question?", "Evidence", {})
    invalid = deepcopy(concise)
    invalid["pipeline"]["query"]["prompt_builders"][0]["template"] = "Unapproved"
    with pytest.raises(ValueError, match="approved"):
        tools.validate_candidate(invalid)
    with pytest.raises(ValueError, match="baseline"):
        tools.run_trial(
            TrialRequest(session_id=session, config=concise, phase="retrieval"),
            lambda _: None,
        )
    store.update(session, progress={"workflow_version": 2, "selected_config": baseline})
    with pytest.raises(ValueError, match="frozen"):
        tools.run_trial(
            TrialRequest(
                session_id=session, config=concise, phase="confirm", split="holdout"
            ),
            lambda _: None,
        )
    store.update(session, progress={})
    with pytest.raises(ValueError):
        tools.validate_candidate(concise)


def test_version_two_workflow_confirms_exact_selected_prompt_and_exports_it(session):
    from langgraph.checkpoint.memory import InMemorySaver
    from ragbench_core.engine.data_loader import load_ground_truth
    from ragbench_core.optimization.search import candidate_hash
    from ragbench_core.optimization.stages import stage_dashboard

    request = store.get(session)["request"]
    request["limits"]["answer_trials"] = 5
    store.update(session, request=request, progress={"workflow_version": 2})
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    tools.call = lambda *args: pytest.fail(
        "Two seeded retrieval trials fit the cap; holdout must never reach planner"
    )
    dispatched = []

    def dispatch(run_id):
        with db.SessionLocal() as conn:
            trial = store.payload(conn.get(store.OptimizationTrial, run_id))
        cfg = trial["config"]
        case = expand_cases(cfg)[0]
        prompt = case.query_config["prompt_builder"].get("preset_id")
        score = (
            0.95
            if prompt == "concise_v1"
            else 0.6
            if case.query_config["retriever"]["type"] == "bm25"
            else 0.3
        )
        dispatched.append((trial["phase"], candidate_hash(cfg)))
        for q in load_ground_truth(cfg["_snapshot"]["ground_truth_file"]):
            db.insert_result(
                f"{run_id}-{q.question_id}",
                run_id,
                case.case_id,
                q.question,
                {
                    "reranked_ndcg@10": score,
                    "reranked_recall@10": 1,
                    "answer_correctness": score,
                    "query_latency_seconds": 0.1,
                },
                "Answer",
                {},
                question_id=q.question_id,
            )
        db.upsert_run(
            run_id,
            run_id,
            "completed",
            cfg,
            dataset_hash=cfg["_snapshot"]["dataset_hash"],
        )
        with db.SessionLocal.begin() as conn:
            row = conn.get(store.OptimizationTrial, run_id)
            row.status = "completed"
            row.summary = {"pricing_known": False}

    graph = build_graph(InMemorySaver(), dispatch, lambda _: tools)
    graph.invoke(
        {"session_id": session, "stage": "retrieval"},
        {"configurable": {"thread_id": session}},
    )
    final = store.get(session)
    assert final["status"] == "completed"
    assert final["progress"]["confirmed"]
    selected = final["progress"]["selected_config"]
    assert (
        selected["pipeline"]["query"]["prompt_builders"][0]["preset_id"] == "concise_v1"
    )
    assert dispatched[-1] == ("confirm", candidate_hash(selected))
    assert (
        tools.read("report.json")["selected_trial_id"]
        == final["progress"]["selected_trial_id"]
    )
    assert "concise_v1" in (tools.root / "recommended.yaml").read_text()
    data = stage_dashboard(final)
    assert data["stages"]["confirm"]["outcome"] == "confirmed improvement"
    selected_trial = next(
        t
        for t in data["stages"]["answer"]["trials"]
        if "selected for holdout" in t["roles"]
    )
    assert selected_trial["source_trial_id"] in {
        t["id"] for t in data["stages"]["retrieval"]["trials"]
    }
    assert any(p["trial_id"] for p in data["stages"]["answer"]["plan"])


def test_historical_lineage_is_only_inferred_when_unambiguous(session):
    from ragbench_core.optimization.stages import stage_dashboard

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    base = tools.baseline()
    save_stage_trial(tools, "origin", base, "retrieval")
    save_stage_trial(tools, "answer", base, "answer")
    data = stage_dashboard(store.get(session))
    assert data["stages"]["answer"]["trials"][0]["source_trial_id"] == "origin"
    save_stage_trial(tools, "ambiguous", base, "retrieval")
    data = stage_dashboard(store.get(session))
    assert data["stages"]["answer"]["trials"][0]["source_trial_id"] is None


@pytest.mark.parametrize("saved_queue", [False, True])
def test_version_two_answer_queue_survives_interrupt_without_redispatch(
    session, monkeypatch, saved_queue
):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    store.update(session, progress={"workflow_version": 2})
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    winner = deepcopy(baseline)
    winner["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    save_stage_trial(tools, "retrieval-winner", winner, "retrieval")
    state = {
        "workflow_version": 2,
        "retrieval_ranking": tools.compare_trials("retrieval"),
    }
    if saved_queue:
        from ragbench_core.optimization.search import candidate_hash

        state.update(
            answer_candidates=[baseline],
            answer_plan=[
                {
                    "candidate_hash": candidate_hash(baseline),
                    "source_trial_id": None,
                    "variant": "baseline",
                    "scheduled": True,
                }
            ],
            retrieval_shortlist=[],
        )
    store.update(session, progress=state)
    calls = []
    graph = build_graph(InMemorySaver(), calls.append, lambda _: tools)
    config = {"configurable": {"thread_id": session}}
    graph.invoke({"session_id": session, "stage": "answer"}, config)
    saved = deepcopy(store.get(session)["progress"])
    assert graph.get_state(config).interrupts
    assert len(calls) == 1
    assert saved["answer_plan"]
    expected = baseline if saved_queue else winner
    assert saved["answer_candidates"][0] == expected
    if saved_queue:
        assert saved == state
    from ragbench_core.optimization import workflow

    def unexpected_replan(*args):
        pytest.fail("A saved answering queue must not be rebuilt")

    monkeypatch.setattr(workflow, "answer_plan", unexpected_replan)
    graph.invoke(Command(resume=True), config)
    assert len(calls) == 1
    assert (
        store.get(session)["progress"]["answer_candidates"]
        == saved["answer_candidates"]
    )
    assert store.get(session)["progress"]["answer_plan"] == saved["answer_plan"]


@pytest.mark.parametrize("has_winner", [False, True])
def test_dashboard_explains_answer_parent_and_baseline_fallback(session, has_winner):
    from ragbench_core.optimization.candidates import answer_plan
    from ragbench_core.optimization.stages import stage_dashboard

    store.update(session, progress={"workflow_version": 2})
    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    parent = deepcopy(baseline)
    parent["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    save_stage_trial(
        tools, "origin", parent, "retrieval", missing=0 if has_winner else 4
    )
    plan = answer_plan(
        tools,
        store.rows(store.OptimizationTrial, session),
        tools.compare_trials("retrieval"),
        1,
    )
    save_stage_trial(tools, "answer", plan["answer_candidates"][0], "answer")
    store.update(
        session,
        stage="answer",
        progress={
            "workflow_version": 2,
            "baseline_config": baseline,
            **plan,
        },
    )
    data = stage_dashboard(store.get(session))
    answer = data["stages"]["answer"]["trials"][0]
    assert answer["source_trial_id"] == ("origin" if has_winner else None)
    assert plan["answer_plan"][0]["reason"] in answer["selection_reasons"]
    assert (
        data["stages"]["answer"]["plan"][0]["reason"]
        == plan["answer_plan"][0]["reason"]
    )
    assert bool(data["stages"]["retrieval"]["shortlist"]) == has_winner


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("none", "confirmation not run"),
        ("failed", "failed confirmation"),
        ("missing", "insufficient evidence"),
        ("inconclusive", "improvement not demonstrated"),
        ("confirmed", "confirmed improvement"),
    ],
)
def test_confirmation_dashboard_outcomes_and_unknown_costs(session, mode, expected):
    from ragbench_core.optimization.stages import stage_dashboard

    tools = OptimizationTools(session, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    candidate = deepcopy(baseline)
    candidate["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    progress = {
        "baseline_config": baseline,
        "selected_config": candidate,
        "confirmed": mode == "confirmed",
    }
    if mode != "none":
        save_stage_trial(tools, "control", baseline, "confirm", pricing=False)
        save_stage_trial(tools, "candidate", candidate, "confirm", pricing=False)
        if mode == "failed":
            with db.SessionLocal.begin() as conn:
                conn.get(store.OptimizationTrial, "candidate").status = "failed"
        progress["comparison"] = {
            "baseline_run_id": "control",
            "run_id": "candidate",
            "compatible": True,
            "metrics": [
                {
                    "metric": "answer_correctness",
                    "ci95": None if mode == "missing" else [-0.1, 0.2],
                    "paired_count": 6,
                    "missing_count": 0,
                },
                {
                    "metric": "total_estimated_cost",
                    "baseline_mean": 0,
                    "candidate_mean": 0,
                    "mean_delta": 0,
                    "ci95": [0, 0],
                },
            ],
        }
    store.update(session, progress=progress, status="completed", stage="done")
    data = stage_dashboard(store.get(session))["stages"]["confirm"]
    assert data["outcome"] == expected
    if mode == "none":
        assert data["status"] == "not run"
    if mode != "none":
        cost = data["comparison"]["metrics"][1]
        assert cost["baseline_mean"] is None and cost["candidate_mean"] is None
        assert cost["mean_delta"] is None and cost["reason"] == "unknown pricing"


def test_evaluation_version_and_invalid_ids_prevent_reuse(
    reusable_origin, reuse_factory, monkeypatch
):
    from ragbench_core.optimization import dataset_reuse

    monkeypatch.setattr(dataset_reuse, "FORMAT_VERSION", "incompatible-future-format")
    tools, _ = reuse_factory("new-version")
    tools.profile_corpus()
    assert store.dataset("new-version") is None
    docs = reusable_origin.read("documents.json")
    raw = deepcopy(store.dataset("origin")["payload"])
    raw[0]["question_id"] = "altered"
    with pytest.raises(ValueError, match="question_id"):
        dataset_reuse.validate_frozen_payload(raw, docs)
