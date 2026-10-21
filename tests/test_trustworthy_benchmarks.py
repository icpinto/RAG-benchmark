import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ragbench_core.components.judges import judge_answer
from ragbench_core.components.metrics import filter_metrics
from ragbench_core.engine.comparison import paired_comparison
from ragbench_core.engine.snapshots import snapshot_inputs, verify_snapshot
from ragbench_core.models.schema import Chunk, GeneratedAnswer, QueryExample


def config_for(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "policy.md").write_text("Refunds within 14 days.", encoding="utf-8")
    qa = root / "qa.json"
    qa.write_text(
        json.dumps(
            [
                {
                    "question": "When?",
                    "expected_answer": "14 days",
                    "ground_truth_doc_ids": ["policy.md"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return {
        "data": {"input_dir": str(raw), "ground_truth_file": str(qa)},
        "runtime": {"artifact_dir": str(root / "artifacts")},
        "metrics": {"retrieval_k": [1]},
        "llm": {"provider": "ollama"},
        "pipeline": {
            "ingestion": {
                "extractors": ["markdown"],
                "chunkers": [{"type": "fixed"}],
                "embeddings": [{"provider": "fake"}],
            },
            "query": {},
        },
    }


def test_snapshot_replays_after_original_inputs_change_and_has_location_independent_identity(
    tmp_path,
):
    config = config_for(tmp_path / "one")
    frozen = snapshot_inputs("run-one", config)
    Path(config["data"]["input_dir"], "policy.md").write_text("Changed")
    Path(config["data"]["ground_truth_file"]).write_text("[]")
    verify_snapshot(frozen)
    assert "14 days" in Path(frozen["_snapshot"]["sources"][0]["path"]).read_text()
    assert (
        len(json.loads(Path(frozen["_snapshot"]["ground_truth_file"]).read_text())) == 1
    )
    relocated = snapshot_inputs("run-two", config_for(tmp_path / "two"))
    assert frozen["_snapshot"]["dataset_hash"] == relocated["_snapshot"]["dataset_hash"]
    Path(frozen["_snapshot"]["sources"][0]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_snapshot(frozen)


def test_remote_content_is_captured_once_and_extracted_offline(tmp_path, monkeypatch):
    import ragbench_core.engine.snapshots as module
    from ragbench_core.components.extractors import HtmlUrlExtractor

    config = config_for(tmp_path)
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.test/policy")
    config["data"]["remote_urls_file"] = str(urls)
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            content=b"<html><title>Policy</title><p>Frozen remote policy</p></html>",
            encoding="utf-8",
            raise_for_status=lambda: None,
        ),
    )
    frozen = snapshot_inputs("remote", config)
    entry = next(e for e in frozen["_snapshot"]["sources"] if e["remote"])
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *a, **k: pytest.fail("Replay must not fetch the URL"),
    )
    docs = HtmlUrlExtractor().extract(
        entry["source"], {"_snapshot_html": entry["path"]}
    )
    assert docs[0].doc_id == entry["doc_id"]
    assert "Frozen remote policy" in docs[0].text
    verify_snapshot(frozen)


def test_duplicate_document_ids_are_rejected(tmp_path):
    config = config_for(tmp_path)
    duplicate = Path(config["data"]["input_dir"]) / "nested" / "policy.md"
    duplicate.parent.mkdir()
    duplicate.write_text("other")
    with pytest.raises(ValueError, match="Duplicate document ID"):
        snapshot_inputs("collision", config)


@pytest.mark.parametrize(
    "selection",
    [{"source_paths": []}, {"question_indices": []}, {"question_indices": [10]}],
)
def test_empty_or_invalid_selections_are_rejected(tmp_path, selection):
    config = config_for(tmp_path)
    config["input_selection"] = selection
    with pytest.raises(ValueError):
        snapshot_inputs("empty", config)


def test_empty_cases_rejected_and_extractor_config_is_forwarded(tmp_path):
    from ragbench_core.engine.config_expander import expand_cases

    config = config_for(tmp_path)
    config["pipeline"]["ingestion"]["extractor_configs"] = {"pdf": {"ocr": True}}
    assert expand_cases(config)[0].ingestion_config["extractor_configs"] == {
        "pdf": {"ocr": True}
    }
    config["pipeline"]["ingestion"]["chunkers"] = []
    with pytest.raises(ValueError, match="case"):
        snapshot_inputs("empty", config)


def test_stable_question_ids_and_metric_selection():
    assert QueryExample("Q").question_id == QueryExample("Q").question_id
    assert (
        QueryExample("Q").question_id
        != QueryExample("Q", expected_answer="A").question_id
    )
    values = {
        "recall@1": 1,
        "reranked_recall@1": 1,
        "precision@1": 1,
        "generation_seconds": 2,
        "answer_correctness_placeholder": 1,
        "retrieval_ground_truth_available": True,
    }
    assert set(filter_metrics(values, {"enabled": ["recall_at_k"]})) == {
        "recall@1",
        "reranked_recall@1",
        "retrieval_ground_truth_available",
    }
    assert filter_metrics(values, {"enabled": []}) == {}


def test_comparison_pairs_different_cases_and_reports_missing_scores():
    baseline = {"id": "old-name", "dataset_hash": "same"}
    candidate = {"id": "new-name", "dataset_hash": "same"}
    left = [
        {"case_id": "a", "question_id": str(i), "metrics": {"score": v}}
        for i, v in enumerate([0, 0.5, None])
    ]
    right = [
        {"case_id": "b", "question_id": str(i), "metrics": {"score": v}}
        for i, v in enumerate([1, 1, 1])
    ]
    result = paired_comparison(baseline, candidate, left, right, "a", "b")
    assert result == paired_comparison(baseline, candidate, left, right, "a", "b")
    metric = result["metrics"][0]
    assert metric["paired_count"] == 2
    assert metric["missing_count"] == 1
    assert metric["mean_delta"] == 0.75
    assert metric["ci95"] == [0.5, 1]
    candidate["dataset_hash"] = "other"
    incompatible = paired_comparison(baseline, candidate, left, right, "a", "b")
    assert not incompatible["compatible"] and not incompatible["deltas"]


def test_comparison_refuses_ambiguous_or_historical_question_ids():
    run = {"id": "one", "dataset_hash": "same"}
    result = paired_comparison(
        run, run, [{"case_id": "a", "question": "Q", "metrics": {}}], [], "a", "b"
    )
    assert not result["compatible"]
    assert "stable question IDs" in result["reason"]
    assert result["deltas"] == []


def judge_fixture(monkeypatch, bad_evidence=False):
    import ragbench_core.components.judges as module

    evidence = [
        {"doc_id": "policy.md", "quote": "invented" if bad_evidence else "14 days"}
    ]
    verdicts = {
        "answer_correctness": {"score": 1, "reason": "Matches reference"},
        "claims": [
            {"claim": "Refund within 14 days", "supported": True, "evidence": evidence}
        ],
        "citations": [
            {"citation": "policy.md", "supported": True, "evidence": evidence}
        ],
    }
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": json.dumps(verdicts)},
        )

    monkeypatch.setattr(module.requests, "post", post)
    answer = GeneratedAnswer(
        "14 days [policy.md]",
        "prompt",
        [Chunk("c", "policy.md", "Refund within 14 days")],
        {"context_text": "Refund within 14 days"},
    )
    return answer, calls


def test_local_judge_saves_evidence_and_uses_separate_model(monkeypatch):
    answer, calls = judge_fixture(monkeypatch)
    result = judge_answer(
        QueryExample("Q", "14 days"), answer, {"model": "judge-model"}
    )
    assert result["scores"] == {
        "answer_correctness": 1,
        "claim_support": 1,
        "citation_support": 1,
    }
    assert result["details"]["status"] == "ok"
    assert result["details"]["rubric_version"]
    assert calls[0]["model"] == "judge-model"


def test_judge_bad_evidence_is_unavailable_not_zero(monkeypatch):
    answer, _ = judge_fixture(monkeypatch, bad_evidence=True)
    result = judge_answer(
        QueryExample("Q", "14 days"), answer, {"model": "judge-model"}
    )
    assert all(value is None for value in result["scores"].values())
    assert result["details"]["status"] == "error"


def test_judge_missing_reference_is_unavailable(monkeypatch):
    answer, _ = judge_fixture(monkeypatch)
    assert (
        judge_answer(QueryExample("Q"), answer, {"model": "judge-model"})["scores"][
            "answer_correctness"
        ]
        is None
    )


@pytest.fixture
def database(tmp_path, monkeypatch):
    import ragbench_core.engine.database as db
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    db.init_db()
    yield db
    engine.dispose()


def launch_setup(tmp_path, monkeypatch, database):
    from services.api.app import main

    config = config_for(tmp_path / "inputs")
    path = tmp_path / "benchmark.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(main, "CONFIG_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        main, "RAW_DATA_DIR", Path(config["data"]["input_dir"]).resolve()
    )
    monkeypatch.setattr(main, "upsert_run", database.upsert_run)
    return main, config, path


def test_queued_worker_uses_persisted_config_after_yaml_and_data_edits(
    tmp_path, monkeypatch, database
):
    from services.worker.app import tasks

    main, config, path = launch_setup(tmp_path, monkeypatch, database)
    queued = []
    monkeypatch.setattr(
        main,
        "run_benchmark_task",
        SimpleNamespace(delay=lambda run_id: queued.append(run_id)),
    )
    response = main.run_benchmark(main.RunRequest(config_path=str(path)))
    path.write_text("invalid: changed after submission")
    Path(config["data"]["ground_truth_file"]).write_text("[]")
    Path(config["data"]["input_dir"], "policy.md").write_text("modified")
    captured = []
    monkeypatch.setattr(tasks, "get_run_config", database.get_run_config)
    monkeypatch.setattr(
        tasks,
        "BenchmarkRunner",
        lambda: SimpleNamespace(run=lambda run_id, frozen: captured.append(frozen)),
    )
    tasks.run_benchmark_task.run(response["run_id"])
    assert queued == [response["run_id"]]
    frozen = captured[0]
    verify_snapshot(frozen)
    assert frozen["runtime"] == config["runtime"]
    assert (
        Path(frozen["_snapshot"]["sources"][0]["path"]).read_text()
        == "Refunds within 14 days."
    )


def test_queue_failure_marks_persisted_run_failed(tmp_path, monkeypatch, database):
    from fastapi import HTTPException

    main, _, path = launch_setup(tmp_path, monkeypatch, database)
    ids = []

    def fail(run_id):
        ids.append(run_id)
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(main, "run_benchmark_task", SimpleNamespace(delay=fail))
    with pytest.raises(HTTPException) as exc:
        main.run_benchmark(main.RunRequest(config_path=str(path)))
    assert exc.value.status_code == 503
    row = database.get_run(ids[0])
    assert row["status"] == "failed"
    assert "broker unavailable" in row["error"]


def fake_runner(monkeypatch, fail_questions=()):
    from ragbench_core.engine.runner import BenchmarkRunner
    from ragbench_core.models.schema import RetrievalResult
    from ragbench_core.registry import register_builtin_components

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.registry = deepcopy(register_builtin_components())
    indexes = {}

    def embed_chunks(chunks, config):
        for chunk in chunks:
            chunk.embedding = [1.0, 0.0]
        return chunks

    runner.registry.embedders["fake"] = SimpleNamespace(
        embed_chunks=embed_chunks, embed_queries=lambda qs, c: [[1.0, 0.0] for _ in qs]
    )

    def index(chunks, name, config):
        indexes[name] = chunks

    runner.registry.indexers["qdrant"] = SimpleNamespace(index=index)
    runner.registry.indexers["opensearch"] = SimpleNamespace(index=index)

    def retrieve(questions, embeddings, name, config, top_k):
        if any(q in fail_questions for q in questions):
            raise RuntimeError("query failed")
        return [[RetrievalResult(indexes[name][0], 1, 1, "dense")] for _ in questions]

    runner.registry.retrievers["dense"] = SimpleNamespace(retrieve_batch=retrieve)
    runner.registry.generators["ollama"] = SimpleNamespace(
        generate=lambda *a: pytest.fail("Answer generation must be skipped")
    )
    return runner, indexes


@pytest.mark.parametrize(
    "questions,failed,status",
    [
        (["good"], (), "completed"),
        (["bad"], ("bad",), "failed"),
        (["good", "bad"], ("bad",), "completed_with_errors"),
    ],
)
def test_runner_retrieval_only_outcomes_preserve_results(
    tmp_path, monkeypatch, database, questions, failed, status
):
    import ragbench_core.engine.runner as module

    config = config_for(tmp_path / "inputs")
    config["runtime"]["mode"] = "retrieval_only"
    config["metrics"]["enabled"] = ["recall_at_k", "answer_correctness_placeholder"]
    config["metrics"]["judge"] = {"enabled": True, "model": "judge"}
    Path(config["data"]["ground_truth_file"]).write_text(
        json.dumps(
            [{"question": q, "ground_truth_doc_ids": ["policy.md"]} for q in questions]
        )
    )
    runner, _indexes = fake_runner(monkeypatch, failed)
    monkeypatch.setattr(module, "upsert_run", database.upsert_run)
    monkeypatch.setattr(module, "insert_result", database.insert_result)
    monkeypatch.setattr(
        module, "judge_answer", lambda *a: pytest.fail("Judge must be skipped")
    )
    summary = runner.run("run-1", config)
    assert database.get_run("run-1")["status"] == status
    rows = database.list_results("run-1")
    assert len(rows) == len(questions)
    assert all("answer_correctness_placeholder" not in row["metrics"] for row in rows)
    assert summary["aggregate"]["result_counts"]["failed"] == len(failed)
    assert summary["questions"] == len(questions) - len(failed)
    assert all(row["question_id"] for row in rows)


def test_identical_runs_have_disjoint_indexes_and_cache_keys(
    tmp_path, monkeypatch, database
):
    from concurrent.futures import ThreadPoolExecutor

    import ragbench_core.engine.runner as module
    from ragbench_core.engine.cache import retrieval_cache_key

    config = config_for(tmp_path / "inputs")
    config["runtime"]["mode"] = "retrieval_only"
    monkeypatch.setattr(module, "upsert_run", database.upsert_run)
    monkeypatch.setattr(module, "insert_result", database.insert_result)
    runner, indexes = fake_runner(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda run_id: runner.run(run_id, deepcopy(config)),
                ["concurrent-one", "concurrent-two"],
            )
        )
    assert len(indexes) == 2
    assert all(o["questions"] == 1 for o in outcomes)
    names = list(indexes)
    assert retrieval_cache_key(
        names[0], "dense", {}, "Q", [1, 0], 1
    ) != retrieval_cache_key(names[1], "dense", {}, "Q", [1, 0], 1)


def test_existing_sqlite_schema_gains_question_id_without_ignored_errors(
    tmp_path, monkeypatch
):
    import ragbench_core.engine.database as db
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.sqlite'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE benchmark_results (id VARCHAR PRIMARY KEY, run_id VARCHAR, case_id VARCHAR, question TEXT, metrics JSON, answer TEXT, case_config JSON, created_at DATETIME)"
            )
        )
    monkeypatch.setattr(db, "engine", engine)
    db.init_db()
    assert "question_id" in {
        c["name"] for c in inspect(engine).get_columns("benchmark_results")
    }
    db.init_db()
    engine.dispose()


def test_end_to_end_judge_failure_preserves_answer_and_null_scores(
    tmp_path, monkeypatch, database
):
    import ragbench_core.engine.runner as module
    from ragbench_core.components import judges

    config = config_for(tmp_path / "inputs")
    config["metrics"].update(
        enabled=["answer_correctness", "claim_support", "citation_support"],
        judge={"enabled": True, "model": "judge-only"},
    )
    runner, _ = fake_runner(monkeypatch)
    runner.registry.generators["ollama"] = SimpleNamespace(
        generate=lambda *a: "14 days [policy.md]"
    )
    monkeypatch.setattr(module, "upsert_run", database.upsert_run)
    monkeypatch.setattr(module, "insert_result", database.insert_result)

    def fail(*a, **k):
        raise TimeoutError("judge unavailable")

    monkeypatch.setattr(judges.requests, "post", fail)
    runner.run("judge-failure", config)
    row = database.list_results("judge-failure")[0]
    assert database.get_run("judge-failure")["status"] == "completed"
    assert row["answer"] == "14 days [policy.md]"
    assert row["metrics"] == {
        "answer_correctness": None,
        "claim_support": None,
        "citation_support": None,
    }
    assert row["case_config"]["judge"]["details"]["status"] == "error"
    assert Path(row["artifact_paths"]["judge"]).exists()


def test_all_ingestion_cases_fail_is_failed_with_failed_coverage(
    tmp_path, monkeypatch, database
):
    import ragbench_core.engine.runner as module

    config = config_for(tmp_path / "inputs")
    runner, _ = fake_runner(monkeypatch)

    def fail(*a, **k):
        raise RuntimeError("index unavailable")

    runner.registry.indexers["qdrant"] = SimpleNamespace(index=fail)
    monkeypatch.setattr(module, "upsert_run", database.upsert_run)
    monkeypatch.setattr(module, "insert_result", database.insert_result)
    summary = runner.run("ingestion-failure", config)
    assert database.get_run("ingestion-failure")["status"] == "failed"
    assert summary["aggregate"]["result_counts"] == {
        "valid": 0,
        "failed": 1,
        "total": 1,
    }


def test_comparison_counts_questions_missing_from_both_result_sets():
    run = {"id": "run", "dataset_hash": "same", "question_ids": ["q1", "q2"]}
    rows = [{"case_id": "a", "question_id": "q1", "metrics": {"recall@1": 1}}]
    result = paired_comparison(run, run, rows, rows, "a", "a")
    assert result["metrics"][0]["missing_count"] == 1


def test_comparison_api_requires_case_selection_and_keeps_incompatible_results(
    monkeypatch, database
):
    from services.api.app import main

    monkeypatch.setattr(main, "get_run", database.get_run)
    monkeypatch.setattr(main, "list_results", database.list_results)
    for run_id, case_id, score in (
        ("renamed-baseline", "a", 0.5),
        ("renamed-candidate", "b", 1),
    ):
        database.upsert_run(run_id, run_id, "completed", {}, dataset_hash="same")
        database.insert_result(
            run_id,
            run_id,
            case_id,
            "Q",
            {"recall@1": score},
            "answer",
            {},
            question_id="q1",
        )
    ids = ["renamed-baseline", "renamed-candidate"]
    assert main.compare_benchmarks(ids)["deltas"] == []
    paired = main.compare_benchmarks(ids, "a", "b")
    assert paired["comparisons"][0]["metrics"][0]["mean_delta"] == 0.5
    database.upsert_run(ids[1], ids[1], "completed", {}, dataset_hash="different")
    incompatible = main.compare_benchmarks(ids, "a", "b")
    assert not incompatible["deltas"]
    assert incompatible["results"][ids[0]]
    assert not incompatible["comparisons"][0]["compatible"]
