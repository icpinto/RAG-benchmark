import json
import sys
import types

sys.modules.setdefault(
    "ragbench_core.engine.database",
    types.SimpleNamespace(upsert_run=lambda *args, **kwargs: None, insert_result=lambda *args, **kwargs: None),
)

import ragbench_core.engine.runner as runner_module
from ragbench_core.engine.runner import BenchmarkRunner
from ragbench_core.models.schema import BenchmarkCase, QueryExample


def test_runner_selects_configured_source_and_question_subsets(tmp_path):
    selected_source = tmp_path / "selected.md"
    skipped_source = tmp_path / "skipped.md"
    selected_source.write_text("selected", encoding="utf-8")
    skipped_source.write_text("skipped", encoding="utf-8")
    examples = [QueryExample(question="Q1?"), QueryExample(question="Q2?"), QueryExample(question="Q3?")]
    runner = BenchmarkRunner.__new__(BenchmarkRunner)

    assert runner._select_sources(
        [str(selected_source), str(skipped_source)],
        [str(selected_source)],
    ) == [str(selected_source)]
    assert [example.question for example in runner._select_examples(examples, [0, 2])] == ["Q1?", "Q3?"]


def test_runner_input_selection_falls_back_to_all_inputs(tmp_path):
    sources = [str(tmp_path / "one.md"), str(tmp_path / "two.md")]
    examples = [QueryExample(question="Q1?"), QueryExample(question="Q2?")]
    runner = BenchmarkRunner.__new__(BenchmarkRunner)

    assert runner._select_sources(sources, None) == sources
    assert runner._select_examples(examples, None) == examples


def test_runner_records_case_errors_and_continues(monkeypatch, tmp_path):
    cases = [
        BenchmarkCase("failed-case", {}, {}, {}, {}),
        BenchmarkCase("successful-case", {}, {}, {}, {}),
    ]
    upsert_calls = []

    monkeypatch.setattr(runner_module, "expand_cases", lambda config: cases)
    monkeypatch.setattr(runner_module, "load_ground_truth", lambda path: [QueryExample(question="Q?")])
    monkeypatch.setattr(runner_module, "list_sources", lambda input_dir, remote_urls_file=None: [str(tmp_path / "doc.md")])
    monkeypatch.setattr(runner_module, "dataset_hash", lambda sources, examples: "dataset-hash")
    monkeypatch.setattr(runner_module, "component_versions", lambda registry: {})
    monkeypatch.setattr(runner_module, "dependency_versions", lambda: {})
    monkeypatch.setattr(runner_module, "export_run", lambda config, run_id, run_payload, results: {})
    monkeypatch.setattr(
        runner_module,
        "upsert_run",
        lambda run_id, name, status, config, summary=None, error=None, **metadata: upsert_calls.append(
            {"status": status, "summary": summary, "error": error, "metadata": metadata}
        ),
    )

    def run_case(run_id, case, sources, examples, full_config):
        if case.case_id == "failed-case":
            raise ValueError("boom")
        return {"questions": len(examples), "metric_rows": []}

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.registry = types.SimpleNamespace()
    monkeypatch.setattr(runner, "_run_case", run_case)

    summary = runner.run(
        "run-with-case-error",
        {
            "experiment": {"name": "case errors"},
            "runtime": {"artifact_dir": str(tmp_path)},
            "data": {"ground_truth_file": "ground-truth.jsonl", "input_dir": "inputs"},
            "pipeline": {"ingestion": {"embeddings": []}, "query": {}},
            "llm": {},
        },
    )

    completed = [call for call in upsert_calls if call["status"] == "completed"][-1]
    case_error = summary["case_errors"][0]

    assert summary["cases"] == 1
    assert summary["questions"] == 1
    assert summary["errors"] == 1
    assert case_error["case_id"] == "failed-case"
    assert case_error["exception_type"] == "ValueError"
    assert case_error["message"] == "boom"
    assert "traceback" in case_error
    assert completed["summary"] == summary

    artifact_path = tmp_path / "run-with-case-error" / "cases" / "failed-case" / "case_error.json"
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["case_id"] == "failed-case"
    assert artifact_payload["exception_type"] == "ValueError"
    assert artifact_payload["message"] == "boom"
    assert case_error["artifact_path"] == str(artifact_path)


def test_run_case_delegates_to_ingest_answer_and_persist(monkeypatch, tmp_path):
    case = BenchmarkCase(
        "case-1",
        {"embedder": {"provider": "fake", "model": "fake-model"}},
        {},
        {},
        {"retrieval_k": 1},
    )
    examples = [QueryExample(question="Q1?"), QueryExample(question="Q2?")]
    calls = []
    ingestion = {"embedding_cache": object(), "embedding_tokens": 6, "chunks": [], "index_name": "idx"}

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.registry = types.SimpleNamespace()

    def ingest_case(received_case, sources, full_config):
        calls.append(("ingest", received_case.case_id, tuple(sources), full_config["runtime"]["collection_prefix"]))
        return ingestion

    def answer_question(example, received_case, chunks, index_name, caches, metric_config):
        calls.append(
            (
                "answer",
                example.question,
                received_case.case_id,
                chunks,
                index_name,
                caches["embedding"] is ingestion["embedding_cache"],
                metric_config["pricing"],
            )
        )
        return {"question": example.question}

    def persist_result(run_id, example, received_case, full_config, metric_config, received_ingestion, answer_payload):
        calls.append(
            (
                "persist",
                run_id,
                example.question,
                received_case.case_id,
                received_ingestion["per_question_embedding_tokens"],
                answer_payload["question"],
            )
        )
        return {"case_id": received_case.case_id, "question": example.question, "metrics": {}}

    monkeypatch.setattr(runner, "_ingest_case", ingest_case)
    monkeypatch.setattr(runner, "_answer_question", answer_question)
    monkeypatch.setattr(runner, "_compute_and_persist_result", persist_result)
    monkeypatch.setattr(runner_module, "build_retrieval_cache", lambda config: "retrieval-cache")

    summary = runner._run_case(
        "run-1",
        case,
        [str(tmp_path / "doc.md")],
        examples,
        {"runtime": {"collection_prefix": "test", "pricing": {"prompt": 1}}},
    )

    assert summary == {
        "questions": 2,
        "metric_rows": [
            {"case_id": "case-1", "question": "Q1?", "metrics": {}},
            {"case_id": "case-1", "question": "Q2?", "metrics": {}},
        ],
    }
    assert calls == [
        ("ingest", "case-1", (str(tmp_path / "doc.md"),), "test"),
        ("answer", "Q1?", "case-1", [], "idx", True, {"prompt": 1}),
        ("persist", "run-1", "Q1?", "case-1", 3, "Q1?"),
        ("answer", "Q2?", "case-1", [], "idx", True, {"prompt": 1}),
        ("persist", "run-1", "Q2?", "case-1", 3, "Q2?"),
    ]
