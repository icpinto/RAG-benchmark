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
