import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException

from services.api.app import main


def test_question_helpers_create_update_delete_ground_truth(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    qa_path = eval_dir / "qa.json"
    qa_path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(main, "EVAL_DATA_DIR", eval_dir.resolve())
    monkeypatch.setattr(main, "GROUND_TRUTH_PATH", qa_path.resolve())

    created = main.create_question(
        main.QuestionRequest(
            question="What is the refund window?",
            expected_answer="14 days.",
            ground_truth_doc_ids=["policy.md"],
            expected_citations=["policy.md"],
        )
    )
    assert created["index"] == 0

    updated = main.update_question(
        0,
        main.QuestionRequest(
            question="What is the current refund window?",
            expected_answer="14 days.",
            ground_truth_doc_ids=["policy.md"],
            expected_citations=["policy.md"],
        ),
    )
    assert updated["question"]["question"] == "What is the current refund window?"
    assert main.list_questions()["questions"][0]["expected_citations"] == ["policy.md"]

    deleted = main.delete_question(0)
    assert deleted["status"] == "deleted"
    assert json.loads(qa_path.read_text(encoding="utf-8")) == []


def test_resolve_data_path_rejects_path_traversal(tmp_path):
    root = (tmp_path / "raw").resolve()
    root.mkdir()

    with pytest.raises(HTTPException) as exc:
        main._resolve_data_path("../secret.txt", root)

    assert exc.value.status_code == 400



def test_run_benchmark_validates_and_embeds_input_selection(tmp_path, monkeypatch):
    raw_dir = (tmp_path / "raw").resolve()
    eval_dir = (tmp_path / "eval").resolve()
    raw_dir.mkdir()
    eval_dir.mkdir()
    source = raw_dir / "docs" / "policy.md"
    source.parent.mkdir()
    source.write_text("policy", encoding="utf-8")
    qa_path = eval_dir / "qa.json"
    qa_path.write_text(json.dumps([
        {"question": "Q1?"},
        {"question": "Q2?"},
    ]), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: selected\ndata: {}\n", encoding="utf-8")

    queued = {}

    class DummyTask:
        @staticmethod
        def delay(run_id, config_path_arg, overrides=None):
            queued["run_id"] = run_id
            queued["config_path"] = config_path_arg
            queued["overrides"] = overrides

    monkeypatch.setattr(main, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(main, "EVAL_DATA_DIR", eval_dir)
    monkeypatch.setattr(main, "GROUND_TRUTH_PATH", qa_path)
    monkeypatch.setattr(main, "run_benchmark_task", DummyTask)
    monkeypatch.setattr(main, "upsert_run", lambda *args, **kwargs: queued.setdefault("upsert", args))

    response = main.run_benchmark(
        main.RunRequest(
            config_path=str(config_path),
            source_paths=["docs/policy.md"],
            question_indices=[1],
        )
    )

    assert response["status"] == "queued"
    assert queued["overrides"] == {
        "input_selection": {
            "source_paths": [str(source)],
            "question_indices": [1],
        }
    }
    saved_config = queued["upsert"][3]
    assert saved_config["input_selection"]["source_paths"] == [str(source)]


def test_run_benchmark_rejects_invalid_source_and_question_selection(tmp_path, monkeypatch):
    raw_dir = (tmp_path / "raw").resolve()
    eval_dir = (tmp_path / "eval").resolve()
    raw_dir.mkdir()
    eval_dir.mkdir()
    qa_path = eval_dir / "qa.json"
    qa_path.write_text(json.dumps([{"question": "Q1?"}]), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: selected\n", encoding="utf-8")

    monkeypatch.setattr(main, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(main, "EVAL_DATA_DIR", eval_dir)
    monkeypatch.setattr(main, "GROUND_TRUTH_PATH", qa_path)

    with pytest.raises(HTTPException) as source_exc:
        main.run_benchmark(main.RunRequest(config_path=str(config_path), source_paths=["missing.md"]))
    assert source_exc.value.status_code == 404

    with pytest.raises(HTTPException) as question_exc:
        main.run_benchmark(main.RunRequest(config_path=str(config_path), question_indices=[3]))
    assert question_exc.value.status_code == 400
