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
