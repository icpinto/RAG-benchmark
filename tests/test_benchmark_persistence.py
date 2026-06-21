from pathlib import Path

from ragbench_core.engine.persistence import artifact_path, dataset_hash, stable_hash
from ragbench_core.models.schema import QueryExample


def test_stable_config_hash_is_order_independent_and_redacts_secrets():
    left={"llm":{"model":"m","api_key":"one"},"runtime":{"artifact_dir":"/tmp/a"}}
    right={"runtime":{"artifact_dir":"/tmp/a"},"llm":{"api_key":"two","model":"m"}}

    assert stable_hash(left) == stable_hash(right)


def test_stable_config_hash_changes_for_material_config_values():
    first={"llm":{"model":"model-a"}}
    second={"llm":{"model":"model-b"}}

    assert stable_hash(first) != stable_hash(second)


def test_artifact_path_uses_runtime_artifact_dir_and_sanitizes_parts(tmp_path):
    config={"runtime":{"artifact_dir":str(tmp_path)}}

    path=artifact_path(config, "run-1", "cases", "case/../a", "prompt.json")

    assert path == tmp_path / "run-1" / "cases" / "case___a" / "prompt.json"
    assert path.parent.exists()


def test_dataset_hash_uses_file_content_and_examples(tmp_path):
    source=tmp_path / "doc.md"
    source.write_text("alpha", encoding="utf-8")
    examples=[QueryExample(question="What?", expected_answer="Alpha")]
    first=dataset_hash([source], examples)

    source.write_text("beta", encoding="utf-8")

    assert dataset_hash([source], examples) != first
