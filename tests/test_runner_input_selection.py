import sys
import types

sys.modules.setdefault(
    "ragbench_core.engine.database",
    types.SimpleNamespace(upsert_run=lambda *args, **kwargs: None, insert_result=lambda *args, **kwargs: None),
)

from ragbench_core.engine.runner import BenchmarkRunner
from ragbench_core.models.schema import QueryExample


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
