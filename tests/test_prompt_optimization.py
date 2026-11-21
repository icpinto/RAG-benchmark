"""Prompt compiler parity, leakage controls, durable jobs, and shared workflows."""

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml
from ragbench_core.engine import database as db
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.persistence import stable_hash
from ragbench_core.engine.snapshots import (
    snapshot_inputs,
)
from ragbench_core.models.schema import QueryExample
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.contracts import OptimizationRequest
from ragbench_core.prompt_optimization.artifacts import (
    CompiledPrompt,
    CompiledPromptBuilder,
    assert_no_training_overlap,
    builder_config,
    render_prompt,
)
from ragbench_core.prompt_optimization.contracts import (
    PRESETS,
    PromptOptimizationConfig,
    PromptOptimizationSettings,
)
from ragbench_core.prompt_optimization.datasets import split_questions
from ragbench_core.prompt_optimization.service import (
    CompilationAccount,
    artifacts_for_job,
    create_job,
    get_job,
    ingestion_root_for_job,
    load_or_build_ingestion,
    run_job,
    task_profile,
    update_job,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'prompts.sqlite'}")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine))
    db.init_db()
    yield db
    engine.dispose()


@pytest.fixture
def examples():
    return [
        QueryExample(
            question=f"Explain requirement {stable_hash(i)[:24]}?",
            expected_answer=f"Policy {i}",
            ground_truth_doc_ids=[f"{i}.txt"],
            question_id=f"q{i}",
        )
        for i in range(15)
    ]


@pytest.fixture
def config(tmp_path, examples):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(15):
        (raw / f"{i}.txt").write_text(f"Policy {i}", encoding="utf-8")
    qa = tmp_path / "eval" / "qa.json"
    qa.parent.mkdir()
    qa.write_text(json.dumps([asdict(e) for e in examples]), encoding="utf-8")
    return {
        "data": {"input_dir": str(raw), "ground_truth_file": str(qa)},
        "pipeline": {
            "ingestion": {
                "extractors": ["txt"],
                "chunkers": [{"type": "fixed"}],
                "embeddings": [{"provider": "fake", "model": "fake"}],
                "indexers": {"dense": {"type": "qdrant"}},
            },
            "query": {},
        },
        "llm": {"provider": "ollama", "model": "test", "max_output_tokens": 128},
        "metrics": {"retrieval_k": [3, 5, 10]},
        "runtime": {"artifact_dir": str(tmp_path / "artifacts"), "max_combinations": 1},
    }


def settings(preset="mipro_zero_shot_v1", **kwargs):
    return PromptOptimizationSettings(
        configs=[PromptOptimizationConfig(preset=preset, **kwargs)]
    )


@pytest.mark.parametrize("name", PRESETS)
def test_presets_resolve_and_roundtrip(name):
    recipe = PromptOptimizationConfig(preset=name)
    assert recipe.optimizer == PRESETS[name]["optimizer"]
    assert PromptOptimizationConfig.model_validate(recipe.model_dump()) == recipe


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "unknown"},
        {"optimizer": "miprov2"},
        {"preset": "mipro_zero_shot_v1", "parameters": {"auto": "light"}},
        {"preset": "gepa_light_v1", "parameters": {"max_full_evals": 2}},
        {"preset": "bootstrap_small_v1", "parameters": {"num_threads": 20}},
        {"preset": "bootstrap_small_v1", "parameters": {"metric_threshold": 0}},
        {"preset": "mipro_zero_shot_v1", "parameters": {"num_trials": True}},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        PromptOptimizationConfig(**kwargs)


def test_hosted_roles_and_prices_are_validated():
    hosted = {
        "provider": "chat_completions",
        "model": "test",
        "base_url": "https://example.test/v1",
    }
    with pytest.raises(ValueError, match="opt-in"):
        OptimizationRequest(
            prompt_optimization={
                "configs": [{"preset": "gepa_light_v1", "proposal_model": hosted}]
            }
        )
    assert OptimizationRequest(
        allow_hosted=True,
        prompt_optimization={
            "configs": [{"preset": "gepa_light_v1", "proposal_model": hosted}]
        },
    ).prompt_optimization.configs


def test_artifact_hash_rendering_and_yaml_roundtrip():
    artifact = CompiledPrompt(
        instructions="Summarize {facts}.",
        demonstrations=[
            {
                "question": "Example?",
                "context": "[doc::chunk]\nFact",
                "answer": "Fact [doc::chunk]",
            }
        ],
    ).model_dump()
    config = yaml.safe_load(yaml.safe_dump(builder_config(artifact)))
    expected = render_prompt(
        artifact["instructions"],
        artifact["demonstrations"],
        "Q {x}?",
        "[new::1]\nContext",
    )
    assert (
        CompiledPromptBuilder().build("Q {x}?", "[new::1]\nContext", config) == expected
    )
    config["artifact"]["instructions"] = "Changed"
    with pytest.raises(ValueError, match="checksum"):
        CompiledPromptBuilder().build("q", "c", config)


def test_splits_are_disjoint_deterministic_and_grouped(examples):
    raw = [asdict(e) for e in examples]
    raw[1]["ground_truth_doc_ids"] = raw[0]["ground_truth_doc_ids"]
    split = split_questions(raw)
    assert split == split_questions(list(reversed(raw)))
    partitions = [set(split[k]) for k in ("training", "validation", "evaluation")]
    assert all(partitions)
    assert set.union(*partitions) == {e.question_id for e in examples}
    assert (
        not partitions[0] & partitions[1]
        and not partitions[0] & partitions[2]
        and not partitions[1] & partitions[2]
    )
    assert any({"q0", "q1"} <= p for p in partitions)


def test_insufficient_groups_and_missing_labels_rejected(examples):
    raw = [asdict(e) for e in examples]
    for item in raw:
        item["ground_truth_doc_ids"] = ["shared"]
    with pytest.raises(ValueError, match="independent"):
        split_questions(raw)
    raw[0]["expected_answer"] = ""
    with pytest.raises(ValueError, match="reference"):
        split_questions(raw)


def test_overlap_checks_include_renamed_questions(examples):
    from ragbench_core.prompt_optimization.datasets import question_fingerprint

    artifact = CompiledPrompt(
        instructions="Answer.",
        provenance={
            "training_question_ids": ["q0"],
            "validation_question_ids": ["q1"],
            "training_question_hashes": [question_fingerprint(examples[0].question)],
        },
    ).model_dump()
    with pytest.raises(ValueError, match="overlap"):
        assert_no_training_overlap(artifact, [examples[0]])
    with pytest.raises(ValueError, match="overlap"):
        assert_no_training_overlap(
            artifact,
            [QueryExample(question=examples[0].question, question_id="renamed")],
        )
    with pytest.raises(ValueError, match="overlap"):
        assert_no_training_overlap(artifact, [examples[1]])
    assert_no_training_overlap(artifact, [examples[1]], allow_validation=True)


def make_job(config, examples, **kwargs):
    frozen = snapshot_inputs("compile-fixture", config)
    splits = split_questions([asdict(e) for e in examples])
    return create_job(frozen, settings(**kwargs), splits, task_profile(config))


def fake_prepare(job, account):
    account.check()
    return []


def test_auto_compilation_reuses_session_ingestion(config, tmp_path):
    job = {
        "id": "prompt-job",
        "session_id": "auto-session",
        "config": config,
    }
    assert ingestion_root_for_job(job) == (
        tmp_path / "artifacts" / "optimizations" / "auto-session" / "ingestion"
    )
    job["session_id"] = None
    assert ingestion_root_for_job(job) == (
        tmp_path / "artifacts" / "prompt_optimizations" / "prompt-job" / "ingestion"
    )


def test_context_preparation_retries_qdrant_timeout(monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException
    from ragbench_core.prompt_optimization import service

    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    saves = []
    reuse = SimpleNamespace(
        load=lambda *args: None,
        index_name=lambda *args: "index",
        save=lambda *args: saves.append(args[-1]),
    )
    attempts = []

    def ingest(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise ResponseHandlingException(TimeoutError("timed out"))
        return {"index_name": "index"}

    runner = SimpleNamespace(_ingest_case=ingest)
    result = load_or_build_ingestion(
        reuse, runner, object(), {}, ["source.txt"], NullAccount()
    )
    assert result == {"index_name": "index"}
    assert len(attempts) == 2
    assert saves == [result]


def fake_compile(recipe, prepared, splits, config, judge, account):
    with account.call("prompt_task", "test", task_profile(config)) as usage:
        usage.update(prompt_tokens=5, completion_tokens=3)
    return CompiledPrompt(
        instructions="Use evidence.", provenance={"split_hash": splits["hash"]}
    ).model_dump(), {"validation_score": 0.9}


def test_completed_job_reuses_artifacts_and_usage(database, config, examples):
    job_id = make_job(config, examples)
    run_job(job_id, compiler=fake_compile, prepare=fake_prepare)
    first = get_job(job_id)
    assert first["status"] == "completed"
    assert len(artifacts_for_job(job_id)) == 1
    assert budget.totals(job_id)["reported_actual_tokens"] == 8
    run_job(job_id, compiler=lambda *args: pytest.fail("completed work was repeated"))
    assert get_job(job_id)["attempts"] == first["attempts"]


def test_identical_artifacts_remain_available_to_later_jobs(database, config, examples):
    first_id = make_job(config, examples)
    first = get_job(first_id)
    second_id = create_job(
        first["config"], settings(), first["splits"], task_profile(config)
    )
    run_job(first_id, compiler=fake_compile, prepare=fake_prepare)
    run_job(second_id, compiler=fake_compile, prepare=fake_prepare)
    assert len(artifacts_for_job(second_id)) == 1
    assert artifacts_for_job(first_id)[0]["id"] == artifacts_for_job(second_id)[0]["id"]


def test_interrupted_attempt_keeps_reservations(database, config, examples):
    job_id = make_job(config, examples)
    update_job(job_id, lease_owner="old", lease_until=0, status="running", attempts=1)
    budget.reserve(job_id, f"{job_id}:config:0:uncertain", "prompt_task", 10000, None)
    run_job(job_id, compiler=fake_compile, prepare=fake_prepare)
    assert get_job(job_id)["attempts"] == 2
    assert budget.totals(job_id)["tokens"] > 10000
    assert len(artifacts_for_job(job_id)) == 1


def test_cancellation_and_token_caps_prevent_calls(database, config, examples):
    job_id = make_job(config, examples, limits={"max_tokens": 1000})
    update_job(job_id, lease_owner="owner", lease_until=0)
    account = CompilationAccount(job_id, 0, "owner")
    with (
        pytest.raises(budget.BudgetExhausted),
        account.call("prompt_task", "x" * 2000, task_profile(config)),
    ):
        pytest.fail("call exceeded cap")
    update_job(job_id, status="cancelled")
    with pytest.raises(budget.OptimizationStopped):
        account.check()
    run_job(job_id, compiler=lambda *args: pytest.fail("cancelled work ran"))


@pytest.mark.parametrize("with_parent", [False, True])
def test_compilation_releases_reported_token_reservations(
    database, config, examples, with_parent
):
    job_id = make_job(config, examples, limits={"max_tokens": 1000, "max_calls": 16})
    if with_parent:
        request = OptimizationRequest(limits={"max_tokens": 1000}).model_dump()
        with db.SessionLocal.begin() as conn:
            conn.add(store.OptimizationSession(id="parent", request=request))
        update_job(job_id, session_id="parent")
    update_job(job_id, lease_owner="owner")
    account = CompilationAccount(job_id, 0, "owner")
    # Repeated task/judge calls would exhaust the cap using their estimates.
    for role in ["prompt_task", "prompt_judge"] * 8:
        with account.call(role, "test", task_profile(config)) as usage:
            usage.update(prompt_tokens=5, completion_tokens=3)
    for owner_id in [job_id, "parent"] if with_parent else [job_id]:
        totals = budget.totals(owner_id)
        assert totals["tokens"] == totals["reported_actual_tokens"] == 128
        assert totals["reservations"] == 16
        assert totals["by_role"]["prompt_judge"]["reserved_tokens"] == 64
    with (
        pytest.raises(budget.BudgetExhausted, match="call/token limit"),
        account.call("prompt_task", "test", task_profile(config)),
    ):
        pytest.fail("call limit was bypassed")


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": 5},
        {"prompt_tokens": 5, "completion_tokens": None},
        {"prompt_tokens": -1, "completion_tokens": 3},
        {"prompt_tokens": True, "completion_tokens": 3},
        {"prompt_tokens": 5, "completion_tokens": "3"},
    ],
)
def test_compilation_keeps_reservation_without_complete_usage(
    database, config, examples, usage
):
    job_id = make_job(config, examples, limits={"max_tokens": 1000})
    update_job(job_id, lease_owner="owner")
    account = CompilationAccount(job_id, 0, "owner")
    with (
        pytest.raises(RuntimeError, match="provider failed"),
        account.call("prompt_judge", "x" * 300, task_profile(config)) as reported,
    ):
        reported.update(usage)
        raise RuntimeError("provider failed")
    assert budget.totals(job_id)["tokens"] == 684
    assert budget.totals(job_id)["unreported_token_operations"] == 1
    with (
        pytest.raises(budget.BudgetExhausted),
        account.call("prompt_judge", "x" * 300, task_profile(config)),
    ):
        pytest.fail("uncertain usage was released")


def test_compilation_keeps_capacity_for_calls_in_progress(database, config, examples):
    job_id = make_job(config, examples, limits={"max_tokens": 1000})
    update_job(job_id, lease_owner="owner")
    account = CompilationAccount(job_id, 0, "owner")
    with account.call("prompt_task", "x" * 300, task_profile(config)) as usage:
        assert budget.totals(job_id)["tokens"] == 684
        with (
            pytest.raises(budget.BudgetExhausted),
            account.call("prompt_task", "x" * 300, task_profile(config)),
        ):
            pytest.fail("in-flight reservation was ignored")
        usage.update(prompt_tokens=5, completion_tokens=3)
    with account.call("prompt_task", "x" * 300, task_profile(config)) as usage:
        usage.update(prompt_tokens=5, completion_tokens=3)
    assert budget.totals(job_id)["tokens"] == 16


class NullAccount:
    def check(self):
        pass

    @contextmanager
    def call(self, *args, **kwargs):
        yield {}


@pytest.fixture
def dspy_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "dspy_cache"))
    return pytest.importorskip("dspy")


def test_real_dspy_adapter_renders_identical_prompt(dspy_runtime, config, monkeypatch):
    from ragbench_core.prompt_optimization import compiler

    observed = []
    monkeypatch.setattr(
        compiler,
        "model_call",
        lambda profile, prompt, **kwargs: (observed.append(prompt) or "Answer", {}),
    )
    task, _, adapter = compiler.make_bridges(
        dspy_runtime,
        NullAccount(),
        config,
        PromptOptimizationConfig(preset="bootstrap_small_v1"),
    )
    signature = dspy_runtime.Signature(
        "question, context -> answer", instructions="Give evidence."
    )
    demo = {"question": "Prior question", "context": "[a::1]\nA", "answer": "A [a::1]"}
    result = adapter(
        task,
        {},
        signature,
        [demo],
        {"question": "Current question", "context": "[b::1]\nB"},
    )
    artifact = CompiledPrompt(
        instructions=signature.instructions, demonstrations=[demo]
    ).model_dump()
    assert result == [{"answer": "Answer"}]
    assert observed == [
        CompiledPromptBuilder().build(
            "Current question", "[b::1]\nB", builder_config(artifact)
        )
    ]


def test_gepa_does_not_swallow_budget_exhaustion_in_trace_bootstrap(
    dspy_runtime, config, examples, monkeypatch
):
    from ragbench_core.prompt_optimization import compiler

    splits = split_questions([asdict(e) for e in examples])

    class BoundedAccount(NullAccount):
        calls = 0

        @contextmanager
        def call(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 2 * len(splits["validation"]):
                raise budget.BudgetExhausted("test budget exhausted")
            yield {}

    account = BoundedAccount()
    monkeypatch.setattr(compiler, "model_call", lambda *args, **kwargs: ("Answer", {}))
    monkeypatch.setattr(
        compiler,
        "judge_answer",
        lambda *args, **kwargs: {
            "scores": {"answer_correctness": 0.9},
            "details": {
                "status": "ok",
                "verdicts": {
                    "answer_correctness": {"score": 0.9, "reason": "Supported"},
                    "claims": [],
                    "citations": [],
                },
            },
        },
    )
    prepared = [
        {**asdict(e), "context": "Evidence", "context_chunks": []} for e in examples
    ]
    recipe = PromptOptimizationConfig(
        preset="gepa_light_v1",
        parameters={"skip_perfect_score": False, "max_metric_calls": 12},
    )
    with pytest.raises(budget.BudgetExhausted, match="test budget"):
        compiler.compile_prompt(
            recipe,
            prepared,
            splits,
            {**config, "_snapshot": {"dataset_hash": "test"}},
            task_profile(config).model_dump(),
            account,
        )
    assert account.calls == 2 * len(splits["validation"]) + 1


def test_compiler_retries_invalid_correctness_verdict(
    dspy_runtime, config, examples, monkeypatch
):
    from ragbench_core.prompt_optimization import compiler

    monkeypatch.setattr(compiler, "model_call", lambda *args, **kwargs: ("Answer", {}))
    calls = []

    def judge(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "scores": {"answer_correctness": None},
                "details": {"status": "error", "error": "invalid JSON"},
            }
        return {
            "scores": {"answer_correctness": 0.9},
            "details": {
                "status": "ok",
                "verdicts": {"answer_correctness": {"score": 0.9, "reason": "Correct"}},
            },
        }

    monkeypatch.setattr(compiler, "judge_answer", judge)
    prepared = [
        {**asdict(e), "context": "Evidence", "context_chunks": []} for e in examples
    ]
    splits = split_questions([asdict(e) for e in examples])
    _, summary = compiler.compile_prompt(
        PromptOptimizationConfig(preset="bootstrap_small_v1"),
        prepared,
        splits,
        {**config, "_snapshot": {"dataset_hash": "test"}},
        task_profile(config).model_dump(),
        NullAccount(),
    )
    assert summary["validation_score"] == pytest.approx(0.9)
    assert len(calls) > len(splits["validation"])
    assert all(call["correctness_only"] for call in calls)


@pytest.mark.parametrize(
    "error",
    [budget.OptimizationStopped("cancelled"), RuntimeError("provider unavailable")],
)
def test_compiler_propagates_cancellation_and_provider_errors(
    dspy_runtime, config, examples, monkeypatch, error
):
    from ragbench_core.prompt_optimization import compiler

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(compiler, "model_call", fail)
    prepared = [
        {**asdict(e), "context": "Evidence", "context_chunks": []} for e in examples
    ]
    with pytest.raises(type(error), match=str(error)):
        compiler.compile_prompt(
            PromptOptimizationConfig(preset="bootstrap_small_v1"),
            prepared,
            split_questions([asdict(e) for e in examples]),
            {**config, "_snapshot": {"dataset_hash": "test"}},
            task_profile(config).model_dump(),
            NullAccount(),
        )


@pytest.mark.parametrize("preset", PRESETS)
def test_real_optimizers_compile_with_deterministic_transport(
    dspy_runtime, preset, config, examples, monkeypatch
):
    from ragbench_core.prompt_optimization import compiler

    outputs = []

    def model_call(profile, prompt, **kwargs):
        outputs.append(prompt)
        if prompt.startswith("You are a careful RAG answer generator."):
            return "Known answer [doc::1]", {}
        if "Your output fields are:" in prompt:
            part = prompt.split("Your output fields are:", 1)[1].split(
                "All interactions", 1
            )[0]
            fields = re.findall(r"\d+\. `([^`]+)`", part)
            return "\n".join(
                f"[[ ## {field} ## ]]\nGive a direct, evidence-grounded answer."
                for field in fields
            ) + "\n[[ ## completed ## ]]", {}
        return "```\nGive a direct, evidence-grounded answer.\n```", {}

    monkeypatch.setattr(compiler, "model_call", model_call)
    monkeypatch.setattr(
        compiler,
        "judge_answer",
        lambda *args, **kwargs: {
            "scores": {"answer_correctness": 0.9},
            "details": {
                "status": "ok",
                "verdicts": {
                    "answer_correctness": {"score": 0.9, "reason": "Supported"},
                    "claims": [],
                    "citations": [],
                },
            },
        },
    )
    prepared = [
        {**asdict(e), "context": "[doc::1]\nKnown answer", "context_chunks": []}
        for e in examples
    ]
    splits = split_questions([asdict(e) for e in examples])
    recipe = PromptOptimizationConfig(
        preset=preset,
        parameters={"max_metric_calls": 18}
        if preset == "gepa_light_v1"
        else {"num_trials": 2, "num_candidates": 2}
        if "mipro" in preset
        else {},
    )
    config = {**config, "_snapshot": {"dataset_hash": "test"}}
    artifact, summary = compiler.compile_prompt(
        recipe,
        prepared,
        splits,
        config,
        task_profile(config).model_dump(),
        NullAccount(),
    )
    assert summary["validation_count"] == len(splits["validation"])
    assert summary["validation_score"] == pytest.approx(0.9)
    assert artifact["provenance"]["training_question_ids"] == splits["training"]
    assert outputs
    candidates = summary["candidate_history"]["candidates"]
    selected = [item for item in candidates if item["selected"]]
    assert len(selected) == 1
    assert selected[0]["instructions"] == artifact["instructions"]
    assert selected[0]["demonstrations"] == artifact["demonstrations"]
    assert selected[0]["evaluations"][-1] == {
        "score": pytest.approx(0.9),
        "sample_count": len(splits["validation"]),
        "partition": "validation",
        "scope": "full",
        "stage": "final_validation",
    }
    assert any("initial" in item["sources"] for item in candidates)
    if "mipro" in preset:
        assert any(
            item["kind"] == "instruction" and not item["evaluations"]
            for item in candidates
        )
        assert any(item["kind"] == "demonstrations" for item in candidates)
        assert any(
            any(source.startswith("mipro_trial:") for source in item["sources"])
            for item in candidates
        )
        assert any(
            any(source.startswith("mipro_proposal:") for source in item["sources"])
            for item in candidates
        )
    if preset != "bootstrap_small_v1":
        evaluations = [
            evaluation
            for item in candidates
            for evaluation in item["evaluations"]
            if evaluation["stage"] == "optimizer"
        ]
        assert evaluations
        assert all(
            evaluation["score"] == pytest.approx(0.9) for evaluation in evaluations
        )
    assert "candidate_history" not in artifact
    # Compare with DSPy's original proposal path. Observations must not change
    # the selected artifact or add task/proposal/judge budget operations.
    observed_calls = []

    class RecordingAccount(NullAccount):
        @contextmanager
        def call(self, role, *args, **kwargs):
            observed_calls.append(role)
            yield {}

    optimizer_factory = compiler.make_optimizer
    captured, _ = compiler.compile_prompt(
        recipe,
        prepared,
        splits,
        config,
        task_profile(config).model_dump(),
        RecordingAccount(),
    )
    capture_calls = list(observed_calls)
    observed_calls.clear()
    monkeypatch.setattr(
        compiler,
        "make_optimizer",
        lambda dspy, recipe, metric, proposal, history: optimizer_factory(
            dspy, recipe, metric, proposal
        ),
    )
    original, _ = compiler.compile_prompt(
        recipe,
        prepared,
        splits,
        config,
        task_profile(config).model_dump(),
        RecordingAccount(),
    )
    assert original == captured == artifact
    assert observed_calls == capture_calls


def test_manual_api_compiles_and_exports_evaluation(
    database, config, examples, tmp_path, monkeypatch
):
    from ragbench_core.prompt_optimization.contracts import CompilationRequest

    from services.api.app import prompt_optimizations as api
    from services.worker.app import prompt_tasks

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "test.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(api, "CONFIG_ROOT", configs)
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(api, "ARTIFACT_ROOT", tmp_path / "artifacts")
    queued = []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kwargs: queued.append(args[0]),
    )
    result = api.compile_prompts(
        CompilationRequest(config_path="test.yaml", prompt_optimization=settings())
    )
    assert queued == [result["id"]]
    run_job(result["id"], compiler=fake_compile, prepare=fake_prepare)
    detail = api.compilation(result["id"])
    artifact_id = detail["artifacts"][0]["id"]
    exported = api.artifact_config(artifact_id)
    assert exported["config"]["_snapshot"]["question_ids"] == detail["splits"][
        "evaluation"
    ] or set(exported["config"]["_snapshot"]["question_ids"]) == set(
        detail["splits"]["evaluation"]
    )
    assert (
        exported["config"]["pipeline"]["query"]["prompt_builders"][0]["type"]
        == "compiled"
    )
    assert (
        snapshot_inputs("replayed", yaml.safe_load(exported["yaml"]))["_snapshot"][
            "dataset_hash"
        ]
        == exported["config"]["_snapshot"]["dataset_hash"]
    )


def fake_compile_with_history(*args):
    from ragbench_core.prompt_optimization.history import CandidateHistory

    artifact, summary = fake_compile(*args)
    history = CandidateHistory([], {})
    history.observe(
        "Unused suggestion 100%", None, kind="instruction", source="proposal"
    )
    history.observe(artifact["instructions"], [], source="initial")
    return artifact, {**summary, "candidate_history": history.finish(artifact, [], 0.9)}


def test_candidate_history_is_job_scoped_and_reuses_artifacts(
    database, config, examples
):
    from services.api.app import prompt_optimizations as api

    first_id = make_job(config, examples)
    first = get_job(first_id)
    second_id = create_job(
        first["config"], settings(), first["splits"], task_profile(config)
    )
    update_job(first_id, attempts=1, lease_until=0, status="running")
    for job_id in (first_id, second_id):
        run_job(job_id, compiler=fake_compile_with_history, prepare=fake_prepare)
    assert artifacts_for_job(first_id)[0]["id"] == artifacts_for_job(second_id)[0]["id"]
    page = api.candidates(first_id, limit=1)
    assert page["total"] == 2
    assert len(page["items"]) == 1
    assert page["history"]["candidate_count"] == 2
    assert page["history"]["pending"] is False
    assert page["items"][0]["attempt"] == 2
    assert "instructions" not in page["items"][0]
    selected = api.candidates(first_id, offset=1, limit=1)["items"][0]
    assert selected["selected"] is True
    full = api.candidate(first_id, selected["id"])
    assert full["instructions"] == "Use evidence."
    assert full["demonstrations"] == []
    assert full["evaluations"][0]["score"] == 0.9
    with pytest.raises(api.HTTPException) as error:
        api.candidate(second_id, selected["id"])
    assert error.value.status_code == 404
    assert api.candidates(first_id, q="100%")["total"] == 1
    assert api.candidates(first_id, q="100_")["total"] == 0
    assert api.candidates(first_id, config_index=1)["total"] == 0
    assert api.candidates(first_id, offset=100)["items"] == []
    before = budget.totals(first_id)
    run_job(first_id, compiler=lambda *args: pytest.fail("completed work repeated"))
    assert api.candidates(first_id)["total"] == 2
    assert budget.totals(first_id) == before


def test_candidate_history_groups_configurations(database, config, examples):
    from services.api.app import prompt_optimizations as api

    first = get_job(make_job(config, examples))
    recipe = settings().configs[0]
    job_id = create_job(
        first["config"],
        PromptOptimizationSettings(configs=[recipe, recipe]),
        first["splits"],
        task_profile(config),
    )
    run_job(job_id, compiler=fake_compile_with_history, prepare=fake_prepare)
    page = api.candidates(job_id)
    assert page["total"] == 4
    assert [item["config_index"] for item in page["items"]] == [0, 0, 1, 1]
    assert len({item["id"] for item in page["items"]}) == 4
    assert api.candidates(job_id, config_index=1)["total"] == 2
    assert len(artifacts_for_job(job_id)) == 1
    assert len(page["history"]["configurations"]) == 2


def test_candidate_history_commits_with_result(database, config, examples, monkeypatch):
    from ragbench_core.prompt_optimization import history_store

    from services.api.app import prompt_optimizations as api

    job_id = make_job(config, examples)
    save = history_store.save_history

    def fail_save(*args):
        save(*args)
        args[0].flush()
        raise RuntimeError("transaction failed")

    monkeypatch.setattr(history_store, "save_history", fail_save)
    run_job(job_id, compiler=fake_compile_with_history, prepare=fake_prepare)
    assert get_job(job_id)["status"] == "failed"
    assert api.candidates(job_id)["items"] == []
    assert artifacts_for_job(job_id) == []


def test_candidate_history_legacy_and_http_validation(database, config, examples):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from services.api.app import prompt_optimizations as api

    job_id = make_job(config, examples)
    assert api.candidates(job_id)["history"]["pending"] is True
    run_job(job_id, compiler=fake_compile, prepare=fake_prepare)
    page = api.candidates(job_id)
    assert page["history"]["available"] is False
    assert len(page["items"]) == 1
    saved = page["items"][0]
    assert saved["historical"] is True
    assert api.candidate(job_id, saved["id"])["instructions"] == "Use evidence."
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        for query in (
            "limit=0",
            "limit=101",
            "offset=-1",
            "config_index=-1",
            "q=" + "x" * 501,
        ):
            assert (
                client.get(
                    f"/prompt-optimizations/{job_id}/candidates?{query}"
                ).status_code
                == 422
            )
        assert client.get("/prompt-optimizations/missing/candidates").status_code == 404
        assert (
            client.get(f"/prompt-optimizations/{job_id}/candidates/missing").status_code
            == 404
        )
        assert (
            client.get(
                f"/prompt-optimizations/{job_id}/candidates/{saved['id']}"
            ).status_code
            == 200
        )


def test_candidate_history_distinguishes_examples_and_evaluation_scopes(dspy_runtime):
    from ragbench_core.prompt_optimization.history import (
        CandidateHistory,
        evaluation_callback,
    )

    prepared = [
        {"question_id": f"q{i}", "question": str(i), "context": "context"}
        for i in range(3)
    ]
    history = CandidateHistory(
        prepared, {"training": ["q0"], "validation": ["q1", "q2"]}
    )
    demo = {"question": "Example?", "context": "Evidence", "answer": "Answer"}
    empty = history.observe("Instructions", [], source="initial")
    with_examples = history.observe("Instructions", [demo])
    assert empty != with_examples
    assert history.observe("Instructions", [], source="executed") == empty
    validation = [SimpleNamespace(question=str(i), context="context") for i in (1, 2)]
    callback = evaluation_callback(None, history)
    program = SimpleNamespace(
        answer=SimpleNamespace(
            signature=SimpleNamespace(instructions="Instructions"), demos=[]
        )
    )
    for call_id, dataset, raw_score in (
        ("a", validation, 0),
        ("b", validation[:1], 50),
        ("c", validation, 90),
    ):
        callback.on_evaluate_start(
            call_id, SimpleNamespace(devset=dataset), {"program": program}
        )
        callback.on_evaluate_end(call_id, SimpleNamespace(score=raw_score))
    evaluations = history.items[empty]["evaluations"]
    assert [item["score"] for item in evaluations] == [0, 0.5, 0.9]
    assert [item["scope"] for item in evaluations] == ["full", "minibatch", "full"]
    assert [item["sample_count"] for item in evaluations] == [2, 1, 2]


def test_gepa_history_includes_rejected_proposals(
    dspy_runtime, config, examples, monkeypatch
):
    from ragbench_core.prompt_optimization import compiler

    def model_call(profile, prompt, **kwargs):
        if prompt.startswith("You are a careful RAG answer generator."):
            return ("bad" if "Worse suggestion" in prompt else "good"), {}
        return "```Worse suggestion```", {}

    def judge(example, answer, *args, **kwargs):
        score = 0.0 if answer.answer == "bad" else 0.8
        return {
            "scores": {"answer_correctness": score},
            "details": {
                "status": "ok",
                "verdicts": {"answer_correctness": {"score": score}},
            },
        }

    monkeypatch.setattr(compiler, "model_call", model_call)
    monkeypatch.setattr(compiler, "judge_answer", judge)
    prepared = [
        {**asdict(e), "context": "Evidence", "context_chunks": []} for e in examples
    ]
    recipe = PromptOptimizationConfig(
        preset="gepa_light_v1", parameters={"max_metric_calls": 30}
    )
    artifact, summary = compiler.compile_prompt(
        recipe,
        prepared,
        split_questions([asdict(e) for e in examples]),
        {**config, "_snapshot": {"dataset_hash": "test"}},
        task_profile(config).model_dump(),
        NullAccount(),
    )
    rejected = next(
        item
        for item in summary["candidate_history"]["candidates"]
        if item["instructions"] == "Worse suggestion"
    )
    assert "gepa_proposal" in rejected["sources"]
    assert "gepa_retained" not in rejected["sources"]
    assert rejected["selected"] is False
    assert rejected["evaluations"]
    assert all(item["score"] == 0 for item in rejected["evaluations"])
    assert "Worse suggestion" not in artifact["instructions"]


def test_auto_plan_prioritizes_selected_dspy_configs(database, config, monkeypatch):
    from ragbench_core.optimization.search import concrete_config
    from ragbench_core.prompt_optimization import auto

    concrete = concrete_config(config, expand_cases(config)[0])
    recipe = settings().model_dump()
    request = OptimizationRequest(prompt_optimization=recipe).model_dump()
    with db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationSession(
                id="auto",
                request=request,
                template=config,
                progress={"workflow_version": 3},
            )
        )
    parent = {
        "candidate_hash": "base",
        "source_trial_id": None,
        "variant": "baseline",
        "scheduled": True,
    }
    old = {
        "candidate_hash": "preset",
        "source_trial_id": None,
        "variant": "concise_v1",
        "scheduled": True,
    }
    monkeypatch.setattr(
        auto,
        "answer_plan",
        lambda *args: {
            "retrieval_shortlist": [],
            "answer_plan": [parent, old],
            "answer_candidates": [concrete, concrete],
        },
    )
    tools = SimpleNamespace(session_id="auto", session=store.get("auto"))
    plan = auto.plan_answers(tools, 2)["prompt_answer_plan"]
    assert [e["variant"] for e in plan] == [
        "baseline",
        "dspy:mipro_zero_shot_v1",
        "concise_v1",
    ]
    assert [e["scheduled"] for e in plan] == [True, True, False]
    tools.session["request"]["prompt_optimization"]["configs"].append(
        deepcopy(recipe["configs"][0])
    )
    repeated = auto.plan_answers(tools, 2)["prompt_answer_plan"]
    assert repeated == plan


def test_context_preparation_preserves_citations_without_generating(
    config, monkeypatch
):
    from ragbench_core.components.context import DefaultContextAssembler
    from ragbench_core.engine import runner as module
    from ragbench_core.engine.runner import BenchmarkRunner
    from ragbench_core.models.schema import Chunk, RetrievalResult

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.control = lambda *args: None
    result = RetrievalResult(
        Chunk("chunk-1", "doc-1", "Source evidence."), 1.0, 1, "dense"
    )
    runner.registry = SimpleNamespace(
        embedders={"fake": object()},
        rewriters={"none": SimpleNamespace(rewrite=lambda question, *args: [question])},
        retrievers={"dense": object()},
        filters={"none": SimpleNamespace(filter=lambda results, *args: results)},
        rerankers={
            "none": SimpleNamespace(rerank=lambda question, results, *args: results)
        },
        compressors={
            "none": SimpleNamespace(compress=lambda question, results, *args: results)
        },
        assemblers={"default": DefaultContextAssembler()},
        prompt_builders={},
        generators={},
    )
    monkeypatch.setattr(module, "embed_queries_cached", lambda *args: [[1.0]])
    monkeypatch.setattr(module, "retrieve_batch_cached", lambda *args: [[result]])
    case = expand_cases(config)[0]
    payload = runner.prepare_context(
        QueryExample("What is the evidence?"),
        case,
        {"chunks": [result.chunk], "index_name": "test"},
        {"embedding": None, "retrieval": None},
    )
    assert "[doc-1::chunk-1]" in payload["context"]
    assert payload["selected_results"] == [result]
    assert payload["answer_text"] == "" and payload["prompt"] == ""
