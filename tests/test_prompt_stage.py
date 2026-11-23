"""Version-six stage handoff, protected capacity, and durable recipe execution."""

from copy import deepcopy

import pytest
from langgraph.types import Command
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.prompt_stage import prepare
from ragbench_core.optimization.stages import stage_dashboard
from ragbench_core.optimization.tools import OptimizationTools
from ragbench_core.optimization.workflow import progress
from ragbench_core.prompt_optimization.artifacts import CompiledPrompt, seed_prompt
from ragbench_core.prompt_optimization.contracts import PromptOptimizationSettings
from ragbench_core.prompt_optimization.service import get_job, run_job, task_profile
from test_adaptive_optimization import complete, graph_for, stop, transport
from test_optimization import (  # noqa: F401
    database,
    fake_registry,
    prepare_dataset,
    session,
    template,
)


@pytest.fixture
def tools(request):
    session_id = request.getfixturevalue("session")
    progress(session_id, workflow_version=6)
    instance = OptimizationTools(session_id, fake_registry())
    prepare_dataset(instance)
    instance.reserve_confirmation()
    return instance


def configure(tools, count=2):
    request = tools.session["request"]
    request["prompt_optimization"] = PromptOptimizationSettings(
        configs=[
            {"preset": name}
            for name in ["mipro_zero_shot_v1", "bootstrap_small_v1"][:count]
        ]
    ).model_dump()
    store.update(tools.session_id, request=request)
    progress(tools.session_id, workflow_version=6)
    return request


@pytest.mark.parametrize("outcome", ["better", "worse", "equal", "failure", "partial"])
def test_ordered_recipes_start_from_answer_winner_and_preserve_fallback(
    tools, monkeypatch, outcome
):
    configure(tools)
    jobs, calls, dispatched, seeds = [], [], [], []
    from services.worker.app import prompt_tasks

    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )

    def decide(data):
        assert all(
            not c.get("variant", "").startswith("dspy:") for c in data["candidates"]
        )
        if (
            data["objective"]["phase"] == "answer"
            and len([d for d in calls if d["objective"]["phase"] == "answer"]) == 1
        ):
            candidate = next(
                c for c in data["candidates"] if c["variant"] == "concise_v1"
            )
            return {
                "reflection": "Try concise answers",
                "memory": {"lessons": []},
                "action": "run",
                "candidate_id": candidate["candidate_id"],
                "hypothesis": "Concise instructions improve correctness",
            }
        return stop(data)

    tools.model_call = transport(decide, calls)

    def dispatch(run_id):
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == run_id
        )
        dispatched.append(trial)
        prompt = expand_cases(trial["config"])[0].query_config["prompt_builder"]
        score = 0.8 if prompt.get("preset_id") == "concise_v1" else 0.2
        if prompt["type"] == "compiled":
            score = (
                0.95
                if outcome in {"better", "partial"}
                else 0.8
                if outcome == "equal"
                else 0.4
            )
        complete(tools, run_id, score)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert tools.session["stage"] == "prompt" and len(jobs) == 1
    first_job = get_job(jobs[0])
    parent = tools.session["progress"]["answer_winner_config"]
    assert seed_prompt(parent)["source"]["preset_id"] == "concise_v1"
    assert tools.session["progress"]["prompt_budget_ready"]
    # A queued job survives orchestration replay without being dispatched twice.
    graph.invoke(Command(resume=True), config)
    assert len(jobs) == 1
    for index in range(2):
        job = get_job(jobs[index])
        assert job["config"]["pipeline"] == parent["pipeline"]
        assert job["splits"] == first_job["splits"]

        def compiler(
            recipe,
            prepared,
            splits,
            config,
            judge,
            account,
            *,
            initial_prompt,
            recipe_index=index,
        ):
            seeds.append(deepcopy(initial_prompt))
            if outcome == "failure" or outcome == "partial" and recipe_index == 0:
                raise RuntimeError("Simulated failure")
            with account.call(
                "prompt_task", "fixed", task_profile(config), tokens=100, cost=0
            ) as usage:
                usage.update(prompt_tokens=10, completion_tokens=5)
            return CompiledPrompt(
                instructions="Compiled " + str(recipe_index),
                provenance={
                    "training_question_ids": splits["training"],
                    "validation_question_ids": splits["validation"],
                },
            ).model_dump(), {"validation_score": 0.9}

        run_job(job["id"], compiler=compiler, prepare=lambda *a: [])
        graph.invoke(Command(resume=True), config)
    assert len(jobs) == 2 and tools.session["status"] == "completed"
    assert seeds[0] == seeds[1] == seed_prompt(parent)
    state = tools.session["progress"]
    assert len(state["prompt_plan"]) == 2
    answers = [t for t in dispatched if t["phase"] == "answer"]
    prompt_trials = [t for t in dispatched if t["phase"] == "prompt"]
    assert len(answers) == 2  # Its cap is independent of prompt evaluations.
    assert len(prompt_trials) == (
        0 if outcome == "failure" else 1 if outcome == "partial" else 2
    )
    assert all(
        set(t["config"]["_snapshot"]["question_ids"])
        == set(first_job["splits"]["validation"])
        for t in answers + prompt_trials
    )
    assert set(first_job["splits"]["training"]).isdisjoint(
        first_job["splits"]["validation"]
    )
    assert all(
        t["phase"] != "confirm"
        for t in dispatched[: len(answers) + len(prompt_trials) + 1]
    )
    promoted = outcome in {"better", "partial"}
    assert (state["selected_phase"] == "prompt") == promoted
    if not promoted:
        assert state["selected_config"]["pipeline"] == parent["pipeline"]
    else:
        winning = deepcopy(state["selected_config"]["pipeline"])
        winning["query"]["prompt_builders"] = deepcopy(
            parent["pipeline"]["query"]["prompt_builders"]
        )
        assert winning == parent["pipeline"]
    assert all(
        u["tokens"] == u["seconds"] == 0
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_prompt"
    )
    charged = [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "prompt_task"
    ]
    assert len(charged) == len(prompt_trials) and all(
        u["tokens"] == 15 for u in charged
    )
    dashboard = stage_dashboard(tools.session)
    assert list(dashboard["stages"]) == ["retrieval", "answer", "prompt", "confirm"]
    assert len(dashboard["stages"]["prompt"]["compilations"]) == 2
    assert (
        tools.read("report.json")["prompt_plan"][0]["parent_answer_trial_id"]
        == state["answer_winner_trial_id"]
    )


def test_prompt_capacity_is_atomic_and_replay_safe(tools):
    request = configure(tools)
    request["limits"]["max_tokens"] = 1000
    store.update(tools.session_id, request=request)
    with pytest.raises(budget.BudgetExhausted, match="Protected DSPy"):
        prepare(tools)
    assert not [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_prompt"
    ]
    request["limits"]["max_tokens"] = 20_000_000
    store.update(tools.session_id, request=request)
    prepare(tools)
    before = budget.totals(tools.session_id)
    prepare(tools)
    assert budget.totals(tools.session_id) == before


def test_disabled_stage_keeps_legacy_layout_and_skips_compilation(tools):
    from ragbench_core.optimization.prompt_stage import search

    progress(tools.session_id, workflow_version=6)
    assert search(tools, None, None) == "confirm"
    assert tools.session["progress"]["prompt_status"] == "disabled"
    assert stage_dashboard(tools.session)["stages"]["prompt"]["status"] == "disabled"
    for version in (3, 4, 5):
        progress(tools.session_id, workflow_version=version)
        assert list(stage_dashboard(tools.session)["stages"]) == [
            "retrieval",
            "answer",
            "confirm",
        ]


def test_search_exhaustion_preserves_capacity_for_prompt_control_and_confirmation(
    tools, monkeypatch
):
    from ragbench_core.optimization import adaptive

    from services.worker.app import prompt_tasks

    configure(tools, 1)
    jobs = []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )
    # No discretionary experiment can be reserved. Protected stages must still run.
    monkeypatch.setattr(
        adaptive,
        "reserve_planning",
        lambda *a: (_ for _ in ()).throw(
            budget.BudgetExhausted("Search allowance exhausted")
        ),
    )
    trials = []

    def dispatch(run_id):
        trials.append(
            next(
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["id"] == run_id
            )
        )
        complete(tools, run_id, 0.5)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert len(jobs) == 1 and [t["phase"] for t in trials] == ["prompt"]
    from ragbench_core.prompt_optimization.service import update_job

    update_job(jobs[0], status="failed", error="Simulated failure")
    graph.invoke(Command(resume=True), config)
    assert tools.session["status"] == "completed"
    assert [t["phase"] for t in trials] == ["prompt", "confirm"]
    assert tools.session["progress"]["search_stops"] == {
        "retrieval": "Search allowance exhausted",
        "answer": "Search allowance exhausted",
    }


def test_cancel_releases_future_capacity_but_keeps_charged_calls(tools):
    from ragbench_core.optimization.prompt_stage import cancel, credit

    configure(tools)
    prepare(tools)
    budget.reserve(
        tools.session_id,
        "charged-call",
        "prompt_task",
        100,
        0,
        seconds=5,
        credit=credit(tools, 0, "compile"),
    )
    budget.settle("charged-call", actual_tokens=15, seconds=1)
    progress(
        tools.session_id, prompt_plan=[{"status": "compiling"}, {"status": "planned"}]
    )
    store.update(tools.session_id, status="cancelled")
    cancel(tools)
    assert [e["status"] for e in tools.session["progress"]["prompt_plan"]] == [
        "cancelled",
        "cancelled",
    ]
    rows = store.rows(store.BudgetUsage, tools.session_id)
    assert next(u for u in rows if u["id"] == "charged-call")["tokens"] == 15
    assert all(
        u["tokens"] == u["seconds"] == 0 for u in rows if u["role"] == "reserved_prompt"
    )


def test_frozen_dataset_prompt_split_failure_precedes_any_search(tools, monkeypatch):
    import ragbench_core.optimization.prompt_stage as stage

    configure(tools)
    monkeypatch.setattr(
        stage,
        "split_questions",
        lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("Need independent evidence groups")
        ),
    )
    graph, config = graph_for(
        tools, lambda _: pytest.fail("No trial may run before split validation")
    )
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert tools.session["status"] == "needs_input"
    assert "independent evidence" in tools.session["error"]


def test_larger_compiled_prompt_must_fit_evaluation_and_confirmation(tools):
    from ragbench_core.optimization.prompt_stage import credit, estimate
    from ragbench_core.prompt_optimization.artifacts import builder_config

    configure(tools, 1)
    prepare(tools)
    config = deepcopy(tools.baseline())
    config["pipeline"]["query"]["prompt_builders"] = [
        builder_config(CompiledPrompt(instructions="large " * 8000).model_dump())
    ]
    tokens, cost, _ = estimate(tools, config)
    before = next(
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["id"] == credit(tools, 0, "evaluate")
    )
    budget.extend_reservation(tools.session_id, before["id"], tokens, cost)
    after = next(
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["id"] == before["id"]
    )
    assert after["tokens"] >= tokens
    request = tools.session["request"]
    request["limits"]["max_tokens"] = budget.totals(tools.session_id)["tokens"]
    store.update(tools.session_id, request=request)
    with pytest.raises(budget.BudgetExhausted, match="protected confirmation"):
        budget.extend_reservation(
            tools.session_id, before["id"], after["tokens"] + 1, cost
        )


@pytest.mark.parametrize("version", [6, 7])
def test_async_evaluation_replay_does_not_replenish_consumed_reservations(
    tools, monkeypatch, version
):
    from ragbench_core.optimization.prompt_stage import credit

    from services.worker.app import prompt_tasks

    configure(tools, 1)
    progress(tools.session_id, workflow_version=version)
    prepare(tools)
    progress(tools.session_id, selected_config=tools.baseline(), selected_trial_id=None)
    if version == 7:
        progress(tools.session_id, answer_screening_complete=True)
    jobs, dispatched = [], []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )

    def dispatch(run_id):
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == run_id
        )
        dispatched.append(trial)
        if trial["phase"] == "confirm":
            complete(
                tools,
                run_id,
                0.9
                if expand_cases(trial["config"])[0].query_config["prompt_builder"][
                    "type"
                ]
                == "compiled"
                else 0.5,
            )

    graph, config = graph_for(tools, dispatch)
    phase = "answer" if version == 7 else "prompt"
    graph.invoke({"session_id": tools.session_id, "stage": phase}, config)
    assert len(dispatched) == 1 and not jobs
    complete(tools, dispatched[0]["id"], 0.5)
    graph.invoke(Command(resume=True), config)
    assert len(jobs) == 1
    run_job(
        jobs[0],
        prepare=lambda *a: [],
        compiler=lambda *a, **kw: (
            CompiledPrompt(instructions="Use exact evidence.").model_dump(),
            {"validation_score": 0.9},
        ),
    )
    graph.invoke(Command(resume=True), config)
    assert len(dispatched) == 2
    held_id = credit(tools, 0, "evaluate")
    held = next(
        u for u in store.rows(store.BudgetUsage, tools.session_id) if u["id"] == held_id
    )
    graph.invoke(Command(resume=True), config)
    assert len(dispatched) == 2  # Waiting for the same benchmark, not a second trial.
    assert (
        next(
            u
            for u in store.rows(store.BudgetUsage, tools.session_id)
            if u["id"] == held_id
        )
        == held
    )
    store.update(tools.session_id, status="paused")
    graph.invoke(Command(resume=True), config)
    assert tools.session["status"] == "paused"
    store.update(tools.session_id, status="queued")
    complete(tools, dispatched[1]["id"], 0.9)
    graph.invoke(Command(resume=True), config)
    assert tools.session["status"] == "completed"
    assert len([t for t in dispatched if t["phase"] == phase]) == 2
    assert tools.session["progress"]["selected_phase"] == phase


@pytest.mark.parametrize("version", [6, 7])
def test_prompt_worker_budget_failure_does_not_stop_later_recipes(tools, monkeypatch, version):
    from types import SimpleNamespace

    from ragbench_core.engine import runner
    from ragbench_core.optimization import tools as tools_module
    from ragbench_core.optimization.contracts import TrialRequest

    from services.worker.app import optimization_tasks

    configure(tools, 1)
    progress(tools.session_id, workflow_version=version)
    prepare(tools)
    progress(tools.session_id, answer_winner_config=tools.baseline())
    if version == 7:
        progress(tools.session_id, answer_screening_complete=True)
    result = tools.run_trial(
        TrialRequest(
            session_id=tools.session_id, config=tools.baseline(), phase="answer" if version == 7 else "prompt"
        ),
        lambda _: None,
    )
    monkeypatch.setattr(tools_module, "OptimizationTools", lambda _: tools)
    monkeypatch.setattr(
        runner,
        "BenchmarkRunner",
        lambda **kw: SimpleNamespace(
            registry=tools.registry,
            run=lambda *a: (_ for _ in ()).throw(
                budget.BudgetExhausted("Evaluation allowance exhausted")
            ),
        ),
    )
    monkeypatch.setattr(
        optimization_tasks.advance_optimization, "apply_async", lambda **kw: None
    )
    optimization_tasks.run_optimization_trial.run(result.artifact_id)
    trial = next(
        t
        for t in store.rows(store.OptimizationTrial, tools.session_id)
        if t["id"] == result.artifact_id
    )
    assert trial["status"] == "failed" and trial["summary"]["budget_exhausted"]
    assert "stop_reason" not in tools.session["progress"]


def test_new_sessions_require_recipe_cost_caps_without_breaking_legacy_parsing(
    tools, tmp_path, monkeypatch
):
    import yaml
    from fastapi import HTTPException
    from ragbench_core.optimization.contracts import OptimizationRequest

    from services.api.app import optimizations as api

    price = {"input_price_per_million": 0, "output_price_per_million": 0}
    request = OptimizationRequest(
        config_path="test.yaml",
        limits={"max_cost": 1},
        planner=price,
        synthesizer=price,
        verifier=price,
        judge=price,
        prompt_optimization={"configs": [{"preset": "mipro_zero_shot_v1"}]},
    )
    # Old saved requests remain deserializable; creation enforces the v6 policy.
    assert request.prompt_optimization.configs[0].limits.max_cost is None
    (tmp_path / "test.yaml").write_text(yaml.safe_dump(tools.session["template"]))
    monkeypatch.setattr(api, "CONFIG_ROOT", tmp_path)
    with pytest.raises(HTTPException) as exc:
        api.create_optimization(request)
    assert (
        exc.value.status_code == 400 and "max_cost for each recipe" in exc.value.detail
    )
