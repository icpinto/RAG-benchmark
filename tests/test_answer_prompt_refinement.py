"""DSPy shortlist refinement stays inside answering, including durable recovery."""

from copy import deepcopy

import pytest
from langgraph.types import Command
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.prompt_stage import prepare
from ragbench_core.optimization.stages import stage_dashboard
from ragbench_core.optimization.workflow import progress
from ragbench_core.prompt_optimization.artifacts import CompiledPrompt, seed_prompt
from ragbench_core.prompt_optimization.service import get_job, run_job
from test_adaptive_optimization import complete, graph_for, stop, transport
from test_optimization import database, session, template  # noqa: F401
from test_prompt_stage import configure
from test_prompt_stage import tools as legacy_tools  # noqa: F401


@pytest.fixture
def tools(request):
    return request.getfixturevalue("legacy_tools")


def configure_answering(tools):
    request = configure(tools, 1)
    request["limits"]["answer_trials"] = 2
    store.update(tools.session_id, request=request)
    progress(tools.session_id, workflow_version=7)
    return request


@pytest.mark.parametrize(
    "outcome,fail_first", [("better", False), ("better", True), ("worse", False)]
)
def test_shortlist_can_promote_the_second_parent_inside_answering(
    tools, monkeypatch, outcome, fail_first
):
    configure_answering(tools)
    jobs, calls, dispatched = [], [], []
    from services.worker.app import prompt_tasks

    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )

    def decide(data):
        answer_calls = [c for c in calls if c["objective"]["phase"] == "answer"]
        if data["objective"]["phase"] == "answer" and len(answer_calls) == 1:
            candidate = next(
                c for c in data["candidates"] if c["variant"].startswith("context:")
            )
            return {
                "reflection": "Compare compressed context",
                "memory": {"lessons": []},
                "action": "run",
                "candidate_id": candidate["candidate_id"],
                "hypothesis": "Compression may help after prompt refinement",
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
        assert tools.session["stage"] != "prompt"
        case = expand_cases(trial["config"])[0]
        compressed = case.query_config["compressor"]["type"] == "top_n"
        compiled = case.query_config["prompt_builder"]["type"] == "compiled"
        complete(
            tools,
            run_id,
            ((0.95 if compressed else 0.85) if outcome == "better" else 0.4)
            if compiled
            else (0.7 if compressed else 0.8),
        )

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert tools.session["stage"] == "answer" and len(jobs) == 1
    state = tools.session["progress"]
    assert len(state["answer_shortlist"]) == 2
    graph.invoke(Command(resume=True), config)
    assert len(jobs) == 1
    seeds = []
    for index in range(2):
        job = get_job(jobs[index])
        seeds.append(job["request"]["initial_prompt"])
        assert job["splits"] == get_job(jobs[0])["splits"]
        parent_id = job["request"]["parent_trial_id"]
        parent = next(t for t in dispatched if t["id"] == parent_id)
        assert job["config"]["pipeline"] == parent["config"]["pipeline"]
        assert seeds[-1] == seed_prompt(parent["config"])

        def compiler(
            recipe, prepared, splits, config, judge, account, parent_index=index, **kw
        ):
            if fail_first and parent_index == 0:
                raise RuntimeError("First recipe failed")
            return CompiledPrompt(
                instructions=f"Use evidence for parent {parent_index}",
                provenance={
                    "training_question_ids": splits["training"],
                    "validation_question_ids": splits["validation"],
                },
            ).model_dump(), {"validation_score": 0.99}

        run_job(job["id"], compiler=compiler, prepare=lambda *a: [])
        graph.invoke(Command(resume=True), config)

    assert tools.session["status"] == "completed"
    state = tools.session["progress"]
    assert (
        state["selected_phase"] == "answer" and state["answer_subphase"] == "completed"
    )
    winner = expand_cases(state["selected_config"])[0]
    assert (winner.query_config["compressor"]["type"] == "top_n") == (
        outcome == "better"
    )
    assert (winner.query_config["prompt_builder"]["type"] == "compiled") == (
        outcome == "better"
    )
    assert (
        state["prompt_plan"][1]["parent_answer_trial_id"]
        != state["answer_winner_trial_id"]
    )
    assert len({e["compilation_credit"] for e in state["prompt_plan"]}) == 2
    assert [e["status"] for e in state["prompt_plan"]] == [
        "failed" if fail_first else "completed",
        "completed",
    ]
    answers = [t for t in dispatched if t["phase"] == "answer"]
    assert len(answers) == (
        3 if fail_first else 4
    )  # Two screening trials plus protected refinements.
    validation = set(get_job(jobs[0])["splits"]["validation"])
    training = set(get_job(jobs[0])["splits"]["training"])
    assert training.isdisjoint(validation)
    assert all(
        set(t["config"]["_snapshot"]["question_ids"]) == validation for t in answers
    )
    assert all(
        not validation.intersection(t["config"]["_snapshot"]["question_ids"])
        and not training.intersection(t["config"]["_snapshot"]["question_ids"])
        for t in dispatched
        if t["phase"] == "confirm"
    )
    dashboard = stage_dashboard(tools.session)
    assert list(dashboard["stages"]) == ["retrieval", "answer", "confirm"]
    answer = dashboard["stages"]["answer"]
    assert (
        len(answer["compilations"]) == 2 and len(answer["refinement"]["shortlist"]) == 2
    )
    assert answer["selected_trial_id"] == state["selected_trial_id"]
    winning_trial = next(
        t for t in answer["trials"] if t["id"] == state["selected_trial_id"]
    )
    if outcome == "better":
        assert (
            winning_trial["source_answer_trial_id"]
            == state["prompt_plan"][1]["parent_answer_trial_id"]
        )
    else:
        assert state["selected_trial_id"] == state["answer_winner_trial_id"]
    assert all(
        u["tokens"] == u["seconds"] == 0
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_prompt"
    )
    assert len(tools.read("report.json")["answer_shortlist"]) == 2


def test_budget_reduces_shortlist_atomically_and_replays_without_replenishing(tools):
    request = configure_answering(tools)
    request["limits"]["max_tokens"] = 11_000_000
    store.update(tools.session_id, request=request)
    prepare(tools)
    state = tools.session["progress"]
    assert state["prompt_parent_slots"] == 1
    assert (
        state["prompt_shortlist_reason"]
        == "Budget fits 1 of 2 requested answer pipelines"
    )
    held = deepcopy(store.rows(store.BudgetUsage, tools.session_id))
    prepare(tools)
    assert store.rows(store.BudgetUsage, tools.session_id) == held
    assert len([u for u in held if u["role"] == "reserved_prompt"]) == 3


def test_answering_without_dspy_skips_refinement_and_has_three_stages(tools):
    progress(tools.session_id, workflow_version=7)
    tools.model_call = transport(stop, [])
    graph, config = graph_for(tools, lambda run_id: complete(tools, run_id))
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert tools.session["status"] == "completed"
    assert tools.session["progress"]["prompt_status"] == "disabled"
    dashboard = stage_dashboard(tools.session)
    assert list(dashboard["stages"]) == ["retrieval", "answer", "confirm"]
    assert dashboard["stages"]["answer"]["refinement"]["status"] == "disabled"


def test_short_shortlist_releases_unused_slots_and_rejects_unplanned_pipelines(
    tools, monkeypatch
):
    from ragbench_core.optimization.contracts import TrialRequest
    from ragbench_core.optimization.prompt_stage import credit
    from ragbench_core.prompt_optimization.service import update_job

    from services.worker.app import prompt_tasks

    configure_answering(tools)
    tools.model_call = transport(stop, [])
    jobs = []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )
    graph, config = graph_for(tools, lambda run_id: complete(tools, run_id))
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    state = tools.session["progress"]
    assert state["prompt_parent_slots"] == 2 and len(state["answer_shortlist"]) == 1
    unused = [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["id"] in {credit(tools, 1, "compile"), credit(tools, 1, "evaluate")}
    ]
    assert len(unused) == 2 and all(u["tokens"] == u["seconds"] == 0 for u in unused)
    changed = deepcopy(state["answer_winner_config"])
    changed["pipeline"]["query"]["context_assemblers"][0]["max_context_chars"] = 12000
    with pytest.raises(ValueError, match="approved compiled candidate"):
        tools.run_trial(
            TrialRequest(session_id=tools.session_id, config=changed, phase="answer"),
            None,
        )
    with pytest.raises(ValueError, match="version-six"):
        tools.run_trial(
            TrialRequest(
                session_id=tools.session_id, config=tools.baseline(), phase="prompt"
            ),
            None,
        )
    update_job(jobs[0], status="failed", error="No prompt candidate")
    graph.invoke(Command(resume=True), config)
    assert tools.session["status"] == "completed" and len(jobs) == 1


def test_search_exhaustion_still_runs_protected_refinement_in_answering(
    tools, monkeypatch
):
    from ragbench_core.optimization import adaptive
    from ragbench_core.prompt_optimization.service import update_job

    from services.worker.app import prompt_tasks

    configure_answering(tools)
    jobs, phases = [], []
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kw: jobs.append(args[0]),
    )
    monkeypatch.setattr(
        adaptive,
        "reserve_planning",
        lambda *a: (_ for _ in ()).throw(budget.BudgetExhausted("Search exhausted")),
    )

    def dispatch(run_id):
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == run_id
        )
        phases.append(trial["phase"])
        complete(tools, run_id)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "freeze"}, config)
    assert (
        tools.session["stage"] == "answer" and phases == ["answer"] and len(jobs) == 1
    )
    update_job(jobs[0], status="failed", error="No valid prompt")
    graph.invoke(Command(resume=True), config)
    assert tools.session["status"] == "completed" and phases == ["answer", "confirm"]
    assert tools.session["progress"]["search_stops"] == {
        "retrieval": "Search exhausted",
        "answer": "Search exhausted",
    }
