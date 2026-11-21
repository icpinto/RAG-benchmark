"""Adaptive decisions, evidence boundaries, accounting and replay without model services."""

import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.optimization import adaptive, budget, memory, store
from ragbench_core.optimization.adaptive_contracts import AnswerDecision
from ragbench_core.optimization.models import render_prompt
from ragbench_core.optimization.tools import OptimizationTools
from ragbench_core.optimization.workflow import build_graph, progress
from test_optimization import (
    database,  # noqa: F401 -- pytest fixture
    fake_registry,
    prepare_dataset,
    session,  # noqa: F401 -- pytest fixture
    template,  # noqa: F401 -- pytest fixture
)


@pytest.fixture
def tools(request):
    session_id = request.getfixturevalue("session")
    progress(session_id, workflow_version=4)
    tools = OptimizationTools(session_id, fake_registry())
    prepare_dataset(tools)
    tools.reserve_confirmation()
    return tools


def stop(data, reason="Enough evidence"):
    return {
        "reflection": "Reviewed the measured observation",
        "action": "stop",
        "reason": reason,
        "memory": {"lessons": []},
    }


def choose(data, pipeline=None, candidate_id=None):
    result = {
        "reflection": "Current evidence suggests another experiment",
        "action": "run",
        "hypothesis": "The selected change will improve the measured score",
        "memory": {"lessons": []},
    }
    if pipeline is not None:
        result["pipeline"] = pipeline
    if candidate_id is not None:
        result["candidate_id"] = candidate_id
    return result


def complete(tools, run_id, score=0.5, failed=False):
    with store.db.SessionLocal.begin() as conn:
        trial = conn.get(store.OptimizationTrial, run_id)
        trial.status = "failed" if failed else "completed"
        trial.summary = {
            "pricing_known": False,
            **({"error_type": "TimeoutError"} if failed else {}),
        }
        run = conn.get(store.db.BenchmarkRun, run_id)
        run.status = trial.status
        config = deepcopy(trial.config)
        case = expand_cases(config)[0]
        if not failed:
            ids = set(config["_snapshot"]["question_ids"])
            for q in store.dataset(tools.session_id)["payload"]:
                if q["question_id"] not in ids:
                    continue
                conn.add(
                    store.db.BenchmarkResult(
                        id=run_id + q["question_id"],
                        run_id=run_id,
                        case_id=case.case_id,
                        question_id=q["question_id"],
                        question=q["question"],
                        answer="Answer using evidence",
                        metrics={
                            "answer_correctness": score,
                            "reranked_ndcg@10": score,
                            "reranked_recall@10": score,
                            "query_latency_seconds": 0.1,
                        },
                        case_config={
                            "judge": {"feedback": "More precise evidence would help"}
                        },
                        artifact_paths={},
                    )
                )
        operation = config["_optimization"]["budget_operation"]
    budget.settle(operation, seconds=0.01)


def graph_for(tools, dispatch):
    graph = build_graph(InMemorySaver(), dispatch, lambda _: tools)
    return graph, {
        "configurable": {"thread_id": tools.session_id},
        "recursion_limit": 1000,
    }


def transport(callback, calls):
    def call(profile, instruction, data):
        calls.append(deepcopy(data))
        return callback(data), {"prompt_tokens": 50, "completion_tokens": 30}

    return call


def test_adaptive_run_observe_plan_order_and_exact_confirmation(tools):
    events, calls = [], []

    def decide(data):
        phase = data["objective"]["phase"]
        events.append("planner:" + phase)
        assert (
            len(render_prompt(memory.INSTRUCTION, data).encode())
            <= memory.MAX_INPUT_BYTES
        )
        if data["rules"]["allowed_actions"] == ["stop"]:
            return stop(data)
        if phase == "retrieval":
            proposal = next(
                c
                for c in data["candidates"]
                if c["pipeline"]["query"]["retrievers"] == [{"type": "bm25"}]
            )
            result = choose(data, pipeline=proposal["pipeline"])
        else:
            proposal = next(
                c for c in data["candidates"] if c["variant"] == "concise_v1"
            )
            result = choose(data, candidate_id=proposal["candidate_id"])
        result["memory"] = {
            "lessons": [
                {
                    "text": "Observe evidence before changing settings",
                    "status": "tentative",
                    "trial_ids": [data["current_observation"]["trial_id"]],
                }
            ]
        }
        return result

    tools.model_call = transport(decide, calls)
    configs = []

    def dispatch(run_id):
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == run_id
        )
        events.append("run:" + trial["phase"])
        configs.append(trial)
        case = expand_cases(trial["config"])[0]
        score = 0.9 if case.query_config["retriever"]["type"] == "bm25" else 0.2
        if case.query_config["prompt_builder"].get("preset_id") == "concise_v1":
            score = 0.98
        complete(tools, run_id, score)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert events[:8] == [
        "run:retrieval",
        "planner:retrieval",
        "run:retrieval",
        "planner:retrieval",
        "run:answer",
        "planner:answer",
        "run:answer",
        "planner:answer",
    ]
    assert events[8:] == ["run:confirm", "run:confirm"]
    assert store.get(tools.session_id)["recommendation"]["confirmed_improvement"]
    assert len(memory.iterations(tools.session_id)) == 4
    assert all(i["status"] == "committed" for i in memory.iterations(tools.session_id))
    assert calls[1]["memory"]["lessons"] and len(calls[1]["history"]) == 2
    assert calls[1]["current_observation"]["comparisons"]["initial"][
        "delta"
    ] == pytest.approx(0.7)
    held = [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_planner"
    ]
    assert all(u["tokens"] == u["seconds"] == 0 for u in held)
    model_usage = [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "planner"
    ]
    assert len(model_usage) == 4 and all(u["actual_tokens"] == 80 for u in model_usage)
    report = tools.read("report.json")
    assert (
        len(report["iterations"]) == 4
        and report["planner_prompt_version"] == memory.PROMPT_VERSION
    )
    assert any(e["status"] == "untested" for e in report["answer_plan"])
    holdout = [
        q for q in store.dataset(tools.session_id)["payload"] if q["split"] == "holdout"
    ]
    serialized = json.dumps(calls)
    assert all(
        q["question"] not in serialized and q["expected_answer"] not in serialized
        for q in holdout
    )
    selected = store.get(tools.session_id)["progress"]["selected_config"]["pipeline"]
    assert configs[-1]["config"]["pipeline"] == selected


def test_failed_run_becomes_feedback_before_next_experiment(tools):
    calls, dispatches = [], []

    def decide(data):
        if data["objective"]["phase"] == "retrieval" and data["rules"][
            "allowed_actions"
        ] != ["stop"]:
            assert data["current_observation"]["status"] == "failed"
            assert data["current_observation"]["score"] is None
            assert data["current_observation"]["errors"]
            return choose(data, pipeline=data["candidates"][0]["pipeline"])
        return stop(data)

    tools.model_call = transport(decide, calls)

    def dispatch(run_id):
        dispatches.append(run_id)
        complete(tools, run_id, failed=len(dispatches) == 1)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert (
        len(
            [
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["phase"] == "retrieval"
            ]
        )
        == 2
    )


def test_invalid_decision_pauses_then_resume_uses_new_attempt(tools):
    calls, dispatches = [], []
    tools.model_call = transport(
        lambda data: choose(data, pipeline=tools.baseline()["pipeline"]), calls
    )

    def dispatch(run_id):
        dispatches.append(run_id)
        complete(tools, run_id)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert store.get(tools.session_id)["status"] == "paused"
    assert len(calls) == 2 and len(dispatches) == 1
    assert calls[1]["corrective_feedback"] == "Configuration has already been attempted"
    paused = memory.iterations(tools.session_id)[0]
    assert len(paused["attempts"]) == 2
    store.update(tools.session_id, status="queued", error=None)
    tools.model_call = transport(stop, calls)
    graph.invoke(Command(resume=True), config)
    assert store.get(tools.session_id)["status"] == "completed"
    recovered = memory.iterations(tools.session_id)[0]
    assert recovered["cycle"] == 1 and len(recovered["attempts"]) == 3
    assert calls[2]["attempt_id"] not in {
        calls[0]["attempt_id"],
        calls[1]["attempt_id"],
    }
    assert (
        len(
            [
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["phase"] == "retrieval"
            ]
        )
        == 1
    )


def test_trial_wait_resume_does_not_repeat_dispatch_or_skip_planner(tools):
    dispatched, calls = [], []
    tools.model_call = transport(stop, calls)
    graph, config = graph_for(tools, dispatched.append)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert len(dispatched) == 1 and not calls
    graph.invoke(Command(resume=True), config)
    assert len(dispatched) == 1 and not calls
    complete(tools, dispatched[0])
    graph.invoke(Command(resume=True), config)
    assert len(calls) == 1 and len(dispatched) == 2
    assert memory.iterations(tools.session_id)[0]["status"] == "committed"


def test_context_compaction_and_oversized_mandatory_input(tools, monkeypatch):
    calls = []
    tools.model_call = transport(stop, calls)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    item = memory.iterations(tools.session_id)[0]
    observation = tools.read(item["observation_artifact"])
    observation["examples"] = [{"question": "x" * 50000}] * 3
    context = memory.build_context(
        tools, item, observation, [], AnswerDecision.model_json_schema()
    )
    assert (
        len(render_prompt(memory.INSTRUCTION, context).encode())
        <= memory.MAX_INPUT_BYTES
    )
    assert len(context["current_observation"]["examples"]) < 3
    monkeypatch.setattr(memory, "MAX_INPUT_BYTES", 10)
    with pytest.raises(memory.ContextTooLarge, match="Mandatory planner context"):
        memory.build_context(
            tools, item, observation, [], AnswerDecision.model_json_schema()
        )


def test_exhaustion_reflection_and_early_stop_preserve_caps(tools):
    request = tools.session["request"]
    request["limits"].update(retrieval_trials=1, answer_trials=1)
    store.update(tools.session_id, request=request)
    calls = []

    def decide(data):
        assert data["rules"]["allowed_actions"] == ["stop"]
        return stop(data, "Trial cap reached")

    tools.model_call = transport(decide, calls)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert len(calls) == 2
    assert len(adaptive.catalog(tools)) > 1
    assert sum(e["status"] == "executed" for e in adaptive.catalog(tools)) == 1


def test_api_history_scopes_redacts_paginates_and_stage_reports(tools):
    from fastapi import FastAPI
    from ragbench_core.optimization.stages import stage_dashboard

    from services.api.app.optimizations import router

    calls = []
    tools.model_call = transport(stop, calls)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get(
        f"/optimizations/{tools.session_id}/iterations?offset=0&limit=1"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2 and len(body["items"]) == 1
    item = body["items"][0]
    detail = client.get(
        f"/optimizations/{tools.session_id}/iterations/{item['id']}"
    ).json()
    assert detail["attempts"][0]["prompt"]["data"]["current_observation"]["trial_id"]
    assert (
        client.get(f"/optimizations/{tools.session_id}/iterations?limit=0").status_code
        == 422
    )
    assert (
        client.get(
            f"/optimizations/{tools.session_id}/iterations?phase=confirm"
        ).status_code
        == 422
    )
    with store.db.SessionLocal.begin() as conn:
        conn.add(
            store.OptimizationSession(
                id="other",
                request=tools.session["request"],
                template=tools.session["template"],
            )
        )
    assert (
        client.get(f"/optimizations/other/iterations/{item['id']}").status_code == 404
    )
    stages = stage_dashboard(store.get(tools.session_id))
    assert stages["stages"]["retrieval"]["planner"]["iteration_count"] == 1
    assert stages["stages"]["answer"]["plan"][0]["status"] == "executed"


def test_completed_observation_survives_budget_stop(tools):
    dispatched = []
    graph, config = graph_for(tools, dispatched.append)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    complete(tools, dispatched[0])
    progress(tools.session_id, stop_reason="Token budget exhausted")
    graph.invoke(Command(resume=True), config)
    assert len(memory.iterations(tools.session_id)) == 1
    assert store.get(tools.session_id)["status"] == "budget_exhausted"


def test_reject_unknown_answer_and_unsupported_lesson(tools):
    adaptive.initialize(tools, "retrieval")
    with pytest.raises(ValueError, match="observed trial IDs"):
        adaptive.validate_decision(
            tools,
            "retrieval",
            {
                **stop({}),
                "memory": {
                    "lessons": [
                        {
                            "text": "Claim",
                            "status": "supported",
                            "trial_ids": ["invented"],
                        }
                    ]
                },
            },
            False,
        )
    with pytest.raises(ValueError, match="untested approved"):
        adaptive.validate_decision(
            tools, "answer", choose({}, candidate_id="unknown"), False
        )


@pytest.mark.parametrize(
    "crash_point", ["observation", "model_response", "decision_commit"]
)
def test_crash_replay_reuses_observations_decisions_and_completed_model_calls(
    tools, monkeypatch, crash_point
):
    calls, dispatches = [], []
    tools.model_call = transport(stop, calls)

    def dispatch(run_id):
        dispatches.append(run_id)
        complete(tools, run_id)

    tripped = False
    if crash_point == "observation":
        original = adaptive.observation_iteration

        def crash(*args, **kwargs):
            nonlocal tripped
            result = original(*args, **kwargs)
            if not tripped:
                tripped = True
                raise RuntimeError("Simulated process exit after observation commit")
            return result

        monkeypatch.setattr(adaptive, "observation_iteration", crash)
    elif crash_point == "model_response":
        original = adaptive.validate_decision

        def crash(*args, **kwargs):
            nonlocal tripped
            if not tripped:
                tripped = True
                raise RuntimeError("Simulated process exit after provider response")
            return original(*args, **kwargs)

        monkeypatch.setattr(adaptive, "validate_decision", crash)
    else:
        original = budget.release_reservation

        def crash(*args, **kwargs):
            nonlocal tripped
            committed = [
                i
                for i in memory.iterations(tools.session_id)
                if i["status"] == "committed"
            ]
            if committed and not tripped:
                tripped = True
                raise RuntimeError("Simulated process exit after decision commit")
            return original(*args, **kwargs)

        monkeypatch.setattr(budget, "release_reservation", crash)
    graph, config = graph_for(tools, dispatch)
    with pytest.raises(RuntimeError, match="Simulated process exit"):
        graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    graph.invoke(None, config)
    assert store.get(tools.session_id)["status"] == "completed"
    assert len(memory.iterations(tools.session_id)) == 2
    assert len(calls) == 2
    assert (
        len(
            [
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["phase"] == "retrieval"
            ]
        )
        == 1
    )
    assert all(
        u["tokens"] == 0
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_planner"
    )


@pytest.mark.parametrize("initial_score,expected_trials", [(0.2, 2), (0.95, 1)])
def test_changed_observation_changes_next_experiment(
    tools, initial_score, expected_trials
):
    calls = []

    def decide(data):
        if (
            data["objective"]["phase"] == "retrieval"
            and data["current_observation"]["score"] < 0.8
        ):
            return choose(data, pipeline=data["candidates"][0]["pipeline"])
        return stop(data)

    tools.model_call = transport(decide, calls)
    graph, config = graph_for(
        tools, lambda rid: complete(tools, rid, initial_score if not calls else 0.95)
    )
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert (
        len(
            [
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["phase"] == "retrieval"
            ]
        )
        == expected_trials
    )


def test_answer_candidate_exhaustion_forces_stop_before_trial_cap(tools, monkeypatch):
    original = adaptive.answer_plan
    monkeypatch.setattr(
        adaptive,
        "answer_plan",
        lambda tools, trials, ranking, maximum: original(tools, trials, ranking, 1),
    )
    calls = []

    def decide(data):
        if data["objective"]["phase"] == "answer":
            assert data["remaining"]["trials"] > 0
            assert data["rules"]["allowed_actions"] == ["stop"]
        return stop(data)

    tools.model_call = transport(decide, calls)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert len(adaptive.catalog(tools)) == 1


def test_additive_iteration_table_and_duplicate_observation(tools):
    from sqlalchemy import inspect

    store.OptimizationIteration.__table__.drop(store.db.engine)
    before = tools.session
    store.db.init_db()
    store.db.init_db()
    assert "optimization_iterations" in inspect(store.db.engine).get_table_names()
    assert tools.session == before
    calls = []
    graph, config = graph_for(tools, calls.append)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    action = adaptive.phase_state(tools, "retrieval")["pending"]
    complete(tools, calls[0])
    adaptive.capture_pending_terminal(tools, "retrieval")
    trial = store.rows(store.OptimizationTrial, tools.session_id)[0]
    adaptive.observation_iteration(tools, "retrieval", action, trial)
    assert len(memory.iterations(tools.session_id)) == 1


def test_out_of_space_and_unapproved_compiled_prompts_rejected(tools):
    from ragbench_core.prompt_optimization.artifacts import (
        CompiledPrompt,
        builder_config,
    )

    config = tools.baseline()
    config["pipeline"]["query"]["retrievers"] = [{"type": "invented"}]
    with pytest.raises(ValueError):
        adaptive.validate_decision(
            tools, "retrieval", choose({}, pipeline=config["pipeline"]), False
        )
    config = tools.baseline()
    config["pipeline"]["query"]["context_assemblers"][0]["max_context_chars"] = 6000
    with pytest.raises(ValueError, match="preserve baseline"):
        adaptive.validate_decision(
            tools, "retrieval", choose({}, pipeline=config["pipeline"]), False
        )
    config = tools.baseline()
    config["pipeline"]["query"]["prompt_builders"] = [
        builder_config(CompiledPrompt(instructions="Unapproved").model_dump())
    ]
    with pytest.raises(ValueError, match="approved"):
        tools.validate_candidate(config)


def test_duplicate_compiled_pipeline_reuses_trial_and_advances_planner(tools):
    calls = []
    tools.model_call = transport(stop, calls)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    entries = adaptive.catalog(tools)
    duplicate = {
        **deepcopy(entries[0]),
        "candidate_id": "duplicate-recipe",
        "recipe_index": 0,
        "compilation_job_id": "duplicate-job",
        "compilation_status": "completed",
        "status": "selected",
    }
    entries.append(duplicate)
    adaptive.save_catalog(tools, entries)
    action = tools.write(
        "duplicate_action.json",
        {"candidate_id": "duplicate-recipe", "hypothesis": "Evaluate compiled recipe"},
    )
    adaptive.save_state(
        tools, "answer", pending=action, active_iteration=None, finished=False
    )
    graph, config = graph_for(
        tools, lambda _: pytest.fail("Identical pipeline must reuse its saved trial")
    )
    graph.invoke({"session_id": tools.session_id, "stage": "answer"}, config)
    assert (
        len(
            [
                t
                for t in store.rows(store.OptimizationTrial, tools.session_id)
                if t["phase"] == "answer"
            ]
        )
        == 1
    )
    observed = [
        i
        for i in memory.iterations(tools.session_id)
        if i["observation_key"] == "compilation:duplicate-job"
    ]
    assert len(observed) == 1 and observed[0]["status"] == "committed"


def test_invalid_json_response_is_cached_inspectable_and_accounted(tools, monkeypatch):
    from types import SimpleNamespace

    from ragbench_core.optimization import models

    provider_calls = []

    def post(url, json, **kwargs):
        provider_calls.append(json)
        output = (
            "This is not JSON"
            if len(provider_calls) == 1
            else '{"reflection":"Reviewed evidence","action":"stop","reason":"Enough evidence"}'
        )
        body = {"response": output, "prompt_eval_count": 40, "eval_count": 30}
        return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)

    monkeypatch.setattr(models.requests, "post", post)
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    item = memory.iterations(tools.session_id)[0]
    first = item["attempts"][0]
    detail = memory.iteration_detail(tools, item)
    assert detail["attempts"][0]["response"] == {"invalid_output": "This is not JSON"}
    assert len(provider_calls) == 3
    usage = [
        u
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "planner"
    ]
    assert len(usage) == 3 and all(u["actual_tokens"] == 70 for u in usage)
    snapshot = tools.read(first["prompt_artifact"])
    with pytest.raises(models.StructuredOutputError):
        tools.call(
            "planner", snapshot["instruction"], snapshot["data"], attempt_id=first["id"]
        )
    assert len(provider_calls) == 3


def test_transport_retry_is_accounted_and_bad_schema_does_not_expose_response(tools):
    calls = []

    def model(profile, instruction, data):
        calls.append(data)
        if len(calls) == 1:
            raise ValueError("Provider error with sensitive credential")
        return stop(data), {"prompt_tokens": 20, "completion_tokens": 10}

    tools.model_call = model
    graph, config = graph_for(tools, lambda rid: complete(tools, rid))
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    first = memory.iterations(tools.session_id)[0]
    assert len(first["attempts"]) == 2
    assert "sensitive credential" not in json.dumps(
        memory.iteration_detail(tools, first)
    )
    assert (
        len(
            [
                u
                for u in store.rows(store.BudgetUsage, tools.session_id)
                if u["role"] == "planner"
            ]
        )
        == 3
    )


@pytest.mark.parametrize("compilation_fails", [False, True])
def test_dspy_candidate_yields_compiles_once_and_uses_validation_feedback(
    tools, monkeypatch, compilation_fails
):
    from ragbench_core.prompt_optimization.artifacts import CompiledPrompt
    from ragbench_core.prompt_optimization.service import get_job, run_job, update_job

    from services.worker.app import prompt_tasks

    request = tools.session["request"]
    request["prompt_optimization"] = {"configs": [{"preset": "mipro_zero_shot_v1"}]}
    request["limits"].update(retrieval_trials=1, answer_trials=2)
    store.update(tools.session_id, request=request)
    calls, jobs, dispatches = [], [], []

    def decide(data):
        if (
            data["objective"]["phase"] == "answer"
            and data["rules"]["allowed_actions"] != ["stop"]
            and not data["current_observation"]["reference"].startswith("compilation:")
        ):
            selected = next(
                c for c in data["candidates"] if c["variant"].startswith("dspy:")
            )
            return choose(data, candidate_id=selected["candidate_id"])
        return stop(data)

    tools.model_call = transport(decide, calls)
    monkeypatch.setattr(
        prompt_tasks.compile_prompt_job,
        "apply_async",
        lambda args, **kwargs: jobs.append(args[0]),
    )

    def dispatch(run_id):
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == run_id
        )
        dispatches.append(trial)
        compiled = (
            expand_cases(trial["config"])[0].query_config["prompt_builder"]["type"]
            == "compiled"
        )
        complete(tools, run_id, 0.95 if compiled else 0.2)

    graph, config = graph_for(tools, dispatch)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    assert len(jobs) == 1 and graph.get_state(config).interrupts
    before = len(calls)
    graph.invoke(Command(resume=True), config)
    assert len(jobs) == 1 and len(calls) == before
    job = get_job(jobs[0])
    if compilation_fails:
        update_job(job["id"], status="failed", error="Simulated compile failure")
    else:

        def compiler(recipe, prepared, splits, config, judge, account):
            return CompiledPrompt(
                instructions="Use precise evidence.",
                provenance={
                    "training_question_ids": splits["training"],
                    "validation_question_ids": splits["validation"],
                },
            ).model_dump(), {"validation_score": 0.95}

        run_job(job["id"], compiler=compiler, prepare=lambda *args: [])
    graph.invoke(Command(resume=True), config)
    assert store.get(tools.session_id)["status"] == "completed"
    assert len(jobs) == 1
    answers = [t for t in dispatches if t["phase"] == "answer"]
    assert len(answers) == (1 if compilation_fails else 2)
    assert all(
        set(t["config"]["_snapshot"]["question_ids"])
        == set(job["splits"]["validation"])
        for t in answers
    )
    answer_calls = [d for d in calls if d["objective"]["phase"] == "answer"]
    training = [
        q
        for q in store.dataset(tools.session_id)["payload"]
        if q["question_id"] in job["splits"]["training"]
    ]
    assert all(q["question"] not in json.dumps(answer_calls) for q in training)
    if compilation_fails:
        assert (
            answer_calls[-1]["current_observation"]["reference"]
            == "compilation:" + job["id"]
        )
        assert (
            len(
                [
                    e
                    for e in adaptive.catalog(tools)
                    if e.get("compilation_status") == "failed"
                ]
            )
            == 1
        )
    else:
        assert store.get(tools.session_id)["recommendation"]["confirmed_improvement"]


def test_observation_sensitive_fields_redacted_and_missing_comparisons_explicit(tools):
    adaptive.initialize(tools, "retrieval")
    dispatches = []
    graph, config = graph_for(tools, dispatches.append)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    complete(tools, dispatches[0], failed=True)
    adaptive.capture_pending_terminal(tools, "retrieval")
    item = memory.iterations(tools.session_id)[0]
    observed = tools.read(item["observation_artifact"])
    assert observed["comparisons"]["initial"]["delta"] is None
    assert observed["operational"]["cost_per_query"] is None
    observed["configuration"]["api_key"] = "private-secret"
    tools.write(item["observation_artifact"], observed)
    assert "private-secret" not in json.dumps(memory.iteration_detail(tools, item))


def test_cancel_after_trial_finishes_records_observation_and_releases_reservation(
    tools,
):
    from services.api.app.optimizations import cancel

    dispatches = []
    graph, config = graph_for(tools, dispatches.append)
    graph.invoke({"session_id": tools.session_id, "stage": "retrieval"}, config)
    complete(tools, dispatches[0])
    cancel(tools.session_id)
    assert memory.iterations(tools.session_id)[0]["status"] == "cancelled"
    assert all(
        u["tokens"] == 0
        for u in store.rows(store.BudgetUsage, tools.session_id)
        if u["role"] == "reserved_planner"
    )
