"""Recommendation acceptance, explanation isolation, and export consistency."""

import json
from copy import deepcopy

import pytest
import yaml
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.models import StructuredOutputError
from ragbench_core.optimization.recommendation import (
    INSTRUCTION,
    MAX_INPUT_BYTES,
    PROMPT_VERSION,
    evaluate_acceptance,
    explain_recommendation,
    explanation_evidence,
)
from ragbench_core.optimization.search import candidate_hash
from ragbench_core.optimization.stages import stage_dashboard
from ragbench_core.optimization.tools import OptimizationTools
from ragbench_core.optimization.workflow import acceptance_decision, progress, step
from test_optimization import (  # noqa: F401 -- shared pytest fixtures
    database,
    fake_registry,
    prepare_dataset,
    save_stage_trial,
    session,
    template,
)


@pytest.fixture
def evidence():
    baseline = {"pipeline": {"query": "dense"}, "llm": {"model": "fixed"}}
    candidate = {"pipeline": {"query": "bm25"}, "llm": {"model": "fixed"}}
    return {
        "baseline": baseline,
        "candidate": candidate,
        "trials": [
            {
                "id": "left",
                "config": baseline,
                "phase": "confirm",
                "status": "completed",
                "summary": {"pricing_known": True},
            },
            {
                "id": "right",
                "config": candidate,
                "phase": "confirm",
                "status": "completed",
                "summary": {"pricing_known": True},
            },
        ],
        "question_ids": [str(i) for i in range(20)],
        "results": {
            side: [
                {
                    "question_id": str(i),
                    "metrics": {
                        "answer_correctness": score,
                        "query_latency_seconds": 1,
                        "total_estimated_cost": 0.01,
                    },
                }
                for i in range(20)
            ]
            for side, score in (("left", 0.5), ("right", 0.55))
        },
        "comparison": {
            "baseline_run_id": "left",
            "run_id": "right",
            "compatible": True,
            "metrics": [
                {
                    "metric": "answer_correctness",
                    "baseline_mean": 0.5,
                    "candidate_mean": 0.55,
                    "mean_delta": 0.05,
                    "ci95": [-0.02, 0.12],
                    "paired_count": 20,
                }
            ],
        },
        "limits": {},
    }


@pytest.mark.parametrize(
    "interval,outcome",
    [
        ([-0.02, 0.12], "promising_candidate"),
        ([0, 0.12], "promising_candidate"),
        ([0.00000001, 0.12], "confirmed_improvement"),
        ([0.02, 0.12], "confirmed_improvement"),
    ],
)
def test_candidate_outcomes_use_full_precision(evidence, interval, outcome):
    evidence["comparison"]["metrics"][0]["ci95"] = interval
    decision = evaluate_acceptance(**evidence)
    assert decision["outcome"] == outcome
    assert decision["recommended_source"] == "candidate"


@pytest.mark.parametrize(
    "problem,code",
    [
        ("tie", "correctness_improved"),
        ("regression", "correctness_improved"),
        ("same", "different_configuration"),
        ("failed", "trials_completed"),
        ("running", "trials_completed"),
        ("mismatched_config", "trials_completed"),
        ("incompatible", "datasets_compatible"),
        ("coverage", "paired_coverage"),
        ("missing_interval", "confidence_interval"),
        ("reversed_interval", "confidence_interval"),
        ("infinite_score", "correctness_available"),
        ("latency", "query_latency"),
        ("missing_latency", "query_latency"),
        ("cost", "query_cost"),
        ("unknown_pricing", "query_cost"),
        ("no_confirmation", "trials_completed"),
    ],
)
def test_rejected_candidates_explain_failed_checks(evidence, problem, code):
    metric = evidence["comparison"]["metrics"][0]
    if problem in {"tie", "regression"}:
        metric["mean_delta"] = 0 if problem == "tie" else -0.01
    elif problem == "same":
        evidence["candidate"] = deepcopy(evidence["baseline"])
    elif problem in {"failed", "running"}:
        evidence["trials"][1]["status"] = problem
    elif problem == "mismatched_config":
        evidence["trials"][1]["config"] = evidence["baseline"]
    elif problem == "incompatible":
        evidence["comparison"]["compatible"] = False
    elif problem == "coverage":
        metric["paired_count"] = 18
        evidence["results"]["right"] = evidence["results"]["right"][2:]
    elif problem == "missing_interval":
        metric["ci95"] = None
    elif problem == "reversed_interval":
        metric["ci95"] = [0.1, 0.01]
    elif problem == "infinite_score":
        metric["candidate_mean"] = float("inf")
    elif problem in {"latency", "missing_latency"}:
        evidence["limits"]["max_query_p95_seconds"] = 0.5
        if problem == "missing_latency":
            for row in evidence["results"]["right"]:
                row["metrics"].pop("query_latency_seconds")
    elif problem in {"cost", "unknown_pricing"}:
        evidence["limits"]["max_query_cost"] = 0.005 if problem == "cost" else 1
        if problem == "unknown_pricing":
            evidence["trials"][1]["summary"]["pricing_known"] = False
    elif problem == "no_confirmation":
        evidence["comparison"] = None
        evidence["trials"] = []
    decision = evaluate_acceptance(**evidence)
    assert decision["outcome"] == "baseline_retained"
    assert decision["recommended_source"] == "baseline"
    assert code in decision["reason_codes"]
    json.dumps(decision, allow_nan=False)


def test_coverage_boundary_and_unconfigured_limits(evidence):
    evidence["comparison"]["metrics"][0]["paired_count"] = 19
    evidence["results"]["right"] = evidence["results"]["right"][1:]
    evidence["trials"][1]["summary"]["pricing_known"] = False
    for row in evidence["results"]["right"]:
        row["metrics"].pop("query_latency_seconds")
    assert evaluate_acceptance(**evidence)["outcome"] == "promising_candidate"


@pytest.fixture
def recommendation_tools(request):
    session_id = request.getfixturevalue("session")
    progress(session_id, workflow_version=2, recommendation_policy_version=2)
    tools = OptimizationTools(session_id, fake_registry())
    prepare_dataset(tools)
    baseline = tools.baseline()
    candidate = deepcopy(baseline)
    candidate["pipeline"]["query"]["retrievers"] = [{"type": "bm25"}]
    save_stage_trial(tools, "left", baseline, "confirm", score=0.5)
    save_stage_trial(tools, "right", candidate, "confirm", score=0.55)
    count = sum(q["split"] == "holdout" for q in store.dataset(session_id)["payload"])
    progress(
        session_id,
        workflow_version=2,
        recommendation_policy_version=2,
        baseline_config=baseline,
        selected_config=candidate,
        comparison={
            "baseline_run_id": "left",
            "run_id": "right",
            "compatible": True,
            "metrics": [
                {
                    "metric": "answer_correctness",
                    "baseline_mean": 0.5,
                    "candidate_mean": 0.55,
                    "mean_delta": 0.05,
                    "ci95": [-0.02, 0.12],
                    "paired_count": count,
                }
            ],
        },
    )
    return tools


@pytest.mark.parametrize(
    "lower,outcome",
    [
        (-0.02, "promising_candidate"),
        (0.02, "confirmed_improvement"),
        (0, "promising_candidate"),
    ],
)
def test_export_dashboard_and_model_call_agree(recommendation_tools, lower, outcome):
    tools = recommendation_tools
    comparison = deepcopy(tools.session["progress"]["comparison"])
    comparison["metrics"][0]["ci95"][0] = lower
    progress(tools.session_id, comparison=comparison)
    calls = []

    def transport(profile, instruction, data):
        calls.append(data)
        assert tools.session["progress"]["acceptance"]["outcome"] == outcome
        assert instruction == INSTRUCTION
        assert profile.model == tools.session["request"]["planner"]["model"]
        assert "question_id" not in json.dumps(data)
        assert "expected_answer" not in json.dumps(data)
        return {
            "outcome": outcome,
            "explanation": "The candidate is recommended. It scored higher on holdout. These results use synthetic evaluation data.",
        }, {"prompt_tokens": 100, "completion_tokens": 30}

    tools.model_call = transport
    assert step("export", tools, lambda _: None) == "done"
    report = tools.read("report.json")
    assert report["outcome"] == outcome
    assert report["confirmed_improvement"] == (outcome == "confirmed_improvement")
    assert report["baseline_retained"] is False
    assert report["recommended_source"] == "candidate"
    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    assert candidate_hash(exported) == candidate_hash(
        tools.session["progress"]["selected_config"]
    )
    dashboard = stage_dashboard(tools.session)["stages"]["confirm"]
    assert dashboard["acceptance"] == report["acceptance"]
    assert dashboard["baseline_retained"] is False
    assert dashboard["explanation"] == report["explanation"]
    assert dashboard["outcome"] == outcome.replace("_", " ")
    from services.api.app.optimizations import export, recommendation

    before = deepcopy(tools.session)
    assert recommendation(tools.session_id)["acceptance"] == report["acceptance"]
    assert (
        export(tools.session_id, "yaml").body
        == (tools.root / "recommended.yaml").read_bytes()
    )
    assert tools.session == before
    assert explain_recommendation(tools, report["acceptance"]) == report["explanation"]
    assert len(calls) == 1
    usage = budget.totals(tools.session_id)
    assert usage["by_role"]["planner"]["reported_tokens"] == 130
    prompt = tools.read("recommendation_explanation_prompt.json")
    assert prompt["prompt_version"] == PROMPT_VERSION


@pytest.mark.parametrize("delta", [0, -0.01])
def test_new_policy_retains_and_exports_baseline_for_tie_or_regression(
    recommendation_tools, delta
):
    tools = recommendation_tools
    comparison = deepcopy(tools.session["progress"]["comparison"])
    comparison["metrics"][0]["mean_delta"] = delta
    comparison["metrics"][0]["candidate_mean"] = 0.5 + delta
    progress(tools.session_id, comparison=comparison)
    tools.model_call = lambda *_: (_ for _ in ()).throw(RuntimeError("offline"))
    step("export", tools, lambda _: None)
    report = tools.read("report.json")
    assert report["outcome"] == "baseline_retained"
    assert report["baseline_retained"] is True
    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    assert candidate_hash(exported) == candidate_hash(tools.baseline())
    assert "correctness_improved" in report["acceptance"]["reason_codes"]


@pytest.mark.parametrize(
    "failure",
    [
        "provider",
        "invalid_json",
        "malformed",
        "wrong_outcome",
        "empty",
        "too_long",
        "budget",
        "runtime_budget",
    ],
)
def test_explanation_failure_preserves_candidate_and_finishes(
    recommendation_tools, failure
):
    tools = recommendation_tools
    calls = []

    def transport(profile, instruction, data):
        calls.append(1)
        if failure == "provider":
            raise RuntimeError("secret provider diagnostic")
        if failure == "invalid_json":
            raise StructuredOutputError(
                "invalid json", {"prompt_tokens": 10, "completion_tokens": 10}
            )
        output = {
            "malformed": [],
            "wrong_outcome": {
                "outcome": "baseline_retained",
                "explanation": "Keep baseline.",
            },
            "empty": {"outcome": "promising_candidate", "explanation": " "},
            "too_long": {
                "outcome": "promising_candidate",
                "explanation": "word " * 121,
            },
        }[failure]
        return output, {"prompt_tokens": 10, "completion_tokens": 10}

    tools.model_call = transport
    if failure == "budget":
        request = deepcopy(tools.session["request"])
        request["limits"]["max_tokens"] = 1000
        store.update(tools.session_id, request=request)
    if failure == "runtime_budget":
        budget.reserve(tools.session_id, "used-runtime", "benchmark", 0, 0)
        budget.settle(
            "used-runtime", seconds=tools.session["request"]["limits"]["active_seconds"]
        )
    assert step("export", tools, lambda _: None) == "done"
    report = tools.read("report.json")
    assert report["outcome"] == "promising_candidate"
    assert report["status"] == tools.session["status"] == "completed"
    assert report["explanation"]["source"] == "fallback"
    assert "recommendation_evidence" not in report
    assert "secret provider diagnostic" not in json.dumps(report)
    assert "not statistically confirmed" in report["explanation"]["text"]
    assert not report["baseline_retained"]
    count = len(calls)
    explain_recommendation(tools, report["acceptance"])
    assert len(calls) == count
    if failure in {"budget", "runtime_budget"}:
        assert calls == []
        assert report["explanation"]["fallback_reason"] == "budget_exhausted"


@pytest.mark.parametrize("positive", [False, True])
def test_confirm_stage_uses_real_paired_results(
    recommendation_tools, monkeypatch, positive
):
    from ragbench_core.optimization import workflow

    tools = recommendation_tools
    trials = store.rows(store.OptimizationTrial, tools.session_id)
    with store.db.SessionLocal.begin() as conn:
        for trial in trials:
            conn.get(
                store.db.BenchmarkRun, trial["id"]
            ).dataset_hash = "same-frozen-holdout"
        if not positive:
            rows = (
                conn.execute(
                    store.select(store.db.BenchmarkResult)
                    .where(store.db.BenchmarkResult.run_id == "right")
                    .order_by(store.db.BenchmarkResult.id)
                )
                .scalars()
                .all()
            )
            for i, row in enumerate(rows):
                row.metrics = {
                    **row.metrics,
                    "answer_correctness": 0.1 if i == 0 else 0.7,
                }
    monkeypatch.setattr(
        workflow,
        "wait_for_trial",
        lambda _, config, *args: next(
            t for t in trials if candidate_hash(t["config"]) == candidate_hash(config)
        ),
    )
    assert step("confirm", tools, lambda _: None) == "export"
    state = tools.session["progress"]
    expected = "confirmed_improvement" if positive else "promising_candidate"
    assert state["acceptance"]["outcome"] == expected
    assert state["confirmed"] is positive
    assert state["baseline_retained"] is False
    assert state["comparison"]["metrics"]


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_explanation_respects_controls_before_and_during_call(
    recommendation_tools, status
):
    tools = recommendation_tools
    decision = acceptance_decision(tools)
    store.update(tools.session_id, status=status)
    with pytest.raises(budget.OptimizationStopped):
        explain_recommendation(tools, decision)
    store.update(tools.session_id, status="running")

    def transport(*_):
        store.update(tools.session_id, status=status)
        raise RuntimeError("transport stopped")

    tools.model_call = transport
    with pytest.raises(budget.OptimizationStopped):
        explain_recommendation(tools, decision)
    assert not (tools.root / "recommendation_explanation.json").exists()


def test_evidence_redaction_and_input_bound(recommendation_tools):
    tools = recommendation_tools
    selected = deepcopy(tools.session["progress"]["selected_config"])
    selected["llm"]["api_key"] = "sensitive-value"
    selected["pipeline"]["query"]["prompt_builders"][0]["template"] = (
        "large prompt " * 5000
    )
    progress(tools.session_id, selected_config=selected)
    data = explanation_evidence(tools, acceptance_decision(tools))
    text = json.dumps(data)
    assert "sensitive-value" not in text
    assert len(text.encode()) <= MAX_INPUT_BYTES
    assert data["configuration_detail_omitted"]


def test_legacy_export_does_not_call_model_or_reclassify(recommendation_tools):
    tools = recommendation_tools
    state = deepcopy(tools.session["progress"])
    state.pop("recommendation_policy_version")
    store.update(tools.session_id, progress=state)
    tools.model_call = lambda *_: pytest.fail("Legacy export must not call the model")
    step("export", tools, lambda _: None)
    report = tools.read("report.json")
    assert "acceptance" not in report
    assert report["baseline_retained"] is True
    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    assert candidate_hash(exported) == candidate_hash(tools.baseline())
