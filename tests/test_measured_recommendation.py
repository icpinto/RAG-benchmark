"""Measured recommendations preserve decisions, evidence, and historical behavior."""

# ruff: noqa: F811 -- pytest resolves the imported shared fixtures by name.

import json
from copy import deepcopy

import pytest
import yaml
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.contracts import OptimizationRequest
from ragbench_core.optimization.recommendation import evaluate_acceptance
from ragbench_core.optimization.recommendation_metrics import (
    PRESENTATION_VERSION,
    check_detail,
    measured_recommendation,
    measured_summary,
    recommendation_evidence,
)
from ragbench_core.optimization.search import candidate_hash
from ragbench_core.optimization.selection import compare_trials
from ragbench_core.optimization.stages import stage_dashboard
from ragbench_core.optimization.workflow import acceptance_decision, progress, step
from test_optimization import (  # noqa: F401 -- shared fixtures
    database,
    session,
    template,
)
from test_recommendation import evidence, recommendation_tools  # noqa: F401


@pytest.mark.parametrize(
    "delta,interval,outcome",
    [
        (0.05, [0.02, 0.12], "confirmed_improvement"),
        (0.05, [-0.02, 0.12], "promising_candidate"),
        (0.05, [0, 0.12], "promising_candidate"),
        (0, [-0.02, 0.02], "baseline_retained"),
        (-0.01, [-0.03, 0.01], "baseline_retained"),
        (1e-10, [1e-12, 2e-10], "confirmed_improvement"),
    ],
)
def test_measured_export_consistent_without_model_calls(
    recommendation_tools, delta, interval, outcome
):
    tools = recommendation_tools
    comparison = deepcopy(tools.session["progress"]["comparison"])
    metric = comparison["metrics"][0]
    metric.update(candidate_mean=0.5 + delta, mean_delta=delta, ci95=interval)
    comparison["metrics"].append(
        {
            "metric": "claim_support",
            "baseline_mean": 0.4,
            "candidate_mean": 0.6,
            "mean_delta": 0.2,
            "ci95": [0.1, 0.3],
            "paired_count": metric["paired_count"],
        }
    )
    progress(
        tools.session_id,
        comparison=comparison,
        recommendation_presentation_version=PRESENTATION_VERSION,
    )
    tools.model_call = lambda *_: pytest.fail(
        "Measured recommendations must not call a model"
    )
    assert step("export", tools, lambda _: None) == "done"
    report = tools.read("report.json")
    assert report["outcome"] == outcome
    assert report["explanation"]["source"] == "measured"
    text = report["explanation"]["text"]
    assert f"The {report['recommended_source']} configuration is recommended" in text
    assert "0.5 baseline" in text
    assert "95% confidence interval" in text
    assert "Paired holdout questions" in text
    if delta == 1e-10:
        assert "+1e-10 score units" in text
        assert "+1e-12" in text
    saved = report["recommendation_evidence"]
    assert saved["metrics"][0]["delta"] == delta
    assert saved["metrics"][0]["ci95"] == interval
    assert saved["metrics"][1]["baseline"] == 0.4
    assert saved["metrics"][1]["candidate"] == 0.6
    assert saved["metrics"][2]["candidate"] is None
    # Known zero measurements are real zeros, not missing data.
    assert saved["metrics"][3]["candidate"] == 0
    assert saved["metrics"][4]["candidate"] == 0
    assert report["limitations"] == []
    assert saved["evaluation"]["split_counts"]["tuning"] > 0
    assert saved["evaluation"]["split_counts"]["holdout"] == metric["paired_count"]
    assert saved["evaluation"]["human_reviewed_questions"] == 0
    assert saved["evaluation"]["dataset_provenance"] == report["evaluation_dataset"]
    assert any(
        c["setting"] == "retriever.type" for c in saved["configuration_differences"]
    )
    assert tools.read("recommendation_evidence.json") == saved
    dashboard = stage_dashboard(tools.session)["stages"]["confirm"]
    assert dashboard["recommendation_evidence"] == saved
    assert dashboard["explanation"] == report["explanation"]
    from services.api.app.optimizations import export, recommendation

    before = deepcopy(tools.session)
    assert recommendation(tools.session_id)["recommendation_evidence"] == saved
    assert json.loads(export(tools.session_id, "report").body) == report
    assert tools.session == before
    evidence_again, explanation_again = measured_recommendation(
        tools, report["acceptance"]
    )
    assert evidence_again == saved
    assert explanation_again == report["explanation"]
    assert not (tools.root / "recommendation_explanation_prompt.json").exists()
    assert budget.totals(tools.session_id)["by_role"].get("planner") is None
    exported = yaml.safe_load((tools.root / "recommended.yaml").read_text())
    expected = (
        tools.baseline()
        if outcome == "baseline_retained"
        else tools.session["progress"]["selected_config"]
    )
    assert candidate_hash(exported) == candidate_hash(expected)


@pytest.mark.parametrize(
    "problem", ["same", "incomplete", "coverage", "incompatible", "no_confirmation"]
)
def test_measured_baseline_reasons(recommendation_tools, problem):
    tools = recommendation_tools
    comparison = deepcopy(tools.session["progress"]["comparison"])
    if problem == "same":
        progress(tools.session_id, selected_config=tools.baseline())
    elif problem == "incomplete":
        with store.db.SessionLocal.begin() as conn:
            conn.get(store.OptimizationTrial, "right").status = "failed"
    elif problem == "coverage":
        comparison["metrics"][0]["paired_count"] = 2
    elif problem == "incompatible":
        comparison["compatible"] = False
    else:
        comparison = None
    progress(tools.session_id, comparison=comparison)
    decision = acceptance_decision(tools)
    saved, explanation = measured_recommendation(tools, decision)
    assert decision["outcome"] == "baseline_retained"
    assert "Acceptance checks not passed" in explanation["text"]
    for check in decision["checks"]:
        if check["required"] and check["status"] != "pass":
            assert saved["check_details"][check["code"]] in explanation["text"]
    if problem == "same":
        assert not saved["configuration_differences"]
        assert "differs from baseline: no" in explanation["text"]
    elif problem == "incomplete":
        assert "candidate failed" in explanation["text"]
    elif problem == "coverage":
        assert "required coverage: 95%" in explanation["text"]
        assert saved["coverage"]["paired"] == 2
    else:
        assert saved["metrics"][0]["baseline"] is None
        assert "unavailable baseline" in explanation["text"]


@pytest.mark.parametrize(
    "pricing,missing", [(True, False), (False, False), (True, True)]
)
def test_operational_values_match_acceptance_and_show_actual_limits(
    recommendation_tools, monkeypatch, pricing, missing
):
    tools = recommendation_tools
    request = deepcopy(tools.session["request"])
    request["limits"].update(max_query_p95_seconds=0.5, max_query_cost=0.005)
    store.update(tools.session_id, request=request)
    with store.db.SessionLocal.begin() as conn:
        conn.get(store.OptimizationTrial, "right").summary = {"pricing_known": pricing}
    original = store.db.list_results

    def rows(trial_id):
        result = deepcopy(original(trial_id))
        for row in result:
            row["metrics"].update(
                query_latency_seconds=None if missing else 1.25,
                total_estimated_cost=0.01,
            )
        return result

    monkeypatch.setattr(store.db, "list_results", rows)
    decision = acceptance_decision(tools)
    saved = recommendation_evidence(tools, decision)
    summary = measured_summary(decision, saved)
    assert "limit: 0.5 s" in summary
    assert "limit: 0.005 USD/query" in summary
    assert "unavailable s" in summary if missing else "1.25 s" in summary
    cost = saved["metrics"][4]
    assert cost["candidate"] == (0.01 if pricing else None)
    assert cost["candidate_reason"] == (None if pricing else "unknown pricing")
    trials = store.rows(store.OptimizationTrial, tools.session_id)
    ranking = compare_trials(
        trials,
        {t["id"]: rows(t["id"]) for t in trials},
        [
            q["question_id"]
            for q in store.dataset(tools.session_id)["payload"]
            if q["split"] == "holdout"
        ],
        "answer",
        request["limits"],
    )
    candidate = next(r for r in ranking if r["trial_id"] == "right")
    assert saved["metrics"][3]["candidate"] == candidate["p95_seconds"]
    assert cost["candidate"] == candidate["cost_per_query"]


def test_evidence_context_and_redaction(recommendation_tools):
    tools = recommendation_tools
    selected = deepcopy(tools.session["progress"]["selected_config"])
    selected["llm"]["api_key"] = "sensitive-value"
    progress(
        tools.session_id,
        selected_config=selected,
        stop_reason="Active runtime budget exhausted",
    )
    # Frozen datasets may carry human review; reading evidence must preserve it.
    with store.db.SessionLocal.begin() as conn:
        row = conn.execute(store.select(store.DatasetVersion)).scalar_one()
        questions = deepcopy(row.payload)
        questions[0]["human_reviewed"] = True
        row.payload = questions
        row.source_dataset_id = row.id
    saved = recommendation_evidence(tools, acceptance_decision(tools))
    assert "sensitive-value" not in json.dumps(saved)
    assert saved["evaluation"]["human_reviewed_questions"] == 1
    assert saved["evaluation"]["dataset_provenance"]["status"] == "reused"
    assert (
        saved["evaluation"]["confirmation_stop_reason"]
        == "Active runtime budget exhausted"
    )
    json.dumps(saved, allow_nan=False)


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_measured_controls_before_and_after_reading_evidence(
    recommendation_tools, monkeypatch, status
):
    tools = recommendation_tools
    decision = acceptance_decision(tools)
    store.update(tools.session_id, status=status)
    with pytest.raises(budget.OptimizationStopped):
        measured_recommendation(tools, decision)
    store.update(tools.session_id, status="running")
    original = recommendation_evidence

    def stop_during_read(tools, decision):
        result = original(tools, decision)
        store.update(tools.session_id, status=status)
        return result

    monkeypatch.setattr(
        "ragbench_core.optimization.recommendation_metrics.recommendation_evidence",
        stop_during_read,
    )
    with pytest.raises(budget.OptimizationStopped):
        measured_recommendation(tools, decision)
    assert not (tools.root / "recommendation_evidence.json").exists()


def test_measured_export_survives_exhausted_runtime(recommendation_tools):
    tools = recommendation_tools
    progress(
        tools.session_id,
        recommendation_presentation_version=PRESENTATION_VERSION,
        stop_reason="runtime budget exhausted",
    )
    budget.reserve(tools.session_id, "spent-runtime", "benchmark", 0, 0)
    budget.settle(
        "spent-runtime", seconds=tools.session["request"]["limits"]["active_seconds"]
    )
    tools.model_call = lambda *_: pytest.fail("No model budget is needed")
    assert step("export", tools, lambda _: None) == "done"
    report = tools.read("report.json")
    assert report["status"] == "budget_exhausted"
    assert report["explanation"]["source"] == "measured"


def test_new_sessions_and_forks_get_presentation_version(
    database, template, tmp_path, monkeypatch
):
    from services.api.app import optimizations as api

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source.txt").write_text("A source document")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "test.yaml").write_text(yaml.safe_dump(template))
    monkeypatch.setattr(api, "CONFIG_ROOT", configs)
    monkeypatch.setattr(api, "DATA_ROOT", raw)
    monkeypatch.setattr(api, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(api.advance_optimization, "apply_async", lambda **_: None)
    created = api.create_optimization(OptimizationRequest(config_path="test.yaml"))
    forked = api.fork(created["id"])
    for item in (created, forked):
        assert (
            store.get(item["id"])["progress"]["recommendation_presentation_version"]
            == PRESENTATION_VERSION
        )


def test_missing_and_invalid_numbers_stay_unavailable(evidence):
    evidence["comparison"]["metrics"][0].update(
        candidate_mean=float("nan"), ci95=[float("inf"), 0]
    )
    decision = evaluate_acceptance(**evidence)
    details = {
        c["code"]: check_detail(c) for c in decision["checks"] if c["status"] != "pass"
    }
    assert "candidate unavailable" in details["correctness_available"]
    assert "[unavailable, 0]" in details["confidence_interval"]


@pytest.mark.parametrize("interval", [None, [0.1, 0.01], [float("inf"), 0.1]])
def test_invalid_intervals_are_not_presented_as_confidence_bounds(
    recommendation_tools, interval
):
    tools = recommendation_tools
    comparison = deepcopy(tools.session["progress"]["comparison"])
    comparison["metrics"][0].update(ci95=interval, candidate_mean=float("nan"))
    progress(tools.session_id, comparison=comparison)
    saved, explanation = measured_recommendation(tools, acceptance_decision(tools))
    assert saved["metrics"][0]["ci95"] is None
    assert saved["metrics"][0]["candidate"] is None
    assert "confidence interval for the change is unavailable" in explanation["text"]
    json.dumps(saved, allow_nan=False)
