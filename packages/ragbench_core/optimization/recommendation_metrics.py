"""Versioned recommendation presentation built only from persisted measurements."""

from __future__ import annotations

from copy import deepcopy

from ragbench_core.optimization import budget, store
from ragbench_core.optimization.recommendation import finite_json
from ragbench_core.optimization.redaction import public_configuration
from ragbench_core.optimization.selection import finite, operational_metrics

PRESENTATION_VERSION = 1


def number(value, *, signed=False):
    if not finite(value):
        return "unavailable"
    return format(value, "+.6g" if signed and value != 0 else ".6g")


def check_detail(check):
    """Describe a check using its recorded measurement and actual threshold."""
    code, value, threshold = check["code"], check["measured"], check["threshold"]
    if code == "paired_coverage":
        value = value or {}
        fraction = value.get("fraction")
        coverage = f"{number(fraction * 100)}%" if finite(fraction) else "unavailable"
        return (
            f"Paired correctness scores: {number(value.get('paired'))} / "
            f"{number(value.get('expected'))} ({coverage}); "
            f"required coverage: {number(threshold * 100)}%."
        )
    if code in {"query_latency", "query_cost"}:
        label, unit = (
            ("Candidate p95 query latency", "s")
            if code == "query_latency"
            else ("Candidate estimated cost per query", "USD/query")
        )
        limit = (
            f"{number(threshold)} {unit}" if threshold is not None else "not configured"
        )
        return f"{label}: {number(value)} {unit}; limit: {limit}."
    if code in {"correctness_improved", "statistical_confirmation"}:
        label = (
            "Holdout correctness score change"
            if code == "correctness_improved"
            else "Lower bound of the 95% correctness interval"
        )
        return (
            f"{label}: {number(value, signed=True)}; required: > {number(threshold)}."
        )
    if code == "trials_completed":
        value = value or {}
        return (
            f"Holdout trials: baseline {value.get('baseline') or 'unavailable'}, "
            f"candidate {value.get('candidate') or 'unavailable'}; "
            "both must be completed and match the tested configurations."
        )
    if code == "different_configuration":
        return "Candidate configuration differs from baseline: " + (
            "yes." if value else "no."
        )
    if code == "datasets_compatible":
        status = "yes" if value is True else "no" if value is False else "unavailable"
        return f"Compatible holdout datasets: {status}; required: yes."
    if code == "correctness_available":
        value = value or {}
        return (
            f"Holdout correctness scores: baseline {number(value.get('baseline_mean'))}, "
            f"candidate {number(value.get('candidate_mean'))}, "
            f"change {number(value.get('mean_delta'), signed=True)}; finite scores required."
        )
    if code == "confidence_interval":
        bounds = (
            "[" + ", ".join(number(v, signed=True) for v in value) + "]"
            if isinstance(value, (list, tuple))
            else "unavailable"
        )
        return f"95% correctness interval: {bounds}; finite, ordered bounds required."
    if code == "candidate_eligible":
        return "Candidate has sufficient shared correctness scores: " + (
            "yes." if value else "no."
        )
    return check["label"]


def recommendation_evidence(tools, decision):
    # Local import avoids a cycle: stage summaries use the acceptance policy.
    from ragbench_core.optimization.stages import config_summary, configuration_diff

    state = tools.session["progress"]
    comparison = state.get("comparison") or {}
    trials = {
        t["id"]: t
        for t in store.rows(store.OptimizationTrial, tools.session_id)
        if t["phase"] == "confirm"
    }
    metrics = []
    for key, label in (
        ("answer_correctness", "Correctness score"),
        ("claim_support", "Claim support score"),
        ("citation_support", "Citation support score"),
    ):
        measured = next(
            (m for m in comparison.get("metrics", []) if m["metric"] == key), {}
        )
        if comparison.get("compatible") is not True:
            measured = {}
        baseline, candidate = (
            measured.get("baseline_mean"),
            measured.get("candidate_mean"),
        )
        interval = measured.get("ci95")
        if not (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and all(finite(v) for v in interval)
            and interval[0] <= interval[1]
        ):
            interval = None
        metrics.append(
            {
                "metric": key,
                "label": label,
                "unit": "score",
                "aggregation": "paired_mean",
                "baseline": baseline if finite(baseline) else None,
                "candidate": candidate if finite(candidate) else None,
                "delta": measured.get("mean_delta")
                if finite(measured.get("mean_delta"))
                else None,
                "ci95": interval,
                "paired_count": measured.get("paired_count"),
                "baseline_reason": None
                if finite(baseline)
                else "paired scores unavailable",
                "candidate_reason": None
                if finite(candidate)
                else "paired scores unavailable",
            }
        )
    operational = {}
    for side, id_key in (("baseline", "baseline_run_id"), ("candidate", "run_id")):
        trial = trials.get(comparison.get(id_key))
        rows = store.db.list_results(trial["id"]) if trial else []
        # Match selection's stable question-ID lookup, including its deduplication.
        rows = list({r["question_id"]: r for r in rows}.values())
        operational[side] = (trial, operational_metrics(trial or {}, rows))
    for key, label, unit, aggregation in (
        ("p95_seconds", "p95 query latency", "seconds", "p95"),
        ("cost_per_query", "Estimated cost per query", "USD/query", "mean"),
    ):
        item = {
            "metric": key,
            "label": label,
            "unit": unit,
            "aggregation": aggregation,
            "ci95": None,
            "paired_count": None,
        }
        for side, (trial, values) in operational.items():
            item[side] = values[key]
            item[f"{side}_reason"] = (
                "confirmation trial unavailable"
                if trial is None
                else "unknown pricing"
                if key == "cost_per_query"
                and not trial.get("summary", {}).get("pricing_known", False)
                else "not measured"
                if values[key] is None
                else None
            )
        item["delta"] = (
            item["candidate"] - item["baseline"]
            if finite(item["baseline"]) and finite(item["candidate"])
            else None
        )
        metrics.append(item)
    questions = (store.dataset(tools.session_id) or {}).get("payload", [])
    coverage = next(
        (c["measured"] for c in decision["checks"] if c["code"] == "paired_coverage"),
        {},
    )
    threshold = next(
        (c["threshold"] for c in decision["checks"] if c["code"] == "paired_coverage"),
        None,
    )
    baseline = public_configuration(config_summary(tools.baseline()))
    selected = state.get("selected_config")
    candidate = public_configuration(config_summary(selected)) if selected else None
    return finite_json(
        {
            "version": PRESENTATION_VERSION,
            "baseline_trial_id": comparison.get("baseline_run_id"),
            "candidate_trial_id": comparison.get("run_id"),
            "metrics": metrics,
            "coverage": {**(coverage or {}), "required_fraction": threshold},
            "check_details": {
                c["code"]: check_detail(c)
                for c in decision["checks"]
                if c["status"] != "pass"
            },
            "configuration_differences": configuration_diff(baseline, candidate)
            if candidate
            else [],
            "evaluation": {
                "synthetic_data": True,
                "dataset_provenance": store.dataset_provenance(tools.session_id),
                "split_counts": {
                    split: sum(q.get("split") == split for q in questions)
                    for split in ("tuning", "holdout")
                },
                "human_reviewed_questions": sum(
                    bool(q.get("human_reviewed")) for q in questions
                ),
                "confirmation_stop_reason": state.get("stop_reason"),
            },
        }
    )


def measured_summary(decision, evidence):
    correctness = evidence["metrics"][0]
    text = f"The {decision['recommended_source']} configuration is recommended. "
    text += (
        f"Holdout correctness score: {number(correctness['baseline'])} baseline → "
        f"{number(correctness['candidate'])} candidate "
        f"(change {number(correctness['delta'], signed=True)} score units). "
    )
    interval = correctness["ci95"]
    if (
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and all(finite(v) for v in interval)
        and interval[0] <= interval[1]
    ):
        text += f"95% confidence interval for the change: [{number(interval[0], signed=True)}, {number(interval[1], signed=True)}]. "
    else:
        text += "The 95% confidence interval for the change is unavailable. "
    if decision["outcome"] == "confirmed_improvement":
        text += "Improvement is statistically confirmed in this evaluation. "
    elif decision["outcome"] == "promising_candidate":
        text += "Improvement is promising but not statistically confirmed because the interval does not lie entirely above zero. "
    coverage = evidence["coverage"]
    fraction = coverage.get("fraction")
    coverage_text = f" ({number(fraction * 100)}%)" if finite(fraction) else ""
    text += (
        f"Paired holdout questions: {number(coverage.get('paired'))} / {number(coverage.get('expected'))}{coverage_text}; "
        f"required coverage: {number(coverage['required_fraction'] * 100)}%."
    )
    failures = [
        c for c in decision["checks"] if c["required"] and c["status"] != "pass"
    ]
    if failures:
        text += " Acceptance checks not passed: " + " ".join(
            evidence["check_details"][c["code"]] for c in failures
        )
    return text


def measured_recommendation(tools, decision):
    """No provider call; budget exhaustion never hides measurements or stops export."""

    def check_control():
        try:
            budget.check_control(tools.session_id)
        except budget.BudgetExhausted:
            pass

    check_control()
    if (tools.root / "recommendation_evidence.json").exists():
        evidence = tools.read("recommendation_evidence.json")
    else:
        evidence = recommendation_evidence(tools, decision)
    explanation = {
        "text": measured_summary(decision, evidence),
        "source": "measured",
        "prompt_version": f"recommendation-measured-v{PRESENTATION_VERSION}",
    }
    check_control()
    tools.write("recommendation_evidence.json", evidence)
    tools.write("recommendation_explanation.json", explanation)
    return deepcopy(evidence), explanation
