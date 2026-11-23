"""Deterministic scoring; generated prose never decides the winner."""

from __future__ import annotations

import math
from functools import cmp_to_key


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def percentile(values, fraction=0.95):
    values = sorted(v for v in values if finite(v))
    return (
        values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def primary_metric(phase, workflow_version=1):
    if phase != "retrieval":
        return "answer_correctness"
    return "raw_retrieval_ndcg@10" if workflow_version >= 5 else "reranked_ndcg@10"


def operational_metrics(trial, rows):
    """Use the same observed latency and known-price cost in selection and reports."""
    p95 = percentile([r["metrics"].get("query_latency_seconds") for r in rows])
    costs = [r["metrics"].get("total_estimated_cost") for r in rows]
    known_cost = (
        all(finite(v) for v in costs)
        and bool(costs)
        and trial.get("summary", {}).get("pricing_known", False)
    )
    return {
        "p95_seconds": p95,
        "cost_per_query": sum(costs) / len(costs) if known_cost else None,
    }


def compare_trials(
    trials, results_by_trial, question_ids, phase, limits, *, workflow_version=1
):
    metric = primary_metric(phase, workflow_version)
    by_trial = {
        t["id"]: {r["question_id"]: r for r in results_by_trial.get(t["id"], [])}
        for t in trials
    }
    eligible = []
    for trial in trials:
        rows = by_trial[trial["id"]]
        valid = {
            q
            for q in question_ids
            if finite(rows.get(q, {}).get("metrics", {}).get(metric))
        }
        if len(valid) / max(1, len(question_ids)) >= 0.95:
            eligible.append((trial, valid))
    # All ranked trials use the same observations to avoid missing-score selection bias.
    common = set(question_ids)
    for _, valid in eligible:
        common &= valid
    if len(common) / max(1, len(question_ids)) < 0.95:
        return []
    ranked = []
    for trial, valid in eligible:
        rows = by_trial[trial["id"]]
        scores = [rows[q]["metrics"][metric] for q in sorted(common)]
        operational = operational_metrics(trial, rows.values())
        p95, cost = operational["p95_seconds"], operational["cost_per_query"]
        violations = []
        if limits.get("max_query_p95_seconds") is not None and (
            p95 is None or p95 > limits["max_query_p95_seconds"]
        ):
            violations.append("query latency")
        if limits.get("max_query_cost") is not None and (
            cost is None or cost > limits["max_query_cost"]
        ):
            violations.append("query cost")
        recall_metric = (
            "raw_retrieval_recall@10"
            if phase == "retrieval" and workflow_version >= 5
            else "reranked_recall@10"
        )
        recall = [rows[q]["metrics"].get(recall_metric) for q in common]
        ranked.append(
            {
                "trial_id": trial["id"],
                "score": sum(scores) / len(scores),
                "metric": metric,
                "paired_count": len(common),
                "missing_count": len(question_ids) - len(valid),
                "p95_seconds": p95,
                "cost_per_query": cost,
                "feasible": not violations,
                "violations": violations,
                "recall": sum(v for v in recall if finite(v)) / len(recall)
                if all(finite(v) for v in recall)
                else None,
            }
        )

    def compare(a, b):
        if a["feasible"] != b["feasible"]:
            return -1 if a["feasible"] else 1
        tolerance = 0 if phase == "retrieval" else 0.01
        if abs(a["score"] - b["score"]) >= max(tolerance, 1e-12):
            return -1 if a["score"] > b["score"] else 1
        for key in (["recall"] if phase == "retrieval" else []) + [
            "p95_seconds",
            "cost_per_query",
        ]:
            left, right = a[key], b[key]
            if left != right:
                if left is None or right is None:
                    return 1 if left is None else -1
                return (
                    (-1 if left > right else 1)
                    if key == "recall"
                    else (-1 if left < right else 1)
                )
        return (
            -1 if a["trial_id"] < b["trial_id"] else int(a["trial_id"] != b["trial_id"])
        )

    return sorted(ranked, key=cmp_to_key(compare))
