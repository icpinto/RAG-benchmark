"""Pair explicitly selected cases by question identity on the same frozen dataset."""

from __future__ import annotations

import math
import random

from ragbench_core.engine.persistence import stable_hash


def numeric(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def paired_comparison(
    baseline: dict,
    candidate: dict,
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    baseline_case_id: str,
    candidate_case_id: str,
) -> dict:
    result = {
        "baseline_run_id": baseline["id"],
        "run_id": candidate["id"],
        "baseline_case_id": baseline_case_id,
        "candidate_case_id": candidate_case_id,
        "compatible": bool(baseline.get("dataset_hash"))
        and baseline.get("dataset_hash") == candidate.get("dataset_hash"),
        "metrics": [],
        "deltas": [],
    }
    if not result["compatible"]:
        result["reason"] = "Dataset snapshots differ or dataset identity is unavailable"
        return result
    left_rows = [r for r in baseline_rows if r["case_id"] == baseline_case_id]
    right_rows = [r for r in candidate_rows if r["case_id"] == candidate_case_id]

    def lookup(rows):
        values = {}
        for row in rows:
            question_id = row.get("question_id") or row.get("case_config", {}).get(
                "question_id"
            )
            if not question_id or question_id in values:
                raise ValueError(
                    "Paired comparison requires unique stable question IDs; rerun historical experiments"
                )
            values[question_id] = row
        return values

    try:
        left, right = lookup(left_rows), lookup(right_rows)
    except ValueError as exc:
        result.update(compatible=False, reason=str(exc))
        return result
    question_ids = sorted(
        set(left) | set(right) | set(baseline.get("question_ids", []))
    )
    metric_names = sorted(
        {
            key
            for r in left_rows + right_rows
            for key, value in (r.get("metrics") or {}).items()
            if value is None or numeric(value)
        }
    )
    for metric in metric_names:
        pairs = []
        for question_id in question_ids:
            a = (left.get(question_id, {}).get("metrics") or {}).get(metric)
            b = (right.get(question_id, {}).get("metrics") or {}).get(metric)
            if numeric(a) and numeric(b):
                delta = b - a
                pairs.append((a, b, delta))
                result["deltas"].append(
                    {
                        "question_id": question_id,
                        "metric": metric,
                        "delta": delta,
                        "baseline_run_id": baseline["id"],
                        "run_id": candidate["id"],
                        "baseline_case_id": baseline_case_id,
                        "candidate_case_id": candidate_case_id,
                    }
                )
        deltas = [p[2] for p in pairs]
        interval = None
        if len(pairs) >= 2:
            rng = random.Random(stable_hash({"metric": metric, "pairs": pairs}))
            samples = sorted(
                sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                for _ in range(2000)
            )
            interval = [samples[49], samples[1949]]
        result["metrics"].append(
            {
                "metric": metric,
                "paired_count": len(pairs),
                "missing_count": len(question_ids) - len(pairs),
                "baseline_missing": sum(
                    not numeric((r.get("metrics") or {}).get(metric))
                    for r in [left.get(q, {}) for q in question_ids]
                ),
                "candidate_missing": sum(
                    not numeric((r.get("metrics") or {}).get(metric))
                    for r in [right.get(q, {}) for q in question_ids]
                ),
                "baseline_mean": sum(p[0] for p in pairs) / len(pairs)
                if pairs
                else None,
                "candidate_mean": sum(p[1] for p in pairs) / len(pairs)
                if pairs
                else None,
                "mean_delta": sum(deltas) / len(deltas) if pairs else None,
                "ci95": interval,
                "kind": "heuristic"
                if "placeholder" in metric or metric == "citation_coverage"
                else "semantic"
                if metric in {"answer_correctness", "claim_support", "citation_support"}
                else "operational_or_retrieval",
            }
        )
    return result
