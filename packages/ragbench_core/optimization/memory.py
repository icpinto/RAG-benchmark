"""Measured observations and bounded, explicitly reconstructed planner context."""

import json
from copy import deepcopy
from pathlib import Path

from ragbench_core.optimization import budget, store
from ragbench_core.optimization.adaptive_contracts import Observation
from ragbench_core.optimization.models import render_prompt
from ragbench_core.optimization.redaction import public_configuration
from ragbench_core.optimization.search import candidate_hash
from ragbench_core.optimization.selection import (
    compare_trials,
    finite,
    percentile,
    primary_metric,
)
from ragbench_core.optimization.stages import config_summary, configuration_diff

PROMPT_VERSION = "ragbench-react-v2"
MAX_INPUT_BYTES = 64 * 1024
INSTRUCTION = (
    "Choose the next RAG experiment from measured tuning evidence. Keep facts separate "
    "from interpretations. Reflect on the previous hypothesis without claiming causation "
    "from an uncontrolled comparison. Return the supplied decision schema as JSON. "
    "A run requires a nonempty testable hypothesis; a stop requires a reason. "
    "Lessons must cite observed trial IDs and be tentative, supported, or contradicted. "
    "Respect allowed_actions, fixed stages, and remaining budgets. For a run, select a "
    "candidate_id from candidates; do not compose a pipeline. Native fixed/recursive "
    "chunk sizes are characters; LlamaIndex uses native units."
)


class ContextTooLarge(ValueError):
    pass


def question_ids(session, phase):
    ids = {
        q["question_id"]
        for q in store.dataset(session["id"])["payload"]
        if q["split"] == "tuning"
    }
    if phase == "answer" and session["progress"].get("prompt_splits"):
        ids &= set(session["progress"]["prompt_splits"]["validation"])
    return ids


def configuration(config):
    summary = config_summary(config) or {}
    prompt = summary.get("prompt", {})
    if prompt.get("type") == "compiled":
        summary["prompt"] = {
            "type": "compiled",
            "artifact_id": prompt.get("artifact", {}).get("content_hash"),
        }
    else:
        summary["prompt"] = {k: v for k, v in prompt.items() if k != "template"}
    return public_configuration(summary)


def scoped_results(trial_id, ids):
    return [r for r in store.db.list_results(trial_id) if r["question_id"] in ids]


def paired_delta(current, reference, metric, ids):
    if reference is None:
        return {"delta": None, "paired_count": 0, "reason": "No reference trial"}
    left = {r["question_id"]: r["metrics"].get(metric) for r in reference}
    right = {r["question_id"]: r["metrics"].get(metric) for r in current}
    common = sorted(q for q in ids if finite(left.get(q)) and finite(right.get(q)))
    enough = bool(common) and len(common) / max(1, len(ids)) >= 0.95
    return {
        "delta": sum(right[q] - left[q] for q in common) / len(common)
        if enough
        else None,
        "paired_count": len(common),
        "reason": None if enough else "Insufficient matched coverage",
    }


def evidence_from_artifact(tools, row, phase=None):
    """Read only the retrieval trace of an already scoped question, within artifact storage."""
    path = row.get("artifact_paths", {}).get("retrieval_traces")
    if not path:
        return None
    path = Path(path).resolve()
    root = Path(tools.session["template"]["runtime"]["artifact_dir"]).resolve()
    if not path.is_relative_to(root):
        return None
    try:
        if path.stat().st_size > 2_000_000:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            "raw"
            if phase == "retrieval"
            and tools.session["progress"].get("workflow_version", 1) >= 5
            else "reranked"
        )
        return json.dumps(public_configuration(payload.get(key, [])), ensure_ascii=False)[:1800]
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def build_observation(tools, phase, trial=None, entry=None):
    session = tools.session
    trials = sorted(
        [
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["phase"] == phase
        ],
        key=lambda t: (t["created_at"], t["id"]),
    )
    config = trial["config"] if trial else entry["config"]
    reference = (
        "trial:" + trial["id"]
        if trial
        else "compilation:" + entry["compilation_job_id"]
    )
    ids = question_ids(session, phase)
    rows = scoped_results(trial["id"], ids) if trial else []
    metric = primary_metric(phase, session["progress"].get("workflow_version", 1))
    ranking = tools.compare_trials(phase)
    ranked = next((r for r in ranking if trial and r["trial_id"] == trial["id"]), None)
    previous = trials[: trials.index(trial)] if trial else trials
    previous_ranking = compare_trials(
        [t for t in previous if t["status"] == "completed"],
        {t["id"]: scoped_results(t["id"], ids) for t in previous},
        sorted(ids),
        phase,
        session["request"]["limits"],
        workflow_version=session["progress"].get("workflow_version", 1),
    )
    best_id = next((r["trial_id"] for r in previous_ranking if r["feasible"]), None)
    initial = trials[0] if trials else None
    initial_config = initial["config"] if initial else config
    valid = sum(finite(r["metrics"].get(metric)) for r in rows)
    failed = sum(r.get("case_config", {}).get("outcome") == "failed" for r in rows)
    questions = {
        q["question_id"]: q
        for q in store.dataset(tools.session_id)["payload"]
        if q["question_id"] in ids
    }
    ordered = sorted(
        rows,
        key=lambda r: (
            finite(r["metrics"].get(metric)),
            r["metrics"].get(metric) if finite(r["metrics"].get(metric)) else -1,
            r["question_id"],
        ),
    )
    examples = []
    for row in ordered[:3]:
        q = questions[row["question_id"]]
        judge = row.get("case_config", {}).get("judge") or {}
        examples.append(
            public_configuration(
                {
                    "question_id": row["question_id"],
                    "question": q["question"][:800],
                    "expected_answer": q["expected_answer"][:1200],
                    "answer": (row.get("answer") or "")[:1200],
                    "score": row["metrics"].get(metric)
                    if finite(row["metrics"].get(metric))
                    else None,
                    "evidence": [
                        {"doc_id": e["doc_id"], "quote": e["quote"][:600]}
                        for e in q["evidence"][:2]
                    ],
                    "retrieval": evidence_from_artifact(tools, row, phase),
                    "judge_feedback": json.dumps(
                        public_configuration(judge), ensure_ascii=False
                    )[:1200],
                }
            )
        )
    summary = (trial or {}).get("summary") or {}
    errors = []
    if summary.get("error_type"):
        errors.append("Execution failed: " + str(summary["error_type"])[:100])
    if (trial and trial["status"] == "failed") or not trial:
        errors.append(
            "Trial failed; inspect worker logs"
            if trial
            else "Prompt compilation failed; inspect compilation history"
        )
    result = Observation(
        reference=reference,
        phase=phase,
        trial_id=trial["id"] if trial else None,
        candidate_id=entry["candidate_id"] if entry else candidate_hash(config),
        status=trial["status"]
        if trial
        else "cancelled"
        if entry.get("compilation_status") == "cancelled"
        else "failed",
        configuration=configuration(config),
        changes=configuration_diff(
            configuration(initial_config), configuration(config)
        ),
        metric=metric,
        score=ranked["score"] if ranked else None,
        coverage={
            "expected": len(ids),
            "valid": valid,
            "failed": failed,
            "missing": max(0, len(ids) - valid - failed),
        },
        ranking=ranked,
        comparisons={
            "initial": {
                "trial_id": initial["id"] if initial else None,
                **paired_delta(
                    rows,
                    scoped_results(initial["id"], ids) if initial else None,
                    metric,
                    ids,
                ),
            },
            "best_previous": {
                "trial_id": best_id,
                **paired_delta(
                    rows, scoped_results(best_id, ids) if best_id else None, metric, ids
                ),
            },
        },
        errors=errors,
        examples=examples,
    ).model_dump()
    result["operational"] = {
        "p95_seconds": percentile(
            [r["metrics"].get("query_latency_seconds") for r in rows]
        ),
        "cost_per_query": (
            sum(r["metrics"]["total_estimated_cost"] for r in rows) / len(rows)
            if rows
            and summary.get("pricing_known")
            and all(finite(r["metrics"].get("total_estimated_cost")) for r in rows)
            else None
        ),
    }
    return public_configuration(result)


def iterations(session_id, phase=None):
    return sorted(
        [
            r
            for r in store.rows(store.OptimizationIteration, session_id)
            if phase is None or r["phase"] == phase
        ],
        key=lambda r: (r["created_at"], r["phase"], r["sequence"]),
    )


def iteration_summary(tools, item):
    observation = tools.read(item["observation_artifact"])
    decision = item["decision"]
    return public_configuration(
        {
            "id": item["id"],
            "phase": item["phase"],
            "sequence": item["sequence"],
            "status": item["status"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "observation": {
                k: observation[k]
                for k in (
                    "reference",
                    "trial_id",
                    "candidate_id",
                    "status",
                    "metric",
                    "score",
                    "coverage",
                    "comparisons",
                    "errors",
                )
            },
            "decision": {k: v for k, v in decision.items() if k != "pipeline"}
            if decision
            else None,
            "attempts": [
                {"id": a["id"], "status": a["status"], "error": a.get("error")}
                for a in item["attempts"]
            ],
        }
    )


def iteration_detail(tools, item):
    return public_configuration(
        {
            **iteration_summary(tools, item),
            "prompt_version": PROMPT_VERSION,
            "observation": tools.read(item["observation_artifact"]),
            "decision": item["decision"],
            "attempts": [
                {
                    **attempt,
                    "prompt": tools.read(attempt["prompt_artifact"])
                    if attempt.get("prompt_artifact")
                    else None,
                    "response": tools.read(attempt["response_artifact"])
                    if attempt.get("response_artifact")
                    else None,
                }
                for attempt in item["attempts"]
            ],
        }
    )


def build_context(
    tools,
    iteration,
    observation,
    catalog,
    schema,
    *,
    force_stop=False,
    feedback=None,
    attempt_id="",
):
    phase = iteration["phase"]
    session = tools.session
    history, hypotheses, rejected, details = [], [], [], []
    for item in iterations(tools.session_id):
        observed = tools.read(item["observation_artifact"])
        history.append(
            {
                k: observed[k]
                for k in (
                    "reference",
                    "phase",
                    "trial_id",
                    "candidate_id",
                    "status",
                    "metric",
                    "score",
                )
            }
        )
        if item["decision"] and item["phase"] == phase:
            hypotheses.append({"iteration_id": item["id"], **item["decision"]})
        for attempt in item["attempts"]:
            if attempt.get("error"):
                rejected.append(
                    {"attempt_id": attempt["id"], "error": attempt["error"]}
                )
        if item["phase"] == phase and item["id"] != iteration["id"]:
            details.append({k: v for k, v in observed.items() if k != "examples"})
    phase_trials = [
        t
        for t in store.rows(store.OptimizationTrial, tools.session_id)
        if t["phase"] == phase
    ]
    ranking = tools.compare_trials(phase)
    best = next((r for r in ranking if r["feasible"]), None)
    best_trial = next(
        (t for t in phase_trials if best and t["id"] == best["trial_id"]), None
    )
    initial = (
        min(phase_trials, key=lambda t: (t["created_at"], t["id"]))
        if phase_trials
        else None
    )
    limits, used = session["request"]["limits"], budget.totals(tools.session_id)
    maximum = limits["retrieval_trials" if phase == "retrieval" else "answer_trials"]
    data = {
        "prompt_version": PROMPT_VERSION,
        "attempt_id": attempt_id,
        "objective": {
            "audience": session["request"]["audience"],
            "task": session["request"]["task"],
            "metric": observation["metric"],
            "phase": phase,
        },
        "rules": {
            "allowed_actions": ["stop"] if force_stop else ["run", "stop"],
            "fixed": "Answer model and generation; source data; filters; extraction and cleaning. "
            + (
                "Compression, context and prompts. "
                + (
                    "Reranking is disabled; use only the none reranker."
                    if session["progress"].get("workflow_version", 1) >= 5
                    else ""
                )
                if phase == "retrieval"
                else "Choose only an available candidate ID; no pipeline composition."
            ),
            "response_schema": schema,
        },
        "current_observation": deepcopy(observation),
        "initial_configuration": configuration(initial["config"]) if initial else None,
        "best_feasible": {
            "trial_id": best_trial["id"],
            "score": best["score"],
            "configuration": configuration(best_trial["config"]),
        }
        if best_trial
        else None,
        "history": history,
        "rejected_proposals": rejected,
        "recent_observations": details[-3:],
        "previous_hypotheses": hypotheses[-10:],
        "memory": deepcopy(session["progress"].get("planner_memory", {"lessons": []})),
        "remaining": {
            "trials": max(0, maximum - len(phase_trials)),
            "tokens": max(0, limits["max_tokens"] - used["tokens"]),
            "seconds": max(
                0,
                limits["active_seconds"]
                - used["active_seconds"]
                - used["reserved_seconds"],
            ),
            "cost": None
            if limits["max_cost"] is None or used["cost"] is None
            else max(0, limits["max_cost"] - used["cost"]),
        },
        "candidates": deepcopy(catalog),
        "corrective_feedback": feedback,
    }
    if phase == "retrieval":
        data["search_space"] = public_configuration(tools.get_search_space("retrieval"))
        data["baseline_pipeline"] = public_configuration(
            tools.retrieval_config()["pipeline"]
        )
    data = public_configuration(data)

    def fits():
        return len(render_prompt(INSTRUCTION, data).encode("utf-8")) <= MAX_INPUT_BYTES

    # Full observations remain in artifacts. Only optional detail is removed here.
    while not fits() and data["current_observation"]["examples"]:
        data["current_observation"]["examples"].pop()
    while not fits() and data["memory"]["lessons"]:
        data["memory"]["lessons"].pop()
    while not fits() and data["recent_observations"]:
        data["recent_observations"].pop(0)
    while not fits() and data["previous_hypotheses"]:
        data["previous_hypotheses"].pop(0)
    if not fits():
        raise ContextTooLarge(
            "Mandatory planner context exceeds 64 KiB; reduce task guidance or candidate/search-space size"
        )
    return data
