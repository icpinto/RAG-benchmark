"""Read-only stage summaries. Metrics never depend on a trial being ranked."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from sqlalchemy import select

from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.optimization import store
from ragbench_core.optimization.candidates import retrieval_identity, screen_config
from ragbench_core.optimization.recommendation import (
    POLICY_VERSION,
    evaluate_acceptance,
)
from ragbench_core.optimization.search import candidate_hash, concrete_config
from ragbench_core.optimization.selection import (
    compare_trials,
    finite,
    percentile,
    primary_metric,
)

PHASES = ("retrieval", "answer", "confirm")
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
RETRIEVAL_METRICS = [
    f"{stage}_{name}@{k}"
    for stage in ("raw_retrieval", "reranked")
    for name in ("ndcg", "recall", "precision", "mrr")
    for k in (3, 5, 10)
]
ANSWER_METRICS = ["answer_correctness", "claim_support", "citation_support"]
OPERATIONAL_METRICS = [
    "query_latency_seconds",
    "retrieval_seconds",
    "reranking_seconds",
    "generation_seconds",
    "chunk_count",
    "context_chunks",
    "context_chars",
    "context_tokens",
    "prompt_tokens",
    "completion_tokens",
    "embedding_tokens",
    "total_estimated_cost",
]


def config_summary(config):
    try:
        case = expand_cases(config)[0]
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    ing, query = case.ingestion_config, case.query_config
    chunker = deepcopy(ing["chunker"])
    # Hierarchical and semantic fallbacks may use character units; don't label all
    # LlamaIndex implementations as token splitters.
    units = (
        "characters"
        if chunker["type"] in {"fixed", "recursive", "llamaindex_structure"}
        else "implementation-native units"
    )
    if chunker["type"] in {"llamaindex_fixed", "llamaindex_sentence"}:
        units = "tokens (characters when using fallback)"
    return {
        "chunker": chunker,
        "chunk_units": units,
        "embedding": ing["embedder"],
        "indexers": ing["indexers"],
        "rewriter": query["rewriter"],
        "retriever": query["retriever"],
        "filters": query["filters"],
        "reranker": query["reranker"],
        "compressor": query["compressor"],
        "context": query["context_assembler"],
        "prompt": query["prompt_builder"],
        "llm": case.llm_config,
    }


def configuration_diff(left, right, path=""):
    changes = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            changes.extend(
                configuration_diff(
                    left.get(key), right.get(key), f"{path}.{key}" if path else key
                )
            )
    elif left != right:
        changes.append({"setting": path, "baseline": left, "candidate": right})
    return changes


def _hash(config, core=False):
    try:
        return retrieval_identity(config) if core else candidate_hash(config)
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def _source(trial, trials, state):
    if trial["phase"] == "retrieval":
        return None
    retrievals = [t for t in trials if t["phase"] == "retrieval"]
    known_ids = {t["id"] for t in retrievals}
    entry = next(
        (
            entry
            for entry in state.get("answer_plan", [])
            if entry["candidate_hash"] == _hash(trial["config"])
        ),
        None,
    )
    if entry is not None:
        return (
            entry.get("source_trial_id")
            if entry.get("source_trial_id") in known_ids
            else None
        )
    identity = _hash(trial["config"], core=True)
    matches = [
        t["id"]
        for t in retrievals
        if identity is not None and _hash(t["config"], core=True) == identity
    ]
    return matches[0] if len(matches) == 1 else None


def _metrics(trial, rows, question_ids):
    pending = trial["status"] in {"queued", "running", "paused"}
    names = set(RETRIEVAL_METRICS + OPERATIONAL_METRICS)
    if trial["phase"] != "retrieval":
        names.update(ANSWER_METRICS)
    names.update(
        key
        for row in rows
        for key, value in (row.get("metrics") or {}).items()
        if value is None or finite(value)
    )
    metrics = {}
    for name in sorted(names):
        values = [(row.get("metrics") or {}).get(name) for row in rows]
        values = [value for value in values if finite(value)]
        reason = None
        if "cost" in name and not trial.get("summary", {}).get("pricing_known", False):
            values, reason = [], "unknown pricing"
        elif not values:
            reason = (
                "pending"
                if pending
                else "missing historical data"
                if not rows
                else "not measured"
            )
        metrics[name] = {
            "mean": sum(values) / len(values) if values else None,
            "valid_count": len(values),
            "missing_count": max(0, len(question_ids) - len(values)),
            "reason": reason,
        }
    return metrics


def stage_dashboard(session, *, redact=lambda value: value):
    """A bounded set of SELECTs; no graph invocation, provider calls, or writes."""
    session_id, state = session["id"], session.get("progress") or {}
    with store.db.SessionLocal() as conn:
        trials = [
            store.payload(t)
            for t in conn.execute(
                select(store.OptimizationTrial)
                .where(store.OptimizationTrial.session_id == session_id)
                .order_by(
                    store.OptimizationTrial.created_at, store.OptimizationTrial.id
                )
            ).scalars()
        ]
        ids = [t["id"] for t in trials]
        runs = (
            {
                r.id: {"status": r.status, "summary": r.summary or {}}
                for r in conn.execute(
                    select(
                        store.db.BenchmarkRun.id,
                        store.db.BenchmarkRun.status,
                        store.db.BenchmarkRun.summary,
                    ).where(store.db.BenchmarkRun.id.in_(ids))
                )
            }
            if ids
            else {}
        )
        results = defaultdict(list)
        if ids:
            for row in conn.execute(
                select(
                    store.db.BenchmarkResult.run_id,
                    store.db.BenchmarkResult.question_id,
                    store.db.BenchmarkResult.metrics,
                    store.db.BenchmarkResult.case_config["outcome"]
                    .as_string()
                    .label("outcome"),
                ).where(store.db.BenchmarkResult.run_id.in_(ids))
            ):
                results[row.run_id].append(
                    {
                        "question_id": row.question_id,
                        "metrics": row.metrics or {},
                        "failed": row.outcome == "failed",
                    }
                )
        dataset = conn.execute(
            select(store.DatasetVersion).where(
                store.DatasetVersion.session_id == session_id
            )
        ).scalar_one_or_none()
        payload = deepcopy(dataset.payload) if dataset else []

    baseline = state.get("baseline_config") or next(
        iter(state.get("answer_candidates") or []), None
    )
    if baseline is None:
        baseline = next(
            (t["config"] for t in trials if t["phase"] == "retrieval"), None
        )
    if baseline is None:
        try:
            baseline = concrete_config(
                session["template"], expand_cases(session["template"])[0]
            )
        except (ValueError, KeyError, TypeError, IndexError):
            baseline = {}
    baseline_hash = _hash(baseline)
    baseline_summary = redact(config_summary(baseline))
    selected_id = state.get("selected_trial_id")
    selected_config = state.get("selected_config")
    if not selected_id and selected_config:
        matches = [
            t["id"]
            for t in trials
            if t["phase"] == "answer" and _hash(t["config"]) == _hash(selected_config)
        ]
        selected_id = matches[0] if len(matches) == 1 else None
    sources = {t["id"]: _source(t, trials, state) for t in trials}
    shortlist = state.get("retrieval_shortlist")
    if shortlist is None:
        shortlist = [
            {"trial_id": source, "reason": "Matched saved answer-stage configuration"}
            for source in sorted(
                {
                    sources[t["id"]]
                    for t in trials
                    if t["phase"] == "answer" and sources[t["id"]]
                }
            )
        ]
    shortlist_ids = {item["trial_id"] for item in shortlist}
    shortlist_decided = (
        "retrieval_shortlist" in state
        or "answer_candidates" in state
        or session["stage"] in {"answer", "confirm", "export", "done"}
    )
    sections = {}
    phases = ("retrieval", "answer", "prompt", "confirm") if state.get("workflow_version", 1) >= 6 else PHASES
    for phase in phases:
        split = "holdout" if phase == "confirm" else "tuning"
        phase_trials = [t for t in trials if t["phase"] == phase]
        if phase == "prompt":
            control_id = state.get("prompt_control_trial_id") or state.get("answer_winner_trial_id")
            phase_trials += [t for t in trials if t["id"] == control_id and t["phase"] != "prompt"]
        question_ids = [q["question_id"] for q in payload if q.get("split") == split]
        if phase in {"answer", "prompt"} and state.get("prompt_splits"):
            question_ids = state["prompt_splits"]["validation"]
            split = "prompt validation"
        if not question_ids:
            question_ids = sorted(
                {
                    q
                    for t in phase_trials
                    for q in t["config"].get("_snapshot", {}).get("question_ids", [])
                }
            )
        question_set = set(question_ids)
        # Completed trials can have descriptive metrics before the workflow saves
        # a final ranking. Reuse the same ranking policy, on this split only.
        completed = [t for t in phase_trials if t["status"] == "completed"]
        ranking = (
            compare_trials(
                completed,
                results,
                question_ids,
                "answer" if phase == "confirm" else phase,
                session["request"]["limits"],
                workflow_version=state.get("workflow_version", 1),
            )
            if question_ids
            else []
        )
        ranks = {r["trial_id"]: {**r, "rank": i + 1} for i, r in enumerate(ranking)}
        output = []
        for trial in phase_trials:
            rows = results[trial["id"]]
            if question_ids:
                rows = [r for r in rows if r["question_id"] in question_set]
            metrics = _metrics(trial, rows, question_ids)
            primary = primary_metric(phase, state.get("workflow_version", 1))
            valid = metrics[primary]["valid_count"]
            ranked = ranks.get(trial["id"])
            unavailable = None
            if not ranked:
                unavailable = (
                    "pending"
                    if trial["status"] in {"queued", "running", "paused"}
                    else "missing historical data"
                    if not question_ids or not rows
                    else "insufficient coverage"
                    if valid / max(1, len(question_ids)) < 0.95
                    else "insufficient shared coverage"
                )
            roles, reasons = [], []
            if phase == "prompt":
                if trial["id"] == state.get("prompt_control_trial_id"):
                    roles.append("unchanged answer winner")
                    reasons.append("Control reused on the identical prompt-validation partition")
                else:
                    reasons.append("Compiled prompt evaluated against the unchanged answer winner")
            if phase == "answer":
                entry = next(
                    (
                        e
                        for e in state.get("answer_plan", [])
                        if e["candidate_hash"] == _hash(trial["config"])
                    ),
                    None,
                )
                if entry and entry.get("reason"):
                    reasons.append(entry["reason"])
            control_hash = (
                _hash(screen_config(baseline, baseline, state["workflow_version"]))
                if baseline and phase == "retrieval" and state.get("workflow_version", 1) >= 5
                else baseline_hash
            )
            if _hash(trial["config"]) == control_hash:
                roles.append("baseline")
            if trial["id"] in shortlist_ids:
                roles.append("shortlisted")
                reasons.extend(
                    item["reason"]
                    for item in shortlist
                    if item["trial_id"] == trial["id"]
                )
            if trial["id"] == selected_id:
                roles.append("selected for holdout")
                reasons.append(
                    "Best feasible complete pipeline on matched tuning questions"
                )
            if phase == "answer" and trial["id"] == state.get("answer_winner_trial_id"):
                roles.append("answer stage winner")
            if (
                phase == "confirm"
                and selected_config
                and _hash(trial["config"]) == _hash(selected_config)
            ):
                roles.append("holdout candidate")
            if ranked and not ranked["feasible"]:
                roles.append("excluded")
                reasons.extend(f"Exceeds {name} limit" for name in ranked["violations"])
            elif unavailable and trial["status"] not in {"queued", "running", "paused"}:
                roles.append("excluded")
                reasons.append(unavailable)
            elif not roles and phase == "retrieval" and shortlist is not None:
                reasons.append(
                    "Not selected as an answering parent: lower rank or saved handoff policy"
                    if shortlist_decided
                    else "Awaiting retrieval shortlist selection"
                )
            if not reasons:
                reasons.append(
                    "Unchanged baseline control"
                    if "baseline" in roles
                    else "Frozen holdout evaluation; no tuning occurs here"
                    if phase == "confirm"
                    else "Not selected: another complete pipeline ranked higher or won a tiebreak"
                    if phase == "answer" and selected_config
                    else "Awaiting completion of answer trials"
                    if phase == "answer"
                    else "Awaiting retrieval shortlist selection"
                )
            failed = sum(r["failed"] for r in rows)
            summary = (
                runs.get(trial["id"], {}).get("summary") or trial.get("summary") or {}
            )
            errors = max(failed, summary.get("errors", 0))
            failed = min(
                len(question_ids),
                max(
                    failed,
                    summary.get("aggregate", {})
                    .get("result_counts", {})
                    .get("failed", 0),
                ),
            )
            config = redact(config_summary(trial["config"]))
            output.append(
                {
                    "id": trial["id"],
                    "phase": trial["phase"],
                    "split": split,
                    "status": trial["status"],
                    "display_status": "Completed with errors"
                    if trial["status"] == "completed"
                    and (
                        errors
                        or runs.get(trial["id"], {}).get("status")
                        == "completed_with_errors"
                    )
                    else trial["status"].replace("_", " ").capitalize(),
                    "metrics": metrics,
                    "ranking": ranked,
                    "score_reason": unavailable,
                    "p95_seconds": percentile(
                        [r["metrics"].get("query_latency_seconds") for r in rows]
                    ),
                    "coverage": {
                        "expected": len(question_ids),
                        "valid": valid,
                        "missing": max(0, len(question_ids) - valid - failed),
                        "failed": failed,
                    },
                    "roles": roles,
                    "selection_reasons": reasons,
                    "source_trial_id": sources[trial["id"]],
                    "source_answer_trial_id": state.get("answer_winner_trial_id") if phase == "prompt" else selected_id
                    if phase == "confirm" and "holdout candidate" in roles
                    else None,
                    "source_answer_phase": state.get("selected_phase", "answer") if phase == "confirm" else "answer",
                    "configuration": config,
                    "changes_from_baseline": configuration_diff(
                        baseline_summary, config
                    )
                    if config
                    else [],
                }
            )
        ordering = {name: i for i, name in enumerate((*phases, "export", "done"))}
        position = ordering.get(session["stage"], -1)
        phase_position = ordering[phase]
        stage_status = (
            "completed"
            if position > phase_position
            else "running"
            if position == phase_position
            else "not started"
        )
        if not phase_trials:
            stage_status = (
                "not run"
                if session["status"] in TERMINAL or position > phase_position
                else "not started"
            )
        if session["status"] in TERMINAL and stage_status != "completed":
            stage_status = "stopped" if phase_trials else "not run"
        if (
            session["status"] in {"paused", "needs_input"}
            and position == phase_position
        ):
            stage_status = session["status"]
        sections[phase] = {
            "phase": phase,
            "split": split,
            "status": stage_status,
            "question_count": len(question_ids),
            "primary_metric": primary_metric(phase, state.get("workflow_version", 1)),
            "trials": output,
        }

    plan = deepcopy(state.get("answer_plan", []))
    adaptive_entries = []
    if state.get("workflow_version", 1) >= 4:
        from ragbench_core.optimization.adaptive import catalog
        from ragbench_core.optimization.tools import OptimizationTools
        adaptive_entries = catalog(OptimizationTools(session_id))
        plan = [{k: v for k, v in e.items() if k != "config"} for e in adaptive_entries]
        iteration_rows = store.rows(store.OptimizationIteration, session_id)
        for phase in ("retrieval", "answer"):
            current = state.get("adaptive", {}).get(phase, {})
            sections[phase]["planner"] = {
                "status": current.get("planner_status", "waiting for observation"),
                "iteration_count": sum(r["phase"] == phase for r in iteration_rows),
                "stop_reason": current.get("stop_reason"),
                "error": current.get("planner_error"),
            }
        # Preserve source lineage without duplicating the candidate catalog in progress.
        for trial in sections["answer"]["trials"]:
            entry = next((e for e in adaptive_entries if e.get("trial_id") == trial["id"]), None)
            if entry:
                trial["source_trial_id"] = entry.get("source_trial_id")
                if entry.get("hypothesis"):
                    trial["selection_reasons"].append(entry["hypothesis"])
    answers_by_hash = {_hash(t["config"]): t for t in trials if t["phase"] == "answer"}
    plan = [
        {
            **entry,
            "trial_id": entry.get("trial_id") or answers_by_hash.get(entry["candidate_hash"], {}).get("id"),
            "status": (entry.get("status") if state.get("workflow_version", 1) >= 4 else None)
            or answers_by_hash.get(entry["candidate_hash"], {}).get("status")
            or (entry.get("compilation_status") if entry.get("compilation_status") != "completed" else None)
            or ("planned" if entry["scheduled"] else "outside trial cap"),
        }
        for entry in plan
    ]
    sections["answer"].update(plan=plan, selected_trial_id=selected_id)
    if state.get("workflow_version", 1) >= 6:
        sections["answer"]["selected_trial_id"] = state.get("answer_winner_trial_id") or selected_id
        prompt_status = state.get("prompt_status", sections["prompt"]["status"])
        if not session["request"].get("prompt_optimization", {}).get("configs"):
            prompt_status = "disabled"
        elif session["stage"] == "prompt" and session["status"] in {
            "paused", "needs_input", "failed", "cancelled", "budget_exhausted"
        }:
            prompt_status = session["status"]
        sections["prompt"].update(
            status=prompt_status,
            plan=[{k: v for k, v in e.items() if k not in {"config", "seed_prompt"}} for e in state.get("prompt_plan", [])],
            parent_answer_trial_id=state.get("answer_winner_trial_id"),
            selected_trial_id=selected_id,
            budget_required=state.get("prompt_budget_required"),
        )
    if state.get("prompt_splits"):
        from ragbench_core.optimization.budget import totals
        from ragbench_core.prompt_optimization.history_store import history_status
        from ragbench_core.prompt_optimization.service import get_job
        entries = state.get("prompt_plan", []) if state.get("workflow_version", 1) >= 6 else adaptive_entries if state.get("workflow_version", 1) >= 4 else state.get("prompt_answer_plan", [])
        job_ids = {e["compilation_job_id"] for e in entries if e.get("compilation_job_id")}
        compilation_section = sections["prompt"] if state.get("workflow_version", 1) >= 6 else sections["answer"]
        compilation_section["compilations"] = []
        for jid in sorted(job_ids):
            job = get_job(jid)
            compilation_section["compilations"].append({
                "id": jid, "status": job["status"], "progress": job["progress"],
                "usage": totals(jid), "history": history_status(job),
            })
    sections["retrieval"]["shortlist"] = shortlist
    comparison = deepcopy(state.get("comparison"))
    if comparison:
        by_id = {t["id"]: t for t in trials}
        for metric in comparison.get("metrics", []):
            if "cost" in metric["metric"]:
                for side, key in (
                    ("baseline", "baseline_run_id"),
                    ("candidate", "run_id"),
                ):
                    if (
                        not by_id.get(comparison.get(key), {})
                        .get("summary", {})
                        .get("pricing_known", False)
                    ):
                        metric[f"{side}_mean"] = None
                        metric.update(
                            mean_delta=None, ci95=None, reason="unknown pricing"
                        )
    confirmation = sections["confirm"]
    correctness = next(
        (
            m
            for m in (comparison or {}).get("metrics", [])
            if m["metric"] == "answer_correctness"
        ),
        {},
    )
    if not confirmation["trials"]:
        outcome = "confirmation not run"
    elif any(t["status"] in {"failed", "cancelled"} for t in confirmation["trials"]):
        outcome = "failed confirmation"
    elif any(t["status"] != "completed" for t in confirmation["trials"]):
        outcome = "confirmation in progress"
    elif selected_config and _hash(selected_config) == baseline_hash:
        outcome = "candidate is baseline"
    elif state.get("confirmed"):
        outcome = "confirmed improvement"
    elif (
        not correctness.get("ci95")
        or correctness.get("paired_count", 0) / max(1, confirmation["question_count"])
        < 0.95
    ):
        outcome = "insufficient evidence"
    else:
        outcome = "improvement not demonstrated"
    confirmation.update(
        outcome=outcome,
        comparison=comparison,
        selected_trial_id=selected_id,
        selected_phase=state.get("selected_phase", "answer"),
        configuration_differences=configuration_diff(
            baseline_summary, redact(config_summary(selected_config))
        )
        if selected_config
        else [],
        baseline_retained=not bool(state.get("confirmed")),
    )
    if state.get("recommendation_policy_version", 1) >= POLICY_VERSION:
        decision = state.get("acceptance") or evaluate_acceptance(
            baseline=baseline, candidate=selected_config, trials=trials, results=results,
            question_ids=[q["question_id"] for q in payload if q.get("split") == "holdout"],
            comparison=state.get("comparison"), limits=session["request"]["limits"],
        )
        # Preserve progress/not-run labels until confirmation finishes. The saved
        # acceptance decision controls final displays and exports everywhere.
        if state.get("acceptance") or (confirmation["trials"] and all(t["status"] == "completed" for t in confirmation["trials"])):
            confirmation["outcome"] = decision["outcome"].replace("_", " ")
        confirmation.update(
            acceptance=decision,
            baseline_retained=decision["recommended_source"] == "baseline",
            explanation=state.get("explanation"),
        )
        if state.get("recommendation_evidence"):
            confirmation["recommendation_evidence"] = deepcopy(state["recommendation_evidence"])
    return {
        "session_id": session_id,
        "workflow_version": state.get("workflow_version", 1),
        "stages": sections,
    }
