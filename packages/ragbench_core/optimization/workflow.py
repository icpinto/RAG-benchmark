"""Durable graph: model proposals are bounded by deterministic tools and policy."""

from __future__ import annotations

from copy import deepcopy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ragbench_core.engine.comparison import paired_comparison
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.candidates import answer_plan, screen_config
from ragbench_core.optimization.contracts import TrialRequest
from ragbench_core.optimization.search import candidate_hash, seed_candidates
from ragbench_core.optimization.selection import compare_trials
from ragbench_core.optimization.tools import OptimizationTools


class State(TypedDict):
    session_id: str
    stage: str


def progress(session_id, **changes):
    with store.locked(session_id) as (_, row):
        row.progress = {**(row.progress or {}), **changes}


def wait_for_trial(tools, config, phase, dispatch):
    result = tools.run_trial(
        TrialRequest(
            session_id=tools.session_id,
            config=config,
            phase=phase,
            split="holdout" if phase == "confirm" else "tuning",
        ),
        dispatch,
    )
    trial_id = result.artifact_id
    while True:
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["id"] == trial_id
        )
        if trial["status"] in {"completed", "failed", "cancelled"}:
            return trial
        store.update(tools.session_id, status="waiting")
        interrupt({"reason": "trial", "trial_id": trial_id})


def build_graph(checkpointer, dispatch, tools_factory=OptimizationTools):
    def node(stage):
        def run(state):
            session_id = state["session_id"]
            tools = tools_factory(session_id)
            session = tools.session
            if session["status"] == "cancelled":
                return {"stage": "done"}
            while session["status"] in {"paused", "needs_input"}:
                interrupt({"reason": session["status"], "stage": stage})
                session = tools.session
            store.update(session_id, status="running", stage=stage)
            store.event(session_id, "stage", stage=stage)
            try:
                if stage != "export":
                    budget.check_control(session_id)
                if session["progress"].get("stop_reason") and stage not in {
                    "export",
                    "confirm",
                }:
                    return {"stage": "export"}
                next_stage = step(stage, tools, dispatch)
            except budget.BudgetExhausted as exc:
                progress(session_id, stop_reason=str(exc))
                # Never evaluate an unselected candidate on holdout after exhaustion.
                next_stage = (
                    "export"
                    if store.dataset(session_id) and store.dataset(session_id)["frozen"]
                    else "done"
                )
                if next_stage == "done":
                    store.update(session_id, status="budget_exhausted", error=str(exc))
            return {"stage": next_stage}

        return run

    graph = StateGraph(State)
    stages = (
        "profile",
        "generate",
        "freeze",
        "retrieval",
        "answer",
        "confirm",
        "export",
    )
    for stage in stages:
        graph.add_node(stage, node(stage))
        graph.add_conditional_edges(
            stage, lambda state: state["stage"], {**{s: s for s in stages}, "done": END}
        )
    graph.add_conditional_edges(
        START, lambda state: state["stage"], {s: s for s in stages}
    )
    return graph.compile(checkpointer=checkpointer)


def step(stage, tools, dispatch):
    session_id = tools.session_id
    if stage == "profile":
        tools.profile_corpus()
        return "generate"
    if stage == "generate":
        tools.generate_evaluation_dataset()
        return "freeze"
    if stage == "freeze":
        if tools.session["request"]["review_before_freeze"] and not tools.session[
            "progress"
        ].get("review_approved"):
            store.update(session_id, status="paused")
            interrupt({"reason": "dataset_review"})
        result = tools.validate_evaluation_dataset()
        if result.summary.get("needs_input"):
            store.update(
                session_id,
                status="needs_input",
                error=result.summary.get(
                    "reason",
                    "Fewer than 30 validated questions; edit the draft or fork with more sources/guidance",
                ),
            )
            interrupt({"reason": "insufficient_dataset", **result.summary})
            return "freeze"
        return "retrieval"
    if stage == "retrieval":
        return search_retrieval(tools, dispatch)
    if stage == "answer":
        return search_answers(tools, dispatch)
    if stage == "confirm":
        selected = tools.session["progress"]["selected_config"]
        baseline = tools.baseline()
        left = wait_for_trial(tools, baseline, "confirm", dispatch)
        right = (
            left
            if candidate_hash(selected) == candidate_hash(baseline)
            else wait_for_trial(tools, selected, "confirm", dispatch)
        )
        comparison = paired_comparison(
            store.db.get_run(left["id"]),
            store.db.get_run(right["id"]),
            store.db.list_results(left["id"]),
            store.db.list_results(right["id"]),
            expand_cases(left["config"])[0].case_id,
            expand_cases(right["config"])[0].case_id,
        )
        metric = next(
            (m for m in comparison["metrics"] if m["metric"] == "answer_correctness"),
            {},
        )
        total = len(
            [q for q in store.dataset(session_id)["payload"] if q["split"] == "holdout"]
        )
        confirmed = (
            left["status"] == right["status"] == "completed"
            and metric.get("ci95")
            and metric["ci95"][0] > 0
            and metric.get("paired_count", 0) / max(1, total) >= 0.95
        )
        validation = compare_trials(
            [left, right] if left["id"] != right["id"] else [left],
            {t["id"]: store.db.list_results(t["id"]) for t in (left, right)},
            [
                q["question_id"]
                for q in store.dataset(session_id)["payload"]
                if q["split"] == "holdout"
            ],
            "answer",
            tools.session["request"]["limits"],
        )
        confirmed = confirmed and any(
            r["trial_id"] == right["id"] and r["feasible"] for r in validation
        )
        progress(session_id, comparison=comparison, confirmed=bool(confirmed))
        return "export"
    if stage == "export":
        session = tools.session
        state = session["progress"]
        config = (
            state.get("selected_config", tools.baseline())
            if state.get("confirmed")
            else tools.baseline()
        )
        report = {
            "config": config,
            "candidate": state.get("selected_config"),
            "confirmed_improvement": bool(state.get("confirmed")),
            "comparison": state.get("comparison"),
            "tuning_ranking": state.get("answer_ranking", []),
            "retrieval_shortlist": state.get("retrieval_shortlist", []),
            "selected_trial_id": state.get("selected_trial_id"),
            "answer_plan": state.get("answer_plan", []),
            "stop_reason": state.get("stop_reason"),
            "baseline_retained": not state.get("confirmed", False),
            "status": "budget_exhausted" if state.get("stop_reason") else "completed",
        }
        tools.export_recommendation(report)
        store.update(
            session_id,
            recommendation={
                k: v for k, v in report.items() if k not in {"config", "candidate"}
            },
            status=report["status"],
            stage="done",
        )
        return "done"
    raise ValueError("Unknown graph stage")


def search_retrieval(tools, dispatch):
    session_id = tools.session_id
    baseline, space = tools.baseline(), tools.get_search_space()
    maximum = tools.session["request"]["limits"]["retrieval_trials"]
    state = tools.session["progress"]
    if state.get("workflow_version", 1) >= 2 and not state.get("baseline_config"):
        progress(session_id, baseline_config=baseline)
    candidates = state.get("retrieval_candidates", [])
    if not candidates:
        for seed in seed_candidates(baseline, space):
            try:
                if state.get("workflow_version", 1) >= 2:
                    seed = screen_config(seed, baseline)
                candidates.append(tools.validate_candidate(seed))
            except (ValueError, KeyError, TypeError):
                continue
        progress(session_id, retrieval_candidates=candidates[:maximum])
        candidates = candidates[:maximum]
    trials = [
        t
        for t in store.rows(store.OptimizationTrial, session_id)
        if t["phase"] == "retrieval"
    ]
    hashes = {candidate_hash(t["config"]) for t in trials}
    for config in candidates:
        if candidate_hash(config) not in hashes or any(
            t["status"] in {"queued", "running", "paused"}
            and candidate_hash(t["config"]) == candidate_hash(config)
            for t in trials
        ):
            wait_for_trial(tools, config, "retrieval", dispatch)
            return "retrieval"
    ranking = tools.compare_trials("retrieval")
    progress(session_id, retrieval_ranking=ranking)
    attempts = tools.session["progress"].get("proposal_attempts", 0)
    if len(trials) >= maximum or attempts >= maximum * 2:
        return "answer"
    # Supply compact tuning summaries only: no holdout questions, answers, or scores.
    response = tools.call(
        "planner",
        'Propose one promising next RAG pipeline from the bounded search space. Return {"pipeline": object, "reason": string}. '
        "Preserve fixed stages. Use one choice per stage, never arrays of variants. Native fixed/recursive chunk sizes are characters; LlamaIndex sizes use their native units.",
        {
            "search_space": space,
            "baseline_pipeline": baseline["pipeline"],
            "tuning_trials": [
                {
                    "pipeline": t["config"]["pipeline"],
                    "status": t["status"],
                    "score": next(
                        (r for r in ranking if r["trial_id"] == t["id"]), None
                    ),
                }
                for t in trials
            ],
            "attempt": attempts,
        },
    )
    progress(session_id, proposal_attempts=attempts + 1)
    try:
        if state.get("workflow_version", 1) >= 2:
            response = screen_config(response, baseline)
        candidate = tools.validate_candidate(response)
        if candidate_hash(candidate) not in hashes:
            progress(session_id, retrieval_candidates=candidates + [candidate])
    except (ValueError, TypeError, KeyError) as exc:
        store.event(session_id, "candidate_rejected", reason=str(exc))
    return "retrieval"


def search_answers(tools, dispatch):
    if tools.session["progress"].get("workflow_version", 1) >= 3:
        from ragbench_core.prompt_optimization.auto import search_answers as search_prompt_answers
        return search_prompt_answers(tools, dispatch, wait_for_trial, progress)
    session_id = tools.session_id
    state = tools.session["progress"]
    maximum = tools.session["request"]["limits"]["answer_trials"]
    candidates = state.get("answer_candidates")
    if candidates is None and state.get("workflow_version", 1) >= 2:
        plan = answer_plan(
            tools,
            store.rows(store.OptimizationTrial, session_id),
            state.get("retrieval_ranking", []),
            maximum,
        )
        progress(session_id, **plan)
        candidates = plan["answer_candidates"]
    if candidates is None:
        trials = {t["id"]: t for t in store.rows(store.OptimizationTrial, session_id)}
        candidates = [tools.baseline()]
        for ranked in state.get("retrieval_ranking", []):
            if ranked["feasible"]:
                candidates.append(
                    tools.validate_candidate(trials[ranked["trial_id"]]["config"])
                )
        candidates = list({candidate_hash(c): c for c in candidates}.values())[:maximum]
        # Reserve at least one slot for context-only optimization when budget permits.
        if maximum > 1 and candidates:
            context = deepcopy(candidates[-1])
            context["pipeline"]["query"]["compressors"] = [
                {"type": "top_n", "max_chunks": 5}
            ]
            context["pipeline"]["query"]["context_assemblers"] = [
                {"type": "default", "max_context_chars": 6000}
            ]
            try:
                context = tools.validate_candidate(context)
                candidates = list(
                    {
                        candidate_hash(c): c
                        for c in candidates[: maximum - 1] + [context]
                    }.values()
                )
            except (ValueError, TypeError, KeyError):
                pass
        progress(session_id, answer_candidates=candidates)
    for config in candidates:
        wait_for_trial(tools, config, "answer", dispatch)
    ranking = tools.compare_trials("answer")
    feasible = [r for r in ranking if r["feasible"]]
    selected = tools.baseline()
    selected_trial_id = None
    if feasible:
        trial = next(
            t
            for t in store.rows(store.OptimizationTrial, session_id)
            if t["id"] == feasible[0]["trial_id"]
        )
        selected = tools.validate_candidate(trial["config"])
        selected_trial_id = trial["id"]
    progress(
        session_id,
        selected_config=selected,
        selected_trial_id=selected_trial_id,
        answer_ranking=ranking,
    )
    return "confirm"
