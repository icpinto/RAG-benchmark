"""Workflow v6: protected, ordered prompt compilation after answer search."""

from copy import deepcopy

from langgraph.errors import GraphInterrupt

from ragbench_core.models.schema import QueryExample
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.datasets import legacy_question
from ragbench_core.optimization.search import candidate_hash
from ragbench_core.optimization.selection import compare_trials
from ragbench_core.prompt_optimization.artifacts import seed_prompt
from ragbench_core.prompt_optimization.auto import prepare_answer_entry
from ragbench_core.prompt_optimization.datasets import split_questions


def enabled(tools):
    return tools.session["progress"].get("workflow_version", 1) >= 6 and bool(
        tools.session["request"]["prompt_optimization"]["configs"]
    )


def credit(tools, index, kind):
    return f"{tools.session_id}:prompt:{index}:{kind}"


def estimate(tools, config):
    config = deepcopy(config)
    config["runtime"]["mode"] = "end_to_end"
    examples = [
        QueryExample(**legacy_question(q))
        for q in store.dataset(tools.session_id)["payload"]
        if q["question_id"] in tools.session["progress"]["prompt_splits"]["validation"]
    ]
    return budget.trial_estimate(
        config,
        examples,
        tools.read("documents.json"),
        tools.session["request"]["judge"],
    )


def prepare(tools):
    """Validate and protect all prompt capacity before any retrieval trials."""
    from ragbench_core.optimization.workflow import progress

    if not enabled(tools):
        return
    state = tools.session["progress"]
    if state.get("prompt_budget_ready"):
        return
    if not state.get("prompt_splits"):
        tuning = [
            q
            for q in store.dataset(tools.session_id)["payload"]
            if q["split"] == "tuning"
        ]
        progress(
            tools.session_id, prompt_splits=split_questions(tuning, evaluation=False)
        )
    config = deepcopy(tools.baseline())
    config["pipeline"]["query"]["context_assemblers"][0]["max_context_chars"] = 24000
    tokens, cost, _ = estimate(tools, config)
    seconds = min(300, tools.session["request"]["limits"]["active_seconds"] // 8)
    requirements = []
    for index, recipe in enumerate(
        tools.session["request"]["prompt_optimization"]["configs"]
    ):
        limits = recipe["limits"]
        requirements.extend(
            [
                (
                    credit(tools, index, "compile"),
                    limits["max_tokens"],
                    limits["max_cost"],
                    limits["max_seconds"],
                ),
                (credit(tools, index, "evaluate"), tokens, cost, seconds),
            ]
        )
    # Also protect a control evaluation if answer search cannot complete one.
    requirements.append((credit(tools, "control", "evaluate"), tokens, cost, seconds))
    needed = {
        "tokens": sum(r[1] for r in requirements),
        "seconds": sum(r[3] for r in requirements),
        "cost": sum(r[2] for r in requirements)
        if all(r[2] is not None for r in requirements)
        else None,
    }
    progress(tools.session_id, prompt_budget_required=needed)
    # reserve_batch is atomic: infeasible preflight never leaves partial holds.
    budget.reserve_batch(tools.session_id, requirements, "reserved_prompt")
    progress(tools.session_id, prompt_budget_ready=True)


def release(tools, index):
    for kind in ("compile", "evaluate"):
        budget.release_reservation(tools.session_id, credit(tools, index, kind))


def cancel(tools):
    from ragbench_core.optimization.workflow import progress

    queue = deepcopy(tools.session["progress"].get("prompt_plan", []))
    for entry in queue:
        if entry["status"] not in {"completed", "failed", "budget_exhausted"}:
            entry["status"] = "cancelled"
    # Active provider calls retain their own accounted rows.
    for row in store.rows(store.BudgetUsage, tools.session_id):
        if row["role"] == "reserved_prompt":
            budget.release_reservation(tools.session_id, row["id"])
    progress(tools.session_id, prompt_plan=queue, prompt_status="cancelled")


def ranking(tools):
    state = tools.session["progress"]
    control = state.get("prompt_control_trial_id") or state.get(
        "answer_winner_trial_id"
    )
    trials = [
        t
        for t in store.rows(store.OptimizationTrial, tools.session_id)
        if t["status"] == "completed" and (t["phase"] == "prompt" or t["id"] == control)
    ]
    return compare_trials(
        trials,
        {t["id"]: store.db.list_results(t["id"]) for t in trials},
        state["prompt_splits"]["validation"],
        "prompt",
        tools.session["request"]["limits"],
    )


def search(tools, dispatch, wait_for_trial):
    from ragbench_core.optimization.workflow import progress

    if not enabled(tools):
        progress(tools.session_id, prompt_status="disabled")
        return "confirm"
    state = tools.session["progress"]
    if "prompt_plan" not in state:
        parent = deepcopy(state.get("selected_config") or tools.baseline())
        seed = seed_prompt(parent)
        parent_id = state.get("selected_trial_id")
        queue = [
            {
                "recipe_index": i,
                "variant": "dspy:" + (recipe.get("preset") or recipe["optimizer"]),
                "source_trial_id": None,
                "parent_answer_trial_id": parent_id,
                "parent_hash": candidate_hash(parent),
                "candidate_hash": candidate_hash(parent),
                "config": deepcopy(parent),
                "seed_prompt": deepcopy(seed),
                "compilation_credit": credit(tools, i, "compile"),
                "compilation_status": "planned",
                "confirmation_reserved": True,
                "scheduled": True,
                "status": "planned",
            }
            for i, recipe in enumerate(
                tools.session["request"]["prompt_optimization"]["configs"]
            )
        ]
        progress(
            tools.session_id,
            answer_winner_config=parent,
            answer_winner_trial_id=parent_id,
            prompt_plan=queue,
            prompt_status="running",
        )
    state = tools.session["progress"]
    control = state.get("answer_winner_trial_id")
    if control is None:
        trial = wait_for_trial(tools, state["answer_winner_config"], "prompt", dispatch)
        control = trial["id"]
    progress(tools.session_id, prompt_control_trial_id=control)
    release(tools, "control")
    queue = deepcopy(tools.session["progress"]["prompt_plan"])

    def save():
        progress(tools.session_id, prompt_plan=queue)

    for entry in queue:
        if entry["status"] in {"completed", "failed", "budget_exhausted", "cancelled"}:
            release(tools, entry["recipe_index"])
            continue
        entry["status"] = "evaluating" if entry.get("evaluation_ready") else "compiling"
        save()
        try:
            if not prepare_answer_entry(tools, entry, save):
                from ragbench_core.prompt_optimization.service import get_job

                job = (
                    get_job(entry["compilation_job_id"])
                    if entry.get("compilation_job_id")
                    else None
                )
                entry["status"] = (
                    "budget_exhausted"
                    if job and job["status"] == "budget_exhausted"
                    else "failed"
                )
                entry["error"] = entry.get("compilation_error")
            else:
                if not entry.get("evaluation_ready"):
                    budget.release_reservation(
                        tools.session_id, entry["compilation_credit"]
                    )
                    tokens, cost, _ = estimate(tools, entry["config"])
                    budget.extend_reservation(
                        tools.session_id,
                        credit(tools, entry["recipe_index"], "evaluate"),
                        tokens,
                        cost,
                    )
                    tools.reserve_prompt_confirmation(entry["config"])
                    entry["evaluation_ready"] = True
                entry["status"] = "evaluating"
                save()
                trial = wait_for_trial(tools, entry["config"], "prompt", dispatch)
                entry.update(
                    trial_id=trial["id"],
                    status="completed"
                    if trial["status"] == "completed"
                    else "budget_exhausted"
                    if trial.get("summary", {}).get("budget_exhausted")
                    else "failed",
                )
                if trial["status"] != "completed":
                    entry["error"] = trial.get("summary", {})
        except budget.BudgetExhausted as exc:
            entry.update(status="budget_exhausted", error=str(exc))
        except (budget.OptimizationStopped, GraphInterrupt):
            raise
        except (
            Exception
        ) as exc:  # Isolate one recipe; provider details remain in worker logs.
            import logging

            logging.getLogger(__name__).exception(
                "Prompt stage recipe %s failed", entry["recipe_index"]
            )
            entry.update(
                status="failed",
                error=f"Prompt recipe failed ({type(exc).__name__}); inspect worker logs",
            )
        save()
        release(tools, entry["recipe_index"])
    ranked = ranking(tools)
    winner = next((r for r in ranked if r["feasible"]), None)
    unchanged = next(
        (r for r in ranked if r["trial_id"] == control and r["feasible"]), None
    )
    if (
        winner
        and unchanged
        and abs(winner["score"] - unchanged["score"]) < 0.01
        and all(
            winner[key] == unchanged[key] for key in ("p95_seconds", "cost_per_query")
        )
    ):
        winner = unchanged  # Prefer the incumbent over an arbitrary trial-ID tiebreak.
    trials = {t["id"]: t for t in store.rows(store.OptimizationTrial, tools.session_id)}
    selected_id = winner["trial_id"] if winner else control
    selected = (
        trials[selected_id]["config"]
        if selected_id in trials
        else state["answer_winner_config"]
    )
    progress(
        tools.session_id,
        prompt_ranking=ranked,
        prompt_status="completed",
        selected_config=tools.validate_candidate(selected),
        selected_trial_id=selected_id,
        selected_phase=trials[selected_id]["phase"]
        if selected_id in trials
        else "answer",
    )
    return "confirm"
