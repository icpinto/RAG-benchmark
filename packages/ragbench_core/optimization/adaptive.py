"""Workflow v4: one durable observation and planner decision between experiments."""

import time
import uuid
from copy import deepcopy

from langgraph.types import interrupt
from pydantic import ValidationError

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.adaptive_contracts import (
    AnswerDecision,
    RetrievalDecision,
)
from ragbench_core.optimization.candidates import answer_plan, screen_config
from ragbench_core.optimization.memory import (
    INSTRUCTION,
    MAX_INPUT_BYTES,
    PROMPT_VERSION,
    ContextTooLarge,
    build_context,
    build_observation,
    configuration,
    iterations,
)
from ragbench_core.optimization.models import StructuredOutputError
from ragbench_core.optimization.profiles import ModelProfile
from ragbench_core.optimization.redaction import public_configuration
from ragbench_core.optimization.search import candidate_hash, seed_candidates


def phase_state(tools, phase):
    return tools.session["progress"].get("adaptive", {}).get(phase, {})


def save_state(tools, phase, **changes):
    with store.locked(tools.session_id) as (_, row):
        adaptive = deepcopy(row.progress.get("adaptive", {}))
        adaptive[phase] = {**adaptive.get(phase, {}), **changes}
        row.progress = {**row.progress, "adaptive": adaptive}


def save_iteration(tools, iteration_id, **changes):
    with store.locked(tools.session_id) as (conn, _):
        row = conn.get(store.OptimizationIteration, iteration_id)
        for key, value in changes.items():
            setattr(row, key, deepcopy(value))
        row.updated_at = time.time()


def reserve_planning(tools, operation):
    profile = ModelProfile.model_validate(tools.session["request"]["planner"])
    tokens = 2 * (MAX_INPUT_BYTES + 128 + profile.max_output_tokens)
    cost = None
    if (
        profile.input_price_per_million is not None
        and profile.output_price_per_million is not None
    ):
        cost = (
            2
            * (
                (MAX_INPUT_BYTES + 128) * profile.input_price_per_million
                + profile.max_output_tokens * profile.output_price_per_million
            )
            / 1e6
        )
    budget.reserve(
        tools.session_id,
        operation,
        "reserved_planner",
        tokens,
        cost,
        2 * profile.timeout,
    )


def catalog(tools):
    path = phase_state(tools, "answer").get("catalog")
    return tools.read(path) if path else []


def save_catalog(tools, entries):
    tools.write("answer_catalog.json", entries)


def initialize(tools, phase):
    if phase_state(tools, phase).get("initialized"):
        return
    from ragbench_core.optimization.workflow import progress

    if phase == "retrieval":
        config = tools.baseline()
        progress(tools.session_id, baseline_config=config)
        action = {
            "config": config,
            "hypothesis": "Measure the baseline retrieval configuration",
        }
    else:
        from ragbench_core.prompt_optimization.auto import plan_answers

        recipes = (
            tools.session["request"].get("prompt_optimization", {}).get("configs", [])
        )
        if recipes and not tools.session["progress"].get("prompt_splits"):
            from ragbench_core.prompt_optimization.datasets import split_questions

            tuning = [
                q
                for q in store.dataset(tools.session_id)["payload"]
                if q["split"] == "tuning"
            ]
            try:
                progress(
                    tools.session_id,
                    prompt_splits=split_questions(tuning, evaluation=False),
                )
            except ValueError:
                store.update(
                    tools.session_id,
                    status="needs_input",
                    error="Too few independent tuning groups for prompt training; start a new dataset version",
                )
                interrupt({"reason": "insufficient_prompt_training_groups"})
                return
        # Resolve the entire approved catalog. Trial caps constrain execution, not visibility.
        plan = (
            plan_answers(tools, 10000)
            if recipes
            else answer_plan(
                tools,
                store.rows(store.OptimizationTrial, tools.session_id),
                tools.session["progress"].get("retrieval_ranking", []),
                10000,
            )
        )
        if recipes:
            entries = plan["prompt_answer_plan"]
        else:
            entries = [
                {**e, "config": c}
                for e, c in zip(plan["answer_plan"], plan["answer_candidates"])
            ]
        for entry in entries:
            entry.update(
                candidate_id=entry["candidate_hash"], scheduled=False, status="untested"
            )
        save_catalog(tools, entries)
        progress(
            tools.session_id,
            retrieval_shortlist=plan["retrieval_shortlist"],
            planner_memory={"lessons": []},
        )
        action = {
            "candidate_id": entries[0]["candidate_id"],
            "hypothesis": "Measure the unchanged retrieval winner as the answer-stage control",
        }
    path = tools.write("initial_" + phase + "_action.json", action)
    save_state(
        tools,
        phase,
        initialized=True,
        pending=path,
        active_iteration=None,
        **({"catalog": "answer_catalog.json"} if phase == "answer" else {}),
    )


def observation_iteration(tools, phase, action_path, trial=None, entry=None):
    saved = iterations(tools.session_id)
    existing = next(
        (r for r in saved if r["source_action_artifact"] == action_path), None
    )
    if existing:
        return existing
    observation = build_observation(tools, phase, trial, entry)
    if trial and any(r["observation_key"] == observation["reference"] for r in saved):
        # Two recipes can compile to the same exact pipeline. Reuse its trial but
        # record this compilation action separately so the outer loop still advances.
        observation["reference"] = "compilation:" + entry["compilation_job_id"]
        observation["errors"].append(
            "Reused a previously evaluated pipeline; this is not an independent trial"
        )
    key = observation["reference"]
    existing = next(
        (r for r in iterations(tools.session_id) if r["observation_key"] == key), None
    )
    if existing:
        return existing
    iteration_id = str(uuid.uuid4())
    path = tools.write("observation_" + iteration_id + ".json", observation)
    reservation = planning_operation(tools, phase, action_path)
    with store.locked(tools.session_id) as (conn, session):
        # The session lease serializes workers; this lock also protects API-side replay.
        existing = conn.execute(
            store.select(store.OptimizationIteration).where(
                store.OptimizationIteration.session_id == tools.session_id,
                store.OptimizationIteration.observation_key == key,
            )
        ).scalar_one_or_none()
        if existing:
            return store.payload(existing)
        sequence = (
            len(
                [
                    r
                    for r in conn.execute(
                        store.select(store.OptimizationIteration).where(
                            store.OptimizationIteration.session_id == tools.session_id,
                            store.OptimizationIteration.phase == phase,
                        )
                    ).scalars()
                ]
            )
            + 1
        )
        row = store.OptimizationIteration(
            id=iteration_id,
            session_id=tools.session_id,
            phase=phase,
            sequence=sequence,
            observation_key=key,
            observation_artifact=path,
            source_action_artifact=action_path,
            reservation=reservation,
            status="pending",
            attempts=[],
            cycle=0,
        )
        conn.add(row)
        conn.flush()
        adaptive = deepcopy(session.progress.get("adaptive", {}))
        adaptive[phase] = {
            **adaptive.get(phase, {}),
            "pending": None,
            "active_iteration": iteration_id,
        }
        session.progress = {**session.progress, "adaptive": adaptive}
        return store.payload(row)


def planning_operation(tools, phase, action_path):
    return tools.session_id + ":planner:" + phase + ":" + stable_hash(action_path)


def capture_pending_terminal(tools, phase):
    """Persist terminal observations even if cancellation/budget prevents another LLM call."""
    path = phase_state(tools, phase).get("pending")
    if not path:
        return
    action = tools.read(path)
    entry = None
    if phase == "answer":
        entries = catalog(tools)
        entry = next(e for e in entries if e["candidate_id"] == action["candidate_id"])
        if "recipe_index" in entry and entry.get("compilation_status") not in {
            "completed",
            "failed",
            "cancelled",
        }:
            if entry.get("compilation_job_id"):
                from ragbench_core.prompt_optimization.service import get_job

                job = get_job(entry["compilation_job_id"])
                if job["status"] in {"failed", "cancelled"}:
                    entry.update(compilation_status=job["status"], status="failed")
                    save_catalog(tools, entries)
            if entry.get("compilation_status") not in {"failed", "cancelled"}:
                return
        if entry.get("compilation_status") in {"failed", "cancelled"}:
            observation_iteration(tools, phase, path, entry=entry)
            return
    config = entry["config"] if entry else action["config"]
    trial = next(
        (
            t
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["phase"] == phase
            and candidate_hash(t["config"]) == candidate_hash(config)
            and t["status"] in {"completed", "failed", "cancelled"}
        ),
        None,
    )
    if trial:
        if entry:
            entries = catalog(tools)
            for e in entries:
                if e["candidate_id"] == entry["candidate_id"]:
                    e.update(
                        trial_id=trial["id"],
                        status="executed"
                        if trial["status"] == "completed"
                        else "failed",
                    )
            save_catalog(tools, entries)
        observation_iteration(tools, phase, path, trial, entry)


def run_pending(tools, phase, dispatch, wait_for_trial):
    path = phase_state(tools, phase)["pending"]
    action = tools.read(path)
    reservation = planning_operation(tools, phase, path)
    reserve_planning(tools, reservation)
    try:
        entry = None
        if phase == "answer":
            from ragbench_core.prompt_optimization.auto import prepare_answer_entry

            entries = catalog(tools)
            entry = next(
                e for e in entries if e["candidate_id"] == action["candidate_id"]
            )
            entry.update(
                scheduled=True, status="selected", hypothesis=action["hypothesis"]
            )
            save_catalog(tools, entries)
            if not prepare_answer_entry(
                tools, entry, lambda: save_catalog(tools, entries)
            ):
                entry["status"] = "failed"
                save_catalog(tools, entries)
                observation_iteration(tools, phase, path, entry=entry)
                return
        config = entry["config"] if entry else action["config"]
        trial = wait_for_trial(tools, config, phase, dispatch)
        if entry:
            entry.update(
                trial_id=trial["id"],
                status="executed" if trial["status"] == "completed" else "failed",
            )
            save_catalog(tools, entries)
        observation_iteration(tools, phase, path, trial, entry)
    except budget.BudgetExhausted:
        capture_pending_terminal(tools, phase)
        budget.release_reservation(tools.session_id, reservation)
        raise


def available_candidates(tools, phase):
    if phase == "answer":
        return [
            {
                "candidate_id": e["candidate_id"],
                "variant": e["variant"],
                "configuration": configuration(e["config"]),
                "recipe_index": e.get("recipe_index"),
            }
            for e in catalog(tools)
            if e["status"] == "untested"
        ]
    tried = {
        candidate_hash(t["config"])
        for t in store.rows(store.OptimizationTrial, tools.session_id)
        if t["phase"] == phase
    }
    seeds = []
    for seed in seed_candidates(tools.baseline(), tools.get_search_space()):
        try:
            seed = tools.validate_candidate(screen_config(seed, tools.baseline()))
        except (ValueError, KeyError, TypeError):
            continue
        if candidate_hash(seed) not in tried:
            seeds.append(
                {
                    "candidate_id": candidate_hash(seed),
                    "pipeline": public_configuration(seed["pipeline"]),
                }
            )
    return seeds


def validate_decision(tools, phase, response, force_stop):
    model = RetrievalDecision if phase == "retrieval" else AnswerDecision
    decision = model.model_validate(response).model_dump()
    observed_ids = {
        tools.read(r["observation_artifact"])["trial_id"]
        for r in iterations(tools.session_id)
    } - {None}
    if any(
        set(lesson["trial_ids"]) - observed_ids
        for lesson in decision["memory"]["lessons"]
    ):
        raise ValueError("Lessons must cite observed trial IDs")
    if decision["action"] == "stop":
        if not decision["reason"].strip():
            raise ValueError("Stopping requires a reason")
        return decision, None
    if force_stop:
        raise ValueError("This stage has no remaining experiment capacity; return stop")
    if not decision["hypothesis"].strip():
        raise ValueError("Running requires a testable hypothesis")
    if phase == "retrieval":
        if not isinstance(decision["pipeline"], dict):
            raise ValueError("A retrieval run requires a pipeline object")
        config = tools.validate_candidate({"pipeline": decision["pipeline"]})
        if candidate_hash(config) != candidate_hash(
            screen_config(config, tools.baseline())
        ):
            raise ValueError(
                "Retrieval must preserve baseline compression, context and prompt"
            )
        if any(
            candidate_hash(t["config"]) == candidate_hash(config)
            for t in store.rows(store.OptimizationTrial, tools.session_id)
            if t["phase"] == phase
        ):
            raise ValueError("Configuration has already been attempted")
        action = {"config": config, "hypothesis": decision["hypothesis"]}
    else:
        entry = next(
            (
                e
                for e in catalog(tools)
                if e["candidate_id"] == decision["candidate_id"]
                and e["status"] == "untested"
            ),
            None,
        )
        if entry is None:
            raise ValueError("Choose an untested approved answer candidate ID")
        tools.validate_candidate(entry["config"])
        action = {
            "candidate_id": entry["candidate_id"],
            "hypothesis": decision["hypothesis"],
        }
    return decision, action


def pause_planner(tools, phase, iteration, reason):
    budget.check_control(tools.session_id)
    save_iteration(tools, iteration["id"], status="paused")
    budget.release_reservation(tools.session_id, iteration["reservation"])
    save_state(tools, phase, planner_status="paused", planner_error=reason)
    store.update(tools.session_id, status="paused", error=reason)
    interrupt(
        {
            "reason": "planner_recovery",
            "iteration_id": iteration["id"],
            "message": reason,
        }
    )


def plan_next(tools, phase, iteration):
    if iteration["status"] == "committed":
        return
    if iteration["status"] == "paused":
        # Explicit resume starts a new accounted cycle; old invalid responses remain audit history.
        cycle = iteration["cycle"] + 1
        reservation = (
            iteration["reservation"].split(":resume:")[0] + ":resume:" + str(cycle)
        )
        save_iteration(
            tools,
            iteration["id"],
            status="pending",
            cycle=cycle,
            reservation=reservation,
        )
        iteration = next(
            r for r in iterations(tools.session_id) if r["id"] == iteration["id"]
        )
    reserve_planning(tools, iteration["reservation"])
    observation = tools.read(iteration["observation_artifact"])
    maximum = tools.session["request"]["limits"][
        "retrieval_trials" if phase == "retrieval" else "answer_trials"
    ]
    count = sum(
        t["phase"] == phase
        for t in store.rows(store.OptimizationTrial, tools.session_id)
    )
    candidates = available_candidates(tools, phase)
    force_stop = count >= maximum or (phase == "answer" and not candidates)
    schema = (
        RetrievalDecision if phase == "retrieval" else AnswerDecision
    ).model_json_schema()
    for _ in range(2):
        attempts = deepcopy(iteration["attempts"])
        current = [a for a in attempts if a["cycle"] == iteration["cycle"]]
        if len(current) >= 2 and current[-1]["status"] == "failed":
            break
        attempt = (
            current[-1] if current and current[-1]["status"] == "running" else None
        )
        if attempt is None:
            attempt = {
                "id": str(uuid.uuid4()),
                "cycle": iteration["cycle"],
                "status": "running",
            }
            feedback = current[-1].get("error") if current else None
            try:
                data = build_context(
                    tools,
                    iteration,
                    observation,
                    candidates,
                    schema,
                    force_stop=force_stop,
                    feedback=feedback,
                    attempt_id=attempt["id"],
                )
            except ContextTooLarge as exc:
                pause_planner(tools, phase, iteration, str(exc))
                return
            attempt["prompt_artifact"] = tools.write(
                "planner_prompt_" + attempt["id"] + ".json",
                {
                    "instruction": INSTRUCTION,
                    "data": data,
                    "prompt_version": PROMPT_VERSION,
                },
            )
            attempts.append(attempt)
            save_iteration(tools, iteration["id"], attempts=attempts, status="planning")
        else:
            data = tools.read(attempt["prompt_artifact"])["data"]
        save_state(tools, phase, planner_status="planning", planner_error=None)
        try:
            response = tools.call(
                "planner",
                INSTRUCTION,
                data,
                attempt_id=attempt["id"],
                credit=iteration["reservation"],
                prompt_version=PROMPT_VERSION,
            )
            budget.check_control(tools.session_id)
            attempt["response_artifact"] = tools.write(
                "planner_response_" + attempt["id"] + ".json",
                public_configuration(response),
            )
            decision, action = validate_decision(tools, phase, response, force_stop)
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, StructuredOutputError):
                attempt["response_artifact"] = tools.write(
                    "planner_response_" + attempt["id"] + ".json",
                    {"invalid_output": public_configuration(exc.output)},
                )
            error = (
                "Invalid decision fields: "
                + ", ".join(
                    ".".join(map(str, e["loc"])) + " " + e["type"]
                    for e in exc.errors(include_input=False)
                )
                if isinstance(exc, ValidationError)
                else str(exc)[:350]
            )
            attempt.update(status="failed", error=error)
            save_iteration(tools, iteration["id"], attempts=attempts)
            iteration["attempts"] = attempts
            continue
        attempt["status"] = "accepted"
        action_path = (
            tools.write("planner_action_" + iteration["id"] + ".json", action)
            if action
            else None
        )
        with store.locked(tools.session_id) as (conn, session):
            row = conn.get(store.OptimizationIteration, iteration["id"])
            row.status, row.decision, row.action_artifact = (
                "committed",
                decision,
                action_path,
            )
            row.attempts, row.updated_at = attempts, time.time()
            adaptive = deepcopy(session.progress["adaptive"])
            adaptive[phase].update(
                pending=action_path,
                active_iteration=None,
                planner_status="ready",
                finished=action is None,
                stop_reason=decision["reason"] if action is None else None,
            )
            if action is None:
                adaptive[phase]["planner_status"] = "stopped"
            session.progress = {
                **session.progress,
                "adaptive": adaptive,
                "planner_memory": decision["memory"],
            }
        budget.release_reservation(tools.session_id, iteration["reservation"])
        return
    pause_planner(
        tools, phase, iteration, "Planner failed twice; resume to retry this decision"
    )


def finish_phase(tools, phase):
    from ragbench_core.optimization.workflow import progress

    ranking = tools.compare_trials(phase)
    if phase == "retrieval":
        progress(tools.session_id, retrieval_ranking=ranking)
        return "answer"
    winner = next((r for r in ranking if r["feasible"]), None)
    trials = store.rows(store.OptimizationTrial, tools.session_id)
    config = next(
        (t["config"] for t in trials if winner and t["id"] == winner["trial_id"]),
        tools.baseline(),
    )
    progress(
        tools.session_id,
        answer_ranking=ranking,
        selected_config=tools.validate_candidate(config),
        selected_trial_id=winner["trial_id"] if winner else None,
    )
    return "confirm"


def stop_planning(tools, status, reason):
    for item in iterations(tools.session_id):
        if item["status"] not in {"committed", "cancelled", "budget_exhausted"}:
            save_iteration(tools, item["id"], status=status)
    for usage in store.rows(store.BudgetUsage, tools.session_id):
        if usage["role"] == "reserved_planner":
            budget.release_reservation(tools.session_id, usage["id"])
    for phase in ("retrieval", "answer"):
        if phase_state(tools, phase) and not phase_state(tools, phase).get("finished"):
            save_state(tools, phase, planner_status=status, stop_reason=reason)


def search(tools, phase, dispatch, wait_for_trial):
    initialize(tools, phase)
    for item in iterations(tools.session_id):
        if item["status"] == "committed":
            budget.release_reservation(tools.session_id, item["reservation"])
    had_pending = bool(phase_state(tools, phase).get("pending"))
    capture_pending_terminal(tools, phase)
    if tools.session["progress"].get("stop_reason"):
        raise budget.BudgetExhausted(tools.session["progress"]["stop_reason"])
    state = phase_state(tools, phase)
    if state.get("finished"):
        return finish_phase(tools, phase)
    if state.get("pending"):
        run_pending(tools, phase, dispatch, wait_for_trial)
        return phase
    if had_pending:
        # Checkpoint the observation before a different kind of interrupt can occur.
        return phase
    state = phase_state(tools, phase)
    if state.get("active_iteration"):
        iteration = next(
            r
            for r in iterations(tools.session_id)
            if r["id"] == state["active_iteration"]
        )
        try:
            plan_next(tools, phase, iteration)
        except budget.BudgetExhausted:
            save_iteration(tools, iteration["id"], status="budget_exhausted")
            budget.release_reservation(tools.session_id, iteration["reservation"])
            raise
    return (
        finish_phase(tools, phase)
        if phase_state(tools, phase).get("finished")
        else phase
    )
