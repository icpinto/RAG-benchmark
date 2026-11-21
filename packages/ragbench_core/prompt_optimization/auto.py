"""Version-three answer planning: bounded compilation followed by shared validation."""

from copy import deepcopy

from langgraph.types import interrupt

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.engine.snapshots import attach_evaluation
from ragbench_core.models.schema import QueryExample
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.candidates import answer_plan
from ragbench_core.optimization.datasets import legacy_question
from ragbench_core.optimization.search import candidate_hash
from ragbench_core.prompt_optimization.artifacts import builder_config
from ragbench_core.prompt_optimization.datasets import split_questions
from ragbench_core.prompt_optimization.service import (
    TERMINAL,
    artifacts_for_job,
    create_job,
    get_job,
    update_job,
)


def validation_ids(session):
    return session["progress"].get("prompt_splits", {}).get("validation")


def plan_answers(tools, maximum):
    session = tools.session
    existing = answer_plan(
        tools,
        store.rows(store.OptimizationTrial, tools.session_id),
        session["progress"].get("retrieval_ranking", []),
        maximum,
    )
    entries = [
        {**entry, "config": config}
        for entry, config in zip(existing["answer_plan"], existing["answer_candidates"])
    ]
    parents = [e for e in entries if e["variant"] in {"baseline", "inherited"}]
    compiled = []
    seen_recipes = set()
    for index, recipe in enumerate(
        session["request"]["prompt_optimization"]["configs"]
    ):
        for parent in parents:
            identity = stable_hash(
                {"recipe": recipe, "parent": candidate_hash(parent["config"])}
            )
            if identity in seen_recipes:
                continue
            seen_recipes.add(identity)
            compiled.append(
                {
                    "candidate_hash": "pending:" + identity,
                    "source_trial_id": parent["source_trial_id"],
                    **({"reason": parent["reason"]} if "reason" in parent else {}),
                    "variant": "dspy:" + (recipe.get("preset") or recipe["optimizer"]),
                    "recipe_index": index,
                    "config": parent["config"],
                    "compilation_status": "planned",
                }
            )
    # Preserve all existing coverage entries even when the bounded plan cannot run them.
    configs_by_hash = {candidate_hash(c): c for c in existing["answer_candidates"]}
    variants = [
        {
            **entry,
            **(
                {"config": configs_by_hash[entry["candidate_hash"]]}
                if entry["candidate_hash"] in configs_by_hash
                else {}
            ),
        }
        for entry in existing["answer_plan"]
        if entry["variant"] not in {"baseline", "inherited"}
    ]
    queue = [
        {**entry, "scheduled": i < maximum}
        for i, entry in enumerate(parents + compiled + variants)
    ]
    return {
        "retrieval_shortlist": existing["retrieval_shortlist"],
        "prompt_answer_plan": queue,
        "answer_plan": [
            {k: v for k, v in entry.items() if k != "config"} for entry in queue
        ],
    }


def search_answers(tools, dispatch, wait_for_trial, save_progress):
    session_id = tools.session_id
    state = tools.session["progress"]
    if "prompt_splits" not in state:
        tuning = [
            q for q in store.dataset(session_id)["payload"] if q["split"] == "tuning"
        ]
        try:
            splits = split_questions(tuning, evaluation=False)
        except ValueError as exc:
            store.update(session_id, status="needs_input", error=str(exc))
            interrupt(
                {"reason": "insufficient_prompt_training_groups", "message": str(exc)}
            )
            return "answer"
        save_progress(session_id, prompt_splits=splits)
    if "prompt_answer_plan" not in state:
        save_progress(
            session_id,
            **plan_answers(tools, tools.session["request"]["limits"]["answer_trials"]),
        )
    queue = tools.session["progress"]["prompt_answer_plan"]

    def save_queue():
        save_progress(
            session_id,
            prompt_answer_plan=queue,
            answer_plan=[{k: v for k, v in e.items() if k != "config"} for e in queue],
        )

    for entry in queue:
        if not entry["scheduled"]:
            continue
        if "recipe_index" in entry and entry.get("compilation_status") not in {
            "completed",
            "failed",
        }:
            if not entry.get("compilation_job_id"):
                recipe = deepcopy(
                    tools.session["request"]["prompt_optimization"]["configs"][
                        entry["recipe_index"]
                    ]
                )
                recipe["proposal_model"] = (
                    recipe.get("proposal_model") or tools.session["request"]["planner"]
                )
                splits = tools.session["progress"]["prompt_splits"]
                examples = [
                    QueryExample(**legacy_question(q))
                    for q in store.dataset(session_id)["payload"]
                    if q["split"] == "tuning"
                ]
                config = deepcopy(entry["config"])
                config["_snapshot"] = attach_evaluation(
                    tools.read("corpus.json"),
                    examples,
                    tools.root
                    / "prompt_inputs"
                    / str(entry["recipe_index"])
                    / candidate_hash(config),
                )
                job_id = create_job(
                    config,
                    {"configs": [recipe]},
                    splits,
                    tools.session["request"]["judge"],
                    allow_hosted=tools.session["request"]["allow_hosted"],
                    session_id=session_id,
                    operation_key=session_id
                    + ":prompt:"
                    + str(entry["recipe_index"])
                    + ":"
                    + candidate_hash(config),
                )
                entry["compilation_job_id"] = job_id
                entry["compilation_status"] = "queued"
                save_queue()
                from services.worker.app.prompt_tasks import compile_prompt_job

                try:
                    compile_prompt_job.apply_async(args=[job_id], queue="optimization")
                    store.event(
                        session_id,
                        "prompt_compilation_queued",
                        job_id=job_id,
                        variant=entry["variant"],
                    )
                except Exception:  # noqa: BLE001 -- isolate queue failure to this candidate.
                    update_job(
                        job_id,
                        status="failed",
                        error="Prompt compilation could not be queued",
                    )
            job = get_job(entry["compilation_job_id"])
            while job["status"] not in TERMINAL:
                store.update(session_id, status="waiting")
                interrupt({"reason": "prompt_compilation", "job_id": job["id"]})
                job = get_job(entry["compilation_job_id"])
            artifacts = artifacts_for_job(job["id"])
            if not artifacts:
                entry.update(
                    compilation_status="failed",
                    compilation_error=job.get("error")
                    or job["progress"].get("results", []),
                )
                save_queue()
                continue
            artifact = artifacts[0]
            entry["config"] = deepcopy(entry["config"])
            entry["config"]["pipeline"]["query"]["prompt_builders"] = [
                builder_config(artifact["payload"])
            ]
            entry.update(
                compilation_status="completed",
                artifact_id=artifact["id"],
                candidate_hash=candidate_hash(entry["config"]),
            )
            store.event(
                session_id,
                "prompt_compilation_completed",
                job_id=job["id"],
                artifact_id=artifact["id"],
            )
            save_queue()
        if entry.get("compilation_status") == "failed":
            continue
        if "recipe_index" in entry and not entry.get("confirmation_reserved"):
            try:
                tools.reserve_prompt_confirmation(entry["config"])
            except budget.BudgetExhausted as exc:
                entry.update(compilation_status="failed", compilation_error=str(exc))
                save_queue()
                continue
            entry["confirmation_reserved"] = True
            save_queue()
        trial = wait_for_trial(tools, entry["config"], "answer", dispatch)
        entry["trial_id"] = trial["id"]
        save_queue()
    ranking = tools.compare_trials("answer")
    feasible = [r for r in ranking if r["feasible"]]
    selected, selected_trial = tools.baseline(), None
    if feasible:
        selected_trial = feasible[0]["trial_id"]
        selected = next(
            t["config"]
            for t in store.rows(store.OptimizationTrial, session_id)
            if t["id"] == selected_trial
        )
        selected = tools.validate_candidate(selected)
    save_progress(
        session_id,
        selected_config=selected,
        selected_trial_id=selected_trial,
        answer_ranking=ranking,
    )
    return "confirm"
