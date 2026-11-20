"""Explicit manual compilation, shared recipes, and reusable prompt artifacts."""

import uuid
from copy import deepcopy
from dataclasses import asdict

import yaml
from fastapi import APIRouter, HTTPException
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.data_loader import load_ground_truth
from ragbench_core.engine.snapshots import attach_evaluation, snapshot_inputs
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.profiles import ModelProfile
from ragbench_core.optimization.search import concrete_config
from ragbench_core.prompt_optimization.artifacts import builder_config
from ragbench_core.prompt_optimization.contracts import (
    PRESETS,
    CompilationRequest,
    PromptOptimizationSettings,
)
from ragbench_core.prompt_optimization.datasets import split_questions
from ragbench_core.prompt_optimization.service import (
    TERMINAL,
    artifacts_for_job,
    create_job,
    get_job,
    task_profile,
    update_job,
)
from ragbench_core.utils.config import load_yaml
from sqlalchemy import select

from services.api.app.optimizations import (
    ARTIFACT_ROOT,
    CONFIG_ROOT,
    DATA_ROOT,
    confined,
    public_configuration,
    reject_secrets,
)

router = APIRouter(tags=["prompt optimization"])


def job_or_404(job_id):
    try:
        return get_job(job_id)
    except KeyError:
        raise HTTPException(404, "Prompt compilation job not found") from None


def artifact_or_404(artifact_id):
    with store.db.SessionLocal() as conn:
        row = conn.get(store.PromptArtifact, artifact_id)
        if row is None:
            raise HTTPException(404, "Prompt artifact not found")
        return store.payload(row)


@router.get("/prompt-optimization/presets")
def presets():
    return {
        "presets": [{"id": key, **value} for key, value in PRESETS.items()],
        "default": "mipro_zero_shot_v1",
        "config_schema": PromptOptimizationSettings.model_json_schema(),
    }


@router.get("/prompt-optimization/cases")
def cases(config_path: str):
    try:
        config = load_yaml(str(confined(config_path, CONFIG_ROOT)))
        return [
            {
                "id": case.case_id,
                "pipeline": {
                    "ingestion": case.ingestion_config,
                    "query": case.query_config,
                },
                "answer_model": public_configuration(case.llm_config),
            }
            for case in expand_cases(config)
        ]
    except (ValueError, OSError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/prompt-optimizations")
def compile_prompts(req: CompilationRequest):
    store.db.init_db()
    try:
        config = load_yaml(str(confined(req.config_path, CONFIG_ROOT)))
        reject_secrets(config)
        config.setdefault("runtime", {})["artifact_dir"] = str(ARTIFACT_ROOT)
        config["data"]["input_dir"] = str(
            confined(config["data"]["input_dir"], DATA_ROOT)
        )
        config["data"]["ground_truth_file"] = str(
            confined(config["data"]["ground_truth_file"], DATA_ROOT.parent / "eval")
        )
        if config["data"].get("remote_urls_file"):
            config["data"]["remote_urls_file"] = str(
                confined(config["data"]["remote_urls_file"], DATA_ROOT.parent)
            )
        selected = expand_cases(config)
        if req.case_id:
            selected = [case for case in selected if case.case_id == req.case_id]
        if len(selected) != 1:
            raise ValueError("Select one concrete pipeline case before compiling")
        config = concrete_config(config, selected[0])
        config["llm"].setdefault("max_output_tokens", 1024)
        settings = req.prompt_optimization or PromptOptimizationSettings.model_validate(
            config.get("prompt_optimization", {})
        )
        settings.validate_policy(req.allow_hosted)
        config["prompt_optimization"] = settings.model_dump()
        if not req.allow_hosted:
            hosted = {"openai", "voyage", "cohere", "jina"}
            if any(
                e.get("provider") in hosted
                for e in config["pipeline"]["ingestion"]["embeddings"]
            ) or any(
                r.get("type") in {"cohere", "jina"}
                for r in config["pipeline"]["query"]["rerankers"]
            ):
                raise ValueError("Hosted components require source-data opt-in")
        profile = task_profile(config)
        judge = (
            req.judge
            or ModelProfile.model_validate(
                {
                    k: v
                    for k, v in config.get("metrics", {}).get("judge", {}).items()
                    if k in ModelProfile.model_fields
                }
            )
            if config.get("metrics", {}).get("judge", {}).get("model")
            else req.judge or profile
        )
        config["pipeline"]["query"]["prompt_builders"] = [{"type": "default"}]
        config["input_selection"] = {
            **(
                {
                    "source_paths": [
                        str(confined(p, DATA_ROOT)) for p in req.source_paths
                    ]
                }
                if req.source_paths is not None
                else {}
            ),
            **(
                {"question_indices": req.question_indices}
                if req.question_indices is not None
                else {}
            ),
        }
        frozen = snapshot_inputs("prompt-input-" + str(uuid.uuid4()), config)
        examples = load_ground_truth(frozen["_snapshot"]["ground_truth_file"])
        splits = split_questions([asdict(e) for e in examples])
        job_id = create_job(
            frozen, settings, splits, judge, allow_hosted=req.allow_hosted
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    from services.worker.app.prompt_tasks import compile_prompt_job

    try:
        compile_prompt_job.apply_async(args=[job_id], queue="optimization")
    except Exception:  # noqa: BLE001 -- queue errors must not expose broker credentials.
        update_job(job_id, status="failed", error="Queue submission failed")
        raise HTTPException(503, "Prompt compilation could not be queued") from None
    return {"id": job_id, "status": "queued"}


@router.get("/prompt-optimizations/{job_id}")
def compilation(job_id: str):
    job = job_or_404(job_id)
    return {
        k: public_configuration(job[k])
        for k in (
            "id",
            "status",
            "request",
            "splits",
            "progress",
            "error",
            "attempts",
            "created_at",
            "updated_at",
        )
    } | {
        "usage": budget.totals(job_id),
        "artifacts": [public_configuration(a) for a in artifacts_for_job(job_id)],
    }


@router.post("/prompt-optimizations/{job_id}/cancel")
def cancel(job_id: str):
    job_or_404(job_id)
    with store.locked_budget_owner(job_id) as (_, row):
        if row.status not in TERMINAL:
            row.status = "cancelled"
    return {"id": job_id, "status": get_job(job_id)["status"]}


@router.get("/prompt-artifacts")
def artifacts():
    with store.db.SessionLocal() as conn:
        return [
            {
                "id": row.id,
                "job_id": row.job_id,
                "summary": public_configuration(row.summary),
                "created_at": row.created_at,
            }
            for row in conn.execute(
                select(store.PromptArtifact)
                .order_by(store.PromptArtifact.created_at.desc())
                .limit(100)
            ).scalars()
        ]


@router.get("/prompt-artifacts/{artifact_id}")
def artifact(artifact_id: str):
    return public_configuration(artifact_or_404(artifact_id))


@router.get("/prompt-artifacts/{artifact_id}/config")
def artifact_config(artifact_id: str):
    artifact = artifact_or_404(artifact_id)
    job = job_or_404(artifact["job_id"])
    config = deepcopy(job["config"])
    if job["session_id"]:
        from ragbench_core.models.schema import QueryExample
        from ragbench_core.optimization.datasets import legacy_question
        from ragbench_core.optimization.tools import OptimizationTools

        examples = [
            QueryExample(**legacy_question(q))
            for q in store.dataset(job["session_id"])["payload"]
            if q["split"] == "holdout"
        ]
        corpus = OptimizationTools(job["session_id"]).read("corpus.json")
    else:
        examples = [
            e
            for e in load_ground_truth(config["_snapshot"]["ground_truth_file"])
            if e.question_id in set(job["splits"]["evaluation"])
        ]
        corpus = config["_snapshot"]
    config["_snapshot"] = attach_evaluation(
        corpus, examples, ARTIFACT_ROOT / "prompt_exports" / artifact_id
    )
    config["data"]["ground_truth_file"] = config["_snapshot"]["ground_truth_file"]
    config["pipeline"]["query"]["prompt_builders"] = [
        builder_config(artifact["payload"])
    ]
    config.pop("input_selection", None)
    return {"config": config, "yaml": yaml.safe_dump(config, sort_keys=False)}
