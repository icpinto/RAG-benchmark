"""Optimization API. Paths and provider policy are resolved before queueing."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from ragbench_core.engine.data_loader import list_sources
from ragbench_core.engine.snapshots import capture_corpus
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.contracts import DatasetEdit, OptimizationRequest
from ragbench_core.optimization.datasets import validate_questions
from ragbench_core.optimization.recommendation import POLICY_VERSION
from ragbench_core.optimization.redaction import public_configuration
from ragbench_core.optimization.tools import OptimizationTools
from ragbench_core.utils.config import load_yaml

from services.worker.app.optimization_tasks import advance_optimization, dispatch_trial

router = APIRouter(prefix="/optimizations", tags=["optimization"])
CONFIG_ROOT = Path(os.getenv("RAGBENCH_CONFIG_DIR", "/app/configs")).resolve()
DATA_ROOT = (Path(os.getenv("RAGBENCH_DATA_DIR", "/app/data")) / "raw").resolve()
ARTIFACT_ROOT = Path(os.getenv("RAGBENCH_ARTIFACT_DIR", "/app/artifacts")).resolve()
TERMINAL = {"completed", "cancelled", "budget_exhausted", "failed"}


def confined(path, root):
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Path is outside the configured root")
    return candidate


def session_or_404(session_id):
    try:
        return store.get(session_id)
    except KeyError:
        raise HTTPException(404, "Optimization session not found") from None


def public_session(session):
    return {
        k: v
        for k, v in session.items()
        if k not in {"template", "request", "lease_owner", "lease_until", "progress"}
    } | {
        "progress": {
            k: v
            for k, v in session["progress"].items()
            if k
            in {
                "review_approved",
                "proposal_attempts",
                "stop_reason",
                "confirmed",
                "retrieval_ranking",
                "answer_ranking",
                "evaluation_dataset",
            }
        },
        "budget": budget.totals(session["id"]),
        "limits": session["request"]["limits"],
    }



def reject_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                str(key).lower()
                in {"api_key", "authorization", "password", "secret", "token"}
                and child
            ):
                raise ValueError(
                    "Optimizer templates must use server environment credentials, not inline secrets"
                )
            reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            reject_secrets(child)
    elif isinstance(value, str) and "://" in value:
        parsed = urlparse(value)
        if parsed.username or parsed.password:
            raise ValueError(
                "Template endpoint URLs must not contain embedded credentials"
            )


@router.post("")
def create_optimization(req: OptimizationRequest):
    store.db.init_db()
    session_id = str(uuid.uuid4())
    capture_started = time.monotonic()
    try:
        config_path = confined(req.config_path, CONFIG_ROOT)
        template = load_yaml(str(config_path))
        if "prompt_optimization" not in req.model_fields_set:
            from ragbench_core.prompt_optimization.contracts import (
                PromptOptimizationSettings,
            )
            req.prompt_optimization = PromptOptimizationSettings.model_validate(template.get("prompt_optimization", {}))
            req.policy()
        template["prompt_optimization"] = req.prompt_optimization.model_dump()
        if req.limits.max_cost is not None and any(
            recipe.limits.max_cost is None for recipe in req.prompt_optimization.configs
        ):
            raise ValueError(
                "Protected DSPy budgets require max_cost for each recipe when the session has a monetary cap"
            )
        reject_secrets(template)
        template.setdefault("runtime", {})["artifact_dir"] = str(ARTIFACT_ROOT)
        template.setdefault("llm", {}).setdefault("max_output_tokens", 1024)
        template["llm"]["timeout"] = min(
            float(template["llm"].get("timeout", 120)), 240
        )
        input_dir = confined(template["data"]["input_dir"], DATA_ROOT)
        remote_file = template["data"].get("remote_urls_file")
        if remote_file:
            remote_file = str(confined(remote_file, DATA_ROOT))
        all_sources = list_sources(str(input_dir), remote_file)
        local_sources = [
            s for s in all_sources if not s.startswith(("http://", "https://"))
        ]
        for source in local_sources:
            confined(source, DATA_ROOT)  # Also reject symlinks escaping the library.
        if req.source_paths is not None:
            selected = [str(confined(p, DATA_ROOT)) for p in req.source_paths]
            if any(s not in local_sources for s in selected):
                raise ValueError("Selected source is outside the configured corpus")
            local_sources = selected
        urls = req.urls or [
            s for s in all_sources if s.startswith(("http://", "https://"))
        ]
        if any(
            urlparse(u).scheme not in {"http", "https"}
            or not urlparse(u).hostname
            or urlparse(u).username
            for u in urls
        ):
            raise ValueError(
                "Sources must be HTTP(S) URLs without embedded credentials"
            )
        if not req.allow_hosted:
            hosted = {"openai", "voyage", "cohere", "jina"}
            if any(
                e.get("provider") in hosted
                for e in template["pipeline"]["ingestion"].get("embeddings", [])
            ):
                raise ValueError(
                    "Template includes hosted embeddings; enable source-data opt-in or use a local template"
                )
            if any(
                r.get("type") == "cohere"
                for r in template["pipeline"]["query"].get("rerankers", [])
            ):
                raise ValueError(
                    "Template includes hosted reranking; enable source-data opt-in or use a local template"
                )
        if req.limits.max_cost is not None or req.limits.max_query_cost is not None:
            pricing = template["runtime"].get("pricing", {})
            if any(
                k not in pricing
                for k in (
                    "generation_input_per_1m",
                    "generation_output_per_1m",
                    "embedding_per_1m",
                )
            ):
                raise ValueError(
                    "Cost caps require explicit pipeline pricing in runtime.pricing"
                )
        from ragbench_core.optimization.search import get_search_space

        get_search_space(template, req.allowed_components)
        root = ARTIFACT_ROOT / "optimizations" / session_id
        corpus = capture_corpus(local_sources + urls, root / "snapshot")
        with store.db.SessionLocal.begin() as conn:
            conn.add(
                store.OptimizationSession(
                    id=session_id,
                    request=req.model_dump(),
                    template=template,
                    progress={"workflow_version": 6, "recommendation_policy_version": POLICY_VERSION},
                )
            )
            conn.add(
                store.BudgetUsage(
                    id=session_id + ":capture",
                    session_id=session_id,
                    role="corpus_capture",
                    tokens=0,
                    actual_tokens=0,
                    cost=0,
                    seconds=time.monotonic() - capture_started,
                    status="settled",
                )
            )
        OptimizationTools(session_id).write("corpus.json", corpus)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- isolate provider/plugin/queue failures at the service boundary.
        raise HTTPException(
            400, f"Could not prepare corpus ({type(exc).__name__})"
        ) from None
    try:
        advance_optimization.apply_async(args=[session_id], queue="optimization")
    except Exception:  # noqa: BLE001 -- record queue failure without exposing provider secrets.
        store.update(session_id, status="failed", error="Queue submission failed")
        raise HTTPException(503, "Optimization could not be queued") from None
    return {"id": session_id, "status": "queued"}


@router.get("")
def list_optimizations():
    with store.db.SessionLocal() as conn:
        sessions = (
            conn.execute(
                store.select(store.OptimizationSession)
                .order_by(store.OptimizationSession.created_at.desc())
                .limit(100)
            )
            .scalars()
            .all()
        )
        return [public_session(store.payload(s)) for s in sessions]


@router.get("/{session_id}")
def get_optimization(session_id: str):
    return public_session(session_or_404(session_id))


@router.get("/{session_id}/configuration")
def get_configuration(session_id: str):
    session = session_or_404(session_id)
    return public_configuration(
        {"request": session["request"], "template": session["template"]}
    )


@router.get("/{session_id}/trials/{trial_id}/config")
def get_trial_configuration(session_id: str, trial_id: str):
    session_or_404(session_id)
    with store.db.SessionLocal() as conn:
        trial = conn.get(store.OptimizationTrial, trial_id)
        if trial is None or trial.session_id != session_id:
            raise HTTPException(404, "Optimization trial not found")
        return {"config": public_configuration(trial.config)}


@router.get("/{session_id}/dataset")
def get_dataset(session_id: str):
    session_or_404(session_id)
    dataset = store.dataset(session_id)
    return (
        {**dataset, "evaluation_dataset": store.dataset_provenance(session_id)}
        if dataset
        else None
    )


@router.put("/{session_id}/dataset")
def edit_dataset(session_id: str, req: DatasetEdit):
    session = session_or_404(session_id)
    dataset = store.dataset(session_id)
    if (
        session["status"] not in {"paused", "needs_input"}
        or dataset is None
        or dataset["frozen"]
    ):
        raise HTTPException(
            409,
            "Only a paused, unfrozen dataset can be edited; fork to change frozen data",
        )
    tools = OptimizationTools(session_id)
    raw = [q.model_dump() for q in req.questions]
    for question in raw:
        question["automated_validated"] = False
        question["human_reviewed"] = True
    accepted, rejected = validate_questions(raw, tools.read("documents.json"))
    if rejected:
        raise HTTPException(400, {"rejected": rejected})
    store.save_dataset(session_id, [q.model_dump() for q in accepted])
    store.record_dataset_provenance(session_id)
    return get_dataset(session_id)


@router.get("/{session_id}/trials")
def get_trials(session_id: str):
    session_or_404(session_id)
    return [
        {
            k: v
            for k, v in t.items()
            if k not in {"config", "lease_owner", "lease_until"}
        }
        for t in store.rows(store.OptimizationTrial, session_id)
    ]


@router.get("/{session_id}/stages")
def get_stages(session_id: str):
    from ragbench_core.optimization.stages import stage_dashboard

    return public_configuration(stage_dashboard(session_or_404(session_id), redact=public_configuration))


@router.get("/{session_id}/events")
def get_events(session_id: str):
    session_or_404(session_id)
    return sorted(
        store.rows(store.OptimizationEvent, session_id),
        key=lambda e: (e["created_at"], e["id"]),
    )[-500:]


@router.get("/{session_id}/iterations")
def get_iterations(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    phase: Literal["retrieval", "answer"] | None = None,
):
    session_or_404(session_id)
    from ragbench_core.optimization.memory import iteration_summary, iterations

    items = iterations(session_id, phase)
    tools = OptimizationTools(session_id)
    return {"items": [iteration_summary(tools, item) for item in items[offset:offset + limit]],
            "total": len(items), "offset": offset, "limit": limit}


@router.get("/{session_id}/iterations/{iteration_id}")
def get_iteration(session_id: str, iteration_id: str):
    session_or_404(session_id)
    from ragbench_core.optimization.memory import iteration_detail

    with store.db.SessionLocal() as conn:
        item = conn.get(store.OptimizationIteration, iteration_id)
        if item is None or item.session_id != session_id:
            raise HTTPException(404, "Iteration not found")
        return iteration_detail(OptimizationTools(session_id), store.payload(item))


@router.post("/{session_id}/pause")
def pause(session_id: str):
    session = session_or_404(session_id)
    if session["status"] in TERMINAL:
        raise HTTPException(409, "Session has already finished")
    store.update(session_id, status="paused")
    return {"status": "paused"}


@router.post("/{session_id}/resume")
def resume(session_id: str):
    session = session_or_404(session_id)
    if session["status"] not in {"paused", "needs_input"}:
        raise HTTPException(409, "Only paused sessions can be resumed")
    from ragbench_core.optimization.workflow import progress

    dataset = store.dataset(session_id)
    if dataset and session["stage"] == "freeze":
        if not dataset["frozen"]:
            reviewed = [{**q, "human_reviewed": True} for q in dataset["payload"]]
            store.save_dataset(session_id, reviewed)
        progress(session_id, review_approved=True)
    store.update(session_id, status="queued", error=None)
    try:
        for trial in store.rows(store.OptimizationTrial, session_id):
            if trial["status"] == "paused":
                dispatch_trial(trial["id"])
        advance_optimization.apply_async(args=[session_id], queue="optimization")
    except Exception:  # noqa: BLE001 -- record queue failure without exposing provider secrets.
        store.update(
            session_id,
            status="paused",
            error="Resume queue submission failed; retry resume",
        )
        raise HTTPException(503, "Could not queue resume") from None
    return {"status": "queued"}


@router.post("/{session_id}/cancel")
def cancel(session_id: str):
    session = session_or_404(session_id)
    store.update(session_id, status="cancelled")
    from ragbench_core.prompt_optimization.service import TERMINAL as PROMPT_TERMINAL
    with store.db.SessionLocal.begin() as conn:
        for job in conn.execute(store.select(store.PromptCompilationJob).where(store.PromptCompilationJob.session_id == session_id)).scalars():
            if job.status not in PROMPT_TERMINAL:
                job.status = "cancelled"
    if session["progress"].get("workflow_version", 1) >= 4:
        from ragbench_core.optimization.adaptive import (
            capture_pending_terminal,
            stop_planning,
        )
        tools = OptimizationTools(session_id)
        for phase in ("retrieval", "answer"):
            capture_pending_terminal(tools, phase)
        stop_planning(tools, "cancelled", "Session cancelled")
        if session["progress"].get("workflow_version", 1) >= 6:
            from ragbench_core.optimization.prompt_stage import cancel
            cancel(tools)
    return {"status": "cancelled"}


@router.post("/{session_id}/fork")
def fork(session_id: str):
    session = session_or_404(session_id)
    return create_optimization(
        OptimizationRequest.model_validate(
            {
                **session["request"],
                "evaluation_dataset_mode": "generate_new",
            }
        )
    )


@router.get("/{session_id}/recommendation")
def recommendation(session_id: str):
    session = session_or_404(session_id)
    if not session["recommendation"]:
        return None
    tools = OptimizationTools(session_id)
    return {
        **tools.read("report.json"),
        "yaml": (tools.root / "recommended.yaml").read_text(encoding="utf-8"),
    }


@router.get("/{session_id}/export/{kind}")
def export(session_id: str, kind: str):
    session = session_or_404(session_id)
    if not session["recommendation"]:
        raise HTTPException(409, "Recommendation is not ready")
    names = {
        "yaml": ("recommended.yaml", "application/yaml"),
        "qa": ("qa.json", "application/json"),
        "report": ("report.json", "application/json"),
    }
    if kind not in names:
        raise HTTPException(404, "Unknown export")
    name, content_type = names[kind]
    return Response(
        (OptimizationTools(session_id).root / name).read_bytes(),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
