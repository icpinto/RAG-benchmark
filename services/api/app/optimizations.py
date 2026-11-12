"""Optimization API. Paths and provider policy are resolved before queueing."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from ragbench_core.engine.data_loader import list_sources
from ragbench_core.engine.snapshots import capture_corpus
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.contracts import DatasetEdit, OptimizationRequest
from ragbench_core.optimization.datasets import validate_questions
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
            }
        },
        "budget": budget.totals(session["id"]),
        "limits": session["request"]["limits"],
    }


def public_configuration(value):
    """Copy saved settings without credentials or execution-only metadata."""
    secret_keys = {
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "accesskey",
        "secretaccesskey",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "secretkey",
    }

    def secret(key):
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        return normalized in secret_keys or normalized.endswith(
            ("apikey", "authorization", "password", "secret", "privatekey")
        )

    def redact(child):
        if isinstance(child, str) and re.fullmatch(
            r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", child
        ):
            return child
        return "[REDACTED]" if child else child

    if isinstance(value, dict):
        return {
            key: redact(child) if secret(key) else public_configuration(child)
            for key, child in value.items()
            if key not in {"_snapshot", "_optimization"}
        }
    if isinstance(value, list):
        return [public_configuration(child) for child in value]
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlparse(value)
            query = parse_qsl(parsed.query, keep_blank_values=True)
            if parsed.username is not None or any(secret(key) for key, _ in query):
                return urlunparse(
                    parsed._replace(
                        netloc=parsed.netloc.rsplit("@", 1)[-1],
                        query=urlencode(
                            [
                                (key, redact(child) if secret(key) else child)
                                for key, child in query
                            ]
                        ),
                    )
                )
        except ValueError:
            return "[REDACTED]"
    return value


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
                    progress={},
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
    return store.dataset(session_id)


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
    return store.dataset(session_id)


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


@router.get("/{session_id}/events")
def get_events(session_id: str):
    session_or_404(session_id)
    return sorted(
        store.rows(store.OptimizationEvent, session_id),
        key=lambda e: (e["created_at"], e["id"]),
    )[-500:]


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
    if dataset and not dataset["frozen"] and session["stage"] == "freeze":
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
    session_or_404(session_id)
    store.update(session_id, status="cancelled")
    return {"status": "cancelled"}


@router.post("/{session_id}/fork")
def fork(session_id: str):
    session = session_or_404(session_id)
    return create_optimization(OptimizationRequest.model_validate(session["request"]))


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
