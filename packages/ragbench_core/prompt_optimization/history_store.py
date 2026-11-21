"""Durable candidate inspection, independent of deduplicated inference artifacts."""

import json
from copy import deepcopy

from sqlalchemy import select

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization import store
from ragbench_core.prompt_optimization.service import TERMINAL, artifacts_for_job


def candidate_summary(candidate):
    return {
        key: deepcopy(value)
        for key, value in candidate.items()
        if key not in {"instructions", "demonstrations"}
    } | {
        "instructions_preview": (candidate.get("instructions") or "")[:160],
        "demonstration_count": (
            len(candidate["demonstrations"])
            if candidate.get("demonstrations") is not None
            else None
        ),
    }


def save_history(conn, job_id, config_index, attempt, preset, history):
    for candidate in history["candidates"]:
        identity = stable_hash([job_id, config_index, attempt, candidate["id"]])
        payload = {
            **deepcopy(candidate),
            "id": identity,
            "config_index": config_index,
            "attempt": attempt,
            "preset": preset,
            "historical": False,
        }
        conn.add(
            store.PromptCandidate(
                id=identity,
                job_id=job_id,
                config_index=config_index,
                attempt=attempt,
                proposal_order=payload["proposal_order"],
                summary=candidate_summary(payload),
                payload=payload,
                search_text=json.dumps(payload, ensure_ascii=False),
            )
        )


def history_status(job):
    results = job["progress"].get("results", [])
    return {
        "available": any("candidate_history" in result for result in results),
        "candidate_count": sum(
            result.get("candidate_history", {}).get("count", 0) for result in results
        ),
        "pending": job["status"] not in TERMINAL,
        "configurations": [
            {
                "index": result["index"],
                "preset": result.get("preset"),
                "status": result["status"],
                **result.get("candidate_history", {}),
            }
            for result in results
        ],
    }


def historical_candidates(job):
    """Only saved outputs can be recovered for pre-history jobs/configurations."""
    legacy = [
        result
        for result in job["progress"].get("results", [])
        if result.get("artifact_id") and "candidate_history" not in result
    ]
    if not legacy:
        return []
    artifacts = {artifact["id"]: artifact for artifact in artifacts_for_job(job["id"])}
    candidates = []
    for result in legacy:
        artifact = artifacts.get(result["artifact_id"])
        if artifact is None:
            continue
        payload = artifact["payload"]
        candidates.append(
            {
                "id": f"artifact:{result['index']}:{artifact['id']}",
                "config_index": result["index"],
                "attempt": None,
                "preset": result.get("preset"),
                "proposal_order": 1,
                "kind": "prompt",
                "instructions": payload["instructions"],
                "demonstrations": payload.get("demonstrations", []),
                "sources": ["saved_output"],
                "selected": True,
                "historical": True,
                "evaluations": [
                    {
                        "stage": "final_validation",
                        "partition": "validation",
                        "scope": "full",
                        "sample_count": result.get("validation_count"),
                        "score": result.get(
                            "validation_score",
                            artifact["summary"].get("validation_score"),
                        ),
                    }
                ],
            }
        )
    return candidates


def list_candidates(job, *, config_index=None, offset=0, limit=25, query=""):
    statement = select(store.PromptCandidate.summary).where(
        store.PromptCandidate.job_id == job["id"]
    )
    if config_index is not None:
        statement = statement.where(store.PromptCandidate.config_index == config_index)
    if query:
        statement = statement.where(
            store.PromptCandidate.search_text.icontains(query, autoescape=True)
        )
    statement = statement.order_by(
        store.PromptCandidate.config_index, store.PromptCandidate.proposal_order
    )
    with store.db.SessionLocal() as conn:
        summaries = [deepcopy(value) for value in conn.execute(statement).scalars()]
    summaries.extend(
        candidate_summary(candidate)
        for candidate in historical_candidates(job)
        if (config_index is None or candidate["config_index"] == config_index)
        and (
            not query
            or query.casefold() in json.dumps(candidate, ensure_ascii=False).casefold()
        )
    )
    summaries.sort(
        key=lambda candidate: (candidate["config_index"], candidate["proposal_order"])
    )
    return {
        "items": summaries[offset : offset + limit],
        "total": len(summaries),
        "offset": offset,
        "limit": limit,
        "history": history_status(job),
    }


def get_candidate(job, candidate_id):
    with store.db.SessionLocal() as conn:
        row = conn.get(store.PromptCandidate, candidate_id)
        if row is not None and row.job_id == job["id"]:
            return deepcopy(row.payload)
    return next(
        (item for item in historical_candidates(job) if item["id"] == candidate_id),
        None,
    )
