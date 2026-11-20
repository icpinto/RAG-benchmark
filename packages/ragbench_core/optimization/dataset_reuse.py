"""Content-based identities and model-free checks for frozen evaluation datasets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from sqlalchemy import select

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization import store
from ragbench_core.optimization.datasets import (
    PROMPT_VERSION,
    freeze_split,
    validate_questions,
)

FORMAT_VERSION = "frozen-evaluation-v1"


def compatibility_key(request, corpus, documents):
    # corpus_hash covers ALL captured sources, including extraction failures and
    # duplicates subsequently excluded from the benchmark's usable manifest.
    if not corpus.get("corpus_hash") or not documents:
        raise ValueError("Dataset source identity is unavailable")
    return stable_hash(
        {
            "format_version": FORMAT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "corpus_hash": corpus["corpus_hash"],
            "documents": sorted(
                [
                    {"doc_id": d["doc_id"], "text_hash": stable_hash(d["text"])}
                    for d in documents
                ],
                key=lambda d: d["doc_id"],
            ),
            "audience": request.get("audience", ""),
            "task": request.get("task", ""),
            "example_questions": request.get("example_questions", []),
            "target_questions": request["limits"]["target_questions"],
        }
    )


def validate_frozen_payload(questions, documents):
    """Check evidence and original IDs/splits without altering saved review flags."""
    if not isinstance(questions, list) or len(questions) < 30:
        raise ValueError("Frozen dataset requires at least 30 validated questions")
    accepted, rejected = validate_questions(deepcopy(questions), documents)
    if rejected or len(accepted) != len(questions):
        raise ValueError("Saved questions or source evidence are invalid")
    if not all(q.automated_validated for q in accepted):
        raise ValueError("Saved questions lack semantic validation")
    expected = freeze_split(accepted)
    for saved, checked in zip(questions, expected):
        for field in ("question_id", "group_id", "split", "evidence"):
            if saved.get(field) != checked[field]:
                raise ValueError(f"Saved dataset has inconsistent {field}")


def generation_artifacts(session):
    root = (
        Path(session["template"]["runtime"]["artifact_dir"])
        / "optimizations"
        / session["id"]
    )

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    validation = read("dataset_validation.json")
    if validation.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Saved dataset uses an incompatible generation version")
    # Pre-reuse releases used this same frozen format without an explicit tag.
    if validation.get("format_version", "frozen-evaluation-v1") != FORMAT_VERSION:
        raise ValueError("Saved dataset uses an incompatible evaluation format")
    if not all(
        isinstance(validation.get(role), dict) for role in ("synthesizer", "verifier")
    ):
        raise ValueError("Saved dataset generation provenance is incomplete")
    return read("corpus.json"), read("documents.json"), validation


def index_historical_datasets():
    """Backfill metadata only; never rewrite historical questions or call models."""
    with store.db.SessionLocal() as conn:
        pending = [
            store.payload(row)
            for row in conn.execute(
                select(store.DatasetVersion).where(
                    store.DatasetVersion.frozen == 1,
                    store.DatasetVersion.compatibility_key.is_(None),
                    store.DatasetVersion.source_dataset_id.is_(None),
                )
            ).scalars()
        ]
    for dataset in pending:
        try:
            session = store.get(dataset["session_id"])
            corpus, documents, _ = generation_artifacts(session)
            validate_frozen_payload(dataset["payload"], documents)
            key = compatibility_key(session["request"], corpus, documents)
        except (ValueError, TypeError, KeyError, OSError, AttributeError):
            # Older/incomplete records remain readable but are not reusable.
            continue
        with store.db.SessionLocal.begin() as conn:
            row = conn.get(store.DatasetVersion, dataset["id"])
            if row and row.frozen and row.compatibility_key is None:
                row.compatibility_key = key


def compatible_datasets(key, session_id):
    index_historical_datasets()
    with store.db.SessionLocal() as conn:
        return [
            store.payload(row)
            for row in conn.execute(
                select(store.DatasetVersion)
                .join(
                    store.OptimizationSession,
                    store.OptimizationSession.id == store.DatasetVersion.session_id,
                )
                .where(
                    store.DatasetVersion.compatibility_key == key,
                    store.DatasetVersion.frozen == 1,
                    store.DatasetVersion.source_dataset_id.is_(None),
                    store.DatasetVersion.session_id != session_id,
                )
                .order_by(
                    store.OptimizationSession.created_at.desc(),
                    store.DatasetVersion.id.desc(),
                )
            ).scalars()
        ]
