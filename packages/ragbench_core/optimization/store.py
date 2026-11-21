"""SQL system of record; graph checkpoints hold references, not source text."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)

from ragbench_core.engine import database as db


class OptimizationSession(db.Base):
    __tablename__ = "optimization_sessions"
    id = Column(String, primary_key=True)
    status = Column(String, default="queued")
    stage = Column(String, default="profile")
    request = Column(JSON)
    template = Column(JSON)
    progress = Column(JSON, default=dict)
    recommendation = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
    lease_owner = Column(String, nullable=True)
    lease_until = Column(Float, default=0)


class DatasetVersion(db.Base):
    __tablename__ = "optimization_datasets"
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True, unique=True)
    version = Column(Integer, default=1)
    frozen = Column(Integer, default=0)
    payload = Column(JSON)
    compatibility_key = Column(String, index=True, nullable=True)
    source_dataset_id = Column(String, nullable=True)


class OptimizationTrial(db.Base):
    __tablename__ = "optimization_trials"
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    operation_key = Column(String)
    phase = Column(String)
    split = Column(String)
    config = Column(JSON)
    status = Column(String, default="queued")
    summary = Column(JSON, default=dict)
    created_at = Column(Float, default=time.time)
    lease_owner = Column(String, nullable=True)
    lease_until = Column(Float, default=0)
    attempts = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint("session_id", "operation_key"),)


class OptimizationEvent(db.Base):
    __tablename__ = "optimization_events"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)
    created_at = Column(Float, default=time.time)
    kind = Column(String)
    payload = Column(JSON)


class BudgetUsage(db.Base):
    __tablename__ = "optimization_usage"
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)
    tokens = Column(Integer, default=0)
    cost = Column(Float, nullable=True)
    seconds = Column(Float, default=0)
    status = Column(String, default="reserved")
    actual_tokens = Column(Integer, nullable=True)


class PromptCompilationJob(db.Base):
    __tablename__ = "prompt_compilation_jobs"
    id = Column(String, primary_key=True)
    operation_key = Column(String, unique=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    status = Column(String, default="queued")
    request = Column(JSON)
    config = Column(JSON)
    splits = Column(JSON)
    progress = Column(JSON, default=dict)
    error = Column(String, nullable=True)
    attempts = Column(Integer, default=0)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
    lease_owner = Column(String, nullable=True)
    lease_until = Column(Float, default=0)


class PromptArtifact(db.Base):
    __tablename__ = "prompt_artifacts"
    id = Column(String, primary_key=True)
    job_id = Column(String, index=True)
    payload = Column(JSON)
    summary = Column(JSON)
    created_at = Column(Float, default=time.time)


class PromptCandidate(db.Base):
    __tablename__ = "prompt_compilation_candidates"
    id = Column(String, primary_key=True)
    job_id = Column(String, index=True, nullable=False)
    config_index = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False)
    proposal_order = Column(Integer, nullable=False)
    summary = Column(JSON, nullable=False)
    payload = Column(JSON, nullable=False)
    search_text = Column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("job_id", "config_index", "attempt", "proposal_order"),)


def budget_owner(owner_id):
    """Compilation jobs share accounting without pretending to be auto sessions."""
    with db.SessionLocal() as conn:
        row = conn.get(OptimizationSession, owner_id)
        if row is None:
            row = conn.get(PromptCompilationJob, owner_id)
        if row is None:
            raise KeyError("Budget owner not found")
        return payload(row)


@contextmanager
def locked_budget_owner(owner_id):
    with db.SessionLocal.begin() as conn:
        row = conn.execute(select(OptimizationSession).where(OptimizationSession.id == owner_id).with_for_update()).scalar_one_or_none()
        if row is None:
            row = conn.execute(select(PromptCompilationJob).where(PromptCompilationJob.id == owner_id).with_for_update()).scalar_one()
        yield conn, row
        row.updated_at = time.time()


def payload(row):
    return {c.name: deepcopy(getattr(row, c.name)) for c in row.__table__.columns}


def get(session_id):
    with db.SessionLocal() as conn:
        row = conn.get(OptimizationSession, session_id)
        if row is None:
            raise KeyError("Optimization session not found")
        return payload(row)


@contextmanager
def locked(session_id):
    with db.SessionLocal.begin() as conn:
        row = conn.execute(
            select(OptimizationSession)
            .where(OptimizationSession.id == session_id)
            .with_for_update()
        ).scalar_one()
        yield conn, row
        row.updated_at = time.time()


def update(session_id, **changes):
    with locked(session_id) as (_, row):
        for key, value in changes.items():
            setattr(row, key, deepcopy(value))


def event(session_id, kind, **data):
    with db.SessionLocal.begin() as conn:
        conn.add(OptimizationEvent(session_id=session_id, kind=kind, payload=data))


def rows(model, session_id):
    with db.SessionLocal() as conn:
        return [
            payload(r)
            for r in conn.execute(
                select(model).where(model.session_id == session_id)
            ).scalars()
        ]


def dataset(session_id):
    items = rows(DatasetVersion, session_id)
    return items[0] if items else None


def save_dataset(
    session_id,
    questions,
    frozen=False,
    *,
    compatibility_key=None,
    source_dataset_id=None,
):
    with locked(session_id) as (conn, _):
        row = conn.execute(
            select(DatasetVersion).where(DatasetVersion.session_id == session_id)
        ).scalar_one_or_none()
        if row and row.frozen:
            raise ValueError("Dataset is frozen; fork the session to change it")
        if row is None:
            row = DatasetVersion(id=str(uuid.uuid4()), session_id=session_id, version=0)
            conn.add(row)
        row.version = (row.version or 0) + 1
        row.payload = deepcopy(questions)
        row.frozen = int(frozen)
        row.compatibility_key = compatibility_key if frozen else None
        row.source_dataset_id = source_dataset_id if frozen else None


def dataset_provenance(session_id):
    current = dataset(session_id)
    if current is None:
        return None
    origin_id = current["source_dataset_id"] or current["id"]
    with db.SessionLocal() as conn:
        origin = conn.get(DatasetVersion, origin_id)
        return {
            "status": "reused" if current["source_dataset_id"] else "generated",
            "dataset_id": current["id"],
            "source_dataset_id": origin_id,
            "source_session_id": origin.session_id if origin else None,
            "question_count": len(current["payload"]),
        }


def record_dataset_provenance(session_id):
    provenance = dataset_provenance(session_id)
    with locked(session_id) as (_, row):
        row.progress = {**(row.progress or {}), "evaluation_dataset": provenance}
    return provenance


def acquire(session_id, owner, seconds=300):
    with locked(session_id) as (_, row):
        if row.lease_until and row.lease_until > time.time():
            return False
        row.lease_owner, row.lease_until = owner, time.time() + seconds
        return True


def release(session_id, owner):
    with locked(session_id) as (_, row):
        if row.lease_owner == owner:
            row.lease_owner, row.lease_until = None, 0


@contextmanager
def heartbeat(session_id, owner, trial_id=None):
    """Renew leases during long extraction, model loading, and index builds."""
    stopped = threading.Event()

    def renew():
        while not stopped.wait(30):
            try:
                with locked(session_id) as (conn, session):
                    row = conn.get(OptimizationTrial, trial_id) if trial_id else session
                    if row is None or row.lease_owner != owner:
                        return
                    row.lease_until = time.time() + 300
            except Exception:  # noqa: BLE001 -- failure leaves the lease to expire and reconciliation retries.
                return

    thread = threading.Thread(target=renew, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
