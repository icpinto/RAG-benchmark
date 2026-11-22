"""Orchestration never occupies the benchmark worker while waiting for a trial."""

from __future__ import annotations

import time
import uuid

from ragbench_core.optimization import budget, store
from sqlalchemy import select

from services.worker.app.celery_app import celery_app


def dispatch_trial(run_id):
    run_optimization_trial.apply_async(args=[run_id], queue="ragbench")


@celery_app.task(
    name="optimization.advance", acks_late=True, reject_on_worker_lost=True
)
def advance_optimization(session_id):
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.types import Command
    from ragbench_core.engine.database import engine, init_db
    from ragbench_core.optimization.tools import OptimizationPreflightError
    from ragbench_core.optimization.workflow import build_graph

    init_db()
    owner = str(uuid.uuid4())
    if not store.acquire(session_id, owner):
        return
    try:
        session = store.get(session_id)
        if session["status"] in {
            "completed",
            "cancelled",
            "budget_exhausted",
            "failed",
            "paused",
            "needs_input",
        }:
            return
        uri = engine.url.set(drivername="postgresql").render_as_string(
            hide_password=False
        )
        with PostgresSaver.from_conn_string(uri) as saver:
            saver.setup()
            graph = build_graph(saver, dispatch_trial)
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 1000,
            }
            snapshot = graph.get_state(config)
            if snapshot.values:
                data = Command(resume=True) if snapshot.interrupts else None
            else:
                data = {"session_id": session_id, "stage": "profile"}
            with store.heartbeat(session_id, owner):
                graph.invoke(data, config, durability="sync")
    except budget.OptimizationStopped:
        pass
    except budget.BudgetExhausted as exc:
        store.update(session_id, status="budget_exhausted", error=str(exc))
    except OptimizationPreflightError as exc:
        store.update(session_id, status="failed", error=str(exc))
        import logging

        logging.getLogger(__name__).error("Optimization %s failed: %s", session_id, exc)
    except Exception as exc:  # noqa: BLE001 -- isolate provider/plugin/queue failures at the service boundary.
        store.update(
            session_id,
            status="failed",
            error=f"Optimization failed ({type(exc).__name__}); inspect worker logs",
        )
        # Preserve operational diagnostics without exposing provider secrets through the API.
        import logging

        logging.getLogger(__name__).error(
            "Optimization %s failed: %s", session_id, type(exc).__name__
        )
    finally:
        store.release(session_id, owner)


@celery_app.task(name="optimization.trial", acks_late=True, reject_on_worker_lost=True)
def run_optimization_trial(run_id):
    from ragbench_core.engine.runner import BenchmarkRunner
    from ragbench_core.optimization.runtime import IngestionReuse, backend_indexes_exist
    from ragbench_core.optimization.tools import OptimizationTools

    with store.db.SessionLocal() as conn:
        row = conn.get(store.OptimizationTrial, run_id)
        if row is None:
            return
        session_id = row.session_id
    owner = str(uuid.uuid4())
    with store.locked(session_id) as (conn, session):
        row = conn.get(store.OptimizationTrial, run_id)
        if (
            row.status in {"completed", "failed", "cancelled"}
            or (row.lease_until or 0) > time.time()
        ):
            return
        if session.status in {"paused", "needs_input", "cancelled"}:
            row.status = "cancelled" if session.status == "cancelled" else "paused"
            return
        row.status, row.lease_owner, row.lease_until = (
            "running",
            owner,
            time.time() + 300,
        )
        config = row.config
        row.attempts = (row.attempts or 0) + 1
        attempt = row.attempts
    started = time.monotonic()
    tools = OptimizationTools(session_id)
    operation = (
        config["_optimization"].get("budget_operation") if attempt == 1 else None
    )
    operation = operation or session_id + ":trial:" + run_id + ":" + owner
    try:
        budget.check_control(session_id)
        # A retry reserves again: interrupted provider calls can have been billed already.
        from ragbench_core.engine.data_loader import load_ground_truth

        examples = load_ground_truth(config["_snapshot"]["ground_truth_file"])
        estimate, cost, pricing_known = budget.trial_estimate(
            config,
            examples,
            tools.read("documents.json"),
            tools.session["request"]["judge"],
        )
        confirmation = config["_optimization"].get("split") == "holdout"
        protected = confirmation or bool(config["_optimization"].get("budget_credit"))
        budget.reserve(
            session_id,
            operation,
            "benchmark",
            estimate,
            cost,
            seconds=min(300, tools.session["request"]["limits"]["active_seconds"] // 8)
            if protected
            else 0,
            credit=config["_optimization"].get("budget_credit") or (session_id + ":holdout" if confirmation else None),
        )
        per_question = time.monotonic()

        def control(stage, item=None):
            nonlocal per_question
            budget.check_control(session_id)
            elapsed = time.monotonic() - started
            already = sum(
                r["seconds"]
                for r in store.rows(store.BudgetUsage, session_id)
                if r["id"] != operation
            )
            if (
                already + elapsed
                >= tools.session["request"]["limits"]["active_seconds"]
            ):
                raise budget.BudgetExhausted("Active runtime budget exhausted")
            with store.locked(session_id) as (conn, _):
                trial = conn.get(store.OptimizationTrial, run_id)
                if trial.lease_owner != owner:
                    raise budget.OptimizationStopped("Trial lease lost")
                trial.lease_until = time.time() + 300
            if stage == "question":
                per_question = time.monotonic()
            if stage == "question_finished":
                store.event(
                    session_id,
                    "question_completed",
                    trial_id=run_id,
                    question_id=item["question_id"],
                    seconds=time.monotonic() - per_question,
                )

        runner = BenchmarkRunner(control=control)
        runner.ingestion_reuse = IngestionReuse(
            tools.root / "ingestion",
            session_id,
            runner.registry,
            index_exists=backend_indexes_exist,
        )
        with store.heartbeat(session_id, owner, run_id):
            summary = runner.run(run_id, config)
        with store.locked(session_id) as (conn, _):
            row = conn.get(store.OptimizationTrial, run_id)
            row.status = (
                "completed"
                if store.db.get_run(run_id)["status"]
                in {"completed", "completed_with_errors"}
                else "failed"
            )
            row.summary = {**summary, "pricing_known": pricing_known}
    except budget.OptimizationStopped:
        with store.locked(session_id) as (conn, session):
            status = "cancelled" if session.status == "cancelled" else "paused"
            conn.get(store.OptimizationTrial, run_id).status = status
        store.db.upsert_run(run_id, "Auto Optimize", status, config)
    except budget.BudgetExhausted as exc:
        with store.locked(session_id) as (conn, session):
            row = conn.get(store.OptimizationTrial, run_id)
            row.status, row.summary = "failed", {"error": str(exc), "budget_exhausted": True}
            if not (session.progress.get("workflow_version", 1) >= 6 and row.phase == "prompt"):
                session.progress = {**session.progress, "stop_reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- isolate provider/plugin/queue failures at the service boundary.
        with store.locked(session_id) as (conn, _):
            row = conn.get(store.OptimizationTrial, run_id)
            row.status, row.summary = "failed", {"error_type": type(exc).__name__}
    finally:
        budget.settle(operation, seconds=time.monotonic() - started)
        with store.locked(session_id) as (conn, _):
            row = conn.get(store.OptimizationTrial, run_id)
            if row.lease_owner == owner:
                row.lease_owner, row.lease_until = None, 0
        if tools.session["status"] == "cancelled" and tools.session["progress"].get("workflow_version", 1) >= 4:
            from ragbench_core.optimization.adaptive import (
                capture_pending_terminal,
                stop_planning,
            )
            phase = config.get("_optimization", {}).get("split")
            if phase == "tuning":
                with store.db.SessionLocal() as conn:
                    trial_phase = conn.get(store.OptimizationTrial, run_id).phase
                capture_pending_terminal(tools, trial_phase)
            stop_planning(tools, "cancelled", "Session cancelled")
        advance_optimization.apply_async(args=[session_id], queue="optimization")


@celery_app.task(name="optimization.reconcile")
def reconcile_optimizations():
    """Beat redelivers abandoned work; leases and operation keys reject duplicates."""
    store.db.init_db()
    from services.worker.app.prompt_tasks import reconcile_prompt_jobs
    reconcile_prompt_jobs()
    with store.db.SessionLocal() as conn:
        sessions = [
            store.payload(s)
            for s in conn.execute(
                select(store.OptimizationSession).where(
                    store.OptimizationSession.status.in_(
                        ["queued", "running", "waiting"]
                    )
                )
            ).scalars()
        ]
    for session in sessions:
        if (session["lease_until"] or 0) > time.time():
            continue
        for trial in store.rows(store.OptimizationTrial, session["id"]):
            if (
                trial["status"] in {"queued", "running", "paused"}
                and (trial["lease_until"] or 0) < time.time()
            ):
                dispatch_trial(trial["id"])
        advance_optimization.apply_async(args=[session["id"]], queue="optimization")
