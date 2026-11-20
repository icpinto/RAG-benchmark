"""Compilation work uses the optimization queue; orchestration never waits here."""

import time

from ragbench_core.optimization import store
from ragbench_core.prompt_optimization.service import run_job
from sqlalchemy import select

from services.worker.app.celery_app import celery_app


@celery_app.task(name="prompt.compile", acks_late=True, reject_on_worker_lost=True)
def compile_prompt_job(job_id):
    store.db.init_db()
    run_job(job_id)
    from services.worker.app.optimization_tasks import advance_optimization

    with store.db.SessionLocal() as conn:
        job = conn.get(store.PromptCompilationJob, job_id)
        session_id = job.session_id
    if session_id:
        advance_optimization.apply_async(args=[session_id], queue="optimization")


def reconcile_prompt_jobs():
    with store.db.SessionLocal() as conn:
        jobs = [
            store.payload(row)
            for row in conn.execute(
                select(store.PromptCompilationJob).where(
                    store.PromptCompilationJob.status.in_(
                        ["queued", "running", "paused"]
                    ),
                    store.PromptCompilationJob.lease_until < time.time(),
                )
            ).scalars()
        ]
    for job in jobs:
        if job["session_id"] and store.get(job["session_id"])["status"] not in {
            "queued",
            "running",
            "waiting",
        }:
            continue
        compile_prompt_job.apply_async(args=[job["id"]], queue="optimization")
