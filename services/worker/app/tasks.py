from services.worker.app.celery_app import celery_app
from ragbench_core.engine.database import get_run_config, upsert_run
from ragbench_core.engine.runner import BenchmarkRunner

@celery_app.task(name="services.worker.app.tasks.run_benchmark_task")
def run_benchmark_task(run_id: str):
    config = get_run_config(run_id)
    try:
        return BenchmarkRunner().run(run_id, config)
    except Exception as exc:
        upsert_run(run_id, config.get("experiment", {}).get("name", "benchmark"), "failed", config, error=str(exc))
        raise
