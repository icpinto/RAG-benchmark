from services.worker.app.celery_app import celery_app
from ragbench_core.utils.config import load_yaml
from ragbench_core.engine.runner import BenchmarkRunner

@celery_app.task(name="services.worker.app.tasks.run_benchmark_task")
def run_benchmark_task(run_id: str, config_path: str):
    config=load_yaml(config_path)
    return BenchmarkRunner().run(run_id, config)
