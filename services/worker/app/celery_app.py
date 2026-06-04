from celery import Celery
from ragbench_core.settings import settings

celery_app = Celery(
    "ragbench",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["services.worker.app.tasks"],
)
celery_app.conf.task_default_queue = "ragbench"
celery_app.conf.task_routes = {"services.worker.app.tasks.*": {"queue": "ragbench"}}
