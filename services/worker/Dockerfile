FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements/base.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY packages /app/packages
COPY services /app/services
ENV PYTHONPATH=/app/packages:/app
CMD ["celery", "-A", "services.worker.app.celery_app", "worker", "--loglevel=INFO"]
