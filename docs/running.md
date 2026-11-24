# Running the platform

This guide covers stack startup, model endpoints, benchmark execution, and
operational checks. See [Platform flow](platform-flow.md) for architecture and
[Auto Optimize](auto-optimize.md#first-run-workflow) for the optimization workflow.

## Start the stack

Install Docker with Docker Compose, then run these commands from the repository
root. Copy the sample environment only on first setup so existing settings survive:

```bash
cp .env.example .env
```

Configure [model endpoints](#model-endpoints), then build and start the stack:

```bash
docker compose up -d --build
docker compose ps
```

| Service | Host URL |
|---|---|
| React UI | http://localhost:8499 |
| FastAPI / OpenAPI | http://localhost:7800/docs |
| API liveness | http://localhost:7800/health |

Postgres, Redis, Qdrant, OpenSearch, and Ollama are internal to the Compose network;
their ports are not published to the host. A successful API liveness response does
not establish that search backends or model endpoints are ready for a benchmark.

The API, worker, optimizer, and scheduler wait for Postgres's TCP health check.
Other dependencies use startup ordering, so check their logs if a first run fails
while they are initializing. MinIO is excluded from normal startup by the `minio`
profile and is reserved for future object storage. Current artifacts use the shared
`artifacts/` directory; normal startup does not need the retained MinIO image.

### Workers and scheduler

Keep these three services running for Auto Optimize:

| Compose service | Work |
|---|---|
| `worker` | Consumes `ragbench`: manual benchmarks and optimizer benchmark trials through the same `BenchmarkRunner`. |
| `optimizer` | Consumes `optimization`: LangGraph orchestration, recovery tasks, and DSPy compilation. It yields while waiting for benchmark trials. |
| `scheduler` | Celery Beat enqueues optimizer reconciliation every 60 seconds. Run only one scheduler per deployment. |

Both workers default to concurrency one. Starting only the API and benchmark
worker is insufficient for optimization and prompt compilation. Preserve Postgres
data and the shared artifact directory across restarts for checkpoints and recovery.

On upgrades, rebuild the API, worker, optimizer, scheduler, and UI together with
`docker compose up -d --build`. Application tables are created additively at startup;
LangGraph prepares its versioned checkpoint tables on first use. To prepare both
explicitly after Postgres and the rebuilt API are available:

```bash
docker compose exec api python -m scripts.migrate_optimization
```

Existing optimizer sessions retain their saved workflow version.

## Model endpoints

The sample `.env.example` uses `OLLAMA_BASE_URL=http://100.110.250.66:11434` and
`OLLAMA_MODEL=qwen2.5:32b`. This remote address is deployment-specific; select an
endpoint reachable from your containers and install the selected models there.

Keep environment substitution in benchmark YAML so `.env` controls generation:

```yaml
llm:
  provider: ollama
  model: "${OLLAMA_MODEL:qwen2.5:32b}"
  base_url: "${OLLAMA_BASE_URL:http://100.110.250.66:11434}"
```

Auto Optimize's **planner**, **synthesizer**, **verifier**, and **judge** use
independent model profiles configured in **New optimization**, or in the API
request. Their defaults are Ollama at `http://100.110.250.66:11434` with
`qwen2.5:32b`; changing `OLLAMA_BASE_URL` does not update these profiles. Set each
role's endpoint and model explicitly when using another server. The answer model
comes from the template and stays fixed during optimization. If DSPy recipes
specify a proposal model, check that profile too.

The optimizer service must reach its model roles; the benchmark worker must reach
the answer and judge models plus any LLM-backed rewriters. Compatible dataset
reuse skips synthesis and verification calls. See [budgets and providers](auto-optimize.md#budgets-and-providers)
for hosted-provider opt-in, environment-variable credentials, and pricing.

### Use bundled Ollama

Set this in `.env` before starting or recreating the services:

```dotenv
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:32b
```

Once the containers are running, pull the model:

```bash
docker compose exec ollama ollama pull qwen2.5:32b
```

Set optimizer model profiles to `http://ollama:11434` as well. Pull any additional
models selected for those roles. The bundled endpoint is reachable inside Compose;
it is not exposed at `localhost:11434` on the host by default.

### Use Ollama on the host

If your machine already runs Ollama, set
`OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` and set optimizer
model profiles to that same URL. The worker and optimizer include a `host-gateway`
mapping so this hostname resolves on Linux Docker Engine as well as Docker Desktop.
The bundled Ollama service can remain unused.

Ensure the host Ollama listener accepts connections from Docker. On Linux Engine,
a listener bound only to `127.0.0.1:11434` cannot accept connections through the
bridge gateway; configure the host service's `OLLAMA_HOST` and restart it. See the
[Ollama server configuration guide](https://docs.ollama.com/faq#how-do-i-configure-ollama-server).

### Apply changes and check connectivity

After changing `.env` or Compose host mappings, recreate the affected application
services. A plain `docker compose restart` does not apply these changes:

```bash
docker compose up -d --no-deps --force-recreate api worker optimizer scheduler
```

Check the benchmark model endpoint from inside the worker:

```bash
docker compose exec worker python -c "import os, requests; r = requests.get(os.environ['OLLAMA_BASE_URL'].rstrip('/') + '/api/tags', timeout=10); r.raise_for_status(); print(r.json())"
```

For optimizer roles, run a check in the optimizer container using the actual URL
configured in the role profile. This example checks bundled Ollama:

```bash
docker compose exec optimizer python -c "import requests; r = requests.get('http://ollama:11434/api/tags', timeout=10); r.raise_for_status(); print(r.json())"
```

Check both reachability and that the response lists the selected model. Endpoint
changes apply to new runs; existing sessions use their stored configuration.

## Run a sample benchmark

After setup, open the UI and use **Knowledge Base** to upload documents and
ground-truth JSON or manage evaluation questions. The repository also includes
sample inputs under `data/`.

In **Benchmark Setup**, load `/app/configs/default.yaml`, select inputs, and choose
**Save and run**. The guided component forms and YAML editor stay synchronized.
Inspect status, scores, traces, retrieval heatmaps, and comparisons in **Results
Explorer**. For automated configuration search, follow the
[Auto Optimize first-run workflow](auto-optimize.md#first-run-workflow).

### API commands

The following examples use Bash/curl syntax. Submit a benchmark using a path
inside the container's config volume:

```bash
curl -X POST http://localhost:7800/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/default.yaml"}'
```

Replace `<run_id>` with the ID returned by submission:

```bash
curl 'http://localhost:7800/benchmarks/<run_id>'
curl 'http://localhost:7800/benchmarks/<run_id>/results'
```

List and inspect saved configurations:

```bash
curl http://localhost:7800/configs
curl 'http://localhost:7800/configs/content?config_path=/app/configs/default.yaml'
```

To save an edited configuration, send its full YAML text in `content`; the API
parses YAML before saving. For example, this creates a minimal draft to edit
further in Benchmark Setup before running:

```bash
curl -X PUT http://localhost:7800/configs/content \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"/app/configs/draft.yaml","content":"experiment:\n  name: edited_benchmark\n"}'
```

See [Auto Optimize APIs and exports](auto-optimize.md#apis-and-exports) for session
creation, lifecycle controls, trial inspection, and recommendation exports.

## Development

For frontend development, use Node 22.12 or later, run `npm ci` and `npm run dev`
in `services/ui`, then open http://localhost:5173. The development server proxies
`/api` to port 7800. See [React workspace development and checks](../services/ui/README.md)
for configuration, type checking, linting, and browser tests, and
[Contributing](../CONTRIBUTING.md) for Python setup and test commands.

After installing the Python package, `python scripts/run_local.py` invokes
`BenchmarkRunner` directly with `configs/default.yaml`. Host execution requires
host-reachable database, index, and model endpoints and writable data/artifact
paths; Compose service names and `/app/...` paths do not resolve on the host by
default. Copying `.env` alone does not configure this script's process environment.

## Troubleshooting

| Symptom | Check |
|---|---|
| API does not start / Postgres is unhealthy | Run `docker compose ps` and `docker compose logs postgres api`; application startup waits for Postgres's TCP health check. |
| Model connection refused or model missing | Check the selected endpoint from the worker and optimizer as above; verify the host listener and install models on the server actually used by each profile. |
| Optimization or compilation does not advance | Run `docker compose logs optimizer scheduler worker`; ensure both queues have their workers and exactly one scheduler is running. Paused or needs-input sessions require the action shown in the UI. |
| Qdrant collection operations time out | Check Qdrant logs; the client timeout defaults to 30 seconds. Set `QDRANT_TIMEOUT_SECONDS` in `.env` and recreate application services if more time is needed. |
| Restart ignores environment edits | Recreate affected services rather than using `docker compose restart`; start a new run for changed saved model settings. |

For optimizer budgets, cancellation boundaries, and recovery guarantees, see
[Auto Optimize budgets and providers](auto-optimize.md#budgets-and-providers).
