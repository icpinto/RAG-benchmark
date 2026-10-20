# Contributing

Thanks for your interest in improving the RAG Benchmark Platform. This guide explains how to set up the project, run it locally, test changes, and prepare issues and pull requests.

## Development setup

The project targets Python 3.11 or newer and can be run either with Docker Compose or with a local Python environment.

1. Fork and clone the repository.
2. Create a local environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install the package in editable mode and add any component-specific dependencies you need for the providers you plan to test:

   ```bash
   python -m pip install -e .
   ```

4. Copy the sample environment file before running services:

   ```bash
   cp .env.example .env
   ```

Some optional components call external providers such as OpenAI, Voyage, Cohere, or Jina. Set the relevant API keys in your shell or `.env` only when you are testing those integrations.

## Run the app locally

The easiest way to run the complete stack is Docker Compose:

```bash
docker compose up --build
```

After the services start, the Streamlit UI is available at <http://localhost:8501> and FastAPI docs are available at <http://localhost:8000/docs>.

Ollama is included in the Compose stack. Pull the default model before running benchmarks that use local generation:

```bash
docker compose exec ollama ollama pull gpt-oss:20b
```

If you already run Ollama on your host, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` and use your host service instead of the bundled container. Recreate the worker with `docker compose up -d --no-deps --force-recreate worker` after changing its environment or host mapping. See [host Ollama setup and connectivity checks](README.md#use-ollama-on-the-host), including the Linux listener configuration.

For a direct sample run from Python, use:

```bash
python scripts/run_local.py
```

## Test commands

Run the full test suite before opening a pull request:

```bash
pytest
```

For targeted work, run the relevant test module, for example:

```bash
pytest tests/test_config_expander.py
pytest tests/test_retrieval_metrics.py
```

When you add or change behavior, include focused tests that cover the new code path. If a test requires external services or credentials, document the requirement clearly and keep the default test suite runnable without private secrets.

## Coding style

- Keep components small and registered through `ragbench_core.registry` rather than hard-coding special cases in the runner.
- Prefer typed, explicit interfaces from `ragbench_core.interfaces` and schema objects from `ragbench_core.models.schema`.
- Keep provider integrations configurable through YAML and environment variables.
- Avoid committing secrets, local datasets with private data, generated artifacts, or heavyweight model caches.
- Preserve deterministic behavior in tests by using local fakes, fixtures, or small sample inputs when possible.
- Update documentation when changing configuration keys, component behavior, service URLs, or local development workflows.

## Issues

Before filing an issue, search existing issues for related reports or proposals. A good issue includes:

- A concise title and description.
- Steps to reproduce, expected behavior, and actual behavior for bugs.
- Relevant config snippets, logs, stack traces, or screenshots.
- Environment details such as operating system, Python version, Docker version, and provider/model versions when applicable.

## Pull requests

Pull requests should be focused and reviewable. Please:

- Link related issues when applicable.
- Explain the motivation and user-visible impact.
- Include tests or a clear explanation when tests are not practical.
- Update README or configuration documentation for behavior changes.
- Keep unrelated formatting churn out of the diff.
- Confirm that `pytest` passes, or describe any environment limitation that prevented running it.

By participating in this project, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
