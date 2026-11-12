# Auto Optimize

Auto Optimize discovers the best **observed** configuration within an approved
component/model search space and budget. Manual Benchmark Setup continues to use
the same benchmark engine.

## Run it

Rebuild the API, worker, optimizer, scheduler, and UI with `docker compose up --build`.
The optimizer and scheduler are new services. The scheduler reconciles abandoned
sessions every minute; the optimizer consumes only the `optimization` queue, and
the benchmark worker consumes `ragbench`. Run only one scheduler per deployment.
Existing databases gain optimizer tables through the additive startup migration;
LangGraph creates its own versioned checkpoint tables on first use. To prepare
both explicitly, run `python -m scripts.migrate_optimization` with database access.

Open **Auto Optimize**, select a benchmark template and documents/URLs, optionally
describe the audience and expected questions, and start. Defaults are 100 accepted
questions, 20 retrieval trials, five answer trials, two reserved confirmation
executions, 20 million conservatively reserved tokens, and two hours of active time.
The template's answer model stays fixed; its output is bounded to 1,024 tokens by
default. Worker-reachable model endpoints can be configured separately for planning,
synthesis, verification, and judging. Ollama models must already be installed.

Enable **Pause for dataset review** to inspect/edit the generated questions and
source quotes before the split is frozen. Fewer than 30 accepted questions, or an
inability to form independent evidence groups, requires input. Add/edit draft
questions or start a new version with more sources or task guidance. Frozen
datasets cannot be changed. The new-version action captures current inputs in a
new session rather than reusing an old holdout for tuning.

## What is measured

- Sources, including URL response bytes, are captured once. Later edits cannot
  change a running experiment. Unreadable documents and identical extracted copies
  are excluded and reported.
- Reference answers require exact document evidence plus a semantic verifier.
  Data remains labeled synthetic; verification and human review are separate.
- Questions derived from the same fixed source window or overlapping evidence
  stay together. Splits target 80/20; indivisible evidence groups can change the
  exact ratio. Small holdouts have correspondingly weak statistical power.
- Retrieval screening ranks NDCG@10, then recall@10. End-to-end ranking uses
  semantic correctness with at least 95% matched observations. Differences below
  0.01 use latency and then known cost as tiebreakers.
- The planner sees tuning summaries, never holdout questions, references, or
  scores. Holdout is used once to compare the selected candidate with baseline.
  A positive paired-bootstrap interval and satisfied constraints confirm an
  improvement; otherwise baseline is retained and the candidate is provisional.
- Shared ingestion artifacts include corpus/configuration and component/dependency
  versions. Query variants reuse the built index. Reuse time is recorded separately
  from build time; measured trials disable retrieval-result caching.

## Budgets and providers

Token/cost reservations are conservative upper estimates, not provider invoices.
The ledger includes planner, synthesizer, verifier, benchmark, and judge work;
actual provider-reported model tokens are also retained when available. Interrupted
calls may have been billed, so retries reserve again. Unknown pricing is displayed
as unknown, never inferred as zero. Query costs exclude offline dataset/planning
work; semantic judge usage is recorded separately in benchmark traces.

For monetary caps, set input/output prices on every optimizer model role, including
explicit zero prices for local roles, and supply `runtime.pricing` for generation
input/output and embeddings in the template. Templates with hosted rerankers whose
request pricing cannot be modeled are unsuitable for monetary caps. A reservation
can stop search before the trial-count limit. A fixed reservation protects final
confirmation capacity; deadlines may still prevent confirmation after slow calls.

Hosted chat-completions adapters use a base URL ending in `/v1` (the adapter appends
`/chat/completions`) and an API key environment variable, defaulting to
`RAGBENCH_OPTIMIZER_API_KEY`. Supply that environment variable to API/worker/optimizer
containers. Never put credentials in the YAML or endpoint URL. Hosted source-data
opt-in also applies to hosted embedding providers in the template. Local Ollama
uses its `/api/generate` structured JSON interface; model output must conform to the
requested JSON schema.

Pause/cancel takes effect at stage/question boundaries and before expensive calls.
An already active call may finish within its timeout. Checkpoints, leases, trial
operation keys, model-response artifacts, and question-result upserts make recovery
safe against duplicate task delivery; in-flight external model calls have at-least-once
semantics. Preserve the shared artifact volume and Postgres data when restarting.

## APIs and exports

Each run page includes a **Run configuration** card with the saved template,
source selection, model roles, fixed answer model, limits, and component
restrictions. Expand the saved request/template YAML to inspect all settings.
These are the stored snapshots used by the run, so later template edits do not
change the display.

**Currently running configs** shows every running trial and its pipeline settings,
updated with trial status polling every three seconds while the session is active.
Queued and paused trials are not labeled running. Use **View config** in the
leaderboard to inspect any trial, including completed or failed trials, and expand
its YAML for runtime, metrics, and judge settings. Configuration views are read-only;
credential values are redacted and snapshot/orchestration internals are omitted.

`POST /optimizations` accepts `config_path`, optional `source_paths`, `urls`,
`audience`, `task`, `example_questions`, four model profiles, `limits`,
`allowed_components`, `allow_hosted`, and `review_before_freeze`.

Read `/optimizations/{id}` plus `/dataset`, `/trials`, `/events`, and
`/recommendation`. Use `PUT .../dataset` only while paused before freezing;
`POST .../pause`, `/resume`, `/cancel`, and `/fork` manage the lifecycle.
The OpenAPI schema documents the typed request contracts.

`GET /optimizations/{id}/configuration` returns `{request, template}` from the
stored session. `GET /optimizations/{id}/trials/{trial_id}/config` returns `{config}`
from a trial belonging to that session; missing or unrelated trials return 404.
These dedicated endpoints keep session/trial lists lightweight. They preserve
credential environment-variable references, redact inline credential values, and
omit `_snapshot` and `_optimization` metadata. A missing saved configuration is
returned as null and shown as unavailable.

Exports at `/export/yaml`, `/export/qa`, and `/export/report` contain the recommended
configuration, legacy-compatible QA JSON, and evidence/limitations report.
**Open in Benchmark Setup** loads the recommendation as an unsaved draft.
The YAML references its immutable server-side snapshot and is runnable on the same
artifact volume. Moving it to another installation requires copying the snapshot
and updating its paths, or replacing snapshot inputs with a new manual benchmark.

Tests use deterministic model/index fixtures and an in-memory LangGraph checkpointer
to cover discovery, replay, selection, and export without external services. The
browser acceptance test stubs the API. A deployment smoke test still needs actual
Postgres, Redis, Qdrant/OpenSearch, and the selected model endpoints.
