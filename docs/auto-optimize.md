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

Open **Auto Optimize** to browse the newest 100 runs. Search, sort, or page through
the history table and choose **View details** to inspect a saved run's configuration,
dataset, trials, recommendation, and activity. Older runs remain accessible through
their direct URLs. Run summaries include their full ID and created/updated dates.

Choose **New optimization**, select a benchmark template and documents/URLs, optionally
describe the audience and expected questions, and start. Defaults are 100 accepted
questions, 20 retrieval trials, five answer trials, two reserved confirmation
executions, 20 million conservatively reserved tokens, and two hours of active time.
The template's answer model stays fixed; its output is bounded to 1,024 tokens by
default. Worker-reachable model endpoints can be configured separately for planning,
synthesis, verification, and judging. Ollama models must already be installed.

### Reuse evaluation questions

**Evaluation questions** defaults to **Reuse matching questions when available**.
After capturing and extracting the sources, Auto Optimize selects the newest
compatible, validated frozen dataset. Matching requires the complete captured
source contents and document IDs, extracted text, audience, task guidance, example
questions, requested question count, and evaluation format/prompt version.
Pipeline settings, answer/judge models, trial budgets, and snapshot locations do
not affect matching. Changed file or URL contents and partial source-set overlap
require new questions. If no usable match exists, questions are generated normally.

Reuse copies the complete saved set, including reference answers, exact evidence,
question IDs, review flags, and tuning/holdout assignments. The original accepted
count is preserved; it is not topped up or truncated. Synthesis and semantic
verification calls are skipped, and those unused model endpoints need not be
available. Trials, judging, and confirmation still run for the new experiment.
The dataset card links to the originating experiment; dataset manifests, session
progress, and report exports retain dataset provenance. Reused holdouts are shared
evaluation data across experiments, not independent fresh confirmation samples.

Select **Generate fresh questions** to bypass reuse. **Start a new version** also
generates fresh questions. **Pause for dataset review** pauses reused datasets for
read-only inspection; resuming does not change inherited human-review flags.
Generate a fresh set to edit questions. Existing frozen datasets become eligible
automatically when their saved source, request, and validation artifacts establish
compatibility. Incomplete historical records and drafts are not reused.

Enable **Pause for dataset review** to inspect/edit the generated questions and
source quotes before the split is frozen. Fewer than 30 accepted questions, or an
inability to form independent evidence groups, requires input. Add/edit draft
questions or start a new version with more sources or task guidance. Frozen
datasets cannot be changed. The new-version action captures current inputs in a
new session rather than reusing an old holdout for tuning.

## What is measured

### Stage dashboards

Each optimization has three linked tabs. Share a view using
`/auto-optimize/{id}?stage=retrieval`, `?stage=answer`, or `?stage=confirm`.
The selected tab survives polling and reloads; active dashboards refresh every
three seconds and fetch a final update when the session finishes.

- **Retrieval / tuning** shows NDCG@10 selection scores, raw versus reranked
  NDCG/recall/precision/MRR at 3, 5, and 10, latency, coverage, and chunk counts.
  Trial rows show chunking units/size/overlap, embeddings, indexes, rewriting,
  retrieval, and reranking. New sessions keep context, compression, and prompts
  at baseline during screening.
- **Answering / tuning** shows correctness, claim/citation support, latency,
  token/context usage, and known estimated costs. Each trial identifies its source
  retrieval trial and its reranker, context, and prompt settings. Configuration
  inspection distinguishes inherited, varied, and fixed settings.
- **Confirmation / holdout** shows baseline/candidate paired metrics, correctness
  delta and its 95% bootstrap interval, coverage, configuration differences, and
  the confirmation outcome. It links back to the selected answer configuration.

Measured descriptive means remain visible even when a trial is not eligible for
ranking. Selection scores use matched questions with at least 95% coverage.
Missing values explain whether they are pending, unmeasured, lack historical data,
or have unknown pricing. A numeric zero remains a zero. Partial benchmark failures
display as **Completed with errors**; expand **View config** for all metrics and
selection/exclusion reasons.

### Retrieval shortlist and answer search

New sessions use workflow version 2. They shortlist up to two distinct feasible
retrieval setups, excluding a duplicate baseline. Setup identity includes
ingestion, rewriting, retrieval, and fixed filters; the highest-ranked
representative is retained when only downstream settings differ.

Answer evaluation runs the unchanged baseline first, then unchanged shortlisted
pipelines. Remaining slots test prompt variants, then bounded context/compression
presets, then approved rerankers, interleaving shortlisted parents. Without a
qualifying shortlist, variants use the baseline as their parent. Every candidate
preserves its parent's ingestion, rewriting, and retrieval settings. The persisted
queue is deterministic, deduplicated, and reused after recovery.

Prompt choices are **Baseline** (the original prompt), **Concise v1**, and
**Evidence-first v1**. All require context-grounded answers, exact citations, and
insufficient-context handling. Trial snapshots and exported YAML retain the preset
ID and resolved template. Custom and model-generated prompts are not supported.

The default answer-trial cap remains five. Search coverage lists scheduled, tested,
and untested variants; small caps can leave entire dimensions unexplored. No cap
is raised automatically. The winning configuration is the best observed complete
pipeline, never an untested combination of individually winning settings.

Confirmation evaluates that exact pipeline and the baseline on holdout, with one
execution if they are identical. Only confirmed improvement replaces the baseline
recommendation. Existing sessions retain their original workflow and saved queues;
historical dashboards infer source trials only when matching is unambiguous.

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
It also accepts `evaluation_dataset_mode`: `reuse_if_compatible` (the default,
including older saved requests) or `generate_new`.

Read `/optimizations/{id}` plus `/dataset`, `/trials`, `/events`, and
`/recommendation`. Use `PUT .../dataset` only while paused before freezing;
`POST .../pause`, `/resume`, `/cancel`, and `/fork` manage the lifecycle.
The OpenAPI schema documents the typed request contracts.

`GET /optimizations/{id}/stages` is a read-only dashboard summary. Its `stages`
mapping contains `retrieval`, `answer`, and `confirm`, with independent trial
metrics (`mean`, `valid_count`, `missing_count`, `reason`), matched-question
rankings, compact redacted configurations, coverage/error counts, selection roles,
and lineage. Answering includes the candidate plan and dispatch status;
confirmation includes the paired comparison and outcome. It reads saved results
in batches and never dispatches trials or changes recommendations. Workflow
version, lineage, and selection metadata use existing JSON storage; no database
migration is needed.

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
The dataset reuse migration test always covers SQLite. Set
`RAGBENCH_TEST_POSTGRES_URL` to a test database to also run it in a temporary,
isolated PostgreSQL schema; the schema is removed after the test.
