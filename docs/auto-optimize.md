# Auto Optimize

Auto Optimize discovers the best **observed** configuration within an approved
component/model search space and budget. Manual Benchmark Setup continues to use
the same benchmark engine.

## Adaptive experiment loop

New sessions and forks use workflow version 5. Existing sessions retain versions
1–4, including their saved queues and ranking policies. Startup creates the additive
`optimization_iterations` table; rebuild the API, worker, optimizer, scheduler,
and UI to activate the new workflow. No existing runs are converted.

Retrieval starts with the baseline retrieval settings and reranking disabled
(`rerankers: [{type: none}]`), even when the template configures a reranker.
Retrieval selection uses raw NDCG@10, with raw recall@10 as the recall tie-breaker.
Compression, context, and prompts remain at baseline throughout retrieval screening.
Reranking is available only during answering and final holdout evaluation.
Answering starts with the highest-ranked feasible retrieval configuration combined
with the template's complete reranker settings, or the original baseline fallback.
The original template baseline remains unchanged for holdout comparison. Each stage then
runs one configuration at a time: **run → persist observation → rebuild context →
planner decision → validate → next experiment or stop**. Seed retrieval configurations
are suggestions, rather than an automatically executed batch.

The answer planner chooses from the complete approved catalog of prompt presets,
context/compression variants, rerankers, and configured DSPy recipes. The trial cap
limits execution, not the catalog. Recipes can be skipped, and the planner can stop
either stage early. It cannot compose arbitrary answer pipelines. DSPy retains its
own internal optimization; successful compilation is evaluated before the next outer
decision. Failed compilation becomes feedback and is not retried automatically.

Each observation contains measured quality, coverage, feasibility, latency, known
cost, configuration changes, and matched-question deltas against the stage control
and best previous feasible run. Up to three bounded tuning examples include answers,
reference evidence, retrieval traces when available, and judge feedback. With DSPy,
outer answer observations use only prompt-validation questions. Holdout data never
enters planner context.

The model has no implicit conversation memory. Each request reconstructs the current
observation, compact attempted-run history, previous hypotheses, evidence-referenced
lessons, remaining budget, and allowed next actions. Serialized prompts, including
instructions, are limited to 64 KiB. Optional examples, lessons, and older detail are
removed first. Mandatory context that cannot fit pauses the session. Complete saved
observations and prompt/response snapshots remain available for inspection.

The planner gets one accounted repair/retry after an invalid decision or transport
failure. Two failures pause the session; **Resume** starts a new attempt cycle for
the same observation. No uninformed fallback configuration is dispatched. Each
experiment reserves a bounded planner call and repair attempt in addition to its
benchmark budget, preserving confirmation capacity; unused capacity is released.
At a trial cap, the final planner call can only reflect and stop. Cancellation or
exhausted budgets can prevent a call, but completed observations are retained.

Stage dashboards show **Experiment history** with observations, hypotheses, lessons,
decisions, failures, and stop reasons. **View planner context** exposes saved inputs
and responses. Search coverage distinguishes selected, executed, failed, and untested
answer candidates. Reports include iterations, planner prompt versions and usage,
stage stop reasons, and the complete answer catalog.

- `GET /optimizations/{id}/iterations?phase=retrieval&offset=0&limit=20`
  returns paginated summaries (`phase` is optional, `limit` is 1–100).
- `GET /optimizations/{id}/iterations/{iteration_id}` returns the full saved
  observation, decision, and attempt snapshots within that session.

Measured results remain authoritative. Existing ranking, coverage requirements,
fixed stages, trial caps, and exact-pipeline holdout confirmation still apply.

## Run it

Rebuild the API, worker, optimizer, scheduler, and UI with `docker compose up --build`.
The optimizer and scheduler are new services. The scheduler reconciles abandoned
sessions every minute; the optimizer consumes only the `optimization` queue, and
the benchmark worker consumes `ragbench`. Run only one scheduler per deployment.
Qdrant collection operations use a 30-second client timeout by default; set
`QDRANT_TIMEOUT_SECONDS` if collection creation needs more time on your deployment.
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

- **Retrieval / tuning** shows raw NDCG@10 selection scores, latency, coverage,
  and chunk counts. Additional measured retrieval metrics are available in each
  trial's inspector.
  Trial rows show chunking units/size/overlap, embeddings, indexes, rewriting,
  and retrieval. Reranking is disabled during retrieval; its settings are varied
  during answering and fixed for confirmation. New sessions keep context,
  compression, and prompts at baseline during screening. Historical sessions
  retain their saved stage annotations and reranked selection metrics. The
  duplicate raw-versus-reranked comparison table is omitted for all versions.
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

Historical answer queues in workflow versions 2 and 3 use the single highest-ranked
feasible retrieval trial as their parent, including when it differs from the
baseline only by reranker settings. Selection retains the existing NDCG@10
ranking, coverage requirements, constraints, and tie-breakers.

Answer evaluation runs that trial's unchanged pipeline first, even with an
answer-trial cap of one. Remaining slots test prompt variants, then bounded
context/compression presets, then approved rerankers, all derived from the same
parent. Without a feasible ranked retrieval trial, answering falls back to the
baseline and records the reason. Every candidate preserves its parent's
ingestion, rewriting, retrieval, and fixed filters. The baseline comparison runs
during holdout confirmation without consuming an independent answering slot.
Those persisted queues are deterministic, deduplicated, and reused after recovery.
Version 4 retains the same parent and approved variants, selecting subsequent
experiments adaptively as described above. Version 5 selects that parent using raw
retrieval metrics, restores the template reranker for the first answering trial,
and explores approved rerankers only in the answering stage. These stage boundaries
also apply to DSPy control and compilation configurations. No historical sessions
are converted and no database migration is required for version 5.

Without DSPy enabled, prompt choices are **Baseline** (the original prompt),
**Concise v1**, and
**Evidence-first v1**. All require context-grounded answers, exact citations, and
insufficient-context handling. Trial snapshots and exported YAML retain the preset
ID and resolved template. DSPy-enabled workflow-v3 sessions also compile approved
prompt configurations; see [DSPy prompt optimization](prompt-optimization.md) for
shared recipes, training/validation splits, compilation budgets, and manual reuse.

The default answer-trial cap remains five. Search coverage lists scheduled, tested,
and untested variants; small caps can leave entire dimensions unexplored. No cap
is raised automatically. The winning configuration is the best observed complete
pipeline, never an untested combination of individually winning settings.

Confirmation evaluates that exact pipeline and the baseline on holdout, with one
execution if they are identical. New sessions use recommendation policy version 2:

- **Confirmed improvement:** the candidate has higher mean holdout correctness,
  and the lower bound of its 95% paired-bootstrap interval is strictly above zero.
- **Promising candidate:** the candidate has higher mean holdout correctness, but
  its interval includes zero, so the improvement is not statistically confirmed.
- **Baseline retained:** the candidate ties or regresses, is identical to baseline,
  confirmation is unavailable, or a required acceptance check fails.

Both candidate outcomes recommend and export the exact tested candidate, including
**Open in Benchmark Setup**. They require completed baseline/candidate trials,
compatible datasets, finite paired correctness scores on at least 95% of holdout
questions, a valid interval, and compliance with configured latency/cost limits.
Missing measurements fail an applicable operational limit. Decisions use full
precision; rounded display values do not decide acceptance. Lower cost or latency
alone does not override equal or lower correctness.

The Recommendation section and confirmation dashboard show the same saved decision,
coverage, and failed or unavailable checks. A separate call using the planner model
explains the result in plain language after acceptance is decided. It receives only
bounded metrics, checks, and redacted configuration summaries, never questions,
references, or source passages. The model explains the decision; it cannot change
the winner or exported configuration. Its prompt is versioned and inspectable in
the session artifacts. Calls use the existing model budget and caching. If the
model fails, returns invalid output, or lacks budget, a labeled factual explanation
is used without discarding the recommendation. Reads and exports never make calls.

Existing sessions without a recommendation policy version keep the strict
confirmed-improvement-only acceptance rule; completed reports are not rewritten.
Existing sessions retain their saved queues and compilation plans;
version-1 sessions retain their original workflow. Historical dashboards infer
source trials only when matching is unambiguous.

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
- The search planner sees tuning summaries, never holdout questions, references,
  or scores. Holdout is used once to compare the selected candidate with baseline.
  Only the final explanation call sees holdout summary scores, after the decision
  is fixed; no subsequent tuning is performed using these results.
- Shared ingestion artifacts include corpus/configuration and component/dependency
  versions. Query variants reuse the built index. Reuse time is recorded separately
  from build time; measured trials disable retrieval-result caching.

## Budgets and providers

Calls reserve conservative token/cost estimates before execution. Complete provider
input/output token counts replace model-call token estimates at settlement, freeing
unused capacity. Missing or incomplete usage retains the token estimate; cost
reservations remain conservative estimates, not provider invoices. The ledger
includes planner, synthesizer, verifier, benchmark, and judge work. Interrupted
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

Each run page includes an expandable **Run configuration** card with the saved
template, source selection and guidance, model roles, fixed answer model, limits,
and component restrictions. It starts collapsed with a template and dataset
summary, and opens automatically when dataset review becomes required. A manual
collapse is preserved across polling updates; switching runs resets the view.
The **Evaluation dataset** subsection contains dataset provenance, version and
freeze status, question evidence, and draft editing tools. It remains available
if the saved configuration cannot be loaded. Expand the saved request/template
YAML to inspect all settings.
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
