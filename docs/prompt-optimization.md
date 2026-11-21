# DSPy prompt optimization

DSPy is an optional workflow feature. Rebuild the API, worker, optimizer, and
scheduler after updating dependencies. Application startup creates the additive
`prompt_compilation_jobs`, `prompt_artifacts`, and `prompt_compilation_candidates`
tables; existing optimization sessions retain their saved workflow and answer queues.

## Shared configurations

Both Benchmark Setup and Auto Optimize accept the same configuration block:

```yaml
prompt_optimization:
  configs:
    - preset: mipro_zero_shot_v1
      seed: 42
      parameters:
        auto: null
        num_candidates: 4
        num_trials: 8
      limits:
        max_tokens: 2000000
        max_seconds: 900
        max_calls: 500
```

An empty `configs` list disables compilation. The UI initially selects
`mipro_zero_shot_v1` when enabling the feature. Presets are versioned:

| Preset | Optimizer | Changes |
| --- | --- | --- |
| `bootstrap_small_v1` | BootstrapFewShot | Up to two bootstrapped and two labeled examples |
| `mipro_zero_shot_v1` | MIPROv2 | Instructions, without examples |
| `mipro_few_shot_v1` | MIPROv2 | Instructions and example selection |
| `gepa_light_v1` | GEPA | Instructions, using judge feedback |

`GET /prompt-optimization/presets` returns the presets and typed settings schema.
Preset values and overrides are resolved and saved before execution. For MIPROv2,
explicit candidate/trial counts require `auto: null`. GEPA accepts exactly one
of `auto`, `max_metric_calls`, and `max_full_evals`; clear a preset's explicit
count when switching to another budget mode. Compilation always uses one thread.

Each recipe can set `proposal_model` using the existing optimizer ModelProfile
contract. Manual compilation defaults to the answer model; Auto Optimize defaults
to the session's planner profile. The answer model and parent retrieval pipeline
stay fixed. Monetary caps require explicit task, embedding, proposal, and judge
prices, including zero prices for local models. Hosted roles/components require
the source-data opt-in. Credentials use environment-variable references.

## Manual workflow

Open **Compile and reuse DSPy prompts** in Benchmark Setup, enable compilation,
and choose presets. Advanced JSON supports custom optimizer parameters, proposal
model profiles, and per-configuration budgets. Compilation saves the draft and
freezes the selected documents and labeled questions. If the draft expands into
multiple pipelines, select one case and compile again.

Questions are split approximately 60/20/20 into training, validation, and evaluation.
Shared evidence and duplicate questions stay together. Legacy QA files fall back
conservatively to shared source-document groups, requiring at least three
independent groups. Indivisible groups can change the exact split proportions.
Compilation requires reference answers and source document IDs/evidence groups.

Inspect the resulting instructions, examples, model settings, validation score,
and usage. Select **Apply prompt and evaluation partition**, then save and run the
benchmark. Applying loads the artifact's concrete pipeline and frozen evaluation
snapshot. Saved artifacts from either workflow can be selected without compiling
again; Auto Optimize artifacts use their originating holdout for manual replay.
Such replay is evaluation reuse, not independent fresh confirmation.

The compiled prompt embeds plain JSON with instructions, examples, a renderer
version, provenance, and checksum. Inference uses the ordinary benchmark generator
without importing DSPy. Changed artifact contents fail checksum validation.
Benchmark snapshots reject question IDs or normalized question texts used for
prompt training or validation. Internal answer-search trials explicitly permit
validation questions; training questions are excluded.

## Auto Optimize

Select recipes in the new optimization form, in execution order. API clients can
send `prompt_optimization` on `POST /optimizations`; omitting it inherits the
template settings, while an explicit empty list disables the feature.

New sessions use workflow version 5, with an adaptive outer experiment loop described
in [Auto Optimize](auto-optimize.md#adaptive-experiment-loop). Existing DSPy sessions
retain their workflow version and saved state. Version 5 disables reranking during
retrieval screening and selects retrieval winners using raw NDCG@10. Answering and
DSPy compilation restore the template reranker on the winning retrieval settings.
The frozen tuning partition is split 75/25 into prompt training and validation,
preserving evidence groups. The existing final holdout remains separate. Too few
independent tuning groups requires a fresh dataset/version with additional evidence.

In historical version-3 sessions, answer slots are allocated to the unchanged highest-ranked feasible retrieval
trial, selected DSPy configurations using that parent, then existing prompt,
context, and reranker variants. Compilation runs only for scheduled slots. The
answer-trial cap is never raised automatically. Failed compilation is recorded
and skipped; other scheduled candidates can still run within the remaining budget.

Version 4 evaluates the unchanged parent first, then lets the planner choose from
all approved candidates, including configured recipes. It may leave recipes untested
or stop early. Compilation is queued only when that recipe is selected. A successful
compilation receives an answer evaluation before another planner decision; a failed
compilation is saved as an observation. DSPy's internal optimization is unchanged.

All answer candidates are ranked on the same prompt-validation questions. Final
confirmation evaluates the exact selected pipeline against baseline on holdout.
Only confirmed improvement replaces the baseline recommendation. Compiled prompts
must come from an approved job; arbitrary generated prompt templates are rejected.

During compilation, the judge requests only the correctness score used by DSPy's
metric. Invalid judge responses receive one accounted retry. Auto Optimize prompt
jobs reuse the session's ingestion index; a transient Qdrant timeout during context
preparation receives one bounded retry.

The Answering dashboard shows compilation progress, artifacts, and offline usage.
Expand **View prompt suggestions** under **DSPy prompt compilation** to inspect
all captured instruction suggestions, example sets, and evaluated prompt
combinations. History becomes available after each configuration completes
successfully. Search the saved content or filter by configuration, then choose
**View prompt** for complete instructions, examples, and repeated evaluations.
Identical content is grouped under its first observation, retaining proposal
sources and all evaluation results. Instruction and example pools are shown
separately; combinations that DSPy never tried are not generated for inspection.

**Selected by DSPy** identifies the configuration's compiled output, separately
from answer-trial ranking and holdout confirmation. Optimizer scores use a 0–1
scale and identify training versus validation and full-set versus minibatch
coverage. Final artifact validation is shown separately. GEPA history includes
proposals rejected during minibatch screening. BootstrapFewShot selects examples
instead of proposing new instructions. Older jobs can show their saved output,
but display **Candidate history was not recorded** for discarded suggestions.
Interrupted or unsuccessful configurations do not publish candidate history.

Search coverage identifies scheduled and untested variants. Reports include
`prompt_compilations`, compilation usage by role, configuration lineage, and the
winning embedded prompt. Workflow-v3 recommendation YAML uses holdout questions
for replay; the QA export still contains the complete frozen dataset.

## Accounting and recovery

Compilation jobs run on the existing `optimization` queue. The orchestrator yields
while waiting, allowing deployments with one optimizer worker to make progress.
Leases and the existing scheduler reconcile abandoned jobs. Completed artifacts
are reused, including artifacts with identical content from another job.

Context preparation, task generation, proposals/reflection, and judging all reserve
usage before calls. Auto compilation also charges the parent session's ledger and
keeps confirmation capacity reserved. Longer compiled prompts must fit an enlarged
confirmation reservation before their answer trials run. Unknown pricing remains
unknown. Calls initially reserve conservative token estimates. Complete provider
input/output token counts replace those estimates after the call in both the job
and parent session ledgers, freeing unused token capacity for later trials. Missing
or incomplete usage keeps the estimate charged. Monetary reservations remain
conservative estimates, not provider invoices.

Cancel/pause checks happen at question and model-call boundaries; active calls can
finish within their timeout. Interrupted compilation restarts rather than resuming
DSPy's internal optimizer state. Previously reserved calls remain charged because
they may have reached the provider. Finished recipes in a multi-recipe job are not
compiled again.

## APIs and smoke tests

- `GET /prompt-optimization/cases?config_path=...`
- `POST /prompt-optimizations`: saved config path, optional source/question selection,
  concrete `case_id`, shared `prompt_optimization`, optional `judge`, and `allow_hosted`.
- `GET /prompt-optimizations/{id}`: status, splits, attempts, results, artifacts, usage.
- `GET /prompt-optimizations/{id}/candidates`: candidate summaries, history status,
  and pagination (`offset`, `limit` up to 100), with optional `config_index` and
  full-content search `q`.
- `GET /prompt-optimizations/{id}/candidates/{candidate_id}`: complete instructions,
  examples, evaluation history, and selected-output marker for one job's candidate.
- `POST /prompt-optimizations/{id}/cancel`
- `GET /prompt-artifacts` and `GET /prompt-artifacts/{id}`
- `GET /prompt-artifacts/{id}/config`: runnable evaluation configuration and YAML.

Tests cover real DSPy optimizer APIs using deterministic model transport, without
external model or vector services. To exercise local model endpoints with fixed
contexts and capped calls/tokens/time:

```powershell
python -m scripts.smoke_prompt_optimizers --base-url http://localhost:11434 --model qwen2.5:14b
```

The smoke test defaults to all three optimizers; `--preset` selects one, including
`mipro_few_shot_v1`. Results and prompt artifacts are written beneath
`artifacts/dspy-smoke`. It validates compilation and judging; deployment testing
with actual corpus ingestion, Postgres, Redis, and vector services remains separate.
