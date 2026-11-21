"""Persistent compilation jobs and conservative accounting for every external call."""

import json
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from ragbench_core.engine.data_loader import load_ground_truth
from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization import budget, store
from ragbench_core.optimization.profiles import ModelProfile
from ragbench_core.prompt_optimization.artifacts import CompiledPrompt
from ragbench_core.prompt_optimization.contracts import (
    PromptOptimizationConfig,
    PromptOptimizationSettings,
)
from ragbench_core.prompt_optimization.datasets import question_fingerprint

TERMINAL = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "failed",
    "budget_exhausted",
}


def get_job(job_id):
    with store.db.SessionLocal() as conn:
        row = conn.get(store.PromptCompilationJob, job_id)
        if row is None:
            raise KeyError("Prompt compilation job not found")
        return store.payload(row)


def update_job(job_id, **changes):
    with store.locked_budget_owner(job_id) as (_, row):
        for key, value in changes.items():
            setattr(row, key, deepcopy(value))


def artifacts_for_job(job_id):
    with store.db.SessionLocal() as conn:
        job = conn.get(store.PromptCompilationJob, job_id)
        ids = (
            [
                r["artifact_id"]
                for r in (job.progress or {}).get("results", [])
                if r.get("artifact_id")
            ]
            if job
            else []
        )
        return [
            store.payload(a)
            for a in conn.execute(
                select(store.PromptArtifact).where(
                    or_(
                        store.PromptArtifact.job_id == job_id,
                        store.PromptArtifact.id.in_(ids),
                    )
                )
            ).scalars()
        ]


def create_job(
    config,
    settings,
    splits,
    judge,
    *,
    allow_hosted=False,
    session_id=None,
    operation_key=None,
):
    settings = PromptOptimizationSettings.model_validate(settings)
    if not settings.configs:
        raise ValueError("Select at least one prompt optimizer config")
    judge = ModelProfile.model_validate(judge)
    if judge.provider != "ollama" and not allow_hosted:
        raise ValueError("Hosted judging requires source-data opt-in")
    settings.validate_policy(allow_hosted)
    for recipe in settings.configs:
        recipe.proposal_model = recipe.proposal_model or task_profile(config)
    request = {
        "prompt_optimization": settings.model_dump(),
        "judge": judge.model_dump(),
        "allow_hosted": allow_hosted,
        "limits": {
            "max_tokens": sum(c.limits.max_tokens for c in settings.configs),
            "active_seconds": sum(c.limits.max_seconds for c in settings.configs),
            "max_cost": sum(c.limits.max_cost for c in settings.configs)
            if all(c.limits.max_cost is not None for c in settings.configs)
            else None,
        },
    }
    for recipe in settings.configs:
        if recipe.limits.max_cost is not None:
            profiles = [judge, recipe.proposal_model or task_profile(config)]
            if any(
                p.input_price_per_million is None or p.output_price_per_million is None
                for p in profiles
            ):
                raise ValueError(
                    "Monetary caps require explicit task, proposal, and judge prices"
                )
            if any(
                k not in config.get("runtime", {}).get("pricing", {})
                for k in (
                    "generation_input_per_1m",
                    "generation_output_per_1m",
                    "embedding_per_1m",
                )
            ):
                raise ValueError(
                    "Monetary caps require runtime.pricing for generation and embeddings"
                )
    key = operation_key or str(uuid.uuid4())
    with store.db.SessionLocal() as conn:
        existing = conn.execute(
            select(store.PromptCompilationJob).where(
                store.PromptCompilationJob.operation_key == key
            )
        ).scalar_one_or_none()
        if existing:
            return existing.id
    job_id = str(uuid.uuid4())
    try:
        with store.db.SessionLocal.begin() as conn:
            conn.add(
                store.PromptCompilationJob(
                    id=job_id,
                    operation_key=key,
                    session_id=session_id,
                    request=request,
                    config=deepcopy(config),
                    splits=deepcopy(splits),
                    progress={"results": []},
                )
            )
    except IntegrityError:
        with store.db.SessionLocal() as conn:
            return conn.execute(
                select(store.PromptCompilationJob.id).where(
                    store.PromptCompilationJob.operation_key == key
                )
            ).scalar_one()
    return job_id


def task_profile(config):
    llm = config["llm"]
    prices = config.get("runtime", {}).get("pricing", {})
    return ModelProfile.model_validate(
        {
            **{k: v for k, v in llm.items() if k in ModelProfile.model_fields},
            "input_price_per_million": prices.get("generation_input_per_1m"),
            "output_price_per_million": prices.get("generation_output_per_1m"),
        }
    )


class CompilationAccount:
    def __init__(self, job_id, index, owner):
        self.job_id, self.index, self.owner = job_id, index, owner
        job = get_job(job_id)
        self.recipe = PromptOptimizationConfig.model_validate(
            job["request"]["prompt_optimization"]["configs"][index]
        )
        self.session_id = job["session_id"]
        self.started = time.monotonic()
        self.prior_seconds = sum(r["seconds"] or 0 for r in self.rows())

    def rows(self):
        prefix = f"{self.job_id}:config:{self.index}:"
        return [
            r
            for r in store.rows(store.BudgetUsage, self.job_id)
            if r["id"].startswith(prefix)
        ]

    def check(self):
        job = get_job(self.job_id)
        if job["lease_owner"] != self.owner:
            raise budget.OptimizationStopped("Compilation lease lost")
        budget.check_control(self.job_id)
        if self.session_id:
            budget.check_control(self.session_id)
        if (
            self.prior_seconds + time.monotonic() - self.started
            >= self.recipe.limits.max_seconds
        ):
            raise budget.BudgetExhausted("Prompt configuration runtime limit reached")
        with store.locked_budget_owner(self.job_id) as (_, row):
            if row.lease_owner == self.owner:
                row.lease_until = time.time() + 300

    @contextmanager
    def call(self, role, prompt, profile, *, tokens=None, cost=None):
        self.check()
        rows = self.rows()
        limits = self.recipe.limits
        input_tokens = len(prompt.encode("utf-8")) + 256
        tokens = (
            tokens if tokens is not None else input_tokens + profile.max_output_tokens
        )
        if (
            cost is None
            and profile.input_price_per_million is not None
            and profile.output_price_per_million is not None
        ):
            cost = (
                input_tokens * profile.input_price_per_million
                + profile.max_output_tokens * profile.output_price_per_million
            ) / 1_000_000
        if (
            len(rows) >= limits.max_calls
            or sum(r["tokens"] for r in rows) + tokens > limits.max_tokens
        ):
            raise budget.BudgetExhausted(
                "Prompt configuration call/token limit reached"
            )
        if limits.max_cost is not None and (
            cost is None
            or any(r["cost"] is None for r in rows)
            or sum(r["cost"] or 0 for r in rows) + cost > limits.max_cost
        ):
            raise budget.BudgetExhausted("Prompt configuration monetary limit reached")
        operation = f"{self.job_id}:config:{self.index}:" + str(uuid.uuid4())
        timeout = min(
            profile.timeout,
            max(
                0,
                limits.max_seconds
                - self.prior_seconds
                - (time.monotonic() - self.started),
            ),
        )
        if self.session_id:
            budget.reserve(
                self.session_id,
                operation + ":session",
                role,
                tokens,
                cost,
                seconds=timeout,
            )
        budget.reserve(self.job_id, operation, role, tokens, cost, seconds=timeout)
        usage, started = {}, time.monotonic()
        try:
            yield usage
        finally:
            reported = budget.reported_token_total(usage)
            seconds = time.monotonic() - started
            budget.settle(
                operation, actual_tokens=reported, seconds=seconds
            )
            if self.session_id:
                budget.settle(
                    operation + ":session",
                    actual_tokens=reported,
                    seconds=seconds,
                )


def ingestion_root_for_job(job):
    artifact_dir = Path(job["config"]["runtime"]["artifact_dir"])
    if job["session_id"]:
        return artifact_dir / "optimizations" / job["session_id"] / "ingestion"
    return artifact_dir / "prompt_optimizations" / job["id"] / "ingestion"


def load_or_build_ingestion(reuse, runner, case, config, source_paths, account):
    from qdrant_client.http.exceptions import ResponseHandlingException

    for attempt in range(2):
        try:
            ingestion = reuse.load(case, config)
            if ingestion is None:
                config["_index_name"] = reuse.index_name(case, config)
                ingestion = runner._ingest_case(case, source_paths, config)
                reuse.save(case, config, ingestion)
            return ingestion
        except ResponseHandlingException:
            if attempt == 1:
                raise
            account.check()
            time.sleep(1)


def prepare_examples(job, account):
    """Materialize retrieval contexts once, using the shared benchmark path."""
    from ragbench_core.engine.cache import build_retrieval_cache
    from ragbench_core.engine.config_expander import expand_cases
    from ragbench_core.engine.runner import BenchmarkRunner
    from ragbench_core.engine.snapshots import verify_snapshot
    from ragbench_core.optimization.runtime import IngestionReuse, backend_indexes_exist

    config = deepcopy(job["config"])
    verify_snapshot(config)
    case = expand_cases(config)[0]
    all_examples = load_ground_truth(config["_snapshot"]["ground_truth_file"])
    allowed = set(job["splits"]["training"] + job["splits"]["validation"])
    examples = [e for e in all_examples if e.question_id in allowed]
    if len(examples) != len(allowed):
        raise ValueError("Compilation partition does not match its frozen snapshot")
    runner = BenchmarkRunner(control=lambda *args: account.check())
    root = Path(config["runtime"]["artifact_dir"]) / "prompt_optimizations" / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    # Reserve the complete preparation before index, embedding or rewriting calls.
    retrieval_config = deepcopy(config)
    retrieval_config.setdefault("runtime", {})["mode"] = "retrieval_only"
    documents = [
        {"text": Path(s["path"]).read_bytes().decode("utf-8", errors="replace")}
        for s in config["_snapshot"]["sources"]
    ]
    tokens, cost, _ = budget.trial_estimate(
        retrieval_config, examples, documents, job["request"]["judge"]
    )
    with account.call(
        "prompt_context", "", task_profile(config), tokens=tokens, cost=cost
    ):
        reuse = IngestionReuse(
            ingestion_root_for_job(job),
            job["session_id"] or job["id"],
            runner.registry,
            index_exists=backend_indexes_exist,
        )
        ingestion = load_or_build_ingestion(
            reuse,
            runner,
            case,
            config,
            [s["path"] for s in config["_snapshot"]["sources"]],
            account,
        )
        caches = {
            "embedding": ingestion["embedding_cache"],
            "retrieval": build_retrieval_cache(
                {
                    **config,
                    "runtime": {
                        **config["runtime"],
                        "retrieval_cache": {"enabled": False},
                    },
                }
            ),
        }
        prepared = []
        for example in examples:
            account.check()
            payload = runner.prepare_context(example, case, ingestion, caches)
            prepared.append(
                {
                    **asdict(example),
                    "context": payload["context"],
                    "context_chunks": [
                        asdict(r.chunk) for r in payload["selected_results"]
                    ],
                }
            )
    (root / "contexts.json").write_text(
        json.dumps(prepared, ensure_ascii=False), encoding="utf-8"
    )
    return prepared


def run_job(job_id, *, compiler=None, prepare=None):
    """At-least-once attempts; committed artifacts are never compiled twice."""
    owner = str(uuid.uuid4())
    with store.locked_budget_owner(job_id) as (_, row):
        if row.status in TERMINAL or (row.lease_until or 0) > time.time():
            return
        row.status, row.lease_owner, row.lease_until = (
            "running",
            owner,
            time.time() + 300,
        )
        row.attempts += 1
    job = get_job(job_id)
    stop = threading.Event()

    def renew():
        while not stop.wait(30):
            with store.locked_budget_owner(job_id) as (_, row):
                if row.lease_owner != owner:
                    return
                row.lease_until = time.time() + 300

    thread = threading.Thread(target=renew, daemon=True)
    thread.start()
    try:
        if compiler is None:
            from ragbench_core.prompt_optimization.compiler import compile_prompt

            compiler = compile_prompt
        for index, raw in enumerate(job["request"]["prompt_optimization"]["configs"]):
            current = get_job(job_id)
            if any(r["index"] == index for r in current["progress"].get("results", [])):
                continue
            account = CompilationAccount(job_id, index, owner)
            update_job(
                job_id,
                progress={
                    **current["progress"],
                    "current_config": index,
                    "phase": "preparing_context",
                },
            )
            try:
                prepared = (prepare or prepare_examples)(job, account)
                update_job(
                    job_id,
                    progress={**get_job(job_id)["progress"], "phase": "compiling"},
                )
                recipe = PromptOptimizationConfig.model_validate(raw)
                payload, summary = compiler(
                    recipe,
                    prepared,
                    job["splits"],
                    job["config"],
                    job["request"]["judge"],
                    account,
                )
                payload = CompiledPrompt.model_validate(payload).model_dump()
                summary = dict(summary)
                history = summary.pop("candidate_history", None)
                account.check()
                result = {
                    "index": index,
                    "preset": recipe.preset,
                    "status": "completed",
                    "artifact_id": payload["content_hash"],
                    **summary,
                }
                if history is not None:
                    result["candidate_history"] = {
                        "version": history["version"], "attempt": job["attempts"],
                        "count": len(history["candidates"]),
                    }
                with store.locked_budget_owner(job_id) as (conn, row):
                    if row.status == "cancelled" or row.lease_owner != owner:
                        raise budget.OptimizationStopped(
                            "Compilation cancelled or lease lost"
                        )
                    if history is not None:
                        from ragbench_core.prompt_optimization.history_store import (
                            save_history,
                        )

                        save_history(conn, job_id, index, job["attempts"], recipe.preset, history)
                    if conn.get(store.PromptArtifact, payload["content_hash"]) is None:
                        conn.add(
                            store.PromptArtifact(
                                id=payload["content_hash"],
                                job_id=job_id,
                                payload=payload,
                                summary=result,
                            )
                        )
                    row.progress = {
                        **row.progress,
                        "results": [*row.progress.get("results", []), result],
                    }
            except budget.OptimizationStopped:
                raise
            except Exception as exc:
                # No provider response text or endpoint credentials in public errors.
                result = {
                    "index": index,
                    "preset": raw.get("preset"),
                    "status": "budget_exhausted"
                    if isinstance(exc, budget.BudgetExhausted)
                    else "failed",
                    "error": str(exc)
                    if isinstance(exc, budget.BudgetExhausted)
                    else f"Compilation failed ({type(exc).__name__}); inspect worker logs",
                }
                import logging

                logging.getLogger(__name__).exception(
                    "Prompt compilation %s config %s failed", job_id, index
                )
                update_job(
                    job_id,
                    progress={
                        **get_job(job_id)["progress"],
                        "results": [
                            *get_job(job_id)["progress"].get("results", []),
                            result,
                        ],
                    },
                )
        results = get_job(job_id)["progress"]["results"]
        success = sum(r["status"] == "completed" for r in results)
        with store.locked_budget_owner(job_id) as (_, row):
            if row.status != "cancelled" and row.lease_owner == owner:
                row.status = (
                    "completed"
                    if success == len(results)
                    else "completed_with_errors"
                    if success
                    else "budget_exhausted"
                    if any(r["status"] == "budget_exhausted" for r in results)
                    else "failed"
                )
    except budget.OptimizationStopped:
        if get_job(job_id)["status"] != "cancelled":
            parent_id = get_job(job_id)["session_id"]
            cancelled = bool(parent_id) and store.get(parent_id)["status"] == "cancelled"
            update_job(job_id, status="cancelled" if cancelled else "paused")
    except Exception as exc:  # noqa: BLE001 -- record worker boundary failures without provider response text.
        update_job(
            job_id, status="failed", error=f"Compilation failed ({type(exc).__name__})"
        )
    finally:
        stop.set()
        thread.join(timeout=1)
        with store.locked_budget_owner(job_id) as (_, row):
            if row.lease_owner == owner:
                row.lease_owner, row.lease_until = None, 0


def provenance(prepared, splits, recipe, config):
    def hashes(partition):
        ids = set(splits[partition])
        return sorted(
            question_fingerprint(e["question"])
            for e in prepared
            if e["question_id"] in ids
        )

    return {
        "optimizer_config": recipe.model_dump(),
        "answer_model": config["llm"],
        "pipeline_hash": stable_hash(config["pipeline"]),
        "dataset_hash": config["_snapshot"]["dataset_hash"],
        "split_hash": splits["hash"],
        "training_question_ids": splits["training"],
        "validation_question_ids": splits["validation"],
        "training_question_hashes": hashes("training"),
        "validation_question_hashes": hashes("validation"),
    }
