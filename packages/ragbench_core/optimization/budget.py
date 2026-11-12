"""Conservative reservations survive retries and interrupted provider calls."""

from __future__ import annotations

from sqlalchemy import select

from ragbench_core.optimization import store


class BudgetExhausted(RuntimeError):
    pass


class OptimizationStopped(RuntimeError):
    pass


def check_control(session_id):
    session = store.get(session_id)
    if session["status"] in {"paused", "cancelled"}:
        raise OptimizationStopped(session["status"])
    usage = store.rows(store.BudgetUsage, session_id)
    if (
        sum(u["seconds"] or 0 for u in usage)
        >= session["request"]["limits"]["active_seconds"]
    ):
        raise BudgetExhausted("Active runtime budget exhausted")


def reserve(session_id, operation, role, tokens, cost, seconds=0, credit=None):
    check_control(session_id)
    with store.locked(session_id) as (conn, session):
        existing = conn.get(store.BudgetUsage, operation)
        if existing:
            return False
        if credit:
            held = conn.get(store.BudgetUsage, credit)
            if held:
                held.tokens = max(0, held.tokens - tokens)
                if cost is not None and held.cost is not None:
                    held.cost = max(0, held.cost - cost)
                held.seconds = max(0, held.seconds - seconds)
                conn.flush()
        usage = (
            conn.execute(
                select(store.BudgetUsage).where(
                    store.BudgetUsage.session_id == session_id
                )
            )
            .scalars()
            .all()
        )
        limits = session.request["limits"]
        if sum(u.tokens for u in usage) + tokens > limits["max_tokens"]:
            raise BudgetExhausted("Token budget exhausted")
        if limits.get("max_cost") is not None:
            if cost is None or any(u.cost is None for u in usage):
                raise BudgetExhausted(
                    "Cannot enforce monetary cap without provider pricing"
                )
            if sum(u.cost for u in usage) + cost > limits["max_cost"]:
                raise BudgetExhausted("Monetary budget exhausted")
        if sum(u.seconds for u in usage) + seconds > limits["active_seconds"]:
            raise BudgetExhausted("Active runtime budget exhausted")
        conn.add(
            store.BudgetUsage(
                id=operation,
                session_id=session_id,
                role=role,
                tokens=tokens,
                cost=cost,
                seconds=seconds,
            )
        )
    return True


def settle(operation, tokens=None, cost=None, seconds=None, actual_tokens=None):
    with store.db.SessionLocal.begin() as conn:
        row = conn.get(store.BudgetUsage, operation)
        if row:
            if tokens is not None:
                row.tokens = max(
                    row.tokens, tokens
                )  # Never release uncertain token reservations.
            if cost is not None:
                row.cost = max(row.cost or 0, cost)
            if seconds is not None:
                row.seconds = seconds
            if actual_tokens is not None:
                row.actual_tokens = actual_tokens
            row.status = "settled"


def totals(session_id):
    rows = store.rows(store.BudgetUsage, session_id)
    return {
        "tokens": sum(r["tokens"] for r in rows),
        "active_seconds": sum(r["seconds"] for r in rows if r["status"] == "settled"),
        "reserved_seconds": sum(r["seconds"] for r in rows if r["status"] != "settled"),
        "cost": None
        if any(r["cost"] is None for r in rows)
        else sum(r["cost"] for r in rows),
        "reservations": len(rows),
        "reported_actual_tokens": sum(r["actual_tokens"] or 0 for r in rows),
        "unreported_token_operations": sum(
            r["actual_tokens"] is None and r["role"] != "reserved_confirmation"
            for r in rows
        ),
        "by_role": {
            role: {
                "reserved_tokens": sum(r["tokens"] for r in rows if r["role"] == role),
                "reported_tokens": sum(
                    r["actual_tokens"] or 0 for r in rows if r["role"] == role
                ),
            }
            for role in sorted({r["role"] for r in rows})
        },
        "unknown_pricing": any(r["cost"] is None for r in rows),
    }


def trial_estimate(config, examples, documents, judge):
    """Conservative token/cost bounds; all remote pipeline roles need explicit prices."""
    from ragbench_core.engine.config_expander import expand_cases

    case = expand_cases(config)[0]
    ingestion = sum(len(d["text"].encode("utf-8")) for d in documents)
    questions = sum(len(q.question.encode("utf-8")) for q in examples)
    context = case.query_config["context_assembler"].get("max_context_chars", 12000) * 4
    rewrite = case.query_config["rewriter"]["type"] != "none"
    output = config["llm"].get("max_output_tokens", 1024)
    end_to_end = config["runtime"]["mode"] != "retrieval_only"
    generation = (
        (questions + len(examples) * (context + output + 1024)) if end_to_end else 0
    )
    rewriting = questions + len(examples) * (output + 1024) if rewrite else 0
    judging = (
        questions
        + len(examples) * (context * 2 + output + judge["max_output_tokens"] + 2048)
        if end_to_end
        else 0
    )
    embedding = (
        ingestion
        + questions * (5 if rewrite else 1)
        + len(examples) * (output if rewrite else 0)
    )
    tokens = generation + rewriting + judging + embedding
    prices = config["runtime"].get("pricing", {})
    required = (
        "generation_input_per_1m",
        "generation_output_per_1m",
        "embedding_per_1m",
    )
    known = all(k in prices for k in required) and (
        not end_to_end
        or (
            judge.get("input_price_per_million") is not None
            and judge.get("output_price_per_million") is not None
        )
    )
    reranker = case.query_config["reranker"]["type"]
    if reranker in {"cohere", "jina"}:
        known = (
            False  # Hosted reranker request billing is not token-priced by this runner.
        )
    cost = None
    if known:
        cost = (
            (generation + rewriting)
            * max(prices["generation_input_per_1m"], prices["generation_output_per_1m"])
            + embedding * prices["embedding_per_1m"]
            + judging
            * max(
                judge.get("input_price_per_million") or 0,
                judge.get("output_price_per_million") or 0,
            )
        ) / 1e6
    return tokens, cost, known
