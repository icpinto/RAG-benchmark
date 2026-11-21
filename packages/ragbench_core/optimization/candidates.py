"""Deterministic stage handoffs for version-two optimization sessions."""

from __future__ import annotations

from copy import deepcopy

from ragbench_core.components.context import PROMPT_PRESETS, preset_config
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization.search import candidate_hash


def retrieval_identity(config):
    case = expand_cases(config)[0]
    return stable_hash(
        {
            "ingestion": case.ingestion_config,
            **{
                key: case.query_config[key]
                for key in ("rewriter", "retriever", "filters")
            },
        }
    )


def screen_config(config, baseline):
    config = deepcopy(config)
    query = config["pipeline"]["query"]
    for key in ("compressors", "context_assemblers", "prompt_builders"):
        query[key] = deepcopy(baseline["pipeline"]["query"][key])
    return config


def answer_plan(tools, trials, ranking, maximum):
    trials = [t for t in trials if t["phase"] == "retrieval"]
    baseline = tools.baseline()
    by_id = {t["id"]: t for t in trials}
    winner = next(
        (
            by_id[row["trial_id"]]
            for row in ranking
            if row["feasible"] and row["trial_id"] in by_id
        ),
        None,
    )
    if winner is not None:
        parent = tools.validate_candidate(winner["config"])
        source = winner["id"]
        reason = "Highest-ranked feasible retrieval trial; sole parent for answer search"
        shortlist = [{"trial_id": source, "reason": reason}]
        parent_variant = "inherited"
    else:
        parent = baseline
        baseline_sources = [
            t["id"]
            for t in trials
            if candidate_hash(t["config"]) == candidate_hash(baseline)
        ]
        source = baseline_sources[0] if len(baseline_sources) == 1 else None
        reason = "Baseline fallback: no feasible ranked retrieval trial was available"
        shortlist = []
        parent_variant = "baseline"
    all_entries, seen = [], set()

    def add(config, source, variant):
        try:
            config = tools.validate_candidate(config)
        except (ValueError, KeyError, TypeError):
            return
        identity = candidate_hash(config)
        if identity in seen:
            return
        seen.add(identity)
        all_entries.append(
            {
                "candidate_hash": identity,
                "source_trial_id": source,
                "variant": variant,
                "reason": reason,
                "config": config,
            }
        )

    add(parent, source, parent_variant)

    space = tools.get_search_space()
    # Each dimension changes independently from the unchanged retrieval winner.
    for dimension in ("prompt", "context", "reranker"):
        if dimension == "prompt":
            for name in PROMPT_PRESETS:
                config = deepcopy(parent)
                config["pipeline"]["query"]["prompt_builders"] = [preset_config(name)]
                add(config, source, name)
        elif dimension == "context":
            for limit, chunks in zip(space["max_context_chars"], space["max_chunks"]):
                config = deepcopy(parent)
                query = config["pipeline"]["query"]
                query["compressors"] = [{"type": "top_n", "max_chunks": chunks}]
                query["context_assemblers"][0]["max_context_chars"] = limit
                add(config, source, f"context:{limit}/{chunks}")
        else:
            for kind in space["rerankers"]:
                config = deepcopy(parent)
                config["pipeline"]["query"]["rerankers"] = [{"type": kind}]
                add(config, source, f"reranker:{kind}")

    return {
        "retrieval_shortlist": shortlist,
        "answer_candidates": [entry["config"] for entry in all_entries[:maximum]],
        "answer_plan": [
            {
                **{k: v for k, v in entry.items() if k != "config"},
                "scheduled": i < maximum,
            }
            for i, entry in enumerate(all_entries)
        ],
    }
