"""Deterministic stage handoffs for version-two optimization sessions."""

from __future__ import annotations

from copy import deepcopy
from itertools import zip_longest

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
    identities = {retrieval_identity(baseline)}
    shortlist = []
    for row in ranking:
        trial = by_id.get(row["trial_id"])
        if not row["feasible"] or not trial:
            continue
        identity = retrieval_identity(trial["config"])
        if identity in identities:
            continue
        identities.add(identity)
        shortlist.append(
            {
                "trial_id": trial["id"],
                "reason": "Highest-ranked feasible representative of a distinct retrieval setup",
            }
        )
        if len(shortlist) == min(2, max(0, maximum - 1)):
            break
    if maximum == 1:
        shortlist = []

    baseline_sources = [
        t["id"]
        for t in trials
        if candidate_hash(t["config"]) == candidate_hash(baseline)
    ]
    baseline_source = baseline_sources[0] if len(baseline_sources) == 1 else None
    parents = [(baseline, baseline_source)] + [
        (tools.validate_candidate(by_id[item["trial_id"]]["config"]), item["trial_id"])
        for item in shortlist
    ]
    # Prefer improving shortlisted setups. If none qualified, search around baseline.
    variant_parents = parents[1:] or parents[:1]
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
                "config": config,
            }
        )

    for config, source in parents:
        add(config, source, "baseline" if not all_entries else "inherited")

    space = tools.get_search_space()
    # Each dimension changes independently; round-robin parents avoid starving a setup.
    for dimension in ("prompt", "context", "reranker"):
        queues = []
        for parent, source in variant_parents:
            options = []
            if dimension == "prompt":
                for name in PROMPT_PRESETS:
                    config = deepcopy(parent)
                    config["pipeline"]["query"]["prompt_builders"] = [
                        preset_config(name)
                    ]
                    options.append((config, source, name))
            elif dimension == "context":
                for limit, chunks in zip(
                    space["max_context_chars"], space["max_chunks"]
                ):
                    config = deepcopy(parent)
                    query = config["pipeline"]["query"]
                    query["compressors"] = [{"type": "top_n", "max_chunks": chunks}]
                    query["context_assemblers"][0]["max_context_chars"] = limit
                    options.append((config, source, f"context:{limit}/{chunks}"))
            else:
                for kind in space["rerankers"]:
                    config = deepcopy(parent)
                    config["pipeline"]["query"]["rerankers"] = [{"type": kind}]
                    options.append((config, source, f"reranker:{kind}"))
            queues.append(options)
        for group in zip_longest(*queues):
            for option in group:
                if option is not None:
                    add(*option)

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
