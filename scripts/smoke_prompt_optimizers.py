"""Bounded live DSPy smoke test, using fixed contexts without vector services.

Run: python -m scripts.smoke_prompt_optimizers --base-url http://localhost:11434 --model qwen2.5:14b
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization.budget import BudgetExhausted
from ragbench_core.prompt_optimization.compiler import compile_prompt
from ragbench_core.prompt_optimization.contracts import PromptOptimizationConfig
from ragbench_core.prompt_optimization.datasets import split_questions
from ragbench_core.prompt_optimization.service import task_profile


class SmokeAccount:
    """Offline validation helper; deployment jobs use the persistent ledger."""

    def __init__(self):
        self.calls = self.tokens = 0
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started > 600:
            raise BudgetExhausted("Smoke test exceeded ten minutes")

    @contextmanager
    def call(self, role, prompt, profile, **kwargs):
        self.check()
        tokens = len(prompt.encode("utf-8")) + profile.max_output_tokens + 256
        if self.calls >= 100 or self.tokens + tokens > 1_000_000:
            raise BudgetExhausted("Smoke test exceeded its call/token cap")
        self.calls += 1
        self.tokens += tokens
        yield {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="artifacts/dspy-smoke")
    parser.add_argument(
        "--preset",
        choices=[
            "bootstrap_small_v1",
            "mipro_zero_shot_v1",
            "mipro_few_shot_v1",
            "gepa_light_v1",
        ],
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    facts = [
        ("How long is the refund period?", "The refund period is fourteen days."),
        (
            "Which team handles password resets?",
            "The support team handles password resets.",
        ),
        (
            "When does the library close on Fridays?",
            "The library closes at six in the evening on Fridays.",
        ),
        (
            "What material is the reusable container made from?",
            "The reusable container is made from glass.",
        ),
        (
            "Where should faulty equipment be returned?",
            "Faulty equipment should be returned to reception.",
        ),
        (
            "Name the two mandatory safety items.",
            "The two mandatory safety items are gloves and goggles.",
        ),
        (
            "How many guests can attend each workshop?",
            "Each workshop can accommodate twelve guests.",
        ),
        (
            "What must visitors do before entering the laboratory?",
            "Visitors must sign the register before entering the laboratory.",
        ),
    ]
    prepared = [
        {
            "question_id": f"smoke-{i}",
            "question": question,
            "expected_answer": answer,
            "ground_truth_doc_ids": [f"source-{i}"],
            "context": f"[source-{i}::chunk-{i}]\n{answer}",
            "context_chunks": [
                {"chunk_id": f"chunk-{i}", "doc_id": f"source-{i}", "text": answer}
            ],
        }
        for i, (question, answer) in enumerate(facts)
    ]
    splits = split_questions(prepared)
    config = {
        "llm": {
            "provider": "ollama",
            "base_url": args.base_url,
            "model": args.model,
            "temperature": 0,
            "max_output_tokens": 512,
            "timeout": 60,
        },
        "pipeline": {},
        "runtime": {"artifact_dir": str(root)},
        "_snapshot": {"dataset_hash": stable_hash(prepared)},
    }
    judge = {**task_profile(config).model_dump(), "max_output_tokens": 2048}
    results = []
    for preset in (
        [args.preset]
        if args.preset
        else ["bootstrap_small_v1", "mipro_zero_shot_v1", "gepa_light_v1"]
    ):
        account = SmokeAccount()
        params = (
            {"num_candidates": 2, "num_trials": 2}
            if "mipro" in preset
            else {"max_metric_calls": 12, "skip_perfect_score": False}
            if "gepa" in preset
            else {}
        )
        try:
            recipe = PromptOptimizationConfig(preset=preset, parameters=params)
            # Holdout examples are not passed to the compiler, even in this smoke test.
            tuning = [
                e
                for e in prepared
                if e["question_id"] in set(splits["training"] + splits["validation"])
            ]
            artifact, summary = compile_prompt(
                recipe, tuning, splits, config, judge, account
            )
            (root / f"{preset}.json").write_text(
                json.dumps({"artifact": artifact, "summary": summary}, indent=2),
                encoding="utf-8",
            )
            result = {
                "preset": preset,
                "status": "passed",
                **summary,
                "calls": account.calls,
                "reserved_tokens": account.tokens,
            }
        except Exception as exc:  # noqa: BLE001 -- report every failed optimizer and continue smoke coverage.
            import traceback

            traceback.print_exc()
            result = {
                "preset": preset,
                "status": "failed",
                "error_type": type(exc).__name__,
                "calls": account.calls,
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        (root / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
