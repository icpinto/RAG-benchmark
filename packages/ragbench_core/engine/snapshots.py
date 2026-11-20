"""Capture benchmark inputs once, before queueing, and verify them before replay."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import requests

from ragbench_core.components.metrics import RetrievalMetrics, validate_metric_selection
from ragbench_core.engine.config_expander import expand_cases
from ragbench_core.engine.data_loader import list_sources, load_ground_truth
from ragbench_core.engine.persistence import artifact_root, stable_hash


def checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_config(config: dict) -> None:
    for key in ("data", "pipeline"):
        if not isinstance(config.get(key), dict):
            raise TypeError(f"{key} must be a mapping")
    for key in ("runtime", "metrics", "input_selection"):
        if not isinstance(config.get(key, {}), dict):
            raise TypeError(f"{key} must be a mapping")
    if not isinstance(config.get("metrics", {}).get("judge", {}), dict):
        raise TypeError("metrics.judge must be a mapping")
    runtime = config.get("runtime", {})
    if runtime.get("mode", "end_to_end") not in {"end_to_end", "retrieval_only"}:
        raise ValueError("runtime.mode must be end_to_end or retrieval_only")
    cap = runtime.get("max_combinations", 32)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_combinations must be a positive integer")
    RetrievalMetrics._validate_retrieval_k(config.get("metrics", {}))
    validate_metric_selection(config.get("metrics", {}))
    judge = config.get("metrics", {}).get("judge", {})
    if judge.get("enabled"):
        if not judge.get("model") or judge.get("provider", "ollama") not in {"ollama", "chat_completions"}:
            raise ValueError("An enabled judge requires a supported provider and model")
        if (
            not isinstance(judge.get("timeout", 240), (int, float))
            or judge.get("timeout", 240) <= 0
        ):
            raise ValueError("Judge timeout must be positive")
    cases = expand_cases(config)
    if not cases:
        raise ValueError("Benchmark must expand to at least one case")
    from ragbench_core.prompt_optimization.artifacts import CompiledPrompt
    from ragbench_core.prompt_optimization.contracts import PromptOptimizationSettings
    PromptOptimizationSettings.model_validate(config.get("prompt_optimization", {}))
    for case in cases:
        if case.query_config["prompt_builder"].get("type") == "compiled":
            CompiledPrompt.model_validate(case.query_config["prompt_builder"]["artifact"])


def capture_corpus(sources: list[str], root: Path) -> dict:
    """Freeze source bytes without requiring evaluation questions."""
    if not sources:
        raise ValueError("Corpus requires at least one source")
    root.mkdir(parents=True, exist_ok=False)
    entries, seen = [], set()
    for index, source in enumerate(sorted(set(sources))):
        remote = source.startswith(("http://", "https://"))
        doc_id = source.replace("https://", "").replace("http://", "").replace("/", "_")[:180] if remote else Path(source).name
        if doc_id in seen:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        seen.add(doc_id)
        encoding = "utf-8"
        if remote:
            response = requests.get(source, timeout=30, headers={"User-Agent": "ragbench/0.1"})
            response.raise_for_status()
            content, encoding = response.content, response.encoding or "utf-8"
        else:
            content = Path(source).read_bytes()
        target = root / "corpus" / str(index) / ("source.html" if remote else Path(source).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append(dict(source=source, doc_id=doc_id, path=str(target.resolve()), sha256=checksum(content), remote=remote, encoding=encoding))
    manifest = {"version": 1, "sources": entries, "corpus_hash": stable_hash(sorted(
        [{"doc_id": e["doc_id"], "sha256": e["sha256"]} for e in entries], key=lambda e: e["doc_id"]))}
    (root / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_corpus(manifest: dict) -> None:
    for entry in manifest["sources"]:
        if checksum(Path(entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {entry['doc_id']}")


def attach_evaluation(manifest: dict, examples: list, root: Path) -> dict:
    """Bind a question split to an already frozen corpus, without fetching again."""
    verify_corpus(manifest)
    root.mkdir(parents=True, exist_ok=True)
    questions = [asdict(e) for e in examples]
    if not questions:
        raise ValueError("Evaluation split is empty")
    ground_truth = root / "ground_truth.json"
    ground_truth.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    identity = {"sources": sorted([{"doc_id": e["doc_id"], "sha256": e["sha256"]} for e in manifest["sources"]], key=lambda e: e["doc_id"]),
                "questions": sorted(questions, key=lambda e: e["question_id"])}
    result = {**manifest, "ground_truth_file": str(ground_truth.resolve()), "ground_truth_sha256": checksum(ground_truth.read_bytes()),
              "question_ids": [q["question_id"] for q in questions], "dataset_hash": stable_hash(identity)}
    path = root / "manifest.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {**result, "manifest_path": str(path.resolve())}


def snapshot_inputs(run_id: str, config: dict) -> dict:
    config = deepcopy(config)
    validate_config(config)
    if "_snapshot" in config:
        verify_snapshot(config)
        validate_prompt_evaluation(config)
        return config
    data = config["data"]
    sources = list_sources(data["input_dir"], data.get("remote_urls_file"))
    selection = config.get("input_selection", {})
    if selection.get("source_paths") is not None:
        selected = {str(Path(p).resolve()) for p in selection["source_paths"]}
        sources = [s for s in sources if str(Path(s).resolve()) in selected]
        if len(sources) != len(selected):
            raise ValueError("Selected source is outside the configured corpus")
    examples = load_ground_truth(data["ground_truth_file"])
    indices = selection.get("question_indices")
    if indices is not None:
        if any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(examples)
            for i in indices
        ):
            raise ValueError("Question index out of range")
        examples = [e for i, e in enumerate(examples) if i in set(indices)]
    if not sources or not examples:
        raise ValueError("Benchmark requires at least one selected source and question")

    root = artifact_root(config, run_id) / "snapshot"
    corpus = capture_corpus(sources, root)
    config["_snapshot"] = attach_evaluation(corpus, examples, root)
    validate_prompt_evaluation(config)
    return config


def validate_prompt_evaluation(config):
    from ragbench_core.prompt_optimization.artifacts import assert_no_training_overlap
    for case in expand_cases(config):
        prompt = case.query_config["prompt_builder"]
        if prompt.get("type") == "compiled":
            internal_validation = bool(config.get("_optimization", {}).get("session_id")) and config.get("_optimization", {}).get("split") == "tuning"
            assert_no_training_overlap(prompt["artifact"], load_ground_truth(config["_snapshot"]["ground_truth_file"]), allow_validation=internal_validation)



def verify_snapshot(config: dict) -> None:
    manifest = config["_snapshot"]
    for entry in manifest["sources"]:
        if checksum(Path(entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {entry['doc_id']}")
    if (
        checksum(Path(manifest["ground_truth_file"]).read_bytes())
        != manifest["ground_truth_sha256"]
    ):
        raise ValueError("Snapshot ground-truth checksum mismatch")
    questions = [asdict(e) for e in load_ground_truth(manifest["ground_truth_file"])]
    identity = {
        "sources": sorted(
            [
                {"doc_id": e["doc_id"], "sha256": e["sha256"]}
                for e in manifest["sources"]
            ],
            key=lambda e: e["doc_id"],
        ),
        "questions": sorted(questions, key=lambda e: e["question_id"]),
    }
    if stable_hash(identity) != manifest["dataset_hash"]:
        raise ValueError("Snapshot dataset identity mismatch")
