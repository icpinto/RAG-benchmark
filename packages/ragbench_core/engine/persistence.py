from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ragbench_core.models.schema import RetrievalResult, Chunk

_VOLATILE_KEYS={"api_key","token","password","secret","created_at","updated_at"}
_DEPENDENCIES=("rag-benchmark-platform","fastapi","streamlit","sqlalchemy","pydantic","pandas","mlflow","wandb")


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def normalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): normalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k).lower() not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_hash(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(normalize_for_hash(value), sort_keys=True, separators=(",",":"), default=_json_default)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def dataset_hash(sources: list[str | Path], examples: list[Any]) -> str:
    payload={"sources": [], "examples": normalize_for_hash(examples)}
    for source in sorted(Path(item) for item in sources):
        entry={"path": str(source)}
        try:
            entry["sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            entry["sha256"]=None
        payload["sources"].append(entry)
    return stable_hash(payload)


def dependency_versions() -> dict[str, str]:
    versions={}
    for name in _DEPENDENCIES:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def component_versions(registry: Any) -> dict[str, dict[str, str]]:
    versions={}
    for family in ["extractors","cleaners","chunkers","metadata_enrichers","embedders","indexers","rewriters","retrievers","filters","rerankers","compressors","assemblers","prompt_builders","generators","metrics","citation_validators"]:
        members=getattr(registry, family, None)
        if not isinstance(members, dict):
            continue
        versions[family]={}
        for name,component in members.items():
            module=getattr(component.__class__, "__module__", "")
            package=module.split(".")[0] if module else ""
            try:
                version=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version=getattr(component, "version", "local")
            versions[family][name]=str(version)
    return versions


def model_identifiers(config: dict[str, Any]) -> dict[str, list[str]]:
    models={"embedders": [], "llms": [], "rerankers": []}
    for item in config.get("pipeline", {}).get("ingestion", {}).get("embeddings", []):
        provider=item.get("provider", "unknown")
        models["embedders"].append(f"{provider}:{item.get('model', provider)}")
    llm=config.get("llm", {})
    if llm:
        models["llms"].append(f"{llm.get('provider','unknown')}:{llm.get('model', llm.get('provider','unknown'))}")
    for item in config.get("pipeline", {}).get("query", {}).get("rerankers", []):
        if "model" in item:
            models["rerankers"].append(f"{item.get('type','reranker')}:{item['model']}")
    return models


def artifact_root(config: dict[str, Any], run_id: str) -> Path:
    root=Path(config.get("runtime", {}).get("artifact_dir") or "/app/artifacts") / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_path(config: dict[str, Any], run_id: str, *parts: str) -> Path:
    safe_parts=[str(part).replace("/", "_").replace("..", "_") for part in parts]
    path=artifact_root(config, run_id).joinpath(*safe_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json_artifact(config: dict[str, Any], run_id: str, *parts: str, payload: Any) -> str:
    path=artifact_path(config, run_id, *parts)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path)


def retrieval_results_payload(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [{"rank": r.rank, "score": r.score, "retriever": r.retriever, "chunk": _chunk_payload(r.chunk)} for r in results]


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "text": chunk.text, "metadata": chunk.metadata}


def export_run(config: dict[str, Any], run_id: str, run_payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, str]:
    exports={}
    export_conf=config.get("runtime", {}).get("exports", {}) or {}
    root=artifact_root(config, run_id) / "exports"
    root.mkdir(parents=True, exist_ok=True)
    if export_conf.get("sqlite", True):
        db_path=root / "results.sqlite"
        with sqlite3.connect(db_path) as conn:
            conn.execute("create table if not exists runs (id text primary key, payload text)")
            conn.execute("create table if not exists results (run_id text, case_id text, question text, payload text)")
            conn.execute("insert or replace into runs values (?, ?)", (run_id, json.dumps(run_payload, default=_json_default)))
            conn.executemany("insert into results values (?, ?, ?, ?)", [(run_id, r.get("case_id"), r.get("question"), json.dumps(r, default=_json_default)) for r in results])
        exports["sqlite"]=str(db_path)
    if export_conf.get("parquet"):
        import pandas as pd
        path=root / "results.parquet"
        pd.json_normalize(results).to_parquet(path, index=False)
        exports["parquet"]=str(path)
    if export_conf.get("mlflow"):
        import mlflow
        with mlflow.start_run(run_name=run_payload.get("name", run_id)):
            mlflow.log_params({"run_id": run_id, "config_hash": run_payload.get("config_hash"), "dataset_hash": run_payload.get("dataset_hash")})
            for key, stats in (run_payload.get("summary", {}).get("aggregate", {}) or {}).items():
                if isinstance(stats, dict) and "mean" in stats:
                    mlflow.log_metric(key, stats["mean"])
            mlflow.log_artifacts(str(artifact_root(config, run_id)))
        exports["mlflow"]="logged"
    if export_conf.get("wandb"):
        import wandb
        wandb.init(project=export_conf.get("wandb_project", "ragbench"), name=run_payload.get("name", run_id), config=config)
        wandb.log({f"aggregate/{k}": v.get("mean") for k,v in (run_payload.get("summary", {}).get("aggregate", {}) or {}).items() if isinstance(v, dict) and "mean" in v})
        wandb.finish()
        exports["wandb"]="logged"
    return exports
