"""Session-scoped ingestion reuse; standalone benchmark indexes stay isolated."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ragbench_core.engine.cache import build_embedding_cache
from ragbench_core.engine.persistence import (
    component_versions,
    dependency_versions,
    stable_hash,
)
from ragbench_core.models.schema import Chunk


class IngestionReuse:
    def __init__(self, root, session_id, registry, index_exists=None):
        self.root, self.session_id, self.registry = Path(root), session_id, registry
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_exists = index_exists
        self.versions = {
            "components": component_versions(registry),
            "dependencies": dependency_versions(),
        }

    def key(self, case, config):
        return stable_hash(
            {
                "corpus": [
                    {"doc_id": e["doc_id"], "sha256": e["sha256"]}
                    for e in config["_snapshot"]["sources"]
                ],
                "ingestion": case.ingestion_config,
                "versions": self.versions,
            }
        )

    def index_name(self, case, config):
        return (
            "opt_"
            + self.session_id.replace("-", "")
            + "_"
            + self.key(case, config)[:24]
        )

    def load(self, case, config):
        started = time.monotonic()
        path = self.root / (self.key(case, config) + ".json")
        if not path.exists():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        if self.index_exists and not self.index_exists(
            case.ingestion_config["indexers"], result["index_name"]
        ):
            return None
        result["chunks"] = [Chunk(**c) for c in result["chunks"]]
        for chunk in result["chunks"]:
            if chunk.sparse_embedding:
                chunk.sparse_embedding = {
                    int(k): v for k, v in chunk.sparse_embedding.items()
                }
        result["embedder"] = self.registry.embedders[
            case.ingestion_config["embedder"]["provider"]
        ]
        result["embedding_cache"] = build_embedding_cache(config)
        result["build_timings"] = result["timings"]
        result["timings"] = {
            "ingestion_reuse_seconds": time.monotonic() - started,
            "ingestion_reused": True,
        }
        result["embedding_tokens"], result["token_usage"], result["client_timings"] = (
            0,
            {},
            {},
        )
        return result

    def save(self, case, config, ingestion):
        data = {
            k: v
            for k, v in ingestion.items()
            if k not in {"chunks", "embedder", "embedding_cache"}
        }
        data["chunks"] = [asdict(c) for c in ingestion["chunks"]]
        path = self.root / (self.key(case, config) + ".json")
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(data), encoding="utf-8")
        pending.replace(path)


def backend_indexes_exist(indexers, index_name):
    from ragbench_core.settings import settings
    from ragbench_core.utils.clients import get_opensearch_client, get_qdrant_client

    for config in indexers.values():
        if not isinstance(config, dict):
            continue
        kind = config.get("type")
        if kind == "qdrant":
            if not get_qdrant_client(settings.qdrant_url).collection_exists(index_name):
                return False
        elif kind == "opensearch":
            if not get_opensearch_client(settings.opensearch_url).indices.exists(
                index=index_name
            ):
                return False
        else:
            return False
    return True
