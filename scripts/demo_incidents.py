"""Generate recoverable cache incidents without external services or models.

Run from the repository root: python -m scripts.demo_incidents
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from ragbench_core.engine.cache import DEMO_CACHE_FAULTS, FileEmbeddingCache, embed_chunks_cached
from ragbench_core.models.schema import Chunk


class DemoEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_chunks(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        self.calls += 1
        for chunk in chunks:
            chunk.embedding = [float(len(chunk.text)), 1.0]
        return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", nargs="+", choices=sorted(DEMO_CACHE_FAULTS), default=sorted(DEMO_CACHE_FAULTS))
    parser.add_argument("--log-file", type=Path, default=Path("artifacts/incident-demo.log"))
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger("ragbench.incident_demo")
    logger.info("event=incident_demo_started demo=true faults=%s", ",".join(args.faults))
    with TemporaryDirectory(prefix="ragbench-incidents-") as directory:
        cache = FileEmbeddingCache(Path(directory), demo_faults=set(args.faults))
        embedder = DemoEmbedder()
        config = {"provider": "demo", "model": "deterministic"}
        first = embed_chunks_cached(cache, embedder, [Chunk("demo-chunk", "demo-doc", "demo text")], config)
        second = embed_chunks_cached(cache, embedder, [Chunk("demo-chunk", "demo-doc", "demo text")], config)
        if embedder.calls != 1 or first[0].embedding != second[0].embedding or second[0].embedding != [9.0, 1.0]:
            raise RuntimeError("Incident demo did not recover to a valid cache hit")
        logger.info(
            "event=incident_demo_completed demo=true cache_hit=true embedding_calls=%s log_file=%s",
            embedder.calls, args.log_file,
        )


if __name__ == "__main__":
    main()
