import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragbench_core.engine.cache import FileEmbeddingCache, build_embedding_cache, embed_chunks_cached
from ragbench_core.models.schema import Chunk
from scripts.demo_incidents import DemoEmbedder


def test_cache_demo_is_disabled_by_default(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("RAGBENCH_DEMO_FAULTS", raising=False)
    cache = FileEmbeddingCache(tmp_path)
    with caplog.at_level(logging.INFO):
        assert cache.get("test") is None
        cache.set("test", [1.0])
        assert cache.get("test") == [1.0]
    assert not caplog.records


@pytest.mark.parametrize("fault,exception_type", [("cache_schema", KeyError), ("cache_serialization", TypeError)])
def test_demo_fault_logs_once_and_cache_recovers(tmp_path, caplog, fault, exception_type):
    cache = FileEmbeddingCache(tmp_path, demo_faults={fault})
    # Seed a real cache hit for the schema fault, so its fallback can be observed.
    (tmp_path / "test.json").write_text("[1.0]", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        assert cache.get("test") == (None if fault == "cache_schema" else [1.0])
        cache.set("test", [2.0])
        assert cache.get("test") == [2.0]
        cache.set("test", [3.0])
        assert cache.get("test") == [3.0]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info[0] is exception_type
    assert f"fault_id={fault}" in errors[0].getMessage()
    assert "cache_key=test" in errors[0].getMessage()
    assert "cache_path=" in errors[0].getMessage()
    assert any("_recovered" in record.getMessage() for record in caplog.records)
    assert json.loads((tmp_path / "test.json").read_text(encoding="utf-8")) == [3.0]
    assert not (tmp_path / "test.tmp").exists()


def test_env_enabled_faults_recover_through_embedding_pipeline(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("RAGBENCH_DEMO_FAULTS", "cache_schema, cache_serialization")
    cache = build_embedding_cache({"runtime": {"embedding_cache": {"enabled": True, "path": str(tmp_path)}}})
    embedder = DemoEmbedder()
    config = {"provider": "demo", "model": "deterministic"}
    with caplog.at_level(logging.INFO):
        first = embed_chunks_cached(cache, embedder, [Chunk("c1", "d1", "demo text")], config)
        second = embed_chunks_cached(cache, embedder, [Chunk("c1", "d1", "demo text")], config)
    assert first[0].embedding == second[0].embedding == [9.0, 1.0]
    assert embedder.calls == 1
    assert len([record for record in caplog.records if record.levelno == logging.ERROR]) == 2


def test_demo_cli_writes_tracebacks_and_completes(tmp_path):
    log_file = tmp_path / "incident-demo.log"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "packages"), str(repo_root), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "scripts.demo_incidents", "--log-file", str(log_file)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    log = log_file.read_text(encoding="utf-8")
    assert log.count("Traceback (most recent call last):") == 2
    assert "KeyError: 'value'" in log
    assert "TypeError: Object of type bytes is not JSON serializable" in log
    assert "event=incident_demo_completed demo=true cache_hit=true embedding_calls=1" in log
