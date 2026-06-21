from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

_CLIENT_TIMINGS: dict[str, float] = {"client_initialization_seconds": 0.0}


def _record_client_init(start: float) -> None:
    _CLIENT_TIMINGS["client_initialization_seconds"] = _CLIENT_TIMINGS.get("client_initialization_seconds", 0.0) + (time.perf_counter() - start)


def consume_client_timings() -> dict[str, float]:
    timings = dict(_CLIENT_TIMINGS)
    for key in _CLIENT_TIMINGS:
        _CLIENT_TIMINGS[key] = 0.0
    return timings


@lru_cache(maxsize=8)
def get_qdrant_client(url: str):
    start = time.perf_counter()
    try:
        from qdrant_client import QdrantClient

        return QdrantClient(url=url)
    finally:
        _record_client_init(start)


@lru_cache(maxsize=8)
def get_opensearch_client(url: str, verify_certs: bool = False, ssl_show_warn: bool = False):
    start = time.perf_counter()
    try:
        from opensearchpy import OpenSearch

        return OpenSearch(url, verify_certs=verify_certs, ssl_show_warn=ssl_show_warn)
    finally:
        _record_client_init(start)


@lru_cache(maxsize=8)
def get_ollama_session(base_url: str):
    start = time.perf_counter()
    try:
        import requests

        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        return session
    finally:
        _record_client_init(start)


def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    transport = getattr(client, "transport", None)
    transport_close = getattr(transport, "close", None)
    if callable(transport_close):
        transport_close()
