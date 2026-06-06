from __future__ import annotations

import json
from typing import Any


def build_evaluation_step(question: str, rewritten_queries: list[str], retrieval_results: list[Any], context: str, query_build: str) -> dict[str, Any]:
    """Capture the context and query-build artifacts for one benchmark result."""
    return _jsonable({
        "question": question,
        "rewritten_queries": rewritten_queries,
        "context": context,
        "query_build": query_build,
        "retrieved_context": [
            {
                "rank": result.rank,
                "score": result.score,
                "retriever": result.retriever,
                "doc_id": result.chunk.doc_id,
                "chunk_id": result.chunk.chunk_id,
                "text": result.chunk.text,
                "metadata": result.chunk.metadata,
            }
            for result in retrieval_results
        ],
    })


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
