from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None

@dataclass
class QueryExample:
    question: str
    expected_answer: str = ""
    ground_truth_doc_ids: list[str] = field(default_factory=list)
    ground_truth_chunk_ids: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str

@dataclass
class GeneratedAnswer:
    answer: str
    prompt: str
    context_chunks: list[Chunk]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkCase:
    case_id: str
    ingestion_config: dict[str, Any]
    query_config: dict[str, Any]
    llm_config: dict[str, Any]
    metric_config: dict[str, Any]
