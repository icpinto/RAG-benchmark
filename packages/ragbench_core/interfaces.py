from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from ragbench_core.models.schema import Document, Chunk, QueryExample, RetrievalResult, GeneratedAnswer

class BaseExtractor(ABC):
    name: str
    @abstractmethod
    def extract(self, source: str, config: dict[str, Any]) -> list[Document]: ...

class BaseCleaner(ABC):
    name: str
    @abstractmethod
    def clean(self, documents: list[Document], config: dict[str, Any]) -> list[Document]: ...

class BaseChunker(ABC):
    name: str
    @abstractmethod
    def chunk(self, documents: list[Document], config: dict[str, Any]) -> list[Chunk]: ...

class BaseMetadataEnricher(ABC):
    name: str
    @abstractmethod
    def enrich(self, chunks: list[Chunk], config: dict[str, Any]) -> list[Chunk]: ...

class BaseEmbedder(ABC):
    name: str
    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk], config: dict[str, Any]) -> list[Chunk]: ...
    @abstractmethod
    def embed_query(self, query: str, config: dict[str, Any]) -> list[float]: ...

class BaseIndexer(ABC):
    name: str
    @abstractmethod
    def index(self, chunks: list[Chunk], index_name: str, config: dict[str, Any]) -> dict[str, Any]: ...

class BaseQueryRewriter(ABC):
    name: str
    @abstractmethod
    def rewrite(self, question: str, config: dict[str, Any], llm_config: dict[str, Any]) -> list[str]: ...

class BaseRetriever(ABC):
    name: str
    @abstractmethod
    def retrieve(self, question: str, query_embedding: list[float] | None, index_name: str, config: dict[str, Any], top_k: int) -> list[RetrievalResult]: ...

class BaseFilter(ABC):
    name: str
    @abstractmethod
    def filter(self, results: list[RetrievalResult], config: dict[str, Any]) -> list[RetrievalResult]: ...

class BaseReranker(ABC):
    name: str
    @abstractmethod
    def rerank(self, question: str, results: list[RetrievalResult], config: dict[str, Any]) -> list[RetrievalResult]: ...

class BaseContextCompressor(ABC):
    name: str
    @abstractmethod
    def compress(self, question: str, results: list[RetrievalResult], config: dict[str, Any]) -> list[RetrievalResult]: ...

class BaseContextAssembler(ABC):
    name: str
    @abstractmethod
    def assemble(self, results: list[RetrievalResult], config: dict[str, Any]) -> str: ...

class BasePromptBuilder(ABC):
    name: str
    @abstractmethod
    def build(self, question: str, context: str, config: dict[str, Any]) -> str: ...

class BaseGenerator(ABC):
    name: str
    @abstractmethod
    def generate(self, prompt: str, config: dict[str, Any]) -> str: ...

class BaseCitationValidator(ABC):
    name: str
    @abstractmethod
    def validate(self, answer: GeneratedAnswer, expected_citations: list[str], config: dict[str, Any]) -> dict[str, Any]: ...

class BaseMetric(ABC):
    name: str
    @abstractmethod
    def compute(self, example: QueryExample, results: list[RetrievalResult], answer: GeneratedAnswer | None, config: dict[str, Any]) -> dict[str, float | int | str]: ...
