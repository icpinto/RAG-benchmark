from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from ragbench_core.interfaces import *

@dataclass
class ComponentRegistry:
    extractors: dict[str, BaseExtractor] = field(default_factory=dict)
    cleaners: dict[str, BaseCleaner] = field(default_factory=dict)
    chunkers: dict[str, BaseChunker] = field(default_factory=dict)
    metadata_enrichers: dict[str, BaseMetadataEnricher] = field(default_factory=dict)
    embedders: dict[str, BaseEmbedder] = field(default_factory=dict)
    indexers: dict[str, BaseIndexer] = field(default_factory=dict)
    rewriters: dict[str, BaseQueryRewriter] = field(default_factory=dict)
    retrievers: dict[str, BaseRetriever] = field(default_factory=dict)
    filters: dict[str, BaseFilter] = field(default_factory=dict)
    rerankers: dict[str, BaseReranker] = field(default_factory=dict)
    compressors: dict[str, BaseContextCompressor] = field(default_factory=dict)
    assemblers: dict[str, BaseContextAssembler] = field(default_factory=dict)
    prompt_builders: dict[str, BasePromptBuilder] = field(default_factory=dict)
    generators: dict[str, BaseGenerator] = field(default_factory=dict)
    citation_validators: dict[str, BaseCitationValidator] = field(default_factory=dict)
    metrics: dict[str, BaseMetric] = field(default_factory=dict)

    def register_extractor(self, x): self.extractors[x.name] = x
    def register_cleaner(self, x): self.cleaners[x.name] = x
    def register_chunker(self, x): self.chunkers[x.name] = x
    def register_metadata_enricher(self, x): self.metadata_enrichers[x.name] = x
    def register_embedder(self, x): self.embedders[x.name] = x
    def register_indexer(self, x): self.indexers[x.name] = x
    def register_rewriter(self, x): self.rewriters[x.name] = x
    def register_retriever(self, x): self.retrievers[x.name] = x
    def register_filter(self, x): self.filters[x.name] = x
    def register_reranker(self, x): self.rerankers[x.name] = x
    def register_compressor(self, x): self.compressors[x.name] = x
    def register_assembler(self, x): self.assemblers[x.name] = x
    def register_prompt_builder(self, x): self.prompt_builders[x.name] = x
    def register_generator(self, x): self.generators[x.name] = x
    def register_citation_validator(self, x): self.citation_validators[x.name] = x
    def register_metric(self, x): self.metrics[x.name] = x

REGISTRY = ComponentRegistry()

def register_builtin_components() -> ComponentRegistry:
    from ragbench_core.components.extractors import TxtExtractor, MarkdownExtractor, PdfExtractor, HtmlUrlExtractor
    from ragbench_core.components.cleaners import BasicCleaner
    from ragbench_core.components.chunkers import (
        FixedSizeChunker,
        RecursiveChunker,
        LlamaIndexFixedSizeChunker,
        LlamaIndexSentenceChunker,
        LlamaIndexHierarchicalChunker,
        LlamaIndexSemanticChunker,
        LlamaIndexDocumentStructureChunker,
    )
    from ragbench_core.components.metadata import BasicMetadataEnricher
    from ragbench_core.components.embedders import SentenceTransformerEmbedder
    from ragbench_core.components.indexers import QdrantIndexer, OpenSearchIndexer
    from ragbench_core.components.rewriters import NoRewrite, HyDERewrite
    from ragbench_core.components.retrievers import DenseRetriever, BM25Retriever, HybridRetriever
    from ragbench_core.components.filters import NoFilter
    from ragbench_core.components.rerankers import NoReranker, CrossEncoderReranker
    from ragbench_core.components.compressors import NoCompressor, TopNCompressor
    from ragbench_core.components.context import DefaultContextAssembler, DefaultPromptBuilder
    from ragbench_core.components.generators import OllamaGenerator
    from ragbench_core.components.validators import SimpleCitationValidator
    from ragbench_core.components.metrics import RetrievalMetrics, OperationalMetric, CitationCoverageMetric, PlaceholderAnswerMetric, PlaceholderFaithfulnessMetric

    for c in [TxtExtractor(), MarkdownExtractor(), PdfExtractor(), HtmlUrlExtractor()]: REGISTRY.register_extractor(c)
    REGISTRY.register_cleaner(BasicCleaner())
    for c in [
        FixedSizeChunker(),
        RecursiveChunker(),
        LlamaIndexFixedSizeChunker(),
        LlamaIndexSentenceChunker(),
        LlamaIndexHierarchicalChunker(),
        LlamaIndexSemanticChunker(),
        LlamaIndexDocumentStructureChunker(),
    ]: REGISTRY.register_chunker(c)
    REGISTRY.register_metadata_enricher(BasicMetadataEnricher())
    REGISTRY.register_embedder(SentenceTransformerEmbedder())
    for c in [QdrantIndexer(), OpenSearchIndexer()]: REGISTRY.register_indexer(c)
    for c in [NoRewrite(), HyDERewrite()]: REGISTRY.register_rewriter(c)
    for c in [DenseRetriever(), BM25Retriever(), HybridRetriever()]: REGISTRY.register_retriever(c)
    REGISTRY.register_filter(NoFilter())
    for c in [NoReranker(), CrossEncoderReranker()]: REGISTRY.register_reranker(c)
    for c in [NoCompressor(), TopNCompressor()]: REGISTRY.register_compressor(c)
    REGISTRY.register_assembler(DefaultContextAssembler())
    REGISTRY.register_prompt_builder(DefaultPromptBuilder())
    REGISTRY.register_generator(OllamaGenerator())
    REGISTRY.register_citation_validator(SimpleCitationValidator())
    for m in [RetrievalMetrics(), OperationalMetric(), CitationCoverageMetric(), PlaceholderAnswerMetric(), PlaceholderFaithfulnessMetric()]: REGISTRY.register_metric(m)
    return REGISTRY
