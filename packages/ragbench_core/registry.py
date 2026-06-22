from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar
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

    _REGISTER_METHOD_GROUPS: ClassVar[dict[str, str]] = {
        "register_extractor": "extractors",
        "register_cleaner": "cleaners",
        "register_chunker": "chunkers",
        "register_metadata_enricher": "metadata_enrichers",
        "register_embedder": "embedders",
        "register_indexer": "indexers",
        "register_rewriter": "rewriters",
        "register_retriever": "retrievers",
        "register_filter": "filters",
        "register_reranker": "rerankers",
        "register_compressor": "compressors",
        "register_assembler": "assemblers",
        "register_prompt_builder": "prompt_builders",
        "register_generator": "generators",
        "register_citation_validator": "citation_validators",
        "register_metric": "metrics",
    }

    def register(self, group: str, component: Any) -> None:
        getattr(self, group)[component.name] = component

    def __getattr__(self, name: str):
        if name in self._REGISTER_METHOD_GROUPS:
            return lambda component: self.register(self._REGISTER_METHOD_GROUPS[name], component)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

REGISTRY = ComponentRegistry()

def register_builtin_components() -> ComponentRegistry:
    from ragbench_core.components.extractors import TxtExtractor, MarkdownExtractor, PdfExtractor, HtmlUrlExtractor, DoclingExtractor
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
    from ragbench_core.components.embedders import (
        BGEM3Embedder,
        BGEM3HybridEmbedder,
        BGEM3SparseEmbedder,
        CohereEmbedder,
        JinaEmbedder,
        OpenAIEmbedder,
        SentenceTransformerEmbedder,
        VoyageEmbedder,
        SPLADESparseEmbedder,
    )
    from ragbench_core.components.indexers import QdrantIndexer, OpenSearchIndexer, OpenSearchNeuralSparseIndexer, QdrantSparseIndexer
    from ragbench_core.components.rewriters import (
        DecomposeRewrite,
        HyDERewrite,
        MultiQueryRewrite,
        NoRewrite,
        StepBackRewrite,
        SynonymExpandRewrite,
    )
    from ragbench_core.components.retrievers import DenseRetriever, DenseFilteredRetriever, ParentDocumentRetriever, BM25Retriever, HybridRetriever, NeuralSparseRetriever, WeightedHybridRetriever, MultiQueryFusionRetriever, AdaptiveHybridRetriever
    from ragbench_core.components.filters import NoFilter
    from ragbench_core.components.rerankers import NoReranker, CrossEncoderReranker, FlashRankReranker, RankLLMReranker, ColBERTReranker, CohereReranker, JinaReranker, DiversityMMRReranker
    from ragbench_core.components.compressors import NoCompressor, TopNCompressor
    from ragbench_core.components.context import DefaultContextAssembler, DefaultPromptBuilder
    from ragbench_core.components.generators import OllamaGenerator
    from ragbench_core.components.validators import SimpleCitationValidator
    from ragbench_core.components.metrics import RetrievalMetrics, OperationalMetric, CitationCoverageMetric, PlaceholderAnswerMetric, PlaceholderFaithfulnessMetric

    builtin_components = {
        "extractors": [TxtExtractor(), MarkdownExtractor(), PdfExtractor(), HtmlUrlExtractor(), DoclingExtractor()],
        "cleaners": [BasicCleaner()],
        "chunkers": [
            FixedSizeChunker(),
            RecursiveChunker(),
            LlamaIndexFixedSizeChunker(),
            LlamaIndexSentenceChunker(),
            LlamaIndexHierarchicalChunker(),
            LlamaIndexSemanticChunker(),
            LlamaIndexDocumentStructureChunker(),
        ],
        "metadata_enrichers": [BasicMetadataEnricher()],
        "embedders": [
            SentenceTransformerEmbedder(),
            OpenAIEmbedder(),
            VoyageEmbedder(),
            CohereEmbedder(),
            JinaEmbedder(),
            BGEM3Embedder(),
            BGEM3SparseEmbedder(),
            SPLADESparseEmbedder(),
            BGEM3HybridEmbedder(),
        ],
        "indexers": [QdrantIndexer(), OpenSearchIndexer(), OpenSearchNeuralSparseIndexer(), QdrantSparseIndexer()],
        "rewriters": [NoRewrite(), HyDERewrite(), MultiQueryRewrite(), StepBackRewrite(), DecomposeRewrite(), SynonymExpandRewrite()],
        "retrievers": [
            DenseRetriever(),
            DenseFilteredRetriever(),
            ParentDocumentRetriever(),
            BM25Retriever(),
            HybridRetriever(),
            NeuralSparseRetriever(),
            WeightedHybridRetriever(),
            MultiQueryFusionRetriever(),
            AdaptiveHybridRetriever(),
        ],
        "filters": [NoFilter()],
        "rerankers": [
            NoReranker(),
            CrossEncoderReranker(),
            FlashRankReranker(),
            RankLLMReranker(),
            ColBERTReranker(),
            CohereReranker(),
            JinaReranker(),
            DiversityMMRReranker(),
        ],
        "compressors": [NoCompressor(), TopNCompressor()],
        "assemblers": [DefaultContextAssembler()],
        "prompt_builders": [DefaultPromptBuilder()],
        "generators": [OllamaGenerator()],
        "citation_validators": [SimpleCitationValidator()],
        "metrics": [
            RetrievalMetrics(),
            OperationalMetric(),
            CitationCoverageMetric(),
            PlaceholderAnswerMetric(),
            PlaceholderFaithfulnessMetric(),
        ],
    }

    for group, components in builtin_components.items():
        for component in components:
            REGISTRY.register(group, component)
    return REGISTRY
