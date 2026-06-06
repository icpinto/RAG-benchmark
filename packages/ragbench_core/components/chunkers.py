from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from typing import Any

from ragbench_core.interfaces import BaseChunker
from ragbench_core.models.schema import Chunk, Document
from ragbench_core.utils.ids import stable_id


def _overlap_chars(chunk_size: int, overlap: int | float | str) -> int:
    overlap_value = float(overlap) if isinstance(overlap, str) and "." in overlap else overlap
    return int(chunk_size * overlap_value) if isinstance(overlap_value, float) else int(overlap_value)


def _sliding_window_split(text: str, chunk_size: int, overlap: int | float | str) -> list[tuple[str, int, int]]:
    overlap_size = _overlap_chars(chunk_size, overlap)
    step = max(1, chunk_size - overlap_size)
    return [
        (text[i:i + chunk_size], i, i + len(text[i:i + chunk_size]))
        for i in range(0, len(text), step)
        if text[i:i + chunk_size].strip()
    ]


def _find_span(text: str, chunk_text: str, start_at: int) -> tuple[int, int]:
    stripped = chunk_text.strip()
    start = text.find(stripped, start_at)
    if start == -1:
        start = start_at
    end = start + len(stripped)
    return start, end


def _sentence_split(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _fallback_sentence_chunks(text: str, chunk_size: int, overlap: int | float | str) -> list[str]:
    overlap_size = _overlap_chars(chunk_size, overlap)
    chunks: list[str] = []
    current = ""
    for sentence in _sentence_split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        prefix = current[-overlap_size:] if overlap_size else ""
        current = f"{prefix} {sentence}".strip() if prefix else sentence
    if current:
        chunks.append(current)
    return chunks


def _split_with_llamaindex(
    splitter_factory: Callable[[int, int], object],
    fallback: Callable[[str, int, int | float | str], list[str]],
    text: str,
    chunk_size: int,
    overlap: int | float | str,
) -> list[str]:
    overlap_size = _overlap_chars(chunk_size, overlap)
    if importlib.util.find_spec("llama_index") is None:
        return fallback(text, chunk_size, overlap)
    if importlib.util.find_spec("llama_index.core") is None:
        return fallback(text, chunk_size, overlap)
    if importlib.util.find_spec("llama_index.core.node_parser") is None:
        return fallback(text, chunk_size, overlap)
    splitter = splitter_factory(chunk_size, overlap_size)
    return splitter.split_text(text)



def _chunk_from_text(
    document: Document,
    chunk_text: str,
    idx: int,
    cursor: int,
    chunk_size: int,
    overlap: int | float | str,
    chunker_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Chunk | None, int]:
    if not chunk_text.strip():
        return None, cursor
    start, end = _find_span(document.text, chunk_text, cursor)
    metadata = dict(document.metadata)
    metadata.update({"chunk_index": idx, "start_char": start, "end_char": end, "chunker": chunker_name})
    if extra_metadata:
        metadata.update(extra_metadata)
    chunk_id = stable_id(document.doc_id, idx, chunk_size, overlap, chunker_name, chunk_text.strip())
    return Chunk(chunk_id, document.doc_id, chunk_text.strip(), metadata), end


def _split_markdown_sections(text: str) -> list[tuple[str, str, int]]:
    sections: list[tuple[str, list[str], int]] = []
    current_heading = ""
    current_lines: list[str] = []
    current_level = 0
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match and current_lines:
            sections.append((current_heading, current_lines, current_level))
            current_lines = []
        if match:
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines, current_level))
    if not sections and text.strip():
        return [("", text.strip(), 0)]
    section_texts = [(heading, "\n".join(lines).strip(), level) for heading, lines, level in sections]
    return [(heading, section_text, level) for heading, section_text, level in section_texts if section_text]


def _structure_fallback_chunks(text: str, chunk_size: int, overlap: int | float | str) -> list[tuple[str, dict]]:
    section_chunks: list[tuple[str, dict]] = []
    sections = _split_markdown_sections(text)
    for section_index, (heading, section_text, heading_level) in enumerate(sections):
        if len(section_text) <= chunk_size:
            section_chunks.append((section_text, {
                "section_index": section_index,
                "section_heading": heading,
                "heading_level": heading_level,
            }))
            continue
        for part_index, part in enumerate(_fallback_sentence_chunks(section_text, chunk_size, overlap)):
            section_chunks.append((part, {
                "section_index": section_index,
                "section_heading": heading,
                "heading_level": heading_level,
                "section_part_index": part_index,
            }))
    return section_chunks


def _fallback_semantic_chunks(text: str, chunk_size: int, overlap: int | float | str) -> list[str]:
    """Group adjacent paragraphs/sentences as a deterministic semantic splitter fallback."""
    units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not units:
        units = _sentence_split(text)
    chunks: list[str] = []
    current = ""
    overlap_size = _overlap_chars(chunk_size, overlap)
    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_fallback_sentence_chunks(unit, chunk_size, overlap))
            continue
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        prefix = current[-overlap_size:] if overlap_size else ""
        current = f"{prefix}\n\n{unit}".strip() if prefix else unit
    if current:
        chunks.append(current)
    return chunks


def _build_parent_child_chunks(
    document: Document,
    child_texts: list[str],
    parent_texts: list[str],
    chunk_size: int,
    overlap: int | float | str,
    chunker_name: str,
) -> list[Chunk]:
    parent_spans: list[tuple[str, int, int, str]] = []
    parent_cursor = 0
    for parent_index, parent_text in enumerate(parent_texts):
        if not parent_text.strip():
            continue
        start, end = _find_span(document.text, parent_text, parent_cursor)
        parent_cursor = end
        parent_id = stable_id(document.doc_id, "parent", parent_index, chunk_size, overlap, chunker_name, parent_text.strip())
        parent_spans.append((parent_id, start, end, parent_text.strip()))

    chunks: list[Chunk] = []
    cursor = 0
    for idx, child_text in enumerate(child_texts):
        if not child_text.strip():
            continue
        start, end = _find_span(document.text, child_text, cursor)
        cursor = end
        parent_id = None
        parent_index = None
        parent_text = None
        for candidate_index, (candidate_id, parent_start, parent_end, candidate_text) in enumerate(parent_spans):
            if parent_start <= start and end <= parent_end:
                parent_id = candidate_id
                parent_index = candidate_index
                parent_text = candidate_text
                break
        if parent_id is None and parent_spans:
            parent_index = min(range(len(parent_spans)), key=lambda i: abs(start - parent_spans[i][1]))
            parent_id, _, _, parent_text = parent_spans[parent_index]
        metadata = dict(document.metadata)
        metadata.update({
            "chunk_index": idx,
            "start_char": start,
            "end_char": end,
            "chunker": chunker_name,
            "parent_chunk_id": parent_id,
            "parent_chunk_index": parent_index,
            "parent_text": parent_text,
            "chunk_role": "child",
        })
        child_id = stable_id(document.doc_id, idx, chunk_size, overlap, chunker_name, child_text.strip(), parent_id)
        chunks.append(Chunk(child_id, document.doc_id, child_text.strip(), metadata))
    return chunks


class FixedSizeChunker(BaseChunker):
    name = "fixed"
    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        chunks=[]
        for d in documents:
            for chunk_text, i, end in _sliding_window_split(d.text, chunk_size, overlap):
                cid=stable_id(d.doc_id, i, chunk_size, overlap)
                md=dict(d.metadata); md.update({"start_char":i,"end_char":end,"chunker":self.name})
                chunks.append(Chunk(cid,d.doc_id,chunk_text,md))
        return chunks

class RecursiveChunker(BaseChunker):
    name = "recursive"
    separators = ["\n\n", "\n", ". ", " ", ""]
    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        overlap_chars = _overlap_chars(chunk_size, overlap)
        chunks=[]
        for d in documents:
            parts=self._split(d.text, chunk_size)
            pos=0
            for idx, part in enumerate(parts):
                if overlap_chars and idx>0:
                    prev=parts[idx-1][-overlap_chars:]
                    part=prev+part
                cid=stable_id(d.doc_id, idx, chunk_size, overlap, self.name)
                md=dict(d.metadata); md.update({"chunk_index":idx,"chunker":self.name})
                chunks.append(Chunk(cid,d.doc_id,part.strip(),md))
        return [c for c in chunks if c.text]

    def _split(self, text: str, size: int) -> list[str]:
        if len(text) <= size: return [text]
        sep = next((s for s in self.separators if s and s in text), "")
        if not sep:
            return [text[i:i+size] for i in range(0, len(text), size)]
        pieces=text.split(sep)
        out=[]; buf=""
        for p in pieces:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= size:
                buf=candidate
            else:
                if buf: out.append(buf)
                if len(p)>size: out.extend(self._split(p,size))
                else: buf=p
        if buf: out.append(buf)
        return out


class LlamaIndexFixedSizeChunker(BaseChunker):
    name = "llamaindex_fixed"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        chunks=[]
        for d in documents:
            split_texts = _split_with_llamaindex(
                self._splitter,
                lambda text, size, overlap_value: [chunk for chunk, _, _ in _sliding_window_split(text, size, overlap_value)],
                d.text,
                chunk_size,
                overlap,
            )
            cursor = 0
            for idx, text in enumerate(split_texts):
                if not text.strip():
                    continue
                start, end = _find_span(d.text, text, cursor)
                cursor = end
                cid=stable_id(d.doc_id, idx, chunk_size, overlap, self.name)
                md=dict(d.metadata); md.update({"chunk_index":idx,"start_char":start,"end_char":end,"chunker":self.name})
                chunks.append(Chunk(cid,d.doc_id,text.strip(),md))
        return chunks

    def _splitter(self, chunk_size: int, chunk_overlap: int):
        from llama_index.core.node_parser import TokenTextSplitter

        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class LlamaIndexSentenceChunker(BaseChunker):
    name = "llamaindex_sentence"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        chunks=[]
        for d in documents:
            split_texts = _split_with_llamaindex(
                self._splitter,
                _fallback_sentence_chunks,
                d.text,
                chunk_size,
                overlap,
            )
            cursor = 0
            for idx, text in enumerate(split_texts):
                if not text.strip():
                    continue
                start, end = _find_span(d.text, text, cursor)
                cursor = end
                cid=stable_id(d.doc_id, idx, chunk_size, overlap, self.name)
                md=dict(d.metadata); md.update({"chunk_index":idx,"start_char":start,"end_char":end,"chunker":self.name})
                chunks.append(Chunk(cid,d.doc_id,text.strip(),md))
        return chunks

    def _splitter(self, chunk_size: int, chunk_overlap: int):
        from llama_index.core.node_parser import SentenceSplitter

        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class LlamaIndexHierarchicalChunker(BaseChunker):
    name = "llamaindex_hierarchical"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("child_chunk_size", config.get("chunk_size", 512)))
        overlap = config.get("overlap", 0)
        parent_chunk_size = int(config.get("parent_chunk_size", max(chunk_size * 2, chunk_size + 1)))
        chunks: list[Chunk] = []
        for document in documents:
            child_texts, parent_texts = self._split(document.text, chunk_size, parent_chunk_size, overlap)
            chunks.extend(_build_parent_child_chunks(
                document,
                child_texts,
                parent_texts,
                chunk_size,
                overlap,
                self.name,
            ))
        return chunks

    def _split(
        self,
        text: str,
        child_chunk_size: int,
        parent_chunk_size: int,
        overlap: int | float | str,
    ) -> tuple[list[str], list[str]]:
        overlap_size = _overlap_chars(child_chunk_size, overlap)
        if importlib.util.find_spec("llama_index") is None or importlib.util.find_spec("llama_index.core.node_parser") is None:
            return self._fallback_split(text, child_chunk_size, parent_chunk_size, overlap)
        from llama_index.core import Document as LlamaDocument
        from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_chunk_size, child_chunk_size],
            chunk_overlap=overlap_size,
        )
        nodes = parser.get_nodes_from_documents([LlamaDocument(text=text)])
        leaf_nodes = get_leaf_nodes(nodes)
        child_texts = [node.get_content().strip() for node in leaf_nodes if node.get_content().strip()]
        parent_texts = [node.get_content().strip() for node in nodes if node.get_content().strip() and node not in leaf_nodes]
        if not parent_texts:
            parent_texts = _fallback_sentence_chunks(text, parent_chunk_size, overlap)
        return child_texts, parent_texts

    def _fallback_split(
        self,
        text: str,
        child_chunk_size: int,
        parent_chunk_size: int,
        overlap: int | float | str,
    ) -> tuple[list[str], list[str]]:
        parent_texts = _fallback_sentence_chunks(text, parent_chunk_size, overlap)
        child_texts: list[str] = []
        for parent_text in parent_texts:
            child_texts.extend(_fallback_sentence_chunks(parent_text, child_chunk_size, overlap))
        return child_texts, parent_texts


class LlamaIndexSemanticChunker(BaseChunker):
    name = "llamaindex_semantic"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        chunks: list[Chunk] = []
        for document in documents:
            split_texts = self._split(document.text, config, chunk_size, overlap)
            cursor = 0
            for idx, text in enumerate(split_texts):
                chunk, cursor = _chunk_from_text(document, text, idx, cursor, chunk_size, overlap, self.name)
                if chunk:
                    chunks.append(chunk)
        return chunks

    def _split(self, text: str, config: dict, chunk_size: int, overlap: int | float | str) -> list[str]:
        if importlib.util.find_spec("llama_index") is None or importlib.util.find_spec("llama_index.core.node_parser") is None:
            return _fallback_semantic_chunks(text, chunk_size, overlap)
        if importlib.util.find_spec("llama_index.core.embeddings") is None:
            return _fallback_semantic_chunks(text, chunk_size, overlap)
        try:
            from llama_index.core import Document as LlamaDocument
            from llama_index.core.embeddings import MockEmbedding
            from llama_index.core.node_parser import SemanticSplitterNodeParser
        except ImportError:
            return _fallback_semantic_chunks(text, chunk_size, overlap)

        embed_dim = int(config.get("embed_dim", 384))
        breakpoint_percentile_threshold = int(config.get("breakpoint_percentile_threshold", 95))
        buffer_size = int(config.get("buffer_size", 1))
        parser = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=MockEmbedding(embed_dim=embed_dim),
        )
        nodes = parser.get_nodes_from_documents([LlamaDocument(text=text)])
        split_texts = [node.get_content().strip() for node in nodes if node.get_content().strip()]
        if not split_texts:
            return _fallback_semantic_chunks(text, chunk_size, overlap)
        return split_texts


class LlamaIndexDocumentStructureChunker(BaseChunker):
    name = "llamaindex_structure"

    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        chunks: list[Chunk] = []
        for document in documents:
            section_chunks = self._split(document.text, chunk_size, overlap)
            cursor = 0
            for idx, (text, metadata) in enumerate(section_chunks):
                chunk, cursor = _chunk_from_text(document, text, idx, cursor, chunk_size, overlap, self.name, metadata)
                if chunk:
                    chunks.append(chunk)
        return chunks

    def _split(self, text: str, chunk_size: int, overlap: int | float | str) -> list[tuple[str, dict]]:
        if importlib.util.find_spec("llama_index") is None or importlib.util.find_spec("llama_index.core.node_parser") is None:
            return _structure_fallback_chunks(text, chunk_size, overlap)
        try:
            from llama_index.core import Document as LlamaDocument
            from llama_index.core.node_parser import MarkdownNodeParser
        except ImportError:
            return _structure_fallback_chunks(text, chunk_size, overlap)

        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([LlamaDocument(text=text)])
        section_chunks: list[tuple[str, dict]] = []
        for node_index, node in enumerate(nodes):
            node_text = node.get_content().strip()
            if not node_text:
                continue
            metadata = dict(getattr(node, "metadata", {}) or {})
            metadata.setdefault("section_index", node_index)
            if len(node_text) <= chunk_size:
                section_chunks.append((node_text, metadata))
                continue
            for part_index, part in enumerate(_fallback_sentence_chunks(node_text, chunk_size, overlap)):
                part_metadata = dict(metadata)
                part_metadata["section_part_index"] = part_index
                section_chunks.append((part, part_metadata))
        if not section_chunks:
            return _structure_fallback_chunks(text, chunk_size, overlap)
        return section_chunks
