from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable

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
