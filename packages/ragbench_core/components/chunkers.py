from __future__ import annotations
from ragbench_core.interfaces import BaseChunker
from ragbench_core.models.schema import Document, Chunk
from ragbench_core.utils.ids import stable_id

class FixedSizeChunker(BaseChunker):
    name = "fixed"
    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        overlap_chars = int(chunk_size * overlap) if isinstance(overlap, float) else int(overlap)
        step = max(1, chunk_size - overlap_chars)
        chunks=[]
        for d in documents:
            text=d.text
            for i in range(0, len(text), step):
                chunk_text=text[i:i+chunk_size]
                if not chunk_text.strip(): continue
                cid=stable_id(d.doc_id, i, chunk_size, overlap)
                md=dict(d.metadata); md.update({"start_char":i,"end_char":i+len(chunk_text),"chunker":self.name})
                chunks.append(Chunk(cid,d.doc_id,chunk_text,md))
        return chunks

class RecursiveChunker(BaseChunker):
    name = "recursive"
    separators = ["\n\n", "\n", ". ", " ", ""]
    def chunk(self, documents: list[Document], config: dict) -> list[Chunk]:
        chunk_size = int(config.get("chunk_size", 512))
        overlap = config.get("overlap", 0)
        overlap_chars = int(chunk_size * overlap) if isinstance(overlap, float) else int(overlap)
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
