from ragbench_core.interfaces import BaseMetadataEnricher
from ragbench_core.models.schema import Chunk

class BasicMetadataEnricher(BaseMetadataEnricher):
    name = "basic"
    def enrich(self, chunks: list[Chunk], config: dict) -> list[Chunk]:
        for c in chunks:
            c.metadata["char_length"] = len(c.text)
            c.metadata["word_count"] = len(c.text.split())
        return chunks
