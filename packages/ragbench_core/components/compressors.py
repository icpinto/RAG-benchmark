from ragbench_core.interfaces import BaseContextCompressor
from ragbench_core.models.schema import RetrievalResult

class NoCompressor(BaseContextCompressor):
    name = "none"
    def compress(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        return results

class TopNCompressor(BaseContextCompressor):
    name = "top_n"
    def compress(self, question: str, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        return results[:int(config.get("max_chunks", 5))]
