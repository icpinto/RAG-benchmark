from ragbench_core.interfaces import BaseFilter
from ragbench_core.models.schema import RetrievalResult

class NoFilter(BaseFilter):
    name = "none"
    def filter(self, results: list[RetrievalResult], config: dict) -> list[RetrievalResult]:
        return results
