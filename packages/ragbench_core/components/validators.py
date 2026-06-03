from ragbench_core.interfaces import BaseCitationValidator
from ragbench_core.models.schema import GeneratedAnswer

class SimpleCitationValidator(BaseCitationValidator):
    name = "simple"
    def validate(self, answer: GeneratedAnswer, expected_citations: list[str], config: dict) -> dict:
        text=answer.answer or ""
        found=[]
        for c in expected_citations:
            if c and c in text: found.append(c)
        return {"expected_citations":len(expected_citations),"found_citations":len(found),"citation_coverage":len(found)/len(expected_citations) if expected_citations else 1.0}
