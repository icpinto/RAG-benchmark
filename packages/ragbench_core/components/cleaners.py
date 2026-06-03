import re
from ragbench_core.interfaces import BaseCleaner
from ragbench_core.models.schema import Document

class BasicCleaner(BaseCleaner):
    name = "basic"
    def clean(self, documents: list[Document], config: dict) -> list[Document]:
        cleaned=[]
        for d in documents:
            text = d.text.replace("\x00", " ")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            cleaned.append(Document(d.doc_id, d.source, text.strip(), d.metadata))
        return cleaned
