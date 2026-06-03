from __future__ import annotations
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ragbench_core.interfaces import BaseExtractor
from ragbench_core.models.schema import Document

class TxtExtractor(BaseExtractor):
    name = "txt"
    def extract(self, source: str, config: dict) -> list[Document]:
        p = Path(source)
        if p.suffix.lower() != ".txt": return []
        return [Document(doc_id=p.name, source=str(p), text=p.read_text(errors="ignore"), metadata={"file_name": p.name, "type":"txt"})]

class MarkdownExtractor(BaseExtractor):
    name = "markdown"
    def extract(self, source: str, config: dict) -> list[Document]:
        p = Path(source)
        if p.suffix.lower() not in [".md", ".markdown"]: return []
        return [Document(doc_id=p.name, source=str(p), text=p.read_text(errors="ignore"), metadata={"file_name": p.name, "type":"markdown"})]

class PdfExtractor(BaseExtractor):
    name = "pdf"
    def extract(self, source: str, config: dict) -> list[Document]:
        p = Path(source)
        if p.suffix.lower() != ".pdf": return []
        reader = PdfReader(str(p))
        docs=[]
        text_parts=[]
        for i, page in enumerate(reader.pages):
            text_parts.append(f"\n\n[page={i+1}]\n" + (page.extract_text() or ""))
        docs.append(Document(doc_id=p.name, source=str(p), text="\n".join(text_parts), metadata={"file_name":p.name,"type":"pdf","pages":len(reader.pages)}))
        return docs

class HtmlUrlExtractor(BaseExtractor):
    name = "html_url"
    def extract(self, source: str, config: dict) -> list[Document]:
        if not source.startswith(("http://", "https://")):
            return []
        resp = requests.get(source, timeout=30, headers={"User-Agent":"ragbench/0.1"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else source
        text = soup.get_text("\n")
        doc_id = source.replace("https://", "").replace("http://", "").replace("/", "_")[:180]
        return [Document(doc_id=doc_id, source=source, text=text, metadata={"url":source,"title":title,"type":"html_url"})]
