from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import mimetypes

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

        raw_text = p.read_text(errors="ignore")
        frontmatter, markdown_text = self._split_frontmatter(raw_text)
        from markdown_it import MarkdownIt

        tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(markdown_text)
        text, metadata = self._tokens_to_text_and_metadata(tokens)
        metadata.update({"file_name": p.name, "type": "markdown"})
        if frontmatter is not None:
            metadata["frontmatter"] = frontmatter
        return [Document(doc_id=p.name, source=str(p), text=text, metadata=metadata)]

    def _split_frontmatter(self, text: str) -> tuple[dict | str | None, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, text
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                frontmatter_text = "\n".join(lines[1:index])
                markdown_text = "\n".join(lines[index + 1 :])
                parsed = self._parse_frontmatter(frontmatter_text)
                return parsed, markdown_text
        return None, text

    def _parse_frontmatter(self, frontmatter_text: str) -> dict | str:
        if not frontmatter_text.strip():
            return {}
        try:
            import yaml
        except ModuleNotFoundError:
            parsed: dict[str, object] = {}
            current_key = None
            for line in frontmatter_text.splitlines():
                if line.startswith("  - ") and current_key:
                    parsed.setdefault(current_key, []).append(line[4:].strip())
                elif ":" in line:
                    key, value = line.split(":", 1)
                    current_key = key.strip()
                    parsed[current_key] = value.strip() or []
            return parsed or frontmatter_text

        parsed = yaml.safe_load(frontmatter_text)
        return parsed if isinstance(parsed, dict) else frontmatter_text

    def _tokens_to_text_and_metadata(self, tokens: list) -> tuple[str, dict]:
        text_parts: list[str] = []
        headings: list[dict] = []
        links: list[dict] = []
        images: list[dict] = []
        code_blocks: list[dict] = []
        section_stack: list[dict] = []
        tables_count = 0
        table_depth = 0

        for token in tokens:
            if token.type == "heading_open":
                level = int(token.tag[1])
                inline = self._next_inline(tokens, token)
                heading_text = self._render_inline_text(inline.children if inline else [])
                section_stack = [item for item in section_stack if item["level"] < level]
                section_stack.append({"level": level, "title": heading_text})
                section_path = [item["title"] for item in section_stack]
                headings.append({"text": heading_text, "level": level, "section_path": section_path})
                text_parts.append(heading_text)
            elif token.type == "inline":
                if token.map and self._previous_token_type(tokens, token) == "heading_open":
                    self._collect_inline_metadata(token.children or [], links, images)
                    continue
                inline_text = self._render_inline_text(token.children or [])
                if inline_text:
                    text_parts.append(inline_text)
                self._collect_inline_metadata(token.children or [], links, images)
            elif token.type == "fence":
                language = token.info.strip().split()[0] if token.info.strip() else ""
                code_blocks.append({"language": language, "section_path": [item["title"] for item in section_stack]})
                if token.content.strip():
                    text_parts.append(token.content.strip())
            elif token.type == "table_open":
                tables_count += 1
                table_depth += 1
            elif token.type == "table_close":
                table_depth = max(0, table_depth - 1)
            elif table_depth and token.type in {"th_open", "td_open"}:
                inline = self._next_inline(tokens, token)
                cell_text = self._render_inline_text(inline.children if inline else [])
                if cell_text:
                    text_parts.append(cell_text)

        metadata = {
            "headings": headings,
            "links": links,
            "images": images,
            "tables_count": tables_count,
            "code_blocks": code_blocks,
            "section_paths": [heading["section_path"] for heading in headings],
        }
        return "\n\n".join(dict.fromkeys(part.strip() for part in text_parts if part.strip())), metadata

    def _next_inline(self, tokens: list, token):
        index = next(i for i, candidate in enumerate(tokens) if candidate is token)
        for next_token in tokens[index + 1 :]:
            if next_token.type == "inline":
                return next_token
            if next_token.nesting == -1:
                return None
        return None

    def _previous_token_type(self, tokens: list, token) -> str | None:
        index = next(i for i, candidate in enumerate(tokens) if candidate is token)
        return tokens[index - 1].type if index > 0 else None

    def _render_inline_text(self, children: list) -> str:
        parts = []
        for child in children:
            if child.type in {"text", "code_inline"}:
                parts.append(child.content)
            elif child.type == "softbreak":
                parts.append("\n")
            elif child.type == "image":
                parts.append(child.content or child.attrGet("alt") or "")
        return "".join(parts).strip()

    def _collect_inline_metadata(self, children: list, links: list[dict], images: list[dict]) -> None:
        link_stack: list[dict] = []
        for child in children:
            if child.type == "link_open":
                link_stack.append({"url": child.attrGet("href") or "", "title": child.attrGet("title") or "", "text": ""})
            elif child.type == "link_close" and link_stack:
                links.append(link_stack.pop())
            elif child.type == "image":
                images.append({"src": child.attrGet("src") or "", "alt": child.content or child.attrGet("alt") or "", "title": child.attrGet("title") or ""})
            elif link_stack and child.type in {"text", "code_inline"}:
                link_stack[-1]["text"] += child.content

class DoclingExtractor(BaseExtractor):
    name = "docling"
    supported_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    def extract(self, source: str, config: dict) -> list[Document]:
        if source.startswith(("http://", "https://")):
            return []

        p = Path(source)
        file_extension = p.suffix.lower()
        if file_extension not in self.supported_extensions:
            return []

        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(p))
            document = result.document
            text = self._export_text(document, config)
            metadata = self._build_metadata(p, result, document)
            return [Document(doc_id=p.name, source=str(p), text=text, metadata=metadata)]
        except Exception as exc:
            if config.get("ignore_errors"):
                return []
            raise RuntimeError(f"Docling failed to extract {p}: {exc}") from exc

    def _export_text(self, document, config: dict) -> str:
        export_format = config.get("export_format", "markdown")
        if export_format == "text" and hasattr(document, "export_to_text"):
            return document.export_to_text()
        if hasattr(document, "export_to_markdown"):
            return document.export_to_markdown()
        if hasattr(document, "export_to_text"):
            return document.export_to_text()
        return str(document)

    def _build_metadata(self, path: Path, result, document) -> dict:
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(path.name)
        pages_count = self._count_collection(getattr(document, "pages", None))
        tables = getattr(document, "tables", None)
        images = self._first_present_collection(document, ["pictures", "images"])
        tables_count = self._count_collection(tables)
        images_count = self._count_collection(images)

        metadata = {
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "type": "docling",
            "mime_type": mime_type,
            "file_size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "docling_format": self._detect_docling_format(result),
            "pages_count": pages_count,
            "tables_count": tables_count,
            "images_count": images_count,
            "has_tables": tables_count > 0,
            "has_images": images_count > 0,
        }

        page_summaries = self._summarize_collection(getattr(document, "pages", None), ["page_no", "page_number", "size"], limit=20)
        table_summaries = self._summarize_collection(tables, ["label", "caption", "num_rows", "num_cols"], limit=20)
        image_summaries = self._summarize_collection(images, ["label", "caption", "image_ref"], limit=20)
        if page_summaries:
            metadata["pages"] = page_summaries
        if table_summaries:
            metadata["tables"] = table_summaries
        if image_summaries:
            metadata["images"] = image_summaries
        return metadata

    def _detect_docling_format(self, result) -> str | None:
        for attr_path in [("input", "format"), ("input", "file", "format"), ("document", "origin", "mimetype")]:
            value = result
            for attr in attr_path:
                value = getattr(value, attr, None)
                if value is None:
                    break
            if value is not None:
                return getattr(value, "value", str(value))
        return None

    def _first_present_collection(self, obj, names: list[str]):
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return None

    def _count_collection(self, collection) -> int:
        if collection is None:
            return 0
        if isinstance(collection, dict):
            return len(collection)
        try:
            return len(collection)
        except TypeError:
            return sum(1 for _ in collection)

    def _summarize_collection(self, collection, attributes: list[str], limit: int) -> list[dict]:
        if collection is None:
            return []
        values = collection.values() if isinstance(collection, dict) else collection
        summaries = []
        for index, item in enumerate(values):
            if index >= limit:
                break
            summary = {"index": index}
            for attr in attributes:
                value = getattr(item, attr, None)
                if value is None:
                    continue
                if hasattr(value, "text"):
                    value = value.text
                summary[attr] = str(value) if not isinstance(value, (str, int, float, bool)) else value
            if len(summary) > 1:
                summaries.append(summary)
        return summaries


class PdfExtractor(BaseExtractor):
    name = "pdf"
    def extract(self, source: str, config: dict) -> list[Document]:
        p = Path(source)
        if p.suffix.lower() != ".pdf": return []
        from pypdf import PdfReader

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
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(source, timeout=30, headers={"User-Agent":"ragbench/0.1"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else source
        text = soup.get_text("\n")
        doc_id = source.replace("https://", "").replace("http://", "").replace("/", "_")[:180]
        return [Document(doc_id=doc_id, source=source, text=text, metadata={"url":source,"title":title,"type":"html_url"})]
