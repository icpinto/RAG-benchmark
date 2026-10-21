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
            metadata = self._build_metadata(p, result, document, config)
            text = self._enrich_text_with_tables_and_images(text, metadata)
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

    def _build_metadata(self, path: Path, result, document, config: dict | None = None) -> dict:
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
        table_summaries = self._summarize_tables(tables)
        image_summaries = self._summarize_images(images, config or {})
        if page_summaries:
            metadata["pages"] = page_summaries
        if table_summaries:
            metadata["tables"] = table_summaries
        if image_summaries:
            metadata["images"] = image_summaries
        return metadata

    def _enrich_text_with_tables_and_images(self, text: str, metadata: dict) -> str:
        parts = [text.strip()] if text and text.strip() else []
        seen = {part for part in parts}

        for table in metadata.get("tables", []):
            for key in ("caption", "markdown", "text"):
                value = table.get(key)
                if value and value not in seen:
                    parts.append(value)
                    seen.add(value)

        for image in metadata.get("images", []):
            for key in ("caption", "ocr_text", "description"):
                value = image.get(key)
                if value and value not in seen:
                    parts.append(value)
                    seen.add(value)

        return "\n\n".join(parts)

    def _summarize_tables(self, tables) -> list[dict]:
        summaries = []
        for index, table in enumerate(self._iter_collection(tables)):
            summary = {"index": index}
            self._copy_first_attr(summary, table, "page_number", ["page_no", "page_number", "page", "prov.page_no"])
            self._copy_first_attr(summary, table, "row_count", ["num_rows", "nrows", "row_count", "rows_count"])
            self._copy_first_attr(summary, table, "column_count", ["num_cols", "ncols", "column_count", "columns_count"])
            self._copy_first_attr(summary, table, "caption", ["caption", "captions", "title"])
            self._copy_first_attr(summary, table, "label", ["label", "name"])

            markdown = self._export_item_text(table, ["export_to_markdown", "to_markdown", "export_to_html", "text"])
            if markdown:
                summary["markdown"] = markdown
                summary.setdefault("text", markdown)

            summaries.append(summary)
        return summaries

    def _summarize_images(self, images, config: dict) -> list[dict]:
        summaries = []
        image_description_config = config.get("image_descriptions", {})
        descriptions_enabled = (
            bool(image_description_config.get("enabled", False))
            if isinstance(image_description_config, dict)
            else bool(image_description_config)
        )
        for index, image in enumerate(self._iter_collection(images)):
            summary = {"index": index}
            self._copy_first_attr(summary, image, "page_number", ["page_no", "page_number", "page", "prov.page_no"])
            self._copy_first_attr(summary, image, "caption", ["caption", "captions", "title"])
            self._copy_first_attr(summary, image, "ocr_text", ["ocr_text", "text", "extracted_text"])
            self._copy_first_attr(summary, image, "dimensions", ["size", "dimensions", "image.size"])
            self._copy_first_attr(summary, image, "source_id", ["image_ref", "self_ref", "ref", "uri", "id", "label"])
            self._copy_first_attr(summary, image, "label", ["label", "name"])

            if descriptions_enabled:
                description = self._available_image_description(image, config)
                if description:
                    summary["description"] = description

            summaries.append(summary)
        return summaries

    def _available_image_description(self, image, config: dict) -> str | None:
        # External vision calls are intentionally not performed here. If Docling (or
        # a caller-supplied preprocessing step) has already attached a description,
        # expose it only when explicitly enabled so normal ingestion needs no API.
        return self._stringify(self._first_attr(image, ["description", "image_description", "alt_text"]))

    def _iter_collection(self, collection):
        if collection is None:
            return []
        return collection.values() if isinstance(collection, dict) else collection

    def _copy_first_attr(self, target: dict, obj, key: str, attr_paths: list[str]) -> None:
        value = self._first_attr(obj, attr_paths)
        if value is not None:
            target[key] = self._stringify(value)

    def _first_attr(self, obj, attr_paths: list[str]):
        for attr_path in attr_paths:
            value = obj
            for attr in attr_path.split("."):
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                value = getattr(value, attr, None) if value is not None else None
                if value is None:
                    break
            if value is not None:
                return value
        return None

    def _export_item_text(self, item, attr_names: list[str]) -> str | None:
        for attr in attr_names:
            value = getattr(item, attr, None)
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            text = self._stringify(value)
            if text:
                return text
        return None

    def _stringify(self, value) -> str | int | float | bool | None:
        if value is None:
            return None
        if hasattr(value, "text"):
            value = value.text
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            values = [self._stringify(item) for item in value]
            values = [item for item in values if item]
            return "\n".join(str(item) for item in values) if values else None
        if isinstance(value, dict):
            return {key: self._stringify(item) for key, item in value.items()}
        return str(value)

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

        if config.get("_snapshot_html"):
            html = Path(config["_snapshot_html"]).read_bytes().decode(config.get("_snapshot_encoding", "utf-8"), errors="replace")
        else:
            resp = requests.get(source, timeout=30, headers={"User-Agent":"ragbench/0.1"})
            resp.raise_for_status()
            html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else source
        text = soup.get_text("\n")
        doc_id = source.replace("https://", "").replace("http://", "").replace("/", "_")[:180]
        return [Document(doc_id=doc_id, source=source, text=text, metadata={"url":source,"title":title,"type":"html_url"})]
