from __future__ import annotations
from pathlib import Path
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
