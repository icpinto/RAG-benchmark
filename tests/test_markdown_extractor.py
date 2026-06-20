from ragbench_core.components.extractors import MarkdownExtractor


def test_markdown_extractor_collects_structured_metadata(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text(
        "---\n"
        "title: Guide\n"
        "tags:\n"
        "  - rag\n"
        "---\n\n"
        "# Intro\n\n"
        "See the [documentation](https://example.com/docs).\n\n"
        "![Architecture](images/architecture.png \"Diagram\")\n\n"
        "## Results\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        "| Recall | 0.95 |\n\n"
        "```python\n"
        "print('hello')\n"
        "```\n",
        encoding="utf-8",
    )

    doc = MarkdownExtractor().extract(str(source), {})[0]

    assert doc.doc_id == "guide.md"
    assert doc.metadata["file_name"] == "guide.md"
    assert doc.metadata["type"] == "markdown"
    assert doc.metadata["frontmatter"] == {"title": "Guide", "tags": ["rag"]}
    assert doc.metadata["headings"] == [
        {"text": "Intro", "level": 1, "section_path": ["Intro"]},
        {"text": "Results", "level": 2, "section_path": ["Intro", "Results"]},
    ]
    assert doc.metadata["section_paths"] == [["Intro"], ["Intro", "Results"]]
    assert doc.metadata["links"] == [
        {"url": "https://example.com/docs", "title": "", "text": "documentation"}
    ]
    assert doc.metadata["images"] == [
        {"src": "images/architecture.png", "alt": "Architecture", "title": "Diagram"}
    ]
    assert doc.metadata["tables_count"] == 1
    assert doc.metadata["code_blocks"] == [
        {"language": "python", "section_path": ["Intro", "Results"]}
    ]

    assert "Intro" in doc.text
    assert "See the documentation." in doc.text
    assert "Metric" in doc.text
    assert "Recall" in doc.text
    assert "print('hello')" in doc.text
    assert "---" not in doc.text


def test_markdown_extractor_ignores_non_markdown_files(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("# Not markdown", encoding="utf-8")

    assert MarkdownExtractor().extract(str(source), {}) == []
