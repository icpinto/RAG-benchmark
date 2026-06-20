import sys
import types

from ragbench_core.components.extractors import DoclingExtractor

def test_docling_extractor_ignores_urls_and_unsupported_suffix(tmp_path):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("plain text", encoding="utf-8")

    extractor = DoclingExtractor()

    assert extractor.extract("https://example.com/file.pdf", {}) == []
    assert extractor.extract(str(unsupported), {}) == []


def test_docling_extractor_converts_supported_file_with_metadata(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF fake")

    class FakeFormat:
        value = "pdf"

    class FakeInput:
        format = FakeFormat()

    class FakePage:
        page_no = 1

    class FakeTable:
        label = "table-1"
        num_rows = 2
        num_cols = 3

    class FakeImage:
        label = "image-1"

    class FakeDocument:
        pages = {1: FakePage()}
        tables = [FakeTable()]
        pictures = [FakeImage()]

        def export_to_markdown(self):
            return "# Converted\n\n| A | B |"

    class FakeResult:
        input = FakeInput()
        document = FakeDocument()

    class FakeDocumentConverter:
        def convert(self, path):
            assert path == str(source)
            return FakeResult()

    fake_module = types.ModuleType("docling.document_converter")
    fake_module.DocumentConverter = FakeDocumentConverter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_module)

    doc = DoclingExtractor().extract(str(source), {})[0]

    assert doc.doc_id == "report.pdf"
    assert doc.source == str(source)
    assert doc.text == "# Converted\n\n| A | B |"
    assert doc.metadata["file_name"] == "report.pdf"
    assert doc.metadata["file_extension"] == ".pdf"
    assert doc.metadata["type"] == "docling"
    assert doc.metadata["mime_type"] == "application/pdf"
    assert doc.metadata["file_size_bytes"] == len(b"%PDF fake")
    assert doc.metadata["modified_at"]
    assert doc.metadata["docling_format"] == "pdf"
    assert doc.metadata["pages_count"] == 1
    assert doc.metadata["tables_count"] == 1
    assert doc.metadata["images_count"] == 1
    assert doc.metadata["has_tables"] is True
    assert doc.metadata["has_images"] is True
    assert doc.metadata["pages"] == [{"index": 0, "page_no": 1}]
    assert doc.metadata["tables"] == [{"index": 0, "label": "table-1", "num_rows": 2, "num_cols": 3}]
    assert doc.metadata["images"] == [{"index": 0, "label": "image-1"}]

