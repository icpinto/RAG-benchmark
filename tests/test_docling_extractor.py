import sys
import types
from importlib.machinery import ModuleSpec

from ragbench_core.components.extractors import DoclingExtractor

def test_docling_extractor_ignores_urls_and_unsupported_suffix(tmp_path):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("plain text", encoding="utf-8")

    extractor = DoclingExtractor()

    assert extractor.extract("https://example.com/file.pdf", {}) == []
    assert extractor.extract(str(unsupported), {}) == []


def test_docling_extractor_converts_supported_file_with_structured_table_and_image_metadata(tmp_path, monkeypatch):
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
        page_no = 2
        num_rows = 2
        num_cols = 3
        caption = "Quarterly results"

        def export_to_markdown(self):
            return "| A | B |\n| - | - |\n| 1 | 2 |"

    class FakeImage:
        label = "image-1"
        page_number = 3
        caption = "Architecture diagram"
        ocr_text = "Gateway -> Index"
        size = {"width": 640, "height": 480}
        image_ref = "#/pictures/0"
        description = "A diagram showing request flow"

    class FakeDocument:
        pages = {1: FakePage()}
        tables = [FakeTable()]
        pictures = [FakeImage()]

        def export_to_markdown(self):
            return "# Converted"

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
    assert "# Converted" in doc.text
    assert "Quarterly results" in doc.text
    assert "| A | B |" in doc.text
    assert "Architecture diagram" in doc.text
    assert "Gateway -> Index" in doc.text
    assert "A diagram showing request flow" not in doc.text
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
    assert doc.metadata["tables"] == [
        {
            "index": 0,
            "page_number": 2,
            "row_count": 2,
            "column_count": 3,
            "caption": "Quarterly results",
            "label": "table-1",
            "markdown": "| A | B |\n| - | - |\n| 1 | 2 |",
            "text": "| A | B |\n| - | - |\n| 1 | 2 |",
        }
    ]
    assert doc.metadata["images"] == [
        {
            "index": 0,
            "page_number": 3,
            "caption": "Architecture diagram",
            "ocr_text": "Gateway -> Index",
            "dimensions": {"width": 640, "height": 480},
            "source_id": "#/pictures/0",
            "label": "image-1",
        }
    ]

    doc_with_descriptions = DoclingExtractor().extract(
        str(source), {"image_descriptions": {"enabled": True}}
    )[0]
    assert "A diagram showing request flow" in doc_with_descriptions.text
    assert doc_with_descriptions.metadata["images"][0]["description"] == "A diagram showing request flow"


def test_docling_extractor_patches_transformers_image_text_alias(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__spec__ = ModuleSpec("transformers", loader=None)
    fake_transformers.AutoModelForVision2Seq = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    DoclingExtractor()._patch_transformers_image_text_alias()

    assert fake_transformers.AutoModelForImageTextToText is fake_transformers.AutoModelForVision2Seq
