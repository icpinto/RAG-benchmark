from ragbench_core.components.chunkers import FixedSizeChunker
from ragbench_core.models.schema import Document

def test_fixed_chunker():
    docs=[Document("d1","src","a"*1000,{})]
    chunks=FixedSizeChunker().chunk(docs,{"chunk_size":100,"overlap":0.2})
    assert len(chunks)>1
    assert chunks[0].doc_id=="d1"
