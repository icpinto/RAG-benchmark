from ragbench_core.interfaces import BaseContextAssembler, BasePromptBuilder
from ragbench_core.models.schema import RetrievalResult

class DefaultContextAssembler(BaseContextAssembler):
    name = "default"
    def assemble(self, results: list[RetrievalResult], config: dict) -> str:
        max_chars=int(config.get("max_context_chars", 12000))
        parts=[]
        total=0
        for r in results:
            citation=f"[{r.chunk.doc_id}::{r.chunk.chunk_id}]"
            block=f"{citation}\n{r.chunk.text}\n"
            if total+len(block)>max_chars: break
            parts.append(block); total+=len(block)
        return "\n---\n".join(parts)

class DefaultPromptBuilder(BasePromptBuilder):
    name = "default"
    def build(self, question: str, context: str, config: dict) -> str:
        return f"""You are a careful RAG answer generator.
Answer only using the provided context.
Include citations using the exact source markers shown in the context.
If the context is insufficient, say that the answer is not available in the provided context.

Context:
{context}

Question: {question}

Answer:"""
