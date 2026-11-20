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


BASELINE_PROMPT_TEMPLATE = DefaultPromptBuilder().build("{question}", "{context}", {})

# Immutable, versioned additions: the default builder above deliberately stays unchanged.
PROMPT_PRESETS = {
    "concise_v1": """You are a careful RAG answer generator.
Answer only using the provided context. Give a brief, direct answer without unnecessary background.
Include citations using the exact source markers shown in the context.
If the context is insufficient, say that the answer is not available in the provided context.

Context:
{context}

Question: {question}

Answer:""",
    "evidence_first_v1": """You are a careful RAG answer generator.
Answer only using the provided context. Organize the answer around the relevant supporting evidence, then state the conclusion.
Include citations using the exact source markers shown in the context for each supported statement.
If the context is insufficient, say that the answer is not available in the provided context.

Context:
{context}

Question: {question}

Answer:""",
}


def preset_config(preset_id: str) -> dict:
    return {"type": "preset", "preset_id": preset_id, "template": PROMPT_PRESETS[preset_id]}


class PresetPromptBuilder(BasePromptBuilder):
    name = "preset"

    def build(self, question: str, context: str, config: dict) -> str:
        if config.get("preset_id") not in PROMPT_PRESETS:
            raise ValueError("Prompt must match a supported versioned preset")
        resolved = preset_config(config["preset_id"])
        if {**config, "template": config.get("template", resolved["template"])} != resolved:
            raise ValueError("Prompt must match a supported versioned preset")
        return resolved["template"].format(question=question, context=context)
