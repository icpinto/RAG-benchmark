from __future__ import annotations

from typing import Any

from ragbench_core.interfaces import BaseGenerator
from ragbench_core.utils.clients import get_ollama_session, record_token_usage


class OllamaGenerator(BaseGenerator):
    name = "ollama"

    def generate(self, prompt: str, config: dict) -> dict[str, Any]:
        base_url=config.get("base_url", "http://ollama:11434").rstrip("/")
        model=config.get("model", "gpt-oss:20b")
        payload={"model":model,"prompt":prompt,"stream":False,"options":{"temperature":config.get("temperature",0)}}
        resp=get_ollama_session(base_url).post(f"{base_url}/api/generate", json=payload, timeout=int(config.get("timeout", 240)))
        resp.raise_for_status()
        body = resp.json()
        record_token_usage(prompt_tokens=body.get("prompt_eval_count"), completion_tokens=body.get("eval_count"))
        return {
            "answer": body.get("response", "").strip(),
            "metadata": {
                "generation_model": model,
                "token_usage": {
                    # Ollama reports these counts when available. The metrics layer
                    # falls back to local tiktoken estimates when they are absent.
                    "prompt_tokens": body.get("prompt_eval_count"),
                    "completion_tokens": body.get("eval_count"),
                },
            },
        }
