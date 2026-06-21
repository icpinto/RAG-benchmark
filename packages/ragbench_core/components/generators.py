from ragbench_core.interfaces import BaseGenerator
from ragbench_core.utils.clients import get_ollama_session

class OllamaGenerator(BaseGenerator):
    name = "ollama"
    def generate(self, prompt: str, config: dict) -> str:
        base_url=config.get("base_url", "http://ollama:11434").rstrip("/")
        model=config.get("model", "gpt-oss:20b")
        payload={"model":model,"prompt":prompt,"stream":False,"options":{"temperature":config.get("temperature",0)}}
        resp=get_ollama_session(base_url).post(f"{base_url}/api/generate", json=payload, timeout=int(config.get("timeout", 240)))
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
