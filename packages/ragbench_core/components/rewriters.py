import requests
from ragbench_core.interfaces import BaseQueryRewriter

class NoRewrite(BaseQueryRewriter):
    name = "none"
    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        return [question]

class HyDERewrite(BaseQueryRewriter):
    name = "hyde"
    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        prompt = f"Write a concise hypothetical document passage that would answer this question. Do not answer directly.\nQuestion: {question}\nPassage:"
        try:
            resp=requests.post(f"{llm_config['base_url'].rstrip('/')}/api/generate", json={"model":llm_config["model"],"prompt":prompt,"stream":False,"options":{"temperature":0}}, timeout=120)
            resp.raise_for_status()
            hypo=resp.json().get("response", "").strip()
            return [question, hypo] if hypo else [question]
        except Exception:
            return [question]
