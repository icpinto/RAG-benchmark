from ragbench_core.interfaces import BaseQueryRewriter
from ragbench_core.utils.clients import get_ollama_session

class NoRewrite(BaseQueryRewriter):
    name = "none"
    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        return [question]

class HyDERewrite(BaseQueryRewriter):
    name = "hyde"
    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        prompt = f"Write a concise hypothetical document passage that would answer this question. Do not answer directly.\nQuestion: {question}\nPassage:"
        try:
            base_url=llm_config['base_url'].rstrip('/')
            resp=get_ollama_session(base_url).post(f"{base_url}/api/generate", json={"model":llm_config["model"],"prompt":prompt,"stream":False,"options":{"temperature":0}}, timeout=120)
            resp.raise_for_status()
            hypo=resp.json().get("response", "").strip()
            return [question, hypo] if hypo else [question]
        except Exception:
            return [question]
