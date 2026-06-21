from __future__ import annotations

import json
import re
from typing import Any

from ragbench_core.interfaces import BaseQueryRewriter
from ragbench_core.utils.clients import get_ollama_session


def _dedupe_nonempty(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        cleaned = " ".join(str(query).split()).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _ollama_generate(prompt: str, llm_config: dict[str, Any], temperature: float = 0.0) -> str:
    base_url = llm_config["base_url"].rstrip("/")
    resp = get_ollama_session(base_url).post(
        f"{base_url}/api/generate",
        json={
            "model": llm_config["model"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _parse_generated_list(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    items: list[str] = []
    for line in stripped.splitlines():
        cleaned = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', "", line).strip()
        cleaned = cleaned.strip('"\'` ,')
        if cleaned and cleaned not in {"[", "]"}:
            items.append(cleaned)
    return items


class NoRewrite(BaseQueryRewriter):
    name = "none"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        return [question]


class HyDERewrite(BaseQueryRewriter):
    name = "hyde"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        prompt = f"Write a concise hypothetical document passage that would answer this question. Do not answer directly.\nQuestion: {question}\nPassage:"
        try:
            hypo = _ollama_generate(prompt, llm_config, temperature=0.0)
            return _dedupe_nonempty([question, hypo]) if hypo else [question]
        except Exception:
            return [question]


class MultiQueryRewrite(BaseQueryRewriter):
    name = "multi_query"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        count = int(config.get("count", config.get("num_queries", 4)))
        count = max(1, count)
        prompt = (
            f"Generate {count} diverse search-query paraphrases for the question below. "
            "Preserve the original meaning and named entities. Return only a JSON array of strings.\n"
            f"Question: {question}"
        )
        try:
            variants = _parse_generated_list(_ollama_generate(prompt, llm_config, temperature=float(config.get("temperature", 0.2))))
            return _dedupe_nonempty([question, *variants])[: count + 1]
        except Exception:
            return [question]


class StepBackRewrite(BaseQueryRewriter):
    name = "step_back"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        prompt = (
            "Write one broader, more conceptual search query that would retrieve background context "
            "needed to answer the specific question. Return only the query text.\n"
            f"Question: {question}\nBroader query:"
        )
        try:
            broader = _ollama_generate(prompt, llm_config, temperature=float(config.get("temperature", 0.0)))
            return _dedupe_nonempty([question, broader]) if broader else [question]
        except Exception:
            return [question]


class DecomposeRewrite(BaseQueryRewriter):
    name = "decompose"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        max_subquestions = int(config.get("max_subquestions", 4))
        max_subquestions = max(1, max_subquestions)
        prompt = (
            "Split the question into atomic subquestions needed for retrieval. "
            f"Return at most {max_subquestions} subquestions as a JSON array of strings.\n"
            f"Question: {question}"
        )
        try:
            subquestions = _parse_generated_list(_ollama_generate(prompt, llm_config, temperature=float(config.get("temperature", 0.0))))
            return _dedupe_nonempty([question, *subquestions])[: max_subquestions + 1]
        except Exception:
            return [question]


class SynonymExpandRewrite(BaseQueryRewriter):
    name = "synonym_expand"

    def rewrite(self, question: str, config: dict, llm_config: dict) -> list[str]:
        synonyms = config.get("synonyms", {})
        expanded = question
        if isinstance(synonyms, dict):
            for term, replacements in synonyms.items():
                if isinstance(replacements, str):
                    replacements = [replacements]
                if not isinstance(replacements, list):
                    continue
                pattern = re.compile(rf"\b{re.escape(str(term))}\b", flags=re.IGNORECASE)
                if pattern.search(expanded):
                    addition = " OR ".join([str(term), *[str(r) for r in replacements]])
                    expanded = pattern.sub(f"({addition})", expanded, count=1)
        elif isinstance(synonyms, list):
            for group in synonyms:
                if not isinstance(group, list) or len(group) < 2:
                    continue
                terms = [str(t) for t in group]
                for term in terms:
                    pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
                    if pattern.search(expanded):
                        expanded = pattern.sub(f"({' OR '.join(terms)})", expanded, count=1)
                        break
        return _dedupe_nonempty([question, expanded])
