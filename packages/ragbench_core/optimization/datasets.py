"""Evidence validation and leakage-safe synthetic evaluation splits."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ragbench_core.engine.persistence import stable_hash
from ragbench_core.optimization.contracts import SyntheticQuestion

PROMPT_VERSION = "synthetic-evidence-v1"


def normalized(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def validate_questions(items: list[dict], documents: list[dict], verifier=None):
    docs = {d["doc_id"]: d["text"] for d in documents}
    accepted, rejected = [], []
    for raw in items:
        if not isinstance(raw, dict):
            rejected.append({"question": "", "reason": "Question must be an object"})
            continue
        try:
            item = SyntheticQuestion.model_validate(raw)
            for evidence in item.evidence:
                text = docs.get(evidence.doc_id, "")
                start = text.find(evidence.quote)
                if start < 0:
                    raise ValueError(
                        "Evidence quote is absent from its claimed document"
                    )
                evidence.start_char, evidence.end_char = (
                    start,
                    start + len(evidence.quote),
                )
            question = normalized(item.question)
            if any(
                SequenceMatcher(None, question, normalized(q.question)).ratio() >= 0.88
                for q in accepted
            ):
                raise ValueError("Duplicate or near-duplicate question")
            if verifier:
                verdict = verifier(item)
                if (
                    verdict.get("answerable") is not True
                    or verdict.get("supported") is not True
                ):
                    raise ValueError(
                        "Verifier rejected answerability or answer support"
                    )
                item.validation_reason = str(
                    verdict.get("reason", "Verified against source evidence")
                )
                item.automated_validated = True
            else:
                # Structural validation alone never confers semantic validation.
                item.automated_validated = bool(raw.get("automated_validated", False))
            item.question_id = stable_hash(
                {
                    "question": question,
                    "answer": item.expected_answer,
                    "evidence": [e.model_dump() for e in item.evidence],
                }
            )[:24]
            item.split = "unassigned"
            accepted.append(item)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"question": raw.get("question", ""), "reason": str(exc)})
    return accepted, rejected


def freeze_split(items: list[SyntheticQuestion]):
    """Connected overlapping evidence regions stay in the same split."""
    parents = list(range(len(items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, left in enumerate(items):
        for j in range(i):
            right = items[j]
            if any(
                normalized(a.quote) == normalized(b.quote)
                or (
                    a.doc_id == b.doc_id
                    and (
                        (a.start_char < b.end_char and b.start_char < a.end_char)
                        or a.start_char // 3000 == b.start_char // 3000
                    )
                )
                for a in left.evidence
                for b in right.evidence
            ):
                parents[find(i)] = find(j)
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    if len(groups) < 2:
        raise ValueError("Need at least two independent evidence groups for a holdout")
    ordered = sorted(
        groups.values(),
        key=lambda group: stable_hash(sorted(q.question_id for q in group)),
    )
    target = max(1, round(len(items) * 0.2))
    counts, total = [], 0
    for group in ordered[:-1]:
        total += len(group)
        counts.append(total)
    boundary = min(range(len(counts)), key=lambda i: abs(counts[i] - target)) + 1
    for index, group in enumerate(ordered):
        split = "holdout" if index < boundary else "tuning"
        group_id = stable_hash(sorted(q.question_id for q in group))[:24]
        for item in group:
            item.group_id, item.split = group_id, split
    return [q.model_dump() for q in items]


def legacy_question(item):
    docs = sorted({e["doc_id"] for e in item["evidence"]})
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "ground_truth_doc_ids": docs,
        "ground_truth_chunk_ids": [],
        "expected_citations": docs,
    }


def passages(documents, size=3000):
    # Fixed source windows are independent of every candidate's chunker.
    # Interleave documents so a long first file cannot consume the whole sample.
    return [
        {
            "doc_id": d["doc_id"],
            "text": d["text"][start : start + size],
            "start_char": start,
        }
        for start in range(0, max((len(d["text"]) for d in documents), default=0), size)
        for d in documents
        if d["text"][start : start + size].strip()
    ]
