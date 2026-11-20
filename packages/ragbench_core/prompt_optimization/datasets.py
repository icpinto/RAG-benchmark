"""Frozen training/validation/evaluation partitions, independent of candidate prompts."""

import re
from difflib import SequenceMatcher

from ragbench_core.engine.persistence import stable_hash


def question_fingerprint(question):
    return stable_hash(" ".join(re.findall(r"\w+", question.lower())))


def split_questions(questions, *, evaluation=True):
    if not questions or any(
        not q.get("expected_answer", "").strip() for q in questions
    ):
        raise ValueError(
            "Prompt compilation requires labeled questions with reference answers"
        )
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in compilation data")
    parents = list(range(len(questions)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def related(a, b):
        if a.get("group_id") and b.get("group_id"):
            return a["group_id"] == b["group_id"]
        left, right = a.get("evidence", []), b.get("evidence", [])
        if left and right:
            return any(
                x["quote"] == y["quote"]
                or (
                    x["doc_id"] == y["doc_id"]
                    and (
                        x.get("start_char", 0) // 3000 == y.get("start_char", 0) // 3000
                        or x.get("start_char", 0) < y.get("end_char", 0)
                        and y.get("start_char", 0) < x.get("end_char", 0)
                    )
                )
                for x in left
                for y in right
            )
        docs_a, docs_b = (
            set(a.get("ground_truth_doc_ids", [])),
            set(b.get("ground_truth_doc_ids", [])),
        )
        if not docs_a or not docs_b:
            raise ValueError(
                "Supply source document IDs or evidence groups to split compilation questions safely"
            )
        return bool(docs_a & docs_b)

    for i, item in enumerate(questions):
        if (
            not item.get("group_id")
            and not item.get("evidence")
            and not item.get("ground_truth_doc_ids")
        ):
            raise ValueError(
                "Supply source document IDs or evidence groups to split compilation questions safely"
            )
        for j in range(i):
            other = questions[j]
            near_duplicate = (
                SequenceMatcher(
                    None, item["question"].lower(), other["question"].lower()
                ).ratio()
                >= 0.88
            )
            if near_duplicate or related(item, other):
                parents[find(i)] = find(j)
    groups = {}
    for i, item in enumerate(questions):
        groups.setdefault(find(i), []).append(item["question_id"])
    ordered = sorted(groups.values(), key=lambda group: stable_hash(sorted(group)))
    needed = 3 if evaluation else 2
    if len(ordered) < needed:
        raise ValueError(
            f"Prompt compilation needs at least {needed} independent evidence groups; add labeled questions from independent sources"
        )

    def take(groups, fraction, leave):
        target = max(1, round(sum(map(len, groups)) * fraction))
        sums, total = [], 0
        for group in groups[:-leave]:
            total += len(group)
            sums.append(total)
        boundary = min(range(len(sums)), key=lambda i: abs(sums[i] - target)) + 1
        return groups[:boundary], groups[boundary:]

    evaluation_groups, remaining = (
        take(ordered, 0.2, 2) if evaluation else ([], ordered)
    )
    validation_groups, training_groups = take(remaining, 0.25, 1)
    result = {
        "training": sorted(q for group in training_groups for q in group),
        "validation": sorted(q for group in validation_groups for q in group),
        "evaluation": sorted(q for group in evaluation_groups for q in group),
        "groups": [sorted(group) for group in ordered],
    }
    return {**result, "hash": stable_hash(result)}
