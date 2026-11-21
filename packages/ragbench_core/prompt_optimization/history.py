"""Compilation-local, JSON-only observations; never part of inference artifacts."""

from copy import deepcopy
from threading import RLock

from ragbench_core.engine.persistence import stable_hash


def demonstrations(items):
    return [
        {
            key: getattr(item, key) if not isinstance(item, dict) else item[key]
            for key in ("question", "context", "answer")
        }
        for item in items
    ]


class CandidateHistory:
    def __init__(self, prepared, splits):
        self.items = {}
        self.lock = RLock()
        self.partitions = {
            partition: {
                (e["question"], e["context"])
                for e in prepared
                if e["question_id"] in set(ids)
            }
            for partition, ids in splits.items()
            if partition in {"training", "validation", "evaluation"}
        }

    def observe(self, instructions, demos, *, kind="prompt", source="executed"):
        content = {
            "kind": kind,
            "instructions": instructions,
            "demonstrations": demonstrations(demos) if demos is not None else None,
        }
        identity = stable_hash(content)
        with self.lock:
            if identity not in self.items:
                self.items[identity] = {
                    "id": identity,
                    "proposal_order": len(self.items) + 1,
                    **content,
                    "sources": [],
                    "evaluations": [],
                    "selected": False,
                }
            item = self.items[identity]
            if source not in item["sources"]:
                item["sources"].append(source)
        return identity

    def program(self, program, *, source="evaluated"):
        predictor = getattr(program, "answer", None)
        if predictor is None:
            return None
        return self.observe(
            predictor.signature.instructions, predictor.demos, source=source
        )

    def evaluation(self, identity, dataset, score, *, stage="optimizer"):
        if identity is None:
            return
        inputs = {(e.question, e.context) for e in dataset}
        partition = next(
            (
                name
                for name, values in self.partitions.items()
                if inputs and inputs <= values
            ),
            "mixed",
        )
        scope = "full" if inputs == self.partitions.get(partition) else "minibatch"
        with self.lock:
            self.items[identity]["evaluations"].append(
                {
                    "score": float(score) if score is not None else None,
                    "sample_count": len(dataset),
                    "partition": partition,
                    "scope": scope,
                    "stage": stage,
                }
            )

    def finish(self, payload, validation, score):
        identity = self.observe(
            payload["instructions"], payload["demonstrations"], source="compiled_output"
        )
        self.items[identity]["selected"] = True
        self.evaluation(identity, validation, score, stage="final_validation")
        return {"version": 1, "candidates": deepcopy(list(self.items.values()))}


def evaluation_callback(dspy, history):
    from dspy.utils.callback import BaseCallback

    class CaptureEvaluations(BaseCallback):
        def __init__(self):
            self.pending = {}
            self.proposals = set()
            self.proposal_count = 0

        def on_module_start(self, call_id, instance, inputs):
            if type(instance).__name__ == "GenerateModuleInstruction":
                self.proposals.add(call_id)

        def on_module_end(self, call_id, outputs, exception=None):
            if call_id not in self.proposals:
                return
            self.proposals.remove(call_id)
            if exception is None and outputs is not None:
                from dspy.propose.utils import strip_prefix

                self.proposal_count += 1
                # MIPRO replaces proposal 0 with the seed instruction after
                # generation. Observe the original before that replacement.
                history.observe(
                    strip_prefix(outputs.proposed_instruction),
                    None,
                    kind="instruction",
                    source=f"mipro_proposal:{self.proposal_count}",
                )

        def on_evaluate_start(self, call_id, instance, inputs):
            identity = history.program(inputs["program"])
            dataset = inputs.get("devset")
            self.pending[call_id] = (
                identity,
                instance.devset if dataset is None else dataset,
            )

        def on_evaluate_end(self, call_id, outputs, exception=None):
            entry = self.pending.pop(call_id, None)
            if entry and exception is None and outputs is not None:
                # DSPy Evaluate reports percentages, unlike our metric and GEPA results.
                history.evaluation(*entry, outputs.score / 100)

    return CaptureEvaluations()
