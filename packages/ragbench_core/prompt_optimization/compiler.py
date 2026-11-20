"""Pinned DSPy boundary. All task prompts use the inference renderer verbatim."""

import importlib.metadata
import json
import os
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import wraps
from pathlib import Path

import requests

from ragbench_core.components.judges import RUBRIC_VERSION, judge_answer
from ragbench_core.models.schema import Chunk, GeneratedAnswer, QueryExample
from ragbench_core.optimization.profiles import ModelProfile
from ragbench_core.prompt_optimization.artifacts import CompiledPrompt, render_prompt
from ragbench_core.prompt_optimization.contracts import DSPY_VERSION
from ragbench_core.prompt_optimization.service import provenance, task_profile


class _CompilationAbort(BaseException):
    """Escape DSPy's catch-and-continue loops; unwrap at our public boundary."""

    def __init__(self, error):
        self.error = error


def _fatal(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- all external failures must escape optimizer retry loops.
            # Optimizers can swallow ordinary errors and keep trying forever.
            raise _CompilationAbort(exc) from None

    return wrapped


def _compiler_boundary(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except _CompilationAbort as abort:
            raise abort.error from None

    return wrapped


class _FatalAccount:
    def __init__(self, account):
        self.account = account

    @_fatal
    def check(self):
        self.account.check()

    @contextmanager
    def call(self, *args, **kwargs):
        try:
            with self.account.call(*args, **kwargs) as usage:
                yield usage
        except Exception as exc:  # noqa: BLE001 -- preserve the original failure at the compiler boundary.
            raise _CompilationAbort(exc) from None


def model_call(profile, prompt, *, generator=None, generation_config=None):
    if generator is not None:
        result = generator.generate(prompt, generation_config)
        return result["answer"], result.get("metadata", {}).get("token_usage", {})
    if profile.provider == "ollama":
        response = requests.post(
            profile.base_url.rstrip("/") + "/api/generate",
            json={
                "model": profile.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": profile.max_output_tokens},
            },
            timeout=profile.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body["response"], {
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
        }
    key = os.environ.get(profile.api_key_env or "RAGBENCH_OPTIMIZER_API_KEY", "")
    response = requests.post(
        profile.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + key},
        json={
            "model": profile.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": profile.max_output_tokens,
        },
        timeout=profile.timeout,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def make_bridges(dspy, account, config, recipe):
    from ragbench_core.registry import register_builtin_components

    profile = task_profile(config)
    generator = register_builtin_components().generators[config["llm"]["provider"]]

    class Bridge(dspy.BaseLM):
        def __init__(self, profile, role, task=False):
            super().__init__(
                model=profile.model,
                model_type="chat",
                temperature=config["llm"].get("temperature", 0),
                max_tokens=profile.max_output_tokens,
                cache=False,
            )
            self.profile, self.role, self.task = profile, role, task

        def __call__(self, prompt=None, messages=None, **kwargs):
            if isinstance(prompt, list):
                messages, prompt = prompt, None
            prompt = (
                prompt
                if prompt is not None
                else "\n\n".join(m["role"] + ":\n" + m["content"] for m in messages)
            )
            if kwargs.get("n", 1) != 1:
                raise ValueError("Compilation supports one completion per model call")
            with account.call(self.role, prompt, self.profile) as usage:
                text, reported = model_call(
                    self.profile,
                    prompt,
                    generator=generator if self.task else None,
                    generation_config={
                        **config["llm"],
                        "temperature": self.kwargs.get("temperature", 0),
                        "max_output_tokens": self.profile.max_output_tokens,
                    },
                )
                usage.update(reported)
            return [text]

        def copy(self, **kwargs):
            result = copy(self)
            result.kwargs = {**self.kwargs, **kwargs}
            result.history = []
            return result

        def __deepcopy__(self, memo):
            return self.copy()

    class AnswerAdapter(dspy.Adapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            if set(signature.output_fields) == {"answer"} and set(
                signature.input_fields
            ) == {"question", "context"}:
                prompt = render_prompt(
                    signature.instructions, demos, inputs["question"], inputs["context"]
                )
                return [{"answer": lm(prompt=prompt, **lm_kwargs)[0]}]
            # DSPy's proposers have their own signatures; use an explicit adapter.
            return dspy.ChatAdapter()(lm, lm_kwargs, signature, demos, inputs)

    return (
        Bridge(profile, "prompt_task", True),
        Bridge(recipe.proposal_model or profile, "prompt_proposal"),
        AnswerAdapter(),
    )


def make_optimizer(dspy, recipe, metric, proposal):
    params = deepcopy(recipe.parameters)
    if recipe.optimizer == "bootstrap_few_shot":
        return dspy.BootstrapFewShot(metric=metric, max_errors=1, **params), {}
    if recipe.optimizer == "miprov2":
        compile_keys = {
            "num_trials",
            "minibatch",
            "minibatch_size",
            "minibatch_full_eval_steps",
        }
        compile_params = {k: params.pop(k) for k in list(params) if k in compile_keys}
        return dspy.MIPROv2(
            metric=metric,
            prompt_model=proposal,
            num_threads=1,
            max_errors=1,
            seed=recipe.seed,
            **params,
        ), compile_params
    return dspy.GEPA(
        metric=metric, reflection_lm=proposal, num_threads=1, seed=recipe.seed, **params
    ), {}


@_compiler_boundary
def compile_prompt(recipe, prepared, splits, config, judge_config, account):
    account = _FatalAccount(account)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault(
        "DSPY_CACHEDIR",
        str(Path(config["runtime"]["artifact_dir"]) / ".dspy_cache"),
    )
    import dspy

    if importlib.metadata.version("dspy") != DSPY_VERSION:
        raise RuntimeError(f"Prompt compilation requires dspy=={DSPY_VERSION}")
    task, proposal, adapter = make_bridges(dspy, account, config, recipe)

    class AnswerSignature(dspy.Signature):
        """Give a direct, evidence-grounded answer to the question."""

        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class AnswerProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.answer = dspy.Predict(AnswerSignature)
            self.answer.set_lm(task)

        def forward(self, question, context):
            return self.answer(question=question, context=context)

    by_inputs = {(e["question"], e["context"]): e for e in prepared}
    judge_profile = ModelProfile.model_validate(judge_config)

    @_fatal
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        account.check()
        raw = by_inputs[(gold.question, gold.context)]
        example = QueryExample(
            **{k: raw[k] for k in QueryExample.__dataclass_fields__ if k in raw}
        )
        answer = GeneratedAnswer(
            pred.answer,
            "",
            [Chunk(**c) for c in raw["context_chunks"]],
            {"context_text": raw["context"]},
        )
        # The judge includes context, reference, claims and document representations.
        bound = (
            json.dumps(
                {
                    "question": gold.question,
                    "reference": gold.answer,
                    "answer": pred.answer,
                    "context": raw["context"],
                    "documents": raw["context_chunks"],
                },
                ensure_ascii=False,
            )
            + " " * 4096
        )
        with account.call("prompt_judge", bound, judge_profile) as usage:
            result = judge_answer(example, answer, judge_profile.model_dump())
            usage.update(result["details"].get("usage", {}))
        if (
            result["details"]["status"] != "ok"
            or result["scores"]["answer_correctness"] is None
        ):
            raise RuntimeError("Prompt optimization judge failed")
        score = result["scores"]["answer_correctness"]
        if recipe.optimizer == "gepa":
            return dspy.Prediction(
                score=score,
                feedback=json.dumps(
                    {
                        "correctness": result["details"]["verdicts"][
                            "answer_correctness"
                        ],
                        "claims": result["details"]["verdicts"]["claims"],
                        "citations": result["details"]["verdicts"]["citations"],
                    },
                    ensure_ascii=False,
                ),
            )
        return score

    def examples(partition):
        ids = set(splits[partition])
        return [
            dspy.Example(
                question=e["question"],
                context=e["context"],
                answer=e["expected_answer"],
            ).with_inputs("question", "context")
            for e in prepared
            if e["question_id"] in ids
        ]

    train, validation = examples("training"), examples("validation")
    if not train or not validation:
        raise ValueError("Prompt training and validation partitions must be nonempty")
    with dspy.context(lm=task, adapter=adapter, max_errors=1):
        optimizer, compile_params = make_optimizer(dspy, recipe, metric, proposal)
    if recipe.optimizer != "bootstrap_few_shot":
        compile_params["valset"] = validation
    if compile_params.get("minibatch"):
        compile_params["minibatch_size"] = min(
            compile_params.get("minibatch_size", 25), len(validation)
        )
    with dspy.context(lm=task, adapter=adapter, max_errors=1):
        program = optimizer.compile(AnswerProgram(), trainset=train, **compile_params)
        predictor = program.answer
        demos = [
            {k: getattr(demo, k) for k in ("question", "context", "answer")}
            for demo in predictor.demos
        ]
        payload = CompiledPrompt(
            instructions=predictor.signature.instructions,
            demonstrations=demos,
            provenance={
                **provenance(prepared, splits, recipe, config),
                "dspy_version": DSPY_VERSION,
                "judge_model": judge_profile.model_dump(),
                "judge_rubric_version": RUBRIC_VERSION,
            },
        ).model_dump()
        scores = []
        for example in validation:
            prediction = program(question=example.question, context=example.context)
            result = metric(example, prediction)
            scores.append(result.score if recipe.optimizer == "gepa" else result)
    return payload, {
        "validation_score": sum(scores) / len(scores),
        "validation_count": len(scores),
        "demonstration_count": len(demos),
    }
