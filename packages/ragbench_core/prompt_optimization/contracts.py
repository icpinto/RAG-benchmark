"""Versioned prompt search recipes. Presets resolve before work is queued."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, model_validator

from ragbench_core.optimization.profiles import Contract, ModelProfile

DSPY_VERSION = "3.0.4"
PRESETS = {
    "bootstrap_small_v1": {
        "optimizer": "bootstrap_few_shot",
        "parameters": {
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 2,
            "max_rounds": 1,
            "metric_threshold": 0.8,
        },
    },
    "mipro_zero_shot_v1": {
        "optimizer": "miprov2",
        "parameters": {
            "auto": None,
            "num_candidates": 4,
            "num_trials": 8,
            "max_bootstrapped_demos": 0,
            "max_labeled_demos": 0,
            "minibatch": False,
        },
    },
    "mipro_few_shot_v1": {
        "optimizer": "miprov2",
        "parameters": {
            "auto": None,
            "num_candidates": 4,
            "num_trials": 8,
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 2,
            "minibatch": False,
        },
    },
    "gepa_light_v1": {
        "optimizer": "gepa",
        "parameters": {
            "auto": None,
            "max_metric_calls": 100,
            "reflection_minibatch_size": 3,
        },
    },
}


class CompilationLimits(Contract):
    max_tokens: int = Field(default=2_000_000, ge=1000)
    max_seconds: int = Field(default=900, ge=10, le=86400)
    max_calls: int = Field(default=500, ge=1, le=10000)
    max_cost: float | None = Field(default=None, gt=0)


class PromptOptimizationConfig(Contract):
    preset: str | None = None
    optimizer: Literal["bootstrap_few_shot", "miprov2", "gepa"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    proposal_model: ModelProfile | None = None
    limits: CompilationLimits = Field(default_factory=CompilationLimits)

    @model_validator(mode="after")
    def resolve(self):
        if self.preset and self.preset not in PRESETS:
            raise ValueError("Unknown prompt optimizer preset")
        base = deepcopy(PRESETS.get(self.preset, {}))
        if self.optimizer and base and self.optimizer != base["optimizer"]:
            raise ValueError("Optimizer does not match preset")
        self.optimizer = self.optimizer or base.get("optimizer")
        if not self.optimizer:
            raise ValueError("Select a prompt optimizer or preset")
        self.parameters = {**base.get("parameters", {}), **self.parameters}
        common = {
            "max_bootstrapped_demos",
            "max_labeled_demos",
            "max_rounds",
            "metric_threshold",
        }
        allowed = {
            "bootstrap_few_shot": common,
            "miprov2": (common - {"max_rounds"})
            | {
                "auto",
                "num_candidates",
                "num_trials",
                "minibatch",
                "minibatch_size",
                "minibatch_full_eval_steps",
            },
            "gepa": {
                "auto",
                "max_metric_calls",
                "max_full_evals",
                "reflection_minibatch_size",
                "skip_perfect_score",
                "use_merge",
            },
        }[self.optimizer]
        if set(self.parameters) - allowed:
            raise ValueError(
                "Unsupported optimizer parameters: "
                + ", ".join(sorted(set(self.parameters) - allowed))
            )
        p = self.parameters
        if p.get("auto") not in {None, "light", "medium", "heavy"}:
            raise ValueError("auto must be light, medium, heavy, or null")
        if self.optimizer == "miprov2":
            if p.get("auto") is not None and any(
                p.get(k) is not None for k in ("num_candidates", "num_trials")
            ):
                raise ValueError(
                    "MIPROv2 auto cannot be combined with explicit candidates/trials; set auto: null"
                )
            if p.get("auto") is None and any(
                p.get(k) is None for k in ("num_candidates", "num_trials")
            ):
                raise ValueError(
                    "MIPROv2 requires num_candidates and num_trials when auto is null"
                )
        if (
            self.optimizer == "gepa"
            and sum(
                p.get(k) is not None
                for k in ("auto", "max_metric_calls", "max_full_evals")
            )
            != 1
        ):
            raise ValueError(
                "GEPA requires exactly one of auto, max_metric_calls, max_full_evals"
            )
        for key, value in p.items():
            if key == "auto" or value is None:
                continue
            if key in {"minibatch", "skip_perfect_score", "use_merge"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be boolean")
            elif key == "metric_threshold":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 < value <= 1
                ):
                    raise ValueError(
                        "metric_threshold must be greater than zero and at most one"
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not (0 if "demos" in key else 1)
                <= value
                <= (16 if "demos" in key else 10000)
            ):
                raise ValueError(f"Invalid bounded integer parameter: {key}")
        return self


class PromptOptimizationSettings(Contract):
    configs: list[PromptOptimizationConfig] = Field(default_factory=list, max_length=12)

    def validate_policy(self, allow_hosted, monetary_cap=None):
        for config in self.configs:
            profile = config.proposal_model
            if profile and profile.provider != "ollama" and not allow_hosted:
                raise ValueError(
                    "Hosted prompt optimization requires source-data opt-in"
                )
            if (
                profile
                and (monetary_cap is not None or config.limits.max_cost is not None)
                and (
                    profile.input_price_per_million is None
                    or profile.output_price_per_million is None
                )
            ):
                raise ValueError(
                    "Prompt optimization monetary caps require model prices"
                )


class CompilationRequest(Contract):
    config_path: str = "/app/configs/default.yaml"
    source_paths: list[str] | None = None
    question_indices: list[int] | None = None
    case_id: str | None = None
    prompt_optimization: PromptOptimizationSettings | None = None
    judge: ModelProfile | None = None
    allow_hosted: bool = False
