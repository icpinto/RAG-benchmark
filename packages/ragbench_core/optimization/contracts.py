"""Public optimization contracts. Secrets are environment references, never values."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ragbench_core.optimization.profiles import Contract, ModelProfile
from ragbench_core.prompt_optimization.contracts import PromptOptimizationSettings


class Limits(Contract):
    target_questions: int = Field(default=100, ge=30, le=1000)
    retrieval_trials: int = Field(default=20, ge=1, le=100)
    answer_trials: int = Field(default=5, ge=1, le=20)
    active_seconds: int = Field(default=7200, ge=60, le=86400)
    max_tokens: int = Field(default=20_000_000, ge=1000)
    max_cost: float | None = Field(default=None, gt=0)
    max_query_p95_seconds: float | None = Field(default=None, gt=0)
    max_query_cost: float | None = Field(default=None, gt=0)


class OptimizationRequest(Contract):
    config_path: str = "/app/configs/default.yaml"
    source_paths: list[str] | None = None
    urls: list[str] = Field(default_factory=list, max_length=100)
    audience: str = Field(default="", max_length=2000)
    task: str = Field(default="", max_length=4000)
    example_questions: list[str] = Field(default_factory=list, max_length=20)
    planner: ModelProfile = Field(default_factory=ModelProfile)
    synthesizer: ModelProfile = Field(default_factory=ModelProfile)
    verifier: ModelProfile = Field(default_factory=ModelProfile)
    judge: ModelProfile = Field(default_factory=ModelProfile)
    limits: Limits = Field(default_factory=Limits)
    allow_hosted: bool = False
    review_before_freeze: bool = False
    evaluation_dataset_mode: Literal["reuse_if_compatible", "generate_new"] = (
        "reuse_if_compatible"
    )
    allowed_components: dict[str, list[str]] = Field(default_factory=dict)
    prompt_optimization: PromptOptimizationSettings = Field(default_factory=PromptOptimizationSettings)

    @model_validator(mode="after")
    def policy(self):
        for role in (self.planner, self.synthesizer, self.verifier, self.judge):
            if role.provider != "ollama" and not self.allow_hosted:
                raise ValueError("Hosted model use requires source-data opt-in")
            if self.limits.max_cost is not None and (
                role.input_price_per_million is None
                or role.output_price_per_million is None
            ):
                raise ValueError(
                    "Monetary caps require prices for every model role (including local models)"
                )
        self.prompt_optimization.validate_policy(self.allow_hosted, self.limits.max_cost)
        return self


class Evidence(Contract):
    doc_id: str
    quote: str = Field(min_length=1)
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)


class SyntheticQuestion(Contract):
    question_id: str = ""
    question: str = Field(min_length=5)
    expected_answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    category: Literal["fact", "paraphrase", "table", "exception", "multi_document"] = (
        "fact"
    )
    group_id: str = ""
    split: Literal["unassigned", "tuning", "holdout"] = "unassigned"
    automated_validated: bool = False
    human_reviewed: bool = False
    validation_reason: str = ""


class DatasetEdit(Contract):
    questions: list[SyntheticQuestion]


class ArtifactRequest(Contract):
    session_id: str


class TrialRequest(ArtifactRequest):
    config: dict
    phase: Literal["retrieval", "answer", "confirm"]
    split: Literal["tuning", "holdout"] = "tuning"


class ToolResult(Contract):
    session_id: str
    artifact_id: str
    summary: dict = Field(default_factory=dict)
