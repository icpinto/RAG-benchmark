"""Versioned, bounded data at the adaptive planner boundary."""

from typing import Literal

from pydantic import Field

from ragbench_core.optimization.profiles import Contract


class Lesson(Contract):
    text: str = Field(min_length=1, max_length=600)
    status: Literal["tentative", "supported", "contradicted"]
    trial_ids: list[str] = Field(min_length=1, max_length=10)


class Memory(Contract):
    lessons: list[Lesson] = Field(default_factory=list, max_length=8)


class Observation(Contract):
    reference: str
    phase: Literal["retrieval", "answer"]
    trial_id: str | None = None
    candidate_id: str
    status: str
    configuration: dict
    changes: list[dict]
    metric: str
    score: float | None = None
    coverage: dict = Field(default_factory=dict)
    ranking: dict | None = None
    comparisons: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    examples: list[dict] = Field(default_factory=list, max_length=3)
    operational: dict = Field(default_factory=dict)


class DecisionBase(Contract):
    reflection: str = Field(min_length=1, max_length=2000)
    memory: Memory = Field(default_factory=Memory)


class RetrievalDecision(DecisionBase):
    action: Literal["run", "stop"]
    hypothesis: str = Field(default="", max_length=1000)
    pipeline: dict | None = None
    reason: str = Field(default="", max_length=1000)


class AnswerDecision(DecisionBase):
    action: Literal["run", "stop"]
    hypothesis: str = Field(default="", max_length=1000)
    candidate_id: str | None = None
    reason: str = Field(default="", max_length=1000)
