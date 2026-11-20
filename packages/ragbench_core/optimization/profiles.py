"""Shared model contracts for orchestration and prompt compilation."""

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelProfile(Contract):
    provider: Literal["ollama", "chat_completions"] = "ollama"
    model: str = Field(default="qwen2.5:32b", min_length=1)
    base_url: str = "http://100.110.250.66:11434"
    api_key_env: str | None = None
    timeout: float = Field(default=120, gt=0, le=240)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def endpoint_policy(self):
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Model endpoint must be an HTTP(S) URL without credentials or query parameters"
            )
        if self.api_key_env and not self.api_key_env.replace("_", "a").isalnum():
            raise ValueError(
                "API key must be referenced by an environment variable name"
            )
        return self
