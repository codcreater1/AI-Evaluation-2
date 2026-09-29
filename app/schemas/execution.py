from typing import Any

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """What the AI system actually produced for one case (plus operational data)."""

    output: dict[str, Any] = Field(default_factory=dict)
    # Retrieved documents (RAG) or any intermediate artifacts: [{"id":..,"url":..,"text":..}]
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    trace_id: str | None = None  # Langfuse trace id, used to attach scores
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)
