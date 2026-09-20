from typing import Any

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """One golden test case. `input`/`expected_output` are free-form so any AI system fits."""

    id: str
    system: str
    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
