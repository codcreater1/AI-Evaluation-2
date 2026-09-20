from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Standard result every evaluator returns.

    score  : numeric, normalized to 0..1 (None = not applicable for this case)
    passed : boolean verdict (None = not applicable)
    label  : categorical result (e.g. "TP", "FN", "error", "not_applicable")
    """

    evaluator: str
    evaluator_version: str = "1"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool | None = None
    label: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
