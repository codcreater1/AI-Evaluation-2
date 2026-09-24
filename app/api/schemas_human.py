from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HumanEvaluationIn(BaseModel):
    run_id: str
    case_id: str
    evaluator: str  # the metric being judged, e.g. "answer_correctness" (must match an evaluator name)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool | None = None
    reason: str = ""
    reviewer: str = "anonymous"


class HumanEvaluationOut(BaseModel):
    id: str
    run_id: str
    case_id: str
    evaluator: str
    score: float | None
    passed: bool | None
    reason: str
    reviewer: str
    trace_id: str | None
    created_at: datetime


class QueueItemOut(BaseModel):
    run_id: str
    case_id: str
    evaluator: str
    judge_score: float | None
    judge_passed: bool | None
    judge_reason: str
    trace_id: str | None


class AgreementOut(BaseModel):
    n: int
    n_scored: int
    n_pass_fail: int
    pass_agreement_rate: float | None
    cohens_kappa: float | None
    mean_abs_score_diff: float | None
    confusion: dict[str, int]
    disagreements: list[dict[str, Any]]
