from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config.loader import RunConfig
from app.schemas import EvaluationCase, ExecutionResult


class RunItem(BaseModel):
    case: EvaluationCase
    execution: ExecutionResult | None = None  # omitted -> engine calls the registered system endpoint


class RunRequest(BaseModel):
    config: RunConfig
    items: list[RunItem]
    meta: dict[str, Any] = Field(default_factory=dict)  # app_version, model, prompt_version, ...


class SystemIn(BaseModel):
    name: str
    description: str = ""
    endpoint_url: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SystemOut(SystemIn):
    created_at: datetime


class ResultOut(BaseModel):
    case_id: str
    evaluator: str
    evaluator_version: str
    score: float | None
    passed: bool | None
    label: str | None
    reason: str
    meta: dict[str, Any]
    trace_id: str | None
    latency_ms: float | None
    cost_usd: float | None


class RunOut(BaseModel):
    id: str
    system: str
    dataset: str
    status: str
    passed: bool
    config: dict[str, Any]
    meta: dict[str, Any]
    summary: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None
    results: list[ResultOut] | None = None


class MetricOut(BaseModel):
    system: str
    evaluator: str
    n: int
    mean_score: float | None
    pass_rate: float | None
