from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config.loader import RunConfig
from app.experiments.gates import QualityGates
from app.schemas import EvaluationCase, ExecutionResult


class DatasetCreate(BaseModel):
    name: str
    system: str
    description: str = ""
    cases: list[EvaluationCase]
    note: str = ""


class VersionCreate(BaseModel):
    base_version: int | None = None  # default: latest
    cases: list[EvaluationCase] | None = None  # full replacement, OR:
    add_cases: list[EvaluationCase] = Field(default_factory=list)
    update_cases: list[EvaluationCase] = Field(default_factory=list)
    remove_case_ids: list[str] = Field(default_factory=list)
    note: str = ""


class FromTraceRequest(BaseModel):
    trace_id: str
    case_id: str
    expected_output: dict[str, Any]
    input: dict[str, Any] | None = None  # override the trace input if needed
    metadata: dict[str, Any] = Field(default_factory=dict)
    base_version: int | None = None
    note: str = "added from production trace"


class VersionOut(BaseModel):
    dataset: str
    version: int
    ref: str
    n_cases: int
    content_hash: str
    parent_version: int | None
    note: str
    created_at: datetime
    langfuse_synced_at: datetime | None


class VersionDetail(VersionOut):
    cases: list[EvaluationCase]


class DatasetOut(BaseModel):
    name: str
    system: str
    description: str
    latest_version: int | None
    versions: list[VersionOut]


class ExperimentRunRequest(BaseModel):
    config: RunConfig  # config.dataset = "ata-rag-golden-v3" (or name without -vN = latest)
    executions: dict[str, ExecutionResult] = Field(default_factory=dict)  # case_id -> output
    meta: dict[str, Any] = Field(default_factory=dict)  # app_version, model, prompt_version, retriever
    name: str | None = None
    set_baseline: bool = False
    baseline_id: str | None = None  # default: current baseline of the system
    langfuse: bool = True


class ExperimentOut(BaseModel):
    id: str
    number: int
    name: str
    system: str
    dataset_ref: str
    dataset_version: int
    dataset_hash: str
    run_id: str
    is_baseline: bool
    meta: dict[str, Any]
    config: dict[str, Any]
    evaluator_versions: dict[str, str]
    aggregates: dict[str, Any]
    operational: dict[str, Any]
    overall: float | None
    gate_result: dict[str, Any] | None
    langfuse: dict[str, Any]
    created_at: datetime


class CompareRequest(BaseModel):
    candidate_id: str
    baseline_id: str | None = None  # default: current baseline of the candidate's system
    gates: QualityGates | None = None  # default: gates stored in the candidate's config
