from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from app.evaluators import create_evaluator
from app.evaluators.base import BaseEvaluator
from app.experiments.gates import QualityGates


class EvaluatorSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ClassificationConfig(BaseModel):
    """For binary decisions (Internship Coordinator): builds TP/TN/FP/FN + accuracy/precision/recall/F1."""

    expected_key: str = "decision"
    output_key: str = "decision"
    positive_label: str


class RunConfig(BaseModel):
    system: str
    dataset: str = "adhoc"
    evaluators: list[EvaluatorSpec]
    thresholds: dict[str, float] = Field(default_factory=dict)  # evaluator -> minimum mean score
    classification: ClassificationConfig | None = None
    gates: QualityGates | None = None  # regression / minimum quality gates (experiments)
    max_workers: int = 1

    @field_validator("evaluators", mode="before")
    @classmethod
    def _allow_plain_names(cls, v):
        return [{"name": e} if isinstance(e, str) else e for e in v]


def load_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def build_evaluators(config: RunConfig) -> list[BaseEvaluator]:
    return [create_evaluator(e.name, **e.params) for e in config.evaluators]
