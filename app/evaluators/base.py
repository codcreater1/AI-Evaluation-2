from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult


class BaseEvaluator(ABC):
    """Every evaluator subclasses this. Params come from the YAML config."""

    name: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    kind: ClassVar[str] = "deterministic"  # or "llm_judge"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def evaluate(self, case: EvaluationCase, execution: ExecutionResult) -> EvaluationResult: ...

    def make_result(self, **kwargs: Any) -> EvaluationResult:
        kwargs.setdefault("evaluator", self.name)
        kwargs.setdefault("evaluator_version", self.version)
        return EvaluationResult(**kwargs)

    def not_applicable(self, reason: str) -> EvaluationResult:
        return self.make_result(score=None, passed=None, label="not_applicable", reason=reason)


# ---- registry: adding an evaluator = adding a file with @register_evaluator, no runner change
EvaluatorFactory = Callable[..., BaseEvaluator]
_REGISTRY: dict[str, EvaluatorFactory] = {}


def register_factory(name: str, factory: EvaluatorFactory) -> None:
    _REGISTRY[name] = factory


def register_evaluator(name: str) -> Callable[[type[BaseEvaluator]], type[BaseEvaluator]]:
    def decorator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
        cls.name = name
        register_factory(name, cls)
        return cls

    return decorator


def create_evaluator(name: str, **params: Any) -> BaseEvaluator:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown evaluator '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def list_evaluators() -> list[str]:
    return sorted(_REGISTRY)
