import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.config.loader import ClassificationConfig, RunConfig, build_evaluators
from app.engine.adapters import SystemAdapter
from app.engine.aggregation import AggregateMetric, aggregate, check_thresholds
from app.engine.classification import classification_metrics, confusion_label
from app.evaluators.base import BaseEvaluator
from app.evaluators.utils import get_path
from app.integrations.langfuse_hook import ScoreSink
from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult

log = logging.getLogger(__name__)


class CaseReport(BaseModel):
    case_id: str
    execution: ExecutionResult | None = None
    results: list[EvaluationResult] = Field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.results) and self.error is None


class RunReport(BaseModel):
    run_id: str
    system: str
    dataset: str
    cases: list[CaseReport]
    aggregates: dict[str, AggregateMetric]
    classification: dict | None = None
    threshold_failures: list[str] = Field(default_factory=list)
    passed: bool = True
    started_at: datetime
    finished_at: datetime

    @property
    def failed_cases(self) -> list[CaseReport]:
        return [c for c in self.cases if not c.passed]


class EvaluationRunner:
    """case list + evaluators -> results + aggregates. Knows nothing about specific evaluators
    or systems: it only uses the BaseEvaluator / SystemAdapter / ScoreSink interfaces."""

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        adapter: SystemAdapter | None = None,
        sinks: list[ScoreSink] | None = None,
        max_workers: int = 1,
        thresholds: dict[str, float] | None = None,
        classification: ClassificationConfig | None = None,
    ) -> None:
        self.evaluators = evaluators
        self.adapter = adapter
        self.sinks = sinks or []
        self.max_workers = max(1, max_workers)
        self.thresholds = thresholds or {}
        self.classification = classification

    @classmethod
    def from_config(cls, config: RunConfig, adapter=None, sinks=None) -> "EvaluationRunner":
        return cls(
            build_evaluators(config), adapter, sinks, config.max_workers,
            config.thresholds, config.classification,
        )

    def _notify_execution(self, run_id, case, execution) -> None:
        for sink in self.sinks:
            hook = getattr(sink, "on_execution", None)
            if hook is None:
                continue
            try:
                hook(run_id, case, execution)
            except Exception:
                log.exception("score sink on_execution failed")

    # ---- single case
    def _run_case(self, run_id, case, provided: ExecutionResult | None) -> CaseReport:
        report = CaseReport(case_id=case.id)
        try:
            execution = provided
            if execution is None:
                if self.adapter is None:
                    raise RuntimeError("no execution provided and no system adapter configured")
                execution = self.adapter.run(case)
            report.execution = execution
            self._notify_execution(run_id, case, execution)
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
            report.results = [
                EvaluationResult(evaluator=ev.name, evaluator_version=ev.version, score=0.0,
                                 passed=False, label="execution_error", reason=report.error)
                for ev in self.evaluators
            ]
            return report

        for ev in self.evaluators:
            try:
                res = ev.evaluate(case, execution)
            except Exception as exc:  # one broken evaluator must not kill the whole run
                log.exception("evaluator %s failed on %s", ev.name, case.id)
                res = EvaluationResult(
                    evaluator=ev.name, evaluator_version=ev.version, score=0.0, passed=False,
                    label="error", reason=f"{type(exc).__name__}: {exc}",
                )
            report.results.append(res)
            for sink in self.sinks:
                try:
                    sink.on_result(run_id, case, execution, res)
                except Exception:
                    log.exception("score sink failed")
        return report

    # ---- whole run
    def run(
        self,
        cases: list[EvaluationCase],
        executions: dict[str, ExecutionResult] | None = None,
        system: str | None = None,
        dataset: str = "adhoc",
        run_id: str | None = None,
    ) -> RunReport:
        run_id = run_id or str(uuid.uuid4())
        executions = executions or {}
        started = datetime.now(UTC)
        with ThreadPoolExecutor(self.max_workers) as pool:
            reports = list(
                pool.map(lambda c: self._run_case(run_id, c, executions.get(c.id)), cases)
            )
        for sink in self.sinks:
            try:
                sink.flush()
            except Exception:
                log.exception("score sink flush failed")

        aggs = aggregate(r for rep in reports for r in rep.results)
        failures = check_thresholds(aggs, self.thresholds)
        return RunReport(
            run_id=run_id,
            system=system or (cases[0].system if cases else "unknown"),
            dataset=dataset,
            cases=reports,
            aggregates=aggs,
            classification=self._classify(cases, reports),
            threshold_failures=failures,
            passed=not failures,
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    def _classify(self, cases, reports) -> dict | None:
        cfg = self.classification
        if cfg is None:
            return None
        pairs = []
        for case, rep in zip(cases, reports, strict=True):
            exp = get_path(case.expected_output, cfg.expected_key)
            pred = get_path(rep.execution.output, cfg.output_key) if rep.execution else None
            if exp is None:
                continue
            pairs.append((str(exp), str(pred)))
            if rep.execution is not None:
                rep.results.append(
                    EvaluationResult(
                        evaluator="confusion_matrix", score=None, passed=None,
                        label=confusion_label(str(exp), str(pred), cfg.positive_label),
                        reason=f"expected={exp} predicted={pred}",
                    )
                )
        return classification_metrics(pairs, cfg.positive_label)
