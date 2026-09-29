from sqlalchemy.orm import Session

from app.config.loader import RunConfig
from app.db.models import EvaluationResultRow, EvaluationRun, System
from app.engine.runner import RunReport


def persist_report(db: Session, report: RunReport, config: RunConfig, meta: dict) -> EvaluationRun:
    if db.get(System, report.system) is None:
        db.add(System(name=report.system))
    run = EvaluationRun(
        id=report.run_id, system=report.system, dataset=report.dataset, status="completed",
        passed=report.passed, config=config.model_dump(), meta=meta,
        summary={
            "aggregates": {k: v.model_dump() for k, v in report.aggregates.items()},
            "classification": report.classification,
            "threshold_failures": report.threshold_failures,
            "n_cases": len(report.cases),
            "n_failed_cases": len(report.failed_cases),
        },
        created_at=report.started_at, finished_at=report.finished_at,
    )
    for rep in report.cases:
        ex = rep.execution
        for r in rep.results:
            run.results.append(EvaluationResultRow(
                case_id=rep.case_id, evaluator=r.evaluator, evaluator_version=r.evaluator_version,
                score=r.score, passed=r.passed, label=r.label, reason=r.reason, meta=r.metadata,
                trace_id=ex.trace_id if ex else None,
                latency_ms=ex.latency_ms if ex else None, cost_usd=ex.cost_usd if ex else None,
            ))
    db.add(run)
    db.commit()
    return run
