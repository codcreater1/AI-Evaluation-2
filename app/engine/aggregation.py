from collections.abc import Iterable

from pydantic import BaseModel

from app.schemas import EvaluationResult

ERROR_LABELS = {"error", "execution_error"}


class AggregateMetric(BaseModel):
    evaluator: str
    version: str = "1"
    n: int = 0  # results produced
    n_scored: int = 0  # results with a numeric score (not_applicable excluded)
    n_errors: int = 0
    mean_score: float | None = None
    pass_rate: float | None = None
    min_score: float | None = None
    max_score: float | None = None


def aggregate(results: Iterable[EvaluationResult]) -> dict[str, AggregateMetric]:
    """Errors count as score 0 / failed (conservative), not_applicable results are skipped."""
    by_eval: dict[str, list[EvaluationResult]] = {}
    for r in results:
        by_eval.setdefault(r.evaluator, []).append(r)
    out: dict[str, AggregateMetric] = {}
    for name, rs in by_eval.items():
        scores = [r.score for r in rs if r.score is not None]
        judged = [r for r in rs if r.passed is not None]
        out[name] = AggregateMetric(
            evaluator=name,
            version=rs[0].evaluator_version,
            n=len(rs),
            n_scored=len(scores),
            n_errors=sum(1 for r in rs if r.label in ERROR_LABELS),
            mean_score=sum(scores) / len(scores) if scores else None,
            pass_rate=(sum(1 for r in judged if r.passed) / len(judged)) if judged else None,
            min_score=min(scores) if scores else None,
            max_score=max(scores) if scores else None,
        )
    return out


def check_thresholds(aggs: dict[str, AggregateMetric], thresholds: dict[str, float]) -> list[str]:
    failures = []
    for name, minimum in thresholds.items():
        m = aggs.get(name)
        if m is None or m.mean_score is None:
            failures.append(f"{name}: no scores produced (required >= {minimum})")
        elif m.mean_score < minimum:
            failures.append(f"{name}: {m.mean_score:.3f} < minimum {minimum}")
    return failures


def compare_aggregates(
    baseline: dict[str, AggregateMetric],
    candidate: dict[str, AggregateMetric],
    max_drop: dict[str, float] | float = 0.02,
) -> dict:
    """Baseline vs candidate: improvements / regressions / unchanged (used later by CI)."""
    rows, regressions = {}, []
    for name in sorted(set(baseline) & set(candidate)):
        b, c = baseline[name].mean_score, candidate[name].mean_score
        if b is None or c is None:
            continue
        delta = c - b
        limit = max_drop.get(name, 0.02) if isinstance(max_drop, dict) else max_drop
        status = "regression" if delta < -limit else "improvement" if delta > 1e-9 else "unchanged"
        if delta < -limit:
            regressions.append(name)
        rows[name] = {"baseline": b, "candidate": c, "delta": delta, "status": status}
    return {"metrics": rows, "regressions": regressions, "passed": not regressions}
