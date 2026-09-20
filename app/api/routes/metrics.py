from fastapi import APIRouter, Depends
from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from app.api.schemas import MetricOut
from app.db.models import EvaluationResultRow as R
from app.db.models import EvaluationRun
from app.db.session import get_db

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=list[MetricOut])
def get_metrics(system: str | None = None, run_id: str | None = None, db: Session = Depends(get_db)):
    """Mean score / pass rate per (system, evaluator), across all runs or a single run."""
    passed_int = func.sum(case((R.passed.is_(True), 1), else_=0).cast(Integer))
    judged = func.count(R.passed)
    q = (
        select(EvaluationRun.system, R.evaluator, func.count(R.score), func.avg(R.score),
               passed_int, judged)
        .join(EvaluationRun, EvaluationRun.id == R.run_id)
        .group_by(EvaluationRun.system, R.evaluator)
        .order_by(EvaluationRun.system, R.evaluator)
    )
    if system:
        q = q.where(EvaluationRun.system == system)
    if run_id:
        q = q.where(R.run_id == run_id)
    return [
        MetricOut(system=s, evaluator=e, n=n, mean_score=avg,
                  pass_rate=(p / j) if j else None)
        for s, e, n, avg, p, j in db.execute(q).all()
    ]
