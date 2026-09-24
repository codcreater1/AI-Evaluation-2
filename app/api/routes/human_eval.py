from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_langfuse
from app.api.schemas_human import AgreementOut, HumanEvaluationIn, HumanEvaluationOut, QueueItemOut
from app.db.session import get_db
from app.human_eval import service
from app.integrations.langfuse_client import LangfuseClient

router = APIRouter(prefix="/human-evaluations", tags=["human-evaluations"])


@router.post("", response_model=HumanEvaluationOut, status_code=201)
def submit(body: HumanEvaluationIn, db: Session = Depends(get_db),
          client: LangfuseClient | None = Depends(get_langfuse)):
    """Records (or updates, if this reviewer already judged this case+evaluator) a human
    verdict, and best-effort pushes it to Langfuse as a `human_<evaluator>` score on the trace."""
    return service.submit(
        db, run_id=body.run_id, case_id=body.case_id, evaluator=body.evaluator,
        score=body.score, passed=body.passed, reason=body.reason, reviewer=body.reviewer,
        client=client,
    )


@router.get("", response_model=list[HumanEvaluationOut])
def list_evaluations(run_id: str, evaluator: str | None = None, db: Session = Depends(get_db)):
    return service.list_for_run(db, run_id, evaluator)


@router.get("/queue", response_model=list[QueueItemOut])
def queue(run_id: str | None = None, evaluator: str | None = None, reviewer: str | None = None,
         limit: int = 100, db: Session = Depends(get_db)):
    """LLM-judged results still waiting for a human verdict (optionally scoped to one reviewer)."""
    rows = service.review_queue(db, run_id, evaluator, reviewer, limit)
    return [QueueItemOut(run_id=r.run_id, case_id=r.case_id, evaluator=r.evaluator,
                         judge_score=r.score, judge_passed=r.passed, judge_reason=r.reason,
                         trace_id=r.trace_id) for r in rows]


@router.get("/agreement", response_model=AgreementOut)
def agreement(run_id: str | None = None, evaluator: str | None = None, db: Session = Depends(get_db)):
    """LLM judge vs. human agreement: pass/fail agreement rate, Cohen's kappa, mean |score diff|,
    confusion counts, and the biggest disagreements. Needs >=1 matched (judge, human) pair;
    validate on a golden set of >=100 reviewed cases for a statistically meaningful number."""
    return service.agreement_report(db, run_id, evaluator).as_dict()
