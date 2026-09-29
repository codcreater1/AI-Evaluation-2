"""Human-in-the-loop review: reviewers judge the same (run, case, evaluator) triple an LLM judge
already scored, so the two can be compared. Everything here is plain DB/SQLAlchemy; the agreement
math itself lives in app.human_eval.agreement so it can be unit tested without a database."""
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EvaluationResultRow, EvaluationRun, HumanEvaluation
from app.errors import Invalid, NotFound
from app.evaluators.llm_judge.prompts import SPECS as LLM_JUDGE_SPECS
from app.human_eval.agreement import AgreementReport, Pair, compute_agreement
from app.integrations.langfuse_client import LangfuseClient

log = logging.getLogger(__name__)

LLM_JUDGE_METRICS = set(LLM_JUDGE_SPECS)


def _get_run(db: Session, run_id: str) -> EvaluationRun:
    run = db.get(EvaluationRun, run_id)
    if run is None:
        raise NotFound(f"evaluation run '{run_id}' not found")
    return run


def submit(
    db: Session, *, run_id: str, case_id: str, evaluator: str,
    score: float | None, passed: bool | None, reason: str = "", reviewer: str = "anonymous",
    client: LangfuseClient | None = None,
) -> HumanEvaluation:
    if score is None and passed is None:
        raise Invalid("give at least one of score or passed")
    if score is not None and not (0.0 <= score <= 1.0):
        raise Invalid("score must be between 0.0 and 1.0")
    _get_run(db, run_id)
    matching = db.scalars(select(EvaluationResultRow).where(
        EvaluationResultRow.run_id == run_id, EvaluationResultRow.case_id == case_id,
        EvaluationResultRow.evaluator == evaluator,
    )).first()
    trace_id = matching.trace_id if matching else None

    existing = db.scalars(select(HumanEvaluation).where(
        HumanEvaluation.run_id == run_id, HumanEvaluation.case_id == case_id,
        HumanEvaluation.evaluator == evaluator, HumanEvaluation.reviewer == reviewer,
    )).first()
    row = existing or HumanEvaluation(id=str(uuid.uuid4()), run_id=run_id, case_id=case_id,
                                      evaluator=evaluator, reviewer=reviewer)
    row.score, row.passed, row.reason = score, passed, reason
    row.trace_id = trace_id
    row.created_at = datetime.now(UTC)
    db.add(row)
    db.commit()

    if client is not None and trace_id:
        try:
            value: float | str = score if score is not None else ("pass" if passed else "fail")
            client.create_score(
                trace_id, f"human_{evaluator}", value, reason,
                {"reviewer": reviewer, "run_id": run_id, "case_id": case_id, "source": "human_review"},
                score_id=client.score_id("human", run_id, case_id, evaluator, reviewer),
            )
        except Exception:  # noqa: BLE001 - human review must never fail because Langfuse is down
            log.exception("failed to push human score to Langfuse")
    return row


def list_for_run(db: Session, run_id: str, evaluator: str | None = None) -> list[HumanEvaluation]:
    q = select(HumanEvaluation).where(HumanEvaluation.run_id == run_id)
    if evaluator:
        q = q.where(HumanEvaluation.evaluator == evaluator)
    return list(db.scalars(q.order_by(HumanEvaluation.created_at.desc())))


def review_queue(
    db: Session, run_id: str | None = None, evaluator: str | None = None,
    reviewer: str | None = None, limit: int = 100,
) -> list[EvaluationResultRow]:
    """LLM-judged results that don't have a human evaluation yet (for this reviewer, if given)."""
    q = select(EvaluationResultRow).where(
        EvaluationResultRow.evaluator.in_([evaluator] if evaluator else LLM_JUDGE_METRICS),
        EvaluationResultRow.score.is_not(None),
    )
    if run_id:
        q = q.where(EvaluationResultRow.run_id == run_id)
    done_q = select(HumanEvaluation.run_id, HumanEvaluation.case_id, HumanEvaluation.evaluator)
    if reviewer:
        done_q = done_q.where(HumanEvaluation.reviewer == reviewer)
    done = set(db.execute(done_q))
    rows = db.scalars(q.order_by(EvaluationResultRow.id)).all()
    pending = [r for r in rows if (r.run_id, r.case_id, r.evaluator) not in done]
    return pending[:limit]


def agreement_report(
    db: Session, run_id: str | None = None, evaluator: str | None = None,
) -> AgreementReport:
    """Joins EvaluationResultRow (the LLM judge's verdict) with HumanEvaluation (the reviewer's
    verdict) on (run_id, case_id, evaluator) and computes agreement over the matched pairs.
    When several reviewers judged the same triple, each (reviewer, triple) is its own pair."""
    judge_q = select(EvaluationResultRow).where(
        EvaluationResultRow.evaluator.in_([evaluator] if evaluator else LLM_JUDGE_METRICS)
    )
    if run_id:
        judge_q = judge_q.where(EvaluationResultRow.run_id == run_id)
    judge_rows = {(r.run_id, r.case_id, r.evaluator): r for r in db.scalars(judge_q)}
    if not judge_rows:
        return compute_agreement([])

    human_q = select(HumanEvaluation).where(
        HumanEvaluation.evaluator.in_([evaluator] if evaluator else LLM_JUDGE_METRICS)
    )
    if run_id:
        human_q = human_q.where(HumanEvaluation.run_id == run_id)
    human_rows = list(db.scalars(human_q))

    pairs = [
        Pair(case_id=h.case_id, judge_score=judge_rows[key].score, judge_passed=judge_rows[key].passed,
            human_score=h.score, human_passed=h.passed)
        for h in human_rows
        if (key := (h.run_id, h.case_id, h.evaluator)) in judge_rows
    ]
    return compute_agreement(pairs)
