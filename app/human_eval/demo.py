"""Built-in demo for the dashboard's 'Run judge-vs-human demo' button: seeds a run with N
LLM-judge results and matching human reviews so the agreement report has something to show
without needing a paid LLM API key or hand-reviewing 100+ cases first.

Judge and human scores are both noisy readings of the same hidden "true quality" per case, which
is what real (LLM judge, human) pairs look like: mostly they agree, occasionally they don't,
and the report should show that rather than a suspiciously perfect 100%."""
import random
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import EvaluationResultRow, EvaluationRun, System
from app.human_eval import service

SYSTEM = "demo-judge-agreement"
EVALUATOR = "answer_correctness"
N_CASES = 120  # comfortably over the >=100 the agreement validation asks for
REVIEWER = "demo-reviewer"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def run_demo(db: Session, seed: int = 42) -> str:
    rnd = random.Random(seed)
    if db.get(System, SYSTEM) is None:
        db.add(System(name=SYSTEM, description="Synthetic system used only to demo judge-vs-human agreement"))

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    run = EvaluationRun(
        id=run_id, system=SYSTEM, dataset="demo-judge-agreement-v1", status="completed",
        passed=True, config={"evaluators": [{"name": EVALUATOR}]}, meta={"note": "synthetic demo run"},
        summary={"n_cases": N_CASES}, created_at=now, finished_at=now,
    )
    reviews: list[tuple[str, float, bool]] = []
    for i in range(1, N_CASES + 1):
        case_id = f"demo-{i:04d}"
        true_quality = rnd.random()
        judge_score = _clip01(rnd.gauss(true_quality, 0.08))
        # the human reviewer is the reference reading: a bit less noisy than the judge
        human_score = round(_clip01(rnd.gauss(true_quality, 0.05)), 3)
        judge_score = round(judge_score, 3)
        run.results.append(EvaluationResultRow(
            case_id=case_id, evaluator=EVALUATOR, evaluator_version="1", score=judge_score,
            passed=judge_score >= 0.7, label=None, reason="synthetic judge score",
            meta={"judge_model": "demo-judge", "prompt_version": "v1"}, trace_id=None,
        ))
        reviews.append((case_id, human_score, human_score >= 0.7))
    db.add(run)
    db.commit()

    for case_id, human_score, human_passed in reviews:
        service.submit(db, run_id=run_id, case_id=case_id, evaluator=EVALUATOR,
                       score=human_score, passed=human_passed,
                       reason="synthetic human review", reviewer=REVIEWER)
    return run_id
