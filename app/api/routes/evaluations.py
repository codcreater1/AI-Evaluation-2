from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import ResultOut, RunOut, RunRequest
from app.db.models import EvaluationRun, System
from app.db.persistence import persist_report
from app.db.session import get_db
from app.engine.adapters import HTTPSystemAdapter
from app.engine.runner import EvaluationRunner
from app.integrations.langfuse_hook import default_sinks

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def _run_out(run: EvaluationRun, with_results: bool, only_failed: bool = False) -> RunOut:
    out = RunOut.model_validate(run, from_attributes=True)
    if with_results:
        rows = [r for r in run.results if not only_failed or r.passed is False]
        out.results = [ResultOut.model_validate(r, from_attributes=True) for r in rows]
    else:
        out.results = None
    return out


@router.post("/run", response_model=RunOut, status_code=201)
def run_evaluation(req: RunRequest, db: Session = Depends(get_db)):
    """Runs synchronously (MVP). Provide `execution` per item, or register the system with an
    endpoint_url and omit it."""
    system = db.get(System, req.config.system)
    adapter = HTTPSystemAdapter(system.endpoint_url) if system and system.endpoint_url else None
    if adapter is None and any(i.execution is None for i in req.items):
        raise HTTPException(422, "items without `execution` need a system registered with endpoint_url")
    try:
        runner = EvaluationRunner.from_config(req.config, adapter=adapter, sinks=default_sinks())
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc

    cases = [i.case for i in req.items]
    executions = {i.case.id: i.execution for i in req.items if i.execution}
    report = runner.run(cases, executions, system=req.config.system, dataset=req.config.dataset)

    run = persist_report(db, report, req.config, req.meta)
    return _run_out(run, with_results=False)


@router.get("", response_model=list[RunOut])
def list_runs(system: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    q = select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
    if system:
        q = q.where(EvaluationRun.system == system)
    return [_run_out(r, False) for r in db.scalars(q)]


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: str, only_failed: bool = False, db: Session = Depends(get_db)):
    run = db.get(EvaluationRun, run_id)
    if run is None:
        raise HTTPException(404, "evaluation run not found")
    return _run_out(run, True, only_failed)
