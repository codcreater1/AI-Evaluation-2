from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_langfuse
from app.api.schemas import ResultOut
from app.api.schemas_exp import CompareRequest, ExperimentOut, ExperimentRunRequest
from app.db.models import EvaluationRun, Experiment
from app.db.session import get_db
from app.experiments import service
from app.experiments.comparison import compare_experiments
from app.experiments.gates import QualityGates, render_gate_report
from app.integrations.langfuse_client import LangfuseClient

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _out(e: Experiment) -> ExperimentOut:
    return ExperimentOut.model_validate(e, from_attributes=True)


@router.post("", response_model=ExperimentOut, status_code=201)
def run_experiment(body: ExperimentRunRequest, db: Session = Depends(get_db),
                   client: LangfuseClient | None = Depends(get_langfuse)):
    """Runs the dataset version through the evaluators, records the experiment, pushes traces /
    scores / dataset run to Langfuse and evaluates the quality gates against the baseline."""
    exp = service.run_experiment(
        db, body.config, executions=body.executions, meta=body.meta, name=body.name,
        set_baseline=body.set_baseline, baseline_id=body.baseline_id,
        client=client if body.langfuse else None)
    return _out(exp)


@router.get("", response_model=list[ExperimentOut])
def list_experiments(system: str | None = None, limit: int = 100, db: Session = Depends(get_db)):
    """History (newest first): the dashboard can plot `overall` / aggregates over time."""
    q = select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
    if system:
        q = q.where(Experiment.system == system)
    return [_out(e) for e in db.scalars(q)]


@router.post("/compare")
def compare(body: CompareRequest, db: Session = Depends(get_db),
            client: LangfuseClient | None = Depends(get_langfuse)):
    """Baseline vs candidate: improvements, regressions, unchanged, newly failing / fixed cases,
    and the PASS/FAIL gate verdict. `report` is a CI-friendly text rendering."""
    cand = service.get_experiment(db, body.candidate_id)
    baseline = (service.get_experiment(db, body.baseline_id) if body.baseline_id
                else service.get_baseline(db, cand.system, exclude_id=cand.id))
    gates = body.gates or (QualityGates.model_validate(cand.config["gates"])
                           if cand.config.get("gates") else None)
    cmp = compare_experiments(db, cand, baseline, gates, client)
    cmp["report"] = render_gate_report(cmp)
    return cmp


@router.get("/{exp_id}", response_model=ExperimentOut)
def get_experiment(exp_id: str, db: Session = Depends(get_db)):
    return _out(service.get_experiment(db, exp_id))


@router.get("/{exp_id}/results", response_model=list[ResultOut])
def experiment_results(exp_id: str, only_failed: bool = False, db: Session = Depends(get_db),
                       client: LangfuseClient | None = Depends(get_langfuse)):
    """Per-case results (failed ones with `only_failed=true`) incl. trace ids for Langfuse."""
    exp = service.get_experiment(db, exp_id)
    run = db.get(EvaluationRun, exp.run_id)
    rows = [r for r in run.results if not only_failed or r.passed is False]
    return [ResultOut.model_validate(r, from_attributes=True) for r in rows]


@router.post("/{exp_id}/baseline", response_model=ExperimentOut)
def set_baseline(exp_id: str, db: Session = Depends(get_db)):
    return _out(service.mark_baseline(db, service.get_experiment(db, exp_id)))


@router.get("/{exp_id}/trace-url/{trace_id}")
def trace_url(exp_id: str, trace_id: str, client: LangfuseClient | None = Depends(get_langfuse)):
    return {"url": client.trace_url(trace_id) if client else None}
