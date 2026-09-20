import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.loader import RunConfig, build_evaluators
from app.datasets.service import (
    get_dataset,
    get_version,
    load_cases,
    make_ref,
    parse_ref,
    sync_to_langfuse,
)
from app.db.models import Experiment, System
from app.db.persistence import persist_report
from app.engine.adapters import HTTPSystemAdapter
from app.engine.runner import EvaluationRunner, RunReport
from app.errors import Invalid, NotFound
from app.experiments.comparison import compare_experiments
from app.experiments.gates import QualityGates
from app.integrations.langfuse_client import LangfuseClient
from app.integrations.langfuse_hook import LangfuseSink
from app.schemas import ExecutionResult

log = logging.getLogger(__name__)


def get_experiment(db: Session, exp_id: str) -> Experiment:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise NotFound(f"experiment '{exp_id}' not found")
    return exp


def get_baseline(db: Session, system: str, exclude_id: str | None = None) -> Experiment | None:
    q = select(Experiment).where(Experiment.system == system, Experiment.is_baseline.is_(True))
    if exclude_id:
        q = q.where(Experiment.id != exclude_id)
    return db.scalars(q.order_by(Experiment.created_at.desc())).first()


def mark_baseline(db: Session, exp: Experiment) -> Experiment:
    for other in db.scalars(select(Experiment).where(
            Experiment.system == exp.system, Experiment.is_baseline.is_(True))):
        other.is_baseline = False
    exp.is_baseline = True
    db.commit()
    return exp


def _operational(report: RunReport) -> dict[str, float | None]:
    execs = [c.execution for c in report.cases if c.execution]

    def mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    costs = [e.cost_usd for e in execs if e.cost_usd is not None]
    return {
        "latency_ms": mean([e.latency_ms for e in execs if e.latency_ms is not None]),
        "cost_usd": mean(costs),
        "tokens": mean([e.total_tokens for e in execs if e.total_tokens is not None]),
        "cost_usd_total": sum(costs) if costs else None,
    }


def run_experiment(
    db: Session, config: RunConfig, *, executions: dict[str, ExecutionResult] | None = None,
    meta: dict[str, Any] | None = None, name: str | None = None, set_baseline: bool = False,
    baseline_id: str | None = None, client: LangfuseClient | None = None,
) -> Experiment:
    """One experiment = one version of an AI system evaluated on one dataset version."""
    meta, executions = meta or {}, executions or {}
    ds_name, ds_ver = parse_ref(config.dataset)
    dataset = get_dataset(db, ds_name)
    if dataset.system != config.system:
        raise Invalid(f"dataset '{ds_name}' belongs to system '{dataset.system}', not '{config.system}'")
    version = get_version(db, ds_name, ds_ver)
    ref = make_ref(ds_name, version.version)
    cases = load_cases(version)

    system = db.get(System, config.system)
    adapter = HTTPSystemAdapter(system.endpoint_url) if system and system.endpoint_url else None
    missing = [c.id for c in cases if c.id not in executions]
    if missing and adapter is None:
        raise Invalid(f"no execution for {len(missing)} case(s) (e.g. {missing[:3]}) and system "
                      f"'{config.system}' has no endpoint_url to call")
    try:
        evaluators = build_evaluators(config)
    except KeyError as exc:
        raise Invalid(str(exc)) from exc

    number = (db.scalar(select(func.max(Experiment.number)).where(
        Experiment.system == config.system)) or 0) + 1
    exp_name = name or f"{config.system} #{number}"

    # ---- Langfuse: dataset sync + sink (best effort, never breaks the evaluation)
    lf: dict[str, Any] = {"enabled": client is not None}
    sinks = []
    if client:
        dataset_ref = None
        try:
            sync_to_langfuse(db, client, version)
            dataset_ref = ref
            lf["dataset"] = ref
        except Exception as exc:
            log.exception("langfuse dataset sync failed")
            lf["sync_error"] = str(exc)[:300]
        sinks.append(LangfuseSink(
            client, run_name=exp_name, dataset_ref=dataset_ref,
            metadata={"system": config.system, "experiment": exp_name, "dataset": ref, **meta},
            description=f"{config.system} on {ref}",
        ))
        lf["run_name"] = exp_name

    gates = config.gates or QualityGates()
    runner = EvaluationRunner(
        evaluators, adapter, sinks, config.max_workers,
        {**gates.minimum, **config.thresholds}, config.classification,
    )
    report = runner.run(cases, executions, system=config.system, dataset=ref)
    run = persist_report(db, report, config, meta)

    agg = {k: v.model_dump() for k, v in report.aggregates.items()}
    overall_keys = [k for k in gates.minimum if agg.get(k, {}).get("mean_score") is not None]
    exp = Experiment(
        id=str(uuid.uuid4()), number=number, name=exp_name, system=config.system,
        dataset_name=ds_name, dataset_version=version.version, dataset_ref=ref,
        dataset_hash=version.content_hash, run_id=run.id, meta=meta, config=config.model_dump(),
        evaluator_versions={e.name: e.version for e in evaluators},
        aggregates=agg, operational=_operational(report),
        overall=(sum(agg[k]["mean_score"] for k in overall_keys) / len(overall_keys))
        if overall_keys else None,
        langfuse=lf, created_at=report.started_at,
    )
    db.add(exp)
    db.commit()

    if config.gates is not None or config.thresholds:
        baseline = (get_experiment(db, baseline_id) if baseline_id
                    else get_baseline(db, config.system, exclude_id=exp.id))
        cmp = compare_experiments(db, exp, baseline, gates, client)
        exp.gate_result = cmp["gate"]
        db.commit()
    if set_baseline:
        mark_baseline(db, exp)
    return exp
