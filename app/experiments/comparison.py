from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EvaluationResultRow, Experiment
from app.experiments.gates import GateResult, QualityGates, evaluate_gates
from app.integrations.langfuse_client import LangfuseClient

EPSILON = 0.001  # score deltas smaller than this count as "unchanged"
OPERATIONAL = ("latency_ms", "cost_usd", "tokens")  # lower is better


def case_outcomes(db: Session, run_id: str) -> dict[str, dict[str, Any]]:
    """case_id -> {passed, failed_evaluators:[{evaluator, reason, score}], trace_id}"""
    rows = db.scalars(select(EvaluationResultRow).where(EvaluationResultRow.run_id == run_id))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        o = out.setdefault(r.case_id, {"passed": True, "failed_evaluators": [], "trace_id": None})
        o["trace_id"] = o["trace_id"] or r.trace_id
        if r.passed is False:
            o["passed"] = False
            o["failed_evaluators"].append(
                {"evaluator": r.evaluator, "reason": r.reason, "score": r.score})
    return out


def _mean_scores(exp: Experiment) -> dict[str, float | None]:
    return {k: v.get("mean_score") for k, v in exp.aggregates.items()
            if k != "confusion_matrix"}


def _status(delta: float, lower_is_better: bool = False) -> str:
    if abs(delta) < EPSILON:
        return "unchanged"
    better = delta < 0 if lower_is_better else delta > 0
    return "improved" if better else "regressed"


def compare_experiments(
    db: Session, candidate: Experiment, baseline: Experiment | None,
    gates: QualityGates | None = None, client: LangfuseClient | None = None,
) -> dict[str, Any]:
    def brief(e: Experiment) -> dict:
        return {"id": e.id, "name": e.name, "dataset_ref": e.dataset_ref, "meta": e.meta,
                "is_baseline": e.is_baseline, "created_at": e.created_at.isoformat()}

    cand_scores = _mean_scores(candidate)
    result: dict[str, Any] = {
        "candidate": brief(candidate), "baseline": brief(baseline) if baseline else None,
        "dataset_mismatch": bool(baseline and baseline.dataset_ref != candidate.dataset_ref),
        "metrics": {}, "operational": {},
        "summary": {"improvements": [], "regressions": [], "unchanged": []},
        "cases": {"compared": 0, "newly_failing": [], "fixed": [], "still_failing": [],
                  "only_in_candidate": 0, "only_in_baseline": 0},
        "gate": None,
    }
    base_scores = _mean_scores(baseline) if baseline else None
    newly_count: int | None = None

    if baseline:
        for name in sorted(set(cand_scores) | set(base_scores or {})):
            b, c = (base_scores or {}).get(name), cand_scores.get(name)
            if b is None or c is None:
                result["metrics"][name] = {"baseline": b, "candidate": c, "delta": None,
                                           "status": "not_comparable"}
                continue
            st = _status(c - b)
            result["metrics"][name] = {"baseline": b, "candidate": c, "delta": c - b, "status": st}
            key = {"improved": "improvements", "regressed": "regressions", "unchanged": "unchanged"}[st]
            result["summary"][key].append(name)
        for name in OPERATIONAL:
            b, c = baseline.operational.get(name), candidate.operational.get(name)
            if b is None or c is None:
                continue
            result["operational"][name] = {
                "baseline": b, "candidate": c, "delta": c - b,
                "delta_pct": ((c - b) / b * 100) if b else None,
                "status": _status((c - b) / b if b else 0.0, lower_is_better=True),
            }

        cand_out, base_out = case_outcomes(db, candidate.run_id), case_outcomes(db, baseline.run_id)
        common = sorted(set(cand_out) & set(base_out))
        cases = result["cases"]
        cases["compared"] = len(common)
        cases["only_in_candidate"] = len(set(cand_out) - set(base_out))
        cases["only_in_baseline"] = len(set(base_out) - set(cand_out))
        for cid in common:
            b, c = base_out[cid], cand_out[cid]
            if b["passed"] and not c["passed"]:
                cases["newly_failing"].append({
                    "case_id": cid, "failed_evaluators": c["failed_evaluators"],
                    "trace_id": c["trace_id"],
                    "trace_url": client.trace_url(c["trace_id"]) if client and c["trace_id"] else None,
                })
            elif not b["passed"] and c["passed"]:
                cases["fixed"].append({"case_id": cid, "trace_id": c["trace_id"]})
            elif not b["passed"] and not c["passed"]:
                cases["still_failing"].append(cid)
        newly_count = len(cases["newly_failing"])
    else:
        for name, c in cand_scores.items():
            result["metrics"][name] = {"baseline": None, "candidate": c, "delta": None,
                                       "status": "no_baseline"}

    if gates is not None:
        gr: GateResult = evaluate_gates(
            gates, cand_scores, base_scores, candidate.operational,
            baseline.operational if baseline else None, newly_count,
        )
        gr.baseline_id = baseline.id if baseline else None
        result["gate"] = gr.model_dump()
    return result
