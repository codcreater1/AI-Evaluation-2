"""Quality gates: absolute minimums + regression limits -> PASS / FAIL."""
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class QualityGates(BaseModel):
    """YAML example:
    gates:
      minimum:                    # absolute floor for the candidate's mean score
        answer_correctness: 0.90
      max_drop:                   # max allowed decrease vs baseline, in score points (0.02 = 2 pts)
        default: 0.02
        answer_correctness: 0.02
      max_increase_pct:           # lower-is-better operational metrics, relative increase in %
        latency_ms: 25
        cost_usd: 25
      max_newly_failing_cases: 3  # cases that passed in baseline but fail now
    """

    minimum: dict[str, float] = Field(default_factory=dict)
    max_drop: dict[str, float] = Field(default_factory=dict)
    max_increase_pct: dict[str, float] = Field(default_factory=dict)
    max_newly_failing_cases: int | None = None


class GateCheck(BaseModel):
    name: str
    kind: Literal["minimum", "max_drop", "max_increase", "newly_failing"]
    status: Literal["pass", "fail", "skipped"]
    candidate: float | None = None
    baseline: float | None = None
    limit: float | None = None
    message: str


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck]
    baseline_id: str | None = None

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if c.status == "fail"]


def load_gates(path: str | Path) -> QualityGates:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return QualityGates.model_validate(data.get("gates", data))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def evaluate_gates(
    gates: QualityGates,
    candidate: dict[str, float | None],
    baseline: dict[str, float | None] | None,
    candidate_ops: dict[str, float | None],
    baseline_ops: dict[str, float | None] | None,
    newly_failing: int | None,
) -> GateResult:
    checks: list[GateCheck] = []

    # 1) absolute minimums
    for name, minimum in gates.minimum.items():
        c = candidate.get(name)
        if c is None:
            checks.append(GateCheck(name=name, kind="minimum", status="fail", limit=minimum,
                                    message=f"{name}: no score produced (required >= {_pct(minimum)})"))
            continue
        ok = c >= minimum
        checks.append(GateCheck(
            name=name, kind="minimum", status="pass" if ok else "fail", candidate=c, limit=minimum,
            message=f"{name}: {_pct(c)} {'>=' if ok else '<'} minimum {_pct(minimum)}"))

    # 2) regression vs baseline (quality)
    default_drop = gates.max_drop.get("default")
    drop_names = [n for n in gates.max_drop if n != "default"]
    if default_drop is not None:
        drop_names += [n for n in candidate if n not in drop_names]
    for name in drop_names:
        limit = gates.max_drop.get(name, default_drop)
        b, c = (baseline or {}).get(name), candidate.get(name)
        if baseline is None or b is None or c is None:
            reason = "no baseline" if baseline is None else "metric missing in baseline or candidate"
            checks.append(GateCheck(name=name, kind="max_drop", status="skipped", limit=limit,
                                    message=f"{name}: regression check skipped ({reason})"))
            continue
        delta = c - b
        ok = delta >= -limit - 1e-12
        checks.append(GateCheck(
            name=name, kind="max_drop", status="pass" if ok else "fail", candidate=c, baseline=b,
            limit=limit,
            message=f"{name}: {_pct(b)} -> {_pct(c)} ({delta * 100:+.1f} pts), "
                    f"max allowed drop {limit * 100:.1f} pts"))

    # 3) operational regressions (relative increase, lower is better)
    for name, max_pct in gates.max_increase_pct.items():
        b, c = (baseline_ops or {}).get(name), candidate_ops.get(name)
        if baseline_ops is None or not b or c is None:
            checks.append(GateCheck(name=name, kind="max_increase", status="skipped", limit=max_pct,
                                    message=f"{name}: increase check skipped (no baseline value)"))
            continue
        inc = (c - b) / b * 100
        ok = inc <= max_pct + 1e-9
        checks.append(GateCheck(
            name=name, kind="max_increase", status="pass" if ok else "fail", candidate=c,
            baseline=b, limit=max_pct,
            message=f"{name}: {b:.4g} -> {c:.4g} ({inc:+.1f}%), max allowed +{max_pct:g}%"))

    # 4) newly failing cases
    if gates.max_newly_failing_cases is not None:
        if newly_failing is None:
            checks.append(GateCheck(name="newly_failing_cases", kind="newly_failing",
                                    status="skipped", limit=gates.max_newly_failing_cases,
                                    message="newly failing cases: skipped (no baseline)"))
        else:
            ok = newly_failing <= gates.max_newly_failing_cases
            checks.append(GateCheck(
                name="newly_failing_cases", kind="newly_failing", status="pass" if ok else "fail",
                candidate=float(newly_failing), limit=float(gates.max_newly_failing_cases),
                message=f"newly failing cases: {newly_failing} "
                        f"(max allowed {gates.max_newly_failing_cases})"))

    return GateResult(passed=not any(c.status == "fail" for c in checks), checks=checks)


def render_gate_report(comparison: dict) -> str:
    """Human/CI readable summary, e.g. for a PR comment."""
    gate = comparison.get("gate")
    cand, base = comparison["candidate"], comparison.get("baseline")
    lines = []
    if gate is None:
        lines.append("AI Evaluation: no quality gates configured")
    elif gate["passed"]:
        lines.append("PASS: AI Evaluation passed")
    else:
        lines.append("FAIL: AI Evaluation failed")
    lines.append(f"candidate: {cand['name']} ({cand['dataset_ref']})")
    if base:
        lines.append(f"baseline : {base['name']} ({base['dataset_ref']})")
    if comparison.get("dataset_mismatch"):
        lines.append("warning: experiments use different dataset versions; compared on common cases")
    for c in (gate or {}).get("checks", []):
        icon = {"pass": "  ok ", "fail": "FAIL ", "skipped": "skip "}[c["status"]]
        lines.append(f"  [{icon}] {c['message']}")
    nf = comparison["cases"]["newly_failing"]
    if nf:
        lines.append(f"newly failing cases ({len(nf)}):")
        for case in nf[:10]:
            ev = ", ".join(f["evaluator"] for f in case["failed_evaluators"])
            lines.append(f"  - {case['case_id']} [{ev}] {case.get('trace_url') or ''}".rstrip())
    if gate is not None and not gate["passed"]:
        lines.append("Deployment should be blocked.")
    return "\n".join(lines)
