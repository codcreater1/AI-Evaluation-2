"""Simple server-rendered dashboard: system overview, experiment history, failed cases with
Langfuse trace links, and data read back live from Langfuse. Open http://localhost:8000/"""
import html
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_langfuse
from app.db.models import EvaluationResultRow, EvaluationRun, Experiment
from app.db.session import get_db
from app.errors import NotFound
from app.experiments import demo
from app.experiments.comparison import compare_experiments
from app.experiments.gates import QualityGates
from app.experiments.service import get_baseline, get_experiment
from app.integrations.langfuse_client import LangfuseClient, LangfuseError, LangfuseRateLimited

router = APIRouter(include_in_schema=False)

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px 32px;color:#1a1a1a;background:#fff}
h1{margin:0 0 4px;font-size:26px} h2{margin:28px 0 8px;font-size:18px}
.sub{color:#666;font-size:13px;margin-bottom:12px} a{color:#0b5cd5;text-decoration:none}
table{border-collapse:collapse;width:100%;font-size:13px} th{background:#f0f0f0;text-align:left}
th,td{border-bottom:1px solid #ddd;padding:6px 8px;vertical-align:top}
.chip{display:inline-block;padding:1px 6px;border-radius:3px;margin:1px 2px;white-space:nowrap}
.ok{background:#d9f2dd;color:#12631f}.bad{background:#fbdcdc;color:#a11}.neutral{background:#eee}
.pass{color:#12631f;font-weight:600}.fail{color:#a11;font-weight:600}
.box{border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin:8px 0;background:#fafafa}
button{padding:8px 14px;font-size:14px;cursor:pointer;border:1px solid #0b5cd5;background:#0b5cd5;
color:#fff;border-radius:4px} .muted{color:#777} code{background:#f0f0f0;padding:1px 4px}
"""


def e(x: Any) -> str:
    return "-" if x is None else html.escape(str(x))


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def num(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def when(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "-"


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>")


def gate_badge(exp: Experiment) -> str:
    g = exp.gate_result
    if not g:
        return "<span class='muted'>-</span>"
    return "<span class='pass'>PASS</span>" if g["passed"] else "<span class='fail'>FAIL</span>"


def thresholds(exp: Experiment) -> dict[str, float]:
    cfg = exp.config or {}
    return {**((cfg.get("gates") or {}).get("minimum") or {}), **(cfg.get("thresholds") or {})}


def lf_warn(x: Experiment) -> str:
    lf = x.langfuse or {}
    if lf.get("errors") or lf.get("sync_error"):
        tip = lf.get("last_error") or lf.get("sync_error") or ""
        return f" <span class='fail' title='{e(tip)}'>&#9888; Langfuse errors</span>"
    return ""


def trace_link(client: LangfuseClient | None, trace_id: str | None) -> str:
    if not trace_id:
        return "<span class='muted'>-</span>"
    if client is None:
        return f"<code>{e(trace_id[:12])}</code>"
    return f"<a href='{e(client.trace_url(trace_id))}' target='_blank'>Open in Langfuse</a>"


# ------------------------------------------------------------------ Langfuse (live) block
def cached(client: LangfuseClient, key: tuple, ttl: float, fn: Callable[[], Any]) -> Any:
    """Langfuse Hobby allows ~30 general API calls/min, so page reads are cached (errors too)."""
    hit = client.read_cache.get(key)
    if hit and hit[0] > time.monotonic():
        if hit[1] == "err":
            raise hit[2]
        return hit[2]
    try:
        value = fn()
    except LangfuseError as exc:
        client.read_cache[key] = (time.monotonic() + min(ttl, 20), "err", exc)
        raise
    client.read_cache[key] = (time.monotonic() + ttl, "ok", value)
    return value


def lf_error(exc: LangfuseError) -> str:
    if isinstance(exc, LangfuseRateLimited):
        return (f"Langfuse rate limit reached (free plan allows about 30 requests/min). "
                f"Wait ~{exc.retry_after:.0f} s and refresh - your data is safe.")
    return e(exc)


def langfuse_block(client: LangfuseClient | None, trace_ids: list[str] | None = None) -> str:
    if client is None:
        return ("<div class='box'>Langfuse is <b>not configured</b>. Set <code>LANGFUSE_PUBLIC_KEY</code>, "
                "<code>LANGFUSE_SECRET_KEY</code> and <code>LANGFUSE_HOST</code> before starting the API.</div>")
    parts = []
    try:
        project = cached(client, ("project",), 300, client.project_name)
        parts.append(f"<div class='box'>Connected to Langfuse project <b>{e(project)}</b> "
                     f"({e(client.host)}) - <a href='{e(client.host)}' target='_blank'>open Langfuse</a></div>")
    except LangfuseError as exc:
        parts.append(f"<div class='box'><span class='fail'>Cannot reach Langfuse:</span> {lf_error(exc)}</div>")
        return "".join(parts)

    try:
        scores = cached(client, ("scores", tuple(trace_ids or ())), 45,
                        lambda: client.list_scores(limit=100 if trace_ids else 15, trace_ids=trace_ids))
        rows = "".join(
            f"<tr><td>{e(s['name'])}</td><td>{e(s['value'] if not isinstance(s['value'], float) else round(s['value'], 3))}</td>"
            f"<td>{trace_link(client, s['trace_id'])}</td><td>{e(s['timestamp'])}</td></tr>"
            for s in scores[:15])
        head = (f"{len(scores)} scores found in Langfuse for this experiment's traces"
                if trace_ids else "Latest scores stored in Langfuse")
        parts.append(f"<h3>{e(head)}</h3><table><tr><th>Score</th><th>Value</th><th>Trace</th>"
                     f"<th>Time</th></tr>{rows or '<tr><td colspan=4 class=muted>none yet (new data can take a few minutes to appear)</td></tr>'}</table>")
    except LangfuseError as exc:
        parts.append(f"<p class='muted'>Could not read scores from Langfuse: {lf_error(exc)}</p>")

    if not trace_ids:
        try:
            datasets = cached(client, ("datasets",), 60, client.list_datasets)
            names = ", ".join(e(d.get("name")) for d in datasets[:20]) or "none"
            parts.append(f"<h3>Datasets in Langfuse</h3><div class='box'>{names}</div>")
        except LangfuseError as exc:
            parts.append(f"<p class='muted'>Could not read datasets from Langfuse: {lf_error(exc)}</p>")
    return "".join(parts)


# ------------------------------------------------------------------ pages
@router.get("/")
def root():
    return RedirectResponse("/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: Session = Depends(get_db), client: LangfuseClient | None = Depends(get_langfuse)):
    exps = list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc()).limit(100)))
    n_cases = {r.id: r.summary.get("n_cases") for r in db.scalars(select(EvaluationRun))}

    latest: dict[str, Experiment] = {}
    for x in exps:
        latest.setdefault(x.system, x)
    overview = ""
    for system, x in sorted(latest.items()):
        th = thresholds(x)
        chips = ""
        for name, minimum in th.items():
            m = (x.aggregates.get(name) or {}).get("mean_score")
            good = m is not None and m >= minimum
            chips += (f"<span class='chip {'ok' if good else 'bad'}'>{e(name)}: {num(m)} "
                      f"(min {minimum:.2f})</span>")
        if not th:
            chips = "".join(f"<span class='chip neutral'>{e(k)}: {num(v.get('mean_score'))}</span>"
                            for k, v in list(x.aggregates.items())[:5])
        overview += (f"<tr><td>{e(system)}</td><td><a href='/dashboard/experiments/{e(x.id)}'>{e(x.name)}</a></td>"
                     f"<td>{e(x.dataset_ref)}</td><td>{e(n_cases.get(x.run_id))}</td><td>{chips}</td>"
                     f"<td>{gate_badge(x)}</td><td>{when(x.created_at)}</td></tr>")

    history = ""
    for x in exps:
        rates = ", ".join(f"{e(k)}: {pct(v.get('pass_rate'))}" for k, v in x.aggregates.items()
                          if k != "confusion_matrix" and v.get("pass_rate") is not None)
        history += (f"<tr><td><a href='/dashboard/experiments/{e(x.id)}'>{e(x.name)}</a>"
                    f"{' &#9733;' if x.is_baseline else ''}{lf_warn(x)}</td><td>{e(x.system)}</td><td>{e(x.dataset_ref)}</td>"
                    f"<td>{e(x.meta.get('model'))}</td><td>{e(x.meta.get('prompt_version'))}</td>"
                    f"<td>{e(n_cases.get(x.run_id))}</td><td>{rates}</td><td>{gate_badge(x)}</td>"
                    f"<td>{when(x.created_at)}</td></tr>")

    body = f"""
<h1>AI Evaluation Platform</h1>
<div class='sub'>System overview, experiment history, failed cases with Langfuse trace links.
&#9733; = baseline experiment.</div>
<form method='post' action='/dashboard/run-demo' onsubmit="this.querySelector('button').disabled=true;
this.querySelector('button').innerText='Running... (about 10-60 s)'">
<button type='submit'>Run test scenario (baseline &rarr; ok change &rarr; broken change)</button>
<span class='muted'> creates a demo dataset + 3 experiments and sends them to Langfuse if configured</span></form>
<h2>System overview (latest experiment per system)</h2>
<table><tr><th>System</th><th>Latest run</th><th>Dataset</th><th>Cases</th><th>Scores vs threshold</th>
<th>Gate</th><th>When</th></tr>{overview or "<tr><td colspan=7 class=muted>No experiments yet. Click the button above.</td></tr>"}</table>
<h2>Experiment history</h2>
<table><tr><th>Run</th><th>System</th><th>Dataset</th><th>Model</th><th>Prompt</th><th>Cases</th>
<th>Pass rates</th><th>Gate</th><th>Created</th></tr>{history or "<tr><td colspan=9 class=muted>-</td></tr>"}</table>
<h2>Langfuse (read live from Langfuse)</h2>{langfuse_block(client)}"""
    return page("AI Evaluation Platform", body)


@router.post("/dashboard/run-demo")
def run_demo_route(db: Session = Depends(get_db), client: LangfuseClient | None = Depends(get_langfuse)):
    demo.run_demo(db, client)
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/dashboard/experiments/{exp_id}", response_class=HTMLResponse)
def experiment_page(exp_id: str, db: Session = Depends(get_db),
                    client: LangfuseClient | None = Depends(get_langfuse)):
    try:
        x = get_experiment(db, exp_id)
    except NotFound:
        return page("Not found", "<p>Experiment not found. <a href='/dashboard'>Back</a></p>")
    run = db.get(EvaluationRun, x.run_id)
    th = thresholds(x)

    agg_rows = ""
    for name, a in x.aggregates.items():
        minimum = th.get(name)
        m = a.get("mean_score")
        chip = "neutral" if minimum is None else ("ok" if (m is not None and m >= minimum) else "bad")
        agg_rows += (f"<tr><td>{e(name)}</td><td>{e(x.evaluator_versions.get(name))}</td>"
                     f"<td><span class='chip {chip}'>{num(m)}</span></td><td>{pct(a.get('pass_rate'))}</td>"
                     f"<td>{a.get('n_scored')}/{a.get('n')}</td><td>{a.get('n_errors')}</td>"
                     f"<td>{'-' if minimum is None else f'{minimum:.2f}'}</td></tr>")
    ops = x.operational or {}
    op_txt = (f"latency {num(ops.get('latency_ms'), 0)} ms &middot; cost ${num(ops.get('cost_usd'), 4)} "
              f"&middot; tokens {num(ops.get('tokens'), 0)}")

    baseline = get_baseline(db, x.system, exclude_id=x.id)
    gates = QualityGates.model_validate((x.config or {}).get("gates") or {}) if (x.config or {}).get("gates") else None
    cmp = compare_experiments(db, x, baseline, gates, client)

    gate_html = ""
    if cmp["gate"]:
        lines = "".join(
            f"<li class='{'pass' if c['status'] == 'pass' else 'fail' if c['status'] == 'fail' else 'muted'}'>"
            f"[{e(c['status'])}] {e(c['message'])}</li>" for c in cmp["gate"]["checks"])
        verdict = ("<span class='pass'>PASS</span>" if cmp["gate"]["passed"]
                   else "<span class='fail'>FAIL - deployment should be blocked</span>")
        gate_html = f"<h2>Quality gate: {verdict}</h2><ul>{lines}</ul>"
    if baseline:
        rows = "".join(
            f"<tr><td>{e(k)}</td><td>{num(v['baseline'])}</td><td>{num(v['candidate'])}</td>"
            f"<td>{'-' if v['delta'] is None else f'{v['delta'] * 100:+.1f} pts'}</td>"
            f"<td><span class='chip {'ok' if v['status'] == 'improved' else 'bad' if v['status'] == 'regressed' else 'neutral'}'>"
            f"{e(v['status'])}</span></td></tr>" for k, v in cmp["metrics"].items())
        gate_html += (f"<h2>Compared with baseline <a href='/dashboard/experiments/{e(baseline.id)}'>{e(baseline.name)}</a></h2>"
                      f"<table><tr><th>Metric</th><th>Baseline</th><th>This run</th><th>Delta</th><th>Status</th></tr>{rows}</table>")
        nf = cmp["cases"]["newly_failing"]
        gate_html += (f"<p>Newly failing cases: <b>{len(nf)}</b> &middot; fixed cases: "
                      f"<b>{len(cmp['cases']['fixed'])}</b></p>")
    else:
        gate_html += "<p class='muted'>No baseline for this system yet (mark one with POST /experiments/{id}/baseline).</p>"

    rows = list(db.scalars(select(EvaluationResultRow).where(
        EvaluationResultRow.run_id == x.run_id, EvaluationResultRow.passed.is_(False))))
    failed = "".join(
        f"<tr><td>{e(r.case_id)}</td><td>{e(r.evaluator)}</td><td>{num(r.score, 2)}</td><td>{e(r.reason)}</td>"
        f"<td>{trace_link(client, r.trace_id)}</td></tr>" for r in rows[:200])

    classification = ""
    cls = (run.summary or {}).get("classification") if run else None
    if cls:
        classification = (f"<h2>Classification</h2><div class='box'>TP {cls['tp']} &middot; TN {cls['tn']} &middot; "
                          f"FP {cls['fp']} &middot; FN {cls['fn']} &middot; accuracy {num(cls['accuracy'])} &middot; "
                          f"precision {num(cls['precision'])} &middot; recall {num(cls['recall'])} &middot; F1 {num(cls['f1'])}</div>")

    all_traces = [r for (r,) in db.execute(select(EvaluationResultRow.trace_id).where(
        EvaluationResultRow.run_id == x.run_id, EvaluationResultRow.trace_id.is_not(None)).distinct())]
    lf = x.langfuse or {}
    if not lf.get("enabled"):
        lf_status = "not used for this run"
    else:
        lf_status = (f"run <b>{e(lf.get('run_name'))}</b> on dataset <b>{e(lf.get('dataset'))}</b>"
                     if lf.get("dataset") else "enabled, but dataset sync failed")
        lf_status += f" &middot; {e(lf.get('events_sent', 0))} events sent"
        if lf.get("errors") or lf.get("sync_error"):
            lf_status += (f"<br><span class='fail'>&#9888; {e(lf.get('errors', 0))} Langfuse request(s) failed:"
                          f"</span> {e(lf.get('last_error') or lf.get('sync_error'))}")
    body = f"""
<p><a href='/dashboard'>&larr; Dashboard</a></p>
<h1>{e(x.name)} {'&#9733; baseline' if x.is_baseline else ''}</h1>
<div class='sub'>{e(x.system)} &middot; dataset <b>{e(x.dataset_ref)}</b> (hash {e(x.dataset_hash[:12])}) &middot;
model {e(x.meta.get('model'))} &middot; prompt {e(x.meta.get('prompt_version'))} &middot; {when(x.created_at)}</div>
<div class='box'>{op_txt}<br>Langfuse: {lf_status}</div>
<h2>Scores</h2><table><tr><th>Evaluator</th><th>Version</th><th>Mean score</th><th>Pass rate</th><th>Scored</th>
<th>Errors</th><th>Threshold</th></tr>{agg_rows}</table>
{classification}{gate_html}
<h2>Failed cases ({len(rows)})</h2>
<table><tr><th>Case</th><th>Evaluator</th><th>Score</th><th>Reason</th><th>Trace</th></tr>
{failed or "<tr><td colspan=5 class=muted>No failed cases</td></tr>"}</table>
<h2>Langfuse check for this experiment</h2>{langfuse_block(client, all_traces[:20])}"""
    return page(x.name, body)
