"""Live demo: baseline -> regression -> automatic detection.
Start the API first:   DATABASE_URL=sqlite:///demo.db uvicorn app.api.main:app
Then run:              python demo.py
"""
import sys
import time

import httpx

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
LF = "--langfuse" in sys.argv  # also push datasets / traces / scores to Langfuse
API = ARGS[0] if ARGS else "http://localhost:8000"
c = httpx.Client(base_url=API, timeout=60)
RUN = str(int(time.time()))[-6:]  # unique dataset name per demo run
DS = f"ata-rag-demo-{RUN}"


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def ok(r):
    if r.status_code >= 400:
        print("ERROR", r.status_code, r.text)
        raise SystemExit(1)
    return r.json()


# 1. golden dataset (v1) -------------------------------------------------
step("1) Create a versioned golden dataset (immutable v1)")
cases = [{"id": f"rag-{i:03d}", "system": "ata-rag",
          "input": {"question": f"Sample question {i}?"},
          "expected_output": {"answer": f"answer {i}", "relevant_doc_ids": [f"doc-{i}"]}}
         for i in range(1, 7)]
v1 = ok(c.post("/datasets", json={"name": DS, "system": "ata-rag", "cases": cases}))
print(f"created {v1['ref']}  cases={v1['n_cases']}  hash={v1['content_hash'][:12]}...")


# 2. helper: what the AI system "produced" ------------------------------
def executions(wrong=(), latency=1200):
    out = {}
    for i in range(1, 7):
        good = i not in wrong
        out[f"rag-{i:03d}"] = {
            "output": {"answer": f"answer {i}" if good else "some wrong answer",
                       "citations": [f"doc-{i}"]},
            "retrieved": [{"id": f"doc-{i}" if good else "doc-x", "text": "..."}],
            "latency_ms": latency, "input_tokens": 400, "output_tokens": 120, "cost_usd": 0.003}
    return out


config = {
    "system": "ata-rag", "dataset": f"{DS}-v1",
    "evaluators": [{"name": "exact_match", "params": {"output_key": "answer"}},
                   {"name": "retrieval_recall_at_k", "params": {"k": 3}},
                   "citation_exists",
                   {"name": "latency_threshold", "params": {"max_ms": 3000}}],
    "gates": {"minimum": {"exact_match": 0.90, "retrieval_recall_at_k": 0.90},
              "max_drop": {"default": 0.02},
              "max_increase_pct": {"latency_ms": 25},
              "max_newly_failing_cases": 1},
}

# 3. baseline ------------------------------------------------------------
step("2) Baseline experiment (prompt v1)")
base = ok(c.post("/experiments", json={
    "config": config, "executions": executions(), "set_baseline": True, "langfuse": LF,
    "meta": {"model": "demo-model", "prompt_version": "v1", "app_version": "1.0"}}))
print(f"{base['name']}  dataset={base['dataset_ref']}  overall={base['overall']:.1%}")
if LF:
    print("langfuse:", base["langfuse"])
for k, v in base["aggregates"].items():
    print(f"   {k:24s} {v['mean_score']:.1%}")
print("gate:", "PASS" if base["gate_result"]["passed"] else "FAIL")

# 4. improved candidate --------------------------------------------------
step("3) Candidate: a change that does NOT hurt quality")
good = ok(c.post("/experiments", json={
    "config": config, "executions": executions(latency=1300), "langfuse": LF,
    "meta": {"model": "demo-model", "prompt_version": "v2", "app_version": "1.1"}}))
print(ok(c.post("/experiments/compare", json={"candidate_id": good["id"]}))["report"])

# 5. regression ----------------------------------------------------------
step("4) Candidate: a deliberately introduced regression")
bad = ok(c.post("/experiments", json={
    "config": config, "executions": executions(wrong=(2, 5), latency=2400), "langfuse": LF,
    "meta": {"model": "demo-model", "prompt_version": "v3-broken", "app_version": "1.2"}}))
cmp = ok(c.post("/experiments/compare", json={"candidate_id": bad["id"]}))
print(cmp["report"])

# 6. drill into failed case ---------------------------------------------
step("5) Inspect a failing case")
case = cmp["cases"]["newly_failing"][0]
print("case:", case["case_id"])
for f in case["failed_evaluators"]:
    print(f"   failed: {f['evaluator']}  ->  {f['reason']}")
print("trace id:", case["trace_id"] or "(none: Langfuse disabled in this demo)")
if case.get("trace_url"):
    print("Langfuse trace:", case["trace_url"])

# 7. immutability --------------------------------------------------------
step("6) Datasets are immutable: a change creates a new version")
v2 = ok(c.post(f"/datasets/{DS}/versions", json={
    "add_cases": [{"id": "rag-011", "system": "ata-rag", "input": {"question": "New case from production?"},
                   "expected_output": {"answer": "answer 11", "relevant_doc_ids": ["doc-11"]}}],
    "note": "added a failure found in production"}))
print(f"created {v2['ref']} (parent v{v2['parent_version']}), v1 stays untouched:")
old = ok(c.get(f"/datasets/{DS}/versions/1"))
print(f"   {old['ref']} still has {old['n_cases']} cases, hash {old['content_hash'][:12]}...")

if LF:
    print(f"\nIn Langfuse look for: Datasets -> {DS}-v1 (runs: ata-rag #1, #2, #3), Traces, Scores")
print("\nExit code for CI would be:", 0 if cmp["gate"]["passed"] else 1, "(non-zero blocks deployment)")
