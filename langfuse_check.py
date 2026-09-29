"""Checks that your Langfuse keys work and that our integration shows up in the Langfuse UI.
Set env vars first (see README), then run:  python langfuse_check.py
Creates: a dataset 'platform-smoke-test-v1' (1 item), a trace with a retrieval span + generation,
a numeric score, a categorical score, and a dataset run 'smoke-test-run'."""
import sys
import uuid

from app.integrations.langfuse_client import LangfuseClient, LangfuseError

client = LangfuseClient.from_env()
if client is None:
    sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in this terminal.")
print(f"host: {client.host}")


def step(name, fn):
    try:
        out = fn()
        print(f"  OK    {name}")
        return out
    except (LangfuseError, Exception) as exc:  # noqa: BLE001
        print(f"  FAIL  {name}\n        {exc}")
        sys.exit(1)


project = step("credentials (GET /api/public/projects)", lambda: client._request("GET", "/api/public/projects"))
try:
    print("        project:", project["data"][0]["name"])
except Exception:  # noqa: BLE001
    pass

ds = "platform-smoke-test-v1"
step("create dataset", lambda: client.upsert_dataset(ds, "smoke test from ai-eval platform", {"test": True}))
step("create dataset item", lambda: client.upsert_dataset_item(
    ds, f"{ds}:case-001", {"question": "How long can an internship last?"}, {"answer": "6 months"},
    {"case_id": "case-001"}))
trace_id = uuid.uuid4().hex
step("create trace (+ retrieval span + generation)", lambda: client.create_trace(
    trace_id, "ata-rag:case-001", {"question": "How long can an internship last?"},
    {"answer": "Up to 6 months"}, {"model": "demo-model", "prompt_version": "v1"}, ["ata-rag"],
    [{"type": "span", "name": "retrieval", "input": {"q": "internship"}, "output": [{"id": "doc-1"}]},
     {"type": "generation", "name": "generation", "model": "demo-model",
      "output": "Up to 6 months", "usageDetails": {"input": 400, "output": 120},
      "costDetails": {"total": 0.003}}]))
step("link trace to dataset item (experiment run)", lambda: client.link_run_item(
    "smoke-test-run", f"{ds}:case-001", trace_id, {"model": "demo-model"}, "smoke test"))
step("numeric score", lambda: client.create_score(trace_id, "answer_correctness", 0.92, "looks right"))
step("categorical score", lambda: client.create_score(trace_id, "confusion_matrix", "TP"))

print(f"\nAll good. Open Langfuse and look for:\n  Traces   -> trace {trace_id}\n"
      f"  Datasets -> {ds} (run: smoke-test-run)\n  trace url: {client.trace_url(trace_id)}")
