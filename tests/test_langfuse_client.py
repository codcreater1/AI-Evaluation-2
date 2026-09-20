import base64

from app.integrations.langfuse_hook import LangfuseSink
from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult


def test_dataset_item_run_item_and_score_requests(lf):
    c = lf.client
    c.upsert_dataset("ata-rag-golden-v1", "d", {"version": 1})
    c.upsert_dataset_item("ata-rag-golden-v1", "ata-rag-golden-v1:rag-001", {"q": 1}, {"a": 2}, {"m": 1})
    c.link_run_item("ATA #1", "ata-rag-golden-v1:rag-001", "trace-1", {"model": "x"})
    c.create_score("trace-1", "answer_correctness", 0.9, "good", score_id="sid")
    c.create_score("trace-1", "confusion_matrix", "FN")
    paths = [p for _, p, _ in lf.requests]
    assert paths == ["/api/public/v2/datasets", "/api/public/dataset-items",
                     "/api/public/dataset-run-items", "/api/public/scores", "/api/public/scores"]
    assert lf.paths("/api/public/dataset-run-items")[0] == {
        "runName": "ATA #1", "runDescription": "", "datasetItemId": "ata-rag-golden-v1:rag-001",
        "traceId": "trace-1", "metadata": {"model": "x"}}
    scores = lf.paths("/api/public/scores")
    assert scores[0]["dataType"] == "NUMERIC" and scores[0]["id"] == "sid"
    assert scores[1]["dataType"] == "CATEGORICAL" and scores[1]["value"] == "FN"


def test_basic_auth_header(lf):
    lf.client.upsert_dataset("x")
    header = lf.client._http.auth._auth_header
    assert header == "Basic " + base64.b64encode(b"pk:sk").decode()


def test_create_trace_batches_trace_and_steps(lf):
    lf.client.create_trace("t1", "ata-rag:c1", {"q": 1}, {"a": 1}, {"m": 1}, ["ata-rag"],
                           [{"type": "span", "name": "retrieval", "output": []},
                            {"type": "generation", "name": "generation", "model": "m"}])
    batch = lf.paths("/api/public/ingestion")[0]["batch"]
    assert [e["type"] for e in batch] == ["trace-create", "span-create", "generation-create"]
    assert all(e["body"].get("traceId", "t1") == "t1" for e in batch[1:])


def test_sink_creates_trace_links_item_and_scores(lf):
    sink = LangfuseSink(lf.client, run_name="ATA #1", dataset_ref="ds-v1", metadata={"model": "m"})
    case = EvaluationCase(id="c1", system="ata-rag", input={"question": "q"})
    ex = ExecutionResult(output={"answer": "a"}, retrieved=[{"id": "d1", "text": "t"}],
                         input_tokens=5, output_tokens=7, cost_usd=0.01)
    sink.on_execution("run-1", case, ex)
    assert ex.trace_id  # engine created a trace because the system had none
    sink.on_result("run-1", case, ex, EvaluationResult(evaluator="e", score=0.5, passed=False))
    sink.on_result("run-1", case, ex, EvaluationResult(evaluator="cm", label="TP"))
    sink.on_result("run-1", case, ex, EvaluationResult(evaluator="na", label="not_applicable"))
    assert len(lf.paths("/api/public/scores")) == 2  # not_applicable is skipped
    link = lf.paths("/api/public/dataset-run-items")[0]
    assert link["datasetItemId"] == "ds-v1:c1" and link["traceId"] == ex.trace_id


def test_sink_keeps_existing_trace_id(lf):
    sink = LangfuseSink(lf.client, run_name="r", dataset_ref="ds-v1")
    ex = ExecutionResult(output={}, trace_id="from-app")
    sink.on_execution("run", EvaluationCase(id="c", system="s"), ex)
    assert ex.trace_id == "from-app" and not lf.paths("/api/public/ingestion")
