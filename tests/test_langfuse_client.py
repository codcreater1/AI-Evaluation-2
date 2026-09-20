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
    assert not lf.paths("/api/public/scores")  # scores are batched through ingestion, not 1 call each
    sink.flush()
    assert len(lf.events("score-create")) == 2  # not_applicable is skipped
    assert len(lf.paths("/api/public/ingestion")) == 1  # one batch for trace + steps + scores
    link = lf.paths("/api/public/dataset-run-items")[0]
    assert link["datasetItemId"] == "ds-v1:c1" and link["traceId"] == ex.trace_id


def test_sink_keeps_existing_trace_id(lf):
    sink = LangfuseSink(lf.client, run_name="r", dataset_ref="ds-v1")
    ex = ExecutionResult(output={}, trace_id="from-app")
    sink.on_execution("run", EvaluationCase(id="c", system="s"), ex)
    sink.flush()
    assert ex.trace_id == "from-app" and not lf.paths("/api/public/ingestion")


def test_429_is_retried_after_waiting_and_then_succeeds():
    import httpx

    from app.integrations.langfuse_client import LangfuseClient

    calls, slept = [], []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"code": "rate_limited"})
        return httpx.Response(200, json={})

    c = LangfuseClient("https://lf.test", "pk", "sk", transport=httpx.MockTransport(handler),
                       sleep=slept.append)
    c.create_score("t", "n", 0.5)
    assert len(calls) == 3 and slept == [3.0, 3.0]  # waited Retry-After (+1s margin) twice


def test_429_reads_fail_fast_and_report_retry_time():
    import httpx
    import pytest

    from app.integrations.langfuse_client import LangfuseClient, LangfuseRateLimited

    slept = []
    body = {"code": "rate_limited", "details": {"retryAfterSeconds": 38}}
    c = LangfuseClient("https://lf.test", "pk", "sk", sleep=slept.append,
                       transport=httpx.MockTransport(lambda r: httpx.Response(429, json=body)))
    with pytest.raises(LangfuseRateLimited) as exc:
        c.list_scores()
    assert exc.value.retry_after == 39 and slept == []  # page reads never sleep


def test_sink_counts_errors_instead_of_hiding_them():
    import httpx

    from app.integrations.langfuse_client import LangfuseClient

    c = LangfuseClient("https://lf.test", "pk", "sk", sleep=lambda s: None,
                       transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    sink = LangfuseSink(c, run_name="r", dataset_ref="ds-v1")
    case = EvaluationCase(id="c1", system="s")
    ex = ExecutionResult(output={"a": 1})
    sink.on_execution("run", case, ex)
    sink.on_result("run", case, ex, EvaluationResult(evaluator="e", score=1.0, passed=True))
    sink.flush()
    assert sink.error_count == 2 and "boom" in sink.last_error  # run-item link + ingestion batch
