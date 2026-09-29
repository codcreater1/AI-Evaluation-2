from app.api.deps import get_langfuse
from app.api.main import app as fastapi_app
from tests.test_experiments import execs, run, setup_dataset


def test_empty_dashboard_renders(api):
    r = api.get("/dashboard")
    assert r.status_code == 200 and "AI Evaluation Platform" in r.text
    assert "No experiments yet" in r.text
    assert api.get("/", follow_redirects=False).headers["location"] == "/dashboard"


def test_dashboard_shows_experiments_gate_and_langfuse_data(api, lf):
    setup_dataset(api)
    run(api, execs(), set_baseline=True)
    bad = run(api, execs(wrong=(1, 2, 3), latency=2000))
    html = api.get("/dashboard").text
    assert "ata-rag #1" in html and "ata-rag #2" in html and "ata-rag-golden-v1" in html
    assert "FAIL" in html and "PASS" in html and "&#9733;" in html  # baseline star
    assert "gemini-x" in html and "v18" in html  # model / prompt from experiment meta
    # data read back live from Langfuse
    assert "fake-project" in html
    assert "https://lf.test/trace/trace-from-langfuse" in html
    assert "ata-rag-golden-v1" in html  # dataset list from Langfuse

    detail = api.get(f"/dashboard/experiments/{bad['id']}").text
    assert "FAIL - deployment should be blocked" in detail
    assert "rag-001" in detail and "rag-003" in detail  # failed cases
    assert "Open in Langfuse" in detail and "https://lf.test/trace/" in detail
    assert "regressed" in detail and "Newly failing cases: <b>3</b>" in detail
    assert "scores found in Langfuse" in detail


def test_dashboard_escapes_html(api, lf):
    evil = "<script>alert(1)</script>"
    cases = [{"id": evil, "system": "ata-rag", "input": {"q": 1}, "expected_output": {"answer": "right"}}]
    api.post("/datasets", json={"name": "evil", "system": "ata-rag", "cases": cases})
    body = {"config": {"system": "ata-rag", "dataset": "evil-v1",
                       "evaluators": [{"name": "exact_match", "params": {"output_key": "answer"}}]},
            "executions": {evil: {"output": {"answer": "wrong"}, "latency_ms": 1}}}
    exp = api.post("/experiments", json=body).json()
    html = api.get(f"/dashboard/experiments/{exp['id']}").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_dashboard_without_langfuse(api):
    fastapi_app.dependency_overrides[get_langfuse] = lambda: None
    setup_dataset(api, 3)
    run(api, execs(3), langfuse=False)
    html = api.get("/dashboard").text
    assert "not configured" in html and "ata-rag #1" in html


def test_run_test_scenario_button(api, lf):
    r = api.post("/dashboard/run-demo", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard"
    exps = api.get("/experiments?system=demo-ata-rag").json()
    assert len(exps) == 3
    by_num = {e["number"]: e for e in exps}
    assert by_num[1]["is_baseline"] and by_num[1]["gate_result"]["passed"] is True
    assert by_num[2]["gate_result"]["passed"] is True
    assert by_num[3]["gate_result"]["passed"] is False  # the deliberately broken one
    assert len(lf.paths("/api/public/dataset-run-items")) == 18  # 6 cases x 3 experiments
    assert len(lf.events("score-create")) == 72  # 18 x 4 evaluators, all delivered
    assert "demo-ata-rag #3" in api.get("/dashboard").text


def test_unknown_experiment_page(api):
    assert "not found" in api.get("/dashboard/experiments/nope").text.lower()


def test_dashboard_shows_langfuse_errors_and_rate_limit(api, lf):
    setup_dataset(api, 3)
    lf.fail_paths = ("/api/public/ingestion",)
    run(api, execs(3))
    html = api.get("/dashboard").text
    assert "Langfuse errors" in html
    exp_id = api.get("/experiments").json()[0]["id"]
    assert "Langfuse request(s) failed" in api.get(f"/dashboard/experiments/{exp_id}").text


def test_dashboard_rate_limit_message_and_cache(api, lf):
    import httpx

    from app.integrations.langfuse_client import LangfuseClient

    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/public/projects":
            return httpx.Response(200, json={"data": [{"name": "p"}]})
        return httpx.Response(429, json={"details": {"retryAfterSeconds": 38}})

    limited = LangfuseClient("https://lf.test", "pk", "sk", transport=httpx.MockTransport(handler))
    fastapi_app.dependency_overrides[get_langfuse] = lambda: limited
    html = api.get("/dashboard").text
    assert "rate limit reached" in html and "Wait ~39 s" in html
    n = len(calls)
    api.get("/dashboard")
    assert len(calls) == n  # second page load served from cache: no extra Langfuse calls
