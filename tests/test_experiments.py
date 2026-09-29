GOOD = {"answer": "right", "citations": ["d1"]}
BAD = {"answer": "wrong", "citations": ["d1"]}
CFG = {
    "system": "ata-rag", "dataset": "ata-rag-golden-v1",
    "evaluators": [{"name": "exact_match", "params": {"output_key": "answer"}},
                   {"name": "latency_threshold", "params": {"max_ms": 3000}}],
    "gates": {"minimum": {"exact_match": 0.8}, "max_drop": {"default": 0.02},
              "max_increase_pct": {"latency_ms": 25}, "max_newly_failing_cases": 1},
}


def setup_dataset(api, n=10):
    cases = [{"id": f"rag-{i:03d}", "system": "ata-rag", "input": {"question": f"q{i}"},
              "expected_output": {"answer": "right"}} for i in range(1, n + 1)]
    assert api.post("/datasets", json={"name": "ata-rag-golden", "system": "ata-rag",
                                       "cases": cases}).status_code == 201


def execs(n=10, wrong=(), latency=1000, cost=0.003):
    return {f"rag-{i:03d}": {"output": BAD if i in wrong else GOOD,
                             "retrieved": [{"id": "d1"}], "latency_ms": latency,
                             "input_tokens": 100, "output_tokens": 50, "cost_usd": cost}
            for i in range(1, n + 1)}


def run(api, executions, **extra):
    body = {"config": CFG, "executions": executions,
            "meta": {"model": "gemini-x", "prompt_version": "v18"}, **extra}
    r = api.post("/experiments", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_baseline_experiment_records_everything(api, lf):
    setup_dataset(api)
    e = run(api, execs(), set_baseline=True)
    assert e["name"] == "ata-rag #1" and e["number"] == 1 and e["is_baseline"] is True
    assert e["dataset_ref"] == "ata-rag-golden-v1" and len(e["dataset_hash"]) == 64
    assert e["meta"]["model"] == "gemini-x" and e["meta"]["prompt_version"] == "v18"
    assert e["evaluator_versions"] == {"exact_match": "1", "latency_threshold": "1"}
    assert e["aggregates"]["exact_match"]["mean_score"] == 1.0
    assert e["operational"]["latency_ms"] == 1000 and e["operational"]["tokens"] == 150
    assert e["gate_result"]["passed"] is True  # minimums only, no baseline yet
    assert e["overall"] == 1.0


def test_langfuse_gets_dataset_traces_run_items_and_scores(api, lf):
    setup_dataset(api, 4)
    e = run(api, execs(4))
    assert lf.paths("/api/public/v2/datasets")[0]["name"] == "ata-rag-golden-v1"
    assert len(lf.paths("/api/public/dataset-items")) == 4
    links = lf.paths("/api/public/dataset-run-items")
    assert len(links) == 4 and {row["runName"] for row in links} == {"ata-rag #1"}
    assert len(lf.events("score-create")) == 8  # 4 cases x 2 evaluators, sent as batches
    assert not lf.paths("/api/public/scores")  # not one request per score (free-plan rate limit)
    assert len(lf.events("trace-create")) == 4
    assert e["langfuse"]["errors"] == 0 and e["langfuse"]["events_sent"] > 0
    assert e["langfuse"]["enabled"] and e["langfuse"]["dataset"] == "ata-rag-golden-v1"
    trace_ids = {row["traceId"] for row in links}
    assert len(trace_ids) == 4
    res = api.get(f"/experiments/{e['id']}/results").json()
    assert {r["trace_id"] for r in res} == trace_ids  # DB keeps the trace link for the dashboard


def test_langfuse_can_be_disabled_or_broken_without_breaking_evaluation(api, lf):
    setup_dataset(api, 3)
    e = run(api, execs(3), langfuse=False)
    assert e["langfuse"]["enabled"] is False and not lf.requests
    lf.fail_paths = ("/api/public/v2/datasets",)
    e2 = run(api, execs(3))
    assert "sync_error" in e2["langfuse"] and e2["aggregates"]["exact_match"]["mean_score"] == 1.0
    lf.fail_paths = ("/api/public/ingestion",)  # partial failure must be visible, not hidden
    e3 = run(api, execs(3))
    assert e3["langfuse"]["errors"] >= 1 and "ingestion" in e3["langfuse"]["last_error"]


def test_regression_is_detected_and_blocks(api, lf):
    setup_dataset(api)
    base = run(api, execs(), set_baseline=True)
    cand = run(api, execs(wrong=(1, 2, 3), latency=2000))  # 70% correct, latency +100%
    gr = cand["gate_result"]
    assert gr["passed"] is False and gr["baseline_id"] == base["id"]
    failed = {c["name"] for c in gr["checks"] if c["status"] == "fail"}
    assert {"exact_match", "latency_ms", "newly_failing_cases"} <= failed

    cmp = api.post("/experiments/compare", json={"candidate_id": cand["id"]}).json()
    assert "exact_match" in cmp["summary"]["regressions"]
    assert cmp["metrics"]["exact_match"]["baseline"] == 1.0
    assert abs(cmp["metrics"]["exact_match"]["delta"] + 0.3) < 1e-9
    assert cmp["operational"]["latency_ms"]["status"] == "regressed"
    nf = cmp["cases"]["newly_failing"]
    assert [c["case_id"] for c in nf] == ["rag-001", "rag-002", "rag-003"]
    assert nf[0]["failed_evaluators"][0]["evaluator"] == "exact_match"
    assert nf[0]["trace_url"] == f"https://lf.test/trace/{nf[0]['trace_id']}"
    assert cmp["report"].startswith("FAIL") and "Deployment should be blocked." in cmp["report"]


def test_improvement_and_fixed_cases_pass(api, lf):
    setup_dataset(api)
    base = run(api, execs(wrong=(1, 2)), set_baseline=True)  # 80% -> meets minimum
    assert base["gate_result"]["passed"] is True
    cand = run(api, execs())
    assert cand["gate_result"]["passed"] is True
    cmp = api.post("/experiments/compare", json={"candidate_id": cand["id"]}).json()
    assert cmp["summary"]["improvements"] == ["exact_match"]
    assert cmp["summary"]["unchanged"] == ["latency_threshold"]
    assert [c["case_id"] for c in cmp["cases"]["fixed"]] == ["rag-001", "rag-002"]
    assert cmp["cases"]["newly_failing"] == []


def test_compare_with_explicit_baseline_and_custom_gates(api, lf):
    setup_dataset(api)
    a = run(api, execs())
    b = run(api, execs(wrong=(1,)))  # -10 pts
    strict = {"max_drop": {"default": 0.02}}
    lenient = {"max_drop": {"default": 0.15}}
    r1 = api.post("/experiments/compare", json={"candidate_id": b["id"], "baseline_id": a["id"],
                                                "gates": strict}).json()
    r2 = api.post("/experiments/compare", json={"candidate_id": b["id"], "baseline_id": a["id"],
                                                "gates": lenient}).json()
    assert r1["gate"]["passed"] is False and r2["gate"]["passed"] is True


def test_baseline_switching_history_and_numbering(api, lf):
    setup_dataset(api, 3)
    e1 = run(api, execs(3), set_baseline=True)
    e2 = run(api, execs(3))
    assert (e1["number"], e2["number"]) == (1, 2)
    assert api.post(f"/experiments/{e2['id']}/baseline").json()["is_baseline"] is True
    hist = api.get("/experiments?system=ata-rag").json()
    assert [h["number"] for h in hist] == [2, 1]
    assert [h["is_baseline"] for h in hist] == [True, False]  # only one baseline per system
    assert api.get(f"/experiments/{e1['id']}").json()["name"] == "ata-rag #1"


def test_experiment_pins_dataset_version(api, lf):
    setup_dataset(api, 3)
    e1 = run(api, execs(3))
    api.post("/datasets/ata-rag-golden/versions", json={"add_cases": [
        {"id": "rag-999", "system": "ata-rag", "input": {"question": "new"},
         "expected_output": {"answer": "right"}}]})
    body = {"config": {**CFG, "dataset": "ata-rag-golden"}, "executions": {
        **execs(3), "rag-999": {"output": GOOD, "retrieved": [{"id": "d1"}], "latency_ms": 1000}}}
    e2 = api.post("/experiments", json=body).json()  # no -vN => latest
    assert e2["dataset_ref"] == "ata-rag-golden-v2" and e1["dataset_ref"] == "ata-rag-golden-v1"
    assert e1["dataset_hash"] != e2["dataset_hash"]
    cmp = api.post("/experiments/compare", json={"candidate_id": e2["id"], "baseline_id": e1["id"]}).json()
    assert cmp["dataset_mismatch"] is True and cmp["cases"]["only_in_candidate"] == 1


def test_errors(api, lf):
    setup_dataset(api, 2)
    bad_ds = {**CFG, "dataset": "nope-v1"}
    assert api.post("/experiments", json={"config": bad_ds, "executions": {}}).status_code == 404
    assert api.post("/experiments", json={"config": CFG, "executions": {}}).status_code == 422
    wrong_sys = {**CFG, "system": "internship-coordinator"}
    assert api.post("/experiments", json={"config": wrong_sys, "executions": execs(2)}).status_code == 422
    unknown_ev = {**CFG, "evaluators": ["does_not_exist"]}
    assert api.post("/experiments", json={"config": unknown_ev, "executions": execs(2)}).status_code == 422
    assert api.get("/experiments/nope").status_code == 404
    assert api.post("/experiments/compare", json={"candidate_id": "nope"}).status_code == 404


def test_cli_gate_exit_codes(api, lf, monkeypatch, capsys):
    from app import cli

    monkeypatch.setattr(cli.httpx, "post",
                        lambda url, json, timeout: api.post(url.replace("http://x", ""), json=json))
    setup_dataset(api)
    base = run(api, execs(), set_baseline=True)
    good = run(api, execs())
    bad = run(api, execs(wrong=(1, 2, 3)))
    assert cli.main(["gate", "--api", "http://x", "--candidate", good["id"]]) == 0
    assert "PASS" in capsys.readouterr().out
    code = cli.main(["gate", "--api", "http://x", "--candidate", bad["id"], "--baseline", base["id"]])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out and "rag-001" in out
