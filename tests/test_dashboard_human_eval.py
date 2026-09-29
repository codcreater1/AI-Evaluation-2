from tests.test_experiments import setup_dataset
from tests.test_human_eval import run_judge


def test_nav_links_present_on_every_page(api):
    for path in ("/dashboard", "/dashboard/human-eval", "/dashboard/agreement"):
        html = api.get(path).text
        assert "/dashboard/human-eval" in html and "/dashboard/agreement" in html


def test_queue_page_lists_pending_items_and_submit_form(api, lf):
    setup_dataset(api, 2)
    exp = run_judge(api, 2, scores=[0.9, 0.1])
    run_id = exp["run_id"]
    html = api.get("/dashboard/human-eval", params={"run_id": run_id}).text
    assert "rag-001" in html and "rag-002" in html and "answer_correctness" in html
    assert "2 pending item(s)" in html

    r = api.post("/dashboard/human-eval/submit", data={
        "run_id": run_id, "case_id": "rag-001", "evaluator": "answer_correctness",
        "score": "0.8", "passed": "true", "reason": "good", "reviewer": "carol",
        "next": f"/dashboard/human-eval?run_id={run_id}",
    }, follow_redirects=False)
    assert r.status_code == 303

    after = api.get("/dashboard/human-eval", params={"run_id": run_id}).text
    assert "1 pending item(s)" in after and "rag-002" in after
    assert "value='rag-001'" not in after  # rag-001 no longer in the pending queue


def test_agreement_page_renders_report_and_demo_button(api, lf):
    setup_dataset(api, 1)
    exp = run_judge(api, 1, scores=[0.9])
    run_id = exp["run_id"]
    api.post("/dashboard/human-eval/submit", data={
        "run_id": run_id, "case_id": "rag-001", "evaluator": "answer_correctness",
        "score": "0.9", "passed": "true", "reviewer": "dana", "next": "/dashboard/agreement",
    })
    html = api.get("/dashboard/agreement", params={"run_id": run_id}).text
    assert "1 matched case(s)" in html
    assert "Cohen&#x27;s kappa" in html or "kappa" in html.lower()
    assert "Run judge-vs-human demo" in html
    assert run_id[:8] in html  # linked in "runs with judge results"


def test_agreement_page_default_view_has_no_run_filter(api, lf):
    html = api.get("/dashboard/agreement").text
    assert "0 matched case(s)" in html
    assert "Fewer than 100 matched cases" in html
