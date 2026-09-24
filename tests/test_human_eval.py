from tests.test_experiments import execs, run, setup_dataset

JUDGE_CFG = {
    "system": "ata-rag", "dataset": "ata-rag-golden-v1",
    "evaluators": [{"name": "answer_correctness", "params": {"answer_key": "answer"}}],
}


class ScriptedLLM:
    """Structurally matches LLMClient (model attribute + complete()); returns scores from a
    fixed script instead of calling a real API, so judge-vs-human tests are deterministic."""

    def __init__(self, scores: list[float]):
        self.scores, self.calls, self.model = scores, 0, "scripted-judge"

    def complete(self, system, prompt):
        v = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return f'{{"score": {v}, "reason": "scripted"}}'


def run_judge(api, n=3, scores=None):
    """A run scored by a scripted LLM judge, so EvaluationResultRow gets llm_judge-kind rows
    (evaluator="answer_correctness") for a human review to react against."""
    import app.experiments.service as svc
    from app.config.loader import build_evaluators
    from app.evaluators.llm_judge.judge import LLMJudgeEvaluator

    scripted = ScriptedLLM(scores or [0.9] * n)
    original = build_evaluators

    def patched(config):
        evs = original(config)
        for e in evs:
            if isinstance(e, LLMJudgeEvaluator):
                e._client = scripted
        return evs

    svc.build_evaluators = patched
    try:
        r = api.post("/experiments", json={"config": JUDGE_CFG, "executions": execs(n),
                                           "meta": {"model": "scripted-judge"}})
        assert r.status_code == 201, r.text
        return r.json()
    finally:
        svc.build_evaluators = original


def test_submit_records_and_links_trace(api, lf):
    setup_dataset(api, 3)
    exp = run_judge(api, 3, scores=[0.8, 0.2, 0.9])
    run_id = exp["run_id"]

    r = api.post("/human-evaluations", json={
        "run_id": run_id, "case_id": "rag-001", "evaluator": "answer_correctness",
        "score": 0.7, "passed": True, "reason": "looks right", "reviewer": "alice@example.com",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["case_id"] == "rag-001" and body["reviewer"] == "alice@example.com"
    assert body["trace_id"]  # copied from the matching EvaluationResultRow

    listed = api.get("/human-evaluations", params={"run_id": run_id}).json()
    assert len(listed) == 1 and listed[0]["score"] == 0.7

    # pushed to Langfuse as a human_<evaluator> score (direct call, not batched via ingestion)
    scores = lf.paths("/api/public/scores")
    human_scores = [s for s in scores if s["name"] == "human_answer_correctness"]
    assert len(human_scores) == 1 and human_scores[0]["value"] == 0.7


def test_resubmit_by_same_reviewer_updates_not_duplicates(api, lf):
    setup_dataset(api, 2)
    exp = run_judge(api, 2)
    run_id = exp["run_id"]
    body = {"run_id": run_id, "case_id": "rag-001", "evaluator": "answer_correctness",
           "score": 0.5, "passed": False, "reviewer": "bob"}
    api.post("/human-evaluations", json=body)
    r = api.post("/human-evaluations", json={**body, "score": 0.9, "passed": True, "reason": "changed my mind"})
    assert r.status_code == 201
    listed = api.get("/human-evaluations", params={"run_id": run_id}).json()
    assert len(listed) == 1
    assert listed[0]["score"] == 0.9 and listed[0]["reason"] == "changed my mind"


def test_different_reviewers_both_count(api, lf):
    setup_dataset(api, 1)
    exp = run_judge(api, 1)
    run_id = exp["run_id"]
    base = {"run_id": run_id, "case_id": "rag-001", "evaluator": "answer_correctness", "score": 0.6}
    api.post("/human-evaluations", json={**base, "reviewer": "alice"})
    api.post("/human-evaluations", json={**base, "reviewer": "bob", "score": 0.4})
    listed = api.get("/human-evaluations", params={"run_id": run_id}).json()
    assert len(listed) == 2


def test_submit_requires_score_or_passed(api, lf):
    setup_dataset(api, 1)
    exp = run_judge(api, 1)
    r = api.post("/human-evaluations", json={"run_id": exp["run_id"], "case_id": "rag-001",
                                             "evaluator": "answer_correctness"})
    assert r.status_code == 422


def test_submit_rejects_unknown_run(api, lf):
    r = api.post("/human-evaluations", json={"run_id": "nope", "case_id": "x",
                                             "evaluator": "answer_correctness", "score": 0.5})
    assert r.status_code == 404


def test_queue_excludes_already_reviewed(api, lf):
    setup_dataset(api, 3)
    exp = run_judge(api, 3)
    run_id = exp["run_id"]
    q0 = api.get("/human-evaluations/queue", params={"run_id": run_id}).json()
    assert {q["case_id"] for q in q0} == {"rag-001", "rag-002", "rag-003"}
    assert all(q["evaluator"] == "answer_correctness" for q in q0)

    api.post("/human-evaluations", json={"run_id": run_id, "case_id": "rag-001",
                                         "evaluator": "answer_correctness", "score": 0.9})
    q1 = api.get("/human-evaluations/queue", params={"run_id": run_id}).json()
    assert {q["case_id"] for q in q1} == {"rag-002", "rag-003"}

    # a per-reviewer queue still offers a case another reviewer already judged
    q2 = api.get("/human-evaluations/queue",
                params={"run_id": run_id, "reviewer": "someone-else"}).json()
    assert {q["case_id"] for q in q2} == {"rag-001", "rag-002", "rag-003"}


def test_queue_excludes_deterministic_evaluators(api, lf):
    setup_dataset(api, 2)
    exp = run(api, execs(2))  # deterministic-only run (exact_match, latency_threshold)
    q = api.get("/human-evaluations/queue", params={"run_id": exp["run_id"]}).json()
    assert q == []


def test_agreement_report_end_to_end(api, lf):
    setup_dataset(api, 4)
    exp = run_judge(api, 4, scores=[0.9, 0.9, 0.2, 0.9])  # judge: pass,pass,fail,pass (threshold 0.7)
    run_id = exp["run_id"]
    reviews = [
        ("rag-001", 0.95, True),   # agrees
        ("rag-002", 0.1, False),  # disagrees (judge said pass)
        ("rag-003", 0.1, False),  # agrees (both fail)
        ("rag-004", 0.85, True),  # agrees
    ]
    for case_id, score, passed in reviews:
        api.post("/human-evaluations", json={"run_id": run_id, "case_id": case_id,
                                             "evaluator": "answer_correctness",
                                             "score": score, "passed": passed, "reviewer": "alice"})
    rep = api.get("/human-evaluations/agreement", params={"run_id": run_id}).json()
    assert rep["n"] == 4 and rep["n_pass_fail"] == 4
    assert rep["confusion"]["both_pass"] == 2 and rep["confusion"]["both_fail"] == 1
    assert rep["confusion"]["human_fail_judge_pass"] == 1
    assert rep["pass_agreement_rate"] == 0.75
    assert rep["cohens_kappa"] is not None
    assert rep["disagreements"][0]["case_id"] == "rag-002"


def test_agreement_report_empty_when_nothing_reviewed(api, lf):
    setup_dataset(api, 2)
    exp = run_judge(api, 2)
    rep = api.get("/human-evaluations/agreement", params={"run_id": exp["run_id"]}).json()
    assert rep["n"] == 0 and rep["pass_agreement_rate"] is None
