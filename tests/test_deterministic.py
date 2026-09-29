from app.evaluators import create_evaluator
from app.schemas import EvaluationCase, ExecutionResult


def ev(name, case, ex, **params):
    return create_evaluator(name, **params).evaluate(case, ex)


def test_exact_match_normalizes(rag_case, rag_exec):
    assert ev("exact_match", rag_case, rag_exec).passed is True
    rag_exec.output["answer"] = "1 year"
    r = ev("exact_match", rag_case, rag_exec)
    assert r.passed is False and r.score == 0.0


def test_exact_match_na_without_expected(rag_exec):
    c = EvaluationCase(id="a", system="s")
    assert ev("exact_match", c, rag_exec).label == "not_applicable"


def test_required_fields(rag_case, rag_exec):
    r = ev("required_fields", rag_case, rag_exec, fields=["answer", "citations", "missing.x"])
    assert r.passed is False and abs(r.score - 2 / 3) < 1e-9
    assert r.metadata["missing"] == ["missing.x"]


def test_json_schema(rag_case, rag_exec):
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    r = ev("json_schema", rag_case, rag_exec, schema=schema)
    assert r.passed is False and "answer" in r.reason
    ok = {"type": "object", "required": ["answer"]}
    assert ev("json_schema", rag_case, rag_exec, schema=ok).passed is True


def test_recall_precision(rag_case, rag_exec):
    # relevant {d1,d2}; top3 = d1,x,d3 -> recall 1/2, precision 1/3
    r = ev("retrieval_recall_at_k", rag_case, rag_exec, k=3)
    assert r.score == 0.5 and r.passed is True
    p = ev("retrieval_precision_at_k", rag_case, rag_exec, k=3)
    assert abs(p.score - 1 / 3) < 1e-9
    # k=1 -> recall 0.5, precision 1.0
    assert ev("retrieval_precision_at_k", rag_case, rag_exec, k=1).score == 1.0


def test_recall_not_applicable_for_unanswerable(rag_exec):
    c = EvaluationCase(id="u", system="ata-rag", expected_output={"answerable": False})
    assert ev("retrieval_recall_at_k", c, rag_exec).label == "not_applicable"


def test_expected_document_present(rag_case, rag_exec):
    assert ev("expected_document_present", rag_case, rag_exec).passed is True
    rag_case.expected_output["expected_documents"] = ["d9"]
    assert ev("expected_document_present", rag_case, rag_exec).passed is False


def test_thresholds(rag_case, rag_exec):
    assert ev("latency_threshold", rag_case, rag_exec, max_ms=2000).passed is True
    assert ev("latency_threshold", rag_case, rag_exec, max_ms=1000).passed is False
    assert ev("cost_threshold", rag_case, rag_exec, max_usd=0.001).passed is False
    assert ev("token_threshold", rag_case, rag_exec, max_tokens=150).passed is True
    missing = ExecutionResult(output={})
    assert ev("latency_threshold", rag_case, missing, max_ms=1).label == "missing_data"


def test_citation_exists(rag_case, rag_exec):
    assert ev("citation_exists", rag_case, rag_exec).passed is True
    rag_exec.output["citations"] = ["d1", "ghost"]
    r = ev("citation_exists", rag_case, rag_exec)
    assert r.score == 0.5 and r.passed is False
    rag_exec.output["citations"] = []
    assert ev("citation_exists", rag_case, rag_exec).passed is False
