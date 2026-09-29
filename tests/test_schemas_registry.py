import pytest
from pydantic import ValidationError

from app.evaluators import BaseEvaluator, create_evaluator, list_evaluators, register_evaluator
from app.schemas import EvaluationCase, EvaluationResult


def test_case_is_free_form():
    c = EvaluationCase(id="i-1", system="internship", input={"documents": [{"type": "cv"}]})
    assert c.expected_output == {} and c.metadata == {}


def test_result_score_range_and_categorical():
    with pytest.raises(ValidationError):
        EvaluationResult(evaluator="x", score=1.5)
    r = EvaluationResult(evaluator="x", label="FN", passed=False)
    assert r.score is None


def test_builtin_evaluators_registered():
    names = set(list_evaluators())
    assert {"exact_match", "required_fields", "json_schema", "retrieval_recall_at_k",
            "retrieval_precision_at_k", "latency_threshold", "cost_threshold", "token_threshold",
            "citation_exists", "expected_document_present", "answer_correctness", "groundedness",
            "citation_correctness", "decision_correctness", "explanation_quality",
            "hallucination_detection"} <= names


def test_plugin_evaluator_without_touching_engine(rag_case, rag_exec):
    @register_evaluator("always_pass_test")
    class AlwaysPass(BaseEvaluator):
        def evaluate(self, case, execution):
            return self.make_result(score=1.0, passed=True)

    ev = create_evaluator("always_pass_test")
    assert ev.evaluate(rag_case, rag_exec).evaluator == "always_pass_test"


def test_unknown_evaluator():
    with pytest.raises(KeyError):
        create_evaluator("nope")
