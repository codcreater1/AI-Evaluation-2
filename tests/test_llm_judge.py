import pytest

from app.evaluators import create_evaluator
from app.evaluators.llm_judge.judge import JudgeError, parse_judge_json
from tests.conftest import FakeLLM


def test_parse_variants():
    assert parse_judge_json('{"score": 0.8, "reason": "r"}') == (0.8, "r")
    assert parse_judge_json('```json\n{"score": 2, "reason": "x"}\n```')[0] == 1.0
    assert parse_judge_json('Sure! {"score": 0.3, "reason": "y"} bye')[0] == 0.3
    with pytest.raises(JudgeError):
        parse_judge_json("no json here")
    with pytest.raises(JudgeError):
        parse_judge_json('{"reason": "no score"}')


def test_judge_records_model_and_prompt_version(rag_case, rag_exec):
    llm = FakeLLM('{"score": 0.95, "reason": "matches"}', model="gemini-x")
    r = create_evaluator("answer_correctness", client=llm).evaluate(rag_case, rag_exec)
    assert r.score == 0.95 and r.passed is True and r.reason == "matches"
    assert r.metadata["judge_model"] == "gemini-x"
    assert r.metadata["prompt_version"] == "v1" and r.evaluator_version == "v1"
    prompt = llm.calls[0][1]
    assert "How long can an internship last?" in prompt and "Up to 6 months" in prompt


def test_pass_threshold_param(rag_case, rag_exec):
    llm = FakeLLM('{"score": 0.6, "reason": "meh"}')
    ev = create_evaluator("groundedness", client=llm, pass_threshold=0.8)
    assert ev.evaluate(rag_case, rag_exec).passed is False
    assert "Internship max 6 months" in llm.calls[0][1]  # retrieved context is in the prompt


def test_internship_judges_use_their_keys(rag_case, rag_exec):
    from app.schemas import EvaluationCase, ExecutionResult

    case = EvaluationCase(id="i1", system="internship", input={"documents": ["cv"]},
                          expected_output={"decision": "eligible"})
    ex = ExecutionResult(output={"decision": "not_eligible", "explanation": "No transcript"})
    llm = FakeLLM()
    create_evaluator("decision_correctness", client=llm).evaluate(case, ex)
    assert "EXPECTED DECISION: eligible" in llm.calls[0][1]
    assert "ACTUAL DECISION: not_eligible" in llm.calls[0][1]
    create_evaluator("explanation_quality", client=llm).evaluate(case, ex)
    assert "No transcript" in llm.calls[1][1]
    create_evaluator("hallucination_detection", client=llm).evaluate(case, ex)
    assert "cv" in llm.calls[2][1]
