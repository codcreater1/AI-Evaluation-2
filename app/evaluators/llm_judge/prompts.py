from dataclasses import dataclass, field

SYSTEM = (
    "You are a strict, impartial evaluator of AI system outputs. "
    'Reply with ONE JSON object only, no markdown: {"score": <float 0.0-1.0>, "reason": "<1-3 sentences>"}.'
)


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    version: str
    template: str  # string.Template, variables: see judge.build_variables
    default_params: dict = field(default_factory=dict)


SPECS: dict[str, JudgeSpec] = {
    # ------------- ATA RAG -------------
    "answer_correctness": JudgeSpec(
        "answer_correctness", "v1",
        """Judge whether the ACTUAL answer is factually correct compared with the EXPECTED answer.
If the expected answer says the information is not available, the actual answer must also
say so (do not reward invented information).
Scoring: 1.0 = fully correct and complete; 0.5 = partially correct or missing key details;
0.0 = wrong, contradictory or fabricated.

QUESTION: $question
EXPECTED ANSWER: $expected
ACTUAL ANSWER: $actual
""",
    ),
    "groundedness": JudgeSpec(
        "groundedness", "v1",
        """Judge whether every claim in the ANSWER is supported by the RETRIEVED CONTEXT only
(ignore outside knowledge, even if true).
Scoring: 1.0 = all claims supported; 0.5 = some claims unsupported; 0.0 = mostly unsupported
or contradicts the context. If the answer honestly says it cannot answer, score 1.0.

QUESTION: $question
RETRIEVED CONTEXT:
$context
ANSWER: $actual
""",
    ),
    "citation_correctness": JudgeSpec(
        "citation_correctness", "v1",
        """Judge whether the CITATIONS given in the answer really support the claims they are
attached to, using the RETRIEVED CONTEXT (each item shows its source).
Scoring: 1.0 = every citation supports its claim; 0.5 = some wrong/irrelevant; 0.0 = citations
are missing for factual claims, fabricated, or unrelated.

ANSWER: $actual
CITATIONS: $citations
RETRIEVED CONTEXT:
$context
""",
    ),
    # ------------- Internship Coordinator -------------
    "decision_correctness": JudgeSpec(
        "decision_correctness", "v1",
        """Judge whether the system's DECISION about the internship application is correct
compared with the EXPECTED decision. Consider the decision label and any conditions it states.
Scoring: 1.0 = same decision; 0.0 = different decision; use 0.5 only when the decision is
ambiguous but reasonably consistent with the expected one.

APPLICATION (input): $input_json
EXPECTED DECISION: $expected
ACTUAL DECISION: $actual
""",
        {"answer_key": "decision"},
    ),
    "explanation_quality": JudgeSpec(
        "explanation_quality", "v1",
        """Judge the quality of the EXPLANATION given for the decision: is it accurate, specific to
the application, consistent with the decision, and useful to a coordinator (names concrete missing
documents / violated rules)?
Scoring: 1.0 = accurate, specific, consistent; 0.5 = generic or partly wrong; 0.0 = wrong,
contradicts the decision, or empty.

APPLICATION (input): $input_json
DECISION: $decision
EXPLANATION: $actual
EXPECTED OUTPUT (reference): $expected_json
""",
        {"answer_key": "explanation"},
    ),
    "hallucination_detection": JudgeSpec(
        "hallucination_detection", "v1",
        """Detect hallucinations: statements in the OUTPUT (facts, names, dates, documents, rules)
that are NOT supported by the SOURCE application data.
Scoring (higher is better): 1.0 = no hallucination; 0.5 = minor unsupported detail;
0.0 = clear fabricated information.

SOURCE (input): $input_json
OUTPUT: $output_json
""",
    ),
}
