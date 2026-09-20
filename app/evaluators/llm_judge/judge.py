import json
import re
from string import Template
from typing import Any

from app.evaluators.base import BaseEvaluator, register_factory
from app.evaluators.llm_judge.client import LLMClient, get_default_client
from app.evaluators.llm_judge.prompts import SPECS, SYSTEM, JudgeSpec
from app.evaluators.utils import get_path
from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult


class JudgeError(ValueError):
    pass


def _js(obj: Any, limit: int = 12000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


def _text(v: Any) -> str:
    if v is None:
        return "(none)"
    return v if isinstance(v, str) else _js(v)


def build_variables(case: EvaluationCase, ex: ExecutionResult, params: dict) -> dict[str, str]:
    qkey = params.get("question_key", "question")
    akey = params.get("answer_key", "answer")
    ekey = params.get("expected_key", akey)
    ctx = "\n\n".join(
        f"[{i}] (source: {d.get('url') or d.get('id') or 'unknown'})\n"
        f"{d.get('text') or d.get('content') or ''}"
        for i, d in enumerate(ex.retrieved, 1)
    )
    max_ctx = int(params.get("max_context_chars", 12000))
    return {
        "question": _text(get_path(case.input, qkey, _js(case.input))),
        "expected": _text(get_path(case.expected_output, ekey)),
        "actual": _text(get_path(ex.output, akey)),
        "decision": _text(get_path(ex.output, params.get("decision_key", "decision"))),
        "citations": _text(ex.output.get(params.get("citations_key", "citations"))),
        "context": (ctx[:max_ctx] + "\n...[truncated]") if len(ctx) > max_ctx else (ctx or "(none)"),
        "input_json": _js(case.input),
        "expected_json": _js(case.expected_output),
        "output_json": _js(ex.output),
    }


def parse_judge_json(text: str) -> tuple[float, str]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise JudgeError(f"judge did not return JSON: {text[:200]!r}") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise JudgeError(f"invalid judge JSON: {e}") from None
    try:
        score = float(obj["score"])
    except (KeyError, TypeError, ValueError):
        raise JudgeError(f"judge JSON has no numeric 'score': {obj!r}") from None
    return min(1.0, max(0.0, score)), str(obj.get("reason", ""))


class LLMJudgeEvaluator(BaseEvaluator):
    """One generic judge; each metric = a prompt spec in prompts.py.
    params: pass_threshold=0.7, client=<LLMClient> (default: env-configured), *_key overrides."""

    kind = "llm_judge"

    def __init__(self, metric: str, **params: Any) -> None:
        if metric not in SPECS:
            raise KeyError(f"unknown judge metric {metric}")
        self.spec: JudgeSpec = SPECS[metric]
        client = params.pop("client", None)
        super().__init__(**{**self.spec.default_params, **params})
        self.name = metric
        self.version = self.spec.version
        self._client: LLMClient | None = client

    @property
    def client(self) -> LLMClient:
        return self._client or get_default_client()

    def evaluate(self, case, execution):
        variables = build_variables(case, execution, self.params)
        prompt = Template(self.spec.template).safe_substitute(variables)
        raw = self.client.complete(SYSTEM, prompt)
        score, reason = parse_judge_json(raw)
        threshold = float(self.params.get("pass_threshold", 0.7))
        return EvaluationResult(
            evaluator=self.name,
            evaluator_version=self.version,
            score=score,
            passed=score >= threshold,
            reason=reason,
            metadata={
                "judge_model": self.client.model,
                "prompt_name": self.spec.name,
                "prompt_version": self.spec.version,
                "pass_threshold": threshold,
            },
        )


for _name in SPECS:
    register_factory(_name, lambda _n=_name, **p: LLMJudgeEvaluator(_n, **p))
