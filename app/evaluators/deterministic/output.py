from jsonschema import Draft202012Validator

from app.evaluators.base import BaseEvaluator, register_evaluator
from app.evaluators.utils import get_path, is_empty


def _norm(v, case_sensitive: bool, collapse: bool):
    if isinstance(v, str):
        s = " ".join(v.split()) if collapse else v.strip()
        return s if case_sensitive else s.casefold()
    return v


@register_evaluator("exact_match")
class ExactMatch(BaseEvaluator):
    """params: output_key='answer', expected_key=<output_key>, case_sensitive=False"""

    def evaluate(self, case, execution):
        okey = self.params.get("output_key", "answer")
        ekey = self.params.get("expected_key", okey)
        cs = self.params.get("case_sensitive", False)
        actual = _norm(get_path(execution.output, okey), cs, True)
        expected = _norm(get_path(case.expected_output, ekey), cs, True)
        if expected is None:
            return self.not_applicable(f"no expected value at '{ekey}'")
        ok = actual == expected
        return self.make_result(
            score=1.0 if ok else 0.0,
            passed=ok,
            reason="match" if ok else f"expected {expected!r}, got {actual!r}",
        )


@register_evaluator("required_fields")
class RequiredFields(BaseEvaluator):
    """params: fields=[..] (dotted paths allowed). Falls back to expected_output.required_fields."""

    def evaluate(self, case, execution):
        fields = self.params.get("fields") or case.expected_output.get("required_fields")
        if not fields:
            return self.not_applicable("no required fields configured")
        missing = [f for f in fields if is_empty(get_path(execution.output, f))]
        score = (len(fields) - len(missing)) / len(fields)
        return self.make_result(
            score=score,
            passed=not missing,
            reason="all fields present" if not missing else f"missing/empty: {missing}",
            metadata={"missing": missing},
        )


@register_evaluator("json_schema")
class JsonSchemaValidation(BaseEvaluator):
    """params: schema={...}. Falls back to expected_output.schema."""

    def evaluate(self, case, execution):
        schema = self.params.get("schema") or case.expected_output.get("schema")
        if not schema:
            return self.not_applicable("no schema configured")
        errors = sorted(
            Draft202012Validator(schema).iter_errors(execution.output), key=lambda e: list(e.path)
        )
        msgs = [f"{'.'.join(map(str, e.path)) or '$'}: {e.message}" for e in errors[:5]]
        return self.make_result(
            score=0.0 if errors else 1.0,
            passed=not errors,
            reason="valid" if not errors else "; ".join(msgs),
            metadata={"error_count": len(errors)},
        )
