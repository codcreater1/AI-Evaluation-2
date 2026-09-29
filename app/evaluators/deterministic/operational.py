from app.evaluators.base import BaseEvaluator, register_evaluator


class _Threshold(BaseEvaluator):
    param = ""
    unit = ""

    def _value(self, execution):
        raise NotImplementedError

    def evaluate(self, case, execution):
        limit = self.params.get(self.param)
        if limit is None:
            return self.not_applicable(f"param '{self.param}' not configured")
        value = self._value(execution)
        if value is None:
            return self.make_result(
                score=0.0, passed=False, label="missing_data", reason="value not reported"
            )
        ok = value <= limit
        return self.make_result(
            score=1.0 if ok else 0.0,
            passed=ok,
            reason=f"{value} {self.unit} {'<=' if ok else '>'} limit {limit} {self.unit}",
            metadata={"value": value, "limit": limit},
        )


@register_evaluator("latency_threshold")
class LatencyThreshold(_Threshold):
    """params: max_ms"""

    param, unit = "max_ms", "ms"

    def _value(self, execution):
        return execution.latency_ms


@register_evaluator("cost_threshold")
class CostThreshold(_Threshold):
    """params: max_usd"""

    param, unit = "max_usd", "USD"

    def _value(self, execution):
        return execution.cost_usd


@register_evaluator("token_threshold")
class TokenThreshold(_Threshold):
    """params: max_tokens (input + output)"""

    param, unit = "max_tokens", "tokens"

    def _value(self, execution):
        return execution.total_tokens
