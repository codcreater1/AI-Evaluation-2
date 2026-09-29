from app.evaluators.base import BaseEvaluator, register_evaluator
from app.evaluators.utils import doc_keys


@register_evaluator("citation_exists")
class CitationExists(BaseEvaluator):
    """Answer must cite sources, and each citation must point to a retrieved document.
    params: citations_key='citations', require_retrieved=True.
    If expected_output.answerable is False, citations are not required."""

    def evaluate(self, case, execution):
        cites = execution.output.get(self.params.get("citations_key", "citations")) or []
        if case.expected_output.get("answerable") is False:
            return self.not_applicable("unanswerable case: citations not required")
        if not cites:
            return self.make_result(score=0.0, passed=False, reason="no citations in answer")
        if not self.params.get("require_retrieved", True) or not execution.retrieved:
            return self.make_result(score=1.0, passed=True, reason=f"{len(cites)} citation(s)")
        pool: set[str] = set()
        for d in execution.retrieved:
            pool |= doc_keys(d)
        bad = [c for c in cites if not (doc_keys(c) & pool)]
        valid = len(cites) - len(bad)
        return self.make_result(
            score=valid / len(cites),
            passed=not bad,
            reason="all citations exist in retrieved docs" if not bad else f"not retrieved: {bad}",
            metadata={"invalid": bad},
        )
