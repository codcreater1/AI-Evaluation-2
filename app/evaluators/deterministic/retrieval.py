from app.evaluators.base import BaseEvaluator, register_evaluator
from app.evaluators.utils import doc_keys


class _RetrievalBase(BaseEvaluator):
    def _prepare(self, case, execution):
        expected = case.expected_output.get(self.params.get("expected_key", "relevant_doc_ids"))
        if not expected:
            return None
        k = int(self.params.get("k", 5))
        top_k = execution.retrieved[:k]
        expected = {str(e) for e in expected}
        hits_per_doc = [bool(doc_keys(d) & expected) for d in top_k]
        found = {e for e in expected if any(e in doc_keys(d) for d in top_k)}
        return expected, k, top_k, hits_per_doc, found


@register_evaluator("retrieval_recall_at_k")
class RecallAtK(_RetrievalBase):
    """Recall@K = relevant docs found in top-K / all relevant docs. params: k=5, min_score=0.5"""

    def evaluate(self, case, execution):
        prep = self._prepare(case, execution)
        if prep is None:
            return self.not_applicable("no relevant_doc_ids (e.g. unanswerable question)")
        expected, k, _, _, found = prep
        score = len(found) / len(expected)
        return self.make_result(
            score=score,
            passed=score >= self.params.get("min_score", 0.5),
            reason=f"Recall@{k} = {len(found)}/{len(expected)}",
            metadata={"k": k, "missing": sorted(expected - found)},
        )


@register_evaluator("retrieval_precision_at_k")
class PrecisionAtK(_RetrievalBase):
    """Precision@K = relevant docs in top-K / docs returned in top-K. params: k=5, min_score=0.3"""

    def evaluate(self, case, execution):
        prep = self._prepare(case, execution)
        if prep is None:
            return self.not_applicable("no relevant_doc_ids (e.g. unanswerable question)")
        _, k, top_k, hits, _ = prep
        score = (sum(hits) / len(top_k)) if top_k else 0.0
        return self.make_result(
            score=score,
            passed=score >= self.params.get("min_score", 0.3),
            reason=f"Precision@{k} = {sum(hits)}/{len(top_k)}",
            metadata={"k": k},
        )


@register_evaluator("expected_document_present")
class ExpectedDocumentPresent(BaseEvaluator):
    """All expected_output.expected_documents must appear in retrieved and/or citations.
    params: source='retrieved'|'citations'|'either'(default), citations_key='citations'"""

    def evaluate(self, case, execution):
        expected = case.expected_output.get("expected_documents")
        if not expected:
            return self.not_applicable("no expected_documents")
        source = self.params.get("source", "either")
        pool: set[str] = set()
        if source in ("retrieved", "either"):
            for d in execution.retrieved:
                pool |= doc_keys(d)
        if source in ("citations", "either"):
            for c in execution.output.get(self.params.get("citations_key", "citations")) or []:
                pool |= doc_keys(c)
        missing = [e for e in expected if str(e) not in pool]
        return self.make_result(
            score=(len(expected) - len(missing)) / len(expected),
            passed=not missing,
            reason="all expected documents present" if not missing else f"missing: {missing}",
            metadata={"missing": missing},
        )
