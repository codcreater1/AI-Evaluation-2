from app.config.loader import RunConfig
from app.engine.aggregation import aggregate, check_thresholds, compare_aggregates
from app.engine.classification import classification_metrics
from app.engine.runner import EvaluationRunner
from app.evaluators import BaseEvaluator, create_evaluator
from app.evaluators.base import register_evaluator
from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult


def _cases():
    return [
        EvaluationCase(id=f"c{i}", system="s", expected_output={"answer": a})
        for i, a in enumerate(["a", "b", "c", "d"])
    ]


def _execs(answers):
    return {f"c{i}": ExecutionResult(output={"answer": a}) for i, a in enumerate(answers)}


def test_runner_end_to_end_with_thresholds():
    runner = EvaluationRunner([create_evaluator("exact_match")], thresholds={"exact_match": 0.9})
    rep = runner.run(_cases(), _execs(["a", "b", "c", "WRONG"]))
    assert rep.aggregates["exact_match"].mean_score == 0.75
    assert rep.passed is False and "exact_match" in rep.threshold_failures[0]
    assert [c.case_id for c in rep.failed_cases] == ["c3"]


def test_runner_passes_when_above_threshold():
    runner = EvaluationRunner([create_evaluator("exact_match")], thresholds={"exact_match": 0.9})
    assert runner.run(_cases(), _execs(["a", "b", "c", "d"])).passed is True


def test_broken_evaluator_is_isolated():
    @register_evaluator("boom_test")
    class Boom(BaseEvaluator):
        def evaluate(self, case, execution):
            raise RuntimeError("kaboom")

    runner = EvaluationRunner([create_evaluator("boom_test"), create_evaluator("exact_match")])
    rep = runner.run(_cases()[:1], _execs(["a"]))
    res = {r.evaluator: r for r in rep.cases[0].results}
    assert res["boom_test"].label == "error" and res["boom_test"].passed is False
    assert res["exact_match"].passed is True
    assert rep.aggregates["boom_test"].n_errors == 1


def test_adapter_failure_becomes_execution_error():
    class Bad:
        def run(self, case):
            raise ConnectionError("down")

    runner = EvaluationRunner([create_evaluator("exact_match")], adapter=Bad())
    rep = runner.run(_cases()[:1])
    assert rep.cases[0].error and rep.cases[0].results[0].label == "execution_error"


def test_adapter_and_parallel():
    class Echo:
        def run(self, case):
            return ExecutionResult(output={"answer": case.expected_output["answer"]})

    runner = EvaluationRunner([create_evaluator("exact_match")], adapter=Echo(), max_workers=4)
    rep = runner.run(_cases())
    assert rep.aggregates["exact_match"].mean_score == 1.0
    assert [c.case_id for c in rep.cases] == ["c0", "c1", "c2", "c3"]  # order preserved


def test_sink_receives_scores():
    got = []

    class Sink:
        def on_result(self, run_id, case, execution, result):
            got.append((case.id, result.evaluator))

        def flush(self):
            got.append("flushed")

    EvaluationRunner([create_evaluator("exact_match")], sinks=[Sink()]).run(
        _cases()[:2], _execs(["a", "b"])
    )
    assert got == [("c0", "exact_match"), ("c1", "exact_match"), "flushed"]


def test_not_applicable_excluded_from_mean():
    rs = [EvaluationResult(evaluator="e", score=1.0, passed=True),
          EvaluationResult(evaluator="e", label="not_applicable")]
    a = aggregate(rs)["e"]
    assert a.n == 2 and a.n_scored == 1 and a.mean_score == 1.0


def test_compare_aggregates_detects_regression():
    b = aggregate([EvaluationResult(evaluator="e", score=0.924, passed=True)])
    c = aggregate([EvaluationResult(evaluator="e", score=0.867, passed=False)])
    cmp = compare_aggregates(b, c, max_drop=0.02)
    assert cmp["passed"] is False and cmp["regressions"] == ["e"]


def test_classification_metrics_and_confusion_labels():
    pairs = [("eligible", "eligible"), ("eligible", "not_eligible"),
             ("not_eligible", "eligible"), ("not_eligible", "not_eligible"),
             ("eligible", "eligible")]
    m = classification_metrics(pairs, "eligible")
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == 0.6
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["f1"] - 2 / 3) < 1e-9


def test_runner_classification_from_config():
    cfg = RunConfig.model_validate({
        "system": "internship", "evaluators": ["exact_match"],
        "classification": {"positive_label": "eligible"},
    })
    runner = EvaluationRunner.from_config(cfg)
    cases = [EvaluationCase(id="1", system="internship", expected_output={"decision": "eligible"}),
             EvaluationCase(id="2", system="internship", expected_output={"decision": "eligible"})]
    ex = {"1": ExecutionResult(output={"decision": "eligible"}),
          "2": ExecutionResult(output={"decision": "not_eligible"})}
    rep = runner.run(cases, ex)
    assert rep.classification["tp"] == 1 and rep.classification["fn"] == 1
    labels = [r.label for c in rep.cases for r in c.results if r.evaluator == "confusion_matrix"]
    assert labels == ["TP", "FN"]
    assert check_thresholds(rep.aggregates, {}) == []
