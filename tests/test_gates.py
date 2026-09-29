from app.experiments.gates import QualityGates, evaluate_gates, load_gates, render_gate_report


def gate(gates, cand, base=None, cops=None, bops=None, newly=None):
    return evaluate_gates(QualityGates(**gates), cand, base, cops or {}, bops, newly)


def test_minimum_pass_and_fail():
    r = gate({"minimum": {"a": 0.9, "b": 0.9}}, {"a": 0.95, "b": 0.85})
    assert not r.passed and [c.name for c in r.failures] == ["b"]
    assert gate({"minimum": {"a": 0.9}}, {"a": 0.9}).passed


def test_missing_metric_fails_minimum():
    assert not gate({"minimum": {"a": 0.9}}, {}).passed


def test_max_drop_uses_default_and_per_metric_override():
    g = {"max_drop": {"default": 0.02, "b": 0.10}}
    r = gate(g, {"a": 0.85, "b": 0.85}, {"a": 0.90, "b": 0.90})
    by = {c.name: c.status for c in r.checks}
    assert by == {"b": "pass", "a": "fail"}  # a dropped 5 pts > 2, b dropped 5 pts <= 10


def test_pdf_example_92_4_to_86_7_blocks_deployment():
    r = gate({"max_drop": {"answer_correctness": 0.02}},
             {"answer_correctness": 0.867}, {"answer_correctness": 0.924})
    assert not r.passed
    assert "92.4% -> 86.7% (-5.7 pts)" in r.failures[0].message


def test_no_baseline_skips_regression_checks_but_keeps_minimum():
    r = gate({"minimum": {"a": 0.9}, "max_drop": {"default": 0.02}, "max_newly_failing_cases": 0},
             {"a": 0.95}, None)
    assert r.passed
    assert {c.status for c in r.checks if c.kind != "minimum"} == {"skipped"}


def test_operational_increase_and_newly_failing():
    r = gate({"max_increase_pct": {"latency_ms": 25, "cost_usd": 25}, "max_newly_failing_cases": 1},
             {}, {}, {"latency_ms": 2000, "cost_usd": 0.004}, {"latency_ms": 1800, "cost_usd": 0.003},
             newly=2)
    by = {c.name: c.status for c in r.checks}
    assert by == {"latency_ms": "pass", "cost_usd": "fail", "newly_failing_cases": "fail"}


def test_load_gates_yaml_and_render(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("gates:\n  minimum:\n    a: 0.9\n")
    assert load_gates(f).minimum == {"a": 0.9}
    cmp = {"candidate": {"name": "c", "dataset_ref": "d-v1"}, "baseline": None,
           "cases": {"newly_failing": []},
           "gate": gate({"minimum": {"a": 0.9}}, {"a": 0.5}).model_dump()}
    text = render_gate_report(cmp)
    assert text.startswith("FAIL") and "Deployment should be blocked." in text
