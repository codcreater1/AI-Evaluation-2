from app.human_eval.agreement import Pair, cohens_kappa, compute_agreement


def test_perfect_agreement():
    # mixed pass/fail so kappa's chance-correction is meaningful (not the degenerate single-class case)
    pairs = [Pair(f"c{i}", 1.0, True, 1.0, True) for i in range(3)] + \
            [Pair(f"f{i}", 0.0, False, 0.0, False) for i in range(2)]
    r = compute_agreement(pairs)
    assert r.n == 5 and r.n_scored == 5 and r.n_pass_fail == 5
    assert r.pass_agreement_rate == 1.0
    assert r.cohens_kappa == 1.0
    assert r.mean_abs_score_diff == 0.0
    assert r.confusion == {"both_pass": 3, "both_fail": 2, "human_pass_judge_fail": 0, "human_fail_judge_pass": 0}
    assert r.disagreements == []


def test_full_disagreement_gives_negative_kappa():
    # both raters use both labels (50/50), but always on opposite cases -> systematic disagreement
    pairs = [Pair(f"a{i}", 1.0, True, 0.0, False) for i in range(2)] + \
            [Pair(f"b{i}", 0.0, False, 1.0, True) for i in range(2)]
    r = compute_agreement(pairs)
    assert r.pass_agreement_rate == 0.0
    assert r.cohens_kappa == -1.0
    assert r.confusion["human_fail_judge_pass"] == 2 and r.confusion["human_pass_judge_fail"] == 2


def test_mean_abs_score_diff_and_worst_disagreements():
    pairs = [
        Pair("c1", 1.0, True, 1.0, True),
        Pair("c2", 0.8, True, 0.5, True),
        Pair("c3", 0.9, True, 0.1, False),
    ]
    r = compute_agreement(pairs)
    assert abs(r.mean_abs_score_diff - ((0 + 0.3 + 0.8) / 3)) < 1e-9
    assert r.disagreements[0]["case_id"] == "c3"
    assert abs(r.disagreements[0]["diff"] - 0.8) < 1e-9
    assert abs(r.disagreements[1]["diff"] - 0.3) < 1e-9


def test_kappa_with_class_imbalance_is_lower_than_raw_agreement():
    # 90% raw agreement, but almost everything is "pass" -> kappa should discount that
    pairs = [Pair(f"c{i}", 1.0, True, 1.0, True) for i in range(9)] + [Pair("c9", 0.0, False, 1.0, True)]
    r = compute_agreement(pairs)
    assert r.pass_agreement_rate == 0.9
    assert r.cohens_kappa is not None and r.cohens_kappa < 0.9


def test_kappa_undefined_when_no_variance():
    assert cohens_kappa([(True, True)] * 5) is None  # pe == 1 -> undefined, not a fake 0/1


def test_empty_inputs_are_handled():
    r = compute_agreement([])
    assert r.n == 0 and r.pass_agreement_rate is None and r.cohens_kappa is None
    assert r.mean_abs_score_diff is None and r.disagreements == []


def test_pairs_missing_one_side_are_excluded_from_that_metric():
    pairs = [
        Pair("c1", 1.0, True, None, None),  # no human score/pass at all
        Pair("c2", 0.7, True, 0.9, True),
        Pair("c3", None, None, 0.5, True),  # no judge score/pass at all
    ]
    r = compute_agreement(pairs)
    assert r.n == 3 and r.n_scored == 1 and r.n_pass_fail == 1
