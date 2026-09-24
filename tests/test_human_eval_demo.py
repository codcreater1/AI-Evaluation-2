from app.human_eval import service
from app.human_eval.demo import N_CASES, run_demo


def test_demo_seeds_over_100_matched_pairs(api, lf):
    Session = api.session_factory
    with Session() as db:
        run_id = run_demo(db)
        db.commit()
    with Session() as db:
        report = service.agreement_report(db, run_id=run_id)
    assert N_CASES >= 100
    assert report.n == N_CASES
    assert report.n_pass_fail == N_CASES and report.n_scored == N_CASES
    # noisy but correlated data: mostly agrees, not suspiciously perfect, not random noise either
    assert 0.7 <= report.pass_agreement_rate < 1.0
    assert report.cohens_kappa is not None and report.cohens_kappa > 0.3
    assert 0 < report.mean_abs_score_diff < 0.2


def test_demo_is_reproducible_with_same_seed(api):
    Session = api.session_factory
    with Session() as db:
        r1 = run_demo(db, seed=7)
    with Session() as db:
        r2 = run_demo(db, seed=7)
    with Session() as db:
        rep1 = service.agreement_report(db, run_id=r1)
        rep2 = service.agreement_report(db, run_id=r2)
    assert rep1.as_dict() == rep2.as_dict()


def test_demo_via_dashboard_button(api, lf):
    r = api.post("/dashboard/human-eval/run-demo", follow_redirects=False)
    assert r.status_code == 303
    html = api.get(r.headers["location"]).text
    assert "demo-judge-agreement" in html or "120" in html
