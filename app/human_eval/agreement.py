"""Pure math: how well does an evaluator (typically an LLM judge) agree with a human reviewer,
given matched (judge, human) pairs on the same (run, case, evaluator) triple. No DB/HTTP here so
it is trivial to unit test with plain tuples."""
from dataclasses import dataclass, field


@dataclass
class Pair:
    case_id: str
    judge_score: float | None
    judge_passed: bool | None
    human_score: float | None
    human_passed: bool | None


@dataclass
class AgreementReport:
    n: int
    n_scored: int  # pairs where both sides have a numeric score
    n_pass_fail: int  # pairs where both sides have a pass/fail verdict
    pass_agreement_rate: float | None
    cohens_kappa: float | None
    mean_abs_score_diff: float | None
    confusion: dict[str, int]  # {both_pass, both_fail, human_pass_judge_fail, human_fail_judge_pass}
    disagreements: list[dict] = field(default_factory=list)  # worst-diff cases, for the reviewer to look at

    def as_dict(self) -> dict:
        return {
            "n": self.n, "n_scored": self.n_scored, "n_pass_fail": self.n_pass_fail,
            "pass_agreement_rate": self.pass_agreement_rate, "cohens_kappa": self.cohens_kappa,
            "mean_abs_score_diff": self.mean_abs_score_diff, "confusion": self.confusion,
            "disagreements": self.disagreements,
        }


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Chance-corrected agreement on a binary (pass/fail) label. None if undefined (no pairs,
    or every pair falls in the same category so chance agreement is 100%)."""
    n = len(pairs)
    if n == 0:
        return None
    both_true = sum(1 for a, b in pairs if a and b)
    both_false = sum(1 for a, b in pairs if not a and not b)
    po = (both_true + both_false) / n
    p_a_true = sum(1 for a, _ in pairs if a) / n
    p_b_true = sum(1 for _, b in pairs if b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def compute_agreement(pairs: list[Pair], top_disagreements: int = 10) -> AgreementReport:
    scored = [p for p in pairs if p.judge_score is not None and p.human_score is not None]
    pf = [p for p in pairs if p.judge_passed is not None and p.human_passed is not None]

    confusion = {"both_pass": 0, "both_fail": 0, "human_pass_judge_fail": 0, "human_fail_judge_pass": 0}
    for p in pf:
        if p.human_passed and p.judge_passed:
            confusion["both_pass"] += 1
        elif not p.human_passed and not p.judge_passed:
            confusion["both_fail"] += 1
        elif p.human_passed and not p.judge_passed:
            confusion["human_pass_judge_fail"] += 1
        else:
            confusion["human_fail_judge_pass"] += 1

    pass_agreement = ((confusion["both_pass"] + confusion["both_fail"]) / len(pf)) if pf else None
    kappa = cohens_kappa([(p.human_passed, p.judge_passed) for p in pf]) if pf else None
    mean_abs_diff = (sum(abs(p.judge_score - p.human_score) for p in scored) / len(scored)) if scored else None

    worst = sorted(scored, key=lambda p: abs(p.judge_score - p.human_score), reverse=True)[:top_disagreements]
    disagreements = [
        {"case_id": p.case_id, "judge_score": p.judge_score, "human_score": p.human_score,
         "diff": abs(p.judge_score - p.human_score)}
        for p in worst if abs(p.judge_score - p.human_score) > 0
    ]

    return AgreementReport(
        n=len(pairs), n_scored=len(scored), n_pass_fail=len(pf),
        pass_agreement_rate=pass_agreement, cohens_kappa=kappa, mean_abs_score_diff=mean_abs_diff,
        confusion=confusion, disagreements=disagreements,
    )
