from collections.abc import Iterable


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def confusion_label(expected: str, predicted: str, positive: str) -> str:
    e, p = expected == positive, predicted == positive
    return "TP" if e and p else "TN" if not e and not p else "FP" if p else "FN"


def classification_metrics(pairs: Iterable[tuple[str, str]], positive: str) -> dict:
    """pairs = (expected, predicted). Returns confusion matrix + accuracy/precision/recall/F1.
    FP = system approved/flagged positive but should not have; FN = missed a real positive."""
    tp = tn = fp = fn = 0
    for exp, pred in pairs:
        lab = confusion_label(exp, pred, positive)
        tp += lab == "TP"
        tn += lab == "TN"
        fp += lab == "FP"
        fn += lab == "FN"
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall) if precision and recall else (
        0.0 if precision is not None and recall is not None else None
    )
    return {
        "positive_label": positive,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
