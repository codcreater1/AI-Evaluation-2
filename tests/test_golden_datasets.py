import json
from pathlib import Path

import pytest

from app.schemas import EvaluationCase

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _load(name: str) -> list[dict]:
    return json.loads((DATASETS_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename,system,minimum", [
    ("ata_rag_golden.json", "ata-rag", 100),
    ("internship_golden.json", "internship-coordinator", 50),
])
def test_golden_dataset_meets_minimum_size_and_schema(filename, system, minimum):
    raw = _load(filename)
    assert len(raw) >= minimum, f"{filename} has {len(raw)} cases, need >= {minimum}"

    ids = [c["id"] for c in raw]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    cases = [EvaluationCase.model_validate(c) for c in raw]  # every case must be a valid EvaluationCase
    assert all(c.system == system for c in cases)
    assert all(c.input for c in cases), "every case needs a non-empty input"
    assert all(c.expected_output for c in cases), "every case needs a non-empty expected_output"
    assert all("source_note" in c["metadata"] for c in raw), "every case should record its provenance"


def test_ata_rag_dataset_answerable_cases_have_retrieval_ground_truth():
    cases = _load("ata_rag_golden.json")
    for c in cases:
        eo = c["expected_output"]
        if eo.get("answerable") is False:
            continue
        # answerable cases used for grounding (not the constructed edge cases) should carry
        # relevant_doc_ids so retrieval/citation evaluators have something to check
        if c["metadata"]["source_note"] in ("live_verified", "paraphrase_of_verified_fact"):
            assert eo.get("relevant_doc_ids"), c["id"]


def test_ata_rag_dataset_languages_cover_all_ui_locales():
    cases = _load("ata_rag_golden.json")
    langs = {c["metadata"]["language"] for c in cases}
    assert {"en", "pl", "uk"} <= langs


def test_internship_dataset_decisions_are_valid_and_all_represented():
    cases = _load("internship_golden.json")
    decisions = {c["expected_output"]["decision"] for c in cases}
    assert decisions == {"eligible", "not_eligible", "needs_review"}
    for c in cases:
        assert set(c["expected_output"]["required_fields"]) == {"decision", "explanation"}
        assert c["expected_output"]["explanation"]  # never empty


def test_internship_dataset_matches_examples_config_positive_label():
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "examples" / "internship.yaml")
                         .read_text())
    assert cfg["classification"]["positive_label"] == "eligible"
    cases = _load("internship_golden.json")
    assert any(c["expected_output"]["decision"] == "eligible" for c in cases)
