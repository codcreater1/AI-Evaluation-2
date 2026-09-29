import pytest

from app.datasets.service import content_hash, make_ref, parse_ref
from app.db.models import DatasetCase, DatasetVersion, ImmutableDatasetError
from app.schemas import EvaluationCase


def case(i, q="q", system="ata-rag"):
    return {"id": f"rag-{i:03d}", "system": system, "input": {"question": q},
            "expected_output": {"answer": f"a{i}"}}


def create(api, n=3, name="ata-rag-golden"):
    return api.post("/datasets", json={"name": name, "system": "ata-rag",
                                       "cases": [case(i) for i in range(1, n + 1)]})


def test_ref_parsing():
    assert parse_ref("ata-rag-golden-v3") == ("ata-rag-golden", 3)
    assert parse_ref("ata-rag-golden") == ("ata-rag-golden", None)
    assert make_ref("x", 2) == "x-v2"


def test_hash_is_order_independent_and_content_sensitive():
    a, b = EvaluationCase(id="1", system="s"), EvaluationCase(id="2", system="s")
    assert content_hash([a, b]) == content_hash([b, a])
    assert content_hash([a]) != content_hash([a.model_copy(update={"input": {"x": 1}})])


def test_create_and_read(api):
    r = create(api)
    assert r.status_code == 201 and r.json()["ref"] == "ata-rag-golden-v1"
    d = api.get("/datasets/ata-rag-golden").json()
    assert d["latest_version"] == 1 and d["versions"][0]["n_cases"] == 3
    v = api.get("/datasets/ata-rag-golden/versions/latest").json()
    assert [c["id"] for c in v["cases"]] == ["rag-001", "rag-002", "rag-003"]
    assert [x["name"] for x in api.get("/datasets").json()] == ["ata-rag-golden"]


def test_validation(api):
    assert create(api, name="bad-name-v2").status_code == 422  # version suffix reserved
    assert create(api).status_code == 201
    assert create(api).status_code == 409  # already exists
    dup = api.post("/datasets", json={"name": "d2", "system": "ata-rag",
                                      "cases": [case(1), case(1)]})
    assert dup.status_code == 422
    other = api.post("/datasets", json={"name": "d3", "system": "ata-rag",
                                        "cases": [case(1, system="internship")]})
    assert other.status_code == 422


def test_changes_create_new_versions_and_keep_old_untouched(api):
    create(api)
    v2 = api.post("/datasets/ata-rag-golden/versions", json={
        "add_cases": [case(4)], "update_cases": [case(1, q="changed")],
        "remove_case_ids": ["rag-003"], "note": "fixes"})
    assert v2.status_code == 201
    assert v2.json()["ref"] == "ata-rag-golden-v2" and v2.json()["parent_version"] == 1
    v1 = api.get("/datasets/ata-rag-golden/versions/1").json()
    assert [c["id"] for c in v1["cases"]] == ["rag-001", "rag-002", "rag-003"]
    assert v1["cases"][0]["input"]["question"] == "q"  # v1 unchanged
    v2d = api.get("/datasets/ata-rag-golden/versions/2").json()
    assert {c["id"] for c in v2d["cases"]} == {"rag-001", "rag-002", "rag-004"}  # stable ids
    assert v1["content_hash"] != v2d["content_hash"]


def test_version_errors(api):
    create(api)
    p = "/datasets/ata-rag-golden/versions"
    assert api.post(p, json={"add_cases": [case(1)]}).status_code == 409  # exists
    assert api.post(p, json={"update_cases": [case(9)]}).status_code == 404
    assert api.post(p, json={"remove_case_ids": ["nope"]}).status_code == 404
    assert api.post(p, json={}).status_code == 409  # no changes
    assert api.post(p, json={"cases": [case(1)], "add_cases": [case(2)]}).status_code == 422
    assert api.get("/datasets/nope").status_code == 404
    assert api.get("/datasets/ata-rag-golden/versions/9").status_code == 404
    assert api.get("/datasets/ata-rag-golden/versions/abc").status_code == 422


def test_orm_blocks_mutation_of_existing_versions(api):
    create(api)
    with api.session_factory() as db:
        c = db.query(DatasetCase).first()
        c.payload = {"tampered": True}
        with pytest.raises(ImmutableDatasetError):
            db.commit()
        db.rollback()
        v = db.query(DatasetVersion).first()
        v.note = "edited"
        with pytest.raises(ImmutableDatasetError):
            db.commit()
        db.rollback()
        db.delete(db.query(DatasetVersion).first())
        with pytest.raises(ImmutableDatasetError):
            db.commit()


def test_sync_to_langfuse_creates_versioned_dataset_and_items(api, lf):
    create(api)
    r = api.post("/datasets/ata-rag-golden/versions/1/sync-langfuse")
    assert r.status_code == 200 and r.json()["langfuse_synced_at"]
    assert lf.paths("/api/public/v2/datasets")[0]["name"] == "ata-rag-golden-v1"
    items = lf.paths("/api/public/dataset-items")
    assert {i["id"] for i in items} == {f"ata-rag-golden-v1:rag-00{n}" for n in (1, 2, 3)}
    n_before = len(lf.requests)
    api.post("/datasets/ata-rag-golden/versions/1/sync-langfuse")  # idempotent
    assert len(lf.requests) == n_before


def test_production_trace_becomes_new_case_version(api, lf):
    create(api)
    r = api.post("/datasets/ata-rag-golden/versions/from-trace", json={
        "trace_id": "t-prod", "case_id": "prod-001", "expected_output": {"answer": "fixed answer"}})
    assert r.status_code == 201 and r.json()["version"] == 2
    new = {c["id"]: c for c in api.get("/datasets/ata-rag-golden/versions/2").json()["cases"]}
    assert new["prod-001"]["input"] == {"question": "Q from prod?"}
    assert new["prod-001"]["metadata"]["source_trace_id"] == "t-prod"
    assert len(new) == 4
