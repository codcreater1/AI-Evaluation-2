import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.evaluators
from app.api.main import app
from app.db import models  # noqa: F401
from app.db.session import Base, get_db
from app.evaluators.llm_judge.client import set_default_client
from tests.conftest import FakeLLM

SAMPLE = json.loads((Path(__file__).parent.parent / "examples/sample_request.json").read_text())


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[get_db] = override
    yield TestClient(app)  # no `with`: skip lifespan (would connect to Postgres)
    app.dependency_overrides.clear()


def test_run_get_metrics_and_systems(client):
    r = client.post("/evaluations/run", json=SAMPLE)
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["passed"] is False  # recall 0.5 < threshold 0.8
    assert run["summary"]["threshold_failures"]
    assert run["meta"]["model"] == "demo-model"

    full = client.get(f"/evaluations/{run['id']}").json()
    assert {x["evaluator"] for x in full["results"]} == {
        "retrieval_recall_at_k", "citation_exists", "exact_match"}
    assert full["results"][0]["case_id"] == "rag-001"

    only_failed = client.get(f"/evaluations/{run['id']}?only_failed=true").json()
    assert [x["evaluator"] for x in only_failed["results"]] == ["retrieval_recall_at_k"]

    assert client.get("/systems").json()[0]["name"] == "ata-rag"  # auto-registered
    metrics = {m["evaluator"]: m for m in client.get("/metrics?system=ata-rag").json()}
    assert metrics["exact_match"]["mean_score"] == 1.0
    assert client.get("/evaluations?system=ata-rag").json()[0]["id"] == run["id"]


def test_not_found_and_validation(client):
    assert client.get("/evaluations/nope").status_code == 404
    bad = json.loads(json.dumps(SAMPLE))
    bad["config"]["evaluators"] = ["does_not_exist"]
    assert client.post("/evaluations/run", json=bad).status_code == 422
    no_exec = json.loads(json.dumps(SAMPLE))
    no_exec["items"][0].pop("execution")
    assert client.post("/evaluations/run", json=no_exec).status_code == 422


def test_register_system_and_list_evaluators(client):
    r = client.post("/systems", json={"name": "internship", "endpoint_url": "http://x/run"})
    assert r.status_code == 201
    assert "exact_match" in client.get("/evaluators").json()["evaluators"]


def test_llm_judge_via_api_is_mocked(client):
    set_default_client(FakeLLM('{"score": 1.0, "reason": "great"}', model="mock"))
    try:
        req = json.loads(json.dumps(SAMPLE))
        req["config"]["evaluators"] = ["answer_correctness"]
        req["config"]["thresholds"] = {"answer_correctness": 0.9}
        run = client.post("/evaluations/run", json=req).json()
        assert run["passed"] is True
        res = client.get(f"/evaluations/{run['id']}").json()["results"][0]
        assert res["meta"]["judge_model"] == "mock"
    finally:
        set_default_client(None)
