import pytest

from app.schemas import EvaluationCase, ExecutionResult


class FakeLLM:
    def __init__(self, reply='{"score": 0.9, "reason": "ok"}', model="fake-judge"):
        self.reply, self.model, self.calls = reply, model, []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture
def rag_case():
    return EvaluationCase(
        id="rag-001", system="ata-rag",
        input={"question": "How long can an internship last?"},
        expected_output={"answer": "Up to 6 months", "relevant_doc_ids": ["d1", "d2"],
                         "expected_documents": ["d1"]},
    )


@pytest.fixture
def rag_exec():
    return ExecutionResult(
        output={"answer": "up to  6 MONTHS", "citations": ["d1"]},
        retrieved=[{"id": "d1", "text": "Internship max 6 months"}, {"id": "x"}, {"id": "d3"}],
        latency_ms=1200, input_tokens=100, output_tokens=50, cost_usd=0.002, trace_id="t-1",
    )


# ---------- shared fixtures for API-level tests ----------
import json as _json  # noqa: E402

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.evaluators  # noqa: E402, F401
from app.api.deps import get_langfuse  # noqa: E402
from app.api.main import app as fastapi_app  # noqa: E402
from app.db import models  # noqa: E402, F401
from app.db.session import Base, get_db  # noqa: E402
from app.integrations.langfuse_client import LangfuseClient  # noqa: E402


class LangfuseRecorder:
    """Fake Langfuse server: records every request, answers 200."""

    def __init__(self, fail_paths: tuple[str, ...] = ()):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.fail_paths = fail_paths
        self.client = LangfuseClient(
            "https://lf.test", "pk", "sk", transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if any(p in request.url.path for p in self.fail_paths):
            return httpx.Response(500, text="boom")
        if request.url.path.startswith("/api/public/traces/"):
            return httpx.Response(200, json={"id": "t-prod", "input": {"question": "Q from prod?"}})
        if request.url.path == "/api/public/ingestion":
            return httpx.Response(207, json={"successes": [], "errors": []})
        return httpx.Response(200, json={})

    def paths(self, path: str) -> list[dict]:
        return [b for _, p, b in self.requests if p == path]


@pytest.fixture
def lf():
    return LangfuseRecorder()


@pytest.fixture
def api(lf):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db():
        with Session() as s:
            yield s

    fastapi_app.dependency_overrides[get_db] = override_db
    fastapi_app.dependency_overrides[get_langfuse] = lambda: lf.client
    client = TestClient(fastapi_app)
    client.session_factory = Session  # for tests that touch the ORM directly
    yield client
    fastapi_app.dependency_overrides.clear()
