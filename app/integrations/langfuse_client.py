"""Small Langfuse REST client (public API). Using REST instead of the Python SDK keeps us
independent of SDK major versions and makes the integration easy to test with a mock transport.

Endpoints used:
  POST /api/public/v2/datasets        upsert dataset
  POST /api/public/dataset-items      upsert dataset item (by id)
  POST /api/public/dataset-run-items  link a trace to a dataset item inside an experiment run
  POST /api/public/scores             attach a score to a trace
  POST /api/public/ingestion          create traces / spans / generations
  GET  /api/public/traces/{id}        read a (production) trace
"""
import hashlib
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clip(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


class LangfuseError(RuntimeError):
    pass


class LangfuseClient:
    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.host = host.rstrip("/")
        self._http = httpx.Client(
            base_url=self.host, auth=(public_key, secret_key), timeout=timeout, transport=transport
        )

    @classmethod
    def from_env(cls) -> "LangfuseClient | None":
        pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            return None
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        return cls(host or "https://cloud.langfuse.com", pk, sk)

    # ---- low level
    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        r = self._http.request(method, path, json=json)
        if r.status_code >= 400:
            raise LangfuseError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    # ---- datasets
    def upsert_dataset(self, name: str, description: str = "", metadata: dict | None = None) -> dict:
        return self._request(
            "POST", "/api/public/v2/datasets",
            {"name": name, "description": description, "metadata": metadata or {}},
        )

    def upsert_dataset_item(
        self, dataset_name: str, item_id: str, input: Any, expected_output: Any, metadata: dict
    ) -> dict:
        return self._request(
            "POST", "/api/public/dataset-items",
            {"datasetName": dataset_name, "id": item_id, "input": input,
             "expectedOutput": expected_output, "metadata": metadata},
        )

    def link_run_item(
        self, run_name: str, dataset_item_id: str, trace_id: str,
        metadata: dict | None = None, description: str = "",
    ) -> dict:
        return self._request(
            "POST", "/api/public/dataset-run-items",
            {"runName": run_name, "runDescription": description, "datasetItemId": dataset_item_id,
             "traceId": trace_id, "metadata": metadata or {}},
        )

    # ---- scores
    def create_score(
        self, trace_id: str, name: str, value: float | str, comment: str = "",
        metadata: dict | None = None, score_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "traceId": trace_id, "name": name, "value": value,
            "dataType": "CATEGORICAL" if isinstance(value, str) else "NUMERIC",
            "comment": _clip(comment, 500), "metadata": metadata or {},
        }
        if score_id:
            body["id"] = score_id
        return self._request("POST", "/api/public/scores", body)

    @staticmethod
    def score_id(*parts: str) -> str:
        """Deterministic id -> re-sending the same score updates it instead of duplicating."""
        return hashlib.sha1("|".join(parts).encode()).hexdigest()

    # ---- traces
    def create_trace(
        self, trace_id: str, name: str, input: Any, output: Any, metadata: dict,
        tags: list[str], steps: list[dict] | None = None,
    ) -> dict:
        """steps: [{"type": "span"|"generation", "name":.., "input":.., "output":.., ...extra body}]"""
        ts = _now()
        batch = [{"id": str(uuid.uuid4()), "type": "trace-create", "timestamp": ts,
                  "body": {"id": trace_id, "name": name, "input": input, "output": output,
                           "metadata": metadata, "tags": tags, "timestamp": ts}}]
        for step in steps or []:
            body = {k: v for k, v in step.items() if k != "type"}
            body.update({"id": str(uuid.uuid4()), "traceId": trace_id, "startTime": ts})
            batch.append({"id": str(uuid.uuid4()), "type": f"{step['type']}-create",
                          "timestamp": ts, "body": body})
        res = self._request("POST", "/api/public/ingestion", {"batch": batch})
        if res.get("errors"):
            raise LangfuseError(f"ingestion errors: {res['errors'][:2]}")
        return res

    def get_trace(self, trace_id: str) -> dict:
        return self._request("GET", f"/api/public/traces/{trace_id}")

    def trace_url(self, trace_id: str) -> str:
        return f"{self.host}/trace/{trace_id}"


def clip_docs(docs: list[dict], max_docs: int = 20) -> list[dict]:
    return [{k: _clip(v, 500) for k, v in d.items()} for d in docs[:max_docs]]
