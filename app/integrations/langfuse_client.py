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
import time
import uuid
from collections.abc import Callable
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


class LangfuseRateLimited(LangfuseError):
    """HTTP 429. Langfuse Cloud Hobby allows only ~30 'general API' requests per minute."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LangfuseClient:
    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sleep = sleep
        self._scores_path: str | None = None  # read endpoint that works (remembered)
        self.read_cache: dict = {}  # used by the dashboard to avoid burning the rate limit
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
    @staticmethod
    def _retry_after(r: httpx.Response) -> float:
        try:
            return float(r.headers["retry-after"]) + 1
        except (KeyError, ValueError):
            pass
        try:
            return float(r.json()["details"]["retryAfterSeconds"]) + 1
        except Exception:  # noqa: BLE001
            return 6.0

    def _request(self, method: str, path: str, json: dict | None = None,
                 params: dict | None = None, timeout: float | None = None,
                 max_retries: int = 3) -> dict:
        """On HTTP 429 waits for Retry-After and retries (max_retries=0: fail fast, used for reads)."""
        kwargs = {"timeout": timeout} if timeout else {}
        attempt = 0
        while True:
            try:
                r = self._http.request(method, path, json=json, params=params, **kwargs)
            except httpx.HTTPError as exc:
                raise LangfuseError(f"{method} {path} failed: {exc}") from exc
            if r.status_code == 429:
                wait = self._retry_after(r)
                if attempt >= max_retries or wait > 70:
                    raise LangfuseRateLimited(
                        f"Langfuse rate limit reached ({method} {path}); retry in ~{wait:.0f}s", wait)
                self._sleep(wait)
                attempt += 1
                continue
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
    @staticmethod
    def _event(kind: str, body: dict) -> dict:
        return {"id": str(uuid.uuid4()), "type": kind, "timestamp": _now(), "body": body}

    def trace_events(
        self, trace_id: str, name: str, input: Any, output: Any, metadata: dict,
        tags: list[str], steps: list[dict] | None = None,
    ) -> list[dict]:
        """steps: [{"type": "span"|"generation", "name":.., "input":.., "output":.., ...extra body}]"""
        ts = _now()
        events = [self._event("trace-create", {
            "id": trace_id, "name": name, "input": input, "output": output,
            "metadata": metadata, "tags": tags, "timestamp": ts})]
        for step in steps or []:
            body = {k: v for k, v in step.items() if k != "type"}
            body.update({"id": str(uuid.uuid4()), "traceId": trace_id, "startTime": ts})
            events.append(self._event(f"{step['type']}-create", body))
        return events

    def score_event(self, trace_id: str, name: str, value: float | str, comment: str = "",
                    metadata: dict | None = None, score_id: str | None = None) -> dict:
        return self._event("score-create", {
            "id": score_id or str(uuid.uuid4()), "traceId": trace_id, "name": name, "value": value,
            "dataType": "CATEGORICAL" if isinstance(value, str) else "NUMERIC",
            "comment": _clip(comment, 500), "metadata": metadata or {}})

    def ingest(self, events: list[dict], chunk: int = 50) -> int:
        """Sends events in batches. Ingestion has a much higher rate limit than the 'general' API
        (1000 batches/min), so traces AND scores go through here."""
        errors: list = []
        for i in range(0, len(events), chunk):
            res = self._request("POST", "/api/public/ingestion", {"batch": events[i:i + chunk]})
            errors += res.get("errors") or []
        if errors:
            raise LangfuseError(f"{len(errors)} ingestion event(s) rejected, e.g. {errors[0]}")
        return len(events)

    def create_trace(
        self, trace_id: str, name: str, input: Any, output: Any, metadata: dict,
        tags: list[str], steps: list[dict] | None = None,
    ) -> int:
        return self.ingest(self.trace_events(trace_id, name, input, output, metadata, tags, steps))

    def get_trace(self, trace_id: str) -> dict:
        return self._request("GET", f"/api/public/traces/{trace_id}")

    # ---- reads (used by the dashboard). Langfuse changed its read APIs (scores v3 on the new
    # platform, v2/legacy on older ones), so we try the newest first and fall back.
    def project_name(self) -> str:
        data = self._request("GET", "/api/public/projects", timeout=8, max_retries=0).get("data") or []
        return data[0].get("name", "?") if data else "?"

    def list_scores(self, limit: int = 20, trace_ids: list[str] | None = None) -> list[dict]:
        paths = ["/api/public/v3/scores", "/api/public/v2/scores", "/api/public/scores"]
        if self._scores_path in paths:
            paths.remove(self._scores_path)
            paths.insert(0, self._scores_path)
        last: LangfuseError | None = None
        for path in paths:
            params: dict[str, Any] = {"limit": limit}
            if trace_ids:
                params["traceId"] = ",".join(trace_ids)
            if path.endswith("/v3/scores"):
                params["fields"] = "details,subject"
            try:
                rows = self._request("GET", path, params=params, timeout=8,
                                     max_retries=0).get("data") or []
            except LangfuseRateLimited:
                raise  # other endpoints share the same bucket, do not waste more requests
            except LangfuseError as exc:
                last = exc
                continue
            self._scores_path = path
            out = []
            for r in rows:
                subj = r.get("subject") or {}
                trace_id = (r.get("traceId") or subj.get("traceId") or subj.get("trace_id")
                            or (subj.get("id") if subj.get("kind") == "trace" else None))
                out.append({"name": r.get("name"), "value": r.get("value"),
                            "data_type": r.get("dataType"), "trace_id": trace_id,
                            "timestamp": r.get("timestamp") or r.get("createdAt")})
            return out
        raise last or LangfuseError("could not read scores")

    def list_datasets(self) -> list[dict]:
        last: LangfuseError | None = None
        for path in ("/api/public/v2/datasets", "/api/public/datasets"):
            try:
                return self._request("GET", path, params={"limit": 50}, timeout=8,
                                     max_retries=0).get("data") or []
            except LangfuseRateLimited:
                raise
            except LangfuseError as exc:
                last = exc
        raise last or LangfuseError("could not read datasets")

    def trace_url(self, trace_id: str) -> str:
        return f"{self.host}/trace/{trace_id}"


def clip_docs(docs: list[dict], max_docs: int = 20) -> list[dict]:
    return [{k: _clip(v, 500) for k, v in d.items()} for d in docs[:max_docs]]
