import time
from typing import Protocol

import httpx

from app.schemas import EvaluationCase, ExecutionResult


class SystemAdapter(Protocol):
    """Connects the engine to an AI system. Implement `run` to onboard a new system."""

    def run(self, case: EvaluationCase) -> ExecutionResult: ...


class HTTPSystemAdapter:
    """POSTs case.input as JSON to the system and wraps the JSON response.
    Reads optional keys: retrieved|sources, usage.{input_tokens,output_tokens}, cost_usd, trace_id."""

    def __init__(self, url: str, timeout: float = 120.0, headers: dict | None = None):
        self.url, self.timeout, self.headers = url, timeout, headers or {}

    def run(self, case: EvaluationCase) -> ExecutionResult:
        start = time.perf_counter()
        r = httpx.post(self.url, json=case.input, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        latency = (time.perf_counter() - start) * 1000
        data = r.json()
        usage = data.get("usage") or {}
        return ExecutionResult(
            output=data,
            retrieved=data.get("retrieved") or data.get("sources") or [],
            latency_ms=latency,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=data.get("cost_usd"),
            trace_id=data.get("trace_id"),
        )
