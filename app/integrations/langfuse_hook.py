"""ScoreSink implementation for Langfuse. The runner only knows the sink protocol
(on_execution / on_result / flush); everything Langfuse-specific lives here.

Rate limits: Langfuse Cloud Hobby allows ~30 'general API' requests/min, but ingestion is far
higher. So traces and scores are BUFFERED and sent as ingestion batches on flush(); only the
dataset-run link (one small request per case) uses the general API."""
import logging
import threading
import uuid
from typing import Protocol

from app.integrations.langfuse_client import LangfuseClient, clip_docs
from app.schemas import EvaluationCase, EvaluationResult, ExecutionResult

log = logging.getLogger(__name__)


class ScoreSink(Protocol):
    def on_result(
        self, run_id: str, case: EvaluationCase, execution: ExecutionResult | None,
        result: EvaluationResult,
    ) -> None: ...

    def flush(self) -> None: ...

    # optional: on_execution(run_id, case, execution) -> None


class LangfuseSink:
    """- ensures every executed case has a Langfuse trace (creates one when the AI system did not),
    - links the trace to its dataset item inside a named experiment run (if dataset_ref is given),
    - writes each evaluator result as a score on the trace (numeric or categorical).
    Errors never stop the evaluation; they are counted (error_count / last_error)."""

    def __init__(
        self,
        client: LangfuseClient,
        run_name: str | None = None,
        dataset_ref: str | None = None,
        metadata: dict | None = None,
        description: str = "",
    ) -> None:
        self.client = client
        self.run_name = run_name
        self.dataset_ref = dataset_ref
        self.metadata = metadata or {}
        self.description = description
        self.error_count = 0
        self.last_error: str | None = None
        self.events_sent = 0
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def item_id(self, case_id: str) -> str:
        return f"{self.dataset_ref}:{case_id}"

    def _fail(self, what: str, exc: Exception) -> None:
        with self._lock:
            self.error_count += 1
            self.last_error = f"{what}: {exc}"[:300]
        log.warning("langfuse %s failed: %s", what, exc)

    def _add(self, events: list[dict]) -> None:
        with self._lock:
            self._buffer.extend(events)

    def on_execution(self, run_id: str, case: EvaluationCase, execution: ExecutionResult) -> None:
        if not execution.trace_id:
            execution.trace_id = uuid.uuid4().hex
            execution.metadata["trace_created_by"] = "evaluation-platform"
            steps: list[dict] = []
            if execution.retrieved:
                steps.append({"type": "span", "name": "retrieval", "input": case.input,
                              "output": clip_docs(execution.retrieved)})
            steps.append({
                "type": "generation", "name": "generation", "model": self.metadata.get("model"),
                "input": case.input, "output": execution.output,
                "usageDetails": {"input": execution.input_tokens or 0,
                                 "output": execution.output_tokens or 0},
                **({"costDetails": {"total": execution.cost_usd}} if execution.cost_usd else {}),
            })
            self._add(self.client.trace_events(
                execution.trace_id, f"{case.system}:{case.id}", case.input, execution.output,
                {**self.metadata, "case_id": case.id, "run_id": run_id,
                 "latency_ms": execution.latency_ms},
                [case.system, *([self.run_name] if self.run_name else [])], steps,
            ))
        if self.dataset_ref and self.run_name:
            try:
                self.client.link_run_item(
                    self.run_name, self.item_id(case.id), execution.trace_id,
                    {"case_id": case.id, **self.metadata}, self.description,
                )
            except Exception as exc:  # noqa: BLE001
                self._fail(f"link run item {case.id}", exc)

    def on_result(self, run_id, case, execution, result: EvaluationResult) -> None:
        if execution is None or not execution.trace_id:
            return
        if result.score is not None:
            value: float | str = result.score
        elif result.label and result.label != "not_applicable":
            value = result.label  # categorical, e.g. TP / FN / error
        else:
            return
        self._add([self.client.score_event(
            execution.trace_id, result.evaluator, value, result.reason,
            {"run_id": run_id, "case_id": case.id, "passed": result.passed,
             "evaluator_version": result.evaluator_version, **result.metadata},
            score_id=self.client.score_id(run_id, execution.trace_id, result.evaluator),
        )])

    def flush(self) -> None:
        with self._lock:
            events, self._buffer = self._buffer, []
        if not events:
            return
        try:
            self.events_sent += self.client.ingest(events)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"ingestion of {len(events)} events", exc)


def default_sinks() -> list[ScoreSink]:
    client = LangfuseClient.from_env()
    return [LangfuseSink(client)] if client else []
