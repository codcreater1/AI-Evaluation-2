"""Built-in test scenario used by the dashboard's 'Run test scenario' button:
golden dataset v1 -> baseline -> harmless change (PASS) -> deliberately broken change (FAIL).
Everything is pushed to Langfuse when it is configured."""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.config.loader import RunConfig
from app.datasets.service import create_dataset
from app.db.models import Experiment
from app.experiments.gates import QualityGates
from app.experiments.service import run_experiment
from app.integrations.langfuse_client import LangfuseClient
from app.schemas import EvaluationCase, ExecutionResult

SYSTEM = "demo-ata-rag"
N = 6  # small on purpose: fits the free Langfuse plan's rate limits


def _executions(wrong: tuple[int, ...] = (), latency: float = 1200) -> dict[str, ExecutionResult]:
    out = {}
    for i in range(1, N + 1):
        good = i not in wrong
        out[f"rag-{i:03d}"] = ExecutionResult(
            output={"answer": f"answer {i}" if good else "some wrong answer", "citations": [f"doc-{i}"]},
            retrieved=[{"id": f"doc-{i}" if good else "doc-x", "text": f"Official text for question {i}"}],
            latency_ms=latency, input_tokens=400, output_tokens=120, cost_usd=0.003,
        )
    return out


def run_demo(db: Session, client: LangfuseClient | None) -> list[Experiment]:
    stamp = datetime.now(UTC).strftime("%m%d-%H%M%S")
    ds = f"ata-rag-demo-{stamp}"
    cases = [EvaluationCase(
        id=f"rag-{i:03d}", system=SYSTEM, input={"question": f"Sample question {i}?"},
        expected_output={"answer": f"answer {i}", "relevant_doc_ids": [f"doc-{i}"]},
    ) for i in range(1, N + 1)]
    create_dataset(db, ds, SYSTEM, "demo golden dataset", cases)
    config = RunConfig(
        system=SYSTEM, dataset=f"{ds}-v1", max_workers=8,
        evaluators=[{"name": "exact_match", "params": {"output_key": "answer"}},
                    {"name": "retrieval_recall_at_k", "params": {"k": 3}},
                    "citation_exists",
                    {"name": "latency_threshold", "params": {"max_ms": 3000}}],
        gates=QualityGates(minimum={"exact_match": 0.90, "retrieval_recall_at_k": 0.90},
                           max_drop={"default": 0.02}, max_increase_pct={"latency_ms": 25},
                           max_newly_failing_cases=1),
    )
    runs = [
        (_executions(), {"prompt_version": "v1", "app_version": "1.0"}, True),
        (_executions(latency=1300), {"prompt_version": "v2", "app_version": "1.1"}, False),
        (_executions(wrong=(2, 5), latency=2400), {"prompt_version": "v3-broken", "app_version": "1.2"}, False),
    ]
    return [
        run_experiment(db, config, executions=ex, meta={"model": "demo-model", **meta},
                       set_baseline=base, client=client)
        for ex, meta, base in runs
    ]
