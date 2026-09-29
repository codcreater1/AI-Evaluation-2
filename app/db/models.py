from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _now() -> datetime:
    return datetime.now(UTC)


class System(Base):
    __tablename__ = "systems"
    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="")
    endpoint_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    system: Mapped[str] = mapped_column(String(100), index=True)
    dataset: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="completed")
    passed: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # evaluators, thresholds
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # app_version, model, prompt_version
    summary: Mapped[dict] = mapped_column(JSON, default=dict)  # aggregates, classification, failures
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    results: Mapped[list["EvaluationResultRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResultRow(Base):
    __tablename__ = "evaluation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(200), index=True)
    evaluator: Mapped[str] = mapped_column(String(100), index=True)
    evaluator_version: Mapped[str] = mapped_column(String(50), default="1")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    run: Mapped[EvaluationRun] = relationship(back_populates="results")


# ---------------------------------------------------------------- datasets (immutable versions)
from sqlalchemy import UniqueConstraint, event, inspect  # noqa: E402


class ImmutableDatasetError(RuntimeError):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    name: Mapped[str] = mapped_column(String(150), primary_key=True)
    system: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", order_by="DatasetVersion.version"
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_name", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(ForeignKey("datasets.name"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    n_cases: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    langfuse_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    cases: Mapped[list["DatasetCase"]] = relationship(
        back_populates="version", order_by="DatasetCase.id"
    )


class DatasetCase(Base):
    __tablename__ = "dataset_cases"
    __table_args__ = (UniqueConstraint("version_id", "case_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(200))  # stable id, same across versions
    payload: Mapped[dict] = mapped_column(JSON)  # full EvaluationCase
    version: Mapped[DatasetVersion] = relationship(back_populates="cases")


# Immutability is enforced in the ORM, not only by convention: once written, a version's content
# can never be updated or deleted. (Only the Langfuse sync timestamp may change.)
@event.listens_for(DatasetCase, "before_update")
@event.listens_for(DatasetCase, "before_delete")
def _cases_immutable(mapper, connection, target):
    raise ImmutableDatasetError("dataset cases are immutable; create a new dataset version instead")


@event.listens_for(DatasetVersion, "before_delete")
def _version_no_delete(mapper, connection, target):
    raise ImmutableDatasetError("dataset versions cannot be deleted")


@event.listens_for(DatasetVersion, "before_update")
def _version_immutable(mapper, connection, target):
    changed = {a.key for a in inspect(target).attrs if a.history.has_changes()}
    if changed - {"langfuse_synced_at"}:
        raise ImmutableDatasetError("dataset versions are immutable; create a new version instead")


# ---------------------------------------------------------------- experiments
class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (UniqueConstraint("system", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    number: Mapped[int] = mapped_column(Integer)  # "ATA RAG #42"
    name: Mapped[str] = mapped_column(String(200))
    system: Mapped[str] = mapped_column(String(100), index=True)
    dataset_name: Mapped[str] = mapped_column(String(150))
    dataset_version: Mapped[int] = mapped_column(Integer)
    dataset_ref: Mapped[str] = mapped_column(String(160))  # e.g. ata-rag-golden-v3
    dataset_hash: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"))
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # app_version, model, prompt_version, retriever
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # evaluators, thresholds, gates
    evaluator_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    aggregates: Mapped[dict] = mapped_column(JSON, default=dict)
    operational: Mapped[dict] = mapped_column(JSON, default=dict)  # mean latency / cost / tokens
    overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    langfuse: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------- human evaluation
class HumanEvaluation(Base):
    """A human reviewer's judgment on one (run, case, evaluator) triple - the same axis an
    EvaluationResultRow uses, so the two can be joined to measure LLM-judge/human agreement."""

    __tablename__ = "human_evaluations"
    __table_args__ = (UniqueConstraint("run_id", "case_id", "evaluator", "reviewer",
                                       name="uq_human_eval_reviewer_target"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(200), index=True)
    evaluator: Mapped[str] = mapped_column(String(100), index=True)  # metric being judged, e.g. answer_correctness
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0..1, same scale as EvaluationResult
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    reviewer: Mapped[str] = mapped_column(String(200), default="anonymous")
    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
