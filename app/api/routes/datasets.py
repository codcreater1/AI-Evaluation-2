from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_langfuse
from app.api.schemas_exp import (
    DatasetCreate,
    DatasetOut,
    FromTraceRequest,
    VersionCreate,
    VersionDetail,
    VersionOut,
)
from app.datasets import service
from app.db.models import DatasetVersion
from app.db.session import get_db
from app.errors import Invalid
from app.integrations.langfuse_client import LangfuseClient

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _v(v: DatasetVersion) -> VersionOut:
    return VersionOut(
        dataset=v.dataset_name, version=v.version, ref=service.make_ref(v.dataset_name, v.version),
        n_cases=v.n_cases, content_hash=v.content_hash, parent_version=v.parent_version,
        note=v.note, created_at=v.created_at, langfuse_synced_at=v.langfuse_synced_at)


def _ds(ds) -> DatasetOut:
    return DatasetOut(name=ds.name, system=ds.system, description=ds.description,
                      latest_version=ds.versions[-1].version if ds.versions else None,
                      versions=[_v(v) for v in ds.versions])


def _parse_version(version: str) -> int | None:
    if version == "latest":
        return None
    try:
        return int(version)
    except ValueError as exc:
        raise Invalid("version must be an integer or 'latest'") from exc


@router.post("", response_model=VersionOut, status_code=201)
def create_dataset(body: DatasetCreate, db: Session = Depends(get_db)):
    """Creates the dataset and its immutable version 1."""
    return _v(service.create_dataset(db, body.name, body.system, body.description, body.cases, body.note))


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return [_ds(d) for d in service.list_datasets(db)]


@router.get("/{name}", response_model=DatasetOut)
def get_dataset(name: str, db: Session = Depends(get_db)):
    return _ds(service.get_dataset(db, name))


@router.post("/{name}/versions", response_model=VersionOut, status_code=201)
def new_version(name: str, body: VersionCreate, db: Session = Depends(get_db)):
    """Existing versions are immutable: every change creates a new version."""
    return _v(service.create_version(
        db, name, base_version=body.base_version, cases=body.cases, add_cases=body.add_cases,
        update_cases=body.update_cases, remove_case_ids=body.remove_case_ids, note=body.note))


@router.get("/{name}/versions/{version}", response_model=VersionDetail)
def get_version(name: str, version: str, db: Session = Depends(get_db)):
    v = service.get_version(db, name, _parse_version(version))
    return VersionDetail(**_v(v).model_dump(), cases=service.load_cases(v))


@router.post("/{name}/versions/from-trace", response_model=VersionOut, status_code=201)
def add_case_from_trace(name: str, body: FromTraceRequest, db: Session = Depends(get_db),
                        client: LangfuseClient | None = Depends(get_langfuse)):
    """Production -> evaluation loop: a bad Langfuse trace becomes a new case in a new version."""
    if client is None:
        raise Invalid("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
    ds = service.get_dataset(db, name)
    case = service.case_from_trace(client, body.trace_id, body.case_id, ds.system,
                                   body.expected_output, body.metadata, body.input)
    return _v(service.create_version(db, name, base_version=body.base_version,
                                     add_cases=[case], note=body.note))


@router.post("/{name}/versions/{version}/sync-langfuse", response_model=VersionOut)
def sync_langfuse(name: str, version: str, force: bool = False, db: Session = Depends(get_db),
                  client: LangfuseClient | None = Depends(get_langfuse)):
    if client is None:
        raise Invalid("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
    v = service.get_version(db, name, _parse_version(version))
    service.sync_to_langfuse(db, client, v, force=force)
    return _v(v)
