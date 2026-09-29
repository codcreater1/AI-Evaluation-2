import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset, DatasetCase, DatasetVersion
from app.errors import Conflict, Invalid, NotFound
from app.integrations.langfuse_client import LangfuseClient
from app.schemas import EvaluationCase

VERSION_RE = re.compile(r"^(?P<name>.+)-v(?P<v>\d+)$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def parse_ref(ref: str) -> tuple[str, int | None]:
    """'ata-rag-golden-v3' -> ('ata-rag-golden', 3); 'ata-rag-golden' -> ('ata-rag-golden', None)."""
    m = VERSION_RE.match(ref)
    return (m["name"], int(m["v"])) if m else (ref, None)


def make_ref(name: str, version: int) -> str:
    return f"{name}-v{version}"


def content_hash(cases: list[EvaluationCase]) -> str:
    canon = json.dumps([c.model_dump() for c in sorted(cases, key=lambda c: c.id)],
                       sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _validate(cases: list[EvaluationCase], system: str) -> None:
    if not cases:
        raise Invalid("a dataset version needs at least one case")
    ids = [c.id for c in cases]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise Invalid(f"duplicate case ids: {dups[:5]}")
    wrong = [c.id for c in cases if c.system != system]
    if wrong:
        raise Invalid(f"cases belong to another system than '{system}': {wrong[:5]}")


def _store(db: Session, ds: Dataset, cases: list[EvaluationCase], parent: int | None,
           note: str) -> DatasetVersion:
    last = max((v.version for v in ds.versions), default=0)
    ver = DatasetVersion(
        dataset_name=ds.name, version=last + 1, content_hash=content_hash(cases),
        parent_version=parent, note=note, n_cases=len(cases),
    )
    ver.cases = [DatasetCase(case_id=c.id, payload=c.model_dump()) for c in cases]
    db.add(ver)
    db.commit()
    return ver


def create_dataset(db: Session, name: str, system: str, description: str,
                   cases: list[EvaluationCase], note: str = "") -> DatasetVersion:
    if not NAME_RE.match(name) or VERSION_RE.match(name):
        raise Invalid("dataset name: lowercase letters/digits/-_. and must not end with '-v<number>'"
                      " (the version suffix is added automatically)")
    if db.get(Dataset, name):
        raise Conflict(f"dataset '{name}' already exists; create a new version instead")
    _validate(cases, system)
    ds = Dataset(name=name, system=system, description=description)
    db.add(ds)
    db.flush()
    return _store(db, ds, cases, None, note or "initial version")


def get_dataset(db: Session, name: str) -> Dataset:
    ds = db.get(Dataset, name)
    if ds is None:
        raise NotFound(f"dataset '{name}' not found")
    return ds


def get_version(db: Session, name: str, version: int | None = None) -> DatasetVersion:
    ds = get_dataset(db, name)
    if not ds.versions:
        raise NotFound(f"dataset '{name}' has no versions")
    if version is None:
        return ds.versions[-1]
    for v in ds.versions:
        if v.version == version:
            return v
    raise NotFound(f"dataset '{name}' has no version {version}")


def load_cases(version: DatasetVersion) -> list[EvaluationCase]:
    return [EvaluationCase.model_validate(c.payload) for c in version.cases]


def create_version(
    db: Session, name: str, *, base_version: int | None = None,
    cases: list[EvaluationCase] | None = None,
    add_cases: list[EvaluationCase] | None = None,
    update_cases: list[EvaluationCase] | None = None,
    remove_case_ids: list[str] | None = None,
    note: str = "",
) -> DatasetVersion:
    """Existing versions never change. Either give the full new case list (`cases`) or derive from
    a base version with add/update/remove. Case ids are stable across versions."""
    ds = get_dataset(db, name)
    add_cases, update_cases, remove_case_ids = add_cases or [], update_cases or [], remove_case_ids or []
    base = get_version(db, name, base_version)
    if cases is not None:
        if add_cases or update_cases or remove_case_ids:
            raise Invalid("give either `cases` or add/update/remove, not both")
        new_cases = cases
    else:
        working = {c.id: c for c in load_cases(base)}
        for c in add_cases:
            if c.id in working:
                raise Conflict(f"case '{c.id}' already exists; use update_cases to change it")
            working[c.id] = c
        for c in update_cases:
            if c.id not in working:
                raise NotFound(f"cannot update unknown case '{c.id}'")
            working[c.id] = c
        for cid in remove_case_ids:
            if cid not in working:
                raise NotFound(f"cannot remove unknown case '{cid}'")
            del working[cid]
        new_cases = list(working.values())
    _validate(new_cases, ds.system)
    if content_hash(new_cases) == base.content_hash:
        raise Conflict("no changes compared to the base version")
    return _store(db, ds, new_cases, base.version, note)


def case_from_trace(client: LangfuseClient, trace_id: str, case_id: str, system: str,
                    expected_output: dict, metadata: dict | None = None,
                    input_override: dict | None = None) -> EvaluationCase:
    """Production -> evaluation feedback loop: turn a bad production trace into a golden case."""
    trace = client.get_trace(trace_id)
    raw = trace.get("input")
    inp = input_override if input_override is not None else (raw if isinstance(raw, dict) else {"input": raw})
    return EvaluationCase(
        id=case_id, system=system, input=inp, expected_output=expected_output,
        metadata={**(metadata or {}), "source": "production", "source_trace_id": trace_id},
    )


def sync_to_langfuse(db: Session, client: LangfuseClient, version: DatasetVersion,
                     force: bool = False) -> str:
    """Creates Langfuse dataset '<name>-v<N>' and its items (item id = '<ref>:<case_id>')."""
    ref = make_ref(version.dataset_name, version.version)
    if version.langfuse_synced_at and not force:
        return ref
    client.upsert_dataset(ref, version.dataset.description, {
        "dataset": version.dataset_name, "version": version.version,
        "content_hash": version.content_hash, "system": version.dataset.system,
    })
    cases = load_cases(version)

    def push(c: EvaluationCase) -> None:
        client.upsert_dataset_item(ref, f"{ref}:{c.id}", c.input, c.expected_output,
                                   {**c.metadata, "case_id": c.id, "system": c.system})

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(push, cases))
    version.langfuse_synced_at = datetime.now(UTC)
    db.commit()
    return ref


def list_datasets(db: Session) -> list[Dataset]:
    return list(db.scalars(select(Dataset).order_by(Dataset.name)))
