from typing import Any


def get_path(data: Any, path: str, default: Any = None) -> Any:
    """Read nested dict values with dotted paths: get_path(d, 'a.b.c')."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def doc_keys(doc: Any) -> set[str]:
    """All identifiers of a retrieved doc/citation (id and url), as strings."""
    if isinstance(doc, dict):
        return {str(doc[k]) for k in ("id", "doc_id", "url", "source") if doc.get(k)}
    return {str(doc)}
