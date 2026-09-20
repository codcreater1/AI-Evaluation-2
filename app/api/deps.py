import os

from app.integrations.langfuse_client import LangfuseClient

_clients: dict[tuple, LangfuseClient] = {}


def get_langfuse() -> LangfuseClient | None:
    """None when LANGFUSE_* env vars are not set -> platform still works, without Langfuse.
    One client per credentials, so its read cache and remembered endpoints survive requests."""
    key = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL"),
           os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY"))
    if not (key[1] and key[2]):
        return None
    if key not in _clients:
        client = LangfuseClient.from_env()
        if client is None:
            return None
        _clients[key] = client
    return _clients[key]
