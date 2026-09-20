from app.integrations.langfuse_client import LangfuseClient


def get_langfuse() -> LangfuseClient | None:
    """None when LANGFUSE_* env vars are not set -> platform still works, without Langfuse."""
    return LangfuseClient.from_env()
