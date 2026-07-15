"""Task-local, content-free model usage metadata for tracing."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_USAGE: ContextVar[dict[str, int] | None] = ContextVar(
    "evidenceflow_model_usage", default=None
)


def clear_usage_metadata() -> None:
    _USAGE.set(None)


def capture_usage_metadata(message: Any) -> None:
    raw = getattr(message, "usage_metadata", None)
    if not isinstance(raw, dict):
        return
    allowed = ("input_tokens", "output_tokens", "total_tokens")
    usage = {
        key: int(raw[key])
        for key in allowed
        if isinstance(raw.get(key), int) and not isinstance(raw.get(key), bool)
    }
    if usage:
        _USAGE.set(usage)


def consume_usage_metadata() -> dict[str, int]:
    usage = _USAGE.get() or {}
    _USAGE.set(None)
    return usage
