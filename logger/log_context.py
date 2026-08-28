"""Dynamic log context carried by context variables."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_EMPTY_LOG_CONTEXT: Mapping[str, object] = MappingProxyType({})

_CURRENT_LOG_CONTEXT: ContextVar[Mapping[str, object]] = ContextVar(
    "current_log_context",
    default=_EMPTY_LOG_CONTEXT,
)


def current_log_context() -> Mapping[str, object]:
    """Return the active dynamic log context."""

    return _CURRENT_LOG_CONTEXT.get()


@contextmanager
def bind_log_context(fields: Mapping[str, object]) -> Iterator[None]:
    """Temporarily bind dynamic fields for project log records."""

    merged = dict(current_log_context())
    merged.update(
        {str(key): value for key, value in fields.items() if value is not None}
    )
    token = _CURRENT_LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _CURRENT_LOG_CONTEXT.reset(token)
