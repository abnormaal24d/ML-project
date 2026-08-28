"""Project logger with structured keyword fields."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import TracebackType
from typing import cast

from logger.log_context import current_log_context
from logger.redaction import redact_log_text, redact_log_value

ExcInfo = (
    bool
    | tuple[
        type[BaseException],
        BaseException,
        TracebackType | None,
    ]
    | tuple[None, None, None]
    | BaseException
    | None
)


class ProjectLogger:
    """Small project logger facade with per-factory context fields."""

    __slots__ = ("_context_items", "_logger")

    def __init__(
        self,
        logger: logging.Logger,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self._logger = logger
        self._context_items = _context_items(context)

    @property
    def name(self) -> str:
        """Return the underlying logger name."""

        return self._logger.name

    def debug(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit a DEBUG record with structured keyword fields."""

        if self.is_enabled_for(logging.DEBUG):
            self._log_with_fields(
                logging.DEBUG,
                msg,
                args,
                kwargs,
            )

    def debug_lazy(
        self,
        msg: object,
        *,
        fields: Mapping[str, object] | None = None,
        field_builders: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Emit DEBUG only after lazily building expensive fields."""

        if self.is_enabled_for(logging.DEBUG):
            self.debug(
                msg,
                **_lazy_fields(fields=fields, field_builders=field_builders),
                **kwargs,
            )

    def info(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit an INFO record with structured keyword fields."""

        if self.is_enabled_for(logging.INFO):
            self._log_with_fields(
                logging.INFO,
                msg,
                args,
                kwargs,
            )

    def info_lazy(
        self,
        msg: object,
        *,
        fields: Mapping[str, object] | None = None,
        field_builders: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Emit INFO only after lazily building expensive fields."""

        if self.is_enabled_for(logging.INFO):
            self.info(
                msg,
                **_lazy_fields(fields=fields, field_builders=field_builders),
                **kwargs,
            )

    def warning(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit a WARNING record with structured keyword fields."""

        if self.is_enabled_for(logging.WARNING):
            self._log_with_fields(
                logging.WARNING,
                msg,
                args,
                kwargs,
            )

    def error(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit an ERROR record with structured keyword fields."""

        if self.is_enabled_for(logging.ERROR):
            self._log_with_fields(
                logging.ERROR,
                msg,
                args,
                kwargs,
            )

    def exception(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit an ERROR record with exception information attached."""

        kwargs.setdefault("exc_info", True)

        self.error(
            msg,
            *args,
            **kwargs,
        )

    def critical(
        self,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit a CRITICAL record with structured keyword fields."""

        if self.is_enabled_for(logging.CRITICAL):
            self._log_with_fields(
                logging.CRITICAL,
                msg,
                args,
                kwargs,
            )

    def log(
        self,
        level: int,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit a record at an arbitrary numeric log level."""

        if self.is_enabled_for(level):
            self._log_with_fields(
                level,
                msg,
                args,
                kwargs,
            )

    def is_enabled_for(
        self,
        level: int,
    ) -> bool:
        """Return whether this logger emits records at the given level."""

        return self._logger.isEnabledFor(level)

    def is_debug_enabled(self) -> bool:
        """Return whether this logger emits debug records."""

        return self.is_enabled_for(logging.DEBUG)

    def _log_with_fields(
        self,
        level: int,
        msg: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        exc_info = cast(
            "ExcInfo",
            kwargs.pop("exc_info", None),
        )

        extra = _as_extra_mapping(kwargs.pop("extra", None))

        message = kwargs.pop("message", None)

        stack_info = bool(kwargs.pop("stack_info", False))

        stacklevel = _as_stacklevel(kwargs.pop("stacklevel", 1))

        record_extra = _record_extra(
            context_items=self._context_items,
            extra=extra,
            fields=kwargs,
            message=message,
            event_name=_event_name(msg),
        )

        self._logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            extra=record_extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
        )


def _as_extra_mapping(
    value: object,
) -> Mapping[str, object] | None:
    if value is None:
        return None

    if isinstance(value, Mapping):
        return value

    raise TypeError("logger extra must be a mapping")


def _as_stacklevel(
    value: object,
) -> int:
    if isinstance(value, int):
        return value

    return 1


def _context_items(
    context: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(key), value)
        for key, value in (context or {}).items()
        if value is not None
    )


def _record_extra(
    *,
    context_items: tuple[tuple[str, object], ...],
    extra: Mapping[str, object] | None,
    fields: Mapping[str, object],
    message: object,
    event_name: str,
) -> dict[str, object] | None:
    """Build the log record payload with static context and dynamic fields."""

    merged: dict[str, object] = {
        "_project_event": event_name,
    }

    field_keys: list[str] = []

    if message is not None:
        merged["_project_message"] = redact_log_value(message)

    for key, value in context_items:
        merged[key] = redact_log_value(value, field_name=key)
        field_keys.append(key)

    for key, value in current_log_context().items():
        if value is None:
            continue

        merged[key] = redact_log_value(value, field_name=key)

        if not key.startswith("_"):
            field_keys.append(key)

    if extra is not None:
        for key, value in extra.items():
            if value is None:
                continue

            field_name = str(key)

            if field_name == "message":
                merged["_project_message"] = redact_log_value(value)
                continue

            merged[field_name] = redact_log_value(
                value,
                field_name=field_name,
            )

            if not field_name.startswith("_"):
                field_keys.append(field_name)

    for key, value in fields.items():
        if value is None:
            continue

        if key == "message":
            merged["_project_message"] = redact_log_value(value)
            continue

        merged[key] = redact_log_value(value, field_name=key)

        if not key.startswith("_"):
            field_keys.append(key)

    if field_keys:
        merged["_project_field_keys"] = tuple(dict.fromkeys(field_keys))

    return merged


def _event_name(
    msg: object,
) -> str:
    return redact_log_text(str(msg))


def _lazy_fields(
    *,
    fields: Mapping[str, object] | None,
    field_builders: Mapping[str, object] | None,
) -> dict[str, object]:
    resolved = dict(fields or {})
    for key, builder in (field_builders or {}).items():
        resolved[str(key)] = builder() if callable(builder) else builder
    return resolved
