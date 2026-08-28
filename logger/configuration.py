from __future__ import annotations

import logging
from importlib.util import find_spec

from config.settings.logging import LoggingSettings
from logger.filters import ConsoleNoiseFilter, EventRateLimitFilter
from logger.formatters import JsonFormatter, PlainFormatter
from logger.handlers import (
    build_project_handlers,
    get_project_handler_identity,
    remove_project_handlers,
)


def configure_logging(settings: LoggingSettings) -> None:
    """Configure the root project logger from settings."""

    root_logger = logging.getLogger(settings.root_name)
    root_level = _resolve_level(settings.level)

    root_logger.setLevel(root_level)
    root_logger.propagate = settings.propagate

    remove_project_handlers(root_logger)

    handlers = build_project_handlers(
        enable_console=settings.enable_console,
        console_level=_resolve_level(settings.console_level or settings.level),
        enable_file=settings.enable_file,
        file_level=_resolve_level(settings.file_level or settings.level),
        file_path=settings.file_path,
        max_bytes=settings.max_bytes,
        backup_count=settings.backup_count,
        console_formatter=_build_console_formatter(settings),
        file_formatter=_build_file_formatter(settings),
    )

    if settings.rate_limit_enabled:
        rate_limit_filter = EventRateLimitFilter(
            default_min_interval_sec=settings.rate_limit_min_interval_sec,
            governance=settings.event_rate_limit_governance,
            max_entries=settings.rate_limit_max_entries,
        )

        for handler in handlers:
            handler.addFilter(rate_limit_filter)

    for handler in handlers:
        identity = get_project_handler_identity(handler)
        if identity is not None and identity.sink == "console":
            handler.addFilter(
                ConsoleNoiseFilter(settings.console_suppressed_events)
            )
        root_logger.addHandler(handler)

    for logger_name, level_name in settings.component_levels.items():
        _validate_configured_logger_name(
            root_name=settings.root_name,
            logger_name=logger_name,
        )

        logging.getLogger(
            _resolve_configured_logger_name(
                root_name=settings.root_name,
                logger_name=logger_name,
            )
        ).setLevel(_resolve_level(level_name))


def _build_console_formatter(
    settings: LoggingSettings,
) -> logging.Formatter:
    if settings.console_format.lower() == "json":
        return JsonFormatter(datefmt=settings.datefmt)

    return PlainFormatter(
        datefmt=settings.datefmt,
        compact_context=settings.console_compact_context,
    )


def _build_file_formatter(
    settings: LoggingSettings,
) -> logging.Formatter:
    if settings.file_format == "json":
        return JsonFormatter(datefmt=settings.datefmt)

    return PlainFormatter(datefmt=settings.datefmt)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    resolved = logging.getLevelName(level.upper())

    if isinstance(resolved, int):
        return resolved

    raise ValueError(f"unknown logging level: {level}")


def _resolve_configured_logger_name(
    *,
    root_name: str,
    logger_name: str,
) -> str:
    if logger_name == root_name or logger_name.startswith(f"{root_name}."):
        return logger_name

    return f"{root_name}.{logger_name}"


def _validate_configured_logger_name(
    *,
    root_name: str,
    logger_name: str,
) -> None:
    if logger_name == root_name:
        return

    module_name = (
        logger_name.removeprefix(f"{root_name}.")
        if logger_name.startswith(f"{root_name}.")
        else logger_name
    )

    if find_spec(module_name) is not None:
        return

    raise ValueError(f"unknown logger component: {logger_name}")
