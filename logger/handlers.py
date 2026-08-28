"""Project-owned logging handlers and handler discovery helpers."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_HANDLER_MARKER = "_project_handler_identity"


@dataclass(frozen=True, slots=True)
class ProjectHandlerIdentity:
    """Identity marker used for handler discovery on reconfiguration."""

    sink: str


def ensure_parent_directory(path: Path) -> None:
    """Create parent directories for a configured file sink."""

    path.parent.mkdir(parents=True, exist_ok=True)


def mark_project_handler(
    handler: logging.Handler,
    *,
    identity: ProjectHandlerIdentity,
) -> logging.Handler:
    """Attach a stable project handler identity to the handler."""

    setattr(handler, PROJECT_HANDLER_MARKER, identity)
    return handler


def get_project_handler_identity(
    handler: logging.Handler,
) -> ProjectHandlerIdentity | None:
    """
    Return the project handler identity if the handler is project-managed.
    """

    identity = getattr(handler, PROJECT_HANDLER_MARKER, None)

    if isinstance(identity, ProjectHandlerIdentity):
        return identity

    return None


def is_project_handler(handler: logging.Handler) -> bool:
    """Return whether the handler belongs to the project logger setup."""

    return get_project_handler_identity(handler) is not None


def find_project_handlers(
    target_logger: logging.Logger,
) -> list[logging.Handler]:
    """Return the handlers owned by the project logger setup."""

    return [
        handler
        for handler in target_logger.handlers
        if is_project_handler(handler)
    ]


def remove_project_handlers(target_logger: logging.Logger) -> None:
    """Detach and close all project-managed handlers."""

    for handler in find_project_handlers(target_logger):
        target_logger.removeHandler(handler)

        try:
            handler.close()
        except (OSError, ValueError):
            continue


def build_console_handler(
    *,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    """Build the configured console sink."""

    handler = mark_project_handler(
        logging.StreamHandler(stream=sys.stdout),
        identity=ProjectHandlerIdentity(sink="console"),
    )

    handler.setLevel(level)
    handler.setFormatter(formatter)

    return handler


def build_file_handler(
    *,
    level: int,
    file_path: Path | str | None,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    """Build the configured file sink."""

    if file_path is None or str(file_path).strip() == "":
        raise ValueError(
            "file logger is enabled but logging.file_path is missing. "
            'Set file_path = "runtime/logs/project.log" '
            "or disable file logging with enable_file = false."
        )

    path = Path(file_path).expanduser().resolve()
    ensure_parent_directory(path)

    handler = mark_project_handler(
        RotatingFileHandler(
            filename=str(path),
            maxBytes=max(1, int(max_bytes)),
            backupCount=max(0, int(backup_count)),
            encoding="utf-8",
            delay=True,
        ),
        identity=ProjectHandlerIdentity(sink="file"),
    )

    handler.setLevel(level)
    handler.setFormatter(formatter)

    return handler


def build_project_handlers(
    *,
    enable_console: bool,
    console_level: int,
    enable_file: bool,
    file_level: int,
    file_path: Path | str | None,
    max_bytes: int,
    backup_count: int,
    console_formatter: logging.Formatter,
    file_formatter: logging.Formatter,
) -> list[logging.Handler]:
    """Return the handlers implied by the logger configuration."""

    handlers: list[logging.Handler] = []

    if enable_console:
        handlers.append(
            build_console_handler(
                level=console_level,
                formatter=console_formatter,
            )
        )

    if enable_file:
        handlers.append(
            build_file_handler(
                level=file_level,
                file_path=file_path,
                max_bytes=max_bytes,
                backup_count=backup_count,
                formatter=file_formatter,
            )
        )

    return handlers
