"""Project-scoped logger resolution utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class ProjectLoggerFactory:
    """Create project-scoped loggers."""

    root_name: str = "project"
    namespace: str = ""
    base_context: Mapping[str, object] = field(default_factory=dict)
    _logger_cache: dict[str, ProjectLogger] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def create_child(
        self,
        namespace: str,
    ) -> ProjectLoggerFactory:
        """Create a child logger factory."""

        resolved_namespace = (
            f"{self.namespace}.{namespace}" if self.namespace else namespace
        )

        return ProjectLoggerFactory(
            root_name=self.root_name,
            namespace=resolved_namespace,
            base_context=self.base_context,
        )

    def get_logger(
        self,
        name: str,
    ) -> ProjectLogger:
        """Return a project logger."""

        resolved_name = self._resolve_name(name)

        cached = self._logger_cache.get(resolved_name)
        if cached is not None:
            return cached

        logger = ProjectLogger(
            logging.getLogger(resolved_name),
            context=self._logger_context(
                logger_name=resolved_name,
            ),
        )
        self._logger_cache[resolved_name] = logger
        return logger

    def get_logger_for(
        self,
        component_type: type[object],
    ) -> ProjectLogger:
        """Return a project logger named after a component module."""

        return self.get_logger(component_type.__module__)

    def _resolve_name(
        self,
        name: str,
    ) -> str:
        """Resolve fully-qualified logger name."""

        full_name = f"{self.namespace}.{name}" if self.namespace else name

        return f"{self.root_name}.{full_name}" if full_name else self.root_name

    def _logger_context(
        self,
        *,
        logger_name: str,
    ) -> dict[str, object]:
        context = {
            str(key): value
            for key, value in self.base_context.items()
            if value is not None
        }

        context.update(
            {
                "logger_root": self.root_name,
                "component": logger_name.rsplit(".", 1)[-1],
            }
        )

        return context
