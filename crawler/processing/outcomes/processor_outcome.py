"""Canonical processor outcome contract.

One immutable processor decision. Explicit policy attributes carry control
flow; ``metadata`` carries only supplementary extension information. Task
facts belong to ``CrawlTask``, result facts to ``FetchResult``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

ProcessorOutcomeStatus = Literal[
    "success",
    "dropped",
    "deferred",
    "failure",
]


RESERVED_PROCESSOR_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        # ProcessorOutcome policy
        "status",
        "stage",
        "reason",
        "detail",
        "retry_after_seconds",
        "retry_class",
        "retry_error_kind",
        "counts_toward_task_retry_budget",
        "terminal_eligible",
        "error_type",
        "error",
        # CrawlTask facts
        "task_id",
        "url",
        "requested_kind",
        "target_kind",
        # FetchResult facts
        "kind",
        "result_kind",
        "content_type",
        "mime_type",
        "bytes",
        "stored",
        "category",
        "relevance_score",
    }
)


@dataclass(frozen=True, slots=True)
class ProcessorOutcome:
    """Describe one immutable processor decision."""

    status: ProcessorOutcomeStatus
    stage: str
    reason: str = ""
    detail: str = ""
    retry_after_seconds: float | None = None
    retry_class: str = ""
    retry_error_kind: str = ""
    counts_toward_task_retry_budget: bool = False
    terminal_eligible: bool = False
    error_type: str = ""
    error: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {
            "success",
            "dropped",
            "deferred",
            "failure",
        }:
            raise ValueError(
                f"unsupported processor outcome status: {self.status!r}"
            )

        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError(
                "ProcessorOutcome.stage must be a non-empty string"
            )

        if not isinstance(self.reason, str):
            raise TypeError("ProcessorOutcome.reason must be a string")

        if not isinstance(self.detail, str):
            raise TypeError("ProcessorOutcome.detail must be a string")

        if self.status != "success" and not self.reason:
            raise ValueError(f"{self.status} outcome requires a reason")

        retry_after_seconds = self.retry_after_seconds

        if retry_after_seconds is not None:
            if isinstance(retry_after_seconds, bool):
                raise TypeError(
                    "ProcessorOutcome.retry_after_seconds must be numeric"
                )

            if not isinstance(retry_after_seconds, (int, float)):
                raise TypeError(
                    "ProcessorOutcome.retry_after_seconds must be numeric"
                )

            retry_after_seconds = float(retry_after_seconds)

            if not math.isfinite(retry_after_seconds):
                raise ValueError(
                    "ProcessorOutcome.retry_after_seconds must be finite"
                )

            if retry_after_seconds < 0:
                raise ValueError(
                    "ProcessorOutcome.retry_after_seconds cannot be negative"
                )

            object.__setattr__(
                self,
                "retry_after_seconds",
                retry_after_seconds,
            )

        if self.status == "deferred":
            if retry_after_seconds is None:
                raise ValueError(
                    "deferred outcome requires retry_after_seconds"
                )
        elif retry_after_seconds is not None:
            raise ValueError(
                "retry_after_seconds is only valid for deferred outcomes"
            )

        for attribute in (
            "retry_class",
            "retry_error_kind",
            "error_type",
            "error",
        ):
            if not isinstance(getattr(self, attribute), str):
                raise TypeError(
                    f"ProcessorOutcome.{attribute} must be a string"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("ProcessorOutcome.metadata must be a mapping")

        copied_metadata: dict[str, object] = {}

        for name, value in self.metadata.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "ProcessorOutcome metadata keys must be non-empty strings"
                )

            if name in RESERVED_PROCESSOR_METADATA_KEYS:
                raise ValueError(
                    f"reserved processor key {name!r} "
                    "cannot be extension metadata"
                )

            copied_metadata[name] = value

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(copied_metadata),
        )

    @classmethod
    def success(
        cls,
        *,
        stage: str,
        detail: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ProcessorOutcome:
        return cls(
            status="success",
            stage=stage,
            detail=detail,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def dropped(
        cls,
        *,
        stage: str,
        reason: str,
        detail: str = "",
        retry_class: str = "",
        retry_error_kind: str = "",
        counts_toward_task_retry_budget: bool = False,
        terminal_eligible: bool = False,
        error_type: str = "",
        error: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ProcessorOutcome:
        return cls(
            status="dropped",
            stage=stage,
            reason=reason,
            detail=detail,
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            counts_toward_task_retry_budget=counts_toward_task_retry_budget,
            terminal_eligible=terminal_eligible,
            error_type=error_type,
            error=error,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def deferred(
        cls,
        *,
        stage: str,
        reason: str,
        retry_after_seconds: float,
        detail: str = "",
        retry_class: str = "",
        retry_error_kind: str = "",
        counts_toward_task_retry_budget: bool = False,
        terminal_eligible: bool = False,
        error_type: str = "",
        error: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ProcessorOutcome:
        return cls(
            status="deferred",
            stage=stage,
            reason=reason,
            detail=detail,
            retry_after_seconds=retry_after_seconds,
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            counts_toward_task_retry_budget=counts_toward_task_retry_budget,
            terminal_eligible=terminal_eligible,
            error_type=error_type,
            error=error,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failure(
        cls,
        *,
        stage: str,
        reason: str,
        detail: str = "",
        retry_class: str = "",
        retry_error_kind: str = "",
        counts_toward_task_retry_budget: bool = False,
        terminal_eligible: bool = False,
        error_type: str = "",
        error: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ProcessorOutcome:
        return cls(
            status="failure",
            stage=stage,
            reason=reason,
            detail=detail,
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            counts_toward_task_retry_budget=counts_toward_task_retry_budget,
            terminal_eligible=terminal_eligible,
            error_type=error_type,
            error=error,
            metadata={} if metadata is None else metadata,
        )

    @property
    def succeeded(self) -> bool:
        """Return whether processing completed successfully."""
        return self.status == "success"
