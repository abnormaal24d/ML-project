"""Fetch acceptance resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import definition_for

from .decision import FetchAcceptance

if TYPE_CHECKING:
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettingsCatalog,
    )

AcceptanceMode = Literal["strict", "exploratory"]


class FetchAcceptanceResolver:
    """Resolve transport-level fetch acceptance from settings.

    Important:
    - `kind` is the requested task intent.
    - `acceptance_mode` controls how broad the transport acceptance may be.
    - This resolver does NOT determine the final modality truth of the
    response.
    """

    _SUPPORTED_ACCEPTANCE_MODES = frozenset(
        {
            "strict",
            "exploratory",
        }
    )

    def __init__(
        self,
        *,
        modality_acceptance: ModalityAcceptanceSettingsCatalog,
    ) -> None:
        self._modality_acceptance = modality_acceptance

    def resolve(
        self,
        *,
        kind: MediaKind,
        acceptance_mode: AcceptanceMode = "strict",
    ) -> FetchAcceptance:
        """Return the effective transport acceptance."""

        normalized_mode = self._normalize_acceptance_mode(acceptance_mode)
        allowed_content_types_by_kind = {
            media_kind: definition_for(media_kind).mime_types
            for media_kind in MediaKind
        }
        max_bytes_by_kind = {
            media_kind: getattr(
                self._modality_acceptance, media_kind.value
            ).fetch_max_bytes
            for media_kind in MediaKind
        }

        content_type_byte_limits = self._content_type_byte_limits(
            allowed_content_types_by_kind=allowed_content_types_by_kind,
            max_bytes_by_kind=max_bytes_by_kind,
        )

        if normalized_mode == "exploratory":
            allowed_content_types = tuple(
                dict.fromkeys(
                    content_type
                    for kind_values in allowed_content_types_by_kind.values()
                    for content_type in kind_values
                    if content_type
                )
            )
            max_bytes = max_bytes_by_kind[kind]
        else:
            allowed_content_types = allowed_content_types_by_kind[kind]
            max_bytes = max_bytes_by_kind[kind]

        kind_acceptance = getattr(self._modality_acceptance, kind.value)
        is_exploratory = normalized_mode == "exploratory"

        return FetchAcceptance(
            requested_kind=kind,
            allowed_content_types=allowed_content_types,
            max_bytes=max_bytes,
            allow_metadata_only_when_oversized=(
                kind_acceptance.allow_metadata_only_when_oversized
            )
            or is_exploratory,
            allow_streaming_when_oversized=(
                kind_acceptance.allow_streaming_when_oversized
            ),
            allow_partial_when_oversized=(
                kind_acceptance.allow_partial_when_oversized
            )
            or is_exploratory,
            max_bytes_by_content_type=content_type_byte_limits,
        )

    def _normalize_acceptance_mode(
        self, acceptance_mode: str
    ) -> AcceptanceMode:
        normalized = str(acceptance_mode).strip().lower()
        if normalized in self._SUPPORTED_ACCEPTANCE_MODES:
            return normalized  # type: ignore[return-value]
        raise ValueError(f"unsupported acceptance mode: {acceptance_mode!r}")

    @staticmethod
    def _content_type_byte_limits(
        *,
        allowed_content_types_by_kind: dict[MediaKind, tuple[str, ...]],
        max_bytes_by_kind: dict[MediaKind, int],
    ) -> dict[str, int]:
        limits: dict[str, int] = {}
        for kind, content_types in allowed_content_types_by_kind.items():
            max_bytes = max_bytes_by_kind[kind]
            for content_type in content_types:
                normalized = content_type.strip().lower()
                if not normalized:
                    continue
                previous = limits.get(normalized)
                limits[normalized] = (
                    max_bytes if previous is None else min(previous, max_bytes)
                )
        return limits
