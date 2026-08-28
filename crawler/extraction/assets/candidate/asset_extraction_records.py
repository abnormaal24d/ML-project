"""Asset extraction records and value coercion helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

AssetKind = Literal["image", "audio", "video", "document"]


@dataclass(frozen=True, slots=True)
class AssetCandidate:
    """Context-rich asset reference before scheduler admission."""

    url: str
    kind: AssetKind
    parent_url: str
    source_attribute: str
    source_tag: str
    alt_text: str | None = None
    caption_text: str | None = None
    surrounding_text: str | None = None
    mime_hint: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_task_context(
        self, *, quality_score: float | None
    ) -> dict[str, object]:
        """Return the serializable crawl-task context for this asset."""

        payload: dict[str, object] = {
            "tag_name": self.source_tag,
            "source_tag": self.source_tag,
            "source_attribute": self.source_attribute,
            "asset_url": self.url,
        }
        _add_if_present(
            payload, "text_hint", self.caption_text or self.alt_text
        )
        _add_if_present(payload, "alt_text", self.alt_text)
        _add_if_present(payload, "caption_text", self.caption_text)
        _add_if_present(payload, "surrounding_text", self.surrounding_text)
        _add_if_present(payload, "mime_hint", self.mime_hint)
        _add_if_present(payload, "width", self.width)
        _add_if_present(payload, "height", self.height)
        _add_if_present(payload, "duration_seconds", self.duration_seconds)
        _add_if_present(payload, "asset_quality_score", quality_score)
        for key, value in self.metadata.items():
            _add_if_present(payload, str(key), value)
        return payload


@dataclass(frozen=True, slots=True)
class AssetDiscoveryResult:
    """Grouped page-asset discovery output before scheduling."""

    parent_url: str
    images: tuple[AssetCandidate, ...] = ()
    audio: tuple[AssetCandidate, ...] = ()
    video: tuple[AssetCandidate, ...] = ()
    documents: tuple[AssetCandidate, ...] = ()
    rejected: tuple[dict[str, object], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def total_media(self) -> int:
        return len(self.images) + len(self.audio) + len(self.video)

    @property
    def total_assets(self) -> int:
        return self.total_media + len(self.documents)

    @property
    def assets(self) -> tuple[AssetCandidate, ...]:
        """Return all accepted assets in deterministic modality order."""

        return self.images + self.audio + self.video + self.documents


@dataclass(slots=True)
class AssetExtractionCandidateState:
    """Mutable extraction collections shared while discovering one document."""

    results: list[AssetCandidate]
    seen: dict[str, int]
    rejected: list[dict[str, object]]
    base_url: str


@dataclass(frozen=True, slots=True)
class AssetCandidateDraft:
    """Unresolved asset candidate plus extraction context."""

    candidate: str
    kind: AssetKind
    source_attribute: str
    source_tag: str
    context: dict[str, object]
    metadata: dict[str, object]


def build_asset_candidate_drafts(
    *,
    references: tuple[tuple[str, str], ...],
    kind: AssetKind,
    source_tag: str,
    context: dict[str, object],
    metadata: Mapping[str, object],
) -> tuple[AssetCandidateDraft, ...]:
    """Materialize approved references as unresolved candidate drafts."""

    return tuple(
        AssetCandidateDraft(
            candidate=candidate,
            kind=kind,
            source_attribute=attribute,
            source_tag=source_tag,
            context=context,
            metadata=dict(metadata),
        )
        for attribute, candidate in references
    )


def build_asset_discovery_result(
    *,
    parent_url: str,
    candidates: tuple[AssetCandidate, ...],
    rejected: tuple[dict[str, object], ...] = (),
    warnings: tuple[str, ...] = (),
) -> AssetDiscoveryResult:
    """Group accepted asset candidates by kind."""

    return AssetDiscoveryResult(
        parent_url=parent_url,
        images=tuple(item for item in candidates if item.kind == "image"),
        audio=tuple(item for item in candidates if item.kind == "audio"),
        video=tuple(item for item in candidates if item.kind == "video"),
        documents=tuple(
            item for item in candidates if item.kind == "document"
        ),
        rejected=rejected,
        warnings=warnings,
    )


def parent_text_metadata(
    *,
    parent_text: str | None,
    parent_title: str | None,
) -> dict[str, object]:
    text = clean_string(parent_text)
    payload: dict[str, object] = {}
    title = clean_string(parent_title)

    if title is not None:
        payload["parent_title"] = title

    if text is not None:
        payload.update(
            {
                "parent_text_hash": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "parent_text_preview": text[:500],
            }
        )

    return payload


def clean_string(value: object) -> str | None:
    if value is None:
        return None

    text = " ".join(str(value).split())
    return text or None


def as_optional_str(value: object) -> str | None:
    return clean_string(value)


def as_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _add_if_present(
    payload: dict[str, object],
    key: str,
    value: object,
) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    payload[key] = value
