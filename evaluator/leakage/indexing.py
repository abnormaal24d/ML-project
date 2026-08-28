"""Streaming leakage fingerprint extraction and bounded indexes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from evaluator.leakage.contracts import (
    CATEGORIES,
    EXACT_CATEGORIES,
    PERCEPTUAL_CATEGORIES,
    PERCEPTUAL_HASH_HEX_LENGTH,
    SHA256_EVIDENCE_CATEGORIES,
)
from evaluator.leakage.schema import (
    LeakageIdentity,
    optional_text,
    required_text,
)


@dataclass(slots=True)
class SideIndex:
    seen_identities: set[tuple[str, str, str, str]]
    identities: dict[tuple[str, str, str, str], LeakageIdentity]
    values: dict[
        str,
        dict[str, list[tuple[str, str, str, str]]],
    ]
    profiles: dict[
        tuple[str, str, str, str],
        tuple[str, ...],
    ]
    perceptual: dict[
        str,
        dict[tuple[str, str, str, str], tuple[str, ...]],
    ]
    eligible: dict[str, int]
    with_evidence: dict[str, int]
    stored_records: dict[str, int]
    overflowed_categories: set[str]
    family_splits: dict[str, set[str]]

    @classmethod
    def empty(cls) -> SideIndex:
        return cls(
            seen_identities=set(),
            identities={},
            values={category: {} for category in EXACT_CATEGORIES},
            profiles={},
            perceptual={category: {} for category in PERCEPTUAL_CATEGORIES},
            eligible={category: 0 for category in CATEGORIES},
            with_evidence={category: 0 for category in CATEGORIES},
            stored_records={category: 0 for category in CATEGORIES},
            overflowed_categories=set(),
            family_splits={},
        )


def build_index(
    records: Iterable[Mapping[str, object]],
    *,
    max_records: int,
) -> SideIndex:
    index = SideIndex.empty()
    for row in records:
        identity = identity_from_row(row)
        if identity.key in index.seen_identities:
            raise ValueError(
                "duplicate leakage identity:"
                f"{identity.dataset_id}:{identity.sample_id}"
            )
        index.seen_identities.add(identity.key)
        index.family_splits.setdefault(identity.lineage_key, set()).add(
            identity.partition
        )
        identity_stored = False
        for category in CATEGORIES:
            if not is_eligible(row, category):
                continue
            index.eligible[category] += 1
            values = validated_values(
                category=category,
                values=extract_values(row, category),
            )
            if not values:
                continue
            index.with_evidence[category] += 1
            if index.stored_records[category] >= max_records:
                index.overflowed_categories.add(category)
                continue
            if not identity_stored:
                index.identities[identity.key] = identity
                identity_stored = True
            index.stored_records[category] += 1
            if category in EXACT_CATEGORIES:
                for value in values:
                    index.values[category].setdefault(value, []).append(
                        identity.key
                    )
            elif category == "near_duplicate_text":
                index.profiles[identity.key] = values
            else:
                index.perceptual[category][identity.key] = values
    return index


def indexed_content_family_violations(
    *,
    left: SideIndex,
    right: SideIndex,
) -> tuple[str, ...]:
    """Return cross-split families already captured by the side indexes."""

    family_splits = {
        family: set(splits) for family, splits in left.family_splits.items()
    }
    for family, splits in right.family_splits.items():
        family_splits.setdefault(family, set()).update(splits)
    return tuple(
        f"content_family_cross_split:{family}:{','.join(sorted(splits))}"
        for family, splits in sorted(family_splits.items())
        if len(splits) > 1
    )


def identity_from_row(row: Mapping[str, object]) -> LeakageIdentity:
    return LeakageIdentity(
        dataset_id=required_text(row, "dataset_id"),
        sample_id=required_text(row, "sample_id"),
        partition=required_text(row, "partition"),
        lineage_key=required_text(row, "lineage_key"),
    )


def fingerprints_from_row(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("content_fingerprints")
    return value if isinstance(value, Mapping) else {}


def modality_from_row(row: Mapping[str, object]) -> str:
    value = row.get("modality")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "leakage record requires a non-empty canonical 'modality' field"
        )
    return value.strip().casefold()


def output_modalities_from_row(row: Mapping[str, object]) -> frozenset[str]:
    target = row.get("task_target")
    if not isinstance(target, Mapping):
        return frozenset()
    raw = target.get("output_modalities")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(str(item).strip().casefold() for item in raw)


def is_eligible(row: Mapping[str, object], category: str) -> bool:
    modalities = {modality_from_row(row), *output_modalities_from_row(row)}
    if category in {
        "canonical_url_sha256",
        "scheme_agnostic_url_sha256",
    }:
        return optional_text(row.get("source_url")) is not None
    if category == "content_hash":
        return True
    if category == "object_sha256":
        objects = row.get("objects")
        return (
            bool(isinstance(objects, (list, tuple)) and objects)
            or optional_text(row.get("object_sha256")) is not None
        )
    if category in {
        "emitted_text_sha256",
        "normalized_text_sha256",
        "near_duplicate_text",
    }:
        return bool(
            modalities & {"text", "document", "image", "audio", "video"}
        )
    if category == "image_phash":
        # Curated documents can carry one pHash per rendered page.  Treat
        # those pages as image evidence instead of silently reporting zero
        # perceptual-image coverage for every document record.
        return bool(modalities & {"image", "document"})
    if category == "image_dhash":
        return "image" in modalities
    if category == "audio_chromaprint":
        return "audio" in modalities or row.get("has_audio") is True
    if category == "video_keyframe_sequence":
        return "video" in modalities
    if category == "document_layout_sha256":
        return "document" in modalities
    return False


def extract_values(
    row: Mapping[str, object],
    category: str,
) -> tuple[str, ...]:
    fingerprints = fingerprints_from_row(row)
    if category == "canonical_url_sha256":
        direct = optional_text(row.get(category))
        value = direct or url_hash(row.get("source_url"))
        return (value,) if value else ()
    if category == "scheme_agnostic_url_sha256":
        direct = optional_text(row.get(category))
        value = direct or url_hash(
            row.get("source_url"),
            scheme_agnostic=True,
        )
        return (value,) if value else ()
    if category == "object_sha256":
        object_values: list[str] = []
        objects = row.get("objects")
        if isinstance(objects, (list, tuple)):
            for item in objects:
                if not isinstance(item, Mapping):
                    continue
                value = optional_text(item.get("object_sha256"))
                if value:
                    object_values.append(value.casefold())
        direct = optional_text(row.get("object_sha256"))
        if direct:
            object_values.append(direct.casefold())
        return tuple(dict.fromkeys(object_values))
    if category == "content_hash":
        value = optional_text(row.get("content_hash"))
        return (value.casefold(),) if value else ()
    if category in {"emitted_text_sha256", "normalized_text_sha256"}:
        value = optional_text(fingerprints.get(category))
        return (value.casefold(),) if value else ()
    if category == "near_duplicate_text":
        raw = fingerprints.get("text_shingle_profile")
        if not isinstance(raw, (list, tuple)):
            return ()
        return tuple(sorted({str(item) for item in raw if str(item)}))
    if category == "image_phash":
        image_values: list[str] = []
        direct_image_hash = optional_text(fingerprints.get("image_phash"))
        if direct_image_hash is not None:
            image_values.append(direct_image_hash)
        page_hashes = fingerprints.get("document_page_phashes")
        if isinstance(page_hashes, (list, tuple)):
            image_values.extend(
                value
                for item in page_hashes
                if (value := optional_text(item)) is not None
            )
        return tuple(dict.fromkeys(value.casefold() for value in image_values))
    if category == "image_dhash":
        value = optional_text(fingerprints.get("image_dhash"))
        return (value.casefold(),) if value else ()
    if category == "audio_chromaprint":
        value = optional_text(fingerprints.get("audio_chromaprint"))
        return (value,) if value else ()
    if category == "video_keyframe_sequence":
        raw = fingerprints.get("video_keyframe_phashes")
        if not isinstance(raw, (list, tuple)):
            return ()
        normalized = tuple(
            value.casefold() for item in raw if (value := optional_text(item))
        )
        return (sha256_text("|".join(normalized)),) if normalized else ()
    if category == "document_layout_sha256":
        value = optional_text(fingerprints.get(category))
        return (value.casefold(),) if value else ()
    return ()


def validated_values(
    *,
    category: str,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """Return only structurally valid evidence for one leakage category."""

    if category in SHA256_EVIDENCE_CATEGORIES:
        return tuple(value for value in values if is_sha256_digest(value))
    if category in PERCEPTUAL_CATEGORIES:
        return tuple(value for value in values if is_perceptual_hash(value))
    if category in {"near_duplicate_text", "audio_chromaprint"}:
        return tuple(value for value in values if value.strip())
    return ()


def is_sha256_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def is_perceptual_hash(value: str) -> bool:
    if len(value) != PERCEPTUAL_HASH_HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def url_hash(
    value: object,
    scheme_agnostic: bool = False,
) -> str | None:
    text = optional_text(value)
    if text is None:
        return None
    try:
        parsed = urlsplit(text)
        host = (parsed.hostname or "").casefold()
        if not parsed.scheme or not host:
            return None
        scheme = "" if scheme_agnostic else parsed.scheme.casefold()
        port = parsed.port
        default_port = (
            parsed.scheme.casefold() == "http"
            and port == 80
            or parsed.scheme.casefold() == "https"
            and port == 443
        )
        netloc = host if port is None or default_port else f"{host}:{port}"
        normalized = urlunsplit(
            (scheme, netloc, parsed.path or "/", parsed.query, "")
        )
    except ValueError:
        return None
    return sha256_text(normalized)


def sha256_text(value: str) -> str:
    """Return the canonical lowercase SHA-256 digest for text evidence."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
