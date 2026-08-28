"""Dataset-level near-duplicate clustering for privacy-cleared text."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.environment.default_values import (
    DEFAULT_CACHE_ITEMS,
    DEFAULT_SHINGLE_CANDIDATE_BANDS,
    DEFAULT_SHINGLE_WIDTH,
)
from mmcrawler_datasets.similarity.text import to_token_shingles
from preprocessing.provenance import stable_identifier, stable_int_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class TextDuplicateAssignment:
    """One deterministic dataset-level near-duplicate assignment."""

    cluster_id: str
    is_duplicate: bool
    similarity: float


class NearTextDeduplicator:
    """Assign near-duplicate clusters without retaining cross-run state."""

    def __init__(
        self,
        *,
        threshold: float,
        shingle_width: int = DEFAULT_SHINGLE_WIDTH,
        candidate_bands: int = DEFAULT_SHINGLE_CANDIDATE_BANDS,
        use_buckets: bool = True,
        candidate_fallback_limit: int = 64,
        profile_cache_size: int = DEFAULT_CACHE_ITEMS,
        cluster_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self._threshold = max(0.0, min(1.0, threshold))
        self._shingle_width = max(1, shingle_width)
        self._candidate_bands = max(1, candidate_bands)
        self._use_buckets = use_buckets
        self._candidate_fallback_limit = max(0, candidate_fallback_limit)
        self._profile_cache_size = max(0, profile_cache_size)
        self._cluster_id_factory = cluster_id_factory or _default_cluster_id

    def assign_clusters(
        self,
        *,
        texts_by_document_id: Mapping[str, str],
    ) -> dict[str, TextDuplicateAssignment]:
        """Assign clusters using run-local state and stable document ordering."""

        index = _NearDuplicateIndex(
            threshold=self._threshold,
            shingle_width=self._shingle_width,
            candidate_bands=self._candidate_bands,
            use_buckets=self._use_buckets,
            candidate_fallback_limit=self._candidate_fallback_limit,
            profile_cache_size=self._profile_cache_size,
            cluster_id_factory=self._cluster_id_factory,
        )
        assignments: dict[str, TextDuplicateAssignment] = {}
        for document_id, text in sorted(texts_by_document_id.items()):
            assignments[document_id] = index.assign_document(
                document_id=document_id,
                text=text,
            )
        return assignments


class _NearDuplicateIndex:
    """Mutable index scoped to exactly one ``assign_clusters`` call."""

    def __init__(
        self,
        *,
        threshold: float,
        shingle_width: int,
        candidate_bands: int,
        use_buckets: bool,
        candidate_fallback_limit: int,
        profile_cache_size: int,
        cluster_id_factory: Callable[[str], str],
    ) -> None:
        self._threshold = threshold
        self._shingle_width = shingle_width
        self._candidate_bands = candidate_bands
        self._use_buckets = use_buckets
        self._candidate_fallback_limit = candidate_fallback_limit
        self._profile_cache_size = profile_cache_size
        self._cluster_id_factory = cluster_id_factory
        self._representatives: list[_Representative] = []
        self._bucket_index: dict[int, list[int]] = {}
        self._fingerprint_index: dict[str, list[int]] = {}
        self._profile_cache: OrderedDict[
            tuple[int, int, str], _ShingleProfile
        ] = OrderedDict()

    def assign_document(
        self,
        *,
        document_id: str,
        text: str,
    ) -> TextDuplicateAssignment:
        profile = self._profile_for_text(text=text)
        candidate_indexes = self._candidate_representative_indexes(
            profile=profile
        )
        best_index: int | None = None
        best_similarity = 0.0

        for representative_index in candidate_indexes:
            representative = self._representatives[representative_index]
            similarity = _jaccard(
                profile.tokens, representative.profile.tokens
            )
            if best_index is None or similarity > best_similarity:
                best_similarity = similarity
                best_index = representative_index

        if best_index is None or best_similarity < self._threshold:
            cluster_id = self._cluster_id_factory(document_id)
            representative_index = len(self._representatives)
            self._representatives.append(
                _Representative(cluster_id=cluster_id, profile=profile)
            )
            self._register_representative_profile(
                representative_index=representative_index,
                profile=profile,
            )
            return TextDuplicateAssignment(
                cluster_id=cluster_id,
                is_duplicate=False,
                similarity=1.0,
            )

        return TextDuplicateAssignment(
            cluster_id=self._representatives[best_index].cluster_id,
            is_duplicate=True,
            similarity=round(best_similarity, 4),
        )

    def _candidate_representative_indexes(
        self,
        *,
        profile: _ShingleProfile,
    ) -> Iterable[int]:
        if not self._representatives:
            return ()
        if not self._use_buckets or self._threshold <= 0.0:
            return range(len(self._representatives))

        indexes = dict.fromkeys(
            self._fingerprint_index.get(profile.digest, ())
        )
        for signal in profile.band_signals:
            for representative_index in self._bucket_index.get(signal, ()):
                indexes.setdefault(representative_index, None)

        if not indexes:
            if len(self._representatives) <= self._candidate_fallback_limit:
                raw_indexes: Iterable[int] = range(len(self._representatives))
            else:
                return ()
        else:
            raw_indexes = indexes.keys()

        return tuple(
            representative_index
            for representative_index in raw_indexes
            if _can_meet_threshold(
                left_size=profile.size,
                right_size=self._representatives[
                    representative_index
                ].profile.size,
                threshold=self._threshold,
            )
        )

    def _register_representative_profile(
        self,
        *,
        representative_index: int,
        profile: _ShingleProfile,
    ) -> None:
        self._fingerprint_index.setdefault(profile.digest, []).append(
            representative_index
        )
        if self._use_buckets:
            for signal in dict.fromkeys(profile.band_signals):
                self._bucket_index.setdefault(signal, []).append(
                    representative_index
                )

    def _profile_for_text(self, *, text: str) -> _ShingleProfile:
        cache_key = _profile_cache_key(text=text, width=self._shingle_width)
        cached_profile = self._profile_cache.get(cache_key)
        if cached_profile is not None:
            self._profile_cache.move_to_end(cache_key)
            return cached_profile

        profile = _build_shingle_profile(
            text=text,
            width=self._shingle_width,
            bands=self._candidate_bands,
        )
        if self._profile_cache_size > 0:
            self._profile_cache[cache_key] = profile
            if len(self._profile_cache) > self._profile_cache_size:
                self._profile_cache.popitem(last=False)
        return profile


@dataclass(frozen=True, slots=True)
class _ShingleProfile:
    tokens: frozenset[int]
    size: int
    band_signals: tuple[int, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class _Representative:
    cluster_id: str
    profile: _ShingleProfile


def _build_shingle_profile(
    *,
    text: str,
    width: int,
    bands: int,
) -> _ShingleProfile:
    tokens = frozenset(
        stable_int_hash(value=shingle)
        for shingle in to_token_shingles(text=text, width=width)
    )
    ordered_tokens = tuple(sorted(tokens))
    return _ShingleProfile(
        tokens=tokens,
        size=len(tokens),
        band_signals=_build_band_signals_from_ordered(
            ordered_tokens=ordered_tokens,
            bands=bands,
        ),
        digest=_digest_ordered_tokens(ordered_tokens=ordered_tokens),
    )


def _build_band_signals_from_ordered(
    *,
    ordered_tokens: tuple[int, ...],
    bands: int,
) -> tuple[int, ...]:
    if not ordered_tokens:
        return ()
    total = len(ordered_tokens)
    resolved_bands = max(1, min(bands, total))
    return tuple(
        ordered_tokens[min(total - 1, (band_index * total) // resolved_bands)]
        for band_index in range(resolved_bands)
    )


def _digest_ordered_tokens(*, ordered_tokens: tuple[int, ...]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for token in ordered_tokens:
        hasher.update(token.to_bytes(8, "big"))
    return hasher.hexdigest()


def _profile_cache_key(*, text: str, width: int) -> tuple[int, int, str]:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return (width, len(text), digest)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    smaller, larger = (
        (left, right) if len(left) <= len(right) else (right, left)
    )
    intersection = sum(1 for token in smaller if token in larger)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _can_meet_threshold(
    *,
    left_size: int,
    right_size: int,
    threshold: float,
) -> bool:
    if threshold <= 0.0:
        return True
    if left_size == 0 and right_size == 0:
        return True
    if left_size == 0 or right_size == 0:
        return False
    return min(left_size, right_size) / max(left_size, right_size) >= threshold


def _default_cluster_id(document_id: str) -> str:
    return stable_identifier(prefix="cluster", parts=(document_id,))


__all__ = ["NearTextDeduplicator", "TextDuplicateAssignment"]
