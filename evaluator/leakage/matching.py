"""Leakage overlap sinks and category-specific matchers."""

from __future__ import annotations

import json
from typing import TextIO

from evaluator.leakage.contracts import (
    ALGORITHMS,
    CATEGORIES,
    EXACT_CATEGORIES,
)
from evaluator.leakage.indexing import SideIndex, sha256_text
from evaluator.leakage.schema import (
    LeakageFingerprintReference,
    LeakageOverlap,
)
from mmcrawler_datasets.similarity.text import text_shingle_similarity


class OverlapSink:
    def __init__(
        self,
        *,
        sample_limit: int,
        detail_handle: TextIO | None,
    ) -> None:
        self._sample_limit = sample_limit
        self._detail_handle = detail_handle
        self.counts = {category: 0 for category in CATEGORIES}
        self.sample: list[LeakageOverlap] = []

    def add(self, overlap: LeakageOverlap) -> None:
        self.counts[overlap.category] += 1
        self.record(overlap)

    def add_count(self, *, category: str, count: int) -> None:
        self.counts[category] += count

    @property
    def sample_full(self) -> bool:
        return len(self.sample) >= self._sample_limit

    @property
    def writes_detail(self) -> bool:
        return self._detail_handle is not None

    def record(self, overlap: LeakageOverlap) -> None:
        if len(self.sample) < self._sample_limit:
            self.sample.append(overlap)
        if self._detail_handle is not None:
            self._detail_handle.write(
                json.dumps(
                    overlap.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def intersect_exact(
    *,
    left: SideIndex,
    right: SideIndex,
    sink: OverlapSink,
) -> None:
    for category in EXACT_CATEGORIES:
        left_values = left.values[category]
        right_values = right.values[category]
        for value in sorted(left_values.keys() & right_values.keys()):
            reference = LeakageFingerprintReference(
                category=category,
                algorithm=ALGORITHMS[category],
                normalized_digest=value,
            )
            left_keys = left_values[value]
            right_keys = right_values[value]
            sink.add_count(
                category=category,
                count=len(left_keys) * len(right_keys),
            )
            if sink.sample_full and not sink.writes_detail:
                continue
            stop = False
            for left_key in left_keys:
                for right_key in right_keys:
                    sink.record(
                        LeakageOverlap(
                            category=category,
                            left=left.identities[left_key],
                            right=right.identities[right_key],
                            fingerprint=reference,
                        )
                    )
                    if sink.sample_full and not sink.writes_detail:
                        stop = True
                        break
                if stop:
                    break


def intersect_near_text(
    *,
    left: SideIndex,
    right: SideIndex,
    sink: OverlapSink,
    threshold: float,
    max_candidates: int,
) -> bool:
    inverted: dict[str, set[tuple[str, str, str, str]]] = {}
    for right_key, profile in right.profiles.items():
        for shingle in profile:
            inverted.setdefault(shingle, set()).add(right_key)
    overflowed = False
    for left_key, left_profile in left.profiles.items():
        candidates: set[tuple[str, str, str, str]] = set()
        for shingle in left_profile:
            candidates.update(inverted.get(shingle, ()))
            if len(candidates) > max_candidates:
                overflowed = True
                break
        if len(candidates) > max_candidates:
            continue
        for right_key in sorted(candidates):
            right_profile = right.profiles[right_key]
            score = text_shingle_similarity(left_profile, right_profile)
            if score < threshold:
                continue
            common = tuple(sorted(set(left_profile) & set(right_profile)))
            sink.add(
                LeakageOverlap(
                    category="near_duplicate_text",
                    left=left.identities[left_key],
                    right=right.identities[right_key],
                    fingerprint=LeakageFingerprintReference(
                        category="near_duplicate_text",
                        algorithm=ALGORITHMS["near_duplicate_text"],
                        normalized_digest=sha256_text("\n".join(common)),
                        metric_name="similarity",
                        metric_value=float(score),
                    ),
                )
            )
    return overflowed


def intersect_perceptual(
    *,
    category: str,
    left: SideIndex,
    right: SideIndex,
    sink: OverlapSink,
    max_distance: int,
    max_candidates: int,
) -> bool:
    band_count = max_distance + 1
    buckets: dict[
        tuple[int, int],
        set[tuple[str, str, str, str]],
    ] = {}
    for right_key, values in right.perceptual[category].items():
        for value in values:
            for band in hash_bands(value, band_count=band_count):
                buckets.setdefault(band, set()).add(right_key)
    overflowed = False
    for left_key, left_values in left.perceptual[category].items():
        candidates: set[tuple[str, str, str, str]] = set()
        for value in left_values:
            for band in hash_bands(value, band_count=band_count):
                candidates.update(buckets.get(band, ()))
                if len(candidates) > max_candidates:
                    overflowed = True
                    break
        if len(candidates) > max_candidates:
            continue
        for right_key in sorted(candidates):
            right_values = right.perceptual[category][right_key]
            best = min(
                (
                    (
                        hamming_distance(left_value, right_value),
                        left_value,
                        right_value,
                    )
                    for left_value in left_values
                    for right_value in right_values
                ),
                default=None,
            )
            if best is None or best[0] > max_distance:
                continue
            distance, left_value, right_value = best
            sink.add(
                LeakageOverlap(
                    category=category,
                    left=left.identities[left_key],
                    right=right.identities[right_key],
                    fingerprint=LeakageFingerprintReference(
                        category=category,
                        algorithm=ALGORITHMS[category],
                        normalized_digest=sha256_text(
                            f"{left_value}:{right_value}"
                        ),
                        metric_name="hamming_distance",
                        metric_value=distance,
                    ),
                )
            )
    return overflowed


def hash_bands(
    value: str,
    *,
    band_count: int,
) -> tuple[tuple[int, int], ...]:
    try:
        number = int(value, 16)
    except ValueError:
        return ()
    bit_count = max(1, len(value) * 4)
    band_width = max(1, (bit_count + band_count - 1) // band_count)
    mask = (1 << band_width) - 1
    return tuple(
        (index, (number >> (index * band_width)) & mask)
        for index in range(band_count)
    )


def hamming_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 10**9
