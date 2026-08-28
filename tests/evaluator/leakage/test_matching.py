from __future__ import annotations

from evaluator.leakage.indexing import build_index
from evaluator.leakage.matching import (
    OverlapSink,
    hamming_distance,
    hash_bands,
    intersect_exact,
    intersect_near_text,
    intersect_perceptual,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": "dataset",
        "sample_id": "sample",
        "partition": "train",
        "lineage_key": "lineage",
        "modality": "text",
        "content_hash": "a" * 64,
    }
    row.update(overrides)
    return row


def test_intersect_exact_finds_shared_content_hash() -> None:
    shared_hash = "c" * 64
    left = build_index(
        [_row(dataset_id="left", content_hash=shared_hash)],
        max_records=10,
    )
    right = build_index(
        [
            _row(
                dataset_id="right",
                sample_id="sample-2",
                lineage_key="lineage-2",
                content_hash=shared_hash,
            )
        ],
        max_records=10,
    )
    sink = OverlapSink(sample_limit=10, detail_handle=None)

    intersect_exact(left=left, right=right, sink=sink)

    assert sink.counts["content_hash"] == 1
    assert len(sink.sample) == 1
    assert sink.sample[0].category == "content_hash"


def test_intersect_exact_ignores_disjoint_values() -> None:
    left = build_index(
        [_row(dataset_id="left", content_hash="a" * 64)],
        max_records=10,
    )
    right = build_index(
        [
            _row(
                dataset_id="right",
                sample_id="sample-2",
                lineage_key="lineage-2",
                content_hash="b" * 64,
            )
        ],
        max_records=10,
    )
    sink = OverlapSink(sample_limit=10, detail_handle=None)

    intersect_exact(left=left, right=right, sink=sink)

    assert sink.counts["content_hash"] == 0
    assert sink.sample == []


def test_intersect_near_text_matches_similar_profiles() -> None:
    fingerprints = {"text_shingle_profile": ["shingle-a", "shingle-b"]}
    left = build_index(
        [
            _row(
                dataset_id="left",
                modality="text",
                content_fingerprints=fingerprints,
            )
        ],
        max_records=10,
    )
    right = build_index(
        [
            _row(
                dataset_id="right",
                sample_id="sample-2",
                lineage_key="lineage-2",
                modality="text",
                content_fingerprints=fingerprints,
            )
        ],
        max_records=10,
    )
    sink = OverlapSink(sample_limit=10, detail_handle=None)

    overflowed = intersect_near_text(
        left=left,
        right=right,
        sink=sink,
        threshold=0.5,
        max_candidates=10,
    )

    assert overflowed is False
    assert sink.counts["near_duplicate_text"] == 1


def test_intersect_perceptual_matches_within_distance() -> None:
    left = build_index(
        [
            _row(
                dataset_id="left",
                modality="image",
                content_fingerprints={"image_phash": "0" * 16},
            )
        ],
        max_records=10,
    )
    right = build_index(
        [
            _row(
                dataset_id="right",
                sample_id="sample-2",
                lineage_key="lineage-2",
                modality="image",
                content_fingerprints={"image_phash": "1" * 16},
            )
        ],
        max_records=10,
    )
    sink = OverlapSink(sample_limit=10, detail_handle=None)

    overflowed = intersect_perceptual(
        category="image_phash",
        left=left,
        right=right,
        sink=sink,
        max_distance=64,
        max_candidates=10,
    )

    assert overflowed is False
    assert sink.counts["image_phash"] == 1


def test_hamming_distance_counts_differing_bits() -> None:
    assert hamming_distance("0" * 16, "0" * 16) == 0
    assert hamming_distance("0", "1") == 1


def test_hamming_distance_returns_large_value_for_invalid_hex() -> None:
    assert hamming_distance("zz", "00") == 10**9


def test_hash_bands_produces_requested_band_count() -> None:
    bands = hash_bands("abcd", band_count=4)

    assert len(bands) == 4
    assert all(isinstance(band, tuple) for band in bands)


def test_hash_bands_returns_empty_for_invalid_hex() -> None:
    assert hash_bands("not-hex", band_count=4) == ()
