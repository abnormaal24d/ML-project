from __future__ import annotations

import pytest

from evaluator.leakage.indexing import (
    build_index,
    is_perceptual_hash,
    is_sha256_digest,
    sha256_text,
    url_hash,
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


def test_build_index_requires_canonical_modality_field() -> None:
    row = _row()
    row.pop("modality")

    with pytest.raises(ValueError, match="canonical 'modality'"):
        build_index((row,), max_records=10)


def test_build_index_rejects_duplicate_identities() -> None:
    row = _row()

    with pytest.raises(ValueError, match="duplicate leakage identity"):
        build_index((row, dict(row)), max_records=10)


def test_build_index_tracks_eligibility_and_evidence() -> None:
    row = _row(content_hash="b" * 64)

    index = build_index((row,), max_records=10)

    assert index.eligible["content_hash"] == 1
    assert index.with_evidence["content_hash"] == 1
    identity_key = next(iter(index.identities))
    assert ("b" * 64) in index.values["content_hash"]
    assert index.values["content_hash"]["b" * 64] == [identity_key]


def test_document_page_hashes_count_as_image_phash_evidence() -> None:
    index = build_index(
        [
            _row(
                modality="document",
                content_fingerprints={
                    "document_page_phashes": ["0" * 16],
                },
            )
        ],
        max_records=10,
    )

    assert index.eligible["image_phash"] == 1
    assert index.with_evidence["image_phash"] == 1


def test_build_index_overflows_when_max_records_exceeded() -> None:
    rows = [
        _row(
            sample_id=f"sample-{i}",
            lineage_key=f"lineage-{i}",
            content_hash=format(i, "064x"),
        )
        for i in range(3)
    ]

    index = build_index(rows, max_records=1)

    assert "content_hash" in index.overflowed_categories
    assert index.stored_records["content_hash"] == 1


def test_is_sha256_digest_validates_length_and_hex() -> None:
    assert is_sha256_digest("a" * 64) is True
    assert is_sha256_digest("a" * 63) is False
    assert is_sha256_digest("z" * 64) is False


def test_is_perceptual_hash_validates_length_and_hex() -> None:
    assert is_perceptual_hash("0" * 16) is True
    assert is_perceptual_hash("0" * 15) is False
    assert is_perceptual_hash("z" * 16) is False


def test_url_hash_is_scheme_and_default_port_normalized() -> None:
    plain = url_hash("https://Example.com:443/path?q=1")
    equivalent = url_hash("https://example.com/path?q=1")

    assert plain == equivalent
    assert plain == sha256_text("https://example.com/path?q=1")


def test_url_hash_scheme_agnostic_ignores_scheme_difference() -> None:
    https_hash = url_hash("https://example.com/path", scheme_agnostic=True)
    http_hash = url_hash("http://example.com/path", scheme_agnostic=True)

    assert https_hash == http_hash


def test_url_hash_returns_none_for_invalid_url() -> None:
    assert url_hash("not a url") is None
    assert url_hash(None) is None
