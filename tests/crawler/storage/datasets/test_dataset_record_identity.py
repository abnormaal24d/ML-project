"""Fetch record ids must respect the canonical media identity."""

from __future__ import annotations

from crawler.storage.datasets.records.dataset_record import (
    derive_fetch_record_id,
)


def test_media_record_identity_prevents_same_url_cross_kind_collisions() -> (
    None
):
    image_id = derive_fetch_record_id(
        run_id="run-1",
        record_identity="image:https://example.test/asset",
        record_version=1,
    )
    video_id = derive_fetch_record_id(
        run_id="run-1",
        record_identity="video:https://example.test/asset",
        record_version=1,
    )
    retry_id = derive_fetch_record_id(
        run_id="run-1",
        record_identity="image:https://example.test/asset",
        record_version=2,
    )

    assert image_id != video_id
    assert image_id != retry_id
