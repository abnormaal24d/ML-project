from __future__ import annotations

import json
from pathlib import Path

from datachecker.inventory.curated_snapshot_inventory import (
    CuratedInventoryReader,
)
from schemas.versions import CURATED_DATASET_SCHEMA_VERSION


def test_curated_inventory_rejects_malformed_jsonl_records(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": CURATED_DATASET_SCHEMA_VERSION,
        "final": True,
        "status": "completed",
        "snapshot_id": "snapshot-1",
        "documents": 1,
        "chunks": 0,
        "images": 0,
        "audio": 0,
        "video": 0,
        "alignments": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    documents_path = tmp_path / "documents.jsonl"
    documents_path.write_text("{not-json}\n", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    valid = CuratedInventoryReader._curated_schema_valid(
        directory=tmp_path,
        manifest_path=manifest_path,
        documents_path=documents_path,
        chunks_path=chunks_path,
        images_path=None,
        audio_path=None,
        video_path=None,
        alignments_path=None,
        manifest_payload=manifest,
    )

    assert valid is False
