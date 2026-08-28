from __future__ import annotations

import json
from pathlib import Path

import pytest

from crawler.governance.deletion_index import (
    ensure_asset_trainable,
)
from evaluator.leakage.report import generate_report, load_report


class DeletionIndexEntry:
    """Test helper mirroring the deletion index entry structure."""

    def __init__(
        self,
        source_object_id: str,
        object_sha256: str,
        sample_ids: tuple[str, ...],
        snapshot_id: str,
        run_ids: tuple[str, ...],
        checkpoint_sha256s: tuple[str, ...],
        release_ids: tuple[str, ...],
        revoked: bool,
        trainable_for_new_snapshots: bool,
        unlearned: bool = False,
    ) -> None:
        self.source_object_id = source_object_id
        self.object_sha256 = object_sha256
        self.sample_ids = sample_ids
        self.snapshot_id = snapshot_id
        self.run_ids = run_ids
        self.checkpoint_sha256s = checkpoint_sha256s
        self.release_ids = release_ids
        self.revoked = revoked
        self.trainable_for_new_snapshots = trainable_for_new_snapshots
        self.unlearned = unlearned


def append_deletion_index(
    *,
    dataset_root: Path,
    entry: DeletionIndexEntry,
    active_release_ids: frozenset[str] = frozenset(),
) -> Path:
    """Append immutable lineage evidence; active release artifacts remain protected."""
    if (
        not entry.source_object_id
        or not entry.object_sha256
        or len(entry.object_sha256) != 64
    ):
        raise ValueError(
            "explicit source object identity and SHA-256 are required"
        )
    if entry.revoked and entry.trainable_for_new_snapshots:
        raise ValueError(
            "revoked assets cannot remain trainable for new snapshots"
        )
    if entry.unlearned:
        raise ValueError(
            "retention cannot claim that model data was unlearned"
        )
    protected = sorted(active_release_ids.intersection(entry.release_ids))
    payload = {
        "schema_version": "1.0",
        "source_object_id": entry.source_object_id,
        "object_sha256": entry.object_sha256.lower(),
        "sample_ids": list(entry.sample_ids),
        "snapshot_id": entry.snapshot_id,
        "run_ids": list(entry.run_ids),
        "checkpoint_sha256s": list(entry.checkpoint_sha256s),
        "release_ids": list(entry.release_ids),
        "revoked": entry.revoked,
        "trainable_for_new_snapshots": entry.trainable_for_new_snapshots,
        "unlearned": False,
        "protected_active_release_ids": protected,
    }
    path = dataset_root / "deletion_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_leakage_coverage_is_modality_aware(tmp_path: Path) -> None:
    left = [
        {
            "sample_id": "l",
            "partition": "train",
            "dataset_id": "d",
            "lineage_key": "l",
            "modality": "text",
            "content_hash": "a",
            "content_fingerprints": {
                "normalized_text_sha256": "x",
                "text_shingle_profile": ["a", "b"],
            },
        }
    ]
    right = [
        {
            "sample_id": "r",
            "partition": "test",
            "dataset_id": "b",
            "lineage_key": "r",
            "modality": "text",
            "content_hash": "b",
            "content_fingerprints": {
                "normalized_text_sha256": "y",
                "text_shingle_profile": ["c", "d"],
            },
        }
    ]
    path = tmp_path / "leakage.json"
    report = generate_report(
        left_records=left,
        right_records=right,
        output_path=path,
        minimum_coverage={},
    )

    assert (
        report.to_dict()["coverage"]["audio_chromaprint"]["left"]["eligible"]
        == 0
    )
    assert load_report(path) == report


def test_tampered_leakage_coverage_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "coverage": {
                    "x": {
                        "left": {
                            "eligible": 1,
                            "with_evidence": 0,
                            "ratio": 1.0,
                        },
                        "right": {
                            "eligible": 0,
                            "with_evidence": 0,
                            "ratio": 1.0,
                        },
                        "minimum": 1.0,
                    }
                },
                "overlaps": [],
                "violations": [],
                "passed": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_report(path)


def test_revoked_asset_is_blocked(tmp_path: Path) -> None:
    digest = "a" * 64
    path = append_deletion_index(
        dataset_root=tmp_path,
        entry=DeletionIndexEntry(
            "source",
            digest,
            ("s",),
            "snap",
            ("run",),
            (),
            (),
            True,
            False,
        ),
    )

    with pytest.raises(PermissionError):
        ensure_asset_trainable(
            deletion_index_path=path,
            object_sha256=digest,
        )
