"""Deletion index governance for asset trainability checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def ensure_asset_trainable(
    *,
    deletion_index_path: Path,
    object_sha256: str,
    max_index_bytes: int = 1_073_741_824,
    max_index_rows: int = 1_000_000,
    max_row_bytes: int = 8_388_608,
) -> None:
    """Fail closed when an asset is revoked or its deletion evidence is malformed.

    Args:
        deletion_index_path: Path to the deletion index JSONL file.
        object_sha256: SHA-256 hash of the object to check.
        max_index_bytes: Maximum allowed size of the deletion index file.
        max_index_rows: Maximum allowed number of rows in the deletion index.
        max_row_bytes: Maximum allowed size of a single row.
    """
    if not deletion_index_path.is_file():
        return

    size = deletion_index_path.stat().st_size
    if size > max_index_bytes:
        raise ValueError(
            f"deletion index exceeds byte limit: {deletion_index_path}"
        )

    try:
        lines = deletion_index_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("deletion index is unreadable") from exc

    for number, line in enumerate(lines, 1):
        if number > max_index_rows:
            raise ValueError("deletion index exceeds row limit")
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > max_row_bytes:
            raise ValueError("deletion index row exceeds byte limit")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid deletion index row {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"invalid deletion index row {number}: expected JSON object"
            )
        evidence_sha256 = row.get("object_sha256")
        revoked = row.get("revoked")
        trainable = row.get("trainable_for_new_snapshots")
        if not isinstance(
            evidence_sha256, str
        ) or not _SHA256_PATTERN.fullmatch(evidence_sha256):
            raise ValueError(
                f"invalid deletion index row {number}: invalid SHA-256"
            )
        if type(revoked) is not bool or type(trainable) is not bool:
            raise ValueError(
                f"invalid deletion index row {number}: governance flags "
                "must be booleans"
            )
        if revoked and trainable:
            raise ValueError(
                f"invalid deletion index row {number}: revoked assets "
                "cannot be trainable"
            )
        if evidence_sha256.lower() == object_sha256.lower() and (
            revoked or not trainable
        ):
            raise PermissionError(
                f"asset blocked by deletion index: {object_sha256}"
            )
