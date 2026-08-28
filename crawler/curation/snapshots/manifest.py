"""Curated snapshot manifest serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CuratedSnapshotManifest:
    """Write finalized curated snapshot manifests."""

    @classmethod
    def write(cls, path: Path, **payload: Any) -> None:
        snapshot_root = path.parent.resolve()

        normalized = {
            key: cls._normalize(
                value,
                snapshot_root=snapshot_root,
            )
            for key, value in payload.items()
        }

        normalized.setdefault("lifecycle_stage", "curated")
        normalized["status"] = "completed"
        normalized["final"] = True
        normalized.setdefault("immutable", True)

        path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def _normalize(cls, value: Any, *, snapshot_root: Path) -> Any:
        if isinstance(value, Path):
            resolved = (
                value.resolve()
                if value.is_absolute()
                else (snapshot_root / value).resolve()
            )

            try:
                relative = resolved.relative_to(snapshot_root)
            except ValueError as exc:
                raise ValueError(
                    "curated snapshot manifest path "
                    f"escapes snapshot root: {value}"
                ) from exc

            return relative.as_posix()

        if isinstance(value, tuple):
            return [
                cls._normalize(
                    item,
                    snapshot_root=snapshot_root,
                )
                for item in value
            ]

        if isinstance(value, list):
            return [
                cls._normalize(
                    item,
                    snapshot_root=snapshot_root,
                )
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): cls._normalize(
                    item,
                    snapshot_root=snapshot_root,
                )
                for key, item in value.items()
            }

        return value
