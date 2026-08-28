"""Shared JSONL file writer used by curated and training builders."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class JsonlWriter:
    """Write iterable dictionaries to newline-delimited JSON files."""

    def write(self, *, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        """Persist rows into JSONL with deterministic key ordering."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
