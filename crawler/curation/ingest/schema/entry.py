"""Raw manifest entry binding parsed record and source run directory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from crawler.curation.ingest.schema.record import RawManifestRecord


@dataclass(frozen=True, slots=True)
class RawManifestEntry:
    """Single manifest record with run-path provenance."""

    run_directory: Path
    record: RawManifestRecord
