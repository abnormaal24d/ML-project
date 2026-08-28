"""Module implementation for the crawler runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class FetchedPayload:
    """Immutable reference to a fetched response payload persisted to disk.

    This object exists to decouple fetch metadata from the physical payload
    storage. The fetch layer is responsible for downloading and persisting the
    payload once; downstream processing layers consume this reference instead
    of
    keeping the whole response body in memory.
    """

    temp_path: Path
    byte_size: int
    sha256_hex: str
    sniff_bytes: bytes
    chunk_count: int
    truncated: bool = False
    source_content_length: int | None = None
    fetch_mode: str = "full"
    is_complete_payload: bool = True
    observed_bytes: int | None = None
    duration_seconds: float | None = None

    def exists(self) -> bool:
        """Return whether the persisted payload file currently exists."""
        return self.temp_path.exists()

    def cleanup(self) -> None:
        """Remove the temporary payload file and its derived scratch siblings."""
        self.temp_path.unlink(missing_ok=True)
        remove_derived_scratch_siblings(self.temp_path)

    def read_bytes(self) -> bytes:
        """Read the complete persisted payload bytes."""
        return self.temp_path.read_bytes()


def remove_derived_scratch_siblings(path: Path) -> None:
    """Remove derived artifact scratch files written next to a payload.

    Handlers write normalized media (``<name>.normalized.*``) and extracted
    audio tracks (``<stem>_audio_*.wav``) as siblings of the temporary fetch
    file. These scratch files are never persisted as-is; removing them with
    the payload keeps the temporary area clean.
    """

    try:
        parent = path.parent
        normalized_prefix = f"{path.name}.normalized"
        audio_prefix = f"{path.stem}_audio_"
        for candidate in parent.iterdir():
            name = candidate.name
            if name.startswith(normalized_prefix) or (
                name.startswith(audio_prefix) and name.endswith(".wav")
            ):
                candidate.unlink(missing_ok=True)
    except OSError:
        pass
