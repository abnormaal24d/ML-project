"""Response body read-plan schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BodyReadPlan:
    """Describe how a fetch response body should be requested and read.

    The fetch executor only needs the resulting plan. Header mutation, byte
    budgeting, range probes, and partial-read acceptance live with the request
    planning collaborators instead of the transport orchestration class.
    """

    headers: dict[str, str]
    max_bytes: int
    allow_partial: bool
    mode: str
    probe_bytes: int | None = None
    resume_partial_path: Path | None = None
    resume_offset: int | None = None
    resume_owner_token: str | None = None
    resume_etag: str | None = None
    resume_last_modified: str | None = None

    @property
    def expects_range_response(self) -> bool:
        """Return whether the plan expects a HTTP 206 range response."""

        return self.mode in {
            "metadata_probe",
            "metadata_only",
            "fetch_partial",
            "resume_partial",
        }
