"""Shared validation utilities."""

from __future__ import annotations

from pathlib import Path


class ArtifactPathPresence:
    """Helper to check for physical artifact presence."""

    @staticmethod
    def missing(*paths: Path | None) -> tuple[str, ...]:
        """Return a tuple of missing paths formatted as strings."""
        missing = []
        for path in paths:
            if path is None:
                continue
            if not path.exists():
                missing.append(str(path))
        return tuple(missing)
