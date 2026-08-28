"""Artifact writer for extracted curated text and markdown payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class CuratedArtifactWriter:
    """
    Persist extracted textual artifacts under curated snapshot directories.
    """

    def __init__(
        self, *, snapshot_directory: Path, artifacts_directory: str
    ) -> None:
        self._snapshot_directory = snapshot_directory
        self._artifacts_root = snapshot_directory / artifacts_directory
        self._text_root = self._artifacts_root / "text" / "plain"
        self._markdown_root = self._artifacts_root / "text" / "markdown"
        self._text_root.mkdir(parents=True, exist_ok=True)
        self._markdown_root.mkdir(parents=True, exist_ok=True)

    def write_text(self, *, object_id: str, text: str) -> Path:
        """
        Persist canonical plain text and return path relative to snapshot.
        """

        absolute_path = self._text_root / f"{object_id}.txt"
        absolute_path.write_bytes(text.encode("utf-8"))
        return absolute_path.relative_to(self._snapshot_directory)

    def write_markdown(self, *, object_id: str, markdown: str) -> Path:
        """Persist canonical markdown and return path relative to snapshot."""

        absolute_path = self._markdown_root / f"{object_id}.md"
        absolute_path.write_bytes(markdown.encode("utf-8"))
        return absolute_path.relative_to(self._snapshot_directory)
