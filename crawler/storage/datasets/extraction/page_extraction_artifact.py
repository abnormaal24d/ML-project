"""Versioned page extraction sidecar write/read with integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PAGE_EXTRACTION_SCHEMA_VERSION = 1
_PAGE_EXTRACTION_RELATIVE_ROOT = Path("extraction") / "page"
_ENRICHMENT_ARTIFACT_KEY = "page_extraction_artifact"


class PageExtractionArtifactError(ValueError):
    """Raised when a page extraction artifact cannot be trusted."""


@dataclass(frozen=True, slots=True)
class PageExtractionArtifact:
    """Full page extraction payload stored as a versioned sidecar."""

    schema_version: int
    text: str
    markdown: str
    headings: tuple[str, ...]
    code_block_count: int
    boilerplate_ratio: float
    extraction_warnings: tuple[str, ...]
    title: str | None
    canonical_url: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "text": self.text,
            "markdown": self.markdown,
            "headings": list(self.headings),
            "code_block_count": self.code_block_count,
            "boilerplate_ratio": self.boilerplate_ratio,
            "extraction_warnings": list(self.extraction_warnings),
            "title": self.title,
            "canonical_url": self.canonical_url,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> PageExtractionArtifact:
        if not isinstance(payload, Mapping):
            raise PageExtractionArtifactError(
                "page extraction artifact must be a mapping"
            )
        schema_version = _require_int(
            payload.get("schema_version"),
            field_name="schema_version",
            minimum=1,
        )
        if schema_version != PAGE_EXTRACTION_SCHEMA_VERSION:
            raise PageExtractionArtifactError(
                f"unsupported page extraction schema_version: {schema_version}"
            )
        text = _require_str(payload.get("text"), field_name="text")
        markdown = _require_str(payload.get("markdown"), field_name="markdown")
        headings = _require_string_tuple(
            payload.get("headings"),
            field_name="headings",
        )
        warnings = _require_string_tuple(
            payload.get("extraction_warnings"),
            field_name="extraction_warnings",
        )
        code_block_count = _require_int(
            payload.get("code_block_count"),
            field_name="code_block_count",
            minimum=0,
        )
        boilerplate_ratio = _require_finite_float(
            payload.get("boilerplate_ratio"),
            field_name="boilerplate_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        return cls(
            schema_version=schema_version,
            text=text,
            markdown=markdown,
            headings=headings,
            code_block_count=code_block_count,
            boilerplate_ratio=boilerplate_ratio,
            extraction_warnings=warnings,
            title=_optional_string(payload.get("title")),
            canonical_url=_optional_string(payload.get("canonical_url")),
        )


@dataclass(frozen=True, slots=True)
class PreparedPageExtractionWrite:
    """Sidecar bytes prepared for a later transactional commit."""

    relative_path: str
    absolute_path: Path
    encoded_bytes: bytes
    sha256: str


class PageExtractionArtifactWriter:
    """Prepare and commit page extraction sidecars under a crawl run directory."""

    def __init__(self, *, run_directory: Path) -> None:
        self._run_directory = run_directory.resolve()

    def prepare(
        self,
        *,
        fetch_record_id: str,
        artifact: PageExtractionArtifact,
    ) -> PreparedPageExtractionWrite:
        """Serialize an artifact without writing it to disk yet."""

        safe_id = _safe_id(fetch_record_id)
        relative = (
            _PAGE_EXTRACTION_RELATIVE_ROOT / f"{safe_id}.json"
        ).as_posix()
        absolute = self._run_directory / relative
        encoded = json.dumps(
            artifact.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return PreparedPageExtractionWrite(
            relative_path=relative,
            absolute_path=absolute,
            encoded_bytes=encoded,
            sha256=digest,
        )

    def commit(self, *, prepared: PreparedPageExtractionWrite) -> None:
        """Persist a prepared artifact with fsync of file and parent dir."""

        absolute = prepared.absolute_path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        temp_path = absolute.with_suffix(absolute.suffix + ".tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(prepared.encoded_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, absolute)
            _fsync_directory(absolute.parent)
        finally:
            # Remove leftover temp when replace/fsync fails mid-commit.
            temp_path.unlink(missing_ok=True)


class PageExtractionArtifactReader:
    """Load and verify page extraction sidecars from a snapshot/run root."""

    def read(
        self,
        *,
        snapshot_directory: Path,
        relative_path: str,
        expected_sha256: str,
        expected_schema_version: int | None = None,
    ) -> PageExtractionArtifact:
        """Return a verified artifact or raise PageExtractionArtifactError."""

        root = snapshot_directory.resolve()
        relative = _require_relative_path(relative_path)
        absolute = (root / relative).resolve()
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise PageExtractionArtifactError(
                "page extraction path escapes snapshot directory"
            ) from exc

        if not absolute.is_file():
            raise PageExtractionArtifactError(
                f"page extraction artifact missing: {relative}"
            )

        raw = absolute.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected = str(expected_sha256 or "").strip().lower()
        if not expected or digest != expected:
            raise PageExtractionArtifactError(
                "page extraction sha256 mismatch"
            )

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PageExtractionArtifactError(
                "page extraction artifact is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise PageExtractionArtifactError(
                "page extraction artifact must be a JSON object"
            )
        artifact = PageExtractionArtifact.from_payload(payload)
        if expected_schema_version is not None:
            expected_version = _require_int(
                expected_schema_version,
                field_name="expected_schema_version",
                minimum=1,
            )
            if artifact.schema_version != expected_version:
                raise PageExtractionArtifactError(
                    "page extraction schema_version mismatch between "
                    "manifest and sidecar"
                )
        return artifact


def build_page_extraction_artifact_from_analysis(
    *,
    analysis: Any,
) -> PageExtractionArtifact:
    """Build a sidecar payload from a PageExtractionResult-like object."""

    text = analysis.text_content
    metadata = analysis.metadata
    return PageExtractionArtifact(
        schema_version=PAGE_EXTRACTION_SCHEMA_VERSION,
        text=_require_str(getattr(text, "text", None), field_name="text"),
        markdown=_require_str(
            getattr(text, "markdown", None), field_name="markdown"
        ),
        headings=_require_string_tuple(
            getattr(text, "headings", ()) or (),
            field_name="headings",
        ),
        code_block_count=_require_int(
            getattr(text, "code_block_count", 0),
            field_name="code_block_count",
            minimum=0,
        ),
        boilerplate_ratio=_require_finite_float(
            getattr(text, "boilerplate_ratio", 0.0),
            field_name="boilerplate_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        extraction_warnings=_require_string_tuple(
            getattr(text, "extraction_warnings", ()) or (),
            field_name="extraction_warnings",
        ),
        title=_optional_string(getattr(metadata, "title", None)),
        canonical_url=_optional_string(
            getattr(metadata, "canonical_url", None)
        ),
    )


def strip_page_extraction_artifact_from_enrichment(
    enrichment: Mapping[str, object] | None,
) -> tuple[dict[str, object], PageExtractionArtifact | None]:
    """Pop the transient bulk artifact key from enrichment."""

    payload = {
        str(key): value for key, value in dict(enrichment or {}).items()
    }
    raw_artifact = payload.pop(_ENRICHMENT_ARTIFACT_KEY, None)
    if raw_artifact is None:
        return payload, None
    if not isinstance(raw_artifact, Mapping):
        raise PageExtractionArtifactError(
            "page_extraction_artifact must be a mapping"
        )
    return payload, PageExtractionArtifact.from_payload(raw_artifact)


def enrichment_artifact_key() -> str:
    return _ENRICHMENT_ARTIFACT_KEY


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PageExtractionArtifactError(f"{field_name} must be a string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PageExtractionArtifactError("optional string field is invalid")
    text = value.strip()
    return text or None


def _require_int(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Reject bools and non-int types; do not accept numeric strings.
        raise PageExtractionArtifactError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise PageExtractionArtifactError(f"{field_name} must be >= {minimum}")
    return value


def _require_finite_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PageExtractionArtifactError(
            f"{field_name} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PageExtractionArtifactError(
            f"{field_name} must be a finite number"
        )
    if minimum is not None and number < minimum:
        raise PageExtractionArtifactError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise PageExtractionArtifactError(f"{field_name} must be <= {maximum}")
    return number


def _require_string_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise PageExtractionArtifactError(
            f"{field_name} must be a list or tuple of strings"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PageExtractionArtifactError(
                f"{field_name} must contain only strings"
            )
        items.append(item)
    return tuple(items)


def _safe_id(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(value).strip()
    )
    if not cleaned:
        raise PageExtractionArtifactError("fetch_record_id is empty")
    return cleaned


def _require_relative_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise PageExtractionArtifactError(
            "page extraction path must be relative and stay in-bounds"
        )
    return path


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
