"""Build canonical preprocessing inputs from normalized raw crawl entries."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from crawler.curation.ingest.schema.record import (
    validate_storage_relative_path,
)
from crawler.curation.preprocessing_governance import governance_payload
from preprocessing.preprocessing_input import (
    ExtractedTextContent,
    LanguageEvidence,
    PreprocessingInput,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from crawler.curation.ingest.schema.entry import RawManifestEntry
    from crawler.curation.ingest.schema.record import RawManifestRecord

MediaModality = Literal["image", "audio", "video"]

_MEDIA_KINDS = frozenset({"image", "audio", "video"})
_HASH_CHUNK_SIZE = 1024 * 1024

_OCR_TEXT_KEYS = {
    "image": ("image_ocr_text", "image_ocr_preview"),
    "video": ("frame_ocr_text", "frame_ocr_preview"),
    "audio": (),
}


class RawPayloadArtifactError(ValueError):
    """Raised when a manifest-backed raw payload cannot be trusted."""


@dataclass(frozen=True, slots=True)
class _VerifiedRawPayload:
    path: Path
    content: bytes | None


@dataclass(frozen=True, slots=True)
class _VerifiedRawPayloadRead:
    content: bytes | None
    file_identity: tuple[int, int, int, int]


def build_preprocessing_inputs(
    *,
    raw_entries: Iterable[RawManifestEntry],
    max_input_bytes: int,
) -> tuple[PreprocessingInput, ...]:
    """Map normalized raw entries to canonical preprocessing inputs."""

    document_inputs: list[PreprocessingInput] = []
    media_inputs: list[PreprocessingInput] = []

    for entry in raw_entries:
        if _is_curatable_document_entry(entry=entry):
            raw_payload = _verified_raw_payload(
                entry=entry,
                capture_bytes_limit=(
                    max_input_bytes
                    if entry.record.kind == "document"
                    else None
                ),
            )
            if raw_payload is None:
                continue
            document_input = _document_input_from_entry(
                entry=entry,
                raw_payload=raw_payload,
            )
            if document_input is not None:
                document_inputs.append(document_input)
            continue

        kind = entry.record.kind.strip().lower()
        if kind not in _MEDIA_KINDS:
            continue
        media_input = _media_input_from_entry(
            entry=entry,
            modality=cast(MediaModality, kind),
        )
        if media_input is not None:
            media_inputs.append(media_input)

    return tuple(document_inputs + media_inputs)


def _document_input_from_entry(
    *,
    entry: RawManifestEntry,
    raw_payload: _VerifiedRawPayload,
) -> PreprocessingInput | None:
    record = entry.record
    if record.kind == "page":
        extracted = _load_page_extracted_text(entry=entry)
        return PreprocessingInput(
            source_id=record.fetch_record_id,
            source_url=record.requested_url,
            normalized_url=record.normalized_url,
            domain=record.domain,
            path=record.path,
            language_evidence=_build_language_evidence(record),
            encoding=record.encoding,
            title=extracted.title,
            modality="text",
            mime_type=record.mime_type,
            byte_size=record.byte_size,
            extracted_text_content=extracted.content,
            payload={
                **dict(record.enrichment),
                **governance_payload(record.governance),
            },
        )

    document_text = _document_text_from_entry(entry=entry)
    if document_text is None and _can_read_raw_text_document(entry=entry):
        if raw_payload.content is not None:
            document_text = _decode_verified_text(
                content=raw_payload.content,
                encoding=record.encoding or "utf-8",
            )

    if document_text is None:
        return None

    return PreprocessingInput(
        source_id=record.fetch_record_id,
        source_url=record.requested_url,
        normalized_url=record.normalized_url,
        domain=record.domain,
        path=record.path,
        language_evidence=_build_language_evidence(record),
        encoding=record.encoding,
        title=_document_title_from_entry(entry=entry),
        modality="document",
        mime_type=record.mime_type,
        media_path=str(raw_payload.path),
        byte_size=record.byte_size,
        ocr_text=document_text,
        payload={
            **dict(record.enrichment),
            **governance_payload(record.governance),
        },
    )


def _media_input_from_entry(
    *,
    entry: RawManifestEntry,
    modality: MediaModality,
) -> PreprocessingInput | None:
    record = entry.record
    raw_payload = _verified_raw_payload(
        entry=entry,
        capture_bytes_limit=None,
    )
    if raw_payload is None and not record.asset_metadata_only:
        return None

    enrichment = dict(record.enrichment)
    asset_context = dict(record.asset_context)
    payload: dict[str, object] = {**asset_context, **enrichment}
    payload.update(governance_payload(record.governance))
    if record.source_page_url:
        payload.setdefault("source_page_url", record.source_page_url)
    if record.embed_host:
        payload.setdefault("embed_host", record.embed_host)
    if record.keyframe_paths:
        payload.setdefault("keyframes", list(record.keyframe_paths))
    if record.thumbnail_path:
        payload.setdefault("thumbnail_path", record.thumbnail_path)

    return PreprocessingInput(
        source_id=record.fetch_record_id,
        source_url=record.requested_url or record.final_url,
        normalized_url=record.normalized_url,
        domain=record.domain,
        path=record.path,
        language_evidence=_build_language_evidence(record),
        encoding=record.encoding,
        title=_optional_text(
            record.parent_title
            or payload.get("title")
            or payload.get("caption_text")
        ),
        modality=modality,
        mime_type=record.mime_type,
        media_path=str(raw_payload.path) if raw_payload is not None else None,
        byte_size=record.byte_size or record.observed_bytes or None,
        duration_seconds=_optional_float(
            payload.get(f"{modality}_duration_seconds")
        ),
        width=_optional_int(payload.get(f"{modality}_width")),
        height=_optional_int(payload.get(f"{modality}_height")),
        transcript_text=_optional_text(payload.get("transcript_text")),
        ocr_text=_optional_text(
            next(
                (
                    payload[key]
                    for key in _OCR_TEXT_KEYS[modality]
                    if payload.get(key)
                ),
                None,
            )
        ),
        payload=payload,
    )


def _build_language_evidence(
    record: RawManifestRecord,
) -> LanguageEvidence | None:
    if all(
        value is None
        for value in (
            record.language,
            record.language_confidence,
            record.language_source,
            record.language_detector_version,
        )
    ):
        return None

    return LanguageEvidence(
        language=record.language,
        confidence=record.language_confidence,
        source=record.language_source,
        detector_version=record.language_detector_version,
    )


def _verified_raw_payload(
    *,
    entry: RawManifestEntry,
    capture_bytes_limit: int | None,
) -> _VerifiedRawPayload | None:
    """Resolve and authenticate one manifest-backed raw payload.

    Missing payloads remain representable for metadata-only media records.
    Every payload that does exist must be an in-tree regular file whose size
    and digest still match the immutable manifest evidence.
    """

    record = entry.record
    try:
        relative_text = validate_storage_relative_path(
            record.storage_relative_path
        )
        root = entry.run_directory.resolve()
        relative = PurePosixPath(relative_text)
        unresolved = root.joinpath(*relative.parts)
        _reject_symlink_components(root=root, relative=relative)
        resolved = unresolved.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RawPayloadArtifactError(
            "raw payload path is not safely contained for "
            f"fetch_record_id={record.fetch_record_id}"
        ) from exc

    if not resolved.exists():
        return None
    if not resolved.is_file():
        raise RawPayloadArtifactError(
            "raw payload is not a regular file for "
            f"fetch_record_id={record.fetch_record_id}"
        )

    verified_read = _verify_raw_payload_integrity(
        path=resolved,
        entry=entry,
        capture_bytes_limit=capture_bytes_limit,
    )

    # Media preprocessors consume a path rather than an open descriptor.  A
    # final identity/containment check narrows the unavoidable interval after
    # verification and before those adapters open the file.
    try:
        _reject_symlink_components(root=root, relative=relative)
        if (
            unresolved.resolve() != resolved
            or _file_identity(resolved.stat()) != verified_read.file_identity
        ):
            raise RawPayloadArtifactError(
                "raw payload path changed after verification"
            )
    except (OSError, ValueError) as exc:
        raise RawPayloadArtifactError(
            "raw payload path changed after verification for "
            f"fetch_record_id={record.fetch_record_id}"
        ) from exc
    return _VerifiedRawPayload(
        path=resolved,
        content=verified_read.content,
    )


def _reject_symlink_components(
    *,
    root: Path,
    relative: PurePosixPath,
) -> None:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RawPayloadArtifactError(
                "raw payload path contains a symbolic link"
            )


def _verify_raw_payload_integrity(
    *,
    path: Path,
    entry: RawManifestEntry,
    capture_bytes_limit: int | None,
) -> _VerifiedRawPayloadRead:
    record = entry.record
    expected_size = record.byte_size
    expected_sha256 = record.content_sha256.strip().lower()

    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise RawPayloadArtifactError(
            "raw payload manifest byte_size is invalid for "
            f"fetch_record_id={record.fetch_record_id}"
        )
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RawPayloadArtifactError(
            "raw payload manifest content_sha256 is invalid for "
            f"fetch_record_id={record.fetch_record_id}"
        )

    try:
        with _open_raw_payload_no_follow(path=path) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RawPayloadArtifactError(
                    "raw payload is not a regular file for "
                    f"fetch_record_id={record.fetch_record_id}"
                )
            if before.st_size != expected_size:
                raise RawPayloadArtifactError(
                    "raw payload size mismatch for "
                    f"fetch_record_id={record.fetch_record_id}: "
                    f"expected={expected_size}, observed={before.st_size}"
                )

            digest = hashlib.sha256()
            capture = (
                bytearray()
                if capture_bytes_limit is not None
                and before.st_size <= capture_bytes_limit
                else None
            )
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
                if capture is not None:
                    capture.extend(chunk)

            after = os.fstat(handle.fileno())
    except RawPayloadArtifactError:
        raise
    except OSError as exc:
        raise RawPayloadArtifactError(
            "raw payload could not be verified for "
            f"fetch_record_id={record.fetch_record_id}"
        ) from exc

    if _file_identity(before) != _file_identity(after):
        raise RawPayloadArtifactError(
            "raw payload changed during verification for "
            f"fetch_record_id={record.fetch_record_id}"
        )
    if digest.hexdigest() != expected_sha256:
        raise RawPayloadArtifactError(
            "raw payload sha256 mismatch for "
            f"fetch_record_id={record.fetch_record_id}"
        )
    return _VerifiedRawPayloadRead(
        content=bytes(capture) if capture is not None else None,
        file_identity=_file_identity(after),
    )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


def _open_raw_payload_no_follow(*, path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _decode_verified_text(*, content: bytes, encoding: str) -> str:
    # TextIOWrapper preserves Path.read_text's universal-newline behavior
    # while decoding the exact bytes authenticated by the open descriptor.
    with io.TextIOWrapper(
        io.BytesIO(content),
        encoding=encoding,
        errors="replace",
    ) as stream:
        return stream.read()


@dataclass(frozen=True, slots=True)
class _PageExtractedLoad:
    content: ExtractedTextContent
    title: str | None


def _load_page_extracted_text(
    *,
    entry: RawManifestEntry,
) -> _PageExtractedLoad:
    from crawler.storage.datasets.extraction.page_extraction_artifact import (
        PAGE_EXTRACTION_SCHEMA_VERSION,
        PageExtractionArtifactError,
        PageExtractionArtifactReader,
    )

    record = entry.record
    enrichment = record.enrichment
    relative_path = str(enrichment.get("page_extraction_path") or "").strip()

    try:
        expected_sha = str(
            enrichment.get("page_extraction_sha256") or ""
        ).strip()
        if not relative_path:
            raise PageExtractionArtifactError(
                "page_extraction_path missing from enrichment"
            )
        if not expected_sha:
            raise PageExtractionArtifactError(
                "page_extraction_sha256 missing from enrichment"
            )

        expected_schema = enrichment.get("page_extraction_schema_version")
        if expected_schema is None:
            expected_schema_version = PAGE_EXTRACTION_SCHEMA_VERSION
        elif isinstance(expected_schema, bool) or not isinstance(
            expected_schema,
            int,
        ):
            raise PageExtractionArtifactError(
                "page_extraction_schema_version must be an integer"
            )
        else:
            expected_schema_version = expected_schema

        artifact = PageExtractionArtifactReader().read(
            snapshot_directory=entry.run_directory,
            relative_path=relative_path,
            expected_sha256=expected_sha,
            expected_schema_version=expected_schema_version,
        )
    except (PageExtractionArtifactError, OSError) as exc:
        raise PageExtractionArtifactError(
            "page extraction load failed for "
            f"fetch_record_id={record.fetch_record_id}, "
            f"path={relative_path or '<missing>'}: {exc}"
        ) from exc

    return _PageExtractedLoad(
        content=ExtractedTextContent(
            text=artifact.text,
            markdown=artifact.markdown,
            headings=artifact.headings,
            code_block_count=artifact.code_block_count,
            boilerplate_ratio=artifact.boilerplate_ratio,
            warnings=artifact.extraction_warnings,
        ),
        title=artifact.title,
    )


def _is_curatable_document_entry(*, entry: RawManifestEntry) -> bool:
    if entry.record.kind != "page":
        return entry.record.kind == "document"
    mime_type = (entry.record.mime_type or "").lower()
    if mime_type.startswith("text/html"):
        return True
    extension = (entry.record.extension or "").lower()
    return extension in {".html", ".htm", ""}


def _document_text_from_entry(*, entry: RawManifestEntry) -> str | None:
    document_text = entry.record.enrichment.get("document_text")
    if not isinstance(document_text, str):
        return None
    return document_text.strip() or None


def _document_title_from_entry(*, entry: RawManifestEntry) -> str | None:
    title = entry.record.parent_title
    return title.strip() if title and title.strip() else None


def _can_read_raw_text_document(*, entry: RawManifestEntry) -> bool:
    mime_type = (entry.record.mime_type or "").lower()
    if mime_type.startswith("text/"):
        return True
    return (entry.record.extension or "").lower() in {
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".json",
        ".vtt",
        ".srt",
        ".ttml",
    }


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)
