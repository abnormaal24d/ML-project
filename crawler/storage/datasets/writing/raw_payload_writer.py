"""Persist raw crawler payloads in content-addressed dataset storage."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from config.path_resolution.project_paths import validate_safe_relative_path

if TYPE_CHECKING:
    from config.settings.datasets import (
        DatasetPathSettings,
        RawDatasetWriterSettings,
    )
    from crawler.fetching.results.result import FetchResult


from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)

_COPY_BUFFER_SIZE = 1024 * 1024
_EXTENSION_FALLBACK = ".bin"
_MIME_EXTENSION_OVERRIDES = {
    "application/javascript": ".js",
    "application/json": ".json",
    "application/xhtml+xml": ".xhtml",
    "application/xml": ".xml",
    "image/jpeg": ".jpg",
    "text/html": ".html",
    "text/javascript": ".js",
    "text/plain": ".txt",
    "text/xml": ".xml",
}
_VALID_EXTENSION_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.+-_")


class RawPayloadWriter:
    """Persist raw crawler payloads in content-addressed dataset storage."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        dataset_paths: DatasetPathSettings,
        run_directory: Path,
    ) -> None:
        self._settings = settings
        self._run_directory = run_directory
        self._objects_relative_directory = _require_relative_path(
            dataset_paths.objects_directory,
            field_name="dataset_paths.objects_directory",
        )

        self.objects_directory.mkdir(parents=True, exist_ok=True)

    @property
    def objects_directory(self) -> Path:
        return self._run_directory / self._objects_relative_directory

    @property
    def run_directory(self) -> Path:
        return self._run_directory

    def absolute_path(self, *, relative_path: Path) -> Path:
        safe_path = _require_relative_path(
            relative_path,
            field_name="relative_path",
        )
        return self._run_directory / safe_path

    def prepare(
        self,
        *,
        result: FetchResult,
        kind: str,
        modality: str,
    ) -> tuple[Path, str, int, bytes | None]:
        payload = result.payload

        if payload is None:
            inline_body = result.read_body_required()
            content_hash = (
                result.body_sha256 or hashlib.sha256(inline_body).hexdigest()
            )
            byte_size = len(inline_body)
        else:
            content_hash = result.body_sha256 or payload.sha256_hex
            byte_size = payload.byte_size
            inline_body = None

        relative_path = self._build_storage_path(
            content_hash=content_hash,
            mime_type=result.mime_type,
            final_url=result.final_url,
            kind=kind,
            modality=modality,
        )
        return relative_path, content_hash, byte_size, inline_body

    def persist_prepared(
        self,
        *,
        relative_path: Path,
        result: FetchResult,
        inline_body: bytes | None,
    ) -> None:
        safe_relative_path = _require_relative_path(
            relative_path,
            field_name="relative_path",
        )
        absolute_path = self._run_directory / safe_relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        payload = result.payload

        expected_sha256, expected_size = _prepared_payload_integrity(
            result=result,
            inline_body=inline_body,
        )
        if absolute_path.exists() or absolute_path.is_symlink():
            if self._settings.deduplicate_objects and _is_verified_duplicate(
                path=absolute_path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            ):
                # A conditional-cache hit already points at this durable
                # object.  Reusing it for another logical media identity must
                # not unlink the object that backs the cached representation.
                self.discard_payload(result=result)
                return
            _require_replaceable_destination(path=absolute_path)

        temp_path = _create_temporary_sibling(absolute_path)

        try:
            if inline_body is not None:
                _write_inline_payload_durable(
                    path=temp_path,
                    data=inline_body,
                )
            else:
                if payload is None:
                    raise ValueError(
                        "FetchResult payload is required when inline_body is "
                        "not provided"
                    )
                _write_temp_payload_durable(
                    src=payload.temp_path,
                    dst=temp_path,
                )

            os.replace(temp_path, absolute_path)
            _fsync_directory(absolute_path.parent)

            if payload is not None:
                payload.temp_path.unlink(missing_ok=True)

        except OSError:
            temp_path.unlink(missing_ok=True)
            raise

    def discard_payload(self, *, result: FetchResult) -> None:
        """Discard only a transient fetch payload, never a durable object."""

        payload = result.payload
        if payload is None:
            return
        try:
            payload.temp_path.resolve().relative_to(self._run_directory)
        except ValueError:
            payload.cleanup()

    def _build_storage_path(
        self,
        *,
        content_hash: str,
        mime_type: str | None,
        final_url: str,
        kind: str,
        modality: str,
    ) -> Path:
        normalized_hash = _normalize_content_hash(content_hash)
        extension = self._resolve_extension(mime_type=mime_type, url=final_url)
        family, variant = self._resolve_storage_parts(
            kind=kind,
            modality=modality,
            mime_type=mime_type,
        )

        return Path(
            self._objects_relative_directory,
            family,
            variant,
            normalized_hash[:2],
            f"{normalized_hash}{extension}",
        )

    @classmethod
    def _resolve_extension(cls, *, mime_type: str | None, url: str) -> str:
        normalized_mime_type = normalize_mime_type(mime_type) or ""

        if normalized_mime_type:
            override = _MIME_EXTENSION_OVERRIDES.get(normalized_mime_type)
            if override is not None:
                return override

            guessed = mimetypes.guess_extension(normalized_mime_type)
            normalized_extension = cls._normalize_extension(guessed)
            if normalized_extension is not None:
                return normalized_extension

        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix
        normalized_suffix = cls._normalize_extension(suffix)
        if normalized_suffix is not None:
            return normalized_suffix

        return _EXTENSION_FALLBACK

    @classmethod
    def _resolve_storage_parts(
        cls,
        *,
        kind: str,
        modality: str,
        mime_type: str | None,
    ) -> tuple[str, str]:
        lowered_kind = str(kind).strip().lower()
        lowered_modality = str(modality).strip().lower()
        lowered_mime = normalize_mime_type(mime_type) or ""

        if lowered_kind == "page":
            if lowered_mime == "text/plain":
                return "page", "plain"
            return "page", "html"

        if lowered_kind == "feed":
            return "feed", "original"

        if lowered_kind == "document":
            return "document", "original"

        if lowered_kind in {"image", "audio", "video"}:
            return lowered_kind, "original"

        if lowered_modality == "text":
            return "text", "original"

        if lowered_modality:
            return lowered_modality, "original"

        return "other", "original"

    @staticmethod
    def _normalize_extension(extension: str | None) -> str | None:
        if not extension:
            return None

        normalized = str(extension).strip().lower()
        if not normalized.startswith("."):
            return None

        body = normalized[1:]
        if not body:
            return None

        if len(body) > 32:
            return None

        if any(character not in _VALID_EXTENSION_CHARS for character in body):
            return None

        return f".{body}"


def _normalize_content_hash(content_hash: str) -> str:
    normalized = str(content_hash).strip().lower()

    if not normalized:
        raise ValueError("content_hash must not be empty")

    if "/" in normalized or "\\" in normalized or ":" in normalized:
        raise ValueError("content_hash must not contain path separators")

    return normalized


def _prepared_payload_integrity(
    *,
    result: FetchResult,
    inline_body: bytes | None,
) -> tuple[str, int]:
    """Return the expected digest and size for a prepared payload.

    Content-addressed paths are safe to reuse only when the existing object
    matches the payload that this write is about to persist.  Keep this in one
    helper so the duplicate check follows the same digest choice as
    :meth:`RawPayloadWriter.prepare`.
    """

    if inline_body is not None:
        return hashlib.sha256(inline_body).hexdigest(), len(inline_body)

    payload = result.payload
    if payload is None:
        raise ValueError(
            "FetchResult payload is required when inline_body is not provided"
        )
    return (
        str(result.body_sha256 or payload.sha256_hex).strip().lower(),
        payload.byte_size,
    )


def _is_verified_duplicate(
    *,
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> bool:
    """Return whether ``path`` is the exact durable object to reuse.

    Never reuse a symlink or another non-regular entry, and avoid hashing when
    its byte count is already wrong.  A same-size corrupt object still needs a
    full digest check before its incoming payload can be discarded.
    """

    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_size != expected_size:
            return False
        return _sha256_file(path) == expected_sha256
    except OSError:
        return False


def _require_replaceable_destination(*, path: Path) -> None:
    """Fail closed for a non-regular existing destination.

    ``os.replace`` can safely replace a regular stale file.  It must not be
    allowed to silently consume the incoming payload when a directory,
    symlink, or special filesystem entry occupies the object path.
    """

    if path.is_symlink() or not path.is_file():
        raise OSError(
            "raw payload destination is not a replaceable regular file: "
            f"{path}"
        )


def _require_relative_path(path: Path | str, *, field_name: str) -> Path:
    return Path(validate_safe_relative_path(path, field_name=field_name))


def _write_inline_payload_durable(*, path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_temp_payload_durable(*, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError:
        if not src.exists():
            raise

    _copy_file_durable(src=src, dst=dst)
    src.unlink(missing_ok=True)


def _copy_file_durable(*, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as source, dst.open("wb") as target:
        shutil.copyfileobj(source, target, length=_COPY_BUFFER_SIZE)
        target.flush()
        os.fsync(target.fileno())


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one regular payload file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return

    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _create_temporary_sibling(path: Path) -> Path:
    """Create a uniquely named temporary sibling file using mkstemp.

    Keep the temporary basename independent of the content-addressed target
    name.  Content hashes already consume most of a Windows path budget, and
    repeating the target name can make an otherwise valid destination exceed
    the Windows path-length limit.  The temporary file remains in the same
    directory, so ``os.replace`` retains its atomic same-volume behavior.
    """
    fd, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp-",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(temporary_path)
