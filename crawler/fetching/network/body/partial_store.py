"""Confined, durable storage for temporary and resumable response bodies."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

MAX_PARTIAL_METADATA_BYTES = 64 * 1024
MAX_PARTIAL_SCAN_ENTRIES = 1_000


class PartialPayloadSecurityError(ValueError):
    """Raised when a partial-payload reference fails ownership checks."""


class PartialPayloadStorage:
    """Own temporary response files below one canonical directory."""

    def __init__(
        self,
        *,
        temporary_directory: str | Path | None = None,
        file_prefix: str = "crawler_body_",
        preserve_partial_files: bool = False,
    ) -> None:
        selected = (
            Path(temporary_directory)
            if temporary_directory is not None
            else Path(tempfile.gettempdir()) / "multimodal-crawler"
        )
        selected.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = selected.resolve()
        self._file_prefix = str(file_prefix)
        if not self._file_prefix:
            raise ValueError("file_prefix must not be empty")
        self._preserve_partial_files = bool(preserve_partial_files)
        self._lock = RLock()

    @property
    def temporary_directory(self) -> Path:
        return self._temporary_directory

    def create_temp_file(
        self,
        *,
        directory: str | Path | None = None,
        prefix: str | None = None,
        suffix: str = "",
    ) -> tuple[int, Path]:
        """Create a private temp file within the configured directory."""

        selected_directory = (
            self._temporary_directory
            if directory is None
            else self._confined_directory(Path(directory))
        )
        selected_directory.mkdir(parents=True, exist_ok=True)
        selected_prefix = prefix or self._file_prefix
        if not selected_prefix.startswith(self._file_prefix):
            raise PartialPayloadSecurityError("temporary prefix is not owned")
        with self._lock:
            fd, raw_path = tempfile.mkstemp(
                prefix=selected_prefix,
                suffix=suffix,
                dir=str(selected_directory),
            )
        try:
            os.chmod(raw_path, 0o600)
        except OSError:
            os.close(fd)
            Path(raw_path).unlink(missing_ok=True)
            raise
        return fd, Path(raw_path)

    def delete(self, *, path: Path) -> None:
        """Delete an owned temporary payload and surface cleanup failures."""

        owned = self._owned_path(path=path, require_partial=False)
        with self._lock:
            owned.unlink(missing_ok=True)

    def discard_partial(self, *, path: Path, owner_token: str) -> None:
        """Delete a partial and sidecar only when their owner token matches."""

        partial = self._owned_path(path=path, require_partial=True)
        metadata_path = self._metadata_path(partial)
        metadata = self.read_metadata(metadata_path=metadata_path)
        if not owner_token or not secrets.compare_digest(
            str(metadata.get("owner_token") or ""), owner_token
        ):
            raise PartialPayloadSecurityError("partial owner token mismatch")
        with self._lock:
            partial.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            self._fsync_directory()

    def complete_resume(self, *, path: Path, owner_token: str) -> None:
        """Remove resume metadata after a partial became a complete payload."""

        partial = self._owned_path(path=path, require_partial=True)
        metadata_path = self._metadata_path(partial)
        metadata = self.read_metadata(metadata_path=metadata_path)
        stored_token = str(metadata.get("owner_token") or "")
        if not owner_token or not secrets.compare_digest(
            stored_token,
            owner_token,
        ):
            raise PartialPayloadSecurityError("partial owner token mismatch")
        with self._lock:
            metadata_path.unlink()
            self._fsync_directory()

    def finalize_incomplete_payload(
        self,
        *,
        path: Path,
        reason: str,
        url: str,
        status_code: int | None,
        content_length: int | None,
        observed_bytes: int,
        chunk_count: int,
        max_bytes: int,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> Path | None:
        """Delete or durably preserve an incomplete owned payload."""

        payload_path = self._owned_path(path=path, require_partial=False)
        with self._lock:
            if not payload_path.exists():
                return None
            if not self._preserve_partial_files:
                self.delete(path=payload_path)
                return None

            actual_size = payload_path.stat().st_size
            if actual_size <= 0 or actual_size != int(observed_bytes):
                self.delete(path=payload_path)
                return None

            partial_path = self._partial_path(payload_path)
            metadata_path = self._metadata_path(partial_path)
            owner_token = secrets.token_urlsafe(32)
            metadata: dict[str, object] = {
                "schema_version": 1,
                "reason": str(reason),
                "url_sha256": self.url_fingerprint(url),
                "status_code": status_code,
                "content_length": content_length,
                "observed_bytes": actual_size,
                "chunk_count": int(chunk_count),
                "max_bytes": int(max_bytes),
                "etag": _clean_validator(etag),
                "last_modified": _clean_validator(last_modified),
                "owner_token": owner_token,
            }
            return self._preserve_with_metadata(
                path=payload_path,
                partial_path=partial_path,
                metadata_path=metadata_path,
                metadata=metadata,
            )

    def read_metadata(self, *, metadata_path: Path) -> dict[str, Any]:
        """Read one bounded, owned sidecar object."""

        path = self._owned_metadata_path(metadata_path)
        if not path.exists() or path.is_symlink():
            return {}
        if path.stat().st_size > MAX_PARTIAL_METADATA_BYTES:
            return {}
        try:
            with path.open("rb") as handle:
                raw = handle.read(MAX_PARTIAL_METADATA_BYTES + 1)
            if len(raw) > MAX_PARTIAL_METADATA_BYTES:
                return {}
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def iter_metadata_paths(self) -> tuple[Path, ...]:
        """Return a bounded snapshot of owned sidecars."""

        paths: list[Path] = []
        with os.scandir(self._temporary_directory) as entries:
            for entry in entries:
                if len(paths) >= MAX_PARTIAL_SCAN_ENTRIES:
                    break
                if (
                    entry.is_file(follow_symlinks=False)
                    and entry.name.startswith(self._file_prefix)
                    and entry.name.endswith(".partial.json")
                ):
                    paths.append(Path(entry.path))
        return tuple(paths)

    def validate_resume(
        self,
        *,
        path: Path,
        owner_token: str,
    ) -> tuple[Path, dict[str, Any], int]:
        """Validate path, sidecar ownership, validators, and actual length."""

        partial = self._owned_path(path=path, require_partial=True)
        if partial.is_symlink() or not partial.is_file():
            raise PartialPayloadSecurityError(
                "partial payload is not a regular file"
            )
        metadata = self.read_metadata(
            metadata_path=self._metadata_path(partial)
        )
        stored_token = str(metadata.get("owner_token") or "")
        if not owner_token or not secrets.compare_digest(
            stored_token, owner_token
        ):
            raise PartialPayloadSecurityError("partial owner token mismatch")
        if not metadata.get("etag") and not metadata.get("last_modified"):
            raise PartialPayloadSecurityError(
                "partial has no representation validator"
            )
        actual_size = partial.stat().st_size
        if (
            actual_size <= 0
            or _strict_int(metadata.get("observed_bytes")) != actual_size
        ):
            raise PartialPayloadSecurityError(
                "partial length does not match metadata"
            )
        return partial, metadata, actual_size

    def open_for_resume(
        self,
        *,
        path: Path,
        owner_token: str,
        max_bytes: int,
        sniff_byte_count: int,
    ) -> tuple[int, Any, bytearray, int]:
        """Open an owned partial without following symlinks and hash it."""

        partial, _metadata, actual_size = self.validate_resume(
            path=path,
            owner_token=owner_token,
        )
        if actual_size >= int(max_bytes):
            raise PartialPayloadSecurityError(
                "partial already exhausts byte budget"
            )
        flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(partial, flags)
        try:
            stat = os.fstat(descriptor)
            if stat.st_size != actual_size:
                raise PartialPayloadSecurityError(
                    "partial changed while opening"
                )
            digest = hashlib.sha256()
            sniff = bytearray()
            os.lseek(descriptor, 0, os.SEEK_SET)
            remaining = actual_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise PartialPayloadSecurityError(
                        "partial changed while hashing"
                    )
                digest.update(chunk)
                if len(sniff) < sniff_byte_count:
                    sniff.extend(chunk[: sniff_byte_count - len(sniff)])
                remaining -= len(chunk)
            os.lseek(descriptor, 0, os.SEEK_END)
            return descriptor, digest, sniff, actual_size
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def url_fingerprint(url: str) -> str:
        return hashlib.sha256(str(url).strip().encode("utf-8")).hexdigest()

    def _preserve_with_metadata(
        self,
        *,
        path: Path,
        partial_path: Path,
        metadata_path: Path,
        metadata: dict[str, object],
    ) -> Path:
        if partial_path.exists() or metadata_path.exists():
            raise FileExistsError(f"partial payload collision: {partial_path}")
        os.link(path, partial_path, follow_symlinks=False)
        path.unlink()
        try:
            self._atomic_write_metadata(path=metadata_path, payload=metadata)
            self._fsync_directory()
        except BaseException:
            partial_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            self._fsync_directory()
            raise
        return partial_path

    def _atomic_write_metadata(
        self,
        *,
        path: Path,
        payload: dict[str, object],
    ) -> None:
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f"{self._file_prefix}metadata_",
            suffix=".tmp",
            dir=str(self._temporary_directory),
        )
        temp_path = Path(raw_temp)
        try:
            os.chmod(temp_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    payload, handle, sort_keys=True, separators=(",", ":")
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temp_path.unlink(missing_ok=True)
            raise

    def _owned_path(self, *, path: Path, require_partial: bool) -> Path:
        candidate = Path(path)
        if candidate.is_symlink():
            raise PartialPayloadSecurityError(
                "symlink payload paths are forbidden"
            )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._temporary_directory)
        except ValueError as exc:
            raise PartialPayloadSecurityError(
                "payload path escapes the temporary directory"
            ) from exc
        if not resolved.name.startswith(self._file_prefix):
            raise PartialPayloadSecurityError("payload path is not owned")
        if require_partial and not resolved.name.endswith(".partial"):
            raise PartialPayloadSecurityError("payload is not a partial file")
        return resolved

    def _owned_metadata_path(self, path: Path) -> Path:
        resolved = self._owned_path(
            path=Path(str(path).removesuffix(".json")),
            require_partial=True,
        )
        return resolved.with_name(f"{resolved.name}.json")

    def _confined_directory(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self._temporary_directory)
        except ValueError as exc:
            raise PartialPayloadSecurityError(
                "temporary directory escapes configured root"
            ) from exc
        return resolved

    @staticmethod
    def _partial_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.partial")

    @staticmethod
    def _metadata_path(partial_path: Path) -> Path:
        return partial_path.with_name(f"{partial_path.name}.json")

    def _fsync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self._temporary_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _clean_validator(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned[:1024] if cleaned else None


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
