"""Stable fingerprint calculators for files, settings, and datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeGuard, cast

from pydantic import BaseModel

DeadlineCheckpoint = Callable[[str], None]


def _no_deadline_checkpoint(_stage: str) -> None:
    """Provide the default for fingerprinting outside a timed check."""

    return None


class FileFingerprintCalculator:
    """Calculate SHA-256 fingerprints for one file."""

    def calculate(
        self,
        *,
        path: Path,
        checkpoint: DeadlineCheckpoint = _no_deadline_checkpoint,
    ) -> str:
        """Return a stable SHA-256 digest for file bytes."""
        digest = hashlib.sha256()

        with path.open("rb") as handle:
            for chunk_index, chunk in enumerate(
                iter(lambda: handle.read(65_536), b""),
                start=1,
            ):
                if chunk_index % 32 == 0:
                    checkpoint("file_fingerprint_scan")
                digest.update(chunk)

        return digest.hexdigest()


class SettingsFingerprintCalculator:
    """Calculate stable hashes for structured settings payloads."""

    @staticmethod
    def _is_dataclass_instance(value: object) -> TypeGuard[Any]:
        """
        Return whether value is a dataclass instance, not a dataclass type.
        """
        return is_dataclass(value) and not isinstance(value, type)

    def calculate(self, *, payload: object) -> str:
        """Return a stable SHA-256 digest for a structured payload."""
        normalized = self.normalize(payload=payload)
        serialized = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def normalize(self, *, payload: object) -> object:
        """Normalize a structured payload into JSON-safe primitives."""
        if isinstance(payload, BaseModel):
            return self.normalize(payload=payload.model_dump(mode="json"))

        if self._is_dataclass_instance(payload):
            return self.normalize(payload=asdict(cast("Any", payload)))

        if isinstance(payload, Path):
            return payload.as_posix()

        if isinstance(payload, Enum):
            return payload.value

        if isinstance(payload, dict):
            return {
                str(key): self.normalize(payload=value)
                for key, value in sorted(
                    payload.items(),
                    key=lambda item: str(item[0]),
                )
            }

        if isinstance(payload, tuple):
            return [self.normalize(payload=value) for value in payload]

        if isinstance(payload, list):
            return [self.normalize(payload=value) for value in payload]

        return payload


class DatasetFingerprintCalculator:
    """Calculate a stable fingerprint across multiple artifact files."""

    def __init__(
        self,
        *,
        file_fingerprint_calculator: FileFingerprintCalculator,
    ) -> None:
        self._file_fingerprint_calculator = file_fingerprint_calculator

    def calculate(
        self,
        *,
        paths: tuple[Path, ...],
        root: Path | None = None,
        checkpoint: DeadlineCheckpoint = _no_deadline_checkpoint,
    ) -> str:
        """Return a stable digest across sorted path names and file hashes."""
        digest = hashlib.sha256()
        resolved_root = root.resolve() if root is not None else None

        for file_index, path in enumerate(sorted(paths), start=1):
            if file_index % 32 == 0:
                checkpoint("dataset_fingerprint_scan")
            digest.update(
                self._fingerprint_path(path=path, root=resolved_root).encode(
                    "utf-8"
                )
            )
            digest.update(b"\0")
            digest.update(
                self._file_fingerprint_calculator.calculate(
                    path=path,
                    checkpoint=checkpoint,
                ).encode("utf-8")
            )
            digest.update(b"\0")

        return digest.hexdigest()

    @staticmethod
    def _fingerprint_path(*, path: Path, root: Path | None) -> str:
        if root is None:
            return path.as_posix()
        try:
            return path.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            return path.name


class SourceFingerprintCalculator:
    """Fingerprint source inputs that influence crawl scope."""

    def __init__(
        self,
        *,
        settings_fingerprint_calculator: SettingsFingerprintCalculator,
    ) -> None:
        self._settings_fingerprint_calculator = settings_fingerprint_calculator

    def calculate(
        self,
        *,
        seed_urls: tuple[str, ...],
        source_profile: object,
    ) -> str:
        """Return a stable digest for the active source profile."""
        payload = {
            "seed_urls": tuple(sorted(seed_urls)),
            "source_profile": source_profile,
        }

        return self._settings_fingerprint_calculator.calculate(
            payload=payload,
        )


class ProjectFingerprintCalculator:
    """Fingerprint the checked-out product source without runtime artifacts."""

    _SOURCE_SUFFIXES = frozenset(
        {".py", ".toml", ".json", ".yaml", ".yml", ".ini"}
    )
    _SOURCE_FILENAMES = frozenset(
        {"Dockerfile", "Dockerfile.cpu", "Dockerfile.gpu", "Makefile"}
    )
    _EXCLUDED_PARTS = frozenset(
        {
            ".git",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "artifacts",
            "build",
            "data",
            "dist",
            "htmlcov",
            "runtime",
        }
    )

    def calculate(self, *, project_root: Path) -> str:
        """Return a Git commit fingerprint or deterministic source-tree hash."""

        root = project_root.resolve()
        git_commit = self._git_commit(root=root)
        if git_commit is not None:
            return f"git:{git_commit}"

        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or self._excluded(path=path, root=root):
                continue
            if (
                path.suffix.lower() not in self._SOURCE_SUFFIXES
                and path.name not in self._SOURCE_FILENAMES
            ):
                continue
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65_536), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return f"source:{digest.hexdigest()}"

    @classmethod
    def dependency_lock_fingerprint(cls, *, project_root: Path) -> str:
        """Hash dependency declarations and lock/constraint files."""

        root = project_root.resolve()
        candidates = [root / "pyproject.toml"]
        requirements = root / "requirements"
        if requirements.is_dir():
            candidates.extend(sorted(requirements.glob("*.txt")))
        digest = hashlib.sha256()
        for path in candidates:
            if not path.is_file():
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @classmethod
    def _excluded(cls, *, path: Path, root: Path) -> bool:
        relative_parts = path.relative_to(root).parts
        return any(part in cls._EXCLUDED_PARTS for part in relative_parts)

    @staticmethod
    def _git_commit(*, root: Path) -> str | None:
        head_path = root / ".git" / "HEAD"
        try:
            head = head_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if head.startswith("ref: "):
            ref_path = root / ".git" / head.removeprefix("ref: ")
            try:
                head = ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        if len(head) < 7 or any(
            character not in "0123456789abcdefABCDEF" for character in head
        ):
            return None
        return head.lower()
