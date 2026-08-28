"""Manifest serialization for workflow artifact manifests."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunArtifactIdentity:
    """Immutable identity and reproducibility evidence for one generation."""

    generation_id: str
    workflow_id: str
    project_fingerprint: str
    config_fingerprint: str
    environment_name: str
    environment_fingerprint: str
    python_version: str
    dependency_lock_fingerprint: str

    def __post_init__(self) -> None:
        """Validate all identity fields."""

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field.name} must be a non-empty string")

    def manifest_fields(self) -> dict[str, str]:
        """Return constructor fields shared by every artifact manifest."""

        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactManifest:
    """
    Base manifest serialization with deterministic JSON-compatible payloads.
    """

    generation_id: str
    workflow_id: str
    project_fingerprint: str
    config_fingerprint: str
    environment_name: str
    environment_fingerprint: str
    python_version: str
    dependency_lock_fingerprint: str

    def __post_init__(self) -> None:
        """Reject manifests that lack generation or reproducibility identity."""

        RunArtifactIdentity(**self.identity_fields())

    def identity_fields(self) -> dict[str, str]:
        """Return the canonical run identity embedded in this manifest."""

        return {
            "generation_id": self.generation_id,
            "workflow_id": self.workflow_id,
            "project_fingerprint": self.project_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "environment_name": self.environment_name,
            "environment_fingerprint": self.environment_fingerprint,
            "python_version": self.python_version,
            "dependency_lock_fingerprint": (self.dependency_lock_fingerprint),
        }

    @classmethod
    def identity_from_payload(
        cls,
        payload: dict[str, object],
    ) -> dict[str, str]:
        """Read required generation identity from a serialized manifest."""

        result: dict[str, str] = {}

        for name in (
            "generation_id",
            "workflow_id",
            "project_fingerprint",
            "config_fingerprint",
            "environment_name",
            "environment_fingerprint",
            "python_version",
            "dependency_lock_fingerprint",
        ):
            value = payload.get(name)

            if not isinstance(value, str):
                raise ValueError(
                    f"manifest field {name} must be a non-empty string"
                )

            normalized = value.strip()

            if not normalized:
                raise ValueError(
                    f"manifest field {name} must be a non-empty string"
                )

            result[name] = normalized

        return result

    def to_payload(self) -> dict[str, object]:
        """
        Return the manifest as a normalized JSON-compatible payload.

        Values are normalized field-by-field to avoid the deep-copy behavior
        of dataclasses.asdict().
        """

        payload: dict[str, object] = {}

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            payload[field.name] = self._normalize_value(value)

        return payload

    @staticmethod
    def _parse_count(
        value: object,
        *,
        field: str = "unknown",
        default: int = 0,
        nonnegative: bool = True,
    ) -> int:
        """Parse one manifest count using the canonical strict schema."""

        if value is None:
            return default

        if isinstance(value, bool):
            raise ValueError(f"{field}: bool not allowed for count")

        if not isinstance(value, int):
            raise ValueError(
                f"{field}: count must be an integer, "
                f"got {type(value).__name__}"
            )

        if nonnegative and value < 0:
            raise ValueError(f"{field}: count must be >= 0, got {value}")

        return value

    @staticmethod
    def as_int(
        value: object,
        default: int = 0,
    ) -> int:
        """Parse an integer manifest value."""

        return ArtifactManifest._parse_count(
            value,
            default=default,
        )

    @staticmethod
    def as_opt_int(
        value: object,
    ) -> int | None:
        """Parse an optional integer manifest value."""

        if value is None:
            return None

        return ArtifactManifest._parse_count(value)

    @staticmethod
    def as_opt_str(
        value: object,
    ) -> str | None:
        """Parse an optional non-empty string."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("manifest text value must be a string")

        text = value.strip()

        return text or None

    @staticmethod
    def as_required_str(
        value: object,
        *,
        field: str,
    ) -> str:
        """Parse a required non-empty string."""

        text = ArtifactManifest.as_opt_str(value)

        if text is None:
            raise ValueError(
                f"manifest field {field} must be a non-empty string"
            )

        return text

    @classmethod
    def as_required_path(
        cls,
        value: object,
        *,
        field: str,
    ) -> Path:
        """Parse a required manifest path."""

        text = cls.as_required_str(
            value,
            field=field,
        )

        path = Path(text)

        if path == Path("."):
            raise ValueError(f"manifest field {field} must not be '.'")

        return path

    @staticmethod
    def as_opt_path(
        value: object,
    ) -> Path | None:
        """Parse an optional manifest path."""

        text = ArtifactManifest.as_opt_str(value)

        if text is None:
            return None

        return Path(text)

    @staticmethod
    def as_bool(
        value: object,
        *,
        default: bool = False,
    ) -> bool:
        """Parse a strict boolean manifest value."""

        if isinstance(value, bool):
            return value

        if value is None:
            return default

        raise ValueError("manifest boolean value must be a boolean")

    @staticmethod
    def _normalize_value(
        value: object,
    ) -> object:
        """
        Normalize a value to a deterministic JSON-compatible representation.

        Unsupported objects are rejected instead of being silently converted
        to strings.
        """

        if isinstance(value, Path):
            return value.as_posix()

        if isinstance(value, Enum):
            return ArtifactManifest._normalize_value(value.value)

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: ArtifactManifest._normalize_value(
                    getattr(value, field.name)
                )
                for field in dataclasses.fields(value)
            }

        if isinstance(value, (set, frozenset)):
            normalized_items = [
                ArtifactManifest._normalize_value(item) for item in value
            ]

            normalized_items.sort(key=ArtifactManifest._stable_sort_key)

            return normalized_items

        if isinstance(value, dict):
            return {
                str(key): ArtifactManifest._normalize_value(item)
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            }

        if isinstance(value, (tuple, list)):
            return [ArtifactManifest._normalize_value(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        raise TypeError(
            "cannot normalize non-JSON value of type "
            f"{type(value).__name__} "
            f"(value repr starts with: {repr(value)[:80]})"
        )

    @staticmethod
    def _stable_sort_key(
        value: object,
    ) -> str:
        """Return a deterministic ordering key for normalized set values."""

        if value is None:
            return ""

        return str(value)


def format_manifest_path(
    path: Path | None,
) -> str | None:
    """Return a stable POSIX path string for a manifest field."""

    if path is None:
        return None

    return path.as_posix()
