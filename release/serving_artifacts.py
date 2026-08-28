"""Fail-closed validation for deployable model release artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence, cast

from schemas.release import ReleaseReason, detail

ServingFormatStatus = Literal["ok", "skipped", "missing", "invalid"]

SERVING_FORMATS: Mapping[str, tuple[str, str]] = {
    "safetensors": (
        "model.safetensors",
        "safetensors_export_status.json",
    ),
    "torchscript": (
        "model.torchscript.pt",
        "torchscript_export_status.json",
    ),
    "onnx": (
        "model.onnx",
        "onnx_export_status.json",
    ),
}

_SKIPPED_STATUSES = frozenset({"skipped", "failed", "error"})
_SUCCESS_STATUSES = frozenset({"ok", "success", "available", "exported"})


class _SafeTensorSlice(Protocol):
    def get_shape(self) -> Sequence[int]: ...


class _SafeTensorHandle(Protocol):
    def keys(self) -> Sequence[str]: ...

    def get_slice(self, key: str) -> _SafeTensorSlice: ...


class _TorchScriptModule(Protocol):
    def eval(self) -> object: ...


class _OnnxChecker(Protocol):
    def check_model(self, model: object, *, _full_check: bool) -> None: ...


class _OnnxModule(Protocol):
    checker: _OnnxChecker

    def load(self, path: str, *, _load_external_data: bool) -> object: ...


@dataclass(frozen=True, slots=True)
class ServingArtifactPolicy:
    """Explicit serving contract for one release mode."""

    required_any_of: frozenset[str] = frozenset()
    required_all_of: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = (
            set(self.required_any_of) | set(self.required_all_of)
        ) - set(SERVING_FORMATS)
        if unknown:
            raise ValueError(
                "unknown serving formats: " + ", ".join(sorted(unknown))
            )


def default_serving_policy(mode: str) -> ServingArtifactPolicy:
    """Return the normative serving contract for a release mode."""

    if mode in {"candidate", "production_model"}:
        return ServingArtifactPolicy(
            required_any_of=frozenset(SERVING_FORMATS),
        )
    return ServingArtifactPolicy()


@dataclass(frozen=True, slots=True)
class ServingArtifactInspection:
    """Per-format validated availability of deployment artifacts."""

    status_by_format: Mapping[str, ServingFormatStatus]
    available: frozenset[str]
    skipped: frozenset[str]
    invalid: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        return frozenset(
            fmt
            for fmt, status in self.status_by_format.items()
            if status == "missing"
        )


@dataclass(frozen=True, slots=True)
class _ExportStatus:
    outcome: Literal["ok", "skipped", "missing", "invalid"]
    sha256: str | None = None


def inspect_serving_artifacts(
    *,
    export_directory: Path,
) -> ServingArtifactInspection:
    """Validate canonical serving formats, their receipts, and checksums.

    An artifact is available only when its status receipt exists, declares a
    successful export, contains a valid SHA-256 digest, matches the artifact,
    and the artifact can be opened by its native runtime/parser.
    """

    status_by_format: dict[str, ServingFormatStatus] = {}
    for fmt, (artifact_name, status_name) in SERVING_FORMATS.items():
        artifact = export_directory / artifact_name
        receipt = _read_export_status(export_directory / status_name)
        status_by_format[fmt] = _classify_artifact(
            fmt=fmt,
            artifact=artifact,
            receipt=receipt,
        )

    available = frozenset(
        fmt for fmt, status in status_by_format.items() if status == "ok"
    )
    skipped = frozenset(
        fmt for fmt, status in status_by_format.items() if status == "skipped"
    )
    invalid = frozenset(
        fmt for fmt, status in status_by_format.items() if status == "invalid"
    )
    return ServingArtifactInspection(
        status_by_format=status_by_format,
        available=available,
        skipped=skipped,
        invalid=invalid,
    )


def check_serving_artifacts(
    *,
    export_directory: Path,
    policy: ServingArtifactPolicy,
) -> tuple[str, ...]:
    """Return every serving-artifact release reason for one policy."""

    required = set(policy.required_any_of) | set(policy.required_all_of)
    if not required:
        return ()

    inspection = inspect_serving_artifacts(export_directory=export_directory)
    reasons: list[str] = []

    for fmt in sorted(policy.required_all_of):
        reason = _failure_reason(fmt=fmt, inspection=inspection)
        if reason is not None:
            reasons.append(reason)

    if policy.required_any_of and not (
        policy.required_any_of & inspection.available
    ):
        for fmt in sorted(policy.required_any_of):
            reason = _failure_reason(fmt=fmt, inspection=inspection)
            if reason is not None:
                reasons.append(reason)

    return tuple(reasons)


def _failure_reason(
    *,
    fmt: str,
    inspection: ServingArtifactInspection,
) -> str | None:
    if fmt in inspection.available:
        return None
    if fmt in inspection.skipped:
        return detail(ReleaseReason.SERVING_EXPORT_SKIPPED, fmt)
    if fmt in inspection.invalid:
        return detail(ReleaseReason.SERVING_ARTIFACT_INVALID, fmt)
    return detail(ReleaseReason.SERVING_ARTIFACT_MISSING, fmt)


def _classify_artifact(
    *,
    fmt: str,
    artifact: Path,
    receipt: _ExportStatus,
) -> ServingFormatStatus:
    if receipt.outcome == "skipped":
        return "skipped"
    if receipt.outcome == "invalid":
        return "invalid"
    if receipt.outcome == "missing":
        return "invalid" if artifact.exists() else "missing"
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        return "invalid"
    if receipt.sha256 is None or _sha256(artifact) != receipt.sha256:
        return "invalid"
    try:
        _validate_artifact_format(fmt=fmt, artifact=artifact)
    except Exception:  # native format parsers expose library-specific errors
        return "invalid"
    return "ok"


def _read_export_status(path: Path) -> _ExportStatus:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _ExportStatus("missing")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _ExportStatus("invalid")
    if not isinstance(payload, Mapping):
        return _ExportStatus("invalid")
    status = payload.get("status")
    if not isinstance(status, str):
        return _ExportStatus("invalid")
    lowered = status.strip().lower()
    if lowered in _SKIPPED_STATUSES:
        return _ExportStatus("skipped")
    if lowered not in _SUCCESS_STATUSES:
        return _ExportStatus("invalid")
    digest = payload.get("sha256")
    if not isinstance(digest, str):
        return _ExportStatus("invalid")
    normalized = digest.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return _ExportStatus("invalid")
    return _ExportStatus("ok", normalized)


def _validate_artifact_format(*, fmt: str, artifact: Path) -> None:
    if fmt == "safetensors":
        from safetensors import safe_open

        typed_safe_open = cast(
            Callable[..., AbstractContextManager[_SafeTensorHandle]],
            safe_open,
        )
        with typed_safe_open(artifact, framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            if not keys:
                raise ValueError("safetensors artifact contains no tensors")
            for key in keys:
                shape = tuple(handle.get_slice(key).get_shape())
                if any(int(dimension) < 0 for dimension in shape):
                    raise ValueError(
                        "safetensors artifact has an invalid shape"
                    )
        return
    if fmt == "torchscript":
        import torch

        jit_load = cast(Callable[..., _TorchScriptModule], torch.jit.load)
        module = jit_load(str(artifact), map_location="cpu")
        module.eval()
        return
    if fmt == "onnx":
        onnx = cast(_OnnxModule, importlib.import_module("onnx"))
        model = onnx.load(str(artifact), load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        return
    raise ValueError(f"unsupported serving format: {fmt}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SERVING_FORMATS",
    "ServingArtifactInspection",
    "ServingArtifactPolicy",
    "ServingFormatStatus",
    "check_serving_artifacts",
    "default_serving_policy",
    "inspect_serving_artifacts",
]
