"""Checksum-verified and atomic checkpoint I/O.

Every checkpoint is persisted as one canonical JSON manifest stored at
``checkpoint_path`` itself. The manifest is either versioned (a model file
lives in an adjacent version directory) or blob-backed (the model lives in
an external content-addressable store). There is no direct-model fallback:
a checkpoint path that is not a valid canonical manifest fails closed.
"""

from __future__ import annotations

import json
import os
import pickle  # nosec B403
import re
import shutil
import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch

from preprocessing.provenance import hash_file

MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_VERSIONED = "versioned"
_MANIFEST_BLOB = "blob"


def _read_json_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def resolve_checkpoint_model_path(checkpoint_path: Path) -> Path | None:
    """Resolve the real model file for a canonical checkpoint manifest.

    Returns ``None`` (rather than guessing) when *checkpoint_path* is not a
    valid canonical manifest, so callers can fail closed.
    """

    data = _read_json_object(checkpoint_path)
    if data is None:
        return None
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return None

    kind = data.get("kind")
    file_name = data.get("file")
    sha256 = data.get("sha256")
    if not isinstance(file_name, str) or not file_name.strip():
        return None

    if kind == _MANIFEST_BLOB:
        blob_storage = data.get("blob_storage")
        if not isinstance(blob_storage, str) or not blob_storage.strip():
            return None
        if not isinstance(sha256, str):
            return None
        expected_sha256 = sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            return None
        root = Path(blob_storage).resolve()
        canonical_path = (
            root / expected_sha256[:2] / expected_sha256 / "checkpoint.pt"
        )
        if not canonical_path.is_file():
            return None
        if canonical_path.is_symlink():
            return None
        try:
            resolved_path = canonical_path.resolve(strict=True)
        except OSError:
            return None
        if not resolved_path.is_relative_to(root):
            return None
        actual_sha256 = _file_sha256(resolved_path)
        if actual_sha256 != expected_sha256:
            return None
        return resolved_path

    if kind != _MANIFEST_VERSIONED:
        return None

    version_dir = data.get("version_dir")
    if not isinstance(version_dir, str) or not version_dir.strip():
        return None
    relative_version_dir = Path(version_dir)
    relative_file = Path(file_name)
    if relative_version_dir.is_absolute() or relative_file.is_absolute():
        return None
    if relative_file.name != file_name:
        return None
    version_root = (
        checkpoint_path.parent / f"{checkpoint_path.name}.d"
    ).resolve()
    candidate = (
        checkpoint_path.parent / relative_version_dir / relative_file
    ).resolve()
    if not candidate.is_relative_to(version_root):
        return None

    if sha256 is not None:
        if not isinstance(sha256, str) or len(sha256) != 64:
            return None
        expected_sha256 = sha256.strip().lower()
        sidecar_path = candidate.with_name(candidate.name + ".sha256")
        if not sidecar_path.is_file():
            return None
        sidecar_content = (
            sidecar_path.read_text(encoding="ascii").strip().split()
        )
        if not sidecar_content:
            return None
        sidecar_sha256 = sidecar_content[0].lower()
        if sidecar_sha256 != expected_sha256:
            return None
        actual_sha256 = _file_sha256(candidate)
        if actual_sha256 != expected_sha256:
            return None

    return candidate


def checkpoint_is_available(checkpoint_path: Path) -> bool:
    """Return whether a complete checksum-backed checkpoint is readable."""

    actual_path = resolve_checkpoint_model_path(checkpoint_path)
    if actual_path is None:
        return False
    return (
        actual_path.is_file()
        and actual_path.with_name(actual_path.name + ".sha256").is_file()
    )


def safe_torch_load(path: Path) -> Any:
    """Load a canonical checkpoint without permitting arbitrary Python objects.

    The real model file is resolved through the canonical manifest and its
    checksum sidecar is verified before deserialization.
    """

    actual_path = resolve_checkpoint_model_path(path)
    if actual_path is None:
        raise FileNotFoundError(
            f"checkpoint is not a canonical manifest: {path}"
        )
    _verify_checkpoint_checksum(checkpoint_path=actual_path)
    try:
        return torch.load(actual_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return _load_with_numpy_safe_globals(actual_path)


def _load_with_numpy_safe_globals(path: Path) -> Any:
    import numpy as np
    from torch.serialization import safe_globals

    try:
        multiarray = import_module("numpy._core.multiarray")
    except ImportError:
        multiarray = import_module("numpy.core.multiarray")

    allowed: list[Callable[..., Any] | tuple[Callable[..., Any], str]] = [
        cast(Callable[..., Any], multiarray._reconstruct),
        cast(Callable[..., Any], np.ndarray),
        cast(Callable[..., Any], np.dtype),
    ]
    for dtype_name in (
        "float32",
        "float64",
        "int32",
        "int64",
        "uint8",
        "uint32",
    ):
        allowed.append(cast(Callable[..., Any], type(np.dtype(dtype_name))))

    with safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_checksum_path(checkpoint_path: Path) -> Path:
    """Return the checksum sidecar for a checkpoint manifest.

    Resolves the real model file through the canonical manifest and returns
    the checksum sidecar next to it.
    """

    actual_path = resolve_checkpoint_model_path(checkpoint_path)
    if actual_path is None:
        raise FileNotFoundError(
            f"checkpoint is not a canonical manifest: {checkpoint_path}"
        )
    return actual_path.with_name(actual_path.name + ".sha256")


def atomic_torch_save(
    *,
    payload: object,
    checkpoint_path: Path,
    blob_storage: Path | None = None,
) -> None:
    """Atomically save a torch payload as a canonical checkpoint manifest.

    Without blob storage the model is written into a private version
    directory and ``checkpoint_path`` is atomically replaced with the
    versioned manifest. With blob storage the model bytes are written
    content-addressably into the store and ``checkpoint_path`` becomes a
    blob-backed manifest pointing at it. There is exactly one commit marker.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_name = checkpoint_path.name

    if blob_storage is not None:
        _atomic_blob_save(
            payload=payload,
            checkpoint_path=checkpoint_path,
            blob_storage=blob_storage,
        )
        return

    version_root = checkpoint_path.parent / f"{checkpoint_name}.d"
    version_root.mkdir(parents=True, exist_ok=True)
    version_dir = Path(tempfile.mkdtemp(dir=str(version_root), prefix="v-"))
    version_model_path = version_dir / checkpoint_name
    version_checksum_path = version_dir / (checkpoint_name + ".sha256")
    relative_version_dir = str(
        version_dir.relative_to(checkpoint_path.parent).as_posix()
    )

    tmp_path: Path | None = None
    published = False
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(version_dir),
            prefix=f".{checkpoint_name}.",
            suffix=".tmp",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        torch.save(payload, tmp_path)
        with tmp_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        checksum = _file_sha256(tmp_path)

        checksum_tmp = version_dir / (checkpoint_name + ".sha256.tmp")
        with checksum_tmp.open("w", encoding="ascii") as handle:
            handle.write(f"{checksum}  {checkpoint_name}\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(str(tmp_path), str(version_model_path))
        tmp_path = None
        os.replace(str(checksum_tmp), str(version_checksum_path))
        _fsync_directory(version_dir)

        _atomic_write_json(
            path=checkpoint_path,
            payload={
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": _MANIFEST_VERSIONED,
                "version_dir": relative_version_dir,
                "file": checkpoint_name,
                "sha256": checksum,
                "timestamp_ns": version_model_path.stat().st_mtime_ns,
            },
        )
        published = True
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        (version_dir / (checkpoint_name + ".sha256.tmp")).unlink(
            missing_ok=True
        )
        if not published:
            shutil.rmtree(version_dir, ignore_errors=True)


def _atomic_blob_save(
    *,
    payload: object,
    checkpoint_path: Path,
    blob_storage: Path,
) -> None:
    root = blob_storage.resolve()
    root.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(root),
        prefix=".blob.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        sha256 = _file_sha256(tmp_path)
        target_dir = root / sha256[:2] / sha256
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "checkpoint.pt"
        checksum_path = target_dir / ("checkpoint.pt.sha256")

        if model_path.exists():
            if _file_sha256(model_path) != sha256:
                raise ValueError(
                    "blob store collision with different content: "
                    f"{model_path}"
                )
        else:
            os.replace(str(tmp_path), str(model_path))
            checksum_tmp = target_dir / "checkpoint.pt.sha256.tmp"
            with checksum_tmp.open("w", encoding="ascii") as handle:
                handle.write(f"{sha256}  checkpoint.pt\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(checksum_tmp), str(checksum_path))
            _fsync_directory(target_dir)

        _atomic_write_json(
            path=checkpoint_path,
            payload={
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": _MANIFEST_BLOB,
                "blob_storage": root.as_posix(),
                "file": "checkpoint.pt",
                "sha256": sha256,
            },
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(*, path: Path, payload: Mapping[str, object]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
        _fsync_directory(path.parent)
    finally:
        tmp_path.unlink(missing_ok=True)


def _verify_checkpoint_checksum(*, checkpoint_path: Path) -> None:
    checksum_path = checkpoint_path.with_name(checkpoint_path.name + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"checkpoint checksum is required: {checksum_path}"
        )
    parts = checksum_path.read_text(encoding="ascii").strip().split()
    if not parts:
        raise ValueError(f"checkpoint checksum is empty: {checksum_path}")
    expected = parts[0].lower()
    actual = _file_sha256(checkpoint_path)
    if expected != actual:
        raise ValueError(
            "checkpoint checksum mismatch before deserialization: "
            f"{checkpoint_path}"
        )


def _file_sha256(path: Path) -> str:
    return hash_file(path)


def checkpoint_sha256(checkpoint_path: Path) -> str:
    """Return the verified SHA-256 digest of the real checkpoint model file."""

    actual_path = resolve_checkpoint_model_path(checkpoint_path)
    if actual_path is None:
        raise FileNotFoundError(
            f"checkpoint is not a canonical manifest: {checkpoint_path}"
        )
    _verify_checkpoint_checksum(checkpoint_path=actual_path)
    return _file_sha256(actual_path)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "checkpoint_checksum_path",
    "checkpoint_is_available",
    "checkpoint_sha256",
    "resolve_checkpoint_model_path",
    "safe_torch_load",
    "atomic_torch_save",
]
