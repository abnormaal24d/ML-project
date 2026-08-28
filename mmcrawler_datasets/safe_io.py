"""Bounded, dataset-root-confined input helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 250_000
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_SHARD_ENTRIES = 100_000
MAX_TAR_MEMBERS = 100_000


class _BinaryReadable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def resolve_dataset_reference(
    *,
    dataset_root: Path,
    reference: str | Path,
    label: str,
    allow_absolute: bool = False,
) -> Path:
    """Resolve a relative reference and reject traversal and symlink escapes."""

    root = Path(dataset_root).resolve()
    raw = Path(reference)
    if raw.is_absolute() and not allow_absolute:
        raise ValueError(f"{label} must be relative to the dataset root")
    if not raw.parts or str(raw).strip() in {"", "."}:
        raise ValueError(f"{label} must not be empty")
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the dataset root") from exc
    return resolved


def load_bounded_json_object(
    *,
    path: Path,
    max_bytes: int = MAX_METADATA_BYTES,
) -> dict[str, Any]:
    """Load a size- and complexity-bounded JSON object."""

    payload = _read_bounded_bytes(path=path, max_bytes=max_bytes)
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON metadata must contain an object: {path}")
    _validate_json_complexity(value=value, path=path)
    return value


def read_jsonl(
    *,
    path: Path,
    max_bytes: int = MAX_RECORD_BYTES,
) -> list[dict[str, Any]]:
    """Read every row of a newline-delimited JSON file."""
    return list(iter_jsonl(path=path, max_bytes=max_bytes))


def iter_bounded_jsonl(
    *,
    path: Path,
    max_bytes: int = MAX_RECORD_BYTES,
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a newline-delimited file with byte-bounded I/O.

    Reads each line with readline(max_bytes + 1), validates byte length before
    decoding, enforces UTF-8, parses JSON, and validates complexity bounds.
    """
    try:
        with path.open("rb") as handle:
            line_number = 0
            while True:
                line = handle.readline(max_bytes + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > max_bytes:
                    raise ValueError(
                        f"JSONL line {line_number} exceeds "
                        f"{max_bytes} bytes: {path}"
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    text = stripped.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"JSONL line {line_number} is not valid UTF-8: {path}"
                    ) from exc
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {path} on line "
                        f"{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"expected JSON object in {path} on line {line_number}"
                    )
                _validate_json_complexity(value=payload, path=path)
                yield payload
    except FileNotFoundError as exc:
        raise ValueError(f"JSONL file is missing: {path}") from exc
    except OSError as exc:
        raise ValueError(f"JSONL file cannot be read: {path}") from exc


def iter_jsonl(
    *,
    path: Path,
    max_bytes: int = MAX_RECORD_BYTES,
) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a newline-delimited file, one per line.

    Delegates to iter_bounded_jsonl for byte-bounded, complexity-checked I/O.
    """
    yield from iter_bounded_jsonl(path=path, max_bytes=max_bytes)


def read_bounded_text(
    *, handle: _BinaryReadable, max_bytes: int, label: str
) -> str:
    """Read at most ``max_bytes`` from a binary stream and decode UTF-8."""

    payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc


def _read_bounded_bytes(*, path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot stat JSON metadata: {path}") from exc
    if size > max_bytes:
        raise ValueError(f"JSON metadata exceeds {max_bytes} bytes: {path}")
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"cannot read JSON metadata: {path}") from exc
    if len(payload) > max_bytes:
        raise ValueError(f"JSON metadata exceeds {max_bytes} bytes: {path}")
    return payload


def _validate_json_complexity(*, value: object, path: Path) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError(f"JSON metadata has too many values: {path}")
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON metadata is nested too deeply: {path}")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
