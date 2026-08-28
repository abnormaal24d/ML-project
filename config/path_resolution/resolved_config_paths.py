"""Portable path formatting helpers for persisted project payloads."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any


def format_project_path(
    value: str | Path,
    *,
    project_root: str | Path | None = None,
) -> str:
    """Return a stable POSIX-style path for JSON payloads and logs."""

    raw_text = str(value)
    if _looks_like_url(raw_text):
        return raw_text

    root = Path(project_root or Path.cwd()).resolve()
    resolved = _format_path_text(raw_text=raw_text, root=root)
    if resolved:
        return resolved

    return raw_text.replace("\\", "/")


def relativize_payload_paths(
    payload: Any,
    *,
    project_root: str | Path | None = None,
) -> Any:
    """Recursively convert filesystem paths in a JSON-like payload."""

    if isinstance(payload, Path):
        return format_project_path(payload, project_root=project_root)

    if isinstance(payload, str):
        return format_project_path(payload, project_root=project_root)

    if isinstance(payload, dict):
        return {
            key: relativize_payload_paths(value, project_root=project_root)
            for key, value in payload.items()
        }

    if isinstance(payload, tuple):
        return tuple(
            relativize_payload_paths(item, project_root=project_root)
            for item in payload
        )

    if isinstance(payload, list):
        return [
            relativize_payload_paths(item, project_root=project_root)
            for item in payload
        ]

    return payload


def _format_path_text(*, raw_text: str, root: Path) -> str | None:
    """Try to render raw path text relative to the project root.

    ``raw_text`` may contain an absolute POSIX path, an absolute Windows path,
    or plain text. ``root`` is the resolved project directory used for
    relativization. The helper returns a POSIX-style relative path when it can
    recognize the input as a filesystem path, otherwise ``None``.
    """

    candidate = Path(raw_text)
    if candidate.is_absolute():
        return _format_absolute_path(candidate=candidate, root=root)

    if _looks_like_windows_path(raw_text):
        return _format_windows_path(raw_text=raw_text, root=root)

    return None


def _format_absolute_path(*, candidate: Path, root: Path) -> str:
    """Return an absolute path relative to ``root`` when possible.

    Path resolution can raise ``OSError`` for invalid or inaccessible paths,
    while ``relative_to`` raises ``ValueError`` when ``candidate`` is outside
    the root. In both cases the absolute POSIX path is returned as a fallback.
    Absolute Windows paths are avoided where possible by using relative or
    best-effort POSIX forms for portable artifacts.
    """

    root_res = root.resolve(strict=False)
    try:
        # Prefer non-strict resolve + relative for robustness on Windows
        cand_res = candidate.resolve(strict=False)
        relative = cand_res.relative_to(root_res)
        return relative.as_posix()
    except (OSError, ValueError, RuntimeError):
        pass

    # Best effort string-based relativization (handles some non-existing and drive cases)
    try:
        root_str = root_res.as_posix().rstrip("/")
        cand_str = candidate.as_posix()
        if cand_str.startswith(root_str + "/"):
            return cand_str[len(root_str) + 1 :]
        # same drive but outside: use relative_to on pure if possible
        if candidate.drive == root.drive:
            rel = candidate.relative_to(root)
            return rel.as_posix()
    except (ValueError, OSError, AttributeError, RuntimeError):
        pass

    # Fallback: environment-bound absolute (documented as non-portable)
    # but normalized to POSIX to avoid raw Windows backslashes
    return candidate.as_posix()


def _format_windows_path(*, raw_text: str, root: Path) -> str:
    """Convert Windows-style path text to a project-relative POSIX path.

    ``raw_text`` is parsed as a ``PureWindowsPath`` and compared against the
    project ``root``. The return value is a POSIX path, either relative to the
    root or absolute when the path is outside the project directory.
    """

    candidate = PureWindowsPath(raw_text)
    root_candidate = PureWindowsPath(str(root))
    try:
        relative = candidate.relative_to(root_candidate)
    except ValueError:
        return candidate.as_posix()

    return relative.as_posix()


def _looks_like_url(value: str) -> bool:
    """Return whether text should be treated as a URL instead of a path.

    The heuristic checks for ``://`` plus common non-file schemes such as
    ``mailto:`` and ``urn:``.
    """

    lower = value.lower()
    return "://" in lower or lower.startswith(("mailto:", "urn:"))


def _looks_like_windows_path(value: str) -> bool:
    """Return whether text resembles an absolute Windows path."""

    normalized = value.replace("/", "\\")
    return len(normalized) >= 3 and normalized[1:3] == ":\\"
