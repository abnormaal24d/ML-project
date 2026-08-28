"""Canonical executable resolution and version verification.

Single source of truth for resolving configured executable names/paths,
verifying exact semantic versions, and returning absolute paths for all
subsequent invocations.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_SEMANTIC_VERSION = r"(?P<version>\d+(?:\.\d+){2})"


def _which(executable: str) -> str | None:
    """Internal which function to allow monkeypatching in tests."""
    return shutil.which(executable)


class ExecutableVerificationError(RuntimeError):
    """A configured executable could not be safely resolved or verified."""


def resolve_executable(configured_executable: str) -> str | None:
    """Resolve a configured executable name or path to an absolute path.

    Returns None if the executable cannot be found on PATH or resolved.
    """
    configured = configured_executable.strip()
    resolved = _which(configured)
    if resolved is None:
        return None

    try:
        return str(Path(resolved).resolve())
    except (OSError, RuntimeError):
        return None


def resolve_and_verify_executable(
    *,
    tool_name: str,
    configured_executable: str,
    expected_version: str | None,
    timeout_seconds: float,
    required: bool,
) -> tuple[str, str | None]:
    """Resolve, verify version, and return absolute path + observed version.

    The essential invariant:
        which(configured)
            → strict absolute resolution
            → version probe with exact same absolute path
            → all later commands use that same path

    Args:
        tool_name: Logical name for error messages (e.g., "ffmpeg", "fpcalc")
        configured_executable: Name or path from configuration
        expected_version: Exact semantic version required, or None to skip check
        timeout_seconds: Timeout for version probe
        required: Whether the executable is mandatory (raises if missing)

    Returns:
        Tuple of (absolute_executable_path, observed_version_or_None)

    Raises:
        ExecutableVerificationError: If required and any check fails
    """
    configured = configured_executable.strip()
    executable_path = resolve_executable(configured)
    must_verify = required or expected_version is not None

    if executable_path is None:
        if must_verify:
            raise ExecutableVerificationError(
                f"{tool_name}_executable_unavailable"
            )
        return configured, None

    try:
        completed = subprocess.run(  # nosec: B603
            (executable_path, "-version"),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        if must_verify:
            raise ExecutableVerificationError(
                f"{tool_name}_version_probe_failed"
            ) from error
        return executable_path, None

    if completed.returncode != 0:
        if must_verify:
            raise ExecutableVerificationError(
                f"{tool_name}_version_probe_failed:exit={completed.returncode}"
            )
        return executable_path, None

    output = "\n".join(
        value for value in (completed.stdout, completed.stderr) if value
    )
    match = re.search(
        rf"(?im)^\s*{re.escape(tool_name)}\s+version\s+"
        rf"{_SEMANTIC_VERSION}\b",
        output,
    )
    if match is None:
        if must_verify:
            raise ExecutableVerificationError(
                f"{tool_name}_version_unreadable"
            )
        return executable_path, None

    observed_version = match.group("version")
    if expected_version is not None and observed_version != expected_version:
        raise ExecutableVerificationError(
            f"{tool_name}_version_mismatch:"
            f"expected={expected_version}:actual={observed_version}"
        )

    return executable_path, observed_version


__all__ = [
    "ExecutableVerificationError",
    "resolve_executable",
    "resolve_and_verify_executable",
]
