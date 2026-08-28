"""Chromaprint fingerprinting through a pinned local argv-only executable."""

from __future__ import annotations

import json
import subprocess  # nosec: B404
from pathlib import Path

from preprocessing.media.adapters.versioned_executable import (
    ExecutableVerificationError,
    resolve_and_verify_executable,
)


class AudioFingerprintError(RuntimeError):
    """Raised when Chromaprint cannot produce trustworthy evidence."""


def calculate_audio_fingerprint(
    path: Path,
    *,
    executable: str,
    expected_version: str,
    timeout_seconds: float,
) -> str:
    """Return a locally calculated Chromaprint fingerprint.

    The executable version is pinned and both subprocesses are checked. Empty,
    malformed, or non-string fingerprints are rejected. A byte hash is never
    used as a fallback because it is not an acoustic fingerprint.
    """

    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        raise AudioFingerprintError("audio_fingerprint_input_missing")

    try:
        resolved_executable, _ = resolve_and_verify_executable(
            tool_name="fpcalc",
            configured_executable=executable,
            expected_version=expected_version,
            timeout_seconds=timeout_seconds,
            required=True,
        )
    except ExecutableVerificationError as error:
        raise AudioFingerprintError(str(error)) from error

    result = _run_fpcalc(
        command=(resolved_executable, "-json", str(candidate)),
        executable=resolved_executable,
        timeout_seconds=timeout_seconds,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioFingerprintError(
            "audio_fingerprint_output_invalid_json"
        ) from exc
    if not isinstance(payload, dict):
        raise AudioFingerprintError("audio_fingerprint_output_invalid")

    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise AudioFingerprintError("audio_fingerprint_empty")
    normalized = fingerprint.strip()
    if len(normalized) < 8 or any(
        character.isspace() for character in normalized
    ):
        raise AudioFingerprintError("audio_fingerprint_malformed")
    return normalized


def _run_fpcalc(
    *,
    command: tuple[str, ...],
    executable: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    if not command or command[0] != executable:
        raise AudioFingerprintError("chromaprint_command_invalid")
    try:
        # The pinned executable receives fixed flags and a resolved input path.
        result = subprocess.run(  # nosec: B603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise AudioFingerprintError("chromaprint_tool_missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioFingerprintError("chromaprint_timeout") from exc
    except OSError as exc:
        raise AudioFingerprintError("chromaprint_execution_failed") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        detail = detail[:240] or "no_output"
        raise AudioFingerprintError(
            f"chromaprint_failed:exit={result.returncode}:detail={detail}"
        )
    return result


__all__ = [
    "AudioFingerprintError",
    "calculate_audio_fingerprint",
]
