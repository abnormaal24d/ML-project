from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from preprocessing.media.audio.audio_fingerprint import (
    AudioFingerprintError,
    calculate_audio_fingerprint,
)


def test_calculate_audio_fingerprint_checks_pinned_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=(r"C:\tools\configured-fpcalc", "-version"),
                returncode=0,
                stdout="fpcalc version 1.5.1\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(r"C:\tools\configured-fpcalc", "-json", str(audio)),
                returncode=0,
                stdout='{"duration": 2, "fingerprint": "AQADtMkkmY"}',
                stderr="",
            ),
        )
    )

    def mock_which(executable: str) -> str:
        return f"/tools/{executable}"

    monkeypatch.setattr(
        "preprocessing.media.adapters.versioned_executable._which",
        mock_which,
    )

    def run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(subprocess, "run", run)

    assert (
        calculate_audio_fingerprint(
            audio,
            executable="configured-fpcalc",
            expected_version="1.5.1",
            timeout_seconds=17.5,
        )
        == "AQADtMkkmY"
    )
    # The resolved path will be used for the second call
    assert calls[0][0] == (r"C:\tools\configured-fpcalc", "-version")
    assert calls[1][0][0] == r"C:\tools\configured-fpcalc"
    assert calls[1][0][1] == "-json"
    assert all(kwargs["timeout"] == 17.5 for _command, kwargs in calls)


def test_calculate_audio_fingerprint_rejects_wrong_tool_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    def mock_which(executable: str) -> str:
        return f"/tools/{executable}"

    monkeypatch.setattr(
        "preprocessing.media.adapters.versioned_executable._which",
        mock_which,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=(r"C:\tools\fpcalc", "-version"),
            returncode=0,
            stdout="fpcalc version 1.4.3\n",
            stderr="",
        ),
    )

    with pytest.raises(AudioFingerprintError, match="version_mismatch"):
        calculate_audio_fingerprint(
            audio,
            executable="fpcalc",
            expected_version="1.5.1",
            timeout_seconds=20.0,
        )


def test_calculate_audio_fingerprint_never_accepts_empty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=(r"C:\tools\fpcalc", "-version"),
                returncode=0,
                stdout="fpcalc version 1.5.1\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(r"C:\tools\fpcalc", "-json", str(audio)),
                returncode=0,
                stdout='{"duration": 2, "fingerprint": ""}',
                stderr="",
            ),
        )
    )

    def mock_which(executable: str) -> str:
        return f"/tools/{executable}"

    monkeypatch.setattr(
        "preprocessing.media.adapters.versioned_executable._which",
        mock_which,
    )

    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: next(responses)
    )

    with pytest.raises(AudioFingerprintError, match="fingerprint_empty"):
        calculate_audio_fingerprint(
            audio,
            executable="fpcalc",
            expected_version="1.5.1",
            timeout_seconds=20.0,
        )
