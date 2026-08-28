from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def production_whisper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide config-only pins for production tests."""

    monkeypatch.setenv(
        "APP_OVERRIDE__preprocessing__transcription__model_name",
        "/tmp/mmcrawler-test-whisper",
    )
    monkeypatch.setenv(
        "APP_OVERRIDE__preprocessing__transcription__model_revision",
        "test-only-revision",
    )
    monkeypatch.setenv(
        "APP_OVERRIDE__preprocessing__transcription__model_artifact_hash",
        "0" * 64,
    )
    monkeypatch.setenv(
        "APP_OVERRIDE__preprocessing__transcription__backend_version",
        "1.1.1",
    )
