from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from config.preprocessing.media_settings import TranscriptionSettings
from preprocessing.media.adapters.whisper_model_loader import (
    WhisperModelLoader,
)


def _settings(
    model_directory: Path, artifact_hash: str
) -> TranscriptionSettings:
    return TranscriptionSettings(
        enabled=True,
        backend="whisper",
        model_name=str(model_directory),
        model_revision="immutable-revision",
        model_artifact_hash=artifact_hash,
        backend_version="1.1.1",
        local_files_only=True,
        production_mode=True,
    )


def test_production_whisper_loads_only_the_hashed_local_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"approved-model")
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    observed: dict[str, object] = {}

    def fake_model(model_name: str, **kwargs: object) -> object:
        observed.update({"model_name": model_name, **kwargs})
        return object()

    with (
        patch(
            "preprocessing.media.adapters.whisper_model_loader.installed_backend_version",
            return_value="1.1.1",
        ),
        patch(
            "preprocessing.media.adapters.whisper_model_loader.import_module",
            return_value=SimpleNamespace(WhisperModel=fake_model),
        ),
    ):
        loader = WhisperModelLoader(
            settings=_settings(tmp_path, expected_hash)
        )
        loader.get_model()

    assert observed["model_name"] == str(tmp_path)
    assert observed["local_files_only"] is True


def test_production_whisper_rejects_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin").write_bytes(b"tampered-model")
    with (
        patch(
            "preprocessing.media.adapters.whisper_model_loader.installed_backend_version",
            return_value="1.1.1",
        ),
        patch(
            "preprocessing.media.adapters.whisper_model_loader.import_module",
            return_value=SimpleNamespace(
                WhisperModel=lambda *a, **k: object()
            ),
        ),
        pytest.raises(RuntimeError, match="artifact hash mismatch"),
    ):
        loader = WhisperModelLoader(settings=_settings(tmp_path, "0" * 64))
        loader.get_model()
