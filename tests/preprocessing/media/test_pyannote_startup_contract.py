"""Pyannote startup contract: static Hub pin check + hard builder failures."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import preprocessing.media.adapters.pyannote_adapter as pyannote_mod
from orchestration.runtime_dependency_preflight import (
    _validate_pyannote_runtime,
)
from preprocessing.media.adapters.pyannote_adapter import (
    load_pyannote_backend,
)


def test_validator_rejects_pyannote_3x_with_hub_1x_for_remote_model() -> None:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from pyannote.audio import Pipeline  # noqa: F401
    except Exception:
        pytest.skip("pyannote stack not installed")

    missing: list[str] = []
    settings = SimpleNamespace(
        local_model_path=None,
        model_name="pyannote/speaker-diarization-3.1",
    )

    def _fake_version(name: str) -> str:
        return {
            "pyannote.audio": "3.4.0",
            "huggingface_hub": "1.23.0",
        }[name]

    with patch("importlib.metadata.version", side_effect=_fake_version):
        _validate_pyannote_runtime(settings=settings, missing=missing)

    assert missing
    joined = " ".join(missing)
    assert "use_auth_token" in joined
    assert "token" in joined
    assert "huggingface_hub" in joined


def test_validator_skips_hub_tuple_check_for_local_model() -> None:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from pyannote.audio import Pipeline  # noqa: F401
    except Exception:
        pytest.skip("pyannote stack not installed")

    missing: list[str] = []
    settings = SimpleNamespace(
        local_model_path="/tmp/local-pipeline",
        model_name="unused",
    )

    def _fake_version(name: str) -> str:
        return {
            "pyannote.audio": "3.4.0",
            "huggingface_hub": "1.23.0",
        }[name]

    with patch("importlib.metadata.version", side_effect=_fake_version):
        _validate_pyannote_runtime(settings=settings, missing=missing)

    assert missing == []


def test_builder_preserves_use_auth_token_typeerror_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name="pyannote/speaker-diarization-3.1",
        local_model_path=None,
        local_files_only=False,
        model_artifact_hash=None,
        model_revision=None,
        token_environment_variable="HF_TOKEN",
        backend_version=None,
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class _Pipeline:
        @staticmethod
        def from_pretrained(origin, **kwargs):
            del origin, kwargs
            raise TypeError(
                "hf_hub_download() got an unexpected keyword argument "
                "'use_auth_token'"
            )

    # Ensure signature exposes use_auth_token so builder passes it.
    def _signature(fn):
        del fn
        return inspect.Signature(
            [
                inspect.Parameter(
                    "checkpoint",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "use_auth_token",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        )

    monkeypatch.setattr(
        pyannote_mod,
        "inspect",
        SimpleNamespace(signature=_signature),
    )

    def _import_side_effect(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(device=lambda x: x)
        if name == "pyannote.audio":
            return SimpleNamespace(Pipeline=_Pipeline)
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_side_effect):
        with pytest.raises(RuntimeError) as raised:
            load_pyannote_backend(settings=settings, device="cpu")

    message = str(raised.value)
    assert "use_auth_token" in message
    assert "TypeError" in message
    assert "pyannote/speaker-diarization-3.1" in message


def test_builder_none_pipeline_is_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name="pyannote/speaker-diarization-3.1",
        local_model_path=None,
        local_files_only=False,
        model_artifact_hash=None,
        model_revision=None,
        token_environment_variable="HF_TOKEN",
        backend_version=None,
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class _Pipeline:
        @staticmethod
        def from_pretrained(origin, **kwargs):
            del origin, kwargs
            return None

    def _signature(fn):
        del fn
        return inspect.Signature(
            [
                inspect.Parameter(
                    "checkpoint",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "use_auth_token",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        )

    monkeypatch.setattr(
        pyannote_mod,
        "inspect",
        SimpleNamespace(signature=_signature),
    )

    def _import_side_effect(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(device=lambda x: x)
        if name == "pyannote.audio":
            return SimpleNamespace(Pipeline=_Pipeline)
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_side_effect):
        with pytest.raises(RuntimeError, match="returned no pipeline"):
            load_pyannote_backend(settings=settings, device="cpu")


def test_remote_model_requires_token_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name="pyannote/speaker-diarization-3.1",
        local_model_path=None,
        local_files_only=False,
        model_artifact_hash=None,
        model_revision=None,
        token_environment_variable="HF_TOKEN",
        backend_version=None,
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def _import_side_effect(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(device=lambda x: x)
        if name == "pyannote.audio":
            return SimpleNamespace(
                Pipeline=SimpleNamespace(
                    from_pretrained=staticmethod(lambda *a, **k: object())
                )
            )
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_side_effect):
        with pytest.raises(RuntimeError, match="token environment variable"):
            load_pyannote_backend(settings=settings, device="cpu")


def test_local_model_skips_token_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_file = tmp_path / "pipeline.yaml"
    model_file.write_text("dummy: true\n", encoding="utf-8")
    settings = SimpleNamespace(
        model_name="ignored-remote",
        local_model_path=str(model_file),
        local_files_only=False,
        model_artifact_hash=None,
        model_revision=None,
        token_environment_variable="HF_TOKEN",
        backend_version="test",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    class _PipelineObj:
        def to(self, device):
            del device
            return self

    class _Pipeline:
        @staticmethod
        def from_pretrained(origin, **kwargs):
            del origin, kwargs
            return _PipelineObj()

    def _import_side_effect(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(device=lambda x: x)
        if name == "pyannote.audio":
            return SimpleNamespace(Pipeline=_Pipeline)
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_side_effect):
        backend = load_pyannote_backend(settings=settings, device="cpu")

    assert backend is not None
    assert backend.model_version == "test"
