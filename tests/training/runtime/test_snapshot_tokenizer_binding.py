from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.load import load_settings
from training.runtime.snapshot_tokenizer_binding import bind_snapshot_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _settings(tmp_path: Path) -> object:
    config_root = tmp_path / "config-root"
    shutil.copytree(
        PROJECT_ROOT / "config/files", config_root / "config/files"
    )
    shutil.copytree(
        PROJECT_ROOT / "config/profiles", config_root / "config/profiles"
    )
    return load_settings(
        "dev",
        project_root=tmp_path / "workspace",
        config_root=config_root,
        environment="dev",
    )


def _bind(settings: object, root: Path) -> object:
    return bind_snapshot_tokenizer(
        training=settings.training,
        manifest_filename=settings.datasets.paths.dataset_manifest_filename,
        training_root=root,
    )


def _special_tokens() -> dict[str, int]:
    return {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<mask>": 4,
    }


def _tokenizer_identity(
    *,
    artifact_bytes: bytes = b"tokenizer artifact",
    **overrides: object,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": "artifacts/tokenizer.json",
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_version": "byte_bpe_v2",
        "vocab_size": 1000,
        "special_tokens": _special_tokens(),
    }
    identity.update(overrides)
    return identity


def _write_snapshot(
    tmp_path: Path,
    *,
    tokenizer_identity: dict[str, object] | None = None,
    artifact_bytes: bytes = b"tokenizer artifact",
    manifest: object | None = None,
) -> Path:
    root = tmp_path / "training" / "snapshot-1"
    root.mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "artifacts" / "tokenizer.json").write_bytes(artifact_bytes)
    if manifest is not None:
        (root / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    elif tokenizer_identity is not None:
        (root / "dataset_manifest.json").write_text(
            json.dumps({"tokenizer": tokenizer_identity}),
            encoding="utf-8",
        )
    return root


def test_valid_snapshot_binds_tokenizer_to_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(tmp_path, tokenizer_identity=_tokenizer_identity())

    bound_training = _bind(settings, root)

    assert bound_training.text_tokenizer_path == str(
        (root / "artifacts" / "tokenizer.json").resolve()
    )
    assert (
        bound_training.text_tokenizer_sha256
        == hashlib.sha256(b"tokenizer artifact").hexdigest()
    )
    assert bound_training.text_tokenizer_artifact_version == "byte_bpe_v2"
    assert bound_training.text_tokenizer_vocab_size == 1000
    assert bound_training.text_tokenizer_special_tokens == _special_tokens()


def test_valid_snapshot_does_not_mutate_input_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(tmp_path, tokenizer_identity=_tokenizer_identity())

    _bind(settings, root)

    assert settings.training.text_tokenizer_path == (
        "artifacts/tokenizer/tokenizer.json"
    )
    assert settings.training.text_tokenizer_sha256 is None


def test_missing_manifest_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = tmp_path / "training" / "snapshot-missing-manifest"
    root.mkdir(parents=True)

    with pytest.raises(ValueError, match="unreadable"):
        _bind(settings, root)


def test_missing_tokenizer_section_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(tmp_path, manifest={"dataset": {"name": "x"}})

    with pytest.raises(ValueError, match="no tokenizer identity"):
        _bind(settings, root)


def test_absolute_artifact_path_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    absolute = (tmp_path / "artifacts" / "tokenizer.json").resolve()
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(path=str(absolute)),
    )

    with pytest.raises(ValueError, match="must be relative"):
        _bind(settings, root)


def test_traversal_artifact_path_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(
            path="../outside/tokenizer.json"
        ),
    )

    with pytest.raises(ValueError, match="escapes training root"):
        _bind(settings, root)


def test_missing_artifact_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(path="artifacts/missing.json"),
    )

    with pytest.raises(ValueError, match="artifact is missing"):
        _bind(settings, root)


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(
            sha256="a" * 64,
        ),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _bind(settings, root)


def test_invalid_vocab_size_is_rejected_by_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(vocab_size=10),
    )

    with pytest.raises(ValidationError, match="vocab_size"):
        _bind(settings, root)


def test_missing_special_tokens_is_rejected_by_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(special_tokens={}),
    )

    with pytest.raises(ValidationError, match="missing required tokens"):
        _bind(settings, root)


def test_duplicate_special_token_ids_are_rejected_by_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    tokens = _special_tokens()
    tokens["<mask>"] = 1
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(special_tokens=tokens),
    )

    with pytest.raises(ValidationError, match="ids must be unique"):
        _bind(settings, root)


def test_invalid_sha256_in_manifest_is_rejected_by_settings(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(
            sha256="not-a-sha256-digest",
            artifact_bytes=b"tokenizer artifact",
        ),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _bind(settings, root)


def test_non_integer_vocab_size_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    root = _write_snapshot(
        tmp_path,
        tokenizer_identity=_tokenizer_identity(vocab_size="many"),
    )

    with pytest.raises(ValueError, match="invalid literal"):
        _bind(settings, root)
