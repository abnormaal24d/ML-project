from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest

from config.load import load_settings as load_config
from config.releases.release_requirements import ReleaseRequirements
from orchestration import settings_loader as settings_loader_module
from orchestration.settings_loader import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_orchestration_loader_composes_multimodal_domain_validation(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config-root"
    shutil.copytree(
        PROJECT_ROOT / "config/files", config_root / "config/files"
    )
    shutil.copytree(
        PROJECT_ROOT / "config/profiles", config_root / "config/profiles"
    )
    profile_path = config_root / "config/profiles/dev.toml"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "text_pretrain",
            "unknown_task",
        ),
        encoding="utf-8",
    )

    settings = load_config(
        "dev",
        project_root=tmp_path / "structural-workspace",
        config_root=config_root,
        environment="dev",
    )
    assert "unknown_task" in settings.training.tasks

    with pytest.raises(
        ValueError, match="configuration references unknown tasks"
    ):
        load(
            project_root=tmp_path / "orchestration-workspace",
            config_root=config_root,
            environment="dev",
            profile="dev",
        )


def test_orchestration_loader_applies_cuda_override(tmp_path: Path) -> None:
    settings = load(
        project_root=tmp_path / "workspace",
        config_root=PROJECT_ROOT,
        environment="dev",
        profile="dev",
        use_cuda=True,
    )

    assert settings.training.device == "cuda"


def test_orchestration_validation_layers_run_once_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = load_config(
        "dev",
        project_root=tmp_path / "workspace",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    requirements = cast(ReleaseRequirements, object())
    calls: list[str] = []

    monkeypatch.setattr(
        settings_loader_module,
        "load_settings",
        lambda **_kwargs: settings,
    )
    monkeypatch.setattr(
        settings_loader_module,
        "validate_multimodal_settings",
        lambda _settings: calls.append("multimodal_base"),
    )
    monkeypatch.setattr(
        settings_loader_module,
        "validate_multimodal_cross_section_settings",
        lambda _settings: calls.append("domain"),
    )
    monkeypatch.setattr(
        settings_loader_module,
        "release_requirements_from_settings",
        lambda _settings: requirements,
    )
    monkeypatch.setattr(
        settings_loader_module,
        "validate_release_requirements",
        lambda **_kwargs: calls.append("release_contract"),
    )
    monkeypatch.setattr(
        settings_loader_module,
        "validate_multimodal_release_settings",
        lambda _settings, _requirements: calls.append("multimodal_release"),
    )

    loaded = settings_loader_module.load(
        project_root=tmp_path / "workspace",
        config_root=PROJECT_ROOT,
        environment="dev",
        profile="dev",
    )

    assert loaded is settings
    assert calls == [
        "multimodal_base",
        "domain",
        "release_contract",
        "multimodal_release",
    ]
