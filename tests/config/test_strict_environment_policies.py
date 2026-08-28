from __future__ import annotations

from pathlib import Path

from config.load import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dev_preserves_explicit_recomputation() -> None:
    settings = load_settings(
        "dev",
        project_root=PROJECT_ROOT,
        environment="dev",
    )
    assert settings.datasets.schemas.raw_schema_version == "3.0"
    assert settings.datasets.schemas.curated_schema_version == "3.0"
    assert settings.datasets.schemas.training_schema_version == "3.0"


def test_production_profiles_are_fail_closed(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        environment="prod",
    )
    assert settings.training.materialized_tensors_enabled is True
    assert settings.training.materialized_tensor_validate_shapes is True
