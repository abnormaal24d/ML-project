"""Canonical configuration tests: profiles, overrides, paths, and release contract."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.environment.source_selection import ConfigSourceResolver
from config.errors import ConfigError
from config.load import load_settings
from config.overrides import ALLOWED_OVERRIDES
from config.path_resolution.project_paths import ProjectPaths
from config.paths import resolve_paths
from config.profiles import normalize_profile
from config.settings.paths import PathSettings
from config.settings.release import MetricReq, ReleaseSettings, TaskReq
from config.settings.root import Settings
from config.validate import validate_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_WHISPER_PINS = {
    "preprocessing.transcription.model_name": "/tmp/mmcrawler-test-whisper",
    "preprocessing.transcription.model_revision": "test-only-revision",
    "preprocessing.transcription.model_artifact_hash": "0" * 64,
    "preprocessing.transcription.backend_version": "1.1.1",
}

PRODUCTION_V1_REQUIRED_TASKS = (
    "text_pretrain",
    "instruction_following",
    "document_text_pair",
    "pdf_text_pair",
    "doc_qa",
    "image_text_pair",
    "ocr_parse",
    "vqa",
    "audio_text_pair",
    "audio_qa",
    "video_text_pair",
    "video_qa",
    "multimodal_retrieval",
    "cross_modal_consistency",
)


def _prod_env(pins: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    env["DATA_ENGINE_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if pins:
        for path, value in _WHISPER_PINS.items():
            env[f"APP_OVERRIDE__{path.replace('.', '__')}"] = value
    return env


# ---------------------------------------------------------------------------
# A. Profile loading and identity
# ---------------------------------------------------------------------------


def test_default_profile_is_dev() -> None:
    assert load_settings().profile == "dev"


def test_app_profile_env_selects_profile() -> None:
    env = dict(os.environ, APP_PROFILE="test")
    settings = load_settings(env=env)
    assert settings.profile == "test"
    assert settings.training.batch_size == 4


def test_explicit_argument_wins_over_env() -> None:
    env = dict(os.environ, APP_PROFILE="test")
    settings = load_settings("dev", env=env)
    assert settings.profile == "dev"


def test_profiles_load_without_pins_or_root() -> None:
    for profile in ("test", "dev"):
        settings = load_settings(profile)
        assert settings.profile == profile
    with pytest.raises(ConfigError, match="requires an explicit project root"):
        load_settings("prod")


def test_artifact_only_config_root_uses_packaged_profile(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config-root"
    shutil.copytree(
        PROJECT_ROOT / "config/files",
        config_root / "config/files",
    )

    settings = load_settings(
        "dev",
        config_root=config_root,
        project_root=tmp_path / "workspace",
        environment="dev",
        env={},
    )

    assert settings.profile == "dev"
    assert settings.training.batch_size == 4


def test_custom_config_root_profile_is_honored(tmp_path: Path) -> None:
    config_root = tmp_path / "config-root"
    shutil.copytree(
        PROJECT_ROOT / "config/files",
        config_root / "config/files",
    )
    shutil.copytree(
        PROJECT_ROOT / "config/profiles",
        config_root / "config/profiles",
    )
    profile_path = config_root / "config/profiles/dev.toml"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace(
            "batch_size = 4",
            "batch_size = 7",
            1,
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        "dev",
        config_root=config_root,
        project_root=tmp_path / "workspace",
        environment="dev",
        env={},
    )

    assert settings.training.batch_size == 7


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        normalize_profile("staging")


def test_unknown_profile_env_rejected() -> None:
    env = dict(os.environ, APP_PROFILE="staging")
    with pytest.raises(ValueError, match="unknown profile"):
        load_settings(env=env)


# ---------------------------------------------------------------------------
# B. Invalid configuration fails closed
# ---------------------------------------------------------------------------


def test_unknown_override_env_rejected() -> None:
    env = dict(os.environ, APP_OVERRIDE__training__epochs="10")
    with pytest.raises(ConfigError, match="unknown runtime override"):
        load_settings("dev", env=env)


def test_unknown_override_cli_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown runtime override"):
        load_settings("dev", overrides=["training.epochs=10"])


def test_duplicate_cli_override_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate runtime override"):
        load_settings(
            "dev",
            overrides=["training.batch_size=8", "training.batch_size=16"],
        )


def test_malformed_cli_override_rejected() -> None:
    with pytest.raises(ConfigError, match="expected path=value"):
        load_settings("dev", overrides=["training.batch_size"])


def test_invalid_override_value_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid value"):
        load_settings("dev", overrides=["training.num_workers=many"])


def test_unknown_setting_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(profile="test", bogus_key=1)


def test_metric_requires_exactly_one_bound() -> None:
    with pytest.raises(ValidationError):
        MetricReq(name="accuracy", min=0.1, max=0.9)
    with pytest.raises(ValidationError):
        MetricReq(name="accuracy")


def test_duplicate_release_tasks_rejected() -> None:
    task = TaskReq(
        name="text_pretrain",
        min_samples=1,
        metrics=(MetricReq(name="m", min=0.0),),
    )
    settings = Settings(
        profile="test",
        release=ReleaseSettings(tasks=(task, task)),
    )
    with pytest.raises(ConfigError, match="duplicate release task"):
        validate_settings(settings)


def test_backend_enabled_mismatch_rejected() -> None:
    from config.preprocessing.media_settings import TranscriptionSettings

    with pytest.raises(ValidationError):
        TranscriptionSettings(enabled=True, backend="disabled")


# ---------------------------------------------------------------------------
# C. Runtime overrides
# ---------------------------------------------------------------------------


def test_allowlist_is_small_and_explicit() -> None:
    assert set(ALLOWED_OVERRIDES) == {
        "logging.level",
        "paths.data",
        "paths.cache",
        "paths.output",
        "training.device",
        "training.num_workers",
        "training.batch_size",
        "preprocessing.transcription.model_name",
        "preprocessing.transcription.model_revision",
        "preprocessing.transcription.model_artifact_hash",
        "preprocessing.transcription.backend_version",
        "preprocessing.ocr.model_artifact_path",
    }


def test_cli_and_env_overrides_are_equivalent() -> None:
    env = dict(os.environ, APP_OVERRIDE__training__batch_size="128")
    from_cli = load_settings("dev", overrides=["training.batch_size=128"])
    from_env = load_settings("dev", env=env)
    assert from_cli.model_dump() == from_env.model_dump()


def test_cli_wins_over_env() -> None:
    env = dict(
        os.environ,
        APP_OVERRIDE__training__batch_size="128",
        APP_OVERRIDE__training__device="cuda",
    )
    settings = load_settings(
        "dev", env=env, overrides=["training.batch_size=256"]
    )
    assert settings.training.batch_size == 256
    assert settings.training.device == "cuda"


def test_override_names_cannot_be_smuggled_via_nested_path() -> None:
    env = dict(os.environ, APP_OVERRIDE__training__batch_size__extra="1")
    with pytest.raises(ConfigError, match="unknown runtime override"):
        load_settings("dev", env=env)


# ---------------------------------------------------------------------------
# D. Paths
# ---------------------------------------------------------------------------


def test_relative_dirs_resolve_under_explicit_root() -> None:
    settings = load_settings("test", project_root=str(PROJECT_ROOT))
    paths = resolve_paths(
        settings.profile,
        settings.paths,
        env={},
        project_root=str(PROJECT_ROOT),
    )
    assert paths.root.samefile(PROJECT_ROOT)
    assert paths.data.relative_to(paths.root) == Path("data")
    assert paths.cache.relative_to(paths.root) == Path("runtime/cache")
    assert paths.output.relative_to(paths.root) == Path("runtime/output")


def test_transcription_cache_resolves_under_explicit_root(
    tmp_path: Path,
) -> None:
    settings = load_settings("dev", project_root=tmp_path)

    cache_directory = Path(
        settings.preprocessing.transcription.cache_directory
    )
    assert cache_directory.relative_to(settings.paths.root) == Path(
        "runtime/cache/transcription"
    )


def test_dev_falls_back_to_cwd_without_explicit_root() -> None:
    settings = load_settings("dev")
    paths = resolve_paths(settings.profile, settings.paths, env={})
    assert paths.root.samefile(Path.cwd())


def test_prod_requires_explicit_root() -> None:
    env = dict(os.environ)
    for key in ("DATA_ENGINE_PROJECT_ROOT", "APP_OVERRIDE__paths__root"):
        env.pop(key, None)
    with pytest.raises(ConfigError, match="requires an explicit project root"):
        resolve_paths("prod", PathSettings(), env=env)


def test_production_source_resolution_rejects_default_project_root(
    tmp_path: Path,
) -> None:
    resolver = ConfigSourceResolver(
        base_project_root=tmp_path,
        base_config_root=PROJECT_ROOT,
    )

    with pytest.raises(ValueError, match="explicit project root"):
        resolver.resolve(environment="prod")


def test_dev_source_resolution_keeps_default_project_root(
    tmp_path: Path,
) -> None:
    resolver = ConfigSourceResolver(
        base_project_root=tmp_path,
        base_config_root=PROJECT_ROOT,
    )

    sources = resolver.resolve(environment="dev")

    assert sources.project_root == tmp_path.resolve()


@pytest.mark.parametrize(
    ("profile", "environment"),
    (
        ("test", "dev"),
        ("dev", "prod"),
        ("prod", "test"),
    ),
)
def test_runtime_environment_rejects_incompatible_profile(
    profile: str,
    environment: str,
) -> None:
    with pytest.raises(ConfigError, match="requires the"):
        load_settings(profile, environment=environment)


def test_root_from_environment_accepted() -> None:
    env = {"DATA_ENGINE_PROJECT_ROOT": str(PROJECT_ROOT)}
    paths = resolve_paths("dev", PathSettings(), env=env)
    assert paths.root.samefile(PROJECT_ROOT)


def test_absolute_path_outside_root_rejected() -> None:
    if sys.platform != "win32":
        # On non-Windows, Windows absolute paths (C:\...) are rejected early
        with pytest.raises(ConfigError, match="Windows absolute path"):
            resolve_paths(
                "dev", PathSettings(data=r"C:\Windows\System32"), env={}
            )
    else:
        # On Windows, C:\Windows\System32 is a valid absolute path that escapes the project root
        with pytest.raises(ConfigError, match="escapes the project root"):
            resolve_paths(
                "dev", PathSettings(data=r"C:\Windows\System32"), env={}
            )


def test_absolute_path_inside_root_allowed(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    inside = root / "data"
    inside.mkdir(parents=True)
    paths = resolve_paths(
        "dev",
        PathSettings(data=str(inside)),
        project_root=root,
        env={},
    )
    assert paths.data.samefile(inside)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path form")
def test_windows_resolves_root_and_absolute_dirs_with_extended_paths(
    tmp_path: Path,
) -> None:
    paths = resolve_paths(
        "dev",
        PathSettings(
            data=str(tmp_path / "data"),
            cache=str(tmp_path / "cache"),
            output=str(tmp_path / "output"),
        ),
        project_root=tmp_path,
    )

    for path in (paths.root, paths.data, paths.cache, paths.output):
        assert str(path).startswith("\\\\?\\")
    assert paths.data.relative_to(paths.root) == Path("data")
    assert paths.cache.relative_to(paths.root) == Path("cache")
    assert paths.output.relative_to(paths.root) == Path("output")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path form")
def test_windows_project_paths_canonicalize_long_absolute_input(
    tmp_path: Path,
) -> None:
    project_paths = ProjectPaths(project_root=tmp_path)
    long_path = (
        tmp_path
        / ("p" * 160)
        / "data"
        / "raw"
        / "runs"
        / "multimodal"
        / ("f" * 64 + ".html")
    )

    resolved = project_paths.resolve(long_path, allow_absolute=True)

    assert len(str(long_path)) >= 260
    assert str(resolved).startswith("\\\\?\\")
    assert resolved.relative_to(project_paths.project_root).name == (
        "f" * 64 + ".html"
    )


def test_relative_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(ConfigError, match="escapes the project root"):
        resolve_paths(
            "dev",
            PathSettings(data="../outside"),
            project_root=root,
        )


def test_symlinked_relative_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    escaped = root / "escaped"
    try:
        escaped.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(
            f"symlinks are unavailable in this test environment: {exc}"
        )

    with pytest.raises(ConfigError, match="escapes the project root"):
        resolve_paths(
            "dev",
            PathSettings(data="escaped"),
            project_root=root,
        )


# ---------------------------------------------------------------------------
# E. Production contract
# ---------------------------------------------------------------------------


def test_prod_starts_as_a_candidate_with_all_required_tasks() -> None:
    settings = load_settings("prod", env=_prod_env())
    assert settings.application.environment == "prod"
    assert settings.training.release_stage == "candidate"
    assert {task.name for task in settings.release.tasks} == set(
        PRODUCTION_V1_REQUIRED_TASKS
    )


def test_prod_tasks_have_positive_minima_with_metrics() -> None:
    settings = load_settings("prod", env=_prod_env())
    for task in settings.release.tasks:
        assert task.min_samples > 0
        assert task.metrics


def test_prod_metrics_are_the_declared_floors_only() -> None:
    settings = load_settings("prod", env=_prod_env())
    # Expected metrics per task based on evaluator/task_metrics.py output
    expected_metrics = {
        "text_pretrain": {"masked_token_accuracy"},
        "instruction_following": {"token_f1"},
        "document_text_pair": {"recall_at_1", "embedding_similarity_mean"},
        "pdf_text_pair": {"recall_at_1", "embedding_similarity_mean"},
        "doc_qa": {"doc_qa_f1"},
        "image_text_pair": {"recall_at_1", "embedding_similarity_mean"},
        "ocr_parse": {"character_error_rate", "word_error_rate"},
        "vqa": {"vqa_accuracy"},
        "audio_text_pair": {"recall_at_1", "embedding_similarity_mean"},
        "audio_qa": {"token_f1"},
        "video_text_pair": {"recall_at_1", "embedding_similarity_mean"},
        "video_qa": {"token_f1"},
        "multimodal_retrieval": {"recall_at_1", "embedding_similarity_mean"},
        "cross_modal_consistency": {
            "recall_at_1",
            "embedding_similarity_mean",
        },
    }
    for task in settings.release.tasks:
        assert {m.name for m in task.metrics} == expected_metrics[task.name], (
            f"Task {task.name} has unexpected metrics"
        )
        for metric in task.metrics:
            if metric.name in ("character_error_rate", "word_error_rate"):
                # OCR metrics use max, not min
                assert metric.max is not None
                assert metric.min is None
            else:
                # Accuracy/F1/recall metrics use min
                assert metric.min is not None
                assert metric.max is None


def test_prod_limits_are_positive() -> None:
    settings = load_settings("prod", env=_prod_env())
    assert settings.release.limits.max_batch_latency_ms > 0
    assert settings.release.limits.max_peak_memory_mb > 0


def test_prod_fails_closed_without_whisper_pins() -> None:
    with pytest.raises(ConfigError, match="production Whisper transcription"):
        load_settings("prod", env=_prod_env(pins=False))


def test_prod_fails_closed_on_partial_pin_set() -> None:
    env = _prod_env(pins=False)
    env["APP_OVERRIDE__preprocessing__transcription__model_name"] = (
        "/tmp/mmcrawler-test-whisper"
    )
    with pytest.raises(ConfigError, match="missing: model_revision"):
        load_settings("prod", env=env)


def test_prod_accepts_pins_from_cli() -> None:
    settings = load_settings(
        "prod",
        project_root=str(PROJECT_ROOT),
        overrides=[f"{path}={value}" for path, value in _WHISPER_PINS.items()],
    )
    assert (
        settings.preprocessing.transcription.model_name
        == "/tmp/mmcrawler-test-whisper"
    )


def test_fingerprint_is_deterministic_and_override_sensitive() -> None:
    first = load_settings("dev")
    second = load_settings("dev")
    assert first.meta is not None
    assert first.meta.sha256 == second.meta.sha256

    changed = load_settings("dev", overrides=["training.batch_size=256"])
    assert changed.meta.sha256 != first.meta.sha256


def test_fingerprint_computed_without_meta() -> None:
    settings = load_settings("dev")
    assert "." not in settings.meta.sha256
    assert settings.meta.profile == "dev"
    fresh = Settings(**settings.model_dump(exclude={"meta"}))
    assert fresh.meta is None


def test_fingerprint_disabled_on_request() -> None:
    settings = load_settings("dev", fingerprint=False)
    assert settings.meta is None


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown environment"):
        load_settings(
            "prod",
            project_root=PROJECT_ROOT,
            config_root=PROJECT_ROOT,
            environment="invalid-environment",
            env=_prod_env(),
        )


def test_prod_environment_preserves_candidate_release_stage() -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
        env=_prod_env(),
    )

    assert settings.application.environment == "prod"
    assert settings.training.release_stage == "candidate"
