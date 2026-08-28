"""Canonical profile loader.

A single profile TOML is read from the selected read-only configuration root,
source-registry selectors are expanded, runtime overrides are applied, paths
are resolved once, and one immutable ``Settings`` tree is returned.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.environment.runtime_environment import all_values
from config.environment.source_selection import (
    profile_for_environment,
    require_environment,
)
from config.errors import ConfigError
from config.overrides import (
    apply_overrides,
    overrides_from_cli,
    overrides_from_env,
)
from config.paths import resolve_dir, resolve_paths
from config.profiles import Profile, normalize_profile
from config.settings.meta import ConfigMeta
from config.settings.paths import PathSettings
from config.settings.root import Settings
from config.source_catalog.registry_expansion import apply_source_registry
from config.validate import validate_settings
from config.validation.cross_section.coordinator import (
    validate_structural_settings,
)

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent.parent


def _resolve_config_root(config_root: str | Path | None) -> Path:
    root = (
        DEFAULT_CONFIG_ROOT
        if config_root is None
        else Path(config_root).expanduser()
    )
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not (root / "config" / "files").is_dir():
        raise ConfigError(
            f"configuration root must contain config/files: {root}"
        )
    return root


def _profile_config_root(config_root: Path, profile: Profile) -> Path:
    """Select an optional custom profile or the packaged canonical profile."""

    custom_profile = config_root / "config" / "profiles" / f"{profile}.toml"
    if custom_profile.is_file():
        return config_root
    return DEFAULT_CONFIG_ROOT


def _read_profile_toml(
    profile: Profile, *, config_root: Path
) -> dict[str, Any]:
    toml_path = config_root / "config" / "profiles" / f"{profile}.toml"
    if not toml_path.is_file():
        raise ConfigError(f"missing profile file {toml_path}")
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    declared = raw.pop("profile", None)
    if declared != profile:
        raise ConfigError(
            f"profile file {toml_path.name} declares {declared!r}, "
            f"expected {profile!r}"
        )
    return raw


def _runtime_environment(profile: Profile, environment: str | None) -> str:
    if environment is None:
        return profile
    try:
        normalized = require_environment(environment)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    required_profile = profile_for_environment(normalized)
    if profile != required_profile:
        raise ConfigError(
            f"{normalized} requires the {required_profile} configuration "
            "profile"
        )
    return normalized


def _expand_sources(
    raw: dict[str, Any],
    *,
    config_root: Path,
    runtime_environment: str,
) -> None:
    sources = raw.get("sources")
    if not isinstance(sources, dict):
        raise ConfigError("profile must declare a [sources] section")

    selectors = sources.pop("selectors", None)
    if selectors is None:
        raise ConfigError(
            "[sources].selectors must be declared by the profile"
        )

    registry_name = str(
        sources.get("registry", "source_registry.json")
    ).strip()
    if not registry_name or Path(registry_name).name != registry_name:
        raise ConfigError("[sources].registry must be a filename")

    sources["registry_sources"] = selectors
    registry_environment = (
        "dev" if runtime_environment == "test" else runtime_environment
    )
    try:
        apply_source_registry(
            raw,
            project_root=config_root,
            environment=registry_environment,
            registry_path=Path("config/files/sources") / registry_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    # Loaded runtime profiles are always required to resolve concrete seeds.
    sources["require_seed_urls"] = True


def _resolve_profile_paths(
    raw: dict[str, Any],
    *,
    profile: Profile,
    env: Mapping[str, str],
    project_root: str | Path | None,
) -> None:
    raw_paths = raw.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ConfigError("[paths] must be an object")
    path_settings = PathSettings.model_validate(raw_paths)
    resolved = resolve_paths(
        profile,
        path_settings,
        env=env,
        project_root=project_root,
    )
    raw["paths"] = {
        "root": resolved.root,
        "data": resolved.data,
        "cache": resolved.cache,
        "output": resolved.output,
    }

    preprocessing = raw.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return
    transcription = preprocessing.get("transcription")
    if not isinstance(transcription, dict):
        return
    cache_directory = transcription.get("cache_directory")
    if cache_directory in (None, ""):
        return
    transcription["cache_directory"] = str(
        resolve_dir(
            resolved.root,
            str(cache_directory),
            kind="preprocessing.transcription.cache_directory",
        )
    )


def _precheck_production_transcription(
    raw: dict[str, Any], *, profile: Profile
) -> None:
    if profile != "prod":
        return
    preprocessing = raw.get("preprocessing", {})
    transcription = (
        preprocessing.get("transcription", {})
        if isinstance(preprocessing, dict)
        else {}
    )
    if not isinstance(transcription, dict) or not transcription.get("enabled"):
        return
    missing = [
        name
        for name in (
            "model_name",
            "model_revision",
            "model_artifact_hash",
            "backend_version",
        )
        if transcription.get(name) in (None, "")
    ]
    if missing:
        raise ConfigError(
            "production Whisper transcription requires deployment pins, "
            f"missing: {', '.join(missing)}",
            setting="preprocessing.transcription",
            issue="required_deployment_pins_missing",
        )


def _fingerprint(settings: Settings) -> str:
    payload = json.dumps(
        settings.model_dump(mode="json", exclude={"meta"}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_loaded_settings(settings: Settings) -> None:
    """Apply structural and config-owned cross-section validation once."""

    validate_settings(settings)
    validate_structural_settings(settings)


def load_settings(
    profile: str | None = None,
    *,
    project_root: str | Path | None = None,
    config_root: str | Path | None = None,
    environment: str | None = None,
    overrides: Sequence[str] | None = None,
    fingerprint: bool = True,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate exactly one canonical runtime settings tree."""

    env = all_values() if env is None else dict(env)
    resolved_profile = normalize_profile(
        profile or env.get("APP_PROFILE") or environment or "dev"
    )
    selected_config_root = _resolve_config_root(config_root)
    profile_config_root = _profile_config_root(
        selected_config_root,
        resolved_profile,
    )
    runtime_environment = _runtime_environment(resolved_profile, environment)

    raw = _read_profile_toml(resolved_profile, config_root=profile_config_root)
    application = raw.setdefault("application", {})
    if not isinstance(application, dict):
        raise ConfigError("[application] must be an object")
    application["environment"] = runtime_environment
    _expand_sources(
        raw,
        config_root=selected_config_root,
        runtime_environment=runtime_environment,
    )

    overrides_env = overrides_from_env(env)
    overrides_env.update(overrides_from_cli(overrides or ()))
    apply_overrides(raw, overrides_env)

    # Resolve filesystem identity before constructing subsystems that may have
    # stricter production validators.  This preserves the fail-closed rule that
    # production must name its writable workspace explicitly.
    _resolve_profile_paths(
        raw,
        profile=resolved_profile,
        env=env,
        project_root=project_root,
    )
    _precheck_production_transcription(raw, profile=resolved_profile)

    settings = Settings(profile=resolved_profile, **raw)
    _validate_loaded_settings(settings)

    if not fingerprint:
        return settings
    meta = ConfigMeta(profile=resolved_profile, sha256=_fingerprint(settings))
    return settings.model_copy(update={"meta": meta})
