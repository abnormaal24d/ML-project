"""Runtime overrides: a small allowlist applied on top of the profile file.

Only explicitly listed settings are runtime-writable. Everything else is
frozen in the profile TOML. Unknown or malformed overrides fail closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from config.errors import ConfigError

ENV_PREFIX: str = "APP_OVERRIDE__"

ALLOWED_OVERRIDES: dict[str, Callable[[str], object]] = {
    "logging.level": str,
    "paths.data": str,
    "paths.cache": str,
    "paths.output": str,
    "training.device": str,
    "training.num_workers": int,
    "training.batch_size": int,
    "preprocessing.transcription.model_name": str,
    "preprocessing.transcription.model_revision": str,
    "preprocessing.transcription.model_artifact_hash": str,
    "preprocessing.transcription.backend_version": str,
    "preprocessing.ocr.model_artifact_path": str,
}


def _convert(path: str, value: str) -> object:
    converter = ALLOWED_OVERRIDES.get(path)
    if converter is None:
        allowed = ", ".join(sorted(ALLOWED_OVERRIDES))
        raise ConfigError(
            f"unknown runtime override {path!r}; allowed: {allowed}"
        )
    try:
        return converter(value)
    except ValueError:
        raise ConfigError(
            f"invalid value {value!r} for runtime override {path!r}"
        ) from None


def overrides_from_env(env: Mapping[str, str]) -> dict[str, object]:
    """Collect APP_OVERRIDE__* variables (parts joined by double underscore)."""

    out: dict[str, object] = {}
    for key, value in env.items():
        if not key.upper().startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower()
        if "__" in path:
            path = path.replace("__", ".")
        out[path] = _convert(path, value)
    return out


def overrides_from_cli(items: Sequence[str]) -> dict[str, object]:
    """Collect CLI overrides given as ``path=value`` entries."""

    out: dict[str, object] = {}
    for item in items:
        path, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(
                f"invalid override {item!r}: expected path=value"
            )
        if path in out:
            raise ConfigError(f"duplicate runtime override {path!r}")
        out[path] = _convert(path, value.strip())
    return out


def apply_overrides(
    raw: dict[str, object], values: Mapping[str, object]
) -> None:
    """Apply typed values into a decoded profile dict (nested keys)."""

    for path, value in values.items():
        parts = path.split(".")
        node: dict[str, object] = raw
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
