"""Structural invariants for resolved settings (code, not knobs).

Production requirements are declared in prod.toml; these checks make sure
the declared requirements are internally consistent and actually strict.
"""

from __future__ import annotations

from config.errors import ConfigError
from config.settings.root import Settings

TRANSCRIPTION_PIN_FIELDS: tuple[str, ...] = (
    "model_name",
    "model_revision",
    "model_artifact_hash",
    "backend_version",
)


def _check_duplicates(settings: Settings) -> None:
    seen: set[str] = set()
    for task in settings.release.tasks:
        if task.name in seen:
            raise ConfigError(f"duplicate release task {task.name!r}")
        seen.add(task.name)
        metric_names: set[str] = set()
        for metric in task.metrics:
            if metric.name in metric_names:
                raise ConfigError(
                    f"duplicate metric {metric.name!r} on task {task.name!r}"
                )
            metric_names.add(metric.name)


def _check_production(settings: Settings) -> None:
    if not settings.release.tasks:
        raise ConfigError("production requires a non-empty [release].tasks")
    for task in settings.release.tasks:
        if task.min_samples <= 0:
            raise ConfigError(
                f"production task {task.name!r} requires min_samples > 0"
            )
        if not task.metrics:
            raise ConfigError(
                f"production task {task.name!r} requires at least one metric"
            )
    limits = settings.release.limits
    if limits.max_batch_latency_ms <= 0 or limits.max_peak_memory_mb <= 0:
        raise ConfigError("production requires positive runtime limits")
    transcription = settings.preprocessing.transcription
    if transcription.enabled:
        missing = [
            field
            for field in TRANSCRIPTION_PIN_FIELDS
            if getattr(transcription, field) in (None, "")
        ]
        if missing:
            raise ConfigError(
                "production Whisper transcription requires deployment "
                f"pins, missing: {', '.join(missing)}"
            )


def validate_settings(settings: Settings) -> None:
    """Raise ConfigError on any violated structural invariant."""

    _check_duplicates(settings)
    if settings.profile == "prod":
        _check_production(settings)
