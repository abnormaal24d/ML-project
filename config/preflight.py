"""Advisory readiness report for the canonical configuration surface.

A structured readiness report for one profile: configuration must load
fail-closed (pins, invariants, path guards) and every issue is classified
as a hard error or a warning. Warnings never block; errors represent an
unusable configuration.

This report is a diagnostic API, not the runtime startup boundary. The CLI
loads through ``orchestration.settings_loader`` and uses its canonical runtime
readiness validation for both ``run`` and ``control validate-config``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from config.errors import ConfigError
from config.load import load_settings
from config.paths import ENV_PROJECT_ROOT, resolve_paths
from config.settings.root import Settings

BASELINE_LATENCY_MS: float = 250.0
BASELINE_MEMORY_MB: int = 16384


class IssueKind(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class PreflightIssue(NamedTuple):
    kind: IssueKind
    message: str


class PreflightResult(NamedTuple):
    profile: str
    issues: tuple[PreflightIssue, ...]
    settings: Settings | None

    @property
    def ok(self) -> bool:
        return not any(issue.kind == IssueKind.ERROR for issue in self.issues)


def preflight(
    profile: str | None = None,
    *,
    project_root: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    check_artifact_paths: bool = False,
) -> PreflightResult:
    """Assess one profile and return a structured report.

    Artifact path validation (``check_artifact_paths``) is advisory:
    pinned local artifacts may legitimately be absent on machines that do
    not run the corresponding backend yet.
    """

    issues: list[PreflightIssue] = []

    try:
        settings = load_settings(
            profile,
            project_root=project_root,
            overrides=overrides,
            env=env,
        )
    except ConfigError as error:
        return PreflightResult(
            profile=str(profile) if profile is not None else "dev",
            issues=(PreflightIssue(IssueKind.ERROR, str(error)),),
            settings=None,
        )

    resolved = resolve_paths(
        settings.profile,
        settings.paths,
        env=env,
        project_root=project_root,
    )
    for kind, _name, path in (
        ("data", "data", resolved.data),
        ("cache", "cache", resolved.cache),
        ("output", "output", resolved.output),
    ):
        if not path.exists():
            issues.append(
                PreflightIssue(
                    IssueKind.WARNING,
                    f"[paths].{kind} does not exist yet: {path}",
                )
            )

    effective_env = {} if env is None else env
    if (
        settings.profile != "prod"
        and project_root is None
        and not effective_env.get(ENV_PROJECT_ROOT)
    ):
        issues.append(
            PreflightIssue(
                IssueKind.WARNING,
                f"{settings.profile} falls back to the current working "
                "directory as project root; production never does",
            )
        )

    limits = settings.release.limits
    if (
        limits.max_batch_latency_ms == BASELINE_LATENCY_MS
        or limits.max_peak_memory_mb == BASELINE_MEMORY_MB
    ):
        issues.append(
            PreflightIssue(
                IssueKind.WARNING,
                "release runtime limits are still the initial baseline "
                "(250 ms, 16384 MB); calibrate from measured runs before "
                "the release gate vouches for them",
            )
        )

    if check_artifact_paths:
        _check_artifact_paths(settings, issues)

    return PreflightResult(settings.profile, tuple(issues), settings)


def _check_artifact_paths(
    settings: Settings, issues: list[PreflightIssue]
) -> None:
    transcription = settings.preprocessing.transcription
    if (
        transcription.enabled
        and transcription.local_files_only
        and transcription.model_name
        and not Path(transcription.model_name).exists()
    ):
        issues.append(
            PreflightIssue(
                IssueKind.WARNING,
                f"pinned Whisper model directory not found: "
                f"{transcription.model_name}",
            )
        )

    ocr = settings.preprocessing.ocr
    if ocr.model_artifact_path and not Path(ocr.model_artifact_path).exists():
        issues.append(
            PreflightIssue(
                IssueKind.WARNING,
                f"pinned OCR model artifact not found: "
                f"{ocr.model_artifact_path}",
            )
        )
