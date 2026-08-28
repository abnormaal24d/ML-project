"""Project root resolution and escape-guarded path resolution.

Production never falls back to the current working directory: it requires
an explicit root (``[paths].root``, ``DATA_ENGINE_PROJECT_ROOT``, or the
``project_root`` loader argument). Resolved data/cache/output directories,
whether configured relatively or absolutely, must stay inside the project
root.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple

from config.errors import ConfigError
from config.path_resolution.project_paths import normalize_project_path
from config.profiles import Profile
from config.settings.paths import PathSettings

ENV_PROJECT_ROOT: str = "DATA_ENGINE_PROJECT_ROOT"

_IS_WINDOWS = sys.platform == "win32"


class ResolvedPaths(NamedTuple):
    root: Path
    data: Path
    cache: Path
    output: Path


def _inside(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def _is_absolute_path(value: str | Path) -> bool:
    """Return True if ``value`` is an absolute path on any platform."""
    text = str(value)
    return (
        Path(text).is_absolute()
        or PurePosixPath(text.replace("\\", "/")).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or bool(PureWindowsPath(text).drive)
    )


def _is_windows_absolute(path: str | Path) -> bool:
    """Return True if ``path`` is a Windows absolute path (e.g. ``C:\\...``)."""
    try:
        return PureWindowsPath(path).is_absolute()
    except Exception:
        return False


def resolve_root(
    profile: Profile,
    paths: PathSettings,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the explicit project root for a profile."""

    env = env if env is not None else {}
    candidate: str | Path | None = project_root
    if candidate is None:
        candidate = env.get(ENV_PROJECT_ROOT)
    if candidate is None and "root" in paths.model_fields_set:
        candidate = paths.root
    if candidate is None:
        if profile == "prod":
            raise ConfigError(
                "production requires an explicit project root "
                f"(set {ENV_PROJECT_ROOT} or [paths].root)"
            )
        candidate = Path.cwd()
    p = Path(candidate).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return normalize_project_path(p)


def resolve_dir(root: Path, value: str | Path, *, kind: str) -> Path:
    """Resolve one relative directory and reject escapes outside the root."""

    # On non-Windows hosts, reject Windows absolute paths (C:\...) early
    # because Path() would treat them as relative paths.
    if not _IS_WINDOWS and _is_windows_absolute(value):
        raise ConfigError(
            f"[paths].{kind} {value!r} is a Windows absolute path; "
            f"use a project-relative path or a POSIX absolute path"
        )

    resolved_root = normalize_project_path(root)
    path = Path(value).expanduser()
    resolved = (
        path.resolve()
        if _is_absolute_path(path)
        else (resolved_root / path).resolve()
    )
    resolved = normalize_project_path(resolved)
    if not _inside(resolved_root, resolved):
        raise ConfigError(
            f"[paths].{kind} {value!r} escapes the project root {root}"
        )
    return resolved


def resolve_paths(
    profile: Profile,
    paths: PathSettings,
    env: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> ResolvedPaths:
    """Resolve and verify all configured directories for one profile."""

    root = resolve_root(profile, paths, env=env, project_root=project_root)
    return ResolvedPaths(
        root=root,
        data=resolve_dir(root, paths.data, kind="data"),
        cache=resolve_dir(root, paths.cache, kind="cache"),
        output=resolve_dir(root, paths.output, kind="output"),
    )
