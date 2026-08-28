"""Resolve runtime configuration sources in one place."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from config.profiles import Profile

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

EnvironmentName = Literal["test", "dev", "prod"]

_ALLOWED_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    ("test", "dev", "prod")
)
PRODUCTION_ENVIRONMENTS: Final[frozenset[EnvironmentName]] = frozenset(
    ("prod",)
)
_PROFILE_BY_ENVIRONMENT: Final[dict[EnvironmentName, Profile]] = {
    "test": "test",
    "dev": "dev",
    "prod": "prod",
}

DEFAULT_CONFIG_DIR: Final[str] = "config/files"
DEFAULT_DOTENV_PATH: Final[str] = "env/.env"

PROJECT_ROOT_SELECTOR_ENV_VARS: Final[tuple[str, ...]] = (
    "DATA_ENGINE_PROJECT_ROOT",
)

CONFIG_ROOT_SELECTOR_ENV_VARS: Final[tuple[str, ...]] = (
    "DATA_ENGINE_CONFIG_ROOT",
)

ENVIRONMENT_SELECTOR_ENV_VARS: Final[tuple[str, ...]] = (
    "DATA_ENGINE_ENVIRONMENT",
    "APP_ENV",
)

_DOTENV_SELECTOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        *PROJECT_ROOT_SELECTOR_ENV_VARS,
        *ENVIRONMENT_SELECTOR_ENV_VARS,
    }
)


def packaged_config_root() -> Path:
    """Return the installed distribution root containing ``config/files``."""

    package_directory = Path(str(files("config"))).resolve()
    root = package_directory.parent
    _require_config_tree(root)
    return root


def _require_config_tree(config_root: Path) -> None:
    config_directory = config_root / DEFAULT_CONFIG_DIR
    if not config_directory.is_dir():
        raise FileNotFoundError(
            f"configuration root must contain config/files: {config_root}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedConfigSources:
    """Concrete selectors for one process or workflow."""

    project_root: Path
    config_root: Path
    environment: EnvironmentName


class ConfigSourceResolver:
    """Resolve writable workspace and read-only configuration roots."""

    def __init__(
        self,
        *,
        base_project_root: Path | None = None,
        base_config_root: Path | None = None,
    ) -> None:
        self._base_project_root = (
            base_project_root.resolve()
            if base_project_root is not None
            else Path.cwd().resolve()
        )
        self._base_config_root = (
            base_config_root.resolve()
            if base_config_root is not None
            else packaged_config_root()
        )

    def resolve(
        self,
        *,
        project_root: str | Path | None = None,
        config_root: str | Path | None = None,
        environment: str | None = None,
    ) -> ResolvedConfigSources:
        """Resolve one explicit runtime configuration schema."""

        resolved_config_root = (
            self._resolve_path(
                _first_text(
                    _path_text(config_root),
                    _read_env_value(CONFIG_ROOT_SELECTOR_ENV_VARS),
                ),
                base_dir=Path.cwd().resolve(),
            )
            or self._base_config_root
        )
        _require_config_tree(resolved_config_root)

        dotenv_values = self._read_dotenv_values(
            config_root=resolved_config_root
        )

        resolved_project_root = (
            self._resolve_path(
                _first_text(
                    _path_text(project_root),
                    _read_env_value(PROJECT_ROOT_SELECTOR_ENV_VARS),
                    _read_mapping_value(
                        dotenv_values,
                        PROJECT_ROOT_SELECTOR_ENV_VARS,
                    ),
                ),
                base_dir=self._base_project_root,
            )
            or self._base_project_root
        )

        environment_os_value = _read_env_value(ENVIRONMENT_SELECTOR_ENV_VARS)
        environment_dotenv_value = _read_mapping_value(
            dotenv_values,
            ENVIRONMENT_SELECTOR_ENV_VARS,
        )
        resolved_environment = require_environment(
            _first_text(
                environment,
                environment_os_value,
                environment_dotenv_value,
            )
        )
        environment_source = _resolve_selector_source(
            explicit=environment,
            os_value=environment_os_value,
            dotenv_value=environment_dotenv_value,
        )
        project_root_source = _resolve_selector_source(
            explicit=_path_text(project_root),
            os_value=_read_env_value(PROJECT_ROOT_SELECTOR_ENV_VARS),
            dotenv_value=_read_mapping_value(
                dotenv_values,
                PROJECT_ROOT_SELECTOR_ENV_VARS,
            ),
        )
        config_root_source = _resolve_selector_source(
            explicit=_path_text(config_root),
            os_value=_read_env_value(CONFIG_ROOT_SELECTOR_ENV_VARS),
            dotenv_value=None,
        )
        if (
            resolved_environment in PRODUCTION_ENVIRONMENTS
            and project_root_source == "default"
        ):
            raise ValueError(
                "production environments require an explicit project root "
                "(set DATA_ENGINE_PROJECT_ROOT or --project-root)"
            )
        _LOGGER.debug(
            "config_selectors_resolved",
            extra={
                "environment": resolved_environment,
                "environment_source": environment_source,
                "project_root_source": project_root_source,
                "project_root": str(resolved_project_root),
                "config_root_source": config_root_source,
                "config_root": str(resolved_config_root),
            },
        )

        return ResolvedConfigSources(
            project_root=resolved_project_root,
            config_root=resolved_config_root,
            environment=resolved_environment,
        )

    def _read_dotenv_values(self, *, config_root: Path) -> dict[str, str]:
        """Read dotenv values from the selected configuration tree."""

        return _read_selector_dotenv(
            config_root / DEFAULT_CONFIG_DIR / DEFAULT_DOTENV_PATH
        )

    def _resolve_path(
        self,
        raw_value: str | None,
        *,
        base_dir: Path,
    ) -> Path | None:
        if raw_value is None:
            return None
        path = Path(raw_value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()


def resolve_environment(
    value: str | None,
    *,
    default: EnvironmentName | None = "dev",
) -> EnvironmentName:
    """Resolve and validate the environment name."""

    if value is None or not str(value).strip():
        if default is None:
            raise ValueError(
                "environment must be explicitly provided for non-test workflows"
            )
        return default

    environment = str(value).strip().lower()
    if environment not in _ALLOWED_ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}. "
            f"Expected one of: {sorted(_ALLOWED_ENVIRONMENTS)}"
        )

    return environment  # type: ignore[return-value]


def require_environment(value: str | None) -> EnvironmentName:
    """Resolve an environment name without implicit dev fallback."""

    return resolve_environment(value, default=None)


def profile_for_environment(environment: EnvironmentName) -> Profile:
    """Return the one configuration profile allowed for an environment."""

    return _PROFILE_BY_ENVIRONMENT[environment]


def _resolve_selector_source(
    *,
    explicit: str | None,
    os_value: str | None,
    dotenv_value: str | None,
) -> str:
    if explicit:
        return "explicit"
    if os_value:
        return "os_environment"
    if dotenv_value:
        return "dotenv"
    return "default"


def _read_env_value(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _read_mapping_value(
    values: dict[str, str],
    names: Sequence[str],
) -> str | None:
    for name in names:
        value = values.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _first_text(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


def _read_selector_dotenv(path: Path) -> dict[str, str]:
    """Read only launcher selectors from the optional dotenv file."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(
                f"invalid dotenv selector on line {line_number} in {path.name}"
            )
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in _DOTENV_SELECTOR_KEYS:
            raise ValueError(f"dotenv key {key!r} is not a launcher selector")
        if key in values:
            raise ValueError(f"duplicate dotenv selector {key!r}")
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"unterminated dotenv selector {key!r}")
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values
