"""Exception type for the new configuration layer."""

from __future__ import annotations


class ConfigError(Exception):
    """Invalid, ambiguous, or unsupported configuration state.

    ``setting`` and ``issue`` are optional, stable operator context.  Raw
    exception messages remain diagnostic-only because they can contain local
    paths or other deployment details.
    """

    def __init__(
        self,
        message: str,
        *,
        setting: str | None = None,
        issue: str | None = None,
    ) -> None:
        super().__init__(message)
        self.setting = setting
        self.issue = issue


class RuntimeDependencyError(RuntimeError):
    """A configured backend requirement is unavailable at startup."""

    def __init__(
        self,
        message: str,
        *,
        setting: str | None = None,
        required_artifact: str | None = None,
        issue: str = "required_runtime_dependency_missing",
    ) -> None:
        super().__init__(message)
        self.setting = setting
        self.required_artifact = required_artifact
        self.issue = issue
