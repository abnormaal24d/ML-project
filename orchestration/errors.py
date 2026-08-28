"""Typed orchestration failures with stable runtime context."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from orchestration.bootstrap.shutdown import ResourceShutdownManager


class OrchestrationError(RuntimeError):
    """Base error carrying the stage, component, cause, and safe context."""

    kind = "orchestration_error"

    def __init__(
        self,
        message: str,
        *,
        stage: str = "runtime",
        component: str = "application",
        cause: BaseException | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if not message.strip():
            raise ValueError("orchestration error message must not be empty")
        if not stage.strip():
            raise ValueError("orchestration error stage must not be empty")
        if not component.strip():
            raise ValueError("orchestration error component must not be empty")
        super().__init__(message)
        self.stage = stage
        self.component = component
        self.cause = cause
        self.context = dict(context or {})


class BootstrapError(OrchestrationError):
    """Base class for failures while assembling application dependencies."""

    kind = "bootstrap_error"


class StartupConfigurationError(BootstrapError):
    """Expected startup/configuration failure safe for operator reporting.

    Human-readable exception messages and causes can contain deployment paths
    or other local details.  CLI error rendering therefore uses only the
    stable ``kind``, ``component``, and explicitly allow-listed context.
    """

    kind = "startup_configuration_error"


class SettingsLoadError(StartupConfigurationError):
    """Settings could not be loaded or validated."""

    kind = "settings_load_error"


class SettingsValidationError(SettingsLoadError, ValueError):
    """Loaded configuration violates a runtime-domain invariant."""

    kind = "settings_validation_error"


class BackendConfigurationError(StartupConfigurationError, ValueError):
    """A configured runtime backend cannot serve the selected workflow."""

    kind = "backend_configuration_error"


class LoggingConfigurationError(StartupConfigurationError):
    """Logging could not be configured safely."""

    kind = "logging_configuration_error"


class ApplicationContainerBuildError(BootstrapError):
    """The application container could not be assembled."""

    kind = "application_container_build_error"


class ApplicationWiringError(StartupConfigurationError):
    """A runtime dependency graph is incomplete or inconsistent."""

    kind = "application_wiring_error"


class RuntimeServicesBuildError(StartupConfigurationError):
    """Crawler runtime services could not be constructed."""

    kind = "runtime_services_build_error"


class BootstrapBuildFailure(BootstrapError):
    """Preserve partial resources when bootstrap fails before shutdown."""

    kind = "bootstrap_build_failure"

    def __init__(
        self,
        *,
        build_error: BaseException,
        shutdown_manager: ResourceShutdownManager | None,
    ) -> None:
        super().__init__(
            f"bootstrap build failed: {build_error}",
            stage="bootstrap",
            component="container",
            cause=build_error,
            context={"build_error_type": type(build_error).__name__},
        )
        self.build_error = build_error
        self.shutdown_manager = shutdown_manager


class ExecutionError(OrchestrationError):
    """The assembled application failed during crawler execution."""

    kind = "execution_error"

    def __init__(self, *, cause: BaseException) -> None:
        super().__init__(
            f"application execution failed: {cause}",
            stage="execution",
            component="crawler",
            cause=cause,
            context={"error_type": type(cause).__name__},
        )


class LifecycleError(OrchestrationError):
    """Both a primary lifecycle operation and its cleanup failed."""

    kind = "lifecycle_error"

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        component: str,
        primary_error: BaseException,
        secondary_error: BaseException,
    ) -> None:
        super().__init__(
            message,
            stage=stage,
            component=component,
            cause=secondary_error,
            context={
                "primary_error_type": type(primary_error).__name__,
                "secondary_error_type": type(secondary_error).__name__,
            },
        )
        self.primary_error = primary_error
        self.secondary_error = secondary_error


class ShutdownStepError(BootstrapError):
    """One named shutdown step failed."""

    kind = "shutdown_step_error"

    def __init__(self, *, step_name: str, cause: BaseException) -> None:
        if not step_name.strip():
            raise ValueError("shutdown step name must not be empty")
        super().__init__(
            f"{step_name} shutdown failed: {cause}",
            stage="shutdown",
            component=step_name,
            cause=cause,
            context={"error_type": type(cause).__name__},
        )
        self.step_name = step_name
        self.original_error = cause


class ShutdownError(BootstrapError):
    """One or more deterministic shutdown steps failed."""

    kind = "shutdown_error"

    def __init__(self, *, errors: tuple[ShutdownStepError, ...]) -> None:
        if not errors:
            raise ValueError("shutdown error requires at least one step error")
        super().__init__(
            "; ".join(str(error) for error in errors),
            stage="shutdown",
            component="runtime",
            cause=errors[0],
            context={
                "error_count": len(errors),
                "failed_steps": [error.step_name for error in errors],
            },
        )
        self.errors = errors
