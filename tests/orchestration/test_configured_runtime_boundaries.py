"""Regression contracts for configured orchestration entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from config.settings.root import Settings
from orchestration import main as runner_module
from orchestration.bootstrap import container as container_module
from orchestration.bootstrap.container import build_application_container
from orchestration.bootstrap.run_context import create_run_context
from orchestration.bootstrap.workflow_runner import execute_workflow_command
from orchestration.cli.argument_parser import RuntimeOptions
from orchestration.errors import (
    BackendConfigurationError,
    BootstrapBuildFailure,
    SettingsLoadError,
)
from orchestration.main import execute_runtime_command
from orchestration.settings_loader import load as load_settings
from orchestration.workflow.curated_snapshot_runtime import (
    CuratedSnapshotRuntimeResult,
)
from shared.runtime_primitives import Clock, IdGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_command_loads_through_orchestration_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The top-level dispatcher uses the same replaceable loader boundary."""

    captured: dict[str, object] = {}
    settings = cast(Settings, object())

    def load(**kwargs: object) -> Settings:
        captured.update(kwargs)
        return settings

    def execute_control_command(*, action: str, settings: Settings) -> int:
        captured["action"] = action
        captured["settings"] = settings
        return 17

    monkeypatch.setattr(runner_module, "load", load)
    monkeypatch.setattr(
        runner_module,
        "execute_control_command",
        execute_control_command,
    )
    options = RuntimeOptions(
        command="control",
        control_action="status",
        project_root=tmp_path / "workspace",
        config_root=tmp_path / "configuration",
        environment="dev",
        profile="dev",
    )

    assert execute_runtime_command(options) == 17
    assert captured["config_root"] == tmp_path / "configuration"
    assert captured["settings"] is settings


def test_validate_config_uses_the_loaded_settings_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    settings = cast(Settings, object())

    def load(**kwargs: object) -> Settings:
        captured["load"] = kwargs
        return settings

    def validate_runtime_configuration(**kwargs: object) -> object:
        captured["validation"] = kwargs
        return object()

    monkeypatch.setattr(runner_module, "load", load)
    monkeypatch.setattr(
        runner_module,
        "validate_runtime_configuration",
        validate_runtime_configuration,
    )
    monkeypatch.setattr(
        runner_module,
        "execute_control_command",
        lambda **_kwargs: pytest.fail(
            "validate-config must not enter the mutating control runner"
        ),
    )
    options = RuntimeOptions(
        command="control",
        control_action="validate-config",
        project_root=tmp_path / "workspace",
        config_root=tmp_path / "configuration",
        environment="dev",
        profile="dev",
    )

    assert execute_runtime_command(options) == 0
    assert captured["load"] == {
        "project_root": tmp_path / "workspace",
        "config_root": tmp_path / "configuration",
        "environment": "dev",
        "profile": "dev",
        "overrides": (),
    }
    assert captured["validation"] == {
        "settings": settings,
        "config_root": tmp_path / "configuration",
    }


def test_run_reuses_the_same_runtime_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    settings = cast(Settings, object())
    readiness = object()

    monkeypatch.setattr(runner_module, "load", lambda **_kwargs: settings)
    monkeypatch.setattr(
        runner_module,
        "validate_runtime_configuration",
        lambda **_kwargs: readiness,
    )

    def execute_workflow_command(**kwargs: object) -> int:
        captured.update(kwargs)
        return 19

    monkeypatch.setattr(
        runner_module,
        "execute_workflow_command",
        execute_workflow_command,
    )
    options = RuntimeOptions(
        command="run",
        project_root=tmp_path / "workspace",
        config_root=tmp_path / "configuration",
        environment="dev",
        profile="dev",
    )

    assert execute_runtime_command(options) == 19
    assert captured == {
        "options": options,
        "settings": settings,
        "runtime_readiness": readiness,
    }


def test_container_forwards_explicit_config_root_when_loading_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public container boundary must not silently use packaged config."""

    captured: dict[str, object] = {}
    config_root = tmp_path / "configuration"

    def fail_after_recording_load(**kwargs: object) -> Settings:
        captured.update(kwargs)
        raise ValueError("expected test settings-load failure")

    monkeypatch.setattr(container_module, "load", fail_after_recording_load)

    with pytest.raises(BootstrapBuildFailure) as exc_info:
        build_application_container(
            project_root=tmp_path / "workspace",
            config_root=config_root,
            environment="dev",
            run_context=create_run_context(stage="test"),
        )

    assert captured["config_root"] == config_root
    assert isinstance(exc_info.value.build_error, SettingsLoadError)


@pytest.mark.parametrize(
    (
        "settings_root",
        "requested_root",
        "settings_environment",
        "environment",
        "message",
    ),
    (
        (
            "configured-workspace",
            "requested-workspace",
            "dev",
            "dev",
            "supplied settings project_root does not match requested project_root",
        ),
        (
            "workspace",
            "workspace",
            "dev",
            "prod",
            "supplied settings environment does not match requested environment",
        ),
    ),
)
def test_container_rejects_mismatched_supplied_settings(
    tmp_path: Path,
    settings_root: str,
    requested_root: str,
    settings_environment: str,
    environment: str,
    message: str,
) -> None:
    """Injected settings and public selector arguments must identify one run."""

    settings = cast(
        Settings,
        SimpleNamespace(
            paths=SimpleNamespace(root=tmp_path / settings_root),
            application=SimpleNamespace(
                resolved_environment=lambda: settings_environment,
            ),
        ),
    )

    with pytest.raises(BootstrapBuildFailure) as exc_info:
        build_application_container(
            project_root=tmp_path / requested_root,
            environment=environment,
            settings=settings,
            run_context=create_run_context(stage="test"),
        )

    assert message in str(exc_info.value.build_error)


def test_curated_snapshot_config_entrypoint_forwards_config_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The standalone snapshot boundary honors the selected config tree."""

    from orchestration.bootstrap import (
        curated_snapshot_runner as runner_module,
    )

    captured: dict[str, object] = {}
    expected = CuratedSnapshotRuntimeResult(
        snapshot_id="snapshot-1",
        snapshot_directory=tmp_path / "snapshot-1",
    )

    def load_settings(**kwargs: object) -> Settings:
        captured.update(kwargs)
        return cast(
            Settings,
            SimpleNamespace(
                logging=SimpleNamespace(
                    root_name="test",
                    base_log_fields={},
                ),
                application=SimpleNamespace(environment="dev"),
            ),
        )

    class _StubRuntime:
        async def build(
            self,
            *,
            snapshot_id: str | None,
            raw_run_selection_mode: str | None,
            selected_run_ids: tuple[str, ...] | None,
        ) -> CuratedSnapshotRuntimeResult:
            del raw_run_selection_mode, selected_run_ids
            assert snapshot_id == "snapshot-1"
            return expected

    monkeypatch.setattr(runner_module, "load", load_settings)
    monkeypatch.setattr(
        runner_module,
        "build_curated_snapshot_runtime",
        lambda **_kwargs: _StubRuntime(),
    )

    result = runner_module.build_curated_snapshot_from_config(
        project_root=tmp_path / "workspace",
        config_root=tmp_path / "configuration",
        environment="dev",
        snapshot_id="snapshot-1",
        clock=cast(Clock, object()),
        id_generator=cast(IdGenerator, object()),
    )

    assert result == expected
    assert captured["config_root"] == tmp_path / "configuration"


@pytest.mark.parametrize("strategy", ("ddp", "fsdp"))
def test_autonomous_workflow_rejects_multi_process_training_strategy(
    tmp_path: Path,
    strategy: str,
) -> None:
    settings = load_settings(
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        profile="dev",
    )
    training = settings.training.model_copy(
        update={"distributed_strategy": strategy}
    )
    configured = settings.model_copy(update={"training": training})
    options = RuntimeOptions(
        command="run",
        project_root=tmp_path,
        config_root=tmp_path,
        environment="dev",
        profile="dev",
    )

    with pytest.raises(
        ValueError, match="autonomous workflow is single-process"
    ) as exc_info:
        execute_workflow_command(options=options, settings=configured)

    assert isinstance(exc_info.value, BackendConfigurationError)
    assert exc_info.value.kind == "backend_configuration_error"


def test_autonomous_workflow_rejects_distributed_launcher_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    settings = load_settings(
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        profile="dev",
    )
    options = RuntimeOptions(
        command="run",
        project_root=tmp_path,
        config_root=tmp_path,
        environment="dev",
        profile="dev",
    )

    with pytest.raises(ValueError, match="WORLD_SIZE=2") as exc_info:
        execute_workflow_command(options=options, settings=settings)

    assert isinstance(exc_info.value, BackendConfigurationError)


def test_autonomous_workflow_classifies_backend_preparation_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from training.runtime import preparation

    settings = load_settings(
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        profile="dev",
    )
    options = RuntimeOptions(
        command="run",
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        environment="dev",
        profile="dev",
    )
    cause = ValueError("configured training backend is not implemented")

    monkeypatch.setattr(
        preparation,
        "prepare_training_backend",
        lambda **_kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(ValueError, match="not implemented") as exc_info:
        execute_workflow_command(options=options, settings=settings)

    assert isinstance(exc_info.value, BackendConfigurationError)
    assert exc_info.value.cause is cause
    assert exc_info.value.component == "training_backend"


def test_runtime_preflight_classifies_missing_torch_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A minimal install must fail through the safe startup error boundary."""

    from orchestration.settings_loader import validate_runtime_configuration

    settings = load_settings(
        project_root=tmp_path,
        config_root=PROJECT_ROOT,
        profile="dev",
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(
        sys.modules, "training.runtime.preparation", raising=False
    )

    with pytest.raises(BackendConfigurationError) as exc_info:
        validate_runtime_configuration(
            settings=settings,
            config_root=PROJECT_ROOT,
        )

    assert isinstance(exc_info.value.cause, ModuleNotFoundError)
    assert exc_info.value.component == "training_backend"
    assert exc_info.value.context == {
        "issue": "training_runtime_dependency_missing"
    }
