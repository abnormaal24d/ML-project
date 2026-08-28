"""CLI process entrypoint for autonomous workflow orchestration.

One CLI invocation: parse options, load settings, dispatch the run or
control command, and translate process-boundary errors into exit codes.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence

from orchestration.bootstrap.application import (
    FAILURE_EXIT_CODE,
    INTERRUPTED_EXIT_CODE,
    SUCCESS_EXIT_CODE,
)
from orchestration.bootstrap.control_command_runner import (
    execute_control_command,
)
from orchestration.bootstrap.workflow_runner import (
    execute_workflow_command,
)
from orchestration.cli.argument_parser import (
    RuntimeOptions,
    parse_runtime_options,
)
from orchestration.errors import StartupConfigurationError
from orchestration.settings_loader import (
    load,
    validate_runtime_configuration,
)

STARTUP_CONFIGURATION_EXIT_CODE = 2


def execute_runtime_command(options: RuntimeOptions) -> int:
    """Configure and execute one validated runtime command."""

    overrides = ("training.device=cuda",) if options.use_cuda else ()
    settings = load(
        project_root=options.project_root,
        config_root=options.config_root,
        environment=options.environment,
        profile=options.profile,
        overrides=overrides,
    )

    if options.command == "run":
        runtime_readiness = validate_runtime_configuration(
            settings=settings,
            config_root=options.config_root,
        )
        return execute_workflow_command(
            options=options,
            settings=settings,
            runtime_readiness=runtime_readiness,
        )

    if options.command == "control":
        control_action = options.control_action

        if control_action is None:
            raise ValueError("Control command requires an action.")

        if control_action == "validate-config":
            validate_runtime_configuration(
                settings=settings,
                config_root=options.config_root,
            )
            print("Configuration valid")
            return int(SUCCESS_EXIT_CODE)

        return execute_control_command(
            action=control_action,
            settings=settings,
        )

    raise ValueError(f"Unsupported runtime command: {options.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI invocation and return its process exit code."""

    try:
        return execute_runtime_command(parse_runtime_options(argv))

    except StartupConfigurationError as error:
        fields = (
            ("kind", error.kind),
            ("component", error.component),
            ("setting", error.context.get("setting")),
            ("file", error.context.get("file")),
            ("issue", error.context.get("issue")),
        )

        print(
            "Startup configuration error:\n"
            + "\n".join(
                f"  {name + ':':<11}{value}"
                for name, value in fields
                if value is not None
            ),
            file=sys.stderr,
        )
        return STARTUP_CONFIGURATION_EXIT_CODE

    except KeyboardInterrupt:
        return INTERRUPTED_EXIT_CODE

    except Exception as error:
        print(f"Application failed: {error}", file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
