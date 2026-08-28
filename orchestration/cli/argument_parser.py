"""Canonical command-line schema for the autonomous production workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from config.environment.source_selection import (
    ConfigSourceResolver,
    profile_for_environment,
)
from config.profiles import Profile, normalize_profile

PROGRAM_NAME = "multimodal-crawler"

RuntimeCommand = Literal["run", "control"]
ControlAction = Literal["pause", "resume", "stop", "status", "validate-config"]


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Resolved command-line options for one workflow execution."""

    command: RuntimeCommand
    project_root: Path
    config_root: Path
    environment: str
    profile: Profile
    control_action: ControlAction | None = None
    use_cuda: bool = False
    fresh_run: bool = False
    resume: bool = True
    checkpoint_headers: bool = False
    checkpoint_blob_storage: Path | None = None
    staging_lock: Path | None = None


def parse_runtime_options(
    argv: Sequence[str] | None = None,
) -> RuntimeOptions:
    """Parse one autonomous workflow or control command."""

    parser = _build_parser()
    cli_options = parser.parse_args(argv)

    try:
        sources = ConfigSourceResolver().resolve(
            project_root=cli_options.project_root,
            config_root=cli_options.config_root,
            environment=cli_options.environment,
        )
    except FileNotFoundError:
        # Selector failures are argparse errors, but the config-root path is
        # deployment topology and must not be echoed to stderr.
        parser.error("configuration root must contain config/files")
    except OSError:
        parser.error("configuration selector files could not be read")
    except ValueError as exc:
        parser.error(str(exc))

    environment = sources.environment

    command = cast(RuntimeCommand, cli_options.command)
    control_action = getattr(cli_options, "control_action", None)

    env = environment
    profile = profile_for_environment(env)

    if getattr(cli_options, "profile", None):
        requested_profile = normalize_profile(cli_options.profile)
        if requested_profile != profile:
            parser.error(
                f"--profile {requested_profile!r} is incompatible with "
                f"--environment {env!r}; use {profile!r}"
            )

    if command == "run" and env == "prod":
        if not getattr(cli_options, "checkpoint_headers", False):
            parser.error(
                "--checkpoint-headers is required for prod environments"
            )
        if getattr(cli_options, "checkpoint_blob_storage", None) is None:
            parser.error(
                "--checkpoint-blob-storage is required for prod environments"
            )
        if getattr(cli_options, "staging_lock", None) is None:
            parser.error("--staging-lock is required for prod environments")

    return RuntimeOptions(
        command=command,
        project_root=sources.project_root,
        config_root=sources.config_root,
        environment=env,
        profile=profile,
        control_action=control_action,
        use_cuda=bool(getattr(cli_options, "use_cuda", False)),
        fresh_run=bool(getattr(cli_options, "fresh_run", False)),
        resume=bool(getattr(cli_options, "resume", True)),
        checkpoint_headers=bool(
            getattr(cli_options, "checkpoint_headers", False)
        ),
        checkpoint_blob_storage=getattr(
            cli_options, "checkpoint_blob_storage", None
        ),
        staging_lock=getattr(cli_options, "staging_lock", None),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Run the autonomous multimodal data workflow or control "
            "an active workflow."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{run,control}",
    )

    common_options = _common_options_parser()

    run_parser = subparsers.add_parser(
        "run",
        parents=(common_options,),
        help=(
            "Inspect the current data state and automatically crawl, "
            "preprocess, augment, or train as required."
        ),
    )
    run_parser.add_argument(
        "--fresh-run",
        action="store_true",
        help=(
            "Remove existing workflow data and artifacts before starting "
            "a new workflow generation."
        ),
    )
    run_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reconcile and resume recoverable interrupted workflow state "
            "before continuing."
        ),
    )
    run_parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA when the workflow determines that training is required.",
    )
    run_parser.add_argument(
        "--checkpoint-headers",
        action="store_true",
        help=(
            "Require standalone checkpoint headers for every persisted "
            "checkpoint (mandatory for prod environments)."
        ),
    )
    run_parser.add_argument(
        "--checkpoint-blob-storage",
        metavar="DIR",
        type=Path,
        help=(
            "Persist checkpoints exclusively in this external "
            "content-addressable store (mandatory for prod environments)."
        ),
    )
    run_parser.add_argument(
        "--staging-lock",
        metavar="PATH",
        type=Path,
        help=(
            "Exclusive release-staging lock path; promotions fail closed "
            "while another workflow holds it (mandatory for prod "
            "environments)."
        ),
    )

    control_parser = subparsers.add_parser(
        "control",
        parents=(common_options,),
        help=(
            "Pause, resume, stop, inspect the active workflow, or validate "
            "configuration readiness."
        ),
    )
    control_parser.add_argument(
        "control_action",
        choices=("pause", "resume", "stop", "status", "validate-config"),
        metavar="{pause,resume,stop,status,validate-config}",
    )

    return parser


def _common_options_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--project-root",
        help=(
            "Writable workspace root for data and runtime artifacts. "
            "Defaults to DATA_ENGINE_PROJECT_ROOT or the current directory."
        ),
    )
    parser.add_argument(
        "--config-root",
        help=(
            "Read-only artifact root containing config/files and optionally "
            "config/profiles. Missing profiles fall back to the installed "
            "package. Defaults to DATA_ENGINE_CONFIG_ROOT or packaged "
            "configuration."
        ),
    )
    parser.add_argument(
        "--environment",
        help=(
            "Runtime environment (dev, test, prod). "
            "May also be provided via DATA_ENGINE_ENVIRONMENT or APP_ENV. "
            "Required for non-test workflows; there is no implicit dev fallback."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("test", "dev", "prod"),
        help=(
            "Configuration profile (test, dev, prod). "
            "Must match the profile required by --environment. "
            "Does not replace the required runtime environment selector."
        ),
    )

    return parser
