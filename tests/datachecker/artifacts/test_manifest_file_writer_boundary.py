"""Regression tests for datachecker manifest writer boundaries."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from datachecker.manifests.artifact_manifest import RunArtifactIdentity
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.crawl_state_manifest_writer import (
    CrawlStateManifestWriter,
)
from datachecker.manifests.manifest_file_writer import ManifestFileWriter
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_datachecker_does_not_import_crawler_runtime() -> None:
    violations: list[str] = []
    datachecker_root = PROJECT_ROOT / "datachecker"

    for path in datachecker_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module,)

            for module in imported:
                if module == "crawler.runtime" or module.startswith(
                    "crawler.runtime."
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"{module}"
                    )

    assert violations == []


def test_manifest_file_writer_uses_injected_history_identity(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 7, 28, 14, 30, tzinfo=UTC)

    writer = ManifestFileWriter(
        now=lambda: fixed_time,
        generate_id=lambda: "history-identifier",
        replace_retry_attempts=1,
        replace_retry_delay_seconds=0,
        replace_retry_jitter_seconds=0,
    )

    path = tmp_path / "manifest.json"

    writer.write(
        path=path,
        payload={"version": 1},
    )
    writer.write(
        path=path,
        payload={"version": 2},
    )

    history_files = tuple((tmp_path / ".history").glob("*.bak"))

    assert len(history_files) == 1
    assert "history-iden" in history_files[0].name
    assert "20260728T143000000000Z" in history_files[0].name
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert not any(tmp_path.glob("*.tmp"))


def test_manifest_file_writer_skips_history_when_preserve_previous_false(
    tmp_path: Path,
) -> None:
    writer = ManifestFileWriter(
        now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        generate_id=lambda: "unused-id",
        replace_retry_attempts=1,
        replace_retry_delay_seconds=0,
        replace_retry_jitter_seconds=0,
    )
    path = tmp_path / "manifest.json"

    writer.write(path=path, payload={"version": 1})
    writer.write(
        path=path,
        payload={"version": 2},
        preserve_previous=False,
    )

    assert not (tmp_path / ".history").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}


def test_crawl_attempt_id_preserves_wire_format_via_started_write(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 7, 28, 15, 0, tzinfo=UTC)
    identity = _identity()
    registry = _Registry(manifest_path=tmp_path / "crawl_state.json")
    file_writer = ManifestFileWriter(
        now=lambda: fixed_time,
        generate_id=lambda: "file-writer-id",
        replace_retry_attempts=1,
        replace_retry_delay_seconds=0,
        replace_retry_jitter_seconds=0,
    )
    writer = CrawlStateManifestWriter(
        artifact_path_registry=registry,
        reference_resolver=_EmptyReferenceResolver(
            project_root=tmp_path,
            registry=registry,
        ),
        logger=_Logger(),
        project_root=tmp_path,
        file_writer=file_writer,
        artifact_identity=identity,
        now=lambda: fixed_time,
        generate_id=lambda: "abcdef1234567890zzzzzzzz",
    )

    manifest = writer.write_crawl_state_started(
        source_registry_hash="source-hash",
        crawl_settings_hash="settings-hash",
    )

    assert manifest.attempt_id == "crawl_attempt_abcdef1234567890zzzzzzzz"
    assert manifest.started_at == fixed_time.isoformat()
    assert manifest.updated_at == fixed_time.isoformat()
    assert manifest.status is WorkflowLifecycleStatus.RUNNING

    payload = json.loads(
        (tmp_path / "crawl_state.json").read_text(encoding="utf-8")
    )
    assert payload["attempt_id"] == "crawl_attempt_abcdef1234567890zzzzzzzz"
    assert payload["started_at"] == fixed_time.isoformat()


def test_manifest_writer_base_uses_injected_now_for_timestamps(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 7, 28, 16, 45, 12, tzinfo=UTC)
    identity = _identity()
    registry = _Registry(manifest_path=tmp_path / "crawl_state.json")
    writer = CrawlStateManifestWriter(
        artifact_path_registry=registry,
        reference_resolver=_EmptyReferenceResolver(
            project_root=tmp_path,
            registry=registry,
        ),
        logger=_Logger(),
        project_root=tmp_path,
        file_writer=ManifestFileWriter(
            now=lambda: fixed_time,
            generate_id=lambda: "history",
            replace_retry_attempts=1,
            replace_retry_delay_seconds=0,
            replace_retry_jitter_seconds=0,
        ),
        artifact_identity=identity,
        now=lambda: fixed_time,
        generate_id=lambda: "attempt-id-value-1234567890",
    )

    assert writer._utc_now_iso() == fixed_time.isoformat()
    manifest = writer.write_crawl_state_started(
        source_registry_hash="source-hash",
        crawl_settings_hash="settings-hash",
    )
    assert manifest.started_at == fixed_time.isoformat()


@pytest.mark.parametrize(
    "status",
    (
        WorkflowLifecycleStatus.RUNNING,
        WorkflowLifecycleStatus.RECOVERING,
    ),
)
def test_crawl_start_refuses_to_overwrite_active_attempt_in_same_generation(
    tmp_path: Path,
    status: WorkflowLifecycleStatus,
) -> None:
    fixed_time = datetime(2026, 7, 28, 17, 0, tzinfo=UTC)
    identity = _identity()
    registry = _Registry(manifest_path=tmp_path / "crawl_state.json")
    resolver = _EmptyReferenceResolver(
        project_root=tmp_path,
        registry=registry,
        current_state=CrawlStateManifest(
            **{
                **identity.manifest_fields(),
                "config_fingerprint": "previous-config",
            },
            status=status,
            attempt_id="crawl_attempt_interrupted",
            started_at=fixed_time.isoformat(),
            updated_at=fixed_time.isoformat(),
            completed_at=(
                fixed_time.isoformat()
                if status is WorkflowLifecycleStatus.RECOVERING
                else None
            ),
            raw_run_directory=None,
            run_summary_path=None,
            previous_status=WorkflowLifecycleStatus.RUNNING,
            previous_raw_run_directory=None,
            last_successful_completed_at=None,
            last_successful_manifest_path=None,
            error_type=None,
            error_message=None,
        ),
    )
    writer = CrawlStateManifestWriter(
        artifact_path_registry=registry,
        reference_resolver=resolver,
        logger=_Logger(),
        project_root=tmp_path,
        file_writer=ManifestFileWriter(
            now=lambda: fixed_time,
            generate_id=lambda: "history",
            replace_retry_attempts=1,
            replace_retry_delay_seconds=0,
            replace_retry_jitter_seconds=0,
        ),
        artifact_identity=identity,
        now=lambda: fixed_time,
        generate_id=lambda: "new-attempt",
    )

    with pytest.raises(RuntimeError, match="must be reconciled"):
        writer.write_crawl_state_started(
            source_registry_hash="source-hash",
            crawl_settings_hash="settings-hash",
        )

    assert not registry.manifest_path.exists()


def _identity() -> RunArtifactIdentity:
    return RunArtifactIdentity(
        generation_id="generation",
        workflow_id="workflow",
        project_fingerprint="project",
        config_fingerprint="config",
        environment_name="dev",
        environment_fingerprint="environment",
        python_version="3.12",
        dependency_lock_fingerprint="lock",
    )


@dataclass
class _Registry:
    manifest_path: Path

    def crawl_state_manifest_path(self) -> Path:
        return self.manifest_path

    def crawl_manifest_path(self) -> Path:
        return self.manifest_path.with_name("crawl.json")


@dataclass
class _EmptyReferenceResolver:
    project_root: Path
    registry: _Registry
    current_state: Any = None

    def read_crawl_state(self) -> Any:
        return self.current_state

    def existing_crawl_manifest(self) -> Any:
        return None

    def existing_crawl_manifest_path(self) -> Path | None:
        path = self.registry.crawl_manifest_path()
        return path if path.exists() else None

    def existing_crawl_manifest_completed_at(self) -> str | None:
        return None


class _Logger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None

    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None
