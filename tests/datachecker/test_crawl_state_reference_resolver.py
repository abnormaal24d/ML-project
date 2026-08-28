from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.crawl_state_reference_resolver import (
    CrawlStateReferenceResolver,
)
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus


@dataclass
class _Registry:
    manifest_path: Path

    def crawl_state_manifest_path(self) -> Path:
        return self.manifest_path


class _Logger:
    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


def _manifest(
    *, raw_directory: Path, summary_path: Path
) -> CrawlStateManifest:
    return CrawlStateManifest(
        generation_id="generation",
        workflow_id="workflow",
        project_fingerprint="project",
        config_fingerprint="config",
        environment_name="dev",
        environment_fingerprint="environment",
        python_version="3.12",
        dependency_lock_fingerprint="lock",
        status=WorkflowLifecycleStatus.RUNNING,
        attempt_id="attempt",
        started_at="2026-07-22T00:00:00+00:00",
        updated_at="2026-07-22T00:00:00+00:00",
        completed_at=None,
        raw_run_directory=raw_directory,
        run_summary_path=summary_path,
        previous_status=None,
        previous_raw_run_directory=None,
        last_successful_completed_at=None,
        last_successful_manifest_path=None,
        error_type=None,
        error_message=None,
        raw_run_id="raw-run",
        crawl_session_id="crawl-session",
    )


def _resolver(
    *, project_root: Path, manifest: CrawlStateManifest
) -> CrawlStateReferenceResolver:
    manifest_path = project_root / "crawl-state.json"
    manifest_path.write_text(
        json.dumps(manifest.to_payload()),
        encoding="utf-8",
    )
    return CrawlStateReferenceResolver(
        artifact_path_registry=_Registry(manifest_path),
        logger=_Logger(),
        project_root=project_root,
    )


def test_resolves_project_relative_raw_run_link(tmp_path: Path) -> None:
    raw_directory = tmp_path / "data" / "raw" / "run"
    raw_directory.mkdir(parents=True)
    summary_path = raw_directory / "run_manifest.json"
    summary_path.write_text("{}", encoding="utf-8")

    resolved = _resolver(
        project_root=tmp_path,
        manifest=_manifest(
            raw_directory=Path("data/raw/run"),
            summary_path=Path("data/raw/run/run_manifest.json"),
        ),
    ).read_crawl_state()

    assert resolved is not None
    assert resolved.raw_run_directory == raw_directory.resolve()
    assert resolved.run_summary_path == summary_path.resolve()
    assert resolved.raw_run_id == "raw-run"
    assert resolved.crawl_session_id == "crawl-session"


def test_missing_raw_run_link_is_cleared_as_one_unit(tmp_path: Path) -> None:
    raw_directory = tmp_path / "data" / "raw" / "run"
    raw_directory.mkdir(parents=True)

    resolved = _resolver(
        project_root=tmp_path,
        manifest=_manifest(
            raw_directory=Path("data/raw/run"),
            summary_path=Path("data/raw/run/missing.json"),
        ),
    ).read_crawl_state()

    assert resolved is not None
    assert resolved.raw_run_directory is None
    assert resolved.run_summary_path is None
    assert resolved.raw_run_id is None
    assert resolved.crawl_session_id is None


def test_crawl_state_reference_cannot_escape_project_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-crawl-state"
    outside.mkdir(exist_ok=True)
    summary_path = outside / "run_manifest.json"
    summary_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes project_root"):
        _resolver(
            project_root=tmp_path,
            manifest=_manifest(
                raw_directory=outside,
                summary_path=summary_path,
            ),
        ).read_crawl_state()
