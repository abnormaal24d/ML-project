"""Contract tests for the process-wide workflow file lock.

Uses real temporary directories and the real ``workflow_file_lock`` API.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestration.bootstrap.workflow_lock import (
    WorkflowLockError,
    workflow_file_lock,
)


def _lock_path(project_root: Path) -> Path:
    return project_root / "runtime" / "locks" / "data_workflow.lock"


def _lock_kwargs(project_root: Path) -> dict[str, object]:
    return {
        "project_root": project_root,
        "workflow_id": "workflow-1",
        "generation_id": "generation-1",
        "heartbeat_interval_seconds": 60.0,
    }


def test_acquire_once_succeeds_and_releases_on_context_exit(
    tmp_path: Path,
) -> None:
    lock_path = _lock_path(tmp_path)

    with workflow_file_lock(**_lock_kwargs(tmp_path)):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_acquired_lock_payload_identifies_the_owner(tmp_path: Path) -> None:
    with workflow_file_lock(**_lock_kwargs(tmp_path)):
        payload = json.loads(_lock_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == socket.gethostname()
    assert payload["workflow_id"] == "workflow-1"
    assert payload["generation_id"] == "generation-1"
    assert isinstance(payload["owner_token"], str) and payload["owner_token"]
    assert isinstance(payload["heartbeat_sequence"], int)
    assert payload["heartbeat_sequence"] >= 1
    assert payload["stale_after_seconds"] == 3600.0


def test_second_concurrent_acquire_fails_while_lock_is_held(
    tmp_path: Path,
) -> None:
    acquired = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            acquired.set()
            assert release.wait(timeout=10)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    try:
        assert acquired.wait(timeout=10)

        with pytest.raises(WorkflowLockError, match="already running"):
            with workflow_file_lock(**_lock_kwargs(tmp_path)):
                raise AssertionError("a second owner must not enter the body")
    finally:
        release.set()
        holder_thread.join(timeout=10)

    assert not holder_thread.is_alive()
    assert not _lock_path(tmp_path).exists()


def test_lock_is_released_when_the_workflow_body_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="workflow body failed"):
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            raise RuntimeError("workflow body failed")

    assert not _lock_path(tmp_path).exists()


def test_stale_lock_is_taken_over_and_released(tmp_path: Path) -> None:
    lock_path = _lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True)
    stale_timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).isoformat()
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "process_start_time": stale_timestamp,
                "workflow_id": "old-workflow",
                "generation_id": "old-generation",
                "project_fingerprint": "old-fingerprint",
                "created_at": stale_timestamp,
                "heartbeat_at": stale_timestamp,
                "owner_token": "stale-owner",
                "heartbeat_sequence": 1,
                "stale_after_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with workflow_file_lock(
        **_lock_kwargs(tmp_path),
        stale_after_seconds=1.0,
    ):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["owner_token"] != "stale-owner"

    assert not lock_path.exists()


def test_fresh_empty_lock_is_not_stolen_within_grace_period(
    tmp_path: Path,
) -> None:
    lock_path = _lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("", encoding="utf-8")

    with pytest.raises(WorkflowLockError, match="already running"):
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            raise AssertionError("a fresh empty lock must stay claimed")


def test_ancient_empty_lock_is_removed_and_reacquired(tmp_path: Path) -> None:
    lock_path = _lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("", encoding="utf-8")
    old_timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).timestamp()
    os.utime(lock_path, (old_timestamp, old_timestamp))

    with workflow_file_lock(**_lock_kwargs(tmp_path)):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()

    assert not lock_path.exists()
