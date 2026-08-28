from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestration.bootstrap import workflow_lock
from orchestration.bootstrap.workflow_lock import (
    HeartbeatShutdownError,
    LockOwnershipLostError,
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


def test_heartbeat_join_failure_still_releases_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StuckHeartbeatThread:
        def __init__(self) -> None:
            self.started = False
            self.join_timeout: float | None = None

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    stuck_thread = _StuckHeartbeatThread()
    monkeypatch.setattr(
        workflow_lock.threading,
        "Thread",
        lambda **_kwargs: stuck_thread,
    )

    with pytest.raises(HeartbeatShutdownError):
        with workflow_file_lock(
            **_lock_kwargs(tmp_path),
            heartbeat_join_timeout_seconds=0.01,
        ):
            assert _lock_path(tmp_path).exists()

    assert stuck_thread.join_timeout == 0.01
    assert not _lock_path(tmp_path).exists()


def test_workflow_exception_releases_the_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workflow failed"):
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            assert _lock_path(tmp_path).exists()
            raise ValueError("workflow failed")

    assert not _lock_path(tmp_path).exists()


def test_initial_payload_failure_releases_the_claimed_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_initial_payload(**_kwargs: object) -> None:
        raise OSError("lock payload write failed")

    monkeypatch.setattr(
        workflow_lock,
        "_write_lock_payload",
        fail_initial_payload,
    )

    with pytest.raises(OSError, match="lock payload write failed"):
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            raise AssertionError("workflow body must not start")

    assert not _lock_path(tmp_path).exists()


def test_lock_cleanup_never_deletes_another_owner_lock(tmp_path: Path) -> None:
    with pytest.raises(LockOwnershipLostError):
        with workflow_file_lock(**_lock_kwargs(tmp_path)):
            lock_path = _lock_path(tmp_path)
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            payload["owner_token"] = "different-owner"
            lock_path.write_text(json.dumps(payload), encoding="utf-8")

    payload = json.loads(_lock_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["owner_token"] == "different-owner"


def test_lock_creation_is_exclusive(tmp_path: Path) -> None:
    with workflow_file_lock(**_lock_kwargs(tmp_path)):
        with pytest.raises(WorkflowLockError, match="already running"):
            with workflow_file_lock(**_lock_kwargs(tmp_path)):
                raise AssertionError("an exclusive lock was not enforced")


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


def test_missing_lock_during_cleanup_is_idempotent(tmp_path: Path) -> None:
    with workflow_file_lock(**_lock_kwargs(tmp_path)):
        _lock_path(tmp_path).unlink()

    assert not _lock_path(tmp_path).exists()
