"""Process-wide file lock for the autonomous data workflow."""

from __future__ import annotations

import _thread
import ctypes
import json
import math
import os
import socket
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from datachecker.fingerprints import ProjectFingerprintCalculator

# Grace period during which an empty/unreadable lock file is treated as live
_LOCK_CLAIM_GRACE_SECONDS = 5.0


class WorkflowLockError(RuntimeError):
    """Raised when another data workflow process already owns the lock."""


class HeartbeatShutdownError(WorkflowLockError):
    """Raised when the workflow-lock heartbeat cannot be stopped safely."""


class LockOwnershipLostError(WorkflowLockError):
    """Raised when cleanup no longer owns the workflow lock it acquired."""


@contextmanager
def workflow_file_lock(
    *,
    project_root: Path,
    workflow_id: str,
    generation_id: str,
    stale_after_seconds: float = 3600.0,
    heartbeat_interval_seconds: float = 30.0,
    heartbeat_join_timeout_seconds: float = 2.0,
) -> Iterator[None]:
    """Acquire an exclusive file lock for generated workflow artifacts."""

    lock_dir = Path(project_root) / "runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "data_workflow.lock"
    owner_token = uuid4().hex
    project_fingerprint = ProjectFingerprintCalculator().calculate(
        project_root=Path(project_root),
    )
    process_start_time = _process_start_time_iso(os.getpid())
    if process_start_time is None:
        raise WorkflowLockError("cannot establish workflow process start time")

    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    heartbeat_errors: list[BaseException] = []
    lock_claimed = False
    lock_payload_written = False
    primary_error: BaseException | None = None

    try:
        try:
            _remove_stale_lock(
                lock_path=lock_path,
                stale_after_seconds=stale_after_seconds,
            )

            # Prepare the initial payload but write it to a temporary file
            # inside the lock directory. Atomically claim the final lock name
            # by creating a hard link to the temp file. The link will fail if
            # the lock already exists, avoiding the empty-file race.
            now = datetime.now(timezone.utc).isoformat()
            initial_payload = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "process_start_time": process_start_time,
                "workflow_id": workflow_id,
                "generation_id": generation_id,
                "project_fingerprint": project_fingerprint,
                "created_at": now,
                "heartbeat_at": now,
                "owner_token": owner_token,
                "heartbeat_sequence": 1,
                "stale_after_seconds": max(1.0, float(stale_after_seconds)),
            }

            fd, temp_name = tempfile.mkstemp(
                prefix=f".{lock_path.name}.",
                suffix=".claim",
                dir=str(lock_dir),
                text=True,
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        initial_payload, handle, indent=2, sort_keys=True
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temp_path, 0o600)

                try:
                    # Attempt to create a hard link; this will fail if the
                    # destination already exists, providing an atomic claim.
                    os.link(str(temp_path), str(lock_path))
                    lock_claimed = True
                except FileExistsError as exc:
                    # Someone else created the lock concurrently.
                    raise WorkflowLockError(
                        f"data workflow is already running: {lock_path}"
                    ) from exc
                finally:
                    # Remove the temporary file if the link succeeded (lock
                    # is now the live file), or if the link failed.
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        # Best-effort cleanup; existence of temp file is not fatal.
                        pass

            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise

        except FileExistsError as exc:
            raise WorkflowLockError(
                f"data workflow is already running: {lock_path}"
            ) from exc

        # Write the initial payload *before* starting heartbeat. If this
        # fails the finally block removes the claimed lockfile.
        # Note: when link succeeded the lock file already contains the
        # initial_payload. We still call _write_lock_payload to normalize
        # created_at and heartbeat_sequence in case callers expect exact
        # formatting produced by the writer.
        _write_lock_payload(
            lock_path=lock_path,
            owner_token=owner_token,
            stale_after_seconds=stale_after_seconds,
            workflow_id=workflow_id,
            generation_id=generation_id,
            project_fingerprint=project_fingerprint,
            process_start_time=process_start_time,
        )
        lock_payload_written = True

        # Only start heartbeat after a successful initial write.
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "lock_path": lock_path,
                "owner_token": owner_token,
                "workflow_id": workflow_id,
                "generation_id": generation_id,
                "project_fingerprint": project_fingerprint,
                "process_start_time": process_start_time,
                "stop_event": stop_heartbeat,
                "interval_seconds": heartbeat_interval_seconds,
                "stale_after_seconds": stale_after_seconds,
                "errors": heartbeat_errors,
            },
            name="data-workflow-lock-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            yield
        except KeyboardInterrupt as exc:
            if heartbeat_errors:
                raise WorkflowLockError(
                    "workflow lock heartbeat failed"
                ) from heartbeat_errors[0]
            raise exc

        if heartbeat_errors:
            raise WorkflowLockError(
                "workflow lock heartbeat failed"
            ) from heartbeat_errors[0]
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        heartbeat_shutdown_error: BaseException | None = None
        lock_release_error: BaseException | None = None

        try:
            _stop_heartbeat(
                stop_event=stop_heartbeat,
                heartbeat_thread=heartbeat_thread,
                join_timeout_seconds=heartbeat_join_timeout_seconds,
            )
        except BaseException as exc:
            heartbeat_shutdown_error = exc
        finally:
            if lock_claimed:
                try:
                    _release_owned_lock(
                        lock_path=lock_path,
                        owner_token=owner_token,
                        allow_empty_payload=not lock_payload_written,
                    )
                except BaseException as exc:
                    lock_release_error = exc

        cleanup_error = heartbeat_shutdown_error or lock_release_error
        if (
            heartbeat_shutdown_error is not None
            and lock_release_error is not None
        ):
            _add_cleanup_note(
                heartbeat_shutdown_error,
                lock_release_error,
            )

        if cleanup_error is not None:
            if primary_error is not None:
                _add_cleanup_note(primary_error, cleanup_error)
            else:
                raise cleanup_error


def _stop_heartbeat(
    *,
    stop_event: threading.Event,
    heartbeat_thread: threading.Thread | None,
    join_timeout_seconds: float,
) -> None:
    """Stop the heartbeat and report a thread that exceeds its join budget."""

    try:
        timeout = float(join_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "heartbeat join timeout must be a non-negative number"
        ) from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(
            "heartbeat join timeout must be a non-negative number"
        )

    stop_event.set()
    if heartbeat_thread is None or not heartbeat_thread.is_alive():
        return

    heartbeat_thread.join(timeout=timeout)
    if heartbeat_thread.is_alive():
        raise HeartbeatShutdownError(
            "workflow lock heartbeat thread did not stop"
        )


def _add_cleanup_note(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> None:
    """Preserve the primary failure while retaining cleanup diagnostics."""

    add_note = getattr(primary_error, "add_note", None)
    if callable(add_note):
        add_note(
            "workflow lock cleanup also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _heartbeat_loop(
    *,
    lock_path: Path,
    owner_token: str,
    stop_event: threading.Event,
    interval_seconds: float,
    stale_after_seconds: float,
    workflow_id: str,
    generation_id: str,
    project_fingerprint: str,
    process_start_time: str,
    errors: list[BaseException],
) -> None:
    """Refresh lock heartbeat until the owning workflow exits."""

    interval = max(1.0, float(interval_seconds))
    while not stop_event.wait(interval):
        try:
            _write_lock_payload(
                lock_path=lock_path,
                owner_token=owner_token,
                stale_after_seconds=stale_after_seconds,
                workflow_id=workflow_id,
                generation_id=generation_id,
                project_fingerprint=project_fingerprint,
                process_start_time=process_start_time,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).error(
                "heartbeat_write_failed: %s",
                exc,
            )
            errors.append(exc)
            stop_event.set()
            _thread.interrupt_main()
            return


def _write_lock_payload(
    *,
    lock_path: Path,
    owner_token: str,
    stale_after_seconds: float,
    workflow_id: str,
    generation_id: str,
    project_fingerprint: str,
    process_start_time: str,
) -> None:
    """Refresh an owned lock without replacing the lock pathname.

    The owner token is verified on the same open file descriptor that is
    updated. If a stale lock was unlinked meanwhile, this descriptor refers
    only to the old inode and can never replace a newer owner's lock file.
    """
    try:
        with lock_path.open("r+", encoding="utf-8") as handle:
            try:
                existing = json.load(handle)
            except json.JSONDecodeError as exc:
                raise LockOwnershipLostError(
                    "workflow lock payload became unreadable before heartbeat"
                ) from exc

            if not isinstance(existing, dict):
                raise LockOwnershipLostError(
                    "workflow lock payload changed before heartbeat"
                )
            if existing.get("owner_token") != owner_token:
                raise LockOwnershipLostError(
                    "workflow lock ownership changed before heartbeat"
                )

            sequence = existing.get("heartbeat_sequence")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 1
            ):
                raise LockOwnershipLostError(
                    "workflow lock heartbeat sequence is invalid"
                )

            now = datetime.now(timezone.utc).isoformat()
            created_at = existing.get("created_at")
            payload = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "process_start_time": process_start_time,
                "workflow_id": workflow_id,
                "generation_id": generation_id,
                "project_fingerprint": project_fingerprint,
                "created_at": (str(created_at) if created_at else now),
                "heartbeat_at": now,
                "owner_token": owner_token,
                "heartbeat_sequence": sequence + 1,
                "stale_after_seconds": max(1.0, float(stale_after_seconds)),
            }
            handle.seek(0)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except FileNotFoundError as exc:
        raise LockOwnershipLostError(
            "workflow lock disappeared before heartbeat"
        ) from exc


def _release_owned_lock(
    *,
    lock_path: Path,
    owner_token: str,
    allow_empty_payload: bool = False,
) -> None:
    payload = _read_lock_payload(lock_path=lock_path)
    if not payload and not lock_path.exists():
        return
    if not payload and allow_empty_payload:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return
        return
    if payload.get("owner_token") != owner_token:
        raise LockOwnershipLostError(
            "workflow lock ownership changed before release"
        )

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _remove_stale_lock(
    *,
    lock_path: Path,
    stale_after_seconds: float,
) -> None:
    if not lock_path.exists():
        return

    payload = _read_lock_payload(lock_path=lock_path)

    # If the payload is missing or invalid, treat very recent files as live
    # to avoid a race where a process creates an empty file and another
    # process removes it before the creator wrote the payload.
    if not payload:
        try:
            mtime = lock_path.stat().st_mtime
        except OSError:
            return
        age_seconds = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(mtime, timezone.utc)
        ).total_seconds()
        if age_seconds < _LOCK_CLAIM_GRACE_SECONDS:
            # File is too fresh to consider stale.
            return
        # Otherwise fall through and remove the stale-looking file.

    if _lock_is_live(
        payload=payload,
        lock_path=lock_path,
        stale_after_seconds=stale_after_seconds,
    ):
        return

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _lock_is_live(
    *,
    payload: dict[str, object],
    lock_path: Path,
    stale_after_seconds: float,
) -> bool:
    if not _valid_lock_payload(payload=payload):
        return False

    pid = _coerce_pid(payload.get("pid"))
    if pid is None:
        return True

    hostname = str(payload["hostname"])
    if hostname == socket.gethostname():
        if not _pid_exists(pid):
            return False
        recorded_start = str(payload["process_start_time"])
        observed_start = _process_start_time_iso(pid)
        if observed_start is not None and observed_start != recorded_start:
            return False
        return True

    stale_after = max(1.0, float(stale_after_seconds))
    heartbeat_at = _parse_iso_timestamp(payload.get("heartbeat_at"))
    if heartbeat_at is None:
        try:
            heartbeat_at = datetime.fromtimestamp(
                lock_path.stat().st_mtime,
                timezone.utc,
            )
        except OSError:
            return False

    age_seconds = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
    if age_seconds < -60.0:
        return False
    try:
        filesystem_age = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(lock_path.stat().st_mtime, timezone.utc)
        ).total_seconds()
    except OSError:
        return False
    return age_seconds < stale_after and filesystem_age < stale_after


def _valid_lock_payload(*, payload: dict[str, object]) -> bool:
    required_text = (
        "hostname",
        "process_start_time",
        "workflow_id",
        "generation_id",
        "project_fingerprint",
        "created_at",
        "heartbeat_at",
        "owner_token",
    )
    if any(
        not isinstance(payload.get(name), str)
        or not str(payload[name]).strip()
        for name in required_text
    ):
        return False
    sequence = payload.get("heartbeat_sequence")
    return (
        _coerce_pid(payload.get("pid")) is not None
        and isinstance(sequence, int)
        and not isinstance(sequence, bool)
        and sequence > 0
    )


def _read_lock_payload(*, lock_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _coerce_pid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_exists(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_exists(pid: int) -> bool:
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None:
        return False

    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        int(pid),
    )
    if handle:
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    return bool(kernel32.GetLastError() == 5)


def _process_start_time_iso(pid: int) -> str | None:
    if os.name == "nt":
        return _windows_process_start_time_iso(pid)
    stat_path = Path(f"/proc/{pid}/stat")
    proc_stat_path = Path("/proc/stat")
    try:
        stat_fields = stat_path.read_text(encoding="utf-8").split()
        start_ticks = int(stat_fields[21])
        boot_line = next(
            line
            for line in proc_stat_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("btime ")
        )
        boot_seconds = int(boot_line.split()[1])
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return None
        ticks_per_second = int(sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    timestamp = boot_seconds + (start_ticks / ticks_per_second)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _windows_process_start_time_iso(pid: int) -> str | None:
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None:
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [
            ("low", ctypes.c_ulong),
            ("high", ctypes.c_ulong),
        ]

    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    creation = FileTime()
    exit_time = FileTime()
    kernel_time = FileTime()
    user_time = FileTime()
    try:
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
    finally:
        kernel32.CloseHandle(handle)
    windows_ticks = (creation.high << 32) + creation.low
    unix_seconds = (windows_ticks / 10_000_000) - 11_644_473_600
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat()
