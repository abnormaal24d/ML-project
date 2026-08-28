"""Crawler runtime checkpoint store."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from config.environment.default_values import (
    DEFAULT_MANIFEST_REPLACE_RETRY_ATTEMPTS,
    DEFAULT_MANIFEST_REPLACE_RETRY_DELAY_SECONDS,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.crawler import CrawlStateStoreSettings
    from crawler.runtime.state.runtime_checkpoint_payload_builder import (
        RuntimeCheckpointPayloadBuilder,
    )
    from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot


class CrawlerCheckpointStore:
    """Persist and restore crawler runtime checkpoints."""

    _lock = Lock()
    _LOCK_ATTEMPTS = 40

    def __init__(
        self,
        *,
        settings: CrawlStateStoreSettings,
        state_directory: Path,
        checkpoint_path: Path,
        logger: ProjectLogger,
        payload_builder: RuntimeCheckpointPayloadBuilder,
    ) -> None:
        self._settings = settings
        self._state_directory = state_directory
        self._checkpoint_path = checkpoint_path
        self._logger = logger
        self._payload_builder = payload_builder
        self._replace_attempts = DEFAULT_MANIFEST_REPLACE_RETRY_ATTEMPTS
        self._replace_delay_seconds = (
            DEFAULT_MANIFEST_REPLACE_RETRY_DELAY_SECONDS
        )
        if self.enabled:
            self._state_directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """Return whether checkpoint persistence is enabled."""

        return self._settings.enabled

    @property
    def checkpoint_path(self) -> Path:
        """Return checkpoint file path."""

        return self._checkpoint_path

    def load_checkpoint(self) -> dict[str, object] | None:
        """Load the raw runtime checkpoint payload if available."""

        if not self.enabled:
            return None

        path = self.checkpoint_path
        if not path.exists():
            return None

        try:
            raw_text = path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except OSError as exc:
            self._logger.warning(
                "crawl_checkpoint_read_failed",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None
        except json.JSONDecodeError as exc:
            self._logger.warning(
                "crawl_checkpoint_invalid",
                path=str(path),
                error_type=type(exc).__name__,
                line=exc.lineno,
                column=exc.colno,
            )
            return None

        if not isinstance(payload, dict):
            self._logger.warning(
                "crawl_checkpoint_invalid_payload",
                path=str(path),
            )
            return None

        return payload

    def load_scheduler_checkpoint_payload(self) -> dict[str, object] | None:
        """Load the persisted scheduler checkpoint payload when available."""

        checkpoint_payload = self.load_checkpoint()
        if checkpoint_payload is None:
            return None

        scheduler_payload = checkpoint_payload.get("scheduler")
        if not isinstance(scheduler_payload, dict):
            self._logger.warning(
                "crawl_checkpoint_invalid_scheduler_payload",
                path=str(self.checkpoint_path),
            )
            return None

        return scheduler_payload

    def load_checkpoint_run_context(self) -> dict[str, object] | None:
        """Load persisted runtime context used to validate checkpoint reuse."""

        checkpoint_payload = self.load_checkpoint()
        if checkpoint_payload is None:
            return None

        run_context = checkpoint_payload.get("run_context")
        if not isinstance(run_context, dict):
            return None

        return run_context

    def write_checkpoint(self, *, payload: dict[str, object]) -> None:
        """Write the raw runtime checkpoint payload atomically."""

        if not self.enabled:
            return

        path = self.checkpoint_path
        self._write_json_atomic(
            path=path,
            payload=payload,
            pretty=self._settings.pretty_checkpoint_json,
        )

        self._logger.debug(
            "crawl_checkpoint_written",
            path=str(path),
        )

    def write_runtime_checkpoint(
        self,
        *,
        final: bool,
        scheduler_state: dict[str, object],
        worker_snapshot: WorkerPoolSnapshot,
        metrics: Any | None,
        run_context: dict[str, object] | None = None,
    ) -> None:
        """Persist the runtime checkpoint envelope for scheduler recovery."""

        payload = self._payload_builder.build(
            final=final,
            scheduler_state=scheduler_state,
            worker_snapshot=worker_snapshot,
            metrics=metrics,
            run_context=run_context,
        )
        self.write_checkpoint(payload=payload)

    def clear_checkpoint(self) -> None:
        """Delete the raw persisted runtime checkpoint file."""

        if not self.enabled:
            return

        self.checkpoint_path.unlink(missing_ok=True)

    def _write_json_atomic(
        self,
        *,
        path: Path,
        payload: dict[str, object],
        pretty: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = self._serialize_json(payload=payload, pretty=pretty)
        temp_path = self._write_temp_json_file(
            parent=path.parent,
            target_name=path.name,
            serialized=serialized,
        )
        lock_path = path.with_suffix(f"{path.suffix}.lock")
        try:
            self._with_sidecar_lock(
                lock_path=lock_path,
                callback=lambda: self._replace_with_retry(
                    temp_path=temp_path,
                    target_path=path,
                ),
            )
        except OSError:
            temp_path.unlink(missing_ok=True)
            raise
        _fsync_directory(path.parent)

    @staticmethod
    def _serialize_json(*, payload: dict[str, object], pretty: bool) -> str:
        if pretty:
            return (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                )
                + "\n"
            )

        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    @staticmethod
    def _write_temp_json_file(
        *,
        parent: Path,
        target_name: str,
        serialized: str,
    ) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f"{target_name}.",
                suffix=".tmp",
                delete=False,
            ) as file_handle:
                temp_path = Path(file_handle.name)
                file_handle.write(serialized)
                file_handle.flush()
                os.fsync(file_handle.fileno())
        except OSError:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        if temp_path is None:
            raise OSError(f"failed to create temp JSON file in {parent}")

        return temp_path

    def _replace_with_retry(
        self,
        *,
        temp_path: Path,
        target_path: Path,
    ) -> None:
        last_error: OSError | None = None

        with self._lock:
            for attempt in range(1, self._replace_attempts + 1):
                try:
                    os.replace(temp_path, target_path)
                    return
                except PermissionError as exc:
                    last_error = exc
                    if attempt >= self._replace_attempts:
                        break

                    self._logger.warning(
                        "atomic_json_replace_retrying",
                        path=str(target_path),
                        attempt=attempt,
                        max_attempts=self._replace_attempts,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    self._wait_before_retry(attempt=attempt)
                except OSError as exc:
                    temp_path.unlink(missing_ok=True)
                    raise OSError(
                        f"failed to replace JSON file: {target_path}"
                    ) from exc

        temp_path.unlink(missing_ok=True)

        if last_error is not None:
            raise last_error

    def _with_sidecar_lock(
        self,
        *,
        lock_path: Path,
        callback: Callable[[], None],
    ) -> None:
        lock_fd: int | None = None

        for attempt in range(1, self._LOCK_ATTEMPTS + 1):
            try:
                lock_fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                break
            except FileExistsError as exc:
                if attempt >= self._LOCK_ATTEMPTS:
                    raise PermissionError(
                        f"timed out waiting for JSON writer lock: {lock_path}"
                    ) from exc
                self._wait_before_retry(attempt=attempt)

        if lock_fd is None:
            raise PermissionError(
                f"could not acquire JSON writer lock: {lock_path}"
            )

        try:
            os.write(
                lock_fd, str(os.getpid()).encode("ascii", errors="ignore")
            )
            callback()
        finally:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)

    def _wait_before_retry(self, *, attempt: int) -> None:
        jitter = random.uniform(0.0, self._replace_delay_seconds)  # nosec B311
        delay_seconds = (self._replace_delay_seconds * attempt) + jitter
        Event().wait(delay_seconds)


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for the containing directory on POSIX systems."""

    if os.name == "nt":
        return

    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
