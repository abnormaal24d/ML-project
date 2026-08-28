"""Cross-process synchronization for crawler dead-letter file adapters."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from threading import Lock, RLock, local
from typing import BinaryIO, cast

_registry_lock = Lock()
_path_locks: dict[str, DeadLetterPathLock] = {}


class DeadLetterPathLock:
    """Serialize append and acknowledgement across threads and processes."""

    def __init__(self, path: Path) -> None:
        self._thread_lock = RLock()
        self._thread_state = local()
        self._lock_path = path.with_suffix(f"{path.suffix}.lock")

    def __enter__(self) -> DeadLetterPathLock:
        self._thread_lock.acquire()
        try:
            depth = int(getattr(self._thread_state, "depth", 0))
            if depth == 0:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = self._lock_path.open("a+b")
                try:
                    _acquire_file_lock(handle)
                except BaseException:
                    handle.close()
                    raise
                self._thread_state.handle = handle
            self._thread_state.depth = depth + 1
            return self
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(self, *_exc: object) -> None:
        try:
            depth = int(getattr(self._thread_state, "depth", 0))
            if depth <= 1:
                handle = getattr(self._thread_state, "handle", None)
                if handle is not None:
                    try:
                        _release_file_lock(handle)
                    finally:
                        handle.close()
                        del self._thread_state.handle
                self._thread_state.depth = 0
            else:
                self._thread_state.depth = depth - 1
        finally:
            self._thread_lock.release()


def dead_letter_path_lock(path: Path) -> DeadLetterPathLock:
    """Return the shared process-aware lock for one resolved JSONL path."""

    key = str(path.resolve())
    with _registry_lock:
        return _path_locks.setdefault(key, DeadLetterPathLock(path.resolve()))


def _acquire_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    fcntl = import_module("fcntl")
    flock = cast(Callable[[int, int], None], fcntl.flock)
    lock_ex = cast(int, fcntl.LOCK_EX)
    flock(handle.fileno(), lock_ex)


def _release_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = import_module("fcntl")
    flock = cast(Callable[[int, int], None], fcntl.flock)
    lock_un = cast(int, fcntl.LOCK_UN)
    flock(handle.fileno(), lock_un)


__all__ = ["DeadLetterPathLock", "dead_letter_path_lock"]
