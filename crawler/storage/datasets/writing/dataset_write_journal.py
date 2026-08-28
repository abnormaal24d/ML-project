"""Durable rollback journal for raw dataset write transactions.

Journals are two-phase:
* ``pending`` — data may still be rolled back / recovered by restore
* ``committed`` — data is durable; only leftover journal cleanup remains

Commit first durably rewrites the journal to ``committed`` (including
directory fsync). Only after that is cleanup best-effort. A cleanup
failure must never cause callers to roll back already-committed data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JOURNAL_STATE_PENDING = "pending"
_JOURNAL_STATE_COMMITTED = "committed"
_VALID_JOURNAL_STATES = frozenset(
    {_JOURNAL_STATE_PENDING, _JOURNAL_STATE_COMMITTED}
)


class TransactionJournalCorruptError(RuntimeError):
    """Raised when a pending write journal cannot be trusted."""


@dataclass(frozen=True, slots=True)
class DatasetWriteTransaction:
    """Reference to one durable pending transaction journal."""

    transaction_id: str
    journal_path: Path


@dataclass(frozen=True, slots=True)
class JournalRecoveryResult:
    """Outcome of startup journal recovery."""

    rolled_back: tuple[str, ...]
    finalized_commits: tuple[str, ...]


class DatasetWriteJournal:
    """Create, commit, roll back, and recover dataset write journals."""

    def __init__(self, *, run_directory: Path) -> None:
        self._run_directory = run_directory.resolve()
        self._journal_directory = self._run_directory / "transactions"
        self._journal_directory.mkdir(parents=True, exist_ok=True)

    def begin(
        self,
        *,
        transaction_id: str,
        tracked_paths: tuple[Path, ...],
        payload_path: Path,
    ) -> DatasetWriteTransaction:
        safe_id = _safe_transaction_id(transaction_id)
        relative_payload = self._relative_path(payload_path)
        snapshots = {
            self._relative_path(path): {
                "existed": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
            for path in tracked_paths
        }
        payload = {
            "schema_version": 1,
            "state": _JOURNAL_STATE_PENDING,
            "transaction_id": safe_id,
            "tracked_paths": snapshots,
            "payload_path": relative_payload,
            "payload_existed": payload_path.exists(),
        }
        payload["checksum"] = _payload_checksum(payload)
        journal_path = self._journal_directory / f"{safe_id}.json"
        _write_json_atomic(path=journal_path, payload=payload)
        return DatasetWriteTransaction(safe_id, journal_path)

    def commit(self, transaction: DatasetWriteTransaction) -> bool:
        """Durably mark the transaction committed, then best-effort cleanup.

        Returns ``True`` when the committed journal was fully removed.
        Returns ``False`` when the durable commit marker is present but
        journal cleanup failed (startup recovery will finalize).

        Raises only when the durable ``committed`` marker cannot be written;
        callers may then roll back using the still-pending journal.
        """

        payload = self._read_verified(transaction.journal_path)
        state = _require_journal_state(payload)
        if state != _JOURNAL_STATE_PENDING:
            raise TransactionJournalCorruptError(
                "only a pending transaction can be committed"
            )

        payload["state"] = _JOURNAL_STATE_COMMITTED
        payload["checksum"] = _payload_checksum(payload)
        _write_json_atomic(
            path=transaction.journal_path,
            payload=payload,
        )
        # From here the dataset write is committed. Cleanup failure must not
        # trigger rollback of durable data.
        try:
            transaction.journal_path.unlink(missing_ok=False)
            _fsync_directory(self._journal_directory)
        except OSError:
            return False
        return True

    def rollback(self, transaction: DatasetWriteTransaction) -> None:
        """Restore the begin snapshot after a failed write or commit.

        A failed committed-marker directory fsync can leave a readable
        journal whose visible state is ``committed`` even though
        ``commit()`` did not complete successfully.

        The caller therefore determines commit success from the return of
        ``commit()``, not solely from the currently visible journal state.
        Any exception before a successful commit return remains rollbackable.
        """

        payload = self._read_verified(transaction.journal_path)
        _require_journal_state(payload)

        self._restore(payload)
        transaction.journal_path.unlink(missing_ok=False)
        _fsync_directory(self._journal_directory)

    def recover_pending(self) -> JournalRecoveryResult:
        rolled_back: list[str] = []
        finalized: list[str] = []
        any_unlinked = False
        for journal_path in sorted(self._journal_directory.glob("*.json")):
            payload = self._read_verified(journal_path)
            state = _require_journal_state(payload)
            transaction_id = str(payload["transaction_id"])

            if state == _JOURNAL_STATE_PENDING:
                self._restore(payload)
                rolled_back.append(transaction_id)
            elif state == _JOURNAL_STATE_COMMITTED:
                # Data is already committed; only clean leftover journal.
                finalized.append(transaction_id)
            else:
                raise TransactionJournalCorruptError(
                    f"unsupported journal state: {state}"
                )

            journal_path.unlink(missing_ok=False)
            any_unlinked = True

        if any_unlinked:
            _fsync_directory(self._journal_directory)
        return JournalRecoveryResult(
            rolled_back=tuple(rolled_back),
            finalized_commits=tuple(finalized),
        )

    def _restore(self, payload: dict[str, Any]) -> None:
        tracked = payload.get("tracked_paths")
        if not isinstance(tracked, dict):
            raise TransactionJournalCorruptError(
                "tracked_paths must be an object"
            )

        # Track which files were truncated and which directories need fsync.
        truncated_files: list[Path] = []
        dirs_to_fsync: set[Path] = set()

        for relative, snapshot in tracked.items():
            if not isinstance(snapshot, dict):
                raise TransactionJournalCorruptError(
                    "invalid tracked path snapshot"
                )
            path = self._absolute_path(str(relative))
            existed = snapshot.get("existed") is True
            size = snapshot.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise TransactionJournalCorruptError(
                    "invalid tracked file size"
                )
            if existed:
                if not path.exists():
                    raise TransactionJournalCorruptError(
                        f"tracked file disappeared: {relative}"
                    )
                # Truncate to the recorded size, then ensure data is flushed to disk.
                os.truncate(path, size)
                truncated_files.append(path)
                dirs_to_fsync.add(path.parent)
            else:
                # Remove files created during the failed transaction and fsync
                # their parent directories after deletion to make the removal
                # durable.
                try:
                    path.unlink(missing_ok=True)
                finally:
                    dirs_to_fsync.add(path.parent)

        payload_path = self._absolute_path(
            str(payload.get("payload_path") or "")
        )
        if payload.get("payload_existed") is not True:
            try:
                payload_path.unlink(missing_ok=True)
            finally:
                dirs_to_fsync.add(payload_path.parent)

        # Flush and fsync truncated files.
        for file_path in truncated_files:
            try:
                # Open for read+write without changing content; fsync underlying inode.
                with file_path.open("rb+") as fh:
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                # If fsync fails, raise to abort rollback — safer than deleting the
                # journal while the filesystem state may be uncertain.
                raise

        # Fsync all directories that were modified (deleted files or truncated files).
        for d in sorted(dirs_to_fsync):
            try:
                _fsync_directory(d)
            except Exception:
                # As above, fail fast to avoid removing the journal before durable
                # persistence of restored state.
                raise

    def _read_verified(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TransactionJournalCorruptError(
                f"cannot read transaction journal: {path.name}"
            ) from exc
        if not isinstance(payload, dict):
            raise TransactionJournalCorruptError(
                "transaction journal must be an object"
            )
        checksum = payload.pop("checksum", None)
        if not isinstance(checksum, str) or checksum != _payload_checksum(
            payload
        ):
            raise TransactionJournalCorruptError(
                "transaction journal checksum mismatch"
            )
        return payload

    def _relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self._run_directory).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"transaction path escapes run directory: {path}"
            ) from exc

    def _absolute_path(self, relative: str) -> Path:
        if not relative:
            raise TransactionJournalCorruptError("transaction path is empty")
        path = (self._run_directory / relative).resolve()
        try:
            path.relative_to(self._run_directory)
        except ValueError as exc:
            raise TransactionJournalCorruptError(
                f"transaction path escapes run directory: {relative}"
            ) from exc
        return path


def _require_journal_state(payload: dict[str, Any]) -> str:
    state = payload.get("state")
    if not isinstance(state, str) or state not in _VALID_JOURNAL_STATES:
        raise TransactionJournalCorruptError(
            "journal state must be pending or committed"
        )
    return state


def _safe_transaction_id(value: str) -> str:
    text = value.strip()
    if not text or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in text
    ):
        raise ValueError("transaction_id contains unsafe characters")
    return text


def _payload_checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(*, path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(".tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        # Leave no .tmp behind when replace/fsync fails mid-write.
        temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
