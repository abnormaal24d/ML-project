"""Parse and atomically commit scheduler checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)
from crawler.scheduling.checkpointing.scheduler_task_envelope import (
    SchedulerTaskEnvelope,
)
from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue

if TYPE_CHECKING:
    from collections.abc import Callable

    from crawler.classification.media_kind import MediaKind
    from crawler.scheduling.checkpointing.scheduler_task_deserializer import (
        SchedulerTaskDeserializer,
    )
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )
    from crawler.scheduling.progress.active_task_registry import (
        ActiveTaskRegistry,
    )
    from crawler.scheduling.progress.scheduler_progress_state import (
        SchedulerProgressRestoreState,
        SchedulerProgressState,
    )
    from crawler.scheduling.queueing.host_task_queue import HostTaskQueue

RestoreSource = Literal[
    "queued",
    "delayed",
    "requeued_inflight",
    "dispatching",
]


@dataclass(frozen=True, slots=True)
class SchedulerRestoreItem:
    """One fully validated frontier item in a restore plan."""

    envelope: SchedulerTaskEnvelope
    source: RestoreSource
    delayed_wait_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SchedulerRestorePlan:
    """Complete validated scheduler replacement ready for commit."""

    items: tuple[SchedulerRestoreItem, ...]
    seen_entries: tuple[tuple[str, float], ...] | None
    progress: SchedulerProgressRestoreState | None
    next_sequence_value: int
    restored_queued_count: int
    restored_delayed_count: int
    restored_requeued_inflight_count: int
    restored_dispatching_count: int


@dataclass(frozen=True, slots=True)
class SchedulerRestoreResult:
    """Stable summary of one scheduler-state restoration pass."""

    restored_count: int
    restored_queued_count: int
    restored_delayed_count: int
    restored_requeued_inflight_count: int
    skipped_tasks: int
    next_sequence_value: int
    queued: int
    pending_hosts: int
    restored_dispatching_count: int = 0


class SchedulerStateRestorer:
    """Build a restore plan before applying any live-state mutation."""

    def __init__(self, *, deserializer: SchedulerTaskDeserializer) -> None:
        self._deserializer = deserializer

    def parse_restore_plan(
        self,
        *,
        payload: dict[str, object],
        clear_existing: bool,
        current_next_sequence: int,
        current_queue_size: int,
        current_ready_pending_by_host: dict[str | None, int],
        current_ready_kind_pending_by_host: dict[
            tuple[str | None, MediaKind], int
        ],
        current_seen_identity_keys: set[str],
        current_sequences: set[int],
        prepare_restored_envelope: Callable[..., SchedulerTaskEnvelope | None],
        parse_progress_state: Callable[
            [dict[str, object]], SchedulerProgressRestoreState
        ],
    ) -> SchedulerRestorePlan:
        """Parse and validate a whole checkpoint without live mutation."""

        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 2
        ):
            raise ValueError(
                "unsupported scheduler checkpoint schema: "
                f"{schema_version} (expected 2)"
            )

        progress_payload = payload.get("progress_counters")
        if not isinstance(progress_payload, dict):
            raise ValueError(
                "scheduler checkpoint missing progress_counters payload"
            )
        parsed_progress = parse_progress_state(progress_payload)
        progress = parsed_progress if clear_existing else None

        seen_entries = self._parse_seen_url_entries(
            payload.get("seen_url_entries"),
        )
        if not clear_existing:
            seen_entries = None

        admission_seen_identity_keys = (
            set() if clear_existing else set(current_seen_identity_keys)
        )
        seen_task_identity_keys: set[str] = set()
        planned_sequences = set() if clear_existing else set(current_sequences)
        ready_pending_by_host = (
            {} if clear_existing else dict(current_ready_pending_by_host)
        )
        ready_kind_pending_by_host = (
            {} if clear_existing else dict(current_ready_kind_pending_by_host)
        )
        queue_size = 0 if clear_existing else current_queue_size
        items: list[SchedulerRestoreItem] = []
        highest_sequence = -1
        counts: dict[RestoreSource, int] = {
            "queued": 0,
            "delayed": 0,
            "requeued_inflight": 0,
            "dispatching": 0,
        }

        restore_sources: tuple[tuple[str, RestoreSource], ...] = (
            ("queued_tasks", "queued"),
            ("delayed_tasks", "delayed"),
            ("requeued_inflight_tasks", "requeued_inflight"),
            ("dispatching_tasks", "dispatching"),
        )
        for key, source in restore_sources:
            raw_items = payload.get(key)
            if not isinstance(raw_items, list):
                raise ValueError(f"checkpoint field {key} must be a list")

            for raw_item in raw_items:
                envelope = self._deserialize_checkpoint_item(item=raw_item)
                sequence = envelope.sequence
                if sequence in planned_sequences:
                    raise ValueError(
                        f"checkpoint field {key} contains duplicate sequence"
                    )
                planned_sequences.add(sequence)

                delayed_wait_seconds = (
                    self._parse_delayed_wait_seconds(raw_item)
                    if source == "delayed"
                    else None
                )
                prepared = prepare_restored_envelope(
                    envelope=envelope,
                    queue_size=queue_size,
                    ready_pending_by_host=ready_pending_by_host,
                    ready_kind_pending_by_host=ready_kind_pending_by_host,
                    seen_identity_keys=admission_seen_identity_keys,
                    use_host_advice=not clear_existing,
                )
                if prepared is None:
                    raise ValueError(
                        f"checkpoint field {key} contains task item that "
                        "was not accepted on restore"
                    )

                identity_key = scheduler_task_identity_key(task=prepared.task)
                if identity_key in seen_task_identity_keys:
                    raise ValueError(
                        f"checkpoint field {key} contains duplicate task identity"
                    )
                seen_task_identity_keys.add(identity_key)
                admission_seen_identity_keys.add(identity_key)

                is_ready = source != "delayed" or not delayed_wait_seconds
                if is_ready:
                    host = prepared.host
                    ready_pending_by_host[host] = (
                        ready_pending_by_host.get(host, 0) + 1
                    )
                    kind_key = (host, prepared.task.kind)
                    ready_kind_pending_by_host[kind_key] = (
                        ready_kind_pending_by_host.get(kind_key, 0) + 1
                    )
                queue_size += 1
                highest_sequence = max(highest_sequence, sequence)
                counts[source] += 1
                items.append(
                    SchedulerRestoreItem(
                        envelope=prepared,
                        source=source,
                        delayed_wait_seconds=delayed_wait_seconds,
                    )
                )

        next_sequence = self._parse_next_sequence(payload.get("next_sequence"))
        minimum_next_sequence = highest_sequence + 1
        if not clear_existing:
            minimum_next_sequence = max(
                minimum_next_sequence,
                current_next_sequence,
            )
        if next_sequence < minimum_next_sequence:
            raise ValueError(
                "scheduler checkpoint next_sequence must exceed every task "
                "sequence and preserve the live sequence"
            )

        return SchedulerRestorePlan(
            items=tuple(items),
            seen_entries=seen_entries,
            progress=progress,
            next_sequence_value=next_sequence,
            restored_queued_count=counts["queued"],
            restored_delayed_count=counts["delayed"],
            restored_requeued_inflight_count=counts["requeued_inflight"],
            restored_dispatching_count=counts["dispatching"],
        )

    @staticmethod
    def commit_restore_plan(
        *,
        plan: SchedulerRestorePlan,
        clear_existing: bool,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        active_registry: ActiveTaskRegistry,
        seen_urls: SeenUrlRegistry,
        host_advice_tracker: HostAdviceTracker,
        progress_state: SchedulerProgressState,
        commit_restored_envelope: Callable[..., None],
    ) -> SchedulerRestoreResult:
        """Apply a fully validated plan as one non-failing commit phase."""

        if clear_existing and plan.progress is None:
            raise RuntimeError("replacement restore plan missing progress")

        if clear_existing:
            host_queue.clear()
            delayed_queue.clear()
            active_registry.clear()
            host_advice_tracker.clear()
            seen_urls.replace_entries(())
            assert plan.progress is not None
            progress_state.apply_restore_state(plan.progress)

        for item in plan.items:
            commit_restored_envelope(
                envelope=item.envelope,
                delayed_wait_seconds=item.delayed_wait_seconds,
            )

        if clear_existing and plan.seen_entries is not None:
            seen_urls.replace_entries(plan.seen_entries)

        total_pending_by_host = DelayedTaskQueue.combine_pending_maps(
            host_queue.pending_count_by_host(),
            delayed_queue.pending_count_by_host(),
        )
        restored_count = len(plan.items)
        return SchedulerRestoreResult(
            restored_count=restored_count,
            restored_queued_count=plan.restored_queued_count,
            restored_delayed_count=plan.restored_delayed_count,
            restored_requeued_inflight_count=(
                plan.restored_requeued_inflight_count
            ),
            restored_dispatching_count=plan.restored_dispatching_count,
            skipped_tasks=0,
            next_sequence_value=plan.next_sequence_value,
            queued=host_queue.queue_size + delayed_queue.queue_size,
            pending_hosts=len(total_pending_by_host),
        )

    def _deserialize_checkpoint_item(
        self,
        *,
        item: object,
    ) -> SchedulerTaskEnvelope:
        if not isinstance(item, dict):
            raise ValueError("checkpoint task item must be a dict")
        envelope = self._deserializer.deserialize(item=item)
        if not isinstance(envelope, SchedulerTaskEnvelope):
            raise ValueError("failed to deserialize checkpoint task item")
        return envelope

    @staticmethod
    def _parse_delayed_wait_seconds(item: object) -> float:
        if not isinstance(item, dict):
            raise ValueError("checkpoint delayed task item must be a dict")
        raw_wait = item.get("delay_remaining_seconds")
        if isinstance(raw_wait, bool) or not isinstance(
            raw_wait, (int, float)
        ):
            raise ValueError(
                "checkpoint delayed task has invalid delay_remaining_seconds"
            )
        wait = float(raw_wait)
        if not isfinite(wait) or wait < 0:
            raise ValueError(
                "checkpoint delayed task has invalid delay_remaining_seconds"
            )
        return wait

    @staticmethod
    def _parse_next_sequence(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "scheduler checkpoint next_sequence must be a non-negative integer"
            )
        return value

    @staticmethod
    def _parse_seen_url_entries(
        value: object,
    ) -> tuple[tuple[str, float], ...] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(
                "checkpoint field seen_url_entries must be a list"
            )

        entries: list[tuple[str, float]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("checkpoint contains invalid seen URL entry")
            raw_url = item.get("url")
            raw_seen_at = item.get("seen_at")
            if not isinstance(raw_url, str) or not raw_url.strip():
                raise ValueError("checkpoint contains invalid seen URL entry")
            if isinstance(raw_seen_at, bool) or not isinstance(
                raw_seen_at, (int, float)
            ):
                raise ValueError(
                    "checkpoint contains invalid seen URL timestamp"
                )
            seen_at = float(raw_seen_at)
            if not isfinite(seen_at):
                raise ValueError(
                    "checkpoint contains invalid seen URL timestamp"
                )
            url = raw_url.strip()
            entries.append((url, seen_at))

        # ``SeenUrlRegistry.replace_entries`` is the single owner of restore
        # semantics: it consumes every entry, keeps the last occurrence of a
        # duplicate, bounds the final registry to its capacity, and purges TTL
        # expiry over the complete restored state. Rejecting here would bypass
        # that canonical behavior before it can run.
        return tuple(entries)
