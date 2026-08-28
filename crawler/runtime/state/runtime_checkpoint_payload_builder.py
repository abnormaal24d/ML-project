"""Runtime checkpoint payload construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from crawler.numeric import coerce_finite_float
from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot
from shared.runtime_primitives import Clock


class RuntimeCheckpointPayloadBuilder:
    """Build JSON-safe crawler runtime checkpoint payloads."""

    SCHEMA_VERSION = 1

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock

    def build(
        self,
        *,
        final: bool,
        scheduler_state: dict[str, object],
        worker_snapshot: WorkerPoolSnapshot,
        metrics: Any | None,
        run_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return the JSON-safe runtime checkpoint payload."""

        payload: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "saved_at": self._clock.now().isoformat(),
            "final": bool(final),
        }

        if run_context:
            payload["run_context"] = self._json_safe(run_context)

        metrics_payload = self._build_metrics_payload(metrics=metrics)
        if metrics_payload is not None:
            payload["metrics"] = metrics_payload

        payload["scheduler"] = self._json_safe(scheduler_state)
        payload["worker_pool"] = self._build_worker_pool_payload(
            worker_snapshot=worker_snapshot,
        )

        return payload

    @classmethod
    def _build_worker_pool_payload(
        cls,
        *,
        worker_snapshot: WorkerPoolSnapshot,
    ) -> dict[str, object]:
        """Return the serialized worker-pool snapshot for checkpoints."""

        return {
            "workers": cls._safe_int(worker_snapshot.size),
            "busy_workers": cls._safe_int(worker_snapshot.busy_worker_count),
            "completed_tasks": cls._safe_int(
                worker_snapshot.completed_task_count
            ),
            "failure_count": cls._safe_int(worker_snapshot.failure_count),
            "average_processing_seconds": cls._safe_float(
                worker_snapshot.average_processing_seconds
            ),
        }

    @classmethod
    def _build_metrics_payload(
        cls,
        *,
        metrics: Any | None,
    ) -> dict[str, object] | None:
        """Return the serialized metrics snapshot for checkpoints."""

        if metrics is None:
            return None

        if not bool(getattr(metrics, "enabled", False)):
            return None

        snapshot = metrics.snapshot(host_limit=0)

        return {
            "requests_total": cls._safe_int(
                getattr(snapshot, "requests_total", 0)
            ),
            "successes_total": cls._safe_int(
                getattr(snapshot, "successes_total", 0)
            ),
            "failures_total": cls._safe_int(
                getattr(snapshot, "failures_total", 0)
            ),
            "skipped_total": cls._safe_int(
                getattr(snapshot, "skipped_total", 0)
            ),
            "bytes_total": cls._safe_int(getattr(snapshot, "bytes_total", 0)),
            "average_latency_seconds": cls._safe_float(
                getattr(snapshot, "average_latency_seconds", 0.0)
            ),
            "quality_score": cls._safe_float(
                getattr(snapshot, "quality_score", 0.0)
            ),
            "blacklist_total": cls._safe_int(
                getattr(snapshot, "blacklist_total", 0)
            ),
            "blacklist_by_stage": cls._stable_mapping(
                getattr(snapshot, "blacklist_by_stage", {})
            ),
            "blacklist_by_reason": cls._stable_mapping(
                getattr(snapshot, "blacklist_by_reason", {})
            ),
            "skipped_by_reason": cls._stable_mapping(
                getattr(snapshot, "skipped_by_reason", {})
            ),
        }

    @classmethod
    def _stable_mapping(cls, value: Any) -> dict[str, object]:
        """
        Return a stable JSON-safe mapping.

        Accepts:
        - dict / Mapping
        - Counter-like objects
        - iterable key/value pairs
        - tuple/list key/value pairs
        """

        if value is None:
            return {}

        if isinstance(value, Mapping):
            items = value.items()
        else:
            try:
                items = dict(value).items()
            except (TypeError, ValueError):
                return {
                    "value": cls._json_safe(value),
                }

        normalized: dict[str, object] = {}

        for key, item_value in items:
            normalized[str(key)] = cls._json_safe(item_value)

        return {key: normalized[key] for key in sorted(normalized)}

    @classmethod
    def _json_safe(cls, value: Any) -> object:
        """Convert common Python objects into JSON-safe values."""

        if value is None:
            return None

        if isinstance(value, float):
            return coerce_finite_float(value, default=0.0)

        if isinstance(value, str | int | bool):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(item_value)
                for key, item_value in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }

        if isinstance(value, tuple | list | set | frozenset):
            return [cls._json_safe(item) for item in value]

        if hasattr(value, "value"):
            return cls._json_safe(value.value)

        if hasattr(value, "isoformat"):
            return value.isoformat()

        return str(value)

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Return value as int, falling back to zero."""

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Return value as float, falling back to zero."""

        return coerce_finite_float(value, default=0.0)
