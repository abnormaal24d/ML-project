"""Manifest model for raw crawl workflow output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from datachecker.manifests.artifact_manifest import ArtifactManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

CRAWL_LIFECYCLE_STAGE = "raw"


def _parse_non_negative_count(
    value: object,
    *,
    field_name: str,
) -> int:
    """Parse and validate a required non-negative integer count."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")

    parsed = ArtifactManifest.as_opt_int(value)

    if parsed is None:
        raise ValueError(f"{field_name} must be an integer")

    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {parsed}")

    return parsed


def _as_string_keyed_mapping(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> dict[str, object]:
    """Validate and return a mapping containing string keys only."""

    if value is None:
        if allow_none:
            return {}

        raise ValueError(f"{field_name} must be a mapping")

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")

    result: dict[str, object] = {}

    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")

        result[key] = item

    return result


def _as_string_list(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> list[str]:
    """Validate and return a list containing strings only."""

    if value is None:
        if allow_none:
            return []

        raise ValueError(f"{field_name} must be a list")

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    result: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")

        result.append(item)

    return result


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Structured crawl coverage summary."""

    modality_counts: dict[str, int] = field(default_factory=dict)
    minimum_modality_counts: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    pipeline_counters: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object] | None,
    ) -> CoverageSummary | None:
        """Build a coverage summary from a serialized mapping."""

        if not data:
            return None

        modality_counts = _as_string_keyed_mapping(
            data.get("modality_counts"),
            field_name="coverage_summary.modality_counts",
        )

        minimum_modality_counts = _as_string_keyed_mapping(
            data.get("minimum_modality_counts"),
            field_name="coverage_summary.minimum_modality_counts",
            allow_none=True,
        )

        missing = _as_string_keyed_mapping(
            data.get("missing"),
            field_name="coverage_summary.missing",
        )

        pipeline_counters = _as_string_keyed_mapping(
            data.get("pipeline_counters"),
            field_name="coverage_summary.pipeline_counters",
            allow_none=True,
        )

        return cls(
            modality_counts={
                key: _parse_non_negative_count(
                    value,
                    field_name=f"modality_counts[{key}]",
                )
                for key, value in modality_counts.items()
            },
            minimum_modality_counts={
                key: _parse_non_negative_count(
                    value,
                    field_name=f"minimum_modality_counts[{key}]",
                )
                for key, value in minimum_modality_counts.items()
            },
            missing={
                key: _parse_non_negative_count(
                    value,
                    field_name=f"missing[{key}]",
                )
                for key, value in missing.items()
            },
            pipeline_counters=pipeline_counters,
        )


@dataclass(frozen=True, slots=True)
class SchemaValidationSummary:
    """Summary of raw crawl schema validation."""

    raw_schema_valid: bool = False
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object] | None,
    ) -> SchemaValidationSummary | None:
        """Build a schema-validation summary from serialized data."""

        if not data:
            return None

        reasons = _as_string_list(
            data.get("reasons"),
            field_name="schema_validation_summary.reasons",
            allow_none=True,
        )

        return cls(
            raw_schema_valid=ArtifactManifest.as_bool(
                data.get("raw_schema_valid")
            ),
            reasons=reasons,
        )


@dataclass(frozen=True, slots=True)
class CrawlManifest(ArtifactManifest):
    """Persisted proof that the crawl phase completed for current inputs."""

    source_registry_hash: str
    crawl_settings_hash: str
    output_fingerprint: str

    raw_run_directory: Path | None
    raw_records_manifest_path: Path | None
    run_summary_path: Path | None

    fetched_url_count: int
    failed_url_count: int
    output_file_count: int

    started_at: str | None
    completed_at: str | None

    lifecycle_stage: str = CRAWL_LIFECYCLE_STAGE
    status: WorkflowLifecycleStatus = WorkflowLifecycleStatus.COMPLETED
    final: bool = True

    raw_errors_manifest_path: Path | None = None
    records_manifest_hash: str | None = None
    raw_object_records_total: int | None = None

    coverage_summary: CoverageSummary | None = None
    schema_validation_summary: SchemaValidationSummary | None = None

    def __post_init__(self) -> None:
        """Validate the persisted crawl-manifest contract."""

        ArtifactManifest.__post_init__(self)

        for field_name, value in (
            ("source_registry_hash", self.source_registry_hash),
            ("crawl_settings_hash", self.crawl_settings_hash),
            ("output_fingerprint", self.output_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.lifecycle_stage != CRAWL_LIFECYCLE_STAGE:
            raise ValueError("crawl manifest lifecycle_stage must be raw")

        if self.status is not WorkflowLifecycleStatus.COMPLETED:
            raise ValueError("crawl manifest status must be completed")

        if not self.final:
            raise ValueError("crawl manifest must be final")

        for field_name, count in (
            ("fetched_url_count", self.fetched_url_count),
            ("failed_url_count", self.failed_url_count),
            ("output_file_count", self.output_file_count),
        ):
            self._validate_count(
                count,
                field_name=field_name,
            )

        if self.raw_object_records_total is not None:
            self._validate_count(
                self.raw_object_records_total,
                field_name="raw_object_records_total",
            )

        for field_name, timestamp in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if timestamp is not None and not self._looks_like_iso(timestamp):
                raise ValueError(
                    f"{field_name} must be an ISO 8601 timestamp "
                    f"or None, got {timestamp!r}"
                )

    @staticmethod
    def _validate_count(
        value: object,
        *,
        field_name: str,
    ) -> None:
        """Validate a direct-construction count value."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")

        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")

    @staticmethod
    def _looks_like_iso(value: str) -> bool:
        """Return whether a string can be parsed as ISO 8601."""

        text = value.strip()

        if not text:
            return False

        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"

        try:
            datetime.fromisoformat(text)
        except ValueError:
            return False

        return True

    @staticmethod
    def _as_optional_dict(
        value: object,
    ) -> dict[str, object] | None:
        """Return a validated string-keyed mapping or None."""

        if value is None:
            return None

        return _as_string_keyed_mapping(
            value,
            field_name="manifest summary value",
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> CrawlManifest:
        """Build a crawl manifest from a serialized payload."""

        coverage_summary = CoverageSummary.from_dict(
            cls._as_optional_dict(payload.get("coverage_summary"))
        )

        schema_validation_summary = SchemaValidationSummary.from_dict(
            cls._as_optional_dict(payload.get("schema_validation_summary"))
        )

        return cls(
            **cls.identity_from_payload(payload),
            source_registry_hash=cls.as_required_str(
                payload.get("source_registry_hash"),
                field="source_registry_hash",
            ),
            crawl_settings_hash=cls.as_required_str(
                payload.get("crawl_settings_hash"),
                field="crawl_settings_hash",
            ),
            output_fingerprint=cls.as_required_str(
                payload.get("output_fingerprint"),
                field="output_fingerprint",
            ),
            raw_run_directory=cls._optional_payload_path(
                payload,
                "raw_run_directory",
            ),
            raw_records_manifest_path=cls._optional_payload_path(
                payload,
                "raw_records_manifest_path",
            ),
            run_summary_path=cls._optional_payload_path(
                payload,
                "run_summary_path",
            ),
            fetched_url_count=_parse_non_negative_count(
                payload.get("fetched_url_count"),
                field_name="fetched_url_count",
            ),
            failed_url_count=_parse_non_negative_count(
                payload.get("failed_url_count"),
                field_name="failed_url_count",
            ),
            output_file_count=_parse_non_negative_count(
                payload.get("output_file_count"),
                field_name="output_file_count",
            ),
            started_at=cls.as_opt_str(payload.get("started_at")),
            completed_at=cls.as_opt_str(payload.get("completed_at")),
            lifecycle_stage=cls.as_required_str(
                payload.get(
                    "lifecycle_stage",
                    CRAWL_LIFECYCLE_STAGE,
                ),
                field="lifecycle_stage",
            ),
            status=WorkflowLifecycleStatus.parse(payload.get("status")),
            final=cls.as_bool(
                payload.get("final"),
                default=True,
            ),
            raw_errors_manifest_path=cls.as_opt_path(
                payload.get("raw_errors_manifest_path")
            ),
            records_manifest_hash=cls.as_opt_str(
                payload.get("records_manifest_hash")
            ),
            raw_object_records_total=cls.as_opt_int(
                payload.get("raw_object_records_total")
            ),
            coverage_summary=coverage_summary,
            schema_validation_summary=schema_validation_summary,
        )

    @classmethod
    def _optional_payload_path(
        cls,
        payload: dict[str, object],
        field_name: str,
    ) -> Path | None:
        """Parse an optional payload path while rejecting '.'."""

        text = cls.as_opt_str(payload.get(field_name))

        if text is None:
            return None

        path = Path(text)

        if path == Path("."):
            return None

        return path
