"""Manifest model for augmentation workflow output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datachecker.manifests.artifact_manifest import ArtifactManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

AUGMENTED_LIFECYCLE_STAGE = "augmented"


@dataclass(frozen=True, slots=True)
class AugmentationManifest(ArtifactManifest):
    """Persisted proof that augmentation is current for preprocessing input."""

    preprocessing_manifest_hash: str
    augmentation_settings_hash: str
    augmentation_strategy_hash: str
    output_fingerprint: str

    training_snapshot_directory: Path | None
    augmented_training_directory: Path | None
    augmented_dataset_manifest_path: Path | None

    input_chunk_count: int
    input_sample_count: int
    augmented_sample_count: int
    rejected_augmented_count: int

    variants_by_modality: dict[str, int]
    variants_by_operation: dict[str, int]
    media_outputs: dict[str, object]
    rejections_by_modality: dict[str, int]

    quality_checks_passed: bool
    built_at: str | None

    lifecycle_stage: str = AUGMENTED_LIFECYCLE_STAGE
    status: WorkflowLifecycleStatus = WorkflowLifecycleStatus.COMPLETED
    final: bool = True

    def __post_init__(self) -> None:
        """Validate the augmentation manifest contract."""

        ArtifactManifest.__post_init__(self)

        self._validate_required_text_fields()
        self._validate_counts()
        self._validate_count_mappings()
        self._validate_media_outputs()
        self._validate_lifecycle()
        self._validate_completed_paths()

        if not isinstance(self.quality_checks_passed, bool):
            raise ValueError("quality_checks_passed must be a boolean")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> AugmentationManifest:
        """Build an augmentation manifest from serialized payload data."""

        return cls(
            **cls.identity_from_payload(payload),
            preprocessing_manifest_hash=cls.as_required_str(
                payload.get("preprocessing_manifest_hash"),
                field="preprocessing_manifest_hash",
            ),
            augmentation_settings_hash=cls.as_required_str(
                payload.get("augmentation_settings_hash"),
                field="augmentation_settings_hash",
            ),
            augmentation_strategy_hash=cls.as_required_str(
                payload.get("augmentation_strategy_hash"),
                field="augmentation_strategy_hash",
            ),
            output_fingerprint=cls.as_required_str(
                payload.get("output_fingerprint"),
                field="output_fingerprint",
            ),
            training_snapshot_directory=cls._optional_payload_path(
                payload,
                "training_snapshot_directory",
            ),
            augmented_training_directory=cls._optional_payload_path(
                payload,
                "augmented_training_directory",
            ),
            augmented_dataset_manifest_path=cls._optional_payload_path(
                payload,
                "augmented_dataset_manifest_path",
            ),
            input_chunk_count=cls._parse_required_count(
                payload.get("input_chunk_count"),
                field_name="input_chunk_count",
            ),
            input_sample_count=cls._parse_required_count(
                payload.get("input_sample_count"),
                field_name="input_sample_count",
            ),
            augmented_sample_count=cls._parse_required_count(
                payload.get("augmented_sample_count"),
                field_name="augmented_sample_count",
            ),
            rejected_augmented_count=cls._parse_required_count(
                payload.get("rejected_augmented_count"),
                field_name="rejected_augmented_count",
            ),
            variants_by_modality=cls._parse_count_mapping(
                payload.get("variants_by_modality"),
                field_name="variants_by_modality",
            ),
            variants_by_operation=cls._parse_count_mapping(
                payload.get("variants_by_operation"),
                field_name="variants_by_operation",
            ),
            media_outputs=cls._parse_media_outputs(
                payload.get("media_outputs"),
            ),
            rejections_by_modality=cls._parse_count_mapping(
                payload.get("rejections_by_modality"),
                field_name="rejections_by_modality",
            ),
            quality_checks_passed=cls.as_bool(
                payload.get("quality_checks_passed")
            ),
            built_at=cls.as_opt_str(payload.get("built_at")),
            lifecycle_stage=cls.as_required_str(
                payload.get(
                    "lifecycle_stage",
                    AUGMENTED_LIFECYCLE_STAGE,
                ),
                field="lifecycle_stage",
            ),
            status=WorkflowLifecycleStatus.parse(payload.get("status")),
            final=cls.as_bool(
                payload.get("final"),
                default=True,
            ),
        )

    def _validate_required_text_fields(self) -> None:
        """Validate required non-empty manifest fingerprint fields."""

        for field_name, value in (
            (
                "preprocessing_manifest_hash",
                self.preprocessing_manifest_hash,
            ),
            (
                "augmentation_settings_hash",
                self.augmentation_settings_hash,
            ),
            (
                "augmentation_strategy_hash",
                self.augmentation_strategy_hash,
            ),
            (
                "output_fingerprint",
                self.output_fingerprint,
            ),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def _validate_counts(self) -> None:
        """Validate direct-construction scalar counts."""

        for field_name, value in (
            (
                "input_chunk_count",
                self.input_chunk_count,
            ),
            (
                "input_sample_count",
                self.input_sample_count,
            ),
            (
                "augmented_sample_count",
                self.augmented_sample_count,
            ),
            (
                "rejected_augmented_count",
                self.rejected_augmented_count,
            ),
        ):
            self._validate_nonnegative_int(
                value,
                field_name=field_name,
            )

    def _validate_count_mappings(self) -> None:
        """Validate all persisted count mappings."""

        for field_name, mapping in (
            (
                "variants_by_modality",
                self.variants_by_modality,
            ),
            (
                "variants_by_operation",
                self.variants_by_operation,
            ),
            (
                "rejections_by_modality",
                self.rejections_by_modality,
            ),
        ):
            if not isinstance(mapping, dict):
                raise ValueError(f"{field_name} must be a mapping")

            for key, value in mapping.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"{field_name} keys must be non-empty strings"
                    )

                self._validate_nonnegative_int(
                    value,
                    field_name=f"{field_name}[{key}]",
                )

    def _validate_media_outputs(self) -> None:
        """Validate media output keys and recursively supported values."""

        if not isinstance(self.media_outputs, dict):
            raise ValueError("media_outputs must be a mapping")

        for key, value in self.media_outputs.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "media_outputs keys must be non-empty strings"
                )

            self._validate_media_output_value(
                value,
                field_name=f"media_outputs[{key}]",
            )

    def _validate_lifecycle(self) -> None:
        """Validate lifecycle state for a completed augmentation manifest."""

        if self.lifecycle_stage != AUGMENTED_LIFECYCLE_STAGE:
            raise ValueError(
                "augmentation manifest lifecycle_stage must be augmented"
            )

        if self.status is not WorkflowLifecycleStatus.COMPLETED:
            raise ValueError("augmentation manifest status must be completed")

        if not self.final:
            raise ValueError("completed augmentation manifest must be final")

    def _validate_completed_paths(self) -> None:
        """Require augmentation output paths for completed manifests."""

        missing: list[str] = []

        if self.training_snapshot_directory is None:
            missing.append("training_snapshot_directory")

        if self.augmented_training_directory is None:
            missing.append("augmented_training_directory")

        if self.augmented_dataset_manifest_path is None:
            missing.append("augmented_dataset_manifest_path")

        if missing:
            raise ValueError(
                "completed augmentation manifest requires paths: "
                + ", ".join(missing)
            )

    @staticmethod
    def _validate_nonnegative_int(
        value: object,
        *,
        field_name: str,
    ) -> None:
        """Validate one non-negative integer without accepting booleans."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")

        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")

    @classmethod
    def _parse_required_count(
        cls,
        value: object,
        *,
        field_name: str,
    ) -> int:
        """Parse one required non-negative integer count."""

        cls._validate_nonnegative_int(
            value,
            field_name=field_name,
        )

        if not isinstance(value, int) or isinstance(value, bool):
            raise AssertionError("validated count did not narrow to int")

        return value

    @classmethod
    def _parse_count_mapping(
        cls,
        value: object,
        *,
        field_name: str,
    ) -> dict[str, int]:
        """Parse a string-keyed mapping of non-negative integer counts."""

        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a mapping")

        result: dict[str, int] = {}

        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"{field_name} keys must be non-empty strings"
                )

            result[key] = cls._parse_required_count(
                item,
                field_name=f"{field_name}[{key}]",
            )

        return result

    @classmethod
    def _parse_media_outputs(
        cls,
        value: object,
    ) -> dict[str, object]:
        """Parse and validate persisted media-output metadata."""

        if value is None:
            return {}

        if not isinstance(value, dict):
            raise ValueError("media_outputs must be a mapping")

        result: dict[str, object] = {}

        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "media_outputs keys must be non-empty strings"
                )

            cls._validate_media_output_value(
                item,
                field_name=f"media_outputs[{key}]",
            )

            result[key] = item

        return result

    @classmethod
    def _validate_media_output_value(
        cls,
        value: object,
        *,
        field_name: str,
    ) -> None:
        """Validate one recursively JSON-compatible media-output value."""

        if value is None:
            return

        if isinstance(value, (str, bool)):
            return

        if isinstance(value, int):
            return

        if isinstance(value, float):
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                cls._validate_media_output_value(
                    item,
                    field_name=f"{field_name}[{index}]",
                )

            return

        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{field_name} mapping keys must be strings"
                    )

                cls._validate_media_output_value(
                    item,
                    field_name=f"{field_name}[{key}]",
                )

            return

        raise ValueError(
            f"{field_name} contains unsupported value type "
            f"{type(value).__name__}"
        )

    @classmethod
    def _optional_payload_path(
        cls,
        payload: dict[str, object],
        field_name: str,
    ) -> Path | None:
        """Parse an optional manifest path while treating '.' as absent."""

        text = cls.as_opt_str(payload.get(field_name))

        if text is None:
            return None

        path = Path(text)

        if path == Path("."):
            return None

        return path
