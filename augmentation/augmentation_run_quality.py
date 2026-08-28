"""Run-level integrity checks for accepted augmentation variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    file_sha256,
    media_variant_id,
    valid_sha256,
)
from mmcrawler_datasets.schema import MultimodalSample

_REQUIRED_MEDIA_LINEAGE_FIELDS = (
    "augmentation_source_sha256",
    "augmentation_output_sha256",
    "augmentation_config_hash",
    "augmentation_implementation_hash",
    "augmentation_implementation_version",
    "augmentation_output_path",
    "augmentation_source_path",
)


@dataclass(frozen=True, slots=True)
class AugmentationQualityAssessment:
    """Integrity result for all accepted generated media variants."""

    passed: bool
    media_outputs: dict[str, object]
    checks: dict[str, int]
    failures: tuple[str, ...]


def assess_augmentation_run_quality(
    *,
    dataset: tuple[MultimodalSample, ...],
    dataset_root: str | Path | None,
) -> AugmentationQualityAssessment:
    """Validate files, hashes, metadata, and deterministic media identities."""

    stats: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    checks = {
        "accepted_media_variants": 0,
        "generated_media_files_checked": 0,
        "missing_generated_files": 0,
        "invalid_output_hashes": 0,
        "invalid_source_hashes": 0,
        "missing_lineage_fields": 0,
        "invalid_variant_ids": 0,
        "invalid_implementation_bindings": 0,
    }
    root = _root_path(dataset_root)

    for sample in dataset:
        metadata = sample.metadata or {}
        if not metadata.get("augmentation_media_transform_applied"):
            continue

        checks["accepted_media_variants"] += 1
        modality = str(metadata.get("output_modality") or sample.modality)
        bucket = stats.setdefault(
            modality,
            {
                "generated_files": 0,
                "generated_media_files_checked": 0,
                "missing_generated_files": 0,
                "invalid_output_hashes": 0,
                "invalid_source_hashes": 0,
                "missing_lineage_fields": 0,
                "invalid_variant_ids": 0,
                "invalid_implementation_bindings": 0,
            },
        )
        bucket["generated_files"] += 1

        missing = [
            field
            for field in _REQUIRED_MEDIA_LINEAGE_FIELDS
            if not metadata.get(field)
        ]
        if missing:
            checks["missing_lineage_fields"] += 1
            bucket["missing_lineage_fields"] += 1
            failures.append(f"{sample.sample_id}:missing:{','.join(missing)}")
            continue

        source_hash = metadata["augmentation_source_sha256"]
        output_hash = metadata["augmentation_output_sha256"]
        config_hash = metadata["augmentation_config_hash"]
        implementation_hash = metadata["augmentation_implementation_hash"]
        operation = metadata.get("augmentation_name")
        source_sample_id = metadata.get("augmentation_source_sample_id")

        if implementation_hash != MEDIA_AUGMENTATION_IMPLEMENTATION_HASH:
            checks["invalid_implementation_bindings"] += 1
            bucket["invalid_implementation_bindings"] += 1
            failures.append(f"{sample.sample_id}:invalid_implementation_hash")

        output_path = _resolve_path(
            value=metadata["augmentation_output_path"], root=root
        )
        source_path = _resolve_path(
            value=metadata["augmentation_source_path"], root=root
        )

        if (
            output_path is None
            or not output_path.is_file()
            or output_path.stat().st_size <= 0
        ):
            checks["missing_generated_files"] += 1
            bucket["missing_generated_files"] += 1
            failures.append(f"{sample.sample_id}:missing_generated_file")
        else:
            checks["generated_media_files_checked"] += 1
            bucket["generated_media_files_checked"] += 1
            actual_output_hash = file_sha256(path=output_path)
            if (
                not valid_sha256(output_hash)
                or actual_output_hash != output_hash
            ):
                checks["invalid_output_hashes"] += 1
                bucket["invalid_output_hashes"] += 1
                failures.append(f"{sample.sample_id}:output_hash_mismatch")

        if source_path is None or not source_path.is_file():
            checks["invalid_source_hashes"] += 1
            bucket["invalid_source_hashes"] += 1
            failures.append(f"{sample.sample_id}:source_file_missing")
        else:
            actual_source_hash = file_sha256(path=source_path)
            if (
                not valid_sha256(source_hash)
                or actual_source_hash != source_hash
            ):
                checks["invalid_source_hashes"] += 1
                bucket["invalid_source_hashes"] += 1
                failures.append(f"{sample.sample_id}:source_hash_mismatch")

        if not all(
            isinstance(value, str) and value
            for value in (config_hash, operation, source_sample_id)
        ) or not valid_sha256(config_hash):
            checks["invalid_variant_ids"] += 1
            bucket["invalid_variant_ids"] += 1
            failures.append(f"{sample.sample_id}:invalid_identity_inputs")
            continue

        expected_id = media_variant_id(
            source_sample_id=cast(str, source_sample_id),
            operation=cast(str, operation),
            source_sha256=cast(str, source_hash),
            config_hash=cast(str, config_hash),
            prefix=_variant_prefix(sample_id=sample.sample_id),
        )
        if (
            sample.sample_id != expected_id
            or metadata.get("augmentation_variant_id") != expected_id
        ):
            checks["invalid_variant_ids"] += 1
            bucket["invalid_variant_ids"] += 1
            failures.append(f"{sample.sample_id}:variant_id_mismatch")

    return AugmentationQualityAssessment(
        passed=not failures,
        media_outputs=cast(dict[str, object], stats),
        checks=checks,
        failures=tuple(failures),
    )


def _root_path(dataset_root: str | Path | None) -> Path | None:
    if dataset_root is None:
        return None
    if isinstance(dataset_root, str) and not dataset_root.strip():
        return None
    return Path(dataset_root)


def _resolve_path(*, value: object, root: Path | None) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if root is None:
        return None
    candidate = (root / path).resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        return None
    return candidate


def _variant_prefix(*, sample_id: str) -> str:
    return (
        "sample_doc_aug"
        if sample_id.startswith("sample_doc_aug_")
        else "sample_media_aug"
    )
