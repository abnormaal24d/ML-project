from __future__ import annotations

RAW_DATASET_SCHEMA_VERSION = "3.0"
CURATED_DATASET_SCHEMA_VERSION = "3.0"
TRAINING_DATASET_SCHEMA_VERSION = "3.0"
WORKFLOW_MANIFEST_SCHEMA_VERSION = "2.0"

SUPPORTED_RAW_DATASET_SCHEMA_VERSIONS = frozenset({RAW_DATASET_SCHEMA_VERSION})
SUPPORTED_CURATED_DATASET_SCHEMA_VERSIONS = frozenset(
    {CURATED_DATASET_SCHEMA_VERSION}
)
SUPPORTED_TRAINING_DATASET_SCHEMA_VERSIONS = frozenset(
    {TRAINING_DATASET_SCHEMA_VERSION}
)
SUPPORTED_WORKFLOW_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {WORKFLOW_MANIFEST_SCHEMA_VERSION}
)


def _is_supported(
    schema_version: str | None,
    supported: frozenset[str],
) -> bool:
    return bool(schema_version and schema_version.strip() in supported)


def is_supported_raw_schema_version(schema_version: str | None) -> bool:
    return _is_supported(schema_version, SUPPORTED_RAW_DATASET_SCHEMA_VERSIONS)


def is_supported_training_schema_version(
    schema_version: str | None,
) -> bool:
    return _is_supported(
        schema_version, SUPPORTED_TRAINING_DATASET_SCHEMA_VERSIONS
    )


def is_supported_workflow_manifest_schema_version(
    schema_version: str | None,
) -> bool:
    return _is_supported(
        schema_version, SUPPORTED_WORKFLOW_MANIFEST_SCHEMA_VERSIONS
    )


def schema_version_error(
    *,
    artifact: str,
    from_version: str | None,
    supported: frozenset[str],
) -> str:
    observed = from_version or "missing"
    expected = ", ".join(sorted(supported))
    return (
        f"Unsupported {artifact} schema version {observed!r}; "
        f"supported version(s): {expected}."
    )


def training_schema_error(from_version: str | None) -> str:
    return schema_version_error(
        artifact="training dataset",
        from_version=from_version,
        supported=SUPPORTED_TRAINING_DATASET_SCHEMA_VERSIONS,
    )
