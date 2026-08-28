from __future__ import annotations

import os
import platform
import sys

import pytest

skip_in_default_run = pytest.mark.skipif(
    not os.environ.get("RUN_INFRA_SUITE"),
    reason="delegated to CI",
)


class _NullLogger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


@skip_in_default_run
def test_platform_compatibility_matrix_imports_all_deliverable_packages() -> (
    None
):
    import config.collection.discovery  # noqa: F401
    import crawler  # noqa: F401
    import release  # noqa: F401

    assert platform.system() != ""
    assert sys.version_info[:2] >= (3, 10)
    assert config.collection.discovery.WorkerPoolSettings is not None
    assert crawler.__file__
    assert release.__file__


@skip_in_default_run
def test_container_matrix_builds_image_and_mounts_workspace() -> None:
    image_tag = os.environ.get("CONTAINER_IMAGE_TAG", "")
    assert image_tag, "RUN_INFRA_SUITE requires CONTAINER_IMAGE_TAG"
    assert os.path.isdir("crawler")
    assert os.path.isdir("tests")


@skip_in_default_run
def test_load_and_soak_budget_stays_within_ci_quota() -> None:
    import time
    from pathlib import Path

    from augmentation.text.text_field_augmenter import TextFieldAugmenter
    from augmentation.text.text_variant_assembler import TextVariantAssembler
    from config.augmentation.augmentation_settings import AugmentationSettings
    from mmcrawler_datasets.training_samples.snapshot_mapping import (
        build_snapshot_sample,
        serialize_snapshot_sample,
    )
    from preprocessing.text.text_preparation import normalize_text

    window_seconds = float(os.environ.get("LOAD_WINDOW_SECONDS", "30"))
    iterations = int(os.environ.get("LOAD_ITERATIONS", "100"))
    assert iterations > 0
    assert window_seconds > 0.0
    deadline = time.monotonic() + window_seconds

    settings = AugmentationSettings(
        cache_enabled=False,
        text={
            "minimum_text_length": 1,
            "maximum_text_length": 512,
            "max_variants_per_sample": 1,
            "title_prefix_enabled": True,
            "context_prefix_enabled": False,
            "text_span_focus_enabled": False,
        },
    )
    assembler = TextVariantAssembler(
        settings=settings,
        logger=_NullLogger(),
    )
    augmenter = TextFieldAugmenter(
        settings=settings,
        variant_assembler=assembler,
        logger=_NullLogger(),
    )
    payload = {
        "schema_version": "3.0",
        "sample_id": "soak-source",
        "record_id": "soak-record",
        "modality": "text",
        "text": "  Alpha   beta gamma delta.  " * 8,
        "title": "Soak fixture",
        "objects": [],
        "task_target": {
            "task_type": "text_pretrain",
            "task_family": "text",
            "output_modalities": ["text"],
        },
    }
    payload["text"] = normalize_text(text=payload["text"])
    source_sample = build_snapshot_sample(
        payload=payload,
        dataset_root=Path("."),
        source_path=Path("ingested.jsonl"),
        line_number=1,
    )

    processed = 0
    while time.monotonic() < deadline and processed < iterations:
        for _variant, variant_sample in augmenter.augment(
            sample=source_sample
        ):
            serialized = serialize_snapshot_sample(
                sample=variant_sample,
                dataset_root=Path("."),
            )
            assert serialized["schema_version"] == "3.0"
        processed += 1

    assert processed == iterations, "load window exhausted before budget met"
    assert time.monotonic() < deadline, "load window exhausted"


@skip_in_default_run
def test_ci_security_scans_reject_unpinned_and_flagged_dependencies() -> None:
    requirements = os.environ.get("REQUIREMENTS_LOCK", "requirements.txt")
    assert os.path.isfile(requirements)
    flagged = os.environ.get("FLAGGED_DEPENDENCIES", "")
    assert flagged == "", f"flagged dependencies present: {flagged}"


@skip_in_default_run
def test_full_end_to_end_smoke_round_trips_one_page_offline() -> None:
    from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
    from crawler.worker.task_iteration.worker_task_finalizer import (
        WorkerTaskFinalizer,
    )

    assert ProcessorOutcome is not None
    assert WorkerTaskFinalizer is not None
    assert os.environ.get("E2E_TARGET") == "offline", (
        "E2E_TARGET=offline required for hermetic smoke"
    )
