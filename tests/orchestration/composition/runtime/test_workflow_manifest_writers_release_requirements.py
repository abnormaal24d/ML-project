from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime
from types import SimpleNamespace

from orchestration.composition.runtime import (
    workflow_manifest_writers as module,
)


def test_crawl_state_manifest_writer_has_canonical_manifest_names() -> None:
    """The aggregate and committer must distinguish manifest state."""

    aggregate_fields = module.WorkflowManifestWriters.__dataclass_fields__
    committer_parameters = inspect.signature(
        module.CrawlPromotionCommitter
    ).parameters

    assert "crawl_state_manifest_writer" in aggregate_fields
    assert "crawl_state_manifest_writer" in committer_parameters


def test_builder_does_not_forward_release_policy_to_manifest_assembler() -> (
    None
):
    """Manifest construction must not own release-policy decisions."""

    source = textwrap.dedent(
        inspect.getsource(module.build_workflow_manifest_writers)
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_assemble_manifest_writers"
    ]

    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert "release_requirements" not in keyword_names
    assert "dataset_validator_settings" not in keyword_names


def test_training_manifest_writer_receives_results_not_release_policy(
    monkeypatch,
) -> None:
    """The manifest writer is passive and receives no policy services."""

    captured: dict[str, dict[str, object]] = {}

    def factory(name: str):
        def build(**kwargs):
            captured[name] = kwargs
            return object()

        return build

    for name in (
        "CrawlManifestWriter",
        "CrawlStateManifestWriter",
        "PreprocessingManifestWriter",
        "AugmentationManifestWriter",
        "TrainingArtifactManifestWriter",
        "CrawlStateReferenceResolver",
    ):
        monkeypatch.setattr(module, name, factory(name))

    settings = SimpleNamespace(
        paths=SimpleNamespace(root=SimpleNamespace()),
        sources=SimpleNamespace(
            active=SimpleNamespace(seed_urls=()),
        ),
        crawl_output_gate=SimpleNamespace(),
        training=SimpleNamespace(
            release_stage="production_model",
        ),
    )
    payloads = SimpleNamespace(
        crawl={},
        preprocessing={},
        normalization={},
        deduplication={},
        splitting={},
        validation={},
        augmentation={},
        augmentation_strategy={},
        training={},
        model={},
    )
    logger_factory = SimpleNamespace(
        get_logger_for=lambda _writer_type: object(),
    )

    writers = module._assemble_manifest_writers(
        settings=settings,
        logger_factory=logger_factory,
        artifact_path_registry=object(),
        raw_inventory_reader=object(),
        curated_inventory_reader=object(),
        training_inventory_reader=object(),
        settings_fingerprint_calculator=object(),
        source_fingerprint_calculator=object(),
        file_fingerprint_calculator=object(),
        payloads=payloads,
        artifact_identity=object(),
        file_writer=object(),
        now=lambda: datetime(2026, 7, 28, tzinfo=UTC),
        generate_id=lambda: "test-id",
    )

    assert writers.training is not None
    training_kwargs = captured["TrainingArtifactManifestWriter"]
    assert training_kwargs["release_stage"] == "production_model"
    assert "release_requirements" not in training_kwargs
    assert "dataset_validator_settings" not in training_kwargs
