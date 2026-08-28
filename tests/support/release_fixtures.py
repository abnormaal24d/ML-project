"""Shared release-promotion fixtures for contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from config.releases.release_requirements import (
    ReleaseRequirements,
    ReproducibilityRequirements,
)
from evaluator.reproducibility import (
    RECEIPT_SCHEMA_VERSION,
    RUN_RECEIPTS_SCHEMA_VERSION,
    stable_payload_fingerprint,
)
from release.release_artifact_validation import _REQUIRED_MODEL_CARD_SECTIONS


def write_valid_model_card(path: Path) -> None:
    """Write a minimal valid model_card.md with all 8 required ``##`` headings.

    The headings are normalized as the parser lowercases and strips ``(required)``.
    Content after each heading is a single sentence.
    """
    lines: list[str] = []
    for section in _REQUIRED_MODEL_CARD_SECTIONS:
        lines.append(f"## {section}")
        lines.append(f"Content about {section}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture_policy() -> ReproducibilityRequirements:
    return ReproducibilityRequirements(
        policy_id="production_v1",
        seeds=(42,),
        require_deterministic_execution=True,
        metric_tolerances={"validation_loss": 0.03, "test_loss": 0.03},
    )


def fixture_requirements() -> ReleaseRequirements:
    return ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=("image",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_tagging",),
        optional_tasks=(),
        blocked_capabilities=(),
        reproducibility=fixture_policy(),
    )


def fixture_receipts(
    policy: ReproducibilityRequirements,
    *,
    run_id_prefix: str = "attempt-fixture",
    metrics: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """One structurally valid immutable run receipt per policy seed."""
    resolved_metrics = metrics or {
        "validation_loss": 1.0,
        "test_loss": 2.0,
    }
    return [
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": f"{run_id_prefix}-{seed}",
            "seed": seed,
            "dataset_manifest_sha256": "d" * 64,
            "experiment_sha256": "e" * 64,
            "tokenizer_sha256": "a" * 64,
            "initialization_policy_sha256": "b" * 64,
            "training_plan_sha256": "c" * 64,
            "container_digest": "c" * 64,
            "model_state_sha256": "f" * 64,
            "hardware": {"platform": "fixture"},
            "determinism": {
                "python_seeded": True,
                "numpy_seeded": True,
                "torch_seeded": True,
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "tf32_disabled": True,
            },
            "metrics": dict(resolved_metrics),
            "resumed_from_run_id": None,
        }
        for seed in policy.seeds
    ]


def write_valid_reproducibility_report(
    path: Path,
    *,
    policy: ReproducibilityRequirements,
    receipts: list[dict[str, object]] | None = None,
) -> None:
    """Persist a schema-3.2 report bound to its immutable run receipts.

    The receipts collection is written next to the report so the evidence
    bundle can reference it; every per-run digest in the report is the
    canonical fingerprint of the receipt it names.
    """
    resolved_receipts = receipts or fixture_receipts(policy)
    run_ids = [str(receipt["run_id"]) for receipt in resolved_receipts]
    path.parent.mkdir(parents=True, exist_ok=True)

    aggregated_metrics = {}
    for metric_name, tolerance in sorted(policy.metric_tolerances.items()):
        values = [
            float(receipt["metrics"][metric_name])
            for receipt in resolved_receipts
        ]
        aggregated_metrics[metric_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "variation": max(values) - min(values),
            "tolerance": float(tolerance),
        }

    path.write_text(
        json.dumps(
            {
                "schema_version": "3.2",
                "kind": "reproducibility_report",
                "run_ids": run_ids,
                "seeds": sorted(int(seed) for seed in policy.seeds),
                "required_seeds": sorted(int(seed) for seed in policy.seeds),
                "release_requirements_id": policy.policy_id,
                "release_requirements_sha256": policy.policy_sha256,
                "reproducibility_deterministic_required": (
                    policy.require_deterministic_execution
                ),
                "reproducibility_metric_tolerances": {
                    str(name): float(tolerance)
                    for name, tolerance in sorted(
                        policy.metric_tolerances.items()
                    )
                },
                "experiment_sha256": "e" * 64,
                "dataset_manifest_sha256": "d" * 64,
                "run_receipts": [
                    {
                        "run_id": receipt["run_id"],
                        "seed": int(receipt["seed"]),
                        "sha256": stable_payload_fingerprint(dict(receipt)),
                    }
                    for receipt in resolved_receipts
                ],
                "metrics": aggregated_metrics,
                "violations": [],
                "passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path.with_name("run_receipts.json").write_text(
        json.dumps(
            {
                "schema_version": RUN_RECEIPTS_SCHEMA_VERSION,
                "run_receipts": resolved_receipts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
