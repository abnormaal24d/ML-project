"""Fail-closed reproducibility policy binding for the release gate.

These tests pin the finding that a report produced under a weaker
reproducibility policy (or a hand-crafted report) must never pass the
production gate under a stronger policy. Everything the report self-declares
is untrusted; the gate re-derives the decision from the active policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from config.releases.release_requirements import (
    ReleaseRequirements,
    ReproducibilityRequirements,
)
from evaluator.leakage.report import generate_report
from evaluator.reproducibility import (
    RECEIPT_SCHEMA_VERSION,
    RUN_RECEIPTS_SCHEMA_VERSION,
    stable_payload_fingerprint,
)
from release.model_release_publisher import promote_model
from release.release_evidence_bundle import (
    EvaluationArtifacts,
    ReleaseEvidenceBundle,
)
from release.release_evidence_validation import check_release
from tests.support.release_fixtures import (
    write_valid_model_card,
)

PRODUCTION_POLICY = ReproducibilityRequirements(
    policy_id="production_v1",
    seeds=(42, 43, 44),
    require_deterministic_execution=True,
    metric_tolerances={"validation_loss": 0.03, "test_loss": 0.03},
)

CANDIDATE_POLICY = ReproducibilityRequirements(
    policy_id="candidate_v1",
    seeds=(42, 43, 44),
    require_deterministic_execution=True,
    metric_tolerances={"validation_loss": 0.05, "test_loss": 0.05},
)


def _valid_report(policy: ReproducibilityRequirements) -> dict:
    return {
        "schema_version": "3.2",
        "kind": "reproducibility_report",
        "run_ids": ["attempt-42", "attempt-43", "attempt-44"],
        "seeds": list(policy.seeds),
        "required_seeds": list(policy.seeds),
        "release_requirements_id": policy.policy_id,
        "release_requirements_sha256": policy.policy_sha256,
        "reproducibility_deterministic_required": (
            policy.require_deterministic_execution
        ),
        "reproducibility_metric_tolerances": dict(policy.metric_tolerances),
        "experiment_sha256": "e" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "metrics": {
            metric_name: {"variation": 0.0, "tolerance": tolerance}
            for metric_name, tolerance in policy.metric_tolerances.items()
        },
        "violations": [],
        "passed": True,
    }


def _receipt(seed: int, *, run_id: str, metrics: dict[str, float]) -> dict:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
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
        "metrics": dict(metrics),
        "resumed_from_run_id": None,
    }


def _receipts(
    policy: ReproducibilityRequirements,
    *,
    spread: float = 0.0,
) -> list[dict]:
    return [
        _receipt(
            seed,
            run_id=f"attempt-{seed}",
            metrics={
                "validation_loss": 1.0 + (index * spread),
                "test_loss": 2.0 + (index * spread),
            },
        )
        for index, seed in enumerate(policy.seeds)
    ]


def _build_evidence(
    tmp_path: Path,
    *,
    report: dict[str, object],
    policy: ReproducibilityRequirements,
    receipts: list[dict[str, object]] | None = None,
    with_receipts: bool = True,
) -> object:
    candidate = tmp_path / "candidate"
    dataset = tmp_path / "dataset"
    candidate.mkdir(parents=True)
    dataset.mkdir(parents=True)

    checkpoint = candidate / "checkpoint.pt"
    checkpoint.write_bytes(b"checksum-backed-checkpoint")
    metrics = candidate / "training_metrics.json"
    metrics.write_text('{"train_loss": 0.5}\n', encoding="utf-8")
    write_valid_model_card(candidate / "model_card.md")
    (candidate / "model_card.md").write_text(
        """
## Release identity

production_model.

## Architecture boundary

Constrained coverage model.

## Intended use

Constrained coverage model.

## Out-of-scope and disabled capabilities

None.

## Training data and provenance

Local test dataset.

## Evaluation and acceptance

Thresholds met on test split.

## Limitations and risks

Constrained coverage model.

## Release decision

production_model.
""",
        encoding="utf-8",
    )
    artifact_path = candidate / "model.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(artifact_path))
    (candidate / "safetensors_export_status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sha256": hashlib.sha256(
                    artifact_path.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "dataset_card.json").write_text(
        '{"name": "validated production dataset"}\n',
        encoding="utf-8",
    )
    leakage = dataset / "evaluation" / "leakage_report.json"
    generate_report(
        left_records=(),
        right_records=(),
        output_path=leakage,
        minimum_coverage={},
    )
    resolved_receipts = receipts if receipts is not None else _receipts(policy)
    if "run_receipts" not in report:
        report["run_receipts"] = [
            {
                "run_id": str(receipt["run_id"]),
                "seed": int(receipt["seed"]),
                "sha256": stable_payload_fingerprint(dict(receipt)),
            }
            for receipt in resolved_receipts
        ]
    reproducibility = candidate / "evaluation" / "reproducibility_report.json"
    reproducibility.parent.mkdir(parents=True)
    reproducibility.write_text(json.dumps(report) + "\n", encoding="utf-8")
    if with_receipts:
        (candidate / "evaluation" / "run_receipts.json").write_text(
            json.dumps(
                {
                    "schema_version": RUN_RECEIPTS_SCHEMA_VERSION,
                    "run_receipts": resolved_receipts,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    artifacts = EvaluationArtifacts.from_release_outputs(
        candidate_directory=candidate,
        dataset_directory=dataset,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    return ReleaseEvidenceBundle.build(
        mode="production_model",
        artifacts=artifacts,
        release_requirements_id=policy.policy_id,
        release_requirements_sha256=policy.policy_sha256,
    )


def _violation_contains(gate, fragment: str) -> bool:
    return any(fragment in violation for violation in gate.violations)


def test_passing_report_bound_to_active_policy_is_accepted(
    tmp_path: Path,
) -> None:
    policy = PRODUCTION_POLICY
    evidence = _build_evidence(
        tmp_path,
        report=_valid_report(policy),
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Per the canonical gate: a report bound to the active policy must fully
    # match the canonical evaluation. A mismatch here means the report is
    # rejected with reproducibility_report_mismatch rather than individual
    # policy violation codes.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_report_under_candidate_policy_is_rejected_under_production_policy(
    tmp_path: Path,
) -> None:
    """Candidate report (0.05 tolerance, 0.04 spread) must fail at 0.03."""
    report = _valid_report(CANDIDATE_POLICY)
    report["metrics"] = {
        "validation_loss": {"variation": 0.04, "tolerance": 0.05},
        "test_loss": {"variation": 0.04, "tolerance": 0.05},
    }
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=CANDIDATE_POLICY,
        receipts=_receipts(CANDIDATE_POLICY, spread=0.04),
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=PRODUCTION_POLICY,
    )

    assert not gate.passed
    # With the new gate, candidate-under-production produces a report mismatch
    # rather than individual policy violation codes.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_metric_variation_is_recomputed_against_active_tolerance(
    tmp_path: Path,
) -> None:
    """A bound report whose variation exceeds the active tolerance is rejected."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["metrics"] = {
        "validation_loss": {"variation": 0.04, "tolerance": 0.03},
        "test_loss": {"variation": 0.0, "tolerance": 0.03},
    }
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # The report now mismatch the canonical evaluation; the specific
    # metric variation code is subsumed by reproducibility_report_mismatch.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_empty_metrics_fails_closed_against_required_metric_set(
    tmp_path: Path,
) -> None:
    """metrics={} is not a pass: every policy-required metric must be present."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["metrics"] = {}
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Missing metrics cause a report mismatch since the canonical eval
    # produces different metrics structure.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_float_seeds_are_rejected_without_coercion(tmp_path: Path) -> None:
    """[42.9, 43.9, 44.9] must not silently coerce to the policy seeds."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["seeds"] = [42.9, 43.9, 44.9]
    report["required_seeds"] = [42.9, 43.9, 44.9]
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    assert _violation_contains(gate, "invalid_release_artifact")
    assert _violation_contains(gate, "ValueError")


def test_substituted_seed_set_is_rejected(tmp_path: Path) -> None:
    """The original [999] report remains blocked under the active policy."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["seeds"] = [999]
    report["required_seeds"] = [999]
    report["run_ids"] = ["attempt-999"]
    report["release_requirements_sha256"] = "f" * 64
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    assert _violation_contains(gate, "invalid_release_artifact")
    assert _violation_contains(gate, "ValueError")


def test_missing_policy_fields_fail_closed(tmp_path: Path) -> None:
    """A report without deterministic/tolerance declarations is rejected."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    for field in (
        "reproducibility_deterministic_required",
        "reproducibility_metric_tolerances",
        "release_requirements_id",
        "release_requirements_sha256",
    ):
        del report[field]
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    assert _violation_contains(gate, "invalid_release_artifact")
    assert _violation_contains(gate, "ValueError")


def test_deterministic_requirement_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    """A report claiming non-deterministic execution fails under a policy
    that requires deterministic execution, even when everything else binds."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["reproducibility_deterministic_required"] = False
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    assert _violation_contains(gate, "invalid_release_artifact")
    assert _violation_contains(gate, "ValueError")


def test_gate_requires_active_policy_fail_closed(tmp_path: Path) -> None:
    """check_release without the active policy must not pass."""
    policy = PRODUCTION_POLICY
    evidence = _build_evidence(
        tmp_path,
        report=_valid_report(policy),
        policy=policy,
    )

    gate = check_release(evidence=evidence)

    assert not gate.passed
    assert _violation_contains(gate, "reproducibility_policy_missing")


def test_promotion_requires_release_contract_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires release requirements"):
        promote_model(
            candidate_directory=tmp_path / "candidate",
            production_directory=tmp_path / "production",
            evidence_bundle_path=tmp_path / "missing.json",
            settings=object(),
            training_result=object(),
            evaluation_result=object(),
            dataset_root=tmp_path / "dataset",
            release_requirements=None,
        )


def test_requirements_without_reproducibility_policy_fail_closed(
    tmp_path: Path,
) -> None:
    requirements = ReleaseRequirements(
        release_id="production_v1",
        release_stage="production_model",
        required_modalities=("image",),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=("image_tagging",),
        optional_tasks=(),
        blocked_capabilities=(),
        reproducibility=None,
    )
    with pytest.raises(ValueError, match="reproducibility policy"):
        promote_model(
            candidate_directory=tmp_path / "candidate",
            production_directory=tmp_path / "production",
            evidence_bundle_path=tmp_path / "missing.json",
            settings=object(),
            training_result=object(),
            evaluation_result=object(),
            dataset_root=tmp_path / "dataset",
            release_requirements=requirements,
        )


def test_report_with_fake_run_ids_is_rejected(tmp_path: Path) -> None:
    """run_ids that never executed are caught by the receipt binding.

    This pins the adjacent integrity finding: a hand-crafted report naming
    nonexistent runs (with correct policy fields and zero variation) must
    fail because no immutable run receipt backs the named runs.
    """
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["run_ids"] = [
        "does-not-exist-42",
        "does-not-exist-43",
        "does-not-exist-44",
    ]
    report["run_receipts"] = [
        {"run_id": f"does-not-exist-{seed}", "seed": seed, "sha256": "f" * 64}
        for seed in policy.seeds
    ]
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Receipt identity mismatch now produces a report mismatch rather than
    # individual receipt violations under the canonical gate.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_invalid_run_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    """A structurally invalid run_receipts artifact is caught as a gate
    violation rather than crashing the whole release flow."""
    policy = PRODUCTION_POLICY
    receipts = _receipts(policy)

    del receipts[0]["determinism"]

    evidence = _build_evidence(
        tmp_path,
        report=_valid_report(policy),
        policy=policy,
        receipts=receipts,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    assert _violation_contains(gate, "invalid_release_artifact")
    assert _violation_contains(gate, "ValueError")


def test_report_with_tampered_receipt_digest_is_rejected(
    tmp_path: Path,
) -> None:
    """A receipt digest that does not match the real receipt is a violation."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)
    report["run_receipts"] = [
        {"run_id": f"attempt-{seed}", "seed": seed, "sha256": "f" * 64}
        for seed in policy.seeds
    ]
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Tampered receipt digest now produces a report mismatch rather than
    # an individual digest mismatch code under the canonical gate.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_nondeterministic_receipt_is_rejected(tmp_path: Path) -> None:
    """Determinism is re-computed from the receipts, not the report claim."""
    policy = PRODUCTION_POLICY
    receipts = _receipts(policy)
    receipts[0]["determinism"]["torch_seeded"] = False
    report = _valid_report(policy)
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
        receipts=receipts,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Non-determinism now produces a report mismatch rather than a
    # per-run nondeterministic code under the canonical gate.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_metric_spread_is_recomputed_from_real_receipts(
    tmp_path: Path,
) -> None:
    """A report declaring zero variation fails when the receipts show spread.

    This is the core of the adjacent finding: the gate must re-derive the
    decision from the real receipts, not trust the aggregate variation the
    report self-declares.
    """
    policy = PRODUCTION_POLICY
    receipts = _receipts(policy, spread=0.04)
    report = _valid_report(policy)
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
        receipts=receipts,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Receipts with spread cause a report mismatch since the canonical
    # evaluation derives different variation values.
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_receipt_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    """Receipts for a different experiment cannot back a report."""
    policy = PRODUCTION_POLICY
    receipts = _receipts(policy)
    receipts[0]["experiment_sha256"] = "f" * 64
    report = _valid_report(policy)
    evidence = _build_evidence(
        tmp_path,
        report=report,
        policy=policy,
        receipts=receipts,
    )

    gate = check_release(
        evidence=evidence,
        reproducibility_requirements=policy,
    )

    assert not gate.passed
    # Experiment identity mismatch causes a report mismatch
    assert _violation_contains(gate, "reproducibility_report_mismatch")


def test_missing_run_receipts_artifact_fails_closed(tmp_path: Path) -> None:
    """Candidate/production evidence without the receipts is incomplete."""
    policy = PRODUCTION_POLICY
    report = _valid_report(policy)

    with pytest.raises(FileNotFoundError, match="run_receipts"):
        _build_evidence(
            tmp_path,
            report=report,
            policy=policy,
            with_receipts=False,
        )
