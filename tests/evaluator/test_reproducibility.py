from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import torch

from evaluator.reproducibility import (
    RECEIPT_SCHEMA_VERSION,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    RUN_RECEIPTS_SCHEMA_VERSION,
    TrainingRunReceipt,
    evaluate_reproducibility,
    load_run_receipts_collection,
    load_run_reproducibility_receipt,
    stable_payload_fingerprint,
    write_reproducibility_report,
    write_run_receipts_collection,
    write_training_reproducibility_receipt,
)


class _Policy:
    def __init__(
        self,
        *,
        seeds: Sequence[int],
        require_deterministic_execution: bool = True,
        metric_tolerances: Mapping[str, float] | None = None,
    ) -> None:
        self.seeds = tuple(seeds)
        self.require_deterministic_execution = require_deterministic_execution
        self.metric_tolerances = dict(metric_tolerances or {})


def _write_fixture_receipt(
    *,
    output_path: Path,
    seed: int,
    checkpoint_payload: Mapping[str, object],
    run_id: str,
    resume_from_checkpoint: str | None = None,
) -> TrainingRunReceipt:
    training_settings: dict[str, object] = {
        "deterministic": True,
        "seed": seed,
        "text_tokenizer_backend": "local",
        "text_tokenizer_name": "fixture",
        "text_tokenizer_artifact_version": "1",
        "text_tokenizer_vocab_size": 32,
        "text_tokenizer_special_tokens": ["<pad>"],
    }
    if resume_from_checkpoint is not None:
        training_settings["resume_from_checkpoint"] = resume_from_checkpoint
    return write_training_reproducibility_receipt(
        output_path=output_path,
        run_id=run_id,
        seed=seed,
        dataset_manifest_sha256="a" * 64,
        checkpoint_payload=checkpoint_payload,
        training_settings=training_settings,
        model_settings={"hidden_dim": 8},
        container_digest="sha256:fixture",
    )


def _checkpoint_payload(*, train_loss: float) -> dict[str, object]:
    return {
        "model_state": {"weight": torch.tensor([[1.0, train_loss]])},
        "metadata": {
            "initialization": {
                "method": "seeded",
                "schema": "fixture_v1",
                "parameter_count": 2,
                "trainable_parameter_count": 2,
            },
            "training_scale_plan": {"optimizer_steps": 10},
            "train_loss": train_loss,
            "val_loss": train_loss - 0.1,
            "test_loss": train_loss - 0.2,
        },
    }


def _write_fixture_receipt_path(
    *,
    output_path: Path,
    run_id: str,
    seed: int,
) -> TrainingRunReceipt:
    return _write_fixture_receipt(
        output_path=output_path,
        run_id=run_id,
        seed=seed,
        checkpoint_payload=_checkpoint_payload(train_loss=0.5),
    )


def test_run_receipt_persists_facts_without_passed_flag(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "training-snapshot" / "evaluation" / "run_receipt.json"
    )
    output_path.parent.mkdir(parents=True)

    receipt = _write_fixture_receipt_path(
        output_path=output_path,
        run_id="attempt-1",
        seed=7,
    )

    assert output_path.is_file()
    assert "passed" not in receipt
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["run_id"] == "attempt-1"
    assert receipt["seed"] == 7
    assert receipt["metrics"] == {
        "train_loss": 0.5,
        "validation_loss": 0.4,
        "test_loss": 0.3,
    }
    assert "experiment_sha256" in receipt
    assert "initialization_policy_sha256" in receipt
    assert "training_plan_sha256" in receipt
    assert "semantic_config_sha256" not in receipt
    assert "initialization_sha256" not in receipt
    assert "dataset_tree_fingerprint" not in receipt
    assert load_run_reproducibility_receipt(output_path) == receipt
    assert json.loads(output_path.read_text(encoding="utf-8"))["run_id"] == (
        "attempt-1"
    )


def test_evaluate_reproducibility_aggregates_multi_seed_campaign(
    tmp_path: Path,
) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    policy = _Policy(
        seeds=[42, 43, 44],
        metric_tolerances={"validation_loss": 0.03},
    )

    report = evaluate_reproducibility(
        receipts=receipts,
        policy=policy,
        release_requirements_id="fixture-policy",
    )

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["kind"] == REPORT_KIND
    assert report["passed"] is True
    assert report["run_ids"] == ["attempt-42", "attempt-43", "attempt-44"]
    assert report["seeds"] == [42, 43, 44]
    assert report["required_seeds"] == [42, 43, 44]
    assert report["release_requirements_id"] == "fixture-policy"
    assert report["reproducibility_deterministic_required"] is True
    assert report["reproducibility_metric_tolerances"] == {
        "validation_loss": 0.03
    }
    assert report["run_receipts"] == [
        {
            "run_id": receipt["run_id"],
            "seed": receipt["seed"],
            "sha256": stable_payload_fingerprint(dict(receipt)),
        }
        for receipt in receipts
    ]
    assert report["violations"] == []
    assert report["metrics"]["validation_loss"]["mean"] == pytest.approx(0.4)
    assert report["metrics"]["validation_loss"]["variation"] == 0.0


def test_evaluate_reproducibility_fails_on_missing_seed_and_tolerance(
    tmp_path: Path,
) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43)
    ]
    policy = _Policy(
        seeds=[42, 43, 44],
        metric_tolerances={"validation_loss": 0.03},
    )
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=policy,
        release_requirements_id="fixture-policy",
    )

    assert report["passed"] is False
    assert "required_seeds_missing" in report["violations"]


def test_duplicate_seed_receipt_cannot_pass_a_campaign(
    tmp_path: Path,
) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"run-{index}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{index}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for index, seed in enumerate((42, 43, 44, 44), start=1)
    ]

    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(
            seeds=[42, 43, 44],
            metric_tolerances={"validation_loss": 0.03},
        ),
        release_requirements_id="fixture-policy",
    )

    assert report["passed"] is False
    assert "required_seeds_missing" in report["violations"]
    assert "duplicate_seed_receipt:44" in report["violations"]
    assert report["metrics"] == {}


def test_resume_receipt_requires_and_preserves_parent_lineage(
    tmp_path: Path,
) -> None:
    resumed_payload = _checkpoint_payload(train_loss=0.5)
    metadata = resumed_payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["resumed_from_run_id"] = "parent-attempt"

    receipt = _write_fixture_receipt(
        output_path=tmp_path / "resumed" / "run_receipt.json",
        seed=42,
        run_id="attempt-42",
        checkpoint_payload=resumed_payload,
        resume_from_checkpoint="checkpoint.pt",
    )

    assert receipt["resumed_from_run_id"] == "parent-attempt"
    with pytest.raises(ValueError, match="lacks metadata.resumed_from_run_id"):
        _write_fixture_receipt(
            output_path=tmp_path / "missing-parent" / "run_receipt.json",
            seed=43,
            run_id="attempt-43",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
            resume_from_checkpoint="checkpoint.pt",
        )


def test_receipts_require_real_sha256_digest_values(tmp_path: Path) -> None:
    receipt = _write_fixture_receipt_path(
        output_path=tmp_path / "run_receipt.json",
        run_id="attempt-42",
        seed=42,
    )
    tampered = dict(receipt)
    tampered["model_state_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="model_state_sha256 must be SHA-256"):
        write_run_receipts_collection(
            path=tmp_path / "run_receipts.json",
            receipts=[tampered],  # type: ignore[list-item]
        )

    tampered["model_state_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="model_state_sha256 must be SHA-256"):
        write_run_receipts_collection(
            path=tmp_path / "upper-case.json",
            receipts=[tampered],  # type: ignore[list-item]
        )


def test_evaluate_reproduce_fails_on_metric_variation(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(
                train_loss=0.5 + (0.05 * (seed - 42))
            ),
        )
        for seed in (42, 43, 44)
    ]
    policy = _Policy(
        seeds=[42, 43, 44],
        metric_tolerances={"validation_loss": 0.03},
    )
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=policy,
        release_requirements_id="fixture-policy",
    )

    assert report["passed"] is False
    assert "metric_variation:validation_loss" in report["violations"]


def test_report_round_trip(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(seeds=[42, 43, 44]),
        release_requirements_id="fixture-policy",
    )
    report_path = tmp_path / "reproducibility_report.json"

    written = write_reproducibility_report(path=report_path, report=report)

    assert written == report_path
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["kind"] == "reproducibility_report"
    assert loaded["schema_version"] == "3.2"
    assert loaded["passed"] is True
    assert loaded["release_requirements_id"] == "fixture-policy"


def test_report_rejects_unknown_run_receipt_identity(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(seeds=[42, 43, 44]),
        release_requirements_id="fixture-policy",
    )
    report["run_receipts"][0]["run_id"] = "does-not-exist-42"

    with pytest.raises(ValueError, match="run receipt identities"):
        write_reproducibility_report(
            path=tmp_path / "reproducibility_report.json",
            report=report,
        )


def test_report_rejects_missing_run_receipt_digests(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(seeds=[42, 43, 44]),
        release_requirements_id="fixture-policy",
    )
    del report["run_receipts"][0]

    with pytest.raises(ValueError, match="run count"):
        write_reproducibility_report(
            path=tmp_path / "reproducibility_report.json",
            report=report,
        )


def test_run_receipts_collection_round_trip(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    collection_path = tmp_path / "run_receipts.json"

    written = write_run_receipts_collection(
        path=collection_path,
        receipts=receipts,
    )

    assert written == collection_path
    payload = load_run_receipts_collection(collection_path)
    assert payload["schema_version"] == RUN_RECEIPTS_SCHEMA_VERSION
    assert payload["run_receipts"] == [dict(receipt) for receipt in receipts]


def test_report_rejects_tampered_policy_digest(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(seeds=[42, 43, 44]),
        release_requirements_id="fixture-policy",
    )
    report["release_requirements_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="policy digest"):
        write_reproducibility_report(
            path=tmp_path / "reproducibility_report.json",
            report=report,
        )


def test_report_rejects_float_seeds(tmp_path: Path) -> None:
    receipts = [
        _write_fixture_receipt(
            output_path=tmp_path / f"seed-{seed}" / "run_receipt.json",
            seed=seed,
            run_id=f"attempt-{seed}",
            checkpoint_payload=_checkpoint_payload(train_loss=0.5),
        )
        for seed in (42, 43, 44)
    ]
    report = evaluate_reproducibility(
        receipts=receipts,
        policy=_Policy(seeds=[42, 43, 44]),
        release_requirements_id="fixture-policy",
    )
    report["seeds"] = [42.9, 43.9, 44.9]

    with pytest.raises(ValueError, match="seeds must be integers"):
        write_reproducibility_report(
            path=tmp_path / "reproducibility_report.json",
            report=report,
        )
