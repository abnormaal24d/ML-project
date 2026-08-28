"""Immutable run receipts and multi-seed reproducibility evaluation.

The module owns two distinct concepts:

* ``TrainingRunReceipt`` — facts about what one training run actually did.
  A single run can never prove reproducibility, so a receipt carries no
  ``passed`` flag.
* ``evaluate_reproducibility`` — a pure comparison of three (or more) run
  receipts against one explicit release reproducibility policy. Its output is
  the only release-decisive report, persisted with
  ``write_reproducibility_report``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, Self, TypedDict, cast

from schemas.canonical import stable_payload_fingerprint

RECEIPT_SCHEMA_VERSION = "3.0"
REPORT_SCHEMA_VERSION = "3.2"
REPORT_KIND = "reproducibility_report"
RUN_RECEIPTS_SCHEMA_VERSION = "run_receipts.v1"

_DETERMINISM_FIELDS = (
    "python_seeded",
    "numpy_seeded",
    "torch_seeded",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "tf32_disabled",
)

# Identity fields that are expected to be invariant across a multi-run
# campaign. Only the seed (and therefore every derived fingerprint) is
# allowed to vary between runs.
type _IdentityField = Literal[
    "dataset_manifest_sha256",
    "experiment_sha256",
    "tokenizer_sha256",
    "initialization_policy_sha256",
    "training_plan_sha256",
    "container_digest",
]

_IDENTITY_FIELDS: tuple[_IdentityField, ...] = (
    "dataset_manifest_sha256",
    "experiment_sha256",
    "tokenizer_sha256",
    "initialization_policy_sha256",
    "training_plan_sha256",
    "container_digest",
)


class _ByteArray(Protocol):
    def tobytes(self) -> bytes: ...


class _FingerprintTensor(Protocol):
    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> Sequence[int]: ...

    def detach(self) -> Self: ...

    def cpu(self) -> Self: ...

    def contiguous(self) -> Self: ...

    def numpy(self) -> _ByteArray: ...


class TrainingRunReceipt(TypedDict):
    """Facts recorded for a single executed training run."""

    schema_version: str
    run_id: str
    seed: int
    dataset_manifest_sha256: str
    experiment_sha256: str
    tokenizer_sha256: str
    initialization_policy_sha256: str
    training_plan_sha256: str
    container_digest: str
    model_state_sha256: str
    hardware: Mapping[str, object]
    determinism: Mapping[str, object]
    metrics: Mapping[str, float]
    resumed_from_run_id: str | None


class ReproducibilityPolicy(Protocol):
    """Release reproducibility contract consumed by pure evaluation."""

    @property
    def seeds(self) -> Sequence[int]: ...

    @property
    def require_deterministic_execution(self) -> bool: ...

    @property
    def metric_tolerances(self) -> Mapping[str, float]: ...


def model_state_fingerprint(state: Mapping[str, _FingerprintTensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def experiment_fingerprint(
    *,
    training_settings: Mapping[str, object],
    model_settings: Mapping[str, object],
) -> str:
    """Fingerprint the experiment, ignoring deliberate per-run variation.

    The seed and resume location are invocation details that must differ
    between campaign runs; excluding them keeps one stable experiment
    identity across all seeds.
    """

    training = dict(training_settings)
    training.pop("seed", None)
    training.pop("resume_from_checkpoint", None)
    return stable_payload_fingerprint(
        {
            "training": training,
            "model": dict(model_settings),
        }
    )


_INITIALIZATION_POLICY_FIELDS = (
    "schema",
    "parameter_count",
    "trainable_parameter_count",
)


def initialization_policy_fingerprint(
    metadata: Mapping[str, object],
) -> str:
    """Fingerprint the initialization *policy*, not the concrete seed.

    The seed drives the concrete parameter values and therefore must be
    excluded when comparing runs against a multi-seed policy.
    """

    missing = [
        field_name
        for field_name in _INITIALIZATION_POLICY_FIELDS
        if field_name not in metadata
    ]
    if missing:
        raise ValueError(
            "initialization policy evidence is missing fields: "
            + ", ".join(missing)
        )
    return stable_payload_fingerprint(
        {field: metadata[field] for field in _INITIALIZATION_POLICY_FIELDS}
    )


def _validate_receipt(p: Mapping[str, object]) -> None:
    required = (
        "run_id",
        "seed",
        "dataset_manifest_sha256",
        "experiment_sha256",
        "tokenizer_sha256",
        "initialization_policy_sha256",
        "training_plan_sha256",
        "container_digest",
        "model_state_sha256",
        "hardware",
        "determinism",
        "metrics",
        "resumed_from_run_id",
    )
    if p.get("schema_version") != RECEIPT_SCHEMA_VERSION or any(
        key not in p for key in required
    ):
        raise ValueError("invalid reproducibility run receipt schema")
    text_fields = (
        "run_id",
        "container_digest",
    )
    digest_fields = (
        "dataset_manifest_sha256",
        "experiment_sha256",
        "tokenizer_sha256",
        "initialization_policy_sha256",
        "training_plan_sha256",
        "model_state_sha256",
    )
    for field in text_fields:
        value = p[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("receipt identity fields must be explicit")
    for field in digest_fields:
        if not _sha256_string(p[field]):
            raise ValueError(f"receipt {field} must be SHA-256")
    seed = p["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("receipt seed must be an integer")
    resumed_from_run_id = p["resumed_from_run_id"]
    if resumed_from_run_id is not None and (
        not isinstance(resumed_from_run_id, str)
        or not resumed_from_run_id.strip()
    ):
        raise ValueError("receipt resume parent must be a string or null")
    if not isinstance(p["hardware"], Mapping):
        raise ValueError("receipt hardware evidence must be an object")
    if not isinstance(p["metrics"], Mapping) or not p["metrics"]:
        raise ValueError("receipt metrics required")
    if not all(
        isinstance(name, str)
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for name, value in p["metrics"].items()
    ):
        raise ValueError("receipt metrics must be finite numbers")
    determinism = p["determinism"]
    if not isinstance(determinism, Mapping) or not all(
        isinstance(determinism.get(key), bool) for key in _DETERMINISM_FIELDS
    ):
        raise ValueError("determinism evidence incomplete")


def run_receipts_collection_payload(
    receipts: Sequence[TrainingRunReceipt],
) -> dict[str, object]:
    """Canonical receipts-collection payload bound into release evidence."""
    return {
        "schema_version": RUN_RECEIPTS_SCHEMA_VERSION,
        "run_receipts": [dict(receipt) for receipt in receipts],
    }


def write_run_receipts_collection(
    *,
    path: Path,
    receipts: Sequence[TrainingRunReceipt],
) -> Path:
    """Atomically persist the immutable receipts backing one report."""
    for receipt in receipts:
        _validate_receipt(receipt)
    _atomic_write_json(
        path=path, payload=run_receipts_collection_payload(receipts)
    )
    return path


def load_run_receipts_collection(path: Path) -> dict[str, object]:
    """Load and structurally validate a persisted receipts collection."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("run receipts collection root must be an object")
    payload = cast(dict[str, object], loaded)
    if payload.get("schema_version") != RUN_RECEIPTS_SCHEMA_VERSION:
        raise ValueError("invalid run receipts collection schema")
    raw_receipts = payload.get("run_receipts")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise ValueError("run receipts collection must contain receipts")
    for entry in raw_receipts:
        if not isinstance(entry, dict):
            raise ValueError("run receipts collection entry must be an object")
        _validate_receipt(entry)
    return payload


def _atomic_write_json(*, path: Path, payload: object) -> None:
    """Crash-durable atomic JSON write: fsync file before rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_run_reproducibility_receipt(
    *,
    output_path: Path,
    run_id: str,
    seed: int,
    dataset_manifest_sha256: str,
    experiment_sha256: str,
    model_state_sha256: str,
    container_digest: str,
    hardware: Mapping[str, object],
    determinism: Mapping[str, object],
    metrics: Mapping[str, float],
    resumed_from_run_id: str | None = None,
    tokenizer_sha256: str,
    initialization_policy_sha256: str,
    training_plan_sha256: str,
) -> TrainingRunReceipt:
    receipt: TrainingRunReceipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "seed": seed,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "experiment_sha256": experiment_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "initialization_policy_sha256": initialization_policy_sha256,
        "training_plan_sha256": training_plan_sha256,
        "container_digest": container_digest,
        "model_state_sha256": model_state_sha256,
        "hardware": dict(hardware),
        "determinism": dict(determinism),
        "metrics": dict(metrics),
        "resumed_from_run_id": resumed_from_run_id,
    }
    _validate_receipt(receipt)
    _atomic_write_json(path=output_path, payload=receipt)
    return receipt


def load_run_reproducibility_receipt(path: Path) -> TrainingRunReceipt:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("reproducibility run receipt root must be an object")
    payload = cast(dict[str, object], loaded)
    _validate_receipt(payload)
    return cast(TrainingRunReceipt, payload)


def write_training_reproducibility_receipt(
    *,
    output_path: Path,
    run_id: str,
    seed: int,
    dataset_manifest_sha256: str,
    checkpoint_payload: Mapping[str, object],
    training_settings: Mapping[str, object],
    model_settings: Mapping[str, object],
    container_digest: str,
    evaluated_metrics: Mapping[str, float] | None = None,
) -> TrainingRunReceipt:
    """Build and persist one immutable run receipt from a verified checkpoint.

    The dataset identity is the already validated immutable manifest digest;
    no full dataset tree scan is performed.
    """

    model_state = checkpoint_payload.get("model_state")
    metadata = checkpoint_payload.get("metadata")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint lacks model_state for reproducibility")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint lacks metadata for reproducibility")
    initialization = metadata.get("initialization")
    training_plan = metadata.get("training_scale_plan")
    if not isinstance(initialization, Mapping):
        raise ValueError("checkpoint lacks initialization evidence")
    if not isinstance(training_plan, Mapping):
        raise ValueError("checkpoint lacks training plan evidence")

    deterministic = bool(training_settings.get("deterministic"))
    determinism = {
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "deterministic_algorithms": deterministic,
        "cudnn_deterministic": deterministic,
        "tf32_disabled": deterministic,
    }
    checkpoint_metrics = {
        name: float(value)
        for name, value in {
            "train_loss": metadata.get("train_loss"),
            "validation_loss": metadata.get("val_loss"),
            "test_loss": metadata.get("test_loss"),
        }.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    metrics = (
        {str(name): float(value) for name, value in evaluated_metrics.items()}
        if evaluated_metrics is not None
        else checkpoint_metrics
    )
    resumed_from_run_id = _resumed_from_run_id(
        metadata=metadata,
        training_settings=training_settings,
    )
    tokenizer_sha = training_settings.get("text_tokenizer_sha256")
    if not isinstance(tokenizer_sha, str) or not tokenizer_sha.strip():
        tokenizer_sha = stable_payload_fingerprint(
            {
                "backend": training_settings.get("text_tokenizer_backend"),
                "name": training_settings.get("text_tokenizer_name"),
                "version": training_settings.get(
                    "text_tokenizer_artifact_version"
                ),
                "vocab_size": training_settings.get(
                    "text_tokenizer_vocab_size"
                ),
                "special_tokens": training_settings.get(
                    "text_tokenizer_special_tokens"
                ),
            }
        )

    return write_run_reproducibility_receipt(
        output_path=output_path,
        run_id=run_id,
        seed=seed,
        dataset_manifest_sha256=dataset_manifest_sha256,
        experiment_sha256=experiment_fingerprint(
            training_settings=training_settings,
            model_settings=model_settings,
        ),
        model_state_sha256=model_state_fingerprint(
            cast(Mapping[str, _FingerprintTensor], model_state)
        ),
        container_digest=container_digest.strip() or "local-unpinned",
        tokenizer_sha256=tokenizer_sha,
        initialization_policy_sha256=initialization_policy_fingerprint(
            initialization
        ),
        training_plan_sha256=stable_payload_fingerprint(training_plan),
        hardware={
            "platform": __import__("platform").platform(),
            "python": __import__("platform").python_version(),
            "torch": str(__import__("torch").__version__),
            "cuda_available": bool(__import__("torch").cuda.is_available()),
        },
        determinism=determinism,
        metrics=metrics,
        resumed_from_run_id=resumed_from_run_id,
    )


def _resumed_from_run_id(
    *,
    metadata: Mapping[str, object],
    training_settings: Mapping[str, object],
) -> str | None:
    """Read the durable resume parent; never silently discard it."""

    raw_parent = metadata.get("resumed_from_run_id")
    parent = raw_parent.strip() if isinstance(raw_parent, str) else None
    resume_requested = training_settings.get("resume_from_checkpoint")
    if isinstance(resume_requested, str):
        resume_requested = bool(resume_requested.strip())
    else:
        resume_requested = bool(resume_requested)
    if resume_requested and not parent:
        raise ValueError(
            "resumed checkpoint lacks metadata.resumed_from_run_id"
        )
    if raw_parent is not None and not parent:
        raise ValueError(
            "checkpoint resumed_from_run_id must be non-empty text"
        )
    return parent


def _require_same(
    receipts: Sequence[TrainingRunReceipt],
    field: _IdentityField,
    violations: list[str],
) -> None:
    values = {str(receipt[field]) for receipt in receipts}
    if len(values) != 1:
        violations.append(f"inconsistent:{field}")


def _determinism_complete(determinism: Mapping[str, object]) -> bool:
    return all(determinism.get(field) is True for field in _DETERMINISM_FIELDS)


def _compare_metrics(
    *,
    receipts: Sequence[TrainingRunReceipt],
    tolerances: Mapping[str, float],
    violations: list[str],
) -> dict[str, dict[str, object]]:
    """Compare only metrics explicitly required by the release policy.

    A metric without a configured tolerance is not compared for seed
    stability; a configured metric that is missing from any run is a
    violation. There is no hidden default tolerance of zero.
    """

    aggregates: dict[str, dict[str, object]] = {}
    for metric_name, configured_tolerance in sorted(tolerances.items()):
        values: list[float] = []
        for receipt in receipts:
            raw = receipt["metrics"].get(metric_name)
            if raw is None or isinstance(raw, bool):
                violations.append(
                    f"missing_metric:{receipt['run_id']}:{metric_name}"
                )
                continue
            value = float(raw)
            if not math.isfinite(value):
                violations.append(
                    f"nonfinite_metric:{receipt['run_id']}:{metric_name}"
                )
                continue
            values.append(value)
        if len(values) != len(receipts):
            continue
        tolerance = float(configured_tolerance)
        spread = max(values) - min(values)
        aggregates[metric_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "variation": spread,
            "tolerance": tolerance,
        }
        if spread > tolerance:
            violations.append(f"metric_variation:{metric_name}")
    return aggregates


def _policy_payload(
    *,
    policy: ReproducibilityPolicy,
    release_requirements_id: str,
) -> dict[str, object]:
    """Canonical reproducibility-policy payload bound into every report.

    The structure must stay byte-identical to
    ``ReproducibilityRequirements.to_payload`` so producer and release gate
    derive the same digest.
    """

    return {
        "policy_id": release_requirements_id,
        "seeds": sorted(int(seed) for seed in policy.seeds),
        "require_deterministic_execution": bool(
            policy.require_deterministic_execution
        ),
        "metric_tolerances": {
            str(name): float(tolerance)
            for name, tolerance in dict(policy.metric_tolerances).items()
        },
    }


def evaluate_reproducibility(
    *,
    receipts: Sequence[TrainingRunReceipt],
    policy: ReproducibilityPolicy,
    release_requirements_id: str,
) -> dict[str, object]:
    """Evaluate a completed multi-seed campaign against the release policy.

    This is the single release-decisive reproducibility evaluation. It is a
    pure function: it reads no files and writes nothing. The report is bound
    to the active release policy by ``release_requirements_id`` and a
    canonical policy digest, so evidence produced under one policy can never
    be accepted under another.
    """

    violations: list[str] = []
    for receipt in receipts:
        _validate_receipt(receipt)
    if not receipts:
        violations.append("no_run_receipts")

    expected_seed_values = tuple(int(seed) for seed in policy.seeds)
    expected_seeds = set(expected_seed_values)
    actual_seed_values = [int(receipt["seed"]) for receipt in receipts]
    actual_seeds = set(actual_seed_values)
    duplicate_seeds = sorted(
        seed for seed in actual_seeds if actual_seed_values.count(seed) != 1
    )
    duplicate_run_ids = sorted(
        run_id
        for run_id in {str(receipt["run_id"]) for receipt in receipts}
        if sum(str(receipt["run_id"]) == run_id for receipt in receipts) != 1
    )
    if (
        len(expected_seed_values) != len(expected_seeds)
        or len(receipts) != len(expected_seed_values)
        or actual_seeds != expected_seeds
        or duplicate_seeds
    ):
        violations.append("required_seeds_missing")
    violations.extend(
        f"duplicate_seed_receipt:{seed}" for seed in duplicate_seeds
    )
    violations.extend(
        f"duplicate_run_id:{run_id}" for run_id in duplicate_run_ids
    )

    for field in _IDENTITY_FIELDS:
        _require_same(receipts, field, violations)

    if policy.require_deterministic_execution:
        for receipt in receipts:
            if not _determinism_complete(receipt["determinism"]):
                violations.append(f"nondeterministic_run:{receipt['run_id']}")

    campaign_shape_valid = (
        not duplicate_seeds
        and not duplicate_run_ids
        and (
            len(receipts) == len(expected_seed_values)
            and actual_seeds == expected_seeds
            and len(expected_seed_values) == len(expected_seeds)
        )
    )
    metrics = (
        _compare_metrics(
            receipts=receipts,
            tolerances=dict(policy.metric_tolerances),
            violations=violations,
        )
        if campaign_shape_valid
        else {}
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "run_ids": [receipt["run_id"] for receipt in receipts],
        "seeds": sorted(actual_seed_values),
        "required_seeds": sorted(expected_seeds),
        "release_requirements_id": release_requirements_id,
        "release_requirements_sha256": stable_payload_fingerprint(
            _policy_payload(
                policy=policy,
                release_requirements_id=release_requirements_id,
            )
        ),
        "reproducibility_deterministic_required": bool(
            policy.require_deterministic_execution
        ),
        "reproducibility_metric_tolerances": {
            str(name): float(tolerance)
            for name, tolerance in dict(policy.metric_tolerances).items()
        },
        "experiment_sha256": (
            receipts[0]["experiment_sha256"] if receipts else None
        ),
        "dataset_manifest_sha256": (
            receipts[0]["dataset_manifest_sha256"] if receipts else None
        ),
        "run_receipts": [
            {
                "run_id": receipt["run_id"],
                "seed": int(receipt["seed"]),
                "sha256": stable_payload_fingerprint(dict(receipt)),
            }
            for receipt in receipts
        ],
        "metrics": metrics,
        "violations": sorted(set(violations)),
        "passed": not violations,
    }


def _validate_report(report: Mapping[str, object]) -> None:
    required = (
        "schema_version",
        "kind",
        "run_ids",
        "seeds",
        "required_seeds",
        "release_requirements_id",
        "release_requirements_sha256",
        "reproducibility_deterministic_required",
        "reproducibility_metric_tolerances",
        "experiment_sha256",
        "dataset_manifest_sha256",
        "run_receipts",
        "metrics",
        "violations",
        "passed",
    )
    if report.get("schema_version") != REPORT_SCHEMA_VERSION or any(
        key not in report for key in required
    ):
        raise ValueError("invalid reproducibility report schema")
    if report.get("kind") != REPORT_KIND:
        raise ValueError("report kind must be reproducibility_report")
    if not isinstance(report.get("violations"), list):
        raise ValueError("report violations must be a list")
    run_ids = report.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids:
        raise ValueError("report must reference at least one run receipt")
    run_count = len(run_ids)
    seeds = report.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != run_count:
        raise ValueError("report seeds must match run count")
    if not _exact_int_list(seeds):
        raise ValueError("report seeds must be integers")
    if not isinstance(report.get("passed"), bool):
        raise ValueError("report passed must be boolean")
    required_seeds = report.get("required_seeds")
    if not isinstance(required_seeds, list):
        raise ValueError("report required_seeds must be a list")
    if not _exact_int_list(required_seeds):
        raise ValueError("report required_seeds must be integers")
    if bool(report["passed"]) and report["violations"]:
        raise ValueError("passed report must not contain violations")
    for field in ("experiment_sha256", "dataset_manifest_sha256"):
        if not _sha256_string(report.get(field)):
            raise ValueError(f"report {field} must be SHA-256")

    run_receipts = report.get("run_receipts")
    if not isinstance(run_receipts, list) or not run_receipts:
        raise ValueError("report must bind run receipt digests")
    if len(run_receipts) != run_count:
        raise ValueError("report run receipt digests must match run count")
    entry_run_ids: list[str] = []
    for entry in run_receipts:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("run_id"), str)
            or not entry["run_id"].strip()
            or isinstance(entry.get("seed"), bool)
            or not isinstance(entry.get("seed"), int)
            or not _sha256_string(entry.get("sha256"))
        ):
            raise ValueError("report run receipt digest entries are invalid")
        entry_run_ids.append(str(entry["run_id"]))
    if sorted(entry_run_ids) != sorted(str(run_id) for run_id in run_ids):
        raise ValueError("report run receipt identities must match run_ids")

    release_requirements_id = report.get("release_requirements_id")
    if (
        not isinstance(release_requirements_id, str)
        or not release_requirements_id.strip()
    ):
        raise ValueError("report release_requirements_id must be non-empty")
    release_requirements_sha256 = report.get("release_requirements_sha256")
    if not _sha256_string(release_requirements_sha256):
        raise ValueError("report release_requirements_sha256 must be SHA-256")
    deterministic_required = report.get(
        "reproducibility_deterministic_required"
    )
    if not isinstance(deterministic_required, bool):
        raise ValueError(
            "report reproducibility_deterministic_required must be boolean"
        )
    tolerances = report.get("reproducibility_metric_tolerances")
    if not isinstance(tolerances, Mapping) or not all(
        isinstance(name, str)
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for name, value in tolerances.items()
    ):
        raise ValueError(
            "report reproducibility_metric_tolerances must be finite numbers"
        )

    declared_policy = {
        "policy_id": release_requirements_id,
        "seeds": sorted(int(seed) for seed in required_seeds),
        "require_deterministic_execution": deterministic_required,
        "metric_tolerances": {
            str(name): float(value) for name, value in tolerances.items()
        },
    }
    if stable_payload_fingerprint(declared_policy) != (
        release_requirements_sha256
    ):
        raise ValueError("report policy digest does not match declared policy")


def _exact_int_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _sha256_string(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.casefold()
    ):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def write_reproducibility_report(
    *,
    path: Path,
    report: Mapping[str, object],
) -> Path:
    """Atomically persist one multi-run reproducibility report."""

    _validate_report(report)
    _atomic_write_json(path=path, payload=report)
    return path


def load_reproducibility_report(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("reproducibility report root must be an object")
    payload = cast(dict[str, object], loaded)
    _validate_report(payload)
    return payload


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "REPORT_KIND",
    "REPORT_SCHEMA_VERSION",
    "RUN_RECEIPTS_SCHEMA_VERSION",
    "TrainingRunReceipt",
    "evaluate_reproducibility",
    "experiment_fingerprint",
    "initialization_policy_fingerprint",
    "load_reproducibility_report",
    "load_run_receipts_collection",
    "load_run_reproducibility_receipt",
    "model_state_fingerprint",
    "run_receipts_collection_payload",
    "stable_payload_fingerprint",
    "write_reproducibility_report",
    "write_run_receipts_collection",
    "write_run_reproducibility_receipt",
    "write_training_reproducibility_receipt",
]
