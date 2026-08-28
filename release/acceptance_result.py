"""Durable reporting for training acceptance decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from release.release_decision import ReleaseDecision, ReleaseStatus
from release.release_evidence_bundle import (
    EvidenceReference,
    ReleaseEvidenceBundle,
    parse_release_mode,
)
from release.release_utilities import atomic_write_json

_SCHEMA_VERSION = "1.0"
_EXPECTED_STATUS_BY_STAGE = {
    "pipeline_smoke": ReleaseStatus.PIPELINE_ACCEPTED,
    "learning_candidate": ReleaseStatus.DATASET_ACCEPTED,
    "candidate": ReleaseStatus.MODEL_CANDIDATE,
    "production_model": ReleaseStatus.MODEL_ACCEPTED,
}


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    schema_version: str
    release_stage: str
    decision: dict[str, object]
    expected_status: str
    violations: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]

    @classmethod
    def build(
        cls,
        *,
        release_stage: str,
        decision: ReleaseDecision,
        expected_status: ReleaseStatus,
        evidence_paths: Mapping[str, Path],
    ) -> AcceptanceReport:
        references: list[EvidenceReference] = []
        violations = list(decision.reasons)
        for name, path in sorted(evidence_paths.items()):
            if not path.is_file():
                violations.append(f"missing_evidence:{name}")
                continue
            references.append(
                EvidenceReference(
                    name=name,
                    path=path.as_posix(),
                    sha256=_sha256(path),
                )
            )
        status = decision.status if not violations else ReleaseStatus.FAILED
        unique_violations = tuple(dict.fromkeys(violations))
        expected_for_stage = _expected_status_for_stage(release_stage)
        if expected_status is not expected_for_stage:
            raise ValueError(
                "acceptance report expected status does not match release stage"
            )
        return cls(
            schema_version=_SCHEMA_VERSION,
            release_stage=release_stage,
            decision={
                "status": status.value,
                "reasons": list(unique_violations),
            },
            expected_status=expected_status.value,
            violations=unique_violations,
            evidence=tuple(references),
        )

    def to_payload(self) -> dict[str, object]:
        passed = (
            self.decision["status"] == self.expected_status
            and not self.violations
        )
        return {
            "schema_version": self.schema_version,
            "release_stage": self.release_stage,
            "decision": self.decision,
            "expected_status": self.expected_status,
            "violations": list(self.violations),
            "evidence": [reference.to_dict() for reference in self.evidence],
            "passed": passed,
        }

    def write(self, path: Path) -> None:
        atomic_write_json(path=path, payload=self.to_payload())

    @classmethod
    def load(cls, path: Path) -> AcceptanceReport:
        """Load and validate one persisted acceptance report."""

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid acceptance report: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("acceptance report root must be an object")
        if set(payload) != {
            "schema_version",
            "release_stage",
            "decision",
            "expected_status",
            "violations",
            "evidence",
            "passed",
        }:
            raise ValueError("acceptance report fields are invalid")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported acceptance report schema")

        release_stage = _required_string(payload, "release_stage")
        expected_status = _release_status(payload, "expected_status")
        if expected_status is not _expected_status_for_stage(release_stage):
            raise ValueError(
                "acceptance report expected status does not match release stage"
            )

        decision = payload["decision"]
        if not isinstance(decision, Mapping) or set(decision) != {
            "status",
            "reasons",
        }:
            raise ValueError("acceptance report decision is invalid")
        decision_status = _release_status(decision, "status")
        reasons = _string_list(decision, "reasons")
        violations = _string_list(payload, "violations")
        raw_evidence = payload["evidence"]
        if not isinstance(raw_evidence, list):
            raise ValueError("acceptance report evidence is invalid")
        try:
            evidence = tuple(
                EvidenceReference.from_mapping(raw_reference)
                for raw_reference in raw_evidence
                if isinstance(raw_reference, Mapping)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("acceptance report evidence is invalid") from exc
        if len(evidence) != len(raw_evidence):
            raise ValueError("acceptance report evidence is invalid")
        report = cls(
            schema_version=_SCHEMA_VERSION,
            release_stage=release_stage,
            decision={
                "status": decision_status.value,
                "reasons": list(reasons),
            },
            expected_status=expected_status.value,
            violations=violations,
            evidence=evidence,
        )
        if payload.get("passed") is not True or report.to_payload() != payload:
            raise ValueError("persisted acceptance report is not passed")
        return report


@dataclass(frozen=True, slots=True)
class TrainingAcceptanceResult:
    """Decision and durable evidence produced before manifest persistence."""

    decision: ReleaseDecision
    acceptance_report: AcceptanceReport
    acceptance_report_path: Path
    evidence_bundle: ReleaseEvidenceBundle | None = None
    evidence_bundle_path: Path | None = None

    def evidence_paths(self, *, checkpoint_path: Path) -> dict[str, Path]:
        """Return every persisted evidence path keyed by its stable name."""

        paths = {"checkpoint": checkpoint_path}
        if self.evidence_bundle is not None:
            paths.update(self.evidence_bundle.evidence_paths)
        if self.evidence_bundle_path is not None:
            paths["release_evidence_bundle"] = self.evidence_bundle_path
        return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"acceptance report {name} must be non-empty")
    return value


def _expected_status_for_stage(release_stage: str) -> ReleaseStatus:
    parse_release_mode(release_stage)
    return _EXPECTED_STATUS_BY_STAGE[release_stage]


def _release_status(
    payload: Mapping[str, object],
    name: str,
) -> ReleaseStatus:
    try:
        return ReleaseStatus(_required_string(payload, name))
    except ValueError as exc:
        raise ValueError(f"acceptance report {name} is invalid") from exc


def _string_list(
    payload: Mapping[str, object],
    name: str,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"acceptance report {name} is invalid")
    values = tuple(value)
    if len(values) != len(set(values)):
        raise ValueError(f"acceptance report {name} contains duplicates")
    return values


__all__ = ["AcceptanceReport", "TrainingAcceptanceResult"]
