"""Release evidence bundle and artifact contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config.environment.default_values import DEFAULT_DATASET_CARD_FILENAME
from evaluator.leakage.report import load_report, violations_for
from evaluator.leakage.schema import LeakageReportV2
from release.release_utilities import (
    atomic_write_json,
    sha256,
)

ReleaseMode = Literal[
    "pipeline_smoke",
    "learning_candidate",
    "candidate",
    "production_model",
]

_REQUIRED_EVIDENCE = {
    "pipeline_smoke": frozenset(),
    "learning_candidate": frozenset({"leakage"}),
    "candidate": frozenset(
        {
            "checkpoint",
            "leakage",
            "dataset_card",
            "model_card",
            "reproducibility",
            "run_receipts",
            "training_metrics",
        }
    ),
    "production_model": frozenset(
        {
            "checkpoint",
            "leakage",
            "dataset_card",
            "model_card",
            "reproducibility",
            "run_receipts",
            "training_metrics",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    """Canonical locations produced by one candidate evaluation stage."""

    checkpoint: Path
    metrics: Path
    manifest: Path
    leakage: Path
    dataset_card: Path
    model_card: Path
    reproducibility: Path
    run_receipts: Path

    @classmethod
    def from_release_outputs(
        cls,
        *,
        candidate_directory: Path,
        dataset_directory: Path,
        checkpoint: Path,
        metrics: Path,
    ) -> EvaluationArtifacts:
        evaluation = candidate_directory / "evaluation"
        return cls(
            checkpoint=checkpoint,
            metrics=metrics,
            manifest=candidate_directory / "training_manifest.json",
            leakage=dataset_directory / "evaluation" / "leakage_report.json",
            dataset_card=dataset_directory / DEFAULT_DATASET_CARD_FILENAME,
            model_card=candidate_directory / "model_card.md",
            reproducibility=evaluation / "reproducibility_report.json",
            run_receipts=evaluation / "run_receipts.json",
        )

    def required_paths(self, mode: ReleaseMode) -> dict[str, Path]:
        candidates = {
            "checkpoint": self.checkpoint,
            "leakage": self.leakage,
            "dataset_card": self.dataset_card,
            "model_card": self.model_card,
            "reproducibility": self.reproducibility,
            "run_receipts": self.run_receipts,
            "training_metrics": self.metrics,
        }
        return {name: candidates[name] for name in _REQUIRED_EVIDENCE[mode]}


def parse_release_mode(value: str) -> ReleaseMode:
    if value == "pipeline_smoke":
        return "pipeline_smoke"
    if value == "learning_candidate":
        return "learning_candidate"
    if value == "candidate":
        return "candidate"
    if value == "production_model":
        return "production_model"
    raise ValueError(f"unsupported release mode: {value!r}")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    name: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.path.strip():
            raise ValueError("evidence name and path must be non-empty")
        if not Path(self.path).is_absolute():
            raise ValueError("release evidence path must be absolute")
        _require_sha256(self.sha256)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> EvidenceReference:
        if set(payload) != {"name", "path", "sha256"}:
            raise ValueError("release evidence reference fields are invalid")
        return cls(
            name=_required_string(payload, "name"),
            path=_required_string(payload, "path"),
            sha256=_required_string(payload, "sha256"),
        )


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceBundle:
    """Complete versioned evidence required before a release gate starts."""

    release_mode: ReleaseMode
    references: tuple[EvidenceReference, ...]
    leakage_report: LeakageReportV2 | None
    schema_version: str = "release_evidence.v1"
    release_requirements_id: str | None = None
    release_requirements_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "release_evidence.v1":
            raise ValueError("unsupported release evidence schema")
        if self.release_mode not in _REQUIRED_EVIDENCE:
            raise ValueError("unsupported release evidence mode")
        if not isinstance(self.references, tuple) or not all(
            isinstance(reference, EvidenceReference)
            for reference in self.references
        ):
            raise TypeError("release evidence references must be typed")
        if self.leakage_report is not None and not isinstance(
            self.leakage_report,
            LeakageReportV2,
        ):
            raise TypeError("release leakage evidence must be typed")
        for field_name, value in (
            ("release_requirements_id", self.release_requirements_id),
            ("release_requirements_sha256", self.release_requirements_sha256),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"release evidence {field_name} must be non-empty"
                )
        if self.release_requirements_sha256 is not None:
            _require_sha256(self.release_requirements_sha256)
        names = tuple(reference.name for reference in self.references)
        if len(names) != len(set(names)):
            raise ValueError("release evidence names must be unique")
        if frozenset(names) != _REQUIRED_EVIDENCE[self.release_mode]:
            raise ValueError("release evidence references are incomplete")
        if self.release_mode != "pipeline_smoke":
            if self.leakage_report is None:
                raise ValueError("release evidence lacks leakage report")
            if violations_for(self.leakage_report):
                raise ValueError("release evidence contains failed leakage")

    @classmethod
    def build(
        cls,
        *,
        mode: ReleaseMode,
        artifacts: EvaluationArtifacts,
        release_requirements_id: str | None = None,
        release_requirements_sha256: str | None = None,
    ) -> ReleaseEvidenceBundle:
        required = artifacts.required_paths(mode)
        missing = tuple(
            name for name, path in required.items() if not path.is_file()
        )
        if missing:
            raise FileNotFoundError(
                "release evidence is incomplete: " + ", ".join(missing)
            )
        references = tuple(
            EvidenceReference(
                name=name,
                path=path.resolve(strict=True).as_posix(),
                sha256=sha256(path),
            )
            for name, path in sorted(required.items())
        )
        leakage_report = (
            load_report(artifacts.leakage)
            if mode != "pipeline_smoke"
            else None
        )
        return cls(
            release_mode=mode,
            references=references,
            leakage_report=leakage_report,
            release_requirements_id=release_requirements_id,
            release_requirements_sha256=release_requirements_sha256,
        )

    @property
    def evidence_paths(self) -> dict[str, Path]:
        return {
            reference.name: Path(reference.path)
            for reference in self.references
        }

    def path(self, name: str) -> Path | None:
        for reference in self.references:
            if reference.name == name:
                return Path(reference.path)
        return None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_mode": self.release_mode,
            "evidence": [reference.to_dict() for reference in self.references],
            "leakage_schema_version": (
                self.leakage_report.schema_version
                if self.leakage_report is not None
                else None
            ),
            "release_requirements_id": self.release_requirements_id,
            "release_requirements_sha256": self.release_requirements_sha256,
            "passed": (
                self.leakage_report is None
                or not violations_for(self.leakage_report)
            ),
        }

    def write(self, path: Path) -> None:
        """Atomically persist the complete evidence-bundle manifest."""
        atomic_write_json(path=path, payload=self.to_payload())

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        approved_roots: tuple[Path, ...],
    ) -> ReleaseEvidenceBundle:
        """Load one complete bundle and its canonical leakage evidence."""

        roots = tuple(root.resolve(strict=True) for root in approved_roots)
        if not roots:
            raise ValueError("release evidence requires approved roots")
        path = _contained_evidence_file(path, approved_roots=roots)
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("release evidence bundle exceeds size limit")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid release evidence bundle") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("release evidence bundle root must be an object")
        if set(payload) != {
            "schema_version",
            "release_mode",
            "evidence",
            "leakage_schema_version",
            "release_requirements_id",
            "release_requirements_sha256",
            "passed",
        }:
            raise ValueError("release evidence bundle fields are invalid")
        if payload.get("schema_version") != "release_evidence.v1":
            raise ValueError("unsupported release evidence schema")
        raw_references = payload.get("evidence")
        if not isinstance(raw_references, list):
            raise ValueError("release evidence references must be a list")
        references = tuple(
            EvidenceReference.from_mapping(reference)
            for reference in raw_references
            if isinstance(reference, Mapping)
        )
        if len(references) != len(raw_references):
            raise ValueError("release evidence contains an invalid reference")
        references = tuple(
            EvidenceReference(
                name=reference.name,
                path=_contained_evidence_file(
                    Path(reference.path),
                    approved_roots=roots,
                ).as_posix(),
                sha256=reference.sha256,
            )
            for reference in references
        )
        for reference in references:
            if _sha256(Path(reference.path)) != reference.sha256:
                raise ValueError(
                    f"release evidence digest mismatch: {reference.name}"
                )
        mode = parse_release_mode(_required_string(payload, "release_mode"))
        release_requirements_id = payload.get("release_requirements_id")
        release_requirements_sha256 = payload.get(
            "release_requirements_sha256"
        )
        if mode in {"candidate", "production_model"}:
            if (
                not isinstance(release_requirements_id, str)
                or not release_requirements_id.strip()
            ):
                raise ValueError(
                    "release evidence requires release policy identity"
                )
            if not isinstance(release_requirements_sha256, str):
                raise ValueError(
                    "release evidence requires release policy digest"
                )
            _require_sha256(release_requirements_sha256)
        elif release_requirements_id is not None or (
            release_requirements_sha256 is not None
        ):
            raise ValueError(
                "release evidence policy identity is invalid for this mode"
            )
        leakage_reference = next(
            (
                reference
                for reference in references
                if reference.name == "leakage"
            ),
            None,
        )
        leakage_report = (
            load_report(Path(leakage_reference.path))
            if leakage_reference is not None
            else None
        )
        bundle = cls(
            release_mode=mode,
            references=references,
            leakage_report=leakage_report,
            release_requirements_id=release_requirements_id,
            release_requirements_sha256=release_requirements_sha256,
        )
        if payload.get("passed") is not True:
            raise ValueError("release evidence bundle is not passed")
        expected_leakage_schema = (
            leakage_report.schema_version
            if leakage_report is not None
            else None
        )
        if payload.get("leakage_schema_version") != expected_leakage_schema:
            raise ValueError("release evidence leakage schema mismatch")
        return bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_evidence_file(
    path: Path,
    *,
    approved_roots: tuple[Path, ...],
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("release evidence file is unavailable") from exc
    if not resolved.is_file():
        raise ValueError("release evidence reference must be a file")
    for root in approved_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    raise ValueError("release evidence reference escapes approved roots")


def _require_sha256(value: str) -> None:
    if len(value) != 64:
        raise ValueError("evidence digest must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("evidence digest must be SHA-256") from exc


def _required_string(
    payload: Mapping[str, object],
    name: str,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"release evidence {name} must be non-empty")
    return value


__all__ = [
    "EvaluationArtifacts",
    "EvidenceReference",
    "ReleaseEvidenceBundle",
    "ReleaseMode",
    "parse_release_mode",
]
