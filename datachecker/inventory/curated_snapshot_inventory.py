"""Curated snapshot artifact inventory reader."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from config.path_resolution.project_paths import validate_safe_relative_path
from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from schemas.versions import CURATED_DATASET_SCHEMA_VERSION

DeadlineCheckpoint = Callable[[str], None]


def no_deadline_checkpoint(stage: str) -> None:
    pass


if TYPE_CHECKING:
    from pathlib import Path

    from config.collection.training_input_gate import DataCheckerSettings
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import DatasetFingerprintCalculator


@dataclass(slots=True, frozen=True)
class CuratedInventory:
    """Resolved curated snapshot artifacts."""

    directory: Path | None
    manifest_path: Path | None
    fingerprint: str | None
    document_count: int
    chunk_count: int
    image_count: int
    audio_count: int
    video_count: int
    alignment_count: int
    rejected_document_count: int
    rejected_image_count: int
    rejected_audio_count: int
    rejected_video_count: int
    image_coverage: dict[str, object]
    audio_coverage: dict[str, object]
    video_coverage: dict[str, object]
    schema_valid: bool
    curated_document_modality_counts: dict[str, int] = field(
        default_factory=dict
    )
    rejected_document_by_reason: dict[str, int] = field(default_factory=dict)


class CuratedInventoryReader:
    """Resolve the latest curated snapshot outputs."""

    def __init__(
        self,
        *,
        settings: DataCheckerSettings,
        artifact_path_registry: ArtifactPathRegistry,
        dataset_fingerprint_calculator: DatasetFingerprintCalculator,
    ) -> None:
        self._settings = settings
        self._artifact_path_registry = artifact_path_registry
        self._dataset_fingerprint_calculator = dataset_fingerprint_calculator

    def read(
        self,
        *,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> CuratedInventory:
        """Build curated snapshot inventory."""
        manifest_path = self._latest_named_file(
            root=self._artifact_path_registry.curated_root(),
            filename=self._artifact_path_registry.dataset_paths.snapshot_manifest_filename,
            checkpoint=checkpoint,
            stage="curated_snapshot_scan",
        )

        directory = manifest_path.parent if manifest_path is not None else None
        payload = {}
        if manifest_path and manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        documents_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="documents_path",
        )
        chunks_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="chunks_path",
        )
        images_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="images_path",
        )
        audio_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="audio_path",
        )
        video_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="video_path",
        )
        alignments_path = self._manifest_path(
            directory=directory,
            manifest_payload=payload,
            key="alignments_path",
        )
        schema_valid = self._curated_schema_valid(
            directory=directory,
            manifest_path=manifest_path,
            documents_path=documents_path,
            chunks_path=chunks_path,
            images_path=images_path,
            audio_path=audio_path,
            video_path=video_path,
            alignments_path=alignments_path,
            manifest_payload=payload,
        )
        if not schema_valid:
            return self._empty_inventory()
        image_coverage = self._coverage_dict(payload.get("image_coverage"))
        audio_coverage = self._coverage_dict(payload.get("audio_coverage"))
        video_coverage = self._coverage_dict(payload.get("video_coverage"))

        return CuratedInventory(
            directory=directory,
            manifest_path=manifest_path,
            fingerprint=self._calculate_fingerprint(
                tuple(
                    p
                    for p in (
                        manifest_path,
                        documents_path,
                        chunks_path,
                        images_path,
                        audio_path,
                        video_path,
                        alignments_path,
                    )
                    if p and p.exists()
                ),
                root=directory,
                checkpoint=checkpoint,
            ),
            document_count=int(payload.get("documents") or 0),
            chunk_count=int(payload.get("chunks") or 0),
            image_count=int(payload.get("images") or 0),
            audio_count=int(payload.get("audio") or 0),
            video_count=int(payload.get("video") or 0),
            alignment_count=int(payload.get("alignments") or 0),
            rejected_document_count=self._count_rejected_documents(
                documents_path=documents_path,
                checkpoint=checkpoint,
            ),
            rejected_image_count=self._rejected_count_from_coverage(
                coverage=image_coverage,
            ),
            rejected_audio_count=self._rejected_count_from_coverage(
                coverage=audio_coverage,
            ),
            rejected_video_count=self._rejected_count_from_coverage(
                coverage=video_coverage,
            ),
            image_coverage=image_coverage,
            audio_coverage=audio_coverage,
            video_coverage=video_coverage,
            schema_valid=True,
            curated_document_modality_counts=self._document_modality_counts(
                documents_path=documents_path,
                checkpoint=checkpoint,
            ),
            rejected_document_by_reason=self._rejected_document_by_reason(
                documents_path=documents_path
            ),
        )

    @staticmethod
    def _empty_inventory() -> CuratedInventory:
        return CuratedInventory(
            directory=None,
            manifest_path=None,
            fingerprint=None,
            document_count=0,
            chunk_count=0,
            image_count=0,
            audio_count=0,
            video_count=0,
            alignment_count=0,
            rejected_document_count=0,
            rejected_image_count=0,
            rejected_audio_count=0,
            rejected_video_count=0,
            image_coverage={},
            audio_coverage={},
            video_coverage={},
            schema_valid=False,
            curated_document_modality_counts={},
            rejected_document_by_reason={},
        )

    def _calculate_fingerprint(
        self,
        paths: tuple[Path, ...],
        *,
        root: Path | None,
        checkpoint: DeadlineCheckpoint,
    ) -> str | None:
        if not paths:
            return None
        return self._dataset_fingerprint_calculator.calculate(
            paths=paths,
            root=root,
            checkpoint=checkpoint,
        )

    @staticmethod
    def _count_rejected_documents(
        *,
        documents_path: Path | None,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> int:
        if documents_path is None or not documents_path.exists():
            return 0

        rejected = 0

        with documents_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number % 256 == 0:
                    checkpoint("curated_rejected_documents_scan")
                stripped = line.strip()

                if not stripped:
                    continue
                if '"quality_bucket"' not in stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                if not isinstance(payload, dict):
                    continue

                quality_bucket = (
                    str(payload.get("quality_bucket", "")).strip().lower()
                )

                if quality_bucket == "reject":
                    rejected += 1

        return rejected

    @staticmethod
    def _coverage_dict(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _rejected_count_from_coverage(
        *,
        coverage: dict[str, object],
    ) -> int:
        rejected_by_reason = coverage.get("rejected_by_reason")
        if not isinstance(rejected_by_reason, dict):
            return 0
        return sum(int(value or 0) for value in rejected_by_reason.values())

    @staticmethod
    def _latest_named_file(
        *,
        root: Path,
        filename: str,
        checkpoint: DeadlineCheckpoint,
        stage: str,
    ) -> Path | None:
        if not root.exists() or not root.is_dir():
            return None
        matches = []
        for i, p in enumerate(root.rglob(filename)):
            if i % 256 == 0:
                checkpoint(stage)
            if p.is_file():
                matches.append(p)
        if not matches:
            return None
        return sorted(matches, key=lambda p: (p.parent.name, p.name))[-1]

    @staticmethod
    def _manifest_path(
        *,
        directory: Path | None,
        manifest_payload: dict[str, object],
        key: str,
    ) -> Path | None:
        if directory is None:
            return None
        value = manifest_payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return None

        try:
            relative_path = validate_safe_relative_path(
                value,
                field_name=key,
            )
        except ValueError:
            return None

        path = (directory / relative_path).resolve()

        if not path.is_relative_to(directory.resolve()):
            return None

        return path

    def _document_modality_counts(
        self,
        *,
        documents_path: Path | None,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> dict[str, int]:
        if documents_path is None or not documents_path.exists():
            return {}
        counts: dict[str, int] = {}
        try:
            with documents_path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if line_number % 256 == 0:
                        checkpoint("curated_documents_scan")
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            mod = str(
                                rec.get("modality") or "document"
                            ).lower()
                            counts[mod] = counts.get(mod, 0) + 1
                        except (
                            json.JSONDecodeError,
                            TypeError,
                            KeyError,
                            AttributeError,
                            ValueError,
                        ):  # exception-rules: best-effort-cleanup
                            counts["document"] = counts.get("document", 0) + 1
        except (OSError, UnicodeError):
            return counts
        return counts

    def _rejected_document_by_reason(
        self, *, documents_path: Path | None
    ) -> dict[str, int]:
        # Structured split per reason for rejected documents.
        # Real implementation should scan rejection manifests/sidecars.
        if documents_path is None or not documents_path.exists():
            return {}
        return {}

    @staticmethod
    def _curated_schema_valid(
        *,
        directory: Path | None,
        manifest_path: Path | None,
        documents_path: Path | None,
        chunks_path: Path | None,
        images_path: Path | None,
        audio_path: Path | None,
        video_path: Path | None,
        alignments_path: Path | None,
        manifest_payload: dict[str, object],
    ) -> bool:
        if directory is None or manifest_path is None:
            return False
        if not directory.exists() or not manifest_path.is_file():
            return False
        if manifest_payload.get("final") is not True:
            return False
        if manifest_payload.get("lifecycle_stage") != "curated":
            return False
        if manifest_payload.get("immutable") is not True:
            return False
        status = str(manifest_payload.get("status") or "").strip()
        if status != "completed":
            return False
        snapshot_id = manifest_payload.get("snapshot_id")
        if not isinstance(snapshot_id, str) or snapshot_id != directory.name:
            return False
        schema_version = manifest_payload.get("schema_version")
        if (
            not isinstance(schema_version, str)
            or schema_version != CURATED_DATASET_SCHEMA_VERSION
        ):
            return False

        artifacts = (
            (
                "documents",
                documents_path,
                CuratedDocumentRecord.from_dict,
                True,
            ),
            ("chunks", chunks_path, ChunkRecord.from_dict, True),
            ("images", images_path, CuratedImageRecord.from_dict, False),
            ("audio", audio_path, CuratedAudioRecord.model_validate, False),
            ("video", video_path, CuratedVideoRecord.model_validate, False),
            ("alignments", alignments_path, _validate_alignment_row, False),
        )
        for count_field, path, parser, required_path in artifacts:
            expected_count = _manifest_count(manifest_payload.get(count_field))
            if expected_count is None:
                return False
            if path is None:
                if required_path or expected_count != 0:
                    return False
                continue
            if not path.is_file():
                return False
            actual_count = _validated_jsonl_count(
                path=path,
                parser=parser,
                snapshot_id=snapshot_id,
                schema_version=schema_version,
            )
            if actual_count is None or actual_count != expected_count:
                return False
        return True


def _manifest_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validated_jsonl_count(
    *,
    path: Path,
    parser: Callable[[Mapping[str, object]], object],
    snapshot_id: str,
    schema_version: str,
) -> int | None:
    """Return a line count only when every JSONL row has a valid schema."""

    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, Mapping):
                    return None
                if row.get("schema_version") != schema_version:
                    return None
                record = parser(row)
                if getattr(record, "snapshot_id", None) != snapshot_id:
                    return None
                count += 1
    except (
        OSError,
        UnicodeError,
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return None
    return count


@dataclass(frozen=True, slots=True)
class _AlignmentRow:
    snapshot_id: str


def _validate_alignment_row(row: Mapping[str, object]) -> _AlignmentRow:
    """Validate the durable identifiers shared by every alignment record."""

    required_text = (
        "schema_version",
        "snapshot_id",
        "alignment_id",
        "object_modality",
        "object_id",
        "alignment_type",
        "relation_type",
    )
    values: dict[str, str] = {}
    for name in required_text:
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("invalid curated alignment row")
        values[name] = value.strip()
    if values["object_modality"].casefold() not in {
        "image",
        "audio",
        "video",
    }:
        raise ValueError("invalid curated alignment modality")
    return _AlignmentRow(snapshot_id=values["snapshot_id"])
