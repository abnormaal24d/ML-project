"""Load raw crawl records and resolve reusable curated snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from crawler.curation.snapshots.dataset_assembly.curated_snapshot_fingerprint import (
    build_curation_input_fingerprint,
)
from crawler.storage.datasets.run_layout.dataset_path_layout import (
    output_directory,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from pathlib import Path

    from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
        SnapshotDirectoryResolver,
    )


@dataclass(frozen=True, slots=True)
class CuratedRawRecordSet:
    """Filtered raw crawl entries and derived snapshot metadata."""

    raw_entries: tuple[Any, ...]
    source_run_ids: tuple[str, ...]
    content_fingerprint: str


class ReusedAssemblyCounts(TypedDict):
    """Typed counts loaded from a reusable curated snapshot manifest."""

    documents: int
    chunks: int
    images: int
    audio: int
    video: int
    alignments: int
    source_run_ids: tuple[str, ...]


CURATED_INPUT_KINDS = frozenset(
    {"page", "image", "audio", "video", "document", "feed"}
)


def load_curated_raw_records(
    *,
    raw_manifest_reader: Any,
    relevant_kinds: frozenset[str],
    settings_payload: dict[str, Any],
) -> CuratedRawRecordSet:
    raw_entries = tuple(
        entry
        for entry in raw_manifest_reader.iter_all()
        if entry.record.kind in relevant_kinds
    )
    source_run_ids = tuple(
        sorted(
            {
                entry.record.run_id
                for entry in raw_entries
                if entry.record.run_id
            }
        )
    )
    content_fingerprint = build_curation_input_fingerprint(
        raw_entries=raw_entries,
        settings_payload=settings_payload,
        relevant_kinds=relevant_kinds,
    )
    return CuratedRawRecordSet(
        raw_entries=raw_entries,
        source_run_ids=source_run_ids,
        content_fingerprint=content_fingerprint,
    )


def find_reusable_snapshot_id(
    *,
    snapshot_root: Path,
    snapshot_manifest_filename: str,
    content_fingerprint: str,
) -> str | None:
    if not snapshot_root.exists():
        return None
    matches: list[tuple[float, str]] = []
    for manifest_path in snapshot_root.glob(f"*/{snapshot_manifest_filename}"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("content_fingerprint") != content_fingerprint:
            continue
        if payload.get("final") is not True:
            continue
        if str(payload.get("status") or "") != "completed":
            continue
        matches.append(
            (manifest_path.stat().st_mtime, manifest_path.parent.name)
        )
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def build_reused_assembly_counts(
    *,
    manifest_path: Path,
) -> ReusedAssemblyCounts:
    payload = read_json_object(path=manifest_path)
    return {
        "documents": payload_int(payload=payload, key="documents"),
        "chunks": payload_int(payload=payload, key="chunks"),
        "images": payload_int(payload=payload, key="images"),
        "audio": payload_int(payload=payload, key="audio"),
        "video": payload_int(payload=payload, key="video"),
        "alignments": payload_int(payload=payload, key="alignments"),
        "source_run_ids": tuple(
            str(item)
            for item in payload.get("source_run_ids", ())
            if str(item).strip()
        ),
    }


def resolve_snapshot_directory(
    *,
    snapshot_directory_resolver: SnapshotDirectoryResolver,
    project_root: Path,
    dataset_paths: Any,
    snapshot_id: str,
) -> Path:
    return snapshot_directory_resolver(
        project_root=project_root,
        base_output_directory=dataset_paths.curated_output_directory,
        configured_subdirectory=dataset_paths.output_subdirectory,
        snapshot_id=snapshot_id,
    )


def build_reused_assembly_result(
    *,
    logger: ProjectLogger,
    snapshot_directory_resolver: SnapshotDirectoryResolver,
    project_root: Path,
    dataset_paths: Any,
    snapshot_id: str,
    content_fingerprint: str,
) -> Any:
    from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
        CuratedAssemblyResult,
    )

    snapshot_directory = resolve_snapshot_directory(
        snapshot_directory_resolver=snapshot_directory_resolver,
        project_root=project_root,
        dataset_paths=dataset_paths,
        snapshot_id=snapshot_id,
    )
    manifest_path = (
        snapshot_directory / dataset_paths.snapshot_manifest_filename
    )
    counts = build_reused_assembly_counts(manifest_path=manifest_path)
    logger.info(
        "curated_snapshot_reused",
        snapshot_id=snapshot_id,
        snapshot_directory=snapshot_directory.as_posix(),
        content_fingerprint=content_fingerprint,
    )
    return CuratedAssemblyResult(
        snapshot_id=snapshot_id,
        snapshot_directory=snapshot_directory,
        documents=int(counts["documents"]),
        chunks=int(counts["chunks"]),
        images=int(counts["images"]),
        audio=int(counts["audio"]),
        video=int(counts["video"]),
        alignments=int(counts["alignments"]),
        source_run_ids=tuple(counts["source_run_ids"]),
    )


def resolve_snapshot_root(
    *,
    project_root: Path,
    base_output_directory: str,
    configured_subdirectory: str | None,
) -> Path:
    """Resolve the snapshot collection root without creating probe paths."""

    return output_directory(
        root=project_root / base_output_directory,
        configured_subdirectory=configured_subdirectory,
    )


def count_raw_kinds(
    *,
    raw_entries: tuple[Any, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in raw_entries:
        kind = str(getattr(entry.record, "kind", "") or "").strip()
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def read_json_object(*, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def payload_int(*, payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0
