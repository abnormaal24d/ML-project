"""Regression guards for curated snapshot ownership and path contracts."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.path_resolution.project_paths import ProjectPaths
from crawler.curation.snapshots.dataset_assembly.curated_record_loader import (
    resolve_snapshot_directory,
    resolve_snapshot_root,
)
from crawler.storage.datasets.run_layout.dataset_path_layout import (
    build_snapshot_directory,
    snapshot_directory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_crawler_product_code_does_not_import_orchestration() -> None:
    violations: list[str] = []
    crawler_root = PROJECT_ROOT / "crawler"

    for path in crawler_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module,)

            for module in imported:
                if module == "orchestration" or module.startswith(
                    "orchestration."
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"{module}"
                    )

    assert violations == []


def test_snapshot_directory_uses_snapshot_id_and_has_no_side_effect(
    tmp_path: Path,
) -> None:
    path = snapshot_directory(
        project_root=tmp_path,
        base_output_directory="datasets",
        configured_subdirectory="multimodal",
        snapshot_id="snapshot-1",
    )

    assert path.relative_to(
        ProjectPaths(project_root=tmp_path).project_root
    ) == (Path("datasets/multimodal/snapshot-1"))
    assert not path.exists()


def test_curated_resolver_call_uses_exact_snapshot_id_contract(
    tmp_path: Path,
) -> None:
    dataset_paths = SimpleNamespace(
        curated_output_directory="curated",
        output_subdirectory="multimodal",
    )

    path = resolve_snapshot_directory(
        snapshot_directory_resolver=snapshot_directory,
        project_root=tmp_path,
        dataset_paths=dataset_paths,
        snapshot_id="snapshot-2",
    )

    assert path.relative_to(
        ProjectPaths(project_root=tmp_path).project_root
    ) == (Path("curated/multimodal/snapshot-2"))
    assert not path.exists()


def test_snapshot_directory_rejects_empty_identifier(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="snapshot_id must be a non-empty relative path"
    ):
        snapshot_directory(
            project_root=tmp_path,
            base_output_directory="datasets",
            configured_subdirectory="multimodal",
            snapshot_id="   ",
        )


def test_build_snapshot_directory_creates_resolved_directory(
    tmp_path: Path,
) -> None:
    path = build_snapshot_directory(
        project_root=tmp_path,
        base_output_directory="datasets",
        configured_subdirectory="multimodal",
        snapshot_id="snapshot-1",
    )

    assert path.is_dir()


def test_snapshot_root_resolution_creates_no_probe_directory(
    tmp_path: Path,
) -> None:
    root = resolve_snapshot_root(
        project_root=tmp_path,
        base_output_directory="curated",
        configured_subdirectory="multimodal",
    )

    assert root == tmp_path / "curated" / "multimodal"
    assert not root.exists()
    assert not (root / "__snapshot_probe__").exists()


def test_snapshot_factories_match_their_application_call_contracts() -> None:
    from orchestration.composition import curated_snapshot_services

    document_parameters = inspect.signature(
        curated_snapshot_services._document_curator
    ).parameters
    writer_parameters = inspect.signature(
        curated_snapshot_services._dataset_writer
    ).parameters

    assert "snapshot_directory" in document_parameters
    assert "snap_dir" not in document_parameters
    assert "snapshot_directory" in writer_parameters
    assert "snap_dir" not in writer_parameters


def test_snapshot_alignment_rows_preserve_page_chunk_and_timed_evidence() -> (
    None
):
    from crawler.curation.snapshots.alignment_rows import CuratedSnapshotRows

    document = SimpleNamespace(document_id="doc-1", title="Document title")
    prepared_text = "Opening text. Spoken evidence. Closing text."
    approved_text = {
        "body": prepared_text,
        "structure:page:0:text": "Opening text.",
    }
    prepared = SimpleNamespace(
        text=prepared_text,
        privacy_clearance=SimpleNamespace(
            permits_training=True,
            output_digest=hashlib.sha256(
                prepared_text.encode("utf-8")
            ).hexdigest(),
            approved_text=approved_text.get,
        ),
        pages=(
            SimpleNamespace(
                page_number=1,
                text_start=0,
                text_end=13,
                rendered_image_path=None,
            ),
        ),
    )
    page_rows = CuratedSnapshotRows.build_page(
        snapshot_id="snapshot-1",
        schema_version="v1",
        documents=(document,),
        preprocessed_documents_by_id={"doc-1": prepared},
    )
    assert page_rows[0]["page_number"] == 1
    assert page_rows[0]["text_span_start"] == 0
    assert page_rows[0]["text_span_end"] == 13
    assert page_rows[0]["rendered_image_path"] is None

    chunk = SimpleNamespace(
        document_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        quality_score=0.95,
        text="Opening text.",
        section_path=("Introduction",),
        start_char=0,
        end_char=13,
    )
    document_rows = CuratedSnapshotRows.build_document(
        snapshot_id="snapshot-1",
        schema_version="v1",
        documents=(document,),
        chunks=(chunk,),
    )
    assert document_rows[0]["alignment_type"] == "document_chunk"
    assert document_rows[0]["chunk_id"] == "chunk-1"
    assert document_rows[0]["text_span_end"] == 13

    segment = SimpleNamespace(
        text="Spoken evidence.",
        source="approved_transcript",
        confidence=0.9,
        start_seconds=1.25,
        end_seconds=2.5,
    )
    media = SimpleNamespace(
        media_id="audio-1",
        parent_document_id="doc-1",
        transcript_segments=(segment,),
        transcript_preview=None,
        transcript_text="Spoken evidence.",
        html_context=None,
        surrounding_text=None,
        context_score=0.8,
    )
    audio_rows = CuratedSnapshotRows.build_audio(
        snapshot_id="snapshot-1",
        rows=(media,),
        schema_version="v1",
        preprocessed_documents_by_id={"doc-1": prepared},
    )
    assert audio_rows[0]["timestamp_start"] == 1.25
    assert audio_rows[0]["timestamp_end"] == 2.5
    assert audio_rows[0]["text_span_start"] == 14
    assert audio_rows[0]["text_span_end"] == 30
