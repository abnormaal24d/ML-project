"""RawManifestEntry to PreprocessingInput contract tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from crawler.curation.ingest.schema.entry import RawManifestEntry
from crawler.curation.ingest.schema.record import RawManifestRecord
from crawler.curation.preprocessing_input_builder import (
    RawPayloadArtifactError,
    build_preprocessing_inputs,
)
from crawler.curation.snapshots.dataset_assembly.curated_sample_pairer import (
    build_curated_samples,
)
from crawler.storage.datasets.extraction.page_extraction_artifact import (
    PAGE_EXTRACTION_SCHEMA_VERSION,
    PageExtractionArtifact,
    PageExtractionArtifactError,
    PageExtractionArtifactWriter,
)
from preprocessing.preprocessing_input import LanguageEvidence


def _manifest_payload(
    *,
    kind: str,
    name: str,
    enrichment: dict[str, object] | None = None,
    language: object = "en",
    language_confidence: object = 0.97,
    language_source: object = "fasttext",
    language_detector_version: object = "crawler-language-v1",
    asset_metadata_only: bool = False,
) -> dict[str, object]:
    modality = "document" if kind in {"page", "document"} else kind
    mime_type = {
        "page": "text/html",
        "document": "text/plain",
        "image": "image/test",
        "audio": "audio/test",
        "video": "video/test",
    }[kind]
    return {
        "schema_version": "3.0",
        "run_id": "run-1",
        "fetch_record_id": f"id-{name}",
        "object_id": f"object-{name}",
        "requested_url": f"https://example.test/{name}",
        "final_url": f"https://example.test/{name}",
        "normalized_url": f"https://example.test/{name}",
        "parent_url": None,
        "kind": kind,
        "modality": modality,
        "depth": 0,
        "source_type": "registry",
        "status_code": 200,
        "content_type": mime_type,
        "mime_type": mime_type,
        "encoding": "utf-8",
        "language": language,
        "language_confidence": language_confidence,
        "language_source": language_source,
        "language_detector_version": language_detector_version,
        "content_sha256": "a" * 64,
        "byte_size": 7,
        "observed_bytes": 7,
        "storage_relative_path": name,
        "domain": "example.test",
        "path": f"/{name}",
        "query": None,
        "extension": Path(name).suffix,
        "fetched_at": "2026-07-19T00:00:00+00:00",
        "enrichment": enrichment or {},
        "asset_context": {},
        "asset_metadata_only": asset_metadata_only,
    }


def _entry(
    *,
    tmp_path: Path,
    kind: str,
    name: str,
    write_file: bool = True,
    content: bytes = b"payload",
    enrichment: dict[str, object] | None = None,
    language: object = "en",
    language_confidence: object = 0.97,
    language_source: object = "fasttext",
    language_detector_version: object = "crawler-language-v1",
    asset_metadata_only: bool = False,
) -> RawManifestEntry:
    if write_file:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    payload = _manifest_payload(
        kind=kind,
        name=name,
        enrichment=enrichment,
        language=language,
        language_confidence=language_confidence,
        language_source=language_source,
        language_detector_version=language_detector_version,
        asset_metadata_only=asset_metadata_only,
    )
    payload.update(
        {
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "byte_size": len(content),
            "observed_bytes": len(content),
        }
    )
    record = RawManifestRecord.from_payload(payload)
    return RawManifestEntry(run_directory=tmp_path, record=record)


def _page_entry(
    *,
    tmp_path: Path,
    name: str = "page.html",
    enrichment_overrides: dict[str, object] | None = None,
    commit_sidecar: bool = True,
    language: object = "en",
    language_confidence: object = 0.97,
    language_source: object = "fasttext",
    language_detector_version: object = "crawler-language-v1",
) -> RawManifestEntry:
    fetch_record_id = f"id-{name}"
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(
        fetch_record_id=fetch_record_id,
        artifact=PageExtractionArtifact(
            schema_version=PAGE_EXTRACTION_SCHEMA_VERSION,
            text="Page text",
            markdown="# Page text",
            headings=("Page text",),
            code_block_count=0,
            boilerplate_ratio=0.0,
            extraction_warnings=(),
            title="Page title",
            canonical_url=None,
        ),
    )
    if commit_sidecar:
        writer.commit(prepared=prepared)
    enrichment: dict[str, object] = {
        "page_extraction_path": prepared.relative_path,
        "page_extraction_sha256": prepared.sha256,
        "page_extraction_schema_version": PAGE_EXTRACTION_SCHEMA_VERSION,
    }
    enrichment.update(enrichment_overrides or {})
    return _entry(
        tmp_path=tmp_path,
        kind="page",
        name=name,
        content=b"<html></html>",
        enrichment=enrichment,
        language=language,
        language_confidence=language_confidence,
        language_source=language_source,
        language_detector_version=language_detector_version,
    )


def test_build_inputs_for_documents_and_media(tmp_path: Path) -> None:
    entries = (
        _entry(tmp_path=tmp_path, kind="image", name="a.jpg"),
        _entry(
            tmp_path=tmp_path,
            kind="audio",
            name="b.wav",
            enrichment={
                "audio_duration_seconds": 3.5,
                "transcript_text": "hi",
            },
        ),
        _entry(
            tmp_path=tmp_path,
            kind="video",
            name="c.mp4",
            enrichment={"video_width": 640, "video_height": 360},
        ),
        _entry(
            tmp_path=tmp_path,
            kind="document",
            name="d.txt",
            content=b"document text",
        ),
    )

    inputs = build_preprocessing_inputs(
        raw_entries=entries,
        max_input_bytes=25_000_000,
    )

    assert {item.modality for item in inputs} == {
        "document",
        "image",
        "audio",
        "video",
    }
    audio = next(item for item in inputs if item.modality == "audio")
    assert audio.duration_seconds == 3.5
    assert audio.transcript_text == "hi"
    video = next(item for item in inputs if item.modality == "video")
    assert video.width == 640
    assert video.height == 360


def test_missing_media_file_skipped_unless_metadata_only(
    tmp_path: Path,
) -> None:
    missing = _entry(
        tmp_path=tmp_path,
        kind="image",
        name="missing.jpg",
        write_file=False,
    )
    metadata_only = _entry(
        tmp_path=tmp_path,
        kind="audio",
        name="meta.wav",
        write_file=False,
        asset_metadata_only=True,
    )

    inputs = build_preprocessing_inputs(
        raw_entries=(missing, metadata_only),
        max_input_bytes=25_000_000,
    )

    assert len(inputs) == 1
    assert inputs[0].modality == "audio"
    assert inputs[0].media_path is None


def test_manifest_payload_happy_path_is_verified(tmp_path: Path) -> None:
    content = b"manifest authenticated payload"
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="objects/verified.txt",
        content=content,
    )

    [item] = build_preprocessing_inputs(
        raw_entries=(entry,),
        max_input_bytes=25_000_000,
    )

    assert item.ocr_text == content.decode("utf-8")
    assert item.media_path == str(
        (tmp_path / "objects/verified.txt").resolve()
    )


def test_builder_rejects_traversal_even_for_direct_record_construction(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "README.md"
    outside.write_bytes(b"outside run")
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="inside.txt",
    )
    malicious = RawManifestEntry(
        run_directory=tmp_path,
        record=replace(
            entry.record,
            storage_relative_path="../../README.md",
            byte_size=outside.stat().st_size,
            content_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        ),
    )

    with pytest.raises(RawPayloadArtifactError, match="safely contained"):
        build_preprocessing_inputs(
            raw_entries=(malicious,),
            max_input_bytes=25_000_000,
        )


def test_builder_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-payload.txt"
    outside.write_bytes(b"outside run")
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="objects/link.txt",
        write_file=False,
        content=outside.read_bytes(),
    )
    link = tmp_path / "objects/link.txt"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable on this platform: {exc}")

    with pytest.raises(RawPayloadArtifactError, match="safely contained"):
        build_preprocessing_inputs(
            raw_entries=(entry,),
            max_input_bytes=25_000_000,
        )


def test_builder_rejects_raw_payload_size_mismatch(tmp_path: Path) -> None:
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="size.txt",
    )
    (tmp_path / "size.txt").write_bytes(b"different length")

    with pytest.raises(RawPayloadArtifactError, match="size mismatch"):
        build_preprocessing_inputs(
            raw_entries=(entry,),
            max_input_bytes=25_000_000,
        )


def test_builder_rejects_raw_payload_sha256_mismatch(tmp_path: Path) -> None:
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="digest.txt",
        content=b"original",
    )
    (tmp_path / "digest.txt").write_bytes(b"tampered")

    with pytest.raises(RawPayloadArtifactError, match="sha256 mismatch"):
        build_preprocessing_inputs(
            raw_entries=(entry,),
            max_input_bytes=25_000_000,
        )


def test_builder_preserves_normalized_language_evidence(
    tmp_path: Path,
) -> None:
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="language.txt",
        content=b"language evidence",
        language=" EN ",
        language_confidence=0.97,
        language_source=" fasttext ",
        language_detector_version=" crawler-language-v1 ",
    )

    [item] = build_preprocessing_inputs(
        raw_entries=(entry,),
        max_input_bytes=25_000_000,
    )

    assert entry.record.language == "en"
    assert item.language_evidence == LanguageEvidence(
        language="en",
        confidence=0.97,
        source="fasttext",
        detector_version="crawler-language-v1",
    )


def test_missing_language_evidence_is_not_materialized(
    tmp_path: Path,
) -> None:
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="without-language.txt",
        content=b"no language evidence",
        language=None,
        language_confidence=None,
        language_source=None,
        language_detector_version=None,
    )

    [item] = build_preprocessing_inputs(
        raw_entries=(entry,),
        max_input_bytes=25_000_000,
    )

    assert item.language_evidence is None


@pytest.mark.parametrize(
    "kind", ("page", "document", "image", "audio", "video")
)
def test_all_modalities_share_one_language_evidence_mapping(
    tmp_path: Path,
    kind: str,
) -> None:
    if kind == "page":
        entry = _page_entry(
            tmp_path=tmp_path,
            language=" NL ",
            language_confidence=0.88,
            language_source=" detector ",
            language_detector_version=" v1 ",
        )
    else:
        entry = _entry(
            tmp_path=tmp_path,
            kind=kind,
            name=f"input-{kind}.txt"
            if kind == "document"
            else f"input-{kind}.bin",
            content=b"shared language mapping",
            language=" NL ",
            language_confidence=0.88,
            language_source=" detector ",
            language_detector_version=" v1 ",
        )

    [item] = build_preprocessing_inputs(
        raw_entries=(entry,),
        max_input_bytes=25_000_000,
    )

    assert item.language_evidence == LanguageEvidence(
        language="nl",
        confidence=0.88,
        source="detector",
        detector_version="v1",
    )


def test_input_order_remains_documents_before_media(tmp_path: Path) -> None:
    entries = (
        _entry(tmp_path=tmp_path, kind="video", name="video.bin"),
        _page_entry(tmp_path=tmp_path),
        _entry(tmp_path=tmp_path, kind="image", name="image.bin"),
        _entry(
            tmp_path=tmp_path,
            kind="document",
            name="document.txt",
            content=b"document",
        ),
        _entry(tmp_path=tmp_path, kind="audio", name="audio.bin"),
    )

    inputs = build_preprocessing_inputs(
        raw_entries=entries,
        max_input_bytes=25_000_000,
    )

    assert [item.source_id for item in inputs] == [
        "id-page.html",
        "id-document.txt",
        "id-video.bin",
        "id-image.bin",
        "id-audio.bin",
    ]


def test_oversized_raw_text_document_is_not_read(tmp_path: Path) -> None:
    entry = _entry(
        tmp_path=tmp_path,
        kind="document",
        name="oversized.txt",
        content=b"more than four bytes",
        enrichment={},
    )

    assert (
        build_preprocessing_inputs(
            raw_entries=(entry,),
            max_input_bytes=4,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("overrides", "commit_sidecar", "expected_path", "expected_detail"),
    (
        (
            {"page_extraction_path": ""},
            True,
            "<missing>",
            "page_extraction_path missing",
        ),
        (
            {"page_extraction_sha256": ""},
            True,
            "extraction/page/id-page.html.json",
            "page_extraction_sha256 missing",
        ),
        (
            {"page_extraction_sha256": "0" * 64},
            True,
            "extraction/page/id-page.html.json",
            "sha256 mismatch",
        ),
        (
            {"page_extraction_schema_version": 2},
            True,
            "extraction/page/id-page.html.json",
            "schema_version mismatch",
        ),
        (
            {},
            False,
            "extraction/page/id-page.html.json",
            "artifact missing",
        ),
    ),
)
def test_page_sidecar_failures_are_contextual_and_fail_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    commit_sidecar: bool,
    expected_path: str,
    expected_detail: str,
) -> None:
    entry = _page_entry(
        tmp_path=tmp_path,
        enrichment_overrides=overrides,
        commit_sidecar=commit_sidecar,
    )

    with pytest.raises(PageExtractionArtifactError) as raised:
        build_preprocessing_inputs(
            raw_entries=(entry,),
            max_input_bytes=25_000_000,
        )

    detail = str(raised.value)
    assert "fetch_record_id=id-page.html" in detail
    assert f"path={expected_path}" in detail
    assert expected_detail in detail


def test_language_evidence_rejects_noncanonical_state() -> None:
    with pytest.raises(ValueError, match="at least one evidence field"):
        LanguageEvidence(language=None)
    with pytest.raises(ValueError, match="normalized lowercase"):
        LanguageEvidence(language="EN")
    with pytest.raises(ValueError, match="source must be stripped"):
        LanguageEvidence(language="en", source=" fasttext ")
    with pytest.raises(ValueError, match="source must not be empty"):
        LanguageEvidence(language="en", source="")


class _CapturingLogger:
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, object]]] = []

    def error(self, event: str, **values: object) -> None:
        self.errors.append((event, values))


@pytest.mark.asyncio
async def test_pairer_logs_and_propagates_page_artifact_failure(
    tmp_path: Path,
) -> None:
    logger = _CapturingLogger()

    def fail_builder(**_: object) -> tuple[()]:
        raise PageExtractionArtifactError("contextual sidecar failure")

    with pytest.raises(
        PageExtractionArtifactError,
        match="contextual sidecar failure",
    ):
        await build_curated_samples(
            snapshot_id="snapshot-1",
            snapshot_directory=tmp_path,
            project_root=tmp_path,
            schema_version="3.0",
            raw_entries=(),
            document_curator_factory=lambda **_: object(),
            preprocessing_input_builder=fail_builder,
            preprocessing_phase_runner=object(),
            logger=logger,  # type: ignore[arg-type]
            chunker=object(),
            sync_row_assembler=lambda **_: (),
        )

    assert logger.errors == [
        (
            "page_extraction_load_failed",
            {
                "reason": "PageExtractionArtifactError",
                "detail": "contextual sidecar failure",
            },
        )
    ]


def test_document_assembler_owns_no_input_construction() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root / "crawler/curation/documents/assembler.py"
    ).read_text(encoding="utf-8")

    for forbidden_name in (
        "prepare_" + "inputs",
        "_build_preprocessing_" + "input",
        "_language_" + "evidence",
        "_load_page_extracted_" + "text",
        "_document_text_from_" + "entry",
    ):
        assert forbidden_name not in source
