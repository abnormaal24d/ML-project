"""Page extraction sidecar integrity and transactional prepare tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from crawler.fetching.response.cache import ConditionalRepresentationCache
from crawler.storage.datasets.extraction.page_extraction_artifact import (
    PAGE_EXTRACTION_SCHEMA_VERSION,
    PageExtractionArtifact,
    PageExtractionArtifactError,
    PageExtractionArtifactReader,
    PageExtractionArtifactWriter,
)


def _artifact(**overrides) -> PageExtractionArtifact:
    payload = dict(
        schema_version=PAGE_EXTRACTION_SCHEMA_VERSION,
        text="Hello world content",
        markdown="# Hello\n\nworld content",
        headings=("Hello",),
        code_block_count=0,
        boilerplate_ratio=0.1,
        extraction_warnings=(),
        title="Hello",
        canonical_url="https://example.test/hello",
    )
    payload.update(overrides)
    return PageExtractionArtifact(**payload)


class _Clock:
    def now(self) -> datetime:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_roundtrip_prepare_commit_read(tmp_path: Path) -> None:
    artifact = _artifact()
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(fetch_record_id="fetch-1", artifact=artifact)
    assert not prepared.absolute_path.exists()
    writer.commit(prepared=prepared)
    assert prepared.absolute_path.is_file()
    loaded = PageExtractionArtifactReader().read(
        snapshot_directory=tmp_path,
        relative_path=prepared.relative_path,
        expected_sha256=prepared.sha256,
        expected_schema_version=PAGE_EXTRACTION_SCHEMA_VERSION,
    )
    assert loaded.text == artifact.text
    assert loaded.markdown == artifact.markdown
    assert loaded.headings == artifact.headings


def test_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifactReader().read(
            snapshot_directory=tmp_path,
            relative_path="../secret.json",
            expected_sha256="0" * 64,
        )


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifactReader().read(
            snapshot_directory=tmp_path,
            relative_path="extraction/page/missing.json",
            expected_sha256="0" * 64,
        )


def test_rejects_sha_mismatch(tmp_path: Path) -> None:
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(fetch_record_id="fetch-2", artifact=_artifact())
    writer.commit(prepared=prepared)
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifactReader().read(
            snapshot_directory=tmp_path,
            relative_path=prepared.relative_path,
            expected_sha256="f" * 64,
        )


def test_rejects_unknown_schema_version() -> None:
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload(
            {
                "schema_version": 99,
                "text": "x",
                "markdown": "x",
                "headings": [],
                "code_block_count": 0,
                "boilerplate_ratio": 0.0,
                "extraction_warnings": [],
            }
        )


def test_rejects_invalid_scalars() -> None:
    base = {
        "schema_version": 1,
        "text": "x",
        "markdown": "x",
        "headings": [],
        "code_block_count": 0,
        "boilerplate_ratio": 0.0,
        "extraction_warnings": [],
    }
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload({**base, "schema_version": "x"})
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload({**base, "code_block_count": "x"})
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload({**base, "code_block_count": True})
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload(
            {**base, "boilerplate_ratio": "nan"}
        )
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload(
            {**base, "boilerplate_ratio": float("nan")}
        )
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload({**base, "boilerplate_ratio": 1.5})
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifact.from_payload({**base, "code_block_count": -1})


def test_rejects_schema_version_mismatch_manifest_vs_sidecar(
    tmp_path: Path,
) -> None:
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(fetch_record_id="fetch-3", artifact=_artifact())
    writer.commit(prepared=prepared)
    with pytest.raises(PageExtractionArtifactError):
        PageExtractionArtifactReader().read(
            snapshot_directory=tmp_path,
            relative_path=prepared.relative_path,
            expected_sha256=prepared.sha256,
            expected_schema_version=99,
        )


def test_prepare_does_not_create_file(tmp_path: Path) -> None:
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(fetch_record_id="fetch-4", artifact=_artifact())
    assert not prepared.absolute_path.exists()


def test_journal_rollback_deletes_sidecar_written_after_begin(
    tmp_path: Path,
) -> None:
    """Sidecar committed inside a journal transaction is removed on rollback."""

    from crawler.storage.datasets.writing.dataset_write_journal import (
        DatasetWriteJournal,
    )

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    writer = PageExtractionArtifactWriter(run_directory=run_directory)
    prepared = writer.prepare(
        fetch_record_id="fetch-rollback", artifact=_artifact()
    )
    assert not prepared.absolute_path.exists()

    journal = DatasetWriteJournal(run_directory=run_directory)
    dummy_payload = run_directory / "objects" / "x.bin"
    dummy_payload.parent.mkdir(parents=True, exist_ok=True)
    transaction = journal.begin(
        transaction_id="tx-1",
        tracked_paths=(prepared.absolute_path,),
        payload_path=dummy_payload,
    )
    writer.commit(prepared=prepared)
    assert prepared.absolute_path.exists()
    journal.rollback(transaction)
    assert not prepared.absolute_path.exists()


def test_commit_removes_tmp_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leftover .tmp files must not survive a failed os.replace()."""

    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(
        fetch_record_id="fetch-tmp", artifact=_artifact()
    )
    temp_path = prepared.absolute_path.with_suffix(
        prepared.absolute_path.suffix + ".tmp"
    )

    def _failing_replace(src: object, dst: object) -> None:
        raise OSError("forced replace failure")

    monkeypatch.setattr(
        "crawler.storage.datasets.extraction.page_extraction_artifact.os.replace",
        _failing_replace,
    )
    with pytest.raises(OSError, match="forced replace failure"):
        writer.commit(prepared=prepared)

    assert not prepared.absolute_path.exists()
    assert not temp_path.exists()


def test_reader_rejects_non_int_expected_schema_version(
    tmp_path: Path,
) -> None:
    writer = PageExtractionArtifactWriter(run_directory=tmp_path)
    prepared = writer.prepare(
        fetch_record_id="fetch-expected", artifact=_artifact()
    )
    writer.commit(prepared=prepared)
    reader = PageExtractionArtifactReader()
    for bad_expected in (True, "1", "x", 1.0):
        with pytest.raises(PageExtractionArtifactError):
            reader.read(
                snapshot_directory=tmp_path,
                relative_path=prepared.relative_path,
                expected_sha256=prepared.sha256,
                expected_schema_version=bad_expected,  # type: ignore[arg-type]
            )


def test_pipeline_rollback_after_sidecar_when_manifest_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full DatasetWritePipeline must roll back sidecar and state on failure."""

    import hashlib

    from config.settings.datasets import (
        DatasetPathSettings,
        RawDatasetWriterSettings,
    )
    from crawler.classification.media_kind import MediaKind
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.payload import FetchedPayload
    from crawler.fetching.results.result import FetchResult
    from crawler.storage.datasets.extraction.page_extraction_artifact import (
        enrichment_artifact_key,
    )
    from crawler.storage.datasets.manifests.dataset_manifest_writer import (
        DatasetManifestWriter,
    )
    from crawler.storage.datasets.records.dataset_record import (
        DatasetRecordCreator,
    )
    from crawler.storage.datasets.records.record_index import (
        DatasetRecordIndex,
    )
    from crawler.storage.datasets.sync_index.sync_index_paths import (
        SyncIndexPaths,
    )
    from crawler.storage.datasets.sync_index.sync_index_reader import (
        SyncIndexReader,
    )
    from crawler.storage.datasets.sync_index.sync_index_updater import (
        SyncIndexUpdater,
    )
    from crawler.storage.datasets.writing.dataset_write_pipeline import (
        DatasetWritePipeline,
    )
    from crawler.storage.datasets.writing.raw_payload_writer import (
        RawPayloadWriter,
    )

    class _Logger:
        def debug(self, *args: object, **kwargs: object) -> None:
            return None

        def info(self, *args: object, **kwargs: object) -> None:
            return None

        def warning(self, *args: object, **kwargs: object) -> None:
            return None

    class _Normalizer:
        def normalize(self, url: str) -> str:
            return str(url).strip()

    class _NoopCompactor:
        def should_write_summary(self, *, manifest_write_count: int) -> bool:
            return False

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    settings = RawDatasetWriterSettings()
    dataset_paths = DatasetPathSettings()
    body = b"<html><body>Hello pipeline rollback</body></html>"
    body_hash = hashlib.sha256(body).hexdigest()
    payload_temp = tmp_path / "incoming.html"
    payload_temp.write_bytes(body)

    task = CrawlTask(
        url="https://example.test/page",
        source_name="example",
        task_id="task-rollback-01",
        kind=MediaKind.PAGE,
        depth=0,
        source_type="seed",
    )
    result = FetchResult(
        url=task.url,
        final_url=task.url,
        status_code=200,
        headers={"content-type": "text/html"},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="text/html",
        mime_type="text/html",
        encoding="utf-8",
        language=None,
        kind=MediaKind.PAGE,
        payload=FetchedPayload(
            temp_path=payload_temp,
            byte_size=len(body),
            sha256_hex=body_hash,
            sniff_bytes=body[:64],
            chunk_count=1,
        ),
        body_sha256=body_hash,
    )

    payload_writer = RawPayloadWriter(
        settings=settings,
        dataset_paths=dataset_paths,
        run_directory=run_directory,
    )
    manifest_path = run_directory / dataset_paths.manifest_filename
    manifest_writer = DatasetManifestWriter(
        settings=settings,
        manifest_path=manifest_path,
    )
    sync_paths = SyncIndexPaths.from_settings(
        run_directory=run_directory,
        dataset_paths=dataset_paths,
    )
    sync_reader = SyncIndexReader(
        paths=sync_paths,
        dataset_paths=dataset_paths,
    )
    sync_updater = SyncIndexUpdater(
        settings=settings,
        reader=sync_reader,
        run_directory=run_directory,
    )
    record_index = DatasetRecordIndex()
    record_creator = DatasetRecordCreator(
        settings=settings,
        run_id="run-rollback-1",
        now=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    pipeline = DatasetWritePipeline(
        settings=settings,
        logger=_Logger(),
        url_normalizer=_Normalizer(),
        payload_writer=payload_writer,
        manifest_writer=manifest_writer,
        sync_updater=sync_updater,
        record_creator=record_creator,
        record_index=record_index,
        conditional_representation_cache=ConditionalRepresentationCache(
            enabled=False,
            max_entries=1,
            ttl_seconds=None,
            clock=lambda: 0.0,
        ),
    )

    relative_path, content_hash, _byte_size, _inline = payload_writer.prepare(
        result=result,
        kind="page",
        modality="page",
    )
    absolute_payload = payload_writer.absolute_path(
        relative_path=relative_path
    )
    expected_sidecar = (
        run_directory / "extraction" / "page" / "placeholder.json"
    )

    def _failing_append(record: object) -> bool:
        # Capture the sidecar path the pipeline would have prepared.
        enrichment = getattr(record, "enrichment", {}) or {}
        path = enrichment.get("page_extraction_path")
        if isinstance(path, str) and path:
            nonlocal expected_sidecar
            expected_sidecar = run_directory / path
        raise RuntimeError("forced manifest append failure")

    monkeypatch.setattr(manifest_writer, "append", _failing_append)

    artifact = _artifact()
    enrichment = {
        enrichment_artifact_key(): artifact.to_payload(),
    }

    assert len(record_index) == 0
    assert manifest_writer.write_count == 0
    assert sync_updater.snapshot()["modality_counts"] == {}

    with pytest.raises(RuntimeError, match="forced manifest append failure"):
        pipeline.execute(
            task=task,
            result=result,
            enrichment=enrichment,
        )

    assert not expected_sidecar.exists(), "sidecar must be removed on rollback"
    assert not expected_sidecar.with_suffix(
        expected_sidecar.suffix + ".tmp"
    ).exists()
    assert not absolute_payload.exists(), "payload must be removed on rollback"
    assert manifest_writer.write_count == 0
    assert sync_updater.snapshot() == {
        "modality_counts": {},
        "relationships_count": 0,
        "metadata_count": 0,
        "updates_count": 0,
        "errors_count": 0,
        "discovered_assets_count": 0,
        "rejected_assets_count": 0,
    }
    assert len(record_index) == 0
    # No pending journal files after rollback.
    assert not list((run_directory / "transactions").glob("*.json"))
    # content_hash only used to keep prepare() side-effect free for assert.
    assert content_hash == body_hash
