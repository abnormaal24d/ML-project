"""Derived media artifacts are committed inside the dataset transaction.

Covers the DerivedMediaArtifactWriter prepare/commit/consume contract and
DatasetWritePipeline relocating normalized media and selected keyframes into
run-owned relative paths tracked by the existing write journal.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings.datasets import (
    DatasetPathSettings,
    RawDatasetWriterSettings,
)
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.fetching.response.cache import ConditionalRepresentationCache
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from crawler.storage.datasets.manifests.dataset_manifest_writer import (
    DatasetManifestWriter,
)
from crawler.storage.datasets.records.dataset_record import (
    DatasetRecordCreator,
)
from crawler.storage.datasets.records.record_index import (
    DatasetRecordIndex,
)
from crawler.storage.datasets.sync_index.sync_index_paths import SyncIndexPaths
from crawler.storage.datasets.sync_index.sync_index_reader import (
    SyncIndexReader,
)
from crawler.storage.datasets.sync_index.sync_index_updater import (
    SyncIndexUpdater,
)
from crawler.storage.datasets.writing.dataset_write_pipeline import (
    DatasetWritePipeline,
)
from crawler.storage.datasets.writing.derived_media_artifact import (
    DerivedMediaArtifactError,
    DerivedMediaArtifactWriter,
    consume_derived_media_source,
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


def _build_pipeline(
    tmp_path: Path,
    *,
    conditional_representation_cache: ConditionalRepresentationCache,
) -> dict[str, object]:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    settings = RawDatasetWriterSettings()
    dataset_paths = DatasetPathSettings()
    body = b"image-or-video-payload-bytes"
    body_hash = hashlib.sha256(body).hexdigest()
    payload_temp = tmp_path / "incoming.bin"
    payload_temp.write_bytes(body)

    task = CrawlTask(
        url="https://example.test/media",
        source_name="example",
        task_id="task-derived-01",
        kind=MediaKind.IMAGE,
        depth=0,
        source_type="seed",
    )
    result = FetchResult(
        url=task.url,
        final_url=task.url,
        status_code=200,
        headers={"content-type": "image/jpeg"},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="image/jpeg",
        mime_type="image/jpeg",
        encoding=None,
        language=None,
        kind=MediaKind.IMAGE,
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
    manifest_writer = DatasetManifestWriter(
        settings=settings,
        manifest_path=run_directory / dataset_paths.manifest_filename,
    )
    sync_paths = SyncIndexPaths.from_settings(
        run_directory=run_directory,
        dataset_paths=dataset_paths,
    )
    sync_updater = SyncIndexUpdater(
        settings=settings,
        reader=SyncIndexReader(paths=sync_paths, dataset_paths=dataset_paths),
        run_directory=run_directory,
    )
    record_index = DatasetRecordIndex()
    record_creator = DatasetRecordCreator(
        settings=settings,
        run_id="run-derived-1",
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
        conditional_representation_cache=conditional_representation_cache,
    )
    return {
        "pipeline": pipeline,
        "task": task,
        "result": result,
        "run_directory": run_directory,
        "payload_writer": payload_writer,
        "manifest_writer": manifest_writer,
        "sync_updater": sync_updater,
        "record_index": record_index,
    }


def _disabled_conditional_representation_cache() -> (
    ConditionalRepresentationCache
):
    return ConditionalRepresentationCache(
        enabled=False,
        max_entries=1,
        ttl_seconds=None,
        clock=lambda: 0.0,
    )


# --- artifact writer contract ----------------------------------------------


def test_artifact_writer_commits_run_owned_copy_and_consumes_source(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    source = tmp_path / "incoming.normalized.jpg"
    source_bytes = b"normalized-image-bytes"
    source.write_bytes(source_bytes)

    writer = DerivedMediaArtifactWriter(run_directory=run_directory)
    prepared = writer.prepare(
        source_path=source,
        fetch_record_id="fetch-1",
        artifact_name="incoming.normalized.jpg",
    )
    assert prepared.relative_path == (
        f"media/derived/fetch-1/{hashlib.sha256(source_bytes).hexdigest()}.jpg"
    )
    assert prepared.sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert not prepared.absolute_path.exists()

    writer.commit(prepared=prepared)
    assert prepared.absolute_path.read_bytes() == source_bytes
    assert source.exists()

    consume_derived_media_source(prepared)
    assert not source.exists()
    assert prepared.absolute_path.exists()


def test_artifact_writer_rejects_missing_source(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    writer = DerivedMediaArtifactWriter(run_directory=run_directory)
    with pytest.raises(DerivedMediaArtifactError):
        writer.prepare(
            source_path=tmp_path / "missing.normalized.jpg",
            fetch_record_id="fetch-1",
            artifact_name="missing.jpg",
        )


def test_artifact_writer_keeps_same_name_different_content_separate(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    source = tmp_path / "normalized.jpg"
    writer = DerivedMediaArtifactWriter(run_directory=run_directory)

    source.write_bytes(b"first")
    first = writer.prepare(
        source_path=source,
        fetch_record_id="fetch-1",
        artifact_name="normalized.jpg",
    )
    source.write_bytes(b"second")
    second = writer.prepare(
        source_path=source,
        fetch_record_id="fetch-1",
        artifact_name="normalized.jpg",
    )
    source.write_bytes(b"first")
    same_content = writer.prepare(
        source_path=source,
        fetch_record_id="fetch-1",
        artifact_name="normalized.jpg",
    )

    assert first.relative_path != second.relative_path
    assert first.relative_path == same_content.relative_path
    assert first.sha256 in first.relative_path
    assert second.sha256 in second.relative_path


def test_artifact_writer_rejects_unsafe_names(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    writer = DerivedMediaArtifactWriter(run_directory=run_directory)
    with pytest.raises(DerivedMediaArtifactError):
        writer.prepare(
            source_path=tmp_path / "x.jpg",
            fetch_record_id="fetch-1",
            artifact_name="../../escape.jpg",
        )


@pytest.mark.parametrize("fetch_record_id", (".", ".."))
def test_artifact_writer_rejects_path_dot_segments_in_record_id(
    tmp_path: Path,
    fetch_record_id: str,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"derived-media")
    writer = DerivedMediaArtifactWriter(run_directory=run_directory)

    with pytest.raises(DerivedMediaArtifactError, match="safe path segment"):
        writer.prepare(
            source_path=source,
            fetch_record_id=fetch_record_id,
            artifact_name="source.jpg",
        )


# --- pipeline relocation ---------------------------------------------------


def test_pipeline_relocates_normalized_image_into_transaction(
    tmp_path: Path,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=(
            _disabled_conditional_representation_cache()
        ),
    )
    pipeline = ctx["pipeline"]
    run_directory = ctx["run_directory"]
    normalized_bytes = b"normalized-jpeg-scratch"
    normalized_source = tmp_path / "incoming.bin.normalized.jpg"
    normalized_source.write_bytes(normalized_bytes)

    outcome = pipeline.execute(
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment={
            "normalized_media_path": str(normalized_source),
            "normalized_image_format": "JPEG",
        },
    )

    record = outcome.record
    assert record is not None
    relative = record.enrichment["normalized_media_path"]
    assert relative == (
        f"media/derived/{record.fetch_record_id}/"
        f"{hashlib.sha256(normalized_bytes).hexdigest()}.jpg"
    )
    assert not Path(str(relative)).is_absolute()
    persisted = run_directory / relative
    assert persisted.read_bytes() == normalized_bytes
    assert not normalized_source.exists()
    assert not list((run_directory / "transactions").glob("*.json"))
    manifest_path = run_directory / "records" / "objects.jsonl"
    assert manifest_path.exists()
    assert relative in manifest_path.read_text(encoding="utf-8")


def test_pipeline_relocates_selected_keyframes_into_transaction(
    tmp_path: Path,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=(
            _disabled_conditional_representation_cache()
        ),
    )
    pipeline = ctx["pipeline"]
    run_directory = ctx["run_directory"]
    keyframe_bytes = b"keyframe-jpeg-scratch"
    keyframe_source = tmp_path / "frame_abc123.jpg"
    keyframe_source.write_bytes(keyframe_bytes)

    outcome = pipeline.execute(
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment={
            "keyframes": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "frame_path": str(keyframe_source),
                    "selection_reason": "scene_change",
                }
            ]
        },
    )

    record = outcome.record
    assert record is not None
    frames = record.enrichment["keyframes"]
    assert isinstance(frames, list)
    relative = frames[0]["frame_path"]
    assert relative == (
        f"media/derived/{record.fetch_record_id}/"
        f"{hashlib.sha256(keyframe_bytes).hexdigest()}.jpg"
    )
    assert not Path(str(relative)).is_absolute()
    assert (run_directory / relative).read_bytes() == keyframe_bytes
    assert not keyframe_source.exists()


def test_pipeline_publishes_validators_only_for_durable_payload(
    tmp_path: Path,
) -> None:
    cache = ConditionalRepresentationCache(
        enabled=True,
        max_entries=10,
        ttl_seconds=60,
        clock=lambda: 0.0,
    )
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=cache,
    )
    result = replace(
        ctx["result"],  # type: ignore[arg-type]
        headers={"content-type": "image/jpeg", "etag": '"v1"'},
    )

    outcome = ctx["pipeline"].execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=result,
        enrichment=None,
    )

    cached = cache.get_representation(result.url)
    assert cached is not None
    assert cached.validators == {"etag": '"v1"'}
    assert cached.result.payload is not None
    assert cached.result.payload.temp_path == (
        ctx["run_directory"] / outcome.record.storage_relative_path  # type: ignore[operator]
    )
    assert cached.result.payload.exists()

    duplicate = ctx["pipeline"].execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=cached.result,
        enrichment=None,
    )
    assert duplicate.duplicate is True
    assert cached.result.payload.exists()


def test_pipeline_reuses_cached_durable_payload_for_distinct_media_identity(
    tmp_path: Path,
) -> None:
    cache = ConditionalRepresentationCache(
        enabled=True,
        max_entries=10,
        ttl_seconds=60,
        clock=lambda: 0.0,
    )
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=cache,
    )
    result = replace(
        ctx["result"],  # type: ignore[arg-type]
        headers={"content-type": "image/jpeg", "etag": '"v1"'},
    )
    first = ctx["pipeline"].execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=result,
        enrichment=None,
    )
    cached = cache.get_representation(result.url)
    assert cached is not None

    distinct_task = replace(
        ctx["task"],  # type: ignore[arg-type]
        task_id="task-derived-02",
        source_type="discovered_link",
    )
    second = ctx["pipeline"].execute(  # type: ignore[union-attr]
        task=distinct_task,
        result=cached.result,
        enrichment=None,
    )

    assert first.record.media_identity != second.record.media_identity
    assert second.duplicate is False
    assert cached.result.payload is not None
    assert cached.result.payload.exists()
    refreshed = cache.get_representation(result.url)
    assert refreshed is not None
    assert refreshed.result.payload is not None
    assert refreshed.result.payload.exists()


def test_pipeline_rollback_never_publishes_conditional_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ConditionalRepresentationCache(
        enabled=True,
        max_entries=10,
        ttl_seconds=60,
        clock=lambda: 0.0,
    )
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=cache,
    )
    result = replace(
        ctx["result"],  # type: ignore[arg-type]
        headers={"content-type": "image/jpeg", "etag": '"v1"'},
    )

    def _fail_append(record: object) -> bool:
        del record
        raise RuntimeError("forced write failure")

    monkeypatch.setattr(ctx["manifest_writer"], "append", _fail_append)
    with pytest.raises(RuntimeError, match="forced write failure"):
        ctx["pipeline"].execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=result,
            enrichment=None,
        )

    assert cache.size == 0


def test_pipeline_rollback_removes_derived_copy_and_consumes_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=(
            _disabled_conditional_representation_cache()
        ),
    )
    pipeline = ctx["pipeline"]
    run_directory = ctx["run_directory"]
    manifest_writer = ctx["manifest_writer"]
    normalized_source = tmp_path / "incoming.bin.normalized.jpg"
    normalized_source.write_bytes(b"normalized-jpeg-scratch")

    def _fail_append(record: object) -> bool:
        del record
        raise RuntimeError("forced write failure")

    monkeypatch.setattr(manifest_writer, "append", _fail_append)

    with pytest.raises(RuntimeError, match="forced write failure"):
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment={"normalized_media_path": str(normalized_source)},
        )

    derived_root = run_directory / "media" / "derived"
    leftover = (
        list(derived_root.rglob("*.jpg")) if derived_root.exists() else []
    )
    assert leftover == []
    assert not list((run_directory / "transactions").glob("*.json"))
    # Scratch sources are consumed even when the transaction rolled back.
    assert not normalized_source.exists()


def test_pipeline_raises_when_derived_source_is_missing(
    tmp_path: Path,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=(
            _disabled_conditional_representation_cache()
        ),
    )
    pipeline = ctx["pipeline"]
    missing = tmp_path / "missing.normalized.jpg"

    with pytest.raises(ValueError, match="derived media artifact source"):
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment={"normalized_media_path": str(missing)},
        )


def test_pipeline_duplicate_skip_cleans_derived_scratch(
    tmp_path: Path,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        conditional_representation_cache=(
            _disabled_conditional_representation_cache()
        ),
    )
    pipeline = ctx["pipeline"]
    normalized_source = tmp_path / "incoming.bin.normalized.jpg"
    normalized_source.write_bytes(b"normalized-jpeg-scratch")
    enrichment = {"normalized_media_path": str(normalized_source)}

    first = pipeline.execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment=enrichment,
    )
    assert first.record is not None
    assert not normalized_source.exists()

    # Second identical write is skipped as a duplicate; the payload temp and
    # any freshly produced scratch sibling must be discarded together.
    recreated_source = tmp_path / "incoming.bin.normalized.jpg"
    recreated_source.write_bytes(b"normalized-jpeg-scratch-again")
    second = pipeline.execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment={"normalized_media_path": str(recreated_source)},
    )
    assert second.duplicate is True
    assert not recreated_source.exists()
    assert not ctx["result"].payload.exists()  # type: ignore[union-attr]
