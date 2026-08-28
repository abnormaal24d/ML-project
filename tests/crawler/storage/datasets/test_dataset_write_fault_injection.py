"""Dataset write fault injection: atomicity and rollback per stage.

Injects fs/os faults at each storage stage of DatasetWritePipeline and
asserts the canonical destination never keeps a partial artifact, prior
valid files stay unchanged, and failures surface as exceptions after
journal rollback. No production code is modified.
"""

from __future__ import annotations

import hashlib
import os
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
from crawler.storage.datasets.writing.dataset_write_journal import (
    DatasetWriteJournal,
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


def _fetch_result(
    *,
    url: str,
    body: bytes,
    payload_temp: Path,
) -> tuple[CrawlTask, FetchResult]:
    body_hash = hashlib.sha256(body).hexdigest()
    task = CrawlTask(
        url=url,
        source_name="example",
        task_id=f"task-{body_hash[:8]}",
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
    return task, result


def _build_pipeline(
    tmp_path: Path,
    *,
    body: bytes,
    url: str,
) -> dict[str, object]:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    settings = RawDatasetWriterSettings()
    dataset_paths = DatasetPathSettings()
    payload_temp = tmp_path / "incoming.html"
    payload_temp.write_bytes(body)

    task, result = _fetch_result(url=url, body=body, payload_temp=payload_temp)
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
        run_id="run-fault-injection",
        now=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    pipeline = DatasetWritePipeline(
        settings=settings,
        logger=_Logger(),  # type: ignore[arg-type]
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
    relative_path, _hash, _size, _inline = payload_writer.prepare(
        result=result,
        kind="page",
        modality="page",
    )
    return {
        "pipeline": pipeline,
        "task": task,
        "result": result,
        "payload_writer": payload_writer,
        "manifest_writer": manifest_writer,
        "record_index": record_index,
        "run_directory": run_directory,
        "manifest_path": manifest_path,
        "absolute_payload": payload_writer.absolute_path(
            relative_path=relative_path
        ),
    }


def _build_second_fetch(
    tmp_path: Path,
    ctx: dict[str, object],
    *,
    body: bytes,
    url: str,
) -> dict[str, object]:
    payload_temp = tmp_path / "incoming-second.html"
    payload_temp.write_bytes(body)
    task, result = _fetch_result(url=url, body=body, payload_temp=payload_temp)
    payload_writer = ctx["payload_writer"]
    relative_path, _hash, _size, _inline = payload_writer.prepare(  # type: ignore[union-attr]
        result=result,
        kind="page",
        modality="page",
    )
    return {
        "task": task,
        "result": result,
        "absolute_payload": payload_writer.absolute_path(  # type: ignore[union-attr]
            relative_path=relative_path
        ),
    }


def _assert_failed_write_never_leaves_artifacts(
    *, ctx: dict[str, object]
) -> None:
    assert not ctx["absolute_payload"].exists()  # type: ignore[union-attr]
    assert not list(
        (ctx["run_directory"] / "objects").rglob("*.tmp")  # type: ignore[operator]
    )
    assert not list(
        (ctx["run_directory"] / "transactions").glob("*.json")  # type: ignore[operator]
    )
    assert not list(
        (ctx["run_directory"] / "transactions").glob("*.tmp")  # type: ignore[operator]
    )
    assert ctx["manifest_writer"].write_count == 0  # type: ignore[union-attr]
    assert len(ctx["record_index"]) == 0  # type: ignore[arg-type]


def _assert_previous_write_untouched(
    *,
    ctx: dict[str, object],
    first_absolute: Path,
    first_body: bytes,
    first_manifest: bytes,
) -> None:
    assert first_absolute.exists()
    assert first_absolute.read_bytes() == first_body
    assert ctx["manifest_path"].read_bytes() == first_manifest  # type: ignore[union-attr]
    assert ctx["manifest_writer"].write_count == 1  # type: ignore[union-attr]
    assert len(ctx["record_index"]) == 1  # type: ignore[arg-type]
    assert not list(
        (ctx["run_directory"] / "transactions").glob("*.json")  # type: ignore[operator]
    )


def test_fault_during_temp_file_write_rolls_back_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        body=b"<html>will-never-commit</html>",
        url="https://example.test/temp-write",
    )
    pipeline = ctx["pipeline"]
    real_replace = os.replace

    def _block_payload_temp_rename(src: object, dst: object) -> None:
        if Path(str(dst)).suffix == ".tmp":
            raise OSError("forced temp-file rename failure")
        return real_replace(src, dst)

    def _fail_copy_bytes(src: object, dst: object, length: int = 0) -> None:
        del dst
        del length
        raise OSError("forced temp-file write failure")

    monkeypatch.setattr(os, "replace", _block_payload_temp_rename)
    monkeypatch.setattr(
        "crawler.storage.datasets.writing.raw_payload_writer.shutil.copyfileobj",
        _fail_copy_bytes,
    )

    with pytest.raises(OSError, match="forced temp-file write failure"):
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=None,
        )

    _assert_failed_write_never_leaves_artifacts(ctx=ctx)


def test_fault_during_temp_file_fsync_leaves_no_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        body=b"<html>never-durable</html>",
        url="https://example.test/temp-fsync",
    )
    pipeline = ctx["pipeline"]
    real_replace = os.replace
    real_fsync = os.fsync
    copy_path_armed = {"armed": False}

    def _block_payload_temp_rename(src: object, dst: object) -> None:
        if Path(str(dst)).suffix == ".tmp":
            copy_path_armed["armed"] = True
            raise OSError("forced temp-file rename failure")
        return real_replace(src, dst)

    def _fail_copy_fsync(fd: int) -> None:
        if copy_path_armed["armed"]:
            copy_path_armed["armed"] = False
            raise OSError("forced temp-file fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", _block_payload_temp_rename)
    monkeypatch.setattr(os, "fsync", _fail_copy_fsync)

    with pytest.raises(OSError, match="forced temp-file fsync failure"):
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=None,
        )

    _assert_failed_write_never_leaves_artifacts(ctx=ctx)


def test_fault_during_canonical_rename_keeps_previous_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        body=b"<html>first</html>",
        url="https://example.test/first",
    )
    pipeline = ctx["pipeline"]
    first_absolute = ctx["absolute_payload"]
    first_body = b"<html>first</html>"

    pipeline.execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment=None,
    )
    first_manifest = ctx["manifest_path"].read_bytes()  # type: ignore[union-attr]

    second = _build_second_fetch(
        tmp_path,
        ctx,
        body=b"<html>second</html>",
        url="https://example.test/second",
    )
    second_absolute = second["absolute_payload"]
    real_replace = os.replace

    def _block_canonical_rename(src: object, dst: object) -> None:
        if Path(str(dst)).resolve() == Path(str(second_absolute)).resolve():
            raise OSError("forced canonical rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _block_canonical_rename)

    with pytest.raises(OSError, match="forced canonical rename failure"):
        pipeline.execute(  # type: ignore[union-attr]
            task=second["task"],  # type: ignore[arg-type]
            result=second["result"],  # type: ignore[arg-type]
            enrichment=None,
        )

    assert not second_absolute.exists()
    _assert_previous_write_untouched(
        ctx=ctx,
        first_absolute=first_absolute,
        first_body=first_body,
        first_manifest=first_manifest,
    )


def test_fault_during_manifest_durability_restores_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(
        tmp_path,
        body=b"<html>first</html>",
        url="https://example.test/m-first",
    )
    pipeline = ctx["pipeline"]
    manifest_writer = ctx["manifest_writer"]
    first_absolute = ctx["absolute_payload"]
    first_body = b"<html>first</html>"

    pipeline.execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment=None,
    )
    first_manifest = ctx["manifest_path"].read_bytes()  # type: ignore[union-attr]
    manifest_handle_fd = manifest_writer._handle.fileno()  # type: ignore[union-attr]

    second = _build_second_fetch(
        tmp_path,
        ctx,
        body=b"<html>second</html>",
        url="https://example.test/m-second",
    )
    second_absolute = second["absolute_payload"]
    real_fsync = os.fsync
    manifest_gate = {"active": True}

    def _fail_manifest_fsync(fd: int) -> None:
        if manifest_gate["active"] and fd == manifest_handle_fd:
            manifest_gate["active"] = False
            raise OSError("forced manifest flush fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fail_manifest_fsync)

    with pytest.raises(OSError, match="forced manifest flush fsync failure"):
        pipeline.execute(  # type: ignore[union-attr]
            task=second["task"],  # type: ignore[arg-type]
            result=second["result"],  # type: ignore[arg-type]
            enrichment=None,
        )

    assert not second_absolute.exists()
    assert manifest_gate["active"] is False
    _assert_previous_write_untouched(
        ctx=ctx,
        first_absolute=first_absolute,
        first_body=first_body,
        first_manifest=first_manifest,
    )
    assert not list(
        (ctx["run_directory"] / "objects").rglob("*.tmp")  # type: ignore[operator]
    )


def test_recovery_restores_prior_manifest_after_mid_flight_crash(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest_path = run_directory / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"row":1}\n', encoding="utf-8")
    payload_path = run_directory / "objects" / "page" / "crash.html"
    payload_path.parent.mkdir(parents=True, exist_ok=True)

    journal = DatasetWriteJournal(run_directory=run_directory)
    transaction = journal.begin(
        transaction_id="tx-crash-1",
        tracked_paths=(manifest_path,),
        payload_path=payload_path,
    )

    payload_path.write_bytes(b"partial-payload-bytes")
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write('{"row":2}\n')

    recovered = DatasetWriteJournal(
        run_directory=run_directory
    ).recover_pending()
    assert recovered.rolled_back == ("tx-crash-1",)
    assert recovered.finalized_commits == ()
    assert manifest_path.read_text(encoding="utf-8") == '{"row":1}\n'
    assert not payload_path.exists()
    assert not transaction.journal_path.exists()
    assert not list((run_directory / "transactions").glob("*.json"))
