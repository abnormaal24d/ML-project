"""Rollback resilience: cleanup must continue after individual step failures."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
from crawler.storage.datasets.extraction.page_extraction_artifact import (
    PAGE_EXTRACTION_SCHEMA_VERSION,
    PageExtractionArtifact,
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
from crawler.storage.datasets.sync_index.sync_index_paths import SyncIndexPaths
from crawler.storage.datasets.sync_index.sync_index_reader import (
    SyncIndexReader,
)
from crawler.storage.datasets.sync_index.sync_index_updater import (
    SyncIndexUpdater,
)
from crawler.storage.datasets.writing.dataset_write_journal import (
    DatasetWriteJournal,
    _write_json_atomic,
)
from crawler.storage.datasets.writing.dataset_write_pipeline import (
    DatasetWritePipeline,
)
from crawler.storage.datasets.writing.raw_payload_writer import (
    RawPayloadWriter,
)


def _artifact() -> PageExtractionArtifact:
    return PageExtractionArtifact(
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
    kind: MediaKind = MediaKind.PAGE,
) -> dict[str, object]:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    settings = RawDatasetWriterSettings()
    dataset_paths = DatasetPathSettings()
    body = b"<html><body>Hello rollback resilience</body></html>"
    body_hash = hashlib.sha256(body).hexdigest()
    payload_temp = tmp_path / "incoming.html"
    payload_temp.write_bytes(body)

    task = CrawlTask(
        url="https://example.test/page",
        source_name="example",
        task_id="task-resilience-01",
        kind=kind,
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
        kind=kind,
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
        run_id="run-resilience-1",
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
    relative_path, _hash, _size, _inline = payload_writer.prepare(
        result=result,
        kind="page",
        modality="page",
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
        "absolute_payload": payload_writer.absolute_path(
            relative_path=relative_path
        ),
        "enrichment": {
            enrichment_artifact_key(): _artifact().to_payload(),
        },
    }


def _sidecar_paths(run_directory: Path) -> list[Path]:
    root = run_directory / "extraction" / "page"
    if not root.exists():
        return []
    return list(root.glob("*.json")) + list(root.glob("*.json.tmp"))


def _assert_rolled_back(
    *,
    run_directory: Path,
    absolute_payload: Path,
    record_index: DatasetRecordIndex,
    manifest_writer: DatasetManifestWriter,
    sync_updater: SyncIndexUpdater,
) -> None:
    assert _sidecar_paths(run_directory) == []
    assert not absolute_payload.exists()
    assert len(record_index) == 0
    assert manifest_writer.write_count == 0
    assert sync_updater.snapshot()["modality_counts"] == {}
    assert not list((run_directory / "transactions").glob("*.json"))
    assert not list((run_directory / "transactions").glob("*.tmp"))


def _flatten_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, ExceptionGroup):
        items: list[BaseException] = []
        for nested in exc.exceptions:
            items.extend(_flatten_exceptions(nested))
        return items
    return [exc]


def _exception_messages(exc: BaseException) -> list[str]:
    return [str(item) for item in _flatten_exceptions(exc)]


def test_rollback_continues_when_manifest_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    manifest_writer = ctx["manifest_writer"]

    def _fail_append(record: object) -> bool:
        del record
        raise RuntimeError("forced write failure")

    def _fail_flush() -> None:
        raise OSError("forced manifest flush failure")

    monkeypatch.setattr(manifest_writer, "append", _fail_append)
    monkeypatch.setattr(manifest_writer, "flush_transaction", _fail_flush)

    with pytest.raises(ExceptionGroup) as raised:
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    # Write failed and cleanup reported the flush failure, but journal still
    # rolled back durable files.
    messages = _exception_messages(raised.value)
    assert any("forced write failure" in msg for msg in messages)
    assert any("forced manifest flush failure" in msg for msg in messages)
    _assert_rolled_back(
        run_directory=ctx["run_directory"],  # type: ignore[arg-type]
        absolute_payload=ctx["absolute_payload"],  # type: ignore[arg-type]
        record_index=ctx["record_index"],  # type: ignore[arg-type]
        manifest_writer=manifest_writer,  # type: ignore[arg-type]
        sync_updater=ctx["sync_updater"],  # type: ignore[arg-type]
    )


def test_failure_after_media_index_registration_restores_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(tmp_path, kind=MediaKind.IMAGE)
    pipeline = ctx["pipeline"]
    manifest_writer = ctx["manifest_writer"]
    record_index = ctx["record_index"]
    index_snapshot = record_index.snapshot()  # type: ignore[union-attr]

    def _fail_flush() -> None:
        raise OSError("forced post-index flush failure")

    monkeypatch.setattr(manifest_writer, "flush_transaction", _fail_flush)

    with pytest.raises(ExceptionGroup):
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    assert record_index.snapshot() == index_snapshot  # type: ignore[union-attr]
    _assert_rolled_back(
        run_directory=ctx["run_directory"],  # type: ignore[arg-type]
        absolute_payload=ctx["absolute_payload"],  # type: ignore[arg-type]
        record_index=record_index,  # type: ignore[arg-type]
        manifest_writer=manifest_writer,  # type: ignore[arg-type]
        sync_updater=ctx["sync_updater"],  # type: ignore[arg-type]
    )


def test_rollback_continues_when_sync_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    sync_updater = ctx["sync_updater"]
    manifest_writer = ctx["manifest_writer"]

    def _fail_append(record: object) -> bool:
        del record
        raise RuntimeError("forced write failure")

    def _fail_sync_flush() -> None:
        raise OSError("forced sync flush failure")

    monkeypatch.setattr(manifest_writer, "append", _fail_append)
    monkeypatch.setattr(sync_updater, "flush_transaction", _fail_sync_flush)

    with pytest.raises(ExceptionGroup) as raised:
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    messages = _exception_messages(raised.value)
    assert any("forced write failure" in msg for msg in messages)
    assert any("forced sync flush failure" in msg for msg in messages)
    _assert_rolled_back(
        run_directory=ctx["run_directory"],  # type: ignore[arg-type]
        absolute_payload=ctx["absolute_payload"],  # type: ignore[arg-type]
        record_index=ctx["record_index"],  # type: ignore[arg-type]
        manifest_writer=manifest_writer,  # type: ignore[arg-type]
        sync_updater=sync_updater,  # type: ignore[arg-type]
    )


def test_rollback_reports_journal_failure_and_still_restores_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    manifest_writer = ctx["manifest_writer"]
    record_index = ctx["record_index"]
    journal = pipeline._journal  # type: ignore[attr-defined]
    original_rollback = journal.rollback

    def _fail_append(record: object) -> bool:
        # Fail append after sidecar/payload; journal rollback is still invoked.
        del record
        raise RuntimeError("forced write failure")

    def _failing_rollback(transaction: object) -> None:
        # Attempt real cleanup first, then surface the journal error path by
        # raising after files were restored when possible.
        try:
            original_rollback(transaction)
        finally:
            raise OSError("forced journal rollback failure")

    monkeypatch.setattr(manifest_writer, "append", _fail_append)
    monkeypatch.setattr(journal, "rollback", _failing_rollback)

    with pytest.raises(ExceptionGroup) as raised:
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    messages = _exception_messages(raised.value)
    assert any("forced write failure" in msg for msg in messages)
    assert any("forced journal rollback failure" in msg for msg in messages)
    # Original rollback still deleted durable artifacts before the synthetic raise.
    assert _sidecar_paths(ctx["run_directory"]) == []  # type: ignore[arg-type]
    assert not ctx["absolute_payload"].exists()  # type: ignore[union-attr]
    assert len(record_index) == 0  # type: ignore[arg-type]
    assert manifest_writer.write_count == 0  # type: ignore[union-attr]


def test_rollback_continues_when_state_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    manifest_writer = ctx["manifest_writer"]
    record_index = ctx["record_index"]

    def _fail_append(record: object) -> bool:
        del record
        raise RuntimeError("forced write failure")

    def _fail_restore(state: object) -> None:
        del state
        raise RuntimeError("forced index restore failure")

    monkeypatch.setattr(manifest_writer, "append", _fail_append)
    monkeypatch.setattr(
        record_index,
        "restore",
        _fail_restore,
    )

    with pytest.raises(ExceptionGroup) as raised:
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    messages = _exception_messages(raised.value)
    assert any("forced write failure" in msg for msg in messages)
    assert any("forced index restore failure" in msg for msg in messages)
    # Journal rollback still removed on-disk artifacts.
    assert _sidecar_paths(ctx["run_directory"]) == []  # type: ignore[arg-type]
    assert not ctx["absolute_payload"].exists()  # type: ignore[union-attr]
    assert not list(
        (ctx["run_directory"] / "transactions").glob("*.json")  # type: ignore[operator]
    )


def test_rollback_method_collects_multiple_cleanup_errors(
    tmp_path: Path,
) -> None:
    """Unit-level: every cleanup step is attempted and errors are grouped."""

    class _BrokenManifest:
        def flush_transaction(self) -> None:
            raise OSError("manifest flush boom")

        def restore_transaction_count(self, count: int) -> None:
            del count
            raise RuntimeError("manifest restore boom")

    class _BrokenSync:
        def flush_transaction(self) -> None:
            raise OSError("sync flush boom")

        def restore(self, state: object) -> None:
            del state
            raise RuntimeError("sync restore boom")

    class _BrokenJournal:
        def rollback(self, transaction: object) -> None:
            del transaction
            raise OSError("journal rollback boom")

    class _BrokenIndex:
        def restore(self, state: object) -> None:
            del state
            raise RuntimeError("index restore boom")

    pipeline = DatasetWritePipeline.__new__(DatasetWritePipeline)
    pipeline._manifest_writer = _BrokenManifest()  # type: ignore[attr-defined]
    pipeline._sync_updater = _BrokenSync()  # type: ignore[attr-defined]
    pipeline._journal = _BrokenJournal()  # type: ignore[attr-defined]
    pipeline._record_index = _BrokenIndex()  # type: ignore[attr-defined]

    with pytest.raises(ExceptionGroup) as raised:
        pipeline._rollback_record_transaction(  # type: ignore[attr-defined]
            transaction=SimpleNamespace(),
            manifest_count=0,
            sync_state={},
            index_state=({}, {}),
        )

    assert len(raised.value.exceptions) == 6
    messages = " | ".join(str(exc) for exc in raised.value.exceptions)
    assert "manifest flush boom" in messages
    assert "sync flush boom" in messages
    assert "journal rollback boom" in messages
    assert "manifest restore boom" in messages
    assert "sync restore boom" in messages
    assert "index restore boom" in messages


def test_journal_write_removes_tmp_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "transactions" / "tx-1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    def _failing_replace(src: object, dst: object) -> None:
        raise OSError("forced journal replace failure")

    monkeypatch.setattr(
        "crawler.storage.datasets.writing.dataset_write_journal.os.replace",
        _failing_replace,
    )
    with pytest.raises(OSError, match="forced journal replace failure"):
        _write_json_atomic(path=path, payload={"schema_version": 1})

    assert not path.exists()
    assert not temp_path.exists()


def test_journal_begin_leaves_no_tmp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    journal = DatasetWriteJournal(run_directory=run_directory)
    payload = run_directory / "objects" / "x.bin"
    payload.parent.mkdir(parents=True, exist_ok=True)

    def _failing_replace(src: object, dst: object) -> None:
        raise OSError("forced begin replace failure")

    monkeypatch.setattr(
        "crawler.storage.datasets.writing.dataset_write_journal.os.replace",
        _failing_replace,
    )
    with pytest.raises(OSError, match="forced begin replace failure"):
        journal.begin(
            transaction_id="tx-begin-tmp",
            tracked_paths=(),
            payload_path=payload,
        )
    assert not list((run_directory / "transactions").glob("*.tmp"))
    assert not list((run_directory / "transactions").glob("*.json"))


def test_committed_marker_directory_fsync_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible committed marker is not success until commit returns."""

    from crawler.storage.datasets.writing import (
        dataset_write_journal as journal_mod,
    )

    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]

    real_fsync_directory = journal_mod._fsync_directory
    failure_triggered = False

    def _fail_when_committed_marker_is_visible(path: Path) -> None:
        nonlocal failure_triggered

        if not failure_triggered:
            for journal_path in path.glob("*.json"):
                payload = json.loads(journal_path.read_text(encoding="utf-8"))

                if payload.get("state") == "committed":
                    failure_triggered = True
                    raise OSError(
                        "forced committed-marker directory fsync failure"
                    )

        real_fsync_directory(path)

    monkeypatch.setattr(
        journal_mod,
        "_fsync_directory",
        _fail_when_committed_marker_is_visible,
    )

    with pytest.raises(
        OSError,
        match="forced committed-marker directory fsync failure",
    ):
        pipeline.execute(
            task=ctx["task"],
            result=ctx["result"],
            enrichment=ctx["enrichment"],
        )

    assert failure_triggered is True

    _assert_rolled_back(
        run_directory=ctx["run_directory"],
        absolute_payload=ctx["absolute_payload"],
        record_index=ctx["record_index"],
        manifest_writer=ctx["manifest_writer"],
        sync_updater=ctx["sync_updater"],
    )


def test_cleanup_failure_after_committed_marker_does_not_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failure after durable commit must not undo the write."""

    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    run_directory = ctx["run_directory"]
    absolute_payload = ctx["absolute_payload"]
    record_index = ctx["record_index"]

    original_unlink = Path.unlink

    def _block_committed_journal_unlink(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            self.suffix == ".json"
            and self.parent.name == "transactions"
            and self.is_file()
        ):
            text = self.read_text(encoding="utf-8")
            if '"state":"committed"' in text or '"state": "committed"' in text:
                raise OSError("forced journal cleanup unlink failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _block_committed_journal_unlink)

    warnings: list[str] = []

    class _WarnLogger(_Logger):
        def warning(self, event: str, **kwargs: object) -> None:
            warnings.append(event)

    pipeline._logger = _WarnLogger()  # type: ignore[attr-defined]

    outcome = pipeline.execute(  # type: ignore[union-attr]
        task=ctx["task"],  # type: ignore[arg-type]
        result=ctx["result"],  # type: ignore[arg-type]
        enrichment=ctx["enrichment"],  # type: ignore[arg-type]
    )

    assert outcome.record is not None
    assert absolute_payload.exists()  # type: ignore[union-attr]
    assert _sidecar_paths(run_directory)  # type: ignore[arg-type]
    assert len(record_index) == 1  # type: ignore[arg-type]
    assert ctx["manifest_writer"].write_count == 1  # type: ignore[union-attr]
    assert "dataset_transaction_committed_journal_cleanup_deferred" in warnings
    leftover = list((run_directory / "transactions").glob("*.json"))  # type: ignore[operator]
    assert leftover
    for journal_path in leftover:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        assert payload["state"] == "committed"


def test_recovery_finalizes_committed_journal_without_deleting_data(
    tmp_path: Path,
) -> None:
    from crawler.storage.datasets.writing.dataset_write_journal import (
        _payload_checksum,
    )

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    journal = DatasetWriteJournal(run_directory=run_directory)

    payload_path = run_directory / "objects" / "page.bin"
    sidecar_path = run_directory / "extraction" / "page" / "fetch.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"payload-bytes")
    sidecar_path.write_text('{"ok":true}', encoding="utf-8")

    body: dict[str, object] = {
        "schema_version": 1,
        "state": "committed",
        "transaction_id": "tx-committed-1",
        "tracked_paths": {
            "extraction/page/fetch.json": {"existed": False, "size": 0},
        },
        "payload_path": "objects/page.bin",
        "payload_existed": False,
    }
    body["checksum"] = _payload_checksum(body)
    journal_path = run_directory / "transactions" / "tx-committed-1.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        json.dumps(body, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    result = journal.recover_pending()
    assert result.rolled_back == ()
    assert result.finalized_commits == ("tx-committed-1",)
    assert not journal_path.exists()
    assert payload_path.exists()
    assert sidecar_path.exists()
    assert payload_path.read_bytes() == b"payload-bytes"


def test_recovery_rolls_back_pending_journal(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    journal = DatasetWriteJournal(run_directory=run_directory)

    payload_path = run_directory / "objects" / "page.bin"
    sidecar_path = run_directory / "extraction" / "page" / "fetch.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    transaction = journal.begin(
        transaction_id="tx-pending-1",
        tracked_paths=(sidecar_path,),
        payload_path=payload_path,
    )
    # Simulate work after begin.
    payload_path.write_bytes(b"new-payload")
    sidecar_path.write_text('{"sidecar":true}', encoding="utf-8")
    assert transaction.journal_path.exists()
    pending = json.loads(transaction.journal_path.read_text(encoding="utf-8"))
    assert pending["state"] == "pending"

    result = journal.recover_pending()
    assert result.rolled_back == ("tx-pending-1",)
    assert result.finalized_commits == ()
    assert not transaction.journal_path.exists()
    assert not payload_path.exists()
    assert not sidecar_path.exists()


def test_failure_before_committed_marker_still_rollbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If durable commit marker write fails, data remains rollbackable."""

    from crawler.storage.datasets.writing import (
        dataset_write_journal as journal_mod,
    )

    ctx = _build_pipeline(tmp_path)
    pipeline = ctx["pipeline"]
    real_write = journal_mod._write_json_atomic
    write_calls = {"n": 0}

    def _fail_commit_marker(*, path: Path, payload: dict) -> None:
        write_calls["n"] += 1
        # begin() succeeds; commit's state rewrite fails.
        if payload.get("state") == "committed":
            raise OSError("forced committed-marker write failure")
        real_write(path=path, payload=payload)

    monkeypatch.setattr(journal_mod, "_write_json_atomic", _fail_commit_marker)

    with pytest.raises(BaseException) as raised:
        pipeline.execute(  # type: ignore[union-attr]
            task=ctx["task"],  # type: ignore[arg-type]
            result=ctx["result"],  # type: ignore[arg-type]
            enrichment=ctx["enrichment"],  # type: ignore[arg-type]
        )

    messages = _exception_messages(raised.value)
    assert any(
        "forced committed-marker write failure" in msg for msg in messages
    )
    _assert_rolled_back(
        run_directory=ctx["run_directory"],  # type: ignore[arg-type]
        absolute_payload=ctx["absolute_payload"],  # type: ignore[arg-type]
        record_index=ctx["record_index"],  # type: ignore[arg-type]
        manifest_writer=ctx["manifest_writer"],  # type: ignore[arg-type]
        sync_updater=ctx["sync_updater"],  # type: ignore[arg-type]
    )
    assert write_calls["n"] >= 2
