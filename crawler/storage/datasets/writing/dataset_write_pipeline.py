"""Execute the raw dataset object and manifest write pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crawler.storage.datasets.records.dataset_record import (
    derive_media_identity,
)
from crawler.storage.datasets.writing.dataset_write_journal import (
    DatasetWriteJournal,
)
from crawler.storage.datasets.writing.derived_media_artifact import (
    PreparedDerivedMediaWrite,
)
from crawler.storage.datasets.writing.write_outcome import (
    WriteOperation,
    WriteOutcome,
    record_is_coverage_eligible,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.fetching.response.cache import ConditionalRepresentationCache
    from crawler.fetching.results.result import FetchResult
    from crawler.storage.datasets.extraction.page_extraction_artifact import (
        PageExtractionArtifact,
        PreparedPageExtractionWrite,
    )
    from crawler.storage.datasets.manifests.dataset_manifest_writer import (
        DatasetManifestWriter,
    )
    from crawler.storage.datasets.records.dataset_record import (
        DatasetRecord,
        DatasetRecordCreator,
    )
    from crawler.storage.datasets.records.record_index import (
        DatasetRecordIndex,
    )
    from crawler.storage.datasets.sync_index.sync_index_updater import (
        SyncIndexUpdater,
    )
    from crawler.storage.datasets.writing.raw_payload_writer import (
        RawPayloadWriter,
    )

_CONTENT_CHANGED_REASON = "content_changed"
_NORMALIZED_MEDIA_KEYS = frozenset(
    {"normalized_media_path", "normalized_video_path"}
)
_KEYFRAMES_KEY = "keyframes"
_FRAME_PATH_KEY = "frame_path"


class DatasetWritePipeline:
    """Coordinate object persistence, manifest append, and sync index updates."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        logger: ProjectLogger,
        url_normalizer: UrlNormalizer,
        payload_writer: RawPayloadWriter,
        manifest_writer: DatasetManifestWriter,
        sync_updater: SyncIndexUpdater,
        record_creator: DatasetRecordCreator,
        record_index: DatasetRecordIndex,
        conditional_representation_cache: ConditionalRepresentationCache,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._url_normalizer = url_normalizer
        self._payload_writer = payload_writer
        self._manifest_writer = manifest_writer
        self._sync_updater = sync_updater
        self._record_creator = record_creator
        self._record_index = record_index
        self._conditional_representation_cache = (
            conditional_representation_cache
        )
        self._journal = DatasetWriteJournal(
            run_directory=payload_writer.run_directory
        )
        self._journal.recover_pending()

    def execute(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        enrichment: Mapping[str, Any] | None,
    ) -> WriteOutcome:
        """Persist one fetched result through the full write pipeline.

        Returns WriteOutcome (with previous/current kind) so coverage
        accounting can handle inserts, updates, and kind changes correctly.
        """

        normalized_final_url = self._url_normalizer.normalize(result.final_url)
        resolved_kind = result.kind.value
        resolved_modality = result.kind.value
        relative_path, content_hash, byte_size, inline_body = (
            self._payload_writer.prepare(
                result=result,
                kind=resolved_kind,
                modality=resolved_modality,
            )
        )
        media_identity = derive_media_identity(
            task=task,
            normalized_url=normalized_final_url,
            kind=resolved_kind,
        )

        existing_record = self._record_index.find_latest(
            normalized_url=normalized_final_url,
            media_identity=media_identity,
        )
        previous_record = self._resolve_previous_record(
            existing_record=existing_record,
            content_hash=content_hash,
        )

        if self._should_skip_existing_record(
            existing_record=existing_record,
            previous_record=previous_record,
        ):
            return self._duplicate_outcome(
                task=task,
                result=result,
                existing_record=existing_record,
                normalized_url=normalized_final_url,
                resolved_kind=resolved_kind,
            )

        if previous_record is not None:
            self._logger.info(
                "dataset_record_updating_existing_url",
                requested_url=task.url,
                final_url=result.final_url,
                normalized_url=normalized_final_url,
                previous_fetch_record_id=previous_record.fetch_record_id,
            )

        compact_enrichment, page_artifact = (
            _extract_page_artifact_from_enrichment(enrichment=enrichment)
        )

        record = self._record_creator.create(
            task=task,
            result=result,
            byte_size=byte_size,
            content_sha256=content_hash,
            relative_path=relative_path,
            normalized_url=normalized_final_url,
            kind=resolved_kind,
            modality=resolved_modality,
            previous_record=previous_record,
            enrichment=compact_enrichment,
        )

        prepared_extraction = None
        if page_artifact is not None and record.kind == "page":
            record, prepared_extraction = _prepare_page_extraction_artifact(
                record=record,
                run_directory=self._payload_writer.run_directory,
                artifact=page_artifact,
            )

        record, prepared_media = _prepare_derived_media_from_enrichment(
            record=record,
            run_directory=self._payload_writer.run_directory,
        )

        self._commit_record(
            record=record,
            result=result,
            previous_record=previous_record,
            relative_path=relative_path,
            inline_body=inline_body,
            prepared_extraction=prepared_extraction,
            prepared_media=prepared_media,
        )

        self._logger.debug(
            "dataset_record_written",
            path=relative_path.as_posix(),
            requested_url=task.url,
            final_url=result.final_url,
            normalized_url=record.normalized_url,
            kind=record.kind,
            modality=record.modality,
            bytes=record.byte_size,
        )

        # New coverage credit only for inserts that are not merely updates
        # of an existing content-hash-differing record. Updates are tracked
        # via previous_record.
        prev_kind = (
            getattr(previous_record, "kind", None)
            if previous_record is not None
            else getattr(existing_record, "kind", None)
        )
        curr_kind = getattr(record, "kind", None) or str(resolved_kind)

        return WriteOutcome(
            record=record,
            operation=(
                WriteOperation.UPDATE
                if previous_record is not None
                else WriteOperation.INSERT
            ),
            previous_kind=str(prev_kind) if prev_kind else None,
            current_kind=str(curr_kind),
            previous_coverage_eligible=record_is_coverage_eligible(
                previous_record or existing_record
            ),
        )

    def _duplicate_outcome(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        existing_record: DatasetRecord | None,
        normalized_url: str,
        resolved_kind: str,
    ) -> WriteOutcome:
        if existing_record is None:
            raise RuntimeError(
                "duplicate dataset skip requires an existing record"
            )
        self._logger.debug(
            "dataset_record_skipped_duplicate_url",
            requested_url=task.url,
            final_url=result.final_url,
            normalized_url=normalized_url,
            original_fetch_record_id=existing_record.fetch_record_id,
        )
        self._payload_writer.discard_payload(result=result)
        previous_kind = getattr(existing_record, "kind", None)
        current_kind = str(previous_kind) if previous_kind else None
        return WriteOutcome(
            record=existing_record,
            operation=WriteOperation.DUPLICATE,
            previous_kind=current_kind,
            current_kind=current_kind or str(resolved_kind),
            previous_coverage_eligible=record_is_coverage_eligible(
                existing_record
            ),
        )

    def _commit_record(
        self,
        *,
        record: DatasetRecord,
        result: FetchResult,
        previous_record: DatasetRecord | None,
        relative_path: Path,
        inline_body: bytes | None,
        prepared_extraction: PreparedPageExtractionWrite | None = None,
        prepared_media: tuple[PreparedDerivedMediaWrite, ...] = (),
    ) -> None:
        from crawler.storage.datasets.extraction.page_extraction_artifact import (
            PageExtractionArtifactWriter,
        )
        from crawler.storage.datasets.writing.derived_media_artifact import (
            DerivedMediaArtifactWriter,
            consume_derived_media_source,
        )

        manifest_count = self._manifest_writer.prepare_transaction()
        sync_state = self._sync_updater.snapshot()
        index_state = self._record_index.snapshot()
        tracked: list[Path] = [
            self._manifest_writer.manifest_path,
            *self._sync_updater.prepare_transaction(modality=record.modality),
        ]
        if prepared_extraction is not None:
            # File must not exist yet so journal records existed=False.
            tracked.append(prepared_extraction.absolute_path)
        tracked.extend(prepared.absolute_path for prepared in prepared_media)
        tracked_paths = tuple(tracked)
        transaction = self._journal.begin(
            transaction_id=record.fetch_record_id,
            tracked_paths=tracked_paths,
            payload_path=self._payload_writer.absolute_path(
                relative_path=relative_path
            ),
        )
        try:
            if prepared_extraction is not None:
                PageExtractionArtifactWriter(
                    run_directory=self._payload_writer.run_directory
                ).commit(prepared=prepared_extraction)
            if prepared_media:
                artifact_writer = DerivedMediaArtifactWriter(
                    run_directory=self._payload_writer.run_directory
                )
                for prepared in prepared_media:
                    artifact_writer.commit(prepared=prepared)
            self._payload_writer.persist_prepared(
                relative_path=relative_path,
                result=result,
                inline_body=inline_body,
            )
            self._manifest_writer.append(record)
            self._sync_updater.append_record_indexes(
                record=record,
                result=result,
            )
            if previous_record is not None:
                self._sync_updater.append_updates_manifest(
                    previous_record=previous_record,
                    updated_record=record,
                    reason=_CONTENT_CHANGED_REASON,
                )
            self._record_index.register(record=record)
            self._manifest_writer.flush_transaction()
            self._sync_updater.flush_transaction()
            cleanup_completed = self._journal.commit(transaction)
            if not cleanup_completed:
                # Durable commit marker is present; leftover journal is
                # finalized on next recover_pending() without rolling back.
                self._logger.warning(
                    "dataset_transaction_committed_journal_cleanup_deferred",
                    transaction_id=transaction.transaction_id,
                )
            self._publish_conditional_representation(
                result=result,
                durable_payload_path=self._payload_writer.absolute_path(
                    relative_path=relative_path
                ),
            )
        except BaseException as operation_error:
            # One cleanup failure must not skip later rollback steps.
            try:
                self._rollback_record_transaction(
                    transaction=transaction,
                    manifest_count=manifest_count,
                    sync_state=sync_state,
                    index_state=index_state,
                )
            except BaseException as rollback_error:
                raise BaseExceptionGroup(
                    "dataset write and rollback both failed",
                    [operation_error, rollback_error],
                ) from operation_error
            raise
        finally:
            # Scratch sources are consumed after the transaction either
            # committed or rolled back; they are never persisted as-is.
            for prepared in prepared_media:
                try:
                    consume_derived_media_source(prepared)
                except OSError as exc:
                    self._logger.warning(
                        "derived_media_source_cleanup_failed",
                        path=str(prepared.source_path),
                        error_type=type(exc).__name__,
                    )

    def _publish_conditional_representation(
        self,
        *,
        result: FetchResult,
        durable_payload_path: Path,
    ) -> None:
        """Publish validators only after the dataset transaction is durable."""

        payload = result.payload
        if payload is None:
            return

        durable_result = replace(
            result,
            payload=replace(payload, temp_path=durable_payload_path),
        )
        try:
            self._conditional_representation_cache.commit_response(
                requested_url=result.url,
                final_url=result.final_url,
                headers=result.headers,
                result=durable_result,
            )
        except Exception as exc:
            # Cache publication is intentionally outside the durable write
            # outcome: a cache issue must never roll back committed data.
            self._logger.warning(
                "conditional_representation_cache_publish_failed",
                requested_url=result.url,
                final_url=result.final_url,
                error_type=type(exc).__name__,
            )

    def _rollback_record_transaction(
        self,
        *,
        transaction: Any,
        manifest_count: int,
        sync_state: dict[str, object],
        index_state: Any,
    ) -> None:
        """Best-effort full rollback; never skip later steps after a failure.

        Collects every cleanup error and raises ``ExceptionGroup`` only after
        all steps have been attempted, so a flush failure cannot leave
        sidecars, payloads, or journal files behind.
        """

        cleanup_errors: list[BaseException] = []

        try:
            self._manifest_writer.flush_transaction()
        except BaseException as exc:
            cleanup_errors.append(exc)

        try:
            self._sync_updater.flush_transaction()
        except BaseException as exc:
            cleanup_errors.append(exc)

        try:
            self._journal.rollback(transaction)
        except BaseException as exc:
            cleanup_errors.append(exc)

        try:
            self._manifest_writer.restore_transaction_count(manifest_count)
        except BaseException as exc:
            cleanup_errors.append(exc)

        try:
            self._sync_updater.restore(sync_state)
        except BaseException as exc:
            cleanup_errors.append(exc)

        try:
            self._record_index.restore(index_state)
        except BaseException as exc:
            cleanup_errors.append(exc)

        if cleanup_errors:
            raise BaseExceptionGroup(
                "dataset transaction rollback failed",
                cleanup_errors,
            )

    def _resolve_previous_record(
        self,
        *,
        existing_record: DatasetRecord | None,
        content_hash: str,
    ) -> DatasetRecord | None:
        if existing_record is None:
            return None

        if not self._settings.deduplicate_within_run_by_normalized_url:
            return None

        if not self._settings.enable_record_updates:
            return None

        if existing_record.content_sha256 == content_hash:
            return None

        return existing_record

    def _should_skip_existing_record(
        self,
        *,
        existing_record: DatasetRecord | None,
        previous_record: DatasetRecord | None,
    ) -> bool:
        if existing_record is None:
            return False

        if not self._settings.deduplicate_within_run_by_normalized_url:
            return False

        return previous_record is None


def _extract_page_artifact_from_enrichment(
    *,
    enrichment: Mapping[str, Any] | None,
) -> tuple[dict[str, object], PageExtractionArtifact | None]:
    from crawler.storage.datasets.extraction.page_extraction_artifact import (
        strip_page_extraction_artifact_from_enrichment,
    )

    return strip_page_extraction_artifact_from_enrichment(enrichment)


def _prepare_page_extraction_artifact(
    *,
    record: DatasetRecord,
    run_directory: Path,
    artifact: PageExtractionArtifact,
) -> tuple[DatasetRecord, PreparedPageExtractionWrite]:
    from crawler.storage.datasets.extraction.page_extraction_artifact import (
        PAGE_EXTRACTION_SCHEMA_VERSION,
        PageExtractionArtifactWriter,
    )

    prepared = PageExtractionArtifactWriter(
        run_directory=run_directory
    ).prepare(
        fetch_record_id=record.fetch_record_id,
        artifact=artifact,
    )
    compact_enrichment = {
        **dict(record.enrichment),
        "page_extraction_path": prepared.relative_path,
        "page_extraction_sha256": prepared.sha256,
        "page_extraction_schema_version": PAGE_EXTRACTION_SCHEMA_VERSION,
    }
    # Never keep bulk text/markdown in the JSONL enrichment line.
    for bulk_key in (
        "page_markdown",
        "page_headings",
        "page_code_block_count",
        "page_boilerplate_ratio",
        "page_text",
        "text",
        "markdown",
    ):
        compact_enrichment.pop(bulk_key, None)

    updated = record.model_copy(update={"enrichment": compact_enrichment})
    return updated, prepared


def _prepare_derived_media_from_enrichment(
    *,
    record: DatasetRecord,
    run_directory: Path,
) -> tuple[DatasetRecord, tuple[PreparedDerivedMediaWrite, ...]]:
    """Relocate derived media references into run-owned relative paths.

    Normalized image/video paths and selected keyframe paths that point at
    temporary scratch files are prepared as run-owned copies to be committed
    inside the existing write journal transaction. The enrichment values are
    rewritten to those relative run paths so the record stays self-contained.
    """

    from crawler.storage.datasets.writing.derived_media_artifact import (
        DerivedMediaArtifactWriter,
    )

    writer = DerivedMediaArtifactWriter(run_directory=run_directory)
    prepared: list[PreparedDerivedMediaWrite] = []
    enrichment = dict(record.enrichment)

    for key in _NORMALIZED_MEDIA_KEYS:
        value = enrichment.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        write = _prepare_media_scratch_file(
            writer=writer,
            record=record,
            source_path=Path(value),
        )
        prepared.append(write)
        enrichment[key] = write.relative_path

    keyframes = enrichment.get(_KEYFRAMES_KEY)
    if isinstance(keyframes, list):
        relocated_frames: list[dict[str, object]] = []
        for frame in keyframes:
            if not isinstance(frame, dict):
                relocated_frames.append(frame)
                continue
            updated_frame = dict(frame)
            frame_path = updated_frame.get(_FRAME_PATH_KEY)
            if isinstance(frame_path, str) and frame_path.strip():
                write = _prepare_media_scratch_file(
                    writer=writer,
                    record=record,
                    source_path=Path(frame_path),
                )
                prepared.append(write)
                updated_frame[_FRAME_PATH_KEY] = write.relative_path
            relocated_frames.append(updated_frame)
        enrichment[_KEYFRAMES_KEY] = relocated_frames

    if not prepared:
        return record, ()

    updated = record.model_copy(update={"enrichment": enrichment})
    return updated, tuple(prepared)


def _prepare_media_scratch_file(
    *,
    writer: Any,
    record: DatasetRecord,
    source_path: Path,
) -> PreparedDerivedMediaWrite:
    """Prepare one scratch file as a run-owned transactional artifact.

    Raises when the referenced scratch file does not exist, so a stale
    enrichment reference never becomes a dataset record pointing outside
    its run directory.
    """

    if not source_path.is_file():
        raise ValueError(
            f"derived media artifact source missing: {source_path}"
        )

    write: PreparedDerivedMediaWrite = writer.prepare(
        source_path=source_path,
        fetch_record_id=record.fetch_record_id,
        artifact_name=source_path.name,
    )
    return write
