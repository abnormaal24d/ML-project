"""Canonical text/document preprocessing orchestrator."""

from __future__ import annotations

import hashlib
import os
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

from preprocessing.preprocessed_document import (
    PreprocessedDocument,
)
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.preprocessing_result import (
    PreprocessingQuarantineRecord,
    PreprocessingResult,
)
from preprocessing.text.text_metadata import build_text_metadata
from preprocessing.text.text_preparation import (
    PreparedTextInput,
    TextInputPreparer,
    build_document_id,
    prepared_text_diagnostics,
    validate_text_input,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from config.preprocessing.text_settings import (
        PreprocessingInputValidationSettings,
    )
    from logger.project_logger import ProjectLogger
    from preprocessing.text.text_quality import TextQualityScorer

_T = TypeVar("_T")
_U = TypeVar("_U")


class TextPreprocessor:
    """Build privacy-cleared, quality-scored text documents."""

    def __init__(
        self,
        *,
        quality_scorer: TextQualityScorer,
        input_validation: PreprocessingInputValidationSettings,
        input_preparer: TextInputPreparer,
        logger: ProjectLogger,
        max_workers: int | None = None,
        batch_size: int = 64,
        parallel_batch_min_size: int = 4,
    ) -> None:
        self._max_workers = _resolve_worker_count(max_workers=max_workers)
        self._parallel_batch_min_size = max(2, parallel_batch_min_size)
        self._batch_size = max(1, batch_size)
        self._input_preparer = input_preparer
        self._quality_scorer = quality_scorer
        self._input_validation = input_validation
        self._logger = logger

    def process(
        self,
        *,
        inputs: Iterable[PreprocessingInput],
    ) -> PreprocessingResult:
        """Process text/document inputs without dataset-level selection."""

        documents: list[PreprocessedDocument] = []
        skipped_sources: dict[str, str] = {}
        quarantine_records: list[PreprocessingQuarantineRecord] = []
        input_count = 0

        executor = (
            ThreadPoolExecutor(max_workers=self._max_workers)
            if self._max_workers > 1
            else None
        )
        try:
            for batch in _iter_batches(items=inputs, size=self._batch_size):
                input_count += len(batch)
                documents.extend(
                    self._process_batch(
                        batch=batch,
                        skipped_sources=skipped_sources,
                        quarantine_records=quarantine_records,
                        executor=executor,
                    )
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        skipped_by_reason = dict(Counter(skipped_sources.values()))
        diagnostics = {
            "content_keys_emitted": len(documents),
            "dataset_selection_owner": "curation",
            "skipped_by_reason": skipped_by_reason,
        }
        self._logger.info(
            "preprocessing_workflow_completed",
            inputs=input_count,
            emitted=len(documents),
            skipped=len(skipped_sources),
            skipped_by_reason=skipped_by_reason,
        )
        return PreprocessingResult(
            documents=tuple(documents),
            skipped_sources=skipped_sources,
            quarantine_records=tuple(quarantine_records),
            diagnostics=diagnostics,
        )

    def _process_batch(
        self,
        *,
        batch: tuple[PreprocessingInput, ...],
        skipped_sources: dict[str, str],
        quarantine_records: list[PreprocessingQuarantineRecord],
        executor: ThreadPoolExecutor | None,
    ) -> list[PreprocessedDocument]:
        valid_inputs: list[tuple[int, PreprocessingInput]] = []
        validation_rejections: dict[int, str] = {}
        for batch_index, item in enumerate(batch):
            if item.modality not in {"text", "document"}:
                validation_rejections[batch_index] = (
                    f"unsupported_text_preprocessor_modality:{item.modality}"
                )
                continue
            rejection = validate_text_input(
                item=item,
                settings=self._input_validation,
            )
            if rejection is not None:
                validation_rejections[batch_index] = rejection
                continue
            valid_inputs.append((batch_index, item))

        prepared_inputs = _map_batch(
            func=self._prepare_safe,
            items=valid_inputs,
            max_workers=self._max_workers,
            parallel_batch_min_size=self._parallel_batch_min_size,
            executor=executor,
        )
        accepted = self._accept_prepared(
            batch=batch,
            validation_rejections=validation_rejections,
            prepared_inputs=prepared_inputs,
            skipped_sources=skipped_sources,
            quarantine_records=quarantine_records,
        )
        return _map_batch(
            func=self._build_document,
            items=accepted,
            max_workers=self._max_workers,
            parallel_batch_min_size=self._parallel_batch_min_size,
            executor=executor,
        )

    def _prepare_safe(
        self,
        indexed: tuple[int, PreprocessingInput],
    ) -> PreparedTextInput:
        batch_index, item = indexed
        try:
            prepared = self._input_preparer.prepare_valid_input(
                batch_index=batch_index,
                item=item,
            )
            if prepared.rejection_reason is None:
                if prepared.extracted_document is None:
                    raise ValueError(
                        "prepared input is missing extracted document"
                    )
                if prepared.approved_structure_payload is None:
                    raise ValueError(
                        "prepared input is missing privacy-cleared structure"
                    )
                if prepared.source_normalized_text is None:
                    raise ValueError(
                        "prepared input is missing normalized source text"
                    )
                if prepared.document_structure is None:
                    raise ValueError(
                        "prepared input is missing canonical structure"
                    )
            return prepared
        except (RuntimeError, OSError, TypeError, ValueError) as error:
            error_type = (
                "ValueError"
                if isinstance(error, ValueError)
                else type(error).__name__
            )
            warning = self._logger.warning
            if callable(warning):
                warning(
                    "preprocessing_item_failed",
                    source_id=item.source_id,
                    modality=item.modality,
                    source_url=item.source_url,
                    error_type=error_type,
                    traceback_hash=_traceback_hash(error=error),
                )
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=None,
                normalized_text="",
                rejection_reason=f"preprocessing_exception:{error_type}",
            )

    def _accept_prepared(
        self,
        *,
        batch: tuple[PreprocessingInput, ...],
        validation_rejections: dict[int, str],
        prepared_inputs: list[PreparedTextInput],
        skipped_sources: dict[str, str],
        quarantine_records: list[PreprocessingQuarantineRecord],
    ) -> list[PreparedTextInput]:
        prepared_by_index = {
            prepared.batch_index: prepared for prepared in prepared_inputs
        }
        accepted: list[PreparedTextInput] = []
        for batch_index, item in enumerate(batch):
            rejection = validation_rejections.get(batch_index)
            if rejection is not None:
                _record_rejection(
                    item=item,
                    reason=rejection,
                    skipped_sources=skipped_sources,
                    quarantine_records=quarantine_records,
                )
                continue
            prepared = prepared_by_index[batch_index]
            if prepared.rejection_reason is not None:
                finding_counts = (
                    prepared.pii_result.finding_counts
                    if prepared.pii_result is not None
                    else None
                )
                pii_spans = (
                    prepared.pii_result.spans
                    if prepared.pii_result is not None
                    else ()
                )
                _record_rejection(
                    item=item,
                    reason=prepared.rejection_reason,
                    skipped_sources=skipped_sources,
                    quarantine_records=quarantine_records,
                    finding_counts=finding_counts,
                    pii_spans=pii_spans,
                )
                continue
            if prepared.exact_duplicate_key is None:
                raise ValueError("prepared input is missing exact content key")
            accepted.append(prepared)
        return accepted

    def _build_document(
        self,
        prepared: PreparedTextInput,
    ) -> PreprocessedDocument:
        if prepared.extracted_document is None:
            raise ValueError("prepared input is missing extracted document")
        if prepared.exact_duplicate_key is None:
            raise ValueError("prepared input is missing exact dedupe key")
        if prepared.approved_structure_payload is None:
            raise ValueError(
                "prepared input is missing privacy-cleared structure"
            )
        if prepared.source_normalized_text is None:
            raise ValueError(
                "prepared input is missing normalized source text"
            )
        if prepared.document_structure is None:
            raise ValueError("prepared input is missing canonical structure")
        item = prepared.item
        quality = self._quality_scorer.score(
            text=prepared.normalized_text,
            boilerplate_ratio=prepared.extracted_document.boilerplate_ratio,
            language_evidence=item.resolved_language_evidence(),
        )
        structure = prepared.document_structure
        metadata = build_text_metadata(
            extracted_document=prepared.extracted_document,
            normalized_text=prepared.normalized_text,
            language=quality.language or item.resolved_language(),
            path=prepared.public_path or item.path,
            title=prepared.extracted_document.title,
            rejection_reason=quality.rejection_reason,
            diagnostics=prepared_text_diagnostics(prepared=prepared),
        )
        return PreprocessedDocument(
            document_id=build_document_id(
                normalized_url=prepared.public_source_url
                or item.normalized_url,
                exact_duplicate_key=prepared.exact_duplicate_key,
            ),
            source_id=item.source_id,
            source_url=prepared.public_source_url or item.normalized_url,
            title=prepared.extracted_document.title,
            text=prepared.normalized_text,
            markdown=prepared.extracted_document.markdown,
            language=quality.language or item.resolved_language(),
            metadata=metadata,
            quality=quality,
            exact_duplicate_key=prepared.exact_duplicate_key,
            near_duplicate_cluster_id=None,
            is_near_duplicate=False,
            warnings=prepared.extracted_document.warnings,
            domain=item.domain,
            path=prepared.public_path or item.path,
            allow_training=_optional_bool(item.payload.get("allow_training")),
            privacy_clearance=prepared.privacy_clearance,
            **structure,
        )


def _iter_batches(
    *,
    items: Iterable[_T],
    size: int,
) -> Iterable[tuple[_T, ...]]:
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _resolve_worker_count(*, max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, max_workers)
    return max(1, min(8, os.cpu_count() or 1))


def _traceback_hash(*, error: BaseException) -> str:
    formatted = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return hashlib.sha256(formatted.encode("utf-8")).hexdigest()[:16]


def _map_batch(
    *,
    func: Callable[[_T], _U],
    items: list[_T],
    max_workers: int,
    parallel_batch_min_size: int,
    executor: ThreadPoolExecutor | None,
) -> list[_U]:
    if not items:
        return []
    if max_workers <= 1 or len(items) < parallel_batch_min_size:
        return [func(item) for item in items]
    if executor is None:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(items))
        ) as local:
            return list(local.map(func, items))
    return list(executor.map(func, items))


def _record_rejection(
    *,
    item: PreprocessingInput,
    reason: str,
    skipped_sources: dict[str, str],
    quarantine_records: list[PreprocessingQuarantineRecord],
    finding_counts: dict[str, int] | None = None,
    pii_spans: tuple[dict[str, object], ...] = (),
) -> None:
    skipped_sources[item.source_id] = reason
    quarantine_records.append(
        PreprocessingQuarantineRecord.from_input(
            item=item,
            reason=reason,
            finding_counts=finding_counts,
            pii_spans=pii_spans,
        )
    )


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
