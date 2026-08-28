"""Training sample construction for curated documents."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from mmcrawler_datasets.assembly.sample_components import (
    _build_document_text_spans,
)
from mmcrawler_datasets.assembly.text_pairing import (
    DocumentTextRejection,
    pairability_score,
    read_document_text,
)
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.splitting.group_keys import document_group
from mmcrawler_datasets.training_samples.common import (
    estimate_token_count,
    stable_sample_id,
)
from mmcrawler_datasets.training_samples.models import (
    GovernanceEvidence,
    TrainingSample,
)
from mmcrawler_datasets.training_samples.targets import (
    ConversationTurn,
    TrainingTaskTarget,
)
from multimodal.tasks.registry import require_task
from preprocessing.privacy.clearance import PrivacyClearance
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION


def build_doc_samples(
    records: tuple[CuratedDocumentRecord, ...],
    splits: dict[str, str],
    *,
    split_assigner: SplitAssigner,
    require_allow_training: bool,
    snapshot_id: str,
    snapshot_directory: Path,
    text_cache: dict[str, str | None],
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION,
    enabled_tasks: frozenset[str] | None = None,
) -> tuple[TrainingSample, ...]:
    """Build document-text, PDF-text, and grounded document-QA samples."""

    samples: list[TrainingSample] = []
    for document in records:
        if (
            document.quality_bucket == "reject"
            or document.privacy_clearance is None
            or not document.privacy_clearance.permits_training
        ):
            continue
        if require_allow_training and document.allow_training is not True:
            continue
        try:
            text = read_document_text(
                snapshot_directory=snapshot_directory,
                text_path=document.text_path,
                privacy_clearance=document.privacy_clearance,
                cache=text_cache,
            )
        except DocumentTextRejection:
            continue
        if not text:
            continue

        emitted_text = text[:2000]
        try:
            clearance = PrivacyClearance.from_dict(
                document.privacy_clearance.bind_training_text(
                    emitted_text,
                    source_name="body",
                    start=0,
                    end=len(emitted_text),
                ).to_dict()
            )
        except ValueError:
            continue

        group_key = document_group(document)
        split = splits.get(group_key) or split_assigner.assign(key=group_key)
        splits[group_key] = split
        spans, page_count = _build_document_text_spans(text=text)
        pair_definition = require_task("document_text_pair")
        pair_score = pairability_score(
            media_record=document.to_dict(),
            parent_text=None,
        )
        base_sample = TrainingSample(
            schema_version=schema_version,
            sample_id=stable_sample_id(
                document.document_id,
                split,
                prefix="doc",
            ),
            snapshot_id=snapshot_id,
            split=split,
            modality="document",
            task_target=TrainingTaskTarget(
                task_type=pair_definition.name,
                task_family=pair_definition.family,
                target_text=emitted_text,
                positive_id=document.document_id,
            ),
            document_id=document.document_id,
            object_id=document.object_id,
            text=emitted_text,
            paired_text_source="document_body",
            token_count_estimate=estimate_token_count(emitted_text),
            language=document.language,
            title=document.title,
            domain=document.domain,
            source_url=document.final_url,
            governance=GovernanceEvidence.from_record(document),
            quality_score=document.quality_score,
            pairability_score=max(pair_score, document.quality_score),
            pair_source="document_body",
            exact_duplicate_key=document.exact_duplicate_key,
            near_duplicate_cluster_id=document.near_duplicate_cluster_id,
            text_spans=spans,
            source_document_id=document.document_id,
            normalized_url=document.normalized_url,
            content_family_id=group_key,
            alignment_group_id=group_key,
            page_range_start=1,
            page_range_end=page_count,
            privacy_clearance=clearance,
            builder_source="curated_document_text_pair",
        )
        samples.append(base_sample)

        if (
            enabled_tasks is not None
            and "pdf_text_pair" in enabled_tasks
            and _is_pdf_document(document)
        ):
            pdf_definition = require_task("pdf_text_pair")
            samples.append(
                replace(
                    base_sample,
                    sample_id=f"{base_sample.sample_id}:pdf_text_pair",
                    task_target=TrainingTaskTarget(
                        task_type=pdf_definition.name,
                        task_family=pdf_definition.family,
                        target_text=emitted_text,
                        positive_id=document.document_id,
                    ),
                    builder_source="curated_pdf_text_pair",
                    content_hash=None,
                )
            )

        if enabled_tasks is not None and "doc_qa" in enabled_tasks:
            qa_sample = _build_document_qa_sample(
                sample=base_sample,
                document=document,
            )
            if qa_sample is not None:
                samples.append(qa_sample)

    return tuple(samples)


def _is_pdf_document(document: CuratedDocumentRecord) -> bool:
    candidates = (
        document.path,
        document.final_url,
        document.raw_storage_path,
    )
    return any(
        Path(urlparse(str(value)).path).suffix.lower() == ".pdf"
        for value in candidates
        if value
    )


def _build_document_qa_sample(
    *,
    sample: TrainingSample,
    document: CuratedDocumentRecord,
) -> TrainingSample | None:
    """Build a verifiable metadata-grounded QA sample when title is cleared."""

    title = str(document.title or "").strip()
    if not title or document.privacy_clearance is None:
        return None
    try:
        clearance = PrivacyClearance.from_dict(
            document.privacy_clearance.bind_training_text(
                title,
                source_name="title",
            ).to_dict()
        )
    except ValueError:
        return None

    title_start = sample.text.casefold().find(title.casefold())
    if title_start < 0:
        return None
    title_end = title_start + len(title)
    page_number = next(
        (
            span.page_number
            for span in sample.text_spans
            if span.text_start is not None
            and span.text_end is not None
            and span.text_start <= title_start < span.text_end
        ),
        1,
    )
    evidence_id = (
        f"{document.document_id}:page:{page_number}:"
        f"span:{title_start}:{title_end}"
    )
    question = "What is the title of this document?"
    definition = require_task("doc_qa")
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:doc_qa",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            question=question,
            target_text=title,
            answer=title,
            user_text=question,
            assistant_text=title,
            conversation_turns=(
                ConversationTurn(role="user", text=question, turn_index=0),
                ConversationTurn(
                    role="assistant",
                    text=title,
                    turn_index=1,
                    answer_evidence_ids=(evidence_id,),
                    is_assistant_answer=True,
                ),
            ),
            answer_evidence_ids=(evidence_id,),
            evidence_records=(
                {
                    "evidence_id": evidence_id,
                    "relation_type": "document_span_contains_answer",
                    "document_id": document.document_id,
                    "page_number": page_number,
                    "text_span_start": title_start,
                    "text_span_end": title_end,
                    "source_field": "title",
                    "confidence": 1.0,
                },
            ),
            output_modalities=("text",),
            sample_source="crawler_derived",
            verification_status="exact_document_title",
        ),
        text=sample.text,
        privacy_clearance=clearance,
        target_source="curated_document_title",
        builder_source="curated_document_qa",
        content_hash=None,
    )


__all__ = ["build_doc_samples"]
