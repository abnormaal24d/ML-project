"""Training sample construction for curated images."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mmcrawler_datasets.assembly.text_pairing import (
    image_quality,
    pairability_score,
    select_image_caption,
    select_image_text,
)
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.splitting.group_keys import image_group
from mmcrawler_datasets.training_samples.artifact_path import (
    _resolve_artifact_path,
)
from mmcrawler_datasets.training_samples.common import stable_sample_id
from mmcrawler_datasets.training_samples.fingerprints import (
    ContentFingerprintInputs,
)
from mmcrawler_datasets.training_samples.models import (
    GovernanceEvidence,
    TrainingSample,
    _build_object,
)
from mmcrawler_datasets.training_samples.targets import (
    ConversationTurn,
    TrainingTaskTarget,
)
from multimodal.tasks.registry import require_task
from preprocessing.privacy.clearance import (
    ApprovedObjectRole,
    PrivacyClearance,
)
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION


def build_image_samples(
    records: tuple[CuratedImageRecord, ...],
    documents: dict[str, CuratedDocumentRecord],
    splits: dict[str, str],
    *,
    split_assigner: SplitAssigner,
    require_allow_training: bool,
    snapshot_id: str,
    snapshot_directory: Path,
    project_root: Path,
    rejections: list[dict[str, object]],
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION,
    enabled_tasks: frozenset[str] | None = None,
) -> tuple[TrainingSample, ...]:
    """Build image-text pairs and supported image-to-text tasks."""

    samples: list[TrainingSample] = []
    for image in records:
        clearance_record = image.privacy_clearance
        if clearance_record is None or not clearance_record.permits_training:
            rejections.append(
                {
                    "image_id": image.image_id,
                    "reason": "text_privacy_incomplete",
                }
            )
            continue
        clearance = PrivacyClearance.from_dict(clearance_record.to_dict())
        if not image.trainable:
            rejections.append(
                {
                    "image_id": image.image_id,
                    "reason": image.curated_rejection_reason
                    or "image_not_trainable",
                }
            )
            continue
        if require_allow_training and image.allow_training is not True:
            continue
        pair = select_image_text(image)
        if pair is None:
            rejections.append(
                {"image_id": image.image_id, "reason": "no_paired_text"}
            )
            continue
        try:
            sample_clearance = clearance.bind_training_text(pair[0])
        except ValueError:
            rejections.append(
                {"image_id": image.image_id, "reason": "text_not_approved"}
            )
            continue
        document = (
            documents.get(image.parent_document_id)
            if image.parent_document_id
            else None
        )
        group_key = image_group(image=image, document=document)
        split = splits.get(group_key) or split_assigner.assign(key=group_key)
        splits.setdefault(group_key, split)
        try:
            image_path = _resolve_artifact_path(
                raw_path=image.media_path,
                project_root=project_root,
                curated_snapshot_directory=snapshot_directory,
            )
        except ValueError:
            rejections.append(
                {
                    "image_id": image.image_id,
                    "reason": "artifact_path_invalid",
                }
            )
            continue
        if image_path is None:
            rejections.append(
                {
                    "image_id": image.image_id,
                    "reason": "artifact_unavailable",
                }
            )
            continue
        image_object = _build_object(
            object_id=image.image_id,
            object_path=image_path,
            object_mime_type=image.image_mime_type,
            role=ApprovedObjectRole.PRIMARY_MEDIA,
        )
        pair_score = pairability_score(
            media_record=image,
            parent_text=document.title if document is not None else None,
        )
        sample = TrainingSample(
            schema_version=schema_version,
            sample_id=stable_sample_id(image.image_id, split, prefix="img"),
            snapshot_id=snapshot_id,
            split=split,
            modality="image",
            task_target=TrainingTaskTarget(
                task_type=require_task("image_text_pair").name,
                task_family=require_task("image_text_pair").family,
                target_text=pair[0],
                positive_id=image.image_id,
            ),
            document_id=image.parent_document_id,
            object_id=image.image_id,
            text=pair[0],
            paired_text_source=pair[1],
            pairability_score=pair_score,
            pair_source=pair[1],
            language=image.ocr_language
            or (document.language if document else None),
            title=image.page_title,
            domain=document.domain if document else "",
            source_url=image.source_url,
            governance=GovernanceEvidence.from_record(image),
            quality_score=image_quality(image),
            context_score=image.context_score,
            exact_duplicate_key=None,
            objects=(image_object,) if image_object is not None else (),
            source_document_id=image.parent_document_id,
            parent_document_id=image.parent_document_id,
            privacy_clearance=sample_clearance,
            fingerprint_inputs=ContentFingerprintInputs(
                image_ahash=image.image_average_hash,
                image_dhash=image.image_difference_hash,
                image_phash=image.image_phash,
            ),
        )
        samples.append(sample)
        if enabled_tasks is not None and "image_captioning" in enabled_tasks:
            caption_sample = _build_image_captioning_sample(
                sample=sample,
                image=image,
                source_clearance=clearance,
            )
            if caption_sample is not None:
                samples.append(caption_sample)
        if enabled_tasks is not None and "vqa" in enabled_tasks:
            vqa_sample = _build_image_vqa_sample(
                sample=sample,
                image=image,
                source_clearance=clearance,
            )
            if vqa_sample is not None:
                samples.append(vqa_sample)
        if enabled_tasks is not None and "ocr_parse" in enabled_tasks:
            ocr_sample = _build_ocr_sample(
                sample=sample,
                image=image,
                source_clearance=clearance,
            )
            if ocr_sample is not None:
                samples.append(ocr_sample)
    return tuple(samples)


def _build_image_captioning_sample(
    *,
    sample: TrainingSample,
    image: CuratedImageRecord,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build one causal image captioning sample from a strict caption source.

    The prompt stays in the decoder sequence; the image is the sole
    encoder context. Caption never reaches encoder context.
    """

    caption_pair = select_image_caption(image)
    if caption_pair is None:
        return None

    caption, caption_source = caption_pair
    try:
        clearance = source_clearance.bind_training_text(
            caption,
            source_name=caption_source,
        )
    except ValueError:
        return None

    definition = require_task("image_captioning")
    instruction = "Describe the image."
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:image_captioning",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            instruction=instruction,
            target_text=caption,
            user_text=instruction,
            assistant_text=caption,
            conversation_turns=(
                ConversationTurn(role="user", text=instruction, turn_index=0),
                ConversationTurn(
                    role="assistant",
                    text=caption,
                    turn_index=1,
                    is_assistant_answer=True,
                ),
            ),
            output_modalities=("text",),
            sample_source="crawler_derived",
        ),
        text=instruction,
        privacy_clearance=clearance,
        target_source=caption_source,
        builder_source="curated_image_captioning",
        content_hash=None,
    )


def _build_image_vqa_sample(
    *,
    sample: TrainingSample,
    image: CuratedImageRecord,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build one region-grounded VQA row from verified OCR evidence.

    Captions remain useful for image-text alignment, but they are deliberately
    not promoted to VQA: the answer must depend on pixels and carry a concrete
    region or line annotation.
    """

    answer = str(image.ocr_text or "").strip()
    evidence_boxes = tuple(
        dict(item) for item in (image.ocr_boxes or image.ocr_lines)
    )
    if not answer or not evidence_boxes:
        return None
    try:
        clearance = source_clearance.bind_training_text(
            answer,
            source_name="ocr_text",
        )
    except ValueError:
        return None

    question = "What text is visible in the highlighted image region?"
    evidence_records = tuple(
        _image_evidence_record(
            image_id=image.image_id,
            box=box,
            index=index,
            relation_type="ocr_originates_from_image",
        )
        for index, box in enumerate(evidence_boxes)
    )
    definition = require_task("vqa")
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:vqa",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            question=question,
            target_text=answer,
            answer=answer,
            object_boxes=evidence_boxes,
            user_text=question,
            assistant_text=answer,
            conversation_turns=(
                ConversationTurn(role="user", text=question, turn_index=0),
                ConversationTurn(
                    role="assistant",
                    text=answer,
                    turn_index=1,
                    answer_evidence_ids=tuple(
                        str(record["evidence_id"])
                        for record in evidence_records
                    ),
                    is_assistant_answer=True,
                ),
            ),
            answer_evidence_ids=tuple(
                str(record["evidence_id"]) for record in evidence_records
            ),
            evidence_records=evidence_records,
            output_modalities=("text",),
            sample_source="crawler_derived",
            verification_status="exact_ocr_text_with_region_evidence",
        ),
        text=question,
        privacy_clearance=clearance,
        target_source="ocr_text",
        builder_source="curated_image_vqa",
        content_hash=None,
    )


def _build_ocr_sample(
    *,
    sample: TrainingSample,
    image: CuratedImageRecord,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build OCR text/layout targets with reading order and confidence evidence."""

    ocr_text = str(image.ocr_text or "").strip()
    layout_boxes = tuple(
        dict(item) for item in (image.ocr_boxes or image.ocr_lines)
    )
    if not ocr_text or not layout_boxes:
        return None
    try:
        clearance = source_clearance.bind_training_text(
            ocr_text,
            source_name="ocr_text",
        )
    except ValueError:
        return None

    evidence_records = tuple(
        _image_evidence_record(
            image_id=image.image_id,
            box=box,
            index=index,
            relation_type="ocr_originates_from_image",
        )
        for index, box in enumerate(layout_boxes)
    )
    reading_order = tuple(
        str(record["evidence_id"]) for record in evidence_records
    )
    fallback_confidence = float(image.ocr_confidence or 0.0)
    confidences = tuple(
        _box_confidence(box, fallback=fallback_confidence)
        for box in layout_boxes
    )

    instruction = "Extract the visible text and preserve its reading layout."
    definition = require_task("ocr_parse")
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:ocr_parse",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            instruction=instruction,
            target_text=ocr_text,
            layout_boxes=layout_boxes,
            reading_order=reading_order,
            ocr_confidences=confidences,
            evidence_records=evidence_records,
            user_text=instruction,
            assistant_text=ocr_text,
            conversation_turns=(
                ConversationTurn(
                    role="user",
                    text=instruction,
                    turn_index=0,
                ),
                ConversationTurn(
                    role="assistant",
                    text=ocr_text,
                    turn_index=1,
                    answer_evidence_ids=reading_order,
                    is_assistant_answer=True,
                ),
            ),
            answer_evidence_ids=reading_order,
            output_modalities=("text", "json"),
            sample_source="crawler_derived",
            verification_status="exact_ocr_text_layout_order_and_confidence",
        ),
        text=instruction,
        privacy_clearance=clearance,
        target_source="ocr_text",
        builder_source="curated_image_ocr",
        content_hash=None,
    )


def _image_evidence_record(
    *,
    image_id: str,
    box: dict[str, object],
    index: int,
    relation_type: str,
) -> dict[str, object]:
    evidence_id = str(
        box.get("id") or box.get("line_id") or f"{image_id}:region:{index}"
    )
    return {
        "evidence_id": evidence_id,
        "object_id": image_id,
        "bounding_box": dict(box),
        "relation_type": relation_type,
        "confidence": _box_confidence(box, fallback=1.0),
        "evidence_source": "curated_ocr",
    }


def _box_confidence(box: dict[str, object], *, fallback: float) -> float:
    for key in ("confidence", "score", "ocr_confidence"):
        raw = box.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(0.0, min(1.0, float(raw)))
    return max(0.0, min(1.0, float(fallback)))


__all__ = ["build_image_samples"]
