"""Training sample construction for curated video."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from mmcrawler_datasets.assembly.sample_components import (
    _build_media_objects,
    _build_media_text_spans,
)
from mmcrawler_datasets.assembly.text_pairing import (
    pairability_score,
    select_media_text,
    select_video_caption,
)
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.timed_media import CuratedVideoRecord
from mmcrawler_datasets.curated.training_projection import (
    TrainingVideoInput,
    project_video_record,
)
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.splitting.group_keys import media_group
from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
    _resolve_artifact_path,
)
from mmcrawler_datasets.training_samples.common import (
    as_opt_float,
    stable_sample_id,
)
from mmcrawler_datasets.training_samples.fingerprints import (
    ContentFingerprintInputs,
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

if TYPE_CHECKING:
    from mmcrawler_datasets.materialization.video_generation import (
        VideoGenerationTargetMaterializer,
    )


def build_video_samples(
    records: tuple[CuratedVideoRecord, ...],
    documents: dict[str, CuratedDocumentRecord],
    splits: dict[str, str],
    *,
    split_assigner: SplitAssigner,
    require_allow_training: bool,
    snapshot_id: str,
    snapshot_directory: Path,
    materialization_directory: Path,
    project_root: Path,
    rejections: list[dict[str, object]],
    materializer_factory: (
        Callable[[Path], VideoGenerationTargetMaterializer] | None
    ),
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION,
    enabled_tasks: frozenset[str] | None = None,
) -> tuple[TrainingSample, ...]:
    """Build video-text pairs and materialized text-to-video tasks."""

    samples: list[TrainingSample] = []
    for persisted_record in records:
        record = project_video_record(persisted_record)
        clearance = record.privacy_clearance
        if clearance is None or not clearance.permits_training:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "text_privacy_incomplete",
                }
            )
            continue
        if require_allow_training and record.allow_training is not True:
            continue
        if not record.trainable:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": (
                        record.curated_rejection_reason
                        or "video_not_trainable"
                    ),
                }
            )
            continue
        pair = select_media_text(record)
        if pair is None:
            rejections.append(
                {"media_id": record.media_id, "reason": "no_paired_text"}
            )
            continue
        try:
            sample_clearance = clearance.bind_training_text(pair[0])
        except ValueError:
            rejections.append(
                {"media_id": record.media_id, "reason": "text_not_approved"}
            )
            continue
        document = (
            documents.get(record.parent_document_id)
            if record.parent_document_id
            else None
        )
        group_key = media_group(record=record, document=document)
        split = splits.get(group_key) or split_assigner.assign(key=group_key)
        splits[group_key] = split
        row = record.to_dict()
        try:
            objects = _build_media_objects(
                row=row,
                object_path=record.media_path,
                media_id=record.media_id,
                project_root=project_root,
                curated_snapshot_directory=snapshot_directory,
            )
            target_path = _resolve_artifact_path(
                raw_path=_video_target(record),
                project_root=project_root,
                curated_snapshot_directory=snapshot_directory,
            )
        except ValueError:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "artifact_path_invalid",
                }
            )
            continue
        if not objects:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "artifact_unavailable",
                }
            )
            continue
        pair_score = pairability_score(
            media_record=record,
            parent_text=document.title if document is not None else None,
        )
        sample = TrainingSample(
            schema_version=schema_version,
            sample_id=stable_sample_id(record.media_id, split, prefix="vid"),
            snapshot_id=snapshot_id,
            split=split,
            modality="video",
            task_target=TrainingTaskTarget(
                task_type=require_task("video_text_pair").name,
                task_family=require_task("video_text_pair").family,
                target_text=pair[0],
                positive_id=record.media_id,
            ),
            document_id=record.parent_document_id,
            object_id=record.media_id,
            text=pair[0],
            paired_text_source=pair[1],
            pairability_score=pair_score,
            pair_source=pair[1],
            objects=objects,
            text_spans=_build_media_text_spans(
                row=row,
                fallback_text=pair[0],
                fallback_source=pair[1],
            ),
            source_document_id=record.parent_document_id,
            parent_document_id=record.parent_document_id,
            content_family_id=group_key,
            alignment_group_id=group_key,
            language=record.language,
            quality_score=record.quality_score,
            context_score=record.context_score,
            is_trainable=record.trainable,
            non_trainable_reason=record.curated_rejection_reason,
            asset_context=record.asset_context,
            domain=record.domain,
            source_url=record.source_url,
            governance=GovernanceEvidence.from_record(record),
            fetch_mode=record.asset_fetch_mode or "full",
            is_complete_payload=record.is_complete_payload is not False,
            near_duplicate_cluster_id=record.near_duplicate_cluster_id,
            privacy_clearance=sample_clearance,
            fingerprint_inputs=ContentFingerprintInputs(
                video_keyframe_phashes=record.video_keyframe_phashes,
            ),
        )
        samples.append(sample)
        if enabled_tasks is not None and "video_qa" in enabled_tasks:
            qa_sample = _build_timed_video_qa(
                sample=sample,
                record=record,
                source_clearance=clearance,
            )
            if qa_sample is not None:
                samples.append(qa_sample)
        if enabled_tasks is not None and "video_captioning" in enabled_tasks:
            caption_sample = _build_video_captioning_sample(
                sample=sample,
                record=record,
                source_clearance=clearance,
            )
            if caption_sample is not None:
                samples.append(caption_sample)

        if enabled_tasks is not None and "text_to_video" in enabled_tasks:
            generation = build_video_generation(
                sample=sample,
                target_path=target_path,
            )
            if generation is None:
                continue
            if materializer_factory is None:
                raise RuntimeError(
                    "video_generation_target_materializer_required"
                )
            materializer = materializer_factory(
                materialization_directory / "video_tokens"
            )
            try:
                generation = materializer.materialize(
                    generation,
                    project_root=project_root,
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed_to_materialize_video_generation_target:"
                    f"{generation.sample_id}"
                ) from exc
            samples.append(generation)
    return tuple(samples)


def _build_video_captioning_sample(
    *,
    sample: TrainingSample,
    record: TrainingVideoInput,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build one causal video captioning sample from a strict caption source.

    The prompt stays in the decoder sequence; the video is the sole
    encoder context. Caption never reaches encoder context.
    """

    caption_pair = select_video_caption(record)
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

    definition = require_task("video_captioning")
    instruction = "Describe what happens in this video."
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:video_captioning",
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
        builder_source="curated_video_captioning",
        content_hash=None,
    )


def _build_timed_video_qa(
    *,
    sample: TrainingSample,
    record: TrainingVideoInput,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build visual video QA only from timed frame/OCR evidence."""

    answer = str(record.frame_ocr_text or "").strip()
    timed_keyframes = tuple(
        (frame, timestamp)
        for frame in record.keyframes
        if (timestamp := as_opt_float(frame.get("timestamp_seconds")))
        is not None
    )
    if not answer or not timed_keyframes:
        return None
    frame, timestamp = timed_keyframes[0]
    frame_index = frame.get("frame_index")
    try:
        clearance = source_clearance.bind_training_text(
            answer, source_name="frame_ocr_text"
        )
    except ValueError:
        return None
    evidence_id = f"{record.media_id}:frame:{frame_index or 0}"
    question = (
        f"What text is visible near {timestamp:.2f} seconds in this video?"
    )
    definition = require_task("video_qa")
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:video_qa:frame",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            question=question,
            target_text=answer,
            answer=answer,
            user_text=question,
            assistant_text=answer,
            conversation_turns=(
                ConversationTurn(role="user", text=question, turn_index=0),
                ConversationTurn(
                    role="assistant",
                    text=answer,
                    turn_index=1,
                    answer_evidence_ids=(evidence_id,),
                    is_assistant_answer=True,
                ),
            ),
            answer_evidence_ids=(evidence_id,),
            evidence_records=(
                {
                    "evidence_id": evidence_id,
                    "relation_type": "object_visible_during_utterance",
                    "object_id": record.media_id,
                    "timestamp_seconds": timestamp,
                    "frame_index": frame_index,
                    "frame_path": frame.get("frame_path"),
                    "source_field": "frame_ocr_text",
                    "confidence": frame.get("confidence"),
                },
            ),
            output_modalities=("text",),
            sample_source="crawler_derived",
            verification_status="exact_timed_frame_ocr",
        ),
        text=question,
        privacy_clearance=clearance,
        target_source="frame_ocr_text",
        builder_source="curated_video_visual_qa",
        content_hash=None,
    )


def build_video_generation(
    *,
    sample: TrainingSample,
    target_path: ValidatedArtifactPath | None,
) -> TrainingSample | None:
    """Create a leakage-safe text-to-video generation target."""

    if not target_path:
        return None
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:text_to_video",
        modality="text",
        task_target=TrainingTaskTarget(
            task_type=require_task("text_to_video").name,
            task_family=require_task("text_to_video").family,
            instruction=sample.text,
            target_video_path=target_path.relative_path,
            output_modalities=("video",),
            alignment_score=sample.task_target.alignment_score,
        ),
        objects=(),
        content_hash=None,
    )


def _video_target(record: TrainingVideoInput) -> str | None:
    return (
        record.target_video_path
        or record.normalized_video_path
        or record.media_path
    )


__all__ = ["build_video_samples"]
