"""Training sample construction for curated audio."""

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
    select_speech_transcript,
)
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.timed_media import CuratedAudioRecord
from mmcrawler_datasets.curated.training_projection import (
    TrainingAudioInput,
    project_audio_record,
)
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.splitting.group_keys import media_group
from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
    _resolve_artifact_path,
)
from mmcrawler_datasets.training_samples.common import stable_sample_id
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
    from mmcrawler_datasets.materialization.audio_generation import (
        AudioGenerationTargetMaterializer,
    )


def build_audio_samples(
    records: tuple[CuratedAudioRecord, ...],
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
        Callable[[Path], AudioGenerationTargetMaterializer] | None
    ),
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION,
    enabled_tasks: frozenset[str] | None = None,
    require_transcript_for_audio_text_pair: bool = False,
) -> tuple[TrainingSample, ...]:
    """Build audio-text pairs and speech-to-audio generation tasks."""

    samples: list[TrainingSample] = []

    for persisted_record in records:
        record = project_audio_record(persisted_record)
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
                        or "audio_not_trainable"
                    ),
                }
            )
            continue

        pair = select_media_text(record)
        if pair is None:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "no_paired_text",
                }
            )
            continue
        if require_transcript_for_audio_text_pair and pair[1] not in {
            "transcript",
            "transcript_preview",
        }:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "audio_pair_requires_transcript",
                }
            )
            continue
        try:
            sample_clearance = clearance.bind_training_text(pair[0])
        except ValueError:
            rejections.append(
                {
                    "media_id": record.media_id,
                    "reason": "text_not_approved",
                }
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
                raw_path=_audio_target(record),
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
            sample_id=stable_sample_id(
                record.media_id,
                split,
                prefix="aud",
            ),
            snapshot_id=snapshot_id,
            split=split,
            modality="audio",
            task_target=TrainingTaskTarget(
                task_type=require_task("audio_text_pair").name,
                task_family=require_task("audio_text_pair").family,
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
                audio_chromaprint=record.audio_chromaprint,
            ),
        )
        samples.append(sample)

        transcript = pair[0].strip()
        if (
            enabled_tasks is not None
            and "speech_transcription" in enabled_tasks
            and transcript
        ):
            instruction = "Transcribe the spoken content exactly."
            definition = require_task("speech_transcription")
            samples.append(
                replace(
                    sample,
                    sample_id=f"{sample.sample_id}:speech_transcription",
                    task_target=TrainingTaskTarget(
                        task_type=definition.name,
                        task_family=definition.family,
                        instruction=instruction,
                        target_text=transcript,
                        user_text=instruction,
                        assistant_text=transcript,
                        conversation_turns=(
                            ConversationTurn(
                                role="user",
                                text=instruction,
                                turn_index=0,
                            ),
                            ConversationTurn(
                                role="assistant",
                                text=transcript,
                                turn_index=1,
                                answer_evidence_ids=(record.media_id,),
                                is_assistant_answer=True,
                            ),
                        ),
                        answer_evidence_ids=(record.media_id,),
                        output_modalities=("text",),
                        sample_source="crawler_derived",
                        verification_status="exact_approved_transcript",
                    ),
                    text=instruction,
                    paired_text_source="transcript",
                    object_id=record.media_id,
                    target_source="approved_transcript",
                    builder_source="curated_audio_transcription",
                    content_hash=None,
                )
            )

        # Build strict speech_transcription sample using strict transcript selector
        if (
            enabled_tasks is not None
            and "speech_transcription" in enabled_tasks
        ):
            speech_pair = select_speech_transcript(record)
            if speech_pair is not None:
                transcript_text = speech_pair[0].strip()
                if transcript_text:
                    try:
                        speech_clearance = clearance.bind_training_text(
                            transcript_text
                        )
                    except ValueError:
                        pass
                    else:
                        instruction = "Transcribe the spoken content exactly."
                        definition = require_task("speech_transcription")
                        samples.append(
                            replace(
                                sample,
                                sample_id=f"{sample.sample_id}:speech_transcription",
                                task_target=TrainingTaskTarget(
                                    task_type=definition.name,
                                    task_family=definition.family,
                                    instruction=instruction,
                                    target_text=transcript_text,
                                    user_text=instruction,
                                    assistant_text=transcript_text,
                                    conversation_turns=(
                                        ConversationTurn(
                                            role="user",
                                            text=instruction,
                                            turn_index=0,
                                        ),
                                        ConversationTurn(
                                            role="assistant",
                                            text=transcript_text,
                                            turn_index=1,
                                            answer_evidence_ids=(
                                                record.media_id,
                                            ),
                                            is_assistant_answer=True,
                                        ),
                                    ),
                                    answer_evidence_ids=(record.media_id,),
                                    output_modalities=("text",),
                                    sample_source="crawler_derived",
                                    verification_status="exact_approved_transcript",
                                ),
                                text=instruction,
                                paired_text_source="transcript",
                                object_id=record.media_id,
                                target_source="approved_transcript",
                                builder_source="curated_audio_transcription",
                                content_hash=None,
                                privacy_clearance=speech_clearance,
                            )
                        )

        if enabled_tasks is not None and "audio_qa" in enabled_tasks:
            qa_sample = _build_timed_audio_qa(
                sample=sample,
                record=record,
                source_clearance=clearance,
            )
            if qa_sample is not None:
                samples.append(qa_sample)

        if enabled_tasks is not None and "speech_to_audio" in enabled_tasks:
            generation = build_audio_generation(
                sample=sample,
                target_path=target_path,
            )
            if generation is None:
                continue
            if materializer_factory is None:
                raise RuntimeError(
                    "audio_generation_target_materializer_required"
                )
            materializer = materializer_factory(
                materialization_directory / "audio_tokens"
            )
            try:
                generation = materializer.materialize(
                    generation,
                    project_root=project_root,
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed_to_materialize_audio_generation_target:"
                    f"{generation.sample_id}"
                ) from exc
            samples.append(generation)

    return tuple(samples)


def _build_timed_audio_qa(
    *,
    sample: TrainingSample,
    record: TrainingAudioInput,
    source_clearance: PrivacyClearance,
) -> TrainingSample | None:
    """Build exact segment QA only when timing and privacy evidence agree."""

    for index, segment in enumerate(record.transcript_segments):
        if segment.start_seconds is None or segment.end_seconds is None:
            continue
        answer = segment.text.strip()
        if len(answer) < 2:
            continue
        try:
            clearance = source_clearance.bind_training_text(
                answer,
                source_name=f"transcript_segment:{index}",
            )
        except ValueError:
            continue
        evidence_id = f"{record.media_id}:segment:{index}"
        question = (
            "What is said between "
            f"{segment.start_seconds:.2f} and {segment.end_seconds:.2f} seconds?"
        )
        definition = require_task("audio_qa")
        return replace(
            sample,
            sample_id=f"{sample.sample_id}:audio_qa:{index}",
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
                        "relation_type": "audio_transcribes_to_text",
                        "object_id": record.media_id,
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                        "source_field": f"transcript_segment:{index}",
                        "confidence": segment.confidence,
                    },
                ),
                output_modalities=("text",),
                sample_source="crawler_derived",
                verification_status="exact_timed_transcript_segment",
            ),
            text=question,
            privacy_clearance=clearance,
            target_source=f"transcript_segment:{index}",
            builder_source="curated_audio_qa",
            content_hash=None,
        )
    return None


def build_audio_generation(
    *,
    sample: TrainingSample,
    target_path: ValidatedArtifactPath | None,
) -> TrainingSample | None:
    """Create a leakage-safe speech-to-audio generation target."""

    if not target_path:
        return None

    return replace(
        sample,
        sample_id=f"{sample.sample_id}:speech_to_audio",
        modality="text",
        task_target=TrainingTaskTarget(
            task_type=require_task("speech_to_audio").name,
            task_family=require_task("speech_to_audio").family,
            instruction=sample.text,
            target_audio_path=target_path.relative_path,
            output_modalities=("audio",),
            alignment_score=sample.task_target.alignment_score,
        ),
        objects=(),
        content_hash=None,
    )


def _audio_target(record: TrainingAudioInput) -> str | None:
    return (
        record.target_audio_path
        or record.normalized_audio_path
        or record.media_path
    )


__all__ = ["build_audio_samples"]
