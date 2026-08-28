"""Training sample construction for curated text chunks."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.splitting.group_keys import document_group
from mmcrawler_datasets.training_samples.common import (
    estimate_token_count,
    stable_sample_id,
)
from mmcrawler_datasets.training_samples.models import (
    GovernanceEvidence,
    TrainingSample,
    TrainingTextSpan,
)
from mmcrawler_datasets.training_samples.targets import (
    ConversationTurn,
    TrainingTaskTarget,
)
from multimodal.tasks.registry import require_task
from preprocessing.privacy.clearance import PrivacyClearance
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION


def build_text_samples(
    records: tuple[ChunkRecord, ...],
    documents: dict[str, CuratedDocumentRecord],
    splits: dict[str, str],
    *,
    split_assigner: SplitAssigner,
    require_allow_training: bool,
    snapshot_id: str,
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION,
    enabled_tasks: frozenset[str] | None = None,
) -> tuple[TrainingSample, ...]:
    """Build deduplicated self-supervised text samples."""

    samples: list[TrainingSample] = []
    seen_text: set[str] = set()
    for chunk in records:
        document = documents.get(chunk.document_id)
        if (
            document is None
            or document.quality_bucket == "reject"
            or document.privacy_clearance is None
            or not document.privacy_clearance.permits_training
        ):
            continue
        if require_allow_training and document.allow_training is not True:
            continue
        text = chunk.text
        if not text.strip():
            continue
        try:
            clearance = PrivacyClearance.from_dict(
                document.privacy_clearance.bind_training_text(
                    text,
                    source_name="body",
                    start=chunk.start_char,
                    end=chunk.end_char,
                ).to_dict()
            )
        except ValueError:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_text:
            continue
        seen_text.add(text_hash)
        group_key = document_group(document)
        split = splits.get(group_key)
        if split is None:
            split = split_assigner.assign(key=group_key)
            splits[group_key] = split
        sample = TrainingSample(
            schema_version=schema_version,
            sample_id=stable_sample_id(chunk.chunk_id, split),
            snapshot_id=snapshot_id,
            split=split,
            modality="text",
            task_target=TrainingTaskTarget(
                task_type=require_task("text_pretrain").name,
                task_family=require_task("text_pretrain").family,
            ),
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            text=text,
            paired_text_source="chunk_text",
            token_count_estimate=estimate_token_count(text),
            language=chunk.language,
            title=chunk.title,
            domain=document.domain,
            source_url=document.final_url,
            governance=GovernanceEvidence.from_record(document),
            quality_score=document.quality_score,
            exact_duplicate_key=chunk.exact_duplicate_key,
            near_duplicate_cluster_id=chunk.near_duplicate_cluster_id,
            target_source="self_supervised",
            builder_source="crawler_chunk_text",
            text_spans=(TrainingTextSpan(text=text, source="chunk_text"),),
            source_document_id=chunk.document_id,
            normalized_url=document.normalized_url,
            content_family_id=group_key,
            privacy_clearance=clearance,
        )
        samples.append(sample)
        if (
            enabled_tasks is not None
            and "instruction_following" in enabled_tasks
        ):
            instruction_sample = _build_instruction_following_sample(sample)
            if instruction_sample is not None:
                samples.append(instruction_sample)
        if (
            enabled_tasks is not None
            and "causal_text_pretrain" in enabled_tasks
        ):
            causal_sample = _build_causal_text_pretrain_sample(sample)
            if causal_sample is not None:
                samples.append(causal_sample)
    return tuple(samples)


def _build_causal_text_pretrain_sample(
    sample: TrainingSample,
) -> TrainingSample | None:
    """Build a pure autoregressive next-token sample from approved text.

    No prompt/answer split, no encoder prefix. The entire text chunk becomes
    the decoder sequence with standard causal shift (input = tokens[:-1],
    labels = tokens[1:]). This ensures zero target leakage into any encoder.
    """
    text = sample.text.strip()
    if len(text) < 24:
        return None

    definition = require_task("causal_text_pretrain")
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:causal_text_pretrain",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            target_text=text,
            system_text=("Predict the next token in the sequence."),
        ),
        target_source="approved_text_continuation",
        builder_source="crawler_chunk_causal_pretrain",
        content_hash=None,
    )


def _build_instruction_following_sample(
    sample: TrainingSample,
) -> TrainingSample | None:
    """Build a deterministic continuation instruction from approved text."""

    text = sample.text.strip()
    if len(text) < 80:
        return None
    split_index = _instruction_split_index(text=text)
    if split_index is None:
        return None

    prompt_text = text[:split_index].strip()
    answer_text = text[split_index:].strip()
    if len(prompt_text) < 48 or len(answer_text) < 24:
        return None

    definition = require_task("instruction_following")
    instruction = "Continue the supplied text exactly from where it stops."
    return replace(
        sample,
        sample_id=f"{sample.sample_id}:instruction_following",
        task_target=TrainingTaskTarget(
            task_type=definition.name,
            task_family=definition.family,
            instruction=instruction,
            target_text=answer_text,
            system_text=(
                "Follow the instruction using only the supplied source text."
            ),
            user_text=f"{instruction}\n\n{prompt_text}",
            assistant_text=answer_text,
            conversation_turns=(
                ConversationTurn(
                    role="system",
                    text=(
                        "Follow the instruction using only the supplied "
                        "source text."
                    ),
                    turn_index=0,
                ),
                ConversationTurn(
                    role="user",
                    text=f"{instruction}\n\n{prompt_text}",
                    turn_index=1,
                ),
                ConversationTurn(
                    role="assistant",
                    text=answer_text,
                    turn_index=2,
                    is_assistant_answer=True,
                ),
            ),
            output_modalities=("text",),
            sample_source="crawler_derived",
            generator_id="deterministic_text_continuation",
            generator_version="1",
            verification_status="deterministic_exact_slice",
        ),
        text=prompt_text,
        target_source="approved_text_continuation",
        builder_source="crawler_chunk_instruction_following",
        content_hash=None,
    )


def _instruction_split_index(*, text: str) -> int | None:
    """Choose a stable sentence boundary without inventing target text."""

    target = max(48, int(len(text) * 0.7))
    sentence_candidates = [
        index + 1
        for index, char in enumerate(text)
        if char in ".!?"
        and index + 1 < len(text)
        and text[index + 1].isspace()
    ]
    eligible = [index for index in sentence_candidates if index >= target]
    if eligible:
        return eligible[0]
    earlier = [index for index in sentence_candidates if index >= 48]
    if earlier:
        return earlier[-1]
    split_index = text.rfind(" ", 0, target)
    if split_index < 48:
        split_index = text.find(" ", target)
    if split_index <= 0 or split_index >= len(text) - 24:
        return None
    return split_index


__all__ = ["build_text_samples"]
