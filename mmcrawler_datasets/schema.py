"""Canonical datatypes for multimodal training samples and annotations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from config.environment.default_values import (
    DEFAULT_TEST_SPLIT_NAME,
    DEFAULT_TRAIN_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
)
from mmcrawler_datasets.training_samples.targets import ConversationTurn

# Type definition for supported modalities.
ModalityType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "document",
    "layout",
    "mask",
    "code",
    "json",
    "table",
    "screen",
    "multimodal",
    "unknown",
]

# Canonical target field names that MultimodalSample can carry. Task
# definitions may only require targets from this set (release evidence).
SUPPORTED_TARGET_FIELDS: frozenset[str] = frozenset(
    {
        "answer",
        "answer_evidence_ids",
        "assistant_text",
        "background_noise_label",
        "chart_data",
        "code_language",
        "emotion_label",
        "form_fields",
        "humor_explanation",
        "instruction",
        "label",
        "layout_boxes",
        "math_expression",
        "math_solution",
        "negative_ids",
        "object_boxes",
        "positive_id",
        "prosody",
        "question",
        "sample_source",
        "scene_graph",
        "speaker_label",
        "speaker_segments",
        "target_audio_path",
        "target_audio_tokens_path",
        "target_code",
        "target_image_path",
        "target_table_structure",
        "target_text",
        "target_video_path",
        "target_video_tokens_path",
        "system_text",
        "tool_arguments_json",
        "tool_name",
        "tool_result_json",
        "user_text",
        "conversation_turns",
        "generator_id",
        "generator_version",
        "verification_status",
        "evidence_records",
        "reading_order",
        "ocr_confidences",
        "negative_verification",
        "chosen_response",
        "rejected_response",
        "preference_reason",
        "preference_source",
        "must_refuse",
        "requires_uncertainty",
        "requires_source_citation",
        "prompt_injection_present",
        "untrusted_document_instruction",
        "sensitive_data_present",
        "requires_tool_confirmation",
    }
)


@dataclass(frozen=True, slots=True)
class ModalityObject:
    """Reference to one modality asset with optional metadata."""

    path: Path | None = None
    url: str | None = None
    mime_type: str | None = None
    byte_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box with optional page and coordinate metadata."""

    x: float
    y: float
    width: float
    height: float
    page: int | None = None
    coordinate_system: str = "relative"


@dataclass(frozen=True, slots=True)
class LayoutBox:
    """Document layout element with text, geometry, and reading order."""

    text: str | None = None
    box: BoundingBox | None = None
    role: str | None = None
    reading_order: int | None = None
    line_id: str | None = None
    column_id: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class UIElement:
    """Screen UI element with optional label, box, and children."""

    element_type: str
    box: BoundingBox | None = None
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: tuple[UIElement, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryAnnotation:
    """Spatial relation between annotated geometry entities."""

    subject_id: str
    relation: str
    object_id: str | None = None
    value: float | str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ObjectBox:
    """Detected object with label, box, and optional attributes."""

    object_id: str
    label: str
    box: BoundingBox | None = None
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """Named chart series with numeric values and labels."""

    name: str
    values: tuple[float, ...]
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChartData:
    """Structured chart metadata with axes, legend, and series."""

    chart_type: str | None = None
    title: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    legend: tuple[str, ...] = ()
    series: tuple[ChartSeries, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """Diarized speech segment with timing and transcript metadata."""

    start_seconds: float
    end_seconds: float
    speaker_id: str
    confidence: float | None = None
    transcript: str | None = None
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class ProsodyFeatures:
    """Prosody measurements and emphasis markers for speech."""

    pitch_hz: float | None = None
    energy: float | None = None
    tempo: float | None = None
    pause_ratio: float | None = None
    emphasis: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MultimodalSample:
    """Canonical parsed sample used by dataset and training code."""

    sample_id: str
    record_id: str = ""
    task_type: str = "representation"
    task_family: str | None = None
    text: str | None = None
    title: str | None = None
    instruction: str | None = None
    question: str | None = None
    answer: str | None = None
    source_url: str | None = None
    label: int | None = None
    target_text: str | None = None
    target_code: str | None = None
    code_language: str | None = None
    target_audio_path: Path | None = None
    target_audio_tokens_path: Path | None = None
    target_image_path: Path | None = None
    target_video_path: Path | None = None
    source_image_path: Path | None = None
    edit_mask_path: Path | None = None
    text_tokens_path: Path | None = None
    image_tensor_path: Path | None = None
    audio_tensor_path: Path | None = None
    video_tensor_path: Path | None = None
    target_image_tensor_path: Path | None = None
    target_video_tensor_path: Path | None = None
    source_image_tensor_path: Path | None = None
    edit_mask_tensor_path: Path | None = None
    target_video_tokens_path: Path | None = None
    output_modalities: tuple[str, ...] = ()
    positive_id: str | None = None
    negative_ids: tuple[str, ...] = ()
    alignment_score: float = 1.0
    dataset_version: str | None = None
    content_hash: str | None = None
    processing_version: str | None = None
    language: str | None = None
    language_confidence: float | None = None
    language_script: str | None = None
    safety_status: str | None = None
    license: str | None = None
    license_url: str | None = None
    robots_status: str | None = None
    terms_source: str | None = None
    usage_rules: str | None = None
    task_target: dict[str, Any] = field(default_factory=dict)
    document: ModalityObject | None = None
    image: ModalityObject | None = None
    audio: ModalityObject | None = None
    video: ModalityObject | None = None
    layout_boxes: tuple[LayoutBox, ...] = ()
    ui_elements: tuple[UIElement, ...] = ()
    geometry_annotations: tuple[GeometryAnnotation, ...] = ()
    object_boxes: tuple[ObjectBox, ...] = ()
    chart_data: ChartData | None = None
    math_expression: str | None = None
    math_solution: str | None = None
    humor_explanation: str | None = None
    target_table_structure: str | dict[str, Any] | None = None
    form_fields: dict[str, Any] = field(default_factory=dict)
    scene_graph: dict[str, Any] = field(default_factory=dict)
    speaker_segments: tuple[SpeakerSegment, ...] = ()
    prosody: ProsodyFeatures | None = None
    emotion_label: str | None = None
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    sarcasm_label: str | None = None
    speaker_label: str | None = None
    background_noise_label: str | None = None
    overlapping_speech: bool | None = None
    system_text: str | None = None
    user_text: str | None = None
    assistant_text: str | None = None
    conversation_turns: tuple[ConversationTurn, ...] = ()
    answer_evidence_ids: tuple[str, ...] = ()
    tool_name: str | None = None
    tool_arguments_json: str | None = None
    tool_result_json: str | None = None
    sample_source: str | None = None
    generator_id: str | None = None
    generator_version: str | None = None
    verification_status: str | None = None
    evidence_records: tuple[dict[str, Any], ...] = ()
    reading_order: tuple[str, ...] = ()
    ocr_confidences: tuple[float, ...] = ()
    negative_verification: tuple[dict[str, Any], ...] = ()
    chosen_response: str | None = None
    rejected_response: str | None = None
    preference_reason: str | None = None
    preference_source: str | None = None
    must_refuse: bool | None = None
    requires_uncertainty: bool | None = None
    requires_source_citation: bool | None = None
    prompt_injection_present: bool | None = None
    untrusted_document_instruction: bool | None = None
    sensitive_data_present: bool | None = None
    requires_tool_confirmation: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modality(self) -> ModalityType:
        """Determines the primary modality of the sample."""
        presence = {
            "document": self.has_document,
            "image": self.has_image,
            "audio": self.has_audio,
            "video": self.has_video,
        }

        # Tel hoeveel media types aanwezig zijn
        active_media = [k for k, v in presence.items() if v]

        if len(active_media) > 1:
            return "multimodal"
        if len(active_media) == 1:
            return active_media[0]  # type: ignore
        if self.has_text:
            return "text"

        return "unknown"

    @property
    def has_text(self) -> bool:
        """Return whether the sample has non-empty input text."""

        return bool(self.input_text.strip())

    @property
    def input_text(self) -> str:
        """Return joined instruction, question, text, and title fields."""

        parts = []
        for value in (self.instruction, self.question, self.text, self.title):
            if value is not None and value.strip():
                parts.append(value.strip())
        return "\n".join(parts)

    @property
    def generative_target_text(self) -> str | None:
        """Return the first non-empty generative target string."""

        for value in (
            self.target_code,
            self.target_text,
            self.answer,
            self.assistant_text,
            self.math_solution,
            self.humor_explanation,
        ):
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @property
    def answer_text(self) -> str | None:
        """Return the primary assistant answer text when present."""

        if self.assistant_text is not None and self.assistant_text.strip():
            return self.assistant_text.strip()
        last_answer = next(
            (
                turn.text.strip()
                for turn in reversed(self.conversation_turns)
                if turn.role == "assistant"
            ),
            "",
        )
        if last_answer:
            return last_answer
        for value in (
            self.answer,
            self.target_text,
            self.target_code,
            self.math_solution,
            self.humor_explanation,
        ):
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @property
    def conversation_prompt_text(self) -> str:
        """Return exactly the conversation context preceding the target."""

        turns = self.conversation_turns
        if turns:
            marked = [
                index
                for index, turn in enumerate(turns)
                if turn.role == "assistant" and turn.is_assistant_answer
            ]
            target_index = (
                marked[0]
                if len(marked) == 1
                else next(
                    (
                        index
                        for index in range(len(turns) - 1, -1, -1)
                        if turns[index].role == "assistant"
                    ),
                    len(turns),
                )
            )
            parts: list[str] = []
            for turn in turns[:target_index]:
                content = turn.text.strip()
                if not content:
                    content = (
                        turn.tool_arguments_json
                        if turn.role == "assistant"
                        else turn.tool_result_json
                    ) or ""
                if content:
                    parts.append(f"<{turn.role}>\n{content}")
            return "\n".join(parts)

        parts = []
        if self.system_text is not None and self.system_text.strip():
            parts.append(f"<system>\n{self.system_text.strip()}")
        if self.user_text is not None and self.user_text.strip():
            parts.append(f"<user>\n{self.user_text.strip()}")
        return "\n".join(parts)

    @property
    def has_conversation(self) -> bool:
        """Return whether the sample carries conversation structure."""

        return bool(
            self.conversation_turns
            or self.system_text
            or self.assistant_text
            or self.user_text
        )

    @property
    def has_image(self) -> bool:
        """Return whether the sample references image inputs."""

        return (
            (
                self.image is not None
                and (self.image.path is not None or self.image.url is not None)
            )
            or self.source_image_path is not None
            or self.image_tensor_path is not None
        )

    @property
    def has_audio(self) -> bool:
        """Return whether the sample references audio inputs."""

        return (
            self.audio is not None
            and (self.audio.path is not None or self.audio.url is not None)
        ) or self.audio_tensor_path is not None

    @property
    def has_video(self) -> bool:
        """Return whether the sample references video inputs."""

        return (
            self.video is not None
            and (self.video.path is not None or self.video.url is not None)
        ) or self.video_tensor_path is not None

    @property
    def has_document(self) -> bool:
        """Return whether the sample includes document content."""

        if self.document is not None and (
            self.document.path is not None or self.document.url is not None
        ):
            return True
        if self.task_type in {
            "document_text_pair",
            "pdf_text_pair",
            "doc_qa",
            "document_summarization",
            "document_comparison",
            "table_extraction",
            "table_qa",
            "spreadsheet_analysis",
            "passage_retrieval",
            "multifile_reasoning",
        }:
            return bool((self.text or "").strip() or self.layout_boxes)
        return False

    @property
    def has_layout(self) -> bool:
        """Return whether the sample includes layout annotations."""

        return bool(self.layout_boxes or self.ui_elements)

    @property
    def has_mask(self) -> bool:
        """Return whether the sample includes an edit mask."""

        return (
            self.edit_mask_path is not None
            or self.edit_mask_tensor_path is not None
        )

    @property
    def has_code(self) -> bool:
        """Return whether the sample includes code-generation targets."""

        return bool(
            (self.target_code or "").strip()
            or (self.code_language or "").strip()
        )

    @property
    def has_json(self) -> bool:
        """Return whether the sample includes structured JSON targets."""

        return bool(
            self.form_fields
            or self.scene_graph
            or self.chart_data is not None
            or self.target_table_structure
        )


class DatasetSplit(StrEnum):
    """Supported dataset split names for multimodal training."""

    TRAIN = DEFAULT_TRAIN_SPLIT_NAME
    VAL = DEFAULT_VAL_SPLIT_NAME
    TEST = DEFAULT_TEST_SPLIT_NAME


class SplitAssigner:
    """Assign deterministic split labels from stable cluster/document keys."""

    def __init__(
        self,
        *,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> None:
        ratios = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
        for name, ratio in ratios.items():
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or ratio < 0.0
            ):
                raise ValueError(
                    f"{name} must be a finite non-negative number"
                )

        self._train_ratio = float(train_ratio)
        self._val_ratio = float(val_ratio)
        self._test_ratio = float(test_ratio)
        total = self._train_ratio + self._val_ratio + self._test_ratio
        if total <= 0:
            self._train_ratio = 1.0
            self._val_ratio = 0.0
            self._test_ratio = 0.0
            total = 1.0
        self._train_ratio /= total
        self._val_ratio /= total
        self._test_ratio /= total

    def assign(self, key: str) -> str:
        """Return stable split label for the provided grouping key."""

        value = _hash_fraction(key=key)
        if value < self._train_ratio:
            return DEFAULT_TRAIN_SPLIT_NAME
        if value < self._train_ratio + self._val_ratio:
            return DEFAULT_VAL_SPLIT_NAME
        return DEFAULT_TEST_SPLIT_NAME

    def assign_many(self, *, keys: Iterable[str]) -> dict[str, str]:
        """
        Assign splits for many group keys with minimum small-set coverage.
        """

        unique_keys = sorted(set(keys), key=_stable_sort_key)
        if not unique_keys:
            return {}

        total = len(unique_keys)
        val_count = _target_partition_count(
            total=total,
            ratio=self._val_ratio,
            minimum=(1 if self._val_ratio > 0.0 and total >= 3 else 0),
        )
        test_count = _target_partition_count(
            total=total,
            ratio=self._test_ratio,
            minimum=(1 if self._test_ratio > 0.0 and total >= 3 else 0),
        )

        while val_count + test_count > max(0, total - 1):
            if test_count >= val_count and test_count > 0:
                test_count -= 1
                continue
            if val_count > 0:
                val_count -= 1
                continue
            break

        assignments: dict[str, str] = {}
        val_keys = unique_keys[:val_count]
        test_keys = unique_keys[val_count : val_count + test_count]
        train_keys = unique_keys[val_count + test_count :]

        for key in train_keys:
            assignments[key] = DEFAULT_TRAIN_SPLIT_NAME
        for key in val_keys:
            assignments[key] = DEFAULT_VAL_SPLIT_NAME
        for key in test_keys:
            assignments[key] = DEFAULT_TEST_SPLIT_NAME
        return assignments


def _hash_fraction(*, key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    integer_value = int(digest, 16)
    max_value = float(0xFFFFFFFFFFFFFFFF)
    return integer_value / max_value


def _stable_sort_key(key: str) -> tuple[str, str]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest, key


def _target_partition_count(
    *,
    total: int,
    ratio: float,
    minimum: int,
) -> int:
    if total <= 0 or ratio <= 0.0:
        return 0
    return max(minimum, int(total * ratio))
