"""Canonical multimodal task definition schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from schemas.multimodal_tasks import (
    TASK_NAME_PATTERN,
    OutputModality,
    TaskModality,
)

TaskMaturity = Literal[
    "stable",
    "beta",
    "experimental",
    "disabled",
]

TaskSensitivity = Literal[
    "standard",
    "sensitive",
    "biometric_identity",
]

TaskSampleSource = Literal[
    "self_supervised",
    "crawler_derived",
    "external",
]

TaskApproval = Literal[
    "beta",
    "sensitive",
]

MODEL_OUTPUT_MODALITIES: frozenset[str] = frozenset(
    {"class", "code", "embedding", "json", "text"}
)

TaskFamily = Literal[
    "text",
    "document",
    "image",
    "audio",
    "video",
    "retrieval",
    "cross_modal",
    "screen_ui",
    "table_spreadsheet",
    "multimodal_reasoning",
]

TaskObjective = Literal[
    "language_modeling",
    "text_mlm",
    "classification",
    "contrastive",
    "ocr_sequence",
    "chart_reasoning",
    "math_reasoning",
    "visual_grounding",
    "emotion",
    "speaker_contrastive",
    "diarization",
    "speech_translation",
    "audio_generation",
    "image_generation",
    "image_editing",
    "video_generation",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDefinition:
    """Canonical schema for one multimodal task."""

    name: str
    family: TaskFamily
    required_input_modalities: tuple[TaskModality, ...]
    output_modalities: tuple[OutputModality, ...]
    required_target_fields: tuple[str, ...]

    evaluation_method: str
    loss_key: TaskObjective
    sample_source: TaskSampleSource
    maturity: TaskMaturity

    evidence_modalities: tuple[TaskModality, ...] = ()
    min_evidence_modalities: int = 0

    optional_annotation_fields: tuple[str, ...] = ()
    sensitivity: TaskSensitivity = "standard"
    supports_hard_negatives: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not TASK_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid canonical task name: {self.name!r}")

        if not self.family.strip():
            raise ValueError(f"task {self.name!r} must declare a family")

        if not self.required_input_modalities:
            raise ValueError(
                f"task {self.name!r} must declare input modalities"
            )

        if not self.output_modalities:
            raise ValueError(
                f"task {self.name!r} must declare output modalities"
            )

        if len(set(self.required_input_modalities)) != len(
            self.required_input_modalities
        ):
            raise ValueError(
                f"task {self.name!r} has duplicate input modalities"
            )

        if len(set(self.output_modalities)) != len(self.output_modalities):
            raise ValueError(
                f"task {self.name!r} has duplicate output modalities"
            )

        if not self.evaluation_method.strip():
            raise ValueError(
                f"task {self.name!r} must declare an evaluation method"
            )

        if not self.loss_key.strip():
            raise ValueError(f"task {self.name!r} must declare a loss key")

        if self.min_evidence_modalities < 0:
            raise ValueError(
                f"task {self.name!r} min_evidence_modalities must be non-negative"
            )
        if self.min_evidence_modalities > len(self.evidence_modalities):
            raise ValueError(
                f"task {self.name!r} min_evidence_modalities cannot exceed "
                f"number of evidence_modalities"
            )
        if len(set(self.evidence_modalities)) != len(self.evidence_modalities):
            raise ValueError(
                f"task {self.name!r} has duplicate evidence modalities"
            )

        for modality in self.evidence_modalities:
            if modality in self.required_input_modalities:
                raise ValueError(
                    f"task {self.name!r} evidence modality {modality!r} "
                    f"must not be in required_input_modalities"
                )

    @property
    def crawler_buildable(self) -> bool:
        return self.sample_source != "external"

    @property
    def requires_external_labels(self) -> bool:
        return self.sample_source == "external"

    @property
    def production_blocked(self) -> bool:
        return (
            self.maturity in {"experimental", "disabled"}
            or self.sensitivity == "biometric_identity"
        )

    @property
    def required_approvals(self) -> frozenset[TaskApproval]:
        if self.production_blocked:
            return frozenset()

        approvals: set[TaskApproval] = set()
        if self.maturity == "beta":
            approvals.add("beta")
        if self.sensitivity == "sensitive":
            approvals.add("sensitive")
        return frozenset(approvals)
