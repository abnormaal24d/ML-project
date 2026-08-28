"""Evaluation contracts, protocols, and typed state."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from torch import Tensor

if TYPE_CHECKING:
    from multimodal.model.contracts import CollatedBatch
    from multimodal.tokenization.text import VocabularyTokenizer


class ExecutionKind(Enum):
    """Execution model for one evaluation method."""

    MLM = "mlm"
    CAUSAL_LM = "causal_lm"
    PAIR_RETRIEVAL = "pair_retrieval"
    SEQUENCE_GENERATION = "sequence_generation"


class MetricStrategy(Protocol):
    """Stateless scoring strategy for one evaluation method."""

    evaluation_method: str

    def accumulate(
        self,
        *,
        state: EvaluationState,
        batch: CollatedBatch,
        outputs: Mapping[str, Tensor],
        tokenizer: VocabularyTokenizer | None,
        model: Any,
    ) -> None:
        """Accumulate metrics for one batch."""

    def synchronize(
        self,
        *,
        state: EvaluationState,
        device: Any,
    ) -> None:
        """Synchronize state across DDP ranks."""

    def finalize(
        self,
        *,
        state: EvaluationState,
    ) -> dict[str, dict[str, float]]:
        """Finalize and return per-task metrics."""


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Complete evaluation plan for one evaluation method."""

    execution_kind: ExecutionKind
    scorer: MetricStrategy


@dataclass(slots=True)
class MLMState:
    correct: int = 0
    total: int = 0


@dataclass(slots=True)
class CausalLMState:
    samples: int = 0
    token_correct: int = 0
    token_total: int = 0
    ce_loss_sum: float = 0.0
    ce_loss_count: int = 0


@dataclass(slots=True)
class SequenceStatistics:
    samples: float = 0.0
    exact: float = 0.0
    token_f1_sum: float = 0.0
    rouge_l_sum: float = 0.0
    token_correct: float = 0.0
    token_total: float = 0.0
    character_edits: float = 0.0
    character_reference_length: float = 0.0
    word_edits: float = 0.0
    word_reference_length: float = 0.0
    layout_squared_error: float = 0.0
    layout_element_count: float = 0.0
    ce_loss_sum: float = 0.0
    ce_loss_count: float = 0.0


@dataclass(slots=True)
class EvaluationState:
    """Mutable state accumulated during evaluation."""

    pair_text_embeddings: dict[str, list[Tensor]] = field(default_factory=dict)
    pair_media_embeddings: dict[str, list[Tensor]] = field(
        default_factory=dict
    )
    sequence_statistics: dict[str, SequenceStatistics] = field(
        default_factory=dict
    )
    mlm_state: MLMState = field(default_factory=MLMState)
    causal_lm_state: CausalLMState = field(default_factory=CausalLMState)


class RuntimeObserver(Protocol):
    """Optional runtime metrics observer."""

    def reset(self, *, device: Any) -> None: ...

    def start_batch(self, *, device: Any) -> None: ...

    def end_batch(self, *, device: Any) -> float: ...

    def peak_memory_mb(self, *, device: Any) -> float | None: ...


@dataclass(slots=True)
class PairRetrievalConfig:
    """Configuration for pair retrieval evaluation."""

    pair_task_order: tuple[str, ...]
    pair_task_modality: dict[str, str]


PAIR_TASK_ORDER = (
    "document_text_pair",
    "pdf_text_pair",
    "image_text_pair",
    "audio_text_pair",
    "video_text_pair",
    "multimodal_retrieval",
    "cross_modal_consistency",
)

PAIR_TASKS = frozenset(PAIR_TASK_ORDER)

PAIR_TASK_MODALITIES = {
    "document_text_pair": "document",
    "pdf_text_pair": "document",
    "image_text_pair": "image",
    "audio_text_pair": "audio",
    "video_text_pair": "video",
}

SEQUENCE_STAT_FIELDS = tuple(
    sequence_field.name for sequence_field in fields(SequenceStatistics)
)

__all__ = [
    "ExecutionKind",
    "MetricStrategy",
    "EvaluationPlan",
    "EvaluationState",
    "MLMState",
    "CausalLMState",
    "SequenceStatistics",
    "RuntimeObserver",
    "PairRetrievalConfig",
    "PAIR_TASK_ORDER",
    "PAIR_TASKS",
    "PAIR_TASK_MODALITIES",
    "SEQUENCE_STAT_FIELDS",
]
