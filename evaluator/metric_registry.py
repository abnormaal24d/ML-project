"""Evaluation method registry mapping to EvaluationPlans."""

from __future__ import annotations

from evaluator.metric_contracts import (
    EvaluationPlan,
    ExecutionKind,
)
from evaluator.metrics.causal_lm import CausalLMStrategy
from evaluator.metrics.mlm import MLMStrategy
from evaluator.metrics.retrieval import PairRetrievalStrategy
from evaluator.metrics.sequence.error_rates import (
    CERWERLayoutStrategy,
    CERWERStrategy,
)
from evaluator.metrics.sequence.exact_match_f1 import ExactMatchF1Strategy
from evaluator.metrics.sequence.rouge_token_f1 import RougeTokenF1Strategy
from evaluator.metrics.sequence.vqa import VQAAccuracyStrategy

EVALUATION_METHODS = {
    "masked_language_modeling": EvaluationPlan(
        execution_kind=ExecutionKind.MLM,
        scorer=MLMStrategy(),
    ),
    "causal_language_modeling": EvaluationPlan(
        execution_kind=ExecutionKind.CAUSAL_LM,
        scorer=CausalLMStrategy(),
    ),
    "retrieval_accuracy": EvaluationPlan(
        execution_kind=ExecutionKind.PAIR_RETRIEVAL,
        scorer=PairRetrievalStrategy(
            evaluation_method="retrieval_accuracy",
        ),
    ),
    "retrieval_or_contrastive": EvaluationPlan(
        execution_kind=ExecutionKind.PAIR_RETRIEVAL,
        scorer=PairRetrievalStrategy(
            evaluation_method="retrieval_or_contrastive",
        ),
    ),
    "exact_match_f1": EvaluationPlan(
        execution_kind=ExecutionKind.SEQUENCE_GENERATION,
        scorer=ExactMatchF1Strategy(),
    ),
    "rouge_or_token_f1": EvaluationPlan(
        execution_kind=ExecutionKind.SEQUENCE_GENERATION,
        scorer=RougeTokenF1Strategy(),
    ),
    "vqa_accuracy": EvaluationPlan(
        execution_kind=ExecutionKind.SEQUENCE_GENERATION,
        scorer=VQAAccuracyStrategy(),
    ),
    "cer_wer": EvaluationPlan(
        execution_kind=ExecutionKind.SEQUENCE_GENERATION,
        scorer=CERWERStrategy(),
    ),
    "cer_wer_layout": EvaluationPlan(
        execution_kind=ExecutionKind.SEQUENCE_GENERATION,
        scorer=CERWERLayoutStrategy(),
    ),
}

SUPPORTED_EVALUATION_METHODS = frozenset(EVALUATION_METHODS.keys())

__all__ = [
    "EVALUATION_METHODS",
    "SUPPORTED_EVALUATION_METHODS",
]
