"""Loss contract types: typed loss terms and collector context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch


@dataclass(frozen=True, slots=True)
class LossTerm:
    """Single loss term with its row-level coverage.

    Attributes:
        name: Unique loss term identifier (matches SUPPORTED_LOSS_KEYS).
        value: Scalar loss tensor.
        coverage: Boolean tensor of shape [batch_size] indicating which
            training rows contributed to this loss term.
    """

    name: str
    value: torch.Tensor
    coverage: torch.Tensor


@dataclass(frozen=True, slots=True)
class LossContext:
    """Immutable context passed to all loss collectors.

    Contains all configuration needed for loss computation without
    coupling collectors to the full training objective module.
    """

    weights: Mapping[str, float]
    contrastive_temperature: float
    alignment_score_exponent: float
    hard_negative_margin: float
    training_backend: str
    training_stage: str
    preference_mode: str
    preference_beta: float


class LossCollector(Protocol):
    """Protocol for loss collector callables."""

    def __call__(
        self,
        *,
        model_output: Mapping[str, torch.Tensor],
        batch: object,
        context: LossContext,
    ) -> tuple[LossTerm, ...]: ...


class LossCollection:
    """Ordered collection of unique loss terms with deduplication check."""

    def __init__(self) -> None:
        self._terms: dict[str, LossTerm] = {}

    def add(self, term: LossTerm) -> None:
        if term.name in self._terms:
            raise RuntimeError(f"duplicate loss term {term.name!r}")
        self._terms[term.name] = term

    def extend(self, terms: tuple[LossTerm, ...]) -> None:
        for term in terms:
            self.add(term)

    def items(self):
        return self._terms.items()

    def values(self):
        return self._terms.values()

    def keys(self):
        return self._terms.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._terms

    def __getitem__(self, name: str) -> LossTerm:
        return self._terms[name]

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)
