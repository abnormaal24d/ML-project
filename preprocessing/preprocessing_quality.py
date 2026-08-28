"""Quality models shared by preprocessing documents and results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PreprocessingQualityResult:
    score: float
    bucket: str
    rejection_reason: str | None
    token_count_estimate: int
    modality: str = "text"
    safety_status: str = "unchecked"
    language: str | None = None
    license: str | None = None
    dedupe_key: str | None = None
    alignment_score: float | None = None
    signals: dict[str, float | int | bool | str | None] = field(
        default_factory=dict
    )
    modality_signals: dict[str, object] = field(default_factory=dict)
