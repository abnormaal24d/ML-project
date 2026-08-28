"""Selection decisions and quota-state contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from mmcrawler_datasets.training_samples.models import TrainingSample


class RejectReason(StrEnum):
    NO_SOURCE = "missing_source_url"
    NO_DOMAIN = "missing_domain"
    ROBOTS = "robots_disallowed"
    USAGE = "usage_rules_blocks_training"
    LICENSE = "license_rules_missing"
    LANGUAGE = "language_rules_reject"
    ALIGNMENT = "low_alignment_score"
    SAFETY = "safety_or_pii_blocked"
    SAFETY_CONTENT = "blocked_safety_content"
    PII_CONTENT = "blocked_pii_content"
    FINGERPRINT = "required_fingerprint_evidence_missing"
    LOW_CAPTION = "low_caption_quality"
    LOW_QUALITY = "low_modality_quality"
    LOW_CONTEXT = "low_context_score"
    CAPTION = "generic_caption"
    GENERIC_TEXT = "generic_or_uninformative_caption"
    DUPLICATE = "duplicate"
    QUOTA = "quota_exceeded"


BLOCKED_USAGE_RULES = frozenset(
    {
        "blocked",
        "deny_training",
        "disallow_training",
        "metadata_only",
        "no-training",
        "no_training",
        "opt_out",
        "restricted",
    }
)


@dataclass(slots=True)
class QuotaState:
    """Mutable counts consumed only after a sample passes every gate."""

    domains: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    domain_modalities: defaultdict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    topics: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    source_types: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record(self, sample: TrainingSample) -> None:
        self.domains[sample.domain] += 1
        self.domain_modalities[(sample.domain, sample.modality)] += 1
        self.topics[_topic(sample)] += 1
        self.source_types[_source_type(sample)] += 1


def _topic(sample: TrainingSample) -> str:
    if sample.title:
        return (
            " ".join(sample.title.casefold().strip().split())[:64] or "unknown"
        )
    return sample.task_target.task_type or "unknown"


def _source_type(sample: TrainingSample) -> str:
    return sample.paired_text_source or sample.modality or "unknown"


__all__ = ["QuotaState", "RejectReason"]
