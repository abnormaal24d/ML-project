"""Shared default values for settings and runtime fallbacks.

Defaults with units include those units in their constant names. ``None`` is
reserved for optional values that mean "use the configured behavior"; numeric
zero remains available as an explicit caller-provided value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class LossWeightDefaults:
    """Canonical multimodal loss weights and exponents.

    Values are dimensionless multipliers. A value of ``0.0`` disables the
    corresponding auxiliary loss by default.
    """

    contrastive_temperature: float = 0.07
    alignment_loss_power: float = 1.0
    mlm_loss_weight: float = 0.15
    image_patch_loss_weight: float = 0.1
    audio_masked_loss_weight: float = 0.1
    audio_token_loss_weight: float = 0.5
    video_temporal_loss_weight: float = 0.05
    hard_negative_loss_weight: float = 0.2
    hard_negative_margin: float = 0.2
    disabled_auxiliary_loss_weight: float = 0.0


DEFAULT_MAX_WORKFLOW_ITERATIONS: Final[int] = 50
DEFAULT_WORKFLOW_ITERATION_PAUSE_SECONDS: Final[float] = 0.05
DEFAULT_WORKFLOW_BLOCKING_TASK_LIMIT: Final[int] = 4
DEFAULT_WORKFLOW_IO_TIMEOUT_SECONDS: Final[float] = 300.0
DEFAULT_DATA_CHECKER_TIMEOUT_SECONDS: Final[float] = 60.0

DEFAULT_MANIFEST_REPLACE_RETRY_ATTEMPTS: Final[int] = 8
DEFAULT_MANIFEST_REPLACE_RETRY_DELAY_SECONDS: Final[float] = 0.1
DEFAULT_MANIFEST_REPLACE_RETRY_JITTER_SECONDS: Final[float] = 0.1
DEFAULT_PROCESSOR_RETRIES: Final[int] = 3
DEFAULT_FEED_PROCESSOR_MAX_ENTRIES: Final[int] = 500
DEFAULT_FEED_ANALYSIS_MAX_ENTRIES: Final[int | None] = None

DEFAULT_AUDIO_SAMPLE_RATE_HZ: Final[int] = 16_000
DEFAULT_MEDIA_ANALYSIS_TIMEOUT_SECONDS: Final[float] = 120.0
DEFAULT_VIDEO_METADATA_USER_AGENT: Final[str] = "DataEngineBot/1.0"

DEFAULT_DATASET_SPLITS_DIRECTORY: Final[str] = "splits"
DEFAULT_TRAIN_SPLIT_NAME: Final[str] = "train"
DEFAULT_VAL_SPLIT_NAME: Final[str] = "val"
DEFAULT_TEST_SPLIT_NAME: Final[str] = "test"
DEFAULT_DATASET_SPLIT_NAMES: Final[tuple[str, str, str]] = (
    DEFAULT_TRAIN_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
    DEFAULT_TEST_SPLIT_NAME,
)
DEFAULT_TRAIN_SPLIT_RATIO: Final[float] = 0.8
DEFAULT_VAL_SPLIT_RATIO: Final[float] = 0.1
DEFAULT_TEST_SPLIT_RATIO: Final[float] = 0.1
DEFAULT_TRAIN_SPLIT_FILENAME: Final[str] = "train.jsonl"
DEFAULT_VAL_SPLIT_FILENAME: Final[str] = "val.jsonl"
DEFAULT_TEST_SPLIT_FILENAME: Final[str] = "test.jsonl"
DEFAULT_DATASET_MANIFEST_FILENAME: Final[str] = "dataset_manifest.json"
DEFAULT_STATS_FILENAME: Final[str] = "stats.json"
DEFAULT_DATASET_CARD_FILENAME: Final[str] = "dataset_card.json"
DEFAULT_VALIDATION_REPORT_FILENAME: Final[str] = "validation_report.json"
DEFAULT_SHINGLE_WIDTH: Final[int] = 4
DEFAULT_SHINGLE_CANDIDATE_BANDS: Final[int] = 4
DEFAULT_CACHE_ITEMS: Final[int] = 4096
DEFAULT_TEXT_SAMPLE_BYTES: Final[int] = 4096

DEFAULT_TEXT_FEATURE_DIM: Final[int] = 512
DEFAULT_IMAGE_FEATURE_DIM: Final[int] = 512
DEFAULT_AUDIO_FEATURE_DIM: Final[int] = 256
DEFAULT_VIDEO_FEATURE_DIM: Final[int] = 512
DEFAULT_RAW_TEXT_MAX_TOKENS: Final[int] = 128
DEFAULT_RAW_TEXT_VOCAB_SIZE: Final[int] = 8192
DEFAULT_RAW_IMAGE_SIZE: Final[int] = 64
DEFAULT_RAW_AUDIO_NUM_SAMPLES: Final[int] = 16_000
DEFAULT_RAW_VIDEO_FRAMES: Final[int] = 8
DEFAULT_MODEL_DROPOUT_PROBABILITY: Final[float] = 0.1
DEFAULT_MLM_PROBABILITY: Final[float] = 0.15
DEFAULT_IMAGE_MASK_PROBABILITY: Final[float] = 0.2
DEFAULT_AUDIO_MASK_PROBABILITY: Final[float] = 0.15

DEFAULT_MEDIA_ATTRIBUTES: Final[tuple[str, ...]] = (
    "src",
    "data-src",
    "href",
    "data-href",
    "poster",
    "data-poster",
    "download",
)

DEFAULT_HOST_PROFILE_FORBIDDEN_COOLDOWN_SECONDS: Final[float] = 120.0
DEFAULT_INFLIGHT_HOST_WAIT_SECONDS: Final[float] = 0.1

ENRICHMENT_PREVIEW_MAX_CHARACTERS: Final[int] = 240
DEFAULT_RESPONSE_SNIFF_BYTE_COUNT: Final[int] = 4096
DEFAULT_LOG_RATE_LIMIT_MAX_ENTRIES: Final[int] = 10_000
DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS: Final[int] = 2
DEFAULT_CONTENT_RELEVANCE_ROUNDING_DIGITS: Final[int] = 3
DEFAULT_LOSS_WEIGHTS: Final[LossWeightDefaults] = LossWeightDefaults()
