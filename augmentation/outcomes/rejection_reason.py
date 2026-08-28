"""Typed rejection reasons emitted by the augmentation workflow."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["AugmentationRejectionReason"]


class AugmentationRejectionReason(StrEnum):
    """Canonical rejection reasons for augmentation attempts."""

    AUDIO_DURATION_TOO_LONG = "audio_duration_too_long"
    AUDIO_SOURCE_HASH_FAILED = "audio_source_hash_failed"
    AUDIO_SOURCE_PATH_INVALID = "audio_source_path_invalid"
    AUDIO_TRANSFORM_FAILED = "audio_transform_failed"
    AUGMENTATION_DISABLED = "augmentation_disabled"
    CONTENT_AWARE_CROP_NO_SAFE_WINDOW = "content_aware_crop_no_safe_window"
    CONTEXT_ALREADY_PRESENT = "context_already_present"
    DATASET_ROOT_MISSING = "dataset_root_missing"
    DOCUMENT_AUGMENTATION_DISABLED = "document_augmentation_disabled"
    DOCUMENT_OUTPUT_DECODE_FAILED = "generated_document_page_decode_failed"
    DOCUMENT_OUTPUT_DIMENSIONS_MISMATCH = (
        "generated_document_page_dimensions_mismatch"
    )
    DOCUMENT_OUTPUT_MIME_MISMATCH = "generated_document_page_mime_mismatch"
    DOCUMENT_OUTPUT_MISSING = "missing_generated_document_page"
    DOCUMENT_OUTPUT_TOO_LARGE = "generated_document_page_too_large"
    DOCUMENT_PAGE_IMAGE_MISSING = "document_page_image_missing"
    DOCUMENT_PAGE_IMAGE_TRANSFORM_FAILED = (
        "document_page_image_transform_failed"
    )
    DOCUMENT_PAGE_IMAGE_UNCHANGED = "document_page_image_unchanged"
    DUPLICATE_MEDIA_VARIANT_DATASET = "duplicate_media_variant_dataset"
    DUPLICATE_TEXT_DATASET = "duplicate_text_dataset"
    DUPLICATE_TEXT_SAMPLE = "duplicate_text_sample"
    GENERATED_AUDIO_CHANNEL_MISMATCH = "generated_audio_channel_mismatch"
    GENERATED_AUDIO_CLIPPING_EXCESSIVE = "generated_audio_clipping_excessive"
    GENERATED_AUDIO_DECODE_FAILED = "generated_audio_decode_failed"
    GENERATED_AUDIO_DURATION_INVALID = "generated_audio_duration_invalid"
    GENERATED_AUDIO_DURATION_MISMATCH = "generated_audio_duration_mismatch"
    GENERATED_AUDIO_FRAME_ALIGNMENT_INVALID = (
        "generated_audio_frame_alignment_invalid"
    )
    GENERATED_AUDIO_INVALID = "generated_audio_invalid"
    GENERATED_AUDIO_PCM_CONTRACT_INVALID = (
        "generated_audio_pcm_contract_invalid"
    )
    GENERATED_AUDIO_SAMPLE_RATE_MISMATCH = (
        "generated_audio_sample_rate_mismatch"
    )
    GENERATED_AUDIO_SIZE_INVALID = "generated_audio_size_invalid"
    GENERATED_IMAGE_DECODE_FAILED = "generated_image_decode_failed"
    GENERATED_IMAGE_DIMENSIONS_INVALID = "generated_image_dimensions_invalid"
    GENERATED_IMAGE_HEIGHT_MISMATCH = "generated_image_height_mismatch"
    GENERATED_IMAGE_INVALID = "generated_image_invalid"
    GENERATED_IMAGE_TOO_LARGE = "generated_image_too_large"
    GENERATED_IMAGE_TOO_MANY_PIXELS = "generated_image_too_many_pixels"
    GENERATED_IMAGE_WIDTH_MISMATCH = "generated_image_width_mismatch"
    IMAGE_BACKEND_MISSING = "image_backend_missing"
    IMAGE_DECODE_FAILED = "image_decode_failed"
    IMAGE_DIMENSIONS_TOO_SMALL = "image_dimensions_too_small"
    IMAGE_TOO_MANY_PIXELS = "image_too_many_pixels"
    IMAGE_TRANSFORM_FAILED = "image_transform_failed"
    IMAGE_VARIANT_SEMANTICALLY_UNCHANGED = (
        "image_variant_semantically_unchanged"
    )
    INVALID_AUDIO = "invalid_audio"
    INVALID_AUDIO_SIZE = "invalid_audio_size"
    INVALID_IMAGE = "invalid_image"
    INVALID_IMAGE_SIZE = "invalid_image_size"
    MAX_VARIANTS_LIMIT_REACHED = "max_variants_limit_reached"
    MAX_VARIANTS_ZERO = "max_variants_zero"
    MEDIA_ANNOTATIONS_NOT_TRANSFORMABLE = "media_annotations_not_transformable"
    MEDIA_AUGMENTATION_FAILED = "media_augmentation_failed"
    MISSING_AUDIO_FILE = "missing_audio_file"
    MISSING_CONTEXT = "missing_context"
    MISSING_GENERATED_AUDIO = "missing_generated_audio"
    MISSING_GENERATED_IMAGE = "missing_generated_image"
    MISSING_IMAGE_FILE = "missing_image_file"
    MISSING_TEXT_SPAN = "missing_text_span"
    MISSING_TITLE = "missing_title"
    MISSING_VIDEO_FILE = "missing_video_file"
    MODALITY_NOT_ALLOWED = "modality_not_allowed"
    NO_ENABLED_TEXT_STRATEGY = "no_enabled_text_strategy"
    NON_TRAIN_SPLIT = "non_train_split"
    RULES_SKIP = "rules_skip"
    SAMPLE_AUGMENTATION_FAILED = "sample_augmentation_failed"
    STRATEGY_NO_OUTPUT = "strategy_no_output"
    TEXT_AUGMENTATION_DISABLED = "text_augmentation_disabled"
    TEXT_SPAN_NOT_APPLICABLE = "text_span_not_applicable"
    TEXT_TOO_LONG = "text_too_long"
    TEXT_TOO_SHORT = "text_too_short"
    TITLE_ALREADY_PRESENT = "title_already_present"
    UNSUPPORTED_AUDIO_MIME_TYPE = "unsupported_audio_mime_type"
    UNSUPPORTED_DOCUMENT_OPERATION = "unsupported_document_operation"
    UNSUPPORTED_IMAGE_MIME_TYPE = "unsupported_image_mime_type"
    UNSUPPORTED_OUTPUT_IMAGE_MIME_TYPE = "unsupported_output_image_mime_type"
    VARIANT_REJECTED = "variant_rejected"
    VARIANT_TEXT_EMPTY = "variant_text_empty"
    VARIANT_TEXT_TOO_LONG = "variant_text_too_long"
    VIDEO_BACKEND_MISSING = "video_backend_missing"
    VIDEO_CLIP_TRANSFORM_FAILED = "video_clip_transform_failed"
    VIDEO_KEYFRAME_TASK_NOT_SUPPORTED = "video_keyframe_task_not_supported"
    VIDEO_KEYFRAME_TRANSFORM_FAILED = "video_keyframe_transform_failed"
    VIDEO_SOURCE_PATH_INVALID = "video_source_path_invalid"
    VIDEO_TOO_LARGE = "video_too_large"
