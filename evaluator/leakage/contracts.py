"""Canonical leakage-v2 constants and category metadata."""

SCHEMA_VERSION = "2.0"
DEFAULT_OVERLAP_SAMPLE_LIMIT = 1_000
DEFAULT_MAX_INDEXED_RECORDS_PER_CATEGORY = 1_000_000
DEFAULT_MAX_CANDIDATES_PER_RECORD = 100_000
MAX_REPORT_BYTES = 16 * 1024 * 1024

CATEGORIES = (
    "canonical_url_sha256",
    "scheme_agnostic_url_sha256",
    "object_sha256",
    "content_hash",
    "emitted_text_sha256",
    "normalized_text_sha256",
    "near_duplicate_text",
    "image_phash",
    "image_dhash",
    "audio_chromaprint",
    "video_keyframe_sequence",
    "document_layout_sha256",
)

ALGORITHMS = {
    "canonical_url_sha256": "sha256:canonical-url-v1",
    "scheme_agnostic_url_sha256": "sha256:scheme-agnostic-url-v1",
    "object_sha256": "sha256:object-bytes",
    "content_hash": "sha256:canonical-content",
    "emitted_text_sha256": "sha256:emitted-training-text",
    "normalized_text_sha256": "sha256:normalized-text",
    "near_duplicate_text": "jaccard:token-shingle-v1",
    "image_phash": "hamming:phash-64-v1",
    "image_dhash": "hamming:dhash-64-v1",
    "audio_chromaprint": "chromaprint:v1",
    "video_keyframe_sequence": "sequence:phash-v1",
    "document_layout_sha256": "sha256:document-layout-v1",
}

EXACT_CATEGORIES = tuple(
    category
    for category in CATEGORIES
    if category not in {"near_duplicate_text", "image_phash", "image_dhash"}
)
PERCEPTUAL_CATEGORIES = ("image_phash", "image_dhash")
SHA256_EVIDENCE_CATEGORIES = tuple(
    category
    for category in EXACT_CATEGORIES
    if category != "audio_chromaprint"
)
PERCEPTUAL_HASH_HEX_LENGTH = 16
