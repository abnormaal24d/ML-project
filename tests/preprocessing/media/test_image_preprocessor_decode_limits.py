"""Image-preprocessing decode policy wiring regressions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from config.collection.modality_acceptance import ImageAcceptanceSettings
from config.preprocessing.media_settings import ImageValidationSettings
from preprocessing.media.image.image_preprocessor import ImagePreprocessor
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)


class _Logger:
    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None


def test_image_preprocessor_uses_its_configured_decode_pixel_limit(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "too-large-for-configured-cap.png"
    Image.new("RGB", (11, 10)).save(image_path, format="PNG")
    preprocessor = ImagePreprocessor(
        logger=_Logger(),
        settings=ImageValidationSettings(
            min_width=1,
            min_height=1,
            require_semantic_text_for_alignment=False,
        ),
        modality_acceptance=ImageAcceptanceSettings(
            fetch_max_bytes=1_000,
            preprocessing_max_bytes=1_000,
            max_decode_pixels=100,
        ),
        pii_detector=None,
        privacy_content_factory=None,
        embedded_metadata_adapter=None,
        now=lambda: datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
        generate_id=lambda: "decode-limit-run",
    )
    item = PreprocessingInput(
        source_id="image",
        source_url="https://example.test/image.png",
        normalized_url="https://example.test/image.png",
        domain="example.test",
        path="/image.png",
        language_evidence=LanguageEvidence(language="en"),
        modality="image",
        mime_type="image/png",
        media_path=str(image_path),
        byte_size=image_path.stat().st_size,
    )

    validation = preprocessor._validate(item=item)

    assert validation.accepted is False
    assert validation.rejection_reason == "decode_failed"
