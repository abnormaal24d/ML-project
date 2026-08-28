"""Composition contracts for preprocessing dependencies."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.collection.modality_acceptance import ImageAcceptanceSettings, ModalityAcceptanceSettingsCatalog
from config.collection.settings import CollectionSettings
from config.media_toolchain import MediaToolchainSettings
from config.preprocessing.media_settings import MediaPrivacySettings
from config.preprocessing.settings import PreprocessingSettings
from config.settings.root import Settings
from orchestration.composition import preprocessing_dependencies


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        return None


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


class _IdGenerator:
    def generate(self) -> str:
        return "preprocessing-run"


def test_image_privacy_factory_receives_acceptance_decode_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class CapturingImagePrivacyContentFactory:
        def __init__(
            self,
            *,
            ocr_engine: object,
            visual_analyzer: object,
            max_decode_pixels: int,
        ) -> None:
            captured.update(
                ocr_engine=ocr_engine,
                visual_analyzer=visual_analyzer,
                max_decode_pixels=max_decode_pixels,
            )

    max_decode_pixels = 123
    settings = Settings(
        profile="dev",
        collection=CollectionSettings(
            modality_acceptance=ModalityAcceptanceSettingsCatalog(
                image=ImageAcceptanceSettings(
                    fetch_max_bytes=1_000_000,
                    preprocessing_max_bytes=1_000_000,
                    max_decode_pixels=max_decode_pixels,
                )
            )
        ),
        media_toolchain=MediaToolchainSettings(),
        preprocessing=PreprocessingSettings(
            media_privacy=MediaPrivacySettings(),
        ),
    )
    monkeypatch.setattr(
        preprocessing_dependencies,
        "LocalImagePrivacyContentFactory",
        CapturingImagePrivacyContentFactory,
    )

    preprocessing_dependencies.build_multimodal_preprocessor(
        settings=settings,
        logger=_Logger(),
        clock=_Clock(),
        id_generator=_IdGenerator(),
    )

    assert captured["max_decode_pixels"] == max_decode_pixels
