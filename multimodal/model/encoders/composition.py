"""Construct and own the configured modality encoders."""

from __future__ import annotations

from torch import nn

from config.multimodal.model_settings import ModelSettings
from multimodal.model.encoders.audio import AutoAudioEncoder
from multimodal.model.encoders.document import LayoutAwareDocumentEncoder
from multimodal.model.encoders.image import AutoVisionEncoder
from multimodal.model.encoders.text import AutoTextEncoder
from multimodal.model.encoders.video import AutoVideoEncoder


class EncoderComposition(nn.Module):
    """Own all modality encoders needed by one model instance."""

    def __init__(
        self,
        *,
        config: ModelSettings,
        enabled_modalities: tuple[str, ...],
    ) -> None:
        super().__init__()
        _reject_pretrained_encoder_configuration(config)

        encoders = nn.ModuleDict()
        if "text" in enabled_modalities:
            encoders["text"] = AutoTextEncoder(
                config.text,
                vocab_size=config.raw_text_vocab_size,
                max_tokens=config.raw_text_max_tokens,
                gradient_checkpointing=config.gradient_checkpointing,
            )

        self.document_text_encoder: nn.Module | None = None
        if "document" in enabled_modalities:
            self.document_text_encoder = AutoTextEncoder(
                config.document,
                vocab_size=config.raw_text_vocab_size,
                max_tokens=config.raw_text_max_tokens,
                gradient_checkpointing=config.gradient_checkpointing,
            )

        if "image" in enabled_modalities:
            encoders["image"] = AutoVisionEncoder(
                config.image,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        if "audio" in enabled_modalities:
            encoders["audio"] = AutoAudioEncoder(
                config.audio,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        if "video" in enabled_modalities:
            encoders["video"] = AutoVideoEncoder(
                config.video,
                max_frames=config.raw_video_frames,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        if "document" in enabled_modalities:
            encoders["document"] = LayoutAwareDocumentEncoder(
                token_dim=config.fusion_dim,
                hidden_dim=config.fusion_dim,
                attention_heads=config.document.attention_heads,
            )

        self.encoders = encoders

    @property
    def document_encoder(self) -> LayoutAwareDocumentEncoder:
        if "document" not in self.encoders:
            raise RuntimeError("document encoder is not enabled")
        module = self.encoders["document"]
        if not isinstance(module, LayoutAwareDocumentEncoder):
            raise RuntimeError("document encoder is not enabled")
        return module


def build_encoder_composition(
    *,
    config: ModelSettings,
    enabled_modalities: tuple[str, ...],
) -> EncoderComposition:
    """Build one self-contained encoder composition."""

    return EncoderComposition(
        config=config,
        enabled_modalities=enabled_modalities,
    )


def _reject_pretrained_encoder_configuration(config: ModelSettings) -> None:
    for field in (
        "pretrained_model",
        "encoder_checkpoint",
        "hf_model_id",
        "pretrained_name",
    ):
        value = getattr(config, field, None)
        if value:
            raise ValueError(
                "Per-encoder pretrained loading is forbidden "
                f"(ADR-0002): {field}={value}"
            )
