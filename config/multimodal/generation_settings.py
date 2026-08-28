"""Generation output settings for text, image, audio, and video."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel
from config.environment.default_values import DEFAULT_AUDIO_SAMPLE_RATE_HZ


class GenerationSettings(SettingsModel):
    """Generation knobs shared by text, image, audio, and video outputs."""

    allow_tensor_artifact_fallback: bool = False
    require_decodeable_generation_outputs: bool = True

    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=4.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0, le=32)
    beam_size: int = Field(default=1, gt=0)
    image_steps: int = Field(default=30, gt=0)
    audio_codec_rate: int = Field(default=50, gt=0)
    video_steps: int = Field(default=30, gt=0)
    streaming_chunk_ms: int = Field(default=40, gt=0)

    @model_validator(mode="after")
    def _validate_beam_search_sampling(self) -> GenerationSettings:
        if self.beam_size > 1 and (
            self.temperature != 1.0 or self.top_p != 1.0
        ):
            raise ValueError(
                "generation beam_size > 1 requires deterministic decoding; "
                "set temperature=1.0 and top_p=1.0 or use beam_size=1"
            )
        return self


class AudioTokenizerSettings(SettingsModel):
    """Audio tokenization settings for speech and codec-style training."""

    enabled: bool = False
    codec: Literal["none", "continuous", "discrete"] = "continuous"
    sample_rate: int = Field(default=DEFAULT_AUDIO_SAMPLE_RATE_HZ, gt=0)
    frame_ms: int = Field(default=20, gt=0)
    hop_ms: int = Field(default=20, gt=0)
    codebook_size: int = Field(default=1024, gt=0)
    # The current generation target path emits one categorical stream. RVQ
    # codebooks remain configured independently on AudioCodecSettings.
    n_codebooks: Literal[1] = 1
    token_dim: int = Field(default=256, gt=0)


class ImageGeneratorSettings(SettingsModel):
    """Image generation strategy and latent-space settings."""

    enabled: bool = False
    strategy: Literal["diffusion", "autoregressive_tokens", "hybrid"] = (
        "diffusion"
    )
    # Image codec settings (moved to ImageCodecSettings)
    # latent_dim: int = Field(default=256, gt=0)
    # image_token_vocab_size: int = Field(default=8192, gt=128)
    # resolution: int = Field(default=512, gt=0)
    # latent_cache_enabled: bool = False
    safety_filter_enabled: bool = True
    watermark_metadata_enabled: bool = True

    # Diffusion-specific settings
    hidden_dim: int = Field(default=512, gt=0)
    num_layers: int = Field(default=12, gt=0)
    num_heads: int = Field(default=8, gt=0)
    num_train_timesteps: int = Field(default=1000, gt=0)
    prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon"


class ImageDecoderSettings(SettingsModel):
    """Enable decoding through the canonical image codec.

    The codec owns its latent channels, output channels, resolution, and
    upsampling topology. Keeping duplicate shape fields here would permit a
    configuration that the actual decoder cannot represent.
    """

    enabled: bool = False


class VideoGeneratorSettings(SettingsModel):
    """Video generation token/latent settings."""

    enabled: bool = False
    strategy: Literal[
        "autoregressive_frame_tokens",
        "diffusion",
        "hybrid",
    ] = "autoregressive_frame_tokens"
    latent_dim: int = Field(default=256, gt=0)
    # 128 is the smallest supported vocabulary.  Keeping this inclusive makes
    # compact video-token models usable in CPU smoke and integration runs.
    video_token_vocab_size: int = Field(default=8192, ge=128)
    frames: int = Field(default=8, gt=0)
    resolution: int = Field(default=128, gt=0)
    grid_height: int = Field(default=16, gt=0)
    grid_width: int = Field(default=16, gt=0)
    temporal_layers: int = Field(default=2, gt=0)
    latent_cache_enabled: bool = False


class ImageCodecSettings(SettingsModel):
    """Settings for the scratch-trained image codec (encoder + decoder)."""

    latent_channels: int = Field(default=8, gt=0)
    downsample_factor: int = Field(default=16, gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    input_resolution: int = Field(default=224, gt=0)
    output_channels: int = Field(default=3, gt=0)
    upsample_blocks: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _validate_spatial_shape_contract(self) -> ImageCodecSettings:
        """Keep the declared codec resolution in sync with its topology."""

        topology_factor = 1 << int(self.upsample_blocks)
        if self.downsample_factor != topology_factor:
            raise ValueError(
                "image_codec.downsample_factor must equal 2 ** "
                "image_codec.upsample_blocks"
            )
        if self.input_resolution % self.downsample_factor != 0:
            raise ValueError(
                "image_codec.input_resolution must be divisible by "
                "image_codec.downsample_factor"
            )
        return self


class VideoDecoderSettings(SettingsModel):
    """Decode generated video tokens or latents back to frames."""

    enabled: bool = False
    latent_channels: int = Field(default=4, gt=0)
    output_channels: int = Field(default=3, gt=0)
    upsample_blocks: int = Field(default=3, gt=0)
    temporal_upsample_blocks: int = Field(default=2, gt=0)


class AudioCodecSettings(SettingsModel):
    """Settings for the scratch-trained audio codec (encoder + RVQ + decoder)."""

    latent_dim: int = Field(default=256, gt=0)
    downsample_factor: int = Field(default=480, gt=0)
    hidden_dim: int = Field(default=512, gt=0)
    n_codebooks: int = Field(default=4, gt=0)
    codebook_size: int = Field(default=1024, gt=0)
    token_dim: int = Field(default=256, gt=0)
    commitment_weight: float = Field(default=0.25, ge=0.0)
    output_channels: int = Field(default=1, gt=0)


class VocoderSettings(SettingsModel):
    """Audio generation output settings."""

    enabled: bool = False
    sample_rate: int = Field(default=24000, gt=0)
    hop_length: int = Field(default=320, gt=0)
    preserve_prosody: bool = True


class GenerationModelSettings(SettingsModel):
    """Generation backends composed into the multimodal model."""

    audio_tokenizer: AudioTokenizerSettings = Field(
        default_factory=AudioTokenizerSettings
    )
    image_generator: ImageGeneratorSettings = Field(
        default_factory=ImageGeneratorSettings
    )
    image_decoder: ImageDecoderSettings = Field(
        default_factory=ImageDecoderSettings
    )
    image_codec: ImageCodecSettings = Field(default_factory=ImageCodecSettings)
    audio_codec: AudioCodecSettings = Field(default_factory=AudioCodecSettings)
    video_generator: VideoGeneratorSettings = Field(
        default_factory=VideoGeneratorSettings
    )
    video_decoder: VideoDecoderSettings = Field(
        default_factory=VideoDecoderSettings
    )
    vocoder: VocoderSettings = Field(default_factory=VocoderSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
