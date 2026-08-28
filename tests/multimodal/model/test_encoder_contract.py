from __future__ import annotations

from math import prod

import pytest
import torch
from torch import nn

from config.multimodal.encoder_settings import EncoderSettings
from config.multimodal.generation_settings import (
    AudioCodecSettings,
    AudioTokenizerSettings,
)
from config.multimodal.model_settings import ModelSettings
from config.multimodal.training_head_settings import DecoderSettings
from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from multimodal.model.contracts import CollatedBatch
from multimodal.model.encoders.document import LayoutAwareDocumentEncoder
from multimodal.model.encoders.runner import encode_active
from multimodal.model.generation.audio_codec import (
    ResidualVectorQuantizer,
    ScratchAudioCodec,
)
from multimodal.model.model import MultimodalModel
from multimodal.model.outputs.builders import GenerationOutputBuilder
from multimodal.model.outputs.projection import ProjectionHeads
from multimodal.tokenization.audio import AudioTokenizer


class _GoodEncoder(nn.Module):
    def forward(
        self,
        values: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del output_keys
        return {"embedding": values}


class _TensorEncoder(nn.Module):
    def forward(  # type: ignore[override]
        self,
        values: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> torch.Tensor:
        del output_keys
        return values


class _FailingEncoder(nn.Module):
    def forward(
        self,
        values: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del values, output_keys
        raise TypeError("internal encoder defect")


def test_project_encoder_contract_requires_embedding_dict() -> None:
    values = torch.ones((2, 3))
    result = encode_active(
        encoder=_GoodEncoder(),
        values=values,
        mask=torch.ones(2, dtype=torch.bool),
        output_dim=3,
        output_keys={"embedding"},
    )
    assert torch.equal(result["embedding"], values)

    with pytest.raises(TypeError, match="must return a dict"):
        encode_active(
            encoder=_TensorEncoder(),
            values=values,
            mask=torch.ones(2, dtype=torch.bool),
            output_dim=3,
            output_keys=None,
        )


def test_internal_type_error_is_not_masked_by_signature_fallback() -> None:
    with pytest.raises(TypeError, match="internal encoder defect"):
        encode_active(
            encoder=_FailingEncoder(),
            values=torch.ones((1, 2)),
            mask=torch.ones(1, dtype=torch.bool),
            output_dim=2,
            output_keys=None,
        )


def test_layout_encoder_zeroes_fully_masked_document_rows() -> None:
    encoder = LayoutAwareDocumentEncoder(
        token_dim=8, hidden_dim=8, attention_heads=2
    )
    output = encoder(
        token_embeddings=torch.randn(2, 3, 8),
        boxes=torch.zeros(2, 3, 4),
        page_ids=torch.zeros(2, 3, dtype=torch.long),
        attention_mask=torch.tensor(
            [[True, True, False], [False, False, False]]
        ),
    )

    assert torch.isfinite(output["tokens"]).all()
    assert torch.isfinite(output["embedding"]).all()
    assert torch.count_nonzero(output["tokens"][1]).item() == 0
    assert torch.count_nonzero(output["embedding"][1]).item() == 0


def test_disabled_generation_backends_do_not_allocate_output_heads() -> None:
    heads = ProjectionHeads(config=ModelSettings())

    assert heads.image_pixel_head is None
    assert heads.image_latent_head is None
    assert heads.audio_generation_head is None
    assert heads.video_generation_head is None


def test_audio_generation_head_uses_the_single_tokenizer_codebook() -> None:
    heads = ProjectionHeads(
        config=ModelSettings(
            audio_tokenizer=AudioTokenizerSettings(
                enabled=True,
                codec="discrete",
            )
        )
    )

    assert heads.audio_n_codebooks == 1
    outputs = GenerationOutputBuilder(heads=heads).build(
        fused=torch.zeros((2, heads.fusion_dim)),
        resolved_heads=frozenset({"audio_token"}),
    )
    assert outputs["audio_token_logits"].shape == (
        2,
        1,
        heads.audio_generation_frames,
        heads.audio_generation_head.out_features,
    )


def test_audio_generation_head_rejects_bypassed_multi_codebook_setting() -> (
    None
):
    tokenizer = AudioTokenizerSettings(
        enabled=True,
        codec="discrete",
    ).model_copy(update={"n_codebooks": 2})

    with pytest.raises(ValueError, match="exactly one tokenizer codebook"):
        ProjectionHeads(config=ModelSettings(audio_tokenizer=tokenizer))


def test_audio_tokenizer_handles_short_and_single_sample_frames() -> None:
    short_frame_tokenizer = AudioTokenizer(
        sample_rate=1_000,
        frame_ms=20,
        hop_ms=20,
    )
    short = short_frame_tokenizer.encode(torch.zeros((1, 1, 3)))

    assert short.tokens.shape == (1, 1, 2)
    assert torch.isfinite(short.tokens).all()

    one_sample_tokenizer = AudioTokenizer(
        sample_rate=1,
        frame_ms=1,
        hop_ms=1,
    )
    one_sample = one_sample_tokenizer.encode(torch.zeros((1, 1, 1)))

    assert one_sample.tokens.shape == (1, 1, 2)
    assert torch.equal(one_sample.tokens[..., 1], torch.zeros((1, 1)))
    assert torch.isfinite(one_sample.tokens).all()


def test_audio_codec_uses_symmetric_configured_strides_and_st_gradients() -> (
    None
):
    config = AudioCodecSettings(
        latent_dim=4,
        downsample_factor=12,
        hidden_dim=16,
        n_codebooks=2,
        codebook_size=8,
        token_dim=4,
    )
    codec = ScratchAudioCodec(config=config)

    assert prod(codec.encoder.stride_factors) == config.downsample_factor
    assert codec.decoder.stride_factors == tuple(
        reversed(codec.encoder.stride_factors)
    )

    waveform = torch.randn((1, 1, 12))
    reconstructed, details = codec(waveform)

    assert reconstructed.shape == waveform.shape
    assert details["tokens"].shape[-1] == config.n_codebooks

    quantizer = ResidualVectorQuantizer(config=config)
    latents = torch.randn((2, config.latent_dim, 3), requires_grad=True)
    quantized, *_ = quantizer(latents)
    quantized.square().sum().backward()

    assert latents.grad is not None
    assert torch.count_nonzero(latents.grad).item() > 0


def _dense_model_settings() -> ModelSettings:
    encoder = EncoderSettings(
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        dropout=0.0,
    )
    return ModelSettings(
        text=encoder,
        document=encoder,
        image=encoder,
        audio=encoder,
        video=encoder,
        fusion_dim=8,
        projection_dim=8,
        raw_text_vocab_size=269,
        raw_text_max_tokens=12,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        enabled_modalities=("text",),
        output_modalities=("text",),
        text_decoder=DecoderSettings(
            enabled=True,
            vocab_size=269,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_target_tokens=12,
            max_context_tokens=8,
            max_text_context_tokens=8,
            max_document_context_tokens=0,
            max_image_context_tokens=0,
            max_audio_context_tokens=0,
            max_video_context_tokens=0,
        ),
    )


def test_preference_router_runs_exactly_chosen_and_rejected_decoder_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MultimodalModel(
        _dense_model_settings(),
        training_backend="dense_transformer",
    ).eval()
    decoder = model.dense_decoder
    assert decoder is not None

    chosen = torch.tensor([[2, 9, 10, 3]], dtype=torch.long)
    rejected = torch.tensor([[2, 9, 11, 3]], dtype=torch.long)
    labels = torch.tensor(
        [[IGNORE_LABEL, IGNORE_LABEL, 10, 3]],
        dtype=torch.long,
    )
    batch = CollatedBatch(
        sample_ids=["preference"],
        text=torch.tensor([[2, 9, 10, 3]], dtype=torch.long),
        image=torch.zeros((1, 8)),
        audio=torch.zeros((1, 8)),
        video=torch.zeros((1, 8)),
        modality_mask=torch.tensor([[True, False, False, False, False]]),
        labels=None,
        task_types=["instruction_following"],
        text_mask=torch.tensor([True]),
        document_mask=torch.tensor([False]),
        image_mask=torch.tensor([False]),
        audio_mask=torch.tensor([False]),
        video_mask=torch.tensor([False]),
        decoder_input_ids=chosen,
        decoder_labels=labels,
        decoder_attention_mask=torch.ones_like(chosen, dtype=torch.bool),
        chosen_input_ids=chosen,
        chosen_labels=labels,
        chosen_attention_mask=torch.ones_like(chosen, dtype=torch.bool),
        rejected_input_ids=rejected,
        rejected_labels=labels,
        rejected_attention_mask=torch.ones_like(rejected, dtype=torch.bool),
    )
    calls: list[torch.Tensor] = []
    original_forward = decoder.forward

    def tracked_forward(**kwargs: torch.Tensor) -> tuple[object, object]:
        calls.append(kwargs["input_ids"].detach().clone())
        return original_forward(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(decoder, "forward", tracked_forward)
    outputs = model(batch, output_heads={"sequence"})

    assert len(calls) == 2
    assert torch.equal(calls[0], chosen)
    assert torch.equal(calls[1], rejected)
    assert torch.equal(
        outputs["sequence_logits"], outputs["chosen_sequence_logits"]
    )
    assert "rejected_sequence_logits" in outputs
