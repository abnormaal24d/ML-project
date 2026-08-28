from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config.multimodal.encoder_settings import EncoderSettings
from config.multimodal.generation_settings import (
    ImageCodecSettings,
    ImageGeneratorSettings,
    VideoGeneratorSettings,
)
from config.multimodal.model_settings import ModelSettings
from mmcrawler_datasets.collation.multimodal import MultimodalCollator
from mmcrawler_datasets.schema import MultimodalSample
from multimodal.model.generation.image_codec import ScratchImageCodec
from multimodal.model.generation.image_diffusion import ImageLatentDiffusion
from multimodal.model.model import MultimodalModel
from multimodal.model.outputs.builders import GenerationOutputBuilder
from multimodal.model.outputs.projection import ProjectionHeads
from multimodal.tokenization.merges import symbol_key
from multimodal.tokenization.text import VocabularyTokenizer


def _image_codec_settings() -> ImageCodecSettings:
    return ImageCodecSettings(
        latent_channels=2,
        downsample_factor=4,
        hidden_dim=16,
        input_resolution=16,
        upsample_blocks=2,
    )


def _tokenizer() -> VocabularyTokenizer:
    special_tokens = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<mask>": 4,
        "<system>": 5,
        "<user>": 6,
        "<assistant>": 7,
        "<tool>": 8,
    }
    byte_tokens = {
        symbol_key(bytes((value,))): value + len(special_tokens)
        for value in range(256)
    }
    token_to_id = {**special_tokens, **byte_tokens}
    return VocabularyTokenizer(
        token_to_id=token_to_id,
        id_to_token={
            token_id: token for token, token_id in token_to_id.items()
        },
        max_tokens=16,
        token_bytes={
            value + len(special_tokens): bytes((value,))
            for value in range(256)
        },
        merges=(),
    )


def _model_settings() -> ModelSettings:
    text = EncoderSettings(
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        dropout=0.0,
    )
    return ModelSettings(
        fusion_dim=8,
        projection_dim=8,
        raw_text_vocab_size=265,
        raw_text_max_tokens=16,
        raw_image_size=16,
        raw_video_frames=2,
        enabled_modalities=("text",),
        output_modalities=("image",),
        text=text,
        image_generator=ImageGeneratorSettings(
            enabled=True,
            hidden_dim=8,
            num_layers=1,
            num_heads=1,
            num_train_timesteps=4,
        ),
        image_codec=_image_codec_settings(),
    )


def test_codec_and_diffusion_share_configured_nchw_shape() -> None:
    codec = ScratchImageCodec(config=_image_codec_settings()).eval()
    images = torch.randn(1, 3, 16, 16)
    latents = codec.encode(images).latents
    assert latents.shape == (1, 2, 4, 4)
    assert codec.decode(latents).shape == images.shape

    diffusion = ImageLatentDiffusion(
        config=ImageGeneratorSettings(
            hidden_dim=8,
            num_layers=1,
            num_heads=1,
            num_train_timesteps=4,
        ),
        latent_channels=2,
        conditioning_dim=8,
    ).eval()
    timestep = torch.tensor([1], dtype=torch.long)
    text_embedding = torch.randn(1, 8)
    first = diffusion.predict_noise(
        latents=latents,
        text_embedding=text_embedding,
        timestep=timestep,
    )
    second = diffusion.predict_noise(
        latents=latents,
        text_embedding=text_embedding,
        timestep=timestep,
    )

    assert first.shape == latents.shape
    assert torch.equal(first, second)


def test_diffusion_prediction_is_pure_at_multiple_nchw_resolutions() -> None:
    diffusion = ImageLatentDiffusion(
        config=ImageGeneratorSettings(
            hidden_dim=8,
            num_layers=1,
            num_heads=1,
            num_train_timesteps=4,
        ),
        latent_channels=2,
        conditioning_dim=8,
    ).eval()
    timestep = torch.tensor([1], dtype=torch.long)
    condition = torch.randn(1, 8)

    for height, width in ((4, 4), (3, 5)):
        clean = torch.randn(1, 2, height, width)
        noisy = diffusion.q_sample(
            latents=clean,
            timestep=timestep,
            noise=torch.ones_like(clean),
        )
        first = diffusion.predict_noise(
            latents=noisy,
            text_embedding=condition,
            timestep=timestep,
        )
        second = diffusion.predict_noise(
            latents=noisy,
            text_embedding=condition,
            timestep=timestep,
        )

        assert first.shape == clean.shape
        assert torch.equal(first, second)


@pytest.mark.parametrize(
    "prediction_type", ("epsilon", "v_prediction", "sample")
)
def test_diffusion_prediction_types_share_training_and_sampling_contract(
    prediction_type: str,
) -> None:
    diffusion = ImageLatentDiffusion(
        config=ImageGeneratorSettings(
            hidden_dim=8,
            num_layers=1,
            num_heads=1,
            num_train_timesteps=4,
            prediction_type=prediction_type,  # type: ignore[arg-type]
        ),
        latent_channels=2,
        conditioning_dim=8,
    ).eval()
    clean = torch.randn((1, 2, 3, 5))
    noise = torch.randn_like(clean)
    timestep = torch.tensor([2], dtype=torch.long)
    noisy = diffusion.q_sample(
        latents=clean,
        timestep=timestep,
        noise=noise,
    )

    target = diffusion.training_target(
        latents=clean,
        noise=noise,
        timestep=timestep,
    )
    recovered_clean, recovered_noise = (
        diffusion.prediction_to_original_and_noise(
            model_prediction=target,
            noisy_latents=noisy,
            timestep=timestep,
        )
    )
    output = diffusion(
        clean,
        torch.zeros((1, 8)),
        timestep=timestep,
        noise=noise,
    )

    torch.testing.assert_close(recovered_clean, clean)
    torch.testing.assert_close(recovered_noise, noise)
    assert output.target is not None
    torch.testing.assert_close(output.target, target)


def test_image_generation_uses_prompt_tokens_cfg_and_local_seed() -> None:
    model = MultimodalModel(_model_settings()).eval()
    tokenizer = _tokenizer()

    torch.manual_seed(73)
    expected_next_random = torch.rand(1)
    torch.manual_seed(73)
    first = model.generate_image(
        ["a red cat"],
        tokenizer=tokenizer,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=11,
    )
    observed_next_random = torch.rand(1)
    same_seed = model.generate_image(
        ["a red cat"],
        tokenizer=tokenizer,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=11,
    )
    other_prompt = model.generate_image(
        ["a blue dog"],
        tokenizer=tokenizer,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=11,
    )
    unguided = model.generate_image(
        ["a red cat"],
        tokenizer=tokenizer,
        num_inference_steps=2,
        guidance_scale=0.0,
        seed=11,
    )

    assert first[0].shape == (3, 16, 16)
    assert torch.equal(first[0], same_seed[0])
    assert not torch.equal(first[0], other_prompt[0])
    assert not torch.equal(first[0], unguided[0])
    assert torch.equal(expected_next_random, observed_next_random)


def test_image_sampler_uses_previous_schedule_step_and_terminal_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MultimodalModel(_model_settings()).eval()
    tokenizer = _tokenizer()
    assert model.image_codec is not None
    assert model.image_diffusion is not None

    # Return the final latent directly so the DDIM update can be checked
    # independently from codec reconstruction weights.
    model.image_codec.decoder = torch.nn.Identity()
    alphas = torch.tensor((0.2, 0.4, 0.6, 0.8), dtype=torch.float32)
    model.image_diffusion.alphas_cumprod.copy_(alphas)

    def constant_noise(**kwargs: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(kwargs["latents"])

    monkeypatch.setattr(model.image_diffusion, "predict_noise", constant_noise)
    generated = model.generate_image(
        ["schedule check"],
        tokenizer=tokenizer,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=17,
    )[0].unsqueeze(0)

    generator = torch.Generator().manual_seed(17)
    initial = torch.randn(
        (1, 2, 4, 4),
        generator=generator,
        dtype=generated.dtype,
    )
    noise = torch.ones_like(initial)
    alpha_t = alphas[3]
    alpha_previous_step = alphas[0]
    first_original = (
        initial - torch.sqrt(1.0 - alpha_t) * noise
    ) / torch.sqrt(alpha_t)
    after_first = (
        torch.sqrt(alpha_previous_step) * first_original
        + torch.sqrt(1.0 - alpha_previous_step) * noise
    )
    expected_terminal = (
        after_first - torch.sqrt(1.0 - alpha_previous_step) * noise
    ) / torch.sqrt(alpha_previous_step)

    torch.testing.assert_close(generated, expected_terminal)


def test_projection_heads_use_generation_frames_and_codec_latent_channels() -> (
    None
):
    settings = _model_settings().model_copy(
        update={
            "video_generator": VideoGeneratorSettings(
                enabled=True,
                frames=8,
            )
        }
    )
    heads = ProjectionHeads(config=settings)

    assert heads.image_latent_head is not None
    assert heads.image_latent_head.out_features == 2
    assert heads.image_codec_decoder_head is not None
    assert heads.image_codec_decoder_head.out_features == 2 * 4 * 4
    assert heads.video_position_embedding is not None
    assert heads.video_position_embedding.num_embeddings == 8


def test_video_generation_targets_and_logits_follow_generation_frames(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw_video.pt"
    target_path = tmp_path / "target_video_tokens.pt"
    torch.save(torch.zeros((2, 3, 16, 16)), input_path)
    torch.save(
        torch.arange(32, dtype=torch.long).reshape(8, 2, 2), target_path
    )

    collator = MultimodalCollator(
        tokenizer=_tokenizer(),
        text_dim=8,
        image_dim=8,
        audio_dim=8,
        video_dim=8,
        raw_text_vocab_size=265,
        raw_text_max_tokens=16,
        raw_image_size=16,
        raw_audio_num_samples=512,
        raw_video_frames=2,
        video_generation_frames=8,
        audio_token_codec="discrete",
        training_backend="pipeline_smoke",
        mlm_probability=0.15,
        image_mask_probability=0.2,
        audio_mask_probability=0.15,
        materialized_dataset_root=tmp_path,
        materialized_tensors_enabled=True,
        base_seed=0,
    )
    batch = collator(
        (
            MultimodalSample(
                sample_id="video-generation",
                task_type="text_to_video",
                video_tensor_path=input_path.relative_to(tmp_path),
                target_video_tokens_path=target_path.relative_to(tmp_path),
            ),
        )
    )
    assert batch.video.shape == (1, 2, 3, 16, 16)
    assert batch.video_token_targets is not None
    assert batch.video_token_targets.shape == (1, 8, 2, 2)
    assert batch.video_token_attention_mask is not None
    assert batch.video_token_attention_mask.shape == (1, 8)
    assert batch.video_token_attention_mask.all()

    settings = _model_settings().model_copy(
        update={
            "video_generator": VideoGeneratorSettings(enabled=True, frames=8)
        }
    )
    heads = ProjectionHeads(config=settings)
    logits = GenerationOutputBuilder(heads=heads).build(
        fused=torch.zeros((1, heads.fusion_dim)),
        resolved_heads=frozenset({"video_generation"}),
    )["video_token_logits"]
    assert logits.shape[1] == batch.video_token_targets.shape[1] == 8
