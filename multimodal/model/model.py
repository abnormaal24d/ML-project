"""Multimodal model composition, dense decoding, and generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Collection, Iterator

import torch
from torch import nn

from multimodal.model.contracts import CollatedBatch
from multimodal.model.encoders.composition import build_encoder_composition
from multimodal.model.forward import (
    DenseCausalDecoder,
    MultimodalForwardRouter,
)
from multimodal.model.fusion import build_fusion_composition
from multimodal.model.generation.image_codec import ScratchImageCodec
from multimodal.model.generation.image_diffusion import ImageLatentDiffusion
from multimodal.model.outputs.builders import (
    GenerationOutputBuilder,
    SupervisedOutputBuilder,
)
from multimodal.model.outputs.projection import build_projection_heads
from multimodal.tokenization.text import VocabularyTokenizer

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings


class MultimodalModel(nn.Module):
    """Shared modality encoders with smoke or dense autoregressive routing."""

    def __init__(
        self,
        config: ModelSettings,
        *,
        training_backend: str = "pipeline_smoke",
    ) -> None:
        super().__init__()
        self.config = config
        self.training_backend = str(training_backend)
        if self.training_backend not in {
            "pipeline_smoke",
            "dense_transformer",
        }:
            raise ValueError(
                f"unsupported MultimodalModel backend: {self.training_backend!r}"
            )
        self.enabled_modalities = tuple(config.enabled_modalities)
        self.encoder_composition = build_encoder_composition(
            config=config,
            enabled_modalities=self.enabled_modalities,
        )
        self.fusion_composition = build_fusion_composition(
            config=config,
            enabled_modalities=self.enabled_modalities,
        )
        self.projection_heads = build_projection_heads(config=config)

        # Generation components (Step 6)
        self.image_codec: ScratchImageCodec | None = None
        self.image_diffusion: ImageLatentDiffusion | None = None
        if config.image_generator.enabled:
            self.image_codec = ScratchImageCodec(config=config.image_codec)
            self.image_diffusion = ImageLatentDiffusion(
                config=config.image_generator,
                latent_channels=config.image_codec.latent_channels,
                conditioning_dim=config.fusion_dim,
            )

        dense_decoder: DenseCausalDecoder | None = None
        if self.training_backend == "dense_transformer":
            dense_decoder = DenseCausalDecoder(config=config)
        self.forward_router = MultimodalForwardRouter(
            config=config,
            enabled_modalities=self.enabled_modalities,
            encoders=self.encoder_composition,
            fusion=self.fusion_composition,
            projection_heads=self.projection_heads,
            loss_output_builder=SupervisedOutputBuilder(
                heads=self.projection_heads,
                enabled_modalities=self.enabled_modalities,
            ),
            generation_output_builder=GenerationOutputBuilder(
                heads=self.projection_heads,
            ),
            dense_decoder=dense_decoder,
        )

    @property
    def dense_decoder(self) -> DenseCausalDecoder | None:
        return self.forward_router.dense_decoder

    @property
    def encoders(self) -> nn.ModuleDict:
        return self.encoder_composition.encoders

    @property
    def fusion(self) -> nn.Module:
        return self.fusion_composition.fusion

    def forward(
        self,
        batch: CollatedBatch,
        *,
        active_modalities: Collection[str] | None = None,
        output_heads: Collection[str] | str | None = None,
        max_active_modalities: int | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_router.forward(
            batch,
            active_modalities=active_modalities,
            output_heads=output_heads,
            max_active_modalities=max_active_modalities,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: CollatedBatch,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        use_kv_cache: bool = True,
        cancelled: Callable[[], bool] | None = None,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """Generate answer tokens from prompts ending in ``<assistant>``."""

        steps = list(
            self.generate_stream(
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                use_kv_cache=use_kv_cache,
                cancelled=cancelled,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        )
        if not steps:
            return torch.empty(
                len(batch.sample_ids),
                0,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        return torch.stack(steps, dim=1)

    @torch.no_grad()
    def generate_stream(
        self,
        batch: CollatedBatch,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        use_kv_cache: bool = True,
        cancelled: Callable[[], bool] | None = None,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> Iterator[torch.Tensor]:
        """Yield one generated token per batch row at each decoding step."""

        decoder = self.dense_decoder
        if decoder is None:
            raise RuntimeError(
                "generate requires training_backend='dense_transformer'"
            )
        input_ids = batch.decoder_input_ids
        attention_mask = batch.decoder_attention_mask
        if input_ids is None or attention_mask is None:
            raise ValueError("generation requires decoder prompt tensors")
        device = next(self.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        prompt_counts = _prompt_lengths(
            batch=batch, attention_mask=attention_mask
        )
        if any(count <= 0 for count in prompt_counts):
            raise ValueError("every generated row requires a non-empty prompt")
        maximum_prompt = max(prompt_counts)
        prompt_ids = input_ids[:, :maximum_prompt]
        positions = torch.arange(maximum_prompt, device=device).unsqueeze(0)
        prompt_mask = positions < torch.tensor(
            prompt_counts, device=device, dtype=torch.long
        ).unsqueeze(1)
        context = self.forward_router.encode_context(batch)
        output, cache = decoder(
            context=context,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            return_cache=True,
        )
        if cache is None:
            raise RuntimeError(
                "dense decoder did not return a generation cache"
            )
        row_indices = torch.arange(prompt_ids.shape[0], device=device)
        last_indices = torch.tensor(
            prompt_counts, device=device, dtype=torch.long
        ).sub(1)
        next_logits = output.logits[row_indices, last_indices]
        settings = self.config.generation
        resolved_max_new = int(max_new_tokens or settings.max_new_tokens)
        resolved_temperature = (
            float(settings.temperature)
            if temperature is None
            else float(temperature)
        )
        resolved_top_p = (
            float(settings.top_p) if top_p is None else float(top_p)
        )
        resolved_top_k = int(settings.top_k) if top_k is None else int(top_k)
        resolved_repetition_penalty = (
            float(settings.repetition_penalty)
            if repetition_penalty is None
            else float(repetition_penalty)
        )
        resolved_no_repeat_ngram = (
            int(settings.no_repeat_ngram_size)
            if no_repeat_ngram_size is None
            else int(no_repeat_ngram_size)
        )
        if resolved_repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        if resolved_no_repeat_ngram < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        active = torch.ones(
            prompt_ids.shape[0], dtype=torch.bool, device=device
        )
        generated_tokens: list[torch.Tensor] = []
        uncached_ids = prompt_ids
        uncached_mask = prompt_mask

        for _ in range(resolved_max_new):
            if cancelled is not None and cancelled():
                break
            controlled_logits = _apply_repetition_controls(
                logits=next_logits,
                generated_tokens=generated_tokens,
                repetition_penalty=resolved_repetition_penalty,
                no_repeat_ngram_size=resolved_no_repeat_ngram,
            )
            sampled = _sample_next_token(
                logits=controlled_logits,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                top_k=resolved_top_k,
            )
            sampled = torch.where(
                active,
                sampled,
                torch.full_like(sampled, int(pad_token_id)),
            )
            yield sampled
            generated_tokens.append(sampled)
            step_valid = active
            if use_kv_cache:
                next_logits, cache = decoder.decode_step(
                    token_ids=sampled,
                    cache=cache,
                    step_valid_mask=step_valid,
                )
            else:
                uncached_ids = torch.cat(
                    (uncached_ids, sampled.unsqueeze(1)), dim=1
                )
                uncached_mask = torch.cat(
                    (uncached_mask, step_valid.unsqueeze(1)), dim=1
                )
                uncached_output, _unused_cache = decoder(
                    context=context,
                    input_ids=uncached_ids,
                    attention_mask=uncached_mask,
                    return_cache=False,
                )
                next_logits = uncached_output.logits[:, -1, :]
            active = active & sampled.ne(int(eos_token_id))
            if not bool(active.any().item()):
                break

    @torch.no_grad()
    def generate_image(
        self,
        prompts: list[str],
        *,
        tokenizer: VocabularyTokenizer,
        num_inference_steps: int | None = None,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        """Generate images from text prompts using latent diffusion.

        Args:
            prompts: List of text prompts
            tokenizer: Canonical tokenizer loaded from the training artifact
            num_inference_steps: Number of denoising steps (default: config.image_steps)
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            device: Device to run generation on

        Returns:
            List of generated images as tensors [3, H, W] in range [-1, 1]
        """
        if self.image_codec is None or self.image_diffusion is None:
            raise RuntimeError(
                "Image generation components not initialized. Enable "
                "image_generator in config."
            )
        if not prompts:
            return []
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("image generation prompts must be strings")
        if len(tokenizer.token_to_id) != self.config.raw_text_vocab_size:
            raise ValueError(
                "image generation tokenizer vocabulary must match "
                "raw_text_vocab_size"
            )
        if tokenizer.max_tokens != self.config.raw_text_max_tokens:
            raise ValueError(
                "image generation tokenizer max_tokens must match "
                "raw_text_max_tokens"
            )
        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and non-negative")

        model_device = next(self.parameters()).device
        resolved_device = (
            model_device if device is None else torch.device(device)
        )
        if resolved_device != model_device:
            raise ValueError(
                "image generation device must match the model device; move "
                "the model before generating"
            )

        resolved_steps = int(
            num_inference_steps or self.config.generation.image_steps
        )
        if resolved_steps <= 0:
            raise ValueError("num_inference_steps must be greater than zero")
        if resolved_steps > self.image_diffusion.num_train_timesteps:
            raise ValueError(
                "num_inference_steps cannot exceed the diffusion training "
                "schedule"
            )
        if "text" not in self.encoders:
            raise RuntimeError(
                "image generation requires the configured text encoder"
            )

        was_training = self.training
        self.eval()
        try:
            prompt_token_ids = torch.tensor(
                [
                    tokenizer.encode(
                        prompt,
                        add_special_tokens=True,
                        pad_to_max_length=True,
                    )
                    for prompt in prompts
                ],
                dtype=torch.long,
                device=resolved_device,
            )
            text_outputs = self.encoders["text"](prompt_token_ids)
            text_embeddings = text_outputs["embedding"]

            codec_encoder = self.image_codec.encoder
            latent_side = codec_encoder.latent_resolution
            codec_dtype = next(self.image_codec.parameters()).dtype
            generator = torch.Generator(device=resolved_device)
            if seed is not None:
                generator.manual_seed(seed)
            latents = torch.randn(
                (
                    len(prompts),
                    codec_encoder.latent_channels,
                    latent_side,
                    latent_side,
                ),
                device=resolved_device,
                dtype=codec_dtype,
                generator=generator,
            )
            timesteps = (
                torch.linspace(
                    self.image_diffusion.num_train_timesteps - 1,
                    0,
                    resolved_steps,
                    device=resolved_device,
                )
                .round()
                .to(dtype=torch.long)
            )

            for index, timestep_value in enumerate(timesteps):
                timestep = torch.full(
                    (len(prompts),),
                    int(timestep_value.item()),
                    device=resolved_device,
                    dtype=torch.long,
                )
                conditional_noise = self.image_diffusion.predict_noise(
                    latents=latents,
                    text_embedding=text_embeddings,
                    timestep=timestep,
                )
                unconditional_noise = self.image_diffusion.predict_noise(
                    latents=latents,
                    text_embedding=torch.zeros_like(text_embeddings),
                    timestep=timestep,
                )
                model_prediction = unconditional_noise + guidance_scale * (
                    conditional_noise - unconditional_noise
                )
                if index + 1 < len(timesteps):
                    alpha_prod_t_prev = self.image_diffusion.alphas_cumprod[
                        timesteps[index + 1]
                    ].to(device=resolved_device, dtype=latents.dtype)
                else:
                    alpha_prod_t_prev = torch.ones(
                        (),
                        device=resolved_device,
                        dtype=latents.dtype,
                    )
                pred_original, noise_prediction = (
                    self.image_diffusion.prediction_to_original_and_noise(
                        model_prediction=model_prediction,
                        noisy_latents=latents,
                        timestep=timestep,
                    )
                )
                latents = (
                    torch.sqrt(alpha_prod_t_prev) * pred_original
                    + torch.sqrt((1.0 - alpha_prod_t_prev).clamp_min(0.0))
                    * noise_prediction
                )

            images = self.image_codec.decode(latents)
            return [image for image in images]
        finally:
            self.train(was_training)


def _prompt_lengths(
    *, batch: CollatedBatch, attention_mask: torch.Tensor
) -> list[int]:
    if len(batch.prompt_token_count) == attention_mask.shape[0]:
        return [int(value) for value in batch.prompt_token_count]
    return [int(value) for value in attention_mask.sum(dim=1).tolist()]


def _apply_repetition_controls(
    *,
    logits: torch.Tensor,
    generated_tokens: list[torch.Tensor],
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    if not generated_tokens:
        return logits
    controlled = logits.clone()
    generated = torch.stack(generated_tokens, dim=1)
    if repetition_penalty > 1.0:
        for row in range(generated.shape[0]):
            token_ids = torch.unique(generated[row])
            values = controlled[row, token_ids]
            controlled[row, token_ids] = torch.where(
                values < 0,
                values * repetition_penalty,
                values / repetition_penalty,
            )
    if no_repeat_ngram_size > 0:
        for row in range(generated.shape[0]):
            banned = _banned_ngram_tokens(
                generated[row].tolist(), ngram_size=no_repeat_ngram_size
            )
            if banned:
                controlled[
                    row, torch.tensor(banned, device=controlled.device)
                ] = float("-inf")
    return controlled


def _banned_ngram_tokens(
    generated: list[int],
    *,
    ngram_size: int,
) -> tuple[int, ...]:
    if ngram_size <= 0 or len(generated) + 1 < ngram_size:
        return ()
    if ngram_size == 1:
        return tuple(sorted(set(generated)))
    prefix = tuple(generated[-(ngram_size - 1) :])
    banned: set[int] = set()
    for index in range(len(generated) - ngram_size + 1):
        candidate_prefix = tuple(generated[index : index + ngram_size - 1])
        if candidate_prefix == prefix:
            banned.add(generated[index + ngram_size - 1])
    return tuple(sorted(banned))


def _sample_next_token(
    *,
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    scores = logits / max(temperature, 1e-6)
    if top_k > 0:
        keep = min(top_k, scores.shape[-1])
        threshold = scores.topk(keep, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(
            scores, descending=True, dim=-1
        )
        sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(-1, sorted_indices, sorted_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
