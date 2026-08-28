"""Forward routing and the dense autoregressive decoder.

The smoke backend keeps pooled encoder/fusion execution.  The dense backend
reuses the same modality encoders for retrieval heads while adding a bounded
multimodal prefix and a causal text decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, cast

import torch
from torch import nn

from multimodal.model.contracts import (
    CollatedBatch,
    DenseDecoderOutput,
    ModalityTokenSequence,
)
from multimodal.model.outputs.batch import resolve_modality_row_masks
from multimodal.model.outputs.routing import resolve_output_heads

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from multimodal.model.encoders.composition import EncoderComposition
    from multimodal.model.fusion import FusedRepresentation, FusionComposition
    from multimodal.model.outputs.builders import (
        GenerationOutputBuilder,
        SupervisedOutputBuilder,
    )
    from multimodal.model.outputs.projection import (
        ProjectedModalities,
        ProjectionHeads,
    )


@dataclass(frozen=True, slots=True)
class DecoderLayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class DenseDecoderCache:
    layers: tuple[DecoderLayerCache, ...]
    key_valid_mask: torch.Tensor


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * scale * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding shared by query and key projections."""

    inverse_frequencies: torch.Tensor

    def __init__(self, *, head_dim: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("rotary head_dim must be even")
        frequencies = 1.0 / (
            float(base)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer(
            "inverse_frequencies", frequencies, persistent=False
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = position_ids.to(dtype=torch.float32).unsqueeze(-1)
        angles = angles * self.inverse_frequencies.view(1, 1, -1)
        cosine = angles.cos().to(dtype=query.dtype).unsqueeze(1)
        sine = angles.sin().to(dtype=query.dtype).unsqueeze(1)
        return (
            _rotate_half_pairs(query, cosine=cosine, sine=sine),
            _rotate_half_pairs(key, cosine=cosine, sine=sine),
        )


def _rotate_half_pairs(
    value: torch.Tensor,
    *,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head attention supporting full prefix-LM and cached decoding."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_rotary_embeddings: bool,
        rotary_base: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = (
            RotaryEmbedding(head_dim=self.head_dim, base=rotary_base)
            if use_rotary_embeddings
            else None
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        allowed_attention: torch.Tensor,
        valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        query, key, value = self._project(hidden, position_ids=position_ids)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        scores = scores.masked_fill(
            ~allowed_attention[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        scores = scores.masked_fill(
            ~valid_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        attended = torch.matmul(self.dropout(weights), value)
        output = self.output(self._merge_heads(attended))
        output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output, DecoderLayerCache(key=key, value=value)

    def step(
        self,
        hidden: torch.Tensor,
        *,
        cache: DecoderLayerCache,
        key_valid_mask: torch.Tensor,
        step_valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        query, key, value = self._project(hidden, position_ids=position_ids)
        key = torch.cat((cache.key, key), dim=2)
        value = torch.cat((cache.value, value), dim=2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        scores = scores.masked_fill(
            ~key_valid_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        attended = torch.matmul(weights, value)
        output = self.output(self._merge_heads(attended))
        output = output * step_valid_mask[:, None, None].to(output.dtype)
        return output, DecoderLayerCache(key=key, value=value)

    def _project(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, hidden_dim = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query_heads = split_heads(query)
        key_heads = split_heads(key)
        if self.rotary is not None:
            query_heads, key_heads = self.rotary(
                query_heads,
                key_heads,
                position_ids=position_ids,
            )
        return query_heads, key_heads, split_heads(value)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, length, _head_dim = tensor.shape
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch, length, self.num_heads * self.head_dim)
        )


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * 4
        self.gate = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.value = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.output = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout.forward(
            self.output.forward(
                torch.nn.functional.silu(self.gate.forward(hidden))
                * self.value.forward(hidden)
            )
        )


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal decoder block with cached attention support."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_rotary_embeddings: bool,
        rotary_base: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim)
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary_embeddings=use_rotary_embeddings,
            rotary_base=rotary_base,
        )
        self.feed_forward_norm = RMSNorm(hidden_dim)
        self.feed_forward = SwiGLU(hidden_dim=hidden_dim, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        allowed_attention: torch.Tensor,
        valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        attended, cache = self.attention(
            self.attention_norm(hidden),
            allowed_attention=allowed_attention,
            valid_mask=valid_mask,
            position_ids=position_ids,
        )
        hidden = hidden + self.residual_dropout(attended)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, cache

    def step(
        self,
        hidden: torch.Tensor,
        *,
        cache: DecoderLayerCache,
        key_valid_mask: torch.Tensor,
        step_valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        attended, updated_cache = self.attention.step(
            self.attention_norm(hidden),
            cache=cache,
            key_valid_mask=key_valid_mask,
            step_valid_mask=step_valid_mask,
            position_ids=position_ids,
        )
        hidden = hidden + self.residual_dropout(attended)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        hidden = hidden * step_valid_mask[:, None, None].to(hidden.dtype)
        return hidden, updated_cache


class DenseCausalDecoder(nn.Module):
    """Scratch causal decoder over a multimodal prefix sequence."""

    def __init__(
        self,
        *,
        config: "ModelSettings",
    ) -> None:
        super().__init__()
        settings = config.text_decoder
        if not settings.enabled:
            raise ValueError(
                "dense_transformer requires text_decoder.enabled=true"
            )
        self.vocab_size = int(settings.vocab_size)
        self.hidden_dim = int(settings.hidden_dim)
        self.max_target_tokens = int(settings.max_target_tokens)
        self.max_position_embeddings = (
            int(settings.max_context_tokens)
            + self.max_target_tokens
            + int(config.generation.max_new_tokens)
        )
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.use_rotary_embeddings = bool(settings.use_rotary_embeddings)
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_dim,
            padding_idx=0,
        )
        self.position_embedding = (
            None
            if self.use_rotary_embeddings
            else nn.Embedding(
                self.max_position_embeddings,
                self.hidden_dim,
            )
        )
        self.input_dropout = nn.Dropout(settings.dropout)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=settings.num_heads,
                dropout=settings.dropout,
                use_rotary_embeddings=self.use_rotary_embeddings,
                rotary_base=float(settings.rotary_base),
            )
            for _ in range(settings.num_layers)
        )
        self.norm = RMSNorm(self.hidden_dim)
        self.language_model_head = nn.Linear(
            self.hidden_dim, self.vocab_size, bias=False
        )
        if settings.tie_input_output_embeddings:
            self.language_model_head.weight = self.token_embedding.weight

    def forward(
        self,
        *,
        context: ModalityTokenSequence,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_cache: bool = False,
    ) -> tuple[DenseDecoderOutput, DenseDecoderCache | None]:
        input_ids, attention_mask = self._validate_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        context.validate()
        if context.tokens.shape[0] != input_ids.shape[0]:
            raise ValueError("context and decoder batch sizes must match")
        if context.tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "context token dimension must match decoder hidden_dim"
            )

        combined_mask = torch.cat(
            (
                context.attention_mask.to(input_ids.device),
                attention_mask,
            ),
            dim=1,
        )
        if combined_mask.shape[1] > self.max_position_embeddings:
            raise ValueError(
                "dense decoder context plus input exceeds positional capacity: "
                f"length={combined_mask.shape[1]}, "
                f"maximum={self.max_position_embeddings}"
            )
        position_ids = _position_ids(
            combined_mask, self.max_position_embeddings
        )
        decoder_tokens = self.token_embedding(input_ids)
        hidden = torch.cat(
            (context.tokens.to(decoder_tokens.device), decoder_tokens), dim=1
        )
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(position_ids)
        hidden = self.input_dropout(hidden)
        hidden = hidden * combined_mask.unsqueeze(-1).to(hidden.dtype)
        context_length = context.tokens.shape[1]
        allowed = _prefix_lm_attention_pattern(
            context_length=context_length,
            decoder_length=input_ids.shape[1],
            device=hidden.device,
        )
        layer_caches: list[DecoderLayerCache] = []
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and not return_cache
            ):
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda value, current_block=block: current_block(
                        value,
                        allowed_attention=allowed,
                        valid_mask=combined_mask,
                        position_ids=position_ids,
                    )[0],
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden, cache = block(
                    hidden,
                    allowed_attention=allowed,
                    valid_mask=combined_mask,
                    position_ids=position_ids,
                )
                layer_caches.append(cache)
        hidden = self.norm(hidden)
        decoder_hidden = hidden[:, context_length:, :]
        logits = self.language_model_head(decoder_hidden)
        result = DenseDecoderOutput(
            hidden_states=decoder_hidden, logits=logits
        )
        result.validate()
        cache = None
        if return_cache:
            cache = DenseDecoderCache(
                layers=tuple(layer_caches),
                key_valid_mask=combined_mask,
            )
        return result, cache

    def decode_step(
        self,
        *,
        token_ids: torch.Tensor,
        cache: DenseDecoderCache,
        step_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, DenseDecoderCache]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_step token_ids must be [batch, 1]")
        step_valid_mask = step_valid_mask.to(
            device=token_ids.device, dtype=torch.bool
        )
        if cache.key_valid_mask.shape[1] >= self.max_position_embeddings:
            raise ValueError(
                "dense decoder generation exceeds positional capacity"
            )
        next_key_mask = torch.cat(
            (
                cache.key_valid_mask.to(token_ids.device),
                step_valid_mask.unsqueeze(1),
            ),
            dim=1,
        )
        position_ids = (
            cache.key_valid_mask.to(token_ids.device)
            .sum(dim=1, keepdim=True)
            .clamp_max(self.max_position_embeddings - 1)
        )
        hidden = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(position_ids)
        hidden = hidden * step_valid_mask[:, None, None].to(hidden.dtype)
        updated: list[DecoderLayerCache] = []
        if len(cache.layers) != len(self.blocks):
            raise ValueError("decoder cache layer count does not match model")
        for block, layer_cache in zip(self.blocks, cache.layers, strict=True):
            typed_block = cast(CausalTransformerBlock, block)
            hidden, layer_cache = typed_block.step(
                hidden,
                cache=layer_cache,
                key_valid_mask=next_key_mask,
                step_valid_mask=step_valid_mask,
                position_ids=position_ids,
            )
            updated.append(layer_cache)
        hidden = self.norm(hidden)
        logits = self.language_model_head(hidden[:, 0, :])
        return logits, DenseDecoderCache(
            layers=tuple(updated),
            key_valid_mask=next_key_mask,
        )

    def _validate_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("decoder input_ids must be [batch, tokens]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("decoder attention mask must match input_ids")
        if input_ids.shape[1] > self.max_target_tokens:
            raise ValueError("decoder input exceeds max_target_tokens")
        input_ids = input_ids.to(
            device=self.token_embedding.weight.device, dtype=torch.long
        )
        attention_mask = attention_mask.to(
            device=input_ids.device, dtype=torch.bool
        )
        invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
        if invalid.any():
            raise ValueError(
                "decoder input contains token ids outside vocabulary"
            )
        return input_ids, attention_mask


class TaskOutputRouter:
    def __init__(
        self,
        *,
        loss_output_builder: "SupervisedOutputBuilder",
        generation_output_builder: "GenerationOutputBuilder",
    ) -> None:
        self._loss_outputs = loss_output_builder
        self._generation_outputs = generation_output_builder

    def route(
        self,
        batch: "CollatedBatch",
        projected: "ProjectedModalities",
        fused: "FusedRepresentation",
        *,
        resolved_heads: frozenset[str],
        dense_output: DenseDecoderOutput | None = None,
        dense_context: ModalityTokenSequence | None = None,
        preference_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {
            "fused": fused.fused,
            "fused_embedding": fused.fused,
            "fusion_weights": fused.fusion_weights,
        }

        outputs.update(
            self._loss_outputs.build(
                fused=fused.fused,
                resolved_heads=resolved_heads,
                encoded_by_modality=projected.encoded_by_modality,
                batch_size=len(batch.sample_ids),
                batch=batch,
            )
        )
        if preference_outputs is not None:
            outputs.update(preference_outputs)
        outputs.update(
            self._generation_outputs.build(
                fused=fused.fused,
                resolved_heads=resolved_heads,
                decoder_hidden_states=(
                    dense_output.hidden_states
                    if dense_output is not None
                    else None
                ),
                sequence_logits=(
                    dense_output.logits if dense_output is not None else None
                ),
            )
        )
        if dense_context is not None:
            outputs["context_attention_mask"] = dense_context.attention_mask
            outputs["context_modality_ids"] = dense_context.modality_ids
            if dense_context.temporal_positions is not None:
                outputs["context_temporal_positions"] = (
                    dense_context.temporal_positions
                )
            if dense_context.spatial_positions is not None:
                outputs["context_spatial_positions"] = (
                    dense_context.spatial_positions
                )
            if dense_context.separator_mask is not None:
                outputs["context_separator_mask"] = (
                    dense_context.separator_mask
                )

        if projected.active_names:
            outputs["active_modality_mask"] = torch.stack(
                [mask.to(fused.fused.device) for mask in fused.modality_masks],
                dim=1,
            )

        return outputs


class MultimodalForwardRouter(nn.Module):
    def __init__(
        self,
        *,
        config: "ModelSettings",
        enabled_modalities: tuple[str, ...],
        encoders: "EncoderComposition",
        fusion: "FusionComposition",
        projection_heads: "ProjectionHeads",
        loss_output_builder: "SupervisedOutputBuilder",
        generation_output_builder: "GenerationOutputBuilder",
        dense_decoder: DenseCausalDecoder | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._enabled_modalities = enabled_modalities
        from multimodal.model.encoders.runner import ModalityEncoderRunner

        encoder_runner = ModalityEncoderRunner(
            config=config,
            enabled_modalities=enabled_modalities,
            encoders=encoders,
            document_fallback_projection=(
                projection_heads.document_fallback_projection
            ),
        )
        self._encoder_runner = encoder_runner
        from multimodal.model.outputs.projection import ProjectionRunner

        self._projection_runner = ProjectionRunner(
            enabled_modalities=enabled_modalities,
            projection_heads=projection_heads,
            encoder_runner=encoder_runner,
        )
        from multimodal.model.fusion import FusionRunner

        self._fusion_runner = FusionRunner(config=config, fusion=fusion)
        self._task_output_router = TaskOutputRouter(
            loss_output_builder=loss_output_builder,
            generation_output_builder=generation_output_builder,
        )
        self.dense_decoder = dense_decoder

    def forward(
        self,
        batch: "CollatedBatch",
        *,
        active_modalities: Collection[str] | None = None,
        output_heads: Collection[str] | str | None = None,
        max_active_modalities: int | None = None,
    ) -> dict[str, torch.Tensor]:
        resolved_heads = resolve_output_heads(
            batch=batch,
            config=self._config,
            requested=output_heads,
        )
        modality_row_masks = resolve_modality_row_masks(
            batch=batch,
            config=self._config,
            enabled_modalities=self._enabled_modalities,
            requested_modalities=active_modalities,
        )
        encoded_by_modality = self._encoder_runner.encode(
            batch,
            resolved_heads=resolved_heads,
            modality_row_masks=modality_row_masks,
        )
        projected = self._projection_runner.project(
            batch,
            encoded_by_modality=encoded_by_modality,
            modality_row_masks=modality_row_masks,
        )
        fused = self._fusion_runner.fuse(
            projected,
            batch,
            max_active_modalities,
        )
        dense_output: DenseDecoderOutput | None = None
        dense_context: ModalityTokenSequence | None = None
        preference_outputs: dict[str, torch.Tensor] | None = None
        if self.dense_decoder is not None and "sequence" in resolved_heads:
            task_types = tuple(batch.task_types) if batch.task_types else None
            dense_context = self._fusion_runner.context_sequence(
                projected=projected,
                batch=batch,
                task_types=task_types,
            )
            chosen_input_ids = batch.chosen_input_ids
            chosen_attention_mask = batch.chosen_attention_mask
            rejected_input_ids = batch.rejected_input_ids
            rejected_attention_mask = batch.rejected_attention_mask
            preference_values = (
                chosen_input_ids,
                chosen_attention_mask,
                rejected_input_ids,
                rejected_attention_mask,
            )
            if any(value is not None for value in preference_values):
                if (
                    chosen_input_ids is None
                    or chosen_attention_mask is None
                    or rejected_input_ids is None
                    or rejected_attention_mask is None
                ):
                    raise ValueError(
                        "preference scoring requires chosen/rejected input "
                        "ids and attention masks"
                    )
                chosen_output, _chosen_cache = self.dense_decoder(
                    context=dense_context,
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                )
                rejected_output, _rejected_cache = self.dense_decoder(
                    context=dense_context,
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                )
                preference_outputs = {
                    "chosen_sequence_logits": chosen_output.logits,
                    "rejected_sequence_logits": rejected_output.logits,
                }
                # Preference ownership lives in the model: the chosen decoder
                # pass is also the primary sequence output.  Do not run it a
                # second time merely because a collator populated the regular
                # decoder fields.
                dense_output = chosen_output
            elif (
                batch.decoder_input_ids is not None
                and batch.decoder_attention_mask is not None
            ):
                dense_output, _cache = self.dense_decoder(
                    context=dense_context,
                    input_ids=batch.decoder_input_ids,
                    attention_mask=batch.decoder_attention_mask,
                )
            else:
                raise ValueError(
                    "dense_transformer sequence output requires decoder "
                    "inputs or complete chosen/rejected preference inputs"
                )
        return self._task_output_router.route(
            batch,
            projected,
            fused,
            resolved_heads=resolved_heads,
            dense_output=dense_output,
            dense_context=dense_context,
            preference_outputs=preference_outputs,
        )

    def encode_context(
        self,
        batch: "CollatedBatch",
        *,
        active_modalities: Collection[str] | None = None,
    ) -> ModalityTokenSequence:
        modality_row_masks = resolve_modality_row_masks(
            batch=batch,
            config=self._config,
            enabled_modalities=self._enabled_modalities,
            requested_modalities=active_modalities,
        )
        encoded_by_modality = self._encoder_runner.encode(
            batch,
            resolved_heads=frozenset({"modality_embeddings"}),
            modality_row_masks=modality_row_masks,
        )
        projected = self._projection_runner.project(
            batch,
            encoded_by_modality=encoded_by_modality,
            modality_row_masks=modality_row_masks,
        )
        return self._fusion_runner.context_sequence(
            projected=projected, batch=batch
        )


def _prefix_lm_attention_pattern(
    *, context_length: int, decoder_length: int, device: torch.device
) -> torch.Tensor:
    total = context_length + decoder_length
    allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
    if context_length:
        allowed[:context_length, :context_length] = True
    if decoder_length:
        allowed[context_length:, :context_length] = True
        allowed[context_length:, context_length:] = torch.tril(
            torch.ones(
                decoder_length,
                decoder_length,
                dtype=torch.bool,
                device=device,
            )
        )
    return allowed


def _position_ids(valid_mask: torch.Tensor, maximum: int) -> torch.Tensor:
    positions = (
        valid_mask.to(dtype=torch.long).cumsum(dim=1).sub(1).clamp_min(0)
    )
    return positions.clamp_max(maximum - 1)
