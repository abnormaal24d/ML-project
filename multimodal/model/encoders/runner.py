"""Execute configured modality encoders over collated batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from multimodal.model.contracts import CollatedBatch

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from multimodal.model.encoders.composition import EncoderComposition


def module_for_name(*, modules: nn.ModuleDict, name: str) -> nn.Module | None:
    return dict(modules.items()).get(name)


def required_embedding(
    *, encoded: dict[str, torch.Tensor], source: str
) -> torch.Tensor:
    value = encoded.get("embedding")
    if not torch.is_tensor(value):
        raise KeyError(f"{source} output dict must contain tensor 'embedding'")
    return value


def required_token_sequence(
    *, encoded: dict[str, torch.Tensor], source: str
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = encoded.get("tokens")
    attention_mask = encoded.get("attention_mask")
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        raise KeyError(
            f"{source} output must contain [batch, tokens, hidden] 'tokens'"
        )
    if (
        not torch.is_tensor(attention_mask)
        or attention_mask.shape != tokens.shape[:2]
    ):
        raise KeyError(
            f"{source} output must contain matching 'attention_mask'"
        )
    return tokens, attention_mask.to(device=tokens.device, dtype=torch.bool)


def scatter_encoded(
    *,
    encoded: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    restored: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        full_value = value.new_zeros((batch_size, *value.shape[1:]))
        full_value.index_copy_(0, indices.to(value.device), value)
        restored[key] = full_value
    return restored


def encode_active(
    *,
    encoder: nn.Module,
    values: torch.Tensor,
    mask: torch.Tensor,
    output_dim: int,
    output_keys: set[str] | None,
) -> dict[str, torch.Tensor]:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        raise ValueError("cannot encode a modality with no active rows")

    indices: torch.Tensor | None = None
    active_values = values

    if not bool(mask.all().item()):
        indices = mask.nonzero(as_tuple=True)[0]
        active_values = values.index_select(0, indices)

    raw_encoded = encoder(active_values, output_keys=output_keys)

    if not isinstance(raw_encoded, dict):
        raise TypeError("encoder must return a dict containing 'embedding'")
    if "embedding" not in raw_encoded:
        raise KeyError("encoder output dict must contain 'embedding'")
    if any(not torch.is_tensor(value) for value in raw_encoded.values()):
        raise TypeError("all encoder output values must be tensors")

    encoded = cast("dict[str, torch.Tensor]", raw_encoded)

    if indices is not None:
        encoded = scatter_encoded(
            encoded=encoded,
            indices=indices,
            batch_size=values.shape[0],
        )

    embedding = required_embedding(encoded=encoded, source="active encoder")
    embedding = embedding * mask.to(dtype=embedding.dtype).unsqueeze(-1)
    if embedding.shape[-1] != output_dim:
        raise ValueError(
            f"encoder output dimension {embedding.shape[-1]} "
            f"does not match fusion_dim {output_dim}"
        )
    result = {**encoded, "embedding": embedding}
    tokens = result.get("tokens")
    attention_mask = result.get("attention_mask")
    if torch.is_tensor(tokens):
        if tokens.ndim != 3 or tokens.shape[0] != values.shape[0]:
            raise ValueError("encoder tokens must be [batch, tokens, hidden]")
        if tokens.shape[-1] != output_dim:
            raise ValueError(
                f"encoder token dimension {tokens.shape[-1]} "
                f"does not match fusion_dim {output_dim}"
            )
        row_mask = mask.to(device=tokens.device, dtype=tokens.dtype)
        result["tokens"] = tokens * row_mask[:, None, None]
        if not torch.is_tensor(attention_mask):
            raise ValueError("encoder token output requires attention_mask")
        result["attention_mask"] = attention_mask.to(
            device=tokens.device, dtype=torch.bool
        ) & mask.to(device=tokens.device, dtype=torch.bool).unsqueeze(1)
    return result


def encoder_output_keys(
    *, modality: str, output_heads: frozenset[str]
) -> set[str] | None:
    keys: set[str] = set()
    if modality == "text" and "text_mlm" in output_heads:
        keys.add("text_mlm_logits")
    if modality == "image" and "image_reconstruction" in output_heads:
        keys.update(("image_reconstruction", "image_region_features"))
    if modality == "audio" and "audio_reconstruction" in output_heads:
        keys.add("audio_reconstruction")
    if modality == "video" and "video_temporal" in output_heads:
        keys.add("video_temporal_logits")
    return keys or None


def text_sequence_length(*, config: ModelSettings) -> int:
    text_decoder = getattr(config, "text_decoder", None)
    value = (
        getattr(text_decoder, "max_target_tokens", None)
        if text_decoder is not None and bool(text_decoder.enabled)
        else None
    )
    return max(1, int(value or config.raw_text_max_tokens))


def text_vocab_size(*, config: ModelSettings) -> int:
    text_decoder = getattr(config, "text_decoder", None)
    value = (
        getattr(text_decoder, "vocab_size", None)
        if text_decoder is not None and bool(text_decoder.enabled)
        else None
    )
    return max(1, int(value or config.raw_text_vocab_size))


def video_generation_vocab_size(*, config: ModelSettings) -> int:
    video_generator = getattr(config, "video_generator", None)
    value = (
        getattr(video_generator, "video_token_vocab_size", None)
        if video_generator is not None
        else None
    )
    if value is None:
        image_generator = getattr(config, "image_generator", None)
        value = (
            getattr(image_generator, "image_token_vocab_size", None)
            if image_generator is not None
            else None
        )
    return max(1, int(value or config.raw_text_vocab_size))


class ModalityEncoderRunner:
    """Encode every active physical and document modality in one batch."""

    def __init__(
        self,
        *,
        config: ModelSettings,
        enabled_modalities: tuple[str, ...],
        encoders: EncoderComposition,
        document_fallback_projection: nn.Module,
    ) -> None:
        self._config = config
        self._enabled_modalities = enabled_modalities
        self._encoders = encoders
        self._document_fallback_projection = document_fallback_projection

    def encode(
        self,
        batch: CollatedBatch,
        *,
        resolved_heads: frozenset[str],
        modality_row_masks: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        encoded_by_modality: dict[str, dict[str, torch.Tensor]] = {}
        for modality in self._enabled_modalities:
            if modality == "document":
                continue
            row_mask = modality_row_masks.get(modality)
            encoder = module_for_name(
                modules=self._encoders.encoders,
                name=modality,
            )
            if (
                row_mask is None
                or encoder is None
                or not bool(row_mask.any().item())
            ):
                continue
            encoded_by_modality[modality] = encode_active(
                encoder=encoder,
                values=getattr(batch, modality),
                mask=row_mask,
                output_dim=self._config.fusion_dim,
                output_keys=encoder_output_keys(
                    modality=modality,
                    output_heads=resolved_heads,
                ),
            )

        document_mask = modality_row_masks.get("document")
        if (
            "document" in self._enabled_modalities
            and document_mask is not None
            and bool(document_mask.any().item())
        ):
            encoded_by_modality["document"] = self._encode_document_modality(
                batch=batch,
                row_mask=document_mask,
                text_encoded=encoded_by_modality.get("text"),
            )

        return encoded_by_modality

    def _encode_document_modality(
        self,
        *,
        batch: CollatedBatch,
        row_mask: torch.Tensor,
        text_encoded: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        source_encoded: dict[str, torch.Tensor]
        if text_encoded is not None:
            source_encoded = text_encoded
        else:
            values = getattr(batch, "document", None)
            if values is None:
                values = batch.text
            document_text_encoder = self._encoders.document_text_encoder
            if document_text_encoder is None:
                raise RuntimeError(
                    "document text encoder is required for document input"
                )
            source_encoded = encode_active(
                encoder=document_text_encoder,
                values=values,
                mask=row_mask,
                output_dim=self._config.fusion_dim,
                output_keys=None,
            )

        text_embedding = required_embedding(
            encoded=source_encoded,
            source="document text encoder",
        )
        source_tokens, source_mask = required_token_sequence(
            encoded=source_encoded,
            source="document text encoder",
        )
        document_embedding, document_tokens, document_attention_mask = (
            self.document_representation(
                batch=batch,
                text_embedding=text_embedding,
                text_tokens=source_tokens,
                text_attention_mask=source_mask,
                document_mask=row_mask.to(text_embedding.device),
            )
        )
        return {
            "embedding": document_embedding,
            "tokens": document_tokens,
            "attention_mask": document_attention_mask,
        }

    def document_representation(
        self,
        *,
        batch: CollatedBatch,
        text_embedding: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = getattr(batch, "document_layout_boxes", None)
        page_ids = getattr(batch, "document_page_ids", None)
        attention_mask = getattr(batch, "document_layout_attention_mask", None)
        if (
            boxes is not None
            and page_ids is not None
            and attention_mask is not None
        ):
            attention_mask = attention_mask.to(
                device=text_embedding.device, dtype=torch.bool
            )
            if bool(attention_mask.any().item()):
                boxes = boxes.to(
                    device=text_embedding.device, dtype=text_embedding.dtype
                )
                page_ids = page_ids.to(
                    device=text_embedding.device, dtype=torch.long
                )
                token_embeddings = _fit_sequence_length(
                    tokens=text_tokens,
                    length=boxes.shape[1],
                )
                encoded = self._encoders.document_encoder(
                    token_embeddings=token_embeddings,
                    boxes=boxes,
                    page_ids=page_ids,
                    attention_mask=attention_mask,
                )
                embedding = required_embedding(
                    encoded=encoded, source="layout document encoder"
                ) * document_mask.unsqueeze(-1)
                document_tokens, document_attention = required_token_sequence(
                    encoded=encoded, source="layout document encoder"
                )
                document_attention = (
                    document_attention & document_mask.unsqueeze(1)
                )
                return embedding, document_tokens, document_attention

        fallback_embedding = cast(
            "torch.Tensor",
            self._document_fallback_projection(text_embedding),
        ) * document_mask.unsqueeze(-1)
        fallback_tokens = self._document_fallback_projection(text_tokens)
        fallback_attention = text_attention_mask & document_mask.unsqueeze(1)
        fallback_tokens = fallback_tokens * fallback_attention.unsqueeze(
            -1
        ).to(fallback_tokens.dtype)
        return fallback_embedding, fallback_tokens, fallback_attention

    def document_embedding(
        self,
        *,
        batch: CollatedBatch,
        text_embedding: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_tokens = text_embedding.unsqueeze(1)
        text_attention_mask = document_mask.unsqueeze(1).to(dtype=torch.bool)
        embedding, _tokens, _attention = self.document_representation(
            batch=batch,
            text_embedding=text_embedding,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            document_mask=document_mask,
        )
        return embedding


def _fit_sequence_length(*, tokens: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("document sequence length must be positive")
    if tokens.shape[1] == length:
        return tokens
    if tokens.shape[1] > length:
        return tokens[:, :length, :]
    padding = tokens.new_zeros(
        tokens.shape[0], length - tokens.shape[1], tokens.shape[2]
    )
    return torch.cat((tokens, padding), dim=1)
