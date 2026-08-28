"""Collate text inputs, labels, and tokenized task metadata."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict

import torch

from mmcrawler_datasets.collation.tensor_ops import (
    IGNORE_LABEL,
    TEXT_TOKEN_DTYPE,
    mask_text,
    sample_generator,
    stack_feature_matrix,
    to_float_tensor,
    to_long_tensor,
)
from multimodal.tasks.registry import task_requires_causal_decoder
from multimodal.tokenization.text import VocabularyTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mmcrawler_datasets.schema import MultimodalSample
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

from pathlib import Path

from mmcrawler_datasets.tensors import load_required_tensor

_LOGGER = logging.getLogger(__name__)


class DecoderTensorBatch(TypedDict):
    decoder_input_ids: torch.Tensor
    decoder_labels: torch.Tensor
    decoder_attention_mask: torch.Tensor
    prompt_token_count: list[int]
    answer_token_count: list[int]


class PreferenceTensorBatch(TypedDict):
    chosen_input_ids: torch.Tensor | None
    chosen_labels: torch.Tensor | None
    chosen_attention_mask: torch.Tensor | None
    rejected_input_ids: torch.Tensor | None
    rejected_labels: torch.Tensor | None
    rejected_attention_mask: torch.Tensor | None


class DecoderSequence(TypedDict, total=False):
    """One typed causal-decoder sequence before tensor collation."""

    prompt_token_count: int
    answer_token_count: int
    prompt_tokens: list[int]
    answer_tokens: list[int]
    is_conversation: bool
    full_sequence: list[int]
    answer_start: int
    answer_end: int
    eos_id: int


class TextCollator:
    """Build masked text tensors and auxiliary text-side batch fields."""

    _CONVERSATION_ROLE_TOKENS: dict[str, str] = {
        "system": "<system>",
        "user": "<user>",
        "assistant": "<assistant>",
        "tool": "<tool>",
    }

    def __init__(
        self,
        *,
        tokenizer: VocabularyTokenizer,
        mlm_probability: float,
        materialized_dataset_root: Path | None = None,
        use_materialized_text_tokens: bool = False,
        base_seed: int = 0,
    ) -> None:
        self._tokenizer = tokenizer
        self.raw_text_max_tokens = tokenizer.max_tokens
        self.pad_token_id = tokenizer.token_to_id[tokenizer.pad_token]
        self.mask_token_id = tokenizer.token_to_id["<mask>"]
        self.mlm_probability = float(mlm_probability)
        self._materialized_dataset_root = (
            Path(materialized_dataset_root).resolve()
            if materialized_dataset_root is not None
            else None
        )
        self._use_materialized_text_tokens = bool(use_materialized_text_tokens)
        self._base_seed = int(base_seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def collate_sample(
        self,
        sample: MultimodalSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked text input and MLM target for one sample."""

        text = (
            sample.input_text
            or sample.title
            or sample.generative_target_text
            or ""
        )
        if self._use_materialized_text_tokens:
            if self._materialized_dataset_root is None:
                raise RuntimeError(
                    "materialized text tokens require materialized_dataset_root"
                )
            if sample.has_text:
                token_ids = load_required_tensor(
                    dataset_root=self._materialized_dataset_root,
                    path=sample.text_tokens_path,
                    expected_shape=(self.raw_text_max_tokens,),
                    dtype=TEXT_TOKEN_DTYPE,
                )
            else:
                token_ids = torch.full(
                    (self.raw_text_max_tokens,),
                    self.pad_token_id,
                    dtype=TEXT_TOKEN_DTYPE,
                )
        else:
            token_ids = torch.tensor(
                self._tokenizer.encode(text),
                dtype=TEXT_TOKEN_DTYPE,
            )
        generator = sample_generator(
            base_seed=self._base_seed,
            epoch=self._epoch,
            sample_id=str(sample.sample_id),
            operation="text_mask",
        )
        masked_text, text_target = mask_text(
            token_ids,
            probability=self.mlm_probability,
            generator=generator,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
        )
        if sample.task_type != "text_pretrain":
            text_target = torch.full_like(text_target, IGNORE_LABEL)
        return masked_text, text_target

    def collate_batch(
        self,
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack masked text tensors for a batch of samples."""

        text_inputs: list[torch.Tensor] = []
        text_targets: list[torch.Tensor] = []
        for sample in samples:
            masked_text, text_target = self.collate_sample(sample)
            text_inputs.append(masked_text)
            text_targets.append(text_target)
        return (
            stack_feature_matrix(text_inputs),
            stack_feature_matrix(text_targets),
        )

    @staticmethod
    def collate_labels(
        samples: Sequence[MultimodalSample],
    ) -> torch.Tensor | None:
        """Collate optional scalar labels, tolerating mixed unlabeled rows."""

        labels_raw = [sample.label for sample in samples]
        if all(label is None for label in labels_raw):
            return None
        if any(label is None for label in labels_raw):
            logging.getLogger(__name__).debug(
                "mixed_labeled_unlabeled_tolerated_for_full_modality_utilization"
            )
        label_list = [
            int(label) if label is not None else IGNORE_LABEL
            for label in labels_raw
        ]
        return to_long_tensor(label_list)

    @staticmethod
    def collate_alignment_scores(
        samples: Sequence[MultimodalSample],
    ) -> torch.Tensor:
        """Collate clamped alignment scores for one batch."""

        values = [
            max(0.0, min(1.0, float(sample.alignment_score)))
            for sample in samples
        ]
        return to_float_tensor(values)

    @classmethod
    def collate_task_ids(
        cls,
        samples: Sequence[MultimodalSample],
    ) -> torch.Tensor:
        """Map task type strings onto stable integer ids."""

        return to_long_tensor(
            [cls.task_id(task_type=sample.task_type) for sample in samples]
        )

    @staticmethod
    def task_id(*, task_type: str) -> int:
        """Return a stable non-negative integer for a task type string."""

        digest = hashlib.blake2b(
            str(task_type or "").encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") % (2**31 - 1)

    def collate_token_ids(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode free-form texts with the configured tokenizer."""

        return stack_feature_matrix(
            [
                torch.tensor(
                    self._tokenizer.encode(text or ""),
                    dtype=TEXT_TOKEN_DTYPE,
                )
                for text in texts
            ]
        )

    def collate_target_attention_mask(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """Build attention masks for generative target token sequences."""

        token_ids = self.collate_token_ids(texts)
        return token_ids.ne(int(self.pad_token_id))

    def _role_token_id(self, role: str) -> int:
        """Return the delimiter token id for a conversation role."""

        token = self._CONVERSATION_ROLE_TOKENS.get(role, f"<{role}>")
        if token not in self._tokenizer.token_to_id:
            raise ValueError(
                f"tokenizer missing conversation role token: {token!r}"
            )
        return int(self._tokenizer.token_to_id[token])

    def _conversation_turns_from_flat_fields(
        self,
        sample: MultimodalSample,
    ) -> tuple[ConversationTurn, ...]:
        """Convert flat conversation fields to canonical role-marked turns."""

        from mmcrawler_datasets.training_samples.targets import (
            ConversationTurn,
        )

        turns: list[ConversationTurn] = []
        turn_index = 0
        if sample.system_text and sample.system_text.strip():
            turns.append(
                ConversationTurn(
                    role="system",
                    text=sample.system_text.strip(),
                    turn_index=turn_index,
                )
            )
            turn_index += 1
        if sample.user_text and sample.user_text.strip():
            turns.append(
                ConversationTurn(
                    role="user",
                    text=sample.user_text.strip(),
                    turn_index=turn_index,
                )
            )
            turn_index += 1
        if sample.assistant_text and sample.assistant_text.strip():
            turns.append(
                ConversationTurn(
                    role="assistant",
                    text=sample.assistant_text.strip(),
                    turn_index=turn_index,
                    is_assistant_answer=True,
                )
            )
        return tuple(turns)

    def _generative_turns_from_sample(
        self,
        sample: MultimodalSample,
    ) -> tuple[ConversationTurn, ...]:
        """Build canonical prompt/answer turns for non-conversation tasks."""

        if not task_requires_causal_decoder(sample.task_type):
            return ()

        from mmcrawler_datasets.training_samples.targets import (
            ConversationTurn,
        )

        target = sample.generative_target_text
        if target is None or not target.strip():
            raise ValueError(
                "causal generative sample requires a non-empty target: "
                f"sample_id={sample.sample_id!r}, task={sample.task_type!r}"
            )

        turns: list[ConversationTurn] = []
        prompt = sample.input_text.strip()
        if prompt:
            turns.append(
                ConversationTurn(
                    role="user",
                    text=prompt,
                    turn_index=0,
                )
            )
        turns.append(
            ConversationTurn(
                role="assistant",
                text=target.strip(),
                turn_index=len(turns),
                is_assistant_answer=True,
            )
        )
        return tuple(turns)

    def _encode_turn_segment(
        self,
        turn: ConversationTurn,
    ) -> list[int]:
        """Encode one complete role/content turn without internal padding."""

        return [
            self._role_token_id(turn.role),
            *(
                int(token_id)
                for token_id in self._tokenizer.encode(
                    turn.text.strip(),
                    add_special_tokens=False,
                    pad_to_max_length=False,
                )
            ),
        ]

    def _select_recent_prompt_segments(
        self,
        *,
        segments: Sequence[tuple[str, list[int]]],
        token_budget: int,
        sample_id: str,
    ) -> list[int]:
        """Keep optional system context and the newest complete prompt turns."""

        if not segments:
            return []
        if token_budget < 2:
            raise ValueError(
                "conversation prompt cannot fit decoder sequence budget: "
                f"sample_id={sample_id!r}, token_budget={token_budget}"
            )

        initial_system: list[int] | None = None
        remaining_segments = list(segments)
        if remaining_segments[0][0] == "system":
            initial_system = remaining_segments.pop(0)[1]

        selected_reversed: list[list[int]] = []
        remaining_budget = token_budget
        for _role, segment in reversed(remaining_segments):
            if len(segment) <= remaining_budget:
                selected_reversed.append(segment)
                remaining_budget -= len(segment)
                continue

            if not selected_reversed:
                if remaining_budget < 2:
                    raise ValueError(
                        "latest conversation turn cannot fit decoder sequence "
                        f"budget: sample_id={sample_id!r}, "
                        f"token_budget={token_budget}"
                    )
                selected_reversed.append(
                    [segment[0], *segment[-(remaining_budget - 1) :]]
                )
                remaining_budget = 0
            break

        selected = list(reversed(selected_reversed))
        if initial_system is not None and remaining_budget > 0:
            if len(initial_system) <= remaining_budget:
                selected.insert(0, initial_system)
            elif not selected and remaining_budget >= 2:
                selected.insert(
                    0,
                    initial_system[:remaining_budget],
                )

        return [token for segment in selected for token in segment]

    def build_causal_pretrain_sequence(
        self,
        sample: MultimodalSample,
    ) -> DecoderSequence:
        """Build a pure autoregressive causal sequence for text pretraining.

        The decoder loss performs the next-token shift itself. This method
        therefore keeps the complete ``[BOS, content..., EOS]`` sequence in
        the inputs and marks every post-BOS position as a target. There are
        no conversation delimiters, system prompts, or role tokens.
        """
        text = (
            sample.input_text
            or sample.title
            or sample.generative_target_text
            or ""
        )
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
            pad_to_max_length=False,
        )

        if not token_ids:
            return {
                "prompt_token_count": 0,
                "answer_token_count": 0,
                "prompt_tokens": [],
                "answer_tokens": [],
                "is_conversation": False,
                "full_sequence": [],
                "answer_start": 0,
                "answer_end": 0,
                "eos_id": int(
                    self._tokenizer.token_to_id[self._tokenizer.eos_token]
                ),
            }

        bos_id = int(self._tokenizer.token_to_id[self._tokenizer.bos_token])
        eos_id = int(self._tokenizer.token_to_id[self._tokenizer.eos_token])

        content_budget = self.raw_text_max_tokens - 2
        full_sequence = [bos_id, *token_ids[:content_budget], eos_id]
        prompt_tokens = [bos_id]
        answer_tokens = full_sequence[1:]
        answer_start = 1
        answer_end = len(full_sequence)

        return {
            "prompt_token_count": len(prompt_tokens),
            "answer_token_count": len(answer_tokens),
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "is_conversation": False,
            "full_sequence": full_sequence,
            "answer_start": answer_start,
            "answer_end": answer_end,
            "eos_id": eos_id,
        }

    def build_decoder_sequence(
        self,
        sample: MultimodalSample,
    ) -> DecoderSequence:
        """Build one compact causal assistant-training sequence.

        Contract B is used: the ``<assistant>`` delimiter belongs to the prompt
        and is excluded from labels. Only assistant content plus ``<eos>`` are
        teacher-forced targets.
        """

        from mmcrawler_datasets.training_samples.targets import (
            validate_conversation_turns,
        )

        bos_id = int(self._tokenizer.token_to_id[self._tokenizer.bos_token])
        eos_id = int(self._tokenizer.token_to_id[self._tokenizer.eos_token])
        assistant_token_id = self._role_token_id("assistant")

        turns = sample.conversation_turns
        if not turns:
            turns = self._conversation_turns_from_flat_fields(sample)
        if not turns:
            turns = self._generative_turns_from_sample(sample)

        if not turns:
            return {
                "prompt_token_count": 0,
                "answer_token_count": 0,
                "prompt_tokens": [],
                "answer_tokens": [],
                "is_conversation": False,
            }

        target_index = validate_conversation_turns(
            turns,
            sample_id=str(sample.sample_id),
        )
        target_turn = turns[target_index]
        answer_tokens = [
            *(
                int(token_id)
                for token_id in self._tokenizer.encode(
                    target_turn.text.strip(),
                    add_special_tokens=False,
                    pad_to_max_length=False,
                )
            ),
            eos_id,
        ]

        maximum_answer_tokens = self.raw_text_max_tokens - 2
        if len(answer_tokens) > maximum_answer_tokens:
            raise ValueError(
                "assistant answer exceeds decoder sequence budget: "
                f"sample_id={sample.sample_id!r}, "
                f"answer_tokens={len(answer_tokens)}, "
                f"maximum={maximum_answer_tokens}"
            )

        prompt_segments = [
            (turn.role, self._encode_turn_segment(turn))
            for turn in turns[:target_index]
        ]
        context_budget = self.raw_text_max_tokens - len(answer_tokens) - 2
        prompt_context = self._select_recent_prompt_segments(
            segments=prompt_segments,
            token_budget=context_budget,
            sample_id=str(sample.sample_id),
        )
        prompt_tokens = [bos_id, *prompt_context, assistant_token_id]
        full_sequence = [*prompt_tokens, *answer_tokens]
        answer_start = len(prompt_tokens)
        answer_end = len(full_sequence)

        return {
            "prompt_token_count": len(prompt_tokens),
            "answer_token_count": len(answer_tokens),
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "is_conversation": True,
            "full_sequence": full_sequence,
            "answer_start": answer_start,
            "answer_end": answer_end,
            "eos_id": eos_id,
        }

    def collate_decoder_tensors(
        self,
        samples: Sequence[MultimodalSample],
    ) -> DecoderTensorBatch:
        """Stack role-marked decoder inputs/labels for a batch of samples."""

        sequences = []
        for sample in samples:
            if sample.task_type == "causal_text_pretrain":
                sequences.append(self.build_causal_pretrain_sequence(sample))
            else:
                sequences.append(self.build_decoder_sequence(sample))

        max_seq_len = max(
            (len(seq.get("full_sequence", [])) for seq in sequences),
            default=0,
        )

        decoder_inputs: list[torch.Tensor] = []
        decoder_labels: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        prompt_counts: list[int] = []
        answer_counts: list[int] = []
        for seq in sequences:
            full_sequence = seq.get("full_sequence", [])
            if not full_sequence:
                if max_seq_len == 0:
                    input_row = torch.zeros((0,), dtype=TEXT_TOKEN_DTYPE)
                    label_row = torch.full(
                        (0,), IGNORE_LABEL, dtype=TEXT_TOKEN_DTYPE
                    )
                    attention_row = torch.zeros((0,), dtype=torch.bool)
                else:
                    input_row = torch.full(
                        (max_seq_len,),
                        int(self.pad_token_id),
                        dtype=TEXT_TOKEN_DTYPE,
                    )
                    label_row = torch.full(
                        (max_seq_len,),
                        IGNORE_LABEL,
                        dtype=TEXT_TOKEN_DTYPE,
                    )
                    attention_row = torch.zeros(
                        (max_seq_len,),
                        dtype=torch.bool,
                    )
                decoder_inputs.append(input_row)
                decoder_labels.append(label_row)
                attention_masks.append(attention_row)
                prompt_counts.append(0)
                answer_counts.append(0)
                continue

            sequence_length = len(full_sequence)
            answer_start = seq["answer_start"]
            answer_end = seq["answer_end"]

            input_row = torch.full(
                (max_seq_len,),
                int(self.pad_token_id),
                dtype=TEXT_TOKEN_DTYPE,
            )
            label_row = torch.full(
                (max_seq_len,),
                IGNORE_LABEL,
                dtype=TEXT_TOKEN_DTYPE,
            )

            input_row[:sequence_length] = torch.tensor(
                full_sequence, dtype=TEXT_TOKEN_DTYPE
            )
            label_row[answer_start:answer_end] = torch.tensor(
                seq["answer_tokens"], dtype=TEXT_TOKEN_DTYPE
            )

            decoder_inputs.append(input_row)
            decoder_labels.append(label_row)
            attention_masks.append(input_row.ne(int(self.pad_token_id)))
            prompt_counts.append(seq["prompt_token_count"])
            answer_counts.append(seq["answer_token_count"])

        if not decoder_inputs:
            decoder_input = torch.zeros((0, 0), dtype=TEXT_TOKEN_DTYPE)
        else:
            decoder_input = stack_feature_matrix(decoder_inputs)

        return {
            "decoder_input_ids": decoder_input,
            "decoder_labels": (
                stack_feature_matrix(decoder_labels)
                if decoder_labels
                else torch.full_like(decoder_input, IGNORE_LABEL)
            ),
            "decoder_attention_mask": (
                stack_feature_matrix(attention_masks)
                if attention_masks
                else torch.zeros_like(decoder_input, dtype=torch.bool)
            ),
            "prompt_token_count": prompt_counts,
            "answer_token_count": answer_counts,
        }

    def collate_preference_tensors(
        self,
        samples: Sequence[MultimodalSample],
    ) -> PreferenceTensorBatch:
        paired = [
            bool(sample.chosen_response and sample.rejected_response)
            for sample in samples
        ]
        if not any(paired):
            return {
                "chosen_input_ids": None,
                "chosen_labels": None,
                "chosen_attention_mask": None,
                "rejected_input_ids": None,
                "rejected_labels": None,
                "rejected_attention_mask": None,
            }
        if not all(paired):
            raise ValueError(
                "preference batches may not mix paired and unpaired samples"
            )
        chosen = self.collate_decoder_tensors(
            [
                self._sample_with_response(
                    sample, response=str(sample.chosen_response)
                )
                for sample in samples
            ]
        )
        rejected = self.collate_decoder_tensors(
            [
                self._sample_with_response(
                    sample, response=str(sample.rejected_response)
                )
                for sample in samples
            ]
        )
        return {
            "chosen_input_ids": chosen["decoder_input_ids"],
            "chosen_labels": chosen["decoder_labels"],
            "chosen_attention_mask": chosen["decoder_attention_mask"],
            "rejected_input_ids": rejected["decoder_input_ids"],
            "rejected_labels": rejected["decoder_labels"],
            "rejected_attention_mask": rejected["decoder_attention_mask"],
        }

    @staticmethod
    def _sample_with_response(
        sample: MultimodalSample,
        *,
        response: str,
    ) -> MultimodalSample:
        turns = sample.conversation_turns
        if turns:
            from mmcrawler_datasets.training_samples.targets import (
                ConversationTurn,
                validate_conversation_turns,
            )

            target_index = validate_conversation_turns(
                turns, sample_id=str(sample.sample_id)
            )
            target = turns[target_index]
            updated_target = ConversationTurn(
                role="assistant",
                text=response,
                turn_index=target.turn_index,
                answer_evidence_ids=target.answer_evidence_ids,
                is_assistant_answer=True,
            )
            return replace(
                sample,
                conversation_turns=tuple(
                    (*turns[:target_index], updated_target)
                ),
                assistant_text=None,
                answer=None,
                target_text=None,
                target_code=None,
            )
        return replace(
            sample,
            assistant_text=response,
            answer=None,
            target_text=None,
            target_code=None,
        )

    @staticmethod
    def collate_safety_targets(
        samples: Sequence[MultimodalSample],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        names = (
            "must_refuse",
            "requires_uncertainty",
            "requires_source_citation",
            "prompt_injection_present",
            "untrusted_document_instruction",
            "sensitive_data_present",
            "requires_tool_confirmation",
        )
        rows: list[list[float]] = []
        masks: list[list[bool]] = []
        for sample in samples:
            row: list[float] = []
            mask: list[bool] = []
            for name in names:
                value = getattr(sample, name)
                row.append(float(bool(value)))
                mask.append(value is not None)
            rows.append(row)
            masks.append(mask)
        mask_tensor = torch.tensor(masks, dtype=torch.bool)
        if not bool(mask_tensor.any().item()):
            return None, None
        return torch.tensor(rows, dtype=torch.float32), mask_tensor

    @staticmethod
    def collate_optional_label_ids(
        labels: Sequence[str | None],
    ) -> torch.Tensor | None:
        """Map optional string labels onto stable integer ids."""

        if all(label is None or not str(label).strip() for label in labels):
            return None
        return to_long_tensor(
            [
                TextCollator.stable_label_id(label=label)
                if label is not None and str(label).strip()
                else IGNORE_LABEL
                for label in labels
            ]
        )

    @staticmethod
    def stable_label_id(*, label: str) -> int:
        """Return a stable non-negative integer for a label string."""

        digest = hashlib.blake2b(
            str(label).casefold().encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") % (2**31 - 1)
