from __future__ import annotations

from pathlib import Path

import pytest

from mmcrawler_datasets.collation.tensor_ops import (
    IGNORE_LABEL,
    TEXT_TOKEN_DTYPE,
)
from mmcrawler_datasets.collation.text import TextCollator
from mmcrawler_datasets.record_components.parsing import (
    parse_modality_objects,
    parse_record,
    require_training_record,
)
from mmcrawler_datasets.schema import MultimodalSample
from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)
from mmcrawler_datasets.training_samples.models import (
    TrainingObject,
    TrainingSample,
)
from mmcrawler_datasets.training_samples.snapshot_mapping import (
    serialize_snapshot_sample,
)
from preprocessing.privacy.clearance import ApprovedObjectRole


def test_canonical_objects_array_resolves_media(tmp_path: Path) -> None:
    asset = tmp_path / "objects" / "image.jpg"
    asset.parent.mkdir()
    asset.write_bytes(b"jpeg")
    objects = parse_modality_objects(
        raw_objects=[
            {
                "object_id": "image-1",
                "object_path": "objects/image.jpg",
                "object_mime_type": "image/jpeg",
                "role": "image",
            }
        ],
        dataset_root=tmp_path,
    )
    assert "image" in objects
    assert objects["image"].path == asset
    assert objects["image"].mime_type == "image/jpeg"


def test_unsupported_object_role_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported object role"):
        parse_modality_objects(
            raw_objects=[
                {
                    "object_path": "image.jpg",
                    "role": "primary_media",
                }
            ],
            dataset_root=tmp_path,
        )


def test_training_sample_serialization_emits_only_canonical_media_shape(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "objects" / "image.jpg"
    asset.parent.mkdir()
    asset.write_bytes(b"image")
    validated = ValidatedArtifactPath(
        relative_path="objects/image.jpg",
        resolved_path=asset,
        project_root=tmp_path,
    )
    sample = TrainingSample(
        sample_id="sample-1",
        object_id="image-1",
        objects=(
            TrainingObject(
                object_id="image-1",
                object_path=validated,
                object_sha256=__import__("hashlib")
                .sha256(b"image")
                .hexdigest(),
                object_mime_type="image/jpeg",
                role=ApprovedObjectRole.PRIMARY_MEDIA,
            ),
        ),
    )
    payload = sample.to_dict()
    assert "object_path" not in payload
    assert "object_mime_type" not in payload
    assert "media_path" not in payload
    assert payload["objects"][0]["object_path"] == "objects/image.jpg"  # type: ignore[index]


def test_training_sample_rejects_flattened_media_fields() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TrainingSample(object_path="removed.jpg")  # type: ignore[call-arg]


def test_require_training_record_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="unsupported training schema"):
        require_training_record(
            {
                "sample_id": "s1",
                "record_id": "r1",
                "task_target": {"task_type": "vqa"},
                "objects": [],
            }
        )


def _conversation_record() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "sample_id": "conv-1",
        "record_id": "record-1",
        "modality": "text",
        "text": "",
        "objects": [],
        "task_target": {
            "task_type": "instruction_following",
            "task_family": "text",
            "system_text": "You are a helpful assistant.",
            "user_text": "What is the capital of France?",
            "assistant_text": "Paris",
            "conversation_turns": [
                {"role": "system", "text": "You are a helpful assistant."},
                {"role": "user", "text": "What is the capital of France?"},
                {"role": "assistant", "text": "Paris"},
            ],
            "answer_evidence_ids": ("doc:fr:capitals",),
        },
    }


def test_conversation_target_round_trip(tmp_path: Path) -> None:
    sample = parse_record(
        record=_conversation_record(),
        dataset_root=tmp_path,
    )
    assert sample.system_text == "You are a helpful assistant."
    assert sample.assistant_text == "Paris"
    assert len(sample.conversation_turns) == 3
    assert sample.conversation_turns[2].role == "assistant"
    assert sample.has_conversation
    assert sample.answer_text == "Paris"

    serialized = serialize_snapshot_sample(
        sample=sample,
        dataset_root=tmp_path,
    )
    task_target = serialized["task_target"]
    assert task_target["system_text"] == "You are a helpful assistant."
    assert task_target["conversation_turns"][2]["role"] == "assistant"
    assert task_target["assistant_text"] == "Paris"
    assert task_target["answer_evidence_ids"] == ["doc:fr:capitals"]

    parsed_2 = parse_record(
        record=serialized,
        dataset_root=tmp_path,
    )
    assert parsed_2.system_text == sample.system_text
    assert parsed_2.assistant_text == sample.assistant_text
    assert len(parsed_2.conversation_turns) == len(sample.conversation_turns)
    assert parsed_2.answer_text == sample.answer_text


class _StubTokenizer:
    """Minimal tokenizer for TextCollator decoder-sequence tests."""

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.token_to_id = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<mask>": 4,
            "<image>": 5,
            "<audio>": 6,
            "<video>": 7,
            "<doc>": 8,
            "<user>": 9,
            "<assistant>": 10,
            "<system>": 11,
            "<tool>": 12,
        }
        for i in range(256):
            key = f"<byte:{i:02x}>"
            if key not in self.token_to_id:
                self.token_to_id[key] = 13 + i
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.max_tokens = 64

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        pad_to_max_length: bool = True,
    ) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id[self.bos_token])
        for char in text:
            byte_val = ord(char.encode("utf-8")[:1])
            key = f"<byte:{byte_val:02x}>"
            if key in self.token_to_id:
                ids.append(self.token_to_id[key])
            else:
                ids.append(self.token_to_id[self.unk_token])
        if add_special_tokens and len(ids) < self.max_tokens:
            ids.append(self.token_to_id[self.eos_token])
        ids = ids[: self.max_tokens]
        if pad_to_max_length:
            pad_id = self.token_to_id[self.pad_token]
            ids.extend([pad_id] * (self.max_tokens - len(ids)))
        return ids


def test_conversation_decoder_sequence_masks_prompt() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="conv-decoder-1",
        system_text="You are helpful.",
        user_text="What is 2+2?",
        conversation_turns=(
            ConversationTurn(
                role="system", text="You are helpful.", turn_index=0
            ),
            ConversationTurn(role="user", text="What is 2+2?", turn_index=1),
            ConversationTurn(
                role="assistant",
                text="Four.",
                turn_index=2,
                is_assistant_answer=True,
            ),
        ),
        assistant_text="Four",
        task_type="instruction_following",
    )
    seq = collator.build_decoder_sequence(sample)
    assert seq["prompt_token_count"] > 0
    assert seq["answer_token_count"] >= 1
    batch = collator.collate_decoder_tensors((sample, sample))
    labels = batch["decoder_labels"]
    inputs = batch["decoder_input_ids"]
    assert inputs.shape == labels.shape
    prompt_len = seq["prompt_token_count"]
    assert labels.shape[1] >= prompt_len
    assert (labels[:, :prompt_len] == IGNORE_LABEL).all()
    assert (labels[:, prompt_len:] != IGNORE_LABEL).any()
    assert inputs.dtype == TEXT_TOKEN_DTYPE


def test_non_conversation_sample_has_all_ignored_labels() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    sample = MultimodalSample(
        sample_id="plain-1",
        text="Hello world",
        task_type="text_pretrain",
    )
    seq = collator.build_decoder_sequence(sample)
    assert seq["prompt_token_count"] == 0
    assert seq["answer_token_count"] == 0
    assert not seq.get("is_conversation", True)
    batch = collator.collate_decoder_tensors((sample,))
    labels = batch["decoder_labels"]
    assert labels.numel() == 0 or (labels == IGNORE_LABEL).all()


def test_causal_pretraining_is_a_supervised_decoder_sequence() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    sample = MultimodalSample(
        sample_id="causal-single-token",
        text="x",
        task_type="causal_text_pretrain",
    )

    sequence = collator.build_causal_pretrain_sequence(sample)
    bos_id = tokenizer.token_to_id["<bos>"]
    eos_id = tokenizer.token_to_id["<eos>"]
    content_id = tokenizer.encode(
        "x",
        add_special_tokens=False,
        pad_to_max_length=False,
    )[0]
    assert sequence["full_sequence"] == [bos_id, content_id, eos_id]
    assert sequence["answer_start"] == 1
    assert sequence["answer_tokens"] == [content_id, eos_id]

    batch = collator.collate_decoder_tensors((sample,))
    assert batch["decoder_input_ids"].tolist() == [sequence["full_sequence"]]
    assert batch["decoder_labels"].tolist() == [
        [IGNORE_LABEL, content_id, eos_id]
    ]
    assert batch["decoder_attention_mask"].all()


def test_causal_pretraining_reserves_eos_when_truncated() -> None:
    tokenizer = _StubTokenizer()
    tokenizer.max_tokens = 5
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    sample = MultimodalSample(
        sample_id="causal-truncated",
        text="abcdef",
        task_type="causal_text_pretrain",
    )

    sequence = collator.build_causal_pretrain_sequence(sample)
    expected_content = tokenizer.encode(
        "abcdef",
        add_special_tokens=False,
        pad_to_max_length=False,
    )[:3]
    assert sequence["full_sequence"] == [
        tokenizer.token_to_id["<bos>"],
        *expected_content,
        tokenizer.token_to_id["<eos>"],
    ]
    assert len(sequence["full_sequence"]) == tokenizer.max_tokens


def test_different_length_samples_produce_correct_labels() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    short_sample = MultimodalSample(
        sample_id="short-1",
        conversation_turns=(
            ConversationTurn(role="user", text="Hi", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="OK",
                turn_index=1,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )
    long_sample = MultimodalSample(
        sample_id="long-1",
        conversation_turns=(
            ConversationTurn(
                role="system", text="You are helpful.", turn_index=0
            ),
            ConversationTurn(role="user", text="What is 2+2?", turn_index=1),
            ConversationTurn(
                role="assistant",
                text="Four.",
                turn_index=2,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )
    batch = collator.collate_decoder_tensors((short_sample, long_sample))
    labels = batch["decoder_labels"]
    inputs = batch["decoder_input_ids"]
    assert inputs.shape == labels.shape
    assert inputs.shape[0] == 2

    short_prompt = collator.build_decoder_sequence(short_sample)[
        "prompt_token_count"
    ]
    short_answer = collator.build_decoder_sequence(short_sample)[
        "answer_token_count"
    ]
    assert (labels[0, :short_prompt] == IGNORE_LABEL).all()
    assert (
        labels[0, short_prompt : short_prompt + short_answer] != IGNORE_LABEL
    ).any()

    long_prompt = collator.build_decoder_sequence(long_sample)[
        "prompt_token_count"
    ]
    long_answer = collator.build_decoder_sequence(long_sample)[
        "answer_token_count"
    ]
    assert (labels[1, :long_prompt] == IGNORE_LABEL).all()
    assert (
        labels[1, long_prompt : long_prompt + long_answer] != IGNORE_LABEL
    ).any()


def test_multi_turn_includes_earlier_assistant_in_context() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="multi-1",
        conversation_turns=(
            ConversationTurn(role="user", text="Q1", turn_index=0),
            ConversationTurn(role="assistant", text="A1", turn_index=1),
            ConversationTurn(role="user", text="Q2", turn_index=2),
            ConversationTurn(
                role="assistant",
                text="A2",
                turn_index=3,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )
    seq = collator.build_decoder_sequence(sample)
    prompt = seq["prompt_tokens"]
    assistant_token_id = tokenizer.token_to_id["<assistant>"]
    assert prompt.count(assistant_token_id) == 2
    assert seq["prompt_token_count"] > 0
    assert seq["answer_token_count"] > 0
    assert seq["is_conversation"]


def test_tokenizer_matches_production_special_tokens() -> None:
    from config.multimodal.training_settings import (
        _DEFAULT_TEXT_SPECIAL_TOKEN_IDS,
    )

    tokenizer = _StubTokenizer()
    for token, expected_id in _DEFAULT_TEXT_SPECIAL_TOKEN_IDS.items():
        assert token in tokenizer.token_to_id, f"missing token: {token}"
        assert tokenizer.token_to_id[token] == expected_id, (
            f"wrong id for {token}: expected {expected_id}, "
            f"got {tokenizer.token_to_id[token]}"
        )
    for role_token in TextCollator._CONVERSATION_ROLE_TOKENS.values():
        assert role_token in _DEFAULT_TEXT_SPECIAL_TOKEN_IDS


def test_mixed_conversation_and_plain_samples_share_decoder_shape() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    conversation = MultimodalSample(
        sample_id="mixed-conversation",
        conversation_turns=(
            ConversationTurn(role="user", text="Hi", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="OK",
                turn_index=1,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )
    plain = MultimodalSample(
        sample_id="mixed-plain",
        text="plain text",
        task_type="text_pretrain",
    )

    batch = collator.collate_decoder_tensors((conversation, plain))
    assert batch["decoder_input_ids"].shape == (2, 8)
    assert batch["decoder_labels"].shape == (2, 8)
    assert batch["decoder_attention_mask"].shape == (2, 8)
    assert (batch["decoder_labels"][1] == IGNORE_LABEL).all()
    assert not batch["decoder_attention_mask"][1].any()


def test_assistant_delimiter_is_prompt_and_eos_is_target() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="assistant-contract",
        conversation_turns=(
            ConversationTurn(role="user", text="Hi", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="OK",
                turn_index=1,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )

    sequence = collator.build_decoder_sequence(sample)
    assert (
        sequence["prompt_tokens"][-1] == tokenizer.token_to_id["<assistant>"]
    )
    assert (
        tokenizer.token_to_id["<assistant>"] not in sequence["answer_tokens"]
    )
    assert sequence["answer_tokens"][-1] == tokenizer.token_to_id["<eos>"]

    batch = collator.collate_decoder_tensors((sample,))
    prompt_length = sequence["prompt_token_count"]
    assert (batch["decoder_labels"][0, :prompt_length] == IGNORE_LABEL).all()
    assert batch["decoder_labels"][0, prompt_length] != IGNORE_LABEL


def test_answer_larger_than_decoder_budget_is_rejected() -> None:
    tokenizer = _StubTokenizer()
    tokenizer.max_tokens = 8
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="oversized-answer",
        conversation_turns=(
            ConversationTurn(role="user", text="Q", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="answer-that-is-too-long",
                turn_index=1,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )

    with pytest.raises(
        ValueError,
        match="assistant answer exceeds decoder sequence budget",
    ):
        collator.build_decoder_sequence(sample)


def test_prompt_truncation_preserves_latest_user_turn() -> None:
    tokenizer = _StubTokenizer()
    tokenizer.max_tokens = 24
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="recent-context",
        conversation_turns=(
            ConversationTurn(
                role="system",
                text="S" * 80,
                turn_index=0,
            ),
            ConversationTurn(
                role="user",
                text="LATESTQUESTION",
                turn_index=1,
            ),
            ConversationTurn(
                role="assistant",
                text="OK",
                turn_index=2,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )

    sequence = collator.build_decoder_sequence(sample)
    latest_user_segment = [
        tokenizer.token_to_id["<user>"],
        *tokenizer.encode(
            "LATESTQUESTION",
            add_special_tokens=False,
            pad_to_max_length=False,
        ),
    ]
    prompt = sequence["prompt_tokens"]
    assert prompt[-(len(latest_user_segment) + 1) : -1] == latest_user_segment
    assert prompt[-1] == tokenizer.token_to_id["<assistant>"]
    assert sequence["answer_tokens"][-1] == tokenizer.token_to_id["<eos>"]


def test_multiple_marked_assistant_targets_are_rejected() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="multiple-targets",
        conversation_turns=(
            ConversationTurn(role="user", text="Q1", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="A1",
                turn_index=1,
                is_assistant_answer=True,
            ),
            ConversationTurn(role="user", text="Q2", turn_index=2),
            ConversationTurn(
                role="assistant",
                text="A2",
                turn_index=3,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )

    with pytest.raises(
        ValueError,
        match="only one marked assistant answer",
    ):
        collator.build_decoder_sequence(sample)


def test_turns_after_target_and_invalid_indexes_are_rejected() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    after_target = MultimodalSample(
        sample_id="turn-after-target",
        conversation_turns=(
            ConversationTurn(role="user", text="Q", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="A",
                turn_index=1,
                is_assistant_answer=True,
            ),
            ConversationTurn(role="user", text="late", turn_index=2),
        ),
        task_type="instruction_following",
    )
    with pytest.raises(ValueError, match="turns after the target answer"):
        collator.build_decoder_sequence(after_target)

    duplicate_indexes = MultimodalSample(
        sample_id="duplicate-indexes",
        conversation_turns=(
            ConversationTurn(role="user", text="Q", turn_index=0),
            ConversationTurn(
                role="assistant",
                text="A",
                turn_index=0,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )
    with pytest.raises(ValueError, match="turn_index values must be unique"):
        collator.build_decoder_sequence(duplicate_indexes)


def test_json_only_tool_call_and_result_turns_are_supported() -> None:
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    assistant_call = ConversationTurn(
        role="assistant",
        text="",
        turn_index=1,
        tool_name="weather",
        tool_arguments_json='{"city":"Ghent"}',
    )
    tool_result = ConversationTurn(
        role="tool",
        text="",
        turn_index=2,
        tool_result_json='{"temperature":18}',
    )

    assert assistant_call.text == ""
    assert assistant_call.tool_name == "weather"
    assert tool_result.text == ""
    assert tool_result.tool_result_json == '{"temperature":18}'


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "role": "assistant",
                "text": "",
                "tool_name": "weather",
                "tool_arguments_json": "not-json",
            },
            "tool_arguments_json must contain valid JSON",
        ),
        (
            {
                "role": "assistant",
                "text": "",
                "tool_name": "weather",
                "tool_arguments_json": "[]",
            },
            "tool_arguments_json must contain a JSON object",
        ),
        (
            {
                "role": "tool",
                "text": "",
                "tool_result_json": "not-json",
            },
            "tool_result_json must contain valid JSON",
        ),
    ],
)
def test_conversation_turn_rejects_invalid_tool_json(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    with pytest.raises(ValueError, match=message):
        ConversationTurn(turn_index=0, **kwargs)  # type: ignore[arg-type]


def test_conversation_properties_match_decoder_context() -> None:
    from mmcrawler_datasets.training_samples.targets import (
        ConversationTurn,
        TrainingTaskTarget,
    )

    turns = (
        ConversationTurn(role="user", text="Q1", turn_index=0),
        ConversationTurn(role="assistant", text="A1", turn_index=1),
        ConversationTurn(
            role="assistant",
            text="",
            turn_index=2,
            tool_name="lookup",
            tool_arguments_json='{"id":1}',
        ),
        ConversationTurn(
            role="tool",
            text="",
            turn_index=3,
            tool_result_json='{"value":"x"}',
        ),
        ConversationTurn(role="user", text="Q2", turn_index=4),
        ConversationTurn(
            role="assistant",
            text="A2",
            turn_index=5,
            is_assistant_answer=True,
        ),
    )
    sample = MultimodalSample(
        sample_id="conversation-properties",
        conversation_turns=turns,
        task_type="instruction_following",
    )
    target = TrainingTaskTarget(user_text="question only")

    prompt = sample.conversation_prompt_text
    assert "<assistant>\nA1" in prompt
    assert '<assistant>\n{"id":1}' in prompt
    assert '<tool>\n{"value":"x"}' in prompt
    assert "<user>\nQ2" in prompt
    assert "A2" not in prompt
    assert target.has_conversation


def test_supported_target_fields_cover_conversation_contract() -> None:
    from mmcrawler_datasets.schema import SUPPORTED_TARGET_FIELDS

    expected = {
        "system_text",
        "user_text",
        "assistant_text",
        "conversation_turns",
        "answer_evidence_ids",
        "tool_name",
        "tool_arguments_json",
        "tool_result_json",
        "sample_source",
        "generator_id",
        "generator_version",
        "verification_status",
    }
    assert expected <= SUPPORTED_TARGET_FIELDS


def test_decoder_tensor_payload_has_no_redundant_conversation_flag() -> None:
    tokenizer = _StubTokenizer()
    collator = TextCollator(
        tokenizer=tokenizer,
        mlm_probability=0.0,
        base_seed=0,
    )
    payload = collator.collate_decoder_tensors(
        (MultimodalSample(sample_id="plain", text="plain"),)
    )
    assert "conversation_flag" not in payload
    assert "conversation_flags" not in payload


def test_cross_modal_builders_derive_retrieval_and_consistency_rows() -> None:
    from mmcrawler_datasets.assembly.build import _build_cross_modal_samples
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    image_pair = TrainingSample(
        sample_id="image-pair",
        split="train",
        modality="image",
        object_id="image-1",
        text="red bicycle",
        content_family_id="family-image",
        task_target=TrainingTaskTarget(
            task_type="image_text_pair",
            task_family="image",
            target_text="red bicycle",
            positive_id="image-1",
        ),
    )
    audio_pair = TrainingSample(
        sample_id="audio-pair",
        split="train",
        modality="audio",
        object_id="audio-1",
        text="spoken bicycle",
        content_family_id="family-audio",
        task_target=TrainingTaskTarget(
            task_type="audio_text_pair",
            task_family="audio",
            target_text="spoken bicycle",
            positive_id="audio-1",
        ),
    )

    derived = _build_cross_modal_samples(
        samples=(image_pair, audio_pair),
        enabled_tasks=frozenset(
            {"multimodal_retrieval", "cross_modal_consistency"}
        ),
    )

    assert [sample.task_target.task_type for sample in derived].count(
        "multimodal_retrieval"
    ) == 2
    assert [sample.task_target.task_type for sample in derived].count(
        "cross_modal_consistency"
    ) == 2
    for sample in derived:
        assert sample.task_target.positive_id in {"image-1", "audio-1"}
        assert sample.task_target.negative_ids
        assert (
            sample.task_target.positive_id
            not in sample.task_target.negative_ids
        )


def test_text_only_tool_turn_is_valid_conversation_context() -> None:
    from mmcrawler_datasets.collation.multimodal import (
        _validate_conversation_samples,
    )
    from mmcrawler_datasets.training_samples.targets import ConversationTurn

    sample = MultimodalSample(
        sample_id="text-tool-result",
        conversation_turns=(
            ConversationTurn(role="user", text="Run it", turn_index=0),
            ConversationTurn(role="tool", text="completed", turn_index=1),
            ConversationTurn(
                role="assistant",
                text="Done",
                turn_index=2,
                is_assistant_answer=True,
            ),
        ),
        task_type="instruction_following",
    )

    _validate_conversation_samples(samples=(sample,))


def test_mapping_preserves_explicit_duplicate_zero_turn_indexes() -> None:
    from mmcrawler_datasets.training_samples.targets import (
        conversation_turns_from_mapping,
        validate_conversation_turns,
    )

    turns = conversation_turns_from_mapping(
        [
            {"role": "user", "text": "Q", "turn_index": 0},
            {
                "role": "assistant",
                "text": "A",
                "turn_index": 0,
                "is_assistant_answer": True,
            },
        ]
    )

    with pytest.raises(ValueError, match="turn_index values must be unique"):
        validate_conversation_turns(turns)
