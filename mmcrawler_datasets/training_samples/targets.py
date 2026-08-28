"""Typed task-target schema and conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from .common import as_opt_float, as_opt_str, coerce_bool

CONVERSATION_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant", "tool"}
)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One explicit role-marked conversation turn."""

    role: str
    text: str
    turn_index: int = 0
    tool_name: str | None = None
    tool_arguments_json: str | None = None
    tool_result_json: str | None = None
    answer_evidence_ids: tuple[str, ...] = ()
    is_assistant_answer: bool = False

    def __post_init__(self) -> None:
        role = self.role.strip().lower()
        if role not in CONVERSATION_ROLES:
            raise ValueError(f"unsupported conversation role: {self.role!r}")
        object.__setattr__(self, "role", role)
        text = str(self.text or "").strip()
        object.__setattr__(self, "text", text)
        tool_name = _clean_optional_text(self.tool_name)
        tool_arguments_json = _validated_json_text(
            self.tool_arguments_json,
            field_name="tool_arguments_json",
            require_object=True,
        )
        tool_result_json = _validated_json_text(
            self.tool_result_json,
            field_name="tool_result_json",
            require_object=False,
        )
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_arguments_json", tool_arguments_json)
        object.__setattr__(self, "tool_result_json", tool_result_json)

        if role in {"system", "user"} and not text:
            raise ValueError(
                f"{role} conversation turn text must be non-empty"
            )
        if role == "assistant" and not text:
            if not (tool_name and tool_arguments_json):
                raise ValueError(
                    "assistant conversation turn requires text or a tool call"
                )
        if role == "tool" and not text and not tool_result_json:
            raise ValueError(
                "tool conversation turn requires text or tool_result_json"
            )
        if role != "assistant" and (tool_name or tool_arguments_json):
            raise ValueError(
                "tool_name and tool_arguments_json are only valid on "
                "assistant turns"
            )
        if role != "tool" and tool_result_json is not None:
            raise ValueError("tool_result_json is only valid on tool turns")
        if self.turn_index < 0:
            raise ValueError("conversation turn_index must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ConversationTurn:
        return cls(
            role=as_opt_str(raw.get("role")) or "user",
            text=as_opt_str(raw.get("text")) or "",
            turn_index=int(as_opt_float(raw.get("turn_index")) or 0),
            tool_name=as_opt_str(raw.get("tool_name")),
            tool_arguments_json=as_opt_str(raw.get("tool_arguments_json")),
            tool_result_json=as_opt_str(raw.get("tool_result_json")),
            answer_evidence_ids=_str_tuple(raw.get("answer_evidence_ids")),
            is_assistant_answer=bool(
                coerce_bool(raw.get("is_assistant_answer")) or False
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "role": self.role,
            "text": self.text,
            "turn_index": self.turn_index,
        }
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.tool_arguments_json is not None:
            payload["tool_arguments_json"] = self.tool_arguments_json
        if self.tool_result_json is not None:
            payload["tool_result_json"] = self.tool_result_json
        if self.answer_evidence_ids:
            payload["answer_evidence_ids"] = list(self.answer_evidence_ids)
        if self.is_assistant_answer:
            payload["is_assistant_answer"] = True
        return payload


def validate_conversation_turns(
    turns: Sequence[ConversationTurn],
    *,
    sample_id: str | None = None,
) -> int:
    """Validate turn ordering and return the sole assistant target index."""

    label = f" for sample {sample_id!r}" if sample_id is not None else ""
    if not turns:
        raise ValueError(f"conversation requires at least one turn{label}")

    indexes = [int(turn.turn_index) for turn in turns]
    if len(set(indexes)) != len(indexes):
        raise ValueError(
            f"conversation turn_index values must be unique{label}"
        )
    if any(
        current <= previous
        for previous, current in zip(indexes, indexes[1:], strict=False)
    ):
        raise ValueError(
            f"conversation turn_index values must be strictly increasing{label}"
        )

    invalid_targets = [
        turn.turn_index
        for turn in turns
        if turn.is_assistant_answer and turn.role != "assistant"
    ]
    if invalid_targets:
        raise ValueError(
            "is_assistant_answer may only be set on assistant turns"
            f"{label}: {invalid_targets}"
        )

    marked_targets = [
        index
        for index, turn in enumerate(turns)
        if turn.role == "assistant" and turn.is_assistant_answer
    ]
    if len(marked_targets) > 1:
        raise ValueError(
            f"conversation may contain only one marked assistant answer{label}"
        )

    if marked_targets:
        target_index = marked_targets[0]
    else:
        target_index = next(
            (
                index
                for index in range(len(turns) - 1, -1, -1)
                if turns[index].role == "assistant"
            ),
            -1,
        )
    if target_index < 0:
        raise ValueError(f"conversation requires an assistant answer{label}")
    if target_index != len(turns) - 1:
        raise ValueError(
            f"conversation may not contain turns after the target answer{label}"
        )
    return target_index


def conversation_turns_from_mapping(
    raw: object,
) -> tuple[ConversationTurn, ...]:
    """Parse a conversation_turns JSON array into typed turns."""
    if not isinstance(raw, (list, tuple)):
        return ()
    parsed: list[ConversationTurn] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        turn = ConversationTurn.from_mapping(item)
        raw_turn_index = item.get("turn_index")
        turn = ConversationTurn(
            role=turn.role,
            text=turn.text,
            turn_index=(
                int(raw_turn_index) if raw_turn_index is not None else index
            ),
            tool_name=turn.tool_name,
            tool_arguments_json=turn.tool_arguments_json,
            tool_result_json=turn.tool_result_json,
            answer_evidence_ids=turn.answer_evidence_ids,
            is_assistant_answer=turn.is_assistant_answer,
        )
        parsed.append(turn)
    return tuple(parsed)


def conversation_turns_to_mapping(
    turns: tuple[ConversationTurn, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(turn.to_mapping() for turn in turns)


@dataclass(frozen=True, slots=True)
class TrainingTaskTarget:
    """Typed task schema for one training sample.

    Parsing and coercion happen once at construction. Consumers read fields
    directly from this object instead of through sample-level property wrappers.
    """

    task_type: str = "representation"
    task_family: str | None = None
    target_text: str | None = None
    instruction: str | None = None
    question: str | None = None
    answer: str | None = None
    target_audio_path: str | None = None
    target_audio_tokens_path: str | None = None
    target_image_path: str | None = None
    target_video_path: str | None = None
    target_video_tokens_path: str | None = None
    target_video_token_metadata_path: str | None = None
    video_token_schema: str | None = None
    source_image_path: str | None = None
    edit_mask_path: str | None = None
    target_code: str | None = None
    code_language: str | None = None
    output_modalities: tuple[str, ...] = ()
    positive_id: str | None = None
    negative_ids: tuple[str, ...] = ()
    alignment_score: float = 1.0
    layout_boxes: tuple[dict[str, object], ...] = ()
    ui_elements: tuple[dict[str, object], ...] = ()
    geometry_annotations: tuple[dict[str, object], ...] = ()
    object_boxes: tuple[dict[str, object], ...] = ()
    chart_data: dict[str, object] | None = None
    math_expression: str | None = None
    math_solution: str | None = None
    humor_explanation: str | None = None
    target_table_structure: str | dict[str, object] | None = None
    form_fields: dict[str, object] | None = None
    scene_graph: dict[str, object] | None = None
    speaker_segments: tuple[dict[str, object], ...] = ()
    prosody: dict[str, object] | None = None
    emotion_label: str | None = None
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    sarcasm_label: str | None = None
    speaker_label: str | None = None
    background_noise_label: str | None = None
    overlapping_speech: bool | None = None
    system_text: str | None = None
    user_text: str | None = None
    assistant_text: str | None = None
    conversation_turns: tuple[ConversationTurn, ...] = ()
    answer_evidence_ids: tuple[str, ...] = ()
    tool_name: str | None = None
    tool_arguments_json: str | None = None
    tool_result_json: str | None = None
    sample_source: str | None = None
    generator_id: str | None = None
    generator_version: str | None = None
    verification_status: str | None = None
    evidence_records: tuple[dict[str, object], ...] = ()
    reading_order: tuple[str, ...] = ()
    ocr_confidences: tuple[float, ...] = ()
    negative_verification: tuple[dict[str, object], ...] = ()
    chosen_response: str | None = None
    rejected_response: str | None = None
    preference_reason: str | None = None
    preference_source: str | None = None
    must_refuse: bool | None = None
    requires_uncertainty: bool | None = None
    requires_source_citation: bool | None = None
    prompt_injection_present: bool | None = None
    untrusted_document_instruction: bool | None = None
    sensitive_data_present: bool | None = None
    requires_tool_confirmation: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_audio_path",
            "target_audio_tokens_path",
            "target_image_path",
            "target_video_path",
            "target_video_tokens_path",
            "target_video_token_metadata_path",
            "source_image_path",
            "edit_mask_path",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"{name} must be a contained project-relative path"
                )
        chosen = _clean_optional_text(self.chosen_response)
        rejected = _clean_optional_text(self.rejected_response)
        object.__setattr__(self, "chosen_response", chosen)
        object.__setattr__(self, "rejected_response", rejected)
        if (chosen is None) != (rejected is None):
            raise ValueError(
                "chosen_response and rejected_response must be provided together"
            )
        if chosen is not None and chosen == rejected:
            raise ValueError(
                "chosen_response and rejected_response must differ"
            )

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, object] | None
    ) -> TrainingTaskTarget:
        """Parse a schema task_target mapping into a typed object."""

        if not isinstance(raw, Mapping):
            return cls()

        return cls(
            task_type=as_opt_str(raw.get("task_type")) or "representation",
            task_family=as_opt_str(raw.get("task_family")),
            target_text=as_opt_str(raw.get("target_text")),
            instruction=as_opt_str(raw.get("instruction")),
            question=as_opt_str(raw.get("question")),
            answer=as_opt_str(raw.get("answer")),
            target_audio_path=as_opt_str(raw.get("target_audio_path")),
            target_audio_tokens_path=as_opt_str(
                raw.get("target_audio_tokens_path")
            ),
            target_image_path=as_opt_str(raw.get("target_image_path")),
            target_video_path=as_opt_str(raw.get("target_video_path")),
            target_video_tokens_path=as_opt_str(
                raw.get("target_video_tokens_path")
            ),
            target_video_token_metadata_path=as_opt_str(
                raw.get("target_video_token_metadata_path")
            ),
            video_token_schema=as_opt_str(raw.get("video_token_schema")),
            source_image_path=as_opt_str(raw.get("source_image_path")),
            edit_mask_path=as_opt_str(raw.get("edit_mask_path")),
            target_code=as_opt_str(raw.get("target_code")),
            code_language=as_opt_str(raw.get("code_language")),
            output_modalities=_str_tuple(raw.get("output_modalities")),
            positive_id=as_opt_str(raw.get("positive_id")),
            negative_ids=_str_tuple(raw.get("negative_ids")),
            alignment_score=_alignment_score(raw.get("alignment_score")),
            layout_boxes=_dict_tuple(raw.get("layout_boxes")),
            ui_elements=_dict_tuple(raw.get("ui_elements")),
            geometry_annotations=_dict_tuple(raw.get("geometry_annotations")),
            object_boxes=_dict_tuple(raw.get("object_boxes")),
            chart_data=_dict_or_none(raw.get("chart_data")),
            math_expression=as_opt_str(raw.get("math_expression")),
            math_solution=as_opt_str(raw.get("math_solution")),
            humor_explanation=as_opt_str(raw.get("humor_explanation")),
            target_table_structure=_table_structure(
                raw.get("target_table_structure")
            ),
            form_fields=_dict_or_none(raw.get("form_fields")),
            scene_graph=_dict_or_none(raw.get("scene_graph")),
            speaker_segments=_dict_tuple(raw.get("speaker_segments")),
            prosody=_dict_or_none(raw.get("prosody")),
            emotion_label=as_opt_str(raw.get("emotion_label")),
            arousal=as_opt_float(raw.get("arousal")),
            valence=as_opt_float(raw.get("valence")),
            dominance=as_opt_float(raw.get("dominance")),
            sarcasm_label=as_opt_str(raw.get("sarcasm_label")),
            speaker_label=as_opt_str(raw.get("speaker_label")),
            background_noise_label=as_opt_str(
                raw.get("background_noise_label")
            ),
            overlapping_speech=coerce_bool(raw.get("overlapping_speech")),
            system_text=as_opt_str(raw.get("system_text")),
            user_text=as_opt_str(raw.get("user_text")),
            assistant_text=as_opt_str(raw.get("assistant_text")),
            conversation_turns=conversation_turns_from_mapping(
                raw.get("conversation_turns")
            ),
            answer_evidence_ids=_str_tuple(raw.get("answer_evidence_ids")),
            tool_name=as_opt_str(raw.get("tool_name")),
            tool_arguments_json=as_opt_str(raw.get("tool_arguments_json")),
            tool_result_json=as_opt_str(raw.get("tool_result_json")),
            sample_source=as_opt_str(raw.get("sample_source")),
            generator_id=as_opt_str(raw.get("generator_id")),
            generator_version=as_opt_str(raw.get("generator_version")),
            verification_status=as_opt_str(raw.get("verification_status")),
            evidence_records=_dict_tuple(raw.get("evidence_records")),
            reading_order=_str_tuple(raw.get("reading_order")),
            ocr_confidences=_float_tuple(raw.get("ocr_confidences")),
            negative_verification=_dict_tuple(
                raw.get("negative_verification")
            ),
            chosen_response=as_opt_str(raw.get("chosen_response")),
            rejected_response=as_opt_str(raw.get("rejected_response")),
            preference_reason=as_opt_str(raw.get("preference_reason")),
            preference_source=as_opt_str(raw.get("preference_source")),
            must_refuse=coerce_bool(raw.get("must_refuse")),
            requires_uncertainty=coerce_bool(raw.get("requires_uncertainty")),
            requires_source_citation=coerce_bool(
                raw.get("requires_source_citation")
            ),
            prompt_injection_present=coerce_bool(
                raw.get("prompt_injection_present")
            ),
            untrusted_document_instruction=coerce_bool(
                raw.get("untrusted_document_instruction")
            ),
            sensitive_data_present=coerce_bool(
                raw.get("sensitive_data_present")
            ),
            requires_tool_confirmation=coerce_bool(
                raw.get("requires_tool_confirmation")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the sparse schema-compatible task_target mapping."""

        defaults = TrainingTaskTarget()
        payload: dict[str, object] = {"task_type": self.task_type}
        for item in fields(self):
            if item.name == "task_type":
                continue
            value = getattr(self, item.name)
            if item.name == "conversation_turns":
                if value:
                    payload[item.name] = list(
                        conversation_turns_to_mapping(value)
                    )
                continue
            if value == getattr(defaults, item.name):
                continue
            payload[item.name] = value
        return payload

    @property
    def has_conversation(self) -> bool:
        """Return whether the target carries explicit conversation turns."""

        return bool(
            self.conversation_turns
            or self.system_text
            or self.user_text
            or self.assistant_text
        )

    @property
    def answer_text(self) -> str | None:
        """Return the primary generative answer text when present."""

        for value in (
            self.assistant_text,
            self.target_text,
            self.answer,
            self.target_code,
            self.math_solution,
            self.humor_explanation,
        ):
            if value is not None and str(value).strip():
                return str(value).strip()
        return None


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validated_json_text(
    value: str | None,
    *,
    field_name: str,
    require_object: bool,
) -> str | None:
    text = _clean_optional_text(value)
    if text is None:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must contain valid JSON") from exc
    if require_object and not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return text


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


def _float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[float] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            rows.append(float(item))
        except (TypeError, ValueError):
            continue
    return tuple(rows)


def _dict_tuple(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _dict_or_none(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _alignment_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    return float(value)


def _table_structure(
    value: object,
) -> str | dict[str, object] | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        return dict(value)
    return None
