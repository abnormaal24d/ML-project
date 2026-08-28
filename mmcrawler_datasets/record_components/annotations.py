"""Canonical structured annotation parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from mmcrawler_datasets.schema import (
    BoundingBox,
    ChartData,
    ChartSeries,
    GeometryAnnotation,
    LayoutBox,
    ObjectBox,
    ProsodyFeatures,
    SpeakerSegment,
    UIElement,
)

from .coercion import (
    optional_bool,
    optional_dict,
    optional_float,
    optional_int,
    optional_string,
    optional_table,
    str_tuple,
)

_BOX_COORDINATES = frozenset({"x", "y", "width", "height"})
_UNSUPPORTED_BOX_KEYS = frozenset({"bbox", "x_min", "y_min", "x_max", "y_max"})
_UNSUPPORTED_PROSODY_KEYS = frozenset(
    {"pitch_mean", "energy_mean", "speaking_rate"}
)


def _items(value: object, *, field_name: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _item(
    value: object,
    *,
    field_name: str,
    index: int,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name}[{index}] must be a JSON object")
    return value


def _reject_aliases(
    value: Mapping[str, object],
    *,
    aliases: frozenset[str],
    field_name: str,
    replacement: str,
) -> None:
    found = sorted(aliases.intersection(value))
    if found:
        names = ", ".join(found)
        raise ValueError(
            f"{field_name} uses unsupported key(s) {names}; use {replacement}"
        )


def _reject_inline_box(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> None:
    inline_keys = _BOX_COORDINATES | _UNSUPPORTED_BOX_KEYS
    if inline_keys.intersection(value):
        raise ValueError(
            f"{field_name} geometry must use the canonical nested box field"
        )


def box(value: object) -> BoundingBox | None:
    """Parse one canonical bounding-box object."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("box must be an object with x, y, width, and height")
    _reject_aliases(
        value,
        aliases=_UNSUPPORTED_BOX_KEYS,
        field_name="box",
        replacement="x, y, width, and height",
    )
    if "box" in value:
        raise ValueError(
            "box coordinates must be direct; remove the nested box wrapper"
        )
    missing = sorted(_BOX_COORDINATES.difference(value))
    if missing:
        raise ValueError(
            "box is missing canonical field(s): " + ", ".join(missing)
        )
    return BoundingBox(
        x=float(value["x"]),
        y=float(value["y"]),
        width=float(value["width"]),
        height=float(value["height"]),
        page=optional_int(value.get("page")),
        coordinate_system=(
            optional_string(value.get("coordinate_system")) or "relative"
        ),
    )


def parse_layout_boxes(value: object) -> tuple[LayoutBox, ...]:
    """Parse layout boxes with canonical nested geometry."""

    boxes: list[LayoutBox] = []
    for index, raw_item in enumerate(_items(value, field_name="layout_boxes")):
        item = _item(
            raw_item,
            field_name="layout_boxes",
            index=index,
        )
        _reject_inline_box(item, field_name=f"layout_boxes[{index}]")
        boxes.append(
            LayoutBox(
                text=optional_string(item.get("text")),
                box=box(item.get("box")),
                role=optional_string(item.get("role")),
                reading_order=optional_int(item.get("reading_order")),
                line_id=optional_string(item.get("line_id")),
                column_id=optional_string(item.get("column_id")),
                confidence=optional_float(item.get("confidence")),
            )
        )
    return tuple(boxes)


def parse_ui_elements(value: object) -> tuple[UIElement, ...]:
    """Parse UI elements using element_type and nested box fields."""

    elements: list[UIElement] = []
    for index, raw_item in enumerate(_items(value, field_name="ui_elements")):
        item = _item(raw_item, field_name="ui_elements", index=index)
        _reject_aliases(
            item,
            aliases=frozenset({"type"}),
            field_name=f"ui_elements[{index}]",
            replacement="element_type",
        )
        _reject_inline_box(item, field_name=f"ui_elements[{index}]")
        element_type = optional_string(item.get("element_type"))
        if element_type is None:
            raise ValueError(f"ui_elements[{index}] requires element_type")
        elements.append(
            UIElement(
                element_type=element_type,
                box=box(item.get("box")),
                label=optional_string(item.get("label")),
                attributes=optional_dict(item.get("attributes")),
                children=parse_ui_elements(item.get("children")),
            )
        )
    return tuple(elements)


def parse_geometry_annotations(
    value: object,
) -> tuple[GeometryAnnotation, ...]:
    """Parse geometry relations using canonical identifier fields."""

    annotations: list[GeometryAnnotation] = []
    for index, raw_item in enumerate(
        _items(value, field_name="geometry_annotations")
    ):
        item = _item(
            raw_item,
            field_name="geometry_annotations",
            index=index,
        )
        _reject_aliases(
            item,
            aliases=frozenset({"subject", "object"}),
            field_name=f"geometry_annotations[{index}]",
            replacement="subject_id and object_id",
        )
        subject_id = optional_string(item.get("subject_id"))
        relation = optional_string(item.get("relation"))
        if subject_id is None or relation is None:
            raise ValueError(
                f"geometry_annotations[{index}] requires subject_id "
                "and relation"
            )
        annotations.append(
            GeometryAnnotation(
                subject_id=subject_id,
                relation=relation,
                object_id=optional_string(item.get("object_id")),
                value=cast("float | str | None", item.get("value")),
                confidence=optional_float(item.get("confidence")),
            )
        )
    return tuple(annotations)


def parse_object_boxes(value: object) -> tuple[ObjectBox, ...]:
    """Parse object detections using label and nested box fields."""

    boxes: list[ObjectBox] = []
    for index, raw_item in enumerate(_items(value, field_name="object_boxes")):
        item = _item(raw_item, field_name="object_boxes", index=index)
        _reject_aliases(
            item,
            aliases=frozenset({"class"}),
            field_name=f"object_boxes[{index}]",
            replacement="label",
        )
        _reject_inline_box(item, field_name=f"object_boxes[{index}]")
        label = optional_string(item.get("label"))
        if label is None:
            raise ValueError(f"object_boxes[{index}] requires label")
        boxes.append(
            ObjectBox(
                object_id=(
                    optional_string(item.get("object_id")) or f"object_{index}"
                ),
                label=label,
                box=box(item.get("box")),
                confidence=optional_float(item.get("confidence")),
                attributes=optional_dict(item.get("attributes")),
            )
        )
    return tuple(boxes)


def parse_chart_data(value: object) -> ChartData | None:
    """Parse chart metadata with the canonical chart_type field."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("chart_data must be a JSON object")
    _reject_aliases(
        value,
        aliases=frozenset({"type"}),
        field_name="chart_data",
        replacement="chart_type",
    )
    raw_series = value.get("series")
    if raw_series is not None and not isinstance(raw_series, list):
        raise ValueError("chart_data.series must be a JSON array")
    series: list[ChartSeries] = []
    for index, raw_item in enumerate(raw_series or []):
        item = _item(raw_item, field_name="chart_data.series", index=index)
        raw_values = item.get("values")
        if not isinstance(raw_values, list):
            raise ValueError(
                f"chart_data.series[{index}].values must be a JSON array"
            )
        raw_labels = item.get("labels")
        if raw_labels is not None and not isinstance(raw_labels, list):
            raise ValueError(
                f"chart_data.series[{index}].labels must be a JSON array"
            )
        series.append(
            ChartSeries(
                name=(optional_string(item.get("name")) or f"series_{index}"),
                values=tuple(float(number) for number in raw_values),
                labels=tuple(str(label) for label in (raw_labels or [])),
            )
        )
    raw_legend = value.get("legend")
    if raw_legend is not None and not isinstance(raw_legend, list):
        raise ValueError("chart_data.legend must be a JSON array")
    return ChartData(
        chart_type=optional_string(value.get("chart_type")),
        title=optional_string(value.get("title")),
        x_axis=optional_string(value.get("x_axis")),
        y_axis=optional_string(value.get("y_axis")),
        legend=tuple(str(label) for label in (raw_legend or [])),
        series=tuple(series),
        values=optional_dict(value.get("values")),
    )


def parse_speaker_segments(value: object) -> tuple[SpeakerSegment, ...]:
    """Parse diarization with canonical speaker and time fields."""

    segments: list[SpeakerSegment] = []
    for index, raw_item in enumerate(
        _items(value, field_name="speaker_segments")
    ):
        item = _item(raw_item, field_name="speaker_segments", index=index)
        _reject_aliases(
            item,
            aliases=frozenset({"speaker", "start", "end"}),
            field_name=f"speaker_segments[{index}]",
            replacement="speaker_id, start_seconds, and end_seconds",
        )
        start = optional_float(item.get("start_seconds"))
        end = optional_float(item.get("end_seconds"))
        speaker_id = optional_string(item.get("speaker_id"))
        if start is None or end is None or speaker_id is None:
            raise ValueError(
                f"speaker_segments[{index}] requires speaker_id, "
                "start_seconds, and end_seconds"
            )
        if start < 0.0 or end < start:
            raise ValueError(
                f"speaker_segments[{index}] has invalid time bounds"
            )
        segments.append(
            SpeakerSegment(
                start_seconds=start,
                end_seconds=end,
                speaker_id=speaker_id,
                confidence=optional_float(item.get("confidence")),
                transcript=optional_string(item.get("transcript")),
                overlap=bool(optional_bool(item.get("overlap"))),
            )
        )
    return tuple(segments)


def parse_prosody(value: object) -> ProsodyFeatures | None:
    """Parse canonical prosody feature values."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("prosody must be a JSON object")
    _reject_aliases(
        value,
        aliases=_UNSUPPORTED_PROSODY_KEYS,
        field_name="prosody",
        replacement="pitch_hz, energy, and tempo",
    )
    emphasis = value.get("emphasis")
    if emphasis is not None and not isinstance(emphasis, list):
        raise ValueError("prosody.emphasis must be a JSON array")
    return ProsodyFeatures(
        pitch_hz=optional_float(value.get("pitch_hz")),
        energy=optional_float(value.get("energy")),
        tempo=optional_float(value.get("tempo")),
        pause_ratio=optional_float(value.get("pause_ratio")),
        emphasis=tuple(str(item) for item in (emphasis or [])),
    )


@dataclass(frozen=True, slots=True)
class ParsedAnnotations:
    layout_boxes: tuple[LayoutBox, ...] = ()
    ui_elements: tuple[UIElement, ...] = ()
    geometry_annotations: tuple[GeometryAnnotation, ...] = ()
    object_boxes: tuple[ObjectBox, ...] = ()
    chart_data: ChartData | None = None
    math_expression: str | None = None
    math_solution: str | None = None
    humor_explanation: str | None = None
    target_table_structure: str | dict[str, Any] | None = None
    form_fields: dict[str, Any] | None = None
    scene_graph: dict[str, Any] | None = None
    speaker_segments: tuple[SpeakerSegment, ...] = ()
    prosody: ProsodyFeatures | None = None
    target_code: str | None = None
    code_language: str | None = None
    output_modalities: tuple[str, ...] = ()
    emotion_label: str | None = None
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    sarcasm_label: str | None = None
    speaker_label: str | None = None
    background_noise_label: str | None = None
    overlapping_speech: bool | None = None


def parse_annotations(*, task_target: dict[str, Any]) -> ParsedAnnotations:
    """Parse all structured annotation and affect fields from task_target."""

    def field_value(key: str) -> object:
        return task_target.get(key)

    return ParsedAnnotations(
        layout_boxes=parse_layout_boxes(field_value("layout_boxes")),
        ui_elements=parse_ui_elements(field_value("ui_elements")),
        geometry_annotations=parse_geometry_annotations(
            field_value("geometry_annotations")
        ),
        object_boxes=parse_object_boxes(field_value("object_boxes")),
        chart_data=parse_chart_data(field_value("chart_data")),
        math_expression=optional_string(field_value("math_expression")),
        math_solution=optional_string(field_value("math_solution")),
        humor_explanation=optional_string(field_value("humor_explanation")),
        target_table_structure=optional_table(
            field_value("target_table_structure")
        ),
        form_fields=optional_dict(field_value("form_fields")),
        scene_graph=optional_dict(field_value("scene_graph")),
        speaker_segments=parse_speaker_segments(
            field_value("speaker_segments")
        ),
        prosody=parse_prosody(field_value("prosody")),
        target_code=optional_string(field_value("target_code")),
        code_language=optional_string(field_value("code_language")),
        output_modalities=str_tuple(field_value("output_modalities")),
        emotion_label=optional_string(field_value("emotion_label")),
        arousal=optional_float(field_value("arousal")),
        valence=optional_float(field_value("valence")),
        dominance=optional_float(field_value("dominance")),
        sarcasm_label=optional_string(field_value("sarcasm_label")),
        speaker_label=optional_string(field_value("speaker_label")),
        background_noise_label=optional_string(
            field_value("background_noise_label")
        ),
        overlapping_speech=optional_bool(field_value("overlapping_speech")),
    )
