"""Transform video spatial annotations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from augmentation.video.annotations.annotation_receipt import (
    VideoAnnotationReceipt,
)
from augmentation.video.annotations.annotation_value_parsing import (
    finite_number as _finite_number,
)
from augmentation.video.video_transform_backend import SpatialTransform
from mmcrawler_datasets.schema import (
    BoundingBox,
    LayoutBox,
    ObjectBox,
    UIElement,
)


def transform_bounding_box(
    *,
    box: BoundingBox,
    spatial: SpatialTransform,
) -> BoundingBox | None:
    """Map a relative or absolute rectangle through letterbox/crop geometry."""

    coordinate_system = box.coordinate_system.strip().lower()
    relative_coordinates = coordinate_system in {
        "relative",
        "normalized",
        "normalised",
    }
    if relative_coordinates:
        x = box.x * spatial.source_width
        y = box.y * spatial.source_height
        width = box.width * spatial.source_width
        height = box.height * spatial.source_height
    elif coordinate_system in {"absolute", "pixel", "pixels"}:
        x, y, width, height = box.x, box.y, box.width, box.height
    else:
        raise ValueError(
            f"unsupported_video_box_coordinate_system:{box.coordinate_system}"
        )
    if width <= 0.0 or height <= 0.0:
        return None
    left = x * spatial.scale + spatial.offset_x
    top = y * spatial.scale + spatial.offset_y
    right = (x + width) * spatial.scale + spatial.offset_x
    bottom = (y + height) * spatial.scale + spatial.offset_y
    left = min(max(0.0, left), float(spatial.output_width))
    top = min(max(0.0, top), float(spatial.output_height))
    right = min(max(0.0, right), float(spatial.output_width))
    bottom = min(max(0.0, bottom), float(spatial.output_height))
    if right <= left or bottom <= top:
        return None
    if relative_coordinates:
        return replace(
            box,
            x=left / spatial.output_width,
            y=top / spatial.output_height,
            width=(right - left) / spatial.output_width,
            height=(bottom - top) / spatial.output_height,
        )
    return replace(
        box,
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
        coordinate_system="absolute",
    )


def _transform_layout_boxes(
    values: tuple[LayoutBox, ...],
    spatial: SpatialTransform,
) -> tuple[tuple[LayoutBox, ...], VideoAnnotationReceipt]:
    output: list[LayoutBox] = []
    transformed = dropped = 0
    for item in values:
        if item.box is None:
            output.append(item)
            continue
        box = transform_bounding_box(box=item.box, spatial=spatial)
        if box is None:
            dropped += 1
            continue
        output.append(replace(item, box=box))
        transformed += 1
    return tuple(output), VideoAnnotationReceipt(
        transformed_boxes=transformed, dropped_boxes=dropped
    )


def _transform_object_boxes(
    values: tuple[ObjectBox, ...],
    spatial: SpatialTransform,
) -> tuple[tuple[ObjectBox, ...], VideoAnnotationReceipt]:
    output: list[ObjectBox] = []
    transformed = dropped = 0
    for item in values:
        if item.box is None:
            output.append(item)
            continue
        box = transform_bounding_box(box=item.box, spatial=spatial)
        if box is None:
            dropped += 1
            continue
        output.append(replace(item, box=box))
        transformed += 1
    return tuple(output), VideoAnnotationReceipt(
        transformed_boxes=transformed, dropped_boxes=dropped
    )


def _transform_ui_elements(
    values: tuple[UIElement, ...],
    spatial: SpatialTransform,
) -> tuple[tuple[UIElement, ...], VideoAnnotationReceipt]:
    output: list[UIElement] = []
    receipt = VideoAnnotationReceipt()
    for item in values:
        children, child_receipt = _transform_ui_elements(
            item.children, spatial
        )
        receipt = receipt.merge(child_receipt)
        box = item.box
        if box is not None:
            box = transform_bounding_box(box=box, spatial=spatial)
            if box is None:
                receipt = receipt.merge(
                    VideoAnnotationReceipt(dropped_boxes=1)
                )
                continue
            receipt = receipt.merge(
                VideoAnnotationReceipt(transformed_boxes=1)
            )
        output.append(replace(item, box=box, children=children))
    return tuple(output), receipt


def _read_box(value: Mapping[str, object]) -> BoundingBox | None:
    candidate: Mapping[str, object] = value
    nested_key = next(
        (
            key
            for key in ("box", "bbox", "bounding_box")
            if isinstance(value.get(key), Mapping)
        ),
        None,
    )
    if nested_key is not None:
        nested = value[nested_key]
        if not isinstance(nested, Mapping):
            return None
        candidate = nested
    if not all(key in candidate for key in ("x", "y", "width", "height")):
        return None
    return BoundingBox(
        x=_finite_number(candidate.get("x"), field="box.x"),
        y=_finite_number(candidate.get("y"), field="box.y"),
        width=_finite_number(candidate.get("width"), field="box.width"),
        height=_finite_number(candidate.get("height"), field="box.height"),
        page=_read_page(candidate.get("page")),
        coordinate_system=str(
            candidate.get("coordinate_system") or "relative"
        ),
    )


def _write_box(
    *, result: dict[str, Any], source: Mapping[str, object], box: BoundingBox
) -> None:
    payload = {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "page": box.page,
        "coordinate_system": box.coordinate_system,
    }
    nested_key = next(
        (
            key
            for key in ("box", "bbox", "bounding_box")
            if isinstance(source.get(key), Mapping)
        ),
        None,
    )
    if nested_key is not None:
        original = source[nested_key]
        if not isinstance(original, Mapping):
            raise ValueError("video_annotation_nested_box_invalid")
        result[nested_key] = {**original, **payload}
    else:
        result.update(payload)


_BOX_FIELDS = frozenset(
    {"x", "y", "width", "height", "box", "bbox", "bounding_box"}
)


def _read_page(value: object) -> int | None:
    if value is None:
        return None
    return int(_finite_number(value, field="box.page"))
