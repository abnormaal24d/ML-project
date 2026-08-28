"""Spatial annotation transforms shared by image and document augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, overload

from mmcrawler_datasets.schema import (
    BoundingBox,
    LayoutBox,
    ObjectBox,
    UIElement,
)


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    """Affine resize/crop transform between two raster coordinate spaces."""

    source_width: int
    source_height: int
    output_width: int
    output_height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    crop_left: float = 0.0
    crop_top: float = 0.0
    minimum_visible_fraction: float = 0.05

    def receipt(self) -> dict[str, object]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "crop_left": self.crop_left,
            "crop_top": self.crop_top,
            "minimum_visible_fraction": self.minimum_visible_fraction,
        }

    def transform_box(self, box: BoundingBox | None) -> BoundingBox | None:
        if box is None:
            return None
        relative = box.coordinate_system.lower() in {
            "relative",
            "normalized",
            "normalised",
        }
        x = box.x * self.source_width if relative else box.x
        y = box.y * self.source_height if relative else box.y
        width = box.width * self.source_width if relative else box.width
        height = box.height * self.source_height if relative else box.height
        original_area = max(0.0, width) * max(0.0, height)
        if original_area <= 0:
            return None

        left = x * self.scale_x - self.crop_left
        top = y * self.scale_y - self.crop_top
        right = (x + width) * self.scale_x - self.crop_left
        bottom = (y + height) * self.scale_y - self.crop_top
        clipped_left = min(max(left, 0.0), float(self.output_width))
        clipped_top = min(max(top, 0.0), float(self.output_height))
        clipped_right = min(max(right, 0.0), float(self.output_width))
        clipped_bottom = min(max(bottom, 0.0), float(self.output_height))
        clipped_width = clipped_right - clipped_left
        clipped_height = clipped_bottom - clipped_top
        transformed_area = max(0.0, (right - left) * (bottom - top))
        visible_area = max(0.0, clipped_width * clipped_height)
        if clipped_width <= 0 or clipped_height <= 0:
            return None
        if (
            transformed_area > 0
            and visible_area / transformed_area < self.minimum_visible_fraction
        ):
            return None

        if relative:
            return replace(
                box,
                x=clipped_left / self.output_width,
                y=clipped_top / self.output_height,
                width=clipped_width / self.output_width,
                height=clipped_height / self.output_height,
            )
        return replace(
            box,
            x=clipped_left,
            y=clipped_top,
            width=clipped_width,
            height=clipped_height,
        )


def transform_layout_boxes(
    boxes: tuple[LayoutBox, ...], transform: SpatialTransform
) -> tuple[LayoutBox, ...]:
    result = []
    for item in boxes:
        box = transform.transform_box(item.box)
        if item.box is None or box is not None:
            result.append(replace(item, box=box))
    return tuple(result)


def transform_object_boxes(
    boxes: tuple[ObjectBox, ...], transform: SpatialTransform
) -> tuple[ObjectBox, ...]:
    result = []
    for item in boxes:
        box = transform.transform_box(item.box)
        if item.box is None or box is not None:
            result.append(replace(item, box=box))
    return tuple(result)


def transform_ui_elements(
    elements: tuple[UIElement, ...], transform: SpatialTransform
) -> tuple[UIElement, ...]:
    result = []
    for item in elements:
        box = transform.transform_box(item.box)
        children = transform_ui_elements(item.children, transform)
        if item.box is None or box is not None:
            result.append(replace(item, box=box, children=children))
    return tuple(result)


@overload
def transform_mapping(
    value: dict[str, Any], transform: SpatialTransform
) -> dict[str, Any]: ...


@overload
def transform_mapping(
    value: object, transform: SpatialTransform
) -> object: ...


def transform_mapping(value: object, transform: SpatialTransform) -> object:
    """Recursively transform canonical nested ``box`` mappings."""

    if isinstance(value, Mapping):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = str(key)
            if normalized_key.lower() == "box" and isinstance(child, Mapping):
                parsed = _mapping_box(child)
                mapped = transform.transform_box(parsed)
                if mapped is None:
                    transformed[normalized_key] = None
                else:
                    transformed[normalized_key] = _box_mapping(mapped)
            else:
                transformed[normalized_key] = transform_mapping(
                    child, transform
                )
        return transformed
    if isinstance(value, tuple):
        return tuple(transform_mapping(item, transform) for item in value)
    if isinstance(value, list):
        return [transform_mapping(item, transform) for item in value]
    return value


def _mapping_box(value: Mapping[object, object]) -> BoundingBox:
    return BoundingBox(
        x=_numeric_value(value.get("x", 0.0), field="box.x"),
        y=_numeric_value(value.get("y", 0.0), field="box.y"),
        width=_numeric_value(value.get("width", 0.0), field="box.width"),
        height=_numeric_value(value.get("height", 0.0), field="box.height"),
        page=_page_value(value.get("page")),
        coordinate_system=str(value.get("coordinate_system", "relative")),
    )


def _numeric_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _page_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("box.page must be an integer")
    return value


def _box_mapping(box: BoundingBox) -> dict[str, object]:
    result: dict[str, object] = {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "coordinate_system": box.coordinate_system,
    }
    if box.page is not None:
        result["page"] = box.page
    return result
