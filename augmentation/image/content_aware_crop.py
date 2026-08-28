"""Deterministic annotation-aware and entropy-aware crop selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class CropWindow:
    left: int
    top: int
    width: int
    height: int
    score: float
    strategy: str
    annotation_coverage: float


def select_crop_windows(
    *,
    image: Image,
    sample: MultimodalSample,
    width: int,
    height: int,
    candidate_count: int,
    variant_count: int,
    minimum_annotation_coverage: float,
    strategy: str,
    seed_key: str,
) -> tuple[CropWindow, ...]:
    source_width, source_height = image.size
    width, height = min(width, source_width), min(height, source_height)
    boxes = tuple(_sample_boxes(sample, source_width, source_height))
    candidates = _candidate_windows(
        source_width, source_height, width, height, candidate_count, seed_key
    )
    ranked: list[CropWindow] = []
    if strategy == "annotation_aware" and boxes:
        for left, top in candidates:
            coverage = _coverage(boxes, left, top, width, height)
            if coverage >= minimum_annotation_coverage:
                ranked.append(
                    CropWindow(
                        left,
                        top,
                        width,
                        height,
                        coverage,
                        "annotation_aware",
                        coverage,
                    )
                )
    elif strategy in {"annotation_aware", "entropy"}:
        gray = image.convert("L")
        for left, top in candidates:
            entropy = gray.crop(
                (left, top, left + width, top + height)
            ).entropy()
            ranked.append(
                CropWindow(left, top, width, height, entropy, "entropy", 1.0)
            )
    else:
        raise ValueError(f"unsupported crop strategy: {strategy}")
    ranked.sort(key=lambda item: (-item.score, item.top, item.left))
    selected: list[CropWindow] = []
    for item in ranked:
        if all(_window_iou(item, existing) < 0.85 for existing in selected):
            selected.append(item)
        if len(selected) >= variant_count:
            break
    return tuple(selected)


def _candidate_windows(
    sw: int, sh: int, width: int, height: int, count: int, seed_key: str
) -> tuple[tuple[int, int], ...]:
    max_left, max_top = max(0, sw - width), max(0, sh - height)
    positions = {
        (0, 0),
        (max_left, 0),
        (0, max_top),
        (max_left, max_top),
        (max_left // 2, max_top // 2),
    }
    for index in range(max(0, count - len(positions))):
        digest = hashlib.sha256(f"{seed_key}:{index}".encode()).digest()
        left = int.from_bytes(digest[:8], "big") % (max_left + 1)
        top = int.from_bytes(digest[8:16], "big") % (max_top + 1)
        positions.add((left, top))
    return tuple(sorted(positions))


def _sample_boxes(
    sample: MultimodalSample, sw: int, sh: int
) -> Iterable[tuple[float, float, float, float]]:
    values = [
        *(item.box for item in sample.layout_boxes),
        *(item.box for item in sample.object_boxes),
    ]
    stack = list(sample.ui_elements)
    while stack:
        item = stack.pop()
        values.append(item.box)
        stack.extend(item.children)
    for box in values:
        if box is None:
            continue
        relative = box.coordinate_system.lower() in {
            "relative",
            "normalized",
            "normalised",
        }
        x = box.x * sw if relative else box.x
        y = box.y * sh if relative else box.y
        w = box.width * sw if relative else box.width
        h = box.height * sh if relative else box.height
        if w > 0 and h > 0:
            yield x, y, x + w, y + h


def _coverage(
    boxes: tuple[tuple[float, float, float, float], ...],
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    right, bottom = left + width, top + height
    total = visible = 0.0
    for x1, y1, x2, y2 in boxes:
        area = max(0, x2 - x1) * max(0, y2 - y1)
        total += area
        visible += max(0, min(x2, right) - max(x1, left)) * max(
            0, min(y2, bottom) - max(y1, top)
        )
    return visible / total if total else 1.0


def _window_iou(a: CropWindow, b: CropWindow) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.left + a.width, b.left + b.width)
    bottom = min(a.top + a.height, b.top + b.height)
    inter = max(0, right - left) * max(0, bottom - top)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0
