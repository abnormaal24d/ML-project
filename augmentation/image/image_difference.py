"""Perceptual difference between a generated image and its source."""

from __future__ import annotations

from pathlib import Path


def image_difference(source: Path, output: Path) -> float:
    from PIL import Image, ImageChops, ImageStat

    with Image.open(source) as a, Image.open(output) as b:
        geometry = 1.0 if a.size != b.size else 0.0
        source_gray = a.convert("L").resize((64, 64))
        output_gray = b.convert("L").resize((64, 64))
        visual = (
            sum(
                ImageStat.Stat(
                    ImageChops.difference(source_gray, output_gray)
                ).mean
            )
            / 255.0
        )
        return max(geometry, visual)
