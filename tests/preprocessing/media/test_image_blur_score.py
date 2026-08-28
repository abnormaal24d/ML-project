"""Image blur resource-lifecycle regressions."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from preprocessing.media.image.image_blur_score import (
    ImageBlurEstimator,
    PillowGrayscaleArrayConverter,
)


class _ClosableImage:
    def __init__(self, *, mode: str = "L") -> None:
        self.mode = mode
        self.close_calls = 0
        self.converted: _ClosableImage | None = None
        self.convert_error: BaseException | None = None

    def convert(self, mode: str) -> _ClosableImage:
        assert mode == "L"
        if self.convert_error is not None:
            raise self.convert_error
        if self.converted is None:
            raise AssertionError("converted image was not configured")
        return self.converted

    def close(self) -> None:
        self.close_calls += 1


class _ImageLoader:
    def __init__(self, image: _ClosableImage) -> None:
        self._image = image

    def open_image(self, *, body: bytes) -> _ClosableImage:
        assert body == b"image"
        return self._image


class _FrameProcessor:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error

    def laplacian_variance(self, grayscale: object) -> float:
        assert grayscale == "grayscale"
        if self._error is not None:
            raise self._error
        return 12.5


def test_estimator_closes_original_image_once() -> None:
    image = _ClosableImage(mode="L")
    converter = SimpleNamespace(
        to_grayscale_array=lambda *, img: "grayscale",
    )
    estimator = ImageBlurEstimator(
        image_loader=_ImageLoader(image),
        grayscale_converter=converter,
        frame_processor=_FrameProcessor(),
    )

    score = estimator.estimate_blur(body=b"image")

    assert score is not None
    assert score.laplacian_variance == 12.5
    assert image.close_calls == 1


def test_estimator_closes_original_image_when_conversion_raises() -> None:
    image = _ClosableImage(mode="L")

    def fail_conversion(*, img: object) -> object:
        del img
        raise RuntimeError("conversion failed")

    estimator = ImageBlurEstimator(
        image_loader=_ImageLoader(image),
        grayscale_converter=SimpleNamespace(
            to_grayscale_array=fail_conversion,
        ),
        frame_processor=_FrameProcessor(),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        estimator.estimate_blur(body=b"image")

    assert image.close_calls == 1


def test_converter_closes_separate_grayscale_image_on_array_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ClosableImage(mode="RGB")
    converted = _ClosableImage(mode="L")
    original.converted = converted

    def fail_array(_image: object) -> object:
        raise ValueError("array conversion failed")

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(array=fail_array),
    )

    result = PillowGrayscaleArrayConverter().to_grayscale_array(img=original)

    assert result is None
    assert converted.close_calls == 1
    assert original.close_calls == 0


def test_estimator_and_converter_do_not_double_close_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ClosableImage(mode="RGB")
    converted = _ClosableImage(mode="L")
    original.converted = converted
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(array=lambda _image: "grayscale"),
    )
    estimator = ImageBlurEstimator(
        image_loader=_ImageLoader(original),
        grayscale_converter=PillowGrayscaleArrayConverter(),
        frame_processor=_FrameProcessor(
            error=RuntimeError("laplacian failed")
        ),
    )

    with pytest.raises(RuntimeError, match="laplacian failed"):
        estimator.estimate_blur(body=b"image")

    assert converted.close_calls == 1
    assert original.close_calls == 1
