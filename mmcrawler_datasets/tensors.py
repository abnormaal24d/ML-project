"""Raw media decoding and required materialized tensor loading."""

from __future__ import annotations

# Imported only to handle errors from weights_only=True safe loading.
import pickle  # nosec B403
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from mmcrawler_datasets.safe_io import resolve_dataset_reference
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image

# --- image_tensor_loader.py ---

MAX_DECODED_MEDIA_PIXELS = 40_000_000
MAX_ABSOLUTE_TENSOR_VALUE = 1_000_000.0
MAX_MATERIALIZED_TENSOR_BYTES = 512 * 1024 * 1024
MAX_MATERIALIZED_TENSOR_ELEMENTS = 100_000_000


class ImageDecodeError(RuntimeError):
    """Raised when an existing image cannot be decoded for training."""


@dataclass(frozen=True, slots=True)
class ImageTensorLoader:
    """Load RGB CHW float images at a fixed spatial size."""

    image_size: int

    def load(self, *, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise ImageDecodeError(
                "Pillow is required for image loading"
            ) from exc

        try:
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ImageDecodeError(
                        f"image has invalid dimensions: {path}"
                    )
                if width * height > MAX_DECODED_MEDIA_PIXELS:
                    raise ImageDecodeError(
                        f"image exceeds {MAX_DECODED_MEDIA_PIXELS} decoded pixels: "
                        f"{path}"
                    )
                rgb = _rgb_image(image=ImageOps.exif_transpose(image)).resize(
                    (self.image_size, self.image_size)
                )
                raw = torch.tensor(list(rgb.tobytes()), dtype=torch.float32)
                return (
                    raw.reshape(self.image_size, self.image_size, 3)
                    .permute(2, 0, 1)
                    .div(255.0)
                    .contiguous()
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ImageDecodeError(f"cannot decode image: {path}") from exc


def _rgb_image(*, image: Image.Image) -> Image.Image:
    from PIL import Image

    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and isinstance(image.info.get("transparency"), bytes)
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


# --- audio_tensor_loader.py ---


class AudioDecodeError(RuntimeError):
    """Raised when an existing audio file cannot be decoded."""


@dataclass(frozen=True, slots=True)
class AudioTensorLoader:
    """Load mono float waveforms at a fixed sample rate and length."""

    target_sample_rate: int
    num_samples: int

    def load(self, *, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            import soundfile as sf  # type: ignore
            import torchaudio.functional as audio_functional  # type: ignore
        except ImportError as exc:
            raise AudioDecodeError(
                "SoundFile and torchaudio are required for audio loading"
            ) from exc

        try:
            info = sf.info(str(path))
            source_sample_rate = int(info.samplerate)
            if source_sample_rate <= 0:
                raise AudioDecodeError(f"invalid audio sample rate: {path}")
            source_frames_needed = max(
                1,
                (self.num_samples * source_sample_rate)
                // self.target_sample_rate
                + 2,
            )
            samples, source_sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
                frames=source_frames_needed,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise AudioDecodeError(f"cannot decode audio: {path}") from exc

        if samples.size == 0:
            raise AudioDecodeError(f"empty audio file: {path}")

        # SoundFile: [time, channels] -> Torch: [channels, time].
        waveform = torch.from_numpy(samples).transpose(0, 1).contiguous()
        waveform = waveform.mean(dim=0, keepdim=True)

        if int(source_sample_rate) != self.target_sample_rate:
            waveform = audio_functional.resample(
                waveform,
                orig_freq=int(source_sample_rate),
                new_freq=self.target_sample_rate,
            )

        result = torch.zeros(1, self.num_samples, dtype=torch.float32)
        copied_samples = min(self.num_samples, int(waveform.shape[-1]))
        result[:, :copied_samples] = waveform[:, :copied_samples]
        return result.clamp(-1.0, 1.0)


# --- video_tensor_loader.py ---


class VideoDecodeError(RuntimeError):
    """Raised when an existing video cannot produce training frames."""


@dataclass(frozen=True, slots=True)
class VideoTensorLoader:
    """Load uniformly sampled RGB video frames as TCHW float tensors."""

    frame_count: int
    image_size: int

    def load(self, *, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise VideoDecodeError(
                "OpenCV and NumPy are required for video loading"
            ) from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodeError(f"cannot open video: {path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise VideoDecodeError(f"video has no frames: {path}")
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            source_width <= 0
            or source_height <= 0
            or source_width * source_height > MAX_DECODED_MEDIA_PIXELS
        ):
            capture.release()
            raise VideoDecodeError(
                f"video frame dimensions exceed rules: {path}"
            )

        indices = np.linspace(
            0,
            total_frames - 1,
            num=self.frame_count,
            dtype=np.int64,
        )

        frames: list[torch.Tensor] = []
        try:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if not success or frame is None:
                    raise VideoDecodeError(
                        f"cannot decode frame {frame_index} from {path}"
                    )
                if (
                    int(frame.shape[0]) * int(frame.shape[1])
                    > MAX_DECODED_MEDIA_PIXELS
                ):
                    raise VideoDecodeError(
                        f"decoded video frame exceeds pixel rules: {path}"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                square = self._resize_and_pad(rgb)
                frames.append(
                    torch.from_numpy(square)
                    .permute(2, 0, 1)
                    .to(dtype=torch.float32)
                    .div(255.0)
                )
        finally:
            capture.release()

        return torch.stack(frames)

    def _resize_and_pad(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
    ) -> NDArray[np.uint8]:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise VideoDecodeError("decoded frame has invalid dimensions")

        scale = min(self.image_size / width, self.image_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        output = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        top = (self.image_size - resized_height) // 2
        left = (self.image_size - resized_width) // 2
        output[
            top : top + resized_height,
            left : left + resized_width,
        ] = resized
        return output


def load_required_tensor(
    *,
    dataset_root: Path,
    path: Path | str | None,
    expected_shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load one materialized tensor or raise when missing/invalid."""

    if path is None:
        raise FileNotFoundError("materialized tensor path is missing")

    tensor_path = resolve_dataset_reference(
        dataset_root=dataset_root,
        reference=path,
        label="materialized tensor path",
        allow_absolute=True,
    )

    value = _load_bounded_tensor(tensor_path=tensor_path)

    expected = tuple(int(dim) for dim in expected_shape)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"invalid tensor shape for {tensor_path}: "
            f"expected {expected}, got {tuple(value.shape)}"
        )

    converted = value.to(dtype=dtype).contiguous()
    if converted.is_floating_point():
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(f"tensor contains NaN or Inf: {tensor_path}")
        if (
            converted.numel()
            and float(converted.abs().max()) > MAX_ABSOLUTE_TENSOR_VALUE
        ):
            raise ValueError(f"tensor values exceed rules: {tensor_path}")
    return converted


def load_required_token_tensor(
    *,
    dataset_root: Path,
    path: Path | str | None,
) -> torch.Tensor:
    """Load a bounded materialized token tensor beneath the dataset root."""

    if path is None:
        raise FileNotFoundError("materialized token tensor path is missing")
    tensor_path = resolve_dataset_reference(
        dataset_root=dataset_root,
        reference=path,
        label="materialized token tensor path",
        allow_absolute=True,
    )
    value = _load_bounded_tensor(tensor_path=tensor_path)
    if value.numel() == 0:
        raise ValueError(f"materialized token tensor is empty: {tensor_path}")
    if value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(
            f"materialized token tensor must use an integer dtype: {tensor_path}"
        )
    if bool((value < 0).any()):
        raise ValueError(
            f"materialized token tensor contains negative ids: {tensor_path}"
        )
    return value.to(dtype=torch.long).contiguous()


def _load_bounded_tensor(*, tensor_path: Path) -> torch.Tensor:
    try:
        byte_size = tensor_path.stat().st_size
        if byte_size <= 0 or byte_size > MAX_MATERIALIZED_TENSOR_BYTES:
            raise ValueError(
                f"materialized tensor byte size violates rules: {tensor_path}"
            )
        value = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise FileNotFoundError(
            f"materialized tensor is missing or unreadable: {tensor_path}"
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"not a tensor: {tensor_path}")
    if value.numel() > MAX_MATERIALIZED_TENSOR_ELEMENTS:
        raise ValueError(
            f"materialized tensor has too many elements: {tensor_path}"
        )
    if value.is_floating_point():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"tensor contains NaN or Inf: {tensor_path}")
        if (
            value.numel()
            and float(value.abs().max()) > MAX_ABSOLUTE_TENSOR_VALUE
        ):
            raise ValueError(f"tensor values exceed rules: {tensor_path}")
    return value


class SampleTensorSource(Protocol):
    @property
    def dataset_root(self) -> Path: ...

    def image_tensor(self, *, sample: MultimodalSample) -> torch.Tensor: ...

    def audio_tensor(self, *, sample: MultimodalSample) -> torch.Tensor: ...

    def video_tensor(self, *, sample: MultimodalSample) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class MaterializedTensorSource:
    dataset_root: Path
    image_size: int
    audio_num_samples: int
    video_frames: int

    def image_tensor(self, *, sample: MultimodalSample) -> torch.Tensor:
        if not sample.has_image:
            return torch.zeros(
                (3, self.image_size, self.image_size),
                dtype=torch.float32,
            )
        return load_required_tensor(
            dataset_root=self.dataset_root,
            path=sample.image_tensor_path,
            expected_shape=(3, self.image_size, self.image_size),
            dtype=torch.float32,
        )

    def audio_tensor(self, *, sample: MultimodalSample) -> torch.Tensor:
        if not sample.has_audio:
            return torch.zeros(
                (1, self.audio_num_samples),
                dtype=torch.float32,
            )
        return load_required_tensor(
            dataset_root=self.dataset_root,
            path=sample.audio_tensor_path,
            expected_shape=(1, self.audio_num_samples),
            dtype=torch.float32,
        )

    def video_tensor(self, *, sample: MultimodalSample) -> torch.Tensor:
        if not sample.has_video:
            return torch.zeros(
                (
                    self.video_frames,
                    3,
                    self.image_size,
                    self.image_size,
                ),
                dtype=torch.float32,
            )
        return load_required_tensor(
            dataset_root=self.dataset_root,
            path=sample.video_tensor_path,
            expected_shape=(
                self.video_frames,
                3,
                self.image_size,
                self.image_size,
            ),
            dtype=torch.float32,
        )


def load_optional_tensor_batch(
    *,
    dataset_root: Path,
    paths: Sequence[Path | None],
    expected_shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Load optional materialized rows while preserving batch alignment."""

    if not any(path is not None for path in paths):
        return None
    shape = tuple(int(dimension) for dimension in expected_shape)
    rows: list[torch.Tensor] = [
        (
            load_required_tensor(
                dataset_root=dataset_root,
                path=path,
                expected_shape=shape,
                dtype=dtype,
            )
            if path is not None
            else torch.zeros(shape, dtype=dtype)
        )
        for path in paths
    ]
    return torch.stack(rows)
