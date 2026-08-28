from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from augmentation.generated_artifact_cache import AugmentationCache
from augmentation.video.annotations.annotation_assembler import (
    transform_video_clip_sample,
)
from augmentation.video.annotations.spatial_annotation_mapping import (
    transform_bounding_box,
)
from augmentation.video.ffmpeg_video_transform import (
    FfmpegVideoTransformBackend,
)
from augmentation.video.video_clip_augmenter import VideoClipAugmenter
from augmentation.video.video_keyframe_augmenter import VideoKeyframeAugmenter
from augmentation.video.video_transform import (
    SpatialTransform,
    TimelineTransform,
    deterministic_crop_start,
)
from config.augmentation.video_settings import VideoAugmentationSettings
from config.media_toolchain import MediaToolchainSettings
from mmcrawler_datasets.schema import (
    BoundingBox,
    LayoutBox,
    ModalityObject,
    MultimodalSample,
    ObjectBox,
    SpeakerSegment,
    UIElement,
)
from mmcrawler_datasets.training_samples.snapshot_mapping import (
    build_snapshot_sample,
    serialize_snapshot_sample,
)
from preprocessing.media.adapters.versioned_executable import (
    resolve_and_verify_executable,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe required for video augmentation integration tests",
)


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        pass

    def warning(self, *args: object, **kwargs: object) -> None:
        pass


@lru_cache(maxsize=1)
def _resolved_media_tools() -> tuple[str, str, str, str]:
    ffmpeg, ffmpeg_version = resolve_and_verify_executable(
        tool_name="ffmpeg",
        configured_executable="ffmpeg",
        expected_version=None,
        timeout_seconds=60.0,
        required=True,
    )
    ffprobe, ffprobe_version = resolve_and_verify_executable(
        tool_name="ffprobe",
        configured_executable="ffprobe",
        expected_version=None,
        timeout_seconds=60.0,
        required=True,
    )
    assert ffmpeg_version is not None
    assert ffprobe_version is not None
    return ffmpeg, ffmpeg_version, ffprobe, ffprobe_version


def _video_settings(**overrides: object) -> VideoAugmentationSettings:
    ffmpeg, ffmpeg_version, ffprobe, ffprobe_version = _resolved_media_tools()
    return VideoAugmentationSettings(
        enabled=True,
        probe_timeout_seconds=60.0,
        **overrides,
    )


def _toolchain(**overrides: object) -> MediaToolchainSettings:
    ffmpeg, ffmpeg_version, ffprobe, ffprobe_version = _resolved_media_tools()
    return MediaToolchainSettings(
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
        ffmpeg_expected_version=ffmpeg_version,
        ffprobe_expected_version=ffprobe_version,
        **overrides,
    )


def _cache(tmp_path: Path) -> AugmentationCache:
    return AugmentationCache(
        enabled=False,
        cache_directory=tmp_path / "cache",
        logger=_Logger(),
    )


def _run_ffmpeg(*args: str) -> None:
    ffmpeg, _, _, _ = _resolved_media_tools()
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _source_with_audio(
    path: Path,
    *,
    duration: float = 4.0,
    audio_duration: float | None = None,
) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=10:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=16000:duration={audio_duration or duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    )


def _red_blue_source(path: Path) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x90:r=10:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x90:r=10:d=2",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )


def _sample(path: Path) -> MultimodalSample:
    return MultimodalSample(
        sample_id="video-source",
        task_type="video_text_pair",
        text="A source video",
        video=ModalityObject(path=path.name, mime_type="video/mp4"),
        object_boxes=(
            ObjectBox(
                object_id="object-1",
                label="subject",
                box=BoundingBox(
                    x=0.1,
                    y=0.2,
                    width=0.3,
                    height=0.4,
                    coordinate_system="relative",
                ),
            ),
        ),
        speaker_segments=(
            SpeakerSegment(start_seconds=0.5, end_seconds=1.5, speaker_id="a"),
            SpeakerSegment(start_seconds=2.0, end_seconds=3.5, speaker_id="b"),
        ),
        task_target={
            "transcript_segments": [
                {"start_seconds": 0.5, "end_seconds": 1.2, "text": "first"},
                {"start_seconds": 2.2, "end_seconds": 3.8, "text": "second"},
            ],
            "frame_indices": [0, 10, 20, 30],
            "timestamps": [0.0, 1.0, 2.0, 3.0],
            "tracked_box": {
                "box": {
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.4,
                    "coordinate_system": "relative",
                }
            },
        },
        metadata={"video_duration_seconds": 4.0, "fps": 10.0},
    )


def test_clip_uses_real_crop_fps_audio_letterbox_and_annotation_mapping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _source_with_audio(source)
    settings = _video_settings(
        max_clip_duration_seconds=2.0,
        output_fps=12.0,
        output_width=128,
        output_height=128,
        resize_mode="letterbox",
        audio_policy="preserve",
        temporal_crop_offset_mode="center",
    )
    augmenter = VideoClipAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    )

    produced, rejected = augmenter.augment(
        sample=_sample(source),
        dataset_root=tmp_path,
    )

    assert rejected == ()
    assert len(produced) == 1
    variant = produced[0][1]
    assert variant.video is not None and variant.video.path is not None
    backend = FfmpegVideoTransformBackend(
        toolchain=_toolchain(),
        settings=settings,
    )
    probe = backend.probe(path=variant.video.path)
    assert probe.width == 128
    assert probe.height == 128
    assert probe.fps == pytest.approx(12.0, abs=0.05)
    assert probe.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert probe.has_audio is True
    backend.decode_check(path=variant.video.path, timeout_seconds=30.0)

    box = variant.object_boxes[0].box
    assert box is not None
    assert box.x == pytest.approx(0.1)
    assert box.y == pytest.approx(0.33125)
    assert box.width == pytest.approx(0.3)
    assert box.height == pytest.approx(0.225)
    assert variant.speaker_segments[0].start_seconds == pytest.approx(0.0)
    assert variant.speaker_segments[0].end_seconds == pytest.approx(0.5)
    assert variant.speaker_segments[1].start_seconds == pytest.approx(1.0)
    assert variant.speaker_segments[1].end_seconds == pytest.approx(2.0)
    assert variant.task_target["frame_indices"] == [0, 12, 23]
    assert variant.task_target["timestamps"] == [0.0, 1.0, 2.0]
    assert variant.metadata["video_duration_seconds"] == pytest.approx(2.0)
    assert variant.metadata["fps"] == pytest.approx(12.0)
    receipt = variant.metadata["augmentation_video_transform"]
    assert receipt["timeline"]["crop_start_seconds"] == pytest.approx(1.0)
    assert receipt["timeline"]["crop_end_seconds"] == pytest.approx(3.0)
    assert receipt["output_audio_status"] == "preserved_transcoded"


def test_audio_remove_policy_is_explicit_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _source_with_audio(source, duration=2.0)
    settings = _video_settings(
        max_clip_duration_seconds=1.5,
        output_width=160,
        output_height=90,
        output_fps=10.0,
        audio_policy="remove",
        temporal_crop_offset_mode="start",
    )
    produced, rejected = VideoClipAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    ).augment(sample=_sample(source), dataset_root=tmp_path)

    assert rejected == ()
    variant = produced[0][1]
    assert variant.video is not None and variant.video.path is not None
    probe = FfmpegVideoTransformBackend(
        toolchain=_toolchain(),
        settings=settings,
    ).probe(path=variant.video.path)
    assert probe.has_audio is False
    assert variant.video.metadata["audio_status"] == "removed"
    assert (
        variant.metadata["augmentation_video_transform"]["audio_policy"]
        == "remove"
    )


def test_keyframe_is_selected_by_timestamp_and_decoded(tmp_path: Path) -> None:
    source = tmp_path / "colors.mp4"
    _red_blue_source(source)
    settings = _video_settings(
        output_width=128,
        output_height=128,
        resize_mode="letterbox",
        keyframe_timestamp_fraction=0.75,
    )
    sample = _sample(source)
    produced, rejected = VideoKeyframeAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    ).augment(sample=sample, dataset_root=tmp_path)

    assert rejected == ()
    variant = produced[0][1]
    assert variant.video is None
    assert variant.task_type == "image_text_pair"
    assert variant.task_family == "image"
    assert variant.output_modalities == ("embedding",)
    assert variant.task_target["task_type"] == "image_text_pair"
    assert variant.task_target["task_family"] == "image"
    assert variant.task_target["output_modalities"] == ["embedding"]
    assert variant.image is not None and variant.image.path is not None
    with Image.open(variant.image.path) as image:
        image.load()
        assert image.size == (128, 128)
        mean = ImageStat.Stat(image.convert("RGB")).mean
    assert mean[2] > mean[0] * 2.0
    assert variant.image.metadata[
        "keyframe_timestamp_seconds"
    ] == pytest.approx(3.0, abs=0.1)
    assert variant.speaker_segments == ()
    assert "transcript_segments" not in variant.task_target
    assert variant.task_target[
        "selected_keyframe_timestamp_seconds"
    ] == pytest.approx(3.0, abs=0.1)
    assert variant.metadata["augmentation_video_validation"]["decoded"] is True

    serialized = serialize_snapshot_sample(
        sample=variant,
        dataset_root=tmp_path,
    )
    assert serialized["modality"] == "image"
    assert all(item["role"] != "video" for item in serialized["objects"])
    reparsed = build_snapshot_sample(
        payload=serialized,
        dataset_root=tmp_path,
        source_path=tmp_path / "augmented.jsonl",
        line_number=1,
    )
    assert reparsed.task_type == "image_text_pair"
    assert reparsed.task_family == "image"
    assert reparsed.output_modalities == ("embedding",)
    assert reparsed.video is None
    assert reparsed.image is not None


def test_preserved_short_audio_does_not_shorten_video_timeline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "short-audio.mp4"
    _source_with_audio(source, duration=4.0, audio_duration=1.0)
    settings = _video_settings(
        max_clip_duration_seconds=3.0,
        output_width=160,
        output_height=90,
        output_fps=10.0,
        audio_policy="preserve",
        temporal_crop_offset_mode="start",
    )
    produced, rejected = VideoClipAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    ).augment(sample=_sample(source), dataset_root=tmp_path)

    assert rejected == ()
    variant = produced[0][1]
    assert variant.video is not None and variant.video.path is not None
    probe = FfmpegVideoTransformBackend(
        toolchain=_toolchain(),
        settings=settings,
    ).probe(path=variant.video.path)
    assert probe.duration_seconds == pytest.approx(3.0, abs=0.1)
    assert probe.has_audio is True


def test_keyframe_blocks_tasks_that_require_temporal_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _source_with_audio(source, duration=2.0)
    sample = MultimodalSample(
        sample_id="video-qa",
        task_type="video_qa",
        video=ModalityObject(path=source.name, mime_type="video/mp4"),
    )
    settings = _video_settings()
    produced, rejected = VideoKeyframeAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    ).augment(sample=sample, dataset_root=tmp_path)

    assert produced == ()
    assert rejected[0].reason == "video_keyframe_task_not_supported"


def test_video_settings_reject_odd_yuv420p_dimensions() -> None:
    with pytest.raises(ValueError, match="must be even"):
        _video_settings(output_width=127)


def test_keyframe_filters_timestamped_boxes_to_selected_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "colors-tracked.mp4"
    _red_blue_source(source)
    sample = MultimodalSample(
        sample_id="tracked-video",
        task_type="video_text_pair",
        video=ModalityObject(path=source.name, mime_type="video/mp4"),
        task_target={
            "tracks": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.5,
                    "box": {
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "coordinate_system": "relative",
                    },
                    "label": "early",
                },
                {
                    "start_seconds": 2.5,
                    "end_seconds": 3.8,
                    "box": {
                        "x": 0.2,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.3,
                        "coordinate_system": "relative",
                    },
                    "label": "late",
                },
            ]
        },
    )
    settings = _video_settings(
        output_width=128,
        output_height=128,
        keyframe_timestamp_fraction=0.75,
    )
    produced, rejected = VideoKeyframeAugmenter(
        settings=settings,
        max_input_bytes=20_000_000,
        cache=_cache(tmp_path),
        backend=FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ),
        logger=_Logger(),
    ).augment(sample=sample, dataset_root=tmp_path)

    assert rejected == ()
    tracks = produced[0][1].task_target["tracks"]
    assert len(tracks) == 1
    assert tracks[0]["label"] == "late"
    assert "start_seconds" not in tracks[0]
    assert "end_seconds" not in tracks[0]
    assert tracks[0]["box"]["y"] > 0.2


def test_center_crop_transforms_and_clips_absolute_boxes() -> None:
    spatial = SpatialTransform.build(
        source_width=160,
        source_height=90,
        output_width=100,
        output_height=100,
        mode="center_crop",
    )
    transformed = transform_bounding_box(
        box=BoundingBox(
            x=40,
            y=9,
            width=80,
            height=72,
            coordinate_system="absolute",
        ),
        spatial=spatial,
    )
    assert transformed is not None
    assert transformed.coordinate_system == "absolute"
    assert 0.0 <= transformed.x < 100.0
    assert 0.0 <= transformed.y < 100.0
    assert transformed.x + transformed.width <= 100.0
    assert transformed.y + transformed.height <= 100.0


def test_spatial_mapping_accepts_normalized_boxes() -> None:
    spatial = SpatialTransform.build(
        source_width=160,
        source_height=90,
        output_width=100,
        output_height=100,
        mode="letterbox",
    )

    transformed = transform_bounding_box(
        box=BoundingBox(
            x=0.25,
            y=0.2,
            width=0.5,
            height=0.4,
            coordinate_system="normalized",
        ),
        spatial=spatial,
    )

    assert transformed is not None
    assert transformed.coordinate_system == "normalized"
    assert 0.0 <= transformed.x < 1.0
    assert 0.0 <= transformed.y < 1.0
    assert transformed.x + transformed.width <= 1.0
    assert transformed.y + transformed.height <= 1.0


def test_normalized_boxes_transform_through_nested_and_ui_annotations() -> (
    None
):
    spatial = SpatialTransform.build(
        source_width=160,
        source_height=90,
        output_width=100,
        output_height=100,
        mode="letterbox",
    )
    timeline = TimelineTransform(
        crop_start_seconds=0.0,
        crop_end_seconds=1.0,
        output_duration_seconds=1.0,
        source_fps=10.0,
        output_fps=10.0,
    )
    sample = MultimodalSample(
        sample_id="normalized-annotations",
        task_type="video_text_pair",
        text="sample",
        layout_boxes=(
            LayoutBox(
                box=BoundingBox(
                    x=0.1,
                    y=0.2,
                    width=0.2,
                    height=0.3,
                    coordinate_system="normalized",
                )
            ),
        ),
        ui_elements=(
            UIElement(
                element_type="panel",
                box=BoundingBox(
                    x=0.2,
                    y=0.2,
                    width=0.4,
                    height=0.4,
                    coordinate_system="normalised",
                ),
                children=(
                    UIElement(
                        element_type="button",
                        box=BoundingBox(
                            x=0.3,
                            y=0.3,
                            width=0.1,
                            height=0.1,
                            coordinate_system="normalized",
                        ),
                    ),
                ),
            ),
        ),
        task_target={
            "tracked": {
                "bounding_box": {
                    "x": 0.2,
                    "y": 0.1,
                    "width": 0.3,
                    "height": 0.2,
                    "coordinate_system": "normalized",
                }
            }
        },
    )

    transformed, receipt = transform_video_clip_sample(
        sample=sample,
        spatial=spatial,
        timeline=timeline,
    )

    assert receipt.transformed_boxes == 4
    assert transformed.layout_boxes[0].box is not None
    assert transformed.layout_boxes[0].box.coordinate_system == "normalized"
    assert transformed.ui_elements[0].box is not None
    assert transformed.ui_elements[0].box.coordinate_system == "normalised"
    assert transformed.ui_elements[0].children[0].box is not None
    assert (
        transformed.task_target["tracked"]["bounding_box"]["coordinate_system"]
        == "normalized"
    )


def test_deterministic_crop_offset_is_stable_and_in_range() -> None:
    first = deterministic_crop_start(
        duration_seconds=20.0,
        clip_duration_seconds=5.0,
        seed_hex="1234567890abcdef",
        mode="deterministic",
    )
    second = deterministic_crop_start(
        duration_seconds=20.0,
        clip_duration_seconds=5.0,
        seed_hex="1234567890abcdef",
        mode="deterministic",
    )
    assert first == second
    assert 0.0 <= first <= 15.0
    assert deterministic_crop_start(
        duration_seconds=20.0,
        clip_duration_seconds=5.0,
        seed_hex="abcdef1234567890",
        mode="center",
    ) == pytest.approx(7.5)


def test_decode_validation_rejects_corrupt_output(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not-a-video")
    settings = _video_settings()
    with pytest.raises(RuntimeError, match="video_backend_failed"):
        FfmpegVideoTransformBackend(
            toolchain=_toolchain(),
            settings=settings,
        ).decode_check(
            path=corrupt,
            timeout_seconds=10.0,
        )
