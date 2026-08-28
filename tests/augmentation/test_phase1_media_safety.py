from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import augmentation.video.ffmpeg_video_transform as ffmpeg_transform
from augmentation.audio.audio_augmenter import AudioAugmenter
from augmentation.augmentation_run_accumulator import AugmentationRunState
from augmentation.augmentation_run_quality import (
    assess_augmentation_run_quality,
)
from augmentation.document.document_augmenter import DocumentAugmenter
from augmentation.document.document_page_augmenter import DocumentPageAugmenter
from augmentation.document.document_text_augmenter import DocumentTextAugmenter
from augmentation.generated_artifact_cache import AugmentationCache
from augmentation.image.image_augmenter import ImageAugmenter
from augmentation.image.image_operation_executor import ImageOperationExecutor
from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
    file_sha256,
    media_variant_id,
)
from augmentation.video.ffmpeg_video_transform import (
    FfmpegVideoTransformBackend,
)
from augmentation.video.video_clip_augmenter import VideoClipAugmenter
from augmentation.video.video_keyframe_augmenter import VideoKeyframeAugmenter
from augmentation.video.video_transform import VideoProbe
from config.augmentation.audio_settings import AudioAugmentationSettings
from config.augmentation.document_settings import DocumentAugmentationSettings
from config.augmentation.image_settings import ImageAugmentationSettings
from config.augmentation.video_settings import VideoAugmentationSettings
from config.media_toolchain import MediaToolchainSettings
from mmcrawler_datasets.schema import (
    ModalityObject,
    MultimodalSample,
    ProsodyFeatures,
)


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        pass

    def warning(self, *args: object, **kwargs: object) -> None:
        pass


def _enabled_video_settings() -> VideoAugmentationSettings:
    return VideoAugmentationSettings(
        enabled=True,
        probe_timeout_seconds=60.0,
    )


def _enabled_toolchain() -> MediaToolchainSettings:
    return MediaToolchainSettings(
        ffmpeg_expected_version="8.1.2",
        ffprobe_expected_version="8.1.2",
    )


class _NeverCalled:
    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError(
            "backend must not be called for unsafe annotations"
        )

    def __getattr__(self, name: str):
        del name
        return self

    def is_available(self) -> bool:
        raise AssertionError(
            "backend must not be queried for unsafe annotations"
        )


class _Cache(AugmentationCache):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            enabled=False, cache_directory=tmp_path, logger=_Logger()
        )


def test_temporal_and_mask_augmenters_block_non_transformable_annotations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"media")
    audio_sample = MultimodalSample(
        sample_id="audio-source",
        audio=ModalityObject(path=source, mime_type="audio/wav", byte_size=5),
        prosody=ProsodyFeatures(energy=0.5),
    )
    video_sample = MultimodalSample(
        sample_id="video-source",
        video=ModalityObject(path=source, mime_type="video/mp4", byte_size=5),
        task_target={"polygon": [[0.0, 0.0], [1.0, 1.0]]},
    )

    audio = AudioAugmenter(
        settings=AudioAugmentationSettings(enabled=True),
        decoder=_NeverCalled(),
        cache=_Cache(tmp_path),
        validate_input=_NeverCalled(),
        validate_output=_NeverCalled(),
        max_duration_seconds=10.0,
        logger=_Logger(),
    )
    video_clip = VideoClipAugmenter(
        settings=_enabled_video_settings(),
        max_input_bytes=100,
        cache=_Cache(tmp_path),
        backend=_NeverCalled(),
        logger=_Logger(),
    )
    video_keyframe = VideoKeyframeAugmenter(
        settings=_enabled_video_settings(),
        max_input_bytes=100,
        cache=_Cache(tmp_path),
        backend=_NeverCalled(),
        logger=_Logger(),
    )

    # Canonical image/document boxes are now transformed. Temporal prosody
    # and unsupported video polygons remain fail-closed.
    for augmenter, sample in (
        (audio, audio_sample),
        (video_clip, video_sample),
        (video_keyframe, video_sample),
    ):
        variants, rejections = augmenter.augment(
            sample=sample,
            dataset_root=tmp_path,
        )
        assert variants == ()
        assert rejections[0].reason == "media_annotations_not_transformable"


def test_ffmpeg_backend_uses_configured_executables_and_probe_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    settings = _enabled_video_settings()
    toolchain = _enabled_toolchain()
    calls: list[tuple[list[str], float]] = []
    resolutions: list[dict[str, object]] = []

    def resolve_tool(**kwargs: object) -> tuple[str, str]:
        # Map new parameter names to old for test assertions
        mapped = {
            "tool_name": kwargs.get("tool_name"),
            "configured": kwargs.get("configured_executable"),
            "expected_version": kwargs.get("expected_version"),
            "timeout_seconds": kwargs.get("timeout_seconds"),
            "required": kwargs.get("required"),
        }
        resolutions.append(mapped)
        return str(kwargs["configured_executable"]), str(
            kwargs["expected_version"]
        )

    def run(
        *,
        command: list[str],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, timeout_seconds))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                '{"streams":[{"codec_type":"video","width":16,'
                '"height":16,"avg_frame_rate":"24/1",'
                '"duration":"2.0","nb_frames":"48"}],'
                '"format":{"duration":"2.0"}}'
            ),
            stderr="",
        )

    # Monkeypatch the function in the module where it's used
    monkeypatch.setattr(
        "augmentation.video.ffmpeg_video_transform.resolve_and_verify_executable",
        resolve_tool,
    )
    monkeypatch.setattr(ffmpeg_transform, "_run", run)

    probe = FfmpegVideoTransformBackend(
        toolchain=toolchain,
        settings=settings,
    ).probe(path=source)

    assert probe.width == 16
    assert probe.fps == 24.0
    assert resolutions == [
        {
            "tool_name": "ffmpeg",
            "configured": "ffmpeg",
            "expected_version": "8.1.2",
            "timeout_seconds": 60.0,
            "required": True,
        },
        {
            "tool_name": "ffprobe",
            "configured": "ffprobe",
            "expected_version": "8.1.2",
            "timeout_seconds": 60.0,
            "required": True,
        },
    ]
    assert calls == [
        (
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            60.0,
        )
    ]


def test_media_identity_changes_with_config_and_is_bound_to_implementation() -> (
    None
):
    source_hash = hashlib.sha256(b"source").hexdigest()
    config_a = hashlib.sha256(b"config-a").hexdigest()
    config_b = hashlib.sha256(b"config-b").hexdigest()

    first = media_variant_id(
        source_sample_id="source-1",
        operation="image_media_transform",
        source_sha256=source_hash,
        config_hash=config_a,
    )
    second = media_variant_id(
        source_sample_id="source-1",
        operation="image_media_transform",
        source_sha256=source_hash,
        config_hash=config_b,
    )

    assert first != second
    assert len(MEDIA_AUGMENTATION_IMPLEMENTATION_HASH) == 64
    assert MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION == "augmentation-media-v4"


def test_quality_checks_verify_files_hashes_lineage_and_variant_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "objects" / "image" / "augmented" / "variant.webp"
    output.parent.mkdir(parents=True)
    source.write_bytes(b"source-bytes")
    output.write_bytes(b"output-bytes")
    source_hash = file_sha256(path=source)
    output_hash = file_sha256(path=output)
    config_hash = hashlib.sha256(b"image-config").hexdigest()
    variant_id = media_variant_id(
        source_sample_id="source-sample",
        operation="image_media_transform",
        source_sha256=source_hash,
        config_hash=config_hash,
    )
    variant = MultimodalSample(
        sample_id=variant_id,
        image=ModalityObject(path=output, mime_type="image/webp"),
        metadata={
            "augmentation_media_transform_applied": True,
            "augmentation_name": "image_media_transform",
            "augmentation_source_sample_id": "source-sample",
            "augmentation_variant_id": variant_id,
            "augmentation_source_path": source.as_posix(),
            "augmentation_output_path": output.relative_to(
                tmp_path
            ).as_posix(),
            "augmentation_source_sha256": source_hash,
            "augmentation_output_sha256": output_hash,
            "augmentation_config_hash": config_hash,
            "augmentation_implementation_version": MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
            "augmentation_implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
            "output_modality": "image",
        },
    )

    assessment = assess_augmentation_run_quality(
        dataset=(variant,),
        dataset_root=tmp_path,
    )
    assert assessment.passed is True
    assert assessment.checks["generated_media_files_checked"] == 1

    output.write_bytes(b"tampered-output")
    state = AugmentationRunState(samples=())
    report = state.build(dataset=(variant,), dataset_root=tmp_path)
    assert report.quality_checks_passed is False
    assert report.quality_checks["invalid_output_hashes"] == 1
    assert report.quality_check_failures


def test_snapshot_publication_fails_closed_when_media_quality_fails() -> None:
    from augmentation.outcomes.augmentation_result import AugmentationReport
    from orchestration.workflow.augmentation.phase_runner import (
        _require_quality_checks_passed,
    )

    report = AugmentationReport(
        enabled=True,
        original_samples=1,
        augmented_samples=2,
        variants_added=1,
        quality_checks_passed=False,
        quality_check_failures=("variant:output_hash_mismatch",),
    )

    try:
        _require_quality_checks_passed(report=report)
    except ValueError as exc:
        assert "augmentation_quality_checks_failed" in str(exc)
        assert "output_hash_mismatch" in str(exc)
    else:
        raise AssertionError("invalid media quality must block publication")


def test_all_media_variants_emit_complete_hash_lineage(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from augmentation.outcomes.media_validation_outcome import (
        MediaValidationOutcome,
    )
    from preprocessing.media.ports import DecodedAudio

    class AudioDecoder:
        def decode(self, *, path: Path, chunk_frames: int) -> DecodedAudio:
            del path, chunk_frames
            return DecodedAudio(
                channels=1,
                sample_width=2,
                sample_rate=8_000,
                duration_sec=0.1,
                frames_iterator=iter((b"\x00\x00" * 800,)),
            )

    class VideoBackend:
        def is_available(self) -> bool:
            return True

        def probe(self, *, path: Path) -> VideoProbe:
            if (
                path.name.startswith("sample_media_aug")
                and path.suffix == ".mp4"
            ):
                return VideoProbe(
                    width=512,
                    height=512,
                    fps=12.0,
                    duration_seconds=2.0,
                    frame_count=24,
                    has_audio=False,
                    video_codec="h264",
                )
            return VideoProbe(
                width=160,
                height=90,
                fps=10.0,
                duration_seconds=2.0,
                frame_count=20,
                has_audio=False,
                video_codec="h264",
            )

        def render_clip(self, **kwargs: object) -> None:
            output_path = kwargs["output_path"]
            assert isinstance(output_path, Path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"valid-video-output")

        def extract_keyframe(self, **kwargs: object) -> None:
            from PIL import Image

            output_path = kwargs["output_path"]
            assert isinstance(output_path, Path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), (10, 20, 30)).save(
                output_path, "JPEG"
            )

        def decode_check(self, *, path: Path, timeout_seconds: float) -> None:
            assert path.is_file()
            assert timeout_seconds > 0

        def validate_keyframe(
            self,
            *,
            path: Path,
            expected_width: int,
            expected_height: int,
            output_max_bytes: int,
            timeout_seconds: float,
        ) -> dict[str, object]:
            assert path.is_file()
            assert path.stat().st_size <= output_max_bytes
            assert expected_width == 512
            assert expected_height == 512
            assert timeout_seconds > 0
            return {
                "byte_size": path.stat().st_size,
                "width": expected_width,
                "height": expected_height,
                "format": "mjpeg",
                "decoded": True,
            }

    source_image = tmp_path / "source.png"
    Image.new("RGB", (128, 96), (120, 80, 40)).save(source_image)
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"audio-source")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video-source")
    accepted = lambda **kwargs: MediaValidationOutcome(None, dict(kwargs))  # noqa: E731
    cache = _Cache(tmp_path)
    logger = _Logger()
    video_settings = _enabled_video_settings()
    image_settings = ImageAugmentationSettings(enabled=True)
    document_settings = DocumentAugmentationSettings(
        enabled=True,
        mode="document_media",
    )

    cases = (
        (
            ImageAugmenter(
                settings=image_settings,
                operation_executor=ImageOperationExecutor(
                    settings=image_settings,
                    cache=cache,
                    validate_output=accepted,
                ),
                validate_input=accepted,
                logger=logger,
            ),
            MultimodalSample(
                sample_id="image",
                text="image text",
                image=ModalityObject(
                    path=source_image,
                    mime_type="image/png",
                ),
            ),
        ),
        (
            AudioAugmenter(
                settings=AudioAugmentationSettings(enabled=True),
                decoder=AudioDecoder(),
                cache=cache,
                validate_input=accepted,
                validate_output=accepted,
                max_duration_seconds=10.0,
                logger=logger,
            ),
            MultimodalSample(
                sample_id="audio",
                text="audio text",
                audio=ModalityObject(
                    path=source_audio,
                    mime_type="audio/wav",
                ),
            ),
        ),
        (
            VideoClipAugmenter(
                settings=video_settings,
                max_input_bytes=1_000,
                cache=cache,
                backend=VideoBackend(),
                logger=logger,
            ),
            MultimodalSample(
                sample_id="video-clip",
                text="video clip text",
                video=ModalityObject(
                    path=source_video,
                    mime_type="video/mp4",
                ),
            ),
        ),
        (
            VideoKeyframeAugmenter(
                settings=video_settings,
                max_input_bytes=1_000,
                cache=cache,
                backend=VideoBackend(),
                logger=logger,
            ),
            MultimodalSample(
                sample_id="video-keyframe",
                text="video keyframe text",
                video=ModalityObject(
                    path=source_video,
                    mime_type="video/mp4",
                ),
            ),
        ),
        (
            DocumentAugmenter(
                settings=document_settings,
                text_augmenter=DocumentTextAugmenter(
                    settings=document_settings
                ),
                page_augmenter=DocumentPageAugmenter(
                    settings=document_settings
                ),
                logger=logger,
            ),
            MultimodalSample(
                sample_id="document",
                task_type="document_text_pair",
                text="document text",
                image=ModalityObject(
                    path=source_image,
                    mime_type="image/png",
                ),
                metadata={"modality": "document"},
            ),
        ),
    )

    variants = []
    for augmenter, sample in cases:
        produced, rejected = augmenter.augment(
            sample=sample,
            dataset_root=tmp_path,
        )
        assert all(
            item.reason == "image_variant_semantically_unchanged"
            for item in rejected
        )
        assert produced or rejected
        for _name, variant in produced:
            metadata = variant.metadata
            assert valid_hash(metadata["augmentation_source_sha256"])
            assert valid_hash(metadata["augmentation_output_sha256"])
            assert valid_hash(metadata["augmentation_config_hash"])
            assert (
                metadata["augmentation_implementation_hash"]
                == MEDIA_AUGMENTATION_IMPLEMENTATION_HASH
            )
            variants.append(variant)

    assessment = assess_augmentation_run_quality(
        dataset=tuple(variants),
        dataset_root=tmp_path,
    )
    assert assessment.passed is True
    assert assessment.checks["accepted_media_variants"] == len(variants)
    assert assessment.checks["generated_media_files_checked"] == len(variants)


def valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
