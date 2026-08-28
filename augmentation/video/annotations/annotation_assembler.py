"""Assemble clip and keyframe samples after annotation transformation."""

from __future__ import annotations

from dataclasses import replace

from augmentation.video.annotations.annotation_mapping import (
    transform_video_annotation_mapping,
)
from augmentation.video.annotations.annotation_receipt import (
    VideoAnnotationReceipt,
)
from augmentation.video.annotations.keyframe_annotation_mapping import (
    _keyframe_mapping,
)
from augmentation.video.annotations.spatial_annotation_mapping import (
    _transform_layout_boxes,
    _transform_object_boxes,
    _transform_ui_elements,
)
from augmentation.video.annotations.temporal_annotation_mapping import (
    _transform_speaker_segments,
)
from augmentation.video.video_transform import TimelineTransform
from augmentation.video.video_transform_backend import SpatialTransform
from mmcrawler_datasets.schema import MultimodalSample


def transform_video_clip_sample(
    *,
    sample: MultimodalSample,
    spatial: SpatialTransform,
    timeline: TimelineTransform,
) -> tuple[MultimodalSample, VideoAnnotationReceipt]:
    """Transform canonical sample annotations to the output clip coordinate space."""

    receipt = VideoAnnotationReceipt()
    layout_boxes, current = _transform_layout_boxes(
        sample.layout_boxes, spatial
    )
    receipt = receipt.merge(current)
    ui_elements, current = _transform_ui_elements(sample.ui_elements, spatial)
    receipt = receipt.merge(current)
    object_boxes, current = _transform_object_boxes(
        sample.object_boxes, spatial
    )
    receipt = receipt.merge(current)
    speaker_segments, current = _transform_speaker_segments(
        sample.speaker_segments,
        timeline,
    )
    receipt = receipt.merge(current)
    task_target, current = transform_video_annotation_mapping(
        value=sample.task_target,
        spatial=spatial,
        timeline=timeline,
    )
    receipt = receipt.merge(current)
    metadata, current = transform_video_annotation_mapping(
        value=sample.metadata,
        spatial=spatial,
        timeline=timeline,
    )
    receipt = receipt.merge(current)
    return (
        replace(
            sample,
            layout_boxes=layout_boxes,
            ui_elements=ui_elements,
            object_boxes=object_boxes,
            speaker_segments=speaker_segments,
            task_target=task_target,
            metadata=metadata,
            video_tensor_path=None,
            target_video_tensor_path=None,
            target_video_tokens_path=None,
        ),
        receipt,
    )


def transform_video_keyframe_sample(
    *,
    sample: MultimodalSample,
    spatial: SpatialTransform,
    timestamp_seconds: float,
    source_fps: float,
) -> tuple[MultimodalSample, VideoAnnotationReceipt]:
    """Build a still-image view and remove timeline-only annotations explicitly."""

    receipt = VideoAnnotationReceipt()
    layout_boxes, current = _transform_layout_boxes(
        sample.layout_boxes, spatial
    )
    receipt = receipt.merge(current)
    ui_elements, current = _transform_ui_elements(sample.ui_elements, spatial)
    receipt = receipt.merge(current)
    object_boxes, current = _transform_object_boxes(
        sample.object_boxes, spatial
    )
    receipt = receipt.merge(current)
    task_target, current = _keyframe_mapping(
        value=sample.task_target,
        spatial=spatial,
        timestamp_seconds=timestamp_seconds,
        source_fps=source_fps,
    )
    receipt = receipt.merge(current)
    metadata, current = _keyframe_mapping(
        value=sample.metadata,
        spatial=spatial,
        timestamp_seconds=timestamp_seconds,
        source_fps=source_fps,
    )
    receipt = receipt.merge(current)
    receipt = receipt.merge(
        VideoAnnotationReceipt(dropped_intervals=len(sample.speaker_segments))
    )
    return (
        replace(
            sample,
            video=None,
            video_tensor_path=None,
            target_video_tensor_path=None,
            target_video_tokens_path=None,
            layout_boxes=layout_boxes,
            ui_elements=ui_elements,
            object_boxes=object_boxes,
            speaker_segments=(),
            prosody=None,
            task_target=task_target,
            metadata=metadata,
        ),
        receipt,
    )
