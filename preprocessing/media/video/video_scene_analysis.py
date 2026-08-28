"""Scene analysis and visual layout proxy graph construction for video."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any

from .video_ocr import (
    _ocr_text_for_scene,
    joined_text,
    safe_float,
)

if TYPE_CHECKING:
    from preprocessing.media.ports import FrameProcessor, VideoReader


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return sqrt(sum((value - average) ** 2 for value in values) / len(values))


def append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def duration_from_probe(
    *, probe_metadata: dict[str, Any] | None
) -> float | None:
    if not probe_metadata:
        return None
    for key in ("duration_seconds", "duration"):
        value = probe_metadata.get(key)
        if value is not None:
            duration = safe_float(value, default=0.0)
            return duration if duration > 0.0 else None
    return None


def timestamp(item: dict[str, Any]) -> float:
    return safe_float(item.get("timestamp_seconds"), default=0.0)


def _unavailable_visual_stats(*, reason: str) -> dict[str, Any]:
    return {
        "brightness": 0.0,
        "contrast": 0.0,
        "motion_score": 0.0,
        "dominant_color": [0, 0, 0],
        "analysis_status": "unavailable",
        "analysis_reasons": (reason,),
    }


def _empty_scene_result(*, analysis_reasons: list[str]) -> dict[str, Any]:
    return {
        "scenes": [],
        "status": "unavailable",
        "scene_count": 0,
        "scene_graph_semantics": "unavailable",
        "scene_graph_type": "unavailable",
        "semantic_scene_graph_available": False,
        "layout_graph_available": False,
        "object_detection_model_available": False,
        "object_detection_status": "not_run",
        "semantic_object_labels_available": False,
        "scene_graph_reasoning_model_available": False,
        "scene_graph_reasoning_status": "not_run",
        "relation_reasoning_status": "not_run",
        "analysis_status": "unavailable",
        "analysis_reasons": tuple(sorted(set(analysis_reasons))),
        "pipeline": {},
    }


def sample_video_frames(
    *,
    video_path: str | None,
    analysis_reasons: list[str],
    video_reader: VideoReader,
    max_frames: int = 96,
) -> list[dict[str, Any]]:
    if not video_path:
        append_reason(analysis_reasons, "no_video_path_for_frames")
        return []
    if not video_reader.is_available():
        append_reason(analysis_reasons, "video_reader_unavailable")
        return []
    try:
        frames = video_reader.sample_uniform(
            video_path,
            n=min(max_frames, 96),
        )
        if frames:
            return frames
        append_reason(analysis_reasons, "frame_sampling_empty")
    except Exception:  # exception-rules: best-effort-cleanup
        append_reason(analysis_reasons, "frame_sampling_failed")
    return []


def detect_scene_boundaries(
    *,
    frames: list[dict[str, Any]],
    duration_seconds: float | None,
    analysis_reasons: list[str],
    frame_processor: FrameProcessor,
) -> list[tuple[float, float]]:
    if not frames:
        return []
    diffs = frame_differences(
        frames=frames,
        frame_processor=frame_processor,
    )
    if not diffs:
        append_reason(analysis_reasons, "insufficient_frame_diffs")
        return []
    boundaries = boundaries_from_keyframes(  # reuse logic
        keyframes=[{"timestamp_seconds": ts} for ts, _ in diffs],
        duration_seconds=duration_seconds,
    )
    if not boundaries and duration_seconds:
        # fallback coarse
        step = max(1.0, duration_seconds / 8)
        boundaries = [
            (i * step, min(duration_seconds, (i + 1) * step)) for i in range(8)
        ]
    return boundaries or []


def frame_differences(
    frames: list[dict[str, Any]],
    *,
    frame_processor: FrameProcessor,
) -> list[tuple[float, float]]:
    processor = frame_processor
    if not processor.is_available() or len(frames) < 2:
        return []
    diffs: list[tuple[float, float]] = []
    prev = None
    for item in frames:
        ts = timestamp(item)
        frame = item.get("frame")
        if frame is None:
            continue
        gray = processor.bgr_to_gray(frame)
        if gray is None:
            continue
        small = processor.resize(gray, width=64, height=36)
        if small is None:
            continue
        if prev is not None:
            d = processor.absdiff(prev, small)
            if d is not None:
                score = float(d.mean())
                diffs.append((ts, score))
        prev = small
    return diffs


def boundaries_from_keyframes(
    *,
    keyframes: list[dict[str, Any]],
    duration_seconds: float | None,
) -> list[tuple[float, float]]:
    if not keyframes:
        return []
    times = sorted({timestamp(kf) for kf in keyframes})
    if len(times) < 2:
        if duration_seconds and times:
            return [(0.0, float(duration_seconds))]
        return []
    bounds: list[tuple[float, float]] = []
    for i in range(len(times)):
        start = times[i]
        end = (
            times[i + 1]
            if i + 1 < len(times)
            else (duration_seconds or times[-1] + 1.0)
        )
        bounds.append((round(start, 3), round(end, 3)))
    return bounds


def visual_stats_for_frames(
    *,
    frames: list[dict[str, Any]],
    analysis_reasons: list[str],
    frame_processor: FrameProcessor,
) -> dict[str, Any]:
    if not frames:
        append_reason(analysis_reasons, "no_scene_frame_evidence")
        return _unavailable_visual_stats(reason="no_scene_frame_evidence")

    processor = frame_processor
    if not processor.is_available():
        append_reason(analysis_reasons, "opencv_unavailable")
        return _unavailable_visual_stats(reason="opencv_unavailable")

    brightness: list[float] = []
    contrast: list[float] = []
    colors: list[list[int]] = []
    previous = None
    motion: list[float] = []
    for item in frames:
        frame = item.get("frame")
        if frame is None:
            continue
        gray = processor.bgr_to_gray(frame)
        if gray is None:
            continue
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        colors.append(
            [int(value) for value in frame.reshape(-1, 3).mean(axis=0)]
        )
        small = processor.resize(gray, width=96, height=54)
        if small is None:
            continue
        if previous is not None:
            diff = processor.absdiff(previous, small)
            if diff is not None:
                motion.append(float(diff.mean()))
        previous = small
    return {
        "brightness": round(mean(brightness), 3),
        "contrast": round(mean(contrast), 3),
        "motion_score": round(mean(motion), 3),
        "dominant_color": [
            round(mean([color[channel] for color in colors]), 3)
            for channel in range(3)
        ]
        if colors
        else [0, 0, 0],
        "analysis_status": "passed",
        "analysis_reasons": (),
    }


def visual_region_nodes(
    *,
    frames: list[dict[str, Any]],
    analysis_reasons: list[str],
    frame_processor: FrameProcessor,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    processor = frame_processor
    if not processor.is_available():
        append_reason(analysis_reasons, "opencv_unavailable")
        return []

    frame = frames[len(frames) // 2].get("frame")
    if frame is None:
        return []
    h, w = frame.shape[:2]
    # simplistic edge regions proxy
    regions_raw = [
        (0.2 * w * h, 0, 0, w // 3, h // 3),
        (0.1 * w * h, w // 2, h // 2, w // 4, h // 4),
    ]
    return _raw_regions_to_nodes(raw_regions=regions_raw, width=w, height=h)


def region_label(
    *, area_ratio: float, width_ratio: float, height_ratio: float
) -> str:
    if area_ratio > 0.4:
        return "dominant_region"
    if max(width_ratio, height_ratio) > 0.6:
        return "large_span"
    return "support_region"


def _raw_regions_to_nodes(
    *,
    raw_regions: list[tuple[float, int, int, int, int]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, (area, x, y, w, h) in enumerate(raw_regions):
        area_ratio = area / max(1.0, width * height)
        regions.append(
            {
                "id": f"region_{index}",
                "label": region_label(
                    area_ratio=area_ratio,
                    width_ratio=w / max(1.0, width),
                    height_ratio=h / max(1.0, height),
                ),
                "label_semantics": "edge_region_proxy",
                "semantic_label": None,
                "node_type": "visual_region_proxy",
                "is_semantic_object": False,
                "object_detection_model_label": None,
                "confidence": round(min(1.0, area_ratio), 4),
                "box": [
                    round(x / width, 4),
                    round(y / height, 4),
                    round((x + w) / width, 4),
                    round((y + h) / height, 4),
                ],
            }
        )
    return regions


def text_nodes_for_scene(
    *,
    transcript_text: str,
    ocr_text: str,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if transcript_text:
        nodes.append(
            {
                "id": "speech_0",
                "label": "speech",
                "text_preview": transcript_text[:120],
            }
        )
    if ocr_text:
        nodes.append(
            {
                "id": "ocr_0",
                "label": "onscreen_text",
                "text_preview": ocr_text[:120],
            }
        )
    return nodes


def segments_overlap(
    *,
    start_seconds: float,
    end_seconds: float,
    segment: dict[str, Any],
) -> bool:
    raw_start = segment.get("start_seconds")
    raw_end = segment.get("end_seconds")
    if raw_start is None or raw_end is None:
        return False
    seg_start = safe_float(raw_start, default=-1.0)
    seg_end = safe_float(raw_end, default=-1.0)
    if seg_start < 0.0 or seg_end <= seg_start:
        return False
    return not (end_seconds <= seg_start or start_seconds >= seg_end)


def relations_for_scene(
    *,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rels = []
    for i, _node in enumerate(nodes):
        for j in range(i + 1, min(i + 3, len(nodes))):
            rels.append(
                {
                    "source": nodes[i]["id"],
                    "target": nodes[j]["id"],
                    "predicate": "co_visible_or_co_speech",
                }
            )
    return rels


def spatial_predicate(node: dict[str, Any]) -> str:
    return str(node.get("label", "region"))


def build_scene(
    *,
    scene_index: int,
    start_seconds: float,
    end_seconds: float,
    frames: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    frame_ocr_results: list[dict[str, Any]],
    analysis_reasons: list[str],
    frame_processor: FrameProcessor,
) -> dict[str, Any]:
    ocr_text = _ocr_text_for_scene(
        frame_ocr_results=frame_ocr_results,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    speech_text = _speech_text_for_scene(  # defined below or dupe call
        transcript_segments=transcript_segments,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    nodes = visual_region_nodes(
        frames=frames,
        analysis_reasons=analysis_reasons,
        frame_processor=frame_processor,
    )
    text_nodes = text_nodes_for_scene(
        transcript_text=speech_text, ocr_text=ocr_text
    )
    all_nodes = nodes + text_nodes
    stats = visual_stats_for_frames(
        frames=frames,
        analysis_reasons=analysis_reasons,
        frame_processor=frame_processor,
    )
    return {
        "scene_index": scene_index,
        "start_seconds": round(start_seconds, 3),
        "end_seconds": round(end_seconds, 3),
        "duration_seconds": round(max(0.0, end_seconds - start_seconds), 3),
        "visual_stats": stats,
        "nodes": all_nodes,
        "relations": relations_for_scene(nodes=all_nodes),
        "text_preview": (speech_text or ocr_text)[:200],
        "evidence_sources": ("visual_proxy", "text_proxy"),
        "audio_events": ["speech"] if speech_text else [],
        "description": scene_description(
            {
                "visual_stats": stats,
                "text_preview": (speech_text or ocr_text)[:200],
            }
        ),
    }


def _speech_text_for_scene(
    *,
    transcript_segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    return joined_text(
        segment.get("text") or segment.get("transcript")
        for segment in transcript_segments
        if segments_overlap(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            segment=segment,
        )
    )


def scene_description(scene: dict[str, Any]) -> str:
    vis = scene.get("visual_stats", {})
    light = "bright" if vis.get("brightness", 0) > 0.5 else "dim"
    motion_text = (
        "with motion" if vis.get("motion_score", 0) > 0.05 else "static"
    )
    evidence = scene.get("text_preview", "")
    if evidence:
        return (
            f"{light} visual scene {motion_text}; evidence: {evidence[:220]}"
        )
    return f"{light} visual scene {motion_text}"


def analyze_video_scenes(
    *,
    video_path: str | None = None,
    keyframes: list[dict[str, Any]] | None = None,
    transcript_segments: list[dict[str, Any]] | None = None,
    frame_ocr_results: list[dict[str, Any]] | None = None,
    probe_metadata: dict[str, Any] | None = None,
    video_reader: VideoReader,
    frame_processor: FrameProcessor,
) -> dict[str, Any]:
    """Build visual layout proxy segments from frame evidence."""

    analysis_reasons: list[str] = []
    frames = sample_video_frames(
        video_path=video_path,
        analysis_reasons=analysis_reasons,
        video_reader=video_reader,
    )
    resolved_keyframes = keyframes or []
    duration_seconds = duration_from_probe(probe_metadata=probe_metadata)
    boundaries = detect_scene_boundaries(
        frames=frames,
        duration_seconds=duration_seconds,
        analysis_reasons=analysis_reasons,
        frame_processor=frame_processor,
    )
    if not boundaries:
        boundaries = boundaries_from_keyframes(
            keyframes=resolved_keyframes,
            duration_seconds=duration_seconds,
        )
    if not boundaries:
        return _empty_scene_result(analysis_reasons=analysis_reasons)

    scenes = [
        build_scene(
            scene_index=index,
            start_seconds=start,
            end_seconds=end,
            frames=frames,
            keyframes=resolved_keyframes,
            transcript_segments=transcript_segments or [],
            frame_ocr_results=frame_ocr_results or [],
            analysis_reasons=analysis_reasons,
            frame_processor=frame_processor,
        )
        for index, (start, end) in enumerate(boundaries[:24])
    ]
    return {
        "scenes": scenes,
        "status": (
            "cv_visual_proxy_scene_graph"
            if frames
            else "keyframe_visual_proxy_scene_graph"
        ),
        "scene_count": len(scenes),
        "scene_graph_semantics": "visual_proxy",
        "scene_graph_type": "visual_layout_proxy",
        "semantic_scene_graph_available": False,
        "layout_graph_available": True,
        "object_detection_model_available": False,
        "object_detection_status": "not_run",
        "semantic_object_labels_available": False,
        "scene_graph_reasoning_model_available": False,
        "scene_graph_reasoning_status": "not_run",
        "relation_reasoning_status": "geometric_proxy",
        "analysis_status": "proxy",
        "analysis_reasons": tuple(sorted(set(analysis_reasons))),
        "pipeline": {
            "shot_boundary": "frame_difference",
            "nodes": "edge_region_layout_proxies_text_audio",
            "relations": "geometric_layout_proxy_and_text_visibility",
        },
    }
