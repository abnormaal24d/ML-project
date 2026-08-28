"""In-process visual privacy observations derived from exact image bytes.

This module deliberately produces *observations*, not clearance decisions.
The central ``DetectorRegistry`` still maps those observations to canonical
privacy findings.  No caller-supplied ``privacy_analysis`` payload is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from preprocessing.privacy.inspection.detector import VisualRegion


@dataclass(frozen=True, slots=True)
class LocalVisualAnalysis:
    """Result of one locally executed visual-analysis pass."""

    regions: tuple[VisualRegion, ...]
    completed: bool
    detector_versions: dict[str, str]
    errors: tuple[str, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()


class OpenCvVisualPrivacyAnalyzer:
    """Conservative OpenCV-backed visual privacy observer.

    The implementation runs directly over decoded bytes/frames.  It covers
    faces, plate-like regions, QR/barcodes, and high-confidence identity-
    document/signature cues.  It is intentionally independent from policy:
    region categories are interpreted later by the trusted detector registry.
    """

    _VERSION = "opencv-local-privacy-v1"
    _IDENTITY_TERMS = re.compile(
        r"\b(passport|identity\s+card|national\s+id|driving\s+licen[cs]e|"
        r"date\s+of\s+birth|national\s+number|document\s+number)\b",
        flags=re.IGNORECASE,
    )
    _MRZ = re.compile(r"[A-Z0-9<]{25,}\n?[A-Z0-9<]{25,}")
    _SIGNATURE_TERMS = re.compile(
        r"\b(signature|signed\s+by|holder'?s\s+signature)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, *, cv2_module: Any | None = None) -> None:
        self._cv2: Any | None = (
            cv2_module if cv2_module is not None else _load_cv2()
        )
        self._face: Any | None = None
        self._plate: Any | None = None
        self._initialization_errors: tuple[str, ...] = ()
        if self._cv2 is not None:
            self._initialize_cascades()

    def analyze_bytes(
        self,
        *,
        payload: bytes,
        ocr_text: str | None = None,
        frame_index: int | None = None,
        timestamp_ms: int | None = None,
    ) -> LocalVisualAnalysis:
        if self._cv2 is None:
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions={},
                errors=("opencv_visual_backend_unavailable",),
                uncertainty_flags=("visual_backend_unavailable",),
            )
        try:
            import numpy as np

            encoded = np.frombuffer(payload, dtype=np.uint8)
            frame = self._cv2.imdecode(encoded, self._cv2.IMREAD_COLOR)
        except Exception as exc:  # pragma: no cover - backend-specific
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions=self._versions(),
                errors=(f"visual_decode:{type(exc).__name__}",),
                uncertainty_flags=("visual_decode_failure",),
            )
        if frame is None:
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions=self._versions(),
                errors=("visual_decode_failed",),
                uncertainty_flags=("visual_decode_failed",),
            )
        return self.analyze_frame(
            frame=frame,
            ocr_text=ocr_text,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def analyze_frame(
        self,
        *,
        frame: Any,
        ocr_text: str | None = None,
        frame_index: int | None = None,
        timestamp_ms: int | None = None,
    ) -> LocalVisualAnalysis:
        if self._cv2 is None:
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions={},
                errors=("opencv_visual_backend_unavailable",),
                uncertainty_flags=("visual_backend_unavailable",),
            )
        if frame is None or not hasattr(frame, "shape"):
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions=self._versions(),
                errors=("visual_frame_invalid",),
                uncertainty_flags=("visual_frame_invalid",),
            )

        errors = list(self._initialization_errors)
        uncertainty_flags: list[str] = []
        regions: list[VisualRegion] = []
        try:
            height, width = int(frame.shape[0]), int(frame.shape[1])
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            return LocalVisualAnalysis(
                regions=(),
                completed=False,
                detector_versions=self._versions(),
                errors=(f"visual_frame_decode:{type(exc).__name__}",),
                uncertainty_flags=("visual_frame_decode_failure",),
            )

        # Visual quality checks
        if width < 64 or height < 64:
            uncertainty_flags.append("visual_resolution_below_validated_range")
        if width > 10000 or height > 10000:
            uncertainty_flags.append(
                "visual_resolution_exceeds_validated_range"
            )

        # Blur detection using Laplacian variance
        try:
            laplacian = self._cv2.Laplacian(gray, self._cv2.CV_64F)
            blur_score = float(laplacian.var())
            if blur_score < 100.0:  # Calibrated threshold
                uncertainty_flags.append("visual_blur_outside_validated_range")
        except Exception:
            uncertainty_flags.append("visual_blur_assessment_failed")

        # Brightness checks
        brightness_mean = float(gray.mean())
        if brightness_mean < 10.0:
            uncertainty_flags.append("visual_brightness_too_low")
        elif brightness_mean > 245.0:
            uncertainty_flags.append("visual_brightness_too_high")

        if self._face is not None:
            try:
                faces = self._face.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(24, 24),
                )
                regions.extend(
                    _regions_from_rectangles(
                        category="face",
                        rectangles=faces,
                        confidence=0.85,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
            except Exception as exc:  # pragma: no cover - backend-specific
                errors.append(f"face_detector:{type(exc).__name__}")

        if self._plate is not None:
            try:
                plates = self._plate.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 12),
                )
                regions.extend(
                    _regions_from_rectangles(
                        category="license_plate",
                        rectangles=plates,
                        confidence=0.75,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
            except Exception as exc:  # pragma: no cover - backend-specific
                errors.append(f"license_plate_detector:{type(exc).__name__}")

        regions.extend(
            self._machine_readable_regions(
                frame=frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                errors=errors,
            )
        )

        normalized_ocr = (ocr_text or "").strip()
        if normalized_ocr and (
            self._IDENTITY_TERMS.search(normalized_ocr)
            or self._MRZ.search(normalized_ocr.upper())
        ):
            regions.append(
                VisualRegion(
                    category="identity_document",
                    confidence=0.95,
                    x=0,
                    y=0,
                    width=max(1, width),
                    height=max(1, height),
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )
        if normalized_ocr and self._SIGNATURE_TERMS.search(normalized_ocr):
            regions.append(
                VisualRegion(
                    category="signature",
                    confidence=0.70,
                    x=0,
                    y=max(0, height // 2),
                    width=max(1, width),
                    height=max(1, height - (height // 2)),
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )

        return LocalVisualAnalysis(
            regions=_deduplicate_regions(regions),
            completed=not errors,
            detector_versions=self._versions(),
            errors=tuple(dict.fromkeys(errors)),
            uncertainty_flags=tuple(dict.fromkeys(uncertainty_flags)),
        )

    def _initialize_cascades(self) -> None:
        cv2 = self._cv2
        if cv2 is None:
            return
        errors: list[str] = []
        try:
            root = cv2.data.haarcascades
            face = cv2.CascadeClassifier(
                f"{root}haarcascade_frontalface_default.xml"
            )
            if face.empty():
                self._face = None
                errors.append("face_detector_unavailable")
            else:
                self._face = face
        except Exception:
            self._face = None
            errors.append("face_detector_unavailable")
        try:
            root = cv2.data.haarcascades
            plate = cv2.CascadeClassifier(
                f"{root}haarcascade_russian_plate_number.xml"
            )
            if plate.empty():
                self._plate = None
                errors.append("license_plate_detector_unavailable")
            else:
                self._plate = plate
        except Exception:
            self._plate = None
            errors.append("license_plate_detector_unavailable")
        self._initialization_errors = tuple(errors)

    def _machine_readable_regions(
        self,
        *,
        frame: Any,
        frame_index: int | None,
        timestamp_ms: int | None,
        errors: list[str],
    ) -> tuple[VisualRegion, ...]:
        cv2 = self._cv2
        if cv2 is None:
            errors.append("opencv_visual_backend_unavailable")
            return ()
        found: list[VisualRegion] = []
        try:
            detector = cv2.QRCodeDetector()
            ok, points = detector.detectMulti(frame)
            if ok and points is not None:
                for box in points:
                    region = _polygon_region(
                        category="qr_code",
                        points=box,
                        confidence=0.95,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                    if region is not None:
                        found.append(region)
        except Exception as exc:  # pragma: no cover - backend-specific
            errors.append(f"qr_detector:{type(exc).__name__}")

        barcode_type = getattr(cv2, "barcode_BarcodeDetector", None)
        if barcode_type is None:
            errors.append("barcode_detector_unavailable")
        else:
            try:
                detector = barcode_type()
                ok, points = detector.detect(frame)
                if ok and points is not None:
                    for box in points:
                        region = _polygon_region(
                            category="barcode",
                            points=box,
                            confidence=0.90,
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                        )
                        if region is not None:
                            found.append(region)
            except Exception as exc:  # pragma: no cover - backend-specific
                errors.append(f"barcode_detector:{type(exc).__name__}")
        return tuple(found)

    def _versions(self) -> dict[str, str]:
        if self._cv2 is None:
            return {}
        version = str(getattr(self._cv2, "__version__", "unknown"))
        return {
            "opencv_visual_privacy": self._VERSION,
            "opencv": version,
        }


def _load_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _regions_from_rectangles(
    *,
    category: str,
    rectangles: Any,
    confidence: float,
    frame_index: int | None,
    timestamp_ms: int | None,
) -> tuple[VisualRegion, ...]:
    output: list[VisualRegion] = []
    for rectangle in rectangles:
        try:
            x, y, width, height = (int(value) for value in rectangle)
            output.append(
                VisualRegion(
                    category=category,
                    confidence=confidence,
                    x=max(0, x),
                    y=max(0, y),
                    width=max(1, width),
                    height=max(1, height),
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(output)


def _polygon_region(
    *,
    category: str,
    points: Any,
    confidence: float,
    frame_index: int | None,
    timestamp_ms: int | None,
) -> VisualRegion | None:
    try:
        xs = [int(round(float(point[0]))) for point in points]
        ys = [int(round(float(point[1]))) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    x, y = max(0, min(xs)), max(0, min(ys))
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    return VisualRegion(
        category=category,
        confidence=confidence,
        x=x,
        y=y,
        width=width,
        height=height,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


def _deduplicate_regions(
    regions: list[VisualRegion],
) -> tuple[VisualRegion, ...]:
    unique: dict[tuple[object, ...], VisualRegion] = {}
    for region in regions:
        key = (
            region.category,
            region.x,
            region.y,
            region.width,
            region.height,
            region.frame_index,
            region.timestamp_ms,
        )
        existing = unique.get(key)
        if existing is None or region.confidence > existing.confidence:
            unique[key] = region
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.frame_index if item.frame_index is not None else -1,
                item.timestamp_ms if item.timestamp_ms is not None else -1,
                item.category,
                item.x,
                item.y,
            ),
        )
    )


__all__ = ["LocalVisualAnalysis", "OpenCvVisualPrivacyAnalyzer"]
