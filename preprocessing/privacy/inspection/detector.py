"""Small detector schemas and execution diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from preprocessing.privacy.inspection.finding import PrivacyFinding


@dataclass(frozen=True, slots=True)
class TextDetectorInput:
    text: str
    field_name: str
    language: str | None = None
    country: str | None = None
    page_number: int | None = None

    def resolved_language(self) -> str | None:
        """Return the normalized language value consumed by detectors."""

        return self.language


@dataclass(frozen=True, slots=True)
class VisualRegion:
    category: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    frame_index: int | None = None
    timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class DetectorRun:
    detector_name: str
    detector_version: str
    completed: bool
    finding_count: int
    elapsed_ms: int
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class VisualDetectorExecution:
    """Execute visual detectors and isolate failures."""

    findings: tuple[PrivacyFinding, ...]
    runs: tuple[DetectorRun, ...]
    failures: tuple[str, ...]

    @property
    def completed(self) -> bool:
        """Return whether all detectors executed successfully."""

        return (
            bool(self.runs)
            and not self.failures
            and all(run.completed for run in self.runs)
        )


class TextDetector(Protocol):
    name: str
    version: str

    def detect(self, item: TextDetectorInput) -> Sequence[PrivacyFinding]: ...


class VisualDetector(Protocol):
    name: str
    version: str

    def detect_regions(
        self,
        *,
        field_name: str,
        regions: Sequence[VisualRegion],
    ) -> Sequence[PrivacyFinding]: ...


def run_visual_detectors(
    *,
    detectors: Sequence[VisualDetector],
    field_name: str,
    regions: Sequence[VisualRegion],
) -> VisualDetectorExecution:
    """Execute visual detectors in stable order and isolate failures."""

    if not detectors:
        return VisualDetectorExecution(
            findings=(),
            runs=(),
            failures=("visual_detector_registry_empty",),
        )

    findings: list[PrivacyFinding] = []
    runs: list[DetectorRun] = []
    failures: list[str] = []

    for detector in detectors:
        started_ns = time.perf_counter_ns()

        try:
            detected = tuple(
                detector.detect_regions(
                    field_name=field_name,
                    regions=regions,
                )
            )
        except Exception as exc:
            failure = f"{detector.name}:{type(exc).__name__}"

            failures.append(failure)
            runs.append(
                DetectorRun(
                    detector_name=detector.name,
                    detector_version=detector.version,
                    completed=False,
                    finding_count=0,
                    elapsed_ms=_elapsed_ms(started_ns),
                    failure=failure,
                )
            )
            continue

        findings.extend(detected)
        runs.append(
            DetectorRun(
                detector_name=detector.name,
                detector_version=detector.version,
                completed=True,
                finding_count=len(detected),
                elapsed_ms=_elapsed_ms(started_ns),
            )
        )

    return VisualDetectorExecution(
        findings=tuple(findings),
        runs=tuple(runs),
        failures=tuple(failures),
    )


def _elapsed_ms(started_ns: int) -> int:
    """Return elapsed milliseconds from a high-resolution start timestamp."""

    return max(
        0,
        (time.perf_counter_ns() - started_ns) // 1_000_000,
    )
