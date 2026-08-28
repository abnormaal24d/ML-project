"""Canonical manifest promotion accepts complete but undercovered raw runs."""

from __future__ import annotations

import pytest

from config.settings.gate import (
    CrawlOutputGateSettings,
    GateMinimumRecordsSettings,
)
from datachecker.manifests.crawl_manifest_writer import (
    CrawlManifestWriteError,
    CrawlManifestWriter,
)


def _writer(
    *, gate: CrawlOutputGateSettings | None = None
) -> CrawlManifestWriter:
    writer = CrawlManifestWriter.__new__(CrawlManifestWriter)
    writer._crawl_output_gate = gate or CrawlOutputGateSettings(
        enabled=True,
        min_raw_objects_total=80,
        min_successful_requests_total=60,
        min_quality_score=0.45,
        minimum_records=GateMinimumRecordsSettings(
            page=1,
            document=15,
            image=20,
            audio=5,
            video=5,
        ),
    )
    return writer


def _readiness(*, ready: bool) -> dict[str, object]:
    return {
        "ready": ready,
        "unmet_requirements": () if ready else ("audio<5", "video<5"),
        "object_records_total": 17,
        "requests_total": 22,
        "successful_requests_total": 20,
        "quality_score": 0.8,
        "modality_counts": {"page": 2, "document": 10, "audio": 3, "video": 2},
    }


def test_undercovered_readiness_is_valid_promotion_evidence() -> None:
    writer = _writer()

    writer._validate_readiness(
        readiness=_readiness(ready=False),
        object_records_total=17,
        modality_counts={"page": 2, "document": 10, "audio": 3, "video": 2},
    )


def test_ready_readiness_must_satisfy_gate_minimums() -> None:
    writer = _writer()

    with pytest.raises(CrawlManifestWriteError, match="raw object minimum"):
        writer._validate_readiness(
            readiness=_readiness(ready=True),
            object_records_total=17,
            modality_counts={
                "page": 2,
                "document": 10,
                "audio": 3,
                "video": 2,
            },
        )


def test_readiness_counts_must_match_inventory() -> None:
    writer = _writer()

    with pytest.raises(CrawlManifestWriteError, match="object count"):
        writer._validate_readiness(
            readiness=_readiness(ready=False),
            object_records_total=99,
            modality_counts={"audio": 3},
        )


def test_readiness_ready_requires_boolean() -> None:
    writer = _writer()

    readiness = _readiness(ready=False)
    readiness["ready"] = "yes"

    with pytest.raises(
        CrawlManifestWriteError, match="ready must be a boolean"
    ):
        writer._validate_readiness(
            readiness=readiness,
            object_records_total=17,
            modality_counts={"audio": 3},
        )


def test_readiness_quality_score_must_be_numeric() -> None:
    writer = _writer()

    readiness = _readiness(ready=False)
    readiness["quality_score"] = None

    with pytest.raises(CrawlManifestWriteError, match="quality_score"):
        writer._validate_readiness(
            readiness=readiness,
            object_records_total=17,
            modality_counts={
                "page": 2,
                "document": 10,
                "audio": 3,
                "video": 2,
            },
        )


def test_disabled_gate_allows_ready_report_below_minimums() -> None:
    writer = _writer(
        gate=CrawlOutputGateSettings(
            enabled=False,
            minimum_records=GateMinimumRecordsSettings(),
        )
    )

    writer._validate_readiness(
        readiness=_readiness(ready=True),
        object_records_total=17,
        modality_counts={
            "page": 2,
            "document": 10,
            "audio": 3,
            "video": 2,
        },
    )
