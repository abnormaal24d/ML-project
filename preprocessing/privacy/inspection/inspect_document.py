"""Inspect document title, pages, metadata and page coverage locally."""

from __future__ import annotations

from preprocessing.privacy.inspection.content_readers.document_content import (
    DocumentContent,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.inspect_text import inspect_text_fields
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult


def inspect_document(
    content: DocumentContent,
    registry: DetectorRegistry,
) -> InspectionResult:
    fields: dict[str, str] = {}
    if content.title:
        fields["title"] = content.title
    for page in content.pages:
        fields[f"page:{page.page_number}"] = page.text
    fields.update(
        {f"metadata:{key}": value for key, value in content.metadata.items()}
    )
    result = inspect_text_fields(
        fields=fields,
        registry=registry,
        language=content.language,
        country=content.country,
        required_fields=frozenset(fields),
        subject_bytes=content.subject_bytes,
    )
    page_numbers = tuple(page.page_number for page in content.pages)
    checked_pages = frozenset(page_numbers)
    expected_pages = content.expected_page_count
    failures = list(result.errors)
    page_failures: list[str] = []
    if expected_pages is None:
        page_failures.append("document_page_count_missing")
    elif expected_pages <= 0:
        page_failures.append("invalid_expected_page_count")
    if (
        any(page_number <= 0 for page_number in page_numbers)
        or len(checked_pages) != len(page_numbers)
        or (
            expected_pages is not None
            and expected_pages > 0
            and any(
                page_number > expected_pages for page_number in page_numbers
            )
        )
    ):
        page_failures.append("invalid_document_page_evidence")
    failures.extend(page_failures)
    pages_complete = (
        expected_pages is not None
        and expected_pages > 0
        and checked_pages == frozenset(range(1, expected_pages + 1))
        and not page_failures
    )
    checked = set(result.coverage.checked_fields)
    if content.subject_bytes:
        checked.add("document_decode")
    if pages_complete:
        checked.add("document_page_coverage")
    required = set(result.coverage.required_fields) | {
        "document_decode",
        "document_page_coverage",
    }
    coverage = InspectionCoverage(
        checked_fields=frozenset(checked),
        required_fields=frozenset(required),
        checked_pages=checked_pages,
        expected_page_count=expected_pages,
        detector_failures=tuple(dict.fromkeys(failures)),
        warnings=result.coverage.warnings,
    )
    return InspectionResult(
        subject_digest=result.subject_digest,
        findings=result.findings,
        coverage=coverage,
        detector_runs=result.detector_runs,
        completed=result.completed and coverage.complete and not failures,
        errors=tuple(dict.fromkeys(failures)),
        detector_versions=result.detector_versions,
    )
