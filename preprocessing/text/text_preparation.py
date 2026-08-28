"""Prepare preprocessing inputs into normalized text documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from preprocessing.preprocessed_document import build_document_structure
from preprocessing.preprocessing_input import (
    PreprocessingInput,
)
from preprocessing.privacy.clearance import (
    PrivacyClearance,
    PrivacyClearanceStatus,
)
from preprocessing.privacy.field_inspection import (
    inspect_text_fields_for_release,
)
from preprocessing.privacy.inspection.inspect_document import inspect_document
from preprocessing.privacy.inspection.inspection_result import InspectionResult
from preprocessing.privacy.inspection.local_content_factories import (
    DocumentPrivacyContentFactory,
)
from preprocessing.privacy.public_provenance import public_source_url
from preprocessing.privacy.text_privacy import PiiDetectionResult, PiiDetector
from preprocessing.provenance import stable_identifier
from preprocessing.text.document_structure_privacy import (
    build_approved_structure_payload,
    collect_structure_text_fields,
)

if TYPE_CHECKING:
    from config.preprocessing.text_settings import (
        PreprocessingInputValidationSettings,
    )
    from preprocessing.preprocessed_document import (
        _CanonicalDocumentStructure,
    )

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
_MANY_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def _exact_text_key(*, normalized_text: str) -> str:
    """Return the SHA-256 key for the final privacy-cleared normalized text."""

    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedTextDocument:
    """Canonical text extraction result used by preprocessing."""

    title: str | None
    text: str
    markdown: str
    headings: tuple[str, ...]
    code_block_count: int
    boilerplate_ratio: float
    warnings: tuple[str, ...]
    removed_text_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class PreparedTextInput:
    batch_index: int
    item: PreprocessingInput
    extracted_document: PreparedTextDocument | None
    normalized_text: str
    source_normalized_text: str | None = None
    pii_result: PiiDetectionResult | None = None
    exact_duplicate_key: str | None = None
    privacy_clearance: PrivacyClearance | None = None
    approved_structure_payload: Mapping[str, object] | None = None
    document_structure: _CanonicalDocumentStructure | None = None
    public_source_url: str | None = None
    public_path: str | None = None
    rejection_reason: str | None = None


def apply_text_privacy(
    *,
    normalized_text: str,
    extracted_document: PreparedTextDocument,
    detector: PiiDetector,
) -> tuple[str, PreparedTextDocument, PiiDetectionResult, str | None]:
    """Inspect and, when safe, remediate a prepared text document."""

    pii_result = detector.detect(text=normalized_text)
    if not pii_result.inspection_complete:
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "privacy_inspection_incomplete",
        )
    if not pii_result.has_findings or not detector.quarantine_on_detection:
        return normalized_text, extracted_document, pii_result, None
    if pii_result.has_secret_findings:
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "secret_detected",
        )
    if pii_result.assessment_outcome == "reject":
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "restricted_identifier_detected",
        )
    if pii_result.assessment_outcome in {
        "review",
        "quarantine",
        "escalate",
    }:
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "privacy_review_required",
        )
    if pii_result.assessment_outcome == "accept":
        return normalized_text, extracted_document, pii_result, None

    normalized_text = detector.redact(text=normalized_text, result=pii_result)
    markdown_pii_result = detector.detect(text=extracted_document.markdown)
    if not markdown_pii_result.inspection_complete:
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "privacy_inspection_incomplete",
        )
    extracted_document = replace(
        extracted_document,
        text=normalized_text,
        markdown=detector.redact(
            text=extracted_document.markdown,
            result=markdown_pii_result,
        ),
    )
    usable_tokens = (
        token
        for token in normalized_text.split()
        if not token.startswith("[REDACTED")
    )
    if sum(1 for _ in usable_tokens) < 8:
        return (
            normalized_text,
            extracted_document,
            pii_result,
            "pii_redaction_left_no_usable_text",
        )
    return normalized_text, extracted_document, pii_result, None


def normalize_text(*, text: str) -> str:
    """Unicode/whitespace normalization for stable preprocessing text."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_PATTERN.sub("", normalized)
    lines = [
        _HORIZONTAL_WHITESPACE_PATTERN.sub(" ", line).strip()
        for line in normalized.split("\n")
    ]
    compact = "\n".join(lines).strip()
    return _MANY_BLANK_LINES_PATTERN.sub("\n\n", compact)


def validate_text_input(
    *,
    item: PreprocessingInput,
    settings: PreprocessingInputValidationSettings,
) -> str | None:
    """Return a quarantine reason, or None when the input is usable."""

    if not settings.enabled:
        return None
    if not str(item.source_id or "").strip():
        return "missing_source_id"
    if not str(item.normalized_url or item.source_url or "").strip():
        return "missing_source_url"
    if item.modality not in {"text", "document"}:
        if not item.media_path and not item.payload:
            return "missing_media_payload"
        if (
            item.byte_size is not None
            and item.byte_size > settings.max_input_bytes
        ):
            return "too_large"
        return None
    if item.extracted_text_content is not None:
        if not str(item.extracted_text_content.text or "").strip():
            return "empty_input"
        return None
    if not _has_document_text_payload(item=item):
        return "empty_input"
    return None


def build_document_id(
    *,
    normalized_url: str,
    exact_duplicate_key: str,
) -> str:
    return stable_identifier(
        prefix="doc",
        parts=(normalized_url, exact_duplicate_key),
    )


def prepared_text_diagnostics(
    *,
    prepared: PreparedTextInput,
) -> dict[str, object]:
    pii_result = prepared.pii_result
    if pii_result is None or not pii_result.has_findings:
        return {
            "pii_redaction_counts": {},
            "pii_redactions_total": 0,
            "source_text_redacted": False,
        }
    return {
        "pii_redaction_counts": dict(pii_result.finding_counts),
        "pii_redactions_total": pii_result.total_findings,
        "source_text_redacted": pii_result.has_personal_findings,
    }


class TextInputPreparer:
    """Prepare crawler-extracted or plain document text for preprocessing."""

    def __init__(
        self,
        *,
        pii_detector: PiiDetector,
        document_content_factory: DocumentPrivacyContentFactory,
    ) -> None:
        self._pii_detector = pii_detector
        self._document_content_factory = document_content_factory

    def prepare_valid_input(
        self,
        *,
        batch_index: int,
        item: PreprocessingInput,
    ) -> PreparedTextInput:
        if item.extracted_text_content is not None:
            content = item.extracted_text_content
            extracted_document = PreparedTextDocument(
                title=item.title,
                text=content.text,
                markdown=content.markdown,
                headings=content.headings,
                code_block_count=content.code_block_count,
                boilerplate_ratio=content.boilerplate_ratio,
                warnings=content.warnings,
                removed_text_ratio=0.0,
            )
            return self._finalize(
                batch_index=batch_index,
                item=item,
                extracted_document=extracted_document,
            )

        if item.modality == "document":
            return self._prepare_document(
                batch_index=batch_index,
                item=item,
            )

        document_text = _document_text_from_input(item=item)
        if document_text is None:
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=None,
                normalized_text="",
                rejection_reason="empty_input",
            )
        return self._finalize(
            batch_index=batch_index,
            item=item,
            extracted_document=_preprocessed_plain_document(
                item=item,
                document_text=document_text,
            ),
        )

    def _prepare_document(
        self,
        *,
        batch_index: int,
        item: PreprocessingInput,
    ) -> PreparedTextInput:
        document_content = self._document_content_factory.build(
            item=item,
            normalized_text="",
            title=item.title,
            metadata={},
        )
        if not document_content.pages:
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=None,
                normalized_text="",
                rejection_reason="empty_input",
            )
        page_texts = [page.text for page in document_content.pages]
        joined_text = "\n".join(page_texts)
        pages_payload = [
            {
                "page_number": page.page_number,
                "text": page_texts[i],
            }
            for i, page in enumerate(document_content.pages)
        ]
        item.payload["pages"] = pages_payload
        if document_content.subject_bytes:
            item.payload["__pdf_bytes__"] = document_content.subject_bytes
        if document_content.expected_page_count is not None:
            item.payload["page_count"] = document_content.expected_page_count
        headings = self._extract_headings_from_pages(page_texts=page_texts)
        extracted_document = PreparedTextDocument(
            title=item.title,
            text=joined_text,
            markdown=joined_text,
            headings=headings,
            code_block_count=joined_text.count("```") // 2,
            boilerplate_ratio=0.0,
            warnings=("pdf_page_aware_extraction",),
            removed_text_ratio=0.0,
        )
        return self._finalize(
            batch_index=batch_index,
            item=item,
            extracted_document=extracted_document,
        )

    def _extract_headings_from_pages(
        self,
        *,
        page_texts: list[str],
    ) -> tuple[str, ...]:
        headings: list[str] = []
        for page_text in page_texts:
            for raw_line in page_text.splitlines():
                heading = raw_line.strip().lstrip("#").strip()
                if heading:
                    headings.append(heading[:160])
                    break
        return tuple(headings)

    def _finalize(
        self,
        *,
        batch_index: int,
        item: PreprocessingInput,
        extracted_document: PreparedTextDocument,
    ) -> PreparedTextInput:
        raw_source_url = item.normalized_url or item.source_url
        source_url = public_source_url(raw_source_url)

        raw_pages_payload = item.payload.get("pages")
        pages_payload: tuple[Mapping[str, object], ...] | None = None
        if isinstance(raw_pages_payload, list) and all(
            isinstance(page, Mapping) for page in raw_pages_payload
        ):
            pages_payload = tuple(raw_pages_payload)
        has_page_aware_payload = pages_payload is not None
        normalized_pages: tuple[str, ...] = ()

        if has_page_aware_payload:
            assert pages_payload is not None
            page_texts = tuple(
                str(page.get("text", "")) for page in pages_payload
            )
            normalized_pages = tuple(
                normalize_text(text=text) for text in page_texts
            )
            normalized = "\n".join(normalized_pages).strip()
        else:
            normalized = normalize_text(text=extracted_document.text)

        if not normalized:
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=extracted_document,
                normalized_text="",
                rejection_reason="empty_normalized_text",
            )

        fields: dict[str, str | None] = {
            "body": normalized,
            "markdown": extracted_document.markdown or normalized,
            "title": extracted_document.title or item.title,
            "source_url": source_url,
            "source_url_raw": raw_source_url,
            "path": item.path,
        }
        extracted_headings = tuple(
            normalized_heading
            for heading in extracted_document.headings
            if (normalized_heading := normalize_text(text=heading))
        )
        canonical_headings = _headings_in_canonical_text(
            headings=extracted_headings,
            normalized_text=normalized,
        )
        for index, heading in enumerate(canonical_headings):
            fields[f"heading:{index}"] = heading
        for index, heading in enumerate(extracted_headings):
            fields[f"extracted_heading:{index}"] = heading
        fields.update(
            collect_structure_text_fields(
                source_payload=item.payload,
                normalized_pages=normalized_pages,
            )
        )
        for key, value in item.payload.items():
            if not isinstance(value, str):
                continue
            if key in {
                "source_page_url",
                "embed_host",
                "license",
                "license_url",
                "governance_note",
                "robots_status",
                "terms_source",
                "usage_rules",
                "training_reason",
            }:
                fields[key] = value
            elif key in {
                "author",
                "creator",
                "subject",
                "description",
                "keywords",
                "filename",
            }:
                fields[f"metadata:{key}"] = value

        document_inspection = None
        if item.modality == "document":
            if has_page_aware_payload:
                assert pages_payload is not None
                from preprocessing.privacy.inspection.content_readers.document_content import (
                    DocumentContent,
                    DocumentPage,
                )

                raw_pages = [
                    DocumentPage(
                        page_number=_document_page_number(
                            page.get("page_number", i + 1)
                        ),
                        text=str(page.get("text", "")),
                    )
                    for i, page in enumerate(pages_payload)
                ]
                expected_page_count = item.payload.get("page_count")
                raw_subject_bytes = item.payload.get("__pdf_bytes__", b"")
                subject_bytes = (
                    bytes(raw_subject_bytes)
                    if isinstance(raw_subject_bytes, (bytes, bytearray))
                    else b""
                )
                document_content = DocumentContent(
                    subject_bytes=subject_bytes,
                    title=extracted_document.title or item.title,
                    pages=tuple(raw_pages),
                    metadata={},
                    language=item.resolved_language(),
                    country=None,
                    expected_page_count=(
                        int(expected_page_count)
                        if isinstance(expected_page_count, (int, float))
                        else None
                    ),
                )
            else:
                document_content = self._document_content_factory.build(
                    item=item,
                    normalized_text=normalized,
                    title=extracted_document.title or item.title,
                    metadata={},
                )
            document_inspection = inspect_document(
                document_content,
                self._pii_detector.registry,
            )

        present_fields = frozenset(
            name
            for name, value in fields.items()
            if isinstance(value, str) and value.strip()
        )
        document_required = (
            document_inspection.coverage.required_fields
            if document_inspection is not None
            else frozenset()
        )
        document_checked = (
            document_inspection.coverage.checked_fields
            if document_inspection is not None
            else frozenset()
        )
        released_fields = frozenset(
            name
            for name in present_fields
            if name != "source_url_raw"
            and not name.startswith("metadata:")
            and not name.startswith("page:")
        )
        inspected = inspect_text_fields_for_release(
            fields=fields,
            detector=self._pii_detector,
            required_fields=present_fields | document_required,
            evidence_fields=document_checked,
            input_digest=(
                document_inspection.subject_digest
                if document_inspection is not None
                else None
            ),
            released_fields=released_fields,
        )
        clearance = inspected.clearance
        if (
            document_inspection is not None
            and not document_inspection.safe_to_assess
        ):
            clearance = _reject_document_clearance(
                clearance=clearance,
                inspection=document_inspection,
            )
        approved_body = next(
            (
                field
                for field in clearance.approved_text_fields
                if field.name == "body"
            ),
            None,
        )
        if clearance.permits_training and approved_body is not None:
            clearance = replace(
                clearance,
                output_digest=approved_body.output_digest,
            )
        pii_result = replace(
            self._pii_detector.detect(text=normalized),
            finding_counts=dict(inspected.finding_counts),
            spans=inspected.spans,
        )
        if not clearance.permits_training:
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=extracted_document,
                normalized_text=inspected.values.get("body", normalized),
                pii_result=pii_result,
                privacy_clearance=clearance,
                public_source_url=inspected.values.get(
                    "source_url", source_url
                ),
                public_path=inspected.values.get("path", item.path),
                rejection_reason=(
                    clearance.reasons[0]
                    if clearance.reasons
                    else f"privacy_{clearance.status.value}"
                ),
            )

        normalized_text = inspected.values["body"]
        try:
            approved_structure_payload = build_approved_structure_payload(
                source_payload=item.payload,
                approved_values=inspected.values,
                original_text=normalized,
                approved_text=normalized_text,
                normalized_pages=normalized_pages,
            )
        except ValueError:
            clearance = clearance.mark_incomplete(
                reason="privacy_structure_remediation_inconsistent"
            )
            return PreparedTextInput(
                batch_index=batch_index,
                item=item,
                extracted_document=extracted_document,
                normalized_text=normalized_text,
                pii_result=pii_result,
                privacy_clearance=clearance,
                public_source_url=inspected.values.get(
                    "source_url", source_url
                ),
                public_path=inspected.values.get("path", item.path),
                rejection_reason="privacy_structure_remediation_inconsistent",
            )
        approved_extracted_headings = tuple(
            inspected.values.get(f"extracted_heading:{index}", heading)
            for index, heading in enumerate(extracted_headings)
        )
        approved_canonical_headings = tuple(
            inspected.values.get(f"heading:{index}", heading)
            for index, heading in enumerate(canonical_headings)
        )
        extracted_document = PreparedTextDocument(
            title=inspected.values.get("title"),
            text=normalized_text,
            markdown=inspected.values.get("markdown", normalized_text),
            headings=approved_extracted_headings,
            code_block_count=extracted_document.code_block_count,
            boilerplate_ratio=extracted_document.boilerplate_ratio,
            warnings=extracted_document.warnings,
            removed_text_ratio=extracted_document.removed_text_ratio,
        )
        document_structure = build_document_structure(
            source_payload=approved_structure_payload,
            source_text=normalized,
            normalized_text=normalized_text,
            headings=approved_canonical_headings,
        )
        return PreparedTextInput(
            batch_index=batch_index,
            item=item,
            extracted_document=extracted_document,
            normalized_text=normalized_text,
            source_normalized_text=normalized,
            pii_result=pii_result,
            exact_duplicate_key=_exact_text_key(
                normalized_text=normalized_text
            ),
            privacy_clearance=clearance,
            approved_structure_payload=approved_structure_payload,
            document_structure=document_structure,
            public_source_url=inspected.values.get("source_url", source_url),
            public_path=inspected.values.get("path", item.path),
        )


def _reject_document_clearance(
    *,
    clearance: PrivacyClearance,
    inspection: InspectionResult,
) -> PrivacyClearance:
    reasons = list(clearance.reasons)
    reasons.append("local_document_inspection_incomplete")
    reasons.extend(
        f"local_document_unchecked:{name}"
        for name in sorted(inspection.coverage.unchecked_fields)
    )
    reasons.extend(
        f"local_document_detector_failure:{error}"
        for error in inspection.errors
    )
    return replace(
        clearance,
        status=PrivacyClearanceStatus.INCOMPLETE,
        output_digest=None,
        remediation_verified=False,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _has_document_text_payload(*, item: PreprocessingInput) -> bool:
    return bool(
        str(item.ocr_text or "").strip()
        or str(item.transcript_text or "").strip()
    )


def _headings_in_canonical_text(
    *,
    headings: tuple[str, ...],
    normalized_text: str,
) -> tuple[str, ...]:
    canonical: list[str] = []
    used_spans: set[tuple[int, int]] = set()
    search_start = 0
    for heading in headings:
        start = normalized_text.find(heading, search_start)
        if start < 0:
            start = normalized_text.find(heading)
        if start < 0:
            continue
        span = (start, start + len(heading))
        if span in used_spans:
            continue
        used_spans.add(span)
        canonical.append(heading)
        search_start = span[1]
    return tuple(canonical)


def _document_page_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int, float)
    ):
        raise ValueError("document page_number must be numeric")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("document page_number must be numeric") from error


def _document_text_from_input(*, item: PreprocessingInput) -> str | None:
    for candidate_text in (
        item.ocr_text,
        item.transcript_text,
    ):
        document_text = str(candidate_text or "").strip()
        if document_text:
            return document_text
    return None


def _preprocessed_plain_document(
    *,
    item: PreprocessingInput,
    document_text: str,
) -> PreparedTextDocument:
    document_title = item.title or _first_plain_document_heading(
        document_text=document_text,
    )
    headings = (document_title,) if document_title else ()
    return PreparedTextDocument(
        title=document_title,
        text=document_text,
        markdown=document_text,
        headings=headings,
        code_block_count=document_text.count("```") // 2,
        boilerplate_ratio=0.0,
        warnings=("plain_document_text",),
        removed_text_ratio=0.0,
    )


def _first_plain_document_heading(*, document_text: str) -> str | None:
    for raw_line in document_text.splitlines():
        heading = raw_line.strip().lstrip("#").strip()
        if heading:
            return heading[:160]
    return None
