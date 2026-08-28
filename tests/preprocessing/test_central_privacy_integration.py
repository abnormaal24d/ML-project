"""Integration coverage for preprocessing.privacy."""

from __future__ import annotations

from mmcrawler_datasets.selection.quality import _classify_pii
from orchestration.composition.privacy.privacy_inspection_services import (
    build_default_detector_registry,
)
from preprocessing.media.video.video_safety_scanner import VideoSafetyScanner
from preprocessing.privacy.inspection.content_readers.text_content import (
    TextContent,
)
from preprocessing.privacy.inspection.inspect_text import inspect_text
from preprocessing.text.text_preparation import (
    PreparedTextDocument,
    apply_text_privacy,
)
from tests.support.privacy import build_test_pii_detector


def _document(text: str) -> PreparedTextDocument:
    return PreparedTextDocument(
        title=None,
        text=text,
        markdown=text,
        headings=(),
        code_block_count=0,
        boilerplate_ratio=0.0,
        warnings=(),
    )


def test_nested_privacy_package_imports_and_detects() -> None:
    registry = build_default_detector_registry()
    result = inspect_text(
        TextContent("Contact jan@example.com using IBAN BE68539007547034"),
        registry,
    )

    assert result.safe_to_assess
    assert result.finding_counts == {"email_address": 1, "iban": 1}


def test_text_preprocessing_uses_central_privacy_engine() -> None:
    text = (
        "This complete example contains jan@example.com but still has enough "
        "ordinary words to remain useful after secure redaction for training."
    )
    detector = build_test_pii_detector()

    redacted, document, result, rejection = apply_text_privacy(
        normalized_text=text,
        extracted_document=_document(text),
        detector=detector,
    )

    assert rejection is None
    assert result.assessment_outcome == "remediate"
    assert result.finding_counts == {"email_address": 1}
    assert "jan@example.com" not in redacted
    assert "[REDACTED_EMAIL_ADDRESS]" in redacted
    assert document.text == redacted
    assert document.markdown == redacted


def test_secrets_and_restricted_identifiers_fail_closed() -> None:
    detector = build_test_pii_detector()
    secret_text = (
        "This unsafe training example contains password=abcdefghijklmnop and "
        "must be quarantined before any downstream processing can occur."
    )
    restricted_text = (
        "This unsafe training example contains card 4111 1111 1111 1111 and "
        "must be rejected before any downstream processing can occur."
    )

    _, _, secret_result, secret_rejection = apply_text_privacy(
        normalized_text=secret_text,
        extracted_document=_document(secret_text),
        detector=detector,
    )
    _, _, restricted_result, restricted_rejection = apply_text_privacy(
        normalized_text=restricted_text,
        extracted_document=_document(restricted_text),
        detector=detector,
    )

    assert secret_result.assessment_outcome == "escalate"
    assert secret_rejection == "secret_detected"
    assert restricted_result.assessment_outcome == "reject"
    assert restricted_rejection == "restricted_identifier_detected"


def test_training_and_video_flows_share_central_detection() -> None:
    unsafe = "Contact jan@example.com and token=abcdefghijklmnop"

    assert _classify_pii(unsafe, build_test_pii_detector()) == "blocked"
    result = VideoSafetyScanner(
        registry=build_default_detector_registry()
    ).scan(
        transcript=unsafe,
        metadata={"license": "CC0"},
    )

    assert result["safety_status"] == "blocked"
    assert result["pii_detected"] is True
    assert "pii_email_detected" in result["safety_reasons"]
    assert "pii_secret_detected" in result["safety_reasons"]
