from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from crawler.classification.content_category_detector import (
    ContentCategoryDetector,
)
from crawler.classification.content_classifier import (
    ContentClassifier,
    ContentClassifierConfig,
)
from crawler.classification.mime_signature_detector import (
    MimeSignatureDetector,
)
from crawler.classification.mime_type_resolver import (
    MimeTypeResolver,
    normalize_mime_type,
)


class _Logger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None


def test_classifier_exposes_only_canonical_sniff_bytes_contract() -> None:
    parameters = inspect.signature(ContentClassifier.classify).parameters
    assert "sniff_bytes" in parameters
    assert "body" not in parameters
    assert (
        "fingerprint_generator"
        not in inspect.signature(ContentClassifier.__init__).parameters
    )
    assert (
        "fingerprint_algorithm"
        not in ContentClassifierConfig.__dataclass_fields__
    )


def test_removed_mime_resolver_wrapper_is_absent() -> None:
    assert not hasattr(MimeTypeResolver, "detect_with_metadata")


def test_signature_detector_is_keyword_only_and_rejects_body_alias() -> None:
    detector = MimeSignatureDetector(
        settings=SimpleNamespace(
            enabled=True,
            sample_size_bytes=1024,
            maximum_signature_size=16,
            use_filetype=False,
        ),
        logger=_Logger(),
    )
    assert detector.detect(sample=b"%PDF-1.7") == "application/pdf"
    with pytest.raises(TypeError):
        detector.detect(body=b"%PDF-1.7")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("sample", "expected"),
    (
        (b"%PDF-1.7\n", "application/pdf"),
        (b"%PDF", None),
        (b"%PDFx-1.7\n", None),
        (b"\xef\xbb\xbf%PDF-1.7\n", None),
        (b" %PDF-1.7\n", None),
        (b"not-a-pdf%PDF-1.7\n", None),
    ),
)
def test_pdf_signature_requires_complete_magic_at_byte_zero(
    sample: bytes,
    expected: str | None,
) -> None:
    detector = MimeSignatureDetector(
        settings=SimpleNamespace(
            enabled=True,
            sample_size_bytes=1024,
            maximum_signature_size=16,
            use_filetype=False,
        ),
        logger=_Logger(),
    )

    assert detector.detect(sample=sample) == expected


def test_category_detector_no_longer_accepts_raw_body() -> None:
    parameters = inspect.signature(ContentCategoryDetector.detect).parameters
    assert "text_sample" in parameters
    assert "body" not in parameters


@pytest.mark.parametrize(
    "value",
    [
        "application/x-pdf",
        "audio/x-wav",
        "image/jpg",
        "text/x-markdown",
    ],
)
def test_noncanonical_mime_values_are_not_silently_rewritten(
    value: str,
) -> None:
    assert normalize_mime_type(value) == value
