"""URL canonicalization invariants shared by extraction and scheduling."""

from __future__ import annotations

import pytest

from config.collection.extraction import UrlNormalizerSettings
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer
from tests.support.logging import TEST_LOGGER


def _normalizer(**settings: object) -> UrlNormalizer:
    return UrlNormalizer(
        settings=UrlNormalizerSettings.model_validate(settings),
        logger=TEST_LOGGER,
        host_normalizer=HostNormalizer(),
    )


def test_disabled_optional_rules_still_apply_structural_canonicalization() -> (
    None
):
    normalizer = _normalizer(
        enabled=False,
    )
    query = "b=2&a=a+b%2Fc&utm_source=keep"

    normalized = normalizer.normalize(
        f"HTTP://BÜCHER.Example.:80/a/./b//../c/%7Eme/?{query}#Frag"
    )

    assert normalized == (
        f"http://xn--bcher-kva.example:80/a/c/~me/?{query}#Frag"
    )
    assert normalizer.normalize(normalized) == normalized


@pytest.mark.parametrize(
    "removed_setting",
    (
        "lowercase_scheme",
        "lowercase_host",
        "lowercase_scheme_and_host",
    ),
)
def test_removed_structural_canonicalization_switches_are_rejected(
    removed_setting: str,
) -> None:
    with pytest.raises(ValueError, match=removed_setting):
        UrlNormalizerSettings.model_validate({removed_setting: False})


def test_disabled_optional_rules_preserve_ipv6_port_and_query_bytes() -> None:
    query = "value=a+b%2fc&empty=&flag"

    normalized = _normalizer(enabled=False).normalize(
        f"HTTP://[2001:0DB8:0:0::1]:80/a/../b?{query}"
    )

    assert normalized == f"http://[2001:db8::1]:80/b?{query}"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/a//b/%2e%2E/c/%2F/d e/é", "/a/c/d%20e/%C3%A9"),
        ("/../../safe", "/safe"),
        ("//a///./b/../", "/a/"),
        (r"/a/\segment/b", "/a/%5Csegment/b"),
        ("/a/%FF/b", "/a/%FF/b"),
    ],
)
def test_structural_path_canonicalization_is_safe_and_idempotent(
    path: str,
    expected: str,
) -> None:
    normalizer = _normalizer(enabled=False)

    normalized = normalizer.normalize(f"https://EXAMPLE.com{path}")

    assert normalized == f"https://example.com{expected}"
    assert normalizer.normalize(normalized) == normalized


def test_relative_leading_dot_segments_are_removed() -> None:
    normalizer = _normalizer(enabled=False)

    assert normalizer.normalize("../a/./b") == "a/b"


@pytest.mark.parametrize(
    "url",
    [
        "http://[not-an-ipv6-address]/path",
        "http://example.com:not-a-port/path",
        "http://exa mple.com/path",
    ],
)
def test_invalid_authority_never_falls_back_to_the_raw_url(url: str) -> None:
    assert _normalizer(enabled=False).normalize(url) == ""


def test_signed_query_is_preserved_byte_for_byte_with_optional_rules_enabled() -> (
    None
):
    query = (
        "b=2&X-Amz-Signature=A%2fb+Z&"
        "X-Amz-Credential=AKIA%2F20260813&utm_source=paid"
    )

    normalized = _normalizer().normalize(
        f"HTTPS://EXAMPLE.com:443/a/../resource?{query}#section"
    )

    assert normalized == f"https://example.com/resource?{query}"


def test_unchanged_unsigned_query_keeps_its_original_encoding() -> None:
    query = "a=a+b%2fc&b=%7e"

    normalized = _normalizer().normalize(f"https://example.com/path?{query}")

    assert normalized == f"https://example.com/path?{query}"


def test_optional_query_equivalence_rules_still_remove_and_sort() -> None:
    normalized = _normalizer().normalize(
        "https://example.com/image.jpg?utm_source=paid&b=2&a=1&width=100"
    )

    assert normalized == "https://example.com/image.jpg?a=1&b=2"


def test_opt_in_http_upgrade_uses_https_default_port() -> None:
    normalizer = _normalizer(upgrade_http_to_https=True)

    assert normalizer.normalize("HTTP://EXAMPLE.com:80/article") == (
        "https://example.com/article"
    )
    assert normalizer.normalize("http://example.com:443/article") == (
        "https://example.com/article"
    )


def test_enabled_setting_documents_that_structure_cannot_be_disabled() -> None:
    description = UrlNormalizerSettings.model_fields["enabled"].description

    assert description is not None
    assert "structural URL canonicalization is always applied" in description
