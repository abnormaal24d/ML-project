from unittest.mock import Mock
from urllib.parse import ParseResult

import pytest

from crawler.governance.url_filter.url_syntax_rules import UrlSyntaxRules


@pytest.fixture
def rules() -> UrlSyntaxRules:
    return UrlSyntaxRules(
        max_page_number=200,
        pagination_query_keys=("page", "paged", "p", "start"),
        blocked_path_fragments=("/private/",),
        blocked_query_keys=(),
        blocked_query_value_patterns={"format": ("print",)},
        tracking_query_tokens=("tracker",),
        low_value_image_path_fragments=(),
        low_value_image_filenames=(),
        social_icon_tokens=(),
        logger=Mock(),
    )


def _parse(rules: UrlSyntaxRules, url: str) -> ParseResult:
    parsed = rules.parse_url(url)
    assert parsed is not None
    return parsed


def _shape_reason(
    rules: UrlSyntaxRules,
    url: str,
    *,
    source_type: str = "link",
) -> str | None:
    parsed = _parse(rules, url)
    return rules.url_shape_rejection_reason(
        url=url,
        path=parsed.path,
        query=parsed.query,
        source_type=source_type,
    )


@pytest.mark.parametrize("key", ["PAGE", "%50AGE"])
def test_pagination_query_keys_are_casefolded_and_decoded(
    rules: UrlSyntaxRules,
    key: str,
) -> None:
    assert (
        _shape_reason(rules, f"https://example.test/items?{key}=201")
        == "pagination_loop_candidate"
    )


@pytest.mark.parametrize(
    "query",
    ["PAGE=201&page=1", "page=1&PAGE=201"],
)
def test_case_colliding_pagination_keys_preserve_every_value(
    rules: UrlSyntaxRules,
    query: str,
) -> None:
    assert (
        _shape_reason(rules, f"https://example.test/items?{query}")
        == "pagination_loop_candidate"
    )


@pytest.mark.parametrize(
    "path",
    ["/page%2F201", "/PAGE/201", "/pa%67e%2f201"],
)
def test_policy_path_is_casefolded_after_one_decode(
    rules: UrlSyntaxRules,
    path: str,
) -> None:
    assert (
        _shape_reason(rules, f"https://example.test{path}")
        == "pagination_loop_candidate"
    )


def test_pagination_boundary_and_single_path_decode(
    rules: UrlSyntaxRules,
) -> None:
    assert _shape_reason(rules, "https://example.test/page/200") is None
    assert (
        _shape_reason(rules, "https://example.test/page/201")
        == "pagination_loop_candidate"
    )
    assert _shape_reason(rules, "https://example.test/page%252F201") is None


def test_deep_pagination_threshold_is_owned_by_constructor() -> None:
    rules = UrlSyntaxRules(
        max_page_number=10,
        pagination_query_keys=("page",),
        blocked_path_fragments=(),
        blocked_query_keys=(),
        blocked_query_value_patterns={},
        tracking_query_tokens=(),
        low_value_image_path_fragments=(),
        low_value_image_filenames=(),
        social_icon_tokens=(),
        logger=Mock(),
    )

    assert (
        _shape_reason(rules, "https://example.test/page/11")
        == "pagination_loop_candidate"
    )


def test_blocked_path_never_matches_query_or_host(
    rules: UrlSyntaxRules,
) -> None:
    assert (
        _shape_reason(
            rules,
            "https://private.example.test/safe?next=%2Fprivate%2F",
        )
        is None
    )
    assert (
        _shape_reason(rules, "https://example.test/PRIVATE/resource")
        == "blocked_path_pattern"
    )


@pytest.mark.parametrize(
    "query",
    ["FORMAT=print&format=html", "format=html&FORMAT=print"],
)
def test_case_colliding_blocked_query_keys_preserve_every_value(
    rules: UrlSyntaxRules,
    query: str,
) -> None:
    assert (
        rules.discovery_noise_reason(
            path="/document",
            query=query,
            kind="document",
        )
        == "blocked_query_pattern"
    )


def test_tracking_policy_uses_decoded_casefolded_query_values(
    rules: UrlSyntaxRules,
) -> None:
    assert (
        rules.discovery_noise_reason(
            path="/image.png",
            query="signed=unchanged&tag=%54RACKER",
            kind="image",
        )
        == "tracking_or_social_asset"
    )


def test_signed_fetch_url_is_unchanged_while_policy_recognizes_page(
    rules: UrlSyntaxRules,
) -> None:
    url = (
        "https://example.test/items?X-Amz-Signature=AbC%2FDeF"
        "&%50AGE=201&CaseSensitive=VaLue%2B1"
    )
    parsed = _parse(rules, url)

    assert parsed.geturl() == url
    assert _shape_reason(rules, url) == "pagination_loop_candidate"
    assert parsed.geturl() == url


def test_seed_is_explicitly_exempt_from_deep_pagination(
    rules: UrlSyntaxRules,
) -> None:
    assert (
        _shape_reason(
            rules,
            "https://example.test/items?PAGE=999",
            source_type="seed",
        )
        is None
    )
