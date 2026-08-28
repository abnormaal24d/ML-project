"""Fail-closed redaction contract for every structured log value."""

from __future__ import annotations

import pytest

from logger.redaction import (
    LOCAL_PATH,
    REDACTED,
    redact_log_text,
    redact_log_value,
)


@pytest.mark.parametrize(
    "field_name",
    [
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "passwd",
        "private_key",
        "secret",
        "session_token",
        "signature",
        "signed_token",
        "token",
        "OAUTH_TOKEN",
        "database-password",
    ],
)
def test_sensitive_field_names_are_redacted(field_name: str) -> None:
    assert (
        redact_log_value("sk-very-secret", field_name=field_name) == REDACTED
    )


def test_non_sensitive_values_pass_through() -> None:
    assert (
        redact_log_value("Public scene", field_name="title") == "Public scene"
    )
    assert redact_log_value(0.9, field_name="quality_score") == 0.9
    assert redact_log_value(42, field_name="duration_ms") == 42


def test_mapping_recursion_redacts_nested_sensitive_keys() -> None:
    value = {
        "caption": "Public",
        "headers": {"authorization": "Bearer abc"},
        "token_value": "t",
    }
    redacted = redact_log_value(value)
    assert redacted["caption"] == "Public"
    assert redacted["headers"]["authorization"] == REDACTED
    assert redacted["token_value"] == REDACTED


def test_tuple_list_and_set_recursion() -> None:
    value = {
        "tuples": ({"password": "pw"}, "plain"),
        "items": [{"secret": "s"}, 1],
        "tags": {"a", "b"},
    }
    redacted = redact_log_value(value)
    assert redacted["tuples"] == ({"password": REDACTED}, "plain")
    assert redacted["items"] == [{"secret": REDACTED}, 1]
    assert redacted["tags"] == {"a", "b"}


@pytest.mark.parametrize(
    "path_value",
    [
        r"C:\Users\abnor\private\file.png",
        r"D:/data/crawl/raw/image.png",
        "/home/user/crawl/raw/image.png",
        "/Users/abnor/work/archive.txt",
    ],
)
def test_absolute_path_in_path_field_is_local_path(path_value: str) -> None:
    assert redact_log_value(path_value, field_name="media_path") == LOCAL_PATH
    assert redact_log_value(path_value, field_name="path") == LOCAL_PATH


def test_relative_path_in_path_field_is_unchanged() -> None:
    assert redact_log_value("raw/image.png", field_name="media_path") == (
        "raw/image.png"
    )


def test_absolute_path_in_text_is_redacted() -> None:
    assert (
        redact_log_text(r"wrote to C:\Users\abnor\tmp\out.json")
        == f"wrote to {LOCAL_PATH}"
    )


def test_non_local_posix_path_is_not_redacted() -> None:
    assert redact_log_text("opened /etc/passwd") == "opened /etc/passwd"


def test_url_query_values_are_redacted_but_host_kept() -> None:
    result = redact_log_text("https://example.test/search?q=abc&token=xyz")
    assert "example.test" in result
    assert "q=" in result
    assert "token=" in result
    assert "xyz" not in result
    assert "%5BREDACTED%5D" in result


def test_plain_url_is_unchanged() -> None:
    assert redact_log_text("https://example.test/image.png") == (
        "https://example.test/image.png"
    )


def test_url_userinfo_is_dropped() -> None:
    result = redact_log_text("https://user:pass@example.test/feed")
    assert "user" not in result
    assert "pass" not in result
    assert "example.test" in result


def test_ipv6_host_with_port_is_kept() -> None:
    result = redact_log_text("https://[::1]:8080/a?x=1")
    assert "[::1]:8080" in result


def test_malformed_url_is_fully_redacted() -> None:
    assert redact_log_text("connect to http:///missing-host") == (
        "connect to " + REDACTED
    )


def test_sensitive_field_wins_over_path_redaction() -> None:
    assert (
        redact_log_value(
            r"C:\Users\abnor\secret.txt",
            field_name="authorization",
        )
        == REDACTED
    )
